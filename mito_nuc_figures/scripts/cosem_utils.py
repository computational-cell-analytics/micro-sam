"""Helpers to read the janelia-cosem-datasets (OpenOrganelle / CellMap) S3 bucket for the nucleus overview figures."""
import time

import requests
import numpy as np


BUCKET_URL = "https://janelia-cosem-datasets.s3.amazonaws.com/"
S3_ROOT = "s3://janelia-cosem-datasets/"


def get_json(url, n_tries=5):
    """Get a json file, retrying on connection errors (the bucket occasionally drops SSL connections)."""
    for attempt in range(n_tries):
        try:
            resp = requests.get(url, timeout=60)
            break
        except requests.exceptions.ConnectionError:
            if attempt == n_tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    resp.raise_for_status()
    return resp.json()


def group_path(dataset_name, recon, subpath):
    return f"{dataset_name}/{dataset_name}.zarr/{recon}/{subpath}"


def get_levels(path):
    """Read the multiscale levels of a zarr v2 or v3 group: {level: dict(scale, translation, shape, dtype)}."""
    try:
        multiscales = get_json(f"{BUCKET_URL}{path}/.zattrs")["multiscales"]
        zarr_format = 2
    except Exception:
        attrs = get_json(f"{BUCKET_URL}{path}/zarr.json")["attributes"]
        multiscales = attrs["ome"]["multiscales"] if "ome" in attrs else attrs["multiscales"]
        zarr_format = 3
    levels = {}
    for ds in multiscales[0]["datasets"]:
        transforms = {t["type"]: t.get(t["type"]) for t in ds["coordinateTransformations"]}
        if zarr_format == 2:
            meta = get_json(f"{BUCKET_URL}{path}/{ds['path']}/.zarray")
            dtype = meta["dtype"]
        else:
            meta = get_json(f"{BUCKET_URL}{path}/{ds['path']}/zarr.json")
            dtype = meta["data_type"]
        levels[ds["path"]] = {
            "scale": tuple(float(s) for s in transforms["scale"]),
            "translation": tuple(float(t) for t in (transforms.get("translation") or (0.0, 0.0, 0.0))),
            "shape": tuple(meta["shape"]), "dtype": dtype,
        }
    return levels


def open_remote(path):
    import zarr
    from zarr.storage import FsspecStore

    store = FsspecStore.from_url(f"{S3_ROOT}{path}", storage_options={"anon": True}, read_only=True)
    return zarr.open(store, mode="r")


def index_map(target_shape, target_scale, target_translation, source_shape, source_scale, source_translation):
    """Nearest-neighbor index maps from a target grid to a source grid, via physical coordinates."""
    maps = []
    for n_t, s_t, o_t, n_s, s_s, o_s in zip(
        target_shape, target_scale, target_translation, source_shape, source_scale, source_translation
    ):
        coords = np.arange(n_t) * s_t + o_t
        maps.append(np.clip(np.round((coords - o_s) / s_s).astype(int), 0, n_s - 1))
    return maps


class LazyView:
    """Lazy (z, y, x) view of a remote array, with optional axis reversal and index maps, caching full z-planes.

    Args:
        array: The source array (zarr or numpy), indexed in its own axis order.
        reverse: Whether the source axis order is (x, y, z) relative to the target.
        maps: Optional per-axis index arrays (in target axis order) mapping target to source indices.
        shape: The target shape. Defaults to the (reversed) source shape.
    """

    def __init__(self, array, reverse=False, maps=None, shape=None):
        self.array = array
        self.reverse = reverse
        source_shape = tuple(array.shape)[::-1] if reverse else tuple(array.shape)
        self.maps = maps if maps is not None else [np.arange(n) for n in source_shape]
        self.shape = tuple(len(m) for m in self.maps) if shape is None else shape
        self.ndim = 3
        self.dtype = array.dtype
        self.planes = {}

    def read(self, indices):
        """Read an orthogonal selection given as one index array per target axis."""
        indices = [np.atleast_1d(idx) for idx in indices]
        source = indices[::-1] if self.reverse else indices
        selection = []
        for idx in source:
            if len(idx) > 1 and np.all(np.diff(idx) == 1):
                selection.append(slice(int(idx[0]), int(idx[-1]) + 1))
            else:
                selection.append(idx)
        if hasattr(self.array, "get_orthogonal_selection"):
            data = self.array.get_orthogonal_selection(tuple(selection))
        else:
            full = [np.arange(s.start, s.stop) if isinstance(s, slice) else s for s in selection]
            data = np.asarray(self.array)[np.ix_(*full)]
        data = np.asarray(data)
        return np.transpose(data) if self.reverse else data

    def get_plane(self, z):
        if z not in self.planes:
            self.planes[z] = self.read([self.maps[0][z:z + 1], self.maps[1], self.maps[2]])[0]
        return self.planes[z]

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        key = key + (slice(None),) * (3 - len(key))
        if isinstance(key[0], (int, np.integer)):
            return self.get_plane(int(key[0]))[key[1:]]
        indices, squeeze = [], []
        for axis, k in enumerate(key):
            if isinstance(k, (int, np.integer)):
                indices.append(self.maps[axis][int(k):int(k) + 1])
                squeeze.append(axis)
            else:
                indices.append(self.maps[axis][k])
        data = self.read(indices)
        return np.squeeze(data, axis=tuple(squeeze)) if squeeze else data


def count_ids(array, chunk_depth=64):
    """Count the non-zero label ids of a (lazy) 3D array, reading it in z-chunks."""
    ids, counts = set(), None
    for z in range(0, array.shape[0], chunk_depth):
        block = np.asarray(array[z:z + chunk_depth])
        if block.dtype.itemsize <= 2 and block.dtype.kind == "u":
            block_counts = np.bincount(block.ravel(), minlength=2 ** (8 * block.dtype.itemsize))
            counts = block_counts if counts is None else counts + block_counts
        else:
            ids.update(np.unique(block[block > 0]).tolist())
    if counts is not None:
        ids.update(np.nonzero(counts[1:])[0].tolist())
    return len(ids)
