"""Build the derived mitochondria data that the mitochondria EM generalist and its evaluation read.

- human_organoids: the instance labels of the manual semantic masks, split by a neck watershed
  -> <output>/human_organoids_mito/<volume>_mito_labels.zarr and <volume>_info.json.
- parlakgul_fasting: one central 64 x 1024 x 1024 box per fasting liver sample
  -> <output>/parlakgul_fasting_liver/<sample>_mito_<hash>.zarr.
- macrophage_fib_control: a block of slices of the Macrophage / A431 'fib_control' volume
  -> <output>/macrophage_a431/fib_control_z<start>-<stop>.h5.
- cellmap: the CellMap Segmentation Challenge crops with a mito label array
  -> <output>/cellmap_mito_crops/<volume>_crop<id>.zarr and index.json.
"""
import os
import json
import time
import shutil
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cc3d
import h5py
import numpy as np
from scipy import ndimage

import zarr
from zarr.codecs import BloscCodec

from skimage.measure import label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from torch_em.data.datasets.electron_microscopy import macrophage_a431, parlakgul_fasting_liver

from micro_sam.v2.datasets.generalist_loader import HUMAN_ORGANOIDS_MITO_ROOT, HUMAN_ORGANOIDS_VOLUMES


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
NECK_RATIO = 0.5
MIN_PIECE = 5000
SEED_HEIGHT = 1.0
CHUNKS = (32, 256, 256)
LABEL_SLAB = 128

# The components of the volume, shared with the worker processes through fork.
COMPONENTS = None

# (sample, (x_min, x_max, y_min, y_max, z_min, z_max)): the center of each sample volume.
PARLAKGUL_FASTING_BOXES = [
    ("8970", (4189, 5213, 2962, 3986, 4193, 4257)),
    ("2679", (2738, 3762, 2738, 3762, 3820, 3884)),
    ("1858", (4175, 5199, 4800, 5824, 2865, 2929)),
]

CELLMAP_ZIP = os.path.join("cellmap-segmentation-challenge", "matched_res_no_pad.zip")
CELLMAP_ZIP_ROOT = "data_matched_res_no_pad"
CELLMAP_MITO_CHUNKS = (64, 256, 256)
# The mito array of this large nucleus crop is all zero, but many mitochondria are visible in it.
CELLMAP_MITO_EXCLUDED = {("jrc_sum159-4", 211)}


def label_slabwise(mask):
    """3D connected components with 26-connectivity (as the training transform) into uint32, in merged z-slabs."""
    instances = np.zeros(mask.shape, dtype="uint32")
    offset = 0
    for z0 in range(0, mask.shape[0], LABEL_SLAB):
        labels, n = cc3d.connected_components(mask[z0:z0 + LABEL_SLAB], connectivity=26, return_N=True)
        labels = labels.astype("uint32", copy=False)
        labels[labels > 0] += offset
        instances[z0:z0 + LABEL_SLAB] = labels
        offset += n
    parent = np.arange(offset + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for z in range(LABEL_SLAB, mask.shape[0], LABEL_SLAB):
        below, above = instances[z - 1], instances[z]
        h, w = below.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                b = below[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
                a = above[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
                touching = (b > 0) & (a > 0)
                for u, v in np.unique(np.stack([b[touching], a[touching]], axis=1), axis=0):
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[max(ru, rv)] = min(ru, rv)
    roots = np.array([find(i) for i in range(offset + 1)])
    unique_roots, lut = np.unique(roots, return_inverse=True)
    lut = lut.astype("uint32")
    for z0 in range(0, mask.shape[0], LABEL_SLAB):
        instances[z0:z0 + LABEL_SLAB] = lut[instances[z0:z0 + LABEL_SLAB]]
    return instances, len(unique_roots) - 1


def oversegment(mask):
    """Seeded watershed on the smoothed distance transform; every connected component gets at least one seed."""
    components = label(mask)
    distance = ndimage.distance_transform_edt(mask).astype("float32")
    smooth = ndimage.gaussian_filter(distance, 1.0)
    markers = label(h_maxima(smooth, SEED_HEIGHT) & mask)
    missing = np.setdiff1d(np.arange(1, components.max() + 1), np.unique(components[markers > 0]))
    rescue = np.isin(components, missing)
    markers[rescue] = components[rescue] + markers.max()
    return distance, watershed(-smooth, markers, mask=mask)


def pair_saddles(segments, distance):
    """Per pair of touching segments: the highest distance on their shared boundary and the boundary size."""
    us, vs, saddles = [], [], []
    for axis in range(segments.ndim):
        seg, dist = np.moveaxis(segments, axis, 0), np.moveaxis(distance, axis, 0)
        left, right = seg[:-1], seg[1:]
        touch = (left != right) & (left > 0) & (right > 0)
        u, v = left[touch].astype("int64"), right[touch].astype("int64")
        us.append(np.minimum(u, v))
        vs.append(np.maximum(u, v))
        saddles.append(np.minimum(dist[:-1], dist[1:])[touch])
    u, v, saddle = np.concatenate(us), np.concatenate(vs), np.concatenate(saddles)
    n = int(segments.max()) + 1
    key = u * n + v
    order = np.argsort(key)
    key, saddle = key[order], saddle[order]
    if len(key) == 0:
        return key, key, saddle, key
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return key[starts] // n, key[starts] % n, np.maximum.reduceat(saddle, starts), np.diff(np.r_[starts, len(key)])


def merge_without_neck(segments, distance):
    """Merge segments that are not separated by a neck, then absorb pieces below MIN_PIECE voxels."""
    n = int(segments.max()) + 1
    peaks = ndimage.maximum(distance, segments, index=np.arange(n))
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    us, vs, saddles, contacts = pair_saddles(segments, distance)
    for u, v, saddle in zip(us, vs, saddles):
        if saddle > NECK_RATIO * min(peaks[u], peaks[v]):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    sizes = np.bincount(segments.ravel(), minlength=n)
    changed = True
    while changed:
        changed = False
        roots = np.array([find(i) for i in range(n)])
        group_size = np.zeros(n, dtype="int64")
        np.add.at(group_size, roots, sizes)
        best = {}
        for u, v, contact in zip(us, vs, contacts):
            ru, rv = roots[u], roots[v]
            if ru == rv:
                continue
            for a, b in ((ru, rv), (rv, ru)):
                if group_size[a] < MIN_PIECE and contact > best.get(a, (None, 0))[1]:
                    best[a] = (b, contact)
        for a, (b, _) in best.items():
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
                changed = True
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1][segments]


def split_component(task):
    """The pieces of one component as a local label array in its padded bounding box, or None if it stays whole."""
    component_id, box = task
    padded = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in box)
    mask = COMPONENTS[padded] == component_id
    if mask.sum() < 2 * MIN_PIECE:
        return component_id, padded, None
    distance, segments = oversegment(mask)
    pieces = merge_without_neck(segments, distance)
    pieces[~mask] = 0
    return component_id, padded, (pieces if pieces.max() > 1 else None)


def split_volume(mask, n_workers):
    global COMPONENTS
    COMPONENTS, n_components = label_slabwise(mask)
    boxes = ndimage.find_objects(COMPONENTS)
    tasks = [(i + 1, box) for i, box in enumerate(boxes) if box is not None]
    next_id, n_split, n_pieces = n_components + 1, 0, 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(split_component, tasks, chunksize=64))
    instances = COMPONENTS
    for component_id, padded, pieces in results:
        if pieces is None:
            continue
        n_split += 1
        n_pieces += int(pieces.max())
        region = instances[padded]
        for piece in range(2, int(pieces.max()) + 1):
            region[pieces == piece] = next_id
            next_id += 1
    COMPONENTS = None
    return instances, {
        "n_components": int(n_components), "n_components_split": n_split, "n_pieces_of_split_components": n_pieces,
        "n_instances": int(n_components - n_split + n_pieces),
    }


def write_labels(instances, out_path):
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    group = zarr.open_group(tmp_path, mode="w")
    array = group.create_array("labels", shape=instances.shape, chunks=CHUNKS, dtype="uint32")
    for z in range(0, instances.shape[0], CHUNKS[0]):
        array[z:z + CHUNKS[0]] = instances[z:z + CHUNKS[0]]
    group.attrs.update({"neck_ratio": NECK_RATIO, "min_piece": MIN_PIECE, "seed_height": SEED_HEIGHT})
    # The labels of an earlier run are replaced.
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)


def preprocess_human_organoids(input_root, output_root, volumes, n_workers):
    """Split the manual semantic mitochondria masks of the human organoids volumes into instances.

    Touching mitochondria form one connected component in the semantic masks. A neck watershed splits each component:
    - A seeded watershed on the distance transform over-segments the component.
    - Two touching segments merge, unless the distance on their boundary is at most NECK_RATIO of the smaller peak.
    - A piece below MIN_PIECE voxels joins the neighbour that shares the most boundary with it.

    The watershed runs per component in its bounding box, which keeps the memory low. The largest volume needs
    about 200 GB of memory, so run this choice on a large-memory node.
    """
    output = os.path.join(output_root, HUMAN_ORGANOIDS_MITO_ROOT)
    os.makedirs(output, exist_ok=True)
    for volume in volumes:
        path = os.path.join(input_root, "human_organoids", "11380", "data", f"{volume}_bin2.h5")
        with h5py.File(path, "r") as f:
            labels = f["mitos"]
            mask = np.zeros(labels.shape, dtype=bool)
            for z in range(0, labels.shape[0], CHUNKS[0]):
                mask[z:z + CHUNKS[0]] = labels[z:z + CHUNKS[0]] > 0
        instances, info = split_volume(mask, n_workers)
        del mask
        write_labels(instances, os.path.join(output, f"{volume}_mito_labels.zarr"))
        print(f"{volume}: {info}", flush=True)
        with open(os.path.join(output, f"{volume}_info.json"), "w") as f:
            json.dump(info, f, indent=2)


def download_parlakgul_fasting(output_root, index):
    """Download one central mitochondria box of a Parlakgul fasting liver sample (EMPIAR-12017, 8 nm).

    The torch-em loader streams the box and caches it as <output>/parlakgul_fasting_liver/<sample>_mito_<hash>.zarr.
    """
    sample, box = PARLAKGUL_FASTING_BOXES[index]
    path = parlakgul_fasting_liver.get_parlakgul_fasting_liver_data(
        os.path.join(output_root, "parlakgul_fasting_liver"), box, sample=sample, label_choice="mito", download=True,
    )
    print(sample, box, path, flush=True)


def download_slice(args, n_attempts=8):
    """Download one slice, with retries: the proxy and EMPIAR drop connections under load."""
    for attempt in range(n_attempts):
        try:
            return macrophage_a431._download_slice(args)
        except Exception as error:
            if attempt == n_attempts - 1:
                raise
            print(f"retrying {os.path.basename(args[0])} after {type(error).__name__}", flush=True)
            if os.path.exists(args[1]):
                os.remove(args[1])
            time.sleep(min(300, 10 * 2 ** attempt))


def download_macrophage_fib_control(output_root, start, stop, n_threads):
    """Download a contiguous block of slices of the Macrophage / A431 'fib_control' volume with all organelle labels.

    The full volume has 2888 slices of 1054 x 5408 pixels (6 x 6 x 12 nm). EMPIAR delivers about 0.5 MB/s per stream.
    Thus, only a block of slices around z = 911 is downloaded, where the sampled slices hold the most mitochondria.
    The file holds 'raw' and the binary 'labels/<organelle>' of the torch-em loader.
    """
    info = macrophage_a431.VOLUMES["fib_control"]
    out_dir = os.path.join(output_root, "macrophage_a431")
    out_path = os.path.join(out_dir, f"fib_control_z{start}-{stop}.h5")
    tmp_dir = f"{out_path}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    raw_args = macrophage_a431._slice_args(info, "raw", tmp_dir, True)[start:stop]
    label_args = macrophage_a431._slice_args(info, "label", tmp_dir, True)[start:stop]
    tmp_path = f"{out_path}.incomplete"
    step = 16
    with h5py.File(tmp_path, "w") as f, ThreadPoolExecutor(n_threads) as pool:
        datasets = {}
        for offset in range(0, len(raw_args), step):
            raw = np.stack(list(pool.map(download_slice, raw_args[offset:offset + step])))
            labels = np.stack(list(pool.map(download_slice, label_args[offset:offset + step])))
            assert raw.shape == labels.shape, f"{raw.shape} != {labels.shape}"
            if not datasets:
                shape = (len(raw_args),) + raw.shape[1:]
                chunks = (min(8, shape[0]), 512, 512)
                datasets["raw"] = f.create_dataset(
                    "raw", shape=shape, dtype=raw.dtype, chunks=chunks, compression="gzip"
                )
                for organelle in info["organelles"]:
                    datasets[organelle] = f.create_dataset(
                        f"labels/{organelle}", shape=shape, dtype="uint8", chunks=chunks, compression="gzip"
                    )
            end = offset + len(raw)
            datasets["raw"][offset:end] = raw
            for organelle, ids in info["organelles"].items():
                datasets[organelle][offset:end] = np.isin(labels, ids).astype("uint8")
            print(f"slices {start + offset}-{start + end} done", flush=True)
        f.attrs["z_range"] = [start, stop]
        f.attrs["voxel_size_nm"] = [12, 6, 6]
    os.replace(tmp_path, out_path)
    for name in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, name))
    os.rmdir(tmp_dir)
    print(out_path)


def read_zip_json(zip_file, key):
    """Read a json document from the zip file, or None if it does not exist."""
    try:
        return json.loads(zip_file.read(key))
    except KeyError:
        return None


def get_cellmap_levels(attrs):
    """Get the (path, scale, translation) of all scale levels from OME-Zarr multiscales attributes."""
    levels = []
    for dataset in attrs["multiscales"][0]["datasets"]:
        transforms = {t["type"]: t for t in dataset["coordinateTransformations"]}
        levels.append((dataset["path"], transforms["scale"]["scale"], transforms["translation"]["translation"]))
    return levels


def get_cellmap_em_levels(zip_file, volume):
    """Get the scale levels of the EM volume."""
    key = f"{CELLMAP_ZIP_ROOT}/{volume}/{volume}.zarr/recon-1/em/fibsem-uint8/.zattrs"
    return get_cellmap_levels(read_zip_json(zip_file, key))


def match_cellmap_level(label_levels, em_levels):
    """Find the finest pair of label level and EM level with the same scale."""
    matches = []
    for label_path, label_scale, label_translation in label_levels:
        for em_path, em_scale, em_translation in em_levels:
            if np.allclose(label_scale, em_scale, rtol=1e-3, atol=1e-6):
                match = (label_path, em_path, label_scale, label_translation, em_translation)
                matches.append((np.prod(label_scale), *match))
    if not matches:
        return None
    matches.sort(key=lambda m: m[0])
    return matches[0][1:]


def list_cellmap_crops(zip_file, class_name):
    """The (volume, crop id) pairs with a label array of one class, sorted by volume and crop id."""
    crops = set()
    for name in zip_file.namelist():
        parts = name.split("/")
        if len(parts) == 9 and parts[4:6] == ["labels", "groundtruth"] and parts[7:] == [class_name, ".zattrs"]:
            crops.add((parts[1], int(parts[6][len("crop"):])))
    return sorted(crops)


def load_cellmap_crop(store, zip_file, volume, crop_id, em_levels, class_name):
    """Load the EM crop and the labels of one class, at the finest label scale that matches an EM scale.

    This follows torch-em's '_download_cellmap_data'. The unknown label value becomes 0. Some crops reach beyond the
    EM volume, so the labels are clipped to the EM extent.
    """
    zarr_root = f"{CELLMAP_ZIP_ROOT}/{volume}/{volume}.zarr/recon-1"
    group = f"{zarr_root}/labels/groundtruth/crop{crop_id}/{class_name}"
    attrs = read_zip_json(zip_file, f"{group}/.zattrs")
    match = match_cellmap_level(get_cellmap_levels(attrs), em_levels)
    assert match is not None, f"No EM scale matches the {class_name} labels of crop {crop_id}"
    label_path, em_path, scale, translation, em_translation = match
    unknown = attrs["cellmap"]["annotation"]["annotation_type"]["encoding"]["unknown"]
    data = zarr.open_array(store=store, path=f"{group}/{label_path}", mode="r", zarr_format=2)[:]
    em = zarr.open_array(store=store, path=f"{zarr_root}/em/fibsem-uint8/{em_path}", mode="r", zarr_format=2)
    starts = [int(round((t - et) / s)) for t, et, s in zip(translation, em_translation, scale)]
    bb = tuple(slice(max(start, 0), min(start + size, n)) for start, size, n in zip(starts, data.shape, em.shape))
    label_bb = tuple(slice(b.start - start, b.stop - start) for b, start in zip(bb, starts))
    raw = em[bb]
    n_unknown = int((data == unknown).sum())
    labels = np.where(data == unknown, 0, data)[label_bb]
    info = {
        "label_level": label_path, "em_level": em_path, "voxel_size_nm": scale, "translation_nm": translation,
        "em_start_voxel": starts, "unknown_value": unknown, "n_unknown_voxels": n_unknown,
        "clipped_to_em": tuple(raw.shape) != tuple(data.shape), "raw_zero_frac": float((raw == 0).mean()),
    }
    return raw, labels, info


def write_cellmap_mito_crop(store, zip_file, volume, crop_id, em_levels, out_dir):
    """Write the EM crop and the mito labels (original ids and dtype) of one crop to one zarr store."""
    raw, labels, info = load_cellmap_crop(store, zip_file, volume, crop_id, em_levels, "mito")
    assert raw.shape == labels.shape, f"{raw.shape} != {labels.shape}"
    name = f"{volume}_crop{crop_id}"
    out_path = os.path.join(out_dir, f"{name}.zarr")
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    chunks = tuple(min(c, s) for c, s in zip(CELLMAP_MITO_CHUNKS, raw.shape))
    group = zarr.open_group(tmp_path, mode="w")
    group.create_array("raw", data=raw, chunks=chunks, compressors=BloscCodec())
    group.create_array("labels", data=labels, chunks=chunks, compressors=BloscCodec())
    group.attrs.update({
        "volume": volume, "crop": crop_id, "label_level": info["label_level"], "em_level": info["em_level"],
        "voxel_size_nm": info["voxel_size_nm"], "translation_nm": info["translation_nm"],
        "em_start_voxel": info["em_start_voxel"], "unknown_value": info["unknown_value"],
        "n_unknown_voxels_mapped_to_0": info["n_unknown_voxels"],
    })
    # The crop of an earlier run is replaced.
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)
    ids = np.unique(labels)
    entry = {
        "name": name, "path": out_path, "volume": volume, "crop": crop_id, "shape": list(raw.shape),
        "voxel_size_nm": info["voxel_size_nm"], "n_ids": int((ids != 0).sum()), "fg_frac": float((labels > 0).mean()),
        "n_unknown_voxels_mapped_to_0": info["n_unknown_voxels"], "n_value_255": int((labels == 255).sum()),
        "dtype": str(labels.dtype),
    }
    print(entry, flush=True)
    return entry


def preprocess_cellmap(input_root, output_root):
    """Cache the CellMap crops with a mito label array, read directly from the zip of the challenge data.

    Every crop is one zarr store with 'raw' and 'labels' at the finest mito label scale that matches an EM scale.
    The labels keep their ids and dtype; only the unknown value becomes 0.
    """
    zip_path = os.path.join(input_root, CELLMAP_ZIP)
    out_dir = os.path.join(output_root, "cellmap_mito_crops")
    os.makedirs(out_dir, exist_ok=True)
    store = zarr.storage.ZipStore(zip_path, mode="r")
    zip_file = zipfile.ZipFile(zip_path)
    index, em_levels = [], {}
    for volume, crop_id in list_cellmap_crops(zip_file, "mito"):
        if (volume, crop_id) in CELLMAP_MITO_EXCLUDED:
            continue
        if volume not in em_levels:
            em_levels[volume] = get_cellmap_em_levels(zip_file, volume)
        index.append(write_cellmap_mito_crop(store, zip_file, volume, crop_id, em_levels[volume], out_dir))
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


def main():
    choices = ["human_organoids", "parlakgul_fasting", "macrophage_fib_control", "cellmap"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=choices, help="The dataset to preprocess.")
    parser.add_argument("--input_root", default=DATA_ROOT, help="The data root to read the source data from.")
    parser.add_argument("--output_root", default=DATA_ROOT, help="The data root to write the derived data to.")
    parser.add_argument("--volumes", nargs="+", default=None, help="The human organoids volumes. Defaults to all.")
    parser.add_argument("--n_workers", type=int, default=16, help="Processes for the per-component watershed.")
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    parser.add_argument("--index", type=int, default=task_id, help="The Parlakgul fasting box (the Slurm array task).")
    parser.add_argument("--start", type=int, default=848, help="The first macrophage fib_control slice (zero-based).")
    parser.add_argument("--stop", type=int, default=976, help="The macrophage fib_control slice after the last one.")
    parser.add_argument("--n_threads", type=int, default=8, help="Parallel macrophage fib_control downloads.")
    args = parser.parse_args()
    if args.dataset == "human_organoids":
        volumes = args.volumes or [name.replace("_bin2.h5", "") for name in HUMAN_ORGANOIDS_VOLUMES]
        preprocess_human_organoids(args.input_root, args.output_root, volumes, args.n_workers)
    elif args.dataset == "parlakgul_fasting":
        download_parlakgul_fasting(args.output_root, args.index)
    elif args.dataset == "macrophage_fib_control":
        download_macrophage_fib_control(args.output_root, args.start, args.stop, args.n_threads)
    else:
        preprocess_cellmap(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
