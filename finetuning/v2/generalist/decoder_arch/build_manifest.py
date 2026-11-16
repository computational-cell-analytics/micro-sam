"""Write the fixed train / validation subsets of the decoder architecture evaluation and check them for leakage.

The manifest lists, per unit (one dataset or collection and split), the raw and label files, the z-ranges of the
volumes and the number of samples drawn per epoch. Every arm trains on the same manifest.

Leakage checks (the run fails if any of them fails):
1. Every 2D plane (each channel) of every train / validation image is hashed exactly and as a downsampled, coarsely
   quantised thumbnail, and compared against the planes of the six 2D test sets (LIVECell, CellPose cyto, TissueNet,
   CovidIF cells, Usiigaci, DSB) and against the other split.
2. Every non-constant z-slice of the train / validation volumes is hashed and compared against the in-domain 3D test
   volumes and PlantSeg ovules. A train volume that shares a slice with a test volume is dropped and reported
   (PlantSeg root ships 'Movie2_T00010' in both its train and its test split).
3. The z-gaps between train and validation regions and the time gaps of the time-lapses are asserted.
4. TissueNet: the converted zarr files carry no metadata. The experiment of every image is read from
   tissuenet_v1.1.zip, after checking at several indices that the zarr index equals the npz index.

Usage (a CPU job, about 30 min):
    python build_manifest.py --workers 16
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_arch_lib as lib  # noqa: E402

SEED = 0
D = lib.DATA_ROOT

# Samples per epoch (train) and fixed validation sizes.
LIVECELL_TRAIN_PER_TYPE, LIVECELL_VAL_PER_TYPE = 100, 15
CELLPOSE_VAL = 80
CELLPOSE_TRAIN_SAMPLES = 800
TISSUENET_TRAIN, TISSUENET_VAL = 800, 120
SAMPLES_3D = {"nis3d": 100, "plantseg_root": 100, "embedseg": 25}  # embedseg: per collection
VAL_CROPS_3D = {"nis3d": 8, "plantseg_root": 8, "embedseg": 4}  # embedseg: per collection

GAP_Z = 8
NIS3D_TRAIN_Z, NIS3D_VAL_Z = (0, 152), (160, 200)
ORGANOID_TRAIN_TIMEPOINTS = list(range(0, 80, 5))
ORGANOID_VAL_TIMEPOINTS = [86, 91]
ORGANOID_TEST_TIMEPOINTS = list(range(97, 108))
PLATY_NUCLEI_TRAIN, PLATY_NUCLEI_VAL, PLATY_NUCLEI_TEST = list(range(7)), [7], [8]
EMBEDSEG_SLAB = {"Mouse-Skull-Nuclei-CBG": 25, "Platynereis-ISH-Nuclei-CBG": 21}  # validation slab depth


#
# Hashing
#


def _thumb_hash(plane: np.ndarray) -> str:
    """A hash that survives re-encoding and resampling: 32 x 32 block means, percentile-normalised, 8 levels."""
    from skimage.transform import resize

    small = resize(plane.astype("float32"), (32, 32), anti_aliasing=True, preserve_range=True)
    lo, hi = np.percentile(small, (1, 99))
    q = np.clip((small - lo) / max(hi - lo, 1e-6), 0, 1)
    return hashlib.sha1(np.round(q * 7).astype("uint8").tobytes()).hexdigest()


def _read(path, key=None):
    if key is None:
        import imageio.v3 as iio
        if path.endswith((".tif", ".tiff")):
            import tifffile
            return tifffile.imread(path)
        return iio.imread(path)
    from elf.io import open_file
    with open_file(path, "r") as f:
        return f[key][:]


def _planes(array: np.ndarray, channel_axis_last: bool):
    if array.ndim == 2:
        yield array
    elif array.ndim == 3:
        planes = np.moveaxis(array, -1, 0) if channel_axis_last else array
        for plane in planes:
            yield plane
    else:
        raise ValueError(array.shape)


def hash_image(args):
    """Exact and thumbnail hashes of the non-constant planes of a 2D image. args: (path, key or keys)."""
    path, keys = args
    keys = keys if isinstance(keys, (list, tuple)) else [keys]
    exact, thumb = [], []
    for key in keys:
        array = np.asarray(_read(path, key))
        last = array.ndim == 3 and array.shape[-1] <= 4 and array.shape[0] > 4
        for plane in _planes(array, last):
            if plane.min() == plane.max():
                continue
            exact.append(hashlib.sha1(np.ascontiguousarray(plane).tobytes()).hexdigest())
            thumb.append(_thumb_hash(plane))
    return path, exact, thumb


def hash_volume(args):
    """Exact hashes of the non-constant z-slices of a volume. args: (path, key, z-range or None)."""
    path, key, zrange = args
    volume = np.asarray(_read(path, key))
    if zrange is not None:
        volume = volume[zrange[0]:zrange[1]]
    hashes = [
        hashlib.sha1(np.ascontiguousarray(plane).tobytes()).hexdigest()
        for plane in volume if plane.min() != plane.max()
    ]
    return path, zrange, hashes, list(volume.shape)


#
# Selection helpers
#


def round_robin(items, key, n, rng):
    """Take n items spread over the groups of `key`: shuffle within each group, then interleave the groups."""
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    order = sorted(groups)
    for k in order:
        rng.shuffle(groups[k])
    selected, depth = [], 0
    while len(selected) < n:
        added = False
        for k in order:
            if depth < len(groups[k]):
                selected.append(groups[k][depth])
                added = True
                if len(selected) == n:
                    break
        if not added:
            raise RuntimeError(f"Only {len(selected)} items, need {n}.")
        depth += 1
    return selected


def stratified_sample(indices_by_group, n, rng):
    """Proportional allocation (largest remainder, at least one per group if possible), random within groups."""
    total = sum(len(v) for v in indices_by_group.values())
    quotas = {g: n * len(v) / total for g, v in indices_by_group.items()}
    alloc = {g: min(len(indices_by_group[g]), max(1, int(np.floor(q)))) for g, q in quotas.items()}
    remainder = sorted(quotas, key=lambda g: quotas[g] - np.floor(quotas[g]), reverse=True)
    i = 0
    while sum(alloc.values()) < n:
        g = remainder[i % len(remainder)]
        if alloc[g] < len(indices_by_group[g]):
            alloc[g] += 1
        i += 1
    while sum(alloc.values()) > n:
        g = max(alloc, key=lambda g: alloc[g])
        alloc[g] -= 1
    selected = []
    for g in sorted(indices_by_group):
        selected.extend(rng.choice(indices_by_group[g], size=alloc[g], replace=False).tolist())
    return sorted(selected), alloc


#
# TissueNet metadata
#


def _npz_member(split):
    outer = zipfile.ZipFile(os.path.join(D, "tissuenet", "tissuenet_v1.1.zip"))
    return zipfile.ZipFile(outer.open(f"tissuenet_v1.1_{split}.npz"))


def tissuenet_experiments(split, cache_dir):
    """The experiment of every image of a TissueNet split, cached as json."""
    cache = os.path.join(cache_dir, f"tissuenet_meta_{split}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    meta = np.load(io.BytesIO(_npz_member(split).read("meta.npy")), allow_pickle=True)
    header, rows = list(meta[0]), meta[1:]
    experiments = [str(row[header.index("experiment")]).split("/")[-1] for row in rows]
    with open(cache, "w") as f:
        json.dump(experiments, f)
    return experiments


def verify_tissuenet_index(split, zarr_paths, check_indices):
    """Stream X.npy of the npz and compare the nucleus channel with the zarr at the given indices."""
    import zarr
    from numpy.lib import format as npf

    member = _npz_member(split).open("X.npy")
    version = npf.read_magic(member)
    read_header = npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
    shape, fortran, dtype = read_header(member)
    assert not fortran and len(shape) == 4, shape
    n_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    results, position = {}, 0
    for index in sorted(check_indices):
        to_skip = (index - position) * n_bytes
        while to_skip > 0:
            chunk = member.read(min(to_skip, 1 << 26))
            to_skip -= len(chunk)
        image = np.frombuffer(member.read(n_bytes), dtype=dtype).reshape(shape[1:])
        position = index + 1
        stored = zarr.open(zarr_paths[index], mode="r")["raw/nucleus"][:]
        corr = float(np.corrcoef(image[..., 0].ravel(), stored.ravel())[0, 1])
        results[index] = corr
    return shape, results


#
# Manifest
#


def unit(name, split, source, ndim, raw_paths, label_paths, n_samples=None, rois=None, **extra):
    return {
        "name": name, "split": split, "source": source, "ndim": ndim,
        "raw_paths": list(map(str, raw_paths)), "label_paths": list(map(str, label_paths)),
        "n_samples": n_samples, "rois": rois, **extra,
    }


def build_2d_units(cache_dir, rng):
    from torch_em.data import datasets
    units, info = [], {}

    # LIVECell: one held-out well per cell type validates; the other wells train.
    root = os.path.join(D, "livecell")
    raw, labels = datasets.livecell.get_livecell_paths(root, split="train")
    val_names = {os.path.basename(p) for p in datasets.livecell.get_livecell_paths(root, split="val")[0]}
    test_raw = datasets.livecell.get_livecell_paths(root, split="test")[0]
    pairs = [(r, lab) for r, lab in zip(raw, labels) if os.path.basename(r) not in val_names]
    info["livecell_dropped_in_both_official_lists"] = len(raw) - len(pairs)

    def parse(path):
        cell_type, _, well, fov, time_point, _ = os.path.basename(path)[:-4].split("_")
        return cell_type, well, fov, time_point

    test_wells = {(parse(p)[0], parse(p)[1]) for p in test_raw}
    by_type = defaultdict(list)
    for pair in pairs:
        by_type[parse(pair[0])[0]].append(pair)
    info["livecell_wells"] = {}
    for cell_type in sorted(by_type):
        wells = sorted({parse(p[0])[1] for p in by_type[cell_type]})
        val_well = wells[rng.integers(len(wells))]
        val_pool = [p for p in by_type[cell_type] if parse(p[0])[1] == val_well]
        train_pool = [p for p in by_type[cell_type] if parse(p[0])[1] != val_well]
        train = round_robin(train_pool, lambda p: parse(p[0])[3], LIVECELL_TRAIN_PER_TYPE, rng)
        val = round_robin(val_pool, lambda p: parse(p[0])[3], LIVECELL_VAL_PER_TYPE, rng)
        train_wells = sorted({parse(p[0])[1] for p in train})
        assert val_well not in train_wells
        assert not any((cell_type, w) in test_wells for w in train_wells + [val_well]), cell_type
        info["livecell_wells"][cell_type] = {
            "train": train_wells, "val": val_well,
            "train_time_points": len({parse(p[0])[3] for p in train}),
        }
        units.append(unit(f"livecell_{cell_type}", "train", "livecell", 2, *zip(*train),
                          n_samples=LIVECELL_TRAIN_PER_TYPE))
        units.append(unit(f"livecell_{cell_type}", "val", "livecell", 2, *zip(*val)))

    # CellPose: 80 validation images, proportional over cyto / cyto2; the rest trains.
    root = os.path.join(D, "cellpose")
    pools = {c: list(zip(*datasets.cellpose.get_cellpose_paths(root, split="train", choice=c))) for c in
             ("cyto", "cyto2")}
    total = sum(len(v) for v in pools.values())
    train, val = [], []
    info["cellpose"] = {}
    for c, pool in pools.items():
        n_val = int(round(CELLPOSE_VAL * len(pool) / total))
        order = rng.permutation(len(pool))
        val += [pool[i] for i in sorted(order[:n_val])]
        train += [pool[i] for i in sorted(order[n_val:])]
        info["cellpose"][c] = {"files": len(pool), "val": n_val}
    assert len(val) == CELLPOSE_VAL, len(val)
    units.append(unit("cellpose", "train", "cellpose", 2, *zip(*train), n_samples=CELLPOSE_TRAIN_SAMPLES))
    units.append(unit("cellpose", "val", "cellpose", 2, *zip(*val)))

    # TissueNet: stratified by experiment (tissue x platform x lab), after checking the zarr <-> npz index.
    info["tissuenet"] = {}
    for split, n, name in (("train", TISSUENET_TRAIN, "train"), ("val", TISSUENET_VAL, "val")):
        paths = sorted(glob(os.path.join(D, "tissuenet", split, "*.zarr")))
        experiments = tissuenet_experiments(split, cache_dir)
        assert len(experiments) == len(paths), (split, len(experiments), len(paths))
        check = [0, 1, len(paths) // 2, len(paths) - 1]
        shape, correlations = verify_tissuenet_index(split, paths, check)
        assert shape[0] == len(paths) and min(correlations.values()) > 0.99, (split, shape, correlations)
        groups = defaultdict(list)
        for i, e in enumerate(experiments):
            groups[e].append(i)
        selected, alloc = stratified_sample(groups, n, rng)
        info["tissuenet"][split] = {
            "files": len(paths), "index_check_correlation": correlations, "allocation": alloc,
        }
        sel = [paths[i] for i in selected]
        units.append(unit("tissuenet", name, "tissuenet", 2, sel, sel,
                          n_samples=n, experiments=[experiments[i] for i in selected]))
    return units, info


def build_3d_units(info):
    import tifffile
    from torch_em.data import datasets
    units = []

    # NIS3D: Drosophila_2 (official train), z 0-152 trains, z 160-200 validates.
    raw, labels = datasets.nis3d.get_nis3d_paths(os.path.join(D, "nis3d"), split="train", split_type="cross-image")
    raw, labels = [p for p in raw if "Drosophila" in p], [p for p in labels if "Drosophila" in p]
    assert len(raw) == 1, raw
    depth = tifffile.TiffFile(raw[0]).series[0].shape[0]
    assert depth == NIS3D_VAL_Z[1], depth
    assert NIS3D_VAL_Z[0] - NIS3D_TRAIN_Z[1] >= GAP_Z
    units.append(unit("nis3d", "train", "nis3d", 3, raw, labels, SAMPLES_3D["nis3d"], [list(NIS3D_TRAIN_Z)]))
    units.append(unit("nis3d", "val", "nis3d", 3, raw, labels, VAL_CROPS_3D["nis3d"], [list(NIS3D_VAL_Z)]))

    # PlantSeg root: official train / val (train volumes that share slices with a test volume are dropped later).
    for split in ("train", "val"):
        paths = sorted(glob(os.path.join(D, "plantseg", f"root_{split}", "*.h5")))
        n = SAMPLES_3D["plantseg_root"] if split == "train" else VAL_CROPS_3D["plantseg_root"]
        units.append(unit("plantseg_root", split, "plantseg_root", 3, paths, paths, n))

    # EmbedSeg.
    root = os.path.join(D, "embedseg")
    for collection in ("Mouse-Organoid-Cells-CBG", "Platynereis-Nuclei-CBG"):
        raw, labels = datasets.embedseg_data.get_embedseg_paths(root, name=collection, split="train")
        raw, labels = sorted(raw), sorted(labels)
        if collection.startswith("Mouse-Organoid"):
            train_t, val_t, test_t = ORGANOID_TRAIN_TIMEPOINTS, ORGANOID_VAL_TIMEPOINTS, ORGANOID_TEST_TIMEPOINTS
            assert len(raw) == 108
        else:
            train_t, val_t, test_t = PLATY_NUCLEI_TRAIN, PLATY_NUCLEI_VAL, PLATY_NUCLEI_TEST
            assert len(raw) == 9
        assert max(train_t) < min(val_t) and max(val_t) < min(test_t)
        source = f"embedseg:{collection}"
        units.append(unit(collection, "train", source, 3, [raw[t] for t in train_t], [labels[t] for t in train_t],
                          SAMPLES_3D["embedseg"], time_points=train_t))
        units.append(unit(collection, "val", source, 3, [raw[t] for t in val_t], [labels[t] for t in val_t],
                          VAL_CROPS_3D["embedseg"], time_points=val_t))
        info.setdefault("embedseg_test", {})[collection] = [raw[t] for t in test_t]

    for collection, slab in EMBEDSEG_SLAB.items():
        raw, labels = datasets.embedseg_data.get_embedseg_paths(root, name=collection, split="train")
        raw, labels = sorted(raw), sorted(labels)
        depths = [tifffile.TiffFile(p).series[0].shape[0] for p in raw]
        largest = int(np.argmax(depths))
        # The slab sits at the end of the labelled extent: Platynereis-ISH X01 is labelled in z 107-365 of 515 only.
        mask = tifffile.imread(labels[largest])
        labelled = np.nonzero(mask.reshape(mask.shape[0], -1).max(axis=1) > 0)[0]
        z_end = int(labelled.max()) + 1
        z_val = (z_end - slab, z_end)
        z_train = (0, z_val[0] - GAP_Z)
        n_ids = len(np.unique(mask[z_val[0]:z_val[1]])) - 1
        assert n_ids >= 3 and z_val[0] - z_train[1] >= GAP_Z, (collection, z_val, n_ids)
        info.setdefault("embedseg_slabs", {})[collection] = {
            "volume": raw[largest], "depth": depths[largest], "labelled_z": [int(labelled.min()), z_end],
            "train_z": list(z_train), "val_z": list(z_val), "val_instances": n_ids,
        }
        rois = [list(z_train) if i == largest else None for i in range(len(raw))]
        source = f"embedseg:{collection}"
        units.append(unit(collection, "train", source, 3, raw, labels, SAMPLES_3D["embedseg"], rois))
        units.append(unit(collection, "val", source, 3, [raw[largest]], [labels[largest]], VAL_CROPS_3D["embedseg"],
                          [list(z_val)]))
        test_raw, _ = datasets.embedseg_data.get_embedseg_paths(root, name=collection, split="test")
        info.setdefault("embedseg_test", {})[collection] = sorted(test_raw)
    return units


def test_sets_2d():
    """(path, key) of every image of the six 2D test sets."""
    from torch_em.data import datasets
    lm = datasets
    items = {}
    items["livecell"] = [(p, None) for p in lm.livecell.get_livecell_paths(os.path.join(D, "livecell"), "test")[0]]
    items["cellpose"] = [(p, None) for p in lm.cellpose.get_cellpose_paths(
        os.path.join(D, "cellpose"), split="test", choice="cyto")[0]]
    items["tissuenet"] = [(p, "raw/rgb") for p in sorted(glob(os.path.join(D, "tissuenet", "test", "*.zarr")))]
    items["covid_if"] = [(p, ["raw/serum_IgG/s0", "raw/nuclei/s0"]) for p in sorted(
        lm.covid_if.get_covid_if_paths(os.path.join(D, "covid_if"), sample_range=(0, 44)))]
    usiigaci = []
    for split in ("train", "val"):
        usiigaci += lm.usiigaci.get_usiigaci_paths(os.path.join(D, "usiigaci"), split=split)[0]
    items["usiigaci"] = [(p, None) for p in usiigaci]
    items["dsb"] = [(p, None) for p in lm.dsb.get_dsb_paths(os.path.join(D, "dsb"), source="reduced", split="test")[0]]
    return items


def test_volumes_3d(info):
    from torch_em.data import datasets
    vols = []
    raw, _ = datasets.nis3d.get_nis3d_paths(os.path.join(D, "nis3d"), split="test", split_type="cross-image")
    vols += [(p, None) for p in raw if "Drosophila" in p]
    vols += [(p, "raw") for p in sorted(glob(os.path.join(D, "plantseg", "root_test", "*.h5")))]
    vols += [(p, "raw") for p in sorted(glob(os.path.join(D, "plantseg", "ovules_test", "*.h5")))]
    for paths in info["embedseg_test"].values():
        vols += [(p, None) for p in paths]
    return vols


def leakage_checks(units, info, workers):
    report = {}
    with ProcessPoolExecutor(workers) as pool:
        # 2D
        own = []
        for u in units:
            if u["ndim"] != 2:
                continue
            key = "raw/rgb" if u["source"] == "tissuenet" else None
            own += [(u["name"], u["split"], (p, key)) for p in u["raw_paths"]]
        own_hashes = list(pool.map(hash_image, [o[2] for o in own], chunksize=8))
        test_items = test_sets_2d()
        test_hashes = {name: list(pool.map(hash_image, items, chunksize=8)) for name, items in test_items.items()}
        test_exact, test_thumb = defaultdict(set), defaultdict(set)
        for name, results in test_hashes.items():
            for path, exact, thumb in results:
                for h in exact:
                    test_exact[h].add(name)
                for h in thumb:
                    test_thumb[h].add(name)
        hits_exact, hits_thumb = [], []
        split_exact = defaultdict(set)
        for (name, split, _), (path, exact, thumb) in zip(own, own_hashes):
            for h in exact:
                split_exact[h].add(split)
                if h in test_exact:
                    hits_exact.append((path, sorted(test_exact[h])))
            for h in thumb:
                if h in test_thumb:
                    hits_thumb.append((path, sorted(test_thumb[h])))
        report["2d"] = {
            "own_images": len(own), "test_images": {k: len(v) for k, v in test_items.items()},
            "exact_hits": hits_exact, "thumbnail_hits": hits_thumb,
            "train_val_shared_planes": sum(1 for s in split_exact.values() if len(s) > 1),
        }

        # 3D
        own3d = []
        for u in units:
            if u["ndim"] != 3:
                continue
            key = "raw" if u["source"] == "plantseg_root" else None
            rois = u["rois"] or [None] * len(u["raw_paths"])
            own3d += [(u["name"], u["split"], (p, key, roi)) for p, roi in zip(u["raw_paths"], rois)]
        own3d_hashes = list(pool.map(hash_volume, [o[2] for o in own3d]))
        test_vols = test_volumes_3d(info)
        test3d = list(pool.map(hash_volume, [(p, k, None) for p, k in test_vols]))
    test_slice = defaultdict(set)
    for path, _, hashes, _ in test3d:
        for h in hashes:
            test_slice[h].add(path)
    hits3d = []
    for (name, split, _), (path, zrange, hashes, shape) in zip(own3d, own3d_hashes):
        shared = {t for h in hashes for t in test_slice.get(h, ())}
        if shared:
            hits3d.append({"unit": name, "split": split, "path": path, "zrange": zrange, "test_volumes": sorted(shared),
                           "shared_slices": sum(1 for h in hashes if h in test_slice)})
    report["3d"] = {"own_volumes": len(own3d), "test_volumes": [p for p, _ in test_vols], "hits": hits3d}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=lib.MANIFEST_PATH)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.time()
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    units_2d, info = build_2d_units(out_dir, rng)
    print(f"2D units built ({time.time() - started:.0f} s)", flush=True)
    units_3d = build_3d_units(info)
    units = units_2d + units_3d
    report = leakage_checks(units, info, args.workers)
    print(f"leakage checks done ({time.time() - started:.0f} s)", flush=True)

    # Drop train / validation volumes that share slices with a test volume.
    dropped = []
    for hit in report["3d"]["hits"]:
        for u in units:
            if u["name"] == hit["unit"] and u["split"] == hit["split"] and hit["path"] in u["raw_paths"]:
                i = u["raw_paths"].index(hit["path"])
                for k in ("raw_paths", "label_paths"):
                    u[k].pop(i)
                if u["rois"] is not None:
                    u["rois"].pop(i)
                dropped.append(hit)
    report["3d"]["dropped"] = dropped
    if report["2d"]["exact_hits"] or report["2d"]["train_val_shared_planes"]:
        raise RuntimeError(f"2D leakage: {json.dumps(report['2d'], indent=1)[:4000]}")

    manifest = {
        "version": 1, "seed": SEED, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patch_shape": list(lib.PATCH_SHAPE), "z_3d": lib.Z_3D, "units": units, "info": info, "leakage": report,
    }
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=1)
    with open(args.output, "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    print(f"wrote {args.output} (md5 {checksum}) in {time.time() - started:.0f} s")
    for u in units:
        print(f"  {u['split']:5s} {u['name']:28s} files {len(u['raw_paths']):4d} samples {u['n_samples']}")
    print("2D thumbnail hits (near duplicates, review):", len(report["2d"]["thumbnail_hits"]))
    for hit in report["2d"]["thumbnail_hits"][:20]:
        print("   ", hit)
    print("3D volumes dropped:", json.dumps(dropped, indent=1))


if __name__ == "__main__":
    main()
