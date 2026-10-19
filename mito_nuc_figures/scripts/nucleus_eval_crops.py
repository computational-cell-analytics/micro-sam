"""Place the fixed evaluation crops of the EM nucleus panel and add them to eval_crops_3d.json.

The crops follow the rule of the file. They are 32 x 512 x 512 tiles, capped by the test region, that tile the test
region without overlap. The annotation covers the test region. The script drops these tiles:
- tiles thinner than 16 slices or 256 px in plane.
- tiles with a constant raw plane.
- tiles whose labels span fewer than 16 slices, because the evaluation trims a crop to its labeled slices.
- tiles with fewer than 20% of the median object count of the dataset.

Of the remaining tiles, the script keeps at most 16 per volume, the ones with the most objects. Objects are the
connected components of at least the ground-truth size floor, as the evaluation counts them. Tiles are read one at a
time, so no volume is loaded whole.
"""
import os
import sys
import json
import argparse

import numpy as np
from skimage.measure import label as connected_components

from elf.io import open_file

EVALUATION_ROOT = "/mnt/vast-kisski/home/archit/u28048/micro-sam/finetuning/v2/evaluation"
sys.path.insert(0, EVALUATION_ROOT)
from common import DATA_ROOT, DATASETS_3D_EM_NUCLEUS, GT_MIN_SIZE_3D, em_roi, get_data_paths  # noqa

CROPS_PATH = os.path.join(EVALUATION_ROOT, "eval_crops_3d.json")
OBJECT_NAMES = {
    "parlakgul_liver": "mitochondria", "cerebellum_fibsem_mito": "mitochondria",
    "macrophage_a431_mito": "mitochondria", "mito_anf": "mitochondria", "snemi_mito": "mitochondria",
    "donglai_wei_mito": "mitochondria",
}
CROP_SHAPE = (32, 512, 512)
MIN_TILE_SHAPE = (16, 256, 256)
MAX_TILES_PER_VOLUME = 16
MIN_RELATIVE_COUNT = 0.2
NOTES = {
    "cellmap_nucleus": "Nucleus panel: the held-out jrc_mus-liver-zon-2 crop357 (32 nm, manual instances).",
    "fafb_nuclei": "Nucleus panel: the four official FAFB test blocks (automatic FlyWire nucleus segmentation).",
    "microns_nuclei": "Nucleus panel: the four official MICrONS test blocks (automatic minnie65 nucleus segmentation).",
    "platynereis_nuclei": "Nucleus panel: Platynereis volume 03 inside its annotated box (manual instances).",
    "parlakgul_liver": (
        "Mitochondria panel (out-of-domain): the eight Parlakgul liver boxes, scored below the tuning slab (the first "
        "13 slices); semantic labels made instances by connected components."
    ),
    "cerebellum_fibsem_mito": (
        "Mitochondria panel (out-of-domain): the Cerebellum FIB-SEM volume below the tuning slab (the first 106 "
        "slices); student-labeled mitochondria instances, not necessarily proofread."
    ),
    "donglai_wei_mito": (
        "Mitochondria panel (out-of-domain): internal data from Donglai Wei (not public), the MICrONS phase 1 volume "
        "below the tuning slab (the first 150 slices). Only a fraction of the mitochondria is annotated, so unlabeled "
        "mitochondria in the crops count as false positives."
    ),
    "snemi_mito": (
        "Mitochondria panel (in-domain): the test split of the SNEMI Mito stack (z 70-99); the train and valid splits "
        "are training data, the valid split tunes. Manual semantic labels, made instances by connected components."
    ),
    "macrophage_a431_mito": (
        "Mitochondria panel (out-of-domain): the Macrophage / A431 fib_control block (z 848-975) below the tuning slab "
        "(the first 26 slices); quick semantic MIB mitochondria labels, made instances by connected components."
    ),
    "mito_anf": (
        "Mitochondria panel (out-of-domain): the two MitoANF boxes below their tuning slabs (the first 26 slices); "
        "only the mitochondria of traced auditory nerve fibers are labeled (about half of the visible ones)."
    ),
    "human_liver_em_nucleus": (
        "Nucleus panel (out-of-domain): three 4x downsampled human liver boxes (automated segmentation, every visible "
        "nucleus labeled)."
    ),
    "etchacell_nucleus": (
        "Nucleus panel (out-of-domain): the scored Etch a Cell ROIs (HeLa SBF-SEM, 4x downsampled in-plane, one "
        "nucleus per ROI, filled from the consensus nuclear envelope labels)."
    ),
}


def dump(obj, level=0):
    """JSON in the layout of eval_crops_3d.json: one space per level, number lists on one line."""
    pad, inner = " " * level, " " * (level + 1)
    if isinstance(obj, dict):
        items = [f"{inner}{json.dumps(k)}: {dump(v, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(not isinstance(x, (list, dict)) for x in obj):
            return json.dumps(obj)
        return "[\n" + ",\n".join(inner + dump(x, level + 1) for x in obj) + "\n" + pad + "]"
    return json.dumps(obj)


def tile_starts(region):
    axes = []
    for sl, size, min_size in zip(region, CROP_SHAPE, MIN_TILE_SHAPE):
        axes.append([s for s in range(sl.start, sl.stop, size) if min(size, sl.stop - s) >= min_size])
    return [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]


def count_tile(raw, labels, start, region, min_size):
    window = tuple(slice(s, min(s + c, r.stop)) for s, c, r in zip(start, CROP_SHAPE, region))
    planes = np.asarray(raw[window]).reshape(window[0].stop - window[0].start, -1)
    if (planes.min(axis=1) == planes.max(axis=1)).any():
        return None
    tile = np.asarray(labels[window]).astype("int64")
    tile[tile < 0] = 0
    labeled = np.nonzero(tile.reshape(tile.shape[0], -1).any(axis=1))[0]
    if len(labeled) == 0 or labeled[-1] - labeled[0] + 1 < MIN_TILE_SHAPE[0]:
        return None
    components = connected_components(tile)
    sizes = np.bincount(components.ravel())[1:]
    return int((sizes >= min_size).sum())


def place_crops(dataset_name):
    raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, DATA_ROOT, split="test")
    min_size = GT_MIN_SIZE_3D.get(dataset_name, 0)
    counts = {}
    for raw_path, label_path in zip(raw_paths, label_paths):
        raw, labels = open_file(raw_path, "r")[raw_key], open_file(label_path, "r")[label_key]
        roi = em_roi(dataset_name, label_path, "test")
        roi = tuple(slice(None) for _ in labels.shape) if roi is None else roi
        region = tuple(slice(*sl.indices(n)[:2]) for sl, n in zip(roi, labels.shape))
        counts[os.path.basename(raw_path)] = {
            start: count_tile(raw, labels, start, region, min_size)
            for start in tile_starts(region)
        }
    valid = [n for tiles in counts.values() for n in tiles.values() if n is not None]
    threshold = MIN_RELATIVE_COUNT * float(np.median(valid))
    starts, kept, n_tiles = {}, [], 0
    for volume, tiles in counts.items():
        n_tiles += len(tiles)
        passing = sorted(
            ((n, start) for start, n in tiles.items() if n is not None and n >= threshold), key=lambda t: -t[0]
        )[:MAX_TILES_PER_VOLUME]
        starts[volume] = [list(start) for _, start in sorted(passing, key=lambda t: t[1])]
        kept.extend(n for n, _ in passing)
    note = (
        f"{NOTES[dataset_name]} {len(kept)} of {n_tiles} tiles are kept (at most {MAX_TILES_PER_VOLUME} per volume, "
        f"the ones with the most objects), holding {min(kept)}-{max(kept)} {OBJECT_NAMES.get(dataset_name, 'nuclei')}."
    )
    print(f"{dataset_name}: {note}")
    return {"crop_shape": list(CROP_SHAPE), "note": note, "starts": starts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS_3D_EM_NUCLEUS, help="The nucleus evaluation sets.")
    parser.add_argument("--dry", action="store_true", help="Print the crops without writing eval_crops_3d.json.")
    args = parser.parse_args()
    placed = {name: place_crops(name) for name in args.datasets}
    # The file is read after the crops are placed, so edits made in the meantime are kept.
    with open(CROPS_PATH) as f:
        crops = json.load(f)
    crops["datasets"].update(placed)
    if not args.dry:
        tmp_path = f"{CROPS_PATH}.tmp{os.getpid()}"
        with open(tmp_path, "w") as f:
            f.write(dump(crops) + "\n")
        os.replace(tmp_path, CROPS_PATH)


if __name__ == "__main__":
    main()
