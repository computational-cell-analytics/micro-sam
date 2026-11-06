"""Place the 512 x 512 test crops of the 2d EM mitochondria datasets and add them to eval_crops.json.

The crops cover each test image with the fewest 512 x 512 tiles. The tiles are spread evenly, so neighbors overlap by
a few pixels. An image axis below 512 pixels gives one tile of the axis length. These tiles are dropped:
- tiles with a constant raw image.
- tiles with fewer than 20% of the median object count of the dataset.

Of the remaining tiles, at most 16 per image stay, the ones with the most objects. Objects are the connected components
of at least the ground-truth size floor, as the evaluation counts them.

Usage:
    python place_eval_crops_2d.py  # all datasets of DATASETS
    python place_eval_crops_2d.py --datasets perez_mito --dry  # print the crops, do not write them
"""

import os
import json
import argparse

import numpy as np
from skimage.measure import label

import common

DATASETS = ["mitolab_tem", "probtem", "tnbc_mito", "perez_mito", "cardiomyocyte_tem_mito"]
CROPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_crops.json")
TILE = 512
MIN_RELATIVE_COUNT = 0.2
MAX_TILES_PER_IMAGE = 16


def dump_crops(obj, level=0):
    """Return the JSON in the layout of eval_crops.json: one space per level, number lists on one line."""
    pad, inner = " " * level, " " * (level + 1)
    if isinstance(obj, dict):
        items = [f"{inner}{json.dumps(k)}: {dump_crops(v, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(not isinstance(x, (list, dict)) for x in obj):
            return json.dumps(obj)
        return "[\n" + ",\n".join(inner + dump_crops(x, level + 1) for x in obj) + "\n" + pad + "]"
    return json.dumps(obj)


def cover_starts(n):
    """Return the tile starts that cover an axis of length n with the fewest tiles, spread evenly, and the tile size."""
    if n <= TILE:
        return [0], n
    k = int(np.ceil(n / TILE))
    return [int(round(i * (n - TILE) / (k - 1))) for i in range(k)], TILE


def count_objects(raw, labels, start, shape, min_size):
    """Return the number of objects in a tile, or None for a tile with a constant raw image."""
    window = tuple(slice(s, s + t) for s, t in zip(start, shape))
    patch = raw[window]
    if patch.min() == patch.max():
        return None
    gt = common.drop_severed_objects(label(labels[window]).astype("uint32"), min_size)
    return int(len(np.unique(gt)) - 1)


def place_crops(dataset_name):
    """Return the eval_crops.json entry of a dataset, with the crop shape of each image in 'crop_shapes'."""
    raw_paths, label_paths, raw_key, label_key = common.get_data_paths(dataset_name, common.DATA_ROOT, split="test")[:4]
    min_size = common.GT_MIN_SIZE_2D.get(dataset_name, 0)
    counts, shapes = {}, {}
    for raw_path, label_path in zip(raw_paths, label_paths):
        raw, labels = common.read_2d(raw_path, raw_key), common.read_2d(label_path, label_key)
        if raw.ndim == 3:
            raw = raw.mean(axis=-1)
        (ys, height), (xs, width) = cover_starts(raw.shape[0]), cover_starts(raw.shape[1])
        name = os.path.basename(raw_path)
        shapes[name] = [height, width]
        counts[name] = {(y, x): count_objects(raw, labels, (y, x), (height, width), min_size) for y in ys for x in xs}
    valid = [n for tiles in counts.values() for n in tiles.values() if n]
    threshold = MIN_RELATIVE_COUNT * float(np.median(valid))
    starts, crop_shapes, kept, n_tiles = {}, {}, [], 0
    for name, tiles in counts.items():
        n_tiles += len(tiles)
        passing = sorted(((n, s) for s, n in tiles.items() if n and n >= threshold), key=lambda t: -t[0])
        passing = passing[:MAX_TILES_PER_IMAGE]
        if not passing:
            continue
        starts[name] = [list(start) for _, start in sorted(passing, key=lambda t: t[1])]
        if shapes[name] != [TILE, TILE]:
            crop_shapes[name] = [shapes[name]] * len(passing)
        kept.extend(n for n, _ in passing)
    note = (
        "512 x 512 tiles that cover each test image, spread evenly with a small overlap; an image axis below 512 px "
        f"is one tile. {len(kept)} of {n_tiles} tiles are kept (at most {MAX_TILES_PER_IMAGE} per image, the ones "
        f"with the most objects), holding {min(kept)}-{max(kept)} mitochondria."
    )
    print(f"{dataset_name}: {note}")
    entry = {"crop_shape": [TILE, TILE], "note": note, "starts": starts}
    if crop_shapes:
        entry["crop_shapes"] = crop_shapes
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS, help="The datasets to place.")
    parser.add_argument("--dry", action="store_true", help="Print the crops and do not write eval_crops.json.")
    args = parser.parse_args()
    placed = {name: place_crops(name) for name in args.datasets}
    if args.dry:
        return
    # The file is read after the crops are placed, so that edits made in the meantime are kept.
    with open(CROPS_PATH) as f:
        crops = json.load(f)
    crops["datasets"].update(placed)
    tmp_path = f"{CROPS_PATH}.tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.write(dump_crops(crops) + "\n")
    os.replace(tmp_path, CROPS_PATH)
    print(f"Saved the crops of {len(placed)} datasets to {CROPS_PATH}.")


if __name__ == "__main__":
    main()
