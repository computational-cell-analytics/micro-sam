"""Cache the CellMap nucleus crops for the nucleus EM generalist.

Only the large tissue crops (32-64 nm, many complete nuclei) are used. The small high-resolution crops are skipped:
most of them hold one nucleus cut by the crop borders, and 15 lie fully inside a nucleus. jrc_mus-liver-zon-2 crop357
is held out for the in-domain evaluation: it does not overlap any other crop (crop358 and crop367 overlap each other,
so neither of them can be held out on its own).

Each crop is written to one zarr store with 'raw' (uint8) and 'labels' (the nucleus instance ids, uint32) at the
label scale, under a temporary name that is moved in place when complete.
"""
import argparse
import json
import os
import shutil
import zipfile

import numpy as np
import zarr

from cellmap import ZIP_PATH, ZIP_ROOT, get_levels, load_crop, read_json
from figure_utils import DATA_ROOT


TRAIN_CROPS = {
    "jrc_mus-liver-zon-1": [289, 336, 337, 349, 351],
    "jrc_mus-liver-zon-2": [358, 367],
    "jrc_mus-kidney": [179],
    "jrc_mus-kidney-3": [472],
    "jrc_zf-cardiac-1": [378, 379],
    "jrc_mus-heart-1": [423, 452],
}
TEST_CROPS = {"jrc_mus-liver-zon-2": [357]}
CHUNKS = (64, 64, 64)


def write_crop(store, zip_file, volume, crop_id, em_levels, out_dir):
    raw, labels, info = load_crop(store, zip_file, volume, crop_id, em_levels, class_map={"nuc": "nucleus"})
    labels = labels["nucleus"].astype("uint32")
    assert raw.shape == labels.shape, f"{raw.shape} != {labels.shape}"
    name = f"{volume}_crop{crop_id}"
    out_path = os.path.join(out_dir, f"{name}.zarr")
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    group = zarr.open_group(tmp_path, mode="w")
    group.create_array("raw", data=raw, chunks=CHUNKS)
    group.create_array("labels", data=labels, chunks=CHUNKS)
    group.attrs.update({
        "volume": volume, "crop": crop_id, "voxel_size_nm": info["voxel_size_nm"],
        "translation_nm": info["translation_nm"], "em_level": info["em_level"],
    })
    os.replace(tmp_path, out_path)
    ids = np.unique(labels)
    entry = {
        "name": name, "path": out_path, "volume": volume, "crop": crop_id, "shape": list(raw.shape),
        "voxel_size_nm": info["voxel_size_nm"], "n_ids": int((ids != 0).sum()), "fg_frac": float((labels > 0).mean()),
        "clipped_to_em": info["clipped_to_em"], "raw_zero_frac": info["raw_zero_frac"],
        "n_unknown_voxels_mapped_to_0": info["labels"]["nucleus"]["n_unknown_voxels"],
    }
    print(entry)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(DATA_ROOT, "cellmap_nucleus_crops"), help="Output folder.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    store = zarr.storage.ZipStore(ZIP_PATH, mode="r")
    zip_file = zipfile.ZipFile(ZIP_PATH)
    index = []
    for split, crops in (("train", TRAIN_CROPS), ("test", TEST_CROPS)):
        for volume, crop_ids in crops.items():
            em_attrs = read_json(zip_file, f"{ZIP_ROOT}/{volume}/{volume}.zarr/recon-1/em/fibsem-uint8/.zattrs")
            em_levels = get_levels(em_attrs)
            for crop_id in crop_ids:
                entry = write_crop(store, zip_file, volume, crop_id, em_levels, args.output)
                index.append({**entry, "split": split})
    with open(os.path.join(args.output, "index.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
