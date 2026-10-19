"""Overview figures for the ASEM mitochondria labels (FIB-SEM cells, s3://asem-project).

The torch-em loader cannot download the data anymore:
- quilt3 is not installed in the env.
- The bucket no longer has the `datasets/cell_<i>/cell_<i>.zarr` layout that the loader expects.
  The cells are now stored under ids like `46/46.zarr`.

This script streams the mito label crops directly from the public bucket with s3fs.
It caches them as zarr v3 stores (raw + labels) under `<DATA_ROOT>/asem/mito_crops/`.
It does not touch the `cell_<i>` layout of the loader.

Volumes with mito labels in the current bucket:
- 46, 58, 61 (5 nm, chemical fixation): the labels have the full raw shape and cover large regions.
  We cache one 256 x 1024 x 1024 crop at the densest labeled window (likely cell_1 / cell_2 / cell_3).
- 64 (5 nm), 3A and 2E (4 nm, HPFS): the labels are small crops with an nm offset into the raw volume.
  They are likely cell_6, cell_13a and cell_13, the 'inconsistent' volumes of the loader.
  We cache the full label crop.
"""
import os
import json

import s3fs
import fsspec
import numpy as np

import zarr
from zarr.codecs import BloscCodec

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "asem"
DATA_PATH = os.path.join(DATA_ROOT, DATASET, "mito_crops")
BUCKET = "asem-project/datasets"
# Crops as (z_min, z_max, y_min, y_max, x_min, x_max) in raw voxels; None means the full label crop.
CROPS = {
    "46": (640, 896, 0, 1024, 1112, 2136),
    "58": (152, 408, 0, 1000, 1480, 2504),
    "61": (16, 272, 0, 1024, 2528, 3552),
    "64": None,
    "3A": None,
    "2E": None,
}
LIKELY_CELL = {
    "46": "cell_1/2/3", "58": "cell_1/2/3", "61": "cell_1/2/3", "64": "cell_6", "3A": "cell_13a", "2E": "cell_13",
}
PREPARATION = {"46": "CF", "58": "CF", "61": "CF", "64": "CF", "3A": "HPFS", "2E": "HPFS"}

ISSUES = (
    "Loader broken: (1) quilt3 is not installed in the super env (asem.py:20), and asem.py:72 calls q3.Bucket even "
    "with download=False, so it raises NameError without quilt3 even if data is local; (2) the bucket layout changed: "
    "s3://asem-project/datasets/ now holds ids like 46/46.zarr, 58/58.zarr, 2E/2E.zarr, there is no cell_<i>/ "
    "folder, so the VOLUMES paths (asem.py:39-47) do not exist. Mapping to the old cell names is not documented; "
    "46/58/61 are the three CF 5 nm volumes with full-size mito labels (cell_1/2/3), 64 matches cell_6 (cropped "
    "labels, CF 5 nm), 3A has mito + np + np_bottom (cell_13a), 2E has ccp + er + mito (cell_13). Crops were "
    "streamed with s3fs instead."
)
OBSERVATIONS = (
    "All aligned. Labels are semantic (a single foreground value; 46 uses the value 2216232741, 2E/3A use 255). "
    "46, 58, 64, 2E: exhaustive within the crop. 61: only part of the visible mitos are labeled (roughly half). "
    "3A: only 2 mitos labeled in a 753 x 454 x 458 crop, most of it unlabeled. 58 and 61 show z-jumps / "
    "intensity steps in the ZX view; raw contrast varies between slices. Labels outside the annotated regions "
    "of 46/58/61 are 0, so unannotated parts of the full volumes cannot be told apart from background."
)


def open_remote(fs, name, key):
    path = f"{BUCKET}/{name}/{name}.zarr/volumes/{key}"
    attrs = json.loads(fs.cat(f"{path}/.zattrs"))
    array = zarr.open_array(fsspec.get_mapper(f"s3://{path}", anon=True), mode="r", zarr_format=2)
    return array, attrs


def get_crop_box(fs, name):
    """Get the crop in raw voxels, from CROPS or from the label offset relative to the raw offset."""
    labels, label_attrs = open_remote(fs, name, "labels/mito")
    raw, raw_attrs = open_remote(fs, name, "raw")
    resolution = np.array(raw_attrs["resolution"])
    assert (np.array(label_attrs["resolution"]) == resolution).all()
    offset = (np.array(label_attrs["offset"]) - np.array(raw_attrs["offset"])) // resolution
    if CROPS[name] is not None:
        assert (offset == 0).all() and labels.shape == raw.shape, (offset, labels.shape, raw.shape)
        return CROPS[name], (0, 0, 0), resolution.tolist(), raw.shape
    box = tuple(int(v) for o, s in zip(offset, labels.shape) for v in (o, o + s))
    return box, tuple(int(o) for o in offset), resolution.tolist(), raw.shape


def download_crop(fs, name):
    box, label_offset, resolution, raw_shape = get_crop_box(fs, name)
    z0, z1, y0, y1, x0, x1 = box
    zarr_path = os.path.join(DATA_PATH, f"{name}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}.zarr")
    root = zarr.open_group(zarr_path, mode="a")
    if "raw" not in root or "labels" not in root:
        raw, _ = open_remote(fs, name, "raw")
        labels, _ = open_remote(fs, name, "labels/mito")
        raw_data = np.asarray(raw[z0:z1, y0:y1, x0:x1])
        lz, ly, lx = (z0 - label_offset[0], y0 - label_offset[1], x0 - label_offset[2])
        label_data = np.asarray(labels[lz:lz + z1 - z0, ly:ly + y1 - y0, lx:lx + x1 - x0])
        assert raw_data.shape == label_data.shape, (raw_data.shape, label_data.shape)
        for key, data, shuffle in (("raw", raw_data, "shuffle"), ("labels", label_data, "bitshuffle")):
            array = root.create_array(
                key, shape=data.shape, chunks=(64, 256, 256), dtype=data.dtype,
                compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
            )
            array[:] = data
        root.attrs["source"] = f"s3://{BUCKET}/{name}/{name}.zarr/volumes/(raw, labels/mito)"
        root.attrs["crop_zyx_raw_voxels"] = list(box)
        root.attrs["resolution_nm"] = resolution
    return zarr_path, box, resolution, raw_shape


def folder_size_gb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        total += sum(os.path.getsize(os.path.join(dirpath, name)) for name in filenames)
    return total / 1e9


def render_volume(fs, name, out_dir):
    zarr_path, box, resolution, raw_shape = download_crop(fs, name)
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["raw"], root["labels"]
    notes = f"FIB-SEM cell ({PREPARATION[name]}), {resolution[0]} nm isotropic, manual semantic mito labels"
    save_volume_figure(raw, {"mito": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    label_data = labels[:]
    return {
        "name": name, "likely_loader_volume": LIKELY_CELL[name], "preparation": PREPARATION[name],
        "shape": raw.shape, "voxel_size_nm": resolution, "crop_zyx": list(box), "full_raw_shape_zyx": raw_shape,
        "path": zarr_path, "label_values": [int(v) for v in np.unique(label_data)],
        "foreground_fraction": float((label_data > 0).mean()),
    }


def main():
    fs = s3fs.S3FileSystem(anon=True)
    out_dir = get_output_dir(DATASET)
    volumes = [render_volume(fs, name, out_dir) for name in CROPS]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "FIB-SEM", "dimensionality": "3d",
        "label_quality": "manual, semantic (one foreground value), annotated regions only",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_PATH,
        "download": f"one crop per mito-labeled volume, {folder_size_gb(DATA_PATH):.2f} GB on disk",
        "notes": "crop_zyx is in raw voxels of s3://asem-project/datasets/<name>/<name>.zarr/volumes/raw",
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
