"""Overview figures for the MitoNet-predicted mouse kidney (jrc_mus-kidney, 16 nm): a few representative boxes.

The boxes are 128 x 1024 x 1024 voxels. Each box is inside one z-row of the 512^3 label chunks and spans 2 x 2
chunks in yx. Thus, 4 label chunks are decoded per box. The boxes sit at the 2 x 2 chunk groups with the largest
compressed label size at three different depths of the volume. The compressed label size is a proxy for mito density.
They are cached with `get_mitonet_predicted_kidney_data`, so they can be passed to the dataset function directly.
"""
import os

import zarr
import fsspec
import numpy as np

from torch_em.data.datasets.electron_microscopy import mitonet_predicted_kidney as kidney

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "mitonet_predicted_kidney"
DATA_PATH = os.path.join(DATA_ROOT, DATASET)
# (z_min, z_max, y_min, y_max, x_min, x_max) at 16 nm isotropic.
BOXES = [
    (2240, 2368, 2560, 3584, 3584, 4608),
    (4800, 4928, 2048, 3072, 4096, 5120),
    (10432, 10560, 1024, 2048, 512, 1536),
]

ISSUES = (
    "Loader fails on proxy-only compute nodes: mitonet_predicted_kidney.py:64 uses fsspec.filesystem('http') "
    "(aiohttp ignores http(s)_proxy -> connection timeout); fixed in this script with trust_env client_kwargs. "
    "Each box decodes whole 512^3 uint32 label chunks (512 MB each in RAM), so boxes should align to the chunk grid."
)
OBSERVATIONS = (
    "Aligned, dense instance pseudo-labels of very high apparent quality: essentially every mito in the "
    "proximal-tubule-like cells is labeled with clean boundaries; ~35-42% mito foreground, 270-420 ids per box. "
    "Tightly packed elongated mitos, a few touching ones possibly merged. Not proofread."
)


def enable_proxy():
    """Let the aiohttp-based fsspec HTTP filesystem use the http(s)_proxy environment variables."""
    for protocol in ("http", "https"):
        fsspec.config.conf[protocol] = {"client_kwargs": {"trust_env": True}}


def folder_size_gb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        total += sum(os.path.getsize(os.path.join(dirpath, name)) for name in filenames)
    return total / 1e9


def render_box(box, out_dir):
    zarr_path = kidney.get_mitonet_predicted_kidney_data(DATA_PATH, box, download=True)
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["raw"], root["labels"]
    name = f"z{box[0]}-{box[1]}_y{box[2]}-{box[3]}_x{box[4]}-{box[5]}"
    notes = "FIB-SEM mouse kidney (jrc_mus-kidney), 16 nm isotropic, MitoNet instance pseudo-labels (not proofread)"
    save_volume_figure(raw, {"mito": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    label_data = labels[:]
    ids = np.unique(label_data)
    return {
        "name": name, "shape": raw.shape, "voxel_size_nm": (16, 16, 16), "bounding_box_zyx": list(box),
        "full_volume_shape_zyx": kidney.FULL_SHAPE, "path": zarr_path,
        "n_objects": int((ids != 0).sum()), "foreground_fraction": float((label_data > 0).mean()),
    }


def main():
    enable_proxy()
    out_dir = get_output_dir(DATASET)
    volumes = [render_box(box, out_dir) for box in BOXES]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "FIB-SEM", "dimensionality": "3d",
        "label_quality": "automatic MitoNet predictions (pseudo-labels), instance, dense over the volume",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_PATH,
        "download": f"{len(volumes)} boxes of 128 x 1024 x 1024, {folder_size_gb(DATA_PATH):.2f} GB on disk",
        "notes": "raw from OpenOrganelle s1 (16 nm), labels decoded chunk-wise from the 19 GB figshare zip",
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
