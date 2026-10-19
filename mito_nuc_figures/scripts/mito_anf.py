"""Overview figures for Mito-ANFs (mouse cochlea SBEM, webKnossos): one labeled subvolume per sample.

The labels are very sparse (only mitos of traced auditory nerve fibers). Thus, each box is the window with the most
labeled voxels in the 32-32-4 mag of the segmentation layer. The box size is 1024 x 1024 x 128 voxels (native
12 x 12 x 50 nm).
The boxes are cached with `get_mito_anf_data`, so they can be passed to `get_mito_anf_dataset` directly.
"""
import os

import zarr
import fsspec
import numpy as np

from torch_em.data.datasets.electron_microscopy import mito_anf

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "mito_anf"
DATA_PATH = os.path.join(DATA_ROOT, DATASET)
# (z_min, z_max, y_min, y_max, x_min, x_max) at native resolution.
BOXES = {
    "M1": [(452, 580, 7776, 8800, 4064, 5088)],
    "M2": [(1404, 1532, 8896, 9920, 5536, 6560)],
}

ISSUES = (
    "Loader fails on proxy-only compute nodes: mito_anf.py:52 opens the webKnossos zarr with "
    "fsspec.get_mapper(url), whose aiohttp client ignores http(s)_proxy (ConnectionTimeoutError). Fixed in this "
    "script with fsspec.config.conf['https'] = {'client_kwargs': {'trust_env': True}}. Remote chunks are 32^3 and "
    "uncompressed, so a 128 x 1024 x 1024 box needs ~8k requests (still < 1 min here)."
)
OBSERVATIONS = (
    "Aligned. Labels are very sparse: only mitos inside traced auditory nerve fibers; many visible mitos in other "
    "processes are unlabeled (roughly half of the visible mitos in the box), so unlabeled != background. Raw is "
    "very noisy SBEM at 12 nm, 50 nm z. ~210-220 ids and ~3% foreground in the densest boxes; the labeled region is a "
    "small part of each volume."
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


def render_box(sample, box, out_dir):
    zarr_path = mito_anf.get_mito_anf_data(DATA_PATH, box, sample=sample, download=True)
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["raw"], root["labels"]
    name = f"{sample}_z{box[0]}-{box[1]}"
    notes = "SBEM mouse cochlea (inner spiral bundle), 12x12x50 nm (xyz), sparse proofread instance mito (ANF only)"
    save_volume_figure(raw, {"mito": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    label_data = labels[:]
    ids = np.unique(label_data)
    return {
        "name": name, "sample": sample, "shape": raw.shape, "voxel_size_nm_zyx": (50, 12, 12),
        "bounding_box_zyx": list(box), "full_volume_shape_zyx": mito_anf.SAMPLES[sample]["shape_zyx"],
        "seg_layer": mito_anf.SAMPLES[sample]["seg_layer"], "path": zarr_path,
        "n_objects": int((ids != 0).sum()), "foreground_fraction": float((label_data > 0).mean()),
    }


def main():
    enable_proxy()
    out_dir = get_output_dir(DATASET)
    volumes = [render_box(sample, box, out_dir) for sample, boxes in BOXES.items() for box in boxes]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "SBEM (serial block-face SEM)", "dimensionality": "3d",
        "label_quality": "3D U-Net prediction + manual proofreading, instance, sparse (only traced ANF mitos)",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_PATH,
        "download": f"one 128 x 1024 x 1024 box per sample, {folder_size_gb(DATA_PATH):.2f} GB on disk",
        "notes": "the whole labeled region is small: ~190 (M1) / ~220 (M2) ids at the 32-32-4 mag, in M1 x 3168-5984, "
        "y 6688-9792, z 240-864 and M2 x 4736-6944, y 7104-10848, z 1064-1804 (native voxels)",
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
