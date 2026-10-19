"""Overview figures for Cmito (whole-animal C. elegans mito instance masks): one subvolume per stage.

Boxes are 128 z-slices x 1024 x 1024 voxels at the native mask resolution. Each box is at the 1024 x 1024 window
with the highest mito fraction in a coarse mask level (sampled on 6 z-slices).

The l1, l2, adult and dauer stages are cached with `get_cmito_data(path, stage, bounding_box)`, so the recorded nm
boxes reuse the cache. The l3 stage is fetched by voxel index. Its mask reports 8 nm but lies on the raw mip3 grid
(6.144 nm). Thus, the nm -> voxel conversion of the loader misaligns raw and mask for every box that does not start
at 0.
"""
import os

import zarr
import numpy as np
from cloudvolume import CloudVolume

from torch_em.data.datasets.electron_microscopy import cmito

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "cmito"
DATA_PATH = os.path.join(DATA_ROOT, DATASET)
# Voxel boxes (x0, x1, y0, y1, z0, z1) at the native mask resolution.
VOXEL_BOXES = {
    "l1": (1280, 2304, 1280, 2304, 118, 246),
    "l2": (1024, 2048, 768, 1792, 365, 493),
    "l3": (1792, 2816, 2560, 3584, 150, 278),
    "adult": (2816, 3840, 1280, 2304, 286, 414),
    "dauer": (0, 1024, 0, 1024, 118, 246),
}
ISSUES = (
    "Loader bug for l3 (cmito.py:102 _cmito_bbox_voxels, used at :203-209): the nm box is converted with each "
    "volume's own resolution, raw mip3 = 6.144 nm vs mask nominal 8 nm, but the l3 mask lies on the raw mip3 grid "
    "(both 4544 x 4288 x ~430). Any box with x_min/y_min > 0 gives raw offset nm/6.144 vs mask offset nm/8, so raw "
    "and labels are misaligned; the default full box starts at 0 (aligned) but is cropped by min() to 3490 x 3294, "
    "dropping ~23% of x and y. l3 was fetched here by voxel index instead. Raw is streamed with fill_missing=True, "
    "so missing sections become black (zero) slices (visible in l1, l3 and adult ZX views). dauer license uncertain."
)
OBSERVATIONS = (
    "All 5 stages: labels aligned with raw, dense instance labels; nearly every visible mito profile is labeled "
    "(adult: a few small dark profiles unlabeled). ~4-5% mito foreground, 280-1250 ids per 128 x 1024 x 1024 box. "
    "Strongly anisotropic (30-50 nm z) with section artifacts, folds and missing slices in the ZX views; raw "
    "contrast and texture differ a lot between stages (l1 5 nm vs adult 16 nm xy)."
)
STAGE_NOTES = {
    "l1": "L1 larva", "l2": "L2 larva", "l3": "L3 larva", "adult": "adult", "dauer": "dauer (license uncertain)",
}


def get_resolution(stage):
    mask_cv = CloudVolume(cmito.CMITO_STAGES[stage]["mask_url"], mip=0, progress=False, fill_missing=True)
    raw_cv = CloudVolume(
        cmito.CMITO_STAGES[stage]["raw_url"], use_https=True, mip=cmito.CMITO_STAGES[stage]["raw_mip"],
        progress=False, fill_missing=True,
    )
    return [float(r) for r in mask_cv.resolution], [float(r) for r in raw_cv.resolution]


def voxel_box_to_nm(voxel_box, resolution):
    """Convert a voxel box to the nm box of `get_cmito_data`, nudged inwards so floor / ceil give the voxel box."""
    x0, x1, y0, y1, z0, z1 = voxel_box
    rx, ry, rz = resolution
    eps = 1e-3
    return (
        round(x0 * rx + eps, 3), round(x1 * rx - eps, 3), round(y0 * ry + eps, 3), round(y1 * ry - eps, 3),
        round(z0 * rz + eps, 3), round(z1 * rz - eps, 3),
    )


def download_l3_by_voxels(voxel_box):
    """Fetch l3 raw (mip3) and mask (mip0) with the same voxel indices, in the loader's zarr layout."""
    stage_info = cmito.CMITO_STAGES["l3"]
    zarr_path = os.path.join(DATA_PATH, f"l3_voxels_{cmito._cmito_bbox_to_str(voxel_box)}.zarr")
    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path
    x0, x1, y0, y1, z0, z1 = voxel_box
    shape = (z1 - z0, y1 - y0, x1 - x0)
    raw_cv = CloudVolume(
        stage_info["raw_url"], use_https=True, mip=stage_info["raw_mip"], progress=False, fill_missing=True
    )
    mask_cv = CloudVolume(stage_info["mask_url"], mip=0, progress=False, fill_missing=True)
    root.attrs["stage"] = "l3"
    root.attrs["voxel_box_xyz"] = list(voxel_box)
    root.attrs["raw_mip"] = stage_info["raw_mip"]
    if "raw" not in root:
        ds_raw = cmito._cmito_create_array(root, "raw", shape, np.dtype("uint8"), is_label=False)
        cmito._cmito_download_to_zarr(raw_cv, ds_raw, x0, y0, z0, name="raw")
    if "labels" not in root:
        ds_lbl = cmito._cmito_create_array(root, "labels", shape, np.dtype("uint16"), is_label=True)
        cmito._cmito_download_to_zarr(mask_cv, ds_lbl, x0, y0, z0, name="labels")
    return zarr_path


def folder_size_gb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        total += sum(os.path.getsize(os.path.join(dirpath, name)) for name in filenames)
    return total / 1e9


def render_stage(stage, out_dir):
    voxel_box = VOXEL_BOXES[stage]
    mask_resolution, raw_resolution = get_resolution(stage)
    if stage == "l3":
        bbox_nm = None
        zarr_path = download_l3_by_voxels(voxel_box)
    else:
        bbox_nm = voxel_box_to_nm(voxel_box, mask_resolution)
        zarr_path = cmito.get_cmito_data(DATA_PATH, stage, bounding_box=bbox_nm, download=True)
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["raw"], root["labels"]
    voxel_size = raw_resolution if stage == "l3" else mask_resolution
    voxel_text = "x".join(f"{v:g}" for v in voxel_size)
    notes = f"ssTEM C. elegans {STAGE_NOTES[stage]}, {voxel_text} nm (xyz), dense instance mito (Cmito pipeline)"
    out_path = os.path.join(out_dir, f"{stage}.png")
    save_volume_figure(raw, {"mito": labels}, out_path, DATASET, stage, notes=notes)
    label_data = labels[:]
    ids = np.unique(label_data)
    return {
        "name": stage, "shape": raw.shape, "voxel_size_nm_xyz": voxel_size,
        "mask_reported_resolution_nm_xyz": mask_resolution, "raw_mip": cmito.CMITO_STAGES[stage]["raw_mip"],
        "voxel_box_xyz": list(voxel_box), "bounding_box_nm": list(bbox_nm) if bbox_nm else None, "path": zarr_path,
        "n_objects": int((ids != 0).sum()), "foreground_fraction": float((label_data > 0).mean()),
    }


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_stage(stage, out_dir) for stage in VOXEL_BOXES]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "serial-section TEM (whole-animal connectome volumes)", "dimensionality": "3d",
        "label_quality": "dense instance masks over the whole animal (Cmito pipeline; proofreading level not verified)",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_PATH,
        "download": f"one 128 x 1024 x 1024 box per stage, {folder_size_gb(DATA_PATH):.2f} GB on disk",
        "notes": "bounding_box_nm is the (x_min, x_max, y_min, y_max, z_min, z_max) argument of get_cmito_data; "
        "voxel_box_xyz is the same box in mask voxels; l3 was fetched by voxels (see issues)",
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
