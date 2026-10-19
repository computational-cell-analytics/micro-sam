import json
import os

import numpy as np
import requests
import zarr

from denoised_view import DenoisedView, DilatedView
from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info
from torch_em.data.datasets.electron_microscopy import drg_axon_mito as drg


DATASET = "drg_axon_mito"
DATA_PATH = os.path.join(DATA_ROOT, DATASET)
CACHE_DIR = os.path.join(DATA_PATH, "overview_cache")
# The pyramid level that brings each tomogram to ~43 A voxels, keyed by the full resolution voxel spacing.
OVERVIEW_LEVEL = {"10.800": 2, "20.579": 1, "21.600": 1}
NOTES = "cryo-ET DRG axons, {vs} A voxels ({level}), mito MEMBRANES 1=outer 2=inner (not filled, dilated for display)"
ISSUES = (
    "Loader bug, drg_axon_mito.py:42,47,49,72,81,82,85,87 (RUNS imm_scale '-2.0' for 10512 Position_2/32/42, "
    "10515 Position_50C, 10516 Position_1/12/20/27): the portal only has source_label_imm_-1.0 for these 8 runs, "
    "the '-2.0' URL returns 404, so _download_ome_zarr raises HTTPError on .zattrs and get_drg_axon_mito_data "
    "aborts at the first of them (10512 Position_2, the 5th run). Workaround here: fall back to '-1.0'. "
    "Full dataset at level 0 is ~80 GB (3 runs at 10.8 A are 500x1855x1919 float32, ~7 GB each), so only a "
    "6-run subset was downloaded with the loader; all 57 runs are rendered from a ~43 A pyramid level."
)
OBSERVATIONS = (
    "The label is mitochondrial MEMBRANES only (1 = outer, 2 = inner incl. cristae), 1-2 voxels thick, not filled "
    "mitochondria; the figures dilate it for display. Aligned in xy in all checked runs (crops at 21.6 A and a "
    "shift scan); 10512 Position_30 (10.8 A run) has a small z offset of ~10 nm (2-3 slices at 43 A), the other "
    "checked runs have none. Membranes are only traced where visible: gaps at the top/bottom of each mito (missing "
    "wedge), so even the outer membrane is not a closed surface. Per tomogram 1-5 mitochondria are labeled; in most "
    "runs all mitochondria in the axon/varicosity are labeled (e.g. 10514 Position_18), in some (10512 Position_15) "
    "nearby mito-like double-membrane organelles are unlabeled. Very low foreground (0.01-0.5% per class). "
    "Loader-merged labels.zarr (level 0) and the portal pyramid level agree visually."
)
DOWNLOAD = (
    "Loader (_download_ome_zarr + _merge_labels, same layout as get_drg_axon_mito_data) for 6 runs at level 0: "
    "10512 Position_1/15, 10513 Position_03, 10514 Position_5, 10515 Position_16B, 10516 Position_7 (4.5 GB). "
    "All 57 runs at the ~43 A pyramid level (level 1, level 2 for the 10.8 A runs) into overview_cache/ (5.1 GB)."
)
LOCAL_SUBSET = [
    (10512, "Position_1"), (10512, "Position_15"), (10513, "Position_03"), (10514, "Position_5"),
    (10515, "Position_16B"), (10516, "Position_7"),
]


def fetch_level(url, level, out_dir):
    """Download one pyramid level of a portal OME-Zarr into a local zarr v2 array (missing chunks are fill value)."""
    array_dir = os.path.join(out_dir, str(level))
    if not os.path.exists(os.path.join(array_dir, ".done")):
        os.makedirs(array_dir, exist_ok=True)
        drg._fetch(f"{url}/{level}/.zarray", os.path.join(array_dir, ".zarray"))
        with open(os.path.join(array_dir, ".zarray")) as f:
            meta = json.load(f)
        grid = [-(-size // chunk) for size, chunk in zip(meta["shape"], meta["chunks"])]
        for z in range(grid[0]):
            for y in range(grid[1]):
                chunk_dir = os.path.join(array_dir, str(z), str(y))
                os.makedirs(chunk_dir, exist_ok=True)
                for x in range(grid[2]):
                    drg._fetch(f"{url}/{level}/{z}/{y}/{x}", os.path.join(chunk_dir, str(x)), optional=True)
        open(os.path.join(array_dir, ".done"), "w").close()
    return zarr.open_array(array_dir, mode="r", zarr_format=2)


def label_url(template, dataset, run, voxel_spacing, scale):
    """Get the annotation url, falling back to the '-1.0' scale suffix if the loader's suffix does not exist."""
    kwargs = {"dataset": dataset, "run": run, "voxel_spacing": voxel_spacing}
    urls = [template.format(omm_scale=s, imm_scale=s, **kwargs) for s in (scale, "-1.0")]
    for url in urls:
        if requests.get(f"{url}/.zattrs", timeout=60).status_code == 200:
            return url, url == urls[0]
    raise RuntimeError(f"No annotation found for {dataset} {run}")


def merge(outer, inner):
    labels = np.zeros(outer.shape, dtype="uint8")
    labels[outer > 0] = 1
    overlap = int(((outer > 0) & (inner > 0)).sum())
    labels[inner > 0] = 2
    return labels, overlap


def render_overview(dataset, run, voxel_spacing, omm_scale, imm_scale, out_dir):
    level = OVERVIEW_LEVEL[voxel_spacing]
    cache = os.path.join(CACHE_DIR, f"{dataset}_{run}")
    raw_url = drg.RAW_URL.format(dataset=dataset, run=run, voxel_spacing=voxel_spacing)
    omm_url, omm_ok = label_url(drg.OMM_URL, dataset, run, voxel_spacing, omm_scale)
    imm_url, imm_ok = label_url(drg.IMM_URL, dataset, run, voxel_spacing, imm_scale)
    raw = fetch_level(raw_url, level, os.path.join(cache, "raw.zarr"))[:]
    outer = fetch_level(omm_url, level, os.path.join(cache, "omm.zarr"))[:]
    inner = fetch_level(imm_url, level, os.path.join(cache, "imm.zarr"))[:]
    assert raw.shape == outer.shape == inner.shape, f"{run}: {raw.shape}, {outer.shape}, {inner.shape}"
    labels, overlap = merge(outer, inner)
    name = f"{dataset}_{run}"
    overview_vs = round(float(voxel_spacing) * 2 ** level, 1)
    notes = NOTES.format(vs=overview_vs, level=f"pyramid level {level}")
    out_path = os.path.join(out_dir, f"{name}.png")
    display_labels = DilatedView(labels, 3)
    save_volume_figure(DenoisedView(raw, 1, 1.0), {"mito": display_labels}, out_path, DATASET, name, notes=notes)
    return {
        "name": name, "portal_dataset": dataset, "run": run, "voxel_size_nm": float(voxel_spacing) / 10,
        "overview_level": level, "overview_voxel_size_nm": overview_vs / 10, "overview_shape": raw.shape,
        "shape": [int(s * 2 ** level) for s in raw.shape], "loader_urls_valid": omm_ok and imm_ok,
        "outer_fraction": round(float((labels == 1).mean()), 4),
        "inner_fraction": round(float((labels == 2).mean()), 4), "omm_imm_overlap_voxels": overlap,
    }


def render_full_resolution(dataset, run, voxel_spacing, out_dir):
    run_dir = os.path.join(DATA_PATH, f"{dataset}_{run}")
    raw = zarr.open_array(os.path.join(run_dir, "raw.zarr", "0"), mode="r")
    labels = zarr.open_array(os.path.join(run_dir, "labels.zarr", "0"), mode="r")
    assert raw.shape == labels.shape, f"{run}: {raw.shape} vs {labels.shape}"
    name = f"{dataset}_{run}_fullres"
    notes = NOTES.format(vs=voxel_spacing, level="level 0, loader output labels.zarr")
    out_path = os.path.join(out_dir, f"{name}.png")
    save_volume_figure(DenoisedView(raw), {"mito": DilatedView(labels, 5)}, out_path, DATASET, name, notes=notes)
    return {"path": run_dir, "full_shape": raw.shape, "label_values": sorted(np.unique(labels[::4, ::4, ::4]).tolist())}


def main():
    out_dir = get_output_dir(DATASET)
    volumes = []
    for dataset, run, voxel_spacing, omm_scale, imm_scale in drg.RUNS:
        volume = render_overview(dataset, run, voxel_spacing, omm_scale, imm_scale, out_dir)
        local_labels = os.path.join(DATA_PATH, f"{dataset}_{run}", "labels.zarr")
        if (dataset, run) in LOCAL_SUBSET and os.path.exists(local_labels):
            volume["local_full_resolution"] = render_full_resolution(dataset, run, voxel_spacing, out_dir)
        print(volume["name"], volume["overview_shape"], volume["outer_fraction"], volume["inner_fraction"], flush=True)
        volumes.append(volume)
    info = {
        "classes": ["mito (outer + inner membrane, multi-class 1/2)"], "modality": "cryo-ET (CryoET Data Portal)",
        "dimensionality": "3d", "label_quality": "membrane segmentation (OMM/IMM), semantic, not filled mitochondria",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_PATH,
        "download": DOWNLOAD, "notes": NOTES.format(vs="21.6/20.6/10.8", level="full res"), "issues": ISSUES,
        "observations": OBSERVATIONS,
    }
    write_info(DATASET, info)


if __name__ == "__main__":
    main()
