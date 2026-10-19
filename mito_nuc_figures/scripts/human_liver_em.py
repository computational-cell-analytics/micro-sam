import io
import os
from glob import glob

import numpy as np
import requests
import tifffile
import zarr
from PIL import Image

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info
from torch_em.data.datasets.electron_microscopy import human_liver_em


DATASET = "human_liver_em"
EMPIAR_BASE = human_liver_em.EMPIAR_BASE
OVERVIEW_Z = (60, 298, 540)
OVERVIEW_STRIDE = 26
CROP_BBOXES = []
NOTES = "SBF-SEM human liver, pixel size undocumented (mito size suggests ~5-10 nm xy), semantic binary masks (0/255)"
VOXEL_SIZE = "undocumented (~5-10 nm xy est.)"
DOWNLOAD = (
    "Nothing new stored in DATA_ROOT (folder is not writable for u28048). Streamed 3 full raw slices "
    "(~830 MB each) plus their mito/nucleus PNGs from EMPIAR-13356 and cached stride-26 versions (~2 MB) in "
    "mito_nuc_figures/mito_and_nucleus/human_liver_em/cache. The local crop (8, 2048, 2048) was already present "
    "(64 MB)."
)
ISSUES = (
    "Pixel size is undocumented; the loader docstring guess of ~50-100 nm/pixel is inconsistent with the data: "
    "mitochondria are ~50-150 px wide at full resolution, which points to roughly 5-10 nm/pixel. The local zarr "
    "holds all 9 classes (written by an older loader version); the current loader only writes er/mito/nucleus. "
    "Only one 8-slice crop is cached locally, so a real 3D training set requires streaming new boxes "
    "(~42 MB per slice per 1024 rows of raw)."
)
OBSERVATIONS = (
    "Full-resolution crop: mito labels are exhaustive and tightly aligned, nucleus mask exact. Whole-slice "
    "overview (z 60/298/540): mito labels only cover part of the tissue (roughly the upper 70% of each slice, "
    "~45-57% of 1000 px blocks contain labels), while mitochondria-rich hepatocytes in the rest of the slice "
    "are unlabeled, so boxes must be picked inside the annotated region. Nucleus labels are sparse (2-8 "
    "nuclei per slice). A large smooth artefact region (lower right) is unlabeled."
)


def block_max(mask, stride):
    h, w = (mask.shape[0] // stride) * stride, (mask.shape[1] // stride) * stride
    return mask[:h, :w].reshape(h // stride, stride, w // stride, stride).max(axis=(1, 3))


def fetch_overview_slice(z, cache_dir):
    cache_path = os.path.join(cache_dir, f"z{z:03d}.npz")
    if os.path.exists(cache_path):
        return dict(np.load(cache_path))
    x_min, x_max, y_min, y_max, _, _ = human_liver_em.HUMAN_LIVER_EM_TISSUE_BBOX
    s = OVERVIEW_STRIDE
    print(f"Fetching full slice {z} ...", flush=True)
    url = f"{EMPIAR_BASE}/humanliver_raw_images/humanliver_raw_{z:03d}.tif"
    content = requests.get(url, timeout=3600).content
    raw = tifffile.imread(io.BytesIO(content))[y_min:y_max, x_min:x_max]
    del content
    out = {"raw": raw[::s, ::s].copy()}
    del raw
    Image.MAX_IMAGE_PIXELS = None
    for name in ("mito", "nucleus"):
        label_dir = human_liver_em.HUMAN_LIVER_EM_LABEL_DIRS[name]
        response = requests.get(f"{EMPIAR_BASE}/{label_dir}/{label_dir}_{z:03d}.png", timeout=600)
        mask = np.array(Image.open(io.BytesIO(response.content)))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask[y_min:y_max, x_min:x_max]
        out[f"{name}_fraction"] = np.array((mask > 0).mean())
        out[name] = block_max(mask, s)[:out["raw"].shape[0], :out["raw"].shape[1]]
    out["raw"] = out["raw"][:out["mito"].shape[0], :out["mito"].shape[1]]
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    return out


def render_overview(out_dir):
    cache_dir = os.path.join(out_dir, "cache", f"overview_slices_stride{OVERVIEW_STRIDE}")
    slices = [fetch_overview_slice(z, cache_dir) for z in OVERVIEW_Z]
    raw = np.stack([sl["raw"] for sl in slices])
    labels = {name: np.stack([sl[name] for sl in slices]) for name in ("mito", "nucleus")}
    name = f"full tissue slices z={list(OVERVIEW_Z)} (panel z=0,1,2), stride {OVERVIEW_STRIDE}, labels block-max"
    save_volume_figure(
        raw, labels, os.path.join(out_dir, "overview_full_slices.png"), DATASET, name, notes=NOTES, ortho_view=False
    )
    fractions = {}
    for z, sl in zip(OVERVIEW_Z, slices):
        fractions[f"z{z}"] = {k: float(sl[f"{k}_fraction"]) for k in ("mito", "nucleus")}
    return {
        "name": "overview_full_slices", "shape": (597, 20000, 20000), "voxel_size_nm": VOXEL_SIZE,
        "path": f"{EMPIAR_BASE} (remote), cached downsampled slices in {cache_dir}",
        "foreground_fraction_in_tissue_bbox": fractions,
    }


def render_crop(path, out_dir):
    root = zarr.open_group(path, mode="r")
    bbox = root.attrs["bounding_box"]
    name = "crop_" + "_".join(str(v) for v in bbox)
    raw = root["raw"]
    labels = {"mito": root["mito"], "nucleus": root["nucleus"]}
    out_path = os.path.join(out_dir, f"{name}.png")
    save_volume_figure(raw, labels, out_path, DATASET, f"{name} (x0,x1,y0,y1,z0,z1)", notes=NOTES)
    info = {"name": name, "shape": raw.shape, "voxel_size_nm": VOXEL_SIZE, "path": path, "bbox_xxyyzz": bbox}
    for class_name, label_volume in labels.items():
        info[f"{class_name}_fraction"] = round(float((label_volume[:] > 0).mean()), 5)
    return info


def main():
    out_dir = get_output_dir(DATASET)
    data_dir = os.path.join(DATA_ROOT, DATASET)
    crop_dir = os.path.join(out_dir, "cache")
    for bbox in CROP_BBOXES:
        human_liver_em.get_human_liver_em_data(crop_dir, bbox, download=True)
    volumes = [render_overview(out_dir)]
    crop_paths = sorted(glob(os.path.join(data_dir, "*.zarr"))) + sorted(glob(os.path.join(crop_dir, "*.zarr")))
    volumes += [render_crop(path, out_dir) for path in crop_paths]
    write_info(DATASET, {
        "classes": ["mito", "nucleus"], "modality": "SBF-SEM", "dimensionality": "3d",
        "label_quality": "semantic binary masks (0/255), regionally partial", "n_volumes": 1, "volumes": volumes,
        "data_path": data_dir, "download": DOWNLOAD, "notes": NOTES, "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
