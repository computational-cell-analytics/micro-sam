import os

import numpy as np
import tifffile

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, write_info
from torch_em.data.datasets.electron_microscopy import tnbc_mito


DATASET = "tnbc_mito"
NOTES = "TEM (2D), pixel size not documented, manual semantic mito masks (0/1)"
MAX_IMAGES = 24
N_SAMPLES = 16


def load_sample(raw_path, label_path):
    raw = tifffile.imread(raw_path)
    labels = tifffile.imread(label_path)
    assert raw.shape[:2] == labels.shape, f"{raw_path}: {raw.shape} vs {labels.shape}"
    name = os.path.splitext(os.path.basename(raw_path))[0]
    return name, raw, labels


def render_group(data_path, cohort, split, out_dir):
    try:
        raw_paths, label_paths = tnbc_mito.get_tnbc_mito_paths(data_path, split, cohort, download=False)
    except AssertionError:
        return None
    indices = np.arange(len(raw_paths))
    if len(indices) > MAX_IMAGES:
        indices = np.sort(np.random.default_rng(0).choice(indices, N_SAMPLES, replace=False))
    samples, shapes, fractions = [], set(), []
    for i in indices:
        name, raw, labels = load_sample(raw_paths[i], label_paths[i])
        samples.append((name, raw, {"mito": labels}))
        shapes.add(tuple(raw.shape))
        fractions.append(float((labels > 0).mean()))
    group = f"{cohort}_{split}"
    shown = "all" if len(indices) == len(raw_paths) else f"{len(indices)} random (seed 0) of {len(raw_paths)}"
    notes = f"{NOTES}; {shown}"
    save_image_grid(samples, os.path.join(out_dir, f"{group}.png"), DATASET, group, notes=notes, ncols=6)
    return {
        "name": group, "n_images": len(raw_paths), "n_shown": len(indices), "shapes": sorted(shapes),
        "voxel_size_nm": "not documented", "path": os.path.dirname(raw_paths[0]),
        "mean_foreground_fraction_shown": round(float(np.mean(fractions)), 4),
    }


def main():
    data_path = os.path.join(DATA_ROOT, DATASET)
    tnbc_mito.get_tnbc_mito_data(data_path, download=True)
    out_dir = get_output_dir(DATASET)
    splits = [(cohort, split) for cohort in tnbc_mito.COHORTS for split in ("tra_val", "tst")]
    groups = [render_group(data_path, cohort, split, out_dir) for cohort, split in splits]
    groups = [group for group in groups if group is not None]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "TEM", "dimensionality": "2d", "label_quality": "manual, semantic (binary)",
        "n_volumes": sum(group["n_images"] for group in groups), "volumes": groups, "data_path": data_path,
        "download": "", "notes": NOTES, "issues": "", "observations": "",
    })


if __name__ == "__main__":
    main()
