import os

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info
from torch_em.data.datasets.electron_microscopy import pombe_nucleus_mito


DATASET = "pombe_nucleus_mito"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
RUNS = [(10001, "TS_0006"), (10000, "TS_045")]
POPSICLE_DIR = os.path.join(DATA_ROOT, "popsicle_yeast", "yeast_organelles")
POPSICLE_RUNS = {"TS_0006": ("train", 6, 2), "TS_045": ("test", 6, 2)}
NOTES = "cryo-ET S. pombe (CryoET Data Portal), 1.348 nm voxels, expert-verified semantic masks (hybrid)"
DISPLAY_NOTE = "raw shown as 9-slice mean + gaussian (sigma 2) for visibility"
HALF_WINDOW = 4
ISSUES = (
    "Both runs are bit-identical (masks, every 4th slice compared) to popsicle yeast_organelles TS_0006 "
    "(train) and TS_045 (test), so this dataset is a duplicate subset of popsicle; do not use both, and "
    "TS_045 is a popsicle test tomogram. pombe_nucleus_mito.py:86 opens the zarr group in mode 'a' before "
    "the download check, so a call with download=False leaves an empty store behind."
)
OBSERVATIONS = (
    "Labels aligned. TS_0006: one small mito clipped at the image edge (0.16% of voxels) and a large "
    "nucleus inside the lamella; TS_045: one very large mito (11.8%) and a nucleus sliver in a corner "
    "(0.24%). Only a central z band holds specimen. Very little mito/nucleus signal overall: 2 tomograms, "
    "~2 objects each."
)


class SmoothedRaw:
    """Lazy view of a noisy tomogram that averages 2 * HALF_WINDOW + 1 slices around the requested
    plane and applies a gaussian filter, to make the structures visible in the overview figures."""

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.ndim = data.ndim

    def __getitem__(self, key):
        axis = next(i for i, k in enumerate(key) if isinstance(k, (int, np.integer)))
        center = int(key[axis])
        window = slice(max(center - HALF_WINDOW, 0), min(center + HALF_WINDOW + 1, self.shape[axis]))
        full_key = tuple(window if i == axis else slice(None) for i in range(self.ndim))
        plane = np.asarray(self.data[full_key], dtype="float32").mean(axis=axis)
        plane = gaussian_filter(plane, sigma=2)
        return plane[tuple(k for i, k in enumerate(key) if i != axis)]


def compare_with_popsicle(run, labels):
    """Compare the masks with the popsicle yeast_organelles masks of the same tomogram on every 4th slice."""
    split, mito_value, nucleus_value = POPSICLE_RUNS[run]
    popsicle = zarr.open_array(os.path.join(POPSICLE_DIR, split, run, "labels.zarr", "0"), mode="r")[::4]
    result = {}
    for name, value in (("mito", mito_value), ("nucleus", nucleus_value)):
        ours = np.asarray(labels[name][::4]) > 0
        result[f"{name}_identical_to_popsicle_{split}"] = bool(np.array_equal(ours, popsicle == value))
    return result


def render_run(dataset_id, run, out_dir):
    path = pombe_nucleus_mito.get_pombe_nucleus_mito_data(DATA_DIR, dataset_id, run, download=True)
    root = zarr.open_group(path, mode="r")
    labels = {"mito": root["mitochondrion"], "nucleus": root["nucleus"]}
    notes = f"{NOTES}\nportal dataset {dataset_id}; {DISPLAY_NOTE}"
    name = f"{dataset_id}_{run}"
    out_path = os.path.join(out_dir, f"{name}.png")
    save_volume_figure(SmoothedRaw(root["raw"]), labels, out_path, DATASET, name, notes=notes)
    info = {
        "name": name, "run": run, "portal_dataset": dataset_id, "shape": root["raw"].shape,
        "voxel_size_nm": (1.348, 1.348, 1.348), "path": path,
        "portal_annotation_folders": pombe_nucleus_mito.RUN_LABELS[(dataset_id, run)],
    }
    for class_name, label_volume in labels.items():
        sub = np.asarray(label_volume[:, ::8, ::8]) > 0
        z_indices = np.where(sub.any(axis=(1, 2)))[0]
        info[f"{class_name}_fraction"] = round(float(sub.mean()), 4)
        info[f"{class_name}_z_range"] = (int(z_indices.min()), int(z_indices.max())) if len(z_indices) else None
    info.update(compare_with_popsicle(run, labels))
    print(info)
    return info


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_run(dataset_id, run, out_dir) for dataset_id, run in RUNS]
    write_info(DATASET, {
        "classes": ["mito", "nucleus"], "modality": "cryo-ET", "dimensionality": "3d",
        "label_quality": "expert-verified semantic masks (portal methodType hybrid)", "n_volumes": len(volumes),
        "volumes": volumes, "data_path": DATA_DIR,
        "download": "streamed TS_0006 and TS_045 from the CryoET Data Portal via the torch-em loader (713 MB)",
        "notes": NOTES, "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
