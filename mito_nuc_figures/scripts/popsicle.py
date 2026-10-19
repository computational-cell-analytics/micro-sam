import os

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info
from torch_em.data.datasets.electron_microscopy.popsicle import YEAST_CLASSES, YEAST_RUNS


DATASET = "popsicle"
DATA_DIR = os.path.join(DATA_ROOT, "popsicle_yeast", "yeast_organelles")
CLASS_VALUES = {"mito": 6, "nucleus": 2}
CLASS_PORTAL_NAMES = ("mitochondrion", "nucleus")
NOTES = "cryo-ET S. pombe (yeast_organelles), 1.348 nm voxels, semantic multi-class masks (2: nucleus, 6: mito)"
DISPLAY_NOTE = "raw shown as 9-slice mean + gaussian (sigma 2) for visibility"
HALF_WINDOW = 4
ISSUES = (
    "Every tomogram is only partly annotated: classes that are not annotated are background, so "
    "per-tomogram class availability must be respected (see 'annotated' per volume): nucleus in "
    "TS_027/037/043/045/0002/0006/0007/0009, mito in TS_029/034/045/0001/0003/0004/0006/0007/0008, "
    "neither in TS_026/028/030/041/0005. TS_045 and TS_0006 masks are bit-identical to pombe_nucleus_mito "
    "(duplicate data; TS_045 is the popsicle test split, so using both leaks test data). Data folder is "
    "popsicle_yeast, module name popsicle."
)
OBSERVATIONS = (
    "Masks align with the structures where visible (after smoothing; native 1.35 nm raw is very noisy). "
    "Annotation is also spatially partial: e.g. TS_0007 mito labeled at z=232 but the same mito unlabeled "
    "at z=266, and its nucleus mask ends at a straight cut; masks usually cover one or two large objects "
    "per tomogram (one giant mito = 11.6% of TS_045). In portal dataset 10001 the lamella only fills a "
    "central z band, the rest is reconstruction noise."
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


class ClassMask:
    """Lazy view of a multi-class label array that returns the mask of one class on slicing."""

    def __init__(self, data, value):
        self.data = data
        self.value = value
        self.shape = data.shape
        self.ndim = data.ndim

    def __getitem__(self, key):
        return (np.asarray(self.data[key]) == self.value).astype("uint8")


def annotated_classes(folders):
    names = {"mitochondrion": "mito", "nucleus": "nucleus"}
    return {names[name]: bool(folder) for name, folder in zip(YEAST_CLASSES, folders) if name in names}


def render_run(run, dataset_id, split, folders, out_dir):
    run_dir = os.path.join(DATA_DIR, split, run)
    raw = SmoothedRaw(zarr.open_array(os.path.join(run_dir, "raw.zarr", "0"), mode="r"))
    label_data = zarr.open_array(os.path.join(run_dir, "labels.zarr", "0"), mode="r")
    annotated = annotated_classes(folders)
    labels = {name: ClassMask(label_data, CLASS_VALUES[name]) for name in ("mito", "nucleus") if annotated[name]}
    other = [name for name, folder in zip(YEAST_CLASSES, folders) if folder and name not in CLASS_PORTAL_NAMES]
    status = ", ".join(f"{name} {'annotated' if flag else 'NOT annotated'}" for name, flag in annotated.items())
    notes = f"{NOTES}\nportal dataset {dataset_id}, split {split}; {status}; {DISPLAY_NOTE}"
    name = f"{split}_{run}"
    save_volume_figure(raw, labels, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    info = {
        "name": name, "run": run, "portal_dataset": dataset_id, "split": split, "shape": raw.shape,
        "voxel_size_nm": (1.348, 1.348, 1.348), "path": run_dir, "annotated": annotated, "other_classes": other,
    }
    sub = np.asarray(label_data[:, ::8, ::8])
    for class_name, value in CLASS_VALUES.items():
        info[f"{class_name}_fraction"] = round(float((sub == value).mean()), 4)
        z_indices = np.where((sub == value).any(axis=(1, 2)))[0]
        info[f"{class_name}_z_range"] = (int(z_indices.min()), int(z_indices.max())) if len(z_indices) else None
    print(info)
    return info


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_run(*run, out_dir) for run in YEAST_RUNS]
    write_info(DATASET, {
        "classes": ["mito", "nucleus"], "modality": "cryo-ET", "dimensionality": "3d",
        "label_quality": "manual/hybrid semantic masks, every tomogram only partly annotated (per-class)",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR,
        "download": "already local (17 GB, yeast_organelles source only)", "notes": NOTES,
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
