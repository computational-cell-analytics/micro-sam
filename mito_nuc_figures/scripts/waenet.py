import os
import re
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info


DATASET = "waenet"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
N_SAMPLES = 16
SUBSETS = {
    1: ("Human pancreatic carcinoid cell line", "FIB-SEM"),
    2: ("BON cell during interphase", "ssTEM"),
    3: ("BON cell during mitosis", "ssTEM"),
    4: ("Human T-cell line Jurkat", "TEM"),
    5: ("Primary human T-cell blood", "TEM"),
    6: ("Murine B-cell line J558L", "TEM"),
    7: ("PHA/IL-2 expanded human T cells", "TEM"),
}
STACKS = (1, 3)
ISSUES = (
    "waenet.py:6-7 swap the descriptions of datasets 2 and 3: dataset 2 holds a mitotic cell (its only "
    "class is condensed chromosomes, shown as 'chromosomes', not nucleus), dataset 3 an interphase cell "
    "with an intact nucleus plus mitochondria. So dataset 2 has no nucleus labels. waenet.py:127 remaps "
    "sorted gray values to class ids per image, which would shift classes in an image that lacks a middle "
    "class (checked: not triggered, dataset 3 images without nucleus also lack mito). waenet.py:169 "
    "deletes Datasets.zip after unzip. waenet.py:205/212 split on a lexicographic file sort (img_10 < "
    "img_2). Pixel size not stored."
)
OBSERVATIONS = (
    "Dataset 1 is a contiguous FIB-SEM stack (consecutive-slice correlation 0.98), usable as a (323, 512, "
    "896) volume; all nuclei labeled and aligned, one seam mid-stack in ZX. Dataset 3 sections are serial "
    "but each is rotated differently (not registered), so treat it as 2D; nucleus exhaustive, mito labels "
    "aligned but tiny (whole cell in 512 px), and some sections with visible mitos have none labeled "
    "(img_9-11, img_49). Datasets 4-7 are single TEM images where only the central cell's nucleus is "
    "labeled (neighbour nuclei unlabeled), with burned-in scale bars and some empty images."
)


def image_number(path):
    return int(re.findall(r"(\d+)\.h5$", path)[0])


def get_paths(dataset_id):
    return sorted(glob(os.path.join(DATA_DIR, f"dataset_{dataset_id}", "*.h5")), key=image_number)


def class_keys(dataset_id):
    if dataset_id == 2:
        return {"chromosomes": "chromosomes"}
    keys = {"nucleus": "nucleus"}
    if dataset_id == 3:
        keys = {"mito": "mitochondria", **keys}
    return keys


def load_image(path, keys):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        labels = {name: f[f"labels/{key}"][:] for name, key in keys.items()}
    return raw, labels


def get_notes(dataset_id):
    description, modality = SUBSETS[dataset_id]
    keys = class_keys(dataset_id)
    classes = ", ".join(f"{name}=labels/{key}" for name, key in keys.items())
    return f"dataset {dataset_id}: {description}, {modality}, pixel size not stored, manual semantic masks ({classes})"


def render_grid(dataset_id, paths, out_dir):
    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(len(paths), size=min(N_SAMPLES, len(paths)), replace=False))
    keys = class_keys(dataset_id)
    samples = []
    for idx in chosen:
        raw, labels = load_image(paths[idx], keys)
        samples.append((os.path.basename(paths[idx]).replace(".h5", ""), raw, labels))
    out_path = os.path.join(out_dir, f"dataset_{dataset_id}_samples.png")
    group = f"dataset_{dataset_id} ({len(samples)} of {len(paths)} images, seeded random)"
    save_image_grid(samples, out_path, DATASET, group, notes=get_notes(dataset_id))


def render_stack(dataset_id, paths, out_dir):
    keys = class_keys(dataset_id)
    images = [load_image(path, keys) for path in paths]
    raw = np.stack([image[0] for image in images])
    labels = {name: np.stack([image[1][name] for image in images]) for name in keys}
    out_path = os.path.join(out_dir, f"dataset_{dataset_id}_stack.png")
    name = f"dataset_{dataset_id} stacked in numeric file order"
    save_volume_figure(raw, labels, out_path, DATASET, name, notes=get_notes(dataset_id))
    return raw.shape


def summarize(dataset_id, paths):
    shapes, n_with = {}, {}
    keys = class_keys(dataset_id)
    for path in paths:
        raw, labels = load_image(path, keys)
        shapes[str(raw.shape)] = shapes.get(str(raw.shape), 0) + 1
        for name, label in labels.items():
            n_with[name] = n_with.get(name, 0) + int(label.any())
    description, modality = SUBSETS[dataset_id]
    return {
        "name": f"dataset_{dataset_id}", "description": description, "modality": modality, "n_images": len(paths),
        "shapes": shapes, "voxel_size_nm": "not stored", "path": os.path.join(DATA_DIR, f"dataset_{dataset_id}"),
        "label_keys": keys, "n_images_with_class": n_with,
    }


def main():
    out_dir = get_output_dir(DATASET)
    volumes = []
    for dataset_id in SUBSETS:
        paths = get_paths(dataset_id)
        info = summarize(dataset_id, paths)
        if dataset_id in STACKS:
            info["stack_shape"] = render_stack(dataset_id, paths, out_dir)
        render_grid(dataset_id, paths, out_dir)
        print(info)
        volumes.append(info)
    write_info(DATASET, {
        "classes": ["mito", "nucleus"], "modality": "FIB-SEM (dataset 1), ssTEM (2, 3), TEM (4-7)",
        "dimensionality": "2d images; dataset 1 is a contiguous FIB-SEM stack, dataset 3 serial sections",
        "label_quality": "manual semantic multi-class masks", "n_volumes": len(volumes), "volumes": volumes,
        "data_path": DATA_DIR, "download": "already local (280 MB, h5 per image; original Datasets.zip not kept)",
        "notes": "", "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
