"""Overview figures for MitoEM v2 (8 vEM datasets with expert-verified mitochondria instance labels).

The volumes are read lazily from the nnU-Net style NIfTI files of the zenodo release. Thus, the figures do not need
the full n5 conversion of the torch-em loader, which loads every volume into memory.
NIfTI stores (x, y, z); the wrapper below exposes (z, y, x) like the n5 files written by the loader.
"""
import os
import json

import numpy as np
import nibabel as nib

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "mitoemv2"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
DESCRIPTIONS = {
    "beta": "pancreatic beta cells",
    "jurkat": "Jurkat T cells",
    "macro": "macrophages",
    "mossy": "mossy fiber boutons",
    "podo": "podocytes",
    "pyra": "pyramidal neurons",
    "sperm": "sperm",
    "stem": "stem cells",
}


class NiftiVolume:
    """Lazy (z, y, x) view of a NIfTI volume stored as (x, y, z), or as (z, y, x) if zyx_order is set."""

    def __init__(self, path, zyx_order=False):
        self.proxy = nib.load(path, keep_file_open=True).dataobj
        self.zyx_order = zyx_order
        shape = tuple(int(s) for s in self.proxy.shape)
        self.shape = shape if zyx_order else shape[::-1]
        self.ndim = 3

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        index = index + (slice(None),) * (3 - len(index))
        if self.zyx_order:
            return np.asarray(self.proxy[index])
        return np.asarray(self.proxy[index[::-1]]).T


def get_split_file(folder):
    """Get the split file, which is called splits.json instead of split.json for the pyra dataset."""
    for file_name in ("split.json", "splits.json"):
        if os.path.exists(os.path.join(folder, file_name)):
            return os.path.join(folder, file_name)
    return None


def get_modality(folder):
    with open(os.path.join(folder, "dataset.json")) as f:
        return json.load(f)["channel_names"]["0"]


def get_splits(folder):
    with open(get_split_file(folder)) as f:
        split_info = json.load(f)[0]
    return [(split, sample) for split in ("train", "val", "test") for sample in split_info.get(split, [])]


def render_sample(name, folder, split, sample, out_dir):
    image_dir, label_dir = ("imagesTs", "labelsTs") if split == "test" else ("imagesTr", "labelsTr")
    image_path = os.path.join(folder, image_dir, f"{sample}_0000.nii.gz")
    label_path = os.path.join(folder, label_dir, f"{sample}.nii.gz")
    header = nib.load(image_path).header
    voxel_size = tuple(float(z) for z in header.get_zooms())[::-1]
    raw = NiftiVolume(image_path)
    label_shape = tuple(int(s) for s in nib.load(label_path).shape)
    # The loader transposes the labels of me2-jurkat_train02, which are stored in (z, y, x) order.
    zyx_labels = label_shape != raw.shape[::-1] and label_shape == raw.shape
    labels = NiftiVolume(label_path, zyx_order=zyx_labels)
    notes = (
        f"{get_modality(folder)} {DESCRIPTIONS[name]}, voxel (z,y,x) {voxel_size} nm, "
        f"expert-verified instance mito labels, split {split}, raw {header.get_data_dtype()}"
    )
    out_path = os.path.join(out_dir, name, f"{sample}.png")
    save_volume_figure(raw, {"mito": labels}, out_path, DATASET, f"{name}/{sample}", notes=notes)
    return {
        "name": sample, "dataset": name, "split": split, "shape": raw.shape, "voxel_size_nm": voxel_size,
        "raw_dtype": str(header.get_data_dtype()), "labels_stored_zyx": zyx_labels, "path": image_path,
        "label_path": label_path,
    }


def main():
    from torch_em.data.datasets.electron_microscopy.mitoemv2 import DATASETS

    out_dir = get_output_dir(DATASET)
    volumes, missing = [], []
    for name, folder_name in DATASETS.items():
        folder = os.path.join(DATA_DIR, folder_name)
        if get_split_file(folder) is None:
            missing.append(name)
            continue
        for split, sample in get_splits(folder):
            volumes.append(render_sample(name, folder, split, sample, out_dir))
            print("rendered", sample, flush=True)
    write_info(DATASET, {
        "classes": ["mito"], "modality": "vEM (FIB-SEM / SBF-SEM / ssSEM, per dataset)", "dimensionality": "3d",
        "label_quality": "expert-verified instance labels",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR,
        "download": "7 zenodo zips (beta, macro, mossy, podo, pyra, sperm, stem), 15.9 GB, unzipped next to the zips; "
        "not converted to n5", "notes": "8 datasets with train / val / test splits from split.json, all labeled",
        "issues": (
            "me2-jurkat_train02 (the jurkat val volume) labels do not match the raw: after the loader's transpose "
            "the shapes agree, but mitos are drawn in empty resin and miss the cell mitos (also checked swapped "
            "y/x and flipped z, none fits), so exclude it; "
            "torch-em mitoemv2.py reads split.json, but pyra ships splits.json, so the pyra paths fail; "
            "me2-jurkat_train02 and me2-macro_train02 raw are uint16; the n5 conversion loads full volumes "
            "(pyra test01 is 500 x 4096 x 4096, labels cast to uint64 -> ~67 GB in memory)"
            + ("; missing datasets: " + ", ".join(missing) if missing else "")
        ),
        "observations": (
            "beta, jurkat train01/test01, macro (incl. uint16 train02), podo, sperm, stem: aligned and exhaustive; "
            "mossy: only mitos inside the mossy fiber boutons are labeled, the surrounding neuropil mitos are not; "
            "pyra train crops (500 x 512 x 512) exhaustive, but pyra test01 (500 x 4096 x 4096) has only ~6-8 "
            "labeled mitos per slice, i.e. very sparse; jurkat train02 misaligned (see issues)"
        ),
    })


if __name__ == "__main__":
    main()
