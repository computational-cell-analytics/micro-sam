import csv
import os
import tempfile
import zipfile

import fsspec
import mrcfile
import numpy as np
import zarr

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info
from torch_em.data.datasets.electron_microscopy import sxt_ins1e_mito


DATASET = "sxt_ins1e_mito"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
CLASS_VALUES = {"mito": 5, "nucleus": 2}
NOTES = "soft X-ray tomography (not EM), INS-1E whole cells, semantic masks (0 ext, 1 cell, 2 nucleus, 5 mito)"
DOWNLOAD = (
    "All 55 cells streamed from Zenodo 20513085 (Raw_tomograms.zip 17.4 GB + Labels.zip 52 MB, read per member "
    "via HTTP range requests) and cached as zarr by the torch-em loader: 15 GB in DATA_ROOT/sxt_ins1e_mito."
)
ISSUES = (
    "sxt_ins1e_mito.py:66 creates fsspec.filesystem('http') without trust_env, so aiohttp ignores the "
    "http(s)_proxy variables and the download fails with ConnectionTimeoutError on this cluster; this script "
    "replaces _read_mrc_member with a copy using client_kwargs={'trust_env': True}. sxt_ins1e_mito.py:94 opens "
    "the zarr group in mode 'a' before the download check, so a failed or download=False call leaves an empty "
    "store that later looks like a partial cache. Labels are uint16 multi-class, select 2 / 5 explicitly. "
    "Axis 1 is the capillary axis (axes 0 and 2 always have equal length), so the ZX panel is the capillary "
    "cross-section; voxel size varies per cell (36.8-45.5 nm, see Cell_Metadata.csv)."
)
OBSERVATIONS = (
    "All 55 cells rendered, every cell has nucleus and mito labels (mito 0.3-2.3%, nucleus 2-8% of the volume). "
    "Labels are aligned with the cells in all views; one nucleus per cell, cleanly delineated; mito labels "
    "follow the absorbing tubular structures around the cell periphery and look plausible but somewhat "
    "conservative (some faint tubular structures unlabeled); masks look semi-automatic (jagged borders). "
    "The glass capillary wall is visible in every tomogram and is background."
)


class ClassMask:
    """Lazy view of a multi-class label array that returns the mask of one class on slicing."""

    def __init__(self, data, value):
        self.data = data
        self.value = value
        self.shape = data.shape
        self.ndim = data.ndim

    def __getitem__(self, key):
        return (np.asarray(self.data[key]) == self.value).astype("uint8")


def read_mrc_member(zip_url, member_name):
    """Same as the torch-em reader, but lets aiohttp use the http(s)_proxy environment variables."""
    fs = fsspec.filesystem("http", client_kwargs={"trust_env": True})
    with fs.open(zip_url, "rb", block_size=2**24) as f, zipfile.ZipFile(f) as zf:
        content = zf.read(member_name)
    with tempfile.NamedTemporaryFile(suffix=".mrc") as tmp:
        tmp.write(content)
        tmp.flush()
        with mrcfile.open(tmp.name, permissive=True) as mrc:
            return np.asarray(mrc.data)


def read_metadata():
    sxt_ins1e_mito.get_sxt_ins1e_mito_cell_names(DATA_DIR, download=True)
    with open(os.path.join(DATA_DIR, "Cell_Metadata.csv")) as f:
        return list(csv.DictReader(f))


def render_cell(row, out_dir):
    cell_name = row["Cell_Name"]
    path = sxt_ins1e_mito.get_sxt_ins1e_mito_data(DATA_DIR, cell_name, download=True)
    root = zarr.open_group(path, mode="r")
    raw, label_data = root["raw"], root["labels"]
    labels = {name: ClassMask(label_data, value) for name, value in CLASS_VALUES.items()}
    voxel_nm = round(float(row["Voxel_Length (micrometer)"]) * 1000, 1)
    condition = f"{row['Condition']}_{row['Timepoint']}"
    notes = f"{NOTES}\ncondition {row['Condition']}, {row['Timepoint']}, {voxel_nm} nm isotropic voxels"
    name = f"{condition}_{cell_name.replace('_pre_rec', '')}"
    save_volume_figure(raw, labels, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    sub = np.asarray(label_data[::4, ::4, ::4])
    info = {
        "name": name, "cell_name": cell_name, "condition": row["Condition"], "timepoint": row["Timepoint"],
        "shape": raw.shape, "voxel_size_nm": (voxel_nm,) * 3, "path": path,
        "label_values": np.unique(sub).tolist(),
    }
    for class_name, value in {"cell": 1, **CLASS_VALUES}.items():
        info[f"{class_name}_fraction"] = round(float((sub == value).mean()), 4)
    print(info)
    return info


def main():
    sxt_ins1e_mito._read_mrc_member = read_mrc_member
    out_dir = get_output_dir(DATASET)
    volumes = [render_cell(row, out_dir) for row in read_metadata()]
    write_info(DATASET, {
        "classes": ["mito", "nucleus"], "modality": "soft X-ray tomography (cryo SXT), not EM",
        "dimensionality": "3d", "label_quality": "semantic multi-class masks", "n_volumes": len(volumes),
        "volumes": volumes, "data_path": DATA_DIR, "download": DOWNLOAD, "notes": NOTES, "issues": ISSUES,
        "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
