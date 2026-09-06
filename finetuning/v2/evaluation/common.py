import os
import re
import ast
import csv
import warnings
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import xxhash
import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components
from sklearn.model_selection import train_test_split

import torch

from elf.io import open_file

from torch_em.data import datasets
from torch_em.util.image import load_image
from torch_em.util.segmentation import size_filter

from micro_sam.v1.evaluation.livecell import _get_livecell_paths
from micro_sam.v2.normalization import normalize_raw
from micro_sam.v2.datasets.generalist_loader import (
    ASTIH_SUBSETS, AXONEM_TEST_VOLUMES, AXONEM_VAL_VOLUMES, FAFB_TEST_BOXES, FAFB_VAL_BOXES, FIB25_TEST_SAMPLE,
    LICONN_ROI, MALECNS_TEST_BOXES, MALECNS_VAL_BOXES, TUMOR_SPHEROID_TEST_SLICES, TUMOR_SPHEROID_VAL_SLICES,
    WILDENBERG_P105_BOX, XPRESS_CORE, ZEBRAFINCH_J0126_BOX, ZEBRAFINCH_J0251_TEST_BOXES, ZEBRAFINCH_J0251_VAL_BOXES,
    BITDEPTH_MAGNIFICATIONS, CARTOCELL_TEST_FOLDER, CARTOCELL_VAL_FOLDER, CELL_ACDC_TEST_MOVIES, CELL_ACDC_VAL_MOVIES,
    CELLBINDB_HE_STAIN, CELLBINDB_STAINS, CELLULAR_TEST_WELLS, CELLULAR_VAL_WELLS, CISD_TEST_SLIDES, CISD_VAL_SLIDES,
    CVZ_TEST_GROUPS, CVZ_VAL_GROUPS, EMBEDSEG_ORGANOID_TEST_TIMEPOINTS, EMBEDSEG_ORGANOID_VAL_TIMEPOINTS,
    EMBEDSEG_PLATY_NUCLEI_TEST_TIMEPOINTS, EMBEDSEG_PLATY_NUCLEI_VAL_TIMEPOINTS, EMBEDSEG_VAL_Z, ENSEG_TEST_ANIMALS,
    ENSEG_VAL_ANIMALS, GONUCLEAR_TEST_SAMPLES, GONUCLEAR_VAL_SAMPLES, NIS3D_VAL_Z, NUCVERSE_GLIA_VAL_VOLUME,
    NUCVERSE_GLIA_VAL_Z, NUCVERSE_VAL_VOLUMES, ORGANOID_SOURCES, PANNUKE_FOLD2_VAL_TILES, PHMAMM_TEST_TIMEPOINTS,
    PHMAMM_VAL_TIMEPOINTS,
    PNAS_TEST_PLANTS, PNAS_VAL_PLANTS, TOIAM_TEST_MOVIES, TOIAM_VAL_MOVIES, WING_DISC_TEST_VOLUMES, WING_DISC_VAL_Z,
    XENIUM_TEST_SAMPLES, XENIUM_VAL_SAMPLES, cell_acdc_movie, cvz_group, dsb_fluorescence_training_paths,
    _train_val_test_split,
)


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# Derived evaluation data (slide tiles, movie frames) is cached here, since not every dataset folder is writable.
EVAL_CACHE_ROOT = os.path.join(DATA_ROOT, "eval_cache")

_MODELS_DIR = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2"

# The pretrained SAM2 backbones. Only SAM2.1 is supported by micro_sam.v2.
CHECKPOINT_PATHS = {
    "hvit_t": os.path.join(_MODELS_DIR, "sam2.1_hiera_tiny.pt"),
    "hvit_s": os.path.join(_MODELS_DIR, "sam2.1_hiera_small.pt"),
    "hvit_b": os.path.join(_MODELS_DIR, "sam2.1_hiera_base_plus.pt"),
    "hvit_l": os.path.join(_MODELS_DIR, "sam2.1_hiera_large.pt"),
}

MODEL_TYPES = list(CHECKPOINT_PATHS)

MODES = ("ais", "apg")

# The 2d patch shape the models were trained on, see 'generalist_loader'.
TRAINING_PATCH_SHAPE = (512, 512)

# LIVECell test images whose annotation is incomplete: 2 labelled objects in a confluent crop, at
# 348x and 24x the annotated foreground. Both lie outside the stratified subset.
LIVECELL_EXCLUDED_TEST_IMAGES = frozenset({
    "BV2_Phase_A4_2_02d04h00m_3.tif",
    "BV2_Phase_A4_2_00d00h00m_1.tif",
})


def drop_excluded_livecell(raw_paths, label_paths) -> Tuple[List[str], List[str]]:
    """Remove the incompletely annotated LIVECell test images from a path pair list."""
    keep = [
        (raw, label) for raw, label in zip(raw_paths, label_paths)
        if os.path.basename(raw) not in LIVECELL_EXCLUDED_TEST_IMAGES
    ]
    if not keep:
        return [], []
    kept_raw, kept_label = zip(*keep)
    return list(kept_raw), list(kept_label)


# Overridable, so a run can point at another training version. A checkpoint that still trains needs
# a frozen copy, because the trainer overwrites 'best.pt' while jobs queue.
#
# Read fresh on every call (not bound as a module-level constant at import time): a constant here
# would silently ignore `os.environ["MICRO_SAM2_JOINT_CHECKPOINT_ROOT"] = ...` set later in the same
# process, since that assignment always runs after `common` has already been imported. That bit a
# whole investigation's worth of scratch scripts, each believing it had switched checkpoints within
# one process when it had not. Set the env var before the process starts (a shell `export`, or a
# subprocess/sbatch job) for the common case, or pass an explicit path to `build_model` directly.
def _joint_checkpoint_root() -> str:
    return os.environ.get("MICRO_SAM2_JOINT_CHECKPOINT_ROOT", os.path.join(_MODELS_DIR, "joint", "v2", "checkpoints"))


# The joint checkpoints are split into loadable weight files here, see 'export_joint_checkpoint'.
def _joint_export_root() -> str:
    return os.environ.get("MICRO_SAM2_JOINT_EXPORT_ROOT", os.path.join(_MODELS_DIR, "exported", "joint", "v2"))


# HPA is evaluated on the microtubules, nuclei and ER channels, stacked in that order.
HPA_CHANNELS = ("raw/microtubules", "raw/nuclei", "raw/er")
SPATCH_DAPI_SUBSETS = ["xenium_ov", "xenium_hcc", "xenium_coad", "cosmx_ov", "cosmx_hcc", "cosmx_coad"]

# The four E. coli pathways on disk; the remaining four are 6 GB archives each and are not fetched.
ECOLI_GENES = ("cib", "crosstalk", "recA", "rpsM")

# Light microscopy, 2d. The in-domain (ID) panels hold the blind test splits of the v5 training datasets that the
# main comparison scores: the sources shared with the baselines. The supplementary panels hold the blind splits of
# the remaining training datasets ("extra training"), resolvable and scorable on request but outside the main
# panel. Out-of-domain (OOD): datasets kept out of training. Datasets with a cell and a nucleus target are listed
# per target, omnipose per subset.
DATASETS_2D_LM_CELL_ID = [
    "tissuenet", "omnipose_bact_fluor", "cvz_fluo_cell", "flywing", "enseg", "neurips_cellseg_fluorescence",
]
DATASETS_2D_LM_CELL_SUPPLEMENTARY = ["dememseg", "pan_multiplex", "xenium_cells", "deepbacs_fluorescence"]
DATASETS_2D_LM_CELL_OOD = ["covid_if_cells", "medussa", "hpa"]
DATASETS_2D_LM_NUCLEUS_ID = [
    "dsb", "cvz_fluo_dapi", "dynamicnuclearnet", "bitdepth_nucseg", "bmgd", "cellbindb", "u20s", "ifnuclei",
    "tsakiroglou",
]
DATASETS_2D_LM_NUCLEUS_SUPPLEMENTARY = ["xenium_nuclei"]
DATASETS_2D_LM_NUCLEUS_OOD = [
    "cardioblast_nuclei", "hela_cytonuc", "covid_if_nuclei", "arvidsson", "mndino", "micro_bench",
]
# sPATCH DAPI shares tissue sections with the sPATCH H&E training data, so it is a held-out platform evaluation
# rather than an independent OOD collection until the patient overlap is resolved.
DATASETS_2D_LM_NUCLEUS_HELD_OUT_PLATFORM = ["spatch_dapi"]
DATASETS_2D_LM_LABEL_FREE_ID = [
    "livecell", "deepbacs_label_free", "omnipose_bact_phase", "yeaz", "neurips_cellseg_label_free", "cell_acdc",
    "cellular", "vicar", "microbeseg",
]
DATASETS_2D_LM_LABEL_FREE_SUPPLEMENTARY = [
    "orgasegment", "organoidnet", "omnipose_worm", "omnipose_worm_high_res", "bccd", "cisd", "orgline", "organoid",
    "mcellseg", "toiam", "bbbc030",
]
DATASETS_2D_LM_LABEL_FREE_OOD = [
    "cellapp", "deepseas", "dic_hepg2", "yeastsam", "bac_mother", "ecoli_microcolony_lineage",
]
# Reserve: kept resolvable, not part of any panel until its labels are checked.
DATASETS_2D_LM_LABEL_FREE_RESERVE = ["yeastcellseg"]
DATASETS_2D_LM = list(dict.fromkeys(
    DATASETS_2D_LM_CELL_ID + DATASETS_2D_LM_CELL_SUPPLEMENTARY + DATASETS_2D_LM_CELL_OOD
    + DATASETS_2D_LM_NUCLEUS_ID + DATASETS_2D_LM_NUCLEUS_SUPPLEMENTARY + DATASETS_2D_LM_NUCLEUS_OOD
    + DATASETS_2D_LM_NUCLEUS_HELD_OUT_PLATFORM + DATASETS_2D_LM_LABEL_FREE_ID + DATASETS_2D_LM_LABEL_FREE_SUPPLEMENTARY
    + DATASETS_2D_LM_LABEL_FREE_OOD + DATASETS_2D_LM_LABEL_FREE_RESERVE
))

# Histopathology nuclei. The main panel holds the official test splits of the v5 training datasets shared with the
# baselines, which the generalist loader never touches. Supplementary: training datasets with a blind split outside
# the main panel; lizard (overlap with CoNSeP and PanNuke), lynsec and tnbc_celltype (baseline exposure) stay there
# until those are resolved. Out-of-domain: datasets kept out of training. lynsec is split by stain so H&E and IHC
# are reported apart.
DATASETS_2D_HP_ID = ["cpm17", "glysac", "histo_miner", "monuseg", "pannuke", "puma"]
DATASETS_2D_HP_SUPPLEMENTARY = [
    "lizard", "lizard_mitosis", "lynsec_he", "lynsec_ihc", "srsanet", "tnbc_celltype", "cellbindb_he",
]
DATASETS_2D_HP_OOD = ["cytodark0", "deepliif", "khoshdeli", "panoptils", "pcns"]
DATASETS_HP = DATASETS_2D_HP_ID + DATASETS_2D_HP_SUPPLEMENTARY + DATASETS_2D_HP_OOD

# Electron microscopy. The 2d sets are ASTIH (myelinated axons, neurite category) and the tumor spheroid slices
# (cell category); their splits and the 3d regions below come from the generalist loader constants.
DATASETS_2D_EM = ["astih", "tumor_spheroid"]

DATASETS_2D = DATASETS_2D_LM + DATASETS_HP + DATASETS_2D_EM

# Ground-truth size floor that drops the crop-severed slivers relabelling promotes to objects. It
# defines the ground truth, so it is measured, never tuned.
GT_MIN_SIZE_2D = {
    "livecell": 50, "dsb": 10,
    "deepbacs_label_free": 50, "deepbacs_fluorescence": 50, "dynamicnuclearnet": 50, "tissuenet": 10,
    "u20s": 10, "vicar": 25, "yeaz": 10,
}

# Light microscopy, 3d, grouped as the 2d datasets. nis3d is the Drosophila pair the loader trains on. Of MorphoNet
# only the C. elegans nuclei (CTC Fluo-N3DH-CE) are scored: its Arabidopsis volumes are the PNAS plant1 time series
# and its Phallusia volumes come from the PhMamm source, both of which train.
DATASETS_3D_LM_CELL_ID = ["plantseg_root", "pnas_arabidopsis", "cartocell", "phmamm", "wing_disc", "embedseg_organoid"]
DATASETS_3D_LM_CELL_OOD = ["plantseg_ovules", "cshaper", "vibrio_cholerae"]
DATASETS_3D_LM_NUCLEUS_ID = [
    "embedseg_mouse_skull", "embedseg_platy_nuclei", "nis3d", "celegans_atlas", "gonuclear", "nucverse3d",
]
DATASETS_3D_LM_NUCLEUS_SUPPLEMENTARY = ["embedseg_platy_ish"]
DATASETS_3D_LM_NUCLEUS_OOD = ["parhyale_regen", "mouse_embryo", "blastospim", "morphonet_celegans"]
DATASETS_3D_LM = (
    DATASETS_3D_LM_CELL_ID + DATASETS_3D_LM_CELL_OOD + DATASETS_3D_LM_NUCLEUS_ID + DATASETS_3D_LM_NUCLEUS_SUPPLEMENTARY
    + DATASETS_3D_LM_NUCLEUS_OOD
)

# Neurite segmentation: the blind regions of the v5 training sets (see EM_ROIS and the path resolver). The main
# panel holds the sources shared with the baselines, the supplementary panel the remaining training sets. nisb is
# synthetic and kept out of training, so it is a synthetic OOD test. synapseweb is scored inside its annotated cores
# only. humanneurons (the cached H01 crop) is no OOD set: H01 trains through EMNeuron, so it is supplementary at most.
DATASETS_3D_EM_NEURITE_ID = ["cremi", "fafb", "hemibrain", "zebrafinch_j0126", "liconn", "xpress"]
DATASETS_3D_EM_NEURITE_SUPPLEMENTARY = [
    "snemi", "axonem", "fib25", "manc", "malecns", "wafer4", "minnie65", "zebrafinch_j0251", "wildenberg",
    "humanneurons",
]
DATASETS_3D_EM_NEURITE_OOD = ["isbi2012", "synapseweb", "nisb"]
# Cell segmentation: Platynereis volume 9 and the DenseCell val volume are blind; the tumor spheroid slices are 2d.
DATASETS_3D_EM_CELL_ID = ["platynereis_cells", "densecell"]

DATASETS_3D_EM = (
    ["platynereis_nuclei"] + DATASETS_3D_EM_NEURITE_ID + DATASETS_3D_EM_NEURITE_SUPPLEMENTARY
    + DATASETS_3D_EM_NEURITE_OOD + DATASETS_3D_EM_CELL_ID
)
DATASETS_EM = DATASETS_2D_EM + DATASETS_3D_EM

DATASETS_3D = DATASETS_3D_LM + DATASETS_3D_EM

# The neurite datasets need the dense (multicut) pipeline and are ranked by the CREMI score. The EM cell datasets and
# platynereis_nuclei segment separable objects, so they stay on the sparse (flow) pipeline and mSA ranking.
DATASETS_DENSE = DATASETS_3D_EM_NEURITE_ID + DATASETS_3D_EM_NEURITE_SUPPLEMENTARY + DATASETS_3D_EM_NEURITE_OOD

# Everything outside the main in-domain and OOD panels: supplementary, held-out platform and reserve sets.
# The submission script skips them unless asked for them.
DATASETS_SUPPLEMENTARY = (
    DATASETS_2D_LM_CELL_SUPPLEMENTARY + DATASETS_2D_LM_NUCLEUS_SUPPLEMENTARY + DATASETS_2D_LM_NUCLEUS_HELD_OUT_PLATFORM
    + DATASETS_2D_LM_LABEL_FREE_SUPPLEMENTARY + DATASETS_2D_LM_LABEL_FREE_RESERVE + DATASETS_3D_LM_NUCLEUS_SUPPLEMENTARY
    + DATASETS_3D_EM_NEURITE_SUPPLEMENTARY + DATASETS_2D_HP_SUPPLEMENTARY
)

# The split to tune on, or None where the loader has no splits and VAL_Z_RANGE holds out a z-slab.
# The tuning data of the light microscopy datasets: 'val' is the split the generalist loader validates on, 'train'
# is used for out-of-domain datasets that have no validation split of their own but do have training data the
# loader never sees. None means the tuning data is a z-slab of the evaluated volumes, see LM_VAL_Z_SLABS, or the
# reserved sample range covid_if uses. Datasets without held-out data are absent and cannot be tuned.
VAL_SPLITS = {
    name: "val" for name in (
        DATASETS_2D_LM_CELL_ID + DATASETS_2D_LM_NUCLEUS_ID + DATASETS_2D_LM_LABEL_FREE_ID
        + DATASETS_2D_LM_CELL_SUPPLEMENTARY + DATASETS_2D_LM_NUCLEUS_SUPPLEMENTARY
        + DATASETS_2D_LM_LABEL_FREE_SUPPLEMENTARY
        + ["plantseg_root", "pnas_arabidopsis", "cartocell", "phmamm", "embedseg_organoid", "embedseg_platy_nuclei",
           "celegans_atlas", "gonuclear", "nucverse3d"]
    )
}
VAL_SPLITS.update({
    "covid_if_cells": "val", "covid_if_nuclei": "val", "medussa": "train", "cardioblast_nuclei": "train",
    "hela_cytonuc": "val", "arvidsson": "val", "mndino": "val", "cellapp": "train", "deepseas": "train",
    "dic_hepg2": "val", "bac_mother": "val", "plantseg_ovules": "val", "cshaper": "train", "mouse_embryo": "train",
    "blastospim": "val",
    "wing_disc": None, "embedseg_mouse_skull": None, "embedseg_platy_ish": None, "nis3d": None,
    "platynereis_nuclei": None, "humanneurons": None,
})

# Volumes whose tuning data is a z-slab of the test volume, as (file name, z-slab), following the loader; the
# evaluation scores the rest of the volume, see val_z_range. nucverse3d holds separate tuning volumes for its liver
# collections and a slab for drosophila_glia, see _get_3d_lm_data_paths.
LM_VAL_Z_SLABS = {
    "wing_disc": {f"{name}.h5": WING_DISC_VAL_Z for name in WING_DISC_TEST_VOLUMES},
    "embedseg_mouse_skull": {"X2_right.tif": EMBEDSEG_VAL_Z["Mouse-Skull-Nuclei-CBG"]},
    "embedseg_platy_ish": {"X02_test.tif": EMBEDSEG_VAL_Z["Platynereis-ISH-Nuclei-CBG"]},
    "nis3d": {"data.tif": NIS3D_VAL_Z},
    "nucverse3d": {NUCVERSE_GLIA_VAL_VOLUME: NUCVERSE_GLIA_VAL_Z},
}

# EM: None means the tuning data is a different region (EM_ROIS) or different files (see _get_3d_em_data_paths) of
# the same dataset, both disjoint from the blind test region. fib25 has no validation data and is not listed.
VAL_SPLITS.update({
    name: None for name in (
        "cremi", "snemi", "axonem", "fafb", "hemibrain", "manc", "malecns", "wafer4", "minnie65",
        "zebrafinch_j0126", "zebrafinch_j0251", "wildenberg", "liconn", "xpress", "nisb", "platynereis_cells",
        "densecell",
        "astih", "tumor_spheroid",
    )
})

# Histopathology tuning splits. The 'val' splits are the loader's validation data, 'train' is used where a
# dataset has no val split (cpm17, monuseg, glysac); both are disjoint from the scored test split. pannuke
# tunes on fold 2 (see _get_hp_data_paths). pcns is out of domain and its train split is otherwise unused.
VAL_SPLITS.update({
    "cpm17": "train", "glysac": "train", "histo_miner": "val", "lizard": "val", "lizard_mitosis": "val",
    "lynsec_he": "val", "lynsec_ihc": "val", "monuseg": "train", "pannuke": "val", "puma": "val",
    "srsanet": "val", "tnbc_celltype": "val", "cytodark0": "val", "deepliif": "val", "pcns": "train",
    "cellbindb_he": "val",
})

# The tuning slab for volumes with no splits, disjoint from the slab the evaluation scores. Indices
# count from what load_volume keeps, so snemi starts at slice 70, and gonuclear skips its sparse start.
VAL_Z_RANGE = {
    "humanneurons": (0, 16),
}

# The regions of the EM volumes the evaluation scores ('test', blind for training) and the parameter search tunes on
# ('val', the loader's validation region), as (z, y, x) slices in the cached volume. Datasets whose test and tuning
# data are different files (cremi, axonem, fafb, malecns, minnie65, zebrafinch_j0251) are resolved by path instead.
EM_ROIS = {
    "snemi": {"test": np.s_[80:, :, :], "val": np.s_[60:80, :, :]},
    "hemibrain": {"test": np.s_[820:, :, :], "val": np.s_[700:820, :, :]},
    "manc": {"test": np.s_[820:, :, :], "val": np.s_[700:820, :, :]},
    "wafer4": {"test": np.s_[100:, :, :], "val": np.s_[80:100, :, :]},
    "wildenberg": {"test": np.s_[120:, :, :], "val": np.s_[100:120, :, :]},
    "zebrafinch_j0126": {"test": np.s_[512:, :, :], "val": np.s_[448:512, :, :]},
    "liconn": {"test": (slice(576, 640), *LICONN_ROI[1:]), "val": (slice(512, 576), *LICONN_ROI[1:])},
    "xpress": {"test": (slice(308, 328), *XPRESS_CORE[1:]), "val": (slice(288, 308), *XPRESS_CORE[1:])},
    "cremi": {"test": np.s_[:, :, :], "val": np.s_[100:, :, :]},
    # Volume 9 (test) and volumes 7-8 (val) are read inside the bounding boxes of their labelled cells.
    "platynereis_cells": {"test": np.s_[10:110, 128:1152, 128:1152], "val": np.s_[:, :, :]},
    "densecell": {"test": np.s_[:, :, :], "val": np.s_[35:, :, :]},
}

# SynapseWeb is annotated in an irregular core of each volume; these are the bounding boxes of the dense cores.
SYNAPSEWEB_CORE_ROIS = {
    "spine": np.s_[0:42, 768:1984, 1024:1984],
    "oblique": np.s_[5:75, 896:3584, 1344:3328],
    "apical": np.s_[5:111, 192:3776, 320:4032],
}


# Volumes that are mostly empty around a small labelled specimen (e.g. one embryo in a 2048x2048 light-sheet
# frame): the evaluation crop is centered on the bounding box of the labels instead of on the volume.
LABEL_CENTERED_VOLUMES = {"blastospim": "labels"}


def _label_bbox_roi(dataset_name, label_path):
    """The bounding box of the non-zero labels of one volume, cached under EVAL_CACHE_ROOT."""
    import json
    cache_dir = os.path.join(EVAL_CACHE_ROOT, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, os.path.basename(label_path) + "_roi.json")
    if not os.path.exists(cache_path):
        labels = np.asarray(open_file(label_path, mode="r")[LABEL_CENTERED_VOLUMES[dataset_name]])
        fg = np.nonzero(labels != 0)
        roi = [[int(ax.min()), int(ax.max()) + 1] for ax in fg]
        with open(cache_path, "w") as f:
            json.dump({"roi": roi}, f)
    with open(cache_path) as f:
        return tuple(slice(a, b) for a, b in json.load(f)["roi"])


def em_roi(dataset_name: str, label_path: str, split: str):
    """The (z, y, x) roi of one volume for the 'test' or 'val' region, or None to read it whole."""
    if dataset_name in LABEL_CENTERED_VOLUMES:
        return _label_bbox_roi(dataset_name, label_path)
    if dataset_name == "axonem":
        # Only a central block of each volume is annotated; its bounding box is cached next to the labels.
        import json
        with open(label_path.replace(".h5", "_roi.json")) as f:
            return tuple(slice(a, b) for a, b in json.load(f)["roi"])
    if dataset_name == "synapseweb":
        region = os.path.basename(label_path).replace("synapseweb_hippocampus_", "").replace(".h5", "")
        return SYNAPSEWEB_CORE_ROIS[region]
    rois = EM_ROIS.get(dataset_name)
    return None if rois is None else rois[split]


# platynereis_nuclei has 12 volumes, all of which the evaluation reads in full (crop centered on each
# volume's own depth), so tuning cannot afford to sweep every volume: instead of an equal z-slab like
# VAL_Z_RANGE, tuning uses only these 3 sample ids (of 12), the ones with the most annotated (valid_roi,
# i.e. label != -1) foreground voxels, each restricted to its own 16-slice z-window that maximizes that
# foreground count while staying clear of the slab the evaluation's own center crop reads. Chosen by
# measuring per-slice foreground density on the actual data; see load_volume for the valid_roi masking.
PLATYNEREIS_NUCLEI_VAL_SAMPLES = {1: (28, 44), 5: (2, 18), 8: (99, 115)}


def platynereis_nuclei_val_z_range(raw_path: str) -> Tuple[int, int]:
    """The z-window PLATYNEREIS_NUCLEI_VAL_SAMPLES picked for one 'train_data_nuclei_%02i.h5' path."""
    sample_id = int(re.search(r"nuclei_(\d+)\.h5$", raw_path).group(1))
    return PLATYNEREIS_NUCLEI_VAL_SAMPLES[sample_id]


def val_z_range(dataset_name: str, raw_path: str, split: str) -> Optional[Tuple[int, Optional[int]]]:
    """The z-range of a volume that holds both tuning and test data: the tuning slab for 'val', its complement
    for 'test', so the scored region never overlaps the validation region. None where the two are separate files.
    """
    if dataset_name == "platynereis_nuclei":
        return platynereis_nuclei_val_z_range(raw_path) if split == "val" else None
    slab = LM_VAL_Z_SLABS.get(dataset_name, {}).get(os.path.basename(raw_path))
    if slab is None:
        return None
    if split == "val":
        return slab.start, slab.stop
    # The slabs sit at one end of the volume, so the complement is one contiguous range.
    return (slab.stop, None) if slab.start == 0 else (0, slab.start)


def _sorted_pairs(raw_paths, label_paths) -> Tuple[List[str], List[str]]:
    """Sort raw and label paths as pairs.

    Sorting the two lists on their own breaks the pairing whenever the label names sort differently,
    which happens when one image name is a prefix of another, e.g. 'x_1.tif' and 'x_11.tif' with
    labels 'x_1_masks.tif' and 'x_11_masks.tif'.
    """
    if len(raw_paths) != len(label_paths):
        raise RuntimeError(
            f"Expect as many raw as label paths, got {len(raw_paths)} and {len(label_paths)}."
        )
    pairs = sorted(zip(raw_paths, label_paths), key=lambda pair: str(pair[0]))
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _tiles_from_stack(stack_path: str, raw_key: str, label_key: str, out_dir: str) -> List[str]:
    """Write every tile of a stacked h5 (raw (C, N, H, W), labels (N, H, W)) into its own h5 file once.

    The evaluation scores one file per sample, so the stacked tile sets of pannuke and lizard_mitosis are
    unpacked next to the data. Each tile file holds 'raw' (C, H, W) and 'labels' (H, W).
    """
    import h5py

    with h5py.File(stack_path, "r") as f:
        n_tiles = f[label_key].shape[0]
    paths = [os.path.join(out_dir, f"tile_{i:05d}.h5") for i in range(n_tiles)]
    if all(os.path.exists(path) for path in paths):
        return paths
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(stack_path, "r") as f:
        raw, labels = f[raw_key], f[label_key]
        for i, path in enumerate(paths):
            if os.path.exists(path):
                continue
            with h5py.File(path, "w") as g:
                g.create_dataset("raw", data=raw[:, i], compression="gzip")
                g.create_dataset("labels", data=labels[i], compression="gzip")
    return paths


def _frames_from_movie(raw_path: str, label_path: str, out_dir: str, stride: int) -> List[str]:
    """Write every stride-th frame of a 2d+t tif pair into its own h5 file once ('raw' and 'labels').

    Consecutive frames of a time-lapse are near-duplicates, so the evaluation scores a regular subset of them.
    """
    import h5py
    import tifffile

    # One cardioblast movie carries more raw than label frames, so the frame range follows the labels.
    with tifffile.TiffFile(raw_path) as f, tifffile.TiffFile(label_path) as g:
        n_frames = min(f.series[0].shape[0], g.series[0].shape[0])
    frames = range(0, n_frames, stride)
    paths = [os.path.join(out_dir, f"frame_{t:04d}.h5") for t in frames]
    if all(os.path.exists(path) for path in paths):
        return paths
    os.makedirs(out_dir, exist_ok=True)
    raw, labels = tifffile.imread(raw_path), tifffile.imread(label_path)
    for t, path in zip(frames, paths):
        if os.path.exists(path):
            continue
        with h5py.File(path, "w") as g:
            g.create_dataset("raw", data=raw[t], compression="gzip")
            g.create_dataset("labels", data=labels[t], compression="gzip")
    return paths


def _tiles_from_slide(
    slide_path: str, keys: Tuple[str, ...], count_key: str, out_dir: str, n_tiles: int,
    tile_shape: Tuple[int, int] = (1024, 1024), min_instances: int = 20, seed: int = 42,
) -> List[str]:
    """Cut n_tiles windows with at least min_instances labelled objects out of a whole-slide h5 once.

    The windows lie on a grid of tile_shape and are drawn in a fixed random order, so the same tiles are scored
    every time. Every key of the slide is copied into the tile file under the same name.
    """
    import h5py

    paths = [os.path.join(out_dir, f"tile_{i:03d}.h5") for i in range(n_tiles)]
    if all(os.path.exists(path) for path in paths):
        return paths
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(slide_path, "r") as f:
        height, width = f[count_key].shape[-2:]
        th, tw = tile_shape
        grid = [(y, x) for y in range(0, height - th + 1, th) for x in range(0, width - tw + 1, tw)]
        np.random.default_rng(seed).shuffle(grid)
        written = 0
        for y, x in grid:
            if written == n_tiles:
                break
            window = (slice(y, y + th), slice(x, x + tw))
            labels = f[count_key][window]
            if len(np.unique(labels)) - 1 < min_instances:
                continue
            with h5py.File(paths[written], "w") as g:
                for key in keys:
                    source = f[key]
                    data = source[(slice(None),) + window] if source.ndim == 3 else source[window]
                    g.create_dataset(key, data=data, compression="gzip")
            written += 1
    if written < n_tiles:
        raise RuntimeError(f"Only {written} of {n_tiles} tiles with {min_instances} objects found in {slide_path}.")
    return paths


# The fluorescence images of the NeurIPS CellSeg 2022 Tuning (val) and public Testing splits, checked by eye. The
# remaining images form the 'label_free' part, which also holds the stained brightfield blood smears next to the
# unstained brightfield, phase contrast and DIC images.
NEURIPS_FLUORESCENCE_IMAGES = {
    "val": tuple(f"cell_{i:05d}.png" for i in range(43, 70))
    + ("cell_00071.tif", "cell_00072.tif", "cell_00073.tif", "cell_00100.tif", "cell_00101.tif"),
    "test": (
        "OpenTest_001.png", "OpenTest_006.png", "OpenTest_013.png", "OpenTest_014.png", "OpenTest_015.tif",
        "OpenTest_016.png", "OpenTest_017.png", "OpenTest_019.tif", "OpenTest_021.png", "OpenTest_023.png",
        "OpenTest_026.tif", "OpenTest_028.tif", "OpenTest_031.png", "OpenTest_035.tif", "OpenTest_041.png",
        "OpenTest_044.png", "OpenTest_045.png", "OpenTest_046.tif", "OpenTest_047.tif",
    ),
}


# The volumes of our BlastoSPIM copy (250 of the 653 released) that the authors' official split archives put in the
# test (low and moderate SNR, both releases) and validation sets; the remaining 200 are official training volumes.
BLASTOSPIM_TEST_VOLUMES = (
    "Blast_074", "Blast_075", "F11_070", "F24_002", "F24_010", "F25_008", "F27_009", "F27_010", "F29_003",
    "F29_004", "F2_012", "F30_004", "F30_008", "F30_009", "F33_067", "F34_073", "F39_117", "F40_136", "F44_087",
    "F49_148", "F9_071", "H1_006", "H2_016", "H3_002", "H4_012", "H5_007", "H7_004", "H7_008", "H8_016", "H8_021",
    "H9_008", "M10_015", "M14_020", "M3_008", "M4_012", "M6_021", "M7_000", "M7_007", "M8_015", "M8_016",
)
BLASTOSPIM_VAL_VOLUMES = (
    "Blast_022", "F19_067", "F30_001", "F32_052", "F38_105", "F38_109", "F41_053", "F42_065", "F46_107", "M6_011",
)

# The PCNS patch ids whose TCGA patient also provides a MoNuSeg training image (pcns_crosswalk.txt: patients
# TCGA-38-6178, TCGA-49-4488, TCGA-CH-5767, TCGA-G2-A2EK, TCGA-G9-6336 and TCGA-G9-6363), left out of the OOD test.
PCNS_MONUSEG_PATCHES = (41, 42, 498, 511, 512, 794, 821, 822)

# The Lizard source cohorts that are independent of our other training data, see the lizard path resolver.
LIZARD_SCORED_SOURCES = ("crag", "dpath", "glas")


def deepbacs_is_fluorescence(path: str) -> bool:
    """Whether a DeepBacs 'mixed' image is fluorescence: the Nile Red S. aureus and the B. subtilis families."""
    name = os.path.basename(path)
    return name.endswith("_NR.tif") or name.startswith(("train", "test"))


def _loader_val_part(raw_paths, label_paths, split):
    """The random 10 % (seed 42) of a training split the generalist loader validates on, or the rest of it."""
    train_r, val_r, train_l, val_l = train_test_split(raw_paths, label_paths, test_size=0.1, random_state=42)
    return (val_r, val_l) if split == "val" else (train_r, train_l)


def _held_out_part(raw_paths, label_paths, split):
    """The blind test part (or the validation part) of the loader's 80 / 10 / 10 split of a dataset."""
    (_, val_r, test_r), (_, val_l, test_l) = _train_val_test_split(raw_paths, label_paths)
    return (val_r, val_l) if split == "val" else (test_r, test_l)


# Channels the 2d light microscopy models see, as indices into the stored channel-last image. A single index gives
# a grayscale image. Datasets not listed are fed as stored; a two-channel image gets a zero third channel, which is
# the TissueNet layout pan_multiplex trains with.
LM_IMAGE_CHANNELS = {
    "cvz_fluo_cell": (0, 2, 1), "cvz_fluo_dapi": (0,), "enseg": (1,), "xenium_cells": (1, 2, 3),
    "hela_cytonuc": (2,), "micro_bench": (2,), "spatch_dapi": (0,), "bac_mother": (0,),
}


def select_channels(image: np.ndarray, dataset_name: str) -> np.ndarray:
    """Pick the channels of a channel-last 2d image the model was trained on, see LM_IMAGE_CHANNELS."""
    channels = LM_IMAGE_CHANNELS.get(dataset_name)
    if channels is None or image.ndim == 2:
        if image.ndim == 3 and image.shape[-1] == 2:
            image = np.concatenate([image, np.zeros_like(image[..., :1])], axis=-1)
        return image
    image = image[..., list(channels)]
    return image[..., 0] if len(channels) == 1 else image


def _get_hp_data_paths(
    dataset_name: str, p: str, download: bool, split: str
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """The histopathology datasets. 'test' is the official test split, see DATASETS_2D_HP_ID."""
    if dataset_name == "cpm17":
        img, gt = datasets.cpm.get_cpm_paths(
            path=os.path.join(p, "cpm17"), data_choice="cpm17", split=split, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "glysac":
        paths = datasets.glysac.get_glysac_paths(path=os.path.join(p, "glysac"), split=split, download=download)
        return sorted(paths), sorted(paths), "raw", "labels/instances"

    if dataset_name == "histo_miner":
        img, gt = datasets.histo_miner.get_histo_miner_paths(
            path=os.path.join(p, "histo_miner"), split=split, task="nuclei", label_choice="instances",
            download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "lizard":
        paths = datasets.lizard.get_lizard_paths(path=os.path.join(p, "lizard"), split=split, download=download)
        # The test split also holds CoNSeP images and images stitched from PanNuke tiles of all folds, both of which
        # train (CoNSeP directly, PanNuke folds 1 and 2), so only the crag, dpath and glas images are scored.
        paths = [path for path in paths if os.path.basename(path).split("_")[0] in LIZARD_SCORED_SOURCES]
        return sorted(paths), sorted(paths), "image", "labels/segmentation"

    if dataset_name == "lizard_mitosis":
        stack = datasets.lizard_mitosis.get_lizard_mitosis_paths(
            path=os.path.join(p, "lizard_mitosis"), subset="mitosis", split=split, download=download,
        )
        paths = _tiles_from_stack(
            stack, "raw", "labels/instances", os.path.join(p, "lizard_mitosis", "mitosis", f"{split}_tiles")
        )
        return paths, paths, "raw", "labels"

    if dataset_name in ("lynsec_he", "lynsec_ihc"):
        choice = "h&e" if dataset_name == "lynsec_he" else "ihc"
        img, gt = datasets.lynsec.get_lynsec_paths(
            path=os.path.join(p, "lynsec"), split=split, choice=choice, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "monuseg":
        img, gt = datasets.monuseg.get_monuseg_paths(path=os.path.join(p, "monuseg"), split=split, download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "pannuke":
        # fold_3 is the blind benchmark; tuning uses the fold_2 tiles the loader validates on.
        fold = "fold_3" if split == "test" else "fold_2"
        stack = datasets.pannuke.get_pannuke_paths(path=os.path.join(p, "pannuke"), folds=[fold], download=download)
        paths = _tiles_from_stack(
            stack[0], "images", "labels/instances", os.path.join(p, "pannuke", f"{fold}_tiles")
        )
        if split != "test":
            paths = paths[PANNUKE_FOLD2_VAL_TILES]
        return paths, paths, "raw", "labels"

    if dataset_name == "puma":
        paths = datasets.puma.get_puma_paths(
            path=os.path.join(p, "puma"), split=split, annotations="nuclei", download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels/instances/nuclei"

    if dataset_name == "srsanet":
        img, gt = datasets.srsanet.get_srsanet_paths(path=os.path.join(p, "srsanet"), split=split, download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "tnbc_celltype":
        paths = datasets.tnbc_celltype.get_tnbc_celltype_paths(
            path=os.path.join(p, "tnbc_celltype"), split=split, download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels/instances"

    if dataset_name == "cellbindb_he":
        img, gt = _held_out_part(*datasets.cellbindb.get_cellbindb_paths(
            path=os.path.join(p, "cellbindb"), data_choice=CELLBINDB_HE_STAIN, download=download,
        ), split)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "cytodark0":
        paths = datasets.cytodark0.get_cytodark0_paths(
            path=os.path.join(p, "cytodark0"), split=split, download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels/instances"

    if dataset_name == "deepliif":
        paths = datasets.deepliif.get_deepliif_paths(path=os.path.join(p, "deepliif"), split=split, download=download)
        return sorted(paths), sorted(paths), "raw/ihc", "labels/instances"

    if dataset_name == "khoshdeli":
        img, gt = datasets.khoshdeli.get_khoshdeli_paths(path=os.path.join(p, "khoshdeli"), download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "panoptils":
        img, gt = datasets.panoptils.get_panoptils_paths(
            path=os.path.join(p, "panoptils"), label_choice="instances", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "pcns":
        paths = datasets.pcns.get_pcns_paths(path=os.path.join(p, "pcns"), split=split, download=download)
        # Patches cut from the TCGA slides or patients of MoNuSeg training images, see PCNS_MONUSEG_PATCHES.
        paths = [path for path in paths if int(os.path.splitext(os.path.basename(path))[0]) not in PCNS_MONUSEG_PATCHES]
        return sorted(paths), sorted(paths), "raw", "labels/instances"

    raise ValueError(f"Unknown histopathology dataset: {dataset_name!r}")


def _get_2d_data_paths(
    dataset_name: str, data_root: str, download: bool = False, split: str = "test"
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    p = data_root

    if dataset_name in DATASETS_HP:
        return _get_hp_data_paths(dataset_name, p, download, split)

    if dataset_name == "astih":
        # The 4 official test images; tuning uses the loader's 80/20 val part of the 22 train images.
        if split == "test":
            paths = datasets.electron_microscopy.astih.get_astih_paths(
                path=os.path.join(p, "astih"), name=ASTIH_SUBSETS, split="test", download=download,
            )
        else:
            train_paths = datasets.electron_microscopy.astih.get_astih_paths(
                path=os.path.join(p, "astih"), name=ASTIH_SUBSETS, split="train", download=download,
            )
            paths = train_test_split(train_paths, test_size=0.2, random_state=42)[1]
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "tumor_spheroid":
        paths, raw_key, label_key = datasets.electron_microscopy.tumor_spheroid_em.get_tumor_spheroid_paths(
            os.path.join(p, "tumor_spheroid_em"), source="2d_manual", resolution="50-50-50", target="cells",
            download=download,
        )
        names = TUMOR_SPHEROID_TEST_SLICES if split == "test" else TUMOR_SPHEROID_VAL_SLICES
        paths = [path for path in paths if os.path.basename(path) in names]
        return paths, paths, raw_key, label_key

    return _get_2d_lm_data_paths(dataset_name, p, download, split)


def _get_2d_lm_data_paths(
    dataset_name: str, p: str, download: bool, split: str
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """The 2d light microscopy datasets. 'test' is the blind split, 'val' the loader's validation data."""
    lm = datasets.light_microscopy

    if dataset_name == "livecell":
        img, gt = _get_livecell_paths(input_folder=os.path.join(p, "livecell"), split=split)
        img, gt = drop_excluded_livecell(img, gt)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name in ("cvz_fluo_cell", "cvz_fluo_dapi"):
        img, gt = lm.cvz_fluo.get_cvz_fluo_paths(
            path=os.path.join(p, "cvz"), stain_choice=dataset_name.split("_")[-1], download=download,
        )
        groups = CVZ_TEST_GROUPS if split == "test" else CVZ_VAL_GROUPS
        keep = [cvz_group(path) in groups for path in img]
        img = [path for path, k in zip(img, keep) if k]
        gt = [path for path, k in zip(gt, keep) if k]
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "dsb":
        # The StarDist fluorescence test split is blind; tuning uses the loader's 10 % of the remaining images.
        if split == "test":
            img, gt = lm.dsb.get_dsb_paths(
                path=os.path.join(p, "dsb"), source="reduced", split="test", download=download,
            )
        else:
            img, gt = _loader_val_part(*dsb_fluorescence_training_paths(os.path.join(p, "dsb")), split)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "tissuenet":
        paths = lm.tissuenet.get_tissuenet_paths(path=os.path.join(p, "tissuenet"), split=split, download=download)
        return sorted(paths), sorted(paths), "raw/rgb", "labels/cell"

    if dataset_name.startswith("omnipose_"):
        choice = dataset_name[len("omnipose_"):]
        if split == "test":
            img, gt = lm.omnipose.get_omnipose_paths(
                path=os.path.join(p, "omnipose"), split="test", data_choice=choice, download=download,
            )
        else:
            img, gt = _loader_val_part(*lm.omnipose.get_omnipose_paths(
                path=os.path.join(p, "omnipose"), split="train", data_choice=choice, download=download,
            ), split)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name in ("neurips_cellseg_fluorescence", "neurips_cellseg_label_free"):
        img, gt = lm.neurips_cell_seg.get_neurips_cellseg_paths(
            root=os.path.join(p, "neurips_cellseg"), split=split, download=download,
        )
        img, gt = _sorted_pairs(img, gt)
        fluorescence = NEURIPS_FLUORESCENCE_IMAGES[split]
        want_fluorescence = dataset_name == "neurips_cellseg_fluorescence"
        keep = [(os.path.basename(path) in fluorescence) == want_fluorescence for path in img]
        img = [path for path, k in zip(img, keep) if k]
        gt = [path for path, k in zip(gt, keep) if k]
        return img, gt, None, None

    if dataset_name == "dememseg":
        paths = lm.dememseg.get_dememseg_paths(path=os.path.join(p, "dememseg"), split=split, download=download)
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "flywing":
        img, gt = lm.flywing.get_flywing_paths(path=os.path.join(p, "flywing"), split=split, download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "enseg":
        animals = ENSEG_TEST_ANIMALS if split == "test" else ENSEG_VAL_ANIMALS
        img, gt = lm.enseg.get_enseg_paths(path=os.path.join(p, "enseg"), animal_tags=list(animals), download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "pan_multiplex":
        paths = []
        for subset in ("codex_colon", "mibi_breast", "mibi_decidua", "vectra_colon", "vectra_pancreas"):
            paths.extend(lm.pan_multiplex.get_pan_multiplex_paths(
                path=os.path.join(p, "pan_multiplex"), subset=subset, split=split, download=download,
            ))
        return sorted(paths), sorted(paths), ("raw/membrane", "raw/nuclei"), "labels/cell"

    if dataset_name in ("xenium_cells", "xenium_nuclei"):
        # Whole slides are scored on a fixed set of 1024x1024 tiles with at least 20 nuclei, cut once per slide.
        samples = XENIUM_TEST_SAMPLES if split == "test" else XENIUM_VAL_SAMPLES
        paths = []
        for sample in samples:
            slide = lm.xenium.get_xenium_paths(path=os.path.join(p, "xenium"), sample=[sample], download=download)[0]
            paths.extend(_tiles_from_slide(
                slide, ("raw/dapi", "raw/stack", "labels/nuclei", "labels/cells"), "labels/nuclei",
                os.path.join(EVAL_CACHE_ROOT, "xenium", sample), n_tiles=60 if split == "test" else 20,
            ))
        if dataset_name == "xenium_cells":
            return paths, paths, "raw/stack", "labels/cells"
        return paths, paths, "raw/dapi", "labels/nuclei"

    if dataset_name == "bitdepth_nucseg":
        img, gt = [], []
        for magnification in BITDEPTH_MAGNIFICATIONS:
            i, g = _held_out_part(*lm.bitdepth_nucseg.get_bitdepth_nucseg_paths(
                path=os.path.join(p, "bitdepth_nucseg"), magnification=magnification, download=download,
            ), split)
            img.extend(i)
            gt.extend(g)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "bmgd":
        paths = lm.bmgd.get_bmgd_paths(path=os.path.join(p, "bmgd"), download=download)
        paths, _ = _held_out_part(paths, paths, split)
        return sorted(paths), sorted(paths), "raw", "labels/instances"

    if dataset_name == "cellbindb":
        img, gt = [], []
        for stain in CELLBINDB_STAINS:
            i, g = _held_out_part(*lm.cellbindb.get_cellbindb_paths(
                path=os.path.join(p, "cellbindb"), data_choice=stain, download=download,
            ), split)
            img.extend(i)
            gt.extend(g)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "dynamicnuclearnet":
        paths = lm.dynamicnuclearnet.get_dynamicnuclearnet_paths(
            path=os.path.join(p, "dynamicnuclearnet"), split=split, download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "u20s":
        img, gt = _held_out_part(*lm.u20s.get_u20s_paths(path=os.path.join(p, "u20s"), download=download), split)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "ifnuclei":
        if split == "test":
            img, gt = lm.ifnuclei.get_ifnuclei_paths(path=os.path.join(p, "ifnuclei"), split="test", download=download)
        else:
            img, gt = _loader_val_part(*lm.ifnuclei.get_ifnuclei_paths(
                path=os.path.join(p, "ifnuclei"), split="train", download=download,
            ), split)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "tsakiroglou":
        tsakiroglou = datasets.histopathology.tsakiroglou
        if split == "test":
            img, gt = tsakiroglou.get_tsakiroglou_paths(os.path.join(p, "tsakiroglou"), split="test", download=download)
        else:
            img, gt = _loader_val_part(
                *tsakiroglou.get_tsakiroglou_paths(os.path.join(p, "tsakiroglou"), split="train", download=download),
                split,
            )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name in ("deepbacs_label_free", "deepbacs_fluorescence"):
        # The 'mixed' archive pools S. aureus in brightfield and Nile Red fluorescence, B. subtilis in membrane
        # fluorescence and E. coli in phase contrast; the file name families tell the modalities apart.
        img_folder, label_folder = lm.deepbacs.get_deepbacs_paths(
            path=os.path.join(p, "deepbacs"), bac_type="mixed", split="test" if split == "test" else "train",
            download=download,
        )
        img = sorted(glob(os.path.join(img_folder, "*.tif")))
        gt = sorted(glob(os.path.join(label_folder, "*.tif")))
        if split != "test":
            img, gt = _loader_val_part(img, gt, split)
        keep = [deepbacs_is_fluorescence(path) == (dataset_name == "deepbacs_fluorescence") for path in img]
        img = [path for path, k in zip(img, keep) if k]
        gt = [path for path, k in zip(gt, keep) if k]
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "orgasegment":
        img, gt = lm.orgasegment.get_orgasegment_paths(
            path=os.path.join(p, "orgasegment"), split="eval" if split == "test" else "val", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "organoidnet":
        img, gt = lm.organoidnet.get_organoidnet_paths(
            path=os.path.join(p, "organoidnet"), split="Test" if split == "test" else "Validation", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "yeaz":
        img, gt = lm.yeaz.get_yeaz_paths(path=os.path.join(p, "yeaz"), choice="bf", split=split, download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "bccd":
        if split == "test":
            paths = lm.bccd.get_bccd_paths(path=os.path.join(p, "bccd"), split="test", download=download)
        else:
            paths = lm.bccd.get_bccd_paths(path=os.path.join(p, "bccd"), split="train", download=download)
            paths, _ = _loader_val_part(paths, paths, split)
        return sorted(paths), sorted(paths), "raw", "labels/instances"

    if dataset_name == "cell_acdc":
        # One movie per split; every 20th frame is scored.
        movies = CELL_ACDC_TEST_MOVIES if split == "test" else CELL_ACDC_VAL_MOVIES
        img, gt = lm.cell_acdc.get_cell_acdc_paths(path=os.path.join(p, "cell_acdc"), download=download)
        paths = []
        for raw_path, label_path in zip(img, gt):
            movie = cell_acdc_movie(raw_path)
            if movie in movies:
                paths.extend(_frames_from_movie(
                    raw_path, label_path, os.path.join(EVAL_CACHE_ROOT, "cell_acdc", movie.replace("/", "_")),
                    stride=20,
                ))
        return paths, paths, "raw", "labels"

    if dataset_name == "cellular":
        wells = CELLULAR_TEST_WELLS if split == "test" else CELLULAR_VAL_WELLS
        paths = lm.cellular.get_cellular_paths(path=os.path.join(p, "cellular"), download=download)
        paths = [path for path in paths if os.path.basename(path).split("_")[3] in wells]
        return sorted(paths), sorted(paths), "raw/brightfield", "labels/instances"

    if dataset_name == "cisd":
        slides = CISD_TEST_SLIDES if split == "test" else CISD_VAL_SLIDES
        img, gt = lm.cisd.get_cisd_paths(path=os.path.join(p, "cisd"), mode="center_slice", download=download)
        keep = [os.path.basename(path).split("_")[0] in slides for path in img]
        img = [path for path, k in zip(img, keep) if k]
        gt = [path for path, k in zip(gt, keep) if k]
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "vicar":
        img, gt = [], []
        for cell_type in lm.vicar.VALID_CELL_TYPES:
            i, g = _held_out_part(*lm.vicar.get_vicar_paths(
                path=os.path.join(p, "vicar"), cell_types=[cell_type], download=download,
            ), split)
            img.extend(i)
            gt.extend(g)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "microbeseg":
        img, gt = lm.microbeseg.get_microbeseg_paths(
            path=os.path.join(p, "microbeseg"), split=split, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "orgline":
        paths = lm.orgline.get_orgline_paths(path=os.path.join(p, "orgline"), split=split, download=download)
        return sorted(paths), sorted(paths), "image", "masks"

    if dataset_name == "organoid":
        paths = []
        for source in ORGANOID_SOURCES:
            paths.extend(lm.organoid.get_organoid_paths(
                path=os.path.join(p, "organoid"), split=split, source=source, download=download,
            ))
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "mcellseg":
        img, gt = _held_out_part(
            *lm.mcellseg.get_mcellseg_paths(path=os.path.join(p, "mcellseg"), download=download), split
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "toiam":
        # One movie per split; every 20th frame is scored.
        movies = TOIAM_TEST_MOVIES if split == "test" else TOIAM_VAL_MOVIES
        img, gt = lm.toiam.get_toiam_paths(path=os.path.join(p, "toiam"), download=download)
        img, gt = _sorted_pairs(img, gt)
        keep = [os.path.basename(os.path.dirname(path)) in movies for path in img]
        img = [path for path, k in zip(img, keep) if k][::20]
        gt = [path for path, k in zip(gt, keep) if k][::20]
        return img, gt, None, None

    if dataset_name == "bbbc030":
        img, gt = lm.bbbc030.get_bbbc030_paths(path=os.path.join(p, "bbbc030"), split=split, download=download)
        return (*_sorted_pairs(img, gt), "raw", "labels")

    if dataset_name in ("covid_if_cells", "covid_if_nuclei"):
        # 49 samples with no split anywhere; the last 5 are reserved for tuning, the first 44 are scored.
        sample_range = (44, 49) if split == "val" else (0, 44)
        paths = lm.covid_if.get_covid_if_paths(
            path=os.path.join(p, "covid_if"), sample_range=sample_range, download=download,
        )
        if dataset_name == "covid_if_cells":
            return sorted(paths), sorted(paths), "raw/serum_IgG/s0", "labels/cells/s0"
        return sorted(paths), sorted(paths), "raw/nuclei/s0", "labels/nuclei/s0"

    if dataset_name == "medussa":
        paths = lm.medussa.get_medussa_paths(path=os.path.join(p, "medussa"), split=split, download=download)
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "hpa":
        paths = lm.hpa.get_hpa_segmentation_paths(path=os.path.join(p, "hpa"), split="val", download=download)
        return sorted(paths), sorted(paths), HPA_CHANNELS, "labels"

    if dataset_name == "cardioblast_nuclei":
        # Time-lapse projections of 350 frames; every 35th frame is scored.
        img, gt = lm.cardioblast_nuclei.get_cardioblast_nuclei_paths(
            path=os.path.join(p, "cardioblast_nuclei"), split=split, download=download,
        )
        paths = []
        for raw_path, label_path in zip(img, gt):
            movie = os.path.splitext(os.path.basename(raw_path))[0]
            paths.extend(_frames_from_movie(
                raw_path, label_path, os.path.join(EVAL_CACHE_ROOT, "cardioblast_nuclei", split, movie), stride=35,
            ))
        return paths, paths, "raw", "labels"

    if dataset_name == "hela_cytonuc":
        img, gt = lm.hela_cytonuc.get_hela_cytonuc_paths(
            path=os.path.join(p, "hela_cytonuc"), split=split, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "arvidsson":
        img, gt = lm.arvidsson.get_arvidsson_paths(path=os.path.join(p, "arvidsson"), split=split, download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "mndino":
        # The nuclei are scored; the disjoint micronuclei labels are left out.
        paths = lm.mndino.get_mndino_paths(path=os.path.join(p, "mndino"), split=split, download=download)
        return sorted(paths), sorted(paths), "raw", "labels/nuclei"

    if dataset_name == "micro_bench":
        img, gt = lm.micro_bench.get_micro_bench_paths(
            path=os.path.join(p, "micro_bench"), source="protein_localization_nuclei", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "spatch_dapi":
        paths = datasets.histopathology.spatch.get_spatch_paths(
            path=os.path.join(p, "spatch"), subset=SPATCH_DAPI_SUBSETS, download=download,
        )
        return sorted(paths), sorted(paths), "raw/rgb", "labels/nuclei"

    if dataset_name == "cellapp":
        # The per-cell-line subsets; HeLa has no training images. Tuning uses the RPE1 and U2OS train splits.
        sources = ("rpe1", "u2os", "hela") if split == "test" else ("rpe1", "u2os")
        paths = []
        for source in sources:
            paths.extend(lm.cellapp.get_cellapp_paths(
                path=os.path.join(p, "cellapp"), source=source, split=split, download=download,
            ))
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "deepseas":
        img, gt = lm.deepseas.get_deepseas_paths(path=os.path.join(p, "deepseas"), split=split, download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "dic_hepg2":
        img, gt = lm.dic_hepg2.get_dic_hepg2_paths(path=os.path.join(p, "dic_hepg2"), split=split, download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "yeastsam":
        img_folder, label_folder = lm.yeastsam.get_yeastsam_paths(path=os.path.join(p, "yeastsam"), download=download)
        img = sorted(glob(os.path.join(img_folder, "*.tif")))
        gt = sorted(glob(os.path.join(label_folder, "*.tif")))
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "yeastcellseg":
        paths = lm.yeastcellseg.get_yeastcellseg_paths(path=os.path.join(p, "yeastcellseg"), download=download)
        return sorted(paths), sorted(paths), "raw", "labels/instances"

    if dataset_name == "bac_mother":
        # Mother-machine frames of 345x41 pixels; every 10th frame is scored at its native size.
        img, gt = lm.bac_mother.get_bac_mother_paths(path=os.path.join(p, "bac_mother"), split=split, download=download)
        img, gt = _sorted_pairs(img, gt)
        return img[::10], gt[::10], None, None

    if dataset_name == "ecoli_microcolony_lineage":
        # Time-lapse frames of growing colonies; every 10th frame is scored.
        img, gt = lm.ecoli_microcolony_lineage.get_ecoli_microcolony_lineage_paths(
            path=os.path.join(p, "ecoli_microcolony_lineage"), genes=list(ECOLI_GENES), download=download,
        )
        img, gt = _sorted_pairs(img, gt)
        return img[::10], gt[::10], None, None

    raise ValueError(f"Unknown 2D light microscopy dataset: {dataset_name!r}")


def _get_3d_lm_data_paths(
    dataset_name: str, data_root: str, download: bool = False, split: str = "test"
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """The 3d light microscopy datasets. 'test' is the blind split, 'val' the loader's validation data.

    Datasets whose tuning data is a z-slab of the test volumes return the same files for both, see LM_VAL_Z_SLABS.
    """
    p = data_root
    lm = datasets.light_microscopy

    if dataset_name in ("plantseg_root", "plantseg_ovules"):
        name = dataset_name.split("_")[1]
        paths = lm.plantseg.get_plantseg_paths(
            path=os.path.join(p, "plantseg"), name=name, split=split, download=download,
        )
        return sorted(paths), sorted(paths), "raw", "label_with_ignore" if name == "ovules" else "label"

    if dataset_name == "pnas_arabidopsis":
        plants = PNAS_TEST_PLANTS if split == "test" else PNAS_VAL_PLANTS
        paths = lm.pnas_arabidopsis.get_pnas_arabidopsis_paths(
            path=os.path.join(p, "pnas_arabidopsis"), plants=plants, download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "cartocell":
        folder = CARTOCELL_TEST_FOLDER if split == "test" else CARTOCELL_VAL_FOLDER
        img = sorted(glob(os.path.join(p, "cartocell", "CartoCell", folder, "x", "*.tif")))
        gt = [path.replace(os.sep + "x" + os.sep, os.sep + "y" + os.sep) for path in img]
        return img, gt, None, None

    if dataset_name == "phmamm":
        timepoints = PHMAMM_TEST_TIMEPOINTS if split == "test" else PHMAMM_VAL_TIMEPOINTS
        img, gt = lm.phmamm.get_phmamm_paths(path=os.path.join(p, "phmamm"), timepoints=timepoints, download=download)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "wing_disc":
        paths = lm.wing_disc.get_wing_disc_paths(
            path=os.path.join(p, "wing_disc"), volumes=WING_DISC_TEST_VOLUMES, download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "embedseg_organoid":
        img, gt = lm.embedseg_data.get_embedseg_paths(
            path=os.path.join(p, "embedseg"), name="Mouse-Organoid-Cells-CBG", split="train", download=download,
        )
        part = EMBEDSEG_ORGANOID_TEST_TIMEPOINTS if split == "test" else EMBEDSEG_ORGANOID_VAL_TIMEPOINTS
        return img[part], gt[part], None, None

    if dataset_name in ("embedseg_mouse_skull", "embedseg_platy_ish"):
        name = "Mouse-Skull-Nuclei-CBG" if dataset_name == "embedseg_mouse_skull" else "Platynereis-ISH-Nuclei-CBG"
        img, gt = lm.embedseg_data.get_embedseg_paths(
            path=os.path.join(p, "embedseg"), name=name, split="test", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "embedseg_platy_nuclei":
        img, gt = lm.embedseg_data.get_embedseg_paths(
            path=os.path.join(p, "embedseg"), name="Platynereis-Nuclei-CBG", split="train", download=download,
        )
        part = EMBEDSEG_PLATY_NUCLEI_TEST_TIMEPOINTS if split == "test" else EMBEDSEG_PLATY_NUCLEI_VAL_TIMEPOINTS
        return img[part], gt[part], None, None

    if dataset_name == "nis3d":
        img, gt = lm.nis3d.get_nis3d_paths(
            path=os.path.join(p, "nis3d"), split="test", split_type="cross-image", download=download,
        )
        keep = ["Drosophila" in path for path in img]
        img = [path for path, k in zip(img, keep) if k]
        gt = [path for path, k in zip(gt, keep) if k]
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "celegans_atlas":
        img, gt = lm.celegans_atlas.get_celegans_atlas_paths(
            path=os.path.join(p, "celegans_atlas"), split=split, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "gonuclear":
        sample_ids = GONUCLEAR_TEST_SAMPLES if split == "test" else GONUCLEAR_VAL_SAMPLES
        paths = lm.gonuclear.get_gonuclear_paths(
            path=os.path.join(p, "gonuclear"), sample_ids=sample_ids, download=download,
        )
        return sorted(paths), sorted(paths), "raw/nuclei", "labels/nuclei"

    if dataset_name == "nucverse3d":
        root = os.path.join(p, "nucverse3d")
        if split == "test":
            paths = lm.nucverse3d.get_nucverse3d_paths(path=root, split="test", download=download)
        else:
            # One training volume per liver collection and a slab of one glia test volume, as in the loader.
            paths = []
            for name in ("liver", "liver_hcc"):
                paths += [
                    path for path in lm.nucverse3d.get_nucverse3d_paths(
                        path=root, dataset=name, split="train", download=download,
                    ) if os.path.basename(path) == NUCVERSE_VAL_VOLUMES[name]
                ]
            paths += [
                path for path in lm.nucverse3d.get_nucverse3d_paths(
                    path=root, dataset="drosophila_glia", split="test", download=download,
                ) if os.path.basename(path) == NUCVERSE_GLIA_VAL_VOLUME
            ]
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "cshaper":
        # Out of domain: the paper's evaluation samples are scored, its training samples are the tuning data.
        img, gt = lm.cshaper.get_cshaper_paths(
            path=os.path.join(p, "cshaper"), split="val" if split == "test" else "train", download=download,
        )
        return (*_sorted_pairs(img, gt), "raw", "labels")

    if dataset_name == "morphonet_celegans":
        # Every 10th of the 184 C. elegans timepoints.
        root = os.path.join(p, "morphonet")
        paths = lm.morphonet.get_morphonet_paths(path=root, organism="caenorhabditis_elegans", download=download)
        paths = sorted(paths)[::10]
        return paths, paths, "raw", "labels"

    if dataset_name == "parhyale_regen":
        paths = lm.parhyale_regen.get_parhyale_regen_paths(path=os.path.join(p, "parhyale_regen"), download=download)
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "vibrio_cholerae":
        img, gt = lm.vibrio_cholerae.get_vibrio_cholerae_paths(
            path=os.path.join(p, "vibrio_cholerae"), download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "mouse_embryo":
        # Out of domain: the official val split is scored, the train split is the tuning data.
        paths = lm.mouse_embryo.get_mouse_embryo_paths(
            path=os.path.join(p, "mouse_embryo"), name="nuclei", split="val" if split == "test" else "train",
            download=download,
        )
        return sorted(paths), sorted(paths), "raw", "label"

    if dataset_name == "blastospim":
        paths = lm.blastospim.get_blastospim_paths(path=os.path.join(p, "blastospim"), download=download)
        volumes = BLASTOSPIM_VAL_VOLUMES if split == "val" else BLASTOSPIM_TEST_VOLUMES
        paths = [path for path in paths if os.path.basename(path).split("_image_")[0] in volumes]
        return sorted(paths), sorted(paths), "raw", "labels"

    raise ValueError(f"Unknown 3D LM dataset: {dataset_name!r}")


def _get_3d_em_data_paths(
    dataset_name: str, data_root: str, download: bool = False, is_val: bool = False
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """The 3d EM datasets. 'is_val' selects the tuning files where they differ from the test files."""
    p = data_root
    em = datasets.electron_microscopy

    if dataset_name == "platynereis_nuclei":
        # The val split restricts to the 3 richest sample ids, see PLATYNEREIS_NUCLEI_VAL_SAMPLES.
        sample_ids = sorted(PLATYNEREIS_NUCLEI_VAL_SAMPLES) if is_val else None
        paths = datasets.platynereis.get_platynereis_paths(
            path=os.path.join(p, "platynereis"), sample_ids=sample_ids, name="nuclei", download=download,
        )
        return paths, paths, "volumes/raw", "volumes/labels/nucleus_instance_labels"

    if dataset_name == "platynereis_cells":
        # Volume 9 is the blind test set, volumes 7 and 8 validate; the neuropil carries the ignore label.
        sample_ids = [7, 8] if is_val else [9]
        paths = em.platynereis.prepare_platynereis_cell_data(
            os.path.join(p, "platynereis"), sample_ids=sample_ids, download=download,
        )
        return paths, paths, "volumes/raw/s1", em.platynereis.get_platynereis_cell_label_key()

    if dataset_name == "cremi":
        samples = ("A", "B") if is_val else ("C",)
        paths = datasets.cremi.get_cremi_paths(path=os.path.join(p, "cremi"), samples=samples, download=download)
        return sorted(paths), sorted(paths), "volumes/raw", "volumes/labels/neuron_ids"

    if dataset_name == "snemi":
        path = datasets.snemi.get_snemi_paths(path=os.path.join(p, "snemi"), sample="train", download=download)
        return [path], [path], "volumes/raw", "volumes/labels/neuron_ids"

    if dataset_name == "axonem":
        raw, labels = em.axonem.get_axonem_paths(
            path=os.path.join(p, "axonem"), samples=("human", "mouse"), download=download,
        )
        names = AXONEM_VAL_VOLUMES if is_val else AXONEM_TEST_VOLUMES
        keep = [i for i, path in enumerate(labels) if os.path.basename(path) in names]
        return [raw[i] for i in keep], [labels[i] for i in keep], "main", "main"

    if dataset_name == "fafb":
        boxes = FAFB_VAL_BOXES if is_val else FAFB_TEST_BOXES
        paths = em.fafb.get_fafb_paths(path=os.path.join(p, "fafb"), bounding_boxes=boxes, download=download)
        return paths, paths, "raw", "labels"

    if dataset_name == "fib25":
        raw, labels = em.fib25.get_fib25_paths(
            path=os.path.join(p, "fib25"), samples=(FIB25_TEST_SAMPLE,), download=download,
        )
        return raw, labels, "raw", "neuron_ids"

    if dataset_name == "hemibrain":
        paths = em.hemibrain.get_hemibrain_paths(path=os.path.join(p, "hemibrain"), download=download)
        return paths, paths, "raw", "labels"

    if dataset_name == "manc":
        paths = em.manc.get_manc_paths(path=os.path.join(p, "manc"), download=download)
        return paths, paths, "raw", "labels"

    if dataset_name == "malecns":
        boxes = MALECNS_VAL_BOXES if is_val else MALECNS_TEST_BOXES
        paths = em.malecns.get_malecns_paths(path=os.path.join(p, "malecns"), bounding_boxes=boxes, download=download)
        return paths, paths, "raw", "labels"

    if dataset_name == "wafer4":
        raw, labels = em.wafer4.get_wafer4_paths(path=os.path.join(p, "wafer4"), download=download)
        return [raw], [labels], "main", "main"

    if dataset_name == "minnie65":
        paths = em.microns.get_microns_minnie65_paths(
            path=os.path.join(p, "microns-minnie65"), split="val" if is_val else "test", download=download,
        )
        return paths, paths, "raw", "labels"

    if dataset_name == "zebrafinch_j0126":
        path = em.zebrafinch.get_zebrafinch_data(
            os.path.join(p, "zebrafinch"), bounding_box=ZEBRAFINCH_J0126_BOX, mip=0, dataset="j0126", download=download,
        )
        return [path], [path], "raw", "labels"

    if dataset_name == "zebrafinch_j0251":
        # The loader constants are mip-0 voxels; torch-em takes nm (10 x 10 x 25 nm at mip 0).
        boxes = ZEBRAFINCH_J0251_VAL_BOXES if is_val else ZEBRAFINCH_J0251_TEST_BOXES
        paths = [
            em.zebrafinch.get_zebrafinch_data(
                os.path.join(p, "zebrafinch"), bounding_box=tuple(v * r for v, r in zip(box, (10, 10, 10, 10, 25, 25))),
                mip=0, dataset="j0251", download=download,
            ) for box in boxes
        ]
        return paths, paths, "raw", "labels"

    if dataset_name == "wildenberg":
        paths = em.wildenberg.get_wildenberg_paths(
            path=os.path.join(p, "wildenberg2023"), experiments=("p105",), label_choice="saturated",
            bounding_box=WILDENBERG_P105_BOX, download=download,
        )
        return paths, paths, "raw", "labels"

    if dataset_name == "liconn":
        path = datasets.light_microscopy.liconn.get_liconn_paths(
            path=os.path.join(p, "liconn"), segmentation="proofread", download=download,
        )
        return [path], [path], "raw", "seg_proofread"

    if dataset_name == "xpress":
        raw, labels = datasets.light_microscopy.xpress.get_xpress_paths(
            path=os.path.join(p, "xpress"), download=download,
        )
        return [raw], [labels], "raw", "labels"

    if dataset_name == "nisb":
        # Synthetic OOD: the official test cube is scored, the official val cube tunes; nisb is not trained on.
        paths = em.nisb.get_nisb_paths(
            os.path.join(p, "nisb"), setting="base", split="val" if is_val else "test", download=download,
        )
        return paths, paths, "img", "seg"

    if dataset_name == "densecell":
        # The val volume is the blind test set; the top 15 sections of the train volume are the tuning data.
        path = em.densecell.get_densecell_paths(
            path=os.path.join(p, "densecell"), split="train" if is_val else "val", download=download,
        )
        em.densecell._add_cell_instances(path)
        return [path], [path], "raw", em.densecell.CELL_INSTANCE_KEY

    if dataset_name == "isbi2012":
        path = em.isbi2012.get_isbi_paths(path=os.path.join(p, "isbi2012"), download=download)
        return [path], [path], "raw", "labels/gt_segmentation"

    if dataset_name == "humanneurons":
        # The cached H01 crop. Resolved directly: it predates the torch-em loader for this dataset.
        paths = sorted(glob(os.path.join(p, "humanneurons", "*.h5")))
        return paths, paths, "raw", "labels"

    if dataset_name == "synapseweb":
        paths = sorted(glob(os.path.join(p, "synapseweb_hippocampus", "synapseweb_hippocampus_*.h5")))
        return paths, paths, "raw", "labels"

    raise ValueError(f"Unknown 3D EM dataset: {dataset_name!r}")


def get_data_paths(
    dataset_name: str, data_root: str, download: bool = False, split: str = "test"
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """Return (raw_paths, label_paths, raw_key, label_key) for a dataset's evaluation split.

    raw_key / label_key are None for plain image files and non-None for H5 / zarr.

    With split='val' this returns data held out from what the evaluation scores, which is what a
    parameter search has to run on, see VAL_SPLITS. Only the datasets listed there support it.
    """
    all_datasets = DATASETS_2D + DATASETS_3D
    assert dataset_name in all_datasets, (
        f"Unsupported dataset: '{dataset_name}'. Choose from {all_datasets}."
    )

    is_val = split == "val"
    if is_val:
        if dataset_name not in VAL_SPLITS:
            raise ValueError(
                f"There is no data held out from the evaluation for '{dataset_name}', so it cannot be "
                f"tuned on a validation split. Datasets that can: {sorted(VAL_SPLITS)}."
            )
        # None means the loader has no split of its own; the holdout is the z-slab in VAL_Z_RANGE
        # (or, for platynereis_nuclei, the sample_ids in PLATYNEREIS_NUCLEI_VAL_SAMPLES), which
        # load_volume / _get_3d_em_data_paths apply on top of the very same volumes.
        split = VAL_SPLITS[dataset_name] or "test"

    if dataset_name in DATASETS_2D:
        return _get_2d_data_paths(dataset_name, data_root, download=download, split=split)
    if dataset_name in DATASETS_3D_LM:
        return _get_3d_lm_data_paths(dataset_name, data_root, download=download, split=split)
    return _get_3d_em_data_paths(dataset_name, data_root, download=download, is_val=is_val)


def _center_crop_roi(shape, crop_shape):
    """Returns a tuple of slices for a center crop."""
    roi = []
    for s, c in zip(shape, crop_shape):
        c = min(c, s)
        start = (s - c) // 2
        roi.append(slice(start, start + c))
    return tuple(roi)


def _read_window(shape, roi, z_range, crop_shape):
    """Compose a dataset roi, a z-slab and the center crop into one tuple of slices on the stored array."""
    roi = tuple(slice(None) for _ in shape) if roi is None else roi
    starts, sizes = [], []
    for axis, (extent, sl) in enumerate(zip(shape, roi)):
        start, stop, _ = sl.indices(extent)
        if axis == 0 and z_range is not None:
            z_stop = stop if z_range[1] is None else min(stop, start + z_range[1])
            start, stop = start + z_range[0], z_stop
        starts.append(start)
        sizes.append(stop - start)
    crop = _center_crop_roi(sizes, crop_shape)
    return tuple(slice(start + c.start, start + c.stop) for start, c in zip(starts, crop))


def load_volume(
    raw_path: str,
    label_path: str,
    raw_key: Optional[str],
    label_key: Optional[str],
    dataset_name: str,
    crop_shape: Tuple[int, ...] = (8, 512, 512),
    ensure_8bit: bool = True,
    ensure_instances: bool = True,
    z_range: Optional[Tuple[int, int]] = None,
    split: str = "test",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load a 3D volume, apply dataset-specific preprocessing, and center-crop.

    valid_roi is a boolean mask that is True where the data is annotated. It is None except for the datasets
    that are annotated only in part (platynereis_nuclei, platynereis_cells, synapseweb).

    'split' selects the test or the tuning region of the EM volumes, see EM_ROIS. 'z_range' restricts the
    volume to a z-slab before the center crop, which is how a dataset without splits holds tuning data out
    of the evaluated slab. See VAL_Z_RANGE.
    """
    # Only the scored region is read: the dataset roi, the z-slab and the center crop are composed into one
    # window first, so the multi-gigavoxel connectome volumes are never loaded whole.
    raw_source = load_image(raw_path) if raw_key is None else open_file(raw_path, mode="r")[raw_key]
    label_source = load_image(label_path) if label_key is None else open_file(label_path, mode="r")[label_key]
    window = _read_window(label_source.shape, em_roi(dataset_name, label_path, split), z_range, crop_shape)
    raw, labels = np.asarray(raw_source[window]), np.asarray(label_source[window])

    valid_roi = None
    if dataset_name == "platynereis_nuclei":
        labels = labels.astype("int64")
        valid_roi = labels != -1
        labels[labels == -1] = 0
    elif dataset_name == "platynereis_cells":
        # The neuropil is not resolved into cells and carries the ignore label; it is excluded from scoring.
        ignore = labels == datasets.electron_microscopy.platynereis.CELL_IGNORE_LABEL
        valid_roi = ~ignore
        labels[ignore] = 0
    elif dataset_name == "synapseweb":
        # Annotation covers only part of the core, so the unlabelled voxels are excluded from scoring.
        valid_roi = labels != 0
    elif dataset_name == "plantseg_root":
        # Label 1 is the background, label 0 the unannotated deeper tissue.
        valid_roi = labels != 0
        labels[labels == 1] = 0
    elif dataset_name == "plantseg_ovules":
        labels = labels.astype("int64")
        valid_roi = labels != -1
        labels[labels == -1] = 0
    elif dataset_name == "pnas_arabidopsis":
        # The background carries id 1.
        labels[labels == 1] = 0

    if ensure_8bit:
        raw = normalize_raw(raw) * 255.0

    # Restrict to the annotated z-range. Interior empty slices stay, or the volume is not contiguous.
    annotated = np.any(labels != 0, axis=tuple(range(1, labels.ndim)))
    if annotated.any():
        z_start = int(np.argmax(annotated))
        z_stop = len(annotated) - int(np.argmax(annotated[::-1]))
        raw, labels = raw[z_start:z_stop], labels[z_start:z_stop]
        if valid_roi is not None:
            valid_roi = valid_roi[z_start:z_stop]

    if ensure_instances:
        labels = connected_components(labels)

    assert raw.shape == labels.shape, f"Shape mismatch: raw {raw.shape} vs labels {labels.shape}"
    return raw.astype("float32"), labels.astype("uint32"), valid_roi


_UNISAM2_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/automatic/v1"
UNISAM2_CHECKPOINT = os.path.join(_UNISAM2_ROOT, "checkpoints", "unisam2-both", "best.pt")


def get_joint_checkpoint(model_type: str, checkpoint: str = "best") -> str:
    """Return the joint trainer checkpoint for a model type, e.g. 'hvit_b'."""
    path = os.path.join(_joint_checkpoint_root(), f"joint_sam2_{model_type}_multi_gpu", f"{checkpoint}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"There is no joint '{checkpoint}' checkpoint for '{model_type}' at '{path}'.")
    return path


def _save_atomic(obj, path: str) -> None:
    """Save to a process-unique temporary file first, so concurrent jobs never read a partial file."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _strip_ddp_prefix(state_dict):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def checkpoint_checksum(path: str) -> str:
    """Return the xxh128 checksum of a checkpoint without loading it into memory."""
    checksum = xxhash.xxh128()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def combine_checkpoint_checksums(*checksums: str) -> str:
    """Combine the content checksums of all weights that determine one evaluation run."""
    if len(checksums) == 1:
        return checksums[0]
    combined = xxhash.xxh128()
    for checksum in checksums:
        combined.update(checksum.encode("ascii"))
        combined.update(b"\0")
    return combined.hexdigest()


def export_joint_checkpoint(
    model_type: str, checkpoint: str = "best", export_root: Optional[str] = None,
    source_checksum: Optional[str] = None,
) -> Tuple[str, str]:
    """Split a joint checkpoint into an interactive and an automatic weight file.

    The joint trainer bundles the SAM2 weights ('model_state'), the UniSAM2 decoder weights
    ('unetr_state') and pickled trainer state in a single file. That file cannot be loaded by
    `sam2.build_sam`, which reads `torch.load(...)['model']` with `weights_only=True`. Both
    exported files are plain tensor dicts, mirroring `scripts/model_export/export_sam2_cells_model.py`.

    The checksum in the name records which checkpoint an export came from, so an export is reused
    only for that content. Every training version has a 'best' checkpoint, so a plain
    'joint_sam2_hvit_t_best' would hand back the previous version's export instead.

    Args:
        model_type: The SAM2 backbone the model was finetuned from, e.g. 'hvit_b'.
        checkpoint: Which trainer checkpoint to export, 'best' or 'latest'.
        export_root: The directory the exported weight files are written to. Defaults to
            `MICRO_SAM2_JOINT_EXPORT_ROOT`, read fresh at call time.
        source_checksum: A previously computed checksum, to avoid reading the checkpoint twice.

    Returns:
        The paths to the interactive (SAM2) and the automatic (UniSAM2 decoder) weight files.
    """
    export_root = export_root or _joint_export_root()
    checkpoint_path = get_joint_checkpoint(model_type, checkpoint)
    source_checksum = source_checksum or checkpoint_checksum(checkpoint_path)
    name = f"joint_sam2_{model_type}_{checkpoint}_{source_checksum}"
    interactive_path = os.path.join(export_root, f"{name}.pt")
    decoder_path = os.path.join(export_root, f"{name}_decoder.pt")
    if os.path.exists(interactive_path) and os.path.exists(decoder_path):
        return interactive_path, decoder_path

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing = [key for key in ("model_state", "unetr_state") if key not in state]
    if missing:
        raise RuntimeError(f"'{checkpoint_path}' is not a joint checkpoint, it is missing {missing}.")

    os.makedirs(export_root, exist_ok=True)
    _save_atomic({"model": _strip_ddp_prefix(state["model_state"]), "model_type": model_type}, interactive_path)
    _save_atomic(_strip_ddp_prefix(state["unetr_state"]), decoder_path)
    print(f"Exported '{checkpoint_path}' to '{interactive_path}' and '{decoder_path}'.")
    return interactive_path, decoder_path


# Keep a volume's tracking state on the device: the same masks propagate 1.2-1.3x faster, for about
# 17 MB of device memory per slice. That is a batch job's to spend, which is why it is not the default.
VOLUME_SPEED_OPTIONS = {"offload_to_cpu": False}


DATASET_SPACING: dict = {
    # z/xy voxel ratios from published acquisition parameters
    "embedseg_mouse_skull": (4, 1, 1),  # z=1µm, xy=0.25µm
    "embedseg_organoid": (6, 1, 1),  # z=1µm, xy=0.1733µm
    "embedseg_platy_nuclei": (5, 1, 1),  # z=2.031µm, xy=0.406µm
    "blastospim": (10, 1, 1),  # SPIM: z≈2µm, xy≈0.208µm
    "mouse_embryo": (4, 1, 1),  # confocal: z≈1µm, xy≈0.22µm
    "densecell": (5, 1, 1),  # SBF-SEM: 50 nm sections, 10 nm pixels
    "nisb": (2.2, 1, 1),  # synthetic: 20 nm sections, 9 nm pixels
}


# The parameters `AutomaticPromptGenerator.generate` accepts, so a run can be described by one dict.
GENERATE_PARAM_KEYS = (
    "candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size",
    "score_threshold", "score_filter", "max_overlap", "min_size", "max_size_factor", "refinement",
    "refinement_kwargs", "multimasking", "multimask_scorer", "multimask_selection",
    "n_objects_per_pass", "early_stop_patience", "propagation_waves", "batch_size", "n_threads",
)


def resolve_params(overrides=None, ndim=2, model_type=None):
    """The generation parameters for one run, with 'overrides' applied on top of the library defaults.

    The single definition of what a run's parameters are, so that a benchmark, a walk-through and a
    sweep all describe the same run. The result is ready to pass to `generate` as keyword arguments.

    Args:
        overrides: The parameters to change, by the name `generate` gives them. A volume also accepts
            'candidate_threshold_3d', which is the name the defaults give its own threshold.
        ndim: The number of spatial dimensions, 2 or 3.
        model_type: The SAM2 backbone the defaults are looked up for, see `default_prompt_generation`.

    Returns:
        The parameters, keyed as `generate` takes them.
    """
    from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION, default_prompt_generation
    from micro_sam.v2.util import DEFAULT_MODEL

    overrides = overrides or {}
    model_type = model_type or DEFAULT_MODEL
    per_model_defaults = default_prompt_generation(model_type, is_volume=False)
    defaults = {**DEFAULT_PROMPT_GENERATION, **per_model_defaults}
    params = {key: defaults[key] for key in GENERATE_PARAM_KEYS}
    params.update(overrides)
    if ndim == 3:
        # A candidate's density scales with the object's size, so a volume has its own threshold.
        default_3d = default_prompt_generation(model_type, is_volume=True)["candidate_threshold"]
        params["candidate_threshold"] = overrides.get("candidate_threshold_3d", default_3d)
    params.pop("candidate_threshold_3d", None)
    return params


def _alias_micro_sam2_modules():
    """Alias the moved 'micro_sam2' modules so checkpoints pickled before the package move load."""
    import sys
    import types
    import micro_sam.v2.datasets.sampler as datasets_sampler
    import micro_sam.v2.datasets.wrapper as datasets_wrapper
    import micro_sam.v2.transforms.labels as transforms_labels
    import micro_sam.v2.transforms.raw as transforms_raw

    root = sys.modules.setdefault("micro_sam2", types.ModuleType("micro_sam2"))
    root.__path__ = []
    datasets_module = sys.modules.setdefault("micro_sam2.datasets", types.ModuleType("micro_sam2.datasets"))
    datasets_module.__path__ = []
    transforms = sys.modules.setdefault("micro_sam2.transforms", types.ModuleType("micro_sam2.transforms"))
    transforms.__path__ = []

    sys.modules["micro_sam2.datasets.sampler"] = datasets_sampler
    sys.modules["micro_sam2.datasets.wrapper"] = datasets_wrapper
    sys.modules["micro_sam2.transforms.labels"] = transforms_labels
    sys.modules["micro_sam2.transforms.raw"] = transforms_raw
    setattr(root, "datasets", datasets_module)
    setattr(root, "transforms", transforms)
    setattr(datasets_module, "sampler", datasets_sampler)
    setattr(datasets_module, "wrapper", datasets_wrapper)
    setattr(transforms, "labels", transforms_labels)
    setattr(transforms, "raw", transforms_raw)


def load_unisam2_model(checkpoint_path, device, encoder="hvit_t", encoder_model_type=None):
    """Load a UniSAM2 model for automatic segmentation.

    Handles the standalone UniSAM2 checkpoints ('model_state'), the joint checkpoints
    ('unetr_state', with the SAM2 encoder wrapped in an adapter) and exported decoder weights.

    Args:
        checkpoint_path: The filepath to the checkpoint.
        device: The torch device.
        encoder: The SAM2 backbone the decoder was trained on, e.g. 'hvit_b', or a prebuilt encoder
            module to reuse.
        encoder_model_type: The SAM2 backbone, when 'encoder' is a prebuilt module rather than a
            name. Sets 'model.model_type', which the postprocessing defaults are looked up by.

    Returns:
        The UniSAM2 model in eval mode.
    """
    from micro_sam.v2.instance_segmentation import get_unisam2_model
    _alias_micro_sam2_modules()
    return get_unisam2_model(checkpoint_path, device=device, encoder=encoder, encoder_model_type=encoder_model_type)


def build_apg_segmenter(
    model_type, ndim, device, joint_checkpoint="best", decoder_path=None, joint_checksum=None,
    interactive_checkpoint_path=None, export_root=None, devices=None,
):
    """Build the automatic prompt generator from both halves of a joint checkpoint.

    The decoder proposes the candidates and the interactive branch scores them, so a run needs both.
    A volume is propagated by the SAM2 video predictor, which is a different model input type.

    Args:
        model_type: The SAM2 backbone of the joint model, e.g. 'hvit_t'.
        ndim: The number of spatial dimensions, 2 or 3.
        device: The torch device.
        joint_checkpoint: The joint trainer checkpoint, without the '.pt' suffix.
        decoder_path: Decoder weights to use instead of the ones exported from the joint checkpoint.
        interactive_checkpoint_path: Standalone interactive weights (e.g. a downloaded registry
            checkpoint) to use instead of the interactive half exported from the joint checkpoint.
            Requires 'decoder_path' too, since there is then no joint checkpoint to export it from.
        export_root: Optional directory for the split checkpoint files. Defaults to
            `MICRO_SAM2_JOINT_EXPORT_ROOT`, read fresh at call time.
        devices: The devices the decoder, the scoring and the propagation spread over. All visible
            GPUs by default; pass a single device to pin the run to it.

    Returns:
        The prompt generator, built through the library factory that the CLI and the API use.
    """
    from micro_sam.v2.util import get_sam2_model
    from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator

    if interactive_checkpoint_path is not None:
        if decoder_path is None:
            raise ValueError("'interactive_checkpoint_path' requires 'decoder_path' too.")
        interactive_path, exported_decoder = interactive_checkpoint_path, None
    else:
        export_kwargs = {} if export_root is None else {"export_root": export_root}
        interactive_path, exported_decoder = export_joint_checkpoint(
            model_type, joint_checkpoint, source_checksum=joint_checksum, **export_kwargs
        )
    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=interactive_path,
        **({"input_type": "videos"} if ndim == 3 else {}),
    )
    decoder = load_unisam2_model(
        decoder_path or exported_decoder, device, encoder=model.image_encoder, encoder_model_type=model_type,
    )
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=ndim,
        inference_device=devices,
    )


def resolve_checkpoint_identity(
    mode, model_type, joint_checkpoint="best", checkpoint_path=None, interactive_checkpoint_path=None,
):
    """Return the content identity of the effective weights and, if needed, the joint checkpoint."""
    if interactive_checkpoint_path is not None:
        checksums = [checkpoint_checksum(interactive_checkpoint_path)]
        if checkpoint_path is not None:
            checksums.append(checkpoint_checksum(checkpoint_path))
        return combine_checkpoint_checksums(*checksums), None

    joint_checksum = None
    if checkpoint_path is None or mode == "apg":
        joint_path = get_joint_checkpoint(model_type, joint_checkpoint)
        joint_checksum = checkpoint_checksum(joint_path)

    if checkpoint_path is None:
        return joint_checksum, joint_checksum

    decoder_checksum = checkpoint_checksum(checkpoint_path)
    if mode == "apg":
        return combine_checkpoint_checksums(joint_checksum, decoder_checksum), joint_checksum
    return decoder_checksum, None


def build_model(
    mode, model_type, device, ndim, joint_checkpoint="best", checkpoint_path=None, joint_checksum=None,
    interactive_checkpoint_path=None, devices=None,
):
    """Load the model a mode runs on, from the two halves of the joint checkpoint.

    Args:
        mode: The segmentation mode, one of MODES.
        model_type: The SAM2 backbone of the joint model, e.g. 'hvit_t'.
        device: The torch device.
        ndim: The number of spatial dimensions, 2 or 3.
        joint_checkpoint: The joint trainer checkpoint, without the '.pt' suffix.
        checkpoint_path: Decoder weights to use instead of the ones exported from the joint checkpoint.
        joint_checksum: The checksum of the joint checkpoint, from `resolve_checkpoint_identity`.
        interactive_checkpoint_path: Standalone interactive weights for 'apg', bypassing the joint
            checkpoint entirely. See `build_apg_segmenter`.
        devices: The devices inference spreads over. All visible GPUs by default.

    Returns:
        The UniSAM2 decoder for 'ais', or the prompt generator for 'apg'.
    """
    if mode == "apg":
        return build_apg_segmenter(
            model_type, ndim, device, joint_checkpoint, decoder_path=checkpoint_path,
            joint_checksum=joint_checksum, interactive_checkpoint_path=interactive_checkpoint_path,
            devices=devices,
        )

    decoder_path = checkpoint_path or export_joint_checkpoint(
        model_type, joint_checkpoint, source_checksum=joint_checksum
    )[1]
    return load_unisam2_model(decoder_path, device, encoder=model_type)


def build_ais_model_from_checkpoint(joint_checkpoint_path, model_type="hvit_t", device="cuda", ndim=2, cache_dir=None):
    """Build the AIS (UniSAM2) model from a specific joint-trainer checkpoint file, by explicit path.

    Unlike `build_model`, which resolves a checkpoint through `_joint_checkpoint_root()` /
    `_joint_export_root()` and the `joint_sam2_{model_type}_multi_gpu` naming convention, this loads
    `joint_checkpoint_path` directly - handy for a one-off checkpoint that lives outside that
    directory layout (e.g. a checkpoint from an ad hoc experiment).

    Args:
        joint_checkpoint_path: Absolute path to a joint checkpoint (has 'model_state', 'unetr_state').
        model_type: The SAM2 backbone, e.g. 'hvit_t'.
        device: The torch device.
        ndim: The number of spatial dimensions, 2 or 3.
        cache_dir: Where to write the extracted decoder-only file. Defaults to a shared tmp dir.

    Returns:
        The UniSAM2 decoder in eval mode, loaded from exactly this checkpoint.
    """
    if cache_dir is None:
        cache_dir = "/tmp/micro_sam2_checkpoint_cache"
    os.makedirs(cache_dir, exist_ok=True)

    key = xxhash.xxh64(joint_checkpoint_path.encode()).hexdigest()
    decoder_path = os.path.join(cache_dir, f"{model_type}_{key}_decoder.pt")

    if not os.path.exists(decoder_path):
        state = torch.load(joint_checkpoint_path, map_location="cpu", weights_only=False)
        _save_atomic(_strip_ddp_prefix(state["unetr_state"]), decoder_path)

    return build_model(mode="ais", model_type=model_type, device=device, ndim=ndim, checkpoint_path=decoder_path)


def build_apg_model_from_checkpoint(joint_checkpoint_path, model_type="hvit_t", device="cuda", ndim=2, cache_dir=None):
    """Build the APG (prompt generator) from a specific joint-trainer checkpoint file, by explicit path.

    Mirrors `build_apg_segmenter`, but loads both halves (interactive model_state and decoder
    unetr_state) directly from `joint_checkpoint_path` instead of through `_joint_checkpoint_root()` /
    `_joint_export_root()` and the `joint_sam2_{model_type}_multi_gpu` naming convention - handy for a
    one-off checkpoint that lives outside that directory layout.

    Args:
        joint_checkpoint_path: Absolute path to a joint checkpoint (has 'model_state', 'unetr_state').
        model_type: The SAM2 backbone, e.g. 'hvit_t'.
        device: The torch device.
        ndim: The number of spatial dimensions, 2 or 3.
        cache_dir: Where to write the extracted interactive/decoder files. Defaults to a shared tmp dir.

    Returns:
        The automatic prompt generator, built through the same library factory `build_apg_segmenter` uses.
    """
    from micro_sam.v2.util import get_sam2_model
    from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator

    if cache_dir is None:
        cache_dir = "/tmp/micro_sam2_checkpoint_cache"
    os.makedirs(cache_dir, exist_ok=True)

    key = xxhash.xxh64(joint_checkpoint_path.encode()).hexdigest()
    interactive_path = os.path.join(cache_dir, f"{model_type}_{key}_interactive.pt")
    decoder_path = os.path.join(cache_dir, f"{model_type}_{key}_decoder.pt")

    if not (os.path.exists(interactive_path) and os.path.exists(decoder_path)):
        state = torch.load(joint_checkpoint_path, map_location="cpu", weights_only=False)
        _save_atomic({"model": _strip_ddp_prefix(state["model_state"]), "model_type": model_type}, interactive_path)
        _save_atomic(_strip_ddp_prefix(state["unetr_state"]), decoder_path)

    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=interactive_path,
        **({"input_type": "videos"} if ndim == 3 else {}),
    )
    decoder = load_unisam2_model(decoder_path, device, encoder=model.image_encoder, encoder_model_type=model_type)
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=ndim,
    )


def predict_unisam2(model, raw, ndim, device, normalization=None, devices=None):
    from micro_sam.v2.instance_segmentation import get_unisam2_segmentation_generator

    is_3d = (ndim == 3)
    has_channels = raw.ndim > ndim
    # Tiling an image that fits the training patch changes the encoder's scale and the normalization.
    is_tiled = is_3d or any(size > TRAINING_PATCH_SHAPE[-1] for size in raw.shape[:2])

    if has_channels and raw.shape[-1] > 1 and not is_tiled:
        # Real multi-channel data (e.g. RGB histopathology): keep the distinct channels, matching
        # training's raw_transform. `_run_full_inference`'s tiling machinery is single-channel only
        # (it averages a trailing channel axis to grayscale, then triplicates that single value into
        # 3 identical channels), which discards exactly the color signal H&E staining encodes and the
        # model was trained on. A non-tiled image is small enough to run directly instead, bypassing
        # that machinery. Tiled (large or 3d) multi-channel inputs still take the grayscale path below.
        from micro_sam.v2.util import to_float32
        from micro_sam.v2.instance_segmentation import ResizeLongestSideWrapper

        chw = np.moveaxis(raw, -1, 0).astype("float32")  # (H, W, C) -> (C, H, W)
        norm = normalize_raw(chw, axis=(-2, -1)) if normalization is None else normalization(chw)
        inp = torch.from_numpy(norm)[None, :, None].to(device)  # (1, C, 1, H, W)
        img_size = getattr(getattr(model, "encoder", None), "img_size", 1024)
        resize_model = ResizeLongestSideWrapper(model, img_size).to(device)
        with torch.no_grad():
            out = to_float32(resize_model(inp))
        return out[0, :, 0].cpu().numpy()

    # UniSAM2 takes single-channel input, so a trailing channel axis is averaged away.
    if has_channels:
        raw = raw.mean(axis=-1)

    segmenter = get_unisam2_segmentation_generator(
        model, is_tiled=is_tiled, device=device, inference_device=devices
    )
    if is_tiled:
        tile_shape = (4, 384, 384) if is_3d else (384, 384)
        halo = (2, 64, 64) if is_3d else (64, 64)
        segmenter.initialize(raw, ndim=ndim, tile_shape=tile_shape, halo=halo, normalization=normalization)
    else:
        segmenter.initialize(raw, ndim=ndim, normalization=normalization)
    return segmenter.get_state()["prediction"]


def postprocess_unisam2(out, dataset_name, model_type, params=None):
    """Turn a (4, *spatial) prediction into an instance segmentation.

    EM datasets use the dense (multicut) mode, all others the sparse (flow) mode. 'params' overrides
    the postprocessing defaults, e.g. with the best combination found by grid_search_automatic_cells.
    Without 'params', 'model_type' selects the per-model library default.
    """
    from micro_sam.v2.postprocessing import flow_instance_segmentation, run_multicut
    params = {} if params is None else params
    fg = out[0]
    if dataset_name in DATASETS_DENSE:
        boundary_map = fg.max() - fg
        boundary_map /= boundary_map.max()
        distances = np.stack([out[2], out[3]])
        seg = run_multicut(boundary_map, distances, model_type=model_type, **params)
    else:
        spacing = DATASET_SPACING.get(dataset_name, None)
        seg = flow_instance_segmentation(fg, out[1:], model_type=model_type, spacing=spacing, **params)
    return seg.astype("uint32")


def run_dataset_evaluation(gt_paths, prediction_paths, dataset_name: str, save_path: str):
    """Score a dataset and write the results to 'save_path'.

    Neuron segmentation in EM is ranked by the CREMI score, not by mSA, so those datasets report the
    VI and adapted-Rand components instead.

    Args:
        gt_paths: The ground-truth label arrays, or the paths to them.
        prediction_paths: The predicted segmentations, or the paths to them.
        dataset_name: The dataset the segmentations belong to.
        save_path: The filepath to write the result CSV to.

    Returns:
        The results as a DataFrame.
    """
    from micro_sam.v1.evaluation.evaluation import run_evaluation

    if dataset_name not in DATASETS_DENSE:
        return run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths, save_path=save_path)

    import pandas as pd
    from elf.evaluation import cremi_score

    rows = []
    for gt, seg in zip(gt_paths, prediction_paths):
        vi_split, vi_merge, adapted_rand, cremi = cremi_score(seg, gt)
        rows.append({
            "cremi": float(cremi),
            "vi_split": float(vi_split),
            "vi_merge": float(vi_merge),
            "adapted_rand": float(adapted_rand),
        })

    results = pd.DataFrame(rows).mean().to_frame().T
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results.to_csv(save_path, index=False)
    return results


def read_tuned_params(
    grid_search_root: str, dataset_name: str, model_type: str, checkpoint_checksum: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the best parameter combination of a grid search as a dict.

    New sweeps are keyed by checkpoint checksum. If no such sweep exists, an old checksum-less sweep
    is still accepted with a warning. The first row is the best combination and values are parsed as
    Python literals, so a tuple-valued 'candidate_threshold' survives the CSV round trip.

    Args:
        grid_search_root: The root the grid search wrote its per-model directories to.
        dataset_name: The dataset whose tuned parameters are read.
        model_type: The SAM2 backbone, which names the subdirectory.
        checkpoint_checksum: The effective weights, for an exact cache lookup. If omitted, read the
            legacy checksum-less location.

    Returns:
        The best combination, ready to be passed to the postprocessing or to 'generate'.
    """
    legacy_path = os.path.join(grid_search_root, model_type, f"{dataset_name}.csv")
    csv_path = (
        legacy_path if checkpoint_checksum is None else
        os.path.join(grid_search_root, model_type, checkpoint_checksum, f"{dataset_name}.csv")
    )
    if not os.path.exists(csv_path) and checkpoint_checksum is not None and os.path.exists(legacy_path):
        warnings.warn(
            f"Using legacy parameter sweep '{legacy_path}'. It has no checkpoint checksum, so its "
            "weights cannot be verified.",
            stacklevel=2,
        )
        csv_path = legacy_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"There is no grid search result at '{csv_path}'.")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"The grid search result at '{csv_path}' is empty.")

    params = {}
    for key, value in rows[0].items():
        if key.endswith(("_mean", "_std")) or key == "n_images":
            continue
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value

    # These are counts, and a column that ever held a NaN comes back as a float.
    for key in ("min_size", "n_iter", "min_candidate_size", "n_objects_per_pass"):
        if key in params:
            params[key] = int(params[key])
    return params


def _check_key(path: str, key: Optional[Union[str, Tuple[str, ...]]], kind: str) -> None:
    if key is None:
        return
    keys = key if isinstance(key, tuple) else (key,)
    try:
        with open_file(path, mode="r") as f:
            for k in keys:
                if k not in f:
                    raise RuntimeError(f"Missing {kind} key '{k}' in '{path}'.")
    except Exception as e:
        raise RuntimeError(f"Could not open {kind} data key '{key}' in '{path}': {e}") from e


def check_data_download(dataset_name: str, data_root: str, download: bool = True, split: str = "test") -> None:
    """Fail fast if a dataset cannot be resolved from the local data root.

    The check goes through `get_data_paths(..., download=download)`, so it catches a missing download,
    an invalid split and an unavailable cached file before the model loads. It may download a missing
    dataset once. The evaluation itself reads the cached local data afterwards.

    Args:
        dataset_name: The dataset to check.
        data_root: The root the data lives in.
        download: Whether a missing dataset may be downloaded.
        split: The split to check, 'test' or the held-out 'val' a parameter search tunes on.
    """
    try:
        raw_paths, label_paths, raw_key, label_key = get_data_paths(
            dataset_name, data_root, download=download, split=split
        )
    except Exception as e:
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}'. "
            "The dataset-specific get_*_paths helper could not resolve local data."
        ) from e

    if not raw_paths:
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}': "
            "no raw paths were found. The data is probably missing or all subsets were skipped."
        )
    if not label_paths:
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}': "
            "no label paths were found. The data is probably missing or all subsets were skipped."
        )
    if len(raw_paths) != len(label_paths):
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}': "
            f"found {len(raw_paths)} raw paths but {len(label_paths)} label paths."
        )

    missing = []
    for raw_path, label_path in zip(raw_paths, label_paths):
        if not os.path.exists(raw_path):
            missing.append(raw_path)
        if label_path != raw_path and not os.path.exists(label_path):
            missing.append(label_path)

    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}': "
            f"{len(missing)} referenced file(s) do not exist:\n{examples}{suffix}"
        )

    _check_key(raw_paths[0], raw_key, "raw")
    _check_key(label_paths[0], label_key, "label")

    print(f"Data check passed for '{dataset_name}': {len(raw_paths)} sample(s).")


CROP_SHAPE_2D = (512, 512)
CROP_SHAPE_3D = (8, 512, 512)


def ensure_8bit_range(raw):
    """Scale raw data into the [0, 255] range the evaluation feeds the models with."""
    if raw.size == 0:
        return raw.astype("float32", copy=False)
    # `read_2d` returns channel-last images. Preserve the contrast of every microscopy channel
    # instead of letting the channel with the largest values determine the shared percentile range.
    spatial_axes = (0, 1) if raw.ndim == 3 and raw.shape[-1] in (1, 2, 3, 4) else None
    return normalize_raw(raw, axis=spatial_axes) * 255.0


def read_2d(path, key):
    """Read a 2d array from an image file, or from an H5 / zarr file using 'key'.

    A tuple of keys reads one channel per key and stacks them channel-last.
    """
    if isinstance(key, tuple):
        with open_file(path, mode="r") as f:
            return np.stack([f[k][:] for k in key], axis=-1)
    if key is not None:
        arr = open_file(path, mode="r")[key][:]
    else:
        arr = np.asarray(imageio.imread(path))
    # Transpose channel-first (C, H, W) to channel-last (H, W, C).
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
        arr = arr.transpose(1, 2, 0)
    # Some 2d datasets mix in multi-frame stacks, e.g. yeaz. Evaluate their first frame.
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[0]
    return arr


def sorted_path_pairs(raw_paths, label_paths):
    """Sort raw and label paths as pairs, so the pairing survives names that sort differently."""
    return sorted(zip(raw_paths, label_paths), key=lambda pair: (str(pair[0]), str(pair[1])))


def interactive_result_name(
    dataset_name, method, model_type, prompt, iteration,
    ndim=2, use_masks=True, mask_threshold=0.0, min_size=0,
):
    """Build the name of the result CSV for one iteration of an interactive run.

    The name encodes every setting that changes the numbers, so one run cannot reuse the results of
    another.
    """
    dim_suffix = "" if ndim == 2 else "_3d"
    tag = interactive_run_tag(ndim, use_masks, mask_threshold, min_size)
    return f"{dataset_name}_{method}_{model_type}{dim_suffix}_{prompt}{tag}_iter{iteration:02d}.csv"


def interactive_run_tag(ndim=2, use_masks=True, mask_threshold=0.0, min_size=0):
    """Build the settings suffix for an interactive run's result names and prediction directory.

    Both use one tag, so a run can never read back the cached predictions of another run.
    """
    # Only the 2d path chooses between mask logits and binarized masks.
    tag = "" if ndim == 3 else ("_with_masks" if use_masks else "_without_masks")
    if ndim == 2 and mask_threshold != 0.0:
        tag += f"_t{mask_threshold:g}"
    if min_size:
        tag += f"_min{min_size}"
    return tag


def apply_min_size(labels, min_size, dataset_name):
    """Drop ground-truth objects below 'min_size' pixels, and warn if that removes too many.

    No single threshold suits every dataset. Gonuclear nuclei have a median of about 3200 pixels per
    object, while cremi neurite cross-sections in a thin crop have a median of about 6.
    """
    if not min_size:
        return labels
    before = len(np.unique(labels)) - 1
    filtered = size_filter(seg=labels, min_size=min_size)
    after = len(np.unique(filtered)) - 1
    if before and (before - after) / before > 0.25:
        warnings.warn(
            f"min_size={min_size} removes {before - after} of {before} ground-truth objects in "
            f"'{dataset_name}'. That is more than a quarter, so the threshold is likely too large "
            f"for this dataset and is discarding real annotations."
        )
    return filtered


def drop_severed_objects(labels, min_size):
    """Drop the objects that a crop face cut down to a sliver, in a ground truth or a prediction.

    Both conditions are needed. A size threshold alone also deletes small interior objects, and
    border contact alone deletes large cells that only reach the edge. The caller filters the ground
    truth and the prediction the same way, so a dropped remnant never becomes a false positive.
    """
    if not min_size:
        return labels
    if labels.ndim == 2:
        edges = (labels[0], labels[-1], labels[:, 0], labels[:, -1])
    else:
        # In-plane faces only, since a thin z-crop cuts almost every object on the first and last slice.
        edges = (labels[:, 0], labels[:, -1], labels[:, :, 0], labels[:, :, -1])
    border_ids = np.unique(np.concatenate([np.unique(edge) for edge in edges]))
    border_ids = border_ids[border_ids != 0]
    if border_ids.size == 0:
        return labels

    ids, sizes = np.unique(labels[labels != 0], return_counts=True)
    severed = np.intersect1d(border_ids, ids[sizes < min_size], assume_unique=True)
    if severed.size == 0:
        return labels
    return np.where(np.isin(labels, severed), 0, labels).astype(labels.dtype)


def severed_objects(gt, max_span=2):
    """The ground-truth objects that occupy no more than 'max_span' slices of a volume.

    A volumetric object is anchored on the slice its density converges on, and that density scales
    with the object's size. An object that the crop reduced to one or two slices never reaches
    'candidate_threshold', so it is never proposed. Separating these says how much of the gap to the
    ground truth is the crop rather than the method.

    Args:
        gt: The ground-truth labels, shape (Z, Y, X).
        max_span: The largest number of slices a severed object may span.

    Returns:
        The labels of the severed objects with everything else zeroed, and their ids.
    """
    ids = np.unique(gt)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.zeros_like(gt), np.array([], dtype=gt.dtype)
    spans = np.array([int((gt == index).any(axis=(1, 2)).sum()) for index in ids])
    thin = ids[spans <= max_span]
    return np.where(np.isin(gt, thin), gt, 0).astype(gt.dtype), thin


def unmatched_objects(gt, segmentation, iou_threshold=0.5):
    """The ground-truth objects that no predicted instance matches at the given IoU.

    Matched the way `mean_segmentation_accuracy` matches at its lowest threshold, so these are the
    objects the result genuinely lost rather than segmented imprecisely.

    Args:
        gt: The ground-truth labels.
        segmentation: The predicted instance segmentation.
        iou_threshold: The IoU a prediction must reach to count as a match.

    Returns:
        The labels of the unmatched objects, with everything else zeroed.
    """
    ids = np.unique(gt)
    ids = ids[ids != 0]
    missed = []
    for index in ids:
        mask = gt == index
        overlapping = segmentation[mask]
        overlapping = overlapping[overlapping != 0]
        if overlapping.size == 0:
            missed.append(index)
            continue
        candidates, counts = np.unique(overlapping, return_counts=True)
        best = int(np.argmax(counts))
        intersection = int(counts[best])
        union = int(mask.sum()) + int((segmentation == candidates[best]).sum()) - intersection
        if intersection / union < iou_threshold:
            missed.append(index)
    return np.where(np.isin(gt, missed), gt, 0).astype(gt.dtype)


def genuine_misses(gt, segmentation, iou_threshold=0.5, max_span=2):
    """How many ground-truth objects the result lost that the crop did not sever.

    An aggregate metric hides which objects went missing. A run that recovers objects while costing
    precision elsewhere is not the same as one that does neither, so this counts the losses the
    method is answerable for.

    Args:
        gt: The ground-truth labels, shape (Z, Y, X).
        segmentation: The predicted instance segmentation.
        iou_threshold: The IoU a prediction must reach to count as a match.
        max_span: The largest number of slices a severed object may span.

    Returns:
        The number of unmatched objects, and how many of those the crop did not sever.
    """
    unmatched_ids = np.unique(unmatched_objects(gt, segmentation, iou_threshold))
    unmatched_ids = unmatched_ids[unmatched_ids != 0]
    _, thin = severed_objects(gt, max_span)
    return len(unmatched_ids), int((~np.isin(unmatched_ids, thin)).sum())


def load_evaluation_sample_2d(raw_path, label_path, raw_key, label_key, dataset_name):
    """Load one 2d sample the way the evaluation scores it.

    The parameter search and the evaluation both call this function, so both use the same data.
    """
    # Normalize before cropping, so that the percentiles cover the whole image.
    image = ensure_8bit_range(select_channels(read_2d(raw_path, raw_key), dataset_name))
    roi = _center_crop_roi(image.shape[:2], CROP_SHAPE_2D)
    labels = read_2d(label_path, label_key)[roi]
    if dataset_name == "astih":
        # Semantic labels (1 myelin, 2 axon); the instances are the axon class, as in training.
        labels = labels == 2
    gt = connected_components(labels).astype("uint32")
    return image[roi], drop_severed_objects(gt, GT_MIN_SIZE_2D.get(dataset_name, 0))


def load_evaluation_sample_3d(
    raw_path, label_path, raw_key, label_key, dataset_name,
    crop_shape=CROP_SHAPE_3D, z_range=None, min_size=0, split="test",
):
    """Load one volumetric sample the way the evaluation scores it."""
    raw, labels, valid_roi = load_volume(
        raw_path, label_path, raw_key, label_key, dataset_name, crop_shape, z_range=z_range, split=split
    )
    return raw, apply_min_size(labels, min_size, dataset_name), valid_roi


def load_data(dataset_name, data_root, ndim, min_size=0, split="test", crop_shape=None, z_range=None):
    """Yield (image_or_volume, labels, valid_roi) triples for the given dataset.

    valid_roi is a boolean mask that is True where the data is annotated. It is None for every
    dataset except platynereis_nuclei, which is annotated only in part.

    The filtering happens here, in the single source of the labels used for both prompting and
    scoring. If only the prompting copy were filtered, the dropped objects would stay in the scored
    ground truth and count as unmatched.

    Args:
        dataset_name: The dataset to load.
        data_root: The root the data lives in.
        ndim: The number of spatial dimensions, 2 or 3.
        min_size: Drop ground-truth objects below this many pixels (3d only).
        split: The split to load, 'test' or the held-out 'val', see VAL_SPLITS.
        crop_shape: The 3d center crop. Defaults to CROP_SHAPE_3D.
        z_range: Restrict a volume to a z-slab before cropping, see VAL_Z_RANGE.

    Yields:
        One (image_or_volume, labels, valid_roi) triple per sample.
    """
    raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root, split=split)
    for raw_path, label_path in sorted_path_pairs(raw_paths, label_paths):
        if ndim == 3:
            sample_z_range = val_z_range(dataset_name, raw_path, split) or z_range
            yield load_evaluation_sample_3d(
                raw_path, label_path, raw_key, label_key, dataset_name,
                crop_shape=crop_shape or CROP_SHAPE_3D, z_range=sample_z_range, min_size=min_size, split=split,
            )
        else:
            image, gt = load_evaluation_sample_2d(raw_path, label_path, raw_key, label_key, dataset_name)
            yield image, gt, None


def n_samples(dataset_name, data_root, split="test"):
    """The number of samples of a split, for a progress bar over `load_data`."""
    return len(get_data_paths(dataset_name, data_root, split=split)[0])


def has_val_split(dataset_name: str) -> bool:
    """Whether a dataset holds data out from the samples the evaluation scores.

    Only these datasets can be tuned honestly: everywhere else a sweep would select its parameters
    on the very samples the reported score is measured on. See VAL_SPLITS and VAL_Z_RANGE.
    """
    return dataset_name in VAL_SPLITS
