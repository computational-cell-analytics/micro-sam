import os
import ast
import csv
import json
import warnings
from glob import glob
from pathlib import Path
from itertools import islice
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
from micro_sam.v2.models.util import joint_unetr_state
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
    XENIUM_TEST_SAMPLES, XENIUM_VAL_SAMPLES, SYNAPSENET_TEST_TOMOGRAMS, cell_acdc_movie, cvz_group,
    dsb_fluorescence_training_paths, synapsenet_roi, synapsenet_tomograms, _train_val_test_split,
    CELLMAP_NUCLEUS_VAL_CROPS, PLATYNEREIS_NUCLEI_IGNORE_LABEL, PLATYNEREIS_NUCLEI_TEST_IDS,
    PLATYNEREIS_NUCLEI_VAL_IDS,
    _compute_platy_rois, SNEMI_MITO_PATH, SNEMI_MITO_VAL_ROI,
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

# The correction clicks of 2d iterative prompting. 'random' samples them at random positions in the error
# regions (IterativePromptGenerator) and picks the best of 3 masks for a first point prompt. 'center' places
# them at the centers of the largest error regions (CenterIterativePromptGenerator) and predicts a single mask
# for a first point prompt.
CLICK_PROTOCOLS = ("random", "center")

# The 2d patch shape the models were trained on, see 'generalist_loader'.
TRAINING_PATCH_SHAPE = (512, 512)

# LIVECell test images whose annotation is incomplete: 2 labelled objects in a confluent crop, at
# 348x and 24x the annotated foreground. Both lie outside the stratified subset.
LIVECELL_EXCLUDED_TEST_IMAGES = frozenset({
    "BV2_Phase_A4_2_02d04h00m_3.tif",
    "BV2_Phase_A4_2_00d00h00m_1.tif",
})

# Images the official LIVECell files list under both train and val, 17 Huh7 and 13 SKOV3. Dropped
# from 'val' only, so a sweep does not tune on data the model trained on.
LIVECELL_TRAIN_VAL_OVERLAP = frozenset({
    "Huh7_Phase_A10_2_00d00h00m_3.tif",
    "Huh7_Phase_A10_2_00d12h00m_1.tif",
    "Huh7_Phase_A10_2_00d16h00m_3.tif",
    "Huh7_Phase_A10_2_00d20h00m_1.tif",
    "Huh7_Phase_A10_2_01d00h00m_1.tif",
    "Huh7_Phase_A10_2_01d00h00m_2.tif",
    "Huh7_Phase_A10_2_01d00h00m_4.tif",
    "Huh7_Phase_A11_2_01d08h00m_4.tif",
    "Huh7_Phase_A11_2_01d16h00m_1.tif",
    "Huh7_Phase_A11_2_01d16h00m_2.tif",
    "Huh7_Phase_A11_2_01d20h00m_2.tif",
    "Huh7_Phase_A11_2_01d20h00m_3.tif",
    "Huh7_Phase_A11_2_02d00h00m_2.tif",
    "Huh7_Phase_A11_2_02d00h00m_4.tif",
    "Huh7_Phase_A11_2_02d04h00m_2.tif",
    "Huh7_Phase_A11_2_02d04h00m_3.tif",
    "Huh7_Phase_A11_2_02d04h00m_4.tif",
    "SKOV3_Phase_G4_1_00d20h00m_3.tif",
    "SKOV3_Phase_G4_1_00d20h00m_4.tif",
    "SKOV3_Phase_G4_1_01d00h00m_3.tif",
    "SKOV3_Phase_G4_1_01d04h00m_3.tif",
    "SKOV3_Phase_G4_1_01d08h00m_1.tif",
    "SKOV3_Phase_G4_1_01d12h00m_4.tif",
    "SKOV3_Phase_H4_1_00d00h00m_2.tif",
    "SKOV3_Phase_H4_1_00d04h00m_1.tif",
    "SKOV3_Phase_H4_1_00d04h00m_2.tif",
    "SKOV3_Phase_H4_1_00d08h00m_2.tif",
    "SKOV3_Phase_H4_1_00d08h00m_4.tif",
    "SKOV3_Phase_H4_1_00d12h00m_1.tif",
    "SKOV3_Phase_H4_1_00d16h00m_1.tif",
})


# Test images with a verified copy in the training data (byte-identical, or the same field re-imaged with a
# normalized cross-correlation above 0.95), dropped from the scored test split. Found by the 2026-09-21 data audit,
# see experiments/micro_sam2/experiments/data_audit_2026_09_21/contaminated_test_images.csv.
EXCLUDED_TEST_IMAGES = {
    # Our random split of a dataset that ships repeated crops: 3 identical, 10 consecutive re-imaged fields.
    "bmgd": frozenset({
        "bmgd_1200pa_img (45).h5",
        "bmgd_1200pa_img (48).h5",
        "bmgd_1200pa_img (70).h5",
        "bmgd_1800pa_img2.0 (147).h5",
        "bmgd_1800pa_img2.0 (15).h5",
        "bmgd_1800pa_img2.0 (20).h5",
        "bmgd_1800pa_img2.0 (49).h5",
        "bmgd_1800pa_img2.0 (58).h5",
        "bmgd_250pa_img287.h5",
        "bmgd_250pa_img294.h5",
        "bmgd_250pa_img440.h5",
        "bmgd_250pa_img488.h5",
        "bmgd_250pa_img490.h5",
    }),
    # Time points within 0.95 correlation of a training frame of the same field of view.
    "yeaz": frozenset({
        "cdc20F2BF_20_crop_1_im.tif",
        "cdc20F9BF_20_crop_1_im.tif",
        "clnF2BF_5_crop_1_im.tif",
        "clnF5BF_10_crop_1_im.tif",
        "ddF3BF_10_crop_1_im.tif",
        "wtF10BF_10_crop_2_im.tif",
        "wtF14BF_10_crop_1_im.tif",
    }),
    # BBBC022 U2OS fields that DSB stage 1 also holds, re-encoded to 8 bit under hashed names, and trains on.
    "u20s": frozenset({
        "IXMtest_D10_s2_w1BB72A093-72AB-476C-9513-2FF43BFB2134.tif",
        "IXMtest_G10_s3_w1C1257E17-1DBA-4619-B06E-D6DBB8A53088.tif",
        "IXMtest_G16_s6_w1C3F175E2-0AF5-401C-AC5C-8E128A237B22.tif",
        "IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2.tif",
        "IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A.tif",
    }),
    # The one StarDist test image that is a BBBC022 field of the u20s training split.
    "dsb": frozenset({
        "5f9d29d6388c700f35a3c29fa1b1ce0c1cba6667d05fdb70bd1e89004dcf71ed.tif",
    }),
    # Official test frames of the 'ftsN' experiment that repeat training frames, three identical and two adjacent.
    "omnipose_bact_phase": frozenset({
        "ftsN_ensemble_30.tif",
        "ftsN_ensemble_36.tif",
        "murA_ensemble_0.tif",
        "murA_ensemble_2.tif",
        "murA_ensemble_4.tif",
    }),
    # Official brain organoid test images that are identical to training images under other organoid numbers.
    "orgline": frozenset({
        "org01_wt2D_d05_LabB.tif.h5",
        "org02_wt2D_d05_LabB.tif.h5",
        "org03_wt2D_d05_LabB.tif.h5",
    }),
    # The authors' split holds three acquisitions of one Vectra pancreas region; the M2 one trains.
    "pan_multiplex": frozenset({
        "820f9ed19222 Pancreas_PANEL7-10_CD40L,_CD40,_PD1,_PDL1,CD8,CK_[49017,8252]_component_data.tif_image.h5",
    }),
    # Our random split; two consecutive G361 acquisitions are identical.
    "vicar": frozenset({
        "00010_G361_img.tif",
    }),
}


def drop_excluded_test_images(dataset_name, raw_paths, label_paths) -> Tuple[List[str], List[str]]:
    """Remove the test images of a dataset that have a copy in the training data, see EXCLUDED_TEST_IMAGES."""
    excluded = EXCLUDED_TEST_IMAGES.get(dataset_name)
    if not excluded:
        return raw_paths, label_paths
    keep = [os.path.basename(raw) not in excluded for raw in raw_paths]
    return [p for p, k in zip(raw_paths, keep) if k], [p for p, k in zip(label_paths, keep) if k]


def drop_excluded_livecell(raw_paths, label_paths, split=None) -> Tuple[List[str], List[str]]:
    """Remove the LIVECell images that must not be scored from a path pair list.

    Args:
        raw_paths: The image paths.
        label_paths: The label paths, in the same order.
        split: The split the paths were read from. 'val' additionally drops the images the official
            files also list under train, see LIVECELL_TRAIN_VAL_OVERLAP.

    Returns:
        The kept image and label paths.
    """
    excluded = set(LIVECELL_EXCLUDED_TEST_IMAGES)
    if split == "val":
        excluded |= LIVECELL_TRAIN_VAL_OVERLAP
    keep = [
        (raw, label) for raw, label in zip(raw_paths, label_paths)
        if os.path.basename(raw) not in excluded
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
# mnDINO images are BBBC022 U2OS fields: 30 of its 64 test images are byte-identical to u20s training images and
# 3 more to DSB training images, so it is not out of domain.
DATASETS_2D_LM_NUCLEUS_SUPPLEMENTARY = ["xenium_nuclei", "mndino"]
DATASETS_2D_LM_NUCLEUS_OOD = ["cardioblast_nuclei", "hela_cytonuc", "covid_if_nuclei", "arvidsson", "micro_bench"]
# sPATCH DAPI shares tissue sections with the sPATCH H&E training data, so it is a held-out platform evaluation
# rather than an independent OOD collection until the patient overlap is resolved.
DATASETS_2D_LM_NUCLEUS_HELD_OUT_PLATFORM = ["spatch_dapi"]
DATASETS_2D_LM_LABEL_FREE_ID = [
    "livecell", "deepbacs_label_free", "omnipose_bact_phase", "neurips_cellseg_label_free", "cell_acdc", "cellular",
    "vicar", "microbeseg",
]
# yeaz is split per image, so every test image is a time point of a field of view that also trains.
DATASETS_2D_LM_LABEL_FREE_SUPPLEMENTARY = [
    "orgasegment", "organoidnet", "omnipose_worm", "omnipose_worm_high_res", "bccd", "cisd", "orgline", "organoid",
    "mcellseg", "toiam", "bbbc030", "yeaz",
]
DATASETS_2D_LM_LABEL_FREE_OOD = [
    "cellapp", "deepseas", "dic_hepg2", "yeastsam", "bac_mother", "ecoli_microcolony_lineage", "bbbc010",
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

# 2D EM: ASTIH (neurites), the tumor spheroid slices (cells) and the mitochondria and nucleus panels.
DATASETS_2D_EM_MITO = [
    "mitolab_tem", "cmito_dauer_manual", "probtem", "tnbc_mito", "elmi_mito",
    "lewy_mito", "perez_mito", "cardiomyocyte_tem_mito",
]
# Cardiomyocyte TEM (out-of-domain): the images are downscaled and cut into the densest tiles.
# Touching mitochondria merge, mostly in 914.1_nucC and 914.22_nucE.
CARDIOMYOCYTE_TEM_SCALE = 2
CARDIOMYOCYTE_TEM_N_TILES = 4
# Perez (out-of-domain): the 40 test tiles are scored; the train tiles equal the excluded MitoLab group.
# The tiles are partly annotated.
PEREZ_MITO_N_TUNING = 10
# ELMI Mito (out-of-domain): only 3-6 mitochondria per image are labeled.
ELMI_MITO_TUNING_IMAGES = ("ELMI_05.tif",)
# Lewy Mito (out-of-domain): crops around single mitochondria; neighbours are often not labeled.
LEWY_MITO_GROUPS = ("SN-PD", "HC-SN")
LEWY_MITO_N_TEST = 300
LEWY_MITO_N_TUNING = 30
LEWY_MITO_MIN_SIDE = 128
# TNBC-Mito (out-of-domain): the low-magnification images miss labels, so they are left out.
TNBC_MITO_COHORTS = ("DRP1-KO", "HCI-010", "PIM001-P")
TNBC_MITO_EXCLUDED_TEST_IMAGES = ("R20_72_G01_2913_03.tif", "R20_72_G01_2913_05.tif", "wt_10Kx_2.tif")
# U2OS FIB-SEM nuclei (out-of-domain): the nucleus is the convex hull of the envelope arc and the border.
# The tuning crops are not adjacent in z.
U2OS_FIBSEM_NUCLEUS_ROOT = "u2os_fibsem_nucleus"
U2OS_FIBSEM_NUCLEUS_TUNING_CROPS = ("05", "11", "17")
DATASETS_2D_EM_NUCLEUS = ["u2os_fibsem_nucleus"]
DATASETS_2D_EM = ["astih", "tumor_spheroid"] + DATASETS_2D_EM_MITO + DATASETS_2D_EM_NUCLEUS

DATASETS_2D = DATASETS_2D_LM + DATASETS_HP + DATASETS_2D_EM

# Ground-truth size floor that drops the crop-severed slivers relabelling promotes to objects. It
# defines the ground truth, so it is measured, never tuned.
GT_MIN_SIZE_2D = {
    "livecell": 50, "dsb": 10,
    "deepbacs_label_free": 50, "deepbacs_fluorescence": 50, "dynamicnuclearnet": 50, "tissuenet": 10,
    "u20s": 10, "vicar": 25, "yeaz": 10, "probtem": 25, "tnbc_mito": 25, "elmi_mito": 25, "lewy_mito": 25,
    "perez_mito": 25, "cardiomyocyte_tem_mito": 25, "u2os_fibsem_nucleus": 25,
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
DATASETS_3D_LM_NUCLEUS_OOD = [
    "parhyale_regen", "mouse_embryo", "blastospim", "morphonet_celegans", "bbbc032", "bbbc033", "bbbc050",
]
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
DATASETS_3D_EM_CELL_ID = ["platynereis_cells", "densecell", "synapsenet_compartments"]

# The CEM-Mito-Benchmark volumes; lucchi_pp equals the Lucchi train volume, so it is not scored.
MITOLAB_VOLUMES = ("c_elegans", "fly_brain", "glycolytic_muscle", "hela_cell", "lucchi_pp", "salivary_gland")
MITOLAB_EVALUATED_VOLUMES = tuple(name for name in MITOLAB_VOLUMES if name != "lucchi_pp")
# The Cmito volumes; their labels are Cmito model output, not hand labels.
CMITO_VOLUMES = {
    "cmito_l1": "l1_ceb79c8268d7.zarr",
    "cmito_l2": "l2_2ea67de84d7e.zarr",
    "cmito_l3": "l3_voxels_13ca91605fde.zarr",
    "cmito_adult": "adult_dba9f64835ae.zarr",
    "cmito_dauer": "dauer_056f5d679440.zarr",
}
# The testis and brainstem boxes of MitoSegEM train, so only the intestine box is scored.
MITO_SEGEM_BOXES = {
    "mito_segem_intestine": ("intestine", (468, 532, 2304, 3328, 2816, 3840)),
}
# The MitoEM val volumes are scored; the test volumes have no labels.
MITOEM_SAMPLES = {"mitoem_human": "human", "mitoem_rat": "rat"}
# The MitoEM v2 test volumes are scored; jurkat tunes on the end of its train volume.
# mossy is scored only inside the bouton mask of its MitoVerse copies.
MITOEMV2_DATASETS = {
    f"mitoemv2_{name}": name for name in ("beta", "jurkat", "macro", "podo", "sperm", "stem", "mossy")
}
# The MitoVerse han24 volumes of the MitoEM v2 mossy samples: test01 -> neg1, test02 -> pos1, val (train03) -> pos2.
MITOEMV2_MOSSY_VOLUMES = {"test": ["neg1", "pos1"], "val": ["pos2"]}
# MitoNet-Predicted-Kidney (in-domain): MitoNet output in boxes that no training crop overlaps.
MITONET_KIDNEY_BOXES = [
    (2240, 2368, 2560, 3584, 3584, 4608), (4800, 4928, 2048, 3072, 4096, 5120), (10432, 10560, 1024, 2048, 512, 1536),
]
# Human liver (out-of-domain): automatic labels; the boxes lie in the annotated region.
HUMAN_LIVER_EM_BOXES = {
    "test": [
        (10752, 11776, 6656, 7680, 440, 472), (11264, 12288, 1024, 2048, 240, 272), (2048, 3072, 11776, 12800, 40, 72),
    ],
    "val": [(9216, 10240, 6656, 7680, 540, 556)],
}
# Parlakgul liver (out-of-domain): the manual crops; the third crop of each sample tunes.
PARLAKGUL_LIVER_EVAL_SAMPLES = ("6461", "6464", "9430", "1857")
PARLAKGUL_LIVER_EVAL_TUNING_CROP = "003"
# Cerebellum FIB-SEM mitochondria (out-of-domain): the first fifth of z tunes.
CEREBELLUM_FIBSEM_PATH = os.path.join("cerebellum_fibsem", "cerebellum_fibsem.h5")
# Macrophage / A431 mitochondria (out-of-domain): the first fifth of the block tunes.
MACROPHAGE_A431_MITO_PATH = os.path.join("macrophage_a431", "fib_control_z848-976.h5")
# MitoANF (out-of-domain): only the mitochondria of traced nerve fibers are labeled.
MITO_ANF_BOXES = ("M1_9cb2de6a1d1f.zarr", "M2_908d64f28407.zarr")
# INTERNAL data from Donglai Wei's lab (not public), out-of-domain; blank raw strips are not scored.
DONGLAI_WEI_MITO_ROOT = os.path.join("donglai_wei_mito", "micron1")
# densecell_mito and hemibrain_mito are in-domain sets of the generalists.
DATASETS_3D_EM_MITO = (
    [f"mitolab_{name}" for name in MITOLAB_EVALUATED_VOLUMES] + ["asem_mito"] + list(CMITO_VOLUMES)
    + ["densecell_mito", "hemibrain_mito", "lucchi", "snemi_mito", "kasthuri"] + list(MITO_SEGEM_BOXES)
    + list(MITOEM_SAMPLES) + list(MITOEMV2_DATASETS) + ["mitonet_kidney", "human_liver_em", "parlakgul_liver_eval"]
    + ["cerebellum_fibsem_mito", "macrophage_a431_mito", "mito_anf", "donglai_wei_mito"]
)
DATASETS_3D_EM_MITO_SUPPLEMENTARY = []

# The nucleus panel: the in-domain sets are held-out parts of the training datasets.
CELLMAP_NUCLEUS_TEST_CROPS = ("jrc_mus-liver-zon-2_crop357",)
HUMAN_LIVER_EM_NUCLEUS_BOXES = {"test": ["test_a", "test_b", "test_c"], "val": ["tune"]}
# Etch-a-Cell nuclei (out-of-domain): the envelope labels are filled into nuclei.
ETCHACELL_NUCLEUS_ROOT = "etchacell_nucleus"
ETCHACELL_NUCLEUS_TUNING_ROIS = ("ROI_1584-6996-1", "ROI_1608-912-1", "ROI_3588-3972-1")
DATASETS_3D_EM_NUCLEUS_ID = ["cellmap_nucleus", "fafb_nuclei", "microns_nuclei", "platynereis_nuclei"]
DATASETS_3D_EM_NUCLEUS_OOD = ["human_liver_em_nucleus", "etchacell_nucleus"]
DATASETS_3D_EM_NUCLEUS = DATASETS_3D_EM_NUCLEUS_ID + DATASETS_3D_EM_NUCLEUS_OOD
DATASETS_EM_NUCLEUS = DATASETS_3D_EM_NUCLEUS + DATASETS_2D_EM_NUCLEUS

DATASETS_3D_EM = (
    DATASETS_3D_EM_NEURITE_ID + DATASETS_3D_EM_NEURITE_SUPPLEMENTARY + DATASETS_3D_EM_NEURITE_OOD
    + DATASETS_3D_EM_CELL_ID + DATASETS_3D_EM_MITO + DATASETS_3D_EM_MITO_SUPPLEMENTARY + DATASETS_3D_EM_NUCLEUS
)
DATASETS_EM = DATASETS_2D_EM + DATASETS_3D_EM
DATASETS_EM_MITO = (
    DATASETS_2D_EM_MITO + DATASETS_3D_EM_MITO + DATASETS_3D_EM_MITO_SUPPLEMENTARY
)

DATASETS_3D = DATASETS_3D_LM + DATASETS_3D_EM

# The neurite datasets need the dense (multicut) pipeline and are ranked by the CREMI score. The EM cell datasets
# segment separable objects, so they stay on the sparse (flow) pipeline and mSA ranking.
DATASETS_DENSE = DATASETS_3D_EM_NEURITE_ID + DATASETS_3D_EM_NEURITE_SUPPLEMENTARY + DATASETS_3D_EM_NEURITE_OOD

# Everything outside the main in-domain and OOD panels: supplementary, held-out platform and reserve sets.
# The submission script skips them unless asked for them.
DATASETS_SUPPLEMENTARY = (
    DATASETS_2D_LM_CELL_SUPPLEMENTARY + DATASETS_2D_LM_NUCLEUS_SUPPLEMENTARY + DATASETS_2D_LM_NUCLEUS_HELD_OUT_PLATFORM
    + DATASETS_2D_LM_LABEL_FREE_SUPPLEMENTARY + DATASETS_2D_LM_LABEL_FREE_RESERVE + DATASETS_3D_LM_NUCLEUS_SUPPLEMENTARY
    + DATASETS_3D_EM_NEURITE_SUPPLEMENTARY + DATASETS_2D_HP_SUPPLEMENTARY + DATASETS_3D_EM_MITO_SUPPLEMENTARY
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
    "blastospim": "val", "bbbc010": "val", "bbbc050": "train",
    "wing_disc": None, "embedseg_mouse_skull": None, "embedseg_platy_ish": None, "nis3d": None,
    "humanneurons": None, "bbbc032": None, "bbbc033": None,
})

# Volumes whose tuning data is a z-slab of the test volume, as (file name, z-slab), following the loader; the
# evaluation scores the rest of the volume, see val_z_range. nucverse3d holds separate tuning volumes for its liver
# collections and a slab for drosophila_glia, see _get_3d_lm_data_paths. bbbc032 and bbbc033 are each a single
# sparsely-instanced volume, so the tuning slab sits at the start and the rest of the volume is scored.
LM_VAL_Z_SLABS = {
    "wing_disc": {f"{name}.h5": WING_DISC_VAL_Z for name in WING_DISC_TEST_VOLUMES},
    "embedseg_mouse_skull": {"X2_right.tif": EMBEDSEG_VAL_Z["Mouse-Skull-Nuclei-CBG"]},
    "embedseg_platy_ish": {"X02_test.tif": EMBEDSEG_VAL_Z["Platynereis-ISH-Nuclei-CBG"]},
    "nis3d": {"data.tif": NIS3D_VAL_Z},
    "nucverse3d": {NUCVERSE_GLIA_VAL_VOLUME: NUCVERSE_GLIA_VAL_Z},
    "bbbc032": {"BMP4blastocystC3.tif": slice(0, 20)},
    "bbbc033": {"BBBC033.h5": slice(0, 6)},
}

# EM: None means the tuning data is a different region (EM_ROIS, EM_VOLUME_ROIS) or different files (see
# _get_3d_em_data_paths) of the same dataset, both disjoint from the blind test region. fib25 has no validation data
# and is not listed.
VAL_SPLITS.update({
    name: None for name in (
        "cremi", "snemi", "axonem", "fafb", "hemibrain", "manc", "malecns", "wafer4", "minnie65",
        "zebrafinch_j0126", "zebrafinch_j0251", "wildenberg", "liconn", "xpress", "nisb", "platynereis_cells",
        "densecell", "synapsenet_compartments", *DATASETS_3D_EM_MITO, *DATASETS_3D_EM_MITO_SUPPLEMENTARY,
        *DATASETS_3D_EM_NUCLEUS,
    )
})
# The 2d EM resolvers return their tuning images for any split other than 'test', and a None split resolves to
# 'test', so they need 'val' here.
VAL_SPLITS.update({name: "val" for name in DATASETS_2D_EM})

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
    # The CEM-Mito-Benchmark volumes are annotated throughout (salivary_gland from z 76). The tuning slab is the
    # start of the annotated stack, about a fifth of it, and the rest is scored.
    "mitolab_c_elegans": {"test": np.s_[48:, :, :], "val": np.s_[:48, :, :]},
    "mitolab_fly_brain": {"test": np.s_[48:, :, :], "val": np.s_[:48, :, :]},
    "mitolab_glycolytic_muscle": {"test": np.s_[48:, :, :], "val": np.s_[:48, :, :]},
    "mitolab_hela_cell": {"test": np.s_[48:, :, :], "val": np.s_[:48, :, :]},
    "mitolab_salivary_gland": {"test": np.s_[250:, :, :], "val": np.s_[:250, :, :]},
    # The Cmito boxes have 128 slices and the dauer worm 364; the first fifth tunes.
    "cmito_l1": {"test": np.s_[26:, :, :], "val": np.s_[:26, :, :]},
    "cmito_l2": {"test": np.s_[26:, :, :], "val": np.s_[:26, :, :]},
    "cmito_l3": {"test": np.s_[26:, :, :], "val": np.s_[:26, :, :]},
    "cmito_adult": {"test": np.s_[26:, :, :], "val": np.s_[:26, :, :]},
    "cmito_dauer": {"test": np.s_[73:, :, :], "val": np.s_[:73, :, :]},
    # The same regions as the DenseCell cells: the val volume is scored, the train volume tunes on z >= 35.
    "densecell_mito": {"test": np.s_[:, :, :], "val": np.s_[35:, :, :]},
    "hemibrain_mito": {"test": np.s_[205:, :, :], "val": np.s_[:205, :, :]},
    # The MitoSegEM boxes have 64 slices; the first fifth tunes.
    **{name: {"test": np.s_[13:, :, :], "val": np.s_[:13, :, :]} for name in MITO_SEGEM_BOXES},
    # The MitoEM val volumes have 100 slices; the first 20 tune.
    **{name: {"test": np.s_[20:, :, :], "val": np.s_[:20, :, :]} for name in MITOEM_SAMPLES},
    # The jurkat test volume is scored whole, its train volume tunes on the slab the generalist holds out (z >= 230).
    "mitoemv2_jurkat": {"test": np.s_[:, :, :], "val": np.s_[230:, :, :]},
    # The MitoNet kidney boxes have 128 slices; the first fifth tunes.
    "mitonet_kidney": {"test": np.s_[26:, :, :], "val": np.s_[:26, :, :]},
    # The Macrophage / A431 fib_control block has 128 slices; the first fifth tunes.
    "macrophage_a431_mito": {"test": np.s_[26:, :, :], "val": np.s_[:26, :, :]},
    # The MitoANF boxes have 128 slices; the first fifth tunes.
    "mito_anf": {"test": np.s_[26:, :, :], "val": np.s_[:26, :, :]},
    # The Donglai Wei MICrONS phase 1 volume has 750 slices; the first fifth tunes, in a window without blank raw.
    "donglai_wei_mito": {"test": np.s_[150:, :, :], "val": np.s_[:150, 512:1024, 2560:3072]},
    # The Cerebellum FIB-SEM volume has 530 slices; the first fifth tunes.
    "cerebellum_fibsem_mito": {"test": np.s_[106:, :, :], "val": np.s_[:106, :, :]},
    # The mitochondria generalist trains on z < 140 of the Lucchi train volume and validates on the rest, which tunes.
    "lucchi": {"test": np.s_[:, :, :], "val": np.s_[140:, :, :]},
    # The SNEMI Mito stack: the mitochondria generalist trains on its train split (z < 50) and validates on its valid
    # split (z 50-69), which tunes; the test split (z 70-99) is scored.
    "snemi_mito": {"test": np.s_[70:, :, :], "val": SNEMI_MITO_VAL_ROI},
}

# The CEM-Mito-Benchmark volumes in the torch-em benchmark numbering (7 is the 2d TEM benchmark).
MITOLAB_BENCHMARK_IDS = {name: i for i, name in enumerate(MITOLAB_VOLUMES, start=1)}
MITOLAB_TEM_BENCHMARK_ID = 7

# The ASEM crops; the generalist trains on 46, 58 and 2E, so only 61 and 64 are scored.
ASEM_MITO_CROPS = {
    "46": (640, 896, 0, 1024, 1112, 2136),
    "58": (152, 408, 0, 1000, 1480, 2504),
    "61": (0, 700, 190, 730, 2630, 3380),
    "64": (400, 641, 326, 802, 2212, 2740),
    "2E": (1015, 1215, 316, 516, 294, 494),
}
ASEM_MITO_EVAL_VOLUMES = ("61", "64")


def asem_crop_name(volume_id):
    """The file name of one ASEM mito crop in <data>/asem/mito_crops."""
    z0, z1, y0, y1, x0, x1 = ASEM_MITO_CROPS[volume_id]
    return f"{volume_id}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}.zarr"


# The test and tuning regions of datasets with several volumes, per volume file name. The tuning slab is the start of
# the annotated stack of each volume, about a fifth of it, and the rest is scored.
EM_VOLUME_ROIS = {
    "asem_mito": {
        asem_crop_name("61"): {"test": np.s_[140:, :, :], "val": np.s_[:140, :, :]},
        asem_crop_name("64"): {"test": np.s_[70:, :, :], "val": np.s_[:70, :, :]},
    },
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
    if dataset_name == "synapsenet_compartments":
        # The test tomograms are read whole; the training tomograms validate on their last 20 % of sections.
        return None if split == "test" else synapsenet_roi(label_path, "val")
    if dataset_name == "axonem":
        # Only a central block of each volume is annotated; its bounding box is cached next to the labels.
        import json
        with open(label_path.replace(".h5", "_roi.json")) as f:
            return tuple(slice(a, b) for a, b in json.load(f)["roi"])
    if dataset_name == "synapseweb":
        region = os.path.basename(label_path).replace("synapseweb_hippocampus_", "").replace(".h5", "")
        return SYNAPSEWEB_CORE_ROIS[region]
    if dataset_name in EM_VOLUME_ROIS:
        return EM_VOLUME_ROIS[dataset_name][os.path.basename(label_path)][split]
    if dataset_name == "platynereis_nuclei":
        # The annotated box of the volume (its valid ROI), as in training; the halo around it is unlabeled.
        sample_id = int(os.path.basename(label_path).replace("train_data_nuclei_", "").replace(".h5", ""))
        rois = _compute_platy_rois(
            os.path.dirname(os.path.dirname(label_path)), [sample_id], ignore_label=PLATYNEREIS_NUCLEI_IGNORE_LABEL,
            file_template="nuclei/train_data_nuclei_%02i.h5", label_key="volumes/labels/nucleus_instance_labels",
            cache_prefix="_nuclei_roi_cache",
        )
        return rois[sample_id]
    rois = EM_ROIS.get(dataset_name)
    return None if rois is None else rois[split]


def val_z_range(dataset_name: str, raw_path: str, split: str) -> Optional[Tuple[int, Optional[int]]]:
    """The z-range of a volume that holds both tuning and test data: the tuning slab for 'val', its complement
    for 'test', so the scored region never overlaps the validation region. None where the two are separate files.
    """
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


# The hand-labeled Cmito dauer patches; 140 of the 200 patches carry labels.
CMITO_MANUAL = "precomputed://https://storage.googleapis.com/gnd-dauer1/mito_label/220114_test"
CMITO_MANUAL_BOX = (32, 544, 32, 544, 0, 200)  # x, y, z: the extent of the labels


def _cmito_dauer_manual_patches(data_root: str, download: bool) -> List[str]:
    """The hand-labeled Cmito dauer patches with labels, one cached file each ('raw', 'labels').

    The block is cached as <data>/cmito/dauer_manual_220114_test.zarr (raw cropped to the labels, z, y, x), streamed
    from the public bucket if it is missing; the patches are written from it to the evaluation cache.
    """
    import h5py
    import zarr

    block_path = os.path.join(data_root, "cmito", "dauer_manual_220114_test.zarr")
    if not os.path.exists(block_path):
        if not download:
            raise FileNotFoundError(f"'{block_path}' does not exist. Pass download=True to stream it from GCS.")
        from cloudvolume import CloudVolume
        from zarr.codecs import BloscCodec

        x0, x1, y0, y1, z0, z1 = CMITO_MANUAL_BOX
        data = {
            key: np.asarray(
                CloudVolume(f"{CMITO_MANUAL}/{name}", use_https=True, progress=False)[x0:x1, y0:y1, z0:z1]
            )[..., 0].transpose(2, 1, 0)
            for key, name in (("raw", "image"), ("labels", "label"))
        }
        tmp_path = f"{block_path}.{os.getpid()}.tmp"
        group = zarr.open_group(tmp_path, mode="w")
        for key, array in data.items():
            group.create_array(
                key, shape=array.shape, chunks=(1, 512, 512), dtype=array.dtype, compressors=BloscCodec(cname="zstd"),
            )[:] = array
        group.attrs["source"] = CMITO_MANUAL
        group.attrs["box_xyz"] = list(CMITO_MANUAL_BOX)
        os.replace(tmp_path, block_path)

    block = zarr.open_group(block_path, mode="r")
    raw, labels = block["raw"][:], block["labels"][:]
    cache_dir = os.path.join(EVAL_CACHE_ROOT, "cmito_dauer_manual")
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for z in np.where(labels.any(axis=(1, 2)))[0]:
        out_path = os.path.join(cache_dir, f"patch_{z:03}.h5")
        if not os.path.exists(out_path):
            tmp_path = f"{out_path}.{os.getpid()}.tmp"
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("raw", data=raw[z], compression="gzip")
                f.create_dataset("labels", data=labels[z], compression="gzip")
            os.replace(tmp_path, out_path)
        paths.append(out_path)
    return paths


def _asem_mito_crop(data_root: str, volume_id: str, download: bool) -> str:
    """The path of one ASEM mito crop (see ASEM_MITO_CROPS), streamed from s3://asem-project if it is missing."""
    path = os.path.join(data_root, "asem", "mito_crops", asem_crop_name(volume_id))
    if os.path.exists(path):
        return path
    if not download:
        raise FileNotFoundError(f"'{path}' does not exist. Pass download=True to stream it from s3://asem-project.")

    import fsspec
    import s3fs
    import zarr
    from zarr.codecs import BloscCodec

    fs = s3fs.S3FileSystem(anon=True)
    remote = f"asem-project/datasets/{volume_id}/{volume_id}.zarr/volumes"

    def open_remote(key):
        array = zarr.open_array(fsspec.get_mapper(f"s3://{remote}/{key}", anon=True), mode="r", zarr_format=2)
        return array, json.loads(fs.cat(f"{remote}/{key}/.zattrs"))

    raw, raw_attrs = open_remote("raw")
    labels, label_attrs = open_remote("labels/mito")
    resolution = np.array(raw_attrs["resolution"])
    # The labels can be a crop of the raw volume, placed by their offset in nm.
    oz, oy, ox = (np.array(label_attrs["offset"]) - np.array(raw_attrs["offset"])) // resolution
    z0, z1, y0, y1, x0, x1 = ASEM_MITO_CROPS[volume_id]
    data = {
        "raw": np.asarray(raw[z0:z1, y0:y1, x0:x1]),
        "labels": np.asarray(labels[z0 - oz:z1 - oz, y0 - oy:y1 - oy, x0 - ox:x1 - ox]),
    }
    assert data["raw"].shape == data["labels"].shape, {key: value.shape for key, value in data.items()}

    tmp_path = f"{path}.{os.getpid()}.tmp"
    group = zarr.open_group(tmp_path, mode="w")
    for key, shuffle in (("raw", "shuffle"), ("labels", "bitshuffle")):
        array = group.create_array(
            key, shape=data[key].shape, chunks=(64, 256, 256), dtype=data[key].dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        array[:] = data[key]
    group.attrs["source"] = f"s3://{remote}/(raw, labels/mito)"
    group.attrs["crop_zyx_raw_voxels"] = list(ASEM_MITO_CROPS[volume_id])
    group.attrs["resolution_nm"] = resolution.tolist()
    os.replace(tmp_path, path)
    return path


# Channels the 2d light microscopy models see, as indices into the stored channel-last image. A single index gives
# a grayscale image. Datasets not listed are fed as stored; a two-channel image gets a zero third channel, which is
# the TissueNet layout pan_multiplex trains with.
LM_IMAGE_CHANNELS = {
    "cvz_fluo_cell": (0, 2, 1), "cvz_fluo_dapi": (0,), "enseg": (1,), "xenium_cells": (1, 2, 3),
    "hela_cytonuc": (2,), "micro_bench": (2,), "spatch_dapi": (0,), "bac_mother": (0,),
    # The Lewy Mito crops are gray values stored as RGB.
    "lewy_mito": (0,),
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

    if dataset_name == "mitolab_tem":
        # The 100 images of the TEM benchmark; a random 10 % (seed 42) tunes, the rest is scored.
        image_dir, mask_dir, _, _, _ = datasets.electron_microscopy.cem.get_benchmark_paths(
            os.path.join(p, "mitolab"), dataset_id=MITOLAB_TEM_BENCHMARK_ID, download=download,
        )
        raw_paths, label_paths = _sorted_pairs(
            sorted(glob(os.path.join(image_dir, "*.tiff"))), sorted(glob(os.path.join(mask_dir, "*.tiff")))
        )
        raw_paths, label_paths = _loader_val_part(raw_paths, label_paths, split)
        return raw_paths, label_paths, None, None

    if dataset_name == "cmito_dauer_manual":
        # The 140 hand-labeled dauer patches; a random 10 % (seed 42) tunes, the rest is scored.
        paths = _cmito_dauer_manual_patches(p, download)
        paths = _loader_val_part(paths, paths, split)[0]
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "probtem":
        # The 6 test images are scored; the 3 training validation images tune.
        root = os.path.join(p, "probtem_full", "processed")
        if split == "test":
            paths = sorted(glob(os.path.join(root, "test", "*.h5")))
        else:
            train_paths = sorted(glob(os.path.join(root, "train", "*.h5")))
            paths = sorted(train_test_split(train_paths, test_size=3, random_state=42)[1])
        return paths, paths, "raw", "labels"

    if dataset_name == "tnbc_mito":
        # The test images of TNBC_MITO_COHORTS without TNBC_MITO_EXCLUDED_TEST_IMAGES are scored, a seeded 10 of their
        # tra_val images tune.
        pairs = []
        for cohort in TNBC_MITO_COHORTS:
            raw_paths, label_paths = datasets.electron_microscopy.tnbc_mito.get_tnbc_mito_paths(
                os.path.join(p, "tnbc_mito"), "tst" if split == "test" else "tra_val", cohort, download=download,
            )
            pairs.extend(zip(raw_paths, label_paths))
        pairs = sorted(pairs)
        if split == "test":
            pairs = [pair for pair in pairs if os.path.basename(pair[0]) not in TNBC_MITO_EXCLUDED_TEST_IMAGES]
        else:
            pairs = sorted(train_test_split(pairs, test_size=10, random_state=42)[1])
        return [raw for raw, _ in pairs], [label for _, label in pairs], None, None

    if dataset_name == "elmi_mito":
        raw_paths, label_paths = datasets.electron_microscopy.elmi_mito.get_elmi_mito_paths(
            os.path.join(p, "elmi_mito"), download=download,
        )
        pairs = [
            pair for pair in sorted(zip(raw_paths, label_paths))
            if (os.path.basename(pair[0]) in ELMI_MITO_TUNING_IMAGES) == (split != "test")
        ]
        return [raw for raw, _ in pairs], [label for _, label in pairs], None, None

    if dataset_name == "lewy_mito":
        return (*_lewy_mito_pairs(os.path.join(p, "lewy_mito"), split, download), None, None)

    if dataset_name == "cardiomyocyte_tem_mito":
        tiles = _cardiomyocyte_tem_eval_tiles(p, download)
        images = sorted({os.path.basename(path).rsplit("_tile", 1)[0] for path in tiles})
        tuning = set(train_test_split(images, test_size=1, random_state=42)[1])
        tiles = [
            path for path in tiles if (os.path.basename(path).rsplit("_tile", 1)[0] in tuning) == (split != "test")
        ]
        return sorted(tiles), sorted(tiles), "raw", "labels"

    if dataset_name == "u2os_fibsem_nucleus":
        paths = sorted(glob(os.path.join(p, U2OS_FIBSEM_NUCLEUS_ROOT, "*.h5")))
        paths = [
            path for path in paths
            if (os.path.basename(path).split("_crop_")[1][:2] in U2OS_FIBSEM_NUCLEUS_TUNING_CROPS) == (split != "test")
        ]
        return paths, paths, "raw", "labels"

    if dataset_name == "perez_mito":
        raw_paths, label_paths = datasets.electron_microscopy.perez.get_perez_paths(
            os.path.join(p, "perez"), "test" if split == "test" else "train", organelle="mito", download=download,
        )
        pairs = sorted(zip(raw_paths, label_paths))
        if split != "test":
            pairs = sorted(train_test_split(pairs, test_size=PEREZ_MITO_N_TUNING, random_state=42)[1])
        return [raw for raw, _ in pairs], [label for _, label in pairs], None, None

    return _get_2d_lm_data_paths(dataset_name, p, download, split)


def _cardiomyocyte_tem_eval_tiles(data_root: str, download: bool) -> List[str]:
    """The cached evaluation tiles of the Cardiomyocyte TEM mitochondria, see CARDIOMYOCYTE_TEM_SCALE."""
    import h5py
    import tifffile
    from skimage.transform import downscale_local_mean

    cache_dir = os.path.join(EVAL_CACHE_ROOT, "cardiomyocyte_tem")
    if os.path.isdir(cache_dir) and os.listdir(cache_dir):
        return sorted(glob(os.path.join(cache_dir, "*.h5")))
    os.makedirs(cache_dir, exist_ok=True)
    raw_paths, label_paths = datasets.electron_microscopy.cardiomyocyte_tem.get_cardiomyocyte_tem_paths(
        os.path.join(data_root, "cardiomyocyte_tem"), "mito", download=download,
    )
    labels_by_name = {os.path.basename(path).replace("KO_", "").replace("_mito.tif", ""): path for path in label_paths}
    tile_shape = tuple(2 * s for s in CROP_SHAPE_2D)
    scale = CARDIOMYOCYTE_TEM_SCALE
    paths = []
    for raw_path in raw_paths:
        name = os.path.basename(raw_path).replace(".tif", "")
        raw = tifffile.imread(raw_path).astype("float32")
        labels = tifffile.imread(labels_by_name[name]) > 0
        raw = downscale_local_mean(raw, (scale, scale))
        labels = labels[::scale, ::scale][:raw.shape[0], :raw.shape[1]]
        candidates = []
        for y in range(0, raw.shape[0] - tile_shape[0] + 1, tile_shape[0]):
            for x in range(0, raw.shape[1] - tile_shape[1] + 1, tile_shape[1]):
                tile = np.s_[y:y + tile_shape[0], x:x + tile_shape[1]]
                n_objects = int(connected_components(labels[tile]).max())
                candidates.append((n_objects, (y, x), tile))
        for n_objects, (y, x), tile in sorted(candidates, key=lambda c: -c[0])[:CARDIOMYOCYTE_TEM_N_TILES]:
            out_path = os.path.join(cache_dir, f"{name}_tile_y{y}_x{x}.h5")
            tmp_path = f"{out_path}.{os.getpid()}.tmp"
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("raw", data=(normalize_raw(raw[tile]) * 255).astype("uint8"), compression="gzip")
                f.create_dataset("labels", data=labels[tile].astype("uint8"), compression="gzip")
                f.attrs["origin_yx_downscaled"] = [y, x]
                f.attrs["scale"] = scale
            os.replace(tmp_path, out_path)
            paths.append(out_path)
    return sorted(paths)


def _lewy_mito_instances(mask: np.ndarray) -> np.ndarray:
    """The instances of a Lewy Mito mask: float mitochondria ids, resampled with interpolation, so every object outline
    carries thin rings of in-between ids. An opening of each id with a disk of radius 2 removes these rings, the
    interiors keep their ids.
    """
    from scipy.ndimage import binary_opening
    from skimage.morphology import disk

    instances = np.zeros(mask.shape, dtype="uint32")
    for object_id in np.unique(mask[mask > 0]):
        instances[binary_opening(mask == object_id, structure=disk(2))] = int(object_id)
    return instances


def _lewy_mito_pairs(root: str, split: str, download: bool) -> Tuple[List[str], List[str]]:
    """The Lewy Mito crops of the test or the tuning subset, see LEWY_MITO_GROUPS."""
    raw_paths, label_paths = datasets.electron_microscopy.lewy_mito.get_lewy_mito_paths(
        root, groups=list(LEWY_MITO_GROUPS), download=download,
    )
    pairs = sorted(zip(raw_paths, label_paths))
    folders = sorted({os.path.dirname(os.path.dirname(raw)) for raw, _ in pairs})
    tuning_folders = set(train_test_split(folders, test_size=0.1, random_state=42)[1])
    pairs = [
        pair for pair in pairs if (os.path.dirname(os.path.dirname(pair[0])) in tuning_folders) == (split != "test")
    ]
    rng = np.random.default_rng(42)
    selected = []
    for index in rng.permutation(len(pairs)):
        raw, label = pairs[index]
        if min(imageio.improps(raw).shape[:2]) >= LEWY_MITO_MIN_SIDE:
            selected.append((raw, label))
        if len(selected) == (LEWY_MITO_N_TEST if split == "test" else LEWY_MITO_N_TUNING):
            break
    selected = sorted(selected)
    return [raw for raw, _ in selected], [label for _, label in selected]


def _get_2d_lm_data_paths(
    dataset_name: str, p: str, download: bool, split: str
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """The 2d light microscopy datasets. 'test' is the blind split, 'val' the loader's validation data."""
    lm = datasets.light_microscopy

    if dataset_name == "livecell":
        img, gt = _get_livecell_paths(input_folder=os.path.join(p, "livecell"), split=split)
        img, gt = drop_excluded_livecell(img, gt, split=split)
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
            if choice == "bact_fluor":
                # The 'wiggins' test folder holds the same 39 images as its train folder, image and mask alike.
                keep = [os.sep + "wiggins" + os.sep not in path for path in img]
                img = [path for path, k in zip(img, keep) if k]
                gt = [path for path, k in zip(gt, keep) if k]
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
            # Read the membrane marker and the nuclei. `select_channels` completes them to the TissueNet layout.
            return sorted(paths), sorted(paths), ("raw/serum_IgG/s0", "raw/nuclei/s0"), "labels/cells/s0"
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

    if dataset_name == "bbbc010":
        # Out of domain: C. elegans worm instances, brightfield channel; the loader's own 65/15/20 split.
        img, gt = lm.bbbc010.get_bbbc010_paths(
            path=os.path.join(p, "bbbc010"), split=split, channel=1, download=download,
        )
        return sorted(img), sorted(gt), "raw", "labels"

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

    if dataset_name == "bbbc032":
        # Out of domain: single sparsely-instanced volume (56 nuclei), tuning uses a z-slab, see LM_VAL_Z_SLABS.
        img, gt = lm.bbbc032.get_bbbc032_paths(path=os.path.join(p, "bbbc032"), channel=3, download=download)
        return img, gt, None, None

    if dataset_name == "bbbc033":
        # Out of domain: single densely-instanced volume (15 nuclei), tuning uses a z-slab, see LM_VAL_Z_SLABS.
        path, raw_key = lm.bbbc033.get_bbbc033_paths(path=os.path.join(p, "bbbc033"), download=download)
        return [path], [path], raw_key, "labels"

    if dataset_name == "bbbc050":
        # Out of domain: the official test split (4 held-out embryos) is scored, the train split tunes.
        img, gt = lm.bbbc050.get_bbbc050_paths(
            path=os.path.join(p, "bbbc050"), split="test" if split == "test" else "train",
            label_type="QCANet", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    raise ValueError(f"Unknown 3D LM dataset: {dataset_name!r}")


def _get_3d_em_data_paths(
    dataset_name: str, data_root: str, download: bool = False, is_val: bool = False
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """The 3d EM datasets. 'is_val' selects the tuning files where they differ from the test files."""
    p = data_root
    em = datasets.electron_microscopy

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

    if dataset_name == "synapsenet_compartments":
        # One tomogram per synapse type is the blind test set, see SYNAPSENET_TEST_TOMOGRAMS; the others validate.
        paths = synapsenet_tomograms(os.path.join(p, "synapsenet_compartments_data"))
        paths = [path for path in paths if (os.path.basename(path) in SYNAPSENET_TEST_TOMOGRAMS) != is_val]
        return paths, paths, "raw", "labels/compartments"

    if dataset_name == "lucchi":
        # Mitochondria. The test volume is scored, the training volume tunes on its validation slab (EM_ROIS);
        # 'load_volume' turns the binary labels into instances.
        path = em.lucchi.get_lucchi_paths(
            path=os.path.join(p, "lucchi"), split="train" if is_val else "test", download=download,
        )
        return [path], [path], "raw", "labels"

    if dataset_name == "snemi_mito":
        # Mitochondria. The stacked SNEMI Mito slices, scored and tuned in the regions of EM_ROIS; 'load_volume' turns
        # the binary labels into instances.
        path = os.path.join(p, SNEMI_MITO_PATH)
        return [path], [path], "raw", "labels"

    if dataset_name == "kasthuri":
        # Mitochondria. The test volume is scored, the train volume tunes; -1 marks the padding around the rotated
        # section, which 'load_volume' excludes from scoring.
        path = em.kasthuri.get_kasthuri_paths(
            path=os.path.join(p, "kasthuri"), split="train" if is_val else "test", download=download,
        )
        return [path], [path], "raw", "labels"

    if dataset_name.startswith("mitolab_"):
        # One volume of the CEM-Mito-Benchmark, scored and tuned in the regions of EM_ROIS.
        raw_path, label_path, _, _, _ = em.cem.get_benchmark_paths(
            os.path.join(p, "mitolab"), dataset_id=MITOLAB_BENCHMARK_IDS[dataset_name.replace("mitolab_", "")],
            download=download,
        )
        return [raw_path], [label_path], None, None

    if dataset_name in CMITO_VOLUMES:
        # One Cmito volume, scored and tuned in the regions of EM_ROIS. Dense instance labels from Cmito model output.
        path = os.path.join(p, "cmito", CMITO_VOLUMES[dataset_name])
        return [path], [path], "raw", "labels"

    if dataset_name in MITOEM_SAMPLES:
        # The MitoEM val volume of one species, scored and tuned in the regions of EM_ROIS. Instance labels.
        paths = em.mitoem.get_mitoem_paths(
            os.path.join(p, "mitoem"), splits=["val"], samples=[MITOEM_SAMPLES[dataset_name]], download=download,
        )
        return paths, paths, "raw", "labels"

    if dataset_name in MITOEMV2_DATASETS:
        # MitoEM v2: the test volumes are scored, the official val volumes tune (jurkat: its train volume, in the
        # region of EM_ROIS). mossy comes from the MitoVerse copies, where 'load_volume' reads the bouton mask.
        name = MITOEMV2_DATASETS[dataset_name]
        if name == "mossy":
            root = os.path.join(p, "mitoverse", "data", "han24")
            volumes = MITOEMV2_MOSSY_VOLUMES["val" if is_val else "test"]
            paths = [os.path.join(root, f"{volume}.zarr") for volume in volumes]
            return paths, paths, "img", "mito"
        mitoemv2_split = "test" if not is_val else ("train" if name == "jurkat" else "val")
        paths = em.mitoemv2.get_mitoemv2_paths(
            os.path.join(p, "mitoemv2"), name, split=mitoemv2_split, download=download,
        )
        return paths, paths, "raw", "labels"

    if dataset_name == "mitonet_kidney":
        # The cached MitoNet-Predicted-Kidney boxes, scored and tuned in the regions of EM_ROIS. Instance labels.
        paths = em.mitonet_predicted_kidney.get_mitonet_predicted_kidney_paths(
            os.path.join(p, "mitonet_predicted_kidney"), bounding_boxes=MITONET_KIDNEY_BOXES, download=download,
        )
        return paths, paths, "raw", "labels"

    if dataset_name == "human_liver_em":
        # The cached human liver boxes: three are scored, a separate one tunes. Semantic labels (0 / 255), which
        # 'load_volume' turns into instances.
        paths = [
            em.human_liver_em.get_human_liver_em_data(os.path.join(p, "human_liver_em_boxes"), bbox, download=download)
            for bbox in HUMAN_LIVER_EM_BOXES["val" if is_val else "test"]
        ]
        return paths, paths, "raw", "mito"

    if dataset_name == "cerebellum_fibsem_mito":
        # The Cerebellum FIB-SEM volume, scored and tuned in the regions of EM_ROIS. Instance labels.
        path = os.path.join(p, CEREBELLUM_FIBSEM_PATH)
        return [path], [path], "raw", "labels/mitochondria"

    if dataset_name == "mito_anf":
        # The cached MitoANF boxes, scored and tuned in the regions of EM_ROIS. Instance labels (partial).
        paths = [os.path.join(p, "mito_anf", name) for name in MITO_ANF_BOXES]
        return paths, paths, "raw", "labels"

    if dataset_name == "donglai_wei_mito":
        # The internal Donglai Wei volume, scored and tuned in the regions of EM_ROIS. Instance labels (partial).
        root = os.path.join(p, DONGLAI_WEI_MITO_ROOT)
        return [os.path.join(root, "microns1_im.h5")], [os.path.join(root, "microns1_mito.h5")], "main", "main"

    if dataset_name == "macrophage_a431_mito":
        # The fib_control block, scored and tuned in the regions of EM_ROIS. Semantic labels, which 'load_volume'
        # turns into instances.
        path = os.path.join(p, MACROPHAGE_A431_MITO_PATH)
        return [path], [path], "raw", "labels/mitochondria"

    if dataset_name == "parlakgul_liver_eval":
        # The manual Parlakgul evaluation crops: the first two crops per sample are scored, the third tunes. Manual
        # instance labels.
        paths = datasets.electron_microscopy.parlakgul_liver_eval.get_parlakgul_liver_eval_paths(
            os.path.join(p, "parlakgul_liver_eval"), samples=list(PARLAKGUL_LIVER_EVAL_SAMPLES), download=download,
        )
        tuning = [path for path in paths if path.endswith(f"_mito_{PARLAKGUL_LIVER_EVAL_TUNING_CROP}.zarr")]
        paths = sorted(tuning if is_val else [path for path in paths if path not in tuning])
        return paths, paths, "raw", "labels"

    if dataset_name in MITO_SEGEM_BOXES:
        # One cached MitoSegEM box, scored and tuned in the regions of EM_ROIS. Instance labels.
        tissue, bounding_box = MITO_SEGEM_BOXES[dataset_name]
        paths = em.mito_segem.get_mito_segem_paths(
            os.path.join(p, "mito_segem"), bounding_boxes=[bounding_box], tissue=tissue, download=download,
        )
        return paths, paths, "raw", "labels"

    if dataset_name == "densecell_mito":
        # The mitochondrion class of DenseCell: the val volume is scored, the train volume tunes on z >= 35.
        path = em.densecell.get_densecell_paths(
            path=os.path.join(p, "densecell"), split="train" if is_val else "val", download=download,
        )
        return [path], [path], "raw", "labels/mitochondrion"

    if dataset_name == "hemibrain_mito":
        # The mitochondria detection of the cached hemibrain box; 'load_volume' keeps its class 1 (mitochondria).
        paths = em.hemibrain.get_hemibrain_paths(
            path=os.path.join(p, "hemibrain"), label_choice="mito", download=download,
        )
        return paths, paths, "raw", "labels"

    if dataset_name == "cellmap_nucleus":
        # The CellMap nucleus crops cached for the nucleus generalist: the held-out crop is scored, the validation crop
        # tunes. Instance labels.
        names = CELLMAP_NUCLEUS_VAL_CROPS if is_val else CELLMAP_NUCLEUS_TEST_CROPS
        paths = [os.path.join(p, "cellmap_nucleus_crops", f"{name}.zarr") for name in names]
        return paths, paths, "raw", "labels"

    if dataset_name in ("fafb_nuclei", "microns_nuclei"):
        # The official test blocks are scored, the val blocks tune. Instance labels of the automatic segmentation.
        paths = sorted(glob(os.path.join(p, dataset_name, "val" if is_val else "test", "*.h5")))
        return paths, paths, "raw", "labels/nuclei"

    if dataset_name == "human_liver_em_nucleus":
        # The 4x downsampled human liver nucleus boxes: three are scored, a separate one tunes. Instance labels.
        names = HUMAN_LIVER_EM_NUCLEUS_BOXES["val" if is_val else "test"]
        paths = [os.path.join(p, "human_liver_em_nucleus_boxes", f"{name}.zarr") for name in names]
        return paths, paths, "raw", "labels"

    if dataset_name == "etchacell_nucleus":
        # The held-out ROIs tune, the other ROIs are scored. Instance labels (one nucleus per ROI).
        paths = [
            path for path in sorted(glob(os.path.join(p, ETCHACELL_NUCLEUS_ROOT, "*.h5")))
            if (os.path.basename(path)[:-3] in ETCHACELL_NUCLEUS_TUNING_ROIS) == is_val
        ]
        return paths, paths, "raw", "labels"

    if dataset_name == "platynereis_nuclei":
        # Volume 03 is scored and volume 12 tunes, inside their valid ROI (see em_roi). Instance labels.
        sample_ids = PLATYNEREIS_NUCLEI_VAL_IDS if is_val else PLATYNEREIS_NUCLEI_TEST_IDS
        paths = [os.path.join(p, "platynereis", "nuclei", f"train_data_nuclei_{i:02}.h5") for i in sample_ids]
        return paths, paths, "volumes/raw", "volumes/labels/nucleus_instance_labels"

    if dataset_name == "asem_mito":
        # The ASEM evaluation crops (64 and the 61 hold-out block), scored and tuned in the regions of
        # EM_VOLUME_ROIS. Semantic labels, which 'load_volume' turns into instances.
        paths = [_asem_mito_crop(p, volume_id, download) for volume_id in ASEM_MITO_EVAL_VOLUMES]
        return paths, paths, "raw", "labels"

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
        # None means the loader has no split of its own; the holdout is the z-slab in VAL_Z_RANGE, which
        # load_volume applies on top of the very same volumes.
        split = VAL_SPLITS[dataset_name] or "test"

    if dataset_name in DATASETS_2D:
        raw_paths, label_paths, raw_key, label_key = _get_2d_data_paths(
            dataset_name, data_root, download=download, split=split
        )
        if split == "test" and not is_val:
            raw_paths, label_paths = drop_excluded_test_images(dataset_name, raw_paths, label_paths)
        return raw_paths, label_paths, raw_key, label_key
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
    crop_start: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load a 3D volume, apply dataset-specific preprocessing, and center-crop.

    valid_roi is a boolean mask that is True where the data is annotated. It is None except for the datasets
    that are annotated only in part (platynereis_cells, synapseweb, plantseg_root, plantseg_ovules, kasthuri) and
    for donglai_wei_mito, whose blank raw strips are excluded.

    'split' selects the test or the tuning region of the EM volumes, see EM_ROIS. 'z_range' restricts the
    volume to a z-slab before the center crop, which is how a dataset without splits holds tuning data out
    of the evaluated slab. See VAL_Z_RANGE. 'crop_start' places the crop at these stored array coordinates
    instead of the center. See EVAL_CROPS_3D.
    """
    # Only the scored region is read: the dataset roi, the z-slab and the center crop are composed into one
    # window first, so the multi-gigavoxel connectome volumes are never loaded whole.
    raw_source = load_image(raw_path) if raw_key is None else open_file(raw_path, mode="r")[raw_key]
    label_source = load_image(label_path) if label_key is None else open_file(label_path, mode="r")[label_key]
    roi = em_roi(dataset_name, label_path, split)
    if crop_start is None:
        window = _read_window(label_source.shape, roi, z_range, crop_shape)
    else:
        region = _read_window(label_source.shape, roi, z_range, label_source.shape)
        if any(start < r.start or start >= r.stop for start, r in zip(crop_start, region)):
            raise ValueError(
                f"The pinned crop {crop_start} of '{dataset_name}' starts outside its test region. "
                "Fix its start in eval_crops_3d.json."
            )
        window = tuple(
            slice(start, min(start + size, r.stop)) for start, size, r in zip(crop_start, crop_shape, region)
        )
    raw, labels = np.asarray(raw_source[window]), np.asarray(label_source[window])

    valid_roi = None
    if dataset_name == "platynereis_cells":
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
    elif dataset_name in ("plantseg_ovules", "kasthuri", "platynereis_nuclei"):
        # -1 marks unannotated voxels (for kasthuri the padding around the rotated section, for platynereis_nuclei
        # the halo around the annotated box).
        labels = labels.astype("int64")
        valid_roi = labels != -1
        labels[labels == -1] = 0
    elif dataset_name == "pnas_arabidopsis":
        # The background carries id 1.
        labels[labels == 1] = 0
    elif dataset_name == "hemibrain_mito":
        # Class 1 of the 4-class detection is mitochondria; 2 (interior), 3 (background) and 4 (unlabeled) are not.
        labels = (labels == 1).astype("uint32")
    elif dataset_name == "donglai_wei_mito":
        # Blank raw marks missing data at the section edges, where the labels continue; it is excluded from scoring.
        valid_roi = raw > 0
        labels[~valid_roi] = 0
    elif dataset_name == "mitoemv2_mossy":
        # Only the mossy fiber boutons (the MitoVerse bouton mask) are annotated; the rest is excluded from scoring.
        valid_roi = np.asarray(open_file(label_path, mode="r")["mask"][window]) > 0
        labels[~valid_roi] = 0

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
    """Return the joint trainer checkpoint for a model type, e.g. 'hvit_b'.

    A training version tags its run directory, as in 'joint_sam2_hvit_t_multi_gpu_v5a', so the name
    is matched by prefix. One root holds one run per model type.
    """
    root = _joint_checkpoint_root()
    runs = sorted(glob(os.path.join(root, f"joint_sam2_{model_type}_multi_gpu*")))
    if len(runs) > 1:
        raise RuntimeError(f"'{root}' holds several runs for '{model_type}': {[os.path.basename(r) for r in runs]}.")
    path = os.path.join(runs[0], f"{checkpoint}.pt") if runs else ""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"There is no joint '{checkpoint}' checkpoint for '{model_type}' under '{root}'.")
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
    ('decoder_state', 'unetr_state' before v6) and pickled trainer state in a single file. That file cannot be loaded by
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
    if "model_state" not in state or not ("decoder_state" in state or "unetr_state" in state):
        raise RuntimeError(f"'{checkpoint_path}' is not a joint checkpoint.")

    os.makedirs(export_root, exist_ok=True)
    _save_atomic({"model": _strip_ddp_prefix(state["model_state"]), "model_type": model_type}, interactive_path)
    _save_atomic(_strip_ddp_prefix(joint_unetr_state(state)), decoder_path)
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
    "densecell_mito": (5, 1, 1),
    "cmito_l1": (50 / 5.12, 1, 1),  # ssTEM: 50 nm sections, 5.12 nm pixels
    "cmito_l2": (30 / 8, 1, 1),  # 30 nm sections, 8 nm pixels
    "cmito_l3": (50 / 6.144, 1, 1),  # 50 nm sections, 6.144 nm pixels
    "cmito_adult": (30 / 16, 1, 1),  # 30 nm sections, 16 nm pixels
    "cmito_dauer": (50 / 8, 1, 1),  # 50 nm sections, 8 nm pixels
    "hemibrain_mito": (1, 1, 1),  # FIB-SEM, 8 nm isotropic
    "kasthuri": (10, 1, 1),  # ATUM-SEM, 3 nm pixels, 30 nm sections
    "snemi_mito": (30 / 6, 1, 1),  # ssSEM, 6 nm pixels, 30 nm sections
    "mito_segem_intestine": (50 / 8, 1, 1),  # SBF-SEM: 50 nm sections, 8 nm pixels
    "human_liver_em": (55 / 8, 1, 1),  # SBF-SEM: ~55 nm sections, 8 nm pixels
    "cellmap_nucleus": (1, 1, 1),  # FIB-SEM, 32 nm isotropic
    "fafb_nuclei": (40 / 32, 1, 1),  # ssTEM: 40 nm sections, 32 nm pixels
    "microns_nuclei": (40 / 64, 1, 1),  # ssTEM: 40 nm sections, 64 nm pixels
    "platynereis_nuclei": (100 / 80, 1, 1),  # SBEM: about 100 nm sections, 80 nm pixels
    "human_liver_em_nucleus": (55 / 32, 1, 1),  # SBF-SEM: ~55 nm sections, 8 nm pixels downsampled 4x
    "etchacell_nucleus": (50 / 40, 1, 1),  # SBF-SEM: 50 nm sections, 10 nm pixels downsampled 4x
    "mitoem_human": (30 / 8, 1, 1),  # ATUM-SEM: 30 nm sections, 8 nm pixels
    "mitoem_rat": (30 / 8, 1, 1),  # ATUM-SEM: 30 nm sections, 8 nm pixels
    **{name: (1, 1, 1) for name in ("mitoemv2_beta", "mitoemv2_jurkat", "mitoemv2_macro", "mitoemv2_podo")},
    "mitoemv2_sperm": (1, 1, 1),  # FIB-SEM, 16 nm isotropic
    "mitoemv2_stem": (30 / 8, 1, 1),  # SBF-SEM, 8 x 8 x 30 nm (MitoEM v2 metadata)
    "mitoemv2_mossy": (30 / 8, 1, 1),  # ssSEM, 8 x 8 x 30 nm
    "mitonet_kidney": (1, 1, 1),  # FIB-SEM, 16 nm isotropic
    "parlakgul_liver_eval": (1, 1, 1),  # FIB-SEM, 8 nm isotropic
    "cerebellum_fibsem_mito": (1, 1, 1),  # FIB-SEM, 10 nm
    "macrophage_a431_mito": (12 / 6, 1, 1),  # FIB-SEM, 6 x 6 x 12 nm
    "mito_anf": (50 / 12, 1, 1),  # SBEM, 12 x 12 x 50 nm
    "donglai_wei_mito": (40 / 8, 1, 1),  # ssEM, 8 x 8 x 40 nm
    "synapsenet_compartments": (1, 1, 1),  # electron tomography, isotropic
    "nisb": (2.2, 1, 1),  # synthetic: 20 nm sections, 9 nm pixels
    "bbbc032": (5, 1, 1),  # spinning disk confocal: z=0.5µm, xy=0.101µm
    "bbbc050": (2.5, 1, 1),  # CV1000 (test split): z=2.0µm, xy=0.8µm
}


# The parameters `AutomaticPromptGenerator.generate` accepts, so a run can be described by one dict.
GENERATE_PARAM_KEYS = (
    "candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size",
    "score_threshold", "max_overlap", "min_size", "max_size_factor", "refinement", "refinement_kwargs",
    "multimasking", "n_objects_per_pass", "early_stop_patience", "propagation_waves", "batch_size", "n_threads",
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


def load_apg_overrides(path, dataset_name):
    """Read one APG configuration file and return its name and the overrides for one dataset.

    The file has the format of the optimization benchmark: ``{"name": ..., "params_2d": {...},
    "params_3d": {...}}``, with an optional ``params_dense``. Images use 'params_2d' and volumes use
    'params_3d'. The dense-neuron EM volumes use 'params_dense' if the file has it. The function returns
    the overrides unresolved, so that they can go on top of tuned parameters. `resolve_params` fills in
    the defaults.

    Args:
        path: The JSON configuration file.
        dataset_name: The dataset that the overrides are for. It selects the section.

    Returns:
        The configuration name and the overrides, keyed as `generate` takes them.
    """
    import json

    with open(path) as f:
        config = json.load(f)
    unknown_top_level = set(config) - {"name", "params_2d", "params_3d", "params_dense"}
    if unknown_top_level:
        raise ValueError(f"Unknown configuration fields in '{path}': {sorted(unknown_top_level)}.")
    if dataset_name in DATASETS_DENSE and "params_dense" in config:
        section = "params_dense"
    else:
        section = "params_3d" if dataset_name in DATASETS_3D else "params_2d"
    overrides = config.get(section, {})
    if not isinstance(overrides, dict):
        raise TypeError(f"'{section}' in '{path}' must be an object.")
    unknown = set(overrides) - set(GENERATE_PARAM_KEYS)
    if unknown:
        raise ValueError(f"Unknown APG parameters in '{path}': {sorted(unknown)}.")
    return str(config.get("name", Path(path).stem)), dict(overrides)


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
    ('decoder_state' next to 'model_state', 'unetr_state' before v6) and exported decoder weights.

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
        joint_checkpoint_path: Absolute path to a joint checkpoint (has 'model_state' and 'decoder_state').
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
        _save_atomic(_strip_ddp_prefix(joint_unetr_state(state)), decoder_path)

    return build_model(mode="ais", model_type=model_type, device=device, ndim=ndim, checkpoint_path=decoder_path)


def build_apg_model_from_checkpoint(joint_checkpoint_path, model_type="hvit_t", device="cuda", ndim=2, cache_dir=None):
    """Build the APG (prompt generator) from a specific joint-trainer checkpoint file, by explicit path.

    Mirrors `build_apg_segmenter`, but loads both halves (interactive model_state and decoder
    unetr_state) directly from `joint_checkpoint_path` instead of through `_joint_checkpoint_root()` /
    `_joint_export_root()` and the `joint_sam2_{model_type}_multi_gpu` naming convention - handy for a
    one-off checkpoint that lives outside that directory layout.

    Args:
        joint_checkpoint_path: Absolute path to a joint checkpoint (has 'model_state' and 'decoder_state').
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
        _save_atomic(_strip_ddp_prefix(joint_unetr_state(state)), decoder_path)

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
    """Turn a (5, *spatial) prediction into an instance segmentation.

    EM datasets use the dense (multicut) mode, all others the sparse (flow) mode. 'params' overrides
    the postprocessing defaults, e.g. with the best combination found by grid_search_automatic_cells.
    Without 'params', 'model_type' selects the per-model library default. The boundary channel 4 is
    forwarded to the sparse mode as 'boundary'.
    """
    from micro_sam.v2.util import UNISAM2_OUTPUT_CHANNELS
    from micro_sam.v2.postprocessing import flow_instance_segmentation, run_multicut
    if out.shape[0] != UNISAM2_OUTPUT_CHANNELS:
        raise ValueError(f"Expected a {UNISAM2_OUTPUT_CHANNELS}-channel UniSAM2 prediction, got {out.shape[0]}.")
    params = {} if params is None else params
    fg = out[0]
    if dataset_name in DATASETS_DENSE:
        boundary_map = fg.max() - fg
        boundary_map /= boundary_map.max()
        distances = np.stack([out[2], out[3]])
        seg = run_multicut(boundary_map, distances, model_type=model_type, **params)
    else:
        spacing = DATASET_SPACING.get(dataset_name, None)
        seg = flow_instance_segmentation(
            fg, out[1:4], boundary=out[4], model_type=model_type, spacing=spacing, **params
        )
    return seg.astype("uint32")


def run_dataset_evaluation(gt_paths, prediction_paths, dataset_name: str, save_path: str):
    """Score a dataset and write the results to 'save_path'.

    The instance segmentations report the symmetric best Dice (SBD) next to mSA. Neuron segmentation in EM is
    ranked by the CREMI score instead of mSA. Those datasets report the VI and adapted-Rand components instead.

    Args:
        gt_paths: The ground-truth label arrays, or the paths to them.
        prediction_paths: The predicted segmentations, or the paths to them.
        dataset_name: The dataset the segmentations belong to.
        save_path: The filepath to write the result CSV to.

    Returns:
        The results as a DataFrame.
    """
    from bioimage_py.evaluation import symmetric_best_dice_score
    from micro_sam.v1.evaluation.evaluation import run_evaluation

    if dataset_name not in DATASETS_DENSE:
        results = run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths, save_path=save_path)
        # Read files like run_evaluation does: it relabels the ground truth, but not the predictions.
        results["SBD"] = float(np.mean([
            symmetric_best_dice_score(
                seg if isinstance(seg, np.ndarray) else imageio.imread(seg),
                gt if isinstance(gt, np.ndarray) else connected_components(imageio.imread(gt)),
            )
            for gt, seg in zip(gt_paths, prediction_paths)
        ]))
        results.to_csv(save_path, index=False)
        return results

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


def sample_row_path(save_path: str, sample_index: int) -> str:
    """Return the path of the result row of one sample."""
    return os.path.join(f"{save_path[:-4]}_samples", f"sample_{sample_index:04d}.csv")


def evaluate_samples(
    segment_fn, dataset_name, data_root, save_path, desc, limit=None, crop_shape=None, sample_index=None,
    extra_columns=None, normalize=True,
):
    """Segment and score the test samples of a dataset, and write the result CSV.

    With 'sample_index', the function scores only that sample and stores its row next to 'save_path'. A task that
    finds the rows of all samples writes the dataset result, so the array tasks of one dataset need no merge step.
    Every metric is a mean over the samples, so the mean of the rows equals the score of one run over all samples.

    Args:
        segment_fn: The function that maps one image or volume to its instance segmentation.
        dataset_name: The dataset to score.
        data_root: The root the data lives in.
        save_path: The path of the result CSV.
        desc: The label of the progress bar.
        limit: The number of samples to score, counted from the first one.
        crop_shape: The 3d center crop.
        sample_index: The index of the only sample to score.
        extra_columns: The columns to add to the result, such as the parameters of the run.
        normalize: Percentile-normalize the image, 2d only. See `load_evaluation_sample_2d`.

    Returns:
        The results as a DataFrame, or None while the rows of other samples are missing.
    """
    import pandas as pd
    from tqdm import tqdm

    ndim = 3 if dataset_name in DATASETS_3D else 2
    total = n_samples(dataset_name, data_root)
    samples = load_data(dataset_name, data_root, ndim, crop_shape=crop_shape, normalize=normalize)
    first = 0
    if sample_index is not None:
        if limit is not None:
            raise ValueError("Got both 'sample_index' and 'limit'. Pass only one of them.")
        if os.path.exists(sample_row_path(save_path, sample_index)):
            samples, total = [], 0
        else:
            samples, total, first = islice(samples, sample_index, sample_index + 1), 1, sample_index
    elif limit is not None:
        samples, total = islice(samples, limit), min(total, limit)

    all_gt, all_seg, misses = [], [], []
    for index, (raw, labels, valid_roi) in enumerate(tqdm(samples, total=total, desc=desc), start=first):
        if labels.max() == 0:  # The sample has no ground truth to score.
            if sample_index is not None:
                row_path = sample_row_path(save_path, index)
                os.makedirs(os.path.dirname(row_path), exist_ok=True)
                pd.DataFrame({"skipped": [True]}).to_csv(f"{row_path}.{os.getpid()}.tmp", index=False)
                os.replace(f"{row_path}.{os.getpid()}.tmp", row_path)
            continue

        seg = segment_fn(raw)
        if valid_roi is not None:
            seg[~valid_roi] = 0
        if ndim == 2:
            # The ground truth has no severed objects either, so predicting one is not a false positive.
            seg = drop_severed_objects(seg, GT_MIN_SIZE_2D.get(dataset_name, 0))
        else:
            misses.append(genuine_misses(labels, seg))

        if sample_index is not None:
            row_path = sample_row_path(save_path, index)
            os.makedirs(os.path.dirname(row_path), exist_ok=True)
            # Rename a complete file into place, since the task that merges the rows can read it at any time.
            tmp_path = f"{row_path}.{os.getpid()}.tmp"
            row = run_dataset_evaluation([labels], [seg], dataset_name, tmp_path)
            if misses:
                row["unmatched"], row["genuine_misses"] = misses[0]
            row.to_csv(tmp_path, index=False)
            os.replace(tmp_path, row_path)
        else:
            all_gt.append(labels)
            all_seg.append(seg)

    if sample_index is None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results = run_dataset_evaluation(all_gt, all_seg, dataset_name, save_path)
        if misses:
            # The aggregate metric hides which objects went missing.
            results["unmatched"] = sum(count[0] for count in misses)
            results["genuine_misses"] = sum(count[1] for count in misses)
    else:
        row_paths = [sample_row_path(save_path, index) for index in range(n_samples(dataset_name, data_root))]
        missing = [path for path in row_paths if not os.path.exists(path)]
        if missing:
            print(f"Scored sample {sample_index}. {len(missing)} of {len(row_paths)} rows are still missing.")
            return None
        frames = [pd.read_csv(path) for path in row_paths]
        # Refuse rows with other metrics, since a metric must cover every sample of the mean.
        if len({tuple(frame.columns) for frame in frames if "skipped" not in frame}) > 1:
            raise RuntimeError(f"The sample rows of '{save_path}' hold different metrics. Re-score the older rows.")
        rows = pd.concat(frames, ignore_index=True).drop(columns="skipped", errors="ignore")
        counts = [column for column in ("unmatched", "genuine_misses") if column in rows]
        results = rows.drop(columns=counts).mean().to_frame().T
        for column in counts:
            results[column] = int(rows[column].sum())

    for column, value in (extra_columns or {}).items():
        results[column] = value
    # Tasks of one dataset can finish at the same time, so rename a complete file into place.
    tmp_path = f"{save_path}.{os.getpid()}.tmp"
    results.to_csv(tmp_path, index=False)
    os.replace(tmp_path, save_path)
    print(results)
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
        # An empty cell is a swept None (e.g. APG's 'no refinement'), which must not come back as ''.
        if value == "":
            params[key] = None
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

# The 2d datasets scored on the whole stored image instead of the center crop, e.g. the TEM benchmark images (most of
# them 1024 x 1024, whose center crop drops half of the mitochondria).
FULL_IMAGE_2D = frozenset(DATASETS_2D_EM_MITO + DATASETS_2D_EM_NUCLEUS)

# Ground-truth size floor of 3d datasets whose instances come from semantic labels: the connected components of
# a semantic mask include specks of a few voxels that are no objects.
GT_MIN_SIZE_3D = {
    "asem_mito": 100, "densecell_mito": 50, "hemibrain_mito": 100, "lucchi": 100, "snemi_mito": 100, "kasthuri": 100,
    **{name: 100 for name in [*MITO_SEGEM_BOXES, *MITOEM_SAMPLES, *MITOEMV2_DATASETS, "mitonet_kidney"]},
    "human_liver_em": 100, "parlakgul_liver_eval": 100, "cerebellum_fibsem_mito": 100,
    "macrophage_a431_mito": 100, "mito_anf": 100, "donglai_wei_mito": 100,
    **{name: 100 for name in DATASETS_3D_EM_NUCLEUS},
}

# The fixed crops of the 3d test volumes, which replace the center crop: the crop shape and the crop starts per volume,
# in stored array coordinates. eval_crops_3d.json also holds the rule that places the crops and a note per dataset.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_crops_3d.json")) as f:
    EVAL_CROPS_3D = {
        name: (tuple(entry["crop_shape"]), {volume: tuple(map(tuple, v)) for volume, v in entry["starts"].items()})
        for name, entry in json.load(f)["datasets"].items()
    }


def eval_crops_3d(dataset_name, raw_path, split, crop_shape):
    """Return the (crop start, crop shape) pairs for one volume: its crops in EVAL_CROPS_3D or one center crop."""
    if split != "test" or dataset_name not in EVAL_CROPS_3D:
        return [(None, crop_shape or CROP_SHAPE_3D)]
    if crop_shape is not None:
        raise ValueError(f"'{dataset_name}' has fixed crops in EVAL_CROPS_3D. Do not pass a crop shape.")
    shape, starts = EVAL_CROPS_3D[dataset_name]
    return [(start, shape) for start in starts[os.path.basename(raw_path)]]


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
    ndim=2, use_masks=True, mask_threshold=0.0, min_size=0, click_protocol="random",
):
    """Build the name of the result CSV for one iteration of an interactive run.

    The name encodes every setting that changes the numbers, so one run cannot reuse the results of
    another.
    """
    dim_suffix = "" if ndim == 2 else "_3d"
    tag = interactive_run_tag(ndim, use_masks, mask_threshold, min_size, click_protocol)
    return f"{dataset_name}_{method}_{model_type}{dim_suffix}_{prompt}{tag}_iter{iteration:02d}.csv"


def interactive_run_tag(ndim=2, use_masks=True, mask_threshold=0.0, min_size=0, click_protocol="random"):
    """Build the settings suffix for an interactive run's result names and prediction directory.

    Both use one tag, so a run can never read back the cached predictions of another run.
    """
    # Only the 2d path chooses between mask logits and binarized masks.
    tag = "" if ndim == 3 else ("_with_masks" if use_masks else "_without_masks")
    if ndim == 2 and mask_threshold != 0.0:
        tag += f"_t{mask_threshold:g}"
    if min_size:
        tag += f"_min{min_size}"
    # Only the 2d path chooses the click protocol, see 'CLICK_PROTOCOLS'.
    if ndim == 2 and click_protocol != "random":
        tag += f"_{click_protocol}"
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


def load_evaluation_sample_2d(raw_path, label_path, raw_key, label_key, dataset_name, normalize=True):
    """Load one 2d sample the way the evaluation scores it.

    The parameter search and the evaluation both call this function, so both use the same data.

    `normalize=False` returns the stored 8-bit values instead. The percentile stretch rescales every
    channel on its own, which restains histopathology RGB, so a model trained on native slide colour
    reads the image differently than it was trained to.
    """
    # Normalize before cropping, so that the percentiles cover the whole image.
    image = select_channels(read_2d(raw_path, raw_key), dataset_name)
    image = ensure_8bit_range(image) if normalize else np.clip(image, 0, 255).astype("uint8")
    if dataset_name in FULL_IMAGE_2D:
        roi = tuple(slice(None) for _ in image.shape[:2])
    else:
        roi = _center_crop_roi(image.shape[:2], CROP_SHAPE_2D)
    labels = read_2d(label_path, label_key)[roi]
    if dataset_name == "astih":
        # Semantic labels (1 myelin, 2 axon); the instances are the axon class, as in training.
        labels = labels == 2
    elif dataset_name == "lewy_mito":
        labels = _lewy_mito_instances(labels)
    gt = connected_components(labels).astype("uint32")
    return image[roi], drop_severed_objects(gt, GT_MIN_SIZE_2D.get(dataset_name, 0))


def load_evaluation_sample_3d(
    raw_path, label_path, raw_key, label_key, dataset_name,
    crop_shape=CROP_SHAPE_3D, z_range=None, min_size=0, split="test", crop_start=None,
):
    """Load one volumetric sample the way the evaluation scores it."""
    raw, labels, valid_roi = load_volume(
        raw_path, label_path, raw_key, label_key, dataset_name, crop_shape, z_range=z_range, split=split,
        crop_start=crop_start,
    )
    min_size = max(min_size, GT_MIN_SIZE_3D.get(dataset_name, 0))
    return raw, apply_min_size(labels, min_size, dataset_name), valid_roi


def load_data(
    dataset_name, data_root, ndim, min_size=0, split="test", crop_shape=None, z_range=None, normalize=True,
    sample_ids=None,
):
    """Yield (image_or_volume, labels, valid_roi) triples for the given dataset.

    valid_roi is a boolean mask that is True where the data is annotated. It is None except for the datasets
    that are annotated only in part, see load_volume.

    The filtering happens here, in the single source of the labels used for both prompting and
    scoring. If only the prompting copy were filtered, the dropped objects would stay in the scored
    ground truth and count as unmatched.

    Args:
        dataset_name: The dataset to load.
        data_root: The root the data lives in.
        ndim: The number of spatial dimensions, 2 or 3.
        min_size: Drop ground-truth objects below this many pixels (3d only).
        split: The split to load, 'test' or the held-out 'val', see VAL_SPLITS.
        crop_shape: The 3d center crop. Defaults to CROP_SHAPE_3D. The test volumes in EVAL_CROPS_3D use
            their fixed crops instead, with one sample per crop.
        z_range: Restrict a volume to a z-slab before cropping, see VAL_Z_RANGE.
        normalize: Percentile-normalize the image, 2d only. See `load_evaluation_sample_2d`.
        sample_ids: Load and yield only the samples at these positions of the full sequence. The other
            samples are skipped without being read.

    Yields:
        One (image_or_volume, labels, valid_roi) triple per sample.
    """
    raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root, split=split)
    sample_id = -1
    for raw_path, label_path in sorted_path_pairs(raw_paths, label_paths):
        if ndim == 3:
            sample_z_range = val_z_range(dataset_name, raw_path, split) or z_range
            for crop_start, sample_crop_shape in eval_crops_3d(dataset_name, raw_path, split, crop_shape):
                sample_id += 1
                if sample_ids is not None and sample_id not in sample_ids:
                    continue
                yield load_evaluation_sample_3d(
                    raw_path, label_path, raw_key, label_key, dataset_name, crop_shape=sample_crop_shape,
                    z_range=sample_z_range, min_size=min_size, split=split, crop_start=crop_start,
                )
        else:
            sample_id += 1
            if sample_ids is not None and sample_id not in sample_ids:
                continue
            image, gt = load_evaluation_sample_2d(
                raw_path, label_path, raw_key, label_key, dataset_name, normalize=normalize
            )
            yield image, gt, None


def n_samples(dataset_name, data_root, split="test"):
    """Return the number of samples that `load_data` yields for a split."""
    raw_paths = get_data_paths(dataset_name, data_root, split=split)[0]
    if split == "test" and dataset_name in EVAL_CROPS_3D:
        starts = EVAL_CROPS_3D[dataset_name][1]
        return sum(len(starts[os.path.basename(path)]) for path in raw_paths)
    return len(raw_paths)


def has_val_split(dataset_name: str) -> bool:
    """Whether a dataset holds data out from the samples the evaluation scores.

    Only these datasets can be tuned honestly: everywhere else a sweep would select its parameters
    on the very samples the reported score is measured on. See VAL_SPLITS and VAL_Z_RANGE.
    """
    return dataset_name in VAL_SPLITS
