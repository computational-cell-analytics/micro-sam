"""Create the overview table of the training and evaluation data of the micro-sam2 generalist.

The table has one row per dataset and these columns:

- ``dataset``: The dataset name. Evaluated datasets use their name in ``finetuning/v2/evaluation/common.py``.
- ``modality``: fluorescence, label-free, EM or histopathology; XPRESS is X-ray. A dataset that mixes modalities
  gets its majority modality, see the comments of cellpose and neurips_cellseg.
- ``structure``: The labeled structure, such as cell, nucleus or neurite.
- ``dimensionality``: 2d or 3d.
- ``role``: ``train`` (training data only), ``ID`` (training data with a scored test split) or ``OOD`` (scored
  test split of a dataset that is meant to be kept out of training). ID and OOD follow the ``PANEL_DATASETS`` of
  the Figure 2 (2d) and Figure 3 (3d) plotting scripts in ``scripts/plotting/v2/for_paper``.
- ``train_images_or_volumes``, ``train_objects``: The images or volumes and the labeled objects of the training
  data (train + val split) of the 'all' generalist, i.e. the 'lm', 'em' and 'hp' domains of ``generalist_loader.py``.
  They are counted with ``micro_sam.v2.datasets.data_counter``, so they follow the splits and the label
  pre-processing of the loader. See its docstring for how stacks, regions of interest and multi-target images count.
- ``test_images_or_volumes``, ``test_objects``: The test samples that the evaluation scores and their objects,
  loaded with ``common.load_data``. The evaluation scores a 512 x 512 center crop of most 2d images and fixed
  crops of the 3d volumes (8 x 512 x 512 by default, see ``EVAL_CROPS_3D``), so for 3d data a test sample is one
  crop and the objects are the ones inside the crops. Samples without labels are not scored and not counted.
  The columns are blank for datasets without a scored test split.
- ``comment``: Notes, e.g. on the subsets, the annotation or the imaging technique of a dataset.

The table (``data-overview/data_overview.csv`` by default) is also the cache: the counts are only computed for
rows whose count cells are empty, or for all requested rows with ``--force``. The table is written after every
finished dataset, so an interrupted run keeps its progress. Computing the training counts first builds the training
datasets of a domain like the loader does. This takes several minutes and up to 15 GB of memory per domain.
Counting all datasets needs a CPU job with about 64 GB of memory and 8 workers.

Usage:

    python create_data_overview.py  # fill in the missing counts
    python create_data_overview.py --datasets livecell  # only process this dataset
    python create_data_overview.py --datasets livecell --force  # recompute its counts
"""

import os
import ast
import csv
import sys
import argparse
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, *[os.pardir] * 3))
EVALUATION_DIR = os.path.join(REPO_ROOT, "finetuning", "v2", "evaluation")
FIGURE_SCRIPTS = {
    2: os.path.join(REPO_ROOT, "scripts", "plotting", "v2", "for_paper", "plots_for_figure2.py"),
    3: os.path.join(REPO_ROOT, "scripts", "plotting", "v2", "for_paper", "plots_for_figure3.py"),
}
DEFAULT_DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "data-overview", "data_overview.csv")

TRAIN_COLUMNS = ["train_images_or_volumes", "train_objects"]
TEST_COLUMNS = ["test_images_or_volumes", "test_objects"]
COLUMNS = ["dataset", "modality", "structure", "dimensionality", "role"] + TRAIN_COLUMNS + TEST_COLUMNS + ["comment"]
ROLE_ORDER = {"ID": 0, "OOD": 1, "train": 2}

INTERNAL = "Internal data, not public."


def _common():
    """Import the evaluation module ``common.py``, which resolves and loads the test data."""
    if EVALUATION_DIR not in sys.path:
        sys.path.insert(0, EVALUATION_DIR)
    import common
    return common


#
# The registry of the datasets.
#

def _path_has(*parts: str) -> Callable:
    """Select the label sources whose path contains one of the parts."""
    return lambda unit: any(part in unit.label_path for part in parts)


def _key_is(key: str) -> Callable:
    """Select the label sources with this internal label key."""
    return lambda unit: unit.label_key == key


def _deepbacs_fluorescence(unit) -> bool:
    return "/mixed/" in unit.label_path and _common().deepbacs_is_fluorescence(unit.label_path)


def _deepbacs_label_free(unit) -> bool:
    return "/e_coli_stationary/" in unit.label_path or not _common().deepbacs_is_fluorescence(unit.label_path)


@dataclass(frozen=True)
class Train:
    """The training label sources of a dataset: a domain of the loader, a top-level data folder and a filter."""
    domain: str
    folder: str
    select: Optional[Callable] = None


@dataclass(frozen=True)
class Entry:
    """One row of the table. ``test`` is the dataset name of the evaluation, if its test split is scored."""
    name: str
    modality: str
    structure: str
    ndim: int
    train: Optional[Train] = None
    test: Optional[str] = None
    comment: str = ""


FL, LF, EM, HP = "fluorescence", "label-free", "EM", "histopathology"


def _lm(folder, select=None):
    return Train("lm", folder, select)


def _em(folder, select=None):
    return Train("em", folder, select)


def _hp(folder, select=None):
    return Train("hp", folder, select)


CTC_LABEL_FREE = ("/BF-C2DL-HSC/", "/BF-C2DL-MuSC/", "/DIC-C2DH-HeLa/", "/PhC-C2DH-U373/", "/PhC-C2DL-PSC/")
NEURIPS_TRAIN = "Training data in row neurips_cellseg, whose images are not split by modality."

ENTRIES = [
    # Light microscopy, 2d.
    Entry("bbbc030", LF, "cell", 2, _lm("bbbc030")),
    Entry("bccd", LF, "cell", 2, _lm("bccd"), comment="Blood cells in brightfield blood smears."),
    Entry("bitdepth_nucseg", FL, "nucleus", 2, _lm("bitdepth_nucseg")),
    Entry("bmgd", FL, "nucleus", 2, _lm("bmgd"), "bmgd"),
    Entry("cell_acdc", LF, "cell", 2, _lm("cell_acdc"), "cell_acdc"),
    Entry("cellbindb", FL, "nucleus", 2, _lm("cellbindb"), "cellbindb", "DAPI, ssDNA and mIF stains."),
    Entry(
        "cellpose", FL, "cell", 2, _lm("cellpose"),
        comment="Mixed modalities (cyto and cyto2), counted as fluorescence, the majority: 45% of the images have a "
        "dark fluorescence background, most others are dense fluorescence tissue or non-microscopy images, "
        "few are brightfield.",
    ),
    Entry("cellular", LF, "cell", 2, _lm("cellular")),
    Entry("cisd", LF, "cell", 2, _lm("cisd")),
    Entry(
        "ctc_label_free", LF, "cell", 2, _lm("ctc", _path_has(*CTC_LABEL_FREE)),
        comment="Cell Tracking Challenge: BF-C2DL-HSC, BF-C2DL-MuSC, DIC-C2DH-HeLa, PhC-C2DH-U373, PhC-C2DL-PSC."
    ),
    Entry(
        "ctc_fluorescence_cells", FL, "cell", 2, _lm("ctc", _path_has("/Fluo-C2DL-Huh7/", "/Fluo-C2DL-MSC/")),
        comment="Cell Tracking Challenge: Fluo-C2DL-Huh7, Fluo-C2DL-MSC."
    ),
    Entry(
        "ctc_fluorescence_nuclei", FL, "nucleus", 2, _lm("ctc", _path_has("/Fluo-N2DH-SIM+/")),
        comment="Cell Tracking Challenge: Fluo-N2DH-SIM+, simulated nuclei."
    ),
    Entry("cvz_fluo_cell", FL, "cell", 2, _lm("cvz", _path_has("_Cell_")), "cvz_fluo_cell"),
    Entry("cvz_fluo_dapi", FL, "nucleus", 2, _lm("cvz", _path_has("_Dapi_"))),
    Entry("dcis_com_nuclei", FL, "nucleus", 2, _lm("dcis_com_nuclei")),
    Entry("deepbacs_fluorescence", FL, "cell", 2, _lm("deepbacs", _deepbacs_fluorescence), "deepbacs_fluorescence"),
    Entry(
        "deepbacs_label_free", LF, "cell", 2, _lm("deepbacs", _deepbacs_label_free),
        comment="Includes the E. coli stationary-phase images (train split only)."
    ),
    Entry("dememseg", FL, "prospore membrane", 2, _lm("dememseg")),
    Entry("dsb", FL, "nucleus", 2, _lm("dsb"), "dsb", "The fluorescence images of DSB 2018."),
    Entry("dynamicnuclearnet", FL, "nucleus", 2, _lm("dynamicnuclearnet"), "dynamicnuclearnet"),
    Entry("enseg", FL, "cell", 2, _lm("enseg"), comment="Enteric neurons."),
    Entry("epicure", FL, "cell", 2, _lm("epicure")),
    Entry("flywing", FL, "cell", 2, _lm("flywing"), "flywing"),
    Entry("ifnuclei", FL, "nucleus", 2, _lm("ifnuclei"), "ifnuclei"),
    Entry("livecell", LF, "cell", 2, _lm("livecell"), "livecell"),
    Entry("lpc_nucseg", FL, "nucleus", 2, _lm("lpc_nucseg")),
    Entry("mcellseg", LF, "cell", 2, _lm("mcellseg")),
    Entry("microbeseg", LF, "cell", 2, _lm("microbeseg")),
    Entry(
        "neurips_cellseg", LF, "cell", 2, _lm("neurips_cellseg"),
        comment="Brightfield, phase contrast, DIC and fluorescence images, counted as label-free, the majority: "
        "385 of the 1,101 images (35%) are fluorescence by a dark-background rule that classifies 98.7% of the "
        "tuning and test images correctly. Its test split is scored as neurips_cellseg_fluorescence and "
        "neurips_cellseg_label_free.",
    ),
    Entry("neurips_cellseg_fluorescence", FL, "cell", 2, test="neurips_cellseg_fluorescence", comment=NEURIPS_TRAIN),
    Entry("neurips_cellseg_label_free", LF, "cell", 2, test="neurips_cellseg_label_free", comment=NEURIPS_TRAIN),
    Entry("omnipose_bact_fluor", FL, "cell", 2, _lm("omnipose", _path_has("/bact_fluor/")), "omnipose_bact_fluor"),
    Entry("omnipose_bact_phase", LF, "cell", 2, _lm("omnipose", _path_has("/bact_phase/")), "omnipose_bact_phase"),
    Entry("omnipose_worm", LF, "worm", 2, _lm("omnipose", _path_has("/worm/"))),
    Entry("omnipose_worm_high_res", LF, "worm", 2, _lm("omnipose", _path_has("/worm_high_res/"))),
    Entry("organoid", LF, "organoid", 2, _lm("organoid")),
    Entry("organoidnet", LF, "organoid", 2, _lm("organoidnet")),
    Entry("orgasegment", LF, "organoid", 2, _lm("orgasegment")),
    Entry("orgline", LF, "organoid", 2, _lm("orgline"), "orgline"),
    Entry(
        "pan_multiplex", FL, "cell", 2, _lm("pan_multiplex"),
        comment="MIBI, CODEX and Vectra images. MIBI is mass spectrometry imaging, not fluorescence."
    ),
    Entry("tissuenet", FL, "cell", 2, _lm("tissuenet"), "tissuenet"),
    Entry("toiam", LF, "cell", 2, _lm("toiam")),
    Entry("tsakiroglou", FL, "nucleus", 2, _lm("tsakiroglou")),
    Entry("u20s", FL, "nucleus", 2, _lm("u20s"), "u20s"),
    Entry("usiigaci", LF, "cell", 2, _lm("usiigaci")),
    Entry("vicar", LF, "cell", 2, _lm("vicar"), "vicar"),
    Entry(
        "xenium_cells", FL, "cell", 2, _lm("xenium", _key_is("labels/cells")),
        comment="The labels are the automatic Xenium segmentation, not manual annotations."
    ),
    Entry(
        "xenium_nuclei", FL, "nucleus", 2, _lm("xenium", _key_is("labels/nuclei")),
        comment="The labels are the automatic Xenium segmentation, not manual annotations."
    ),
    Entry("yeaz", LF, "cell", 2, _lm("yeaz")),
    # Light microscopy, 2d, OOD.
    Entry("arvidsson", FL, "nucleus", 2, test="arvidsson"),
    Entry(
        "bac_mother", LF, "cell", 2, test="bac_mother",
        comment="E. coli in a mother machine. Every 10th frame of the test movies is scored."
    ),
    Entry("bbbc010", LF, "worm", 2, test="bbbc010", comment="C. elegans."),
    Entry("cardioblast_nuclei", FL, "nucleus", 2, test="cardioblast_nuclei"),
    Entry(
        "cellapp", LF, "cell", 2, test="cellapp",
        comment="The HeLa, RPE1 and U2OS subsets. The labels are automatic, so not every cell is annotated."
    ),
    Entry("covid_if_cells", FL, "cell", 2, test="covid_if_cells"),
    Entry("covid_if_nuclei", FL, "nucleus", 2, test="covid_if_nuclei"),
    Entry("deepseas", LF, "cell", 2, test="deepseas"),
    Entry(
        "ecoli_microcolony_lineage", LF, "cell", 2, test="ecoli_microcolony_lineage",
        comment="E. coli microcolonies (cib, crosstalk, recA, rpsM). Every 10th frame of the movies is scored."
    ),
    Entry("fusionx", FL, "cell", 2, test="fusionx"),
    Entry("hela_cytonuc", FL, "nucleus", 2, test="hela_cytonuc"),
    Entry("hpa", FL, "cell", 2, test="hpa"),
    Entry("medussa", FL, "cell", 2, test="medussa"),
    Entry("micro_bench", FL, "nucleus", 2, test="micro_bench", comment="The OpenCell nuclei of µBench."),
    Entry("mif_tonsil", FL, "nucleus", 2, test="mif_tonsil"),
    Entry("mif_tonsil_cells", FL, "cell", 2, test="mif_tonsil_cells"),
    Entry("wing_disc_timelapse", FL, "cell", 2, test="wing_disc_timelapse"),
    Entry("yeastsam", LF, "cell", 2, test="yeastsam"),
    # Light microscopy, 3d.
    Entry("cartocell", FL, "cell", 3, _lm("cartocell"), "cartocell"),
    Entry("celegans_atlas", FL, "nucleus", 3, _lm("celegans_atlas"), "celegans_atlas"),
    Entry(
        "cochleanet_ihc", FL, "cell", 3, _lm("cochleanet_data", _path_has("/ihc/")),
        comment=f"Inner hair cells. {INTERNAL}"
    ),
    Entry(
        "cochleanet_sgn", FL, "cell", 3, _lm("cochleanet_data", _path_has("/sgn/")),
        comment=f"Spiral ganglion neuron somata. {INTERNAL}"
    ),
    Entry(
        "embedseg_mouse_skull", FL, "nucleus", 3, _lm("embedseg", _path_has("/Mouse-Skull-Nuclei-CBG/")),
        "embedseg_mouse_skull"
    ),
    Entry(
        "embedseg_organoid", FL, "cell", 3, _lm("embedseg", _path_has("/Mouse-Organoid-Cells-CBG/")),
        "embedseg_organoid"
    ),
    Entry("embedseg_platy_ish", FL, "nucleus", 3, _lm("embedseg", _path_has("/Platynereis-ISH-Nuclei-CBG/"))),
    Entry(
        "embedseg_platy_nuclei", FL, "nucleus", 3, _lm("embedseg", _path_has("/Platynereis-Nuclei-CBG/")),
        "embedseg_platy_nuclei"
    ),
    Entry("gonuclear", FL, "nucleus", 3, _lm("gonuclear"), "gonuclear"),
    Entry("nis3d", FL, "nucleus", 3, _lm("nis3d"), "nis3d"),
    Entry("nucverse3d", FL, "nucleus", 3, _lm("nucverse3d"), "nucverse3d"),
    Entry("phmamm", FL, "cell", 3, _lm("phmamm"), "phmamm"),
    Entry("plantseg_root", FL, "cell", 3, _lm("plantseg"), "plantseg_root"),
    Entry("pnas_arabidopsis", FL, "cell", 3, _lm("pnas_arabidopsis"), "pnas_arabidopsis"),
    Entry("wing_disc", FL, "cell", 3, _lm("wing_disc"), "wing_disc"),
    # Light microscopy, 3d, OOD.
    Entry("bbbc032", FL, "nucleus", 3, test="bbbc032"),
    Entry("bbbc033", FL, "nucleus", 3, test="bbbc033"),
    Entry("bbbc050", FL, "nucleus", 3, test="bbbc050"),
    Entry("blastospim", FL, "nucleus", 3, test="blastospim"),
    Entry("cshaper", FL, "cell", 3, test="cshaper"),
    Entry("morphonet_celegans", FL, "nucleus", 3, test="morphonet_celegans"),
    Entry(
        "morphonet_patiria", FL, "cell", 3, test="morphonet_patiria",
        comment="The MorphoNet starfish (Patiria miniata) embryo with its curated labels."
    ),
    Entry(
        "nus_organoids", FL, "nucleus", 3, test="nus_organoids",
        comment="Three of the four manually annotated volumes of the nucleus benchmark of the Digitalized "
        "Organoid paper: cell monolayer, colon organoid and breast cancer spheroid."
    ),
    Entry("plantseg_ovules", FL, "cell", 3, test="plantseg_ovules"),
    # Electron microscopy (and EM-like), 2d.
    Entry(
        "astih_bf", LF, "axon", 2, _em("astih", _path_has("/BF1/", "/BF2/")),
        comment="ASTIH BF1 and BF2: brightfield light microscopy of nerve sections, myelinated axons."
    ),
    Entry("astih_sem", EM, "axon", 2, _em("astih", _path_has("/SEM1/")), comment="ASTIH SEM1: myelinated axons."),
    Entry("tumor_spheroid", EM, "cell", 2, _em("tumor_spheroid_em")),
    # Electron microscopy (and EM-like), 3d.
    Entry("axonem", EM, "neurite", 3, _em("axonem")),
    Entry("cremi", EM, "neurite", 3, _em("cremi"), "cremi"),
    Entry("densecell", EM, "cell", 3, _em("densecell")),
    Entry(
        "emneuron", EM, "neurite", 3, _em("emneuron"),
        comment="A compilation of vEM volumes (without its AxonEM, CREMI and FIB-25 folders). It also holds "
        "Kasthuri (ATUM) and J0126 volumes, the sources of the snemi and zebrafinch_j0126 test data. Its "
        "out-of-distribution validation volumes, the sources of ionsem, microns_basil and synapseweb, do not train.",
    ),
    Entry("fafb", EM, "neurite", 3, _em("fafb"), "fafb"),
    Entry("fib25", EM, "neurite", 3, _em("fib25")),
    Entry("hemibrain", EM, "neurite", 3, _em("hemibrain"), "hemibrain"),
    Entry("igor_cells", EM, "cell", 3, _em("igor_cells"), comment=f"Internal annotations. {INTERNAL}"),
    Entry("liconn", FL, "neurite", 3, _em("liconn"), "liconn", comment="Expansion light microscopy (EM-like), not EM."),
    Entry("malecns", EM, "neurite", 3, _em("malecns")),
    Entry("manc", EM, "neurite", 3, _em("manc")),
    Entry("microns_minnie65", EM, "neurite", 3, _em("microns-minnie65")),
    Entry("microns_pinky", EM, "neurite", 3, _em("microns")),
    Entry("platynereis_cells", EM, "cell", 3, _em("platynereis")),
    Entry("snemi", EM, "neurite", 3, _em("snemi"), "snemi"),
    Entry(
        "synapsenet_compartments", EM, "synaptic compartment", 3, _em("synapsenet_compartments_data"), comment=INTERNAL
    ),
    Entry("wafer4", EM, "neurite", 3, _em("wafer4")),
    Entry("wildenberg", EM, "neurite", 3, _em("wildenberg2023"), comment="The labels are an automatic segmentation."),
    Entry(
        "xpress", "X-ray", "axon", 3, _em("xpress"),
        comment="X-ray holographic nano-tomography (EM-like), not EM. Myelinated axons."
    ),
    Entry("zebrafinch_j0126", EM, "neurite", 3, _em("zebrafinch", _path_has("/j0126_")), "zebrafinch_j0126"),
    Entry("zebrafinch_j0251", EM, "neurite", 3, _em("zebrafinch", _path_has("/j0251_"))),
    # Electron microscopy, 3d, OOD.
    Entry(
        "humanneurons", EM, "neurite", 3, test="humanneurons",
        comment="A 64 x 2048 x 2048 crop of the H01 human cortex volume."
    ),
    Entry(
        "ionsem", EM, "neurite", 3, test="ionsem",
        comment="Ionsem, an out-of-distribution validation volume of EMNeuron."
    ),
    Entry("isbi2012", EM, "neurite", 3, test="isbi2012"),
    Entry(
        "microns_basil", EM, "neurite", 3, test="microns_basil",
        comment="MICrONS basil (mouse visual cortex, ssTEM), an out-of-distribution validation volume of "
        "EMNeuron."
    ),
    Entry("nisb", EM, "neurite", 3, test="nisb", comment="Synthetic data."),
    Entry("synapseweb", EM, "neurite", 3, test="synapseweb"),
    # Histopathology, 2d.
    Entry("cellbindb_he", HP, "nucleus", 2, _hp("cellbindb")),
    Entry("consep", HP, "nucleus", 2, _hp("consep")),
    Entry("cpm15", HP, "nucleus", 2, _hp("cpm15")),
    Entry("cpm17", HP, "nucleus", 2, _hp("cpm17"), "cpm17"),
    Entry("cryonuseg", HP, "nucleus", 2, _hp("cryonuseg")),
    Entry("dsb_he", HP, "nucleus", 2, _hp("dsb"), comment="The H&E images of DSB 2018."),
    Entry("glysac", HP, "nucleus", 2, _hp("glysac"), "glysac"),
    Entry("histo_miner", HP, "nucleus", 2, _hp("histo_miner"), "histo_miner"),
    Entry("lizard", HP, "nucleus", 2, _hp("lizard")),
    Entry("lizard_mitosis", HP, "nucleus", 2, _hp("lizard_mitosis")),
    Entry("lynsec_he", HP, "nucleus", 2, _hp("lynsec", _path_has("/h&e/"))),
    Entry("lynsec_ihc", HP, "nucleus", 2, _hp("lynsec", _path_has("/ihc/")), comment="IHC."),
    Entry("monuseg", HP, "nucleus", 2, _hp("monuseg"), "monuseg"),
    Entry("nuinsseg", HP, "nucleus", 2, _hp("nuinsseg")),
    Entry("pannuke", HP, "nucleus", 2, _hp("pannuke"), "pannuke"),
    Entry("puma", HP, "nucleus", 2, _hp("puma"), "puma"),
    Entry("spatch_he", HP, "nucleus", 2, _hp("spatch"), comment="The H&E subsets of sPATCH."),
    Entry("srsanet", HP, "nucleus", 2, _hp("srsanet"), comment="IHC."),
    Entry("tnbc_celltype", HP, "nucleus", 2, _hp("tnbc_celltype")),
    # Histopathology, 2d, OOD.
    Entry(
        "amonuseg", HP, "nucleus", 2, test="amonuseg",
        comment="H&E. Semantic masks: the instances are connected components, so touching nuclei merge."
    ),
    Entry("beacon", HP, "nucleus", 2, test="beacon", comment="H&E biopsies of Barrett's esophagus."),
    Entry("cd47_ihcnusc", HP, "nucleus", 2, test="cd47_ihcnusc", comment="CD47 IHC of esophageal cancer."),
    Entry("cytodark0", HP, "nucleus", 2, test="cytodark0"),
    Entry("khoshdeli", HP, "nucleus", 2, test="khoshdeli"),
    Entry("pcns", HP, "nucleus", 2, test="pcns"),
]


def read_figure_literal(path: str, name: str):
    """Read the literal value of a module-level assignment from a plotting script, without importing it."""
    with open(path) as f:
        tree = ast.parse(f.read())
    nodes = [
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(nodes) != 1:
        raise RuntimeError(f"Expected one {name} assignment in {path}, found {len(nodes)}.")
    return ast.literal_eval(nodes[0])


def read_panels(figure_scripts: Optional[Dict[int, str]] = None) -> Dict[str, Tuple[int, str]]:
    """Read the ID and OOD datasets of the figure panels from the plotting scripts.

    Args:
        figure_scripts: The plotting script of each dimensionality. ``FIGURE_SCRIPTS`` (Figures 2 and 3) by default.

    Returns:
        The dimensionality and the role ('ID' or 'OOD') of every evaluated dataset.
    """
    panels = {}
    for ndim, path in (FIGURE_SCRIPTS if figure_scripts is None else figure_scripts).items():
        for panel in read_figure_literal(path, "PANEL_DATASETS").values():
            for role, names in panel.items():
                for name in names:
                    if name in panels and panels[name] != (ndim, role):
                        raise RuntimeError(f"{name} has two roles in the figure panels.")
                    panels[name] = (ndim, role)
    return panels


def check_registry(entries: List[Entry], panels: Dict[str, Tuple[int, str]]) -> Dict[str, str]:
    """Check the registry against the figure panels and return the role of each entry."""
    names = [entry.name for entry in entries]
    duplicates = [name for name, n in Counter(names).items() if n > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate dataset names in the registry: {duplicates}")

    roles = {}
    for entry in entries:
        if entry.test is None:
            if entry.train is None:
                raise RuntimeError(f"{entry.name} has neither training nor test data.")
            roles[entry.name] = "train"
            continue
        if entry.test not in panels:
            raise RuntimeError(f"The test data {entry.test} of {entry.name} is not in the figure panels.")
        ndim, role = panels[entry.test]
        if ndim != entry.ndim:
            raise RuntimeError(f"{entry.name} is {entry.ndim}d in the registry, but in a {ndim}d figure.")
        roles[entry.name] = role

    missing = sorted(set(panels) - {entry.test for entry in entries})
    if missing:
        raise RuntimeError(f"The registry has no row for these evaluated datasets: {missing}")
    return roles


def assign_units(units, entries: List[Entry]) -> Dict[str, list]:
    """Assign every training label source of the built domains to exactly one registry entry.

    Raises:
        RuntimeError: If a label source matches no entry or several entries.
    """
    domains = {unit.domain for unit in units}
    candidates = [entry for entry in entries if entry.train is not None and entry.train.domain in domains]
    assigned = {entry.name: [] for entry in candidates}
    unassigned, ambiguous = [], []
    for unit in units:
        matches = [
            entry for entry in candidates if entry.train.domain == unit.domain and entry.train.folder == unit.dataset
            and (entry.train.select is None or entry.train.select(unit))
        ]
        if not matches:
            unassigned.append(f"{unit.domain}: {unit.label_path} ({unit.label_key})")
        elif len(matches) > 1:
            ambiguous.append(f"{unit.label_path}: {[entry.name for entry in matches]}")
        else:
            assigned[matches[0].name].append(unit)
    if unassigned or ambiguous:
        raise RuntimeError(
            "The registry does not partition the training data. Update its entries.\n"
            f"Unassigned label sources ({len(unassigned)}): {unassigned[:10]}\n"
            f"Label sources with several entries ({len(ambiguous)}): {ambiguous[:10]}"
        )
    empty = [name for name, entry_units in assigned.items() if not entry_units]
    if empty:
        raise RuntimeError(f"These entries select no training data: {empty}")
    return assigned


#
# Counting.
#

def count_test_shard(dataset_name: str, ndim: int, data_root: str, shard: int, n_shards: int) -> Tuple[int, int]:
    """Count the scored test samples and their objects for every n_shards-th sample of a dataset.

    Args:
        dataset_name: The dataset name of the evaluation.
        ndim: The dimensionality of the data.
        data_root: The root folder of the data.
        shard: The index of the first sample of this shard.
        n_shards: The number of shards.

    Returns:
        The number of samples with labels and the number of their objects.
    """
    common = _common()
    total = common.n_samples(dataset_name, data_root)
    sample_ids = set(range(shard, total, n_shards))
    n_samples, n_objects = 0, 0
    if not sample_ids:
        return n_samples, n_objects
    for _, labels, _ in common.load_data(dataset_name, data_root, ndim, normalize=False, sample_ids=sample_ids):
        if labels.max() == 0:  # The evaluation does not score samples without labels.
            continue
        n_samples += 1
        n_objects += int(np.count_nonzero(np.unique(labels)))
    return n_samples, n_objects


def build_training_units(domains, data_root):
    """Build the training datasets of the domains like the loader and collect their label sources."""
    from micro_sam.v2.datasets.data_counter import build_units

    units, transforms = [], {}
    for domain in domains:
        print(f"Build the training datasets of the domain '{domain}'. This takes several minutes.", flush=True)
        domain_units, domain_transforms = build_units(domain, data_root)
        units.extend(domain_units)
        transforms.update(domain_transforms)
    return units, transforms


def submit_train(pool, units, transforms):
    """Submit the counting of the training label sources of one dataset.

    Returns:
        The counting jobs and their futures. A file split into several regions of interest is one volume, see
        ``micro_sam.v2.datasets.data_counter.counting_jobs``.
    """
    from micro_sam.v2.datasets.data_counter import count_job, counting_jobs

    jobs = counting_jobs(units, transforms)
    return [(job, pool.submit(count_job, job)) for job in jobs]


def finish_train(name, units, futures):
    """Sum the counts of the training label sources of one dataset."""
    from micro_sam.v2.datasets.data_counter import summarize

    counts = {
        unit.key: count for job, future in futures for unit, count in zip(job.units, future.result())
    }
    _, rows = summarize([replace(unit, dataset=name) for unit in units], counts)
    return sum(row["n_images"] for row in rows), sum(row["n_objects"] for row in rows)


def submit_test(pool, dataset_names, ndim, data_root, n_workers):
    """Submit the counting of the scored test samples of one or several pooled datasets, split into shards."""
    futures = []
    for dataset_name in dataset_names:
        n_samples = _common().n_samples(dataset_name, data_root)
        n_shards = max(1, min(n_workers, n_samples))
        futures.extend(
            pool.submit(count_test_shard, dataset_name, ndim, data_root, i, n_shards) for i in range(n_shards)
        )
    return futures


def finish_test(futures):
    """Sum the counts of the test shards of one dataset."""
    results = [future.result() for future in futures]
    return sum(r[0] for r in results), sum(r[1] for r in results)


#
# The table.
#

def read_table(path: str) -> Dict[str, dict]:
    """Read the existing table. A missing table is empty."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row["dataset"]: row for row in csv.DictReader(f)}


def write_table(path: str, rows: Dict[str, dict]) -> None:
    """Write the table sorted by dimensionality, modality, structure and role, via a temporary file."""
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            row["dimensionality"], row["modality"], row["structure"], ROLE_ORDER[row["role"]], row["dataset"]
        ),
    )
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(tmp_path, path)


def _futures(submitted) -> list:
    """The futures of the submitted counts: the train counts are (job, future) pairs, the test counts futures."""
    return [item[1] if isinstance(item, tuple) else item for item in submitted]


def _is_missing(row: dict, columns: List[str]) -> bool:
    return any(row.get(column, "") in ("", None) for column in columns)


def run(
    output: str, data_root: str, datasets: Optional[List[str]], force: bool, n_workers: int,
    registry: Optional[List[Entry]] = None, roles: Optional[Dict[str, str]] = None,
    pooled: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Fill in the missing counts of the table, or recompute them with 'force'.

    Args:
        output: The path of the CSV table, which is read first and also serves as the cache.
        data_root: The root folder of the data.
        datasets: Only process these datasets. All by default.
        force: Recompute the counts of the processed datasets even if the table has them.
        n_workers: The number of processes that count in parallel.
        registry: The rows of the table. ``ENTRIES`` by default.
        roles: The role of each entry. By default derived from the Figure 2 and 3 panels with ``check_registry``.
        pooled: The evaluation datasets that a figure pools into one test dataset, by the name of the pooled
            dataset. The test counts of a pooled dataset are the sums over its parts.
    """
    registry = ENTRIES if registry is None else registry
    if roles is None:
        roles = check_registry(registry, read_panels())
    entries = {entry.name: entry for entry in registry}
    if datasets is not None:
        unknown = sorted(set(datasets) - set(entries))
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}. Choose from {sorted(entries)}.")

    # The static columns always come from the registry, the counts from the existing table.
    old_rows = read_table(output)
    dropped = sorted(set(old_rows) - set(entries))
    if dropped:
        warnings.warn(f"Dropping the rows that are no longer in the registry: {dropped}")
    rows = {}
    for name, entry in entries.items():
        old = old_rows.get(name, {})
        # Counts of a split the entry no longer has, e.g. after a dataset left the evaluation, are cleared.
        kept = (TRAIN_COLUMNS if entry.train is not None else []) + (TEST_COLUMNS if entry.test is not None else [])
        rows[name] = {
            "dataset": name, "modality": entry.modality, "structure": entry.structure,
            "dimensionality": f"{entry.ndim}d", "role": roles[name], "comment": entry.comment,
            **{column: old.get(column, "") if column in kept else "" for column in TRAIN_COLUMNS + TEST_COLUMNS},
        }

    selected = list(entries) if datasets is None else datasets
    need_train = [
        name for name in selected if entries[name].train is not None
        and (force or _is_missing(rows[name], TRAIN_COLUMNS))
    ]
    need_test = [
        name for name in selected if entries[name].test is not None and (force or _is_missing(rows[name], TEST_COLUMNS))
    ]
    todo = [name for name in selected if name in need_train or name in need_test]
    if not todo:
        write_table(output, rows)
        print(f"All counts are in {output}. Pass --force to recompute them.")
        return

    units_per_entry, transforms = {}, {}
    if need_train:
        domains = sorted({entries[name].train.domain for name in need_train})
        units, transforms = build_training_units(domains, data_root)
        units_per_entry = assign_units(units, registry)

    failed = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pending = {}
        for name in todo:
            jobs = {}
            try:
                if name in need_train:
                    jobs["train"] = submit_train(pool, units_per_entry[name], transforms)
                if name in need_test:
                    test = entries[name].test
                    parts = (pooled or {}).get(test, [test])
                    jobs["test"] = submit_test(pool, parts, entries[name].ndim, data_root, n_workers)
            except Exception as e:
                warnings.warn(f"Could not start counting {name}: {e!r}")
                failed.append(name)
            if jobs:
                pending[name] = jobs

        with tqdm(total=len(todo), desc="Count the datasets") as pbar:
            pbar.update(len(todo) - len(pending))
            while pending:
                futures = [f for jobs in pending.values() for fs in jobs.values() for f in _futures(fs)]
                wait(futures, return_when=FIRST_COMPLETED)
                for name in [
                    n for n, jobs in pending.items() if all(f.done() for fs in jobs.values() for f in _futures(fs))
                ]:
                    jobs = pending.pop(name)
                    try:
                        if "train" in jobs:
                            values = finish_train(name, units_per_entry[name], jobs["train"])
                            rows[name].update(zip(TRAIN_COLUMNS, values))
                        if "test" in jobs:
                            rows[name].update(zip(TEST_COLUMNS, finish_test(jobs["test"])))
                    except Exception as e:
                        tqdm.write(f"Counting {name} failed: {e!r}")
                        failed.append(name)
                    write_table(output, rows)
                    pbar.set_postfix_str(name)
                    pbar.update(1)

    write_table(output, rows)
    print(f"Wrote {output}.")
    if failed:
        print(f"The counts of these datasets failed and stay empty: {sorted(set(failed))}")


def main():
    parser = argparse.ArgumentParser(
        description="Create the overview table of the micro-sam2 generalist training and evaluation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="The CSV table, also used as the cache.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="The root folder of the data.")
    parser.add_argument("-d", "--datasets", nargs="+", help="Only process these datasets. All by default.")
    parser.add_argument("-f", "--force", action="store_true", help="Recompute the counts that the table has.")
    parser.add_argument(
        "-n", "--n_workers", type=int, default=min(8, len(os.sched_getaffinity(0))),
        help="The number of processes that count in parallel. Use at most 8 on a 64 GB allocation.",
    )
    args = parser.parse_args()
    run(args.output, args.data_root, args.datasets, args.force, args.n_workers)


if __name__ == "__main__":
    main()
