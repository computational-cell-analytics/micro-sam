"""The shared style of all micro-sam2 paper plots: the matplotlib settings and the fixed color of each method.

Every figure imports its colors from here, so that a method has the same color in every plot of the paper.
"""

import os
from glob import glob

import matplotlib.pyplot as plt
from matplotlib import font_manager


# The colors of the methods, after the palette of the micro-sam (v1) paper plots. micro-sam2 gets the dark teal of
# the v1 method and the white hatch that marks it, micro-sam (and PathoSAM, its histopathology model) the lighter
# teal and CellposeSAM the pink of CellPose. The SAM family goes from sand over orange to brick red, the other cell
# models are lilac and plum. The histopathology and nucleus models are slate blue (StarDist), brown and dark brown
# (CellViT, CellViT++) and olive green (InstanSeg). Each color has its own hue name, so a caption can name it, and
# every pair is at least 12 OKLab units apart for normal vision.
METHOD_COLORS = {
    "StarDist": "#5966A6",
    "SAM": "#FCDE9C",
    "CellViT": "#8F6E56",
    "micro-sam": "#089099",
    "CellSAM": "#C3A4DC",
    "Cellpose3": "#6E2D7E",
    "SAM2": "#F39B4F",
    "InstanSeg": "#6A8A28",
    "CellViT++": "#3C2920",
    "CellposeSAM": "#E0479B",
    "SAM3": "#A8321F",
    "micro-sam2": "#045275",
    # The 3d-only methods: FOCUS-3D mint, SegNeuron (EM) cornflower blue and nnInteractive light blue. SegNeuron is
    # not grey, which marks the rows of methods that do not apply to a modality.
    "FOCUS-3D": "#7CCBA2",
    "SegNeuron": "#6495ED",
    "nnInteractive": "#9ECAE1",
    # The EM organelle models of empanada: MitoNet violet and NucleoNet green, the hues furthest from the others.
    "MitoNet": "#9E39E6",
    "NucleoNet": "#30BF30",
    # The feature models of the classification plots: DINOv3 indigo and UNIv2 vermilion, at least 15 OKLab units
    # from all colors above. The end-to-end baselines are dark grey with a dashed line.
    "DINOv3": "#1E1E9E",
    "UNIv2": "#FF4500",
    "End-to-end": "#52514E",
    # ilastik (classical features, classification plots only) royal blue: at least 18 OKLab units from every method
    # of those plots, also for protan and deutan vision, and at least 12 from every color above.
    "ilastik": "#1861F2",
}
TEXT_COLOR = "#212427"


# Nature Methods asks for Arial or Helvetica, and for all text between 5 and 7 pt. Liberation Sans has the letter
# widths of Arial and an open license; it is read from FONT_DIR when it is there, else DejaVu Sans is used.
FONT_DIR = os.path.expanduser("~/.local/share/fonts/liberation")
MAX_FONT_SIZE = 7


def set_style():
    """Set the matplotlib style that all plots of the paper share."""
    for path in glob(os.path.join(FONT_DIR, "*.ttf")):
        font_manager.fontManager.addfont(path)
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [name for name in ("Arial", "Helvetica", "Liberation Sans") if name in available] + [
            "DejaVu Sans"
        ],
        "font.size": MAX_FONT_SIZE,
        "axes.titlesize": MAX_FONT_SIZE,
        "axes.labelsize": MAX_FONT_SIZE,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": MAX_FONT_SIZE,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "xtick.labelcolor": TEXT_COLOR,
        "ytick.labelcolor": TEXT_COLOR,
        "hatch.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
