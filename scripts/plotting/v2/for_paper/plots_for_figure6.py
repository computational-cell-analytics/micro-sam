"""Plots for Figure 6 of the micro-sam2 paper.

The figure compares 3d automatic and interactive segmentation of EM mitochondria and nuclei with micro-sam2
(the 'hvit_s_em_mito' and 'hvit_s_em_nuclei' models) and the baselines. It uses the layout and the style of Figure 3
(see plots_for_figure3.py). The script writes these plots to 'figures/figure6/<automatic|interactive>/quantitative':
- 'automatic_3d': the score of each method per organelle, for the in-domain and the out-of-domain datasets.
- 'interactive_3d': the score over the correction iterations, with a point or a box as the first prompt.

Usage:

    python plots_for_figure6.py
    python plots_for_figure6.py --only automatic_3d

The inputs are the EM organelle tables in 'figures', see 'collect_em_organelle_results.py' and
'collect_em_interactive_results.py' in the control folder of the baselines_2d experiment.
"""

import os
import argparse

import pandas as pd

import plots_for_figure2 as fig2
from plot_style import set_style


ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(ROOT, "figures")
OUTPUT_DIR = os.path.join(FIGURES, "figure6")

# The (results table, interactive table, structure, title) of each organelle. The automatic tables have no NucMM
# (user, 2026-10-30), so the interactive plot leaves it out too: both plots show the same datasets.
PANELS = {
    "em_mito": (
        "em_mito_automatic_segmentation_3d_results.csv", "em_mito_interactive_segmentation_3d_results.csv",
        "mitochondrion", "Electron Microscopy (Mitochondrion)",
    ),
    "em_nuclei": (
        "em_nuc_automatic_segmentation_3d_results.csv", "em_nuc_interactive_segmentation_3d_results.csv",
        "nucleus", "Electron Microscopy (Nucleus)",
    ),
}
SCORE_LABEL = "Mean Segmentation Accuracy ↑"

# The 'method', 'model' and 'mode' values of each automatic method in the 3d results. micro-sam shows AIS, its
# default mode (see fig2.FIGURE_MODES), and micro-sam2 AIS and APG.
METHODS = {
    "MitoNet": ("mitonet", ("MitoNet_v1",), None),
    "NucleoNet": ("nucleonet", ("NucleoNet_base_v2",), None),
    "micro-sam": ("micro-sam", ("vit_l_em_organelles",), "apg"),
    "micro-sam2": ("micro-sam2", ("hvit_s",), "apg"),
}
PANEL_METHODS = {
    "em_mito": ["MitoNet", "micro-sam", "micro-sam2"],
    "em_nuclei": ["NucleoNet", "micro-sam", "micro-sam2"],
}
# micro-sam2 runs a different model per organelle ('hvit_s_em_mito', 'hvit_s_em_nuclei'), so it gets a star in the
# automatic plot.
AUTOMATIC_STARRED_METHODS = {"micro-sam2"}
STARRED_METHODS = set()

# The largest model of each method, as in Figure 3. SAM and micro-sam correct with clicks only, as the others do.
INTERACTIVE_METHODS = {
    "SAM": ("sam", ("vit_l",), "points"),
    "SAM2": ("sam2", ("hvit_l",), ""),
    "SAM3": ("sam3", ("default",), ""),
    "micro-sam": ("micro-sam", ("vit_l_em_organelles",), "points"),
    "micro-sam2": ("micro-sam2", ("hvit_s",), ""),
}

AUTOMATIC_ROWS = [["em_mito", "em_nuclei"]]


def panel_datasets(name):
    """Return the in-domain and out-of-domain datasets of an organelle, in the order of its automatic table."""
    table = pd.read_csv(os.path.join(FIGURES, PANELS[name][0])).drop_duplicates("dataset")
    return {domain: list(table[table["domain"] == domain]["dataset"]) for domain in fig2.DOMAIN_TITLES}


def plot_automatic():
    """Plot the automatic results of both organelles, with micro-sam2 in AIS and APG."""
    results = {}
    for name, (table, _, structure, _) in PANELS.items():
        rows = fig2.load_modes("EM", structure, path=os.path.join(FIGURES, table), methods=METHODS)
        results[name] = fig2.one_mode(fig2.select_datasets(rows, panel_datasets(name)), fig2.FIGURE_MODES)
    panels = {name: (title, SCORE_LABEL, PANEL_METHODS[name]) for name, (_, _, _, title) in PANELS.items()}
    return fig2.plot_automatic(
        results, AUTOMATIC_ROWS, panels, AUTOMATIC_STARRED_METHODS, domain_offset=0.27, ranks=True,
    )


def plot_interactive():
    """Plot the interactive results of both organelles over the correction iterations.

    The legend stands in the point half of the out-of-domain mitochondria panel, whose curves stay low. The value
    axes end at 1, the highest mSA: the shared row would otherwise be raised over the nucleus curves for the legend.
    """
    results = {
        name: fig2.load_interactive(
            "EM", structure, panel_datasets(name), path=os.path.join(FIGURES, table), methods=INTERACTIVE_METHODS,
        ).dropna(subset=["score"])
        for name, (_, table, structure, _) in PANELS.items()
    }
    titles = {name: title for name, (_, _, _, title) in PANELS.items()}
    fig = fig2.plot_interactive_by_domain(
        results, list(PANELS), titles, STARRED_METHODS, methods=INTERACTIVE_METHODS,
        legend_panel=("OOD", "em_mito"), legend_left=True, shared_rows=True,
    )
    for ax in fig.axes:
        low, high = ax.get_ylim()
        if high > 1:
            ticks = ax.get_yticks()
            ax.set_ylim(low, 1)
            ax.set_yticks(ticks[ticks <= 1 + 1e-6])
    return fig


PLOTS = {
    "automatic_3d": ("automatic", plot_automatic),
    "interactive_3d": ("interactive", plot_interactive),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(PLOTS), default=list(PLOTS), help="Make only these plots.")
    args = parser.parse_args()
    set_style()
    fig2.check_style()
    fig2.OUTPUT_DIR = OUTPUT_DIR
    for name in args.only:
        kind, plot = PLOTS[name]
        fig2.save_figure(plot(), name, kind)


if __name__ == "__main__":
    main()
