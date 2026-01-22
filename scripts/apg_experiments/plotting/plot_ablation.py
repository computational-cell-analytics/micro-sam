import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_util import (
    msa_results,
    msa_results_fluorescence,
    msa_results_label_free,
    msa_results_histopathology,
)

AIS_METHOD = "AIS - without grid search"
APG_WO_BD = "APG - without grid search (bd)"
APG_WO_CC = "APG - without grid search (cc)"
APG_GS_BD = "APG - with grid search (bd)"
APG_GS_CC = "APG - with grid search (cc)"

plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
})


def compute_apg_vs_ais_relative(msa_results_dict):
    rel_dict = {}

    for dataset, entries in msa_results_dict.items():
        m2v = {e["method"]: e["mSA"] for e in entries}

        v_ais = m2v.get(AIS_METHOD, None)
        v_wo_bd = m2v.get(APG_WO_BD, None)
        v_wo_cc = m2v.get(APG_WO_CC, None)
        v_gs_bd = m2v.get(APG_GS_BD, None)
        v_gs_cc = m2v.get(APG_GS_CC, None)

        if (
            v_ais is None or v_ais == 0.0 or
            v_wo_bd is None or
            v_wo_cc is None or
            v_gs_bd is None or
            v_gs_cc is None
        ):
            continue

        def rel(apg_val, ais_val):
            return float(100.0 * (apg_val - ais_val) / ais_val)

        rel_wo_bd = rel(v_wo_bd, v_ais)
        rel_wo_cc = rel(v_wo_cc, v_ais)
        rel_gs_bd = rel(v_gs_bd, v_ais)
        rel_gs_cc = rel(v_gs_cc, v_ais)

        rel_dict[dataset] = {
            "rel_wo_bd": rel_wo_bd,
            "rel_wo_cc": rel_wo_cc,
            "rel_gs_bd": rel_gs_bd,
            "rel_gs_cc": rel_gs_cc,
        }

    return rel_dict


def plot_apg_vs_ais_relative(
    msa_nuclei,
    msa_fluo_cells,
    msa_label_free,
    msa_histo,
    save_path="apg_vs_ais_relative_top3.png",
):
    rel_nuclei = compute_apg_vs_ais_relative(msa_nuclei)
    rel_fluo = compute_apg_vs_ais_relative(msa_fluo_cells)
    rel_label_free = compute_apg_vs_ais_relative(msa_label_free)
    rel_histo = compute_apg_vs_ais_relative(msa_histo)

    modality_rel = [
        ("Label-Free Microscopy (Cell Segmentation)", rel_label_free),
        ("Fluorescence Microscopy (Cell Segmentation)", rel_fluo),
        ("Fluorescence Microscopy (Nucleus Segmentation)", rel_nuclei),
        ("Histopathology (Nucleus Segmentation)", rel_histo),
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11, 6),
        sharey=False,
    )
    axes = axes.flatten()

    color_wo_bd = "#7CCBA2"  # APG (bd), w/o grid
    color_wo_cc = "#FCDE9C"  # APG (cc), w/o grid
    color_gs_bd = "#045275"  # APG (bd), with grid
    color_gs_cc = "#90477F"  # APG (cc), with grid

    for ax_idx, (ax, (title, rel_dict)) in enumerate(zip(axes, modality_rel)):
        if not rel_dict:
            ax.set_visible(False)
            continue

        all_dsets = list(rel_dict.keys())

        def max_improvement(d):
            vals = rel_dict[d]
            return max(
                vals["rel_wo_bd"],
                vals["rel_wo_cc"],
                vals["rel_gs_bd"],
                vals["rel_gs_cc"],
            )

        all_dsets.sort(key=max_improvement, reverse=True)

        datasets = all_dsets[:3]
        if not datasets:
            ax.set_visible(False)
            continue

        rel_wo_bd = [rel_dict[d]["rel_wo_bd"] for d in datasets]
        rel_wo_cc = [rel_dict[d]["rel_wo_cc"] for d in datasets]
        rel_gs_bd = [rel_dict[d]["rel_gs_bd"] for d in datasets]
        rel_gs_cc = [rel_dict[d]["rel_gs_cc"] for d in datasets]

        x = np.arange(len(datasets))
        width = 0.18

        vals_this_subplot = (
            rel_wo_bd + rel_wo_cc + rel_gs_bd + rel_gs_cc
        )
        min_val = min(vals_this_subplot)
        max_val = max(vals_this_subplot)
        max_abs = max(abs(min_val), abs(max_val))

        y_lim = max(5.0, max_abs * 1.2)
        ax.set_ylim(-y_lim, y_lim)

        if min_val < 0:
            ax.axhspan(min_val, 0, color="#fee0d2", alpha=0.8, zorder=0)
        if max_val > 0:
            ax.axhspan(0, max_val, color="#e0f3db", alpha=0.8, zorder=0)

        bars_wo_bd = ax.bar(x - 1.5 * width, rel_wo_bd, width, color=color_wo_bd, zorder=1)
        bars_wo_cc = ax.bar(x - 0.5 * width, rel_wo_cc, width, color=color_wo_cc, zorder=1)
        bars_gs_bd = ax.bar(x + 0.5 * width, rel_gs_bd, width, color=color_gs_bd, zorder=1)
        bars_gs_cc = ax.bar(x + 1.5 * width, rel_gs_cc, width, color=color_gs_cc, zorder=1)

        best_mask_wo_bd = []
        best_mask_wo_cc = []
        best_mask_gs_bd = []
        best_mask_gs_cc = []

        for i in range(len(datasets)):
            vals_i = [rel_wo_bd[i], rel_wo_cc[i], rel_gs_bd[i], rel_gs_cc[i]]
            max_i = max(vals_i)

            best_mask_wo_bd.append(rel_wo_bd[i] == max_i)
            best_mask_wo_cc.append(rel_wo_cc[i] == max_i)
            best_mask_gs_bd.append(rel_gs_bd[i] == max_i)
            best_mask_gs_cc.append(rel_gs_cc[i] == max_i)

        ax.axhline(0.0, color="black", linewidth=0.8, zorder=2)

        def annotate_bars(bars, vals, best_mask):
            for b, v, is_best in zip(bars, vals, best_mask):
                if v is None:
                    continue
                h = b.get_height()
                if h == 0:
                    continue

                offset = 0.8
                if h > 0:
                    y_text = -offset
                    va = "top"
                else:
                    y_text = offset
                    va = "bottom"

                fontweight = "bold" if is_best else "normal"

                ax.text(
                    b.get_x() + b.get_width() / 2,
                    y_text,
                    f"{v:+.1f}%",
                    ha="center",
                    va=va,
                    fontsize=9,
                    rotation=90,
                    fontweight=fontweight,
                )

        annotate_bars(bars_wo_bd, rel_wo_bd, best_mask_wo_bd)
        annotate_bars(bars_wo_cc, rel_wo_cc, best_mask_wo_cc)
        annotate_bars(bars_gs_bd, rel_gs_bd, best_mask_gs_bd)
        annotate_bars(bars_gs_cc, rel_gs_cc, best_mask_gs_cc)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=0, ha="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, fontweight="bold")

    fig.tight_layout(rect=[0.06, 0.18, 1, 0.97])
    fig.text(
        0.06, 0.575,
        "Relative Mean Segmentation Accuracy (compared to AIS)",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=10,
        fontweight="bold",
    )

    legend_patches = [
        Patch(facecolor=color_wo_bd, label="APG (Boundary Distance) - Default"),
        Patch(facecolor=color_wo_cc, label="APG (Components) - Default"),
        Patch(facecolor=color_gs_bd, label="APG (Boundary Distance) - Grid Search"),
        Patch(facecolor=color_gs_cc, label="APG (Components) - Grid Search"),
    ]

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=2,
        fontsize=8,
        frameon=True,
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        root, _ = os.path.splitext(save_path)
        svg_path = root + ".svg"
        fig.savefig(svg_path, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":
    plot_apg_vs_ais_relative(
        msa_results,
        msa_results_fluorescence,
        msa_results_label_free,
        msa_results_histopathology,
        save_path="apg_vs_ais_relative_top3.png",
    )
