import os

import numpy as np
import matplotlib.pyplot as plt

from util import (
    msa_results,
    msa_results_fluorescence,
    msa_results_label_free,
    msa_results_histopathology,
)

AVG_METHODS = [
    "AMG (vit_b) - without grid search",
    "AIS - without grid search",
    "SAM3",
    "CellPose3",
    "CellPoseSAM",
    "CellSAM",
    "APG - without grid search (cc)",
]

APG_METHOD = "APG - without grid search (cc)"

AVG_DISPLAY_NAME_MAP = {
    "AMG (vit_b) - without grid search": "SAM",
    "AIS - without grid search": "AIS (µSAM)",
    "SAM3": "SAM3",
    "CellPose3": "CellPose 3",
    "CellPoseSAM": "CellPoseSAM",
    "CellSAM": "CellSAM",
    "APG - without grid search (cc)": "APG (µSAM)",
}

AVG_DISPLAY_NAME_MAP_HISTO = AVG_DISPLAY_NAME_MAP.copy()
AVG_DISPLAY_NAME_MAP_HISTO["AIS - without grid search"] = "AIS (PathoSAM)"

plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def compute_method_means(msa_results_dict):
    values_per_method = {}

    for dataset, entries in msa_results_dict.items():
        for e in entries:
            m = e["method"]
            v = e["mSA"]
            if v is None:
                continue
            values_per_method.setdefault(m, []).append(v)

    mean_msa = {}
    for m, vals in values_per_method.items():
        mean_msa[m] = float(np.mean(vals))

    return mean_msa


def compute_method_avg_ranks(msa_results_dict, methods_filter=None):
    ranks_per_method = {}

    for dataset, entries in msa_results_dict.items():
        methods = []
        scores = []

        for e in entries:
            m = e["method"]
            v = e["mSA"]

            if methods_filter is not None and m not in methods_filter:
                continue
            if v is None:
                continue

            methods.append(m)
            scores.append(v)

        if len(methods) == 0:
            continue

        scores = np.array(scores, dtype=float)
        idx_sorted = np.argsort(-scores)

        for rank_pos, idx_m in enumerate(idx_sorted):
            m = methods[idx_m]
            rank = rank_pos + 1
            ranks_per_method.setdefault(m, []).append(rank)

    avg_rank = {}
    for m, ranks in ranks_per_method.items():
        avg_rank[m] = float(np.mean(ranks))

    sorted_methods = sorted(avg_rank.keys(), key=lambda mm: avg_rank[mm])
    final_rank = {}
    for idx, m in enumerate(sorted_methods):
        final_rank[m] = idx + 1

    return avg_rank, final_rank


def plot_overall_averages(
    msa_nuclei,
    msa_fluo_cells,
    msa_label_free,
    msa_histo,
    save_path="msa_overall_averages.png",
):
    mean_nuclei = compute_method_means(msa_nuclei)
    mean_fluo = compute_method_means(msa_fluo_cells)
    mean_label_free = compute_method_means(msa_label_free)
    mean_histo = compute_method_means(msa_histo)

    avg_rank_nuclei, _ = compute_method_avg_ranks(msa_nuclei, AVG_METHODS)
    avg_rank_fluo, _ = compute_method_avg_ranks(msa_fluo_cells, AVG_METHODS)
    avg_rank_label_free, _ = compute_method_avg_ranks(msa_label_free, AVG_METHODS)
    avg_rank_histo, _ = compute_method_avg_ranks(msa_histo, AVG_METHODS)

    def filtered_methods_for_modality(mean_dict):
        return [m for m in AVG_METHODS if m in mean_dict]

    methods_nuclei = filtered_methods_for_modality(mean_nuclei)
    methods_fluo = filtered_methods_for_modality(mean_fluo)
    methods_label_free = filtered_methods_for_modality(mean_label_free)
    methods_histo = filtered_methods_for_modality(mean_histo)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 6),
        sharey=True,
    )
    axes = axes.flatten()

    color_top1 = "#1f77b4"   # darkest
    color_top2 = "#6baed6"
    color_top3 = "#c6dbef"
    color_rest = "#d9d9d9"

    modality_data = [
        ("Label-Free Microscopy (Cell Segmentation)",
         mean_label_free,
         methods_label_free,
         AVG_DISPLAY_NAME_MAP,
         avg_rank_label_free),
        ("Fluorescence Microscopy (Cell Segmentation)",
         mean_fluo,
         methods_fluo,
         AVG_DISPLAY_NAME_MAP,
         avg_rank_fluo),
        ("Fluorescence Microscopy (Nucleus Segmentation)",
         mean_nuclei,
         methods_nuclei,
         AVG_DISPLAY_NAME_MAP,
         avg_rank_nuclei),
        ("Histopathology (Nucleus Segmentation)",
         mean_histo,
         methods_histo,
         AVG_DISPLAY_NAME_MAP_HISTO,
         avg_rank_histo),
    ]

    for ax, (title, mean_dict, methods, disp_map, avg_rank_dict) in zip(axes, modality_data):
        if not methods:
            ax.set_visible(False)
            continue

        vals = np.array([mean_dict[m] for m in methods], dtype=float)
        x = np.arange(len(methods))

        colors = [color_rest] * len(methods)
        valid_mask = ~np.isnan(vals)
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) > 0:
            sorted_valid = valid_idx[np.argsort(vals[valid_idx])[::-1]]
            top_indices = sorted_valid[:3]
            if len(top_indices) > 0:
                colors[top_indices[0]] = color_top1
            if len(top_indices) > 1:
                colors[top_indices[1]] = color_top2
            if len(top_indices) > 2:
                colors[top_indices[2]] = color_top3
        else:
            top_indices = []

        ax.bar(x, vals, color=colors)
        apg_indices = [i for i, m in enumerate(methods) if m == APG_METHOD]
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue

            y_text = min(v + 0.01, 0.98)
            method_name = methods[i]
            avg_rank_val = avg_rank_dict.get(method_name, None)

            if avg_rank_val is not None:
                label = f"{v:.3f}\n({avg_rank_val:.2f})"
            else:
                label = f"{v:.3f}"

            fontweight = "bold" if i in apg_indices else "normal"

            ax.text(
                x[i],
                y_text,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight=fontweight,
            )

        disp_names = [disp_map[m] for m in methods]
        ax.set_xticks(x)
        ax.set_xticklabels(disp_names, rotation=45, ha="right")

        xticklabels = ax.get_xticklabels()
        for idx_lbl, lbl in enumerate(xticklabels):
            if idx_lbl in apg_indices:
                lbl.set_fontweight("bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.0, 1.0)

    fig.text(
        0.05, 0.65,
        "Mean Segmentation Accuracy (mSA)",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0.06, 0.02, 1, 0.97])
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        root, _ = os.path.splitext(save_path)
        svg_path = root + ".svg"
        fig.savefig(svg_path, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":
    plot_overall_averages(
        msa_results,
        msa_results_fluorescence,
        msa_results_label_free,
        msa_results_histopathology,
        save_path="msa_overall_averages.png",
    )
