import os

import numpy as np
import matplotlib.pyplot as plt

from util import (
    msa_results,
    msa_results_fluorescence,
    msa_results_label_free,
    msa_results_histopathology,
    DISPLAY_NAME_MAP, METHOD_ORDER, IN_DOMAIN
)

DISPLAY_NAME_MAP_HISTO = DISPLAY_NAME_MAP.copy()
if "AMG - without grid search" in DISPLAY_NAME_MAP_HISTO:
    DISPLAY_NAME_MAP_HISTO["AMG - without grid search"] = "AMG (PathoSAM)"
if "AIS - without grid search" in DISPLAY_NAME_MAP_HISTO:
    DISPLAY_NAME_MAP_HISTO["AIS - without grid search"] = "AIS (PathoSAM)"

plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def plot_msa_grid(
    msa_results,
    ncols=3,
    figsize_per_subplot=(4, 3),
    save_path=None,
    suptitle="Preliminary mSA results",
    display_name_map=DISPLAY_NAME_MAP,
):
    datasets = list(msa_results.keys())
    n_exp = len(datasets)
    nrows = int(np.ceil(n_exp / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_subplot[0] * ncols,
                 figsize_per_subplot[1] * nrows),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    color_top1 = "#1f77b4"   # darkest
    color_top2 = "#6baed6"
    color_top3 = "#c6dbef"
    color_rest = "#d9d9d9"

    for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        all_entries = msa_results[dataset]
        entries = [e for e in all_entries if e["method"] in display_name_map]
        entries.sort(
            key=lambda e: METHOD_ORDER.index(e["method"])
            if e["method"] in METHOD_ORDER else len(METHOD_ORDER)
        )

        if not entries:
            ax.set_visible(False)
            continue

        raw_methods = [e["method"] for e in entries]
        methods = [display_name_map[m] for m in raw_methods]
        values = [np.nan if e["mSA"] is None else e["mSA"] for e in entries]
        vals = np.array(values, dtype=float)
        x = np.arange(len(methods))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        colors = [color_rest] * len(values)

        valid_mask = ~np.isnan(vals)
        valid_idx = np.where(valid_mask)[0]

        top_indices = []
        if len(valid_idx) > 0:
            sorted_valid = valid_idx[np.argsort(vals[valid_idx])[::-1]]
            top_indices = list(sorted_valid[:3])

            if len(top_indices) > 0:
                colors[top_indices[0]] = color_top1
            if len(top_indices) > 1:
                colors[top_indices[1]] = color_top2
            if len(top_indices) > 2:
                colors[top_indices[2]] = color_top3

        bars = ax.bar(x, vals, color=colors)

        for i, bar in enumerate(bars):
            raw_name = raw_methods[i]
            if IN_DOMAIN.get((dataset, raw_name), False):
                bar.set_hatch("////")

        apg_indices = [i for i, name in enumerate(raw_methods)
                       if name.startswith("APG")]

        ais_idx = None
        for i, name in enumerate(raw_methods):
            if name == "AIS - without grid search":
                ais_idx = i
                break

        ais_value = None
        if ais_idx is not None and not np.isnan(vals[ais_idx]):
            ais_value = vals[ais_idx]

        for i in top_indices:
            if np.isnan(vals[i]):
                continue
            if i in apg_indices:
                continue

            v = vals[i]
            bar_height = v
            y_text = min(bar_height + 0.01, 0.98)

            ax.text(
                x[i],
                y_text,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if ais_value is not None:
            for i in apg_indices:
                v = vals[i]
                if np.isnan(v):
                    continue

                diff = v - ais_value
                if diff == 0:
                    continue

                diff_pct = diff * 100.0
                color = "green" if diff > 0 else "red"

                bar_height = v
                y_text = min(bar_height + 0.02, 0.99)

                ax.text(
                    x[i],
                    y_text,
                    f"{diff_pct:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )

        ax.set_title(dataset, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=60, ha="right")
        ax.set_ylim(0.0, 1.0)

    for j in range(len(datasets), len(axes)):
        fig.delaxes(axes[j])

    if suptitle is not None:
        fig.tight_layout(rect=[0.08, 0, 1, 0.95])
    else:
        fig.tight_layout(rect=[0.08, 0, 1, 1])

    fig.text(
        0.075, 0.5,
        "Mean Segmentation Accuracy",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=11,
        fontweight="bold",
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        root, _ = os.path.splitext(save_path)
        svg_path = root + ".svg"
        fig.savefig(svg_path, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":
    plot_msa_grid(
        msa_results,
        ncols=3,
        save_path="msa_nuclei.png",
        suptitle="Nuclei datasets (LM model)",
        display_name_map=DISPLAY_NAME_MAP,
    )

    plot_msa_grid(
        msa_results_fluorescence,
        ncols=3,
        save_path="msa_fluorescence.png",
        suptitle="Fluorescence datasets",
        display_name_map=DISPLAY_NAME_MAP,
    )

    plot_msa_grid(
        msa_results_label_free,
        ncols=3,
        save_path="msa_label_free.png",
        suptitle="Label-free datasets",
        display_name_map=DISPLAY_NAME_MAP,
    )

    plot_msa_grid(
        msa_results_histopathology,
        ncols=3,
        save_path="msa_histopathology.png",
        suptitle="Histopathology datasets",
        display_name_map=DISPLAY_NAME_MAP_HISTO,
    )
