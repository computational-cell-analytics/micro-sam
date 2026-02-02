import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, wilcoxon


METRIC = "msa"
CRITERION = 0.05


def statistical_analysis_dataset(dataset, method1_path, method2_path, verbose=True):
    res1 = pd.read_csv(f"./results/{method1_path}/{dataset}.csv")[METRIC].values
    res2 = pd.read_csv(f"./results/{method2_path}/{dataset}.csv")[METRIC].values
    assert res1.shape == res2.shape

    diff = res1 - res2

    _, p_gauss = shapiro(diff)
    if verbose:
        print("P-value for gaussian distribution:", p_gauss)

    is_better = diff.sum() > 0
    _, p = wilcoxon(diff, alternative="greater" if is_better else "less")
    is_significant = p < CRITERION

    if verbose:
        print(
            "Hypothesis:", method1_path if is_better else method2_path, "is better than",
            method2_path if is_better else method1_path
        )
        print("Result:", "True" if is_significant else "False", f"(p = {p:.4f})")

    return is_better, is_significant


def statistical_analysis_pair(datasets, method1_path, method2_path, verbose=False):
    better1 = 0
    better2 = 0
    neutral = 0

    for ds in datasets:
        is_better, is_significant = statistical_analysis_dataset(ds, method1_path, method2_path, verbose=verbose)
        if is_significant and is_better:
            better1 += 1
        elif is_significant:
            better2 += 1
        else:
            neutral += 1

    assert better1 + better2 + neutral == len(datasets)
    if verbose:
        print(method1_path, "better than", method2_path, ":", better1)
        print(method2_path, "better than", method1_path, ":", better2)
        print("No difference:", neutral)
    return better1, better2, neutral


def get_datasets(domain):
    domain_to_ds = {
        "fluo_cells": [
            "cellpose",
            "covid_if",
            "hpa",
            "plantseg_root",
            "plantseg_ovules",
            "pnas_arabidopsis",
            "tissuenet",
            "cellbindb",
            "mouse_embryo",
        ],
        "fluo_nuclei": [
            "arvidsson",
            "bitdepth_nucseg",
            "dsb",
            "dynamicnuclearnet",
            "gonuclear",
            "ifnuclei",
            "nis3d",
            "parhyale_regen",
            "u20s",
        ],
        "label_free": [
            "deepbacs",
            "deepseas",
            "livecell",
            "omnipose",
            "usiigaci",
            "vicar",
            "toiam",
            "yeaz",
            "segpc",
        ],
        "histopatho": [
            "cytodark0",
            "ihc_tma",
            "monuseg",
            "lynsec",
            "nuinsseg",
            "pannuke",
            "puma",
            "tnbc",
            "cryonuseg",
        ],
    }
    datasets = domain_to_ds[domain]
    assert len(datasets) == 9
    return datasets


def _plot_comparison_heatmap(domain, comparison_df, title=None):
    # Extract wins for method in row vs method in column
    n = len(comparison_df)
    win_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                parts = comparison_df.iloc[i, j].split(' / ')
                win_matrix[i, j] = int(parts[0])  # wins for row method

    # Mask the diagonal to exclude it from coloring
    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        win_matrix, annot=comparison_df.values, fmt='',
        cmap='RdYlGn', center=len(get_datasets(domain))/2,
        xticklabels=comparison_df.columns,
        yticklabels=comparison_df.index,
        cbar_kws={'label': 'Wins'}, ax=ax,
        mask=mask,
        linewidths=0.5, linecolor='black'
    )

    # Use custom title if provided, otherwise use default
    if title is None:
        title = f'{domain.replace("_", " ").title()} - Method Comparison'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'comparison_heatmap_{domain}.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_all():
    # Define method paths: method/variant or just method
    method_configs = {
        "amg": "amg/vit_b",
        "ais_lm": "ais/vit_b_lm",
        "ais_histo": "ais/vit_b_histopathology",
        "cellpose3": "cellpose/cyto3",
        "cellpose4": "cellpose/cpsam",
        "cellsam": "cellsam/cellsam",
        "sam3": "sam3/cell",
        "apg_lm": "apg/vit_b_lm",
        "apg_histo": "apg/vit_b_histopathology",
    }

    # Define which methods to compare for each domain (ordered as in screenshot)
    domain_methods = {
        "fluo_cells": ["amg", "ais_lm", "cellpose3", "cellpose4", "cellsam", "sam3", "apg_lm"],
        "fluo_nuclei": ["amg", "ais_lm", "cellpose3", "cellpose4", "cellsam", "sam3", "apg_lm"],
        "label_free": ["amg", "ais_lm", "cellpose3", "cellpose4", "cellsam", "sam3", "apg_lm"],
        "histopatho": ["amg", "ais_histo", "cellpose3", "cellpose4", "cellsam", "sam3", "apg_histo"],
    }

    # Map internal keys to display names
    display_names = {
        "amg": "AMG (SAM)",
        "ais_lm": "AIS (μSAM)",
        "ais_histo": "AIS (μSAM)",
        "cellsam": "CellSAM",
        "cellpose3": "Cellpose 3",
        "cellpose4": "CellposeSAM",
        "sam3": "SAM3",
        "apg_lm": "APG (μSAM)",
        "apg_histo": "APG (μSAM)",
    }

    # Custom titles for each domain
    custom_titles = {
        "fluo_cells": "Fluorescence Microscopy (Cell Segmentation)",
        "fluo_nuclei": "Fluorescence Microscopy (Nucleus Segmentation)",
        "label_free": "Label-Free Microscopy (Cell Segmentation)",
        "histopatho": "Histopathology (Nucleus Segmentation)",
    }

    for domain in ["fluo_cells", "fluo_nuclei", "label_free", "histopatho"]:
        datasets = get_datasets(domain)
        methods = domain_methods[domain]
        n_methods = len(methods)

        comparison = np.empty((n_methods, n_methods), dtype="U15")

        for i in range(n_methods):
            for j in range(n_methods):
                if i == j:
                    comparison[i, j] = "-"
                    continue

                method_row = methods[i]
                method_col = methods[j]
                method_row_path = method_configs[method_row]
                method_col_path = method_configs[method_col]

                better_row, better_col, neutral = statistical_analysis_pair(
                    datasets, method_row_path, method_col_path
                )
                comparison[i, j] = f"{better_row} / {better_col} / {neutral}"

        # Use display names for the DataFrame
        display_method_names = [display_names[m] for m in methods]
        comparison = pd.DataFrame(comparison, index=display_method_names, columns=display_method_names)

        # Visualize with custom title
        _plot_comparison_heatmap(domain, comparison, title=custom_titles[domain])
        print(f"Generated heatmap for {domain}: comparison_heatmap_{domain}.png")


def main():
    compare_all()


if __name__ == "__main__":
    main()
