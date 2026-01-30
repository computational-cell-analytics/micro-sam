import numpy as np
import pandas as pd

from scipy.stats import shapiro, wilcoxon


metric = "msa"
criterion = 0.05


def statistical_analysis_dataset(dataset, method1, method2, verbose=True):
    res1 = pd.read_csv(f"./results/{method1}/{dataset}.csv")[metric].values
    res2 = pd.read_csv(f"./results/{method2}/{dataset}.csv")[metric].values
    assert res1.shape == res2.shape

    diff = res1 - res2

    _, p_gauss = shapiro(diff)
    if verbose:
        print("P-value for gaussian distribution:", p_gauss)

    is_better = diff.sum() > 0
    _, p = wilcoxon(diff, alternative="greater" if is_better else "less")
    is_significant = p < criterion
    if verbose:
        print("Hypothesis:", method1 if is_better else method2, "is better than", method2 if is_better else method1)
        print("Result:", "True" if is_significant else "False", f"(p = {p:04f})")

    return is_better, is_significant


def statistical_analysis_pair(datasets, method1, method2, verbose=False):
    better1 = 0
    better2 = 0
    neutral = 0

    for ds in datasets:
        is_better, is_significant = statistical_analysis_dataset(ds, method1, method2, verbose=verbose)
        if is_significant and is_better:
            better1 += 1
        elif is_significant:
            better2 += 1
        else:
            neutral += 1

    assert better1 + better2 + neutral == len(datasets)
    if verbose:
        print(method1, "better than", method2, ":", better1)
        print(method2, "better than", method1, ":", better2)
        print("No difference:", neutral)
    return better1, better2, neutral


def get_datasets(domain):
    # TODO this needs to be double checked.
    domain_to_ds = {
        "fluo_cells": [  # 9 datasets
            "cellpose",
            "covid",
            "hpa",
            "plantseg_root",
            "plantseg_ovules",
            "pnas",
            "tissuenet",
            "cellbindb",
            "mouse",
        ],
        "fluo_nuclei": [  # 9 datasets
            "arvidsson",
            "bitdepth",
            "dsb",
            "dynamicnuclearnet",
            "gonuclear",
            "ifnuclei",
            "nis3d",
            "parhyale",
            "u20s",
        ],
        "label_free": [  # 9 datasets
            "deepbacs",
            "deepseas",
            "livecell",
            "omnipose",
            "usiigaci",
            "vicar",
            "yeaz",
            "toiam",
            "segpc",
        ],
        "histopatho": [  # 9 datasets
            "cryonuseg",
            "cytodark0",
            "ihc",
            "monuseg",
            "lynsec",
            "nuinsseg",
            "pannuke",
            "puma",
            "tnbc",
        ],
    }
    datasets = domain_to_ds[domain]
    # assert len(datasets) == 9
    return datasets


def compare_all():
    methods = ["apg", "ais", "cpsam"]
    domains = ["fluo_cells", "fluo_nuclei", "label_free", "histopatho"]

    n_methods = len(methods)
    for domain in domains:
        datasets = get_datasets(domain)
        comparison = comparison = np.empty((n_methods, n_methods), dtype="U10")

        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method1, method2 = methods[i], methods[j]
                better1, better2, neutral = statistical_analysis_pair(datasets, method1, method2)
                comparison[i, j] = f"{better1} / {better2} / {neutral}"
                comparison[j, i] = f"{better2} / {better1} / {neutral}"

        comparison = pd.DataFrame(comparison, index=methods, columns=methods)
        print(domain, ":")
        print(comparison.to_string())
        print()
        print()


def main():
    compare_all()


if __name__ == "__main__":
    main()
