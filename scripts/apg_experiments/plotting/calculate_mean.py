import numpy as np

from plot_util import (
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

AVG_DISPLAY_NAME_MAP = {
    "AMG (vit_b) - without grid search": "SAM",
    "AIS - without grid search": "AIS (µSAM)",
    "SAM3": "SAM3",
    "CellPose3": "CellPose 3",
    "CellPoseSAM": "CellPoseSAM",
    "CellSAM": "CellSAM",
    "APG - without grid search (cc)": "APG (µSAM)",
}


def compute_method_means(msa_results_dict, methods_filter=None):
    vals_per_method = {}

    for _, entries in msa_results_dict.items():
        for e in entries:
            m = e["method"]
            v = e["mSA"]
            if methods_filter is not None and m not in methods_filter:
                continue
            if v is None:
                continue
            vals_per_method.setdefault(m, []).append(float(v))

    mean_msa = {m: float(np.mean(vs)) for m, vs in vals_per_method.items()}
    return mean_msa


def format_float(v):
    return f"{v:.3f}"


if __name__ == "__main__":
    datasets = {
        "Label-Free (Cell)": msa_results_label_free,
        "Fluorescence (Cell)": msa_results_fluorescence,
        "Fluorescence (Nucleus)": msa_results,
        "Histopathology (Nucleus)": msa_results_histopathology,
    }

    means_by_modality = {}
    for mod_name, data in datasets.items():
        means_by_modality[mod_name] = compute_method_means(data, methods_filter=AVG_METHODS)

    # Print a compact validation table
    col_order = list(datasets.keys())
    print("Averages (mSA) per modality:")
    print("Method".ljust(22), *(c[:18].rjust(18) for c in col_order))
    for m in AVG_METHODS:
        disp = AVG_DISPLAY_NAME_MAP.get(m, m)
        row = [disp.ljust(22)]
        for mod in col_order:
            v = means_by_modality[mod].get(m, np.nan)
            row.append((format_float(v) if np.isfinite(v) else "--").rjust(18))
        print("".join(row))
