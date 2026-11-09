"""Collect the per-run result tables of the automatic and the interactive tracking into one table each.

The per-run tables are joined into 'automatic_tracking_results.csv' and 'interactive_tracking_results.csv' in the
output root. With '--figure_folder', the script also writes the central tables of the paper, in the layout of the
segmentation tables ('cells_automatic_segmentation_2d_results.csv'), to that folder:
- 'cells_automatic_tracking_results.csv': one row per dataset, method, mode and Trackastra model, averaged over the
  sequences of the dataset.
- 'cells_interactive_tracking_results.csv': one row per dataset, method, first prompt, correction and correction
  iteration, averaged over the movies of the dataset.

Usage example:
    python collect_tracking_results.py --figure_folder ../../../scripts/plotting/v2/for_paper/figures
"""

import os
import glob
import argparse

import pandas as pd

from tracking_util import EXPERIMENT_ROOT

COLUMNS = ["TRA", "DET", "LNK", "SEG", "Division F1 (Frame Buffer 0)", "Division F1 (Frame Buffer 2)"]

# The scores of the central tables, with their names there.
SCORES = {
    "TRA": "tra", "DET": "det", "LNK": "lnk", "SEG": "seg", "AOGM": "aogm",
    "Division F1 (Frame Buffer 0)": "division_f1", "Division F1 (Frame Buffer 2)": "division_f1_buffer2",
    "Mitotic Branching Correctness (Frame Buffer 0)": "mbc",
}

# The modality of each dataset, as (category, structure). The CTC names give it: BF, DIC and PhC are label-free,
# Fluo is fluorescence, C is a cell and N a nucleus. TOIAM and BacMother are bacteria in phase contrast.
# tracking_fm_real mixes C. elegans nuclei (fluorescence) and other movies.
MODALITIES = {
    "BacMother": ("label-free", "cell"), "TOIAM": ("label-free", "cell"),
    "dynamicnuclearnet": ("fluorescence", "nucleus"), "toiam": ("label-free", "cell"),
    "bacmother": ("label-free", "cell"), "tracking_fm_real": ("mixed", "mixed"),
    # The datasets of 'prepare_tracking_data.py'. ALFI labels the nucleus of an interphase cell and the whole mitotic
    # cell; it counts as label-free cells.
    "EColi": ("label-free", "cell"), "ecoli": ("label-free", "cell"), "ALFI": ("label-free", "cell"),
    "alfi": ("label-free", "cell"), "Cardioblast": ("fluorescence", "nucleus"),
    "cardioblast": ("fluorescence", "nucleus"),
    "WingDisc": ("fluorescence", "cell"), "wingdisc": ("fluorescence", "cell"),
    "YeaZ": ("label-free", "cell"), "yeaz": ("label-free", "cell"), "CellACDC": ("label-free", "cell"),
    "cellacdc": ("label-free", "cell"), "DynamicNuclearNet": ("fluorescence", "nucleus"),
}

# The datasets that the micro-sam LM model (vit_l_lm) trained on, after the v1 training script
# ('finetuning/v1/generalists/training/light_microscopy/obtain_lm_datasets.py'): all CTC training sets but
# Fluo-N2DH-GOWT1 and Fluo-N2DL-HeLa, and DynamicNuclearNet. The scores use the same CTC training sequences.
MICROSAM_TRAINING = {
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-C3DH-A549",
    "Fluo-C3DH-A549-SIM", "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N2DH-SIM+", "Fluo-N3DH-CE", "Fluo-N3DH-CHO",
    "Fluo-N3DH-SIM+", "Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "PhC-C2DH-U373", "PhC-C2DL-PSC", "dynamicnuclearnet",
}

# The (method, model) names of the central tables, as in the segmentation tables.
AUTOMATIC_METHODS = {"microsam2": ("micro-sam2", "hvit_l"), "microsam_v1": ("micro-sam", "vit_l_lm"),
                     "cpsam": ("cellpose", "cpsam")}
INTERACTIVE_METHODS = {"hvit_l_cells": ("micro-sam2", "hvit_l"), "hvit_l": ("sam2", "hvit_l"),
                       "vit_l_lm": ("micro-sam", "vit_l_lm")}


def collect(folder):
    """Concatenate all result tables of one pipeline, or None without any."""
    paths = sorted(glob.glob(os.path.join(folder, "results", "*.csv")))
    if not paths:
        return None
    return pd.concat([pd.read_csv(path, dtype={"sample": str}) for path in paths], ignore_index=True)


def modality(dataset):
    if dataset in MODALITIES:
        return MODALITIES[dataset]
    prefix, kind = dataset.split("-")[:2]
    return ("fluorescence" if prefix == "Fluo" else "label-free", "nucleus" if kind.startswith("N") else "cell")


def scores_of(table, keys):
    """Average the scores over the sequences or movies of each group, and name them as in the central tables."""
    scores = [column for column in SCORES if column in table.columns]
    grouped = table.groupby(keys, dropna=False)
    result = grouped[scores].mean().rename(columns=SCORES)
    result.insert(0, "n_samples", grouped.size())
    return result.reset_index()


def central_automatic(table):
    """The automatic table of the paper: one row per dataset, method, mode and Trackastra model."""
    # Only the default 2/98 percentile normalization; the other normalizations were tests.
    table = table[table["norm_percentiles"].isna() | (table["norm_percentiles"] == "2,98")].copy()
    # The first micro-sam2 runs have no method column.
    table["method"] = table["method"].fillna("microsam2")
    table = table.drop_duplicates(["dataset", "sequence", "method", "mode", "tracking_model"], keep="last")
    table["model"] = [AUTOMATIC_METHODS[method][1] for method in table["method"]]
    table["method"] = [AUTOMATIC_METHODS[method][0] for method in table["method"]]
    # micro-sam runs AIS; the baselines have no mode.
    table.loc[table["method"] == "micro-sam", "mode"] = "ais"
    table.loc[table["method"] == "cellpose", "mode"] = None
    table["ndim"] = table["ndim"].str[0].astype(int)
    keys = ["dataset", "ndim", "in_training", "method", "model", "mode", "tracking_model", "linking_mode"]
    result = scores_of(table, keys)
    result.insert(1, "category", [modality(d)[0] for d in result["dataset"]])
    result.insert(2, "structure", [modality(d)[1] for d in result["dataset"]])
    # The domain is that of micro-sam2: in domain if micro-sam2 trained on the dataset.
    result.insert(3, "domain", ["ID" if seen else "OOD" for seen in result.pop("in_training")])
    result.insert(5, "in_microsam_training", result["dataset"].isin(MICROSAM_TRAINING))
    result["primary_metric"], result["lower_is_better"] = "tra", False
    return result.sort_values(["ndim", "dataset", "method", "mode", "tracking_model"], na_position="first")


def central_interactive(table):
    """The interactive table of the paper: one row per dataset, method, first prompt, correction and iteration."""
    # Only the default 2/98 percentile normalization; the 1/99.8 runs were tests.
    table = table[table["norm_percentiles"].isna() | (table["norm_percentiles"] == "2,98")].copy()
    table["model"], table["method"] = (
        [INTERACTIVE_METHODS[model][1] for model in table["model"]],
        [INTERACTIVE_METHODS[model][0] for model in table["model"]],
    )
    # The first runs corrected with points and have no correction column.
    table["correction"] = table["correction"].fillna("points")
    table = table.rename(columns={"first_prompt": "prompt", "n_corrections": "iteration"})
    keys = ["dataset", "domain", "method", "model", "prompt", "correction", "accept_iou", "iteration"]
    result = scores_of(table, keys)
    result["n_prompts"] = table.groupby(keys, dropna=False)["n_prompts"].mean().values
    result.insert(1, "category", [modality(d)[0] for d in result["dataset"]])
    result.insert(2, "structure", [modality(d)[1] for d in result["dataset"]])
    result["domain"] = result["domain"].map({"in-domain": "ID", "out-of-domain": "OOD"})
    result.insert(4, "in_microsam_training", result["dataset"].isin(MICROSAM_TRAINING))
    result["primary_metric"], result["lower_is_better"] = "tra", False
    return result.sort_values(["dataset", "method", "correction", "iteration"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output_root", type=str, default=EXPERIMENT_ROOT)
    parser.add_argument("--figure_folder", type=str, default=None, help="Where to write the central tables.")
    args = parser.parse_args()

    for pipeline, keys, central in (
        ("automatic", ["dataset", "sequence", "ndim", "in_training", "mode"], central_automatic),
        ("interactive", ["dataset", "domain", "sample", "n_corrections", "n_prompts"], central_interactive),
    ):
        table = collect(os.path.join(args.output_root, pipeline))
        if table is None:
            print(f"No {pipeline} results yet.")
            continue
        save_path = os.path.join(args.output_root, f"{pipeline}_tracking_results.csv")
        table.to_csv(save_path, index=False)
        columns = keys + [column for column in COLUMNS if column in table.columns]
        print(f"{pipeline} ({save_path}):")
        print(table[columns].round(3).to_string(index=False))
        if args.figure_folder is not None:
            path = os.path.join(args.figure_folder, f"cells_{pipeline}_tracking_results.csv")
            central(table).to_csv(path, index=False)
            print(f"Wrote the central table '{path}'.")


if __name__ == "__main__":
    main()
