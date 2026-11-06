"""Collect the per-run result tables of the automatic and the interactive tracking into one table each.

Usage example:
    python collect_tracking_results.py
"""

import os
import glob
import argparse

import pandas as pd

from tracking_util import EXPERIMENT_ROOT

COLUMNS = ["TRA", "DET", "LNK", "SEG", "Division F1 (Frame Buffer 0)", "Division F1 (Frame Buffer 2)"]


def collect(folder):
    """Concatenate all result tables of one pipeline, or None without any."""
    paths = sorted(glob.glob(os.path.join(folder, "results", "*.csv")))
    if not paths:
        return None
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output_root", type=str, default=EXPERIMENT_ROOT)
    args = parser.parse_args()

    for pipeline, keys in (
        ("automatic", ["dataset", "sequence", "ndim", "in_training", "mode"]),
        ("interactive", ["dataset", "domain", "sample", "n_corrections", "n_prompts"]),
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


if __name__ == "__main__":
    main()
