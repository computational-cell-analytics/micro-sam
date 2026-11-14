"""Summarize the data overview table that ``create_data_overview.py`` writes.

The readout is chosen with the positional argument:

- ``train`` (default): The number of annotated objects in the training data (train + val split) in five groups:
  cells and nuclei in 2d and 3d light microscopy (LM), and all structures in electron microscopy (EM).
  EM covers the EM datasets and the EM-like LICONN (expansion microscopy) and XPRESS (X-ray) datasets. Everything
  else is LM, including histopathology and label-free data. LM rows labeled 'nucleus' count as nuclei, all other
  LM structures (cells, organoids, worms, prospore membranes, axons) count as cells.
- ``test-2d``: The number of annotated objects in the scored 2d test data (see ``create_data_overview.py``) for
  the four panels of Figure 2: label-free cells, fluorescence cells, fluorescence nuclei and histopathology nuclei.
  For each the ID, the OOD and the summed (ID + OOD) objects are printed; the summed objects of all groups are all
  annotations of the scored 2d test splits. All label-free datasets count as cells, including the organoids of
  OrgLine and the C. elegans of BBBC010, which are in the label-free cells panel.
- ``test-3d``: The number of annotated objects in the scored 3d test data (test crops, see
  ``create_data_overview.py``) for the three panels of Figure 3: LM cells, LM nuclei and EM neurites. For each the
  ID, the OOD and the summed (ID + OOD) objects are printed. LICONN (expansion microscopy) counts as EM, as in the
  EM neurites panel of Figure 3.
- ``detail``: A Markdown table with the structures and the images or volumes per modality, structure and
  dimensionality, for the training data and the scored 2d and 3d test data (ID and OOD). The training rows use the
  groups of ``train``, split by modality: label-free and histopathology datasets count as cells and nuclei, EM and
  the EM-like LICONN and XPRESS count as 'Cell / Neurite'. Cellpose (fluorescence) and NeurIPS CellSeg
  (label-free) mix modalities and count under their majority modality. The test rows are the groups of
  ``test-2d`` and ``test-3d``; for 3d the images / volumes are the scored test crops.

Usage:

    python summarize_data_overview.py  # same as 'train'
    python summarize_data_overview.py train
    python summarize_data_overview.py test-2d
    python summarize_data_overview.py test-3d
    python summarize_data_overview.py detail
"""

import csv
import argparse
from collections import defaultdict

from create_data_overview import DEFAULT_OUTPUT

# The datasets that count as EM although their modality is not EM.
EM_LIKE = {"liconn", "xpress"}
GROUPS = ["2d LM cells", "2d LM nuclei", "3d LM cells", "3d LM nuclei", "EM structures"]
# The groups of the 2d test data, the panels of Figure 2.
TEST_2D_GROUPS = ["label-free cells", "fluorescence cells", "fluorescence nuclei", "histopathology nuclei"]
# The groups of the 3d test data, the panels of Figure 3.
TEST_3D_GROUPS = ["LM cells", "LM nuclei", "EM neurites"]


def read_table(path):
    """Read the rows of the data overview table."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select_train_rows(rows):
    """Return the rows with training counts. Raise an error if a dataset with training data has none."""
    missing = [
        row["dataset"] for row in rows if row["role"] in ("train", "ID") and row["train_objects"] == ""
        and not row["dataset"].startswith("neurips_cellseg_")  # their training data is the 'neurips_cellseg' row
    ]
    if missing:
        raise RuntimeError(
            f"The table has no training counts for {missing}. Run create_data_overview.py to fill them in."
        )
    return [row for row in rows if row["role"] in ("train", "ID") and row["train_objects"] != ""]


def select_test_rows(rows, ndim):
    """Return the rows of the given dimensionality ('2d' or '3d') with a scored test split.

    Raise an error if one has no test counts.
    """
    test_rows = [row for row in rows if row["dimensionality"] == ndim and row["role"] in ("ID", "OOD")]
    missing = [row["dataset"] for row in test_rows if row["test_objects"] == ""]
    if missing:
        raise RuntimeError(f"The table has no test counts for {missing}. Run create_data_overview.py to fill them in.")
    return test_rows


def train_group(row):
    """Return the group of a dataset with training data."""
    if row["modality"] == "EM" or row["dataset"] in EM_LIKE:
        return "EM structures"
    kind = "nuclei" if row["structure"] == "nucleus" else "cells"
    return f"{row['dimensionality']} LM {kind}"


def print_train(rows):
    """Print the annotated objects of the training data per group."""
    train_rows = select_train_rows(rows)
    totals = defaultdict(lambda: {"objects": 0, "images": 0, "datasets": 0, "structures": set()})
    for row in train_rows:
        total = totals[train_group(row)]
        total["objects"] += int(row["train_objects"])
        total["images"] += int(row["train_images_or_volumes"])
        total["datasets"] += 1
        total["structures"].add(row["structure"])

    print("Annotated objects in the training data (train + val split):")
    print(f"{'group':<15} {'objects':>12} {'images / volumes':>17} {'datasets':>9}  structures")
    for group in GROUPS:
        total = totals[group]
        print(
            f"{group:<15} {total['objects']:>12,} {total['images']:>17,} {total['datasets']:>9}  "
            f"{', '.join(sorted(total['structures']))}"
        )
    print(
        f"{'total':<15} {sum(t['objects'] for t in totals.values()):>12,} "
        f"{sum(t['images'] for t in totals.values()):>17,} {sum(t['datasets'] for t in totals.values()):>9}"
    )


def test_2d_group(row):
    """Return the group (the Figure 2 panel) of a 2d dataset with a scored test split."""
    if row["modality"] == "label-free":
        return "label-free cells"
    if row["modality"] == "histopathology":
        return "histopathology nuclei"
    if row["modality"] == "fluorescence":
        return "fluorescence nuclei" if row["structure"] == "nucleus" else "fluorescence cells"
    raise ValueError(f"{row['dataset']} has the modality {row['modality']}, which is in no 2d test group.")


def test_3d_group(row):
    """Return the group (the Figure 3 panel) of a 3d dataset with a scored test split."""
    if row["modality"] == "EM" or row["dataset"] in EM_LIKE:
        return "EM neurites"
    if row["modality"] == "fluorescence":
        return "LM nuclei" if row["structure"] == "nucleus" else "LM cells"
    raise ValueError(f"{row['dataset']} has the modality {row['modality']}, which is in no 3d test group.")


def print_test(rows, ndim, group_fn, groups):
    """Print the annotated objects of the scored test splits per group, for ID, OOD and both."""
    test_rows = select_test_rows(rows, ndim)
    totals = defaultdict(lambda: {"ID": 0, "OOD": 0, "images": 0, "datasets": 0})
    for row in test_rows:
        total = totals[group_fn(row)]
        total[row["role"]] += int(row["test_objects"])
        total["images"] += int(row["test_images_or_volumes"])
        total["datasets"] += 1

    # The groups must cover every scored object.
    expected = sum(int(row["test_objects"]) for row in test_rows)
    grand_total = sum(t["ID"] + t["OOD"] for t in totals.values())
    assert grand_total == expected, (grand_total, expected)

    images = "images" if ndim == "2d" else "crops"
    print(f"Annotated objects in the scored {ndim} test data:")
    print(f"{'group':<22} {'ID':>10} {'OOD':>10} {'ID + OOD':>10} {images:>8} {'datasets':>9}")
    for group in groups:
        total = totals[group]
        print(
            f"{group:<22} {total['ID']:>10,} {total['OOD']:>10,} {total['ID'] + total['OOD']:>10,} "
            f"{total['images']:>8,} {total['datasets']:>9}"
        )
    print(
        f"{'total':<22} {sum(t['ID'] for t in totals.values()):>10,} {sum(t['OOD'] for t in totals.values()):>10,} "
        f"{grand_total:>10,} {sum(t['images'] for t in totals.values()):>8,} "
        f"{sum(t['datasets'] for t in totals.values()):>9}"
    )


# The rows of the detail table: (modality, structure, dimensionality).
DETAIL_TRAIN_ROWS = [
    ("Label-free", "Cell", "2"),
    ("Fluorescence", "Cell", "2"),
    ("Fluorescence", "Nucleus", "2"),
    ("Histopathology", "Nucleus", "2"),
    ("Electron Microscopy", "Cell / Neurite", "2"),
    ("Fluorescence", "Cell", "3"),
    ("Fluorescence", "Nucleus", "3"),
    ("Electron Microscopy", "Cell / Neurite", "3"),
]
DETAIL_TEST_2D_ROWS = {
    "label-free cells": ("Label-free", "Cell", "2"),
    "fluorescence cells": ("Fluorescence", "Cell", "2"),
    "fluorescence nuclei": ("Fluorescence", "Nucleus", "2"),
    "histopathology nuclei": ("Histopathology", "Nucleus", "2"),
}
DETAIL_TEST_3D_ROWS = {
    "LM cells": ("Fluorescence", "Cell", "3"),
    "LM nuclei": ("Fluorescence", "Nucleus", "3"),
    "EM neurites": ("Electron Microscopy", "Cell / Neurite", "3"),
}


def detail_train_row(row):
    """Return the detail table row (modality, structure, dimensionality) of a dataset with training data."""
    ndim = row["dimensionality"].rstrip("d")
    if row["modality"] == "EM" or row["dataset"] in EM_LIKE:
        return ("Electron Microscopy", "Cell / Neurite", ndim)
    if row["modality"] == "label-free":
        return ("Label-free", "Cell", ndim)
    if row["modality"] == "histopathology":
        return ("Histopathology", "Nucleus", ndim)
    if row["modality"] == "fluorescence":
        return ("Fluorescence", "Nucleus" if row["structure"] == "nucleus" else "Cell", ndim)
    raise ValueError(f"{row['dataset']} has the modality {row['modality']}, which is in no detail row.")


def print_detail(rows):
    """Print the detail table of the training data and the scored 2d and 3d test data as Markdown."""
    train_rows = select_train_rows(rows)
    train = defaultdict(lambda: {"objects": 0, "images": 0})
    for row in train_rows:
        key = detail_train_row(row)
        if key not in DETAIL_TRAIN_ROWS:
            raise ValueError(f"{row['dataset']} falls into {key}, which is not a row of the detail table.")
        train[key]["objects"] += int(row["train_objects"])
        train[key]["images"] += int(row["train_images_or_volumes"])
    assert sum(t["objects"] for t in train.values()) == sum(int(row["train_objects"]) for row in train_rows)

    test = defaultdict(lambda: {"ID": 0, "OOD": 0, "ID images": 0, "OOD images": 0})
    for ndim, group_fn, detail_rows in [
        ("2d", test_2d_group, DETAIL_TEST_2D_ROWS), ("3d", test_3d_group, DETAIL_TEST_3D_ROWS)
    ]:
        test_rows = select_test_rows(rows, ndim)
        for row in test_rows:
            key = detail_rows[group_fn(row)]
            test[key][row["role"]] += int(row["test_objects"])
            test[key][f"{row['role']} images"] += int(row["test_images_or_volumes"])
        assert sum(test[key]["ID"] + test[key]["OOD"] for key in detail_rows.values()) == sum(
            int(row["test_objects"]) for row in test_rows
        )

    print("| Modality | Structure | Dimensionality | Split | # Structures | # Images / Volumes |")
    print("|---|---|---|---|---|---|")
    for key in DETAIL_TRAIN_ROWS:
        print(f"| {' | '.join(key)} | Train | {train[key]['objects']:,} | {train[key]['images']:,} |")
    for key in [*DETAIL_TEST_2D_ROWS.values(), *DETAIL_TEST_3D_ROWS.values()]:
        counts = test[key]
        print(
            f"| {' | '.join(key)} | Test | {counts['ID']:,} (ID); {counts['OOD']:,} (OOD) | "
            f"{counts['ID images']:,} (ID); {counts['OOD images']:,} (OOD) |"
        )


def print_test_2d(rows):
    """Print the annotated objects of the scored 2d test splits per Figure 2 panel, for ID, OOD and both."""
    print_test(rows, "2d", test_2d_group, TEST_2D_GROUPS)


def print_test_3d(rows):
    """Print the annotated objects of the scored 3d test crops per Figure 3 panel, for ID, OOD and both."""
    print_test(rows, "3d", test_3d_group, TEST_3D_GROUPS)


READOUTS = {"train": print_train, "test-2d": print_test_2d, "test-3d": print_test_3d, "detail": print_detail}


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the data overview table.", formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("readout", nargs="?", default="train", choices=list(READOUTS), help="The readout to print.")
    parser.add_argument("-i", "--input", default=DEFAULT_OUTPUT, help="The data overview table.")
    args = parser.parse_args()
    READOUTS[args.readout](read_table(args.input))


if __name__ == "__main__":
    main()
