"""Summarize the EM organelle data overview table that ``create_data_overview_em_organelles.py`` writes.

The readout is chosen with the positional argument, as for ``summarize_data_overview.py``:

- ``train`` (default): The number of annotated objects in the training data (train + val split) of the two EM
  organelle models: mitochondria and nuclei. 2d and 3d data are not distinguished, so the images / volumes column
  sums 2d images and 3d volumes. The structure column of the table decides the group, so SXT INS-1E (soft X-ray
  tomography, EM-like) counts under nuclei.
- ``test``: The number of annotated objects in the scored test data (test crops, see
  ``create_data_overview_em_organelles.py``) of the two panels of Figure 6: mitochondria and nuclei. For each the ID,
  the OOD and the summed (ID + OOD) objects are printed. A dataset that Figure 6 pools from several volumes counts
  as one dataset.
- ``detail``: A Markdown table with the structures and the images or volumes per structure and dimensionality, for
  the training data (2d and 3d) and the scored test data (ID and OOD, all 3d), in the format of the ``detail``
  readout of ``summarize_data_overview.py``. SXT INS-1E counts as EM, as in ``train``. For the test data the
  images / volumes are the scored test crops.

Usage:

    python summarize_data_overview_em_organelles.py          # same as 'train'
    python summarize_data_overview_em_organelles.py train
    python summarize_data_overview_em_organelles.py test
    python summarize_data_overview_em_organelles.py detail
"""

import argparse
from collections import defaultdict

from create_data_overview_em_organelles import DEFAULT_OUTPUT, ENTRIES, MITO, NUCLEUS
from summarize_data_overview import read_table, select_test_rows

# The groups of the training data: the structure of the table and the name it is printed with.
GROUPS = {MITO: "mitochondria", NUCLEUS: "nuclei"}


def select_train_rows(rows):
    """Return the rows with training data. Raise an error if one has no training counts.

    Unlike in the generalist table, some ID rows have test data only, so the registry decides which rows have
    training data.
    """
    with_train = {entry.name for entry in ENTRIES if entry.train is not None}
    train_rows = [row for row in rows if row["dataset"] in with_train]
    missing = sorted(with_train - {row["dataset"] for row in train_rows if row["train_objects"] != ""})
    if missing:
        raise RuntimeError(
            f"The table has no training counts for {missing}. Run create_data_overview_em_organelles.py to fill "
            "them in."
        )
    return train_rows


def print_train(rows):
    """Print the annotated objects of the training data per structure."""
    train_rows = select_train_rows(rows)
    totals = defaultdict(lambda: {"objects": 0, "images": 0, "datasets": 0, "modalities": set()})
    for row in train_rows:
        if row["structure"] not in GROUPS:
            raise ValueError(f"{row['dataset']} has the structure {row['structure']}, which is in no group.")
        total = totals[row["structure"]]
        total["objects"] += int(row["train_objects"])
        total["images"] += int(row["train_images_or_volumes"])
        total["datasets"] += 1
        total["modalities"].add(row["modality"])

    print("Annotated objects in the EM organelle training data (train + val split, 2d and 3d):")
    print(f"{'group':<15} {'objects':>12} {'images / volumes':>17} {'datasets':>9}  modalities")
    for structure, group in GROUPS.items():
        total = totals[structure]
        print(
            f"{group:<15} {total['objects']:>12,} {total['images']:>17,} {total['datasets']:>9}  "
            f"{', '.join(sorted(total['modalities']))}"
        )
    print(
        f"{'total':<15} {sum(t['objects'] for t in totals.values()):>12,} "
        f"{sum(t['images'] for t in totals.values()):>17,} {sum(t['datasets'] for t in totals.values()):>9}"
    )


def print_test(rows):
    """Print the annotated objects of the scored test crops per structure, for ID, OOD and both."""
    test_rows = select_test_rows(rows, "3d")
    totals = defaultdict(lambda: {"ID": 0, "OOD": 0, "crops": 0, "datasets": 0})
    for row in test_rows:
        if row["structure"] not in GROUPS:
            raise ValueError(f"{row['dataset']} has the structure {row['structure']}, which is in no group.")
        total = totals[row["structure"]]
        total[row["role"]] += int(row["test_objects"])
        total["crops"] += int(row["test_images_or_volumes"])
        total["datasets"] += 1

    # The groups must cover every scored object; the test data of Figure 6 is all 3d.
    expected = sum(int(row["test_objects"]) for row in rows if row["role"] in ("ID", "OOD"))
    grand_total = sum(t["ID"] + t["OOD"] for t in totals.values())
    assert grand_total == expected, (grand_total, expected)

    print("Annotated objects in the scored EM organelle test data:")
    print(f"{'group':<15} {'ID':>10} {'OOD':>10} {'ID + OOD':>10} {'crops':>8} {'datasets':>9}")
    for structure, group in GROUPS.items():
        total = totals[structure]
        print(
            f"{group:<15} {total['ID']:>10,} {total['OOD']:>10,} {total['ID'] + total['OOD']:>10,} "
            f"{total['crops']:>8,} {total['datasets']:>9}"
        )
    print(
        f"{'total':<15} {sum(t['ID'] for t in totals.values()):>10,} {sum(t['OOD'] for t in totals.values()):>10,} "
        f"{grand_total:>10,} {sum(t['crops'] for t in totals.values()):>8,} "
        f"{sum(t['datasets'] for t in totals.values()):>9}"
    )


# The rows of the detail table: (modality, structure, dimensionality) and the structure of the table.
DETAIL_STRUCTURES = {MITO: "Mitochondrion", NUCLEUS: "Nucleus"}
DETAIL_TRAIN_ROWS = [(MITO, "2"), (NUCLEUS, "2"), (MITO, "3"), (NUCLEUS, "3")]
DETAIL_TEST_ROWS = [(MITO, "3"), (NUCLEUS, "3")]
DETAIL_MODALITY = "Electron microscopy"


def _detail_key(row):
    if row["structure"] not in GROUPS:
        raise ValueError(f"{row['dataset']} has the structure {row['structure']}, which is in no group.")
    return row["structure"], row["dimensionality"].rstrip("d")


def _detail_label(key):
    structure, ndim = key
    return f"{DETAIL_MODALITY} | {DETAIL_STRUCTURES[structure]} | {ndim}"


def print_detail(rows):
    """Print the detail table of the training data and the scored test data as Markdown."""
    train_rows = select_train_rows(rows)
    train = defaultdict(lambda: {"objects": 0, "images": 0})
    for row in train_rows:
        key = _detail_key(row)
        if key not in DETAIL_TRAIN_ROWS:
            raise ValueError(f"{row['dataset']} falls into {key}, which is not a row of the detail table.")
        train[key]["objects"] += int(row["train_objects"])
        train[key]["images"] += int(row["train_images_or_volumes"])
    assert sum(t["objects"] for t in train.values()) == sum(int(row["train_objects"]) for row in train_rows)

    test_rows = select_test_rows(rows, "3d")
    test = defaultdict(lambda: {"ID": 0, "OOD": 0, "ID images": 0, "OOD images": 0})
    for row in test_rows:
        key = _detail_key(row)
        test[key][row["role"]] += int(row["test_objects"])
        test[key][f"{row['role']} images"] += int(row["test_images_or_volumes"])
    expected = sum(int(row["test_objects"]) for row in rows if row["role"] in ("ID", "OOD"))
    assert sum(t["ID"] + t["OOD"] for t in test.values()) == expected

    print("| Modality | Structure | Dimensionality | Split | # Structures | # Images / Volumes |")
    print("|---|---|---|---|---|---|")
    for key in DETAIL_TRAIN_ROWS:
        print(f"| {_detail_label(key)} | Train | {train[key]['objects']:,} | {train[key]['images']:,} |")
    for key in DETAIL_TEST_ROWS:
        counts = test[key]
        print(
            f"| {_detail_label(key)} | Test | {counts['ID']:,} (ID); {counts['OOD']:,} (OOD) | "
            f"{counts['ID images']:,} (ID); {counts['OOD images']:,} (OOD) |"
        )


READOUTS = {"train": print_train, "test": print_test, "detail": print_detail}


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the EM organelle data overview table.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("readout", nargs="?", default="train", choices=list(READOUTS), help="The readout to print.")
    parser.add_argument("-i", "--input", default=DEFAULT_OUTPUT, help="The EM organelle data overview table.")
    args = parser.parse_args()
    READOUTS[args.readout](read_table(args.input))


if __name__ == "__main__":
    main()
