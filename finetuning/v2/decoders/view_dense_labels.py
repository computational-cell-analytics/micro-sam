"""Look at the label 0 of the dense datasets in napari: the raw data and the ground truth of every tuning crop.

    python view_dense_labels.py                       # the tuning crops, dataset by dataset
    python view_dense_labels.py --list                # print the crops and their share of label 0, without napari
    python view_dense_labels.py --datasets liconn xpress
    python view_dense_labels.py --split test --max_crops 5   # test crops, e.g. for the datasets without tuning data
    python view_dense_labels.py --ood                 # the test crops of the out-of-domain datasets
    python view_dense_labels.py --ood --max_crops 5 --list

The crops are loaded like the parameter search and the evaluation load them (`common.load_data`), so the labels are
the ones that are scored. The CREMI score ignores label 0 for every dense dataset (`common.cremi_metrics`); before
2026-11-27 it counted label 0 as one object for most of them. The datasets with the largest share of label 0 come
first.

Layers: 'raw', 'labels' (the ground truth), 'label 0' (the voxels with label 0, in red) and, for the partially
annotated datasets, 'outside valid_roi' (where the segmentation is cleared before scoring, hidden).
"Next" / "Back" (keys N / B) move through the crops.
"""

import argparse

import numpy as np

import evaluate_decoder
import tune_set

# The in-domain dense datasets with tuning data, the ones with the most label 0 first.
DATASETS = (
    "liconn", "xpress", "zebrafinch_j0126", "snemi", "hemibrain", "platynereis_cells", "synapsenet_compartments",
    "cremi", "fafb",
)


def ood_datasets(search_common):
    """The out-of-domain datasets of the dense setting, from the figure panels. They only have test data."""
    return [
        dataset for dataset, _, _, domain in evaluate_decoder.panel_datasets()
        if domain == "OOD" and dataset in search_common.DATASETS_DENSE
    ]


def list_crops(search_common, datasets, split, max_crops=None):
    """The crops to show as (dataset, sample index, number of samples of the dataset), in the order of 'datasets'.

    Args:
        search_common: The 'common' module of the evaluation.
        datasets: The datasets to go through.
        split: The split to load, 'val' (the tuning crops) or 'test'.
        max_crops: Show at most this many crops per dataset, spread evenly over its samples.

    Returns:
        The list of crops.
    """
    crops = []
    for dataset in datasets:
        if split == "val" and not search_common.has_val_split(dataset):
            print(f"'{dataset}' has no tuning data, skipping it (try --split test).")
            continue
        n_samples = search_common.n_samples(dataset, search_common.DATA_ROOT, split=split)
        ids = range(n_samples)
        if max_crops is not None and n_samples > max_crops:
            ids = sorted(set(np.linspace(0, n_samples - 1, max_crops).round().astype(int).tolist()))
        crops += [(dataset, index, n_samples) for index in ids]
    return crops


def load_crop(search, search_common, dataset, index, split):
    """Load one crop of a dataset as the search (split 'val') or the evaluation (split 'test') scores it."""
    # The search passes its crop shape; the evaluation uses the fixed test crops or its default center crop.
    crop_shape = search.PIPELINE_CROP_3D if split == "val" else None
    samples = search_common.load_data(
        dataset, search_common.DATA_ROOT, 3, split=split, crop_shape=crop_shape,
        z_range=search_common.VAL_Z_RANGE.get(dataset) if split == "val" else None, sample_ids={index},
    )
    return next(iter(samples))


def crop_statistics(search_common, dataset, labels, valid_roi):
    """The numbers shown for a crop: its share of label 0, in total and inside the annotated region."""
    zero = labels == 0
    return {
        "shape": "x".join(map(str, labels.shape)), "n_objects": int(len(np.setdiff1d(np.unique(labels), [0]))),
        "label_0": float(zero.mean()),
        "label_0_in_valid_roi": float(zero[valid_roi].mean()) if valid_roi is not None else float(zero.mean()),
        "has_valid_roi": valid_roi is not None,
        "zero_ignored_by_score": dataset in search_common.DATASETS_DENSE,
    }


def describe(dataset, index, n_samples, split, statistics):
    scoring = "ignored by the CREMI score" if statistics["zero_ignored_by_score"] else "scored as one object"
    return (
        f"{dataset} | {split} crop {index + 1}/{n_samples} | label 0: {statistics['label_0']:.1%} ({scoring}) | "
        f"{statistics['n_objects']} objects | {statistics['shape']}"
    )


class CropBrowser:
    """Shows the crops one after the other.

    Args:
        viewer: The napari viewer.
        crops: The crops from `list_crops`.
        split: The split the crops are loaded from.
        set_info: A function that takes the description of the shown crop, e.g. to show it in a widget.
    """

    def __init__(self, viewer, crops, split, set_info=None):
        self.viewer, self.crops, self.split, self.set_info = viewer, crops, split, set_info
        self.search, self.search_common = tune_set.import_search()
        self.position = 0
        self.show()

    def show(self):
        """Load and show the crop at the current position."""
        dataset, index, n_samples = self.crops[self.position]
        raw, labels, valid_roi = load_crop(self.search, self.search_common, dataset, index, self.split)
        statistics = crop_statistics(self.search_common, dataset, labels, valid_roi)

        self.viewer.layers.clear()
        self.viewer.add_image(raw, name="raw")
        if valid_roi is not None:
            self.viewer.add_image(
                (~valid_roi).astype("uint8"), name="outside valid_roi", colormap="blue", blending="additive",
                opacity=0.4, contrast_limits=(0, 1), visible=False,
            )
        self.viewer.add_labels(labels, name="labels", opacity=0.5)
        self.viewer.add_image(
            (labels == 0).astype("uint8"), name="label 0", colormap="red", blending="additive", opacity=0.5,
            contrast_limits=(0, 1),
        )
        self.viewer.dims.set_point(0, labels.shape[0] // 2)

        text = describe(dataset, index, n_samples, self.split, statistics)
        self.viewer.title = text
        if statistics["has_valid_roi"]:
            text += f" | label 0 inside valid_roi: {statistics['label_0_in_valid_roi']:.1%}"
        text += f" | crop {self.position + 1}/{len(self.crops)} of this run"
        print(text, flush=True)
        if self.set_info is not None:
            self.set_info(text.replace(" | ", "\n") + "\nKeys: N next, B back")

    def next(self):
        """Go to the next crop."""
        if self.position < len(self.crops) - 1:
            self.position += 1
            self.show()

    def back(self):
        """Go to the crop before."""
        if self.position > 0:
            self.position -= 1
            self.show()


def run_viewer(crops, split):
    import napari
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

    viewer = napari.Viewer()
    info = QLabel()
    info.setWordWrap(True)
    browser = CropBrowser(viewer, crops, split, set_info=info.setText)
    buttons = QHBoxLayout()
    for name, callback in (("Back", browser.back), ("Next", browser.next)):
        button = QPushButton(name)
        button.clicked.connect(callback)
        buttons.addWidget(button)
    layout = QVBoxLayout()
    layout.addWidget(info)
    layout.addLayout(buttons)
    widget = QWidget()
    widget.setLayout(layout)
    viewer.window.add_dock_widget(widget, area="right", name="Dense labels")
    viewer.bind_key("n", lambda _: browser.next())
    viewer.bind_key("b", lambda _: browser.back())
    napari.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", help="Any dataset of the dense setting. Default: DATASETS.")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="'val' are the tuning crops.")
    parser.add_argument(
        "--ood", action="store_true", help="Show the test crops of the out-of-domain datasets of the dense setting.",
    )
    parser.add_argument("--max_crops", type=int, help="At most this many crops per dataset, spread evenly.")
    parser.add_argument("--list", action="store_true", help="Print the crops and their share of label 0 and exit.")
    args = parser.parse_args()

    search, search_common = tune_set.import_search()
    if args.ood:
        if args.datasets is not None:
            parser.error("Pass either --ood or --datasets.")
        args.datasets, args.split = ood_datasets(search_common), "test"
    elif args.datasets is None:
        args.datasets = list(DATASETS)
    unknown = [dataset for dataset in args.datasets if dataset not in search_common.DATASETS_DENSE]
    if unknown:
        parser.error(f"Not datasets of the dense setting: {unknown}.")
    crops = list_crops(search_common, args.datasets, args.split, args.max_crops)
    if not crops:
        parser.error("No crop to show.")
    if args.list:
        for dataset, index, n_samples in crops:
            _, labels, valid_roi = load_crop(search, search_common, dataset, index, args.split)
            print(describe(dataset, index, n_samples, args.split,
                           crop_statistics(search_common, dataset, labels, valid_roi)), flush=True)
        return
    run_viewer(crops, args.split)


if __name__ == "__main__":
    main()
