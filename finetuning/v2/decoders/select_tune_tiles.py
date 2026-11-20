"""Pick the tuning tiles of the low-scoring 3d datasets by hand, in napari.

    python select_tune_tiles.py                 # go through the tiles that have no decision yet
    python select_tune_tiles.py --list          # print the tiles and their decisions, without napari
    python select_tune_tiles.py --datasets cartocell plantseg_root

The candidates come from 'tune_set.py report' (candidates_scored.csv). Per dataset they are filtered (at least 16
slices and 10 objects; the object filter is skipped for the datasets in NO_OBJECT_FILTER, which have too few such
tiles) and sorted by the best Hiera-U score of the grid, highest first. The center crops are left out: they
overlap the tiles and cannot be placed in the tune set.

Each tile is shown with its raw data and labels; "Include" / "Exclude" (keys I / E) store the decision in
tile_selection.csv and move on, "Back" (key B) takes the last decision back. A restart continues with the first
tile that has no decision, so the table is the only state.
"""

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import tune_set

MIN_SLICES = 16
MIN_OBJECTS = 10
# Datasets that would keep (almost) no tile with the object filter.
NO_OBJECT_FILTER = ("synapsenet_compartments",)
SELECTION_COLUMNS = [
    "key", "dataset", "volume", "start", "decision", "n_objects", "hiera_u_best", "hiera_u_defaults",
    "hvit_t_cells_best", "hvit_t_cells_defaults", "time",
]


def candidate_tiles(scored_path, datasets=tune_set.LOW_SCORING, ascending=False):
    """The tiles to decide on: filtered, in the order of 'datasets' and sorted by the best Hiera-U score.

    Args:
        scored_path: The table written by 'tune_set.py report'.
        datasets: The datasets to go through, in this order.
        ascending: Whether to show the lowest scoring tiles first.

    Returns:
        The table of tiles.
    """
    table = pd.read_csv(scored_path)
    table = table[(table["start"] != "center") & table["hiera_u_best"].notna()]
    n_slices = table["shape"].str.split("x").str[0].astype(int)
    enough_objects = (table["n_objects"] >= MIN_OBJECTS) | table["dataset"].isin(NO_OBJECT_FILTER)
    table = table[(n_slices >= MIN_SLICES) & enough_objects]
    tiles = [
        table[table["dataset"] == dataset].sort_values(["hiera_u_best", "key"], ascending=[ascending, True])
        for dataset in datasets
    ]
    return pd.concat(tiles, ignore_index=True)


def load_selection(path):
    """The decisions made so far (empty if there are none)."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=SELECTION_COLUMNS)


def save_selection(selection, path):
    """Write the decisions, replacing the file in one step so that an interrupted write cannot damage it."""
    selection.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def _channel_axis(raw):
    if raw.ndim == 3:
        return None
    return int(np.argmin(raw.shape))


class TileSelector:
    """Shows the tiles one after the other and records the decisions.

    Args:
        tiles: The table of tiles from 'candidate_tiles'.
        selection_path: The table the decisions are stored in.
        show_prediction: Whether to add the foreground prediction of Hiera-U as a hidden layer.
    """

    def __init__(self, tiles, selection_path, show_prediction=True):
        import napari
        from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

        self.tiles = tiles
        self.selection_path = selection_path
        self.show_prediction = show_prediction
        self.selection = load_selection(selection_path)
        self.search, self.search_common = tune_set.import_search()
        self._regions = {}
        self.current = None

        self.viewer = napari.Viewer()
        self.info = QLabel()
        self.info.setWordWrap(True)
        buttons = QHBoxLayout()
        for name, callback in (("Include", self.include), ("Exclude", self.exclude), ("Back", self.back)):
            button = QPushButton(name)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout = QVBoxLayout()
        layout.addWidget(self.info)
        layout.addLayout(buttons)
        widget = QWidget()
        widget.setLayout(layout)
        self.viewer.window.add_dock_widget(widget, area="right", name="Tile selection")
        self.viewer.bind_key("i", lambda _: self.include())
        self.viewer.bind_key("e", lambda _: self.exclude())
        self.viewer.bind_key("b", lambda _: self.back())
        self.show_next()

    def _load(self, row):
        if row.dataset not in self._regions:
            self._regions[row.dataset] = {
                volume: (paths, z_range) for volume, paths, z_range, _ in self.search.tile_regions(
                    row.dataset, self.search_common.DATA_ROOT, 3
                )
            }
        paths, z_range = self._regions[row.dataset][row.volume]
        return self.search_common.load_evaluation_sample_3d(
            *paths, crop_shape=self.search.TUNING_CROP_3D, z_range=z_range, split="val",
            crop_start=tune_set._start(row.start),
        )

    def _pending(self):
        return self.tiles[~self.tiles["key"].isin(self.selection["key"])]

    def show_next(self):
        """Show the first tile without a decision."""
        pending = self._pending()
        self.viewer.layers.clear()
        if len(pending) == 0:
            self.current = None
            counts = self.selection[self.selection["key"].isin(self.tiles["key"])]["decision"].value_counts()
            text = f"All {len(self.tiles)} tiles are done: {counts.to_dict()}. Table: {self.selection_path}"
            self.viewer.title = "Tile selection: done"
            self.info.setText(text)
            print(text)
            return
        row = next(pending.itertuples())
        self.current = row
        raw, labels, valid_roi = self._load(row)

        self.viewer.add_image(raw, name="raw", channel_axis=_channel_axis(raw))
        if valid_roi is not None:
            self.viewer.add_labels((~valid_roi).astype("uint8"), name="not annotated", opacity=0.3, visible=False)
        prediction_path = os.path.join(
            tune_set.OUT_ROOT, "predictions", "hiera_u", row.key.replace("|", "__") + ".npy"
        )
        if self.show_prediction and os.path.exists(prediction_path):
            foreground = np.load(prediction_path, mmap_mode="r")[0].astype("float32")
            self.viewer.add_image(
                foreground, name="hiera_u foreground", colormap="magenta", blending="additive", visible=False
            )
        self.viewer.add_labels(labels, name="labels")
        self.viewer.dims.set_point(0, labels.shape[0] // 2)

        in_dataset = self.tiles[self.tiles["dataset"] == row.dataset]
        position = int((in_dataset["key"] == row.key).to_numpy().nonzero()[0][0]) + 1
        done = len(self.tiles) - len(pending)
        title = (
            f"{row.dataset} [{position}/{len(in_dataset)}] | objects: {row.n_objects} | "
            f"Hiera-U best {row.hiera_u_best:.3f}, defaults {row.hiera_u_defaults:.3f} | "
            f"hvit_t best {row.hvit_t_cells_best:.3f}, defaults {row.hvit_t_cells_defaults:.3f}"
        )
        self.viewer.title = title
        self.info.setText(
            title.replace(" | ", "\n") + f"\n{row.volume}, start {row.start}, shape {row.shape}\n"
            f"median object size {row.median_size:.0f}, cut by the tile {row.cut_fraction:.2f}, "
            f"annotated {row.annotated_fraction:.2f}\n{done}/{len(self.tiles)} tiles done\n"
            "Keys: I include, E exclude, B back"
        )

    def _decide(self, decision):
        row = self.current
        if row is None:
            return
        entry = {column: getattr(row, column) for column in SELECTION_COLUMNS if hasattr(row, column)}
        entry.update(decision=decision, time=datetime.now().isoformat(timespec="seconds"))
        new = pd.DataFrame([entry], columns=SELECTION_COLUMNS)
        self.selection = new if len(self.selection) == 0 else pd.concat([self.selection, new], ignore_index=True)
        save_selection(self.selection, self.selection_path)
        print(f"{decision}: {row.key}", flush=True)
        self.show_next()

    def include(self):
        """Include the shown tile and go to the next one."""
        self._decide("include")

    def exclude(self):
        """Exclude the shown tile and go to the next one."""
        self._decide("exclude")

    def back(self):
        """Take back the last decision on a tile of this run's datasets and show that tile again."""
        mine = self.selection[self.selection["key"].isin(self.tiles["key"])]
        if len(mine) == 0:
            return
        self.selection = self.selection.drop(index=mine.index[-1]).reset_index(drop=True)
        save_selection(self.selection, self.selection_path)
        self.show_next()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(tune_set.LOW_SCORING))
    parser.add_argument("--scored", default=os.path.join(tune_set.OUT_ROOT, "candidates_scored.csv"))
    parser.add_argument("--output", default=os.path.join(tune_set.OUT_ROOT, "tile_selection.csv"))
    parser.add_argument("--ascending", action="store_true", help="Show the lowest scoring tiles first.")
    parser.add_argument("--no_prediction", action="store_true", help="Do not add the foreground prediction layer.")
    parser.add_argument("--list", action="store_true", help="Print the tiles and their decisions and exit.")
    args = parser.parse_args()

    tiles = candidate_tiles(args.scored, args.datasets, args.ascending)
    if args.list:
        decisions = load_selection(args.output).set_index("key")["decision"]
        tiles = tiles.assign(decision=tiles["key"].map(decisions).fillna(""))
        columns = ["dataset", "volume", "start", "shape", "n_objects", "hiera_u_best", "hiera_u_defaults", "decision"]
        print(tiles[columns].to_string(index=False))
        print(tiles.groupby("dataset", sort=False)["decision"].value_counts().unstack(fill_value=0))
        return

    import napari

    TileSelector(tiles, args.output, show_prediction=not args.no_prediction)
    napari.run()


if __name__ == "__main__":
    main()
