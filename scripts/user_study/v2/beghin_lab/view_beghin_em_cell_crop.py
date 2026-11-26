"""View the cells of the EM spheroid crop (explore_beghin_em_data.py watershed_cell_crop) in napari, at s0.

Layers (all on the voxel grid of the s0 crop, 128 x 1024 x 1024):
- raw (inverted): the inverted s0 raw (255 - raw) the cells model saw.
- boundary: the boundary prediction of the cells model, the height map of the watershed.
- foreground mask (closed): the mask of the watershed, foreground > threshold closed with a ball of closing_radius.
- foreground mask: the mask before the closing (hidden).
- nuclei: the nuclei that seed the cells, from s1 (hidden).
- cells: the watershed of the expanded nuclei, a cell has the id of its nucleus.
- cells (no closing): the same watershed with the mask before the closing (seed_cell_crop, hidden).
"""
import os

import h5py
import napari
import numpy as np
import zarr

ZARR_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids/PF_Dish_C_pin3new_Spheroid3.ome.zarr"
PREDICTION_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions"
CELL_CROP_DIR = os.path.join(PREDICTION_DIR, "cell_crop")
LEVEL = 0
WATERSHED_PATH = os.path.join(CELL_CROP_DIR, f"cells_watershed_s{LEVEL}.h5")
PREDICTION_PATH = os.path.join(CELL_CROP_DIR, f"cells_ais_s{LEVEL}.h5")
NO_CLOSING_PATH = os.path.join(CELL_CROP_DIR, f"cells_seeded_s{LEVEL}.h5")
NUCLEI_PATH = os.path.join(CELL_CROP_DIR, "nucleus_prompts.h5")


def get_bb(f):
    return tuple(slice(start, stop) for start, stop in zip(f.attrs["bb_start"], f.attrs["bb_stop"]))


def main():
    root = zarr.open_group(ZARR_PATH, mode="r")
    with h5py.File(WATERSHED_PATH, "r") as f:
        bb = get_bb(f)
        cells, mask = f["segmentation"][:], f["mask"][:]
        threshold, closing_radius = f.attrs["foreground_threshold"], f.attrs["closing_radius"]
    with h5py.File(PREDICTION_PATH, "r") as f:
        assert get_bb(f) == bb, (get_bb(f), bb)
        boundary = f["prediction/boundary"][:].astype("float32")
        mask_open = (f["prediction/foreground"][:] > threshold).astype("uint8")
    # The nuclei are at s1, the first voxel of each 2 x 2 block of s0 in y and x.
    with h5py.File(NUCLEI_PATH, "r") as f:
        nuclei = np.repeat(np.repeat(f["nuclei"][:], 2, axis=1), 2, axis=2)
    no_closing = None
    if os.path.exists(NO_CLOSING_PATH):
        with h5py.File(NO_CLOSING_PATH, "r") as f:
            no_closing = f["watershed"][:]
    print(f"s{LEVEL} crop (ZYX):", ", ".join(f"{b.start}:{b.stop}" for b in bb))
    print(f"{len(np.unique(cells)) - 1} cells, closing radius {closing_radius} nm, the closing added "
          f"{(mask.astype(bool) & ~mask_open.astype(bool)).mean():.3f} of the crop to the mask")

    scale = tuple(root.attrs["ome"]["multiscales"][0]["datasets"][LEVEL]["coordinateTransformations"][0]["scale"])
    viewer = napari.Viewer()
    viewer.add_image(
        255 - root[f"s{LEVEL}"][bb], name="raw (inverted)", scale=scale, units="nm", contrast_limits=(0, 255),
    )
    viewer.add_image(
        boundary, name="boundary", scale=scale, units="nm", contrast_limits=(0, 1), colormap="magenta",
        blending="additive", opacity=0.5, visible=False,
    )
    viewer.add_image(
        mask, name=f"foreground mask (closed, {closing_radius:.0f} nm)", scale=scale, units="nm",
        contrast_limits=(0, 1), colormap="green", blending="additive", opacity=0.4, visible=False,
    )
    viewer.add_image(
        mask_open, name="foreground mask", scale=scale, units="nm", contrast_limits=(0, 1), colormap="green",
        blending="additive", opacity=0.4, visible=False,
    )
    viewer.add_labels(nuclei, name="nuclei", scale=scale, units="nm", visible=False)
    if no_closing is not None:
        viewer.add_labels(no_closing, name="cells (no closing)", scale=scale, units="nm", visible=False)
    viewer.add_labels(cells, name="cells", scale=scale, units="nm")
    viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
