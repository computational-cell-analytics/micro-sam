"""View the 3d AIS test on the inverted raw of the EM spheroid (explore_beghin_em_data.py segment_inverted_crop) in
napari, at s1.

Layers (all on the voxel grid of the s1 crop, 256 x 512 x 512):
- raw: the inverted s1 raw (255 - raw) the s1 run saw.
- s1 inverted: foreground, boundary (hidden) and the AIS segmentation.
- s0 inverted: the same of the s0 run, reduced to s1 by taking every second voxel in y and x.
- s1 volume: the foreground of the whole s1 volume (predict_volume, also inverted), in the same crop.
- nuclei_clean: the 3d APG nuclei of the s0 volume (non-inverted), at s1.
The fraction of the nuclei_clean voxels with foreground above 0.5 and the number of AIS objects are printed.
"""
import os

import h5py
import napari
import numpy as np
import zarr

ZARR_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids/PF_Dish_C_pin3new_Spheroid3.ome.zarr"
PREDICTION_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions"
CROP_DIR = os.path.join(PREDICTION_DIR, "inverted_crop")
VOLUME_PREDICTION_PATH = os.path.join(PREDICTION_DIR, "nuclei_s1_predictions.zarr")


def load_run(level):
    """The foreground, boundary and segmentation of a run, reduced to the s1 grid, and the bounding box."""
    path = os.path.join(CROP_DIR, f"nuclei_s{level}.h5")
    if not os.path.exists(path):
        print(f"No results for s{level} yet ({path})")
        return None
    step = 2 if level == 0 else 1
    with h5py.File(path, "r") as f:
        run = {
            "foreground": f["prediction/foreground"][:, ::step, ::step].astype("float32"),
            "boundary": f["prediction/boundary"][:, ::step, ::step].astype("float32"),
            "segmentation": f["segmentation"][:, ::step, ::step],
            "bb": tuple(slice(start, stop) for start, stop in zip(f.attrs["bb_start"], f.attrs["bb_stop"])),
        }
        if level == 1:
            run["raw"] = f["raw"][:]
    return run


def main():
    root = zarr.open_group(ZARR_PATH, mode="r")
    runs = {level: load_run(level) for level in (1, 0)}
    runs = {level: run for level, run in runs.items() if run is not None}
    assert 1 in runs, "The s1 run is needed for the crop."
    bb = runs[1]["bb"]
    if 0 in runs:
        bb0 = runs[0]["bb"]
        assert bb0[0] == bb[0] and all(b0.start == 2 * b.start for b0, b in zip(bb0[1:], bb[1:])), (bb0, bb)
    print("s1 crop (ZYX):", ", ".join(f"{b.start}:{b.stop}" for b in bb))

    scale = tuple(root.attrs["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]["scale"])
    labels = root["labels/nuclei_clean/s1"][bb]
    volume = zarr.open_array(os.path.join(VOLUME_PREDICTION_PATH, "foreground"), mode="r")[bb].astype("float32")

    mask = labels > 0
    print(f"nuclei_clean: {len(np.unique(labels)) - 1} objects, {mask.mean():.3f} of the crop")
    print(f"s1 volume: {(volume[mask] > 0.5).mean():.3f} of the nuclei_clean voxels with foreground > 0.5, "
          f"{(volume[~mask] > 0.5).mean():.3f} of the rest")
    for level, run in runs.items():
        fg = run["foreground"]
        n_objects = len(np.unique(run["segmentation"])) - 1
        print(f"s{level} inverted: {(fg[mask] > 0.5).mean():.3f} of the nuclei_clean voxels with foreground > 0.5, "
              f"{(fg[~mask] > 0.5).mean():.3f} of the rest, {n_objects} AIS objects")

    viewer = napari.Viewer()
    viewer.add_image(runs[1]["raw"], name="raw (inverted)", scale=scale, units="nm", contrast_limits=(0, 255))
    viewer.add_image(
        volume, name="s1 volume foreground", scale=scale, units="nm", contrast_limits=(0, 1), colormap="magenta",
        blending="additive", opacity=0.5, visible=False,
    )
    for level, run in runs.items():
        viewer.add_image(
            run["foreground"], name=f"s{level} inverted foreground", scale=scale, units="nm", contrast_limits=(0, 1),
            colormap="green", blending="additive", opacity=0.5, visible=level == 1,
        )
        viewer.add_image(
            run["boundary"], name=f"s{level} inverted boundary", scale=scale, units="nm", contrast_limits=(0, 1),
            colormap="magenta", blending="additive", opacity=0.5, visible=False,
        )
        viewer.add_labels(
            run["segmentation"], name=f"s{level} inverted AIS", scale=scale, units="nm", visible=level == 1,
        )
    viewer.add_labels(labels, name="nuclei_clean", scale=scale, units="nm", visible=False)
    viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
