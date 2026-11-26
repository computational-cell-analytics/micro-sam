"""View the partial nuclei of the EM spheroid of the Beghin lab and their candidates from prompts in napari.

With the id of a nucleus: the crop of s1 in which the nucleus was segmented again (reprompt_partial_nuclei.py), with
the raw data, all nuclei ('nuclei'), the nucleus itself ('object'), the candidate of each strategy and its prompt
points. The suggested candidate is visible, the others are hidden. The measurements of the object and of the
candidates are printed.

Without an id: the whole volume at s1, read slice by slice from the zarr arrays, with all nuclei, the suggested
candidates of all partial nuclei ('suggested') and one point per partial nucleus and per fragment of the detection
(detect_partial_nuclei.py) with its id, in the slice of its largest cross-section. The other columns of the table
are the features of the points.

All layers are at s1 and placed in nanometer, the coordinates are those of view_beghin_em_volume.py. The raw is shown
inverted (colormap gray_r), as the models see it.

Example:
    python view_beghin_em_reprompted_nuclei.py 25
    python view_beghin_em_reprompted_nuclei.py
"""
import os
import argparse

import numpy as np
import pandas as pd
import zarr

ZARR_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids/PF_Dish_C_pin3new_Spheroid3.ome.zarr"
PREDICTION_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions"
PARTIAL_NUCLEI_DIR = os.path.join(PREDICTION_DIR, "partial_nuclei")
LEVEL = 1
COLORS = {"partial": "red", "fragment": "dodgerblue"}


def get_transform(root, level):
    """The scale and translation (nanometer, ZYX) of a level of the raw from the OME metadata."""
    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    dataset = next(ds for ds in datasets if ds["path"] == f"s{level}")
    transforms = {tr["type"]: tr for tr in dataset["coordinateTransformations"]}
    scale = tuple(transforms["scale"]["scale"])
    translation = tuple(transforms["translation"]["translation"]) if "translation" in transforms else (0.0,) * 3
    return scale, translation


def add_nucleus_layers(viewer, nucleus_id):
    """Add the layers of the crop of one nucleus that was segmented again, see the description of the script."""
    root = zarr.open_group(ZARR_PATH, mode="r")
    scale, translation = get_transform(root, LEVEL)
    out = zarr.open_group(os.path.join(PARTIAL_NUCLEI_DIR, "reprompted.zarr"), mode="r")
    if str(nucleus_id) not in out["candidates"]:
        ids = sorted(int(name) for name, _ in out["candidates"].groups())
        raise ValueError(f"The nucleus {nucleus_id} was not segmented again. The nuclei are {ids}.")
    group = out["candidates"][str(nucleus_id)]
    offset, suggested = group.attrs["offset"], group.attrs["suggested"]
    bb = tuple(slice(o, o + sh) for o, sh in zip(offset, group.attrs["shape"]))
    placement = {
        "scale": scale, "translate": tuple(tr + o * sc for tr, o, sc in zip(translation, offset, scale)),
        "units": "nm",
    }

    nuclei = zarr.open_group(out.attrs["nuclei"], mode="r")["segmentation"][bb]
    viewer.add_image(root[f"s{LEVEL}"][bb], name="raw", contrast_limits=(0, 255), colormap="gray_r", **placement)
    viewer.add_labels(nuclei, name="nuclei", visible=False, **placement)
    viewer.add_labels((nuclei == nucleus_id).astype("uint8"), name="object", colormap={None: "red"}, **placement)
    colors = ("yellow", "cyan", "lime", "magenta", "orange")
    for (strategy, array), color in zip(sorted(group.arrays()), colors):
        viewer.add_labels(
            array[:].astype("uint8"), name=strategy, visible=strategy == suggested, colormap={None: color},
            opacity=0.5, **placement,
        )
        points = np.array(group.attrs["prompts"][strategy]["points"], dtype="float64")
        viewer.add_points(
            points * scale + translation, name=f"prompts {strategy}", visible=strategy == suggested, size=600,
            face_color=color, border_color="black", units="nm",
        )
    viewer.dims.set_point(0, translation[0] + group.attrs["prompts"][suggested or "center"]["points"][0][0] * scale[0])

    pd.set_option("display.width", 250, "display.max_columns", 40)
    detected = pd.read_csv(os.path.join(PARTIAL_NUCLEI_DIR, "detection.csv"), index_col="id")
    columns = ["volume_um3", "n_slices", "max_change", "n_jumps", "class", "reason"]
    print(f"Nucleus {nucleus_id}:\n{detected.loc[[nucleus_id], columns].round(2).to_string()}")
    candidates = pd.read_csv(os.path.join(PARTIAL_NUCLEI_DIR, "reprompted_candidates.csv"))
    columns = ["strategy", "volume_um3", "n_slices", "max_change", "iou_old", "covers_old", "agreement",
               "fraction_in_other", "absorbs", "at_crop_border"]
    print(f"Candidates (suggested: {suggested}):\n"
          f"{candidates.loc[candidates['id'] == nucleus_id, columns].round(2).to_string(index=False)}")


def add_volume_layers(viewer):
    """Add the layers of the whole volume at s1, see the description of the script."""
    root = zarr.open_group(ZARR_PATH, mode="r")
    scale, translation = get_transform(root, LEVEL)
    placement = {"scale": scale, "translate": translation, "units": "nm"}
    viewer.add_image(root[f"s{LEVEL}"], name="raw", contrast_limits=(0, 255), colormap="gray_r", **placement)
    detected = pd.read_csv(os.path.join(PARTIAL_NUCLEI_DIR, "detection.csv"))
    reprompted_path = os.path.join(PARTIAL_NUCLEI_DIR, "reprompted.zarr")
    nuclei_path = os.path.join(PREDICTION_DIR, f"nuclei_s{LEVEL}_joined.zarr")
    viewer.add_labels(zarr.open_group(nuclei_path, mode="r")["segmentation"], name="nuclei", **placement)
    if os.path.exists(reprompted_path) and "suggested" in zarr.open_group(reprompted_path, mode="r"):
        suggested = zarr.open_group(reprompted_path, mode="r")["suggested"]
        viewer.add_labels(suggested, name="suggested", **placement)
    for name, color in COLORS.items():
        objects = detected[detected["class"] == name]
        centers = np.stack([
            objects["max_area_z"], (objects["y0"] + objects["y1"]) / 2, (objects["x0"] + objects["x1"]) / 2,
        ], axis=1)
        features = objects[["id", "volume_um3", "n_slices", "max_change", "reason"]].reset_index(drop=True)
        viewer.add_points(
            centers * scale + translation, name=name, features=features, size=1500, face_color="transparent",
            border_color=color, border_width=0.15, units="nm", out_of_slice_display=True,
            text={"string": "{id}", "color": color, "size": 10, "translation": (0, -1000, 0)},
        )
        print(f"{name}: {len(objects)} objects")


def main():
    import napari

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nucleus_id", type=int, nargs="?", help="The id of a nucleus that was segmented again.")
    args = parser.parse_args()
    if args.nucleus_id is None:
        viewer = napari.Viewer(title="Partial nuclei")
        add_volume_layers(viewer)
    else:
        viewer = napari.Viewer(title=f"Nucleus {args.nucleus_id}")
        add_nucleus_layers(viewer, args.nucleus_id)
    viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
