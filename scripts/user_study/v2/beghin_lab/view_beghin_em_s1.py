"""View the s1 level of the Beghin lab EM spheroid in napari: the raw data, the 3d decoder outputs of
'explore_beghin_em_data.py predict_volume', the nucleus segmentation of 'watershed_volume' (size filtered; its seeds
and the unfiltered watershed are hidden) and optionally the label images of the OME-Zarr (--labels, e.g.
nuclei_clean, the earlier s0 run, nearest downsampled).
The raw is shown inverted (colormap gray_r), as the decoder sees it.

Without --crop_size the whole level is opened lazily from the zarr arrays. With --crop_size (one value for z, y and x
or three values ZYX) a central crop of that size is loaded into memory.

Example:
    python view_beghin_em_s1.py --crop_size 128 512 512
"""
import os
import argparse

import napari
import numpy as np
import zarr

ZARR_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids/PF_Dish_C_pin3new_Spheroid3.ome.zarr"
VOLUME_PREDICTION_DIR = (
    "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions"
)
PREDICTION_PATH = os.path.join(VOLUME_PREDICTION_DIR, "nuclei_s1_predictions.zarr")
WATERSHED_PATH = os.path.join(VOLUME_PREDICTION_DIR, "nuclei_s1_watershed.zarr")
LEVEL = 1
LABEL_NAMES = ()
# The decoder channels with their contrast limits; the distances are hidden at the start.
PREDICTION_CHANNELS = {
    "foreground": (0.0, 1.0), "boundary": (0.0, 1.0),
    "distance_z": (-1.0, 1.0), "distance_y": (-1.0, 1.0), "distance_x": (-1.0, 1.0),
}


def get_voxel_size(group, path):
    """The ZYX scale of the dataset 'path' from the OME metadata of a multiscale group."""
    datasets = group.attrs["ome"]["multiscales"][0]["datasets"]
    dataset = next(ds for ds in datasets if ds["path"] == path)
    scale = next(tr for tr in dataset["coordinateTransformations"] if tr["type"] == "scale")
    return tuple(scale["scale"])


def get_central_crop(shape, crop_size):
    """The slices of a central crop of crop_size (clipped to the shape), or the whole volume for None."""
    if crop_size is None:
        return tuple(slice(None) for _ in shape)
    if len(crop_size) == 1:
        crop_size = crop_size * len(shape)
    assert len(crop_size) == len(shape), crop_size
    crop_size = [min(cs, sh) for cs, sh in zip(crop_size, shape)]
    starts = [(sh - cs) // 2 for sh, cs in zip(shape, crop_size)]
    return tuple(slice(start, start + cs) for start, cs in zip(starts, crop_size))


def load(array, bb, crop_size):
    """The crop of an array in memory, or the zarr array itself (lazy) for the whole volume."""
    return array if crop_size is None else array[bb]


def view_s1(
    zarr_path, prediction_path, crop_size=None, label_names=LABEL_NAMES, show_predictions=True,
    watershed_path=WATERSHED_PATH,
):
    root = zarr.open_group(zarr_path, mode="r")
    raw = root[f"s{LEVEL}"]
    bb = get_central_crop(raw.shape, crop_size)
    # All layers get the scale only: the raw and the predictions share the voxel grid, the labels of s1 are the first
    # s0 voxel of each block (the half voxel translation of the mean downsampled raw is left out).
    scale = get_voxel_size(root, f"s{LEVEL}")
    if crop_size is not None:
        print("Central crop (ZYX of s1):", ", ".join(f"{b.start}:{b.stop}" for b in bb))

    viewer = napari.Viewer()
    viewer.add_image(
        load(raw, bb, crop_size), name="raw", scale=scale, units="nm", contrast_limits=(0, 255),
        colormap="gray_r",
    )

    if show_predictions:
        predictions = zarr.open_group(prediction_path, mode="r")
        assert predictions.attrs["scale_level"] == LEVEL
        assert tuple(predictions["foreground"].shape) == tuple(raw.shape)
        for name, limits in PREDICTION_CHANNELS.items():
            data = load(predictions[name], bb, crop_size)
            if crop_size is not None:
                data = data.astype("float32")
            viewer.add_image(
                data, name=name, scale=scale, units="nm", contrast_limits=limits, blending="additive",
                colormap="green" if name == "foreground" else "magenta" if name == "boundary" else "gray",
                visible=name == "foreground", opacity=0.5 if name in ("foreground", "boundary") else 1.0,
            )

    if watershed_path is not None and os.path.exists(watershed_path):
        watershed = zarr.open_group(watershed_path, mode="r")
        for name in ("seeds", "watershed", "segmentation"):
            if name not in watershed:
                continue
            data = load(watershed[name], bb, crop_size)
            if crop_size is not None:
                print(f"{name}: {len(np.unique(data)) - 1} objects in the crop")
            viewer.add_labels(data, name=name, scale=scale, units="nm", visible=name == "segmentation")

    for name in label_names:
        labels = root[f"labels/{name}/s{LEVEL}"]
        assert tuple(labels.shape) == tuple(raw.shape), (name, labels.shape)
        data = load(labels, bb, crop_size)
        if crop_size is not None:
            print(f"{name}: {len(np.unique(data)) - 1} objects in the crop")
        viewer.add_labels(data, name=name, scale=scale, units="nm")

    viewer.scale_bar.visible = True
    napari.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop_size", type=int, nargs="+", default=None,
                        help="Size of the central crop in s1 voxels: one value for z, y and x, or three values ZYX.")
    parser.add_argument("--zarr_path", default=ZARR_PATH)
    parser.add_argument("--prediction_path", default=PREDICTION_PATH)
    parser.add_argument("--labels", nargs="*", default=list(LABEL_NAMES),
                        help="Label images of the OME-Zarr to show (e.g. nuclei nuclei_clean), none for no labels.")
    parser.add_argument("--no_predictions", action="store_true", help="Do not show the decoder outputs.")
    parser.add_argument("--watershed_path", default=WATERSHED_PATH)
    parser.add_argument("--no_watershed", action="store_true", help="Do not show the watershed segmentation.")
    args = parser.parse_args()
    assert os.path.exists(args.prediction_path) or args.no_predictions, args.prediction_path
    view_s1(
        args.zarr_path, args.prediction_path, args.crop_size, args.labels, not args.no_predictions,
        None if args.no_watershed else args.watershed_path,
    )


if __name__ == "__main__":
    main()
