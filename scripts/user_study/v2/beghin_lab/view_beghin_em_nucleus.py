"""View the surroundings of one nucleus of the EM spheroid of the Beghin lab in napari: the raw data, the foreground
and boundary predictions of the nuclei and cells models (explore_beghin_em_data.py predict_volume) and the nuclei
(watershed_volume) and cells (segment_cells_volume).

The nucleus is given by its id in the nucleus segmentation, which is also the id of its cell and the id shown by
view_beghin_em_volume.py at every level. The crop is centered on the bounding box of the nucleus and given in voxels
of s1 (shifted to stay inside the volume). The nuclei segmented from point prompts (segment_nuclei_from_prompts) are
shown as well if they exist, and their ids can be given too. So are the cells with the background seed on the slabs
of check_border_seed (hidden, empty outside of these slabs).
If the merged nuclei (merge_prompted_nuclei: without the fragments of the prompted nuclei, with the prompted nuclei)
and the cells from them with the background seed ('segment_cells_volume --border_seed --merged_nuclei') exist, they
are the visible segmentations ('nuclei merged', 'cells merged') and the previous ones are hidden. The prompted nuclei
that are only seeds for their cell, not full nuclei, are printed.
The same holds for the joined nuclei (join_nucleus_fragments: the fragments of a nucleus as one object) and their
cells ('segment_cells_volume --border_seed --nuclei joined'), which replace the merged ones as visible layers
('nuclei joined', 'cells joined'). A fragment keeps its id in 'nuclei' and has the lowest id of its group there.

Every layer is loaded at the scale it was computed at and placed in nanometer with the scale and translation of that
level, so the view is the s1 crop and nothing is resampled: the raw, the predictions of the nuclei and the nuclei are
at s1, the predictions of the cells and the cells at s0 (twice the resolution of s1 in y and x). The coordinates are
those of view_beghin_em_volume.py. The raw is shown inverted (colormap gray_r), as the models see it.

With --intermediates the steps of the two segmentations are added as hidden layers: the seeds of the nuclei
(foreground - boundary > threshold), their watershed before the size filter, the seeds of the cells (the expanded
nuclei) and the closed foreground mask of the cells. The ids of 'nuclei seeds' and 'nuclei watershed' are not the ids
of the nuclei (the size filter relabels them).

Example:
    python view_beghin_em_nucleus.py 42
    python view_beghin_em_nucleus.py 42 --crop_size 128 512 512 --intermediates
"""
import os
import argparse

import napari
import numpy as np
import zarr

ZARR_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids/PF_Dish_C_pin3new_Spheroid3.ome.zarr"
PREDICTION_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions"
PYRAMID_PATH = os.path.join(PREDICTION_DIR, "segmentation_pyramids.zarr")
# The level the view is defined at (raw and crop) and, per target, the level it was predicted and segmented at.
VIEW_LEVEL = 1
LEVELS = {"nuclei": 1, "cells": 0}
PREDICTION_PATHS = {
    target: os.path.join(PREDICTION_DIR, f"{target}_s{level}_predictions.zarr") for target, level in LEVELS.items()
}
NUCLEI_PATH = os.path.join(PREDICTION_DIR, f"nuclei_s{LEVELS['nuclei']}_watershed.zarr")
PROMPTED_PATH = os.path.join(PREDICTION_DIR, f"nuclei_s{LEVELS['nuclei']}_prompted.zarr")
BORDER_SEED_PATH = os.path.join(PREDICTION_DIR, f"cells_s{LEVELS['cells']}_border_seed_check.zarr")
CELLS_PATH = os.path.join(PREDICTION_DIR, f"cells_s{LEVELS['cells']}_segmentation.zarr")
# The later versions of the nuclei, each with the cells from it (with the background seed), in their order.
VERSIONS = ("merged", "joined")
VERSION_PATHS = {
    version: (
        os.path.join(PREDICTION_DIR, f"nuclei_s{LEVELS['nuclei']}_{version}.zarr"),
        os.path.join(PREDICTION_DIR, f"cells_s{LEVELS['cells']}_segmentation_{version}_nuclei_border_seed.zarr"),
    ) for version in VERSIONS
}
CROP_SIZE = (128, 256, 256)
# The colormaps of the predictions: target -> (foreground, boundary).
COLORMAPS = {"nuclei": ("green", "magenta"), "cells": ("cyan", "yellow")}


def get_transform(root, level):
    """The scale and translation (nanometer, ZYX) of a level of the raw from the OME metadata."""
    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    dataset = next(ds for ds in datasets if ds["path"] == f"s{level}")
    transforms = {tr["type"]: tr for tr in dataset["coordinateTransformations"]}
    scale = tuple(transforms["scale"]["scale"])
    translation = tuple(transforms["translation"]["translation"]) if "translation" in transforms else (0.0,) * 3
    return scale, translation


def get_factor(root, level, reference_level):
    """The integer ZYX factor between the voxels of a level and of a finer reference level."""
    scale, reference = get_transform(root, level)[0], get_transform(root, reference_level)[0]
    factor = tuple(int(round(s / r)) for s, r in zip(scale, reference))
    assert all(abs(f * r - s) < 1e-6 for f, r, s in zip(factor, reference, scale)), (scale, reference)
    return factor


def find_nucleus(root, nucleus_id):
    """The bounding box (slices, ZYX in voxels of s1) and the number of voxels of a nucleus.

    The nucleus is located in the pyramid of the nuclei or of the prompted nuclei (the coarsest level that holds it,
    the voxel i of a level is the voxel factor * i of s1) and its bounding box is then taken from s1 within that box,
    grown by one voxel of the pyramid level.
    """
    level = LEVELS["nuclei"]
    pyramids = zarr.open_group(PYRAMID_PATH, mode="r")
    sources = {"nuclei": NUCLEI_PATH, "nuclei_prompted": PROMPTED_PATH}
    all_ids = []
    for name, path in sources.items():
        if name not in pyramids:
            continue
        pyramid = pyramids[name]
        assert pyramid.attrs["source_level"] == level, pyramid.attrs["source_level"]
        pyramid_levels = sorted((int(key[1:]) for key, _ in pyramid.arrays()), reverse=True)
        for pyramid_level in pyramid_levels:
            coords = np.nonzero(pyramid[f"s{pyramid_level}"][:] == nucleus_id)
            if len(coords[0]) > 0:
                break
        if len(coords[0]) > 0:
            break
        ids = np.unique(pyramid[f"s{pyramid_levels[-1]}"][:])
        all_ids.extend(ids[ids > 0].tolist())
    else:
        raise ValueError(f"There is no nucleus with the id {nucleus_id}. The ids go from {min(all_ids)} to "
                         f"{max(all_ids)} ({len(all_ids)} nuclei).")
    nuclei = zarr.open_array(os.path.join(path, "segmentation"), mode="r")
    factor = get_factor(root, pyramid_level, level)
    search = tuple(
        slice(max((int(c.min()) - 1) * f, 0), min((int(c.max()) + 2) * f, sh))
        for c, f, sh in zip(coords, factor, nuclei.shape)
    )
    coords = np.nonzero(nuclei[search] == nucleus_id)
    assert len(coords[0]) > 0, f"The nucleus {nucleus_id} is at s{pyramid_level} of the pyramid but not at s{level}."
    bb = tuple(slice(sl.start + int(c.min()), sl.start + int(c.max()) + 1) for sl, c in zip(search, coords))
    return bb, len(coords[0])


def get_crop(center, crop_size, shape):
    """The slices of a crop of crop_size around a center, shifted to stay inside the shape (clipped if larger)."""
    crop_size = [min(cs, sh) for cs, sh in zip(crop_size, shape)]
    starts = [min(max(ce - cs // 2, 0), sh - cs) for ce, cs, sh in zip(center, crop_size, shape)]
    return tuple(slice(start, start + cs) for start, cs in zip(starts, crop_size))


def scale_crop(root, bb, level):
    """The crop of the view level in the voxels of a finer or equal level."""
    factor = get_factor(root, VIEW_LEVEL, level)
    return tuple(slice(sl.start * f, sl.stop * f) for sl, f in zip(bb, factor))


def get_placement(root, bb, level):
    """The napari keywords that place the crop bb of a level in nanometer."""
    scale, translation = get_transform(root, level)
    translate = tuple(tr + sl.start * sc for tr, sl, sc in zip(translation, bb, scale))
    return {"scale": scale, "translate": translate, "units": "nm"}


def view_nucleus(nucleus_id, crop_size=CROP_SIZE, intermediates=False):
    """Build the viewer for the crop around a nucleus.

    Args:
        nucleus_id: The id of the nucleus in the nucleus segmentation.
        crop_size: The ZYX size of the crop in voxels of s1.
        intermediates: Whether to add the seeds, the unfiltered watershed and the mask of the segmentations.

    Returns:
        The napari viewer.
    """
    root = zarr.open_group(ZARR_PATH, mode="r")
    raw = root[f"s{VIEW_LEVEL}"]
    nucleus_bb, nucleus_size = find_nucleus(root, nucleus_id)
    center = [(sl.start + sl.stop) // 2 for sl in nucleus_bb]
    bb = get_crop(center, crop_size, raw.shape)
    crops = {target: scale_crop(root, bb, level) for target, level in LEVELS.items()}
    placements = {target: get_placement(root, crops[target], level) for target, level in LEVELS.items()}

    def format_bb(box):
        return ", ".join(f"{sl.start}:{sl.stop}" for sl in box)

    print(f"Nucleus {nucleus_id}: {nucleus_size} voxels, bounding box (ZYX of s{LEVELS['nuclei']}) "
          f"{format_bb(nucleus_bb)}")
    print(f"Crop (ZYX of s{VIEW_LEVEL}): {format_bb(bb)}")
    for target, level in LEVELS.items():
        if level != VIEW_LEVEL:
            print(f"Crop of the {target} (ZYX of s{level}): {format_bb(crops[target])}")

    viewer = napari.Viewer(title=f"Nucleus {nucleus_id}")
    viewer.add_image(
        raw[bb], name=f"raw s{VIEW_LEVEL}", contrast_limits=(0, 255), colormap="gray_r",
        **get_placement(root, bb, VIEW_LEVEL),
    )

    for target, level in LEVELS.items():
        predictions = zarr.open_group(PREDICTION_PATHS[target], mode="r")
        assert predictions.attrs["scale_level"] == level, (target, predictions.attrs["scale_level"])
        for name, colormap in zip(("foreground", "boundary"), COLORMAPS[target]):
            viewer.add_image(
                predictions[name][crops[target]].astype("float32"), name=f"{target} {name} s{level}",
                contrast_limits=(0, 1), colormap=colormap, blending="additive", opacity=0.5, visible=False,
                **placements[target],
            )

    nuclei_group, cells_group = zarr.open_group(NUCLEI_PATH, mode="r"), zarr.open_group(CELLS_PATH, mode="r")
    labels = []
    if intermediates:
        labels.extend([
            ("nuclei seeds", "nuclei", nuclei_group["seeds"]),
            ("nuclei watershed (no size filter)", "nuclei", nuclei_group["watershed"]),
            ("cells seeds", "cells", cells_group["seeds"]),
            ("cells mask", "cells", cells_group["mask"]),
        ])
    if os.path.exists(BORDER_SEED_PATH):
        labels.append(("cells border seed", "cells", zarr.open_group(BORDER_SEED_PATH, mode="r")["segmentation"]))
    labels.extend([("cells", "cells", cells_group["segmentation"]), ("nuclei", "nuclei", nuclei_group["segmentation"])])
    if os.path.exists(PROMPTED_PATH):
        labels.append(("nuclei prompted", "nuclei", zarr.open_group(PROMPTED_PATH, mode="r")["segmentation"]))
    segmentations = ["cells", "nuclei", "nuclei prompted"]
    visible = segmentations
    # The latest version of the nuclei and its cells replace the previous segmentations, which stay as hidden layers.
    comments = []
    for version in VERSIONS:
        nuclei_path, cells_path = VERSION_PATHS[version]
        if not os.path.exists(nuclei_path):
            continue
        version_group = zarr.open_group(nuclei_path, mode="r")
        labels.append((f"nuclei {version}", "nuclei", version_group["segmentation"]))
        objects = version_group.attrs["objects"]
        comments = [f"Nucleus {obj['id']}: {obj['comment']}" for obj in objects if obj["seed_only"]]
        for obj in version_group.attrs.get("joined", []):
            comments.append(f"Nucleus {obj['id']}: joined from the fragments {obj['fragments']}")
            comments.extend(f"Fragment {fragment}: {comment}" for fragment, comment in obj["comments"].items())
        visible = [f"nuclei {version}"]
        version_cells = zarr.open_group(cells_path, mode="r") if os.path.exists(cells_path) else None
        if version_cells is not None and "segmentation" in version_cells.attrs.get("done_stages", []):
            labels.insert(-1, (f"cells {version}", "cells", version_cells["segmentation"]))
            visible.append(f"cells {version}")
        else:
            visible.append(next(name for name in segmentations[::-1] if name.startswith("cells")))
        segmentations.extend([f"cells {version}", f"nuclei {version}"])
    for comment in comments:
        print(comment)
    for name, target, array in labels:
        data = array[crops[target]]
        is_segmentation = name in segmentations
        if name != "cells mask":
            print(f"{name}: {len(np.unique(data[data > 0]))} objects in the crop")
        layer = viewer.add_labels(
            data, name=f"{name} s{LEVELS[target]}", visible=name in visible, **placements[target],
        )
        # The cell of a nucleus has its id (also its seed, the expanded nucleus).
        if is_segmentation or name in ("cells seeds", "cells border seed"):
            layer.selected_label = nucleus_id

    # Start at the center of the nucleus (the crop is not centered on it at the border of the volume).
    scale, translation = get_transform(root, LEVELS["nuclei"])
    viewer.dims.set_point(0, translation[0] + center[0] * scale[0])
    viewer.scale_bar.visible = True
    return viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nucleus_id", type=int, help="The id of the nucleus in the nucleus segmentation.")
    parser.add_argument("--crop_size", type=int, nargs="+", default=list(CROP_SIZE),
                        help="Size of the crop in s1 voxels: one value for z, y and x, or three values ZYX. "
                        f"By default {' '.join(str(cs) for cs in CROP_SIZE)}.")
    parser.add_argument("--intermediates", action="store_true",
                        help="Add the seeds, the unfiltered watershed and the mask of the segmentations (hidden).")
    args = parser.parse_args()
    crop_size = args.crop_size * 3 if len(args.crop_size) == 1 else args.crop_size
    assert len(crop_size) == 3, crop_size
    view_nucleus(args.nucleus_id, tuple(crop_size), args.intermediates)
    napari.run()


if __name__ == "__main__":
    main()
