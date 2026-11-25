"""View the whole EM spheroid of the Beghin lab in napari at one scale level: the raw data, the nuclei
(explore_beghin_em_data.py watershed_volume, computed at s1), the cells (segment_cells_volume, computed at s0) and,
if they exist, the nuclei segmented from point prompts (segment_nuclei_from_prompts, computed at s1, with ids after
those of the nuclei) and the cells with the background seed on the slabs of check_border_seed (computed at s0, empty
outside of these slabs; hidden at the start).
If the merged nuclei (merge_prompted_nuclei: without the fragments of the prompted nuclei, with the prompted nuclei)
and the cells from them with the background seed ('segment_cells_volume --border_seed --merged_nuclei') have a
pyramid, these cells are the visible segmentation and the other ones are hidden. The same holds for the joined nuclei
(join_nucleus_fragments: the fragments of a nucleus as one object) and their cells, which replace the merged ones.

The levels above the one a segmentation was computed at come from the pyramids of downsample_segmentations, which take
the first voxel of each block. All layers are placed in nanometer: the raw with the scale and translation of its OME
metadata (mean downsampled), the cells with the scale of the level (first voxel of each block of s0), the nuclei with
the translation of s1 (first voxel of each block of s1). At s0 the nuclei are shown at s1.
The raw is shown inverted (colormap gray_r), as the models see it.

Levels s2 and s3 are loaded into memory, s0 and s1 are read lazily from the zarr arrays (or always with --lazy).

Example:
    python view_beghin_em_volume.py  # s3
    python view_beghin_em_volume.py --level 2
"""
import os
import argparse

import napari
import numpy as np
import zarr

ZARR_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids/PF_Dish_C_pin3new_Spheroid3.ome.zarr"
PREDICTION_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions"
PYRAMID_PATH = os.path.join(PREDICTION_DIR, "segmentation_pyramids.zarr")
SEGMENTATIONS = (
    "nuclei", "cells", "cells_border_seed", "nuclei_prompted", "nuclei_merged", "cells_merged_nuclei_border_seed",
    "nuclei_joined", "cells_joined_nuclei_border_seed",
)
# The versions of the cells, the latest one that exists is visible.
CELLS = ("cells", "cells_merged_nuclei_border_seed", "cells_joined_nuclei_border_seed")
IN_MEMORY_LEVELS = (2, 3)


def get_transform(root, level):
    """The scale and translation (nanometer, ZYX) of a level of the raw from the OME metadata."""
    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    dataset = next(ds for ds in datasets if ds["path"] == f"s{level}")
    transforms = {tr["type"]: tr for tr in dataset["coordinateTransformations"]}
    scale = tuple(transforms["scale"]["scale"])
    translation = tuple(transforms["translation"]["translation"]) if "translation" in transforms else (0.0,) * 3
    return scale, translation


def get_segmentation(name, level):
    """The array of a segmentation at a level (or at the level it was computed at, if that is higher), that level
    and the level it was computed at. None for a segmentation without a pyramid."""
    pyramid = zarr.open_group(PYRAMID_PATH, mode="r")
    if name not in pyramid:
        return None
    group = pyramid[name]
    source_level = group.attrs["source_level"]
    if level <= source_level:
        return zarr.open_array(group.attrs["source"], mode="r"), source_level, source_level
    return group[f"s{level}"], level, source_level


def load(array, in_memory):
    return array[:] if in_memory else array


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, default=3, choices=[0, 1, 2, 3], help="The scale level, s3 by default.")
    parser.add_argument("--lazy", action="store_true", help="Read every level lazily from the zarr arrays.")
    args = parser.parse_args()

    root = zarr.open_group(ZARR_PATH, mode="r")
    level = args.level
    in_memory = level in IN_MEMORY_LEVELS and not args.lazy
    raw = root[f"s{level}"]
    scale, translation = get_transform(root, level)
    print(f"s{level}: shape {raw.shape}, voxel size {scale} nm, {'in memory' if in_memory else 'lazy'}")

    pyramid = zarr.open_group(PYRAMID_PATH, mode="r")
    visible = [name for name in CELLS if name in pyramid][-1:]
    if visible == ["cells"]:
        visible.append("nuclei_prompted")
    viewer = napari.Viewer()
    viewer.add_image(
        load(raw, in_memory), name=f"raw s{level}", scale=scale, translate=translation, units="nm",
        contrast_limits=(0, 255), colormap="gray_r",
    )
    for name in SEGMENTATIONS:
        segmentation = get_segmentation(name, level)
        if segmentation is None:
            continue
        data, data_level, source_level = segmentation
        data_scale, _ = get_transform(root, data_level)
        # The first voxel of each block of the level the segmentation was computed at, which has the translation of
        # that level of the raw.
        _, data_translation = get_transform(root, source_level)
        data = load(data, in_memory)
        if in_memory:
            print(f"{name} s{data_level}: {len(np.unique(data)) - 1} objects")
        elif data_level != level:
            print(f"{name}: shown at s{data_level}")
        viewer.add_labels(
            data, name=f"{name} s{data_level}", scale=data_scale, translate=data_translation, units="nm",
            visible=name in visible,
        )
    viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
