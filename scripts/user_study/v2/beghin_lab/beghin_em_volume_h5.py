"""Export one scale level of the EM spheroid of the Beghin lab into a single hdf5 file and view this file in napari,
for inspecting the data on a machine without access to the zarr files.

The file contains what view_beghin_em_volume.py shows at this level: the raw data, the nuclei, the cells, the nuclei
segmented from point prompts and the cells with the background seed (only filled in the slabs of check_border_seed,
see the attribute 'slabs' of that dataset), the merged nuclei (without the fragments of the prompted nuclei, with
the prompted nuclei) and the cells from them with the background seed, the joined nuclei (the fragments of a nucleus
as one object, listed in the attribute 'joined' of that dataset) and their cells, and in addition the points that
were used as prompts. The prompted nuclei that are only seeds for their cell, not full nuclei, are marked in the
attribute 'objects' of the points ('seed_only', 'comment'). The latest version of the cells is visible in 'view'.
Each dataset has the attributes 'scale' and 'translation' (nanometer, ZYX), so that all layers can be placed in
nanometer. The points are stored in voxel coordinates of the raw at the level of the file.

'view' only needs h5py and napari, so this script can be copied together with the file. It avoids layer arguments
that only exist in recent napari versions.

Example:
    python beghin_em_volume_h5.py export  # writes beghin_em_s3.h5 to the prediction folder
    python beghin_em_volume_h5.py view beghin_em_s3.h5
"""
import os
import argparse
import json

import h5py
import numpy as np

ZARR_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids/PF_Dish_C_pin3new_Spheroid3.ome.zarr"
PREDICTION_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions"
PYRAMID_PATH = os.path.join(PREDICTION_DIR, "segmentation_pyramids.zarr")
PROMPTED_PATH = os.path.join(PREDICTION_DIR, "nuclei_s1_prompted.zarr")
MERGED_NUCLEI_PATH = os.path.join(PREDICTION_DIR, "nuclei_s1_merged.zarr")
JOINED_NUCLEI_PATH = os.path.join(PREDICTION_DIR, "nuclei_s1_joined.zarr")
BORDER_SEED_PATH = os.path.join(PREDICTION_DIR, "cells_s0_border_seed_check.zarr")
SEGMENTATIONS = (
    "nuclei", "cells", "cells_border_seed", "nuclei_prompted", "nuclei_merged", "cells_merged_nuclei_border_seed",
    "nuclei_joined", "cells_joined_nuclei_border_seed",
)
# The versions of the cells, the latest one in a file is visible.
CELLS = ("cells", "cells_merged_nuclei_border_seed", "cells_joined_nuclei_border_seed")
CHUNKS = (32, 128, 128)


def get_transform(root, level):
    """The scale and translation (nanometer, ZYX) of a level of the raw from the OME metadata."""
    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    dataset = next(ds for ds in datasets if ds["path"] == f"s{level}")
    transforms = {tr["type"]: tr for tr in dataset["coordinateTransformations"]}
    scale = tuple(transforms["scale"]["scale"])
    translation = tuple(transforms["translation"]["translation"]) if "translation" in transforms else (0.0,) * 3
    return scale, translation


def get_factor(root, level, target_level):
    """The downscaling factor (ZYX) from a level to a higher one."""
    return np.round(np.array(get_transform(root, target_level)[0]) / get_transform(root, level)[0]).astype("int")


def write_dataset(f, name, data, scale, translation, **attrs):
    chunks = tuple(min(ch, sh) for ch, sh in zip(CHUNKS, data.shape)) if data.ndim == 3 else None
    ds = f.create_dataset(name, data=data, chunks=chunks, compression="gzip" if chunks else None)
    ds.attrs["scale"] = scale
    ds.attrs["translation"] = translation
    for key, value in attrs.items():
        ds.attrs[key] = value
    print(f"{name}: shape {data.shape}, {data.dtype}")


def export(output_path, level):
    """Write the raw, the segmentations and the prompt points of a level into a hdf5 file.

    Args:
        output_path: The path of the hdf5 file.
        level: The scale level. It must be in the pyramids of all segmentations.
    """
    import zarr

    root = zarr.open_group(ZARR_PATH, mode="r")
    pyramid = zarr.open_group(PYRAMID_PATH, mode="r")
    scale, translation = get_transform(root, level)

    with h5py.File(output_path, "w") as f:
        f.attrs["source"] = ZARR_PATH
        f.attrs["level"] = level
        f.attrs["unit"] = "nanometer"
        f.attrs["axes"] = "zyx"
        write_dataset(f, "raw", root[f"s{level}"][:], scale, translation)

        for name in SEGMENTATIONS:
            if name not in pyramid:
                print(f"{name}: no pyramid, skipped")
                continue
            group = pyramid[name]
            source_level = group.attrs["source_level"]
            # The first voxel of each block of the level the segmentation was computed at, which has the
            # translation of that level of the raw.
            attrs = {"source": group.attrs["source"], "source_level": source_level}
            if name == "cells_border_seed":
                # The slabs with a segmentation, as the z-range at the level it was computed at and at this level.
                factor = int(get_factor(root, source_level, level)[0])
                slabs = zarr.open_group(BORDER_SEED_PATH, mode="r").attrs["slabs"]
                slabs = sorted([int(z) for z in slab.split(":")] for slab in slabs)
                attrs["slabs"] = json.dumps([
                    {f"s{source_level}": [z0, z1], f"s{level}": [-(-z0 // factor), -(-z1 // factor)]}
                    for z0, z1 in slabs
                ])
            if name == "nuclei_joined":
                attrs["joined"] = json.dumps(zarr.open_group(JOINED_NUCLEI_PATH, mode="r").attrs["joined"])
            write_dataset(f, name, group[f"s{level}"][:], scale, get_transform(root, source_level)[1], **attrs)

        if os.path.exists(PROMPTED_PATH):
            prompted = zarr.open_group(PROMPTED_PATH, mode="r").attrs
            objects = prompted["objects"]
            if os.path.exists(MERGED_NUCLEI_PATH):
                # With the marks of the prompted nuclei that are only seeds.
                objects = zarr.open_group(MERGED_NUCLEI_PATH, mode="r").attrs["objects"]
            # From the center of a voxel of the level of the prompts to voxel coordinates of this level.
            factor = get_factor(root, prompted["level"], level)
            points = (np.array([obj["point"] for obj in objects]) - (factor - 1) / 2) / factor
            write_dataset(
                f, "nuclei_points", points, scale, translation, ids=[obj["id"] for obj in objects],
                objects=json.dumps(objects), points_level=prompted["level"],
            )
    print(f"Written to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def view(path):
    """Show the datasets of a file written by export in napari, all placed in nanometer.

    Args:
        path: The path of the hdf5 file.
    """
    import napari

    viewer = napari.Viewer()
    with h5py.File(path, "r") as f:
        level = f.attrs["level"]
        # The latest cells replace the previous segmentations, which stay as hidden layers.
        visible = ["raw", "nuclei_points"] + [name for name in CELLS if name in f][-1:]
        if "cells" in visible:
            visible.append("nuclei_prompted")
        for name, ds in f.items():
            kwargs = dict(
                name=f"{name} s{level}", scale=ds.attrs["scale"], translate=ds.attrs["translation"],
                visible=name in visible,
            )
            if name == "raw":
                # The raw is shown inverted, as the models see it.
                viewer.add_image(ds[:], contrast_limits=(0, 255), colormap="gray_r", **kwargs)
            elif name == "nuclei_points":
                for obj in json.loads(ds.attrs["objects"]):
                    print(f"nucleus {obj['id']}: point at s{ds.attrs['points_level']} {obj['point']},",
                          f"replaces {obj['replaces']}{'; ' + obj['comment'] if obj.get('seed_only') else ''}")
                viewer.add_points(
                    ds[:], features={"id": ds.attrs["ids"]}, text={"string": "{id}", "color": "yellow", "size": 10},
                    size=6, face_color="yellow", opacity=0.7, **kwargs,
                )
            else:
                data = ds[:]
                print(f"{name}: {len(np.unique(data)) - 1} objects")
                for obj in json.loads(ds.attrs["joined"]) if "joined" in ds.attrs else []:
                    print(f"nucleus {obj['id']}: joined from the fragments {obj['fragments']}")
                    for fragment, comment in obj["comments"].items():
                        print(f"  fragment {fragment}: {comment}")
                if "slabs" in ds.attrs:
                    slabs = [slab[f"s{level}"] for slab in json.loads(ds.attrs["slabs"])]
                    print(f"{name}: only filled in the z-ranges {slabs}")
                viewer.add_labels(data, **kwargs)
    # The raw at the bottom, the points on top.
    order = sorted(range(len(viewer.layers)), key=lambda i: (
        not isinstance(viewer.layers[i], napari.layers.Image), isinstance(viewer.layers[i], napari.layers.Points)
    ))
    viewer.layers[:] = [viewer.layers[i] for i in order]
    viewer.scale_bar.visible = True
    # The unit of the scale bar: from the layers in recent napari versions, from the scale bar in older ones.
    for layer in viewer.layers:
        if hasattr(layer, "units"):
            layer.units = ("nm",) * 3
    if not hasattr(viewer.layers[0], "units"):
        viewer.scale_bar.unit = "nm"
    napari.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="Write the hdf5 file.")
    export_parser.add_argument("--level", type=int, default=3, choices=[2, 3], help="The scale level, s3 by default.")
    export_parser.add_argument(
        "--output_path", help="The path of the hdf5 file, by default beghin_em_s<level>.h5 in the prediction folder."
    )
    view_parser = subparsers.add_parser("view", help="View a hdf5 file in napari.")
    view_parser.add_argument("path", help="The path of the hdf5 file.")
    args = parser.parse_args()

    if args.command == "export":
        output_path = args.output_path or os.path.join(PREDICTION_DIR, f"beghin_em_s{args.level}.h5")
        export(output_path, args.level)
    else:
        view(args.path)


if __name__ == "__main__":
    main()
