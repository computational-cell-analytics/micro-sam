# we have two datasets here for the light microscopy experiments:
# - PlantSeg (Root): in-domain case
# - PlantSeg (Ovules): out-of-domain case
#
# NOTE:
# IMPORTANT: ideally, we need to stay consistent with 2d inference
#   1. plantseg root:
#       a. for validation: I'll take the first volume from sorted glob.
#           - shape: 100, 620, 768; 56 instances
#       b. for testing: I'll take the first volume from sorted glob
#           - shape: 486, 620, 768; 61 instances
#   2. plantseg ovules:
#       a. for validation: I'll take the first volume from sorted glob.
#           - shape: 25, 768, 768; 237 instances
#       b. for testing: I'll take the first volume from sorted glob
#           - shape: 320, 768, 768; 2638 instances


import os
from glob import glob

import h5py
from skimage.measure import label

from util import (
    _3d_automatic_instance_segmentation_with_decoder,
    _3d_interactive_instance_segmentation,
    _get_default_args
)


def get_raw_and_label_volumes(data_dir, species, split):
    volume_paths = sorted(glob(os.path.join(data_dir, f"{species}_{split}", "*")))
    chosen_volume_path = volume_paths[0]  # we choose the first path
    with h5py.File(chosen_volume_path, "r") as f:
        raw = f["raw"][:]
        labels = f["label"][:]

    # let's take a crop
    if species == "root":
        if split == "val":
            raw, labels = raw[:, :, :768], labels[:, :, :768]
        else:  # test
            raw, labels = raw[:, :, :768], labels[:, :, :768]

    elif species == "ovules":
        if split == "val":
            raw, labels = raw[175:200, :768, :768], labels[175:200, :768, :768]
        else:  # test
            raw, labels = raw[:, :768, :768], labels[:, :768, :768]

    assert raw.shape == labels.shape

    # applying connected components to get instances
    labels = label(labels)

    return raw, labels


def for_plantseg(args):
    test_raw, test_labels = get_raw_and_label_volumes(args.input_path, args.species, "test")

    if args.ais:
        # this should be experiment specific, so PlantSeg Ovules and PlantSeg Root in this case
        auto_3d_seg_kwargs = {
            "center_distance_threshold": 0.3,
            "boundary_distance_threshold": 0.4,
            "distance_smoothing": 2.2,
            "min_size": 200,
            "gap_closing": 2,
            "min_z_extent": 2
        }
        _3d_automatic_instance_segmentation_with_decoder(
            test_raw=test_raw,
            test_labels=test_labels,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            result_dir=args.resdir,
            embedding_dir=args.embedding_path,
            auto_3d_seg_kwargs=auto_3d_seg_kwargs,
            species=args.species
        )

    if args.int:
        val_raw, val_labels = get_raw_and_label_volumes(args.input_path, args.species, "val")
        _3d_interactive_instance_segmentation(
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            result_dir=args.resdir,
            embedding_dir=args.embedding_path,
            species=args.species
        )


def main(args):
    assert args.species is not None, "Choose from 'root' / 'ovules'"
    for_plantseg(args)


if __name__ == "__main__":
    args = _get_default_args("/scratch/projects/nim00007/sam/data/plantseg/")
    main(args)
