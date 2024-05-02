# we have two datasets here for the light microscopy experiments:
# - PlantSeg (Root): in-domain case
# - PlantSeg (Ovules): out-of-domain case
#
# NOTE:
# IMPORTANT: ideally, we need to stay consistent with 2d inference
#   1. plantseg root (see `./check_volumes.py` for details)
#   2. plantseg ovules (see `./check_volumes.py` for details)


import os
from glob import glob

import h5py
from skimage.measure import label

from util import (
    _3d_automatic_instance_segmentation_with_decoder,
    _3d_interactive_instance_segmentation,
    _get_default_args
)


VOLUME_CHOICE = {
    "plantseg_ovules_val": "plantseg_ovules_val_N_420_ds2x.h5",
    "plantseg_ovules_test": "plantseg_ovules_test_N_435_final_crop_ds2.h5",
    "plantseg_root_val": "plantseg_root_val_Movie1_t00040_crop_gt.h5",
    "plantseg_root_test": "plantseg_root_test_Movie1_t00045_crop_gt.h5"

}


def get_raw_and_label_volumes(volume_path):
    if isinstance(volume_path, list):
        assert len(volume_path) == 1
        volume_path = volume_path[0]

    with h5py.File(volume_path, "r") as f:
        raw = f["volume/cropped/raw"][:]
        labels = f["volume/cropped/labels"][:]

    assert raw.shape == labels.shape

    # applying connected components to get instances
    labels = label(labels)

    return raw, labels


def for_plantseg(args):
    test_raw, test_labels = get_raw_and_label_volumes(
        glob(os.path.join(args.input_path, f"plantseg_{args.species}_test_*.h5"))
    )

    experiment_folder = args.experiment_folder
    embedding_path = os.path.join(experiment_folder, "embeddings")
    result_dir = os.path.join(experiment_folder, "results")

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
            result_dir=result_dir,
            embedding_dir=embedding_path,
            auto_3d_seg_kwargs=auto_3d_seg_kwargs,
            species=args.species
        )

    if args.int:
        val_raw, val_labels = get_raw_and_label_volumes(
            glob(os.path.join(args.input_path, f"plantseg_{args.species}_val_*.h5"))
        )
        _3d_interactive_instance_segmentation(
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            result_dir=result_dir,
            embedding_dir=embedding_path,
            species=args.species,
        )


def main(args):
    assert args.species is not None, "Choose from 'root' / 'ovules'"
    for_plantseg(args)


if __name__ == "__main__":
    args = _get_default_args("/scratch/usr/nimanwai/data/for_micro_sam/")
    main(args)
