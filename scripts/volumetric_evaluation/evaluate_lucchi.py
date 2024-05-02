# out-of-domain case for electron microscopy experiments
#
# NOTE:
# IMPORTANT: ideally, we need to stay consistent with 2d inference
#  1. for validation: we take the entire train volume
#       - shape: (165, 768, 1024)
#  2. for testing: we take the entire test volume
#      - shape: (165, 768, 1024)


import os

import h5py
from skimage.measure import label

from util import (
    _3d_automatic_instance_segmentation_with_decoder,
    _3d_interactive_instance_segmentation,
    _get_default_args
)


def get_raw_and_label_volumes(data_dir, split):
    if split == "val":
        chosen_set = "train"
    elif split == "test":
        chosen_set = "test"
    else:
        raise ValueError

    volume_path = os.path.join(data_dir, f"lucchi_{chosen_set}.h5")
    with h5py.File(volume_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]

    assert raw.shape == labels.shape

    # applying connected components to get instances
    labels = label(labels)

    return raw, labels


def for_lucchi(args):
    test_raw, test_labels = get_raw_and_label_volumes(args.input_path, "test")

    experiment_folder = args.experiment_folder
    embedding_path = os.path.join(experiment_folder, "embeddings")
    result_dir = os.path.join(experiment_folder, "results")

    if args.ais:
        # this should be experiment specific, so Lucchi in this case
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
        )

    if args.int:
        val_raw, val_labels = get_raw_and_label_volumes(args.input_path, "val")
        _3d_interactive_instance_segmentation(
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            result_dir=result_dir,
            embedding_dir=embedding_path,
        )


def main(args):
    assert args.species is None
    for_lucchi(args)


if __name__ == "__main__":
    args = _get_default_args("/scratch/projects/nim00007/sam/data/lucchi")
    main(args)
