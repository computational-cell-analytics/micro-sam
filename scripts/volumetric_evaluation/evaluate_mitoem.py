# in-domain case for electron microscopy experiments
#
# NOTE:
# IMPORTANT: ideally, we need to stay consistent with 2d inference
#   1. for the em organelles generalist, we use the following rois:
#       - for training = [np.s_[100:110, :, :], np.s_[100:110, :, :]]
#       - for validation = [np.s_[0:5, :, :], np.s_[0:5, :, :]]
#   2. for the grid search below:
#       - for validation: we take the valudation set and sample one volume with most instances
#           - shape: (25, 768, 768)
#       - for testing: we take the validation set and sample one volume with most instances
#           - shape: (100, 768, 768)


import os

import h5py
import numpy as np
from skimage.measure import label

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_mitoem_loader

from micro_sam.training import identity

from util import (
    _3d_automatic_instance_segmentation_with_decoder,
    _3d_interactive_instance_segmentation,
    _get_default_args
)


def create_raw_and_label_volumes(data_path, species):
    def _get_volumes_from_loaders(split):
        if split == "train":  # this will be used for testing
            rois = [np.s_[:100, :, :]]
            patch_shape = (100, 768, 768)
        elif split == "val":  # this will be used for grid-search
            rois = [np.s_[5:, :, :]]
            patch_shape = (25, 768, 768)
        else:
            raise ValueError

        loader = get_mitoem_loader(
            path=data_path,
            splits=split,
            patch_shape=patch_shape,
            batch_size=1,
            samples=[species],
            sampler=MinInstanceSampler(),
            raw_transform=identity,
            rois=rois,
            num_workers=16,
        )

        max_val = 0
        for x, y in loader:
            num_instances = len(np.unique(y))
            if max_val < num_instances:
                max_val = num_instances
                chosen_raw, chosen_labels = x, y
        print(f"The chosen volume has {max_val} instances")
        return chosen_raw, chosen_labels

    _path_to_new_volume = os.path.join(data_path, "for_micro_sam", f"mitoem_{species}.h5")
    if os.path.exists(_path_to_new_volume):
        print("The volume is already computed and stored at:", _path_to_new_volume)
        return _path_to_new_volume
    else:
        os.makedirs(os.path.split(_path_to_new_volume)[0], exist_ok=True)

    print(f"Creating the volumes for {species}...")
    val_raw, val_labels = _get_volumes_from_loaders("val")
    test_raw, test_labels = _get_volumes_from_loaders("train")

    # now let's save them
    with h5py.File(_path_to_new_volume, "a") as f:
        f.create_dataset("volume/val/raw", data=val_raw.numpy().squeeze(), compression="gzip")
        f.create_dataset("volume/val/labels", data=val_labels.numpy().squeeze(), compression="gzip")
        f.create_dataset("volume/test/raw", data=test_raw.numpy().squeeze(), compression="gzip")
        f.create_dataset("volume/test/labels", data=test_labels.numpy().squeeze(), compression="gzip")

    print("The volume has been computed and stored at", _path_to_new_volume)
    return _path_to_new_volume


def get_raw_and_label_volumes(volume_path, split):
    with h5py.File(volume_path, "r") as f:
        raw = f[f"volume/{split}/raw"][:]
        labels = f[f"volume/{split}/labels"][:]

    assert raw.shape == labels.shape

    # applying connected components to get instances
    labels = label(labels)

    return raw, labels


def for_one_species(args):
    volume_path = create_raw_and_label_volumes(args.input_path, args.species)

    test_raw, test_labels = get_raw_and_label_volumes(volume_path, "test")

    experiment_folder = args.experiment_folder
    embedding_path = os.path.join(experiment_folder, "embeddings")
    result_dir = os.path.join(experiment_folder, "results")

    if args.ais:
        # this should be experiment specific, so MitoEM in this case
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
        val_raw, val_labels = get_raw_and_label_volumes(volume_path, "val")
        _3d_interactive_instance_segmentation(
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            result_dir=result_dir,
            embedding_dir=embedding_path,
            species=args.species
        )


def main(args):
    assert args.species is not None, "Choose from 'human' / 'rat'"
    for_one_species(args)


if __name__ == "__main__":
    args = _get_default_args("/scratch/projects/nim00007/sam/data/mitoem")
    main(args)
