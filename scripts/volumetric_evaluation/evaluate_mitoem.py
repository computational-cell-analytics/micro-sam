# in-domain case for electron microscopy generalist
#
# NOTE:
#   1. for the em organelles generalist, we use the following rois:
#       - for training = [np.s_[100:110, :, :], np.s_[100:110, :, :]]
#       - for validation = [np.s_[0:5, :, :], np.s_[0:5, :, :]]
#   2. for the grid search below:
#       - for validation
#       - for testing


import os

import h5py
import numpy as np

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_mitoem_loader

from micro_sam.training import identity


def create_raw_and_label_volumes(data_path, species):
    def _get_volumes_from_loaders(split):
        if split == "train":
            rois = [np.s_[:100, :, :]]
        else:
            rois = [np.s_[5:, :, :]]

        loader = get_mitoem_loader(
            path=data_path,
            splits=split,
            patch_shape=(50, 768, 768),
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

        return chosen_raw, chosen_labels

    _path_to_new_volume = os.path.join(data_path, "for_micro_sam", f"mitoem_{species}.h5")
    if os.path.exists(_path_to_new_volume):
        print("The volume is already computed and stored at", _path_to_new_volume)
        return
    else:
        os.makedirs(os.path.split(_path_to_new_volume)[0], exist_ok=True)

    val_raw, val_labels = _get_volumes_from_loaders("train")
    test_raw, test_labels = _get_volumes_from_loaders("val")

    # now let's save them
    with h5py.File(_path_to_new_volume, "a") as f:
        f.create_dataset("volume/val/raw", data=val_raw, compression="gzip")
        f.create_dataset("volume/val/labels", data=val_labels, compression="gzip")
        f.create_dataset("volume/test/raw", data=test_raw, compression="gzip")
        f.create_dataset("volume/test/labels", data=test_labels, compression="gzip")

    print("The volume has been computed and stored at", _path_to_new_volume)
    return


def _3d_automatic_instance_segmentation_with_decoder():
    ...


def _3d_interactive_instance_segmentation():
    ...


def main(args):
    create_raw_and_label_volumes(args.input_path, "human")
    create_raw_and_label_volumes(args.input_path, "rat")


if __name__ == "__main__":
    from util import _get_default_args
    args = _get_default_args("/scratch/projects/nim00007/sam/data/mitoem")
    main(args)
