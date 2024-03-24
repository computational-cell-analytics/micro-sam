import os

import numpy as np
import imageio.v3 as imageio

from util import get_paths, DATASETS


SAVE_ROOT = "/scratch/projects/nim00007/sam/data/for_mitonet/"


def evaluate_mitonet_predictions(dataset_name, gt_paths):
    raise NotImplementedError


def make_stack_from_inputs(dataset_name):
    assert dataset_name in DATASETS
    print("Creating the volumes for:", dataset_name)
    test_image_paths, test_gt_paths = get_paths(dataset_name, "test")

    print([imageio.imread(image_path).shape for image_path in test_image_paths])

    image_stack = np.stack([imageio.imread(image_path) for image_path in test_image_paths])
    gt_stack = np.stack([imageio.imread(gt_path) for gt_path in test_gt_paths])

    assert image_stack.shape == gt_stack.shape

    dname_split = dataset_name.split("/")
    if len(dname_split) > 1:
        _save_fname = f"{dname_split[0]}_{dname_split[1]}"
    else:
        _save_fname = dataset_name

    # now let's save it in a target directory
    _save_root = os.path.join(SAVE_ROOT, _save_fname)
    raw_volume_path = os.path.join(_save_root, f"{_save_fname}_raw.tif")
    labels_volume_path = os.path.join(_save_root, f"{_save_fname}_labels.tif")

    if os.path.exists(_save_root):
        print("The stacks are already saved at:", _save_root)
        return
    else:
        os.makedirs(_save_root, exist_ok=True)

    imageio.imwrite(raw_volume_path, image_stack, compression="zlib")
    imageio.imwrite(labels_volume_path, gt_stack, compression="zlib")


def make_stacks(specific_dataset=None):
    # let's create the volumes
    if specific_dataset is None:
        make_stack_from_inputs("lucchi")
        make_stack_from_inputs("mitoem/rat")
        make_stack_from_inputs("mitoem/human")
        make_stack_from_inputs("uro_cell")
        make_stack_from_inputs("mitolab/c_elegans")
        make_stack_from_inputs("mitolab/fly_brain")
        make_stack_from_inputs("mitolab/glycolytic_muscle")
        make_stack_from_inputs("mitolab/hela_cell")
        make_stack_from_inputs("mitolab/lucchi_pp")
        make_stack_from_inputs("mitolab/salivary_gland")
    else:
        make_stack_from_inputs(specific_dataset)


def main():
    make_stacks()


if __name__ == "__main__":
    main()
