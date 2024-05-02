# NOTE:
# This script below the following:
#  - stacks indivudual slice images into one volume for segmenting in empanada-napari
#  - evaluates the segmentations with the mean segmentation accuracy metric


import os
from glob import glob

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy

from util import get_paths, DATASETS


SAVE_ROOT = "/scratch/projects/nim00007/sam/data/for_mitonet/"


def make_stack_from_inputs(dataset_name, data_type=None):
    assert dataset_name in DATASETS
    print("Creating the volumes for:", dataset_name)
    test_image_paths, test_gt_paths = get_paths(dataset_name, "test")

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

    image_stack = np.stack([imageio.imread(image_path) for image_path in test_image_paths])
    gt_stack = np.stack([imageio.imread(gt_path) for gt_path in test_gt_paths])

    assert image_stack.shape == gt_stack.shape

    if data_type is not None:
        image_stack = image_stack.astype(data_type)

    imageio.imwrite(raw_volume_path, image_stack, compression="zlib")
    imageio.imwrite(labels_volume_path, gt_stack, compression="zlib")


def make_stacks(specific_dataset=None):
    # let's create the volumes
    if specific_dataset is None:
        make_stack_from_inputs("lucchi")
        make_stack_from_inputs("mitoem/rat")
        make_stack_from_inputs("mitoem/human")
        make_stack_from_inputs("uro_cell", data_type="uint8")
        make_stack_from_inputs("mitolab/c_elegans")
        make_stack_from_inputs("mitolab/fly_brain")
        make_stack_from_inputs("mitolab/glycolytic_muscle")
        make_stack_from_inputs("mitolab/hela_cell")
        make_stack_from_inputs("mitolab/lucchi_pp")
        make_stack_from_inputs("mitolab/salivary_gland")
        make_stack_from_inputs("vnc")
        make_stack_from_inputs("asem/mito", data_type="uint8")
    else:
        make_stack_from_inputs(specific_dataset)


def _evaluate_mitonet_predictions(view=False):
    # ROOT = "/media/anwai/ANWAI/data/"
    ROOT = "/scratch/projects/nim00007/sam/data"

    all_inputs_dir = sorted(glob(os.path.join(ROOT, "for_mitonet", "*")))
    tem_dir = os.path.join(ROOT, "for_mitonet", "tem")
    all_inputs_dir = [
        input_dir for input_dir in all_inputs_dir if input_dir != tem_dir
    ]

    # let's do evaluation for the tem dataset
    image_paths = sorted(glob(os.path.join(tem_dir, "raw", "*")))
    gt_paths = sorted(glob(os.path.join(tem_dir, "labels", "*")))
    seg_paths = sorted(glob(os.path.join(tem_dir, "segmentations", "*")))

    assert len(image_paths) == len(gt_paths) == len(seg_paths)

    msa_list, sa50_list, sa75_list = [], [], []
    for image_path, gt_path, seg_path in zip(image_paths, gt_paths, seg_paths):
        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)
        seg = imageio.imread(seg_path)

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(gt, visible=False)
            v.add_labels(seg)
            napari.run()

        msa, sa = mean_segmentation_accuracy(seg, gt, return_accuracies=True)
        msa_list.append(msa)
        sa50_list.append(sa[0])
        sa75_list.append(sa[4])

    res_dict = {
        "msa": np.mean(msa_list),
        "sa50": np.mean(sa50_list),
        "sa75": np.mean(sa75_list)
    }
    res_df = pd.DataFrame.from_dict([res_dict])
    res_df.to_csv("./mitonet_tem.csv")

    # let's do this for all other datasets
    for this_dir in all_inputs_dir:
        name = os.path.split(this_dir)[-1]
        raw_path = glob(os.path.join(this_dir, "*_raw.tif"))[0]
        label_path = glob(os.path.join(this_dir, "*_labels.tif"))[0]
        seg_path = glob(os.path.join(this_dir, "*_mitonet_seg.tif"))[0]

        raw = imageio.imread(raw_path)
        labels = imageio.imread(label_path)
        segmentation = imageio.imread(seg_path)

        assert raw.shape == labels.shape == segmentation.shape

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(labels.astype("uint8"), visible=False)
            v.add_labels(segmentation)
            napari.run()

        # let's compute slice-wise msa
        msa_list, sa50_list, sa75_list = [], [], []
        for _seg, _label in zip(segmentation, labels):
            msa, sa = mean_segmentation_accuracy(_seg, _label, return_accuracies=True)
            msa_list.append(msa)
            sa50_list.append(sa[0])
            sa75_list.append(sa[4])

        res_dict = {
            "msa": np.mean(msa_list),
            "sa50": np.mean(sa50_list),
            "sa75": np.mean(sa75_list)
        }
        res_df = pd.DataFrame.from_dict([res_dict])
        res_df.to_csv(f"./mitonet_{name}.csv")


def main():
    # make_stacks()
    _evaluate_mitonet_predictions()


if __name__ == "__main__":
    main()
