import os
import h5py
import itertools
import pandas as pd
import numpy as np
from math import floor
from typing import Union

from skimage.measure import label

from micro_sam import util
from micro_sam.inference import batched_inference
from micro_sam.prompt_generators import PointAndBoxPromptGenerator
from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume

from elf.evaluation import mean_segmentation_accuracy


def get_raw_and_label_volumes(volume_path):
    with h5py.File(volume_path, "r") as f:
        raw = f["raw"][:]
        label = f["labels"][:]

    return raw, label


def segment_lucchi_from_slices(
    input_path: str,
    model_type: str,
    checkpoint: str,
    embedding_path: str,
    iou_threshold: float = 0.8,
    projection: Union[str, dict] = "mask",
    box_extension: Union[float, int] = 0.025
):
    _get_model = util.get_sam_model if checkpoint is None else util.get_custom_sam_model
    predictor = _get_model(model_type=model_type, checkpoint_path=checkpoint)

    test_volume_path = os.path.join(input_path, "lucchi_test.h5")
    volume, labels = get_raw_and_label_volumes(test_volume_path)

    # precompute embeddings
    image_embeddings = util.precompute_image_embeddings(predictor, volume, save_path=embedding_path, ndim=3)

    # let's get the instances from the label volume
    instance_labels = label(labels)

    label_ids = np.unique(instance_labels)[1:]
    final_segmentation = np.zeros_like(instance_labels)
    for label_id in label_ids:
        # the binary volume segmentation
        this_seg = np.zeros_like(instance_labels)
        this_seg[instance_labels == label_id] = 1

        # we search which slices have the current object
        slice_range = np.where(this_seg)[0]

        # we choose the middle slice of the current object
        slice_range = (slice_range.min(), slice_range.max())
        slice_choice = floor(np.mean(slice_range))
        print("The object lies in slice range:", slice_range)

        # we get the box prompts for segmentation
        prompt_generator = PointAndBoxPromptGenerator(
            0, 0, dilation_strength=10, get_point_prompts=False, get_box_prompts=True
        )
        _, bbox_coordinates = util.get_centers_and_bounding_boxes(this_seg[slice_choice])
        _, _, box_prompts, _ = prompt_generator(this_seg[slice_choice], [bbox_coordinates[1]])

        # now, we perform interactive segmentation on one slice
        output_slice = batched_inference(predictor, volume[slice_choice], 1, boxes=box_prompts.numpy())
        output_seg = np.zeros_like(instance_labels)
        output_seg[slice_choice][output_slice == 1] = 1

        this_seg = segment_mask_in_volume(
            output_seg, predictor, image_embeddings, segmented_slices=np.array(slice_choice), stop_lower=False,
            stop_upper=False, iou_threshold=iou_threshold, projection=projection, box_extension=box_extension
        )

        final_segmentation[this_seg == 1] = label_id

    msa = mean_segmentation_accuracy(final_segmentation, instance_labels)
    return msa


def main(args):
    iou_thresholds = np.arange(0.5, 1, 0.1)
    all_projection_methods = ["mask", "points", "bounding_box"]
    all_extension_factors = np.arange(0, 0.5, 0.025)

    parameter_combinations = list(
        itertools.product(
            [round(iou_threshold, 1) for iou_threshold in iou_thresholds],
            all_projection_methods,
            [round(box_extension, 3) for box_extension in all_extension_factors]
        )
    )

    results = []
    for parameters in parameter_combinations:
        iou_threshold, projection, box_extension = parameters

        msa = segment_lucchi_from_slices(
            input_path=args.input_path,
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            embedding_path=args.embedding_path,
            iou_threshold=iou_threshold,
            projection=projection,
            box_extension=box_extension
        )
        this_result = {
            "iou_threshold": iou_threshold,
            "projection": projection,
            "box_extension": box_extension,
            "msa": msa
        }
        results.append(pd.DataFrame.from_dict([this_result]))

    results = pd.concat(results, ignore_index=True)
    results.to_csv("./results.csv")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="/home/anwai/data/lucchi", help="Path to volume")
    parser.add_argument("-m", "--model_type", type=str, default="vit_b", help="Name of the image encoder")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="The custom checkpoint path.")
    parser.add_argument("-e", "--embedding_path", type=str, default=None, help="Path to save embeddings")
    args = parser.parse_args()
    main(args)
