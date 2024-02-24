import os
import h5py
import numpy as np
from math import floor

from skimage.measure import label

from micro_sam import util
from micro_sam.inference import batched_inference
from micro_sam.prompt_generators import PointAndBoxPromptGenerator
from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume


ROOT = "/scratch/projects/nim00007/sam/data"


def segment_lucchi_from_slices():
    predictor = util.get_sam_model(model_type="vit_b")
    embedding_path = "./embeddings/"

    test_volume_path = os.path.join(ROOT, "lucchi", "lucchi_test.h5")
    with h5py.File(test_volume_path, "r") as f:
        volume = f["raw"][:]
        labels = f["labels"][:]

        # precompute embeddings
        image_embeddings = util.precompute_image_embeddings(predictor, volume, save_path=embedding_path, ndim=3)

        # let's get the instances from the label volume
        instance_labels = label(labels)

        label_ids = np.unique(instance_labels)[1:]
        for label_id in label_ids:
            # the binary volume segmentation
            this_seg = np.zeros_like(instance_labels)
            this_seg[instance_labels == label_id] = 1

            # we search for which slices have the object
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

            # now, we need to perform interactive segmentation on one slice
            output_slice = batched_inference(predictor, volume[slice_choice], 1, boxes=box_prompts.numpy())
            output_seg = np.zeros_like(instance_labels)
            output_seg[slice_choice][output_slice] = 1

            this_seg = segment_mask_in_volume(
                output_seg, predictor, image_embeddings, segmented_slices=np.array(slice_choice),
                stop_lower=False, stop_upper=False, iou_threshold=0.8, projection="mask", box_extension=0.0
            )

            breakpoint()


def main():
    segment_lucchi_from_slices()


if __name__ == "__main__":
    main()
