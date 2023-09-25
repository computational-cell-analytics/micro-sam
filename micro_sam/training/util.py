import os
from typing import List, Optional, Union

import torch
import numpy as np

from ..prompt_generators import PointAndBoxPromptGenerator
from ..util import get_centers_and_bounding_boxes, get_sam_model
from .trainable_sam import TrainableSAM


def get_trainable_sam_model(
    model_type: str = "vit_h",
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    freeze: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> TrainableSAM:
    """Get the trainable sam model.

    Args:
        model_type: The type of the segment anything model.
        checkpoint_path: Path to a custom checkpoint from which to load the model weights.
        freeze: Specify parts of the model that should be frozen.
            By default nothing is frozen and the full model is updated.
        device: The device to use for training.

    Returns:
        The trainable segment anything model.
    """
    # set the device here so that the correct one is passed to TrainableSAM below
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _, sam = get_sam_model(device, model_type, checkpoint_path, return_sam=True)

    # freeze components of the model if freeze was passed
    # ideally we would want to add components in such a way that:
    # - we would be able to freeze the choice of encoder/decoder blocks, yet be able to add components to the network
    #   (for e.g. encoder blocks to "image_encoder")
    if freeze is not None:
        for name, param in sam.named_parameters():
            if isinstance(freeze, list):
                # we would want to "freeze" all the components in the model if passed a list of parts
                for l_item in freeze:
                    if name.startswith(f"{l_item}"):
                        param.requires_grad = False
            else:
                # we "freeze" only for one specific component when passed a "particular" part
                if name.startswith(f"{freeze}"):
                    param.requires_grad = False

    # convert to trainable sam
    trainable_sam = TrainableSAM(sam, device)
    return trainable_sam


class ConvertToSamInputs:
    """Convert outputs of data loader to the expected batched inputs of the SegmentAnything model.

    Args:
        dilation_strength: The dilation factor.
            It determines a "safety" border from which prompts are not sampled to avoid ambiguous prompts
            due to imprecise groundtruth masks.
        box_distortion_factor: Factor for distorting the box annotations derived from the groundtruth masks.
            Not yet implemented.
    """
    def __init__(
        self,
        dilation_strength: int = 10,
        box_distortion_factor: Optional[float] = None,
    ) -> None:
        self.dilation_strength = dilation_strength
        # TODO implement the box distortion logic
        if box_distortion_factor is not None:
            raise NotImplementedError

    def _get_prompt_generator(self, n_positive_points, n_negative_points, get_boxes, get_points):
        """Returns the prompt generator w.r.t. the "random" attributes inputed."""

        # the class initialization below gets the random choice of n_positive and n_negative points as inputs
        # (done in the trainer)
        # in case of dynamic choice while choosing between points and/or box, it gets those as well
        prompt_generator = PointAndBoxPromptGenerator(n_positive_points=n_positive_points,
                                                      n_negative_points=n_negative_points,
                                                      dilation_strength=self.dilation_strength,
                                                      get_box_prompts=get_boxes,
                                                      get_point_prompts=get_points)
        return prompt_generator

    def _get_prompt_lists(self, gt, n_samples, n_positive_points, n_negative_points, get_boxes,
                          get_points, prompt_generator, point_coordinates, bbox_coordinates):
        """Returns a list of "expected" prompts subjected to the random input attributes for prompting."""
        box_prompts = []
        point_prompts = []
        point_label_prompts = []

        # getting the cell instance except the bg
        cell_ids = np.unique(gt)[1:]

        accepted_cell_ids = []
        sampled_cell_ids = []

        # while conditions gets all the prompts until it satisfies the requirement
        while len(accepted_cell_ids) < min(n_samples, len(cell_ids)):
            if len(sampled_cell_ids) == len(cell_ids):  # we did not find enough cells
                break

            my_cell_id = np.random.choice(np.setdiff1d(cell_ids, sampled_cell_ids))
            sampled_cell_ids.append(my_cell_id)

            obj_mask = (gt == my_cell_id)
            bbox = bbox_coordinates[my_cell_id]
            # points = point_coordinates[my_cell_id]
            # removed "points" to randomly choose fg points
            coord_list, label_list, bbox_list = prompt_generator(obj_mask, bbox)

            if get_boxes is True and get_points is False:  # only box
                bbox_list = bbox_list[0]
                box_prompts.append([bbox_list[1], bbox_list[0],
                                    bbox_list[3], bbox_list[2]])
                accepted_cell_ids.append(my_cell_id)

            if get_points:  # one with points expected
                # check for the minimum point requirement per object in the batch
                if len(label_list) == n_negative_points + n_positive_points:
                    point_prompts.append(np.array([ip[::-1] for ip in coord_list]))
                    point_label_prompts.append(np.array(label_list))
                    accepted_cell_ids.append(my_cell_id)
                    if get_boxes:  # one with boxes expected with points as well
                        bbox_list = bbox_list[0]
                        box_prompts.append([bbox_list[1], bbox_list[0],
                                            bbox_list[3], bbox_list[2]])

        point_prompts = np.array(point_prompts)
        point_label_prompts = np.array(point_label_prompts)
        return box_prompts, point_prompts, point_label_prompts, accepted_cell_ids

    def __call__(self, x, y, n_pos, n_neg, get_boxes=False, n_samples=None):
        """Convert the outputs of dataloader and prompt settings to the batch format expected by SAM.
        """

        # condition to see if we get point prompts, then we (ofc) use point-prompting
        # else we don't use point prompting
        if n_pos == 0 and n_neg == 0:
            get_points = False
        else:
            get_points = True

        # keeping the solution open by checking for deterministic/dynamic choice of point prompts
        prompt_generator = self._get_prompt_generator(n_pos, n_neg, get_boxes, get_points)

        batched_inputs = []
        batched_sampled_cell_ids_list = []
        for i, gt in enumerate(y):
            gt = gt.squeeze().numpy().astype(np.int32)
            point_coordinates, bbox_coordinates = get_centers_and_bounding_boxes(gt)

            this_n_samples = len(point_coordinates) if n_samples is None else n_samples
            box_prompts, point_prompts, point_label_prompts, sampled_cell_ids = self._get_prompt_lists(
                gt, this_n_samples,
                n_pos, n_neg,
                get_boxes,
                get_points,
                prompt_generator,
                point_coordinates,
                bbox_coordinates
            )

            # check to be sure about the expected size of the no. of elements in different settings
            if get_boxes is True and get_points is False:
                assert len(sampled_cell_ids) == len(box_prompts), \
                    print(len(sampled_cell_ids), len(box_prompts))

            elif get_boxes is False and get_points is True:
                assert len(sampled_cell_ids) == len(point_prompts) == len(point_label_prompts), \
                    print(len(sampled_cell_ids), len(point_prompts), len(point_label_prompts))

            elif get_boxes is True and get_points is True:
                assert len(sampled_cell_ids) == len(box_prompts) == len(point_prompts) == len(point_label_prompts), \
                    print(len(sampled_cell_ids), len(box_prompts), len(point_prompts), len(point_label_prompts))

            batched_sampled_cell_ids_list.append(sampled_cell_ids)

            batched_input = {"image": x[i], "original_size": x[i].shape[1:]}
            if get_boxes:
                batched_input["boxes"] = torch.tensor(box_prompts)
            if get_points:
                batched_input["point_coords"] = torch.tensor(point_prompts)
                batched_input["point_labels"] = torch.tensor(point_label_prompts)

            batched_inputs.append(batched_input)

        return batched_inputs, batched_sampled_cell_ids_list
