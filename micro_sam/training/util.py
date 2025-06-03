import os
from math import ceil, floor
from functools import partial
from typing import Dict, List, Optional, Union, Tuple, Callable

import numpy as np

import torch

from segment_anything.utils.transforms import ResizeLongestSide

from ..prompt_generators import PointAndBoxPromptGenerator
from ..util import (
    get_centers_and_bounding_boxes, get_sam_model, get_device,
    segmentation_to_one_hot, _DEFAULT_MODEL,
)
from .. import models as custom_models
from .trainable_sam import TrainableSAM

from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.transform.raw import normalize_percentile, normalize
from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb


def identity(x):
    """Identity transformation.

    This is a helper function to skip data normalization when finetuning SAM.
    Data normalization is performed within the model and should thus be skipped as
    a preprocessing step in training.
    """
    return x


def require_8bit(x):
    """Transformation to require 8bit input data range (0-255).
    """
    if x.max() < 1:
        x = x * 255
    return x


def _raw_transform(image: np.ndarray, raw_trafo: Callable) -> np.ndarray:
    return raw_trafo(image) * 255


def _normalize_percentile(image: np.ndarray) -> np.ndarray:
    image = normalize_percentile(image)  # Use 1st and 99th percentile values for min-max normalization.
    image = np.clip(image, 0, 1)  # Clip the values to be in range [0, 1].
    return image


def get_raw_transform(preprocess: Optional[str] = None) -> Optional[Callable]:
    """Transformation functions to normalize inputs.

    Args:
        preprocess: By default, the transformation function is set to 'None'.
            The user can choose from 'normalize_minmax' / 'normalize_percentile'.

    Returns:
        The transformation function.
    """

    if preprocess is None:  # Ensures that inputs are 8-bit.
        return require_8bit
    else:
        if preprocess == "normalize_minmax":
            raw_trafo = normalize
        elif preprocess == "normalize_percentile":
            raw_trafo = _normalize_percentile
        else:
            raise ValueError(f"'{preprocess}' is not a supported preprocessing.")

        return partial(_raw_transform, raw_trafo=raw_trafo)


def get_trainable_sam_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    freeze: Optional[List[str]] = None,
    return_state: bool = False,
    peft_kwargs: Optional[Dict] = None,
    flexible_load_checkpoint: bool = False,
    **model_kwargs
) -> TrainableSAM:
    """Get the trainable sam model.

    Args:
        model_type: The Segment Anything model that should be finetuned. The weights of this model
            will be used for initialization, unless a custom weight file is passed via `checkpoint_path`.
        device: The device to use for training. By default, automatically chooses the best available model.
        checkpoint_path: Path to a custom checkpoint from which to load the model weights.
        freeze: Specify parts of the model that should be frozen, namely: `image_encoder`, `prompt_encoder` and
            `mask_decoder`. By default nothing is frozen and the full model is updated.
        return_state: Whether to return the full checkpoint state. By default, set to 'False'.
        peft_kwargs: Keyword arguments for the PEFT wrapper class.
        flexible_load_checkpoint: Whether to adjust mismatching params while loading pretrained checkpoints.
            By default, set to 'False'.
        model_kwargs: Additional keyword arguments for the `util.get_sam_model`.

    Returns:
        The trainable segment anything model.
    """
    # set the device here so that the correct one is passed to TrainableSAM below
    device = get_device(device)
    _, sam, state = get_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        return_sam=True,
        return_state=True,
        flexible_load_checkpoint=flexible_load_checkpoint,
        **model_kwargs
    )

    # NOTE: This is done exclusive to "get_sam_model" here to use PEFT's layer-specific initialization on top.
    # Whether to use Parameter Efficient Finetuning methods to wrap around Segment Anything.
    # Overwrites the SAM model by freezing the backbone and allow PEFT methods.
    if peft_kwargs and isinstance(peft_kwargs, dict):
        if model_type[:5] == "vit_t":
            raise ValueError("'micro-sam' does not support parameter efficient finetuning for 'mobile-sam'.")

        sam = custom_models.peft_sam.PEFT_Sam(sam, **peft_kwargs).sam

    # freeze components of the model if freeze was passed
    # ideally we would want to add components in such a way that:
    # - we would be able to freeze the choice of encoder/decoder blocks, yet be able to add components to the network
    #   (for e.g. encoder blocks to "image_encoder")
    if freeze is not None:
        for name, param in sam.named_parameters():
            if not isinstance(freeze, list):
                # we "freeze" only for one specific component when passed a "particular" part
                freeze = [freeze]

            # we would want to "freeze" all the components in the model if passed a list of parts
            for l_item in freeze:
                # in case PEFT is switched on, we cannot freeze the image encoder
                if (peft_kwargs and peft_kwargs.get('rank') is not None) and (l_item == "image_encoder"):
                    raise ValueError("You cannot use PEFT & freeze the image encoder at the same time.")

                if name.startswith(f"{l_item}"):
                    param.requires_grad = False

    # convert to trainable sam
    trainable_sam = TrainableSAM(sam)

    if return_state:
        return trainable_sam, state
    return trainable_sam


class ConvertToSamInputs:
    """Convert outputs of data loader to the expected batched inputs of the Segment Anything model.

    Args:
        transform: The transformation to resize the prompts. Should be the same transform used in the
            model to resize the inputs. If `None` the prompts will not be resized.
        dilation_strength: The dilation factor.
            It determines a "safety" border from which prompts are not sampled to avoid ambiguous prompts
            due to imprecise groundtruth masks. By default, set to '10'.
        box_distortion_factor: Factor for distorting the box annotations derived from the groundtruth masks.
    """
    def __init__(
        self,
        transform: Optional[ResizeLongestSide],
        dilation_strength: int = 10,
        box_distortion_factor: Optional[float] = None,
    ) -> None:
        self.dilation_strength = dilation_strength
        self.transform = identity if transform is None else transform
        self.box_distortion_factor = box_distortion_factor

    def _distort_boxes(self, bbox_coordinates, shape):
        distorted_boxes = []
        for bbox in bbox_coordinates:
            # The bounding box is parametrized by y0, x0, y1, x1.
            y0, x0, y1, x1 = bbox
            ly, lx = y1 - y0, x1 - x0
            y0 = int(round(max(0, y0 - np.random.uniform(0, self.box_distortion_factor) * ly)))
            y1 = int(round(min(shape[0], y1 + np.random.uniform(0, self.box_distortion_factor) * ly)))
            x0 = int(round(max(0, x0 - np.random.uniform(0, self.box_distortion_factor) * lx)))
            x1 = int(round(min(shape[1], x1 + np.random.uniform(0, self.box_distortion_factor) * lx)))
            distorted_boxes.append([y0, x0, y1, x1])
        return distorted_boxes

    def _get_prompt_lists(self, gt, n_samples, prompt_generator):
        """Returns a list of "expected" prompts subjected to the random input attributes for prompting."""

        _, bbox_coordinates = get_centers_and_bounding_boxes(gt, mode="p")

        # get the segment ids
        cell_ids = np.unique(gt)[1:]
        if n_samples is None:  # n-samples is set to None, so we use all ids
            sampled_cell_ids = cell_ids

        else:  # n-samples is set, so we subsample the cell ids
            sampled_cell_ids = np.random.choice(cell_ids, size=min(n_samples, len(cell_ids)), replace=False)
            sampled_cell_ids = np.sort(sampled_cell_ids)

        # only keep the bounding boxes for sampled cell ids
        bbox_coordinates = [bbox_coordinates[sampled_id] for sampled_id in sampled_cell_ids]
        if self.box_distortion_factor is not None:
            bbox_coordinates = self._distort_boxes(bbox_coordinates, shape=gt.shape[-2:])

        # convert the gt to the one-hot-encoded masks for the sampled cell ids
        object_masks = segmentation_to_one_hot(gt, None if n_samples is None else sampled_cell_ids)

        # derive and return the prompts
        point_prompts, point_label_prompts, box_prompts, _ = prompt_generator(object_masks, bbox_coordinates)
        return box_prompts, point_prompts, point_label_prompts, sampled_cell_ids

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
        prompt_generator = PointAndBoxPromptGenerator(
            n_positive_points=n_pos,
            n_negative_points=n_neg,
            dilation_strength=self.dilation_strength,
            get_box_prompts=get_boxes,
            get_point_prompts=get_points
        )

        batched_inputs = []
        batched_sampled_cell_ids_list = []

        for image, gt in zip(x, y):
            gt = gt.squeeze().numpy().astype(np.int64)
            box_prompts, point_prompts, point_label_prompts, sampled_cell_ids = self._get_prompt_lists(
                gt, n_samples, prompt_generator,
            )

            # check to be sure about the expected size of the no. of elements in different settings
            if get_boxes:
                assert len(sampled_cell_ids) == len(box_prompts), f"{len(sampled_cell_ids)}, {len(box_prompts)}"

            if get_points:
                assert len(sampled_cell_ids) == len(point_prompts) == len(point_label_prompts), \
                    f"{len(sampled_cell_ids)}, {len(point_prompts)}, {len(point_label_prompts)}"

            batched_sampled_cell_ids_list.append(sampled_cell_ids)

            batched_input = {"image": image, "original_size": image.shape[1:]}
            if get_boxes:
                batched_input["boxes"] = self.transform.apply_boxes_torch(
                    box_prompts, original_size=gt.shape[-2:]
                ) if self.transform is not None else box_prompts

            if get_points:
                batched_input["point_coords"] = self.transform.apply_coords_torch(
                    point_prompts, original_size=gt.shape[-2:]
                ) if self.transform is not None else point_prompts
                batched_input["point_labels"] = point_label_prompts

            batched_inputs.append(batched_input)

        return batched_inputs, batched_sampled_cell_ids_list


class ConvertToSemanticSamInputs:
    """Convert outputs of data loader to the expected batched inputs of the Segment Anything model
    for semantic segmentation.
    """
    def __call__(self, x, y):
        """Convert the outputs of dataloader to the batched format of inputs expected by SAM.
        """
        batched_inputs = []
        for image in x:
            batched_input = {"image": image, "original_size": image.shape[-2:]}
            batched_inputs.append(batched_input)

        return batched_inputs


#
# Raw and Label Transformations for the Generalist and Specialist finetuning
#


def normalize_to_8bit(raw):
    raw = normalize(raw) * 255
    return raw


class ResizeRawTrafo:
    def __init__(
        self,
        desired_shape: Tuple[int, ...],
        do_rescaling: bool = False,
        valid_channels: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: str = "constant",
        ensure_rgb: bool = True,
    ):
        self.desired_shape = desired_shape
        self.do_rescaling = do_rescaling
        self.valid_channels = valid_channels
        self.padding = padding
        self.ensure_rgb = ensure_rgb

    def __call__(self, raw):
        if self.ensure_rgb:
            raw = to_rgb(raw)  # Ensure all images are in 3-channels: triplicate one channel to three channels.

        if self.do_rescaling:
            raw = normalize_percentile(raw, axis=self.valid_channels)
            raw = normalize(raw)
            raw = raw * 255

        # Pad the inputs to the desired shape.
        tmp_ddim = [desired - curr for desired, curr in zip(self.desired_shape, raw.shape)]
        ddim = [(per_dim / 2) for per_dim in tmp_ddim]
        pad_width = [(ceil(d), floor(d)) for d in ddim]
        raw = np.pad(raw, pad_width=pad_width, mode=self.padding)

        assert raw.shape == self.desired_shape
        return raw


class ResizeLabelTrafo:
    def __init__(
        self, desired_shape: Tuple[int, ...], min_size: int = 0, padding: str = "constant",
    ):
        self.desired_shape = desired_shape
        self.min_size = min_size
        self.padding = padding

    def __call__(self, labels):
        distance_trafo = PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True,
            min_size=self.min_size
        )
        labels = distance_trafo(labels)

        # choosing H and W from labels (4, H, W), from above dist trafo outputs
        tmp_ddim = (self.desired_shape[0] - labels.shape[1], self.desired_shape[0] - labels.shape[2])
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
        labels = np.pad(
            labels,
            pad_width=((0, 0), (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
            mode=self.padding
        )
        assert labels.shape[1:] == self.desired_shape, labels.shape
        return labels
