import warnings
from typing import Optional, Tuple

import torch
from torch import nn

from segment_anything.predictor import SamPredictor

try:
    # Avoid import warnings from mobile_sam
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mobile_sam import sam_model_registry
except ImportError:
    from segment_anything import sam_model_registry


class PredictorAdaptor(nn.Module):
    """Wrapper around the SamPredictor.

    This model supports the same functionality as SamPredictor and can provide mask segmentations
    from box, point or mask input prompts.

    Args:
        model_type: The type of the model for the image encoder.
            Can be one of 'vit_b', 'vit_l', 'vit_h' or 'vit_t'.
            For 'vit_t' support the 'mobile_sam' package has to be installed.
    """
    def __init__(self, model_type: str) -> None:
        super().__init__()
        sam_model = sam_model_registry[model_type]()
        self.sam = SamPredictor(sam_model)

    def load_state_dict(self, state):
        self.sam.model.load_state_dict(state)

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
        box_prompts: Optional[torch.Tensor] = None,
        point_prompts: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        mask_prompts: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            image: torch inputs of dimensions B x C x H x W
            box_prompts: box coordinates of dimensions B x OBJECTS x 4
            point_prompts: point coordinates of dimension B x OBJECTS x POINTS x 2
            point_labels: point labels of dimension B x OBJECTS x POINTS
            mask_prompts: mask prompts of dimension B x OBJECTS x 256 x 256
            embeddings: precomputed image embeddings B x 256 x 64 x 64

        Returns:
        """
        batch_size = image.shape[0]
        if batch_size != 1:
            raise ValueError

        # We have image embeddings set and image embeddings were not passed.
        if self.sam.is_image_set and embeddings is None:
            pass   # do nothing

        # The embeddings are passed, so we set them.
        elif embeddings is not None:
            self.sam.features = embeddings
            self.sam.orig_h, self.sam.orig_w = image.shape[2:]
            self.sam.input_h, self.sam.input_w = self.sam.transform.apply_image_torch(image).shape[2:]
            self.sam.is_image_set = True

        # We don't have image embeddings set and they were not passed.
        elif not self.sam.is_image_set:
            input_ = self.sam.transform.apply_image_torch(image)
            self.sam.set_torch_image(input_, original_image_size=image.shape[2:])
            self.sam.orig_h, self.sam.orig_w = self.sam.original_size
            self.sam.input_h, self.sam.input_w = self.sam.input_size

        assert self.sam.is_image_set, "The predictor has not yet been initialized."

        # Ensure input size and original size are set.
        self.sam.input_size = (self.sam.input_h, self.sam.input_w)
        self.sam.original_size = (self.sam.orig_h, self.sam.orig_w)

        if box_prompts is None:
            boxes = None
        else:
            boxes = self.sam.transform.apply_boxes_torch(box_prompts, original_size=self.sam.original_size)

        if point_prompts is None:
            point_coords = None
        else:
            assert point_labels is not None
            point_coords = self.sam.transform.apply_coords_torch(point_prompts, original_size=self.sam.original_size)[0]
            point_labels = point_labels[0]

        if mask_prompts is None:
            mask_input = None
        else:
            mask_input = mask_prompts[0]

        masks, scores, _ = self.sam.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
            multimask_output=False
        )

        assert masks.shape[2:] == image.shape[2:], \
            f"{masks.shape[2:]} is not as expected ({image.shape[2:]})"

        # Ensure batch axis.
        if masks.ndim == 4:
            masks = masks[None]
            assert scores.ndim == 2
            scores = scores[None]

        embeddings = self.sam.get_image_embedding()
        return masks.to(dtype=torch.uint8), scores, embeddings
