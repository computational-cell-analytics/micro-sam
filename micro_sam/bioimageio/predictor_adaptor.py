import warnings
from typing import Optional, Tuple

import numpy as np

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

    If it was loaded from a checkpoint that also contains the state of an instance segmentation
    decoder, then calling it without any prompts will run automatic instance segmentation (AIS):
    the UNETR decoder predicts foreground and distance maps from the image embeddings and the
    instances are computed from them via a seeded watershed.
    Running AIS requires the `micro_sam` package.

    Args:
        model_type: The type of the model for the image encoder.
            Can be one of 'vit_b', 'vit_l', 'vit_h' or 'vit_t'.
            For 'vit_t' support the 'mobile_sam' package has to be installed.
    """
    def __init__(self, model_type: str) -> None:
        super().__init__()
        self.sam_model = sam_model_registry[model_type]()
        self.sam = SamPredictor(self.sam_model)
        self.decoder = None

    def load_state_dict(self, state, **kwargs):
        # Finetuning checkpoints store SAM and decoder weights separately.
        if "model_state" in state:
            load_result = self.sam.model.load_state_dict(state["model_state"], **kwargs)
            decoder_state = state.get("decoder_state")
            if decoder_state is not None:
                from micro_sam.instance_segmentation import get_decoder
                device = next(self.sam.model.parameters()).device
                self.decoder = get_decoder(self.sam.model.image_encoder, decoder_state, device=device)
            return load_result

        return self.sam.model.load_state_dict(state, **kwargs)

    def _automatic_instance_segmentation(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run automatic instance segmentation with the decoder for the image embeddings
        that were set in the forward method.

        Returns the instances as a stack of binary masks, so that the output signature
        matches the output of the prompt-based segmentation.
        """
        if self.decoder is None:
            raise ValueError(
                "This model was exported without an instance segmentation decoder, "
                "so it does not support automatic instance segmentation. "
                "At least one prompt input (box, point or mask) is required."
            )
        from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder

        segmenter = InstanceSegmentationWithDecoder(self.sam, self.decoder)
        image_embeddings = {
            "features": self.sam.features,
            "input_size": tuple(self.sam.input_size),
            "original_size": tuple(self.sam.original_size),
        }
        # The image is unused because the embeddings are precomputed.
        segmenter.initialize(image=image[0].permute(1, 2, 0).cpu().numpy(), image_embeddings=image_embeddings)
        segmentation = segmenter.generate(output_mode="instance_segmentation")
        seg_ids = np.unique(segmentation)
        seg_ids = seg_ids[seg_ids != 0]
        instance_masks = [segmentation == seg_id for seg_id in seg_ids]

        height, width = self.sam.original_size
        if len(instance_masks) == 0:
            masks = torch.zeros((1, 0, 1, height, width), dtype=torch.uint8)
        else:
            masks = torch.from_numpy(np.stack(instance_masks)[None, :, None].astype("uint8"))
        # AIS does not predict mask quality.
        scores = torch.ones((1, masks.shape[1], 1), dtype=torch.float32)
        return masks, scores

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
            The segmentation masks.
            The scores for prediction quality.
            The computed image embeddings.
        """
        batch_size = image.shape[0]
        if batch_size != 1:
            raise ValueError

        # Cast to float for MPS compatibility: F.interpolate with antialias=True
        # only supports floating-point dtypes on MPS (Apple Silicon).
        image_float = image.float() if not image.is_floating_point() else image

        # We have image embeddings set and image embeddings were not passed.
        if self.sam.is_image_set and embeddings is None:
            pass  # do nothing

        # The embeddings are passed, so we set them.
        elif embeddings is not None:
            self.sam.features = embeddings
            self.sam.orig_h, self.sam.orig_w = image.shape[2:]
            self.sam.input_h, self.sam.input_w = self.sam.transform.apply_image_torch(image_float).shape[2:]
            self.sam.is_image_set = True

        # We don't have image embeddings set and they were not passed.
        elif not self.sam.is_image_set:
            input_ = self.sam.transform.apply_image_torch(image_float)
            self.sam.set_torch_image(input_, original_image_size=image.shape[2:])
            self.sam.orig_h, self.sam.orig_w = self.sam.original_size
            self.sam.input_h, self.sam.input_w = self.sam.input_size

        assert self.sam.is_image_set, "The predictor has not yet been initialized."

        # Ensure input size and original size are set.
        self.sam.input_size = (self.sam.input_h, self.sam.input_w)
        self.sam.original_size = (self.sam.orig_h, self.sam.orig_w)

        # Preserve prompt-free SamPredictor inference without a decoder.
        prompts = (box_prompts, point_prompts, mask_prompts)
        if self.decoder is not None and all(prompt is None for prompt in prompts):
            masks, scores = self._automatic_instance_segmentation(image)
            embeddings = self.sam.get_image_embedding()
            return masks, scores, embeddings

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
