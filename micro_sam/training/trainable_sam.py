from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from segment_anything.modeling import Sam


# simple wrapper around SAM in order to keep things trainable
class TrainableSAM(nn.Module):
    """Wrapper to make the SegmentAnything model trainable.

    Args:
        sam: The SegmentAnything Model.
        device: The device for training.
    """
    def __init__(
        self,
        sam: Sam,
        device: Union[str, torch.device],
    ) -> None:
        super().__init__()
        self.sam = sam
        self.device = device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input.

        Args:
            x: The input tensor.

        Returns:
            The normalized and padded tensor.
        """
        # Normalize colors
        x = (x - self.sam.pixel_mean) / self.sam.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def image_embeddings_oft(self, input_images):
        """@private"""
        image_embeddings = self.sam.image_encoder(input_images)
        return image_embeddings

    # batched inputs follow the same syntax as the input to sam.forward
    def forward(
        self,
        batched_inputs: List[Dict[str, Any]],
        multimask_output: bool = False,
        image_embeddings: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Forward pass.

        Args:
            batched_inputs: The batched input images and prompts.
            multimask_output: Whether to predict mutiple or just a single mask.
            image_embeddings: The precompute image embeddings. If not passed then they will be computed.

        Returns:
            The predicted segmentation masks and iou values.
        """
        input_images = torch.stack([self.preprocess(x=x["image"].to(self.device)) for x in batched_inputs], dim=0)
        if image_embeddings is None:
            image_embeddings = self.sam.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_inputs, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"].to(self.device), image_record["point_labels"].to(self.device))
            else:
                points = None

            if "boxes" in image_record:
                boxes = image_record.get("boxes").to(self.device)
            else:
                boxes = None

            if "mask_inputs" in image_record:
                masks = image_record.get("mask_inputs").to(self.device)
            else:
                masks = None

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "low_res_masks": low_res_masks,
                    "masks": masks,
                    "iou_predictions": iou_predictions
                }
            )

        return outputs
