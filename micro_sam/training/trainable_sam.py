from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide


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
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)

    def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Resize, normalize pixel values and pad to a square input.

        Args:
            x: The input tensor.

        Returns:
            The resized, normalized and padded tensor.
            The shape of the image after resizing.
        """

        # Resize longest side to match the image encoder.
        x = self.transform.apply_image_torch(x)
        input_size = x.shape[-2:]

        # Normalize colors
        x = (x - self.sam.pixel_mean.unsqueeze(0)) / self.sam.pixel_std.unsqueeze(0)

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x, input_size

    def image_embeddings_oft(self, batched_inputs):
        # Compute the input images.
        input_images, input_size = self.preprocess(
            torch.stack([x["image"] for x in batched_inputs], dim=0).to(self.device)
        )
        # Update the input size for each input in the batch.
        for i in range(len(batched_inputs)):
            batched_inputs[i]["input_size"] = input_size
        # Compute the image embeddings.
        image_embeddings = self.sam.image_encoder(input_images)
        return image_embeddings, batched_inputs

    # batched inputs follow the same syntax as the input to sam.forward
    def forward(
        self,
        batched_inputs: List[Dict[str, Any]],
        image_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> List[Dict[str, Any]]:
        """Forward pass.

        Args:
            batched_inputs: The batched input images and prompts.
            image_embeddings: The precompute image embeddings. If not passed then they will be computed.
            multimask_output: Whether to predict mutiple or just a single mask.

        Returns:
            The predicted segmentation masks and iou values.
        """
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
                input_size=image_record["input_size"],
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
