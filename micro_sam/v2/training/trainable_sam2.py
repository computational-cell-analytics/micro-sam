from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM2Transform:
    """Scale point/box coordinates from original image space to SAM2's input space."""

    def __init__(self, image_size: int = 1024) -> None:
        self.image_size = image_size

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Scale point coords (..., 2) from original_size to image_size x image_size."""
        h, w = original_size
        scale = torch.tensor(
            [self.image_size / w, self.image_size / h],
            dtype=coords.dtype, device=coords.device,
        )
        return coords * scale

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Scale boxes (N, 4) from original_size to image_size x image_size."""
        h, w = original_size
        scale = torch.tensor(
            [self.image_size / w, self.image_size / h,
             self.image_size / w, self.image_size / h],
            dtype=boxes.dtype, device=boxes.device,
        )
        return boxes * scale


class TrainableSAM2(nn.Module):
    """SAM2 wrapper for 2D interactive segmentation training.

    Handles normalization and resizing to 1024x1024, computes image embeddings
    once per training step, and runs the prompt encoder + mask decoder without
    video memory (each 2D patch is treated as an independent image).

    Args:
        sam2: A SAM2Base model instance.
    """

    _PIXEL_MEAN = [0.485, 0.456, 0.406]
    _PIXEL_STD = [0.229, 0.224, 0.225]

    def __init__(self, sam2: nn.Module) -> None:
        super().__init__()
        self.sam2 = sam2
        self.image_size = sam2.image_size  # 1024
        self.transform = SAM2Transform(self.image_size)
        self.register_buffer(
            "_pixel_mean", torch.tensor(self._PIXEL_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_pixel_std", torch.tensor(self._PIXEL_STD).view(1, 3, 1, 1)
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet stats and resize to image_size x image_size.

        Args:
            x: (B, 3, H, W) tensor with values in [0, 1].

        Returns:
            (B, 3, image_size, image_size) normalized tensor.
        """
        x = (x - self._pixel_mean) / self._pixel_std
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return x

    def image_embeddings_oft(
        self, batched_inputs: List[Dict[str, Any]]
    ) -> Tuple[Tuple[torch.Tensor, List[torch.Tensor]], List[Dict[str, Any]]]:
        """Compute image features once per training step.

        Caches backbone + high-res features so that all sub-iterations of
        iterative prompting share the same image features.

        Args:
            batched_inputs: List of dicts, each with key "image" (3, H, W).

        Returns:
            ((backbone_feats, high_res_feats), batched_inputs).
        """
        device = next(self.parameters()).device
        x = torch.stack([inp["image"] for inp in batched_inputs]).to(device, non_blocking=True)
        x = self.preprocess(x)

        backbone_out = self.sam2.forward_image(x)
        _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)

        B = x.shape[0]
        # Lowest-resolution features: (HW, B, C) → (B, C, H, W)
        backbone_feats = vision_feats[-1].permute(1, 2, 0).view(
            B, -1, feat_sizes[-1][0], feat_sizes[-1][1]
        )
        # High-res features for the SAM decoder (all levels except the lowest-res)
        high_res_feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1][:-1]

        return (backbone_feats, high_res_feats), batched_inputs

    def forward(
        self,
        batched_inputs: List[Dict[str, Any]],
        image_embeddings: Tuple[torch.Tensor, List[torch.Tensor]],
        multimask_output: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run the prompt encoder and mask decoder for each image.

        Args:
            batched_inputs: List of dicts with keys:
                - "image": (3, H, W) input tensor
                - "original_size": (H, W) of the original patch
                - "point_coords": (N_obj, P, 2) coords in SAM2 input space (optional)
                - "point_labels": (N_obj, P) labels (optional)
                - "boxes": (N_obj, 4) boxes in SAM2 input space (optional)
                - "mask_inputs": (N_obj, 1, H, W) mask prompts (optional)
            image_embeddings: Output of image_embeddings_oft.
            multimask_output: Return 3 mask candidates per object if True.

        Returns:
            List of dicts with keys "masks", "low_res_masks", "iou_predictions".
            "masks" are at the original patch resolution for direct loss computation.
        """
        backbone_feats, high_res_feats = image_embeddings
        device = backbone_feats.device
        outputs = []

        for i, record in enumerate(batched_inputs):
            # Determine number of objects from available prompts.
            if "point_coords" in record:
                n_objects = record["point_coords"].shape[0]
            elif "boxes" in record:
                n_objects = record["boxes"].shape[0]
            else:
                n_objects = 1

            # Expand per-image features to (n_objects, C, H, W).
            curr_bb = backbone_feats[i:i+1].expand(n_objects, -1, -1, -1)
            curr_hr = [f[i:i+1].expand(n_objects, -1, -1, -1) for f in high_res_feats]

            point_inputs: Optional[Dict[str, torch.Tensor]] = None
            if "point_coords" in record:
                point_inputs = {
                    "point_coords": record["point_coords"].to(device),
                    "point_labels": record["point_labels"].to(device),
                }

            boxes = record.get("boxes")
            mask_inputs = record.get("mask_inputs")
            if mask_inputs is not None:
                mask_inputs = mask_inputs.to(device)

            if boxes is not None:
                # Call prompt encoder directly so boxes are embedded alongside points.
                boxes = boxes.to(device)
                points_arg = (
                    (point_inputs["point_coords"], point_inputs["point_labels"])
                    if point_inputs is not None else None
                )
                sparse_emb, dense_emb = self.sam2.sam_prompt_encoder(
                    points=points_arg, boxes=boxes, masks=mask_inputs,
                )
                low_res_ms, high_res_ms, ious, _, _, _, _ = self.sam2.sam_mask_decoder(
                    image_embeddings=curr_bb,
                    image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=multimask_output,
                    repeat_image=False,
                    high_res_features=curr_hr,
                )
            else:
                low_res_ms, high_res_ms, ious, _, _, _, _ = self.sam2._forward_sam_heads(
                    backbone_features=curr_bb,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=curr_hr,
                    multimask_output=multimask_output,
                )

            # Resize high-res masks back to the original patch resolution for loss.
            original_size = record["original_size"]
            if high_res_ms.shape[-2:] != original_size:
                high_res_ms = F.interpolate(
                    high_res_ms, size=original_size, mode="bilinear", align_corners=False,
                )

            outputs.append({
                "masks": high_res_ms,       # (N_obj, M, H, W) at original patch size
                "low_res_masks": low_res_ms,  # (N_obj, M, 256, 256) — kept for mask prompt input
                "iou_predictions": ious,      # (N_obj, M)
            })

        return outputs
