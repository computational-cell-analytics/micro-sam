"""The micro-sam2 model for the BioImage.IO model format.

BioImage.IO tools build this module, load the exported weights into it and call it.
"""
from typing import Optional, Tuple

import numpy as np

import torch
from torch import nn


class SAM2PredictorAdaptor(nn.Module):
    """A micro-sam2 model that does interactive and automatic instance segmentation.

    The module segments one object per prompt when you give box, point or mask prompts.
    It does automatic instance segmentation (AIS) when you give no prompts and the weights hold
    a UniSAM2 decoder. The module needs the 'micro_sam' and 'sam2' packages.

    Args:
        model_type: The SAM2 backbone, one of 'hvit_t', 'hvit_s', 'hvit_b' or 'hvit_l'.
    """
    def __init__(self, model_type: str) -> None:
        super().__init__()
        from micro_sam.v2.util import CFG_PATHS, _build_sam2_backbone

        self.model_type = model_type
        # The weights come from 'load_state_dict', so the base weights are not loaded.
        self.sam2 = _build_sam2_backbone(CFG_PATHS[model_type], None, "cpu", "images")
        self.decoder = None
        self._predictor = None
        self._segmenter = None
        self._cached_image = None

    def load_state_dict(self, state, strict: bool = True, **kwargs):
        """Load the weights of the SAM2 model and, if the state has one, of the UniSAM2 decoder.

        Args:
            state: The SAM2 weights under 'model_state' and the decoder weights under 'decoder_state'.
                The state can also hold only the SAM2 weights.
            strict: Whether the keys must match the model exactly.
            kwargs: Other arguments for `torch.nn.Module.load_state_dict`.

        Returns:
            The missing and unexpected keys of the SAM2 model.
        """
        model_state = state.get("model_state", state)
        result = self.sam2.load_state_dict(model_state, strict=strict, **kwargs)

        decoder_state = state.get("decoder_state")
        if decoder_state is None:
            self.decoder = None
        else:
            from micro_sam.v2.models.util import UniSAM2, joint_unetr_state
            output_channels, initial_features = decoder_state["out_conv.weight"].shape[:2]
            # The decoder reuses the image encoder of the SAM2 model.
            self.decoder = UniSAM2(
                encoder=self.sam2.image_encoder, output_channels=int(output_channels),
                initial_features=int(initial_features), device=self._device,
            )
            self.decoder.load_state_dict(joint_unetr_state(state), strict=strict)
            self.decoder.eval()
            self.decoder.model_type = self.model_type
        self._predictor = None
        self._segmenter = None
        self._cached_image = None
        return result

    @property
    def _device(self) -> torch.device:
        return next(self.sam2.parameters()).device

    def _get_predictor(self):
        from micro_sam.v2.util import get_sam2_image_predictor
        if self._predictor is None or self._predictor.device != self._device:
            self.sam2.model_type = self.model_type
            self.sam2.model_name = self.model_type
            self.sam2._hash = None
            self._predictor = get_sam2_image_predictor(self.sam2)
            self._cached_image = None
        return self._predictor

    def _set_image(self, image: np.ndarray) -> None:
        """Compute the image embeddings, but only if the image changed since the last call."""
        if self._cached_image is not None and np.array_equal(self._cached_image, image):
            return
        from micro_sam.v2.normalization import to_image
        from micro_sam.v2.util import encode_image
        encode_image(self._get_predictor(), to_image(image))
        self._cached_image = image.copy()

    def _automatic_instance_segmentation(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment all objects with the decoder and return one binary mask per object."""
        if self.decoder is None:
            raise ValueError(
                "These weights have no decoder for automatic instance segmentation, because the model "
                "was trained for interactive segmentation only. Give box, point or mask prompts."
            )
        from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation
        from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator

        predictor = self._get_predictor()
        if self._segmenter is None:
            self._segmenter = get_instance_segmentation_generator(
                model=self.sam2, decoder=self.decoder, segmentation_mode="ais", device=self._device,
            )
        segmentation = automatic_instance_segmentation(
            predictor, self._segmenter, image, ndim=2, verbose=False, device=self._device,
        )

        seg_ids = np.unique(segmentation)
        seg_ids = seg_ids[seg_ids != 0]
        if len(seg_ids) == 0:
            masks = np.zeros((1, 0, 1) + segmentation.shape, dtype="uint8")
        else:
            masks = np.stack([segmentation == seg_id for seg_id in seg_ids])[None, :, None].astype("uint8")
        # AIS does not predict a mask quality, so all scores are 1.
        scores = np.ones((1, masks.shape[1], 1), dtype="float32")
        return torch.from_numpy(masks), torch.from_numpy(scores)

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
        box_prompts: Optional[torch.Tensor] = None,
        point_prompts: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        mask_prompts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment objects in an image, with prompts or automatically.

        All prompts of one object are in the same position of the object axis. Each prompt type that you
        give must have the same number of objects.

        Args:
            image: The image, with shape 1 x C x H x W and 1 or 3 channels. The module normalizes the intensities.
            box_prompts: The boxes, with shape 1 x OBJECTS x 4, in the order x_min, y_min, x_max, y_max.
            point_prompts: The point coordinates, with shape 1 x OBJECTS x POINTS x 2, in the order x, y.
            point_labels: The point labels, with shape 1 x OBJECTS x POINTS: 1 is positive, 0 is negative.
            mask_prompts: The mask logits, with shape 1 x OBJECTS x 1 x 256 x 256.

        Returns:
            The masks, with shape 1 x OBJECTS x 1 x H x W.
            The predicted mask quality scores, with shape 1 x OBJECTS x 1.
        """
        if image.shape[0] != 1:
            raise ValueError(f"The batch size must be 1, got {image.shape[0]}.")
        if image.shape[1] not in (1, 3):
            raise ValueError(f"The image must have 1 or 3 channels, got {image.shape[1]}.")
        if (point_prompts is None) != (point_labels is None):
            raise ValueError("Give 'point_prompts' and 'point_labels' together.")

        # The micro-sam2 functions expect a channel-last numpy image.
        image_np = image[0].permute(1, 2, 0).float().cpu().numpy()
        if image_np.shape[-1] == 1:
            image_np = image_np[..., 0]

        if box_prompts is None and point_prompts is None and mask_prompts is None:
            return self._automatic_instance_segmentation(image_np)

        self._set_image(image_np)
        kwargs = {}
        if box_prompts is not None:
            kwargs["box"] = box_prompts[0].float().cpu().numpy()
        if point_prompts is not None:
            kwargs["point_coords"] = point_prompts[0].float().cpu().numpy()
            kwargs["point_labels"] = point_labels[0].cpu().numpy()
        if mask_prompts is not None:
            kwargs["mask_input"] = mask_prompts[0].float().cpu().numpy()

        n_objects = {len(value) for value in kwargs.values()}
        if len(n_objects) != 1:
            raise ValueError(f"All prompts must have the same number of objects, got {sorted(n_objects)}.")

        from micro_sam.v2.util import autocast
        predictor = self._get_predictor()
        with autocast(predictor.device):
            masks, scores, _ = predictor.predict(multimask_output=False, **kwargs)

        # SAM2 removes the object axis for one object.
        masks = np.asarray(masks).reshape((-1,) + image_np.shape[:2])
        scores = np.asarray(scores, dtype="float32").reshape(-1)
        masks = torch.from_numpy(masks[None, :, None].astype("uint8"))
        scores = torch.from_numpy(scores[None, :, None])
        return masks, scores
