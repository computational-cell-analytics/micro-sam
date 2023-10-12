from typing import Optional

import torch

from segment_anything.predictor import SamPredictor


class PredictorAdaptor(SamPredictor):
    """Wrapper around the SamPredictor to be used by BioImage.IO model format.

    This model supports the same functionality as SamPredictor and can provide mask segmentations
    from box, point or mask input prompts.
    """
    def __call__(
            self,
            input_image: torch.Tensor,
            image_embeddings: Optional[torch.Tensor] = None,
            box_prompts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Expected inputs:
            - input_image: torch inputs of dimensions B x C x H x W
            - image_embeddings: precomputed image embeddings
            - box_prompts: box prompts of dimensions C x 4
        """
        # We have image embeddings set and image embeddings were not passed.
        if self.is_image_set and image_embeddings is None:
            pass   # do nothing

        # We have image embeddings set and image embeddings were passed.
        elif self.is_image_set and image_embeddings is not None:
            self.features = image_embeddings

        # We don't have image embeddings set and image embeddings were passed.
        elif image_embeddings is not None:
            self.features = image_embeddings

        # We don't have image embeddings set and they were not apassed
        elif not self.is_image_set:
            image = self.transform.apply_image_torch(input_image)
            self.set_torch_image(image, original_image_size=input_image.numpy().shape[2:])

        boxes = self.transform.apply_boxes_torch(box_prompts, original_size=input_image.numpy().shape[2:])

        masks, scores, _ = self.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes,
            multimask_output=False
        )

        assert masks.shape[2:] == input_image.shape[2:],\
            f"{masks.shape[2:]} is not as expected ({input_image.shape[2:]})"

        image_embeddings = self.features
        return masks, scores, image_embeddings
