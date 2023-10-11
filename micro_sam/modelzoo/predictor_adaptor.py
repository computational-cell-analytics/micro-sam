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
        if self.is_image_set and image_embeddings is None:  # we have embeddings set and not passed
            pass   # do nothing
        elif self.is_image_set and image_embeddings is not None:
            raise NotImplementedError  # TODO: replace the image embeedings
        elif image_embeddings is not None:
            pass   # TODO set the image embeddings
            # self.features = image_embeddings
        elif not self.is_image_set:
            self.set_torch_image(input_image)   # compute the image embeddings

        instance_segmentation, _, _ = self.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=box_prompts,
            multimask_output=False
        )
        # TODO get the image embeddings via image_embeddings =  self.features
        # and return them
        return instance_segmentation
