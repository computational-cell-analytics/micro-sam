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
        if self.is_image_set and image_embeddings is None:  # we have embeddings set and not passed
            pass   # do nothing
        elif self.is_image_set and image_embeddings is not None:
            raise NotImplementedError  # TODO: replace the image embeedings
        elif image_embeddings is not None:
            pass   # TODO set the image embeddings
            # self.features = image_embeddings
        elif not self.is_image_set:
            image = self.transform.apply_image_torch(input_image)
            self.set_torch_image(image, original_image_size=input_image.numpy().shape[2:])  # compute the image embeddings

        boxes = self.transform.apply_boxes_torch(box_prompts, original_size=input_image.numpy().shape[2:])  # type: ignore

        instance_segmentation, _, _ = self.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes,
            multimask_output=False
        )

        assert instance_segmentation.shape[2:] == input_image.shape[2:], f"{instance_segmentation.shape[2:]} is not as expected ({input_image.shape[2:]})"

        # TODO get the image embeddings via image_embeddings =  self.features
        # and return them
        return instance_segmentation
