"""
Classes for generating prompts from ground-truth segmentation masks.
For training or evaluation of prompt-based segmentation.
"""

from collections.abc import Mapping
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation


class PointAndBoxPromptGenerator:
    """Generate point and/or box prompts from an instance segmentation.

    You can use this class to derive prompts from an instance segmentation, either for
    evaluation purposes or for training Segment Anything on custom data.
    In order to use this generator you need to precompute the bounding boxes and center
    coordiantes of the instance segmentation, using e.g. `util.get_centers_and_bounding_boxes`.
    Here's an example for how to use this class:
    ```python
    # Initialize generator for 1 positive and 4 negative point prompts.
    prompt_generator = PointAndBoxPromptGenerator(1, 4, dilation_strength=8)
    # Precompute the bounding boxes for the given segmentation
    bounding_boxes, _ = util.get_centers_and_bounding_boxes(segmentation)
    # generate point prompts for the object with id 1 in 'segmentation'
    seg_id = 1
    points, point_labels, _, _ = prompt_generator(segmentation, seg_id, bounding_boxes)
    ```

    Args:
        n_positive_points: The number of positive point prompts to generate per mask.
        n_negative_points: The number of negative point prompts to generate per mask.
        dilation_strength: The factor by which the mask is dilated before generating prompts.
        get_point_prompts: Whether to generate point prompts.
        get_box_prompts: Whether to generate box prompts.
    """
    def __init__(
        self,
        n_positive_points: int,
        n_negative_points: int,
        dilation_strength: int,
        get_point_prompts: bool = True,
        get_box_prompts: bool = False
    ) -> None:
        self.n_positive_points = n_positive_points
        self.n_negative_points = n_negative_points
        self.dilation_strength = dilation_strength
        self.get_box_prompts = get_box_prompts
        self.get_point_prompts = get_point_prompts

        if self.get_point_prompts is False and self.get_box_prompts is False:
            raise ValueError("You need to request box prompts, point prompts or both.")

    def __call__(
        self,
        segmentation: np.ndarray,
        segmentation_id: int,
        bbox_coordinates: Mapping[int, tuple],
        center_coordinates: Optional[Mapping[int, np.ndarray]] = None
    ) -> tuple[
        Optional[list[tuple]], Optional[list[int]], Optional[list[tuple]], np.ndarray
    ]:
        """Generate the prompts for one object in the segmentation.

        Args:
            segmentation: The instance segmentation.
            segmentation_id: The ID of the instance.
            bbox_coordinates: The precomputed bounding boxes of all objects in the segmentation.
            center_coordinates: The precomputed center coordinates of all objects in the segmentation.
                If passed, these coordinates will be used as the first positive point prompt.
                If not passed a random point from within the object mask will be used.

        Returns:
            List of point coordinates. Returns None, if get_point_prompts is false.
            List of point labels. Returns None, if get_point_prompts is false.
            List containing the object bounding box. Returns None, if get_box_prompts is false.
            Object mask.
        """
        coord_list = []
        label_list = []

        if center_coordinates is not None:
            # getting the center coordinate as the first positive point (OPTIONAL)
            coord_list.append(tuple(map(int, center_coordinates)))  # to get int coords instead of float
            label_list.append(1)

            # getting the additional positive points by randomly sampling points
            # from this mask except the center coordinate
            n_positive_remaining = self.n_positive_points - 1

        else:
            # need to sample "self.n_positive_points" number of points
            n_positive_remaining = self.n_positive_points

        if self.get_box_prompts:
            bbox_list = [bbox_coordinates]

        object_mask = segmentation == segmentation_id

        if n_positive_remaining > 0:
            # all coordinates of our current object
            object_coordinates = np.where(object_mask)

            # ([x1, x2, ...], [y1, y2, ...])
            n_coordinates = len(object_coordinates[0])

            # randomly sampling n_positive_remaining_points from these coordinates
            positive_indices = np.random.choice(
                n_coordinates, replace=False,
                size=min(n_positive_remaining, n_coordinates)  # handles the cases with insufficient fg pixels
            )
            for positive_index in positive_indices:
                positive_coordinates = int(object_coordinates[0][positive_index]), \
                    int(object_coordinates[1][positive_index])

                coord_list.append(positive_coordinates)
                label_list.append(1)

        # getting the negative points
        # for this we do the opposite and we set the mask to the bounding box - the object mask
        # we need to dilate the object mask before doing this: we use scipy.ndimage.binary_dilation for this
        dilated_object = binary_dilation(object_mask, iterations=self.dilation_strength)
        background_mask = np.zeros(segmentation.shape)
        background_mask[bbox_coordinates[0]:bbox_coordinates[2], bbox_coordinates[1]:bbox_coordinates[3]] = 1
        background_mask = binary_dilation(background_mask, iterations=self.dilation_strength)
        background_mask = abs(
            background_mask.astype(np.float32) - dilated_object.astype(np.float32)
        )  # casting booleans to do subtraction

        n_negative_remaining = self.n_negative_points
        if n_negative_remaining > 0:
            # all coordinates of our current object
            background_coordinates = np.where(background_mask)

            # ([x1, x2, ...], [y1, y2, ...])
            n_coordinates = len(background_coordinates[0])

            # randomly sample n_positive_remaining_points from these coordinates
            negative_indices = np.random.choice(
                n_coordinates, replace=False,
                size=min(n_negative_remaining, n_coordinates)  # handles the cases with insufficient bg pixels
            )
            for negative_index in negative_indices:
                negative_coordinates = int(background_coordinates[0][negative_index]), \
                    int(background_coordinates[1][negative_index])

                coord_list.append(negative_coordinates)
                label_list.append(0)

        # returns object-level masks per instance for cross-verification (TODO: fix it later)
        if self.get_point_prompts is True and self.get_box_prompts is True:  # we want points and box
            return coord_list, label_list, bbox_list, object_mask

        elif self.get_point_prompts is True and self.get_box_prompts is False:  # we want only points
            return coord_list, label_list, None, object_mask

        elif self.get_point_prompts is False and self.get_box_prompts is True:  # we want only boxes
            return None, None, bbox_list, object_mask
