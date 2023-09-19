"""
Classes for generating prompts from ground-truth segmentation masks.
For training or evaluation of prompt-based segmentation.
"""

from collections.abc import Mapping
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

import torch


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

    def _sample_positive_points(self, object_mask, center_coordinates, coord_list, label_list):
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

        return coord_list, label_list

    def _sample_negative_points(self, object_mask, bbox_coordinates, coord_list, label_list):
        # getting the negative points
        # for this we do the opposite and we set the mask to the bounding box - the object mask
        # we need to dilate the object mask before doing this: we use scipy.ndimage.binary_dilation for this
        dilated_object = binary_dilation(object_mask, iterations=self.dilation_strength)
        background_mask = np.zeros(object_mask.shape)
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

        return coord_list, label_list

    def _ensure_num_points(self, object_mask, coord_list, label_list):
        num_points = self.n_positive_points + self.n_negative_points

        # fill up to the necessary number of points if we did not sample enough of them
        if len(coord_list) != num_points:
            # to stay consistent, we add random points in the background of an object
            # if there's no neg region around the object - usually happens with small rois
            needed_points = num_points - len(coord_list)
            more_neg_points = np.where(object_mask == 0)
            chosen_idx = np.random.choice(len(more_neg_points[0]), size=needed_points)

            coord_list.extend([
                (more_neg_points[0][idx], more_neg_points[1][idx]) for idx in chosen_idx
            ])
            label_list.extend([0] * needed_points)

        assert len(coord_list) == len(label_list) == num_points
        return coord_list, label_list

    def _sample_points(self, object_mask, bbox_coordinates, center_coordinates):
        coord_list, label_list = [], []

        coord_list, label_list = self._sample_positive_points(object_mask, center_coordinates, coord_list, label_list)
        coord_list, label_list = self._sample_negative_points(object_mask, bbox_coordinates, coord_list, label_list)
        coord_list, label_list = self._ensure_num_points(object_mask, coord_list, label_list)

        return coord_list, label_list

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
        object_mask = segmentation == segmentation_id

        if self.get_point_prompts:
            coord_list, label_list = self._sample_points(object_mask, bbox_coordinates, center_coordinates)
        else:
            coord_list, label_list = None, None

        if self.get_box_prompts:
            bbox_list = [bbox_coordinates]
        else:
            bbox_list = None

        return coord_list, label_list, bbox_list, object_mask


class IterativePromptGenerator:
    """Generate point prompts from an instance segmentation iteratively.
    """
    def _get_positive_points(self, pos_region, overlap_region):
        positive_locations = [torch.where(pos_reg) for pos_reg in pos_region]
        # we may have objects without a positive region (= missing true foreground)
        # in this case we just sample a point where the model was already correct
        positive_locations = [
            torch.where(ovlp_reg) if len(pos_loc[0]) == 0 else pos_loc
            for pos_loc, ovlp_reg in zip(positive_locations, overlap_region)
        ]
        # we sample one location for each object in the batch
        sampled_indices = [np.random.choice(len(pos_loc[0])) for pos_loc in positive_locations]
        # get the corresponding coordinates (Note that we flip the axis order here due to the expected order of SAM)
        pos_coordinates = [
            [pos_loc[-1][idx], pos_loc[-2][idx]] for pos_loc, idx in zip(positive_locations, sampled_indices)
        ]

        # make sure that we still have the correct batch size
        assert len(pos_coordinates) == pos_region.shape[0]
        pos_labels = [1] * len(pos_coordinates)

        return pos_coordinates, pos_labels

    # TODO get rid of this looped implementation and use proper batched computation instead
    def _get_negative_points(self, negative_region_batched, true_object_batched, gt_batched):
        device = negative_region_batched.device

        negative_coordinates, negative_labels = [], []
        for neg_region, true_object, gt in zip(negative_region_batched, true_object_batched, gt_batched):

            tmp_neg_loc = torch.where(neg_region)
            if torch.stack(tmp_neg_loc).shape[-1] == 0:
                tmp_true_loc = torch.where(true_object)
                x_coords, y_coords = tmp_true_loc[1], tmp_true_loc[2]
                bbox = torch.stack([torch.min(x_coords), torch.min(y_coords),
                                    torch.max(x_coords) + 1, torch.max(y_coords) + 1])
                bbox_mask = torch.zeros_like(true_object).squeeze(0)

                custom_df = 3  # custom dilation factor to perform dilation by expanding the pixels of bbox
                bbox_mask[max(bbox[0] - custom_df, 0): min(bbox[2] + custom_df, gt.shape[-1]),
                          max(bbox[1] - custom_df, 0): min(bbox[3] + custom_df, gt.shape[-2])] = 1
                bbox_mask = bbox_mask[None].to(device)

                background_mask = abs(bbox_mask - true_object)
                tmp_neg_loc = torch.where(background_mask)

                # there is a chance that the object is small to not return a decent-sized bounding box
                # hence we might not find points sometimes there as well, hence we sample points from true background
                if torch.stack(tmp_neg_loc).shape[-1] == 0:
                    tmp_neg_loc = torch.where(gt == 0)

            neg_index = np.random.choice(len(tmp_neg_loc[1]))
            neg_coordinates = [tmp_neg_loc[1][neg_index], tmp_neg_loc[2][neg_index]]
            neg_coordinates = neg_coordinates[::-1]
            neg_labels = 0

            negative_coordinates.append(neg_coordinates)
            negative_labels.append(neg_labels)

        return negative_coordinates, negative_labels

    def __call__(
        self,
        gt: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the prompts for each object iteratively in the segmentation.

        Args:
            The groundtruth segmentation.
            The predicted objects.

        Returns:
            The updated point prompt coordinates.
            The updated point prompt labels.
        """
        assert gt.shape == object_mask.shape
        device = object_mask.device

        true_object = gt.to(device)
        expected_diff = (object_mask - true_object)
        neg_region = (expected_diff == 1).to(torch.float32)
        pos_region = (expected_diff == -1)
        overlap_region = torch.logical_and(object_mask == 1, true_object == 1).to(torch.float32)

        pos_coordinates, pos_labels = self._get_positive_points(pos_region, overlap_region)
        neg_coordinates, neg_labels = self._get_negative_points(neg_region, true_object, gt)
        assert len(pos_coordinates) == len(pos_labels) == len(neg_coordinates) == len(neg_labels)

        pos_coordinates = torch.tensor(pos_coordinates)[:, None]
        neg_coordinates = torch.tensor(neg_coordinates)[:, None]
        pos_labels, neg_labels = torch.tensor(pos_labels)[:, None], torch.tensor(neg_labels)[:, None]

        net_coords = torch.cat([pos_coordinates, neg_coordinates], dim=1)
        net_labels = torch.cat([pos_labels, neg_labels], dim=1)

        return net_coords, net_labels
