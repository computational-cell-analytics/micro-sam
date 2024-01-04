"""
Classes for generating prompts from ground-truth segmentation masks.
For training or evaluation of prompt-based segmentation.
"""

from typing import List, Optional, Tuple

import numpy as np
from kornia import morphology

import torch


class PromptGeneratorBase:
    """PromptGeneratorBase is an interface to implement specific prompt generators.
    """
    def __call__(
            self,
            segmentation: torch.Tensor,
            prediction: Optional[torch.Tensor] = None,
            bbox_coordinates: Optional[List[tuple]] = None,
            center_coordinates: Optional[List[np.ndarray]] = None
    ) -> Tuple[
        Optional[torch.Tensor],  # the point coordinates
        Optional[torch.Tensor],  # the point labels
        Optional[torch.Tensor],  # the bounding boxes
        Optional[torch.Tensor],  # the mask prompts
    ]:
        """Return the point prompts given segmentation masks and optional other inputs.

        Args:
            segmentation: The object masks derived from instance segmentation groundtruth.
                Expects a float tensor of shape NUM_OBJECTS x 1 x H x W.
                The first axis corresponds to the binary object masks.
            prediction: The predicted object masks corresponding to the segmentation.
                Expects the same shape as the segmentation
            bbox_coordinates: Precomputed bounding boxes for the segmentation.
                Expects a list of length NUM_OBJECTS.
            center_coordinates: Precomputed center coordinates for the segmentation.
                Expects a list of length NUM_OBJECTS.

        Returns:
            The point prompt coordinates. Int tensor of shape NUM_OBJECTS x NUM_POINTS x 2.
                The point coordinates are retuned in XY axis order. This means they are reversed compared
                to the standard YX axis order used by numpy.
            The point prompt labels. Int tensor of shape NUM_OBJECTS x NUM_POINTS.
            The box prompts. Int tensor of shape NUM_OBJECTS x 4.
                The box coordinates are retunred as MIN_X, MIN_Y, MAX_X, MAX_Y.
            The mask prompts. Float tensor of shape NUM_OBJECTS x 1 x H' x W'.
                With H' = W'= 256.
        """
        raise NotImplementedError("PromptGeneratorBase is just a class template. \
                                  Use a child class that implements the specific generator instead")


class PointAndBoxPromptGenerator(PromptGeneratorBase):
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

    # generate point prompts for the objects with ids 1, 2 and 3
    seg_ids = (1, 2, 3)
    object_mask = np.stack([segmentation == seg_id for seg_id in seg_ids])[:, None]
    this_bounding_boxes = [bounding_boxes[seg_id] for seg_id in seg_ids]
    point_coords, point_labels, _, _ = prompt_generator(object_mask, this_bounding_boxes)
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

            # getting the additional positive points by randomly sampling points
            # from this mask except the center coordinate
            n_positive_remaining = self.n_positive_points - 1

        else:
            # need to sample "self.n_positive_points" number of points
            n_positive_remaining = self.n_positive_points

        if n_positive_remaining > 0:
            object_coordinates = torch.where(object_mask)
            n_coordinates = len(object_coordinates[0])

            # randomly sampling n_positive_remaining_points from these coordinates
            indices = np.random.choice(
                n_coordinates, size=n_positive_remaining,
                # Allow replacing if we can't sample enough coordinates otherwise
                replace=True if n_positive_remaining > n_coordinates else False,
            )
            coord_list.extend([
                [object_coordinates[0][idx], object_coordinates[1][idx]] for idx in indices
            ])

        label_list.extend([1] * self.n_positive_points)
        assert len(coord_list) == len(label_list) == self.n_positive_points
        return coord_list, label_list

    def _sample_negative_points(self, object_mask, bbox_coordinates, coord_list, label_list):
        if self.n_negative_points == 0:
            return coord_list, label_list

        # getting the negative points
        # for this we do the opposite and we set the mask to the bounding box - the object mask
        # we need to dilate the object mask before doing this: we use kornia.morphology.dilation for this
        dilated_object = object_mask[None, None]
        for _ in range(self.dilation_strength):
            dilated_object = morphology.dilation(dilated_object, torch.ones(3, 3), engine="convolution")
        dilated_object = dilated_object.squeeze()

        background_mask = torch.zeros(object_mask.shape, device=object_mask.device)
        _ds = self.dilation_strength
        background_mask[max(bbox_coordinates[0] - _ds, 0): min(bbox_coordinates[2] + _ds, object_mask.shape[-2]),
                        max(bbox_coordinates[1] - _ds, 0): min(bbox_coordinates[3] + _ds, object_mask.shape[-1])] = 1
        background_mask = torch.abs(background_mask - dilated_object)

        # the valid background coordinates
        background_coordinates = torch.where(background_mask)
        n_coordinates = len(background_coordinates[0])

        # randomly sample the negative points from these coordinates
        indices = np.random.choice(
            n_coordinates, replace=False,
            size=min(self.n_negative_points, n_coordinates)  # handles the cases with insufficient bg pixels
        )
        coord_list.extend([
            [background_coordinates[0][idx], background_coordinates[1][idx]] for idx in indices
        ])
        label_list.extend([0] * len(indices))

        return coord_list, label_list

    def _ensure_num_points(self, object_mask, coord_list, label_list):
        num_points = self.n_positive_points + self.n_negative_points

        # fill up to the necessary number of points if we did not sample enough of them
        if len(coord_list) != num_points:
            # to stay consistent, we add random points in the background of an object
            # if there's no neg region around the object - usually happens with small rois
            needed_points = num_points - len(coord_list)
            more_neg_points = torch.where(object_mask == 0)
            indices = np.random.choice(len(more_neg_points[0]), size=needed_points, replace=False)

            coord_list.extend([
                (more_neg_points[0][idx], more_neg_points[1][idx]) for idx in indices
            ])
            label_list.extend([0] * needed_points)

        assert len(coord_list) == len(label_list) == num_points
        return coord_list, label_list

    # Can we batch this properly?
    def _sample_points(self, segmentation, bbox_coordinates, center_coordinates):
        all_coords, all_labels = [], []

        center_coordinates = [None] * len(segmentation) if center_coordinates is None else center_coordinates
        for object_mask, bbox_coords, center_coords in zip(segmentation, bbox_coordinates, center_coordinates):
            coord_list, label_list = [], []
            coord_list, label_list = self._sample_positive_points(
                object_mask[0], center_coords, coord_list, label_list
            )
            coord_list, label_list = self._sample_negative_points(
                object_mask[0], bbox_coords, coord_list, label_list
            )
            coord_list, label_list = self._ensure_num_points(object_mask[0], coord_list, label_list)

            all_coords.append(coord_list)
            all_labels.append(label_list)

        return all_coords, all_labels

    def __call__(
        self,
        segmentation: torch.Tensor,
        bbox_coordinates: List[Tuple],
        center_coordinates: Optional[List[np.ndarray]] = None,
        **kwargs,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None
    ]:
        """Generate the prompts for one object in the segmentation.

        Args:
            The groundtruth segmentation. Expects a float tensor of shape NUM_OBJECTS x 1 x H x W.
            bbox_coordinates: The precomputed bounding boxes of particular object in the segmentation.
            center_coordinates: The precomputed center coordinates of particular object in the segmentation.
                If passed, these coordinates will be used as the first positive point prompt.
                If not passed a random point from within the object mask will be used.

        Returns:
            Coordinates of point prompts. Returns None, if get_point_prompts is false.
            Point prompt labels. Returns None, if get_point_prompts is false.
            Bounding box prompts. Returns None, if get_box_prompts is false.
        """
        if self.get_point_prompts:
            coord_list, label_list = self._sample_points(segmentation, bbox_coordinates, center_coordinates)
            # change the axis convention of the point coordinates to match the expected coordinate order of SAM
            coord_list = np.array(coord_list)[:, :, ::-1].copy()
            coord_list = torch.from_numpy(coord_list)
            label_list = torch.tensor(label_list)
        else:
            coord_list, label_list = None, None

        if self.get_box_prompts:
            # change the axis convention of the point coordinates to match the expected coordinate order of SAM
            bbox_list = np.array(bbox_coordinates)[:, [1, 0, 3, 2]]
            bbox_list = torch.from_numpy(bbox_list)
        else:
            bbox_list = None

        return coord_list, label_list, bbox_list, None


class IterativePromptGenerator(PromptGeneratorBase):
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
    def _get_negative_points(self, negative_region_batched, true_object_batched):
        device = negative_region_batched.device

        negative_coordinates, negative_labels = [], []
        for neg_region, true_object in zip(negative_region_batched, true_object_batched):

            tmp_neg_loc = torch.where(neg_region)
            if torch.stack(tmp_neg_loc).shape[-1] == 0:
                tmp_true_loc = torch.where(true_object)
                x_coords, y_coords = tmp_true_loc[1], tmp_true_loc[2]
                bbox = torch.stack([torch.min(x_coords), torch.min(y_coords),
                                    torch.max(x_coords) + 1, torch.max(y_coords) + 1])
                bbox_mask = torch.zeros_like(true_object).squeeze(0)

                custom_df = 3  # custom dilation factor to perform dilation by expanding the pixels of bbox
                bbox_mask[max(bbox[0] - custom_df, 0): min(bbox[2] + custom_df, true_object.shape[-2]),
                          max(bbox[1] - custom_df, 0): min(bbox[3] + custom_df, true_object.shape[-1])] = 1
                bbox_mask = bbox_mask[None].to(device)

                background_mask = torch.abs(bbox_mask - true_object)
                tmp_neg_loc = torch.where(background_mask)

                # there is a chance that the object is small to not return a decent-sized bounding box
                # hence we might not find points sometimes there as well, hence we sample points from true background
                if torch.stack(tmp_neg_loc).shape[-1] == 0:
                    tmp_neg_loc = torch.where(true_object == 0)

            neg_index = np.random.choice(len(tmp_neg_loc[1]))
            neg_coordinates = [tmp_neg_loc[1][neg_index], tmp_neg_loc[2][neg_index]]
            neg_coordinates = neg_coordinates[::-1]
            neg_labels = 0

            negative_coordinates.append(neg_coordinates)
            negative_labels.append(neg_labels)

        return negative_coordinates, negative_labels

    def __call__(
        self,
        segmentation: torch.Tensor,
        prediction: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """Generate the prompts for each object iteratively in the segmentation.

        Args:
            The groundtruth segmentation. Expects a float tensor of shape NUM_OBJECTS x 1 x H x W.
            The predicted objects. Epects a float tensor of the same shape as the segmentation.

        Returns:
            The updated point prompt coordinates.
            The updated point prompt labels.
        """
        assert segmentation.shape == prediction.shape
        device = prediction.device

        true_object = segmentation.to(device)
        expected_diff = (prediction - true_object)
        neg_region = (expected_diff == 1).to(torch.float32)
        pos_region = (expected_diff == -1)
        overlap_region = torch.logical_and(prediction == 1, true_object == 1).to(torch.float32)

        pos_coordinates, pos_labels = self._get_positive_points(pos_region, overlap_region)
        neg_coordinates, neg_labels = self._get_negative_points(neg_region, true_object)
        assert len(pos_coordinates) == len(pos_labels) == len(neg_coordinates) == len(neg_labels)

        pos_coordinates = torch.tensor(pos_coordinates)[:, None]
        neg_coordinates = torch.tensor(neg_coordinates)[:, None]
        pos_labels, neg_labels = torch.tensor(pos_labels)[:, None], torch.tensor(neg_labels)[:, None]

        net_coords = torch.cat([pos_coordinates, neg_coordinates], dim=1)
        net_labels = torch.cat([pos_labels, neg_labels], dim=1)

        return net_coords, net_labels, None, None
