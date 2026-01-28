from typing import Optional, Union, List

import numpy as np

from micro_sam.prompt_based_segmentation import _process_box


def promptable_segmentation_2d(
    predictor,
    image: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    boxes: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    batched: Optional[bool] = None,
):
    """@private"""

    if image is not None:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        assert image.ndim == 3

        if image.shape[0] == 3:  # Make channels last, as expected in RGB images.
            image = image.transpose(1, 2, 0)

        # Set the predictor state.
        predictor.set_image(image.astype("uint8"))

    assert len(points) == len(labels)
    have_points = points is not None and len(points) > 0
    have_boxes = boxes is not None and len(boxes) > 0

    # If no prompts are provided, return 'None'.
    if not have_points and not have_boxes:
        return

    kwargs = {}
    if have_points:
        kwargs["point_coords"] = points[:, ::-1].copy()  # Ensure contiguous array convention so that PyTorch likes it.
        kwargs["point_labels"] = labels
    if have_boxes:
        shape = predictor._orig_hw[0]
        kwargs["box"] = np.array([_process_box(b, shape) for b in boxes])

    # Run interactive segmentation.
    masks, scores, logits = predictor.predict(
        # mask_input=masks,
        multimask_output=False,  # NOTE: Hard-coded to 'False' atm.
        **kwargs
    )

    # Get the count of points / boxes.
    n_points = len(points) if have_points else 0
    n_boxes = len(boxes) if have_boxes else 0

    if n_points > 1 or n_boxes > 1:  # Has more than one object, expected instance segmentation.
        out = np.zeros(masks.shape[-2:])
        for i, curr_mask in enumerate(masks, start=1):
            out[curr_mask.squeeze() > 0] = i
    else:
        out = masks.squeeze()

    # HACK: Hard-code the expected data type for labels for napari labels layer: uint8
    out = out.astype("uint8")

    return out


def promptable_segmentation_3d(
    predictor,
    volume: np.ndarray,
    frame_id: int,
    volume_embeddings: Optional[...] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    boxes: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
):
    """@private"""

    assert volume.ndim == 3

    # Initialize the inference state
    inference_state = predictor.init_state(video_path=None, volume=volume)

    assert len(points) == len(labels)
    have_points = points is not None and len(points) > 0
    have_boxes = boxes is not None and len(boxes) > 0

    # If no prompts are provided, return 'None'.
    if not have_points and not have_boxes:
        return

    kwargs = {}
    if have_points:
        kwargs["points"] = points[:, ::-1].copy()  # Ensure contiguous array convention so that PyTorch likes it.
        kwargs["labels"] = labels
    if have_boxes:
        shape = volume.shape[-2:]
        kwargs["box"] = np.array([_process_box(b, shape) for b in boxes])

    # Add point/box prompts in a single frame.
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=int(frame_id),
        obj_id=1,  # NOTE: Setting a fixed object id, assuming only one object is being segmented.
        clear_old_points=True,  # Whether to make use of old points in memory.
        **kwargs
    )

    # TODO: Figure out how to integrate mask prompts in 3d.

    # Next, propagate the masklets throughout the frames using the input prompts in selected frames.
    forward_video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        forward_video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Let's do the propagation reverse in time now.
    reverse_video_segments = {}
    if len(forward_video_segments) < volume.shape[0]:  # Perform reverse propagation only if necessary
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, reverse=True,
        ):
            reverse_video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }
        # NOTE: The order is reversed to stitch the reverse propagation with forward.
        reverse_video_segments = dict(reversed(list(reverse_video_segments.items())))

    # We stitch the segmented slices together.
    video_segments = {**reverse_video_segments, **forward_video_segments}

    # Now, let's merge the segmented objects per frame back together as instances per slice.
    segmentation = []
    for slice_idx in video_segments.keys():
        per_slice_seg = np.zeros(volume.shape[-2:])
        for _instance_idx, _instance_mask in video_segments[slice_idx].items():
            per_slice_seg[_instance_mask.squeeze()] = _instance_idx
        segmentation.append(per_slice_seg)

    segmentation = (np.stack(segmentation) > 0).astype("uint64")

    # Reset the state after finishing the segmentation round.
    predictor.reset_state(inference_state)

    return segmentation


class PromptableSegmentation3D:
    """Promptable segmentation class for volumetric data.
    """
    def __init__(self, predictor, volume, volume_embeddings=None):
        self.predictor = predictor
        self.volume = volume
        self.volume_embeddings = volume_embeddings

        if self.volume.ndim != 3:
            raise AssertionError(f"The dimensionality of the volume should be 3, got '{self.volume.ndim}'")

        self.init_predictor()

        # Store prompts per instance.
        self.running_point_frame_ids: Optional[Union[List[int]]] = None
        self.running_points: Optional[np.ndarray] = None
        self.running_point_labels: Optional[np.ndarray] = None

        self.running_box_frame_ids: Optional[Union[int, List[int]]] = None
        self.running_boxes: Optional[np.ndarray] = None

        self.running_mask_frame_ids: Optional[Union[int, List[int]]] = None
        self.running_masks: Optional[np.ndarray] = None

    def init_predictor(self):
        # Initialize the inference state.
        self.inference_state = self.predictor.init_state(volume=self.volume, volume_embeddings=self.volume_embeddings)

    def reset_predictor(self):
        # Reset the state after finishing the segmentation round.
        self.predictor.reset_state(self.inference_state)

    def _as_array(self, x):
        return None if x is None else np.asarray(x)

    def _is_array_equal(self, a, b):
        if a is None and b is None:
            return True
        if (a is None) != (b is None):
            return False
        a = np.asarray(a)
        b = np.asarray(b)
        return a.shape == b.shape and np.array_equal(a, b)

    def _is_prefix(self, old, new) -> bool:
        """Checks whether the new object is a prefix element of the older object."""
        if old is None:
            return True
        if new is None:
            return False

        old = np.asarray(old)
        new = np.asarray(new)

        if old.ndim == 0 or new.ndim == 0:
            return False
        if old.shape[1:] != new.shape[1:]:
            return False
        if old.shape[0] > new.shape[0]:
            return False
        return np.array_equal(old, new[: old.shape[0]])

    def _tail(self, old, new):
        """Returns the trailing tail by eliminating the prefix from new object."""
        if old is None:
            return new
        old = np.asarray(old)
        new = np.asarray(new)
        if old.shape[0] == new.shape[0]:
            return None
        return new[old.shape[0]:]

    def get_valid_prompts(self, frame_ids, points=None, labels=None, boxes=None, masks=None):
        """Returns the valid prompts to add for promptable segmentation.

        This workflow manages and returns prompts to add to make sure of the following:
        1. Either use new (unused) prompts, or.
        2. Reprompt all prompts in case an old prompt is deleted.
        """
        have_points = (points is not None) or (labels is not None)
        if have_points and (points is None or labels is None):
            raise ValueError("For using point prompts, both 'points' and 'labels' must be provided.")
        have_boxes = boxes is not None
        have_masks = masks is not None

        valid_prompt_combinations = sum([have_points, have_boxes, have_masks])
        if valid_prompt_combinations == 0:
            raise ValueError("You must provide a valid prompt combination.")
        elif valid_prompt_combinations > 1:
            raise ValueError("Please choose only one of the prompt combinations.")

        # The core manager for maintaining prompts in memory and returning valid prompts.
        if have_points:
            points = self._as_array(points)
            labels = self._as_array(labels)

            # Let's perform a quick point prompt sanity check.
            if points.ndim == 0:
                raise ValueError("'points' must be array-like, not a scalar.")
            if labels.ndim == 0:
                raise ValueError("'labels' must be array-like, not a scalar.")
            if points.shape[0] == 0:
                raise ValueError("'points' must contain at least one point.")
            if labels.shape[0] != points.shape[0]:
                raise ValueError("'labels' must have the same length as 'points'.")

            # If the prompt arrive here for the first time, remember me :)
            if self.running_point_frame_ids is None:
                self.running_point_frame_ids = frame_ids
                self.running_points = points
                self.running_point_labels = labels
                return {"mode": "all", "frame_ids": frame_ids, "points": points, "labels": labels}

            # If the 'frame_ids' change, the safest would be to reprompt all and overwrite.
            if frame_ids != self.running_point_frame_ids:
                self.running_point_frame_ids = frame_ids
                self.running_points = points
                self.running_point_labels = labels
                return {"mode": "all", "frame_ids": frame_ids, "points": points, "labels": labels}

            # If the prompt arrive and exactly match the stored prompts, return no prompts.
            if (
                self._is_array_equal(points, self.running_points) and
                self._is_array_equal(labels, self.running_point_labels)
            ):
                return {}

            # If the prompts arrive and have some new prompts compared to stored ones, only return the new ones.
            if self._is_prefix(self.running_points, points) and self._is_prefix(self.running_point_labels, labels):
                new_points = self._tail(self.running_points, points)
                new_labels = self._tail(self.running_point_labels, labels)

                # Let's update the prompt storage to the full incoming prompts.
                self.running_points = points
                self.running_point_labels = labels

                if new_points is None:
                    return {}
                return {"mode": "tail", "frame_ids": frame_ids, "points": new_points, "labels": new_labels}

            # If the prompts arrive and have some old stored prompts deleted, return all arrived prompts as is.
            # NOTE: It could be deletion / modification / reordering, we simply reprompt all prompts again.
            self.running_points = points
            self.running_point_labels = labels
            return {"mode": "all", "frame_ids": frame_ids, "points": points, "labels": labels}

    def add_point_prompts(
        self,
        frame_ids: Union[int, List[int]],
        points: np.ndarray,
        point_labels: np.ndarray,
        object_id: Optional[Union[List[int], int]] = None,
        multiple_objects: bool = False,  # Enables multi-object segmentation.
    ):
        """
        """
        # Support multi-object segmentation.
        if multiple_objects and object_id is not None:
            raise ValueError("Well you can't segment multiple objects and provide a specific id, duh!")

        # In case there is no multi-object segmentation happening and the user forgot to specify object, pin obj_id=1.
        if object_id is None:
            object_id = 1

        # If no point prompts are provided, return 'None'.
        if points is None or len(points) == 0:
            return

        # Check what's been provided by the user.
        if not isinstance(frame_ids, list):
            frame_ids = [frame_ids]

        if len(points) != len(point_labels):
            raise AssertionError("The number of points and corresponding labels for it are mismatching.")

        # Prepare the point prompts.
        expected_prompts = self.get_valid_prompts(frame_ids=frame_ids, points=points, labels=point_labels)
        if not expected_prompts:  # If there are no new prompts, we should not add them.
            return

        mode = expected_prompts["mode"]
        frame_ids = expected_prompts["frame_ids"]
        points = expected_prompts["points"]
        point_labels = expected_prompts["labels"]

        clear_old_points = (mode == "all")  # TODO: Make use of this in a smarter way!
        points = points[:, ::-1].copy()  # Ensure contiguous array convention so that PyTorch likes it.

        # Make object ids consistent to our per-prompt addition strategy
        if not isinstance(object_id, list):
            object_id = [object_id]

        # Now that we have lists, they should match the total number of prompts (hint: going towards multiple objects)
        if len(object_id) != len(point_labels) and len(object_id == 1):
            object_id = object_id * len(point_labels)

        # At this stage, the length of points, point_labels and object_id should match.
        assert len(object_id) == len(point_labels) == len(points), "Number of object ids should match total prompts."

        # Add point prompts in a particular frame.
        for i, (curr_frame_id, curr_point, curr_point_label, curr_obj_id) in enumerate(
            zip(frame_ids, points, point_labels, object_id)
        ):
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=int(curr_frame_id),
                obj_id=curr_obj_id,  # NOTE: Setting a fixed object id, assuming only one object is being segmented.
                clear_old_points=False,  # HACK: Hard-coded atm # Whether to make use of old points in memory.
                points=np.array([curr_point]),
                labels=np.array([curr_point_label]),
            )

    def add_box_prompts(self, frame_ids: Union[int, List[int]], boxes: Optional[np.ndarray] = None):
        # Check what's been provided by the user.
        have_boxes = boxes is not None and len(boxes) > 0

        # If no boxes prompts are provided, return 'None'.
        if not have_boxes:
            return

        if not isinstance(frame_ids, List):
            frame_ids = [frame_ids]

        # Prepare the box prompts.
        # TODO: Validate based on running prompts.
        clear_old_points = True  # TODO: Must depend on the running prompt logic.
        boxes = np.array([_process_box(b, self.volume.shape[-2:]) for b in boxes])

        # Add box prompts in a particular frame.
        for curr_frame_id, curr_box in zip(frame_ids, boxes):
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=int(curr_frame_id),
                obj_id=1,  # NOTE: Setting a fixed object id, assuming only one object is being segmented.
                clear_old_points=clear_old_points,  # Whether to make use of old points in memory.
                box=np.array([curr_box]),
            )

    def add_mask_prompts(
        self, frame_ids: Union[int, List[int]], masks: Optional[np.ndarray] = None,
    ):
        raise NotImplementedError

    def propagate_prompts(self):
        # First, we propagate the masklets throughout the frames using the input prompts in selected frames.
        forward_video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            forward_video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Next, we do the propagation reverse in time.
        reverse_video_segments = {}
        if len(forward_video_segments) < self.volume.shape[0]:  # Perform reverse propagation only if necessary
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, reverse=True,
            ):
                reverse_video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                }
            # NOTE: The order is reversed to stitch the reverse propagation with forward.
            reverse_video_segments = dict(reversed(list(reverse_video_segments.items())))

        # Now, we should stitch the segmented slices together.
        video_segments = {**reverse_video_segments, **forward_video_segments}
        return video_segments

    def segment_slice(
        self,
        frame_idx: int,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[List] = None,
        masks: Optional[List] = None,
        object_id: int = 1,
    ):
        """Segment a single slice using SAM2 video predictor.

        Args:
            frame_idx: Slice index to segment.
            points: Point prompts (N, 2) array.
            labels: Point labels (N,) array.
            boxes: List of box prompts.
            masks: List of mask prompts (can be None).
            object_id: Object ID to use for the segmentation (default: 1).

        Returns:
            Segmentation mask for the slice (2D array), or None if no valid prompts provided.
        """
        # Validate prompts
        have_points = points is not None and len(points) > 0
        have_boxes = boxes is not None and len(boxes) > 0

        if not have_points and not have_boxes:
            return None

        try:
            # Prepare prompts
            box = boxes[0] if have_boxes else None

            # Add prompts to the specific frame
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                points=points if have_points else None,
                labels=labels if have_points else None,
                box=box,
            )

            # Extract the mask from logits
            # out_mask_logits shape: (num_objects, 1, H, W)
            mask_logits = out_mask_logits[0]  # Get first object
            seg = (mask_logits.squeeze() > 0.0).cpu().numpy()

            # Ensure correct output type
            seg = seg.astype("uint32")

        finally:
            # Reset the state to clear this object's prompts
            # This ensures the next segmentation starts fresh
            self.predictor.reset_state(self.inference_state)

        return seg

    def predict(self):
        # First, we propagate prompts.
        video_segments = self.propagate_prompts()

        # Next, let's merge the segmented objects per frame back together as instances per slice.
        segmentation = []
        for slice_idx in video_segments.keys():
            per_slice_seg = np.zeros(self.volume.shape[-2:])
            for _instance_idx, _instance_mask in video_segments[slice_idx].items():
                per_slice_seg[_instance_mask.squeeze()] = _instance_idx
            segmentation.append(per_slice_seg)

        segmentation = np.stack(segmentation).astype("uint64")

        return segmentation
