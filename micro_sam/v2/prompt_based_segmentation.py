from typing import Optional, Union, List

import numpy as np

from micro_sam.v1.prompt_based_segmentation import _process_box


def _crop_to_original_shape(mask, shape):
    """Crop a SAM2 video-predictor mask back to the original slice shape.

    The video predictor pads non-square frames to a square of side max(H, W) (padding appended at the
    bottom/right) and returns masks at that padded size. The image content occupies the top-left
    [0:H, 0:W] region, so cropping recovers the original (H, W) mask. For square volumes this is a no-op.
    """
    return mask[:shape[0], :shape[1]]


def _tile_index_for(tiling, halo, y, x):
    """Return the id of the tile whose inner (halo-free) block contains the point (y, x)."""
    for tile_id in range(tiling.number_of_blocks):
        inner = tiling.get_block_with_halo(tile_id, list(halo)).inner_block
        if inner.begin[0] <= y < inner.end[0] and inner.begin[1] <= x < inner.end[1]:
            return tile_id
    return 0


def _inner_block_slices(tiling, halo, tile_id):
    """Return the (local, global) inner-block slices for placing a tile result into the full array."""
    block = tiling.get_block_with_halo(tile_id, list(halo))
    local = tuple(slice(b, e) for b, e in zip(block.inner_block_local.begin, block.inner_block_local.end))
    glob = tuple(slice(b, e) for b, e in zip(block.inner_block.begin, block.inner_block.end))
    return local, glob


def _box_to_tiles(tiling, halo, box):
    """Assign a box to every tile whose inner block it overlaps, clipped to the tile's outer block.

    A single box can span several tiles (e.g. at a 4-tile junction); each tile segments the box's
    portion that falls in it and the results are unioned. The box and the returned clipped boxes are
    in (y0, x0, y1, x1) order.

    Returns:
        Mapping from tile id to the clipped box (in global coordinates).
    """
    y0, x0, y1, x1 = box
    assignments = {}
    for tile_id in range(tiling.number_of_blocks):
        block = tiling.get_block_with_halo(tile_id, list(halo))
        inner, outer = block.inner_block, block.outer_block
        # Skip tiles whose inner (halo-free) block the box does not overlap.
        if y1 <= inner.begin[0] or y0 >= inner.end[0] or x1 <= inner.begin[1] or x0 >= inner.end[1]:
            continue
        assignments[tile_id] = np.array([
            max(y0, outer.begin[0]), max(x0, outer.begin[1]),
            min(y1, outer.end[0]), min(x1, outer.end[1]),
        ])
    return assignments


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

    # Batched multi-object segmentation: each positive point and each box defines a separate object.
    if batched:
        return _batched_promptable_segmentation_2d(predictor, points, labels, boxes)

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


def _batched_promptable_segmentation_2d(predictor, points, labels, boxes):
    """Batched 2D segmentation where each positive point and each box defines a separate object.

    Negative points are shared as negative prompts for every object. This matches the batched
    convention of the SAM v1 annotator and the 3D unified segment widget.
    """
    shape = predictor._orig_hw[0]

    points = np.zeros((0, 2)) if points is None else np.asarray(points)
    labels = np.zeros((0,), dtype=int) if labels is None else np.asarray(labels)
    positive_points = points[labels == 1]
    negative_points = points[labels != 1]
    n_neg = len(negative_points)

    seg = np.zeros(tuple(shape), dtype="uint32")
    object_id = 0

    def _assign(masks):
        nonlocal object_id
        masks = np.asarray(masks)
        if masks.ndim == 3:  # A single object is returned as (C, H, W); add the object dimension.
            masks = masks[None]
        for curr_mask in masks:
            object_id += 1
            seg[curr_mask[0] > 0] = object_id

    # One object per positive point, each combined with the shared negative points.
    if len(positive_points) > 0:
        obj_points, obj_labels = [], []
        for pos in positive_points:
            pts = np.concatenate([pos[None], negative_points], axis=0) if n_neg else pos[None]
            lbs = np.concatenate([[1], np.zeros(n_neg, dtype=int)]) if n_neg else np.array([1])
            obj_points.append(pts)
            obj_labels.append(lbs)
        # Reverse the last axis to convert (row, col) to the (x, y) convention SAM2 expects.
        batched_points = np.stack(obj_points)[..., ::-1].copy()
        batched_labels = np.stack(obj_labels)
        masks, _, _ = predictor.predict(
            point_coords=batched_points, point_labels=batched_labels, multimask_output=False,
        )
        _assign(masks)

    # One object per box, each combined with the shared negative points.
    if boxes is not None and len(boxes) > 0:
        processed_boxes = np.array([_process_box(b, shape) for b in boxes])
        kwargs = {"box": processed_boxes, "multimask_output": False}
        if n_neg:
            neg = np.repeat(negative_points[None], len(processed_boxes), axis=0)[..., ::-1].copy()
            kwargs["point_coords"] = neg
            kwargs["point_labels"] = np.zeros((len(processed_boxes), n_neg), dtype=int)
        masks, _, _ = predictor.predict(**kwargs)
        _assign(masks)

    if object_id == 0:
        return None

    return seg


def tiled_promptable_segmentation_2d(
    predictor,
    image_embeddings: dict,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    boxes: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    batched: Optional[bool] = None,
):
    """Tiled 2d promptable segmentation for the SAM2 image predictor.

    Routes the prompts to the tile-column they fall in, sets that tile's precomputed embeddings on
    the predictor (via `set_precomputed`), runs `promptable_segmentation_2d` on the tile, and
    stitches the per-tile mask into the full image. Points are in (y, x) order, as passed by the
    annotator. Same return convention as `promptable_segmentation_2d`.
    """
    from bioimage_cpp.utils import Blocking
    from micro_sam.v2.util import set_precomputed

    feats = image_embeddings["features"]
    shape = tuple(int(s) for s in feats.attrs["shape"])
    tile_shape = tuple(int(s) for s in feats.attrs["tile_shape"])
    halo = tuple(int(s) for s in feats.attrs["halo"])
    tiling = Blocking([0, 0], list(shape), list(tile_shape))

    have_points = points is not None and len(points) > 0
    have_boxes = boxes is not None and len(boxes) > 0
    if not have_points and not have_boxes:
        return None

    # Group the prompts by the tile each falls in, so an object spanning multiple tiles is segmented
    # in every tile it has prompts in. Points are (y, x); boxes are (y0, x0, y1, x1).
    tile_points, tile_labels, tile_boxes = {}, {}, {}
    if have_points:
        for point, label in zip(np.asarray(points), np.asarray(labels)):
            tid = _tile_index_for(tiling, halo, int(round(point[0])), int(round(point[1])))
            tile_points.setdefault(tid, []).append(point)
            tile_labels.setdefault(tid, []).append(label)
    if have_boxes:
        for box in boxes:
            # A box may span several tiles; segment its clipped portion in each (boxes are y0,x0,y1,x1).
            for tid, clipped in _box_to_tiles(tiling, halo, np.asarray(box)).items():
                tile_boxes.setdefault(tid, []).append(clipped)

    out = np.zeros(shape, dtype="uint32")
    found = False
    for tile_id in sorted(set(tile_points) | set(tile_boxes)):
        tpoints = np.asarray(tile_points.get(tile_id, [])).reshape(-1, 2)
        tlabels = np.asarray(tile_labels.get(tile_id, []), dtype=int)
        tboxes = tile_boxes.get(tile_id, [])
        # Only segment tiles that have a positive cue (a positive point or a box); a tile with only
        # negative points has nothing to segment there.
        if not ((tlabels == 1).any() or len(tboxes) > 0):
            continue

        outer = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
        y0, x0 = int(outer.begin[0]), int(outer.begin[1])
        local_points = (tpoints - np.array([y0, x0])) if len(tpoints) else tpoints
        local_boxes = [b - np.array([y0, x0, y0, x0]) for b in tboxes] if tboxes else None

        set_precomputed(predictor, image_embeddings, tile_id=tile_id)
        tile_seg = promptable_segmentation_2d(
            predictor, image=None, points=local_points, labels=tlabels, boxes=local_boxes,
            masks=masks, batched=batched,
        )
        if tile_seg is None:
            continue

        local, glob = _inner_block_slices(tiling, halo, tile_id)
        region = tile_seg[local]
        # Union the per-tile result into the output, preserving object ids (an object spanning tiles
        # keeps the same id and is merged across the tile boundary).
        sub = out[glob]
        positive = region > 0
        sub[positive] = region[positive]
        found = True

    return out if found else None


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
            per_slice_seg[_crop_to_original_shape(_instance_mask.squeeze(), volume.shape[-2:])] = _instance_idx
        segmentation.append(per_slice_seg)

    segmentation = (np.stack(segmentation) > 0).astype("uint64")

    # Reset the state after finishing the segmentation round.
    predictor.reset_state(inference_state)

    return segmentation


class PromptableSegmentation3D:
    """Promptable segmentation class for volumetric data.
    """
    def __init__(
        self, predictor, volume, volume_embeddings, device=None,
        offload_video_to_cpu=True, offload_state_to_cpu=True,
    ):
        self.predictor = predictor
        self.volume = volume
        self.volume_embeddings = volume_embeddings
        # 'device=None' uses the predictor's auto-detected device. Offloading the frames and tracking
        # state to CPU keeps GPU memory bounded for large volumes (a no-op when already on CPU).
        self.device = device
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu

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
        self.inference_state = self.predictor.init_state(
            volume=self.volume, volume_embeddings=self.volume_embeddings, device=self.device,
            offload_video_to_cpu=self.offload_video_to_cpu, offload_state_to_cpu=self.offload_state_to_cpu,
        )

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

    def add_box_prompts(
        self,
        frame_ids: Union[int, List[int]],
        boxes: Optional[np.ndarray] = None,
        object_id: Optional[Union[int, List[int]]] = None,
    ):
        # Check what's been provided by the user.
        have_boxes = boxes is not None and len(boxes) > 0

        # If no boxes prompts are provided, return 'None'.
        if not have_boxes:
            return

        if not isinstance(frame_ids, list):
            frame_ids = [frame_ids]
        # A single frame id applies to every box (the annotator passes one frame with all its boxes).
        if len(frame_ids) == 1:
            frame_ids = frame_ids * len(boxes)

        # One object id per box. Default to a single object (id 1) for non-batched segmentation.
        if object_id is None:
            object_id = [1] * len(boxes)
        elif not isinstance(object_id, list):
            object_id = [object_id] * len(boxes)

        # Prepare the box prompts.
        # TODO: Validate based on running prompts.
        clear_old_points = True  # TODO: Must depend on the running prompt logic.
        boxes = np.array([_process_box(b, self.volume.shape[-2:]) for b in boxes])

        # Add box prompts in a particular frame.
        for curr_frame_id, curr_box, curr_obj_id in zip(frame_ids, boxes, object_id):
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=int(curr_frame_id),
                obj_id=int(curr_obj_id),
                clear_old_points=clear_old_points,  # Whether to make use of old points in memory.
                box=np.array([curr_box]),
            )

    def add_mask_prompts(
        self, frame_ids: Union[int, List[int]], masks: Optional[np.ndarray] = None,
    ):
        raise NotImplementedError

    def _propagate_in_direction(
        self, reverse, update_progress=None, early_stop_patience=None, z_range=None, seen_frames=None
    ):
        """Run SAM2 propagation in one temporal direction, optionally stopping early.

        Each step of the SAM2 video predictor runs the full memory attention and mask decoder
        for one frame, which is the dominant cost of volumetric segmentation (especially on CPU).
        Early stopping reads the masks we already compute and breaks out of the propagation once
        the object has clearly left the volume, so we do not keep running the network on frames
        that no longer contain the object.

        Args:
            reverse: Propagate backwards in time (towards lower slice indices) if True.
            update_progress: Optional callback invoked with the number of newly processed frames.
            early_stop_patience: If given, stop this direction after this many consecutive frames
                in which every tracked object's mask is empty (i.e. the object has left the volume).
                'None' disables early stopping and propagates to the end of the volume.
            z_range: If given, an inclusive '(z_min, z_max)' bound on the slice indices propagation
                may cover. Propagation stops at the range edge in this direction. 'None' propagates
                to the end of the volume.
            seen_frames: Optional set of already-counted frame indices, shared across both directions,
                so a frame processed in both (the conditioning frame) advances 'update_progress' once.

        Returns:
            Mapping from frame index to per-object boolean masks, in the order frames were yielded.
        """
        video_segments = {}
        consecutive_empty = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state, reverse=reverse,
        ):
            # Hard z-range bound: stop once propagation would leave the user-selected slice range.
            if z_range is not None and not (z_range[0] <= out_frame_idx <= z_range[1]):
                break

            per_object = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }
            video_segments[out_frame_idx] = per_object
            # Count each slice at most once across both directions (the conditioning frame is yielded
            # by both), so the progress bar reaches its total exactly on a full pass without overshoot.
            if update_progress is not None and (seen_frames is None or out_frame_idx not in seen_frames):
                update_progress(1)
            if seen_frames is not None:
                seen_frames.add(out_frame_idx)

            # Early stopping: once every tracked object has been absent for 'early_stop_patience'
            # consecutive frames, the object has left the volume and there is nothing more to track.
            # A single empty frame is not enough (SAM2 can momentarily drop and recover a mask), so
            # we require a run of empty frames before breaking.
            if early_stop_patience is not None:
                frame_is_empty = not any(mask.any() for mask in per_object.values())
                consecutive_empty = consecutive_empty + 1 if frame_is_empty else 0
                if consecutive_empty >= early_stop_patience:
                    break

            # Hard z-range bound: we have just stored the edge slice, so stop before the predictor
            # steps outside the range (and pays for a frame we would discard).
            if z_range is not None and out_frame_idx == (z_range[0] if reverse else z_range[1]):
                break

        return video_segments

    def propagate_prompts(self, update_progress=None, early_stop_patience=None, z_range=None):
        # First, we propagate the masklets forward in time using the input prompts in selected frames.
        # 'update_progress' is an optional callback that is called with the number of newly processed
        # frames, so callers (e.g. the napari annotator) can report propagation progress to the user.
        # 'early_stop_patience' bounds the propagation by stopping a direction once the object has been
        # absent for that many consecutive frames (see '_propagate_in_direction'). 'z_range' is an
        # inclusive '(z_min, z_max)' hard bound on the slices propagation may cover.
        # Shared across both directions so the conditioning frame (yielded by both) is counted once.
        seen_frames = set()
        forward_video_segments = self._propagate_in_direction(
            reverse=False, update_progress=update_progress, early_stop_patience=early_stop_patience,
            z_range=z_range, seen_frames=seen_frames,
        )

        # Next, we do the propagation reverse in time.
        reverse_video_segments = {}
        if len(forward_video_segments) < self.volume.shape[0]:  # Perform reverse propagation only if necessary
            reverse_video_segments = self._propagate_in_direction(
                reverse=True, update_progress=update_progress, early_stop_patience=early_stop_patience,
                z_range=z_range, seen_frames=seen_frames,
            )
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

            # Crop back to the original slice shape (the video predictor pads non-square frames).
            seg = _crop_to_original_shape(seg, self.volume.shape[-2:]).astype("uint32")

        finally:
            # Reset the state to clear this object's prompts
            # This ensures the next segmentation starts fresh
            self.predictor.reset_state(self.inference_state)

        return seg

    def predict(self, update_progress=None, early_stop_patience=None, z_range=None):
        # First, we propagate prompts.
        video_segments = self.propagate_prompts(
            update_progress=update_progress, early_stop_patience=early_stop_patience, z_range=z_range,
        )

        # Next, let's merge the segmented objects per frame back together as instances per slice.
        # We allocate the full-volume output and index it by the slice id so that frames skipped by
        # early stopping (which are absent from 'video_segments') stay as background instead of
        # shifting the remaining slices out of alignment with the volume.
        shape = self.volume.shape[-2:]
        segmentation = np.zeros((self.volume.shape[0],) + tuple(shape), dtype="uint64")
        for slice_idx, instances in video_segments.items():
            per_slice_seg = segmentation[slice_idx]
            for instance_idx, instance_mask in instances.items():
                mask = _crop_to_original_shape(instance_mask.squeeze(), shape)
                per_slice_seg[mask] = instance_idx

        return segmentation


class TiledPromptableSegmentation3D:
    """Tiled promptable segmentation for volumetric data.

    Routes each prompt to the in-plane tile-column it falls in, runs a per-tile
    `PromptableSegmentation3D` on the tile's sub-volume (reusing the precomputed tiled embeddings),
    and stitches the per-tile results into the full volume. Exposes the same interface as
    `PromptableSegmentation3D`, so it is a drop-in replacement when the embeddings are tiled.

    Args:
        predictor: The SAM2 video predictor.
        volume: The input volume, shape (Z, Y, X).
        volume_embeddings: The precomputed tiled 3d embeddings (with per-tile `features`/`pos_enc`/
            `fpn` groups and `shape`/`tile_shape`/`halo` attrs). See `precompute_image_embeddings`.
        device: The device to run inference on.
    """

    def __init__(self, predictor, volume, volume_embeddings, device=None, **kwargs):
        from bioimage_cpp.utils import Blocking

        self.predictor = predictor
        self.volume = volume
        self.volume_embeddings = volume_embeddings
        self.device = device
        self._kwargs = kwargs

        feats = volume_embeddings["features"]
        self.shape = tuple(int(s) for s in feats.attrs["shape"])
        self.tile_shape = tuple(int(s) for s in feats.attrs["tile_shape"])
        self.halo = tuple(int(s) for s in feats.attrs["halo"])
        self.tiling = Blocking([0, 0], list(self.shape[1:]), list(self.tile_shape))

        # Per-tile state, built lazily for the tiles that actually receive prompts.
        self._segmenters = {}

    def init_predictor(self):
        # Per-tile inference states are created lazily in '_get_segmenter'.
        pass

    def reset_predictor(self):
        for segmenter in self._segmenters.values():
            segmenter.reset_predictor()
        self._segmenters = {}

    def _tile_index(self, y, x):
        """Return the id of the tile whose inner (halo-free) block contains the point (y, x)."""
        for tile_id in range(self.tiling.number_of_blocks):
            inner = self.tiling.get_block_with_halo(tile_id, list(self.halo)).inner_block
            if inner.begin[0] <= y < inner.end[0] and inner.begin[1] <= x < inner.end[1]:
                return tile_id
        return 0

    def _outer_offset(self, tile_id):
        """Return the (y0, x0) origin of the tile's outer (halo-included) block."""
        outer = self.tiling.get_block_with_halo(tile_id, list(self.halo)).outer_block
        return int(outer.begin[0]), int(outer.begin[1])

    def _get_segmenter(self, tile_id):
        if tile_id not in self._segmenters:
            from micro_sam.v2.util import _load_list_datasets

            outer = self.tiling.get_block_with_halo(tile_id, list(self.halo)).outer_block
            bb = (slice(int(outer.begin[0]), int(outer.end[0])), slice(int(outer.begin[1]), int(outer.end[1])))
            sub_volume = np.ascontiguousarray(self.volume[:, bb[0], bb[1]])

            feats = self.volume_embeddings["features"]
            tile_dataset = feats[str(tile_id)]
            tile_embeddings = {
                "features": np.asarray(tile_dataset),
                "pos_enc": _load_list_datasets(self.volume_embeddings["pos_enc"], str(tile_id), lazy_loading=False),
                "fpn": _load_list_datasets(self.volume_embeddings["fpn"], str(tile_id), lazy_loading=False),
                "input_size": tile_dataset.attrs["input_size"],
                "original_size": tile_dataset.attrs["original_size"],
            }
            self._segmenters[tile_id] = PromptableSegmentation3D(
                self.predictor, sub_volume, tile_embeddings, device=self.device, **self._kwargs
            )
        return self._segmenters[tile_id]

    def _inner_slices(self, tile_id):
        """Return the (local, global) inner-block slices for placing a tile result into the volume."""
        block = self.tiling.get_block_with_halo(tile_id, list(self.halo))
        local = tuple(slice(b, e) for b, e in zip(block.inner_block_local.begin, block.inner_block_local.end))
        glob = tuple(slice(b, e) for b, e in zip(block.inner_block.begin, block.inner_block.end))
        return local, glob

    def segment_slice(self, frame_idx, points=None, labels=None, boxes=None, masks=None, object_id=1):
        """Segment a single slice. Points are (x, y), boxes (x0, y0, x1, y1), as passed by the annotator.

        Groups the prompts by the tile they fall in, segments every tile with a positive cue, and
        unions the per-tile masks - so an object spanning tiles is segmented on both sides.
        """
        have_points = points is not None and len(points) > 0
        have_boxes = boxes is not None and len(boxes) > 0
        if not have_points and not have_boxes:
            return None

        tile_points, tile_labels, tile_boxes = {}, {}, {}
        if have_points:
            for point, label in zip(np.asarray(points), np.asarray(labels)):
                tid = self._tile_index(int(round(point[1])), int(round(point[0])))  # (y, x) from (x, y)
                tile_points.setdefault(tid, []).append(point)
                tile_labels.setdefault(tid, []).append(label)
        if have_boxes:
            for box in boxes:
                box = np.asarray(box)  # (x0, y0, x1, y1)
                box_yx = np.array([box[1], box[0], box[3], box[2]])
                for tid, clipped in _box_to_tiles(self.tiling, self.halo, box_yx).items():
                    tile_boxes.setdefault(tid, []).append(np.array([clipped[1], clipped[0], clipped[3], clipped[2]]))

        out = np.zeros(self.shape[1:], dtype="uint32")
        found = False
        for tile_id in sorted(set(tile_points) | set(tile_boxes)):
            tpoints = np.asarray(tile_points.get(tile_id, [])).reshape(-1, 2)
            tlabels = np.asarray(tile_labels.get(tile_id, []), dtype=int)
            tboxes = tile_boxes.get(tile_id, [])
            if not ((tlabels == 1).any() or len(tboxes) > 0):
                continue
            y0, x0 = self._outer_offset(tile_id)
            local_points = (tpoints - np.array([x0, y0])) if len(tpoints) else None
            local_boxes = [b - np.array([x0, y0, x0, y0]) for b in tboxes] if tboxes else None
            tile_seg = self._get_segmenter(tile_id).segment_slice(
                frame_idx, points=local_points, labels=(tlabels if len(tlabels) else None),
                boxes=local_boxes, masks=masks, object_id=object_id,
            )
            if tile_seg is None:
                continue
            local, glob = self._inner_slices(tile_id)
            region = tile_seg[local]
            sub = out[glob]
            positive = region > 0
            sub[positive] = region[positive]
            found = True

        return out if found else None

    def add_point_prompts(self, frame_ids, points, point_labels, object_id=None, multiple_objects=False):
        """Add point prompts. Points are in (y, x) order; each is routed to the tile it falls in, so
        an object with prompts in several tiles is added to each of those tiles."""
        if points is None or len(points) == 0:
            return
        if object_id is None:
            object_id = 1

        tile_points, tile_labels = {}, {}
        for point, label in zip(np.asarray(points), np.asarray(point_labels)):
            tid = self._tile_index(int(round(point[0])), int(round(point[1])))
            tile_points.setdefault(tid, []).append(point)
            tile_labels.setdefault(tid, []).append(label)

        for tile_id, tpoints in tile_points.items():
            y0, x0 = self._outer_offset(tile_id)
            local_points = np.asarray(tpoints) - np.array([y0, x0])
            self._get_segmenter(tile_id).add_point_prompts(
                frame_ids=frame_ids, points=local_points, point_labels=np.asarray(tile_labels[tile_id]),
                object_id=object_id, multiple_objects=multiple_objects,
            )

    def add_box_prompts(self, frame_ids, boxes=None, object_id=None):
        """Add box prompts (y0, x0, y1, x1). A box spanning several tiles is added, clipped, to each."""
        if boxes is None or len(boxes) == 0:
            return
        # One object id per box; default to a single object (id 1) when not batched.
        if object_id is None:
            object_id = [1] * len(boxes)
        elif not isinstance(object_id, list):
            object_id = [object_id] * len(boxes)
        # Group the (clipped) boxes and their object ids by the tile each falls in.
        tile_boxes, tile_ids = {}, {}
        for box, obj_id in zip(boxes, object_id):
            for tid, clipped in _box_to_tiles(self.tiling, self.halo, np.asarray(box)).items():
                tile_boxes.setdefault(tid, []).append(clipped)
                tile_ids.setdefault(tid, []).append(obj_id)
        for tile_id, tboxes in tile_boxes.items():
            y0, x0 = self._outer_offset(tile_id)
            local_boxes = [b - np.array([y0, x0, y0, x0]) for b in tboxes]
            self._get_segmenter(tile_id).add_box_prompts(
                frame_ids=frame_ids, boxes=np.array(local_boxes), object_id=tile_ids[tile_id],
            )

    def add_mask_prompts(self, frame_ids, masks=None):
        raise NotImplementedError

    def predict(self, update_progress=None, early_stop_patience=None, z_range=None):
        """Propagate the prompts in every active tile and stitch the results into the full volume.

        Object ids are preserved across tiles (the inner blocks are disjoint), so an object that was
        prompted in several tiles keeps one id and is merged across the tile boundaries.
        """
        segmentation = np.zeros(self.shape, dtype="uint64")
        for tile_id in sorted(self._segmenters):
            tile_seg = self._segmenters[tile_id].predict(
                update_progress=update_progress, early_stop_patience=early_stop_patience, z_range=z_range,
            )
            local, glob = self._inner_slices(tile_id)
            inner = tile_seg[(slice(None),) + local]
            region = segmentation[(slice(None),) + glob]
            positive = inner != 0
            region[positive] = inner[positive]
        return segmentation
