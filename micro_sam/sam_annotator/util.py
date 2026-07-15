import os
import pickle
import warnings
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import napari
import numpy as np
from skimage import draw
from scipy.ndimage import shift

from .. import util
from ..v1 import prompt_based_segmentation
from .. import _model_settings as model_settings
from ..v1.multi_dimensional_segmentation import _validate_projection

# Green and Red
LABEL_COLOR_CYCLE = ["#00FF00", "#FF0000"]
"""@private"""


#
# Misc helper functions
#


def _channels_to_rgb(image: np.ndarray) -> np.ndarray:
    """Map a 2D image's trailing channel axis to exactly 3 channels.

    A single channel is replicated, two channels are padded with a zero channel, three channels are
    left as-is, and more than three channels are reduced to the first three (with a warning).
    """
    n_channels = image.shape[-1]
    if n_channels == 3:
        return image
    if n_channels == 1:
        return np.concatenate([image] * 3, axis=-1)
    if n_channels == 2:
        zero_channel = np.zeros(image.shape[:-1] + (1,), dtype=image.dtype)
        return np.concatenate([image, zero_channel], axis=-1)
    warnings.warn(f"You provided an input with {n_channels} channels. Only the first three will be used.")
    return image[..., :3]


def prepare_annotation_image(image: np.ndarray, ndim: Optional[int] = None) -> Tuple[np.ndarray, int, bool]:
    """Normalize an image for annotation: squeeze singletons and map 2D channels to RGB.

    Singleton axes (commonly exposed by formats like CZI) are squeezed out across all axes.
    For a 2D image, the trailing channel axis is mapped to 3 channels: a 2-channel input is
    padded with a zero channel and a 4-channel input is reduced to the first three (with a
    warning). A 3D volume with a channel axis (3D+C) is not supported.

    Args:
        image: The input image data.
        ndim: Optional override for the spatial dimensionality (2 or 3). With ``None`` (the default)
            the dimensionality is auto-detected from the shape (a trailing axis of size 3 -> RGB 2D,
            of size 2 or 4 -> channels mapped to RGB 2D, otherwise a 3D volume). With ``2`` a 3D array
            is read as a 2D multi-channel image, taking the smallest axis as the channel axis (so a
            channels-first ``(C, H, W)`` array also works) and mapping the channels to RGB. With ``3``
            a 3D array is read as a ``(Z, H, W)`` volume.

    Returns:
        A tuple of the normalized image, its spatial dimensionality (2 or 3), and whether
        it has a trailing RGB channel axis.

    Raises:
        ValueError: If the (overridden) dimensionality cannot be applied to the image shape, or the
            squeezed image is neither a 2D image nor a grayscale 3D volume.
    """
    if ndim not in (None, 2, 3):
        raise ValueError(f"Invalid ndim override: {ndim}. Expected None, 2 or 3.")

    image = np.squeeze(image)

    # Forced 2D: read a 3D array as a 2D multi-channel image. The channel axis is taken to be the
    # smallest axis (so both channels-first (C, H, W) and channels-last (H, W, C) work); it is moved
    # to the trailing position and mapped to 3 channels.
    if ndim == 2:
        if image.ndim == 2:
            return image, 2, False
        if image.ndim == 3:
            channel_axis = int(np.argmin(image.shape))
            image = np.moveaxis(image, channel_axis, -1)
            return _channels_to_rgb(image), 2, True
        raise ValueError(f"Cannot interpret shape {image.shape} as a 2D image.")

    # Forced 3D: read a 3D array as a (Z, H, W) volume. A channel axis (4D, or 3D+C) is not supported.
    if ndim == 3:
        if image.ndim == 3:
            return image, 3, False
        raise ValueError(
            f"Cannot interpret shape {image.shape} as a 3D volume (3D data with channels is not supported yet)."
        )

    # Auto-detect. A 4D array is either a 3D volume with a channel axis (Z, H, W, C) or a volumetric
    # time series (T, Z, H, W). Neither is supported: the v2 3D path assumes a grayscale (Z, H, W)
    # volume, so a channel axis would otherwise produce wrong-shaped masks.
    if image.ndim == 4:
        if image.shape[-1] in (2, 3, 4):
            raise ValueError(
                f"3D volumes with a channel axis are not supported yet, got shape {image.shape}."
            )
        raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D image data (3D+t is not supported).")

    # 2D image with a 2- or 4-channel trailing axis: map it to 3 channels. A trailing axis of any
    # other size is left alone, so a size-3 axis stays RGB and anything else is treated as a volume.
    if image.ndim == 3 and image.shape[-1] in (2, 4):
        image = _channels_to_rgb(image)

    # Map the (possibly normalized) shape to a spatial dimensionality and rgb flag.
    if image.ndim == 2:
        return image, 2, False
    elif image.ndim == 3:
        if image.shape[-1] == 3:
            return image, 2, True
        return image, 3, False
    raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D image data.")


def toggle_label(prompts):
    """@private"""
    # get the currently selected label
    current_properties = prompts.current_properties
    current_label = current_properties["label"][0]
    new_label = "negative" if current_label == "positive" else "positive"
    current_properties["label"] = np.array([new_label])
    prompts.current_properties = current_properties
    prompts.refresh()
    prompts.refresh_colors()


def clear_annotations(viewer: napari.Viewer, clear_segmentations=True) -> None:
    """@private"""
    viewer.layers["point_prompts"].data = []
    viewer.layers["point_prompts"].refresh()
    if "prompts" in viewer.layers:
        # Select all prompts and then remove them.
        # This is how it worked before napari 0.5.
        # viewer.layers["prompts"].data = []
        viewer.layers["prompts"].selected_data = set(range(len(viewer.layers["prompts"].data)))
        viewer.layers["prompts"].remove_selected()
        viewer.layers["prompts"].refresh()
    if not clear_segmentations:
        return
    viewer.layers["current_object"].data = np.zeros(viewer.layers["current_object"].data.shape, dtype="uint32")
    viewer.layers["current_object"].refresh()


def clear_annotations_slice(viewer: napari.Viewer, i: int, clear_segmentations=True) -> None:
    """@private"""
    point_prompts = viewer.layers["point_prompts"].data
    point_prompts = point_prompts[point_prompts[:, 0] != i]
    viewer.layers["point_prompts"].data = point_prompts
    viewer.layers["point_prompts"].refresh()
    if "prompts" in viewer.layers:
        prompts = viewer.layers["prompts"].data
        prompts = [prompt for prompt in prompts if not (prompt[:, 0] == i).all()]
        viewer.layers["prompts"].data = prompts
        viewer.layers["prompts"].refresh()
    if not clear_segmentations:
        return
    viewer.layers["current_object"].data[i] = 0
    viewer.layers["current_object"].refresh()


#
# Helper functions to extract prompts from napari layers.
#


def point_layer_to_prompts(
    layer: napari.layers.Points, i=None, track_id=None, with_stop_annotation=True, exclude_states=None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract point prompts for SAM from a napari point layer.

    Args:
        layer: The point layer from which to extract the prompts.
        i: Index for the data (required for 3d or timeseries data).
        track_id: Id of the current track (required for tracking data).
        with_stop_annotation: Whether a single negative point will be interpreted
            as stop annotation or just returned as normal prompt.
        exclude_states: Track-states to drop (e.g. ('division',)); such points mark a lineage
            event rather than a segmentation prompt and must not be fed to the predictor.

    Returns:
        The point coordinates for the prompts.
        The labels (positive or negative / 1 or 0) for the prompts.
    """

    points = layer.data
    labels = layer.properties["label"]
    assert len(points) == len(labels)

    # Drop points tagged with an excluded track-state (division markers are not prompts).
    keep = np.ones(len(points), dtype=bool)
    if exclude_states is not None and "state" in layer.properties:
        keep = ~np.isin(np.asarray(layer.properties["state"]), list(exclude_states))

    if i is None:
        assert points.shape[1] == 2, f"{points.shape}"
        this_points, this_labels = points[keep], labels[keep]
    else:
        assert points.shape[1] == 3, f"{points.shape}"
        mask = (np.round(points[:, 0]) == i) & keep
        this_points = points[mask][:, 1:]
        this_labels = labels[mask]
    assert len(this_points) == len(this_labels)

    if track_id is not None:
        assert i is not None
        track_ids = np.array(list(map(int, layer.properties["track_id"])))[mask]
        track_id_mask = track_ids == track_id
        this_labels, this_points = this_labels[track_id_mask], this_points[track_id_mask]
    assert len(this_points) == len(this_labels)

    this_labels = np.array([1 if label == "positive" else 0 for label in this_labels])
    # a single point with a negative label is interpreted as 'stop' signal
    # in this case we return None
    if with_stop_annotation and (len(this_points) == 1 and this_labels[0] == 0):
        return None

    return this_points, this_labels


def shape_layer_to_prompts(
    layer: napari.layers.Shapes, shape: Tuple[int, int], i=None, track_id=None
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
    """Extract prompts for SAM from a napari shape layer.

    Extracts the bounding box for 'rectangle' shapes and the bounding box and corresponding mask
    for 'ellipse' and 'polygon' shapes.

    Args:
        prompt_layer: The napari shape layer.
        shape: The image shape.
        i: Index for the data (required for 3d or timeseries data).
        track_id: Id of the current track (required for tracking data).

    Returns:
        The box prompts.
        The mask prompts.
    """

    def _to_prompts(shape_data, shape_types):
        boxes, masks = [], []

        for data, type_ in zip(shape_data, shape_types):

            if type_ == "rectangle":
                boxes.append(data)
                masks.append(None)

            elif type_ == "ellipse":
                boxes.append(data)
                center = np.mean(data, axis=0)
                radius_r = ((data[2] - data[1]) / 2)[0]
                radius_c = ((data[1] - data[0]) / 2)[1]
                rr, cc = draw.ellipse(center[0], center[1], radius_r, radius_c, shape=shape)
                mask = np.zeros(shape, dtype=bool)
                mask[rr, cc] = 1
                masks.append(mask)

            elif type_ == "polygon":
                boxes.append(data)
                rr, cc = draw.polygon(data[:, 0], data[:, 1], shape=shape)
                mask = np.zeros(shape, dtype=bool)
                mask[rr, cc] = 1
                masks.append(mask)

            else:
                warnings.warn(f"Shape type {type_} is not supported and will be ignored.")

        # map to correct box format
        boxes = [
            np.array([box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()]) for box in boxes
        ]
        return boxes, masks

    shape_data, shape_types = layer.data, layer.shape_type
    assert len(shape_data) == len(shape_types)
    if len(shape_data) == 0:
        return [], []

    if i is not None:
        if track_id is None:
            prompt_selection = [j for j, data in enumerate(shape_data) if (data[:, 0] == i).all()]
        else:
            track_ids = np.array(list(map(int, layer.properties["track_id"])))
            prompt_selection = [
                j for j, (data, this_track_id) in enumerate(zip(shape_data, track_ids))
                if ((data[:, 0] == i).all() and this_track_id == track_id)
            ]

        shape_data = [shape_data[j][:, 1:] for j in prompt_selection]
        shape_types = [shape_types[j] for j in prompt_selection]

    boxes, masks = _to_prompts(shape_data, shape_types)
    return boxes, masks


def prompt_layer_to_state(prompt_layer: napari.layers.Points, i: int) -> str:
    """Get the state of the track from a point layer for a given timeframe.

    Only relevant for annotator_tracking.

    Args:
        prompt_layer: The napari layer.
        i: Timeframe of the data.

    Returns:
        The state of this frame (either "division" or "track").
    """
    state = prompt_layer.properties["state"]

    points = prompt_layer.data
    assert points.shape[1] == 3, f"{points.shape}"
    mask = points[:, 0] == i
    this_points = points[mask][:, 1:]
    this_state = state[mask]
    assert len(this_points) == len(this_state)

    # we set the state to 'division' if at least one point in this frame has a division label
    if any(st == "division" for st in this_state):
        return "division"
    else:
        return "track"


def prompt_layers_to_state(point_layer: napari.layers.Points, box_layer: napari.layers.Shapes, i: int) -> str:
    """Get the state of the track from a point layer and shape layer for a given timeframe.

    Only relevant for annotator_tracking.

    Args:
        point_layer: The napari point layer.
        box_layer: The napari box layer.
        i: Timeframe of the data.

    Returns:
        The state of this frame (either "division" or "track").
    """
    state = point_layer.properties["state"]

    points = point_layer.data
    assert points.shape[1] == 3, f"{points.shape}"
    mask = points[:, 0] == i
    if mask.sum() > 0:
        this_state = state[mask].tolist()
    else:
        this_state = []

    box_states = box_layer.properties["state"]
    this_box_states = [
        state for box, state in zip(box_layer.data, box_states)
        if (box[:, 0] == i).all()
    ]
    this_state.extend(this_box_states)

    # we set the state to 'division' if at least one point in this frame has a division label
    if any(st == "division" for st in this_state):
        return "division"
    else:
        return "track"


#
# Helper functions to run (multi-dimensional) segmentation on napari layers.
#


def segment_slices_with_prompts(
    predictor, point_prompts, box_prompts, image_embeddings, shape, track_id=None, update_progress=None,
):
    """@private"""
    assert len(shape) == 3
    image_shape = shape[1:]
    seg = np.zeros(shape, dtype="uint32")

    z_values = np.round(point_prompts.data[:, 0])
    z_values_boxes = np.concatenate([box[:1, 0] for box in box_prompts.data]) if box_prompts.data else\
        np.zeros(0, dtype="int")

    if track_id is not None:
        track_ids_points = np.array(list(map(int, point_prompts.properties["track_id"])))
        assert len(track_ids_points) == len(z_values)
        z_values = z_values[track_ids_points == track_id]

        if len(z_values_boxes) > 0:
            track_ids_boxes = np.array(list(map(int, box_prompts.properties["track_id"])))
            assert len(track_ids_boxes) == len(z_values_boxes), f"{len(track_ids_boxes)}, {len(z_values_boxes)}"
            z_values_boxes = z_values_boxes[track_ids_boxes == track_id]

    slices = np.unique(np.concatenate([z_values, z_values_boxes])).astype("int")
    stop_lower, stop_upper = False, False

    if update_progress is None:
        def update_progress(*args):
            pass

    for i in slices:
        points_i = point_layer_to_prompts(point_prompts, i, track_id)

        # do we end the segmentation at the outer slices?
        if points_i is None:

            if i == slices[0]:  # The bottom slice is a stop slice.
                stop_lower = True
                seg[i] = 0
            elif i == slices[-1]:  # The top sloce is a stop slice.
                stop_upper = True
                seg[i] = 0
            else:  # We have a stop annotation somewhere in the middle. Ignore this.
                # Remove this slice from the annotated slices, so that it is segmented via
                # projection in the next step.
                slices = np.setdiff1d(slices, i)
                print(f"You have provided a stop annotation (single red point) in slice {i},")
                print("but you have annotated slices above or below it. This stop annotation will")
                print(f"be ignored and the slice {i} will be segmented normally.")

            update_progress(1)
            continue

        boxes, masks = shape_layer_to_prompts(box_prompts, image_shape, i=i, track_id=track_id)
        points, labels = points_i

        seg_i = prompt_segmentation(
            predictor, points, labels, boxes, masks, image_shape, multiple_box_prompts=False,
            image_embeddings=image_embeddings, i=i
        )
        if seg_i is None:
            print(f"The prompts at slice or frame {i} are invalid and the segmentation was skipped.")
            print("This will lead to a wrong segmentation across slices or frames.")
            print(f"Please correct the prompts in {i} and rerun the segmentation.")
            continue

        seg[i] = seg_i
        update_progress(1)

    return seg, slices, stop_lower, stop_upper


# For advanced batching: match prompts to already segmented objects and continue segmentation.
def _match_prompts(previous_segmentation, points, boxes, seg_ids):
    # Create a mapping between ids and prompts.
    batched_prompts = {}
    # seg_boundaries = find_boundaries(previous_segmentation, mode="inner")
    # indices = distance_transform_edt(seg_boundaries, return_distance=False, return_index=True)
    return batched_prompts


def _batched_interactive_segmentation(predictor, points, labels, boxes, image_embeddings, i, previous_segmentation):
    prev_seg = previous_segmentation if i is None else previous_segmentation[i]
    seg = np.zeros(prev_seg.shape, dtype="uint32")

    # seg_ids = np.unique(previous_segmentation)
    # assert seg_ids[0] == 0

    batched_points, batched_labels = [], []
    negative_points, negative_labels = [], []
    for j in range(len(points)):
        if labels[j] == 1:  # positive point
            batched_points.append(points[j:j+1])
            batched_labels.append(labels[j:j+1])
        else:  # negative points
            negative_points.append(points[j:j+1])
            negative_labels.append(labels[j:j+1])

    batched_prompts = [(None, point, label) for point, label in zip(batched_points, batched_labels)]
    batched_prompts.extend([(box, None, None) for box in boxes])
    batched_prompts = {i: prompt for i, prompt in enumerate(batched_prompts, 1)}

    # For advanced batching: match prompts to already segmented objects and continue segmentation.
    # (This is left here as a reference for how this can be implemented.
    #  I have not decided yet if this is actually a good idea or not.)
    # # If we have no objects: this is the first call for a batched segmentation.
    # # We treat each positive point or box as a separate object.
    # if len(seg_ids) == 1:
    #     # Create a list of all prompts.
    #     batched_prompts = [(None, point, label) for point, label in zip(batched_points, batched_labels)]
    #     batched_prompts.extend([(box, None, None) for box in boxes])
    #     batched_prompts = {i: prompt for i, prompt in enumerate(batched_prompts, 1)}

    # # Otherwise we match the prompts to existing objects.
    # else:
    #     batched_prompts = _match_prompts(prev_seg, batched_points, boxes, seg_ids)

    for seg_id, prompt in batched_prompts.items():
        box, point, label = prompt
        if len(negative_points) > 0:
            if point is None:
                point, label = negative_points, negative_labels
            else:
                point = np.concatenate([point] + negative_points)
                label = np.concatenate([label] + negative_labels)

        if (box is not None) and (point is not None):
            prediction = prompt_based_segmentation.segment_from_box_and_points(
                predictor, box, point, label, image_embeddings=image_embeddings, i=i
            ).squeeze()
        elif (box is not None) and (point is None):
            prediction = prompt_based_segmentation.segment_from_box(
                predictor, box, image_embeddings=image_embeddings, i=i
            ).squeeze()
        else:
            prediction = prompt_based_segmentation.segment_from_points(
                predictor, point, label, image_embeddings=image_embeddings, i=i
            ).squeeze()

        seg[prediction] = seg_id

    return seg


def prompt_segmentation(
    predictor, points, labels, boxes, masks, shape, multiple_box_prompts,
    image_embeddings=None, i=None, box_extension=0, batched=None, previous_segmentation=None,
):
    """@private"""
    assert len(points) == len(labels)
    have_points = len(points) > 0
    have_boxes = len(boxes) > 0

    # No prompts were given, return None.
    if not have_points and not have_boxes:
        return

    # Batched interactive segmentation.
    elif batched:
        assert previous_segmentation is not None
        seg = _batched_interactive_segmentation(
            predictor, points, labels, boxes, image_embeddings, i, previous_segmentation
        )

    # Box and point prompts were given.
    elif have_points and have_boxes:
        if len(boxes) > 1:
            print("You have provided point prompts and more than one box prompt.")
            print("This setting is currently not supported.")
            print("When providing both points and prompts you can only segment one object at a time.")
            return
        mask = masks[0]
        if mask is None:
            seg = prompt_based_segmentation.segment_from_box_and_points(
                predictor, boxes[0], points, labels, image_embeddings=image_embeddings, i=i
            ).squeeze()
        else:
            seg = prompt_based_segmentation.segment_from_mask(
                predictor, mask, box=boxes[0], points=points, labels=labels, image_embeddings=image_embeddings, i=i
            ).squeeze()

    # Only point prompts were given.
    elif have_points and not have_boxes:
        seg = prompt_based_segmentation.segment_from_points(
            predictor, points, labels, image_embeddings=image_embeddings, i=i
        ).squeeze()

    # Only box prompts were given.
    elif not have_points and have_boxes:
        seg = np.zeros(shape, dtype="uint32")

        if len(boxes) > 1 and not multiple_box_prompts:
            print("You have provided more than one box annotation. This is not yet supported in the 3d annotator.")
            print("You can only segment one object at a time in 3d.")
            return

        # Batch this?
        for seg_id, (box, mask) in enumerate(zip(boxes, masks), 1):
            if mask is None:
                prediction = prompt_based_segmentation.segment_from_box(
                    predictor, box, image_embeddings=image_embeddings, i=i
                ).squeeze()
            else:
                prediction = prompt_based_segmentation.segment_from_mask(
                    predictor, mask, box=box, image_embeddings=image_embeddings, i=i,
                    box_extension=box_extension,
                ).squeeze()
            seg[prediction] = seg_id

    return seg


def _compute_movement(seg, t0, t1):

    def compute_center(t):
        # computation with center of mass
        center = np.where(seg[t] == 1)
        center = np.array([np.mean(center[0]), np.mean(center[1])])
        return center

    center0 = compute_center(t0)
    center1 = compute_center(t1)

    move = center0 - center1
    return move.astype("float64")


def _shift_object(mask, motion_model):
    mask_shifted = np.zeros_like(mask)
    shift(mask, motion_model, output=mask_shifted, order=0, prefilter=False)
    return mask_shifted


def track_from_prompts(
    point_prompts, box_prompts, seg, predictor, slices, image_embeddings,
    stop_upper, threshold, projection, motion_smoothing=0.5, box_extension=0, update_progress=None,
):
    """@private
    """
    use_box, use_mask, use_points, use_single_point = _validate_projection(projection)

    if update_progress is None:
        def update_progress(*args):
            pass

    # shift the segmentation based on the motion model and update the motion model
    def _update_motion_model(seg, t, t0, motion_model):
        if t in (t0, t0 + 1):  # this is the first or second frame, we don't have a motion yet
            pass
        elif t == t0 + 2:  # this the third frame, we initialize the motion model
            current_move = _compute_movement(seg, t - 1, t - 2)
            motion_model = current_move
        else:  # we already have a motion model and update it
            current_move = _compute_movement(seg, t - 1, t - 2)
            alpha = motion_smoothing
            motion_model = alpha * motion_model + (1 - alpha) * current_move

        return motion_model

    has_division = False
    motion_model = None
    verbose = False

    t0 = int(slices.min())
    t = t0 + 1
    while True:

        # update the motion model
        motion_model = _update_motion_model(seg, t, t0, motion_model)

        # use the segmentation from prompts if we are in a slice with prompts
        if t in slices:
            seg_prev = None
            seg_t = seg[t]
            # currently using the box layer doesn't work for keeping track of the track state
            # track_state = prompt_layers_to_state(point_prompts, box_prompts, t)
            track_state = prompt_layer_to_state(point_prompts, t)

        # otherwise project the mask (under the motion model) and segment the next slice from the mask
        else:
            if verbose:
                print(f"Tracking object in frame {t} with movement {motion_model}")

            seg_prev = seg[t - 1]
            # shift the segmentation according to the motion model
            if motion_model is not None:
                seg_prev = _shift_object(seg_prev, motion_model)

            seg_t = prompt_based_segmentation.segment_from_mask(
                predictor, seg_prev, image_embeddings=image_embeddings, i=t,
                use_mask=use_mask, use_box=use_box, use_points=use_points,
                box_extension=box_extension, use_single_point=use_single_point,
            )
            track_state = "track"

            # are we beyond the last slice with prompt?
            # if no: we continue tracking because we know we need to connect to a future frame
            # if yes: we only continue tracking if overlaps are above the threshold
            if t < slices[-1]:
                seg_prev = None

            update_progress(1)

        if (threshold is not None) and (seg_prev is not None):
            iou = util.compute_iou(seg_prev, seg_t)
            if iou < threshold:
                msg = f"Segmentation stopped at frame {t} due to IOU {iou} < {threshold}."
                print(msg)
                break

        # stop if we have a division
        if track_state == "division":
            has_division = True
            break

        seg[t] = seg_t
        t += 1

        # stop tracking if we have stop upper set (i.e. single negative point was set to indicate stop track)
        if t == slices[-1] and stop_upper:
            break

        # stop if we are at the last slce
        if t == seg.shape[0]:
            break

    return seg, has_division


def _sync_embedding_widget(widget, model_type, save_path, checkpoint_path, device, tile_shape, halo):

    # VFM families (DINO / UNI / SAM3) live in the classification widget's advanced tier; let the widget
    # place the selection. The SAM family/size parsing below is skipped for these names: it parses the
    # size positionally (vit_<size>), which does not apply and even index-errors on short names ('sam3').
    from ..models.vfm import is_vfm_model
    is_vfm = is_vfm_model(model_type)
    if is_vfm and hasattr(widget, "set_model_family_size"):
        family, size = widget._family_and_size_for_model(model_type)
        widget.set_model_family_size(family, size)

    if not is_vfm:
        # Update the index for model family, eg. 'Natural Images (SAM)', 'Light Microscopy', etc.
        supported_dropdown_maps = {
            "lm": "Light Microscopy",
            "em_organelles": "Electron Microscopy",
            "medical_imaging": "Medical Imaging",
            "histopathology": "Histopathology",
        }

        if model_type.startswith("hvit"):  # SAM2 models, eg. 'hvit_t'.
            # Finetuned SAM2 families carry a suffix (e.g. 'hvit_t_cells' -> 'Microscopy'); the plain
            # backbones ('hvit_t', ...) are natural-image models.
            if model_type.endswith("_cells"):
                model_family = "Microscopy"
            else:
                model_family = "Natural Images"
        else:
            model_family = "Natural Images (SAM)"  # No suffix match: stick to 'Natural Images (SAM)'.
            for k, v in supported_dropdown_maps.items():
                if model_type.endswith(k):
                    model_family = v
                    break

        index = widget.model_family_dropdown.findText(model_family)
        if index >= 0:
            widget.model_family_dropdown.setCurrentIndex(index)

        # Update the index for model size, eg. 'base', 'tiny', etc.
        size_map = {"t": "tiny", "s": "small", "b": "base", "l": "large", "h": "huge"}
        size_idx = 5 if model_type.startswith("h") else 4
        model_size = size_map.get(model_type[size_idx])

        if model_size is not None:
            index = widget.model_size_dropdown.findText(model_size)
            if index >= 0:
                widget.model_size_dropdown.setCurrentIndex(index)

    if save_path is not None and isinstance(save_path, str):
        widget.embeddings_save_path_param.setText(str(save_path))

    if checkpoint_path is not None:
        widget.custom_weights_param.setText(str(checkpoint_path))

    if device is not None:
        widget.device = device
        index = widget.device_dropdown.findText(device)
        widget.device_dropdown.setCurrentIndex(index)

    if tile_shape is not None:
        widget.tile_x_param.setValue(tile_shape[0])
        widget.tile_y_param.setValue(tile_shape[1])
        # Enable tiling so the loaded tile shape is used and shown.
        widget.tiling_dropdown.setCurrentText("yes")

    if halo is not None:
        widget.halo_x_param.setValue(halo[0])
        widget.halo_y_param.setValue(halo[1])


# Read parameters from checkpoint path if it is given instead.
def _sync_autosegment_widget(widget, model_type, checkpoint_path, update_decoder=None):
    if update_decoder is not None:
        widget._reset_segmentation_mode(update_decoder)

    # Apply per-model default settings for the v1 generators if the widget exposes them
    # (e.g. the automatic tracking widget). The new dense/sparse segmentation widget does not
    # expose these parameters, since its backend is deferred, so the updates are skipped there.
    if getattr(widget, "with_decoder", False):
        settings = model_settings.AIS_SETTINGS.get(model_type, {})
        params = ("center_distance_thresh", "boundary_distance_thresh")
    else:
        settings = model_settings.AMG_SETTINGS.get(model_type, {})
        params = ("pred_iou_thresh", "stability_score_thresh", "min_object_size")

    for param in params:
        if param in settings and hasattr(widget, f"{param}_param"):
            getattr(widget, f"{param}_param").setValue(settings[param])


# Read parameters from checkpoint path if it is given instead.
def _sync_ndsegment_widget(widget, model_type, checkpoint_path):
    settings = model_settings.ND_SEGMENT_SETTINGS.get(model_type, {})

    if "projection_mode" in settings:
        projection_mode = settings["projection_mode"]
        widget.projection = projection_mode
        index = widget.projection_dropdown.findText(projection_mode)
        if index > 0:
            widget.projection_dropdown.setCurrentIndex(index)

    params = ("iou_threshold", "box_extension")
    for param in params:
        if param in settings:
            getattr(widget, f"{param}_param").setValue(settings[param])


def _load_amg_state(embedding_path):
    if embedding_path is None or not os.path.exists(embedding_path):
        return {"cache_folder": None}

    cache_folder = os.path.join(embedding_path, "amg_state")
    os.makedirs(cache_folder, exist_ok=True)
    amg_state = {"cache_folder": cache_folder}

    state_paths = glob(os.path.join(cache_folder, "*.pkl"))
    for path in state_paths:
        with open(path, "rb") as f:
            state = pickle.load(f)
        i = int(Path(path).stem.split("-")[-1])
        amg_state[i] = state
    return amg_state


def _load_is_state(embedding_path):
    if embedding_path is None or not os.path.exists(embedding_path):
        return {"cache_path": None}

    cache_path = os.path.join(embedding_path, "is_state.h5")
    is_state = {"cache_path": cache_path}

    with h5py.File(cache_path, "a") as f:
        for name, g in f.items():
            i = int(name.split("-")[-1])
            state = {
                "foreground": g["foreground"][:],
                "boundary_distances": g["boundary_distances"][:],
                "center_distances": g["center_distances"][:],
            }
            is_state[i] = state

    return is_state
