import argparse
import warnings
from typing import List, Optional, Tuple

import napari
import numpy as np

from scipy.ndimage import shift
from skimage import draw

from .. import prompt_based_segmentation, util
from ._state import AnnotatorState

# Green and Red
LABEL_COLOR_CYCLE = ["#00FF00", "#FF0000"]
"""@private"""


#
# Misc helper functions
#


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


def _initialize_parser(description, with_segmentation_result=True, with_show_embeddings=True):

    available_models = list(util.get_model_names())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-i", "--input", required=True,
        help="The filepath to the image data. Supports all data types that can be read by imageio (e.g. tif, png, ...) "
        "or elf.io.open_file (e.g. hdf5, zarr, mrc) For the latter you also need to pass the 'key' parameter."
    )
    parser.add_argument(
        "-k", "--key",
        help="The key for opening data with elf.io.open_file. This is the internal path for a hdf5 or zarr container, "
        "for a image series it is a wild-card, e.g. '*.png' and for mrc it is 'data'."
    )

    parser.add_argument(
        "-e", "--embedding_path",
        help="The filepath for saving/loading the pre-computed image embeddings. "
        "NOTE: It is recommended to pass this argument and store the embeddings, "
        "otherwise they will be recomputed every time (which can take a long time)."
    )

    if with_segmentation_result:
        parser.add_argument(
            "-s", "--segmentation_result",
            help="Optional filepath to a precomputed segmentation. If passed this will be used to initialize the "
            "'committed_objects' layer. This can be useful if you want to correct an existing segmentation or if you "
            "have saved intermediate results from the annotator and want to continue with your annotations. "
            "Supports the same file formats as 'input'."
        )
        parser.add_argument(
            "-sk", "--segmentation_key",
            help="The key for opening the segmentation data. Same rules as for 'key' apply."
        )

    parser.add_argument(
        "--model_type", default=util._DEFAULT_MODEL,
        help=f"The segment anything model that will be used, one of {available_models}."
    )
    parser.add_argument(
        "--tile_shape", nargs="+", type=int, help="The tile shape for using tiled prediction", default=None
    )
    parser.add_argument(
        "--halo", nargs="+", type=int, help="The halo for using tiled prediction", default=None
    )

    return parser


def clear_annotations(viewer: napari.Viewer, clear_segmentations=True) -> None:
    """@private"""
    viewer.layers["point_prompts"].data = []
    viewer.layers["point_prompts"].refresh()
    if "prompts" in viewer.layers:
        viewer.layers["prompts"].data = []
        viewer.layers["prompts"].refresh()
    if not clear_segmentations:
        return
    viewer.layers["current_object"].data = np.zeros(viewer.layers["current_object"].data.shape, dtype="uint32")
    viewer.layers["current_object"].refresh()


#
# Helper functions to extract prompts from napari layers.
#


def point_layer_to_prompts(
    layer: napari.layers.Points, i=None, track_id=None, with_stop_annotation=True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract point prompts for SAM from a napari point layer.

    Args:
        layer: The point layer from which to extract the prompts.
        i: Index for the data (required for 3d or timeseries data).
        track_id: Id of the current track (required for tracking data).
        with_stop_annotation: Whether a single negative point will be interpreted
            as stop annotation or just returned as normal prompt.

    Returns:
        The point coordinates for the prompts.
        The labels (positive or negative / 1 or 0) for the prompts.
    """

    points = layer.data
    labels = layer.properties["label"]
    assert len(points) == len(labels)

    if i is None:
        assert points.shape[1] == 2, f"{points.shape}"
        this_points, this_labels = points, labels
    else:
        assert points.shape[1] == 3, f"{points.shape}"
        mask = points[:, 0] == i
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


def prompt_layers_to_state(
    point_layer: napari.layers.Points, box_layer: napari.layers.Shapes, i: int
) -> str:
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
    predictor, point_prompts, box_prompts, image_embeddings, shape, progress_bar=None, track_id=None
):
    """@private"""
    assert len(shape) == 3
    image_shape = shape[1:]
    seg = np.zeros(shape, dtype="uint32")

    z_values = point_prompts.data[:, 0]
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

    def _update_progress():
        if progress_bar is not None:
            progress_bar.update(1)

    for i in slices:
        points_i = point_layer_to_prompts(point_prompts, i, track_id)

        # do we end the segmentation at the outer slices?
        if points_i is None:

            if i == slices[0]:
                stop_lower = True
            elif i == slices[-1]:
                stop_upper = True
            else:
                raise RuntimeError("Stop slices can only be at the start or end")

            seg[i] = 0
            _update_progress()
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
        _update_progress()

    return seg, slices, stop_lower, stop_upper


def prompt_segmentation(
    predictor, points, labels, boxes, masks, shape, multiple_box_prompts,
    image_embeddings=None, i=None, box_extension=0, multiple_point_prompts=None,
):
    """@private"""
    assert len(points) == len(labels)
    have_points = len(points) > 0
    have_boxes = len(boxes) > 0

    # no prompts were given, return None
    if not have_points and not have_boxes:
        return

    # we use the batched point prompt segmentation mode, but
    # have a box prompt -> this does not work
    elif multiple_point_prompts and have_boxes:
        print("You have activated batched point segmentation but have passed a box prompt.")
        print("This setting is currently not supported.")
        print("Provide a single positive point prompt per object when using batched point segmentation.")
        return

    # box and point prompts were given
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

    # only point prompts were given
    elif have_points and not have_boxes:

        if multiple_point_prompts:
            seg = np.zeros(shape, dtype="uint32")
            batched_points, batched_labels = [], []
            negative_points, negative_labels = [], []
            for j in range(len(points)):
                if labels[j] == 1:  # positive point
                    batched_points.append(points[j:j+1])
                    batched_labels.append(labels[j:j+1])
                else:  # negative points
                    negative_points.append(points[j:j+1])
                    negative_labels.append(labels[j:j+1])

            # Batch this?
            for seg_id, (point, label) in enumerate(zip(batched_points, batched_labels), 1):
                if len(negative_points) > 0:
                    point = np.concatenate([point] + negative_points)
                    label = np.concatenate([label] + negative_labels)
                prediction = prompt_based_segmentation.segment_from_points(
                    predictor, point, label, image_embeddings=image_embeddings, i=i
                ).squeeze()
                seg[prediction] = seg_id
        else:
            seg = prompt_based_segmentation.segment_from_points(
                predictor, points, labels, image_embeddings=image_embeddings, i=i
            ).squeeze()

    # only box prompts were given
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
        center = np.array(np.mean(center[0]), np.mean(center[1]))
        return center

    center0 = compute_center(t0)
    center1 = compute_center(t1)

    move = center1 - center0
    return move.astype("float64")


def _shift_object(mask, motion_model):
    mask_shifted = np.zeros_like(mask)
    shift(mask, motion_model, output=mask_shifted, order=0, prefilter=False)
    return mask_shifted


def track_from_prompts(
    point_prompts, box_prompts, seg, predictor, slices, image_embeddings,
    stop_upper, threshold, projection,
    progress_bar=None, motion_smoothing=0.5, box_extension=0,
):
    """@private
    """
    assert projection in ("mask", "bounding_box")
    if projection == "mask":
        use_mask, use_box = True, True
    else:
        use_mask, use_box = False, True

    def _update_progress():
        if progress_bar is not None:
            progress_bar.update(1)

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
                use_mask=use_mask, use_box=use_box, box_extension=box_extension
            )
            track_state = "track"

            # are we beyond the last slice with prompt?
            # if no: we continue tracking because we know we need to connect to a future frame
            # if yes: we only continue tracking if overlaps are above the threshold
            if t < slices[-1]:
                seg_prev = None

            _update_progress()

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
