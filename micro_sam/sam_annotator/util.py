import argparse
import os
import pickle
from typing import Optional, Tuple

import napari
import numpy as np

from magicgui import magicgui
from magicgui.widgets import ComboBox, Container

from .. import instance_segmentation, util
from ..prompt_based_segmentation import segment_from_box, segment_from_box_and_points, segment_from_points

# Green and Red
LABEL_COLOR_CYCLE = ["#00FF00", "#FF0000"]
"""@private"""


def clear_annotations(v: napari.Viewer, clear_segmentations=True) -> None:
    """@private"""
    v.layers["prompts"].data = []
    v.layers["prompts"].refresh()
    if "box_prompts" in v.layers:
        v.layers["box_prompts"].data = []
        v.layers["box_prompts"].refresh()
    if not clear_segmentations:
        return
    if "current_object" in v.layers:
        v.layers["current_object"].data = np.zeros(v.layers["current_object"].data.shape, dtype="uint32")
        v.layers["current_object"].refresh()
    if "current_track" in v.layers:
        v.layers["current_track"].data = np.zeros(v.layers["current_track"].data.shape, dtype="uint32")
        v.layers["current_track"].refresh()


@magicgui(call_button="Clear Annotations [Shfit + C]")
def _clear_widget(v: napari.Viewer) -> None:
    clear_annotations(v)


@magicgui(call_button="Commit [C]", layer={"choices": ["current_object", "auto_segmentation"]})
def _commit_segmentation_widget(v: napari.Viewer, layer: str = "current_object") -> None:
    seg = v.layers[layer].data
    shape = seg.shape

    id_offset = int(v.layers["committed_objects"].data.max())
    mask = seg != 0

    v.layers["committed_objects"].data[mask] = (seg[mask] + id_offset)
    v.layers["committed_objects"].refresh()

    v.layers[layer].data = np.zeros(shape, dtype="uint32")
    v.layers[layer].refresh()

    if layer == "current_object":
        clear_annotations(v)


def create_prompt_menu(points_layer, labels, menu_name="prompt", label_name="label"):
    """@private"""
    label_menu = ComboBox(label=menu_name, choices=labels)
    label_widget = Container(widgets=[label_menu])

    def update_label_menu(event):
        new_label = str(points_layer.current_properties[label_name][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    points_layer.events.current_properties.connect(update_label_menu)

    def label_changed(new_label):
        current_properties = points_layer.current_properties
        current_properties[label_name] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    label_menu.changed.connect(label_changed)

    return label_widget


def prompt_layer_to_points(
    prompt_layer: napari.layers.Points, i=None, track_id=None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract point prompts for SAM from a napari point layer.

    Args:
        prompt_layer: The point layer from which to extract the prompts.
        i: Index for the data (required for 3d or timeseries data).
        track_id: Id of the current track (required for tracking data).

    Returns:
        The point coordinates for the prompts.
        The labels (positive or negative / 1 or 0) for the prompts.
    """

    points = prompt_layer.data
    labels = prompt_layer.properties["label"]
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
        track_ids = np.array(list(map(int, prompt_layer.properties["track_id"])))[mask]
        track_id_mask = track_ids == track_id
        this_labels, this_points = this_labels[track_id_mask], this_points[track_id_mask]
    assert len(this_points) == len(this_labels)

    this_labels = np.array([1 if label == "positive" else 0 for label in this_labels])
    # a single point with a negative label is interpreted as 'stop' signal
    # in this case we return None
    if len(this_points) == 1 and this_labels[0] == 0:
        return None

    return this_points, this_labels


def prompt_layer_to_boxes(prompt_layer: napari.layers.Shapes, i=None, track_id=None) -> np.ndarray:
    """Extract box prompts for SAM from a napari shape layer.

    Args:
        prompt_layer: The napari shape layer.
        i: Index for the data (required for 3d or timeseries data).
        track_id: Id of the current track (required for tracking data).

    Returns:
        The box prompts.
    """
    shape_data = prompt_layer.data
    shape_types = prompt_layer.shape_type
    assert len(shape_data) == len(shape_types)
    if len(shape_data) == 0:
        return []

    if i is None:
        # select all boxes that are rectangles
        boxes = [data for data, stype in zip(shape_data, shape_types) if stype == "rectangle"]
    else:
        # we are currently only supporting rectangle shapes.
        # other shapes could be supported by providing them as rough mask
        # (and also providing the corresponding bounding box)
        # but for this we need to figure out the mask prompts for non-square shapes
        non_rectangle = [stype != "rectangle" for stype in shape_types]
        if any(non_rectangle):
            print(f"You have provided {sum(non_rectangle)} shapes that are not rectangles.")
            print("We currently do not support these as prompts and they will be ignored.")

        if track_id is None:
            boxes = [
                data[:, 1:] for data, stype in zip(shape_data, shape_types)
                if (stype == "rectangle" and (data[:, 0] == i).all())
            ]
        else:
            track_ids = np.array(list(map(int, prompt_layer.properties["track_id"])))
            assert len(track_ids) == len(shape_data), f"{len(track_ids)}, {len(shape_data)}"
            boxes = [
                data[:, 1:] for data, stype, this_track_id in zip(shape_data, shape_types, track_ids)
                if (stype == "rectangle" and (data[:, 0] == i).all() and this_track_id == track_id)
            ]

    # map to correct box format
    boxes = [
        np.array([box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()]) for box in boxes
    ]
    return boxes


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
        points_i = prompt_layer_to_points(point_prompts, i, track_id)

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

        boxes = prompt_layer_to_boxes(box_prompts, i, track_id)
        points, labels = points_i

        seg_i = prompt_segmentation(
            predictor, points, labels, boxes, image_shape, multiple_box_prompts=False,
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
    predictor, points, labels, boxes, shape, multiple_box_prompts, image_embeddings=None, i=None
):
    """@private"""
    assert len(points) == len(labels)
    have_points = len(points) > 0
    have_boxes = len(boxes) > 0

    # no prompts were given, return None
    if not have_points and not have_boxes:
        return

    # box and ppint prompts were given
    elif have_points and have_boxes:
        if len(boxes) > 1:
            print("You have provided point prompts and more than one box prompt.")
            print("This setting is currently not supported.")
            print("When providing both points and prompts you can only segment one object at a time.")
            return
        seg = segment_from_box_and_points(
            predictor, boxes[0], points, labels, image_embeddings=image_embeddings, i=i
        ).squeeze()

    # only point prompts were given
    elif have_points and not have_boxes:
        seg = segment_from_points(predictor, points, labels, image_embeddings=image_embeddings, i=i).squeeze()

    # only box prompts were given
    elif not have_points and have_boxes:
        seg = np.zeros(shape, dtype="uint32")

        if len(boxes) > 1 and not multiple_box_prompts:
            print("You have provided more than one box annotation. This is not yet supported in the 3d annotator.")
            print("You can only segment one object at a time in 3d.")
            return

        seg_id = 1
        for box in boxes:
            mask = segment_from_box(predictor, box, image_embeddings=image_embeddings, i=i).squeeze()
            seg[mask] = seg_id
            seg_id += 1

    return seg


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

    if with_show_embeddings:
        parser.add_argument(
            "--show_embeddings", action="store_true",
            help="Visualize the embeddings computed by SegmentAnything. This can be helpful for debugging."
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
