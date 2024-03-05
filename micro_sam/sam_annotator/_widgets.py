"""Implements the widgets used in the annotation plugins.
"""

import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Literal

import numpy as np
import zarr

from magicgui import magic_factory, widgets
from magicgui.widgets import ComboBox, Container
from napari.qt.threading import thread_worker
from napari.utils import progress
from zarr.errors import PathNotFoundError

from ._state import AnnotatorState
from . import util as vutil
from .. import instance_segmentation, util
from ..multi_dimensional_segmentation import segment_mask_in_volume

if TYPE_CHECKING:
    import napari


def _reset_tracking_state(viewer):
    """Reset the tracking state.

    This helper function is needed by the widgets clear_track and by commit_track.
    """
    state = AnnotatorState()

    # Reset the lineage and track id.
    state.current_track_id = 1
    state.lineage = {1: []}

    # Reset the choices in the track_id menu.
    track_ids = list(map(str, state.lineage.keys()))
    state.tracking_widget[1].choices = track_ids

    viewer.layers["point_prompts"].property_choices["track_id"] = ["1"]
    viewer.layers["prompts"].property_choices["track_id"] = ["1"]


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear(viewer: "napari.viewer.Viewer") -> None:
    """Widget for clearing the current annotations."""
    vutil.clear_annotations(viewer)


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear_track(viewer: "napari.viewer.Viewer") -> None:
    """Widget for clearing all tracking annotations and state."""
    _reset_tracking_state(viewer)
    vutil.clear_annotations(viewer)


@magic_factory(call_button="Commit [C]", layer={"choices": ["current_object", "auto_segmentation"]})
def commit(viewer: "napari.viewer.Viewer", layer: str = "current_object") -> None:
    """Widget for committing the segmented objects from automatic or interactive segmentation."""
    seg = viewer.layers[layer].data
    shape = seg.shape

    id_offset = int(viewer.layers["committed_objects"].data.max())
    mask = seg != 0

    viewer.layers["committed_objects"].data[mask] = (seg[mask] + id_offset)
    viewer.layers["committed_objects"].refresh()

    viewer.layers[layer].data = np.zeros(shape, dtype="uint32")
    viewer.layers[layer].refresh()

    if layer == "current_object":
        vutil.clear_annotations(viewer)


@magic_factory(call_button="Commit [C]", layer={"choices": ["current_object"]})
def commit_track(viewer: "napari.viewer.Viewer", layer: str = "current_object") -> None:
    state = AnnotatorState()

    seg = viewer.layers[layer].data

    id_offset = int(viewer.layers["committed_objects"].data.max())
    mask = seg != 0

    viewer.layers["committed_objects"].data[mask] = (seg[mask] + id_offset)
    viewer.layers["committed_objects"].refresh()

    shape = state.image_shape
    viewer.layers[layer].data = np.zeros(shape, dtype="uint32")
    viewer.layers[layer].refresh()

    updated_lineage = {
        parent + id_offset: [child + id_offset for child in children] for parent, children in state.lineage.items()
    }
    state.committed_lineages.append(updated_lineage)

    _reset_tracking_state(viewer)
    vutil.clear_annotations(viewer, clear_segmentations=False)


@magic_factory(call_button="Save Lineage")
def save_lineage(viewer: "napari.viewer.Viewer", path: Path) -> None:
    state = AnnotatorState()
    path = path.with_suffix(".json")
    with open(path, "w") as f:
        json.dump(state.committed_lineages, f)


def create_prompt_menu(points_layer, labels, menu_name="prompt", label_name="label"):
    """Create the menu for toggling point prompt labels."""
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


# TODO add options for tiling, see https://github.com/computational-cell-analytics/micro-sam/issues/331
@magic_factory(
    pbar={"visible": False, "max": 0, "value": 0, "label": "working..."},
    call_button="Compute image embeddings",
    save_path={"mode": "d"},  # choose a directory
)
def embedding(
    pbar: widgets.ProgressBar,
    image: "napari.layers.Image",
    model: Literal[tuple(util.models().urls.keys())] = util._DEFAULT_MODEL,
    device: Literal[tuple(["auto"] + util._available_devices())] = "auto",
    save_path: Optional[Path] = None,  # where embeddings for this image are cached (optional)
    custom_weights: Optional[Path] = None,  # A filepath or URL to custom model weights.
) -> util.ImageEmbeddings:
    """Widget to compute the embeddings for a napari image layer."""
    state = AnnotatorState()
    state.reset_state()

    # Get image dimensions.
    if image.rgb:
        ndim = image.data.ndim - 1
        state.image_shape = image.data.shape[:-1]
    else:
        ndim = image.data.ndim
        state.image_shape = image.data.shape

    @thread_worker(connect={"started": pbar.show, "finished": pbar.hide})
    def _compute_image_embedding(
        state, image_data, save_path, ndim=None,
        device="auto", model=util._DEFAULT_MODEL,
        custom_weights=None,
    ):
        # Make sure save directory exists and is an empty directory
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            if not save_path.is_dir():
                raise NotADirectoryError(
                    f"The user selected 'save_path' is not a direcotry: {save_path}"
                )
            if len(os.listdir(save_path)) > 0:
                try:
                    zarr.open(save_path, "r")
                except PathNotFoundError:
                    raise RuntimeError(
                        "The user selected 'save_path' is not a zarr array "
                        f"or empty directory: {save_path}"
                    )

        state.initialize_predictor(
            image_data, model_type=model, save_path=save_path, ndim=ndim, device=device,
            checkpoint_path=custom_weights,
        )
        return state  # returns napari._qt.qthreading.FunctionWorker

    return _compute_image_embedding(
        state, image.data, save_path, ndim=ndim, device=device, model=model,
        custom_weights=custom_weights
    )


@magic_factory(
    call_button="Update settings",
    cache_directory={"mode": "d"},  # choose a directory
)
def settings_widget(
    cache_directory: Optional[Path] = util.get_cache_directory(),
) -> None:
    """Widget to update global micro_sam settings."""
    os.environ["MICROSAM_CACHEDIR"] = str(cache_directory)
    print(f"micro-sam cache directory set to: {cache_directory}")


# TODO fail more gracefully in all widgets if image embeddings have not been initialized
# See https://github.com/computational-cell-analytics/micro-sam/issues/332
#
# Widgets for interactive segmentation:
# - segment: for the 2d annotation tool
# - segment_slice: segment object a single slice for the 3d annotation tool
# - segment_volume: segment object in 3d for the 3d annotation tool
# - segment_frame: segment object in frame for the tracking annotation tool
# - track_object: track object over time for the tracking annotation tool
#


# TODO support extra mode for one point per object
# See https://github.com/computational-cell-analytics/micro-sam/issues/333
@magic_factory(call_button="Segment Object [S]")
def segment(viewer: "napari.viewer.Viewer", box_extension: float = 0.1) -> None:
    shape = viewer.layers["current_object"].data.shape

    # get the current box and point prompts
    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape)
    points, labels = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], with_stop_annotation=False)

    predictor = AnnotatorState().predictor
    image_embeddings = AnnotatorState().image_embeddings
    if image_embeddings["original_size"] is None:  # tiled prediction
        seg = vutil.prompt_segmentation(
            predictor, points, labels, boxes, masks, shape, image_embeddings=image_embeddings,
            multiple_box_prompts=True, box_extension=box_extension,
        )
    else:  # normal prediction and we have set the precomputed embeddings already
        seg = vutil.prompt_segmentation(
            predictor, points, labels, boxes, masks, shape, multiple_box_prompts=True, box_extension=box_extension,
        )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    viewer.layers["current_object"].data = seg
    viewer.layers["current_object"].refresh()


@magic_factory(call_button="Segment Slice [S]")
def segment_slice(viewer: "napari.viewer.Viewer", box_extension: float = 0.1) -> None:
    shape = viewer.layers["current_object"].data.shape[1:]
    position = viewer.cursor.position
    z = int(position[0])

    point_prompts = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], z)
    # this is a stop prompt, we do nothing
    if not point_prompts:
        return

    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape, i=z)
    points, labels = point_prompts

    state = AnnotatorState()
    seg = vutil.prompt_segmentation(
        state.predictor, points, labels, boxes, masks, shape, multiple_box_prompts=False,
        image_embeddings=state.image_embeddings, i=z, box_extension=box_extension,
    )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    viewer.layers["current_object"].data[z] = seg
    viewer.layers["current_object"].refresh()


# TODO should probably be wrappred in a thread worker
# See https://github.com/computational-cell-analytics/micro-sam/issues/334
@magic_factory(
    call_button="Segment All Slices [Shift-S]",
    projection={"choices": ["default", "bounding_box", "mask", "points"]},
)
def segment_object(
    viewer: "napari.viewer.Viewer",
    iou_threshold: float = 0.8,
    projection: str = "default",
    box_extension: float = 0.05,
) -> None:

    # we have the following projection modes:
    # bounding_box: uses only the bounding box as prompt
    # mask: uses the bounding box and the mask
    # points: uses the bounding box, mask and points derived from the mask
    # by default we choose mask, which qualitatively seems to work the best
    projection = "mask" if projection == "default" else projection

    state = AnnotatorState()
    shape = state.image_shape

    with progress(total=shape[0]) as progress_bar:

        # step 1: segment all slices with prompts
        seg, slices, stop_lower, stop_upper = vutil.segment_slices_with_prompts(
            state.predictor, viewer.layers["point_prompts"], viewer.layers["prompts"],
            state.image_embeddings, shape,
            progress_bar=progress_bar,
        )

        # step 2: segment the rest of the volume based on smart prompting
        seg = segment_mask_in_volume(
            seg, state.predictor, state.image_embeddings, slices,
            stop_lower, stop_upper,
            iou_threshold=iou_threshold, projection=projection,
            progress_bar=progress_bar, box_extension=box_extension,
        )

    viewer.layers["current_object"].data = seg
    viewer.layers["current_object"].refresh()


def _update_lineage(viewer):
    """Updated the lineage after recording a division event.
    This helper function is needed by 'track_object_widget'.
    """
    state = AnnotatorState()
    tracking_widget = state.tracking_widget

    mother = state.current_track_id
    assert mother in state.lineage
    assert len(state.lineage[mother]) == 0

    daughter1, daughter2 = state.current_track_id + 1, state.current_track_id + 2
    state.lineage[mother] = [daughter1, daughter2]
    state.lineage[daughter1] = []
    state.lineage[daughter2] = []

    # Update the choices in the track_id menu so that it contains the new track ids.
    track_ids = list(map(str, state.lineage.keys()))
    tracking_widget[1].choices = track_ids

    viewer.layers["point_prompts"].property_choices["track_id"] = [str(track_id) for track_id in track_ids]
    viewer.layers["prompts"].property_choices["track_id"] = [str(track_id) for track_id in track_ids]


@magic_factory(call_button="Segment Frame [S]")
def segment_frame(viewer: "napari.viewer.Viewer") -> None:
    state = AnnotatorState()
    shape = state.image_shape[1:]
    position = viewer.cursor.position
    t = int(position[0])

    point_prompts = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], i=t, track_id=state.current_track_id)
    # this is a stop prompt, we do nothing
    if not point_prompts:
        return

    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape, i=t, track_id=state.current_track_id)
    points, labels = point_prompts

    seg = vutil.prompt_segmentation(
        state.predictor, points, labels, boxes, masks, shape, multiple_box_prompts=False,
        image_embeddings=state.image_embeddings, i=t
    )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    # clear the old segmentation for this track_id
    old_mask = viewer.layers["current_object"].data[t] == state.current_track_id
    viewer.layers["current_object"].data[t][old_mask] = 0
    # set the new segmentation
    new_mask = seg.squeeze() == 1
    viewer.layers["current_object"].data[t][new_mask] = state.current_track_id
    viewer.layers["current_object"].refresh()


# TODO should probably be wrappred in a thread worker
@magic_factory(call_button="Track Object [Shift-S]", projection={"choices": ["default", "bounding_box", "mask"]})
def track_object(
    viewer: "napari.viewer.Viewer",
    iou_threshold: float = 0.5,
    projection: str = "default",
    motion_smoothing: float = 0.5,
    box_extension: float = 0.1,
) -> None:
    state = AnnotatorState()
    shape = state.image_shape

    # we use the bounding box projection method as default which generally seems to work better for larger changes
    # between frames (which is pretty tyipical for tracking compared to 3d segmentation)
    projection_ = "mask" if projection == "default" else projection

    with progress(total=shape[0]) as progress_bar:
        # step 1: segment all slices with prompts
        seg, slices, _, stop_upper = vutil.segment_slices_with_prompts(
            state.predictor, viewer.layers["point_prompts"], viewer.layers["prompts"],
            state.image_embeddings, shape,
            progress_bar=progress_bar, track_id=state.current_track_id
        )

        # step 2: track the object starting from the lowest annotated slice
        seg, has_division = vutil.track_from_prompts(
            viewer.layers["point_prompts"], viewer.layers["prompts"], seg,
            state.predictor, slices, state.image_embeddings, stop_upper,
            threshold=iou_threshold, projection=projection_,
            progress_bar=progress_bar, motion_smoothing=motion_smoothing,
            box_extension=box_extension,
        )

    # If a division has occurred and it's the first time it occurred for this track
    # then we need to create the two daughter tracks and update the lineage.
    if has_division and (len(state.lineage[state.current_track_id]) == 0):
        _update_lineage(viewer)

    # clear the old track mask
    viewer.layers["current_object"].data[viewer.layers["current_object"].data == state.current_track_id] = 0
    # set the new object mask
    viewer.layers["current_object"].data[seg == 1] = state.current_track_id
    viewer.layers["current_object"].refresh()


#
# Widgets for automatic segmentation:
# - amg_2d: AMG widget for the 2d annotation tool
# - amg_3d: AMG widget for the 3d annotation tool
#


# TODO should be wrapped in a threadworker
@magic_factory(
    call_button="Automatic Segmentation",
    min_object_size={"min": 0, "max": 10000},
)
def amg_2d(
    viewer: "napari.viewer.Viewer",
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_object_size: int = 100,
    with_background: bool = True,
) -> None:
    state = AnnotatorState()

    is_tiled = state.image_embeddings["input_size"] is None
    if state.amg is None:
        state.amg = instance_segmentation.get_amg(state.predictor, is_tiled)

    shape = state.image_shape
    if not state.amg.is_initialized:
        # We don't need to pass the actual image data here, since the embeddings are passed.
        # (The image data is only used by the amg to compute image embeddings, so not needed here.)
        dummy_image = np.zeros(shape, dtype="uint8")
        state.amg.initialize(dummy_image, image_embeddings=state.image_embeddings, verbose=True)

    seg = state.amg.generate(pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)

    seg = instance_segmentation.mask_data_to_segmentation(
        seg, with_background=with_background, min_object_size=min_object_size
    )
    assert isinstance(seg, np.ndarray)

    viewer.layers["auto_segmentation"].data = seg
    viewer.layers["auto_segmentation"].refresh()


# TODO should be wrapped in a threadworker
@magic_factory(
    call_button="Automatic Segmentation",
    min_object_size={"min": 0, "max": 10000}
)
def amg_3d(
    viewer: "napari.viewer.Viewer",
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_object_size: int = 100,
    with_background: bool = True,
) -> None:
    state = AnnotatorState()

    is_tiled = state.image_embeddings["input_size"] is None
    if state.amg is None:
        state.amg = instance_segmentation.get_amg(state.predictor, is_tiled)

    i = int(viewer.cursor.position[0])
    shape = state.image_shape[-2:]

    if i in state.amg_state:
        amg_state_i = state.amg_state[i]
        state.amg.set_state(amg_state_i)

    else:
        # we don't need to pass the actual image data here, since the embeddings are passed
        # (the image data is only used by the amg to compute image embeddings, so not needed here)
        dummy_image = np.zeros(shape, dtype="uint8")

        state.amg.initialize(dummy_image, image_embeddings=state.image_embeddings, verbose=True, i=i)
        amg_state_i = state.amg.get_state()

        state.amg_state[i] = amg_state_i
        cache_folder = state.amg_state["cache_folder"]
        if cache_folder is not None:
            cache_path = os.path.join(cache_folder, f"state-{i}.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump(amg_state_i, f)

    seg = state.amg.generate(pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)

    seg = instance_segmentation.mask_data_to_segmentation(
        seg, with_background=with_background, min_object_size=min_object_size
    )
    assert isinstance(seg, np.ndarray)

    viewer.layers["auto_segmentation"].data[i] = seg
    viewer.layers["auto_segmentation"].refresh()
