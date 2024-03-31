"""Implements the widgets used in the annotation plugins.
"""

import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import Optional

import elf.parallel
import h5py
import napari
import numpy as np
import zarr
import z5py

from qtpy import QtWidgets
from qtpy.QtCore import QObject, Signal
from superqt import QCollapsible
from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container, create_widget
from napari.qt.threading import thread_worker
from napari.utils import progress
from zarr.errors import PathNotFoundError

from ._state import AnnotatorState
from . import util as vutil
from .. import instance_segmentation, util
from ..multi_dimensional_segmentation import segment_mask_in_volume, merge_instance_segmentation_3d, PROJECTION_MODES


#
# Convenience functionality for creating QT UI and manipulating the napari viewer.
#


def _select_layer(viewer, layer_name):
    viewer.layers.selection.select_only(viewer.layers[layer_name])


# Create a collapsible around the widget
def _make_collapsible(widget, title):
    parent_widget = QtWidgets.QWidget()
    parent_widget.setLayout(QtWidgets.QVBoxLayout())
    collapsible = QCollapsible(title, parent_widget)
    collapsible.addWidget(widget)
    parent_widget.layout().addWidget(collapsible)
    return parent_widget


# Base class for a widget with convenience functionality for adding parameters.
class _WidgetBase(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QtWidgets.QVBoxLayout())

    def _add_boolean_param(self, name, value, title=None):
        checkbox = QtWidgets.QCheckBox(name if title is None else title)
        checkbox.setChecked(value)
        checkbox.stateChanged.connect(lambda val: setattr(self, name, val))
        return checkbox

    def _add_float_param(self, name, value, title=None, min_val=0.0, max_val=1.0, decimals=2, step=0.01, layout=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel(name if title is None else title))
        param = QtWidgets.QDoubleSpinBox()
        param.setRange(min_val, max_val)
        param.setDecimals(decimals)
        param.setValue(value)
        param.setSingleStep(step)
        param.valueChanged.connect(lambda val: setattr(self, name, val))
        layout.addWidget(param)
        return param, layout

    def _add_int_param(self, name, value, min_val, max_val, title=None, step=1, layout=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel(name if title is None else title))
        param = QtWidgets.QSpinBox()
        param.setRange(min_val, max_val)
        param.setValue(value)
        param.setSingleStep(step)
        param.valueChanged.connect(lambda val: setattr(self, name, val))
        layout.addWidget(param)
        return param, layout

    def _add_choice_param(self, name, value, options, title=None, layout=None, update=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel(name if title is None else title))

        # Create the dropdown menu via QComboBox, set the available values.
        dropdown = QtWidgets.QComboBox()
        dropdown.addItems(options)
        if update is None:
            dropdown.currentIndexChanged.connect(lambda index: setattr(self, name, options[index]))
        else:
            dropdown.currentIndexChanged.connect(update)

        # Set the correct value for the value.
        dropdown.setCurrentIndex(dropdown.findText(value))

        layout.addWidget(dropdown)
        return dropdown, layout

    def _add_shape_param(self, names, values, min_val, max_val, step=1):
        layout = QtWidgets.QHBoxLayout()

        x_layout = QtWidgets.QVBoxLayout()
        x_param, _ = self._add_int_param(
            names[0], values[0], min_val=min_val, max_val=max_val, layout=x_layout, step=step
        )
        layout.addLayout(x_layout)

        y_layout = QtWidgets.QVBoxLayout()
        y_param, _ = self._add_int_param(
            names[1], values[1], min_val=min_val, max_val=max_val, layout=y_layout, step=step
        )
        layout.addLayout(y_layout)

        return x_param, y_param, layout


# Custom signals for managing progress updates.
class PBarSignals(QObject):
    pbar_total = Signal(int)
    pbar_update = Signal(int)
    pbar_description = Signal(str)
    pbar_stop = Signal()


# Set up the progress bar. We handle this via custom signals that are passed as callbacks to the
# function that does the actual work. We need callbacks for initializing the progress bar,
# updating it and for stopping the progress bar.
def _create_pbar_for_threadworker():
    pbar = progress()
    pbar_signals = PBarSignals()
    pbar_signals.pbar_total.connect(lambda total: setattr(pbar, "total", total))
    pbar_signals.pbar_update.connect(lambda update: pbar.update(update))
    pbar_signals.pbar_description.connect(lambda description: pbar.set_description(description))
    pbar_signals.pbar_stop.connect(lambda: pbar.close())
    return pbar, pbar_signals


def _reset_tracking_state(viewer):
    """Reset the tracking state.

    This helper function is needed by the widgets clear_track and by commit_track.
    """
    state = AnnotatorState()

    # Reset the lineage and track id.
    state.current_track_id = 1
    state.lineage = {1: []}

    # Reset the layer properties.
    viewer.layers["point_prompts"].property_choices["track_id"] = ["1"]
    viewer.layers["prompts"].property_choices["track_id"] = ["1"]

    # Reset the choices in the track_id menu.
    state.widgets["tracking"][1].value = "1"
    state.widgets["tracking"][1].choices = ["1"]


#
# Widgets implemented with magicgui.
#


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear(viewer: "napari.viewer.Viewer") -> None:
    """Widget for clearing the current annotations."""
    vutil.clear_annotations(viewer)


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear_volume(viewer: "napari.viewer.Viewer", all_slices: bool = True) -> None:
    """Widget for clearing the current annotations in 3D."""
    if all_slices:
        vutil.clear_annotations(viewer)
    else:
        i = int(viewer.cursor.position[0])
        vutil.clear_annotations_slice(viewer, i=i)


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear_track(viewer: "napari.viewer.Viewer", all_frames: bool = True) -> None:
    """Widget for clearing all tracking annotations and state."""
    if all_frames:
        _reset_tracking_state(viewer)
        vutil.clear_annotations(viewer)
    else:
        i = int(viewer.cursor.position[0])
        vutil.clear_annotations_slice(viewer, i=i)


def _commit_impl(viewer, layer):
    # Check if we have a z_range. If yes, use it to set a bounding box.
    state = AnnotatorState()
    if state.z_range is None:
        bb = np.s_[:]
    else:
        z_min, z_max = state.z_range
        bb = np.s_[z_min:(z_max+1)]

    seg = viewer.layers[layer].data[bb]
    shape = seg.shape

    # We parallelize these operatios because they take quite long for large volumes.

    # Compute the max id in the commited objects.
    # id_offset = int(viewer.layers["committed_objects"].data.max())
    full_shape = viewer.layers["committed_objects"].data.shape
    id_offset = int(
        elf.parallel.max(viewer.layers["committed_objects"].data, block_shape=util.get_block_shape(full_shape))
    )

    # Compute the mask for the current object.
    # mask = seg != 0
    mask = np.zeros(seg.shape, dtype="bool")
    mask = elf.parallel.apply_operation(
        seg, 0, np.not_equal, out=mask, block_shape=util.get_block_shape(shape)
    )

    # Write the current object to committed objects.
    seg[mask] += id_offset
    viewer.layers["committed_objects"].data[bb][mask] = seg[mask]
    viewer.layers["committed_objects"].refresh()

    return id_offset, seg, mask, bb


# TODO also keep track of the model being used and the micro-sam version.
def _commit_to_file(path, viewer, layer, seg, mask, bb, extra_attrs=None):

    # NOTE: Zarr is incredibly inefficient and writes empty blocks.
    # So we have to use z5py here.

    # Deal with issues z5py has with empty folders and require the json.
    if os.path.exists(path):
        required_json = os.path.join(path, ".zgroup")
        if not os.path.exists(required_json):
            with open(required_json, "w") as f:
                json.dump({"zarr_format": 2}, f)

    f = z5py.ZarrFile(path, "a")

    # Write the segmentation.
    full_shape = viewer.layers["committed_objects"].data.shape
    block_shape = util.get_block_shape(full_shape)
    ds = f.require_dataset(
        "committed_objects", shape=full_shape, chunks=block_shape, compression="gzip", dtype=seg.dtype
    )
    ds.n_threads = mp.cpu_count()
    data = ds[bb]
    data[mask] = seg[mask]
    ds[bb] = data

    # Write additional information to attrs.
    if extra_attrs is not None:
        f.attrs.update(extra_attrs)

    # If we run commit from the automatic segmentation we don't have
    # any prompts and so don't need to commit anything else.
    if layer == "auto_segmentation":
        return

    def write_prompts(object_id, prompts, point_prompts):
        g = f.create_group(f"prompts/{object_id}")
        if prompts is not None and len(prompts) > 0:
            data = np.array(prompts)
            g.create_dataset("prompts", data=data, chunks=data.shape)
        if point_prompts is not None and len(point_prompts) > 0:
            g.create_dataset("point_prompts", data=point_prompts, chunks=point_prompts.shape)

    # Commit the prompts for all the objects in the commit.
    object_ids = np.unique(seg[mask])
    if len(object_ids) == 1:  # We only have a single object.
        write_prompts(object_ids[0], viewer.layers["prompts"].data, viewer.layers["point_prompts"].data)
    else:
        have_prompts = len(viewer.layers["prompts"].data) > 0
        have_point_prompts = len(viewer.layers["point_prompts"].data) > 0
        if have_prompts and not have_point_prompts:
            prompts = viewer.layers["prompts"].data
            point_prompts = None
        elif not have_prompts and have_point_prompts:
            prompts = None
            point_prompts = viewer.layers["point_prompts"].data
        else:
            msg = "Got multiple objects from interactive segmentation with box and point prompts." if (
                have_prompts and have_point_prompts
            ) else "Got multiple objects from interactive segmentation with neither box or point prompts."
            raise RuntimeError(msg)

        for i, object_id in enumerate(object_ids):
            write_prompts(
                object_id,
                None if prompts is None else prompts[i:i+1],
                None if point_prompts is None else point_prompts[i:i+1]
            )


@magic_factory(
    call_button="Commit [C]",
    layer={"choices": ["current_object", "auto_segmentation"]},
    commit_path={"mode": "d"},  # choose a directory
)
def commit(
    viewer: "napari.viewer.Viewer",
    layer: str = "current_object",
    commit_path: Optional[Path] = None,
) -> None:
    """Widget for committing the segmented objects from automatic or interactive segmentation."""
    _, seg, mask, bb = _commit_impl(viewer, layer)

    if commit_path is not None:
        _commit_to_file(commit_path, viewer, layer, seg, mask, bb)

    if layer == "current_object":
        vutil.clear_annotations(viewer)
    else:
        viewer.layers["auto_segmentation"].data = np.zeros(
            viewer.layers["auto_segmentation"].data.shape, dtype="uint32"
        )
        viewer.layers["auto_segmentation"].refresh()
        _select_layer(viewer, "committed_objects")


@magic_factory(
    call_button="Commit [C]",
    layer={"choices": ["current_object"]},
    commit_path={"mode": "d"},  # choose a directory
)
def commit_track(
    viewer: "napari.viewer.Viewer",
    layer: str = "current_object",
    commit_path: Optional[Path] = None,
) -> None:
    """Widget for committing the segmented objects from interactive tracking."""
    # Commit the segmentation layer.
    id_offset, seg, mask, bb = _commit_impl(viewer, layer)

    # Update the lineages.
    state = AnnotatorState()
    updated_lineage = {
        parent + id_offset: [child + id_offset for child in children] for parent, children in state.lineage.items()
    }
    state.committed_lineages.append(updated_lineage)

    if commit_path is not None:
        _commit_to_file(
            commit_path, viewer, layer, seg, mask, bb,
            extra_attrs={"committed_lineages": state.committed_lineages}
        )

    if layer == "current_object":
        vutil.clear_annotations(viewer)

    # Reset the tracking state.
    _reset_tracking_state(viewer)


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


@magic_factory(call_button="Segment Object [S]")
def segment(viewer: "napari.viewer.Viewer", batched: bool = False) -> None:
    shape = viewer.layers["current_object"].data.shape

    # get the current box and point prompts
    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape)
    points, labels = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], with_stop_annotation=False)

    predictor = AnnotatorState().predictor
    image_embeddings = AnnotatorState().image_embeddings
    seg = vutil.prompt_segmentation(
        predictor, points, labels, boxes, masks, shape, image_embeddings=image_embeddings,
        multiple_box_prompts=True, multiple_point_prompts=batched,
    )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    viewer.layers["current_object"].data = seg
    viewer.layers["current_object"].refresh()


@magic_factory(call_button="Segment Slice [S]")
def segment_slice(viewer: "napari.viewer.Viewer") -> None:
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
        image_embeddings=state.image_embeddings, i=z,
    )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    viewer.layers["current_object"].data[z] = seg
    viewer.layers["current_object"].refresh()


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


#
# Functionality and widget to compute the image embeddings.
#


def _process_tiling_inputs(tile_shape_x, tile_shape_y, halo_x, halo_y):
    tile_shape = (tile_shape_x, tile_shape_y)
    halo = (halo_x, halo_y)
    # check if tile_shape/halo are not set: (0, 0)
    if all(item == 0 for item in tile_shape):
        tile_shape = None
    # check if at least 1 param is given
    elif tile_shape[0] == 0 or tile_shape[1] == 0:
        max_val = max(tile_shape[0], tile_shape[1])
        if max_val < 256:  # at least tile shape >256
            max_val = 256
        tile_shape = (max_val, max_val)
    # if both inputs given, check if smaller than 256
    elif tile_shape[0] != 0 and tile_shape[1] != 0:
        if tile_shape[0] < 256:
            tile_shape = (256, tile_shape[1])  # Create a new tuple
        if tile_shape[1] < 256:
            tile_shape = (tile_shape[0], 256)  # Create a new tuple with modified value
    if all(item == 0 for item in halo):
        if tile_shape is not None:
            halo = (0, 0)
        else:
            halo = None
    # check if at least 1 param is given
    elif halo[0] != 0 or halo[1] != 0:
        max_val = max(halo[0], halo[1])
        # don't apply halo if there is no tiling
        if tile_shape is None:
            halo = None
        else:
            halo = (max_val, max_val)
    return tile_shape, halo


class EmbeddingWidget(_WidgetBase):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Create a nested layout for the sections.
        # Section 1: Image and Model.
        section1_layout = QtWidgets.QHBoxLayout()
        section1_layout.addLayout(self._create_image_section())
        section1_layout.addLayout(self._create_model_section())
        self.layout().addLayout(section1_layout)

        # Section 2: Settings (collapsible).
        self.layout().addWidget(self._create_settings_widget())

        # Section 3: The button to trigger the embedding computation.
        self.run_button = QtWidgets.QPushButton("Compute Embeddings")
        self.run_button.clicked.connect(self._compute_image_embeddings)
        self.layout().addWidget(self.run_button)

    def _create_image_section(self):
        image_section = QtWidgets.QVBoxLayout()
        image_section.addWidget(QtWidgets.QLabel("Image Layer:"))

        # Setting a napari layer in QT, see:
        # https://github.com/pyapp-kit/magicgui/blob/main/docs/examples/napari/napari_combine_qt.py
        self.image_selection = create_widget(annotation=napari.layers.Image)
        image_section.addWidget(self.image_selection.native)

        return image_section

    def _update_model(self, index):
        self.model_type = self.model_options[index]
        state = AnnotatorState()
        if "autosegment" in state.widgets:
            vutil._sync_autosegment_widget(state.widgets["autosegment"], self.model_type, self.custom_weights)
        if "segment_nd" in state.widgets:
            vutil._sync_ndsegment_widget(state.widgets["segment_nd"], self.model_type, self.custom_weights)

    def _create_model_section(self):
        self.model_type = util._DEFAULT_MODEL
        self.model_options = list(util.models().urls.keys())
        layout = QtWidgets.QVBoxLayout()
        self.model_dropdown, layout = self._add_choice_param(
            "model_type", self.model_type, self.model_options, title="Model:",
            layout=layout, update=self._update_model
        )
        return layout

    def _create_settings_widget(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # Create UI for the device.
        self.device = "auto"
        device_options = ["auto"] + util._available_devices()
        self.device_dropdown, layout = self._add_choice_param("device", self.device, device_options)
        setting_values.layout().addLayout(layout)

        # TODO
        # save_path: Optional[Path] = None,  # where embeddings for this image are cached (optional, zarr file = folder)
        # custom_weights: Optional[Path] = None,  # A filepath or URL to custom model weights.
        # Create UI for the save path.
        self.save_path = None
        # Create UI for the custom weights.
        self.custom_weights = None

        # Create UI for the tile shape.
        self.tile_x, self.tile_y = 0, 0
        self.tile_x_param, self.tile_y_param, layout = self._add_shape_param(
            ("tile_x", "tile_y"), (self.tile_x, self.tile_y), min_val=0, max_val=2048, step=16
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the halo.
        self.halo_x, self.halo_y = 0, 0
        self.halo_x_param, self.halo_y_param, layout = self._add_shape_param(
            ("halo_x", "halo_y"), (self.halo_x, self.halo_y), min_val=0, max_val=512
        )
        setting_values.layout().addLayout(layout)

        settings = _make_collapsible(setting_values, title="Settings")
        return settings

    def _compute_image_embeddings(self):

        # Get the image and model.
        image = self.image_selection.get_value()

        # TODO Do a check if we actually need to recompute the embeddings.

        # Update the image embeddings:
        # Reset the state.
        state = AnnotatorState()
        state.reset_state()

        # Get image dimensions.
        if image.rgb:
            ndim = image.data.ndim - 1
            state.image_shape = image.data.shape[:-1]
        else:
            ndim = image.data.ndim
            state.image_shape = image.data.shape

        # Process tile_shape and halo.
        tile_shape, halo = _process_tiling_inputs(self.tile_x, self.tile_y, self.halo_x, self.halo_y)

        # Set up progress bar and signals for using it within a threadworker.
        pbar, pbar_signals = _create_pbar_for_threadworker()

        @thread_worker()
        def compute_image_embedding(
            state, image_data, save_path, ndim, device, model_type, custom_weights, tile_shape, halo
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

            def pbar_init(total, description):
                pbar_signals.pbar_total.emit(total)
                pbar_signals.pbar_description.emit(description)

            state.initialize_predictor(
                image_data, model_type=model_type, save_path=save_path, ndim=ndim, device=device,
                checkpoint_path=custom_weights, tile_shape=tile_shape, halo=halo,
                pbar_init=pbar_init,
                pbar_update=lambda update: pbar_signals.pbar_update.emit(update),
                pbar_stop=lambda: pbar_signals.pbar_stop.emit()
            )

        worker = compute_image_embedding(
            state, image.data, self.save_path, ndim=ndim, device=self.device, model_type=self.model_type,
            custom_weights=self.custom_weights, tile_shape=tile_shape, halo=halo
        )
        # Note: this is how we can handle the worker when it's done.
        # We can use this e.g. to add an indicator that the embeddings are computed or not.
        worker.returned.connect(lambda _: print("Embeddings for", self.model_type, "have been computed."))
        worker.start()
        return worker


#
# Functionality and widget for nd segmentation.
#


def _update_lineage(viewer):
    """Updated the lineage after recording a division event.
    This helper function is needed by 'track_object'.
    """
    state = AnnotatorState()
    tracking_widget = state.widgets["tracking"]

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


class SegmentNDWidget(_WidgetBase):
    def __init__(self, viewer, tracking, parent=None):
        super().__init__(parent=parent)
        self._viewer = viewer
        self.tracking = tracking

        # Add the settings.
        self.settings = self._create_settings()
        self.layout().addWidget(self.settings)

        # Add the run button.
        button_title = "Segment All Frames [Shift-S]" if self.tracking else "Segment All Slices [Shift-S]"
        self.run_button = QtWidgets.QPushButton(button_title)
        self.run_button.clicked.connect(self._run_segmentation)
        self.layout().addWidget(self.run_button)

    def _create_settings(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # Create the UI for the projection modes.
        self.projection = "points"
        self.projection_dropdown, layout = self._add_choice_param("projection", self.projection, PROJECTION_MODES)
        setting_values.layout().addLayout(layout)

        # Create the UI element for the IOU threshold.
        self.iou_threshold = 0.5
        self.iou_threshold_param, layout = self._add_float_param("iou_threshold", self.iou_threshold)
        setting_values.layout().addLayout(layout)

        # Create the UI element for the box extension.
        self.box_extension = 0.05
        self.box_extension_param, layout = self._add_float_param("box_extension", self.box_extension)
        setting_values.layout().addLayout(layout)

        # Create the UI element for the motion smoothing (if we have the tracking widget).
        if self.tracking:
            self.motion_smoothing = 0.5
            self.motion_smoothing_param, layout = self._add_float_param("motion_smoothing", self.motion_smoothing)
            setting_values.layout().addLayout(layout)

        settings = _make_collapsible(setting_values, title="Settings")
        return settings

    def _run_tracking(self):
        state = AnnotatorState()
        shape = state.image_shape

        with progress(total=shape[0]) as progress_bar:
            # Step 1: Segment all slices with prompts.
            seg, slices, _, stop_upper = vutil.segment_slices_with_prompts(
                state.predictor, self._viewer.layers["point_prompts"], self._viewer.layers["prompts"],
                state.image_embeddings, shape,
                progress_bar=progress_bar, track_id=state.current_track_id
            )

            # Step 2: Track the object starting from the lowest annotated slice.
            seg, has_division = vutil.track_from_prompts(
                self._viewer.layers["point_prompts"], self._viewer.layers["prompts"], seg,
                state.predictor, slices, state.image_embeddings, stop_upper,
                threshold=self.iou_threshold, projection=self.projection,
                progress_bar=progress_bar, motion_smoothing=self.motion_smoothing,
                box_extension=self.box_extension,
            )

        # If a division has occurred and it's the first time it occurred for this track
        # then we need to create the two daughter tracks and update the lineage.
        if has_division and (len(state.lineage[state.current_track_id]) == 0):
            _update_lineage(self._viewer)

        # Clear the old track mask.
        self._viewer.layers["current_object"].data[
            self._viewer.layers["current_object"].data == state.current_track_id
        ] = 0
        # Set the new object mask.
        self._viewer.layers["current_object"].data[seg == 1] = state.current_track_id
        self._viewer.layers["current_object"].refresh()

    def _run_volumetric_segmentation(self):
        state = AnnotatorState()
        shape = state.image_shape

        # TODO
        # pbar, pbar_signals = _create_pbar_for_threadworker()

        with progress(total=shape[0]) as progress_bar:

            # Step 1: Segment all slices with prompts.
            seg, slices, stop_lower, stop_upper = vutil.segment_slices_with_prompts(
                state.predictor, self.viewer.layers["point_prompts"], self.viewer.layers["prompts"],
                state.image_embeddings, shape,
                progress_bar=progress_bar,
            )

            # Step 2: Segment the rest of the volume based on projecting prompts.
            seg, (z_min, z_max) = segment_mask_in_volume(
                seg, state.predictor, state.image_embeddings, slices,
                stop_lower, stop_upper,
                iou_threshold=self.iou_threshold, projection=self.projection,
                progress_bar=progress_bar, box_extension=self.box_extension,
            )

        state.z_range = (z_min, z_max)
        self._viewer.layers["current_object"].data = seg
        self._viewer.layers["current_object"].refresh()

    def _run_segmentation(self):
        if self.tracking:
            self._run_tracking()
        else:
            self._run_volumetric_segmentation()


#
# The functionality and widgets for automatic segmentation.
#


def _instance_segmentation_impl(viewer, with_background, min_object_size, i=None, skip_update=False, **kwargs):
    state = AnnotatorState()

    if state.amg is None:
        is_tiled = state.image_embeddings["input_size"] is None
        state.amg = instance_segmentation.get_amg(state.predictor, is_tiled, decoder=state.decoder)

    shape = state.image_shape

    # Further optimization: refactor parts of this so that we can also use it in the automatic 3d segmentation fucnction
    # For 3D we store the amg state in a dict and check if it is computed already.
    if state.amg_state is not None:
        assert i is not None
        if i in state.amg_state:
            amg_state_i = state.amg_state[i]
            state.amg.set_state(amg_state_i)

        else:
            dummy_image = np.zeros(shape[-2:], dtype="uint8")
            state.amg.initialize(dummy_image, image_embeddings=state.image_embeddings, verbose=True, i=i)
            amg_state_i = state.amg.get_state()
            state.amg_state[i] = amg_state_i

            cache_folder = state.amg_state.get("cache_folder", None)
            if cache_folder is not None:
                cache_path = os.path.join(cache_folder, f"state-{i}.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(amg_state_i, f)

            cache_path = state.amge_state.get("cache_path", None)
            if cache_path is not None:
                save_key = f"state-{i}"
                with h5py.File(cache_path, "a") as f:
                    g = f.create_group(save_key)
                    g.create_dataset("foreground", data=state["foreground"], compression="gzip")
                    g.create_dataset("boundary_distances", data=state["boundary_distances"], compression="gzip")
                    g.create_dataset("center_distances", data=state["center_distances"], compression="gzip")

    # Otherwise (2d segmentation) we just check if the amg is initialized or not.
    elif not state.amg.is_initialized:
        assert i is None
        # We don't need to pass the actual image data here, since the embeddings are passed.
        # (The image data is only used by the amg to compute image embeddings, so not needed here.)
        dummy_image = np.zeros(shape, dtype="uint8")
        state.amg.initialize(dummy_image, image_embeddings=state.image_embeddings, verbose=True)

    seg = state.amg.generate(**kwargs)
    if len(seg) == 0:
        seg = np.zeros(shape[-2:], dtype=viewer.layers["auto_segmentation"].data.dtype)
    else:
        seg = instance_segmentation.mask_data_to_segmentation(
            seg, with_background=with_background, min_object_size=min_object_size
        )
    assert isinstance(seg, np.ndarray)

    if skip_update:
        return seg

    if i is None:
        viewer.layers["auto_segmentation"].data = seg
    else:
        viewer.layers["auto_segmentation"].data[i] = seg
    viewer.layers["auto_segmentation"].refresh()

    return seg


def _segment_volume(viewer, with_background, min_object_size, gap_closing, min_extent, **kwargs):
    segmentation = np.zeros_like(viewer.layers["auto_segmentation"].data)

    offset = 0
    # Further optimization: parallelize if state is precomputed for all slices
    for i in progress(range(segmentation.shape[0]), desc="Segment slices"):
        seg = _instance_segmentation_impl(
            viewer, with_background, min_object_size, i=i, skip_update=True, **kwargs
        )
        seg_max = seg.max()
        if seg_max == 0:
            continue
        seg[seg != 0] += offset
        offset = seg_max + offset
        segmentation[i] = seg

    segmentation = merge_instance_segmentation_3d(
        segmentation, beta=0.5, with_background=with_background, gap_closing=gap_closing,
        min_z_extent=min_extent,
    )

    viewer.layers["auto_segmentation"].data = segmentation
    viewer.layers["auto_segmentation"].refresh()


class AutoSegmentWidget(_WidgetBase):
    def __init__(self, viewer, with_decoder, volumetric, parent=None):
        super().__init__(parent)

        self._viewer = viewer
        self.with_decoder = with_decoder
        self.volumetric = volumetric

        # Add the switch for segmenting the slice vs. the volume if we have a volume.
        if self.volumetric:
            self.layout().addWidget(self._create_volumetric_switch())

        # Add the nested settings widget.
        self.settings = self._create_settings()
        self.layout().addWidget(self.settings)

        # Add the run button.
        self.run_button = QtWidgets.QPushButton("Automatic Segmentation")
        self.run_button.clicked.connect(self._run_segmentation)
        self.layout().addWidget(self.run_button)

    def _create_volumetric_switch(self):
        self.apply_to_volume = False
        return self._add_boolean_param("apply_to_volume", self.apply_to_volume, title="Apply to Volume")

    def _add_common_settings(self, settings):
        # Create the UI element for min object size.
        self.min_object_size = 100
        self.min_obbject_size_param, layout = self._add_int_param(
            "min_object_size", self.min_object_size, min_val=0, max_val=int(1e4)
        )
        settings.layout().addLayout(layout)

        # Create the UI element for with background.
        self.with_background = True
        settings.layout().addWidget(self._add_boolean_param("with_background", self.with_background))

        # Add extra settings for volumetric segmentation: gap_closing and min_extent.
        if self.volumetric:
            self.gap_closing = 2
            self.gap_closing_param, layout = self._add_int_param("gap_closing", self.gap_closing, min_val=0, max_val=10)
            settings.layout().addLayout(layout)

            self.min_extent = 2
            self.min_extent_param, layout = self._add_int_param("min_extent", self.min_extent, min_val=0, max_val=10)
            settings.layout().addLayout(layout)

    def _ais_settings(self):
        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())

        # Create the UI element for center_distance_threshold.
        self.center_distance_thresh = 0.5
        self.center_distance_thresh_param, layout = self._add_float_param(
            "center_distance_thresh", self.center_distance_thresh
        )
        settings.layout().addLayout(layout)

        # Create the UI element for boundary_distance_threshold.
        self.boundary_distance_thresh = 0.5
        self.boundary_distance_thresh_param, layout = self._add_float_param(
            "boundary_distance_thresh", self.boundary_distance_thresh
        )
        settings.layout().addLayout(layout)

        # Add min_object_size and with_background
        self._add_common_settings(settings)

        return settings

    def _amg_settings(self):
        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())

        # Create the UI element for pred_iou_thresh.
        self.pred_iou_thresh = 0.88
        self.pred_iou_thresh_param, layout = self._add_float_param("pred_iou_thresh", self.pred_iou_thresh)
        settings.layout().addLayout(layout)

        # Create the UI element for stability score thresh.
        self.stability_score_thresh = 0.95
        self.stability_score_thresh_param, layout = self._add_float_param(
            "stability_score_thresh", self.stability_score_thresh
        )
        settings.layout().addLayout(layout)

        # Create the UI element for box nms thresh.
        self.box_nms_thresh = 0.7
        self.box_nms_thresh_param, layout = self._add_float_param("box_nms_thresh", self.box_nms_thresh)
        settings.layout().addLayout(layout)

        # Add min_object_size and with_background
        self._add_common_settings(settings)

        return settings

    def _create_settings(self):
        setting_values = self._ais_settings() if self.with_decoder else self._amg_settings()
        settings = _make_collapsible(setting_values, title="Settings")
        return settings

    def _run_segmentation_2d(self, kwargs):
        _instance_segmentation_impl(self._viewer, self.with_background, self.min_object_size, **kwargs)

    def _run_segmentation_3d(self, kwargs):
        if self.apply_to_volume:
            # We refuse to run 3D segmentation with the AMG unless we have a GPU or all embeddings
            # are precomputed. Otherwise this would take too long.
            state = AnnotatorState()
            predictor = state.predictor
            if str(predictor.device) == "cpu" or str(predictor.device) == "mps":
                n_slices = self._viewer.layers["auto_segmentation"].data.shape[0]
                embeddings_are_precomputed = len(state.amg_state) > n_slices
                if not embeddings_are_precomputed:
                    print("Volumetric segmentation with AMG is only supported if you have a GPU.")
                    return

            kwargs.update({"gap_closing": self.gap_closing, "min_extent": self.min_extent})
            _segment_volume(self._viewer, self.with_background, self.min_object_size, **kwargs)

        else:
            i = int(self._viewer.cursor.position[0])
            _instance_segmentation_impl(self._viewer, self.with_background, self.min_object_size, i=i, **kwargs)

    # TODO wrap the computation in a threadworker
    def _run_segmentation(self):
        if self.with_decoder:
            kwargs = {
                "center_distance_threshold": self.center_distance_thresh,
                "boundary_distance_threshold": self.boundary_distance_thresh,
                "min_size": self.min_object_size,
            }
        else:
            kwargs = {
                "pred_iou_thresh": self.pred_iou_thresh,
                "stability_score_thresh": self.stability_score_thresh,
                "box_nms_thresh": self.box_nms_thresh,
            }
        if self.volumetric:
            self._run_segmentation_3d(kwargs)
        else:
            self._run_segmentation_2d(kwargs)
        _select_layer(self._viewer, "auto_segmentation")
