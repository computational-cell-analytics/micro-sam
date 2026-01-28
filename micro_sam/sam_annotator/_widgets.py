"""Implements the widgets used in the annotation plugins.
"""

import os
import gc
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Optional

import h5py
import json
import zarr
import z5py
import napari
import numpy as np

import nifty.ground_truth as ngt

import elf.parallel

from qtpy import QtWidgets
from qtpy.QtCore import QObject, Signal
from superqt import QCollapsible
from napari.utils.notifications import show_info
from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container, create_widget
# We have disabled the thread workers for now because they result in a
# massive slowdown in napari >= 0.5.
# See also https://forum.image.sc/t/napari-thread-worker-leads-to-massive-slowdown/103786
# from napari.qt.threading import thread_worker
from napari.utils import progress

from segment_anything import SamPredictor

from . import util as vutil
from ._tooltips import get_tooltip
from ._state import AnnotatorState
from .. import instance_segmentation, util
from ..multi_dimensional_segmentation import (
    segment_mask_in_volume, merge_instance_segmentation_3d, track_across_frames, PROJECTION_MODES, get_napari_track_data
)


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

    def _add_boolean_param(self, name, value, title=None, tooltip=None):
        checkbox = QtWidgets.QCheckBox(name if title is None else title)
        checkbox.setChecked(value)
        checkbox.stateChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            checkbox.setToolTip(tooltip)
        return checkbox

    def _add_string_param(self, name, value, title=None, placeholder=None, layout=None, tooltip=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)
        param = QtWidgets.QLineEdit()
        param.setText(value)
        if placeholder is not None:
            param.setPlaceholderText(placeholder)
        param.textChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            param.setToolTip(tooltip)
        layout.addWidget(param)
        return param, layout

    def _add_float_param(self, name, value, title=None, min_val=0.0, max_val=1.0, decimals=2,
                         step=0.01, layout=None, tooltip=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)
        param = QtWidgets.QDoubleSpinBox()
        param.setRange(min_val, max_val)
        param.setDecimals(decimals)
        param.setValue(value)
        param.setSingleStep(step)
        param.valueChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            param.setToolTip(tooltip)
        layout.addWidget(param)
        return param, layout

    def _add_int_param(self, name, value, min_val, max_val, title=None, step=1, layout=None, tooltip=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)
        param = QtWidgets.QSpinBox()
        param.setRange(min_val, max_val)
        param.setValue(value)
        param.setSingleStep(step)
        param.valueChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            param.setToolTip(tooltip)
        layout.addWidget(param)
        return param, layout

    def _add_choice_param(self, name, value, options, title=None, layout=None, update=None, tooltip=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)

        # Create the dropdown menu via QComboBox, set the available values.
        dropdown = QtWidgets.QComboBox()
        dropdown.addItems(options)
        if update is None:
            dropdown.currentIndexChanged.connect(lambda index: setattr(self, name, options[index]))
        else:
            dropdown.currentIndexChanged.connect(update)

        # Set the correct value for the value.
        dropdown.setCurrentIndex(dropdown.findText(value))

        if tooltip:
            dropdown.setToolTip(tooltip)

        layout.addWidget(dropdown)
        return dropdown, layout

    def _add_shape_param(self, names, values, min_val, max_val, step=1, title=None, tooltip=None):
        layout = QtWidgets.QHBoxLayout()

        x_layout = QtWidgets.QVBoxLayout()
        x_param, _ = self._add_int_param(
            names[0], values[0], min_val=min_val, max_val=max_val, layout=x_layout, step=step,
            title=title[0] if title is not None else title, tooltip=tooltip
        )
        layout.addLayout(x_layout)

        y_layout = QtWidgets.QVBoxLayout()
        y_param, _ = self._add_int_param(
            names[1], values[1], min_val=min_val, max_val=max_val, layout=y_layout, step=step,
            title=title[1] if title is not None else title, tooltip=tooltip
        )
        layout.addLayout(y_layout)

        return x_param, y_param, layout

    def _add_path_param(self, name, value, select_type, title=None, placeholder=None, tooltip=None):
        assert select_type in ("directory", "file", "both")

        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)

        path_textbox = QtWidgets.QLineEdit()
        path_textbox.setText(str(value))
        if placeholder is not None:
            path_textbox.setPlaceholderText(placeholder)
        path_textbox.textChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            path_textbox.setToolTip(tooltip)

        layout.addWidget(path_textbox)

        def add_path_button(select_type, tooltip=None):
            # Adjust button text.
            button_text = f"Select {select_type.capitalize()}"
            path_button = QtWidgets.QPushButton(button_text)

            # Call appropriate function based on select_type.
            path_button.clicked.connect(lambda: getattr(self, f"_get_{select_type}_path")(name, path_textbox))
            if tooltip:
                path_button.setToolTip(tooltip)
            layout.addWidget(path_button)

        if select_type == "both":
            add_path_button("file")
            add_path_button("directory")

        else:
            add_path_button(select_type)

        return path_textbox, layout

    def _get_directory_path(self, name, textbox, tooltip=None):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        if tooltip:
            directory.setToolTip(tooltip)
        if directory and Path(directory).is_dir():
            textbox.setText(str(directory))
        else:
            # Handle the case where the selected path is not a directory
            print("Invalid directory selected. Please try again.")

    def _get_file_path(self, name, textbox, tooltip=None):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select File", "", "All Files (*)"
        )
        if tooltip:
            file_path.setToolTip(tooltip)
        if file_path and Path(file_path).is_file():
            textbox.setText(str(file_path))
        else:
            # Handle the case where the selected path is not a file
            print("Invalid file selected. Please try again.")

    def _get_model_size_options(self):
        # We store the actual model names mapped to UI labels.
        self.model_size_mapping = {}
        if self.model_family == "Natural Images (SAM)":
            self.model_size_options = list(self._model_size_map.values())
            self.model_size_mapping = {self._model_size_map[k]: f"vit_{k}" for k in self._model_size_map.keys()}
        elif self.model_family == "Natural Images (SAM2)":
            self.model_size_options = list(self._model_size_map.values())
            self.model_size_mapping = {self._model_size_map[k]: f"hvit_{k}" for k in self._model_size_map.keys()}
        else:
            model_suffix = self.supported_dropdown_maps[self.model_family]
            self.model_size_options = []

            for option in self.model_options:
                if option.endswith(model_suffix):
                    # Extract model size character on-the-fly.
                    key = next((k for k in self._model_size_map .keys() if f"vit_{k}" in option), None)
                    if key:
                        size_label = self._model_size_map[key]
                        self.model_size_options.append(size_label)
                        self.model_size_mapping[size_label] = option  # Store the actual model name.

        # We ensure an assorted order of model sizes ('tiny' to 'huge')
        self.model_size_options.sort(key=lambda x: ["tiny", "base", "large", "huge"].index(x))

    def _update_model_type(self):
        # Get currently selected model size (before clearing dropdown)
        current_selection = self.model_size_dropdown.currentText()
        self._get_model_size_options()  # Update model size options dynamically

        # NOTE: We need to prevent recursive updates for this step temporarily.
        self.model_size_dropdown.blockSignals(True)

        # Let's clear and recreate the dropdown.
        self.model_size_dropdown.clear()
        self.model_size_dropdown.addItems(self.model_size_options)

        # We restore the previous selection, if still valid.
        if current_selection in self.model_size_options:
            self.model_size = current_selection
        else:
            if self.model_size_options:  # Default to the first available model size
                self.model_size = self.model_size_options[0]

        # Let's map the selection to the correct model type (eg. "tiny" -> "vit_t")
        size_key = next(
            (k for k, v in self._model_size_map.items() if v == self.model_size), "b"
        )
        if "SAM2" in self.model_family:
            self.model_type = f"hvit_{size_key}"
        else:
            self.model_type = f"vit_{size_key}" + self.supported_dropdown_maps[self.model_family]

        self.model_size_dropdown.setCurrentText(self.model_size)  # Apply the selected text to the dropdown

        # We force a refresh for UI here.
        self.model_size_dropdown.update()

        # NOTE: And finally, we should re-enable signals again.
        self.model_size_dropdown.blockSignals(False)

    def _create_model_section(self, default_model: str = util._DEFAULT_MODEL, create_layout: bool = True):

        # Create a list of support dropdown values and correspond them to suffixes.
        self.supported_dropdown_maps = {
            "Natural Images (SAM)": "",
            "Natural Images (SAM2)": "_sam2",
            "Light Microscopy": "_lm",
            "Electron Microscopy": "_em_organelles",
            "Medical Imaging": "_medical_imaging",
            "Histopathology": "_histopathology",
        }

        # NOTE: The available options for all are either 'tiny', 'base', 'large' or 'huge'.
        self._model_size_map = {"t": "tiny", "b": "base", "l": "large", "h": "huge"}

        self._default_model_choice = default_model
        # Let's set the literally default model choice depending on 'micro-sam'.
        self.model_family = {v: k for k, v in self.supported_dropdown_maps.items()}[self._default_model_choice[5:]]

        kwargs = {}
        if create_layout:
            layout = QtWidgets.QVBoxLayout()
            kwargs["layout"] = layout

        # NOTE: We stick to the base variant for each model family.
        # i.e. 'Natural Images (SAM)', 'Light Microscopy', 'Electron Microscopy', 'Medical_Imaging', 'Histopathology'.
        self.model_family_dropdown, layout = self._add_choice_param(
            "model_family", self.model_family, list(self.supported_dropdown_maps.keys()),
            title="Model:", tooltip=get_tooltip("embedding", "model_family"), **kwargs,
        )
        self.model_family_dropdown.currentTextChanged.connect(self._update_model_type)
        return layout

    def _create_model_size_section(self):

        # Create UI for the model size.
        # This would combine with the chosen 'self.model_family' and depend on 'self._default_model_choice'.
        self.model_size = self._model_size_map[self._default_model_choice[4]]

        # Get all model options.
        self.model_options = list(util.models().urls.keys())
        # Filter out the decoders from the model list.
        self.model_options = [model for model in self.model_options if not model.endswith("decoder")]

        # Now, we get the available sizes per model family.
        self._get_model_size_options()

        self.model_size_dropdown, layout = self._add_choice_param(
            "model_size", self.model_size, self.model_size_options,
            title="model size:", tooltip=get_tooltip("embedding", "model_size"),
        )
        self.model_size_dropdown.currentTextChanged.connect(self._update_model_type)
        return layout

    def _validate_model_type_and_custom_weights(self):
        # Let's get all model combination stuff into the desired `model_type` structure.
        if "SAM2" in self.model_family:
            self.model_type = "hvit_" + self.model_size[0]
        else:
            self.model_type = "vit_" + self.model_size[0] + self.supported_dropdown_maps[self.model_family]

        # For 'custom_weights', we remove the displayed text on top of the drop-down menu.
        if self.custom_weights:
            # NOTE: We prevent recursive updates for this step temporarily.
            self.model_family_dropdown.blockSignals(True)
            self.model_family_dropdown.setCurrentIndex(-1)  # This removes the displayed text.
            self.model_family_dropdown.update()
            # NOTE: And re-enable signals again.
            self.model_family_dropdown.blockSignals(False)


# Custom signals for managing progress updates.
class PBarSignals(QObject):
    pbar_total = Signal(int)
    pbar_update = Signal(int)
    pbar_description = Signal(str)
    pbar_stop = Signal()
    pbar_reset = Signal()


class InfoDialog(QtWidgets.QDialog):
    def __init__(self, title, message):
        super().__init__()
        self.setWindowTitle(title)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel(message))

        # Add buttons
        button_box = QtWidgets.QHBoxLayout()  # Use QHBoxLayout for buttons side-by-side
        accept_button = QtWidgets.QPushButton("OK")
        accept_button.clicked.connect(lambda: self.button_clicked(accept_button))  # Connect to clicked signal
        button_box.addWidget(accept_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(lambda: self.button_clicked(cancel_button))  # Connect to clicked signal
        button_box.addWidget(cancel_button)

        layout.addLayout(button_box)
        self.setLayout(layout)

    def button_clicked(self, button):
        if button.text() == "OK":
            self.accept()  # Accept the dialog
        else:
            self.reject()  # Reject the dialog (Cancel)


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
    pbar_signals.pbar_reset.connect(lambda: pbar.reset())
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
    """Widget for clearing the current annotations.

    Args:
        viewer: The napari viewer.
    """
    vutil.clear_annotations(viewer)

    # Perform garbage collection.
    gc.collect()


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear_volume(viewer: "napari.viewer.Viewer", all_slices: bool = True) -> None:
    """Widget for clearing the current annotations in 3D.

    Args:
        viewer: The napari viewer.
        all_slices: Choose whether to clear the annotations for all or only the current slice.
    """
    state = AnnotatorState()

    if all_slices:
        vutil.clear_annotations(viewer)
    else:
        i = int(viewer.dims.point[0])
        vutil.clear_annotations_slice(viewer, i=i)

    # If it's a SAM2 promptable segmentation workflow, we should reset the prompts after clear annotations has been clicked.
    if state.interactive_segmenter is not None:
        state.interactive_segmenter.reset_predictor()

    # Perform garbage collection.
    gc.collect()


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear_track(viewer: "napari.viewer.Viewer", all_frames: bool = True) -> None:
    """Widget for clearing all tracking annotations and state.

    Args:
        viewer: The napari viewer.
        all_frames: Choose whether to clear the annotations for all or only the current frame.
    """
    if all_frames:
        _reset_tracking_state(viewer)
        vutil.clear_annotations(viewer)
    else:
        i = int(viewer.dims.point[0])
        vutil.clear_annotations_slice(viewer, i=i)

    # Perform garbage collection.
    gc.collect()


def _mask_matched_objects(seg, prev_seg, preservation_threshold):
    prev_ids = np.unique(prev_seg)
    ovlp = ngt.overlap(prev_seg, seg)

    mask_ids, prev_mask_ids = [], []
    for prev_id in prev_ids:
        seg_ids, overlaps = ovlp.overlapArrays(prev_id, True)
        if seg_ids[0] != 0 and overlaps[0] >= preservation_threshold:
            mask_ids.append(seg_ids[0])
            prev_mask_ids.append(prev_id)

    preserve_mask = np.logical_or(np.isin(seg, mask_ids), np.isin(prev_seg, prev_mask_ids))
    return preserve_mask


def _commit_impl(viewer, layer, preserve_mode, preservation_threshold):
    state = AnnotatorState()

    # Check whether all layers exist as expected or create new ones automatically.
    state.annotator._require_layers(layer_choices=[layer, "committed_objects"])

    # Check if we have a z_range. If yes, use it to set a bounding box.
    if state.z_range is None:
        bb = np.s_[:]
    else:
        z_min, z_max = state.z_range
        bb = np.s_[z_min:(z_max+1)]

    # Cast the dtype of the segmentation we work with correctly.
    # Otherwise we run into type conversion errors later.
    dtype = viewer.layers["committed_objects"].data.dtype
    seg = viewer.layers[layer].data[bb].astype(dtype)
    shape = seg.shape

    # We parallelize these operations because they take quite long for large volumes.

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
    if preserve_mode != "none":
        prev_seg = viewer.layers["committed_objects"].data[bb]
        # The mode 'pixels' corresponds to a naive implementation where only committed pixels are preserved.
        preserve_mask = prev_seg != 0
        # If the preserve mask is empty we don't need to do anything else here, because we don't have prev objects.
        if preserve_mask.sum() != 0:
            # In the mode 'objects' we preserve committed objects instead, by comparing the overlaps
            # of already committed and newly committed objects.
            if preserve_mode == "objects":
                preserve_mask = _mask_matched_objects(seg, prev_seg, preservation_threshold)
            mask[preserve_mask] = 0

    # Write the current object to committed objects.
    seg[mask] += id_offset
    viewer.layers["committed_objects"].data[bb][mask] = seg[mask]
    viewer.layers["committed_objects"].refresh()

    # If it's a SAM2 promptable segmentation workflow, we should reset the prompts after commit has been clicked.
    if state.interactive_segmenter is not None:
        state.interactive_segmenter.reset_predictor()

    return id_offset, seg, mask, bb


def _get_auto_segmentation_options(state, object_ids):
    widget = state.widgets["autosegment"]

    segmentation_options = {"object_ids": [int(object_id) for object_id in object_ids]}
    if widget.with_decoder:
        segmentation_options["boundary_distance_thresh"] = widget.boundary_distance_thresh
        segmentation_options["center_distance_thresh"] = widget.center_distance_thresh
    else:
        segmentation_options["pred_iou_thresh"] = widget.pred_iou_thresh
        segmentation_options["stability_score_thresh"] = widget.stability_score_thresh
        segmentation_options["box_nms_thresh"] = widget.box_nms_thresh

    segmentation_options["min_object_size"] = widget.min_object_size
    if widget.volumetric:
        segmentation_options["apply_to_volume"] = widget.apply_to_volume
        segmentation_options["gap_closing"] = widget.gap_closing
        segmentation_options["min_extent"] = widget.min_extent

    return segmentation_options


def _get_promptable_segmentation_options(state, object_ids):
    segmentation_options = {"object_ids": [int(object_id) for object_id in object_ids]}
    is_tracking = False
    if "segment_nd" in state.widgets:
        widget = state.widgets["segment_nd"]
        segmentation_options["projection"] = widget.projection
        segmentation_options["iou_threshold"] = widget.iou_threshold
        segmentation_options["box_extension"] = widget.box_extension
        if widget.tracking:
            segmentation_options["motion_smoothing"] = widget.motion_smoothing
            is_tracking = True
    return segmentation_options, is_tracking


def _commit_to_file(path, viewer, layer, seg, mask, bb, extra_attrs=None):

    # NOTE: zarr-python is quite inefficient and writes empty blocks.
    # So we have to use z5py here.

    # Deal with issues z5py has with empty folders and require the json.
    if os.path.exists(path):
        required_json = os.path.join(path, ".zgroup")
        if not os.path.exists(required_json):
            with open(required_json, "w") as f:
                json.dump({"zarr_format": 2}, f)

    f = z5py.ZarrFile(path, "a")
    state = AnnotatorState()

    def _save_signature(f, data_signature):
        embeds = state.widgets["embeddings"]
        tile_shape, halo = _process_tiling_inputs(embeds.tile_x, embeds.tile_y, embeds.halo_x, embeds.halo_y)
        signature = util._get_embedding_signature(
            input_=None,  # We don't need this because we pass the data signature.
            predictor=state.predictor,
            tile_shape=tile_shape,
            halo=halo,
            data_signature=data_signature,
        )
        for key, val in signature.items():
            f.attrs[key] = val

    # If the data signature is saved in the file already,
    # then we check if saved data signature and data signature of our image agree.
    # If not, this file was used for committing objects from another file.
    if "data_signature" in f.attrs:
        saved_signature = f.attrs["data_signature"]
        current_signature = state.data_signature
        if saved_signature != current_signature:  # Signatures disagree.
            msg = f"The commit_path {path} was already used for saving annotations for different image data:\n"
            msg += f"The data signatures are different: {saved_signature} != {current_signature}.\n"
            msg += "Press 'Ok' to remove the data already stored in that file and continue annotation.\n"
            msg += "Otherwise please select a different file path."
            skip_clear = _generate_message("info", msg)
            if skip_clear:
                return
            else:
                f = z5py.ZarrFile(path, "w")
                _save_signature(f, current_signature)
    # Otherwise (data signature not saved yet), write the current signature.
    else:
        _save_signature(f, state.data_signature)

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

    # Get the commit history and the objects that are being commited.
    commit_history = f.attrs.get("commit_history", [])
    object_ids = np.unique(seg[mask])

    # We committed an automatic segmentation.
    if layer == "auto_segmentation":
        # Save the settings of the segmentation widget.
        segmentation_options = _get_auto_segmentation_options(state, object_ids)
        commit_history.append({"auto_segmentation": segmentation_options})

        # Write the commit history.
        f.attrs["commit_history"] = commit_history

        # If we run commit from the automatic segmentation we don't have
        # any prompts and so don't need to commit anything else.
        return

    segmentation_options, is_tracking = _get_promptable_segmentation_options(state, object_ids)
    commit_history.append({"current_object": segmentation_options})

    def write_prompts(object_id, prompts, point_prompts, point_labels, track_state=None):
        g = f.create_group(f"prompts/{object_id}")
        if prompts is not None and len(prompts) > 0:
            data = np.array(prompts)
            g.create_dataset("prompts", data=data, shape=data.shape, chunks=data.shape)
        if point_prompts is not None and len(point_prompts) > 0:
            g.create_dataset("point_prompts", data=point_prompts, shape=data.shape, chunks=point_prompts.shape)
            ds = g.create_dataset("point_labels", data=point_labels, shape=data.shape, chunks=point_labels.shape)
            if track_state is not None:
                ds.attrs["track_state"] = track_state.tolist()

    # Get the prompts from the layers.
    prompts = viewer.layers["prompts"].data
    point_layer = viewer.layers["point_prompts"]
    point_prompts = point_layer.data
    point_labels = point_layer.properties["label"]
    if len(point_prompts) > 0:
        point_labels = np.array([1 if label == "positive" else 0 for label in point_labels])
        assert len(point_prompts) == len(point_labels), \
            f"Number of point prompts and labels disagree: {len(point_prompts)} != {len(point_labels)}"

    # Commit the prompts for all the objects in the commit.
    if len(object_ids) == 1:  # We only have a single object.
        write_prompts(object_ids[0], prompts, point_prompts, point_labels)

    elif is_tracking:  # We have multiple objects from tracking a lineage with divisions.
        track_ids_points = np.array(point_layer.properties["track_id"])
        track_ids_prompts = np.array(viewer.layers["prompts"].properties["track_id"])

        unique_track_ids = np.unique(track_ids_points)
        assert len(unique_track_ids) == len(object_ids)
        track_state = np.array(point_layer.properties["state"])
        for track_id, object_id in zip(unique_track_ids, object_ids):
            this_prompts = None if len(prompts) == 0 else prompts[track_ids_prompts == track_id]
            point_mask = track_ids_points == track_id
            this_points, this_labels, this_track_state = \
                point_prompts[point_mask], point_labels[point_mask], track_state[point_mask]
            write_prompts(object_id, this_prompts, this_points, this_labels, track_state=this_track_state)

    else:  # We have multiple objects, which are the result from batched interactive segmentation.
        # Note: we can't match exact object ids to their prompts, for batched segmentation.
        # We first write the objects from box prompts, then from point prompts.
        n_prompts, n_points = len(prompts), len(point_prompts)
        assert n_prompts + n_points == len(object_ids), \
            f"Number of prompts and objects disagree: {n_prompts} + {n_points} != {len(object_ids)}"
        for i, object_id in enumerate(object_ids):
            if i < n_prompts:
                this_prompts, this_points, this_labels = prompts[i:i+1], None, None
            else:
                j = i - n_prompts
                this_prompts, this_points, this_labels = None, point_prompts[j:j+1], point_labels[j:j+1]
            write_prompts(object_id, this_prompts, this_points, this_labels)

    # Write the commit history.
    f.attrs["commit_history"] = commit_history


@magic_factory(
    call_button="Commit [C]",
    layer={"choices": ["current_object", "auto_segmentation"], "tooltip": get_tooltip("commit", "layer")},
    preserve_mode={"choices": ["objects", "pixels", "none"], "tooltip": get_tooltip("commit", "preserve_mode")},
    commit_path={"mode": "d", "tooltip": get_tooltip("commit", "commit_path")},
)
def commit(
    viewer: "napari.viewer.Viewer",
    layer: str = "current_object",
    preserve_mode: str = "objects",
    preservation_threshold: float = 0.75,
    commit_path: Optional[Path] = None,
) -> None:
    """Widget for committing the segmented objects from automatic or interactive segmentation.

    Args:
        viewer: The napari viewer.
        layer: Select the layer to commit. Can be either 'current_object' to commit interacitve segmentation results.
            Or 'auto_segmentation' to commit automatic segmentation results.
        preserve_mode: The mode for preserving already committed objects, in order to prevent over-writing
            them by a new commit. Supports the modes 'objects', which preserves on the object level and is the default,
            'pixels', which preserves on the pixel-level, or 'none', which does not preserve commited objects.
        preservation_threshold: The overlap threshold for preserving objects. This is only used if
            preservation_mode is set to 'objects'.
        commit_path: Select a file path where the committed results and prompts will be saved.
            This feature is still experimental.
    """
    # Commit the segmentation layer.
    _, seg, mask, bb = _commit_impl(viewer, layer, preserve_mode, preservation_threshold)

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

    # Perform garbage collection
    gc.collect()


@magic_factory(
    call_button="Commit [C]",
    layer={"choices": ["current_object", "auto_segmentation"]},
    preserve_mode={"choices": ["objects", "pixels", "none"]},
    commit_path={"mode": "d"},  # choose a directory
)
def commit_track(
    viewer: "napari.viewer.Viewer",
    layer: str = "current_object",
    preserve_mode: str = "objects",
    preservation_threshold: float = 0.75,
    commit_path: Optional[Path] = None,
) -> None:
    """Widget for committing the objects from interactive tracking.

    Args:
        viewer: The napari viewer.
        layer: Select the layer to commit. Can be either 'current_object' to commit interacitve segmentation results.
            Or 'auto_segmentation' to commit automatic segmentation results.
        preserve_mode: The mode for preserving already committed objects, in order to prevent over-writing
            them by a new commit. Supports the modes 'objects', which preserves on the object level and is the default,
            'pixels', which preserves on the pixel-level, or 'none', which does not preserve commited objects.
        preservation_threshold: The overlap threshold for preserving objects. This is only used if
            preservation_mode is set to 'objects'.
        commit_path: Select a file path where the committed results and prompts will be saved.
            This feature is still experimental.
    """
    # Commit the segmentation layer.
    id_offset, seg, mask, bb = _commit_impl(viewer, layer, preserve_mode, preservation_threshold)

    # Update the lineages.
    state = AnnotatorState()
    lineage = state.lineage

    if isinstance(lineage, list):  # This is a list of lineages from auto-tracking.
        assert id_offset == 0
        assert len(state.committed_lineages) == 0
        state.committed_lineages.extend(lineage)
    else:  # This is a single lineage from interactive tracking.
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

    # Create / update the tracking layer.
    layer_name = "tracks"
    segmentation = viewer.layers["committed_objects"].data
    track_data, parent_graph = get_napari_track_data(segmentation, state.committed_lineages)
    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]
        layer.data = track_data
        layer.graph = parent_graph
    else:
        viewer.add_tracks(track_data, name=layer_name, graph=parent_graph)

    # Reset the tracking state.
    _reset_tracking_state(viewer)

    # Perform garbage collection.
    gc.collect()


def create_prompt_menu(points_layer, labels, menu_name="prompt", label_name="label"):
    """Create the menu for toggling point prompt labels."""
    label_menu = ComboBox(label=menu_name, choices=labels, tooltip=get_tooltip("prompt_menu", "labels"))
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
def settings_widget(cache_directory: Optional[Path] = util.get_cache_directory()) -> None:
    """Widget to update global micro_sam settings.

    Args:
        cache_directory: Select the path for the micro_sam cache directory. `$HOME/.cache/micro_sam`.
    """
    os.environ["MICROSAM_CACHEDIR"] = str(cache_directory)
    print(f"micro-sam cache directory set to: {cache_directory}")


def _generate_message(message_type: str, message: str) -> bool:
    """
    Displays a message dialog based on the provided message type.

    Args:
        message_type: The type of message to display. Valid options are:
            - "error": Displays a critical error message with an "Ok" button.
            - "info": Displays an informational message in a separate dialog box.
                 The user can dismiss it by either clicking "Ok" or closing the dialog.
        message: The message content to be displayed in the dialog.

    Returns:
        A flag indicating whether the user aborted the operation based on the
        message type. This flag is only set for "info" messages where the user
        can choose to cancel (rejected).

    Raises:
        ValueError: If an invalid message type is provided.
    """
    # Set button text and behavior based on message type
    if message_type == "error":
        QtWidgets.QMessageBox.critical(None, "Error", message, QtWidgets.QMessageBox.Ok)
        abort = True
        return abort
    elif message_type == "info":
        info_dialog = InfoDialog(title="Validation Message", message=message)
        result = info_dialog.exec_()
        if result == QtWidgets.QDialog.Rejected:  # Check for cancel
            abort = True  # Set flag directly in calling function
            return abort
    else:
        raise ValueError(f"Invalid message type {message_type}")


def _validate_embeddings(viewer: "napari.viewer.Viewer"):
    state = AnnotatorState()
    if state.image_embeddings is None:
        msg = "Image embeddings are not yet computed. Press 'Compute Embeddings' to compute them for your image."
        return _generate_message("error", msg)
    else:
        return False

    # This code is for checking the data signature of the current image layer and the data signature
    # of the embeddings. However, the code has some disadvantages, for example assuming the position of the
    # image layer and also having to compute the data signature every time.
    # That's why we are not using this for now, but may want to revisit this in the future. See:
    # https://github.com/computational-cell-analytics/micro-sam/issues/504

    # embeddings_save_path = state.embedding_path
    # embedding_data_signature = None
    # image = None
    # if isinstance(viewer.layers[0], napari.layers.Image):  # Assuming the image layer is at index 0
    #     image = viewer.layers[0]
    # else:
    #     # Handle the case where the first layer isn't an Image layer
    #     raise ValueError("Expected an Image layer in viewer.layers")
    # img_signature = util._compute_data_signature(image.data)
    # if embeddings_save_path is not None:
    #     # Check for existing embeddings
    #     if os.listdir(embeddings_save_path):
    #         try:
    #             with zarr.open(embeddings_save_path, "a") as f:
    #                 # If data_signature exists, compare and return validation message
    #                 if "data_signature" in f.attrs:
    #                     embedding_data_signature = f.attrs["data_signature"]
    #         except RuntimeError as e:
    #             val_results = {
    #                 "message_type": "error",
    #                 "message": f"Failed to load image embeddings: {e}"
    #             }
    #     else:
    #         val_results = {"message_type": "info", "message": "No existing embeddings found at the specified path."}
    # else:  # load from state object
    #     embedding_data_signature = state.data_signature
    # # compare image data signature with embedding data signature
    # if img_signature != embedding_data_signature:
    #     val_results = {
    #         "message_type": "error",
    #         "message": f"The embeddings don't match with the image: {img_signature} {embedding_data_signature}"
    #     }
    # else:
    #     val_results = None
    # if val_results:
    #     return _generate_message(val_results["message_type"], val_results["message"])
    # else:
    #     return False


def _validation_window_for_missing_layer(layer_choice):
    if layer_choice == "committed_objects":
        msg = "The 'committed_objects' layer to commit masks is missing. Please try to commit again."
    else:
        msg = f"The '{layer_choice}' layer to commit is missing. Please re-annotate and try again."

    return _generate_message(message_type="error", message=msg)


def _validate_layers(viewer: "napari.viewer.Viewer", automatic_segmentation: bool = False) -> bool:
    # Check whether all layers exist as expected or create new ones automatically.
    state = AnnotatorState()
    state.annotator._require_layers()

    if not automatic_segmentation:
        # Check prompts layer.
        if len(viewer.layers["prompts"].data) == 0 and len(viewer.layers["point_prompts"].data) == 0:
            msg = "No prompts were given. Please provide prompts to run interactive segmentation."
            return _generate_message("error", msg)
        else:
            return False


@magic_factory(call_button="Segment Object [S]")
def segment(viewer: "napari.viewer.Viewer", batched: bool = False) -> None:
    """Segment object(s) for the current prompts.

    Args:
        viewer: The napari viewer.
        batched: Choose if you want to segment multiple objects with point prompts.
    """
    if _validate_embeddings(viewer):
        return None
    if _validate_layers(viewer):
        return None

    shape = viewer.layers["current_object"].data.shape

    # get the current box and point prompts
    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape)
    points, labels = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], with_stop_annotation=False)

    predictor = AnnotatorState().predictor
    image_embeddings = AnnotatorState().image_embeddings

    if isinstance(predictor, SamPredictor):  # This is SAM1 predictor.
        seg = vutil.prompt_segmentation(
            predictor, points, labels, boxes, masks, shape, image_embeddings=image_embeddings,
            multiple_box_prompts=True, batched=batched, previous_segmentation=viewer.layers["current_object"].data,
        )
    else:  # This would be SAM2 predictors.
        from micro_sam.v2.prompt_based_segmentation import promptable_segmentation_2d
        seg = promptable_segmentation_2d(
            predictor=predictor,
            points=points,
            labels=labels,
            boxes=boxes,
            masks=masks,
        )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    viewer.layers["current_object"].data = seg
    viewer.layers["current_object"].refresh()


@magic_factory(call_button="Segment Slice [S]")
def segment_slice(viewer: "napari.viewer.Viewer") -> None:
    """Segment object for to the current prompts.

    Args:
        viewer: The napari viewer.
    """
    if _validate_embeddings(viewer):
        return None
    if _validate_layers(viewer):
        return None

    shape = viewer.layers["current_object"].data.shape[1:]

    position_world = viewer.dims.point
    position = viewer.layers["point_prompts"].world_to_data(position_world)
    z = int(position[0])

    point_prompts = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], z)
    # this is a stop prompt, we do nothing
    if not point_prompts:
        return

    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape, i=z)
    points, labels = point_prompts

    state = AnnotatorState()

    # Check if using SAM1 or SAM2
    if isinstance(state.predictor, SamPredictor):
        # SAM1 path (existing code)
        seg = vutil.prompt_segmentation(
            state.predictor, points, labels, boxes, masks, shape, multiple_box_prompts=False,
            image_embeddings=state.image_embeddings, i=z,
        )
    else:
        # Use the segment_slice method for SAM2.
        boxes = [box[[1, 0, 3, 2]] for box in boxes]
        seg = state.interactive_segmenter.segment_slice(
            frame_idx=z,
            points=points[:, ::-1].copy(),
            labels=labels,
            boxes=boxes,
            masks=masks
        )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    viewer.layers["current_object"].data[z] = seg
    viewer.layers["current_object"].refresh()


@magic_factory(call_button="Segment Frame [S]")
def segment_frame(viewer: "napari.viewer.Viewer") -> None:
    """Segment object for the current prompts.

    Args:
        viewer: The napari viewer.
    """
    if _validate_embeddings(viewer):
        return None
    if _validate_layers(viewer):
        return None

    state = AnnotatorState()
    shape = state.image_shape[1:]
    position = viewer.dims.point
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
    if all(item in (0, None) for item in tile_shape):
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
    if all(item in (0, None) for item in halo):
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
        section1_layout.addLayout(self._create_model_section())  # Creates the model family widget section.
        self.layout().addLayout(section1_layout)

        # Section 2: Settings (collapsible).
        self.layout().addWidget(self._create_settings_widget())

        # Section 3: The button to trigger the embedding computation.
        self.run_button = QtWidgets.QPushButton("Compute Embeddings")
        self.run_button.clicked.connect(self._initialize_image)
        self.run_button.clicked.connect(self.__call__)
        self.run_button.setToolTip(get_tooltip("embedding", "run_button"))
        self.layout().addWidget(self.run_button)

    def _initialize_image(self):
        state = AnnotatorState()
        layer = self.image_selection.get_value()

        # This is encountered when there is no image layer available / selected.
        # In this case, we need not specify other image-level parameters to the state. Hence, we skip them.
        # NOTE: On code-level, this happens as the first step when "Compute Embedding" click is triggered.
        if layer is None:
            return

        image_shape = layer.data.shape
        image_scale = tuple(layer.scale)
        state.image_shape = image_shape
        state.image_scale = image_scale
        state.image_name = layer.name

    def _create_image_section(self):
        image_section = QtWidgets.QVBoxLayout()
        image_layer_widget = QtWidgets.QLabel("Image Layer:")
        # image_layer_widget.setToolTip(get_tooltip("embedding", "image")) #  this adds tooltip to label
        image_section.addWidget(image_layer_widget)

        # Setting a napari layer in QT, see:
        # https://github.com/pyapp-kit/magicgui/blob/main/docs/examples/napari/napari_combine_qt.py
        self.image_selection = create_widget(annotation=napari.layers.Image)
        self.image_selection.native.setToolTip(get_tooltip("embedding", "image"))
        image_section.addWidget(self.image_selection.native)

        return image_section

    def _update_model(self, state):
        _model_type = state.predictor.model_type if self.custom_weights else self.model_type

        # Provide a detailed message for the model family and model size per chosen combination.
        msg = "Computed embeddings for "
        if self.custom_weights:  # Whether the user provided a filepath to custom finetuned model weights.
            msg += f"the model located at '{os.path.abspath(self.custom_weights)}' "
            msg += f"of size '{self._model_size_map[_model_type[4]]}'."
        else:
            msg += f"the '{self.model_family}' model of size '{self.model_size}'."

        show_info(msg)

        state = AnnotatorState()
        # Update the widget itself. This is necessary because we may have loaded
        # some settings from the embedding file and have to reflect them in the widget.
        vutil._sync_embedding_widget(
            self,
            model_type=_model_type,
            save_path=self.embeddings_save_path,
            checkpoint_path=self.custom_weights,
            device=self.device,
            tile_shape=[self.tile_x, self.tile_y],
            halo=[self.halo_x, self.halo_y]
        )

        # Set the default settings for this model in the autosegment widget if it is part of
        # the currently used plugin.
        if "autosegment" in state.widgets:
            with_decoder = state.decoder is not None
            vutil._sync_autosegment_widget(
                state.widgets["autosegment"], _model_type, self.custom_weights, update_decoder=with_decoder
            )
            # Load the AMG/AIS state if we have a 3d segmentation plugin.
            if state.widgets["autosegment"].volumetric and with_decoder:
                state.amg_state = vutil._load_is_state(state.embedding_path)
            elif state.widgets["autosegment"].volumetric and not with_decoder:
                state.amg_state = vutil._load_amg_state(state.embedding_path)

        # Set the default settings for this model in the nd-segmentation widget if it is part of
        # the currently used plugin.
        if "segment_nd" in state.widgets:
            vutil._sync_ndsegment_widget(state.widgets["segment_nd"], _model_type, self.custom_weights)

    def _create_settings_widget(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # Add the model size widget section.
        layout = self._create_model_size_section()
        setting_values.layout().addLayout(layout)

        # Create UI for the device.
        self.device = "auto"
        device_options = ["auto"] + util._available_devices()

        self.device_dropdown, layout = self._add_choice_param(
            "device", self.device, device_options, tooltip=get_tooltip("embedding", "device")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the save path.
        self.embeddings_save_path = None
        self.embeddings_save_path_param, layout = self._add_path_param(
            "embeddings_save_path", self.embeddings_save_path, "directory", title="embeddings save path:",
            tooltip=get_tooltip("embedding", "embeddings_save_path")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the custom weights.
        self.custom_weights = None
        self.custom_weights_param, layout = self._add_path_param(
            "custom_weights", self.custom_weights, "file", title="custom weights path:",
            tooltip=get_tooltip("embedding", "custom_weights")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the tile shape.
        self.tile_x, self.tile_y = 0, 0
        self.tile_x_param, self.tile_y_param, layout = self._add_shape_param(
            ("tile_x", "tile_y"), (self.tile_x, self.tile_y), min_val=0, max_val=2048, step=16,
            tooltip=get_tooltip("embedding", "tiling")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the halo.
        self.halo_x, self.halo_y = 0, 0
        self.halo_x_param, self.halo_y_param, layout = self._add_shape_param(
            ("halo_x", "halo_y"), (self.halo_x, self.halo_y), min_val=0, max_val=512,
            tooltip=get_tooltip("embedding", "halo")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the choice of automatic segmentation mode.
        self.automatic_segmentation_mode = "auto"
        auto_seg_options = ["auto", "amg", "ais"]
        self.automatic_segmentation_mode_dropdown, layout = self._add_choice_param(
            "automatic_segmentation_mode", self.automatic_segmentation_mode, auto_seg_options,
            title="automatic segmentation mode", tooltip=get_tooltip("embedding", "automatic_segmentation_mode")
        )
        setting_values.layout().addLayout(layout)

        settings = _make_collapsible(setting_values, title="Embedding Settings")
        return settings

    def _validate_inputs(self):
        """Validates the inputs for the annotation process and returns a dictionary
        containing information for message generation, or False if no messages are needed.

        This function performs the following checks:

        - If an `embeddings_save_path` is provided:
            - Validates the image data signature by comparing it with the signature
            of the image data in the viewer's selection.
            - Checks for existing embeddings at the specified path.
                - If existing embeddings are found, it attempts to load parameters
                like tile shape, halo, and model type from the Zarr attributes.
                - An informational message is generated based on the loaded parameters.
                - If loading existing embeddings fails, an error message is generated.
                - If no existing embeddings are found, an informational message is generated.
        - If no `embeddings_save_path` is provided, the function returns None.

        Returns:
            bool: True if the computation should be aborted, otherwise False.
        """

        # Check if we have an existing input image to compute the embeddings.
        image = self.image_selection.get_value()
        if image is None:
            return _generate_message("error", "No image has been selected.")

        # Check if we have an existing embedding path.
        # If yes we check the data signature of these embeddings against the selected image
        # and we ask the user if they want to load these embeddings.
        if self.embeddings_save_path and os.listdir(self.embeddings_save_path):
            try:
                f = zarr.open(self.embeddings_save_path, mode="a")

                # Validate that the embeddings are complete.
                # Note: 'input_size' is the last value set in the attrs of f,
                # so we can use it as a proxy to check if the embeddings are fully computed
                if "input_size" not in f.attrs:
                    msg = (f"The embeddings at {self.embeddings_save_path} are incomplete. "
                           "Specify a different path or remove them.")
                    return _generate_message("error", msg)

                # Validate image data signature.
                if "data_signature" in f.attrs:
                    image = self.image_selection.get_value()
                    img_signature = util._compute_data_signature(image.data)
                    if img_signature != f.attrs["data_signature"]:
                        msg = f"The embeddings don't match with the image: {img_signature} {f.attrs['data_signature']}"
                        return _generate_message("error", msg)

                # Load existing parameters.
                self.model_type = f.attrs.get("model_name", f.attrs["model_type"])
                if "tile_shape" in f.attrs and f.attrs["tile_shape"] is not None:
                    self.tile_x, self.tile_y = f.attrs["tile_shape"]
                    self.halo_x, self.halo_y = f.attrs["halo"]
                    val_results = {
                        "message_type": "info",
                        "message": (f"Load embeddings for model: {self.model_type} with tile shape: "
                                    f"{self.tile_x}, {self.tile_y} and halo: {self.halo_x}, {self.halo_y}.")
                    }
                else:
                    self.tile_x, self.tile_y = 0, 0
                    self.halo_x, self.halo_y = 0, 0
                    val_results = {
                        "message_type": "info",
                        "message": f"Load embeddings for model: {self.model_type}."
                    }

                return _generate_message(val_results["message_type"], val_results["message"])

            except RuntimeError as e:
                val_results = {
                    "message_type": "error",
                    "message": f"Failed to load image embeddings: {e}"
                }
                return _generate_message(val_results["message_type"], val_results["message"])

        # Otherwise we either don't have an embedding path or it is empty. We can proceed in both cases.
        return False

    def _validate_existing_embeddings(self, state):
        if state.image_embeddings is None:
            return False
        else:
            val_results = {
                "message_type": "info",
                "message": "Embeddings have already been precomputed. Press OK to recompute the embeddings."
            }
            return _generate_message(val_results["message_type"], val_results["message"])

    def __call__(self, skip_validate=False):
        self._validate_model_type_and_custom_weights()

        # Validate user inputs.
        if not skip_validate and self._validate_inputs():
            return

        # Get the image.
        image = self.image_selection.get_value()

        # Update the image embeddings:
        state = AnnotatorState()
        if self._validate_existing_embeddings(state):
            # Whether embeddings already exist to control existing objects in layers.
            state.skip_recomputing_embeddings = True
            return

        state.skip_recomputing_embeddings = False
        # Reset the state.
        state.reset_state()

        # Get image dimensions.
        if image.rgb:
            ndim = image.data.ndim - 1
            state.image_shape = image.data.shape[:-1]
        else:
            ndim = image.data.ndim
            state.image_shape = image.data.shape

        # Set layer scale
        state.image_scale = tuple(image.scale)

        # Process tile_shape and halo, set other data.
        tile_shape, halo = _process_tiling_inputs(self.tile_x, self.tile_y, self.halo_x, self.halo_y)
        save_path = None if self.embeddings_save_path == "" else self.embeddings_save_path
        image_data = image.data

        # Set up progress bar and signals for using it within a threadworker.
        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker()
        def compute_image_embedding():

            def pbar_init(total, description):
                pbar_signals.pbar_total.emit(total)
                pbar_signals.pbar_description.emit(description)

            # Whether to prefer decoder.
            # With 'amg', it is set to 'False', else it is 'True' for the default 'auto' and 'ais' mode.
            prefer_decoder = True
            if self.automatic_segmentation_mode == "amg":
                prefer_decoder = False

            # Define a predictor for SAM2 models.
            predictor = None
            if self.model_type.startswith("h"):  # i.e. SAM2 models.
                from micro_sam.v2.util import get_sam2_model

                if ndim == 2:  # Get the SAM2 model and prepare the image predictor.
                    model = get_sam2_model(model_type=self.model_type, input_type="images")
                    # Prepare the SAM2 predictor.
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                    predictor = SAM2ImagePredictor(model)
                elif ndim == 3:  # Get SAM2 video predictor
                    predictor = get_sam2_model(model_type=self.model_type, input_type="videos")
                else:
                    raise ValueError

            state.initialize_predictor(
                image_data,
                model_type=self.model_type,
                save_path=save_path,
                ndim=ndim,
                device=self.device,
                checkpoint_path=self.custom_weights,
                predictor=predictor,
                tile_shape=tile_shape,
                halo=halo,
                prefer_decoder=prefer_decoder,
                pbar_init=pbar_init,
                pbar_update=lambda update: pbar_signals.pbar_update.emit(update),
                is_sam2=self.model_type.startswith("h"),
            )
            pbar_signals.pbar_stop.emit()

        compute_image_embedding()
        self._update_model(state)
        # worker = compute_image_embedding()
        # worker.returned.connect(self._update_model)
        # worker.start()
        # return worker


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
        self.run_button.clicked.connect(self.__call__)
        self.layout().addWidget(self.run_button)

    def _create_settings(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setToolTip(get_tooltip("segmentnd", "settings"))
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # Create the UI for the projection modes.
        self.projection = "single_point"
        self.projection_dropdown, layout = self._add_choice_param(
            "projection", self.projection, PROJECTION_MODES, tooltip=get_tooltip("segmentnd", "projection_dropdown")
            )
        setting_values.layout().addLayout(layout)

        # Create the UI element for the IOU threshold.
        self.iou_threshold = 0.5
        self.iou_threshold_param, layout = self._add_float_param(
            "iou_threshold", self.iou_threshold, tooltip=get_tooltip("segmentnd", "iou_threshold")
            )
        setting_values.layout().addLayout(layout)

        # Create the UI element for the box extension.
        self.box_extension = 0.05
        self.box_extension_param, layout = self._add_float_param(
            "box_extension", self.box_extension, tooltip=get_tooltip("segmentnd", "box_extension")
            )
        setting_values.layout().addLayout(layout)

        # Create the UI element in form of a checkbox for multi-object segmentation.
        self.batched = False
        setting_values.layout().addWidget(
            self._add_boolean_param(
                "batched", self.batched, title="batched", tooltip=get_tooltip("segmentnd", "batched")
            )
        )

        # Create the UI element for the motion smoothing (if we have the tracking widget).
        if self.tracking:
            self.motion_smoothing = 0.5
            self.motion_smoothing_param, layout = self._add_float_param(
                "motion_smoothing", self.motion_smoothing, tooltip=get_tooltip("segmentnd", "motion_smoothing")
                )
            setting_values.layout().addLayout(layout)

        settings = _make_collapsible(setting_values, title="Segmentation Settings")
        return settings

    def _run_tracking(self):
        state = AnnotatorState()
        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker
        def tracking_impl():
            shape = state.image_shape

            pbar_signals.pbar_total.emit(shape[0])
            pbar_signals.pbar_description.emit("Track object")

            # Step 1: Segment all slices with prompts.
            seg, slices, _, stop_upper = vutil.segment_slices_with_prompts(
                state.predictor, self._viewer.layers["point_prompts"], self._viewer.layers["prompts"],
                state.image_embeddings, shape, track_id=state.current_track_id,
                update_progress=lambda update: pbar_signals.pbar_update.emit(update),
            )

            # Step 2: Track the object starting from the lowest annotated slice.
            seg, has_division = vutil.track_from_prompts(
                self._viewer.layers["point_prompts"], self._viewer.layers["prompts"], seg,
                state.predictor, slices, state.image_embeddings, stop_upper,
                threshold=self.iou_threshold, projection=self.projection,
                motion_smoothing=self.motion_smoothing,
                box_extension=self.box_extension,
                update_progress=lambda update: pbar_signals.pbar_update.emit(update),
            )

            pbar_signals.pbar_stop.emit()
            return seg, has_division

        def update_segmentation(ret_val):
            seg, has_division = ret_val
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

        ret_val = tracking_impl()
        update_segmentation(ret_val)
        # worker = tracking_impl()
        # worker.returned.connect(update_segmentation)
        # worker.start()
        # return worker

    def _run_volumetric_segmentation(self):
        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker
        def volumetric_segmentation_impl():
            state = AnnotatorState()
            shape = state.image_shape

            pbar_signals.pbar_total.emit(shape[0])
            pbar_signals.pbar_description.emit("Segment object")

            if isinstance(state.predictor, SamPredictor):  # This is SAM2 predictor.
                # Step 1: Segment all slices with prompts.
                seg, slices, stop_lower, stop_upper = vutil.segment_slices_with_prompts(
                    state.predictor, self._viewer.layers["point_prompts"], self._viewer.layers["prompts"],
                    state.image_embeddings, shape,
                    update_progress=lambda update: pbar_signals.pbar_update.emit(update),
                )

                # Step 2: Segment the rest of the volume based on projecting prompts.
                seg, (z_min, z_max) = segment_mask_in_volume(
                    seg, state.predictor, state.image_embeddings, slices,
                    stop_lower, stop_upper,
                    iou_threshold=self.iou_threshold, projection=self.projection,
                    box_extension=self.box_extension,
                    update_progress=lambda update: pbar_signals.pbar_update.emit(update),
                )

                state.z_range = (z_min, z_max)

            else:  # This would be SAM2 predictors.
                # Prepare the prompts
                point_prompts = self._viewer.layers["point_prompts"]
                box_prompts = self._viewer.layers["prompts"]
                z_values_points = np.round(point_prompts.data[:, 0])
                z_values_boxes = np.concatenate(
                    [box[:1, 0] for box in box_prompts.data]
                ) if box_prompts.data else np.zeros(0, dtype="int")

                # Whether the user decide to provide batched prompts for multi-object segmentation.
                is_batched = bool(self.batched)

                # Let's do points first.
                for curr_z_values_point in z_values_points:
                    # Extract the point prompts from the points layer first.
                    points, labels = vutil.point_layer_to_prompts(layer=point_prompts, i=curr_z_values_point)

                    # Add prompts one after the other.
                    [
                        state.interactive_segmenter.add_point_prompts(
                            frame_ids=curr_z_values_point,
                            points=np.array([curr_point]),
                            point_labels=np.array([curr_label]),
                            object_id=i if is_batched else None,
                        ) for i, (curr_point, curr_label) in enumerate(zip(points, labels), start=1)
                    ]

                # Next, we add box prompts.
                for curr_z_values_box in z_values_boxes:
                    # Extract the box prompts from the shapes layer first.
                    boxes, _ = vutil.shape_layer_to_prompts(
                        layer=box_prompts, shape=state.image_shape, i=curr_z_values_box,
                    )

                    # Add prompts one after the other.
                    state.interactive_segmenter.add_box_prompts(frame_ids=curr_z_values_box, boxes=boxes)

                # Propagate the prompts throughout the volume and combine the propagated segmentations.
                seg = state.interactive_segmenter.predict()

            pbar_signals.pbar_stop.emit()

            return seg

        def update_segmentation(seg):
            self._viewer.layers["current_object"].data = seg
            self._viewer.layers["current_object"].refresh()

        seg = volumetric_segmentation_impl()
        self._viewer.layers["current_object"].data = seg
        self._viewer.layers["current_object"].refresh()
        # worker = volumetric_segmentation_impl()
        # worker.returned.connect(update_segmentation)
        # worker.start()
        # return worker

    def __call__(self):
        if _validate_embeddings(self._viewer):
            return None
        if _validate_layers(self._viewer):
            return None

        if self.tracking:
            return self._run_tracking()
        else:
            return self._run_volumetric_segmentation()


#
# The functionality and widgets for automatic segmentation.
#


# Messy amg state handling, would be good to refactor this properly at some point.
def _handle_amg_state(state, i, pbar_init, pbar_update):
    if state.amg is None:
        is_tiled = state.image_embeddings["input_size"] is None
        state.amg = instance_segmentation.get_instance_segmentation_generator(
            state.predictor, is_tiled=is_tiled, decoder=state.decoder
        )

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
            state.amg.initialize(
                dummy_image, image_embeddings=state.image_embeddings, i=i,
                verbose=pbar_init is not None, pbar_init=pbar_init, pbar_update=pbar_update,
            )
            amg_state_i = state.amg.get_state()
            state.amg_state[i] = amg_state_i

            cache_folder = state.amg_state.get("cache_folder", None)
            if cache_folder is not None:
                cache_path = os.path.join(cache_folder, f"state-{i}.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(amg_state_i, f)

            cache_path = state.amg_state.get("cache_path", None)
            if cache_path is not None:
                save_key = f"state-{i}"
                with h5py.File(cache_path, "a") as f:
                    g = f.create_group(save_key)
                    g.create_dataset("foreground", data=amg_state_i["foreground"], compression="gzip")
                    g.create_dataset("boundary_distances", data=amg_state_i["boundary_distances"], compression="gzip")
                    g.create_dataset("center_distances", data=amg_state_i["center_distances"], compression="gzip")

    # Otherwise (2d segmentation) we just check if the amg is initialized or not.
    elif not state.amg.is_initialized:
        assert i is None
        # We don't need to pass the actual image data here, since the embeddings are passed.
        # (The image data is only used by the amg to compute image embeddings, so not needed here.)
        dummy_image = np.zeros(shape, dtype="uint8")
        state.amg.initialize(
            dummy_image, image_embeddings=state.image_embeddings,
            verbose=pbar_init is not None, pbar_init=pbar_init, pbar_update=pbar_update
        )


def _instance_segmentation_impl(min_object_size, i=None, pbar_init=None, pbar_update=None, **kwargs):
    state = AnnotatorState()
    _handle_amg_state(state, i, pbar_init, pbar_update)
    seg = state.amg.generate(**kwargs)
    assert isinstance(seg, np.ndarray)
    return seg


class AutoSegmentWidget(_WidgetBase):
    def __init__(self, viewer, with_decoder, volumetric, parent=None):
        super().__init__(parent)

        self._viewer = viewer
        self.with_decoder = with_decoder
        self.volumetric = volumetric
        self._create_widget()

    def _create_widget(self):
        # Add the switch for segmenting the slice vs. the volume if we have a volume.
        if self.volumetric:
            self.layout().addWidget(self._create_volumetric_switch())

        # Add the nested settings widget.
        self.settings = self._create_settings()
        self.layout().addWidget(self.settings)

        # Add the run button.
        self.run_button = QtWidgets.QPushButton("Automatic Segmentation")
        self.run_button.clicked.connect(self.__call__)
        self.run_button.setToolTip(get_tooltip("autosegment", "run_button"))
        self.layout().addWidget(self.run_button)

    def _reset_segmentation_mode(self, with_decoder):
        # If we already have the same segmentation mode we don't need to do anything.
        if with_decoder == self.with_decoder:
            return

        # Otherwise we change the value of with_decoder.
        self.with_decoder = with_decoder

        # Then we clear the whole widget.
        layout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # And then we reset it.
        self._create_widget()

    def _create_volumetric_switch(self):
        self.apply_to_volume = False
        return self._add_boolean_param(
            "apply_to_volume", self.apply_to_volume, title="Apply to Volume",
            tooltip=get_tooltip("autosegment", "apply_to_volume")
        )

    def _add_common_settings(self, settings):
        # Create the UI element for min object size.
        self.min_object_size = 100
        self.min_object_size_param, layout = self._add_int_param(
            "min_object_size", self.min_object_size, min_val=0, max_val=int(1e4),
            tooltip=get_tooltip("autosegment", "min_object_size")
        )
        settings.layout().addLayout(layout)

        # Add extra settings for volumetric segmentation: gap_closing and min_extent.
        if self.volumetric:
            self.gap_closing = 2
            self.gap_closing_param, layout = self._add_int_param(
                "gap_closing", self.gap_closing, min_val=0, max_val=10,
                tooltip=get_tooltip("autosegment", "gap_closing")
                )
            settings.layout().addLayout(layout)

            self.min_extent = 2
            self.min_extent_param, layout = self._add_int_param(
                "min_extent", self.min_extent, min_val=0, max_val=10,
                tooltip=get_tooltip("autosegment", "min_extent")
                )
            settings.layout().addLayout(layout)

    def _ais_settings(self):
        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())

        # Create the UI element for center_distance_threshold.
        self.center_distance_thresh = 0.5
        self.center_distance_thresh_param, layout = self._add_float_param(
            "center_distance_thresh", self.center_distance_thresh,
            tooltip=get_tooltip("autosegment", "center_distance_thresh")
        )
        settings.layout().addLayout(layout)

        # Create the UI element for boundary_distance_threshold.
        self.boundary_distance_thresh = 0.5
        self.boundary_distance_thresh_param, layout = self._add_float_param(
            "boundary_distance_thresh", self.boundary_distance_thresh,
            tooltip=get_tooltip("autosegment", "boundary_distance_thresh")
        )
        settings.layout().addLayout(layout)

        # Add min_object_size.
        self._add_common_settings(settings)

        return settings

    def _amg_settings(self):
        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())

        # Create the UI element for pred_iou_thresh.
        self.pred_iou_thresh = 0.88
        self.pred_iou_thresh_param, layout = self._add_float_param(
            "pred_iou_thresh", self.pred_iou_thresh,
            tooltip=get_tooltip("autosegment", "pred_iou_thresh")
            )
        settings.layout().addLayout(layout)

        # Create the UI element for stability score thresh.
        self.stability_score_thresh = 0.95
        self.stability_score_thresh_param, layout = self._add_float_param(
            "stability_score_thresh", self.stability_score_thresh,
            tooltip=get_tooltip("autosegment", "stability_score_thresh")
        )
        settings.layout().addLayout(layout)

        # Create the UI element for box nms thresh.
        self.box_nms_thresh = 0.7
        self.box_nms_thresh_param, layout = self._add_float_param(
            "box_nms_thresh", self.box_nms_thresh,
            tooltip=get_tooltip("autosegment", "box_nms_thresh")
            )
        settings.layout().addLayout(layout)

        # Add min_object_size.
        self._add_common_settings(settings)

        return settings

    def _create_settings(self):
        setting_values = self._ais_settings() if self.with_decoder else self._amg_settings()
        settings = _make_collapsible(setting_values, title="Automatic Segmentation Settings")
        return settings

    def _empty_segmentation_warning(self):
        msg = "The automatic segmentation result does not contain any objects."
        msg += "Setting a smaller value for 'min_object_size' may help."
        if not self.with_decoder:
            msg += "Setting smaller values for 'pred_iou_thresh' and 'stability_score_thresh' may also help."
        val_results = {"message_type": "error", "message": msg}
        return _generate_message(val_results["message_type"], val_results["message"])

    def _run_segmentation_2d(self, kwargs, i=None):
        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker
        def seg_impl():
            def pbar_init(total, description):
                pbar_signals.pbar_total.emit(total)
                pbar_signals.pbar_description.emit(description)

            seg = _instance_segmentation_impl(
                self.min_object_size, i=i, pbar_init=pbar_init,
                pbar_update=lambda update: pbar_signals.pbar_update.emit(update),
                **kwargs
            )
            pbar_signals.pbar_stop.emit()
            return seg

        def update_segmentation(seg):
            is_empty = seg.max() == 0
            if is_empty:
                self._empty_segmentation_warning()

            if i is None:
                self._viewer.layers["auto_segmentation"].data = seg
            else:
                self._viewer.layers["auto_segmentation"].data[i] = seg
            self._viewer.layers["auto_segmentation"].refresh()

        # Validate all layers.
        _validate_layers(self._viewer, automatic_segmentation=True)

        seg = seg_impl()
        update_segmentation(seg)
        # worker = seg_impl()
        # worker.returned.connect(update_segmentation)
        # worker.start()
        # return worker

    # We refuse to run 3D segmentation with the AMG unless we have a GPU or all embeddings
    # are precomputed. Otherwise this would take too long.
    def _allow_segment_3d(self):
        if self.with_decoder:
            return True
        state = AnnotatorState()
        predictor = state.predictor
        if str(predictor.device) == "cpu" or str(predictor.device) == "mps":
            n_slices = self._viewer.layers["auto_segmentation"].data.shape[0]
            embeddings_are_precomputed = (state.amg_state is not None) and (len(state.amg_state) > n_slices)
            if not embeddings_are_precomputed:
                return False
        return True

    def _run_segmentation_3d(self, kwargs):
        allow_segment_3d = self._allow_segment_3d()
        if not allow_segment_3d:
            val_results = {
                "message_type": "error",
                "message": "Volumetric segmentation with AMG is only supported if you have a GPU."
            }
            return _generate_message(val_results["message_type"], val_results["message"])

        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker
        def seg_impl():
            segmentation = np.zeros_like(self._viewer.layers["auto_segmentation"].data)
            offset = 0

            def pbar_init(total, description):
                pbar_signals.pbar_total.emit(total)
                pbar_signals.pbar_description.emit(description)

            pbar_init(segmentation.shape[0], "Segment volume")

            # Further optimization: parallelize if state is precomputed for all slices
            for i in range(segmentation.shape[0]):
                seg = _instance_segmentation_impl(self.min_object_size, i=i, **kwargs)
                seg_max = seg.max()
                if seg_max == 0:
                    continue
                seg[seg != 0] += offset
                offset = seg_max + offset
                segmentation[i] = seg
                pbar_signals.pbar_update.emit(1)

            pbar_signals.pbar_reset.emit()
            segmentation = merge_instance_segmentation_3d(
                segmentation, beta=0.5,  gap_closing=self.gap_closing, min_z_extent=self.min_extent,
                verbose=True, pbar_init=pbar_init, pbar_update=lambda update: pbar_signals.pbar_update.emit(1),
            )
            pbar_signals.pbar_stop.emit()
            return segmentation

        def update_segmentation(segmentation):
            is_empty = segmentation.max() == 0
            if is_empty:
                self._empty_segmentation_warning()
            self._viewer.layers["auto_segmentation"].data = segmentation
            self._viewer.layers["auto_segmentation"].refresh()

        seg = seg_impl()
        update_segmentation(seg)
        # worker = seg_impl()
        # worker.returned.connect(update_segmentation)
        # worker.start()
        # return worker

    def __call__(self):
        if _validate_embeddings(self._viewer):
            return None

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
        if self.volumetric and self.apply_to_volume:
            worker = self._run_segmentation_3d(kwargs)
        elif self.volumetric and not self.apply_to_volume:
            i = int(self._viewer.dims.point[0])
            worker = self._run_segmentation_2d(kwargs, i=i)
        else:
            worker = self._run_segmentation_2d(kwargs)
        _select_layer(self._viewer, "auto_segmentation")
        return worker


class AutoTrackWidget(AutoSegmentWidget):
    def _create_tracking_switch(self):
        self.apply_to_volume = False
        return self._add_boolean_param(
            "apply_to_volume", self.apply_to_volume, title="Track Timeseries",
            tooltip=get_tooltip("autotrack", "run_tracking")
        )

    def _create_widget(self):
        # Add the switch for segmenting the slice vs. tracking the timeseries.
        self.layout().addWidget(self._create_tracking_switch())

        # Add the nested settings widget.
        self.settings = self._create_settings()
        self.layout().addWidget(self.settings)

        # Add the run button.
        self.run_button = QtWidgets.QPushButton("Automatic Tracking")
        self.run_button.clicked.connect(self.__call__)
        self.run_button.setToolTip(get_tooltip("autotrack", "run_button"))
        self.layout().addWidget(self.run_button)

    def _run_segmentation_3d(self, kwargs):
        allow_segment_3d = self._allow_segment_3d()
        if not allow_segment_3d:
            return _generate_message("error", "Tracking with AMG is only supported if you have a GPU.")

        state = AnnotatorState()
        if len(state.committed_lineages) > 0:
            return _generate_message(
                "error",
                "Automatic tracking can only be called if you haven't commited results from interactive tracking yet."
            )
        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker
        def seg_impl():
            image_name = state.get_image_name(self._viewer)
            timeseries = self._viewer.layers[image_name].data
            segmentation = np.zeros_like(self._viewer.layers["auto_segmentation"].data)
            offset = 0

            def pbar_init(total, description):
                pbar_signals.pbar_total.emit(total)
                pbar_signals.pbar_description.emit(description)

            pbar_init(segmentation.shape[0], "Run tracking")

            # Further optimization: parallelize if state is precomputed for all slices
            for i in range(segmentation.shape[0]):
                seg = _instance_segmentation_impl(self.min_object_size, i=i, **kwargs)
                seg_max = seg.max()
                if seg_max == 0:
                    continue
                seg[seg != 0] += offset
                offset = seg_max + offset
                segmentation[i] = seg
                pbar_signals.pbar_update.emit(1)

            pbar_signals.pbar_reset.emit()
            segmentation, lineages = track_across_frames(
                timeseries, segmentation,
                verbose=True, pbar_init=pbar_init,
                pbar_update=lambda update: pbar_signals.pbar_update.emit(1),
            )
            pbar_signals.pbar_stop.emit()
            return (segmentation, lineages)

        def update_segmentation(result):
            segmentation, lineages = result
            is_empty = segmentation.max() == 0
            if is_empty:
                self._empty_segmentation_warning()

            state = AnnotatorState()
            state.lineage = lineages

            self._viewer.layers["auto_segmentation"].data = segmentation
            self._viewer.layers["auto_segmentation"].refresh()

        result = seg_impl()
        update_segmentation(result)
        # worker = seg_impl()
        # worker.returned.connect(update_segmentation)
        # worker.start()
        # return worker
