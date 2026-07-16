"""Implements the widgets used in the annotation plugins."""

import gc
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
import z5py

from bioimage_cpp.utils import segmentation_overlap
from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container, create_widget
# We have disabled the thread workers for now because they result in a
# massive slowdown in napari >= 0.5.
# See also https://forum.image.sc/t/napari-thread-worker-leads-to-massive-slowdown/103786
# from napari.qt.threading import thread_worker
from napari.utils import progress
from napari.utils.notifications import show_info
from qtpy import QtWidgets
from qtpy.QtCore import QObject, Signal, Qt
from superqt import QCollapsible, QLabeledRangeSlider

from .. import util
from ..v1 import instance_segmentation
from ..v1.multi_dimensional_segmentation import (
    PROJECTION_MODES,
    export_tracking_result_to_ctc,
    export_tracking_result_to_geff,
    export_tracking_result_to_trackmate_xml,
    get_napari_track_data,
    merge_instance_segmentation_3d,
    segment_mask_in_volume,
    track_across_frames,
)
from . import util as vutil
from ._state import AnnotatorState
from ._tooltips import get_tooltip

#
# Convenience functionality for creating QT UI and manipulating the napari viewer.
#


def _select_layer(viewer, layer_name):
    viewer.layers.selection.select_only(viewer.layers[layer_name])


# Create a collapsible around the widget
def _make_collapsible(widget, title, tooltip=None):
    parent_widget = QtWidgets.QWidget()
    parent_widget.setLayout(QtWidgets.QVBoxLayout())
    collapsible = QCollapsible(title, parent_widget)
    if tooltip:
        collapsible.setToolTip(tooltip)
        # Also set it on the header toggle button, since that is what the user hovers (Qt does not
        # fall back to the parent's tooltip for child widgets).
        toggle_btn = getattr(collapsible, "_toggle_btn", None)
        if toggle_btn is not None:
            toggle_btn.setToolTip(tooltip)
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

    def _update_batched_visibility(self):
        """Hide the 'Batched' checkbox while embeddings are tiled (batched prompting is unsupported
        with tiling). No-op for widgets without a batched checkbox."""
        checkbox = getattr(self, "batched_checkbox", None)
        if checkbox is None:
            return
        is_tiled = _embeddings_are_tiled(AnnotatorState())
        if is_tiled and getattr(self, "batched", False):
            checkbox.setChecked(False)  # reset to single-object (also updates 'self.batched')
        checkbox.setVisible(not is_tiled)

    def _add_string_param(
        self,
        name,
        value,
        title=None,
        placeholder=None,
        layout=None,
        tooltip=None,
    ):
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

    def _add_float_param(
        self,
        name,
        value,
        title=None,
        min_val=0.0,
        max_val=1.0,
        decimals=2,
        step=0.01,
        layout=None,
        tooltip=None,
    ):
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

    def _add_int_param(
        self,
        name,
        value,
        min_val,
        max_val,
        title=None,
        step=1,
        layout=None,
        tooltip=None,
    ):
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

    def _make_int_field(self, name, value, min_val, max_val, step=1, title=None, tooltip=None):
        # A single labeled int spinbox wrapped in its own widget, so it can be placed inside a row
        # next to other fields and shown / hidden independently (e.g. the z fields, which only apply
        # to 3d data). Returns the spinbox and the wrapping widget.
        field = QtWidgets.QWidget()
        field_layout = QtWidgets.QVBoxLayout()
        field_layout.setContentsMargins(0, 0, 0, 0)
        param, _ = self._add_int_param(
            name, value, min_val=min_val, max_val=max_val, step=step,
            layout=field_layout, title=title, tooltip=tooltip,
        )
        field.setLayout(field_layout)
        return param, field

    def _add_choice_param(
        self,
        name,
        value,
        options,
        title=None,
        layout=None,
        update=None,
        tooltip=None,
    ):
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
            dropdown.currentIndexChanged.connect(
                lambda index: setattr(self, name, options[index])
            )
        else:
            dropdown.currentIndexChanged.connect(update)

        # Set the correct value for the value.
        dropdown.setCurrentIndex(dropdown.findText(value))

        if tooltip:
            dropdown.setToolTip(tooltip)

        layout.addWidget(dropdown)
        return dropdown, layout

    def _add_shape_param(
        self, names, values, min_val, max_val, step=1, title=None, tooltip=None
    ):
        layout = QtWidgets.QHBoxLayout()

        x_layout = QtWidgets.QVBoxLayout()
        x_param, _ = self._add_int_param(
            names[0],
            values[0],
            min_val=min_val,
            max_val=max_val,
            layout=x_layout,
            step=step,
            title=title[0] if title is not None else title,
            tooltip=tooltip,
        )
        layout.addLayout(x_layout)

        y_layout = QtWidgets.QVBoxLayout()
        y_param, _ = self._add_int_param(
            names[1],
            values[1],
            min_val=min_val,
            max_val=max_val,
            layout=y_layout,
            step=step,
            title=title[1] if title is not None else title,
            tooltip=tooltip,
        )
        layout.addLayout(y_layout)

        return x_param, y_param, layout

    def _add_path_param(
        self,
        name,
        value,
        select_type,
        title=None,
        placeholder=None,
        tooltip=None,
    ):
        assert select_type in ("directory", "file", "both")

        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)

        path_textbox = QtWidgets.QLineEdit()
        path_textbox.setText("" if value is None else str(value))
        if placeholder is not None:
            path_textbox.setPlaceholderText(placeholder)

        # An empty path means that no optional path was selected. Keep this as ``None`` in the
        # widget state instead of an empty (or whitespace-only) string: downstream model loading
        # distinguishes ``None`` (use the registered model) from a custom checkpoint path.
        path_textbox.textChanged.connect(
            lambda val: setattr(self, name, val if val.strip() else None)
        )
        if tooltip:
            path_textbox.setToolTip(tooltip)

        layout.addWidget(path_textbox)

        def add_path_button(select_type, tooltip=None):
            # Adjust button text.
            button_text = f"Select {select_type.capitalize()}"
            path_button = QtWidgets.QPushButton(button_text)

            # Call appropriate function based on select_type.
            path_button.clicked.connect(
                lambda: getattr(self, f"_get_{select_type}_path")(
                    name, path_textbox
                )
            )
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

    def _align_widths(self, widgets):
        # Give a set of widgets a uniform (max) fixed width so rows line up symmetrically.
        widgets = [w for w in widgets if w is not None]
        if not widgets:
            return
        width = max(w.sizeHint().width() for w in widgets)
        for w in widgets:
            w.setFixedWidth(width)

    def _get_model_size_options(self):
        # The available model sizes depend on the selected family: the base SAM2 family supports all
        # sizes, while finetuned families may only be available for specific sizes (e.g. 'Microscopy'
        # is an 'hvit_t' model). We store the UI labels mapped to the corresponding model names.
        sizes = self.model_family_config[self.model_family]["sizes"]
        self.model_size_options = [self._model_size_map[k] for k in sizes]
        self.model_size_mapping = {self._model_size_map[k]: f"hvit_{k}" for k in sizes}

        # We ensure an assorted order of model sizes ('tiny' to 'large').
        self.model_size_options.sort(
            key=lambda x: ["tiny", "small", "base", "large"].index(x)
        )

    def _update_model_type(self):
        # Sync the selected family; both the available sizes and the model-type suffix depend on it.
        self.model_family = self.model_family_dropdown.currentText() or self.model_family

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
            if (
                self.model_size_options
            ):  # Default to the first available model size
                self.model_size = self.model_size_options[0]

        # Let's map the selection to the correct model type (eg. "tiny" -> "hvit_t").
        size_key = next(
            (
                k
                for k, v in self._model_size_map.items()
                if v == self.model_size
            ),
            "t",
        )
        # Append the family suffix (e.g. 'tiny' + 'Microscopy' -> 'hvit_t_cells'; base -> 'hvit_t').
        suffix = self.model_family_config[self.model_family]["suffix"]
        self.model_type = f"hvit_{size_key}{suffix}"

        self.model_size_dropdown.setCurrentText(
            self.model_size
        )  # Apply the selected text to the dropdown

        # We force a refresh for UI here.
        self.model_size_dropdown.update()

        # NOTE: And finally, we should re-enable signals again.
        self.model_size_dropdown.blockSignals(False)

    def _create_model_section(
        self,
        default_model: Optional[str] = None,
        create_layout: bool = True,
    ):
        # The widget encodes its default as the synthetic 'vit_<size><suffix>' selector string. For
        # SAM2 model ids ('hvit_...') this is just the id without the leading 'h', so we derive it
        # from the single-source 'DEFAULT_MODEL' (e.g. 'hvit_t_cells' -> 'vit_t_cells' -> Microscopy/tiny).
        if default_model is None:
            from ..v2.util import DEFAULT_MODEL
            default_model = DEFAULT_MODEL[1:]

        # Create a list of supported dropdown values and correspond them to suffixes (used to parse
        # the synthetic default-model string). Additional SAM2 families can be added here in future.
        self.supported_dropdown_maps = {
            "Natural Images": "_sam2",
            "Microscopy": "_cells",
        }

        # Per-family backend config: the model-type suffix appended after 'hvit_{size}' and the
        # available model sizes. The base SAM2 family supports all sizes; finetuned families (e.g.
        # 'Microscopy', the joint SAM2 + UniSAM2 'hvit_t_cells' model) may exist only for some sizes.
        self.model_family_config = {
            "Natural Images": {"suffix": "", "sizes": ["t", "s", "b", "l"]},
            "Microscopy": {"suffix": "_cells", "sizes": ["t"]},
        }

        # NOTE: The available SAM2 model sizes are 'tiny', 'small', 'base' and 'large'.
        self._model_size_map = {
            "t": "tiny",
            "s": "small",
            "b": "base",
            "l": "large",
        }

        self._default_model_choice = default_model
        # Let's set the literally default model choice depending on 'micro-sam'.
        self.model_family = {
            v: k for k, v in self.supported_dropdown_maps.items()
        }[self._default_model_choice[5:]]

        kwargs = {}
        if create_layout:
            layout = QtWidgets.QVBoxLayout()
            kwargs["layout"] = layout

        # NOTE: We stick to the base variant for each model family.
        # i.e. 'Natural Images (SAM)', 'Light Microscopy', 'Electron Microscopy', 'Medical_Imaging', 'Histopathology'.
        self.model_family_dropdown, layout = self._add_choice_param(
            "model_family",
            self.model_family,
            list(self.supported_dropdown_maps.keys()),
            title="Model:",
            tooltip=get_tooltip("embedding", "model_family"),
            **kwargs,
        )
        self.model_family_dropdown.currentTextChanged.connect(
            self._update_model_type
        )
        return layout

    def _create_model_size_section(self):

        # Create UI for the model size.
        # This combines with the chosen 'self.model_family' and depends on 'self._default_model_choice'.
        self.model_size = self._model_size_map[self._default_model_choice[4]]

        # Now, we get the available sizes per model family.
        self._get_model_size_options()

        self.model_size_dropdown, layout = self._add_choice_param(
            "model_size",
            self.model_size,
            self.model_size_options,
            title="model size:",
            tooltip=get_tooltip("embedding", "model_size"),
        )
        self.model_size_dropdown.currentTextChanged.connect(
            self._update_model_type
        )
        return layout

    def _validate_model_type_and_custom_weights(self):
        # Map the selected family + size to the SAM2 `model_type`, appending the family suffix
        # (e.g. 'tiny' + 'Microscopy' -> 'hvit_t_cells'; 'tiny' + base family -> 'hvit_t').
        suffix = self.model_family_config.get(self.model_family, {}).get("suffix", "")
        self.model_type = f"hvit_{self.model_size[0]}{suffix}"

        # For 'custom_weights', we remove the displayed text on top of the drop-down menu.
        if self.custom_weights:
            # NOTE: We prevent recursive updates for this step temporarily.
            self.model_family_dropdown.blockSignals(True)
            self.model_family_dropdown.setCurrentIndex(
                -1
            )  # This removes the displayed text.
            self.model_family_dropdown.update()
            # NOTE: And re-enable signals again.
            self.model_family_dropdown.blockSignals(False)

    def _validate_model_support(self):
        if getattr(self, "sam2_only", False) and not self.model_type.startswith("hvit_"):
            return _generate_message(
                "error",
                "The tracking annotator only supports micro-sam2/SAM2 models. "
                f"Got unsupported model '{self.model_type}'.",
            )
        return False

    def _validate_vfm_requirements(self):
        # For gated VFM models (DINOv3 via 'transformers', UNI / UNI2-h via 'timm') check that the backend
        # package is importable and HuggingFace access is set up, surfacing a clear message if not. DINOv2
        # ('torch_hub') is ungated and auto-downloads, so it is not checked. A no-op for SAM models.
        import importlib
        from ..models.vfm import is_vfm_model, VFM_MODELS

        if not is_vfm_model(self.model_type):
            return False

        backend = VFM_MODELS[self.model_type]["backend"]
        if backend == "torch_hub":  # DINOv2: ungated, weights auto-download.
            return False

        package = "transformers" if backend == "hf" else "timm"
        try:
            importlib.import_module(package)
        except ImportError:
            return _generate_message(
                "error",
                f"The model '{self.model_type}' requires the '{package}' package, which is not installed. "
                f"Install it (e.g. 'pip install {package}') and try again."
            )

        # These weights are gated on HuggingFace; warn (but allow continuing, e.g. if already cached).
        try:
            from huggingface_hub import get_token
            has_token = get_token() is not None
        except Exception:
            has_token = False
        if not has_token:
            return _generate_message(
                "info",
                f"'{self.model_type}' is a gated model on Hugging Face. Request access on its Hugging Face "
                "page and authenticate via 'huggingface-cli login' or the 'HF_TOKEN' environment variable. "
                "If the weights are already downloaded you can continue; otherwise the download will fail."
            )
        return False


# Custom signals for managing progress updates.
class PBarSignals(QObject):
    pbar_total = Signal(int)
    pbar_update = Signal(int)
    pbar_description = Signal(str)
    pbar_stop = Signal()
    pbar_reset = Signal()


class InfoDialog(QtWidgets.QDialog):
    def __init__(self, title, message, buttons=("OK", "Cancel")):
        super().__init__()
        self.setWindowTitle(title)
        # Label of the button the user clicked (None if the dialog was closed without a button).
        self.clicked_label = None
        # The first button accepts the dialog; the rest reject it.
        self._accept_label = buttons[0]

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel(message))

        # Buttons side-by-side; the first is the default so Enter triggers it.
        button_box = QtWidgets.QHBoxLayout()
        for i, label in enumerate(buttons):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, lbl=label: self.button_clicked(lbl))
            if i == 0:
                button.setDefault(True)
                button.setFocus()
            button_box.addWidget(button)

        layout.addLayout(button_box)
        self.setLayout(layout)

    def button_clicked(self, label):
        self.clicked_label = label
        if label == self._accept_label:
            self.accept()
        else:
            self.reject()


# Set up the progress bar. We handle this via custom signals that are passed as callbacks to the
# function that does the actual work. We need callbacks for initializing the progress bar,
# updating it and for stopping the progress bar.
def _create_pbar_for_threadworker():
    pbar = progress()
    pbar_signals = PBarSignals()
    pbar_signals.pbar_total.connect(
        lambda total: setattr(pbar, "total", total)
    )
    pbar_signals.pbar_update.connect(lambda update: pbar.update(update))
    pbar_signals.pbar_description.connect(
        lambda description: pbar.set_description(description)
    )
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

    # Reset the choices in the track_id menu (index 2: prompt, track_state, track_id).
    state.annotator._tracking_widget[2].value = "1"
    state.annotator._tracking_widget[2].choices = ["1"]


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
def clear_volume(
    viewer: "napari.viewer.Viewer", all_slices: bool = True
) -> None:
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

    # If it's a SAM2 promptable segmentation workflow,
    # we should reset the prompts after clear annotations has been clicked.
    if state.interactive_segmenter is not None:
        state.interactive_segmenter.reset_predictor()

    # Perform garbage collection.
    gc.collect()


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear_track(
    viewer: "napari.viewer.Viewer", all_frames: bool = True
) -> None:
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
    ovlp = segmentation_overlap(prev_seg, seg)

    mask_ids, prev_mask_ids = [], []
    for prev_id in prev_ids:
        ovlp_table = ovlp.overlaps_for_label_a(prev_id)
        seg_ids, overlaps = ovlp_table["label"], ovlp_table["count"]
        if seg_ids[0] != 0 and overlaps[0] >= preservation_threshold:
            mask_ids.append(seg_ids[0])
            prev_mask_ids.append(prev_id)

    preserve_mask = np.logical_or(
        np.isin(seg, mask_ids), np.isin(prev_seg, prev_mask_ids)
    )
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
        bb = np.s_[z_min : (z_max + 1)]  # noqa

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
        elf.parallel.max(
            viewer.layers["committed_objects"].data,
            block_shape=util.get_block_shape(full_shape),
        )
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
                preserve_mask = _mask_matched_objects(
                    seg, prev_seg, preservation_threshold
                )
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

    segmentation_options = {
        "object_ids": [int(object_id) for object_id in object_ids]
    }
    if widget.with_decoder:
        segmentation_options["boundary_distance_thresh"] = (
            widget.boundary_distance_thresh
        )
        segmentation_options["center_distance_thresh"] = (
            widget.center_distance_thresh
        )
    else:
        segmentation_options["pred_iou_thresh"] = widget.pred_iou_thresh
        segmentation_options["stability_score_thresh"] = (
            widget.stability_score_thresh
        )
        segmentation_options["box_nms_thresh"] = widget.box_nms_thresh

    segmentation_options["min_object_size"] = widget.min_object_size
    if widget.volumetric:
        segmentation_options["apply_to_volume"] = widget.apply_to_volume
        segmentation_options["gap_closing"] = widget.gap_closing
        segmentation_options["min_extent"] = widget.min_extent

    return segmentation_options


def _get_promptable_segmentation_options(state, object_ids):
    segmentation_options = {
        "object_ids": [int(object_id) for object_id in object_ids]
    }
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
        tile_shape, halo = _process_tiling_inputs(
            embeds.tile_x, embeds.tile_y, embeds.halo_x, embeds.halo_y
        )
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
        "committed_objects",
        shape=full_shape,
        chunks=block_shape,
        compression="gzip",
        dtype=seg.dtype,
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
        segmentation_options = _get_auto_segmentation_options(
            state, object_ids
        )
        commit_history.append({"auto_segmentation": segmentation_options})

        # Write the commit history.
        f.attrs["commit_history"] = commit_history

        # If we run commit from the automatic segmentation we don't have
        # any prompts and so don't need to commit anything else.
        return

    segmentation_options, is_tracking = _get_promptable_segmentation_options(
        state, object_ids
    )
    commit_history.append({"current_object": segmentation_options})

    def write_prompts(
        object_id, prompts, point_prompts, point_labels, track_state=None
    ):
        g = f.create_group(f"prompts/{object_id}")
        if prompts is not None and len(prompts) > 0:
            data = np.array(prompts)
            g.create_dataset(
                "prompts", data=data, shape=data.shape, chunks=data.shape
            )
        if point_prompts is not None and len(point_prompts) > 0:
            g.create_dataset(
                "point_prompts",
                data=point_prompts,
                shape=data.shape,
                chunks=point_prompts.shape,
            )
            ds = g.create_dataset(
                "point_labels",
                data=point_labels,
                shape=data.shape,
                chunks=point_labels.shape,
            )
            if track_state is not None:
                ds.attrs["track_state"] = track_state.tolist()

    # Get the prompts from the layers.
    prompts = viewer.layers["prompts"].data
    point_layer = viewer.layers["point_prompts"]
    point_prompts = point_layer.data
    point_labels = point_layer.properties["label"]
    if len(point_prompts) > 0:
        point_labels = np.array(
            [1 if label == "positive" else 0 for label in point_labels]
        )
        assert len(point_prompts) == len(
            point_labels
        ), f"Number of point prompts and labels disagree: {len(point_prompts)} != {len(point_labels)}"

    # Commit the prompts for all the objects in the commit.
    if len(object_ids) == 1:  # We only have a single object.
        write_prompts(object_ids[0], prompts, point_prompts, point_labels)

    elif (
        is_tracking
    ):  # We have multiple objects from tracking a lineage with divisions.
        track_ids_points = np.array(point_layer.properties["track_id"])
        track_ids_prompts = np.array(
            viewer.layers["prompts"].properties["track_id"]
        )

        unique_track_ids = np.unique(track_ids_points)
        assert len(unique_track_ids) == len(object_ids)
        track_state = np.array(point_layer.properties["state"])
        for track_id, object_id in zip(unique_track_ids, object_ids):
            this_prompts = (
                None
                if len(prompts) == 0
                else prompts[track_ids_prompts == track_id]
            )
            point_mask = track_ids_points == track_id
            this_points, this_labels, this_track_state = (
                point_prompts[point_mask],
                point_labels[point_mask],
                track_state[point_mask],
            )
            write_prompts(
                object_id,
                this_prompts,
                this_points,
                this_labels,
                track_state=this_track_state,
            )

    else:  # We have multiple objects, which are the result from batched interactive segmentation.
        # Note: we can't match exact object ids to their prompts, for batched segmentation.
        # We first write the objects from box prompts, then from point prompts.
        n_prompts, n_points = len(prompts), len(point_prompts)
        assert n_prompts + n_points == len(
            object_ids
        ), f"Number of prompts and objects disagree: {n_prompts} + {n_points} != {len(object_ids)}"
        for i, object_id in enumerate(object_ids):
            if i < n_prompts:
                this_prompts, this_points, this_labels = (
                    prompts[i : i + 1],  # noqa
                    None,
                    None,
                )
            else:
                j = i - n_prompts
                this_prompts, this_points, this_labels = (
                    None,
                    point_prompts[j : j + 1],  # noqa
                    point_labels[j : j + 1],  # noqa
                )
            write_prompts(object_id, this_prompts, this_points, this_labels)

    # Write the commit history.
    f.attrs["commit_history"] = commit_history


def _call_button_tooltip(widget_type, name):
    # Returns a magic_factory 'widget_init' that sets the call button's tooltip (magicgui call buttons
    # cannot be given a tooltip via the decorator directly).
    def _init(widget):
        widget.call_button.tooltip = get_tooltip(widget_type, name)
    return _init


@magic_factory(
    call_button="Commit [C]",
    widget_init=_call_button_tooltip("commit", "commit_button"),
    layer={
        "choices": ["current_object", "auto_segmentation"],
        "tooltip": get_tooltip("commit", "layer"),
    },
    preserve_mode={
        "choices": ["objects", "pixels", "none"],
        "tooltip": get_tooltip("commit", "preserve_mode"),
    },
    commit_path={"mode": "d", "tooltip": get_tooltip("commit", "commit_path")},
)
def commit(
    viewer: "napari.viewer.Viewer",
    layer: str = "current_object",
    preserve_mode: str = "pixels",
    preservation_threshold: float = 0.75,
    commit_path: Optional[Path] = None,
) -> None:
    """Widget for committing the segmented objects from automatic or interactive segmentation.

    Args:
        viewer: The napari viewer.
        layer: Select the layer to commit. Can be either 'current_object' to commit interacitve segmentation results.
            Or 'auto_segmentation' to commit automatic segmentation results.
        preserve_mode: The mode for preserving already committed objects, in order to prevent over-writing
            them by a new commit. Supports the modes 'objects', which preserves on the object level,
            'pixels', which preserves on the pixel-level and is the default, or 'none', which does not
            preserve commited objects.
        preservation_threshold: The overlap threshold for preserving objects. This is only used if
            preservation_mode is set to 'objects'.
        commit_path: Select a file path where the committed results and prompts will be saved.
            This feature is not implemented yet and will be supported in a future release.
    """
    # Saving committed results to file is not supported yet.
    if commit_path is not None:
        raise NotImplementedError(
            "Saving committed results to 'commit_path' is not supported yet and will be added in a future release."
        )

    # Commit the segmentation layer.
    _commit_impl(viewer, layer, preserve_mode, preservation_threshold)

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
    widget_init=_call_button_tooltip("commit", "commit_button"),
    layer={
        "choices": ["current_object", "auto_segmentation"],
        "tooltip": get_tooltip("commit", "layer"),
    },
    preserve_mode={
        "choices": ["objects", "pixels", "none"],
        "tooltip": get_tooltip("commit", "preserve_mode"),
    },
    commit_path={"mode": "d", "tooltip": get_tooltip("commit", "commit_path")},
)
def commit_track(
    viewer: "napari.viewer.Viewer",
    layer: str = "current_object",
    preserve_mode: str = "pixels",
    preservation_threshold: float = 0.75,
    commit_path: Optional[Path] = None,
) -> None:
    """Widget for committing the objects from interactive tracking.

    Args:
        viewer: The napari viewer.
        layer: Select the layer to commit. Can be either 'current_object' to commit interacitve segmentation results.
            Or 'auto_segmentation' to commit automatic segmentation results.
        preserve_mode: The mode for preserving already committed objects, in order to prevent over-writing
            them by a new commit. Supports the modes 'objects', which preserves on the object level,
            'pixels', which preserves on the pixel-level and is the default, or 'none', which does not
            preserve commited objects.
        preservation_threshold: The overlap threshold for preserving objects. This is only used if
            preservation_mode is set to 'objects'.
        commit_path: Select a file path where the committed results and prompts will be saved.
            This feature is still experimental.
    """
    # Commit the segmentation layer.
    id_offset, seg, mask, bb = _commit_impl(
        viewer, layer, preserve_mode, preservation_threshold
    )

    # Update the lineages.
    state = AnnotatorState()
    lineage = state.lineage

    if isinstance(
        lineage, list
    ):  # This is a list of lineages from auto-tracking.
        assert id_offset == 0
        assert len(state.committed_lineages) == 0
        state.committed_lineages.extend(lineage)
    else:  # This is a single lineage from interactive tracking.
        updated_lineage = {
            parent + id_offset: [child + id_offset for child in children]
            for parent, children in state.lineage.items()
        }
        state.committed_lineages.append(updated_lineage)

    if commit_path is not None:
        _commit_to_file(
            commit_path,
            viewer,
            layer,
            seg,
            mask,
            bb,
            extra_attrs={"committed_lineages": state.committed_lineages},
        )

    if layer == "current_object":
        vutil.clear_annotations(viewer)

    # Create / update the tracking layer.
    layer_name = "tracks"
    segmentation = viewer.layers["committed_objects"].data
    track_data, parent_graph = get_napari_track_data(
        segmentation, state.committed_lineages
    )
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


@magic_factory(
    call_button="Export",
    widget_init=_call_button_tooltip("annotator_tracking", "export_button"),
    export_format={"choices": ["CTC", "GEFF", "TrackMate XML"]},
    export_folder={"mode": "d"},  # choose a directory
)
def export_track(
    viewer: "napari.viewer.Viewer",
    export_format: str = "CTC",
    export_folder: Path = Path.cwd(),
) -> None:
    """Widget for exporting the committed tracking result.

    Args:
        viewer: The napari viewer.
        export_format: The tracking export format. Supports 'CTC', 'GEFF' and 'TrackMate XML'.
        export_folder: The folder where the export is written. By default the current working directory is used.
    """
    if "committed_objects" not in viewer.layers or viewer.layers["committed_objects"].data.max() == 0:
        _generate_message("error", "There are no committed tracking results to export yet.")
        return

    segmentation = viewer.layers["committed_objects"].data
    lineages = AnnotatorState().committed_lineages
    if export_format == "CTC":
        export_tracking_result_to_ctc(segmentation, lineages, export_folder)
        show_info(f"Exported the tracking result to CTC format in '{export_folder}'.")
    elif export_format == "GEFF":
        export_path = export_tracking_result_to_geff(segmentation, lineages, export_folder)
        show_info(f"Exported the tracking result to GEFF format in '{export_path}'.")
    elif export_format == "TrackMate XML":
        export_path = export_tracking_result_to_trackmate_xml(segmentation, lineages, export_folder)
        show_info(f"Exported the tracking result to TrackMate XML format in '{export_path}'.")
    else:
        _generate_message("error", f"Unsupported tracking export format: {export_format}.")


def create_prompt_menu(
    points_layer, labels, menu_name="prompt", label_name="label", linked_layers=None,
):
    """Create a menu that keeps point and optional shape prompt labels synchronized."""
    prompt_layers = [points_layer] + ([] if linked_layers is None else list(linked_layers))
    label_menu = ComboBox(
        label=menu_name,
        choices=labels,
        tooltip=get_tooltip("prompt_menu", "labels"),
    )
    label_widget = Container(widgets=[label_menu])

    def update_label_menu(event):
        new_label = str(event.source.current_properties[label_name][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    for layer in prompt_layers:
        layer.events.current_properties.connect(update_label_menu)

    def label_changed(new_label):
        for layer in prompt_layers:
            if label_name == "label":
                vutil.set_prompt_label(layer, new_label)
            else:
                current_properties = layer.current_properties
                current_properties[label_name] = np.array([new_label])
                layer.current_properties = current_properties
                layer.refresh_colors()

    label_menu.changed.connect(label_changed)

    return label_widget


@magic_factory(
    call_button="Update settings",
    cache_directory={"mode": "d"},  # choose a directory
)
def settings_widget(
    cache_directory: Optional[Path] = util.get_cache_directory(),
) -> None:
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
        QtWidgets.QMessageBox.critical(
            None, "Error", message, QtWidgets.QMessageBox.Ok
        )
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


def _ask_load_or_recompute(message: str) -> str:
    """Ask the user whether to load existing embeddings or recompute them.

    Returns 'load', 'recompute' or 'cancel'.
    """
    dialog = InfoDialog(title="Existing embeddings found", message=message, buttons=("Load", "Recompute", "Cancel"))
    dialog.exec_()
    return {"Load": "load", "Recompute": "recompute"}.get(dialog.clicked_label, "cancel")


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


def _validate_layers(
    viewer: "napari.viewer.Viewer", automatic_segmentation: bool = False
) -> bool:
    # Check whether all layers exist as expected or create new ones automatically.
    state = AnnotatorState()
    state.annotator._require_layers()

    if not automatic_segmentation:
        # Check prompts layer.
        if (
            len(viewer.layers["prompts"].data) == 0
            and len(viewer.layers["point_prompts"].data) == 0
        ):
            msg = "No prompts were given. Please provide prompts to run interactive segmentation."
            return _generate_message("error", msg)
        else:
            return False


def _embeddings_are_tiled(state):
    """Whether the current embeddings are tiled (tiled embeddings have a top-level 'input_size' of None)."""
    return state.image_embeddings is not None and state.image_embeddings.get("input_size") is None


def _batched_disabled_when_tiled(state, batched):
    """Batched (multi-object) prompting is not supported with tiling: each tile is segmented
    independently, so object ids would collide across tiles. Force single-object and warn."""
    if batched and _embeddings_are_tiled(state):
        show_info("Batched (multi-object) prompting is not supported with tiling. Running single-object.")
        return False
    return batched


def _segment_object_2d(viewer, batched=False):
    """Segment object(s) in 2d for the current prompts.

    This is the shared implementation used by the `segment` widget and the
    `InteractiveSegmentationWidget`.

    Args:
        viewer: The napari viewer.
        batched: Choose if you want to segment multiple objects with point prompts.
    """
    if _validate_embeddings(viewer):
        return None
    if _validate_layers(viewer):
        return None

    shape = viewer.layers["current_object"].data.shape

    # Get the current box, point and open-stroke prompts. Scribbles are encoded through SAM's
    # existing sparse point prompt embeddings, so the predictor interface remains unchanged.
    boxes, masks = vutil.shape_layer_to_prompts(
        viewer.layers["prompts"], shape
    )
    points, labels = vutil.point_layer_to_prompts(
        viewer.layers["point_prompts"], with_stop_annotation=False
    )
    scribble_points, scribble_labels = vutil.scribble_layer_to_prompts(
        viewer.layers["prompts"], image_shape=shape
    )
    points, labels = vutil.merge_point_prompts(
        (points, labels), (scribble_points, scribble_labels)
    )

    have_scribbles = len(scribble_points) > 0
    if have_scribbles and not len(boxes) and not np.any(labels == 1):
        msg = "A negative scribble needs a positive point, positive scribble, box or mask prompt."
        return _generate_message("error", msg)

    state = AnnotatorState()
    predictor = state.predictor
    image_embeddings = state.image_embeddings
    batched = _batched_disabled_when_tiled(state, batched)
    if have_scribbles and batched:
        show_info("Batched segmentation is not supported with scribble prompts. Running single-object.")
        batched = False

    if state.is_sam2:
        # When the embeddings are tiled (top-level 'input_size' is None), route the prompts to the
        # matching tile and stitch; otherwise the predictor already holds the single image embedding.
        if image_embeddings is not None and image_embeddings.get("input_size") is None:
            from micro_sam.v2.prompt_based_segmentation import tiled_promptable_segmentation_2d
            seg = tiled_promptable_segmentation_2d(
                predictor=predictor, image_embeddings=image_embeddings,
                points=points, labels=labels, boxes=boxes, masks=masks, batched=batched,
            )
        else:
            from micro_sam.v2.prompt_based_segmentation import promptable_segmentation_2d
            seg = promptable_segmentation_2d(
                predictor=predictor,
                points=points,
                labels=labels,
                boxes=boxes,
                masks=masks,
                batched=batched,
            )
    else:
        seg = vutil.prompt_segmentation(
            predictor,
            points,
            labels,
            boxes,
            masks,
            shape,
            image_embeddings=image_embeddings,
            multiple_box_prompts=True,
            batched=batched,
            previous_segmentation=viewer.layers["current_object"].data,
        )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print(
            "You either haven't provided any prompts or invalid prompts. The segmentation will be skipped."
        )
        return

    viewer.layers["current_object"].data = seg
    viewer.layers["current_object"].refresh()


@magic_factory(call_button="Segment Object [S]")
def segment(viewer: "napari.viewer.Viewer", batched: bool = False) -> None:
    """Segment object(s) for the current prompts.

    Args:
        viewer: The napari viewer.
        batched: Choose if you want to segment multiple objects with point prompts.
    """
    _segment_object_2d(viewer, batched=batched)


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
            tile_shape = (
                tile_shape[0],
                256,
            )  # Create a new tuple with modified value
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
    # Whether to show the CPU info popup for expensive (many-tile / 3D) computations.
    warn_on_cpu = True

    # A tiled 2D image only gets slow on the CPU once it has many tiles; warn from this many on.
    cpu_warn_tiles = 64

    # Whether to offer the 'cache automatic segmentation state' option (segmentation / tracking only,
    # not the classification tools which have no automatic segmentation).
    supports_state_caching = True

    def __init__(self, parent=None, sam2_only=False, ndim_choice=False, is_timeseries=False):
        super().__init__(parent=parent)
        self.sam2_only = sam2_only
        # Whether to expose the 'image dimensions' (ndim) override dropdown. Only the segmentation
        # annotator wires it into image normalization, so it is off by default (hidden for tracking
        # and the classifiers, which do not use it).
        self.ndim_choice = ndim_choice
        # The tracking annotator operates on a (T, H, W) timeseries, not a 3D volume; relabel the
        # embedding progress accordingly (the underlying compute path is the same as for 3D).
        self.is_timeseries = is_timeseries

        # Create a nested layout for the sections.
        # Section 1: Image and Model.
        section1_layout = QtWidgets.QHBoxLayout()
        section1_layout.addLayout(self._create_image_section())
        # Default to the single-source 'DEFAULT_MODEL' (the 'Microscopy' / 'hvit_t_cells' model);
        # '_create_model_section' derives the synthetic selector string from it when no value is given.
        section1_layout.addLayout(
            self._create_model_section()
        )  # Creates the model family widget section.
        self.layout().addLayout(section1_layout)

        # Section 2: Settings (collapsible).
        self.layout().addWidget(self._create_settings_widget())

        # Enable sensible default tiling when a large image is selected.
        self.image_selection.changed.connect(self._set_default_tiling)
        self._set_default_tiling()

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

        # Drop the channel axis for RGB images so the shape and ndim describe the spatial dims only.
        if layer.rgb:
            state.ndim = layer.data.ndim - 1
            state.image_shape = layer.data.shape[:-1]
        else:
            state.ndim = layer.data.ndim
            state.image_shape = layer.data.shape
        state.image_scale = tuple(layer.scale)
        state.image_name = layer.name

    def _create_image_section(self):
        image_section = QtWidgets.QVBoxLayout()
        layer_label = "Timeseries Layer:" if self.is_timeseries else "Image Layer:"
        self.image_layer_label = QtWidgets.QLabel(layer_label)
        # self.image_layer_label.setToolTip(get_tooltip("embedding", "image")) #  this adds tooltip to label
        image_section.addWidget(self.image_layer_label)

        # Setting a napari layer in QT, see:
        # https://github.com/pyapp-kit/magicgui/blob/main/docs/examples/napari/napari_combine_qt.py
        self.image_selection = create_widget(annotation=napari.layers.Image)
        self.image_selection.native.setToolTip(
            get_tooltip("embedding", "image")
        )
        image_section.addWidget(self.image_selection.native)

        return image_section

    def _update_model(self, state):
        _model_type = (
            state.predictor.model_type
            if self.custom_weights
            else self.model_type
        )

        # Provide a detailed message for the model family and model size per chosen combination.
        msg = "Computed embeddings for "
        if (
            self.custom_weights
        ):  # Whether the user provided a filepath to custom finetuned model weights.
            msg += f"the model located at '{os.path.abspath(self.custom_weights)}' "
            size_key = _model_type[4] if len(_model_type) > 4 else ""
            msg += f"of size '{self._model_size_map.get(size_key, self.model_size)}'."
        else:
            msg += (
                f"the '{self.model_family}' model of size '{self.model_size}'."
            )

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
            # Only forward tiling params when tiling is actually enabled; otherwise '_sync_embedding_widget'
            # would force the tiling dropdown to "yes" using the (always-nonzero) default tile values.
            tile_shape=[self.tile_x, self.tile_y] if self.tiling == "yes" else None,
            halo=[self.halo_x, self.halo_y] if self.tiling == "yes" else None,
        )

        # Set the default settings for this model in the autosegment widget if it is part of
        # the currently used plugin.
        if "autosegment" in state.widgets:
            with_decoder = state.decoder is not None
            vutil._sync_autosegment_widget(
                state.widgets["autosegment"],
                _model_type,
                self.custom_weights,
                update_decoder=with_decoder,
            )
            # Load the AMG/AIS state cache. For SAM2 the state cache (grid masks or decoder
            # predictions) is recorded via '_autoseg_state_descriptor' and the widget reads/writes it
            # on demand; SAM1 preloads the per-slice 3d state as before.
            if state.is_sam2:
                state.autoseg_state = vutil._autoseg_state_descriptor(
                    state.embedding_path, "ais" if with_decoder else "amg",
                )
            elif state.widgets["autosegment"].volumetric and with_decoder:
                state.autoseg_state = vutil._load_is_state(state.embedding_path)
            elif state.widgets["autosegment"].volumetric and not with_decoder:
                state.autoseg_state = vutil._load_amg_state(state.embedding_path)

        # Set the default settings for this model in the nd-segmentation widget if it is part of
        # the currently used plugin.
        if "segment_nd" in state.widgets:
            vutil._sync_ndsegment_widget(
                state.widgets["segment_nd"], _model_type, self.custom_weights
            )

        # Now that the (possibly tiled) embeddings are known, refresh the 'Batched' control on the
        # segment/track widgets: batched prompting is unsupported with tiling, so it is hidden then.
        for widget in state.widgets.values():
            update_batched = getattr(widget, "_update_batched_visibility", None)
            if callable(update_batched):
                update_batched()

    def _create_settings_widget(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # Optional image dimensionality override. 'auto' detects 2d/3d (including channels) from the
        # selected image; '2d'/'3d' force the interpretation, e.g. to read a channels-first
        # (C, H, W) array as a 2d multi-channel image.
        if self.ndim_choice:
            self.image_ndim_mode = "auto"
            self.image_ndim_dropdown, ndim_layout = self._add_choice_param(
                "image_ndim_mode", self.image_ndim_mode, ["auto", "2d", "3d"], title="image dimensions:",
                tooltip="Spatial dimensionality of the image. 'auto' detects it (a channels-first "
                "array is read as a volume); set '2d' to read a multi-channel array (e.g. (C, H, W) "
                "or (H, W, C)) as a single 2d image, or '3d' to force a (Z, H, W) volume.",
            )
            setting_values.layout().addLayout(ndim_layout)

        # Create UI for tiling. A dropdown toggles whether tiling is used; when enabled,
        # the tile shape and halo fields are revealed with sensible defaults.
        self.tiling = "no"
        self.tiling_dropdown, layout = self._add_choice_param(
            "tiling",
            self.tiling,
            ["no", "yes"],
            title="tiling:",
            update=self._update_tiling_visibility,
            tooltip=get_tooltip("embedding", "tiling"),
        )
        setting_values.layout().addLayout(layout)

        # Container holding the tile shape and halo fields (hidden unless tiling is 'yes').
        self._tiling_widget = QtWidgets.QWidget()
        self._tiling_widget.setLayout(QtWidgets.QVBoxLayout())
        self._tiling_widget.layout().setContentsMargins(0, 0, 0, 0)

        # In-plane (xy) tile shape and halo, used when tiling is enabled. The defaults come from the
        # central v2 tiling values. The z block / halo are not set here: they only affect 3d automatic
        # segmentation (not the embeddings, which are tiled in-plane only), so they live in the
        # automatic segmentation settings instead.
        from micro_sam.v2.util import DEFAULT_TILE_SHAPE, DEFAULT_HALO
        self.tile_x, self.tile_y = DEFAULT_TILE_SHAPE
        self.tile_x_param, self.tile_y_param, tile_layout = self._add_shape_param(
            ("tile_x", "tile_y"),
            (self.tile_x, self.tile_y),
            min_val=0,
            max_val=2048,
            step=16,
            tooltip=get_tooltip("embedding", "tiling"),
        )
        self._tiling_widget.layout().addLayout(tile_layout)

        self.halo_x, self.halo_y = DEFAULT_HALO
        self.halo_x_param, self.halo_y_param, halo_layout = self._add_shape_param(
            ("halo_x", "halo_y"),
            (self.halo_x, self.halo_y),
            min_val=0,
            max_val=512,
            title=("overlap_x", "overlap_y"),
            tooltip=get_tooltip("embedding", "halo"),
        )
        self._tiling_widget.layout().addLayout(halo_layout)

        self._tiling_widget.setVisible(False)
        setting_values.layout().addWidget(self._tiling_widget)

        # Add the model size widget section.
        layout = self._create_model_size_section()
        setting_values.layout().addLayout(layout)

        # Create UI for the device.
        self.device = "auto"
        device_options = ["auto"] + util._available_devices()

        self.device_dropdown, layout = self._add_choice_param(
            "device",
            self.device,
            device_options,
            tooltip=get_tooltip("embedding", "device"),
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the save path.
        self.embeddings_save_path = None
        self.embeddings_save_path_param, save_layout = self._add_path_param(
            "embeddings_save_path",
            self.embeddings_save_path,
            "directory",
            title="embeddings save path:",
            tooltip=get_tooltip("embedding", "embeddings_save_path"),
        )
        setting_values.layout().addLayout(save_layout)

        # Opt-in disk caching of the automatic-segmentation state (off by default). When on, the state
        # is precomputed to disk (next to the embeddings) while the embeddings are computed and reused
        # across runs / sessions. The automatic segmentation widget reads this flag from here. Not
        # offered by the classification tools (no automatic segmentation).
        self.cache_state = False
        if self.supports_state_caching:
            self.cache_state_checkbox = self._add_boolean_param(
                "cache_state", self.cache_state, title="cache automatic segmentation state",
                tooltip=get_tooltip("embedding", "cache_state"),
            )
            setting_values.layout().addWidget(self.cache_state_checkbox)

        # Create UI for the custom weights.
        self.custom_weights = None
        self.custom_weights_param, weights_layout = self._add_path_param(
            "custom_weights",
            self.custom_weights,
            "file",
            title="custom weights path:",
            tooltip=get_tooltip("embedding", "custom_weights"),
        )
        setting_values.layout().addLayout(weights_layout)

        # Make the two path rows symmetric: equal label widths, so their text boxes match too.
        self._align_widths([save_layout.itemAt(0).widget(), weights_layout.itemAt(0).widget()])
        # Move the horizontal space trimmed from the text boxes into the browse buttons (rather than
        # leaving a gap): widen them uniformly by that amount ('Select Directory' is wider than
        # 'Select File', which would otherwise offset the rows).
        path_button_extra = 20
        save_button = save_layout.itemAt(save_layout.count() - 1).widget()
        weights_button = weights_layout.itemAt(weights_layout.count() - 1).widget()
        button_width = max(save_button.sizeHint().width(), weights_button.sizeHint().width()) + path_button_extra
        for button in (save_button, weights_button):
            button.setFixedWidth(button_width)

        # Hook for subclasses to add extra model controls at the end of the settings (no-op by default).
        self._add_extra_model_settings(setting_values.layout())

        settings = _make_collapsible(
            setting_values, title="Embedding Settings", tooltip=get_tooltip("embedding", "settings"),
        )
        return settings

    def _add_extra_model_settings(self, layout):
        """Hook to add extra model controls to the embedding settings. No-op by default; the
        classification embedding widget uses it to add the optional advanced-model selector."""
        pass

    def _apply_loaded_model_selection(self, model_name):
        """Reflect a loaded model in the model family / size dropdowns. No-op by default: the
        post-compute '_sync_embedding_widget' already syncs the SAM2-only widget, whose family
        names match. Subclasses with custom family handling (classification) override this."""
        pass

    def _selected_image_ndim(self):
        # Spatial dimensionality of the currently selected image layer (2 or 3), or None if no image.
        image = self.image_selection.get_value()
        if image is None:
            return None
        shape = image.data.shape[:-1] if image.rgb else image.data.shape
        return len(shape)

    def _ndim_override(self):
        # The user-selected image dimensionality override: None for 'auto', else 2 or 3. Read the
        # dropdown directly so the value is correct regardless of signal ordering.
        dropdown = getattr(self, "image_ndim_dropdown", None)
        mode = dropdown.currentText() if dropdown is not None else "auto"
        return {"auto": None, "2d": 2, "3d": 3}.get(mode)

    def _update_tiling_visibility(self, index=None):
        # Show the in-plane tile shape and halo fields only when tiling is enabled.
        self.tiling = self.tiling_dropdown.currentText()
        self._tiling_widget.setVisible(self.tiling == "yes")

    def _apply_default_tiling_for_shape(self, shape):
        # Enable tiling by default for large in-plane images, using the central v2 tiling defaults.
        # 'shape' is the spatial image shape (channel axis already removed). Shared by the layer-based
        # auto-tiling and the batch launcher (which judges from the first file in the folder).
        from micro_sam.v2.util import needs_default_tiling, DEFAULT_TILE_SHAPE, DEFAULT_HALO

        if shape is not None and needs_default_tiling(shape):
            self.tile_x, self.tile_y = DEFAULT_TILE_SHAPE
            self.halo_x, self.halo_y = DEFAULT_HALO
            self.tile_x_param.setValue(self.tile_x)
            self.tile_y_param.setValue(self.tile_y)
            self.halo_x_param.setValue(self.halo_x)
            self.halo_y_param.setValue(self.halo_y)
            self.tiling_dropdown.setCurrentText("yes")

        # Refresh which tiling fields are visible now that the image (and its dimensionality) changed.
        self._update_tiling_visibility()

    def _set_default_tiling(self, *args):
        image = self.image_selection.get_value()
        if image is None:
            return
        shape = image.data.shape[:-1] if image.rgb else image.data.shape
        self._apply_default_tiling_for_shape(shape)

    def _reset_inputs_to_defaults(self):
        """Reset the user inputs to their fresh-open defaults.

        Called when the selected image changes, so a new image starts exactly as a freshly opened
        tool would: the default model family/size, no custom weights, default tiling parameters and
        no embeddings save path. The auto-tiling rule for the new image is re-applied at the end via
        '_set_default_tiling' (which also refreshes field visibility), matching a fresh open.
        """
        # Clear custom weights first, then restore the default model family + size. (Setting custom
        # weights blanks the family dropdown, so clearing it before re-selecting keeps them in sync.)
        self.custom_weights_param.setText("")
        default_family = {v: k for k, v in self.supported_dropdown_maps.items()}[self._default_model_choice[5:]]
        default_size = self._model_size_map[self._default_model_choice[4]]
        self.model_family_dropdown.setCurrentText(default_family)
        self.model_size_dropdown.setCurrentText(default_size)

        # Reset the in-plane tiling parameters to their creation defaults and the save path; the on/off
        # state is then decided by '_set_default_tiling' (auto-enabled for large images, as on a fresh open).
        from micro_sam.v2.util import DEFAULT_TILE_SHAPE, DEFAULT_HALO
        self.tile_x_param.setValue(DEFAULT_TILE_SHAPE[0])
        self.tile_y_param.setValue(DEFAULT_TILE_SHAPE[1])
        self.halo_x_param.setValue(DEFAULT_HALO[0])
        self.halo_y_param.setValue(DEFAULT_HALO[1])
        self.tiling_dropdown.setCurrentText("no")
        self.embeddings_save_path_param.setText("")

        # Reset the image-dimensionality override back to 'auto' so a new image is re-detected.
        # Block the signal so this does not re-trigger the annotator's normalization mid-reset.
        if getattr(self, "ndim_choice", False):
            self.image_ndim_dropdown.blockSignals(True)
            self.image_ndim_dropdown.setCurrentText("auto")
            self.image_ndim_mode = "auto"
            self.image_ndim_dropdown.blockSignals(False)

        self._set_default_tiling()

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
        if (
            self.embeddings_save_path and os.path.isdir(self.embeddings_save_path)
            and os.listdir(self.embeddings_save_path)
        ):
            try:
                f = util._open_embeddings(self.embeddings_save_path, mode="a")

                # Validate that the embeddings are complete.
                # Note: 'input_size' is the last value set in the attrs of f,
                # so we can use it as a proxy to check if the embeddings are fully computed
                if "input_size" not in f.attrs:
                    msg = (
                        f"The embeddings at {self.embeddings_save_path} are incomplete. "
                        "Specify a different path or remove them."
                    )
                    return _generate_message("error", msg)

                # Validate image data signature.
                if "data_signature" in f.attrs:
                    image = self.image_selection.get_value()
                    img_signature = util._compute_data_signature(image.data)
                    if img_signature != f.attrs["data_signature"]:
                        msg = f"The embeddings don't match with the image: {img_signature} {f.attrs['data_signature']}"
                        return _generate_message("error", msg)

                # The model the saved embeddings were computed with.
                saved_model = f.attrs.get("model_name") or f.attrs.get("model_type")
                if saved_model is None:
                    return _generate_message(
                        "error", f"The embeddings at '{self.embeddings_save_path}' do not record a model."
                    )

                # Ask the user whether to load the saved embeddings or recompute them. The message
                # reflects what changed vs the saved embeddings: a model swap, custom weights (whose
                # identity we cannot verify here, so always flag them), or an exact match.
                if self.custom_weights:
                    message = (
                        f"Saved embeddings use '{saved_model}'; custom weights are now selected. "
                        "Load the saved embeddings or recompute?"
                    )
                elif saved_model != self.model_type:
                    message = (
                        f"Saved embeddings use '{saved_model}', but '{self.model_type}' is selected. "
                        "Load the saved embeddings or recompute?"
                    )
                else:
                    message = f"Embeddings for '{saved_model}' already exist. Load or recompute?"
                choice = _ask_load_or_recompute(message)

                if choice == "cancel":
                    return True

                if choice == "recompute":
                    # Recompute with the user's current selection: clear the saved file so the backend
                    # recomputes from scratch (works for any model and even when the model is unchanged).
                    # Tiling and model stay as the user set them in the widget.
                    util._open_embeddings(self.embeddings_save_path, mode="w")
                    return False

                # 'load': adopt the saved model and tiling, then load the existing embeddings. Clear any
                # custom weights so the predictor matches the loaded embeddings ('saved_model'), not a
                # mismatched custom checkpoint.
                self.custom_weights = None
                self.custom_weights_param.setText("")
                self.model_type = saved_model
                if self._validate_model_support():
                    return True
                # Reflect the loaded model in the model family / size dropdowns.
                self._apply_loaded_model_selection(saved_model)
                if "tile_shape" in f.attrs and f.attrs["tile_shape"] is not None:
                    self.tile_x, self.tile_y = f.attrs["tile_shape"]
                    self.halo_x, self.halo_y = f.attrs["halo"]
                    # Reflect the loaded tiling parameters in the UI.
                    self.tile_x_param.setValue(self.tile_x)
                    self.tile_y_param.setValue(self.tile_y)
                    self.halo_x_param.setValue(self.halo_x)
                    self.halo_y_param.setValue(self.halo_y)
                    self.tiling_dropdown.setCurrentText("yes")
                else:
                    self.tiling_dropdown.setCurrentText("no")
                return False

            except RuntimeError as e:
                val_results = {
                    "message_type": "error",
                    "message": f"Failed to load image embeddings: {e}",
                }
                return _generate_message(
                    val_results["message_type"], val_results["message"]
                )

        # Otherwise we either don't have an embedding path or it is empty. We can proceed in both cases.
        return False

    def _validate_existing_embeddings(self, state):
        # When an embeddings save path is set, '_validate_inputs' already offered load-vs-recompute,
        # so don't prompt again here. This only handles the in-memory case (no save path).
        if self.embeddings_save_path:
            return False
        if state.image_embeddings is None:
            return False
        return _generate_message(
            "info", "Embeddings have already been precomputed. Press OK to recompute the embeddings."
        )

    @staticmethod
    def _clear_autosegment_cache(state):
        """Discard predictions derived from the embeddings that were just replaced."""
        widget = state.widgets.get("autosegment")
        if widget is not None:
            widget._segmenter = None
            widget._segmenter_key = None

    def _n_tiles(self, tile_shape, shape):
        """The number of tiles the embedding computation is split into (1 without tiling)."""
        if tile_shape is None:
            return 1
        from bioimage_cpp.utils import Blocking
        return Blocking([0, 0], list(shape[:2]), list(tile_shape)).number_of_blocks

    def _maybe_warn_cpu(self, ndim, tile_shape, shape):
        """Show a one-time-per-session info popup for expensive CPU computations (many tiles or 3D)."""
        state = AnnotatorState()
        if state.cpu_info_shown or not self.warn_on_cpu:
            return
        # 3D runs the encoder per slice; 2D is a single pass per tile, which only adds up for many tiles.
        if ndim < 3 and self._n_tiles(tile_shape, shape) < self.cpu_warn_tiles:
            return
        if str(util.get_device(self.device)) != "cpu":
            return
        state.cpu_info_shown = True
        data_kind = "timeseries" if self.is_timeseries else "3D"  # A timeseries uses the 3D compute path.
        QtWidgets.QMessageBox.information(
            self, "Running on CPU",
            f"micro_sam is running on the CPU, so computations can be slow for tiled or {data_kind} data. "
            "Using a GPU is recommended.",
            QtWidgets.QMessageBox.Ok,
        )

    def __call__(self, skip_validate=False):
        self._validate_model_type_and_custom_weights()
        if self._validate_model_support():
            return

        # For gated advanced (DINO / UNI) models, check the backend package + HuggingFace access.
        if self._validate_vfm_requirements():
            return

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
        state.ndim = ndim

        # Set layer scale
        state.image_scale = tuple(image.scale)

        # Process tile_shape and halo, set other data. Tiling is only applied when enabled.
        if self.tiling == "yes":
            tile_shape, halo = _process_tiling_inputs(
                self.tile_x, self.tile_y, self.halo_x, self.halo_y
            )
            # Reflect the values actually used (after normalization) back in the UI so they are retained.
            if tile_shape is not None:
                self.tile_x, self.tile_y = tile_shape
                self.tile_x_param.setValue(self.tile_x)
                self.tile_y_param.setValue(self.tile_y)
            if halo is not None:
                self.halo_x, self.halo_y = halo
                self.halo_x_param.setValue(self.halo_x)
                self.halo_y_param.setValue(self.halo_y)
        else:
            tile_shape, halo = None, None
        save_path = (
            None
            if self.embeddings_save_path == ""
            else self.embeddings_save_path
        )
        image_data = image.data

        # Warn CPU users once per session that processing can be slow.
        self._maybe_warn_cpu(ndim, tile_shape, state.image_shape)

        # Eager caching of the automatic-segmentation state: if the 'cache automatic segmentation state'
        # option (in these embedding settings) is on, precompute the state to disk while computing the
        # embeddings. It persists wherever the embeddings live on disk - the given save path, or the
        # ephemeral zarr the eager setup creates for SAM2 volumes / tiled images (removed on reset).
        # Plain in-memory 2d has no disk location, so it needs a save path to persist.
        is_sam2 = self.model_type.startswith("h")
        embeddings_on_disk = save_path is not None or (is_sam2 and (ndim == 3 or tile_shape is not None))
        precompute_autoseg_state = self.cache_state and embeddings_on_disk
        if self.cache_state and not embeddings_on_disk:
            show_info("Set an embeddings save path to cache the automatic segmentation state.")

        # Set up progress bar and signals for using it within a threadworker.
        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker()
        def compute_image_embedding():

            # The computation runs synchronously on the main thread, so pump the Qt event loop on
            # every progress step; otherwise the napari progress bar only repaints once at the end
            # (it just jumps to 100%). This matters most for tiled embeddings (many tiles / slices).
            def pbar_init(total, description):
                if self.is_timeseries:  # A timeseries goes through the 3D compute path; relabel it.
                    description = description.replace("3D", "Timeseries")
                # Reset the counter to 0 so each phase starts fresh: the embeddings, then (when caching
                # is on) the automatic-segmentation state precompute reuse the same bar, and without a
                # reset the second phase inherits the first's completed count and sits stuck at full.
                pbar_signals.pbar_reset.emit()
                pbar_signals.pbar_total.emit(total)
                pbar_signals.pbar_description.emit(description)
                QtWidgets.QApplication.processEvents()

            def pbar_update(update):
                pbar_signals.pbar_update.emit(update)
                QtWidgets.QApplication.processEvents()

            state.initialize_predictor(
                image_data,
                model_type=self.model_type,
                save_path=save_path,
                ndim=ndim,
                device=self.device,
                checkpoint_path=self.custom_weights,
                tile_shape=tile_shape,
                halo=halo,
                prefer_decoder=True,
                precompute_autoseg_state=precompute_autoseg_state,
                pbar_init=pbar_init,
                pbar_update=pbar_update,
            )
            pbar_signals.pbar_stop.emit()

        compute_image_embedding()
        self._clear_autosegment_cache(state)
        self._update_model(state)
        # worker = compute_image_embedding()
        # worker.returned.connect(self._update_model)
        # worker.start()
        # return worker


class ClassificationEmbeddingWidget(EmbeddingWidget):
    """Embedding widget for the classification tools (pixel and object classification).

    The model selection mirrors the segmentation/tracking `EmbeddingWidget` exactly: the same single
    'Model:' dropdown with the SAM2 'Natural Images' and 'Microscopy' families (same names and config).
    Since classification operates directly on the image-encoder embeddings, it can additionally use
    models beyond SAM2. An opt-in 'Advanced Models' checkbox in the embedding settings swaps that one
    dropdown to the advanced families instead of adding a second dropdown. The advanced tier holds both
    the SAM1 families and the VFM (DINO / UNI) families (`_advanced_family_suffixes` and `_dino_families`).
    """

    # The classification tools have no automatic segmentation, so the state-caching option is hidden.
    supports_state_caching = False

    size_order = ["tiny", "small", "base", "large", "huge", "giant"]

    # The classifiers only run the image encoder and a lightweight classifier on top, so they stay
    # fast on the CPU and don't need the info popup.
    warn_on_cpu = False

    # Advanced SAM1 families: UI label -> model-name suffix on the SAM1 'vit_' prefix, resolved in
    # '_get_model_size_options'.
    _advanced_family_suffixes = {
        "Natural Images (SAM1)": "",
        "Light Microscopy (SAM1)": "_lm",
        "Electron Microscopy (SAM1)": "_em_organelles",
        "Medical Imaging (SAM1)": "_medical_imaging",
        "Histopathology (SAM1)": "_histopathology",
    }
    _advanced_size_map = {"t": "tiny", "b": "base", "l": "large", "h": "huge"}
    # Vision Foundation Model families beyond SAM: UI label -> the registry model_types in that family
    # (ordered by size). Sizes/names come from 'micro_sam.models.vfm.VFM_MODELS'/'VFM_SIZE_LABELS', not the
    # SAM1 naming scheme. DINOv2/v3 are natural-image (LVD-1689M) models; UNI/UNI2-h are histopathology.
    _dino_families = {
        "Natural Images (DINOv2)": ("vit_s_dinov2", "vit_b_dinov2", "vit_l_dinov2", "vit_g_dinov2"),
        "Natural Images (DINOv3)": ("vit_s_dinov3", "vit_b_dinov3", "vit_l_dinov3"),
        "Histopathology (UNI)": ("vit_uni", "vit_univ2"),
        "Natural Images (SAM3)": ("vit_sam3",),
    }
    # Older saved classifiers stored the primary SAM2 families under '(SAM2)' labels; map them to the
    # current names so loading such a classifier still restores the right family.
    _primary_family_aliases = {"Natural Images (SAM2)": "Natural Images", "Microscopy (SAM2)": "Microscopy"}

    def _all_advanced_families(self):
        # The advanced tier combines the SAM1 families and the VFM (DINO / UNI) families.
        return list(self._advanced_family_suffixes) + list(self._dino_families)

    def _is_dino_active(self):
        return getattr(self, "model_family", None) in self._dino_families

    def _advanced_active(self):
        # The single 'Model:' dropdown holds either the primary or the advanced families, so the
        # current family's membership is the source of truth (robust to the dropdown being blanked).
        return getattr(self, "model_family", None) in self._all_advanced_families()

    def _add_extra_model_settings(self, layout):
        # 'Advanced Models' swaps the single 'Model:' dropdown above between the primary (SAM2) and the
        # advanced (SAM1) families - one dropdown only, to avoid confusion. Added last in the settings.
        self._primary_families = list(self.supported_dropdown_maps.keys())
        # The default primary family ('Microscopy'), restored when advanced is switched back off.
        self._default_primary_family = self.model_family
        self.advanced = False
        self.advanced_checkbox = self._add_boolean_param(
            "advanced", self.advanced, title="Advanced Models",
            tooltip=get_tooltip("embedding", "advanced_model"),
        )
        self.advanced_checkbox.stateChanged.connect(self._on_advanced_toggled)
        layout.addWidget(self.advanced_checkbox)

        # The inherited dropdowns auto-bind their attribute by indexing the option list captured at
        # creation; we swap those lists (the families, and the per-family sizes), which makes the
        # captured index stale (wrong value, or out of range). Drop that auto-bind and let
        # '_update_model_type' (wired to 'currentTextChanged') sync the attribute from the text.
        for dropdown in (self.model_family_dropdown, self.model_size_dropdown):
            try:
                dropdown.currentIndexChanged.disconnect()
            except TypeError:
                pass

    def _set_family_choices(self, families, select=None):
        # Replace the 'Model:' dropdown items, select 'select' (default first), then resolve the model.
        if select not in families:
            select = families[0]
        self.model_family_dropdown.blockSignals(True)
        self.model_family_dropdown.clear()
        self.model_family_dropdown.addItems(families)
        self.model_family_dropdown.setCurrentText(select)
        self.model_family_dropdown.blockSignals(False)
        self.model_family = select
        self._update_model_type()

    def _on_advanced_toggled(self, state):
        advanced = self.advanced_checkbox.isChecked()
        if advanced:
            self._set_family_choices(self._all_advanced_families())
        else:  # Back to the SAM2 families, defaulting to 'Microscopy' rather than the first entry.
            self._set_family_choices(self._primary_families, select=self._default_primary_family)
        # Reflect the active tier in the 'Model:' dropdown tooltip.
        self.model_family_dropdown.setToolTip(
            get_tooltip("embedding", "model_family_advanced" if advanced else "model_family")
        )

    def _reset_inputs_to_defaults(self):
        # Switching off advanced restores the primary families; the base reset then selects the default.
        if getattr(self, "advanced_checkbox", None) is not None and self.advanced_checkbox.isChecked():
            self.advanced_checkbox.setChecked(False)
        super()._reset_inputs_to_defaults()

    def set_model_family_size(self, family, size):
        """Restore a saved (family, size): swap the dropdown to the matching tier, then select it."""
        family = self._primary_family_aliases.get(family, family)
        self.advanced_checkbox.setChecked(family in self._all_advanced_families())
        self.model_family_dropdown.setCurrentText(family)
        if size:
            self.model_size_dropdown.setCurrentText(size)

    def _validate_model_support(self):
        if super()._validate_model_support():
            return True
        # The vit-tiny backbone needs MobileSAM; warn (instead of crashing later) if it is selected
        # without MobileSAM installed. The model stays selectable - we just block this compute.
        from ..util import VIT_T_SUPPORT
        if not VIT_T_SUPPORT and (self.model_type or "").startswith("vit_t"):
            return _generate_message(
                "error",
                f"'{self.model_type}' (vit-tiny) requires MobileSAM. Install MobileSAM or pick another size.",
            )
        return False

    def _family_and_size_for_model(self, model_name):
        """Map a stored model name to its (family label, size label) for this widget's dropdowns."""
        full_size_map = {"t": "tiny", "s": "small", "b": "base", "l": "large", "h": "huge"}
        for family, models in self._dino_families.items():  # VFM families (DINO / UNI).
            if model_name in models:
                from ..models.vfm import VFM_SIZE_LABELS
                return family, VFM_SIZE_LABELS.get(model_name)
        if model_name.startswith("hvit_"):  # SAM2 (primary families).
            size = full_size_map.get(model_name[5])
            family = "Microscopy" if model_name.endswith("_cells") else "Natural Images"
        else:  # SAM1 (advanced families): 'vit_<size><suffix>'.
            size = full_size_map.get(model_name[4])
            suffix = model_name[5:]
            family = {v: k for k, v in self._advanced_family_suffixes.items()}.get(suffix, "Natural Images (SAM1)")
        return family, size

    def _apply_loaded_model_selection(self, model_name):
        # Set the family (primary or advanced) and size dropdowns to match the loaded embeddings.
        family, size = self._family_and_size_for_model(model_name)
        self.set_model_family_size(family, size)

    def _get_model_size_options(self):
        # Primary (SAM2) sizes come from the inherited logic; advanced families resolve to SAM1/DINO names.
        if not self._advanced_active():
            return super()._get_model_size_options()
        if self._is_dino_active():
            from ..models.vfm import VFM_SIZE_LABELS
            models = self._dino_families[self.model_family]
            self.model_size_mapping = {VFM_SIZE_LABELS[m]: m for m in models}
            self.model_size_options = sorted(self.model_size_mapping.keys(), key=self.size_order.index)
            return
        from ..v1.util import get_model_names
        suffix = self._advanced_family_suffixes[self.model_family]
        available = {m for m in get_model_names() if not m.endswith("decoder")}
        self.model_size_mapping = {}
        for key, label in self._advanced_size_map.items():
            name = f"vit_{key}{suffix}"
            if name in available:
                self.model_size_mapping[label] = name
        self.model_size_options = sorted(self.model_size_mapping.keys(), key=self.size_order.index)

    def _update_model_type(self):
        # Sync the family from the dropdown first: the inherited auto-bind closure captures the original
        # (primary) option list, so after the dropdown is swapped to the advanced families it can set a
        # stale value; re-reading the current text here is authoritative and decides the branch below.
        self.model_family = self.model_family_dropdown.currentText() or self.model_family
        # Primary mode defers to the inherited SAM2 logic; advanced mode rebuilds the size dropdown for
        # the current SAM1 family and resolves its model name.
        if not self._advanced_active():
            return super()._update_model_type()
        current_selection = self.model_size_dropdown.currentText()
        self._get_model_size_options()

        self.model_size_dropdown.blockSignals(True)
        self.model_size_dropdown.clear()
        self.model_size_dropdown.addItems(self.model_size_options)
        if current_selection in self.model_size_options:
            self.model_size = current_selection
        elif self.model_size_options:
            self.model_size = self.model_size_options[0]
        self.model_type = self.model_size_mapping.get(self.model_size)
        self.model_size_dropdown.setCurrentText(self.model_size)
        self.model_size_dropdown.update()
        self.model_size_dropdown.blockSignals(False)

    def _validate_model_type_and_custom_weights(self):
        # DINO families always resolve from the registry. DINOv3 weights load from HuggingFace using the
        # user's own HF access (huggingface-cli login / HF_TOKEN), not the custom-weights field.
        if self._is_dino_active():
            self._get_model_size_options()
            self.model_type = self.model_size_mapping.get(self.model_size, self.model_type)
            return
        # Advanced SAM1 mode (without custom weights): resolve the SAM1 model name from family + size.
        if self._advanced_active() and not self.custom_weights:
            self._get_model_size_options()
            self.model_type = self.model_size_mapping.get(self.model_size, self.model_type)
            return
        # Primary mode, or custom weights: the inherited logic resolves the type and blanks the dropdown.
        super()._validate_model_type_and_custom_weights()


#
# Functionality and widget for nd segmentation.
#


def _division_frame_for_track(point_layer, track_id):
    """Frame at which 'track_id' is annotated to divide, or None.

    A division is a point tagged with the 'division' track-state for this track; the earliest such
    frame is the mother track's last frame (propagation is bounded above there).
    """
    props = point_layer.properties
    if not len(point_layer.data) or "state" not in props or "track_id" not in props:
        return None
    states = np.asarray(props["state"])
    track_ids_prop = np.asarray(props["track_id"])
    z_all = np.round(point_layer.data[:, 0]).astype(int)
    div_mask = (states == "division") & (track_ids_prop == str(track_id))
    return int(z_all[div_mask].min()) if np.any(div_mask) else None


def _mother_division_frame(point_layer, lineage, track_id):
    """Division frame of 'track_id's mother, or None if it is not a daughter of a dividing track.

    A daughter does not exist before its mother divides, so its mask must start the frame after.
    """
    for mother, daughters in lineage.items():
        if track_id in daughters:
            return _division_frame_for_track(point_layer, mother)
    return None


def _update_lineage(viewer, mother=None):
    """Record a division for 'mother' by seeding two daughter track ids and refreshing the menus.

    Args:
        viewer: The napari viewer.
        mother: The track id that divides. Defaults to the current track id.
    """
    state = AnnotatorState()
    tracking_widget = state.annotator._tracking_widget

    if mother is None:
        mother = state.current_track_id
    # Only seed daughters once per division: skip unknown tracks or tracks that already divided.
    if mother not in state.lineage or len(state.lineage[mother]) > 0:
        return

    daughter1 = max(state.lineage.keys()) + 1
    daughter2 = daughter1 + 1
    state.lineage[mother] = [daughter1, daughter2]
    state.lineage[daughter1] = []
    state.lineage[daughter2] = []

    # Update the choices in the track_id menu so that it contains the new track ids.
    # (index 2: prompt, track_state, track_id).
    track_ids = list(map(str, state.lineage.keys()))
    tracking_widget[2].choices = track_ids

    viewer.layers["point_prompts"].property_choices["track_id"] = list(track_ids)
    viewer.layers["prompts"].property_choices["track_id"] = list(track_ids)


class UnifiedSegmentWidget(_WidgetBase):
    """Unified widget for 3D segmentation (per-slice and volumetric).
    """

    def __init__(self, viewer, tracking=False, parent=None):
        """Initialize the unified segmentation widget.

        Args:
            viewer: The napari viewer.
            tracking: Whether this is used for tracking (vs volumetric).
            parent: Parent Qt widget.
        """
        super().__init__(parent=parent)
        self._viewer = viewer
        self.tracking = tracking

        # Initialize volume mode state
        self.apply_to_volume = False

        # Create the widget UI
        self._create_widget()

    def _create_widget(self):
        """Create the widget UI elements."""
        # 1. Add volume mode checkbox
        volume_title = (
            "Apply to All Frames" if self.tracking else "Apply to Volume"
        )
        self.volume_checkbox = self._add_boolean_param(
            "apply_to_volume",
            self.apply_to_volume,
            title=volume_title,
            tooltip=get_tooltip("unified_segment", "apply_to_volume"),
        )
        self.volume_checkbox.stateChanged.connect(self._on_volume_mode_changed)
        self.layout().addWidget(self.volume_checkbox)

        # 2. Add batched checkbox (initially hidden, shown for SAM2 when not tiled)
        self.batched = False
        self.batched_checkbox = self._add_boolean_param(
            "batched",
            self.batched,
            title="Batched",
            tooltip=get_tooltip("unified_segment", "batched"),
        )
        self.batched_checkbox.setVisible(False)  # Initially hidden
        self.layout().addWidget(self.batched_checkbox)

        # 3. Create settings panel (initially hidden)
        self.settings = self._create_settings()
        self.settings.setVisible(False)
        self.layout().addWidget(self.settings)

        # 4. Add run button
        self.run_button = QtWidgets.QPushButton(self._get_button_text())
        self.run_button.setToolTip(get_tooltip("unified_segment", "segment_button"))
        self.run_button.clicked.connect(self.__call__)
        self.layout().addWidget(self.run_button)

        # 5. Initialize batched checkbox visibility based on SAM version
        self._update_batched_visibility()

    def _create_settings(self):
        """Create the collapsible settings panel.
        """
        setting_values = QtWidgets.QWidget()
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # Projection mode dropdown
        self.projection = "single_point"
        self.projection_dropdown, layout = self._add_choice_param(
            "projection",
            self.projection,
            PROJECTION_MODES,
            tooltip=get_tooltip("segmentnd", "projection_dropdown"),
        )
        setting_values.layout().addLayout(layout)

        # IOU threshold
        self.iou_threshold = 0.5
        self.iou_threshold_param, layout = self._add_float_param(
            "iou_threshold",
            self.iou_threshold,
            tooltip=get_tooltip("segmentnd", "iou_threshold"),
        )
        setting_values.layout().addLayout(layout)

        # Box extension
        self.box_extension = 0.05
        self.box_extension_param, layout = self._add_float_param(
            "box_extension",
            self.box_extension,
            tooltip=get_tooltip("segmentnd", "box_extension"),
        )
        setting_values.layout().addLayout(layout)

        # SAM2 volume-mode propagation controls. The engine only holds the values; the visible
        # InteractiveSegmentationWidget owns the user-facing controls and writes these attributes.
        # 'early_stop_patience': stop after this many consecutive empty slices (0 -> disabled).
        # 'z_range': inclusive (z_min, z_max) hard bound on propagation, or 'None' for the full volume.
        self.early_stop_patience = 2
        self.z_range = None

        # Motion smoothing (tracking only)
        if self.tracking:
            self.motion_smoothing = 0.5
            self.motion_smoothing_param, layout = self._add_float_param(
                "motion_smoothing",
                self.motion_smoothing,
                tooltip=get_tooltip("segmentnd", "motion_smoothing"),
            )
            setting_values.layout().addLayout(layout)

        settings = _make_collapsible(
            setting_values, title="Segmentation Settings", tooltip=get_tooltip("unified_segment", "settings"),
        )
        return settings

    def _on_volume_mode_changed(self, state):
        """Handle volume mode checkbox state change.

        Args:
            state: Qt checkbox state (0=unchecked, 2=checked).
        """
        is_checked = bool(state)

        # Show/hide settings panel
        self.settings.setVisible(is_checked)

        # Update button text
        self.run_button.setText(self._get_button_text())

        # Update batched checkbox visibility
        self._update_batched_visibility()

    def _update_batched_visibility(self):
        """Show/hide batched checkbox based on SAM version and tiling."""
        state = AnnotatorState()
        is_sam2 = state.is_sam2 if state.is_sam2 is not None else False

        # Show batched for SAM2 models when the embeddings are not tiled (batched prompting is
        # unsupported with tiling). Available in both slice/frame and volume/all-frames mode.
        should_show = is_sam2 and not _embeddings_are_tiled(state)
        if not should_show and getattr(self, "batched", False):
            self.batched_checkbox.setChecked(False)
        self.batched_checkbox.setVisible(should_show)

    def _get_button_text(self):
        """Get dynamic button text based on current mode."""
        if self.tracking:
            return (
                "Segment All Frames [S]"
                if self.apply_to_volume
                else "Segment Frame [S]"
            )
        else:
            return (
                "Segment Volume [S]"
                if self.apply_to_volume
                else "Segment Slice [S]"
            )

    def __call__(self, viewer=None):
        """Execute segmentation based on current mode.

        Args:
            viewer: Optional napari viewer (for keybinding compatibility).
        """
        # Validation
        if _validate_embeddings(self._viewer):
            return None
        if _validate_layers(self._viewer):
            return None

        # Route to appropriate implementation
        if self.apply_to_volume:
            if self.tracking:
                return self._run_tracking()
            else:
                return self._run_volumetric_segmentation()
        else:
            if self.tracking:
                return self._run_frame_segmentation()
            else:
                return self._run_slice_segmentation()

    def _run_slice_segmentation(self):
        """Execute per-slice segmentation.
        """
        shape = self._viewer.layers["current_object"].data.shape[1:]

        position_world = self._viewer.dims.point
        position = self._viewer.layers["point_prompts"].world_to_data(
            position_world
        )
        z = int(position[0])

        prompt_layer = self._viewer.layers["prompts"]
        scribble_points, scribble_labels = vutil.scribble_layer_to_prompts(
            prompt_layer, image_shape=shape, i=z
        )
        have_scribbles = len(scribble_points) > 0
        point_prompts = vutil.point_layer_to_prompts(
            self._viewer.layers["point_prompts"], z, with_stop_annotation=not have_scribbles
        )
        # this is a stop prompt, we do nothing
        if not point_prompts:
            return

        boxes, masks = vutil.shape_layer_to_prompts(
            prompt_layer, shape, i=z
        )
        points, labels = vutil.merge_point_prompts(
            point_prompts, (scribble_points, scribble_labels)
        )
        if have_scribbles and not boxes and not np.any(labels == 1):
            return _generate_message(
                "error",
                "A negative scribble needs a positive point, positive scribble, box or mask prompt.",
            )

        state = AnnotatorState()
        batched = _batched_disabled_when_tiled(state, bool(self.batched))
        if have_scribbles and batched:
            show_info("Batched segmentation is not supported with scribble prompts. Running single-object.")
            batched = False

        if state.is_sam2:
            # Use the segment_slice method for SAM2.
            boxes = [box[[1, 0, 3, 2]] for box in boxes]
            if batched:
                seg = self._segment_slice_batched(z, points, labels, boxes, masks, shape)
            else:
                seg = state.interactive_segmenter.segment_slice(
                    frame_idx=z,
                    points=points[:, ::-1].copy(),
                    labels=labels,
                    boxes=boxes,
                    masks=masks,
                )
        else:
            seg = vutil.prompt_segmentation(
                state.predictor,
                points,
                labels,
                boxes,
                masks,
                shape,
                multiple_box_prompts=False,
                image_embeddings=state.image_embeddings,
                batched=batched,
                i=z,
            )

        # No prompts were given or prompts were invalid, skip segmentation.
        if seg is None:
            print(
                "You either haven't provided any prompts or invalid prompts. The segmentation will be skipped."
            )
            return

        self._viewer.layers["current_object"].data[z] = seg
        self._viewer.layers["current_object"].refresh()

    def _segment_slice_batched(self, z, points, labels, boxes, masks, shape):
        """Batched multi-object segmentation for a single slice with SAM2.

        Mirrors the 2d batched convention: one object per positive point (each combined with the
        shared negative points) and one object per box. A box from a polygon/ellipse (its entry in
        `masks` is not None) also carries its soft mask cue. The boxes are expected in the reordered
        layout used by `segment_slice`.
        """
        state = AnnotatorState()
        points = np.zeros((0, 2)) if points is None else np.asarray(points)
        labels = np.zeros((0,), dtype=int) if labels is None else np.asarray(labels)
        positive_points = points[labels == 1]
        negative_points = points[labels != 1]
        n_neg = len(negative_points)

        seg = np.zeros(tuple(shape), dtype="uint32")
        object_id = 0

        # One object per positive point, each combined with the shared negative points.
        for pos in positive_points:
            pts = np.concatenate([pos[None], negative_points], axis=0) if n_neg else pos[None]
            lbs = np.concatenate([[1], np.zeros(n_neg, dtype=int)]) if n_neg else np.array([1])
            object_id += 1
            mask = state.interactive_segmenter.segment_slice(
                frame_idx=z, points=pts[:, ::-1].copy(), labels=lbs, object_id=object_id,
            )
            if mask is not None:
                seg[mask > 0] = object_id

        # One object per box (with its mask cue if it is a polygon/ellipse), each combined with the
        # shared negative points.
        for bidx, box in enumerate(boxes):
            neg = negative_points[:, ::-1].copy() if n_neg else None
            neg_labels = np.zeros(n_neg, dtype=int) if n_neg else None
            object_id += 1
            mask = state.interactive_segmenter.segment_slice(
                frame_idx=z, points=neg, labels=neg_labels, boxes=[box],
                masks=[masks[bidx]] if masks is not None else None, object_id=object_id,
            )
            if mask is not None:
                seg[mask > 0] = object_id

        return seg

    def _segment_track_on_frame(self, state, t, track_id, shape):
        """Segment a single track's object on frame 't'. Returns the binary mask or None."""
        point_prompts = vutil.point_layer_to_prompts(
            self._viewer.layers["point_prompts"], i=t, track_id=track_id,
        )
        # A single negative point is a stop prompt: nothing to segment for this track here.
        if not point_prompts:
            return None

        boxes, masks = vutil.shape_layer_to_prompts(
            self._viewer.layers["prompts"], shape, i=t, track_id=track_id,
        )
        points, labels = point_prompts

        # The tracking annotator is SAM2-only: segment the frame via the video predictor.
        # Points are reordered to (x, y) and boxes to (x0, y0, x1, y1), matching the per-slice path.
        sam2_boxes = [box[[1, 0, 3, 2]] for box in boxes]
        seg = state.interactive_segmenter.segment_slice(
            frame_idx=t,
            points=points[:, ::-1].copy() if len(points) else points,
            labels=labels,
            boxes=sam2_boxes,
            masks=masks,
        )
        return None if seg is None else (seg.squeeze() == 1)

    def _run_frame_segmentation(self):
        """Execute per-frame segmentation for the current track (tracking mode)."""
        state = AnnotatorState()
        shape = state.image_shape[1:]
        t = int(self._viewer.dims.point[0])

        track_id = state.current_track_id
        new_mask = self._segment_track_on_frame(state, t, track_id, shape)
        if new_mask is None:
            print(
                "You either haven't provided any prompts or invalid prompts. The segmentation will be skipped."
            )
            return

        # Clear the old segmentation for this track id, then set the new one.
        layer = self._viewer.layers["current_object"]
        layer.data[t][layer.data[t] == track_id] = 0
        layer.data[t][new_mask] = track_id
        layer.refresh()

    def _run_volumetric_segmentation(self):
        """Execute volumetric segmentation.
        """
        pbar, pbar_signals = _create_pbar_for_threadworker()

        def emit_progress(update):
            # The run is synchronous, so pump the Qt event loop to repaint the bar live instead of
            # only updating the terminal tqdm.
            pbar_signals.pbar_update.emit(update)
            QtWidgets.QApplication.processEvents()

        # @thread_worker
        def volumetric_segmentation_impl():
            state = AnnotatorState()
            shape = state.image_shape

            pbar_signals.pbar_total.emit(shape[0])
            pbar_signals.pbar_description.emit("Segment object")

            if state.is_sam2:
                # Prepare the prompts
                point_prompts = self._viewer.layers["point_prompts"]
                box_prompts = self._viewer.layers["prompts"]
                z_values_points = np.round(point_prompts.data[:, 0])
                z_values_scribbles = vutil.get_scribble_slices(box_prompts)
                have_scribbles = len(z_values_scribbles) > 0
                z_values_boxes = np.round(
                    np.asarray([
                        shape[0, 0]
                        for shape, shape_type in zip(box_prompts.data, box_prompts.shape_type)
                        if shape_type not in vutil.SCRIBBLE_SHAPE_TYPES
                    ])
                ).astype("int")

                # Whether the user decide to provide batched prompts for multi-object segmentation.
                is_batched = _batched_disabled_when_tiled(state, bool(self.batched))
                if have_scribbles and is_batched:
                    show_info("Batched segmentation is not supported with scribble prompts. Running single-object.")
                    is_batched = False

                # A scribble is expanded into several points. Rebuild the persistent video-predictor
                # state so deleting or relabelling a stroke cannot leave stale samples behind.
                if have_scribbles:
                    state.interactive_segmenter.reset_predictor()

                # Check batched mode validity and show warning if needed
                if is_batched and not state.is_sam2:
                    show_info(
                        "Batched segmentation is only supported with SAM2 models (hvit_*). "
                        "Running in standard mode."
                    )
                    is_batched = False

                # Object-id counter for batched multi-object segmentation: each box and each point
                # becomes its own object, so boxes and points draw distinct ids from one shared
                # counter. In non-batched mode every prompt feeds a single object (id 1).
                object_id = 0

                # Add box prompts first: SAM2 requires a box before any point on the same object/frame,
                # so adding boxes ahead of points lets a box and its correction points combine.
                # Rectangles are box prompts; polygons/ellipses instead carry a filled mask prompt.
                shape_yx = state.image_shape[-2:]
                for curr_z in np.unique(z_values_boxes):
                    boxes, shape_masks = vutil.shape_layer_to_prompts(layer=box_prompts, shape=shape_yx, i=curr_z)
                    if not boxes:
                        continue
                    rect_boxes = [b for b, m in zip(boxes, shape_masks) if m is None]
                    poly_masks = [m for m in shape_masks if m is not None]
                    if rect_boxes:
                        box_ids = list(range(object_id + 1, object_id + 1 + len(rect_boxes))) if is_batched else None
                        object_id += len(rect_boxes)
                        state.interactive_segmenter.add_box_prompts(
                            frame_ids=curr_z, boxes=rect_boxes, object_id=box_ids
                        )
                    if poly_masks:
                        mask_ids = list(range(object_id + 1, object_id + 1 + len(poly_masks))) if is_batched else None
                        object_id += len(poly_masks)
                        state.interactive_segmenter.add_mask_prompts(
                            frame_ids=curr_z, masks=poly_masks, object_id=mask_ids
                        )

                # Then add the point prompts. Iterate unique frames so each frame's points are added
                # together; the segmenter skips points already pushed, so re-runs only add new ones.
                point_slices = np.unique(np.concatenate([z_values_points, z_values_scribbles])).astype("int")
                merged_prompts = []
                for curr_z in point_slices:
                    scribble_points, scribble_labels = vutil.scribble_layer_to_prompts(
                        box_prompts, image_shape=shape_yx, i=curr_z
                    )
                    # A slice whose only prompt is a single negative point is a 'stop' annotation; skip it.
                    prompts = vutil.point_layer_to_prompts(
                        layer=point_prompts,
                        i=curr_z,
                        with_stop_annotation=len(scribble_points) == 0,
                    )
                    if prompts is None:
                        continue
                    points, labels = vutil.merge_point_prompts(
                        prompts, (scribble_points, scribble_labels)
                    )
                    merged_prompts.append((curr_z, points, labels))

                have_positive_points = any(np.any(labels == 1) for _, _, labels in merged_prompts)
                have_shape_cue = len(z_values_boxes) > 0
                if have_scribbles and not have_positive_points and not have_shape_cue:
                    pbar_signals.pbar_stop.emit()
                    _generate_message(
                        "error",
                        "A negative scribble needs a positive point, positive scribble, box or mask prompt.",
                    )
                    return None

                for curr_z, points, labels in merged_prompts:
                    if is_batched:
                        point_ids = list(range(object_id + 1, object_id + 1 + len(points)))
                        object_id += len(points)
                    else:
                        point_ids = None
                    state.interactive_segmenter.add_point_prompts(
                        frame_ids=curr_z,
                        points=np.asarray(points),
                        point_labels=np.asarray(labels),
                        object_id=point_ids,
                    )

                # Propagate the prompts throughout the volume and combine the propagated segmentations.
                # Report each slice propagation step.
                # A patience of 0 disables early stopping (propagate through the whole volume).
                early_stop_patience = self.early_stop_patience if self.early_stop_patience > 0 else None
                # Tiled segmenters count one step per slice in each tile activated above.
                n_propagation_steps = state.interactive_segmenter.get_progress_total(self.z_range)
                pbar_signals.pbar_total.emit(n_propagation_steps)
                pbar_signals.pbar_description.emit("Propagate in volume")
                seg = state.interactive_segmenter.predict(
                    update_progress=emit_progress,
                    early_stop_patience=early_stop_patience, z_range=self.z_range,
                )

            else:
                # Step 1: Segment all slices with prompts.
                seg, slices, stop_lower, stop_upper = (
                    vutil.segment_slices_with_prompts(
                        state.predictor,
                        self._viewer.layers["point_prompts"],
                        self._viewer.layers["prompts"],
                        state.image_embeddings,
                        shape,
                        update_progress=emit_progress,
                    )
                )
                if len(slices) == 0:
                    pbar_signals.pbar_stop.emit()
                    _generate_message(
                        "error",
                        "No valid slice prompts remain. Add a positive point, scribble, box or mask prompt.",
                    )
                    return None

                # Step 2: Segment the rest of the volume based on projecting prompts.
                seg, (z_min, z_max) = segment_mask_in_volume(
                    seg,
                    state.predictor,
                    state.image_embeddings,
                    slices,
                    stop_lower,
                    stop_upper,
                    iou_threshold=self.iou_threshold,
                    projection=self.projection,
                    box_extension=self.box_extension,
                    update_progress=emit_progress,
                )

                state.z_range = (z_min, z_max)

            pbar_signals.pbar_stop.emit()

            return seg

        def update_segmentation(seg):
            self._viewer.layers["current_object"].data = seg
            self._viewer.layers["current_object"].refresh()

        seg = volumetric_segmentation_impl()
        if seg is None:
            return None
        self._viewer.layers["current_object"].data = seg
        self._viewer.layers["current_object"].refresh()
        # worker = volumetric_segmentation_impl()
        # worker.returned.connect(update_segmentation)
        # worker.start()
        # return worker

    def _run_tracking(self):
        """Execute interactive tracking by propagating the current track's prompts across frames.

        Design boundary: SAM2 only does per-object mask propagation across frames. It has no concept
        of a division / lineage event and never signals one. Everything division-related here is
        mechanistic orchestration on top of the model (reading the user's 'division' annotation,
        bounding propagation, seeding daughters, recording lineage edges) - none of it is a SAM2
        capability. Genuine automatic division detection lives in the automatic-tracking path via
        trackastra, which predicts lineages from per-frame segmentations; it is not SAM2.
        """
        state = AnnotatorState()
        pbar, pbar_signals = _create_pbar_for_threadworker()

        def emit_progress(update):
            # Synchronous run: pump the Qt event loop so the bar repaints live, not only in the terminal.
            pbar_signals.pbar_update.emit(update)
            QtWidgets.QApplication.processEvents()

        def propagate_track(track_id, division_frame):
            # Propagate a single track's prompts forward in time with the SAM2 predictor. Tracking
            # is forward-only: it starts at the first prompted frame and never runs to earlier
            # frames. A frame whose only prompt for this track is a single negative point is a
            # 'stop' annotation; a stop on the highest annotated frame bounds propagation above.
            shape = state.image_shape
            point_layer = self._viewer.layers["point_prompts"]
            box_layer = self._viewer.layers["prompts"]

            # Reset so a re-run does not accumulate prompts from a previous propagation.
            state.interactive_segmenter.reset_predictor()
            pbar_signals.pbar_description.emit(f"Track object {track_id}")

            # Add the point prompts for this track, one frame at a time, recording the prompted
            # frames and the stop annotations.
            prompted_frames, stop_frames = [], []
            z_points = (
                np.unique(np.round(point_layer.data[:, 0]).astype(int))
                if len(point_layer.data) else np.zeros(0, dtype=int)
            )
            for t in z_points:
                # Exclude division markers: they signal a lineage event and bound propagation
                # (see below), but must not be fed to SAM2 as conditioning prompts - doing so
                # adds a second conditioning frame that corrupts the mother track's propagation.
                prompts = vutil.point_layer_to_prompts(
                    point_layer, i=int(t), track_id=track_id, exclude_states=("division",)
                )
                if prompts is None:  # Single negative point: a stop annotation for this track.
                    stop_frames.append(int(t))
                    continue
                points, labels = prompts
                if len(points) == 0:  # This track has no point prompts on this frame.
                    continue
                for point, label in zip(points, labels):
                    state.interactive_segmenter.add_point_prompts(
                        frame_ids=int(t), points=np.array([point]), point_labels=np.array([label]),
                    )
                prompted_frames.append(int(t))

            # Add the box prompts for this track.
            z_boxes = (
                np.unique(np.concatenate([box[:1, 0] for box in box_layer.data]).round().astype(int))
                if box_layer.data else np.zeros(0, dtype=int)
            )
            for t in z_boxes:
                boxes, _ = vutil.shape_layer_to_prompts(box_layer, shape=shape, i=int(t), track_id=track_id)
                for box in boxes:
                    state.interactive_segmenter.add_box_prompts(frame_ids=int(t), boxes=[box])
                if boxes:
                    prompted_frames.append(int(t))

            if not prompted_frames:
                return None

            # Forward-only propagation: start at the first prompted frame and never go to earlier
            # frames. A stop annotation on the highest annotated frame bounds propagation above
            # ('predict' enforces the z-range); a division bounds the mother at the division frame -
            # its last frame (only reached when the mother is not segmented yet, see 'tracking_impl').
            annotated = sorted(set(prompted_frames) | set(stop_frames))
            stop_upper = annotated[-1] in stop_frames
            z_lo = min(prompted_frames)
            z_hi = max(prompted_frames) if stop_upper else shape[0] - 1
            if division_frame is not None:  # The division frame is the mother's last frame.
                z_hi = min(z_hi, division_frame)

            if z_hi < z_lo:  # The division precedes the track's first frame: nothing to segment.
                return None

            pbar_signals.pbar_total.emit(
                state.interactive_segmenter.get_progress_total((z_lo, z_hi))
            )
            seg = state.interactive_segmenter.predict(
                update_progress=emit_progress,
                early_stop_patience=None, z_range=(z_lo, z_hi),
            )
            return seg

        def tracking_impl():
            # Propagate the current track. Its propagated mask is labelled with its track id.
            track_ids = [state.current_track_id]
            point_layer = self._viewer.layers["point_prompts"]
            seg_layer = self._viewer.layers["current_object"]
            results = {}
            for track_id in track_ids:
                division_frame = _division_frame_for_track(point_layer, track_id)
                # A division is a cleanup, not a (re)segmentation: when the mother is already
                # segmented, its frames up to and including the division are correct (tracking is
                # forward-only), so we just erase it AFTER the division frame and seed the daughters
                # - no SAM2 re-run. Only when the mother is not segmented yet do we fall back to a
                # bounded propagation so it still gets created up to the division frame.
                segmented = division_frame is not None and bool(
                    np.any(seg_layer.data[:division_frame + 1] == track_id)
                )
                if segmented:
                    results[track_id] = {"truncate_from": division_frame + 1}
                    _update_lineage(self._viewer, mother=track_id)
                    continue

                seg = propagate_track(track_id, division_frame)
                if seg is not None:
                    res = {"seg": seg}
                    # A daughter must not occupy its mother's frames: clip its mask to start the
                    # frame after the mother's division, no matter where the user prompted it.
                    mother_division = _mother_division_frame(point_layer, state.lineage, track_id)
                    if mother_division is not None:
                        res["min_frame"] = mother_division + 1
                    results[track_id] = res
                # A division seeds two daughter tracks, so the user can continue from the division.
                if division_frame is not None:
                    _update_lineage(self._viewer, mother=track_id)
            pbar_signals.pbar_stop.emit()
            return results

        def update_segmentation(results):
            if not results:
                print("No prompts were given for the track(s). The tracking will be skipped.")
                return

            layer = self._viewer.layers["current_object"]
            for track_id, res in results.items():
                if "truncate_from" in res:
                    # Division cleanup: erase the mother after the division frame, leaving its
                    # frames up to and including the division (already correct) untouched.
                    f = res["truncate_from"]
                    layer.data[f:][layer.data[f:] == track_id] = 0
                else:
                    # Clear the old mask for this track, then set the propagated one. A daughter is
                    # clipped to start the frame after its mother's division (it did not exist yet).
                    seg = res["seg"]
                    min_frame = res.get("min_frame")
                    if min_frame is not None:
                        seg = seg.copy()
                        seg[:min_frame] = 0
                    layer.data[layer.data == track_id] = 0
                    layer.data[seg == 1] = track_id
            layer.refresh()

        results = tracking_impl()
        update_segmentation(results)


#
# The functionality and widgets for automatic segmentation.
#


# Messy automatic-segmentation state handling, would be good to refactor this properly at some point.
def _handle_autoseg_state(state, i, pbar_init, pbar_update):
    if state.automatic_segmenter is None:
        is_tiled = state.image_embeddings["input_size"] is None
        state.automatic_segmenter = instance_segmentation.get_instance_segmentation_generator(
            state.predictor, is_tiled=is_tiled, decoder=state.decoder
        )

    shape = state.image_shape

    # Further optimization: refactor parts of this so that we can also use it in the automatic 3d segmentation fucnction
    # For 3D we store the amg state in a dict and check if it is computed already.
    if state.autoseg_state is not None:
        assert i is not None
        if i in state.autoseg_state:
            segmentation_state_i = state.autoseg_state[i]
            state.automatic_segmenter.set_state(segmentation_state_i)

        else:
            dummy_image = np.zeros(shape[-2:], dtype="uint8")
            state.automatic_segmenter.initialize(
                dummy_image,
                image_embeddings=state.image_embeddings,
                i=i,
                verbose=pbar_init is not None,
                pbar_init=pbar_init,
                pbar_update=pbar_update,
            )
            segmentation_state_i = state.automatic_segmenter.get_state()
            state.autoseg_state[i] = segmentation_state_i

            cache_folder = state.autoseg_state.get("cache_folder", None)
            if cache_folder is not None:
                cache_path = os.path.join(cache_folder, f"state-{i}.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(segmentation_state_i, f)

            cache_path = state.autoseg_state.get("cache_path", None)
            if cache_path is not None:
                save_key = f"state-{i}"
                with h5py.File(cache_path, "a") as f:
                    g = f.create_group(save_key)
                    g.create_dataset(
                        "foreground",
                        data=segmentation_state_i["foreground"],
                        compression="gzip",
                    )
                    g.create_dataset(
                        "boundary_distances",
                        data=segmentation_state_i["boundary_distances"],
                        compression="gzip",
                    )
                    g.create_dataset(
                        "center_distances",
                        data=segmentation_state_i["center_distances"],
                        compression="gzip",
                    )

    # Otherwise (2d segmentation) we just check if the amg is initialized or not.
    elif not state.automatic_segmenter.is_initialized:
        assert i is None
        # We don't need to pass the actual image data here, since the embeddings are passed.
        # (The image data is only used by the amg to compute image embeddings, so not needed here.)
        dummy_image = np.zeros(shape, dtype="uint8")
        state.automatic_segmenter.initialize(
            dummy_image,
            image_embeddings=state.image_embeddings,
            verbose=pbar_init is not None,
            pbar_init=pbar_init,
            pbar_update=pbar_update,
        )


def _instance_segmentation_impl(
    min_object_size, i=None, pbar_init=None, pbar_update=None, **kwargs
):
    state = AnnotatorState()
    _handle_autoseg_state(state, i, pbar_init, pbar_update)
    seg = state.automatic_segmenter.generate(**kwargs)
    assert isinstance(seg, np.ndarray)
    return seg


class InteractiveSegmentationWidget(_WidgetBase):
    """Interactive segmentation widget combining the prompt menu, segmentation and clearing.

    This widget adapts to the data dimensionality and works with both SAM and SAM2 models.
    For 3d data it exposes a single 'Apply to Volume' checkbox that governs both segmentation and
    clearing (current slice vs. the whole volume / all slices), plus a batched toggle (SAM2 volume
    mode) for multi-object segmentation.

    Args:
        viewer: The napari viewer.
        ndim: The number of spatial dimensions of the data (2 or 3).
        prompt_widget: The point prompt label menu created by `create_prompt_menu`.
        parent: The parent Qt widget.
    """

    def __init__(self, viewer, ndim, prompt_widget, parent=None):
        super().__init__(parent=parent)
        self._viewer = viewer
        self._ndim = ndim
        self._prompt_widget = prompt_widget
        self.batched = False
        self.apply_to_volume = False
        self._segment_widget = None
        self._propagation_settings = None
        self._create_widget()

    def _create_widget(self):
        # Prompt label menu.
        self.layout().addWidget(self._prompt_widget.native)

        self.clear_button = QtWidgets.QPushButton("Clear Annotations [Shift + C]")
        self.clear_button.setToolTip(get_tooltip("unified_segment", "clear_button"))
        self.clear_button.clicked.connect(lambda: self.clear())

        self.segment_button = QtWidgets.QPushButton("Segment Object [S]")
        self.segment_button.setToolTip(get_tooltip("unified_segment", "segment_button"))
        self.segment_button.clicked.connect(lambda: self.segment())

        # Segmentation controls.
        if self._ndim == 2:
            # 2d: a 'batched' toggle above the side-by-side segment and clear buttons.
            self.batched_checkbox = self._add_boolean_param(
                "batched",
                self.batched,
                title="Batched",
                tooltip=get_tooltip("unified_segment", "batched"),
            )
            self.layout().addWidget(self.batched_checkbox)
        else:
            # 3d: use the volumetric segmentation widget purely as the segmentation engine.
            # Its own controls are not shown; the interactive widget owns the 'Apply to Volume'
            # and 'Batched' checkboxes and drives the engine's attributes. The 'Apply to Volume'
            # checkbox governs both segmentation (slice vs. volume) and clearing (current slice
            # vs. all slices). It is kept hidden but parented so it does not float as a window.
            self._segment_widget = UnifiedSegmentWidget(self._viewer, tracking=False)
            self._segment_widget.setVisible(False)
            self.layout().addWidget(self._segment_widget)

            # Batched multi-object segmentation. Works for both a single slice and the whole
            # volume, so it is always available.
            self.batched_checkbox = self._add_boolean_param(
                "batched",
                self.batched,
                title="Batched",
                tooltip=get_tooltip("unified_segment", "batched"),
            )
            self.batched_checkbox.stateChanged.connect(self._on_batched_changed)

            self.apply_to_volume_checkbox = self._add_boolean_param(
                "apply_to_volume",
                self.apply_to_volume,
                title="Apply to Volume",
                tooltip=get_tooltip("unified_segment", "apply_to_volume"),
            )
            self.apply_to_volume_checkbox.stateChanged.connect(self._on_apply_to_volume_changed)

            # Place the two checkboxes side by side: 'Batched' on the left, 'Apply to Volume'
            # pushed to the right.
            checkbox_row = QtWidgets.QHBoxLayout()
            checkbox_row.addWidget(self.batched_checkbox)
            checkbox_row.addStretch()
            checkbox_row.addWidget(self.apply_to_volume_checkbox)
            self.layout().addLayout(checkbox_row)

            # SAM2 volume-mode propagation controls (early stopping + z-range). Hidden until the
            # user enables 'Apply to Volume' with a SAM2 model; the controls drive the engine.
            self._propagation_settings = self._create_propagation_settings()
            self._propagation_settings.setVisible(False)
            self.layout().addWidget(self._propagation_settings)

        # Place the segment and clear buttons side by side.
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.segment_button)
        button_row.addWidget(self.clear_button)
        self.layout().addLayout(button_row)

        # Scribbles describe corrections for one object and cannot be assigned unambiguously to
        # separate objects in batched mode. Keep the control in sync as scribbles are added or
        # removed from the shared prompt layer.
        self._viewer.layers["prompts"].events.data.connect(self._update_batched_visibility)

        # Hide the batched control if the (already loaded) embeddings are tiled.
        self._update_batched_visibility()

    def _update_batched_visibility(self, event=None):
        """Disable batched segmentation while one or more scribble prompts are present."""
        super()._update_batched_visibility()

        prompt_layer = self._viewer.layers["prompts"]
        have_scribbles = any(
            shape_type in vutil.SCRIBBLE_SHAPE_TYPES for shape_type in prompt_layer.shape_type
        )
        if have_scribbles and self.batched_checkbox.isChecked():
            self.batched_checkbox.setChecked(False)

        self.batched_checkbox.setEnabled(not have_scribbles)
        tooltip_key = "batched_scribble_disabled" if have_scribbles else "batched"
        self.batched_checkbox.setToolTip(get_tooltip("unified_segment", tooltip_key))

    def _create_propagation_settings(self):
        """Build the SAM2 volume-mode propagation controls (early stopping + z-range slider).

        The controls live on the visible interactive widget and write their values into the hidden
        'UnifiedSegmentWidget' engine, which reads 'early_stop_patience' and 'z_range' at run time.
        """
        container = QtWidgets.QWidget()
        container.setLayout(QtWidgets.QVBoxLayout())

        # Stop after this many consecutive empty slices (0 disables early stopping).
        self.early_stop_patience = 2
        self.early_stop_patience_param, layout = self._add_int_param(
            "early_stop_patience", self.early_stop_patience, min_val=0, max_val=100,
            title="Stop after empty slices", tooltip=get_tooltip("segmentnd", "early_stop_patience"),
        )
        self.early_stop_patience_param.valueChanged.connect(self._sync_propagation_settings)
        container.layout().addLayout(layout)

        # Full-volume propagation (default) vs. a restricted z-range.
        self.use_full_z_range = True
        self.full_z_range_checkbox = self._add_boolean_param(
            "use_full_z_range", self.use_full_z_range,
            title="Propagate through all slices",
            tooltip=get_tooltip("segmentnd", "use_full_z_range"),
        )
        container.layout().addWidget(self.full_z_range_checkbox)

        # The labeled range slider collapses to zero height when sharing a row with a text label,
        # so it gets its own full-width row with the caption above it and a minimum height.
        z_range_label = QtWidgets.QLabel("Propagation z-range")
        z_range_label.setToolTip(get_tooltip("segmentnd", "z_range"))
        container.layout().addWidget(z_range_label)
        self.z_range_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self.z_range_slider.setRange(0, 1)
        self.z_range_slider.setValue((0, 1))
        self.z_range_slider.setToolTip(get_tooltip("segmentnd", "z_range"))
        self.z_range_slider.setEnabled(not self.use_full_z_range)
        self.z_range_slider.setMinimumHeight(40)
        self.z_range_slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        container.layout().addWidget(self.z_range_slider)

        self.full_z_range_checkbox.stateChanged.connect(self._on_full_z_range_changed)
        self.z_range_slider.valueChanged.connect(self._sync_propagation_settings)

        return _make_collapsible(
            container, title="Volume Propagation Settings", tooltip=get_tooltip("segmentnd", "settings"),
        )

    def _on_full_z_range_changed(self, state):
        """Enable/disable the z-range slider and push the change to the engine."""
        self.use_full_z_range = bool(state)
        self.z_range_slider.setEnabled(not self.use_full_z_range)
        self._sync_propagation_settings()

    def _update_z_range_slider(self):
        """Match the z-range slider extent to the depth of the loaded volume."""
        state = AnnotatorState()
        if state.image_shape is None:
            return
        z_max = int(state.image_shape[0]) - 1
        if z_max < 0:
            return
        lo, hi = self.z_range_slider.value()
        self.z_range_slider.setRange(0, z_max)
        if not self.use_full_z_range and hi <= z_max and lo < hi:
            self.z_range_slider.setValue((lo, hi))
        else:
            self.z_range_slider.setValue((0, z_max))

    def _sync_propagation_settings(self, *args):
        """Write the propagation controls into the hidden segmentation engine."""
        if self._segment_widget is None:
            return
        self._segment_widget.early_stop_patience = int(self.early_stop_patience_param.value())
        if bool(self.use_full_z_range):
            self._segment_widget.z_range = None
        else:
            self._segment_widget.z_range = tuple(int(v) for v in self.z_range_slider.value())

    def _on_apply_to_volume_changed(self, state):
        """Sync the shared volume mode to the segmentation engine."""
        self.apply_to_volume = bool(state)
        self._segment_widget.apply_to_volume = self.apply_to_volume

        # Show the propagation controls only in volume mode with a SAM2 model, and size the slider.
        if self._propagation_settings is not None:
            annotator_state = AnnotatorState()
            is_sam2 = bool(annotator_state.is_sam2) if annotator_state.is_sam2 is not None else False
            show = self.apply_to_volume and is_sam2
            self._propagation_settings.setVisible(show)
            if show:
                self._update_z_range_slider()
                self._sync_propagation_settings()

    def _on_batched_changed(self, state):
        """Sync the batched mode to the segmentation engine."""
        self.batched = bool(state)
        self._segment_widget.batched = self.batched

    def segment(self, viewer=None):
        """Run interactive segmentation for the current prompts."""
        if self._ndim == 2:
            _segment_object_2d(self._viewer, batched=bool(self.batched))
        else:
            self._sync_propagation_settings()
            self._segment_widget(self._viewer)

    def clear(self, viewer=None):
        """Clear the current annotations."""
        if self._ndim == 2 or self.apply_to_volume:
            vutil.clear_annotations(self._viewer)
        else:
            i = int(self._viewer.dims.point[0])
            vutil.clear_annotations_slice(self._viewer, i=i)

        # If it's a SAM2 promptable segmentation workflow,
        # we reset the prompts after the annotations have been cleared.
        state = AnnotatorState()
        if state.interactive_segmenter is not None:
            state.interactive_segmenter.reset_predictor()

        gc.collect()


class InteractiveTrackingWidget(_WidgetBase):
    """Merged interactive widget for the tracking annotator.

    Combines the prompt label menu, the track id / track state menus and the segment / clear
    controls into a single container, mirroring the segmentation annotator's
    'InteractiveSegmentationWidget'. The 'Apply to All Frames' checkbox governs both segmentation
    (current frame vs. propagating across the whole video) and clearing (current frame vs. all
    frames). A hidden 'UnifiedSegmentWidget' is used purely as the segmentation engine.

    Args:
        viewer: The napari viewer.
        tracking_widget: The combined prompt label / track id / track state menu created by
            'create_tracking_menu'.
        parent: The parent Qt widget.
    """

    def __init__(self, viewer, tracking_widget, parent=None):
        super().__init__(parent=parent)
        self._viewer = viewer
        self._tracking_widget = tracking_widget
        self.apply_to_volume = False
        self._segment_widget = None
        self._create_widget()

    def _create_widget(self):
        # The combined prompt label / track id / track state menus (one container so the three
        # dropdowns share an aligned label column). Lay each row out as label-left, box-right.
        self.layout().addWidget(self._tracking_widget.native)
        self._align_menu_rows()

        # Hidden segmentation engine: owns the per-frame and propagation logic, driven via its
        # 'apply_to_volume' attribute. Its own controls are not shown.
        self._segment_widget = UnifiedSegmentWidget(self._viewer, tracking=True)
        self._segment_widget.setVisible(False)
        self.layout().addWidget(self._segment_widget)

        # 'Apply to All Frames' toggles between segmenting the current frame and propagating the track.
        self.apply_to_volume_checkbox = self._add_boolean_param(
            "apply_to_volume", self.apply_to_volume, title="Apply to All Frames",
            tooltip=get_tooltip("unified_segment", "apply_to_volume"),
        )
        self.apply_to_volume_checkbox.stateChanged.connect(self._on_apply_to_volume_changed)
        self.layout().addWidget(self.apply_to_volume_checkbox)

        # 'Segment Object' / 'Clear Annotations' side by side.
        self.segment_button = QtWidgets.QPushButton("Segment Object [S]")
        self.segment_button.setToolTip(get_tooltip("unified_segment", "segment_button"))
        self.segment_button.clicked.connect(lambda: self.segment())
        self.clear_button = QtWidgets.QPushButton("Clear Annotations [Shift + C]")
        self.clear_button.setToolTip(get_tooltip("unified_segment", "clear_button"))
        self.clear_button.clicked.connect(lambda: self.clear())
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.segment_button)
        button_row.addWidget(self.clear_button)
        self.layout().addLayout(button_row)

    def _align_menu_rows(self):
        # Each menu row is a QHBoxLayout of [QLabel, QComboBox]. Insert a stretch between them so the
        # label stays left and the (fixed-width) combo box is right-aligned. Idempotent, since the
        # menu container persists across layout rebuilds.
        container_layout = self._tracking_widget.native.layout()
        for i in range(container_layout.count()):
            row = container_layout.itemAt(i).widget()
            row_layout = None if row is None else row.layout()
            if row_layout is None or row_layout.count() < 2:
                continue
            combo = row_layout.itemAt(row_layout.count() - 1).widget()
            if combo is not None:
                combo.setFixedWidth(300)
            if row_layout.count() < 3:  # No stretch inserted yet.
                row_layout.insertStretch(1)

    def _on_apply_to_volume_changed(self, state):
        self.apply_to_volume = bool(state)
        self._segment_widget.apply_to_volume = self.apply_to_volume

    def segment(self, viewer=None):
        """Run interactive tracking segmentation for the current prompts."""
        self._segment_widget(self._viewer)

    def clear(self, viewer=None):
        """Clear the annotations: the current frame, or all frames in 'Apply to All Frames' mode."""
        if self.apply_to_volume:
            _reset_tracking_state(self._viewer)
            vutil.clear_annotations(self._viewer)
        else:
            i = int(self._viewer.dims.point[0])
            vutil.clear_annotations_slice(self._viewer, i=i)

        state = AnnotatorState()
        if state.interactive_segmenter is not None:
            state.interactive_segmenter.reset_predictor()
        gc.collect()


class AutoSegmentV1Widget(_WidgetBase):
    """Automatic segmentation widget for the SAM (v1) AMG/AIS generators.

    This implementation backs the automatic tracking widget. The SAM2 segmentation annotator
    uses `AutoSegmentWidget` (dense/sparse modes) instead.
    """

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
            "apply_to_volume",
            self.apply_to_volume,
            title="Apply to Volume",
            tooltip=get_tooltip("autosegment", "apply_to_volume"),
        )

    def _add_common_settings(self, settings):
        # Create the UI element for min object size.
        self.min_object_size = 100
        self.min_object_size_param, layout = self._add_int_param(
            "min_object_size",
            self.min_object_size,
            min_val=0,
            max_val=int(1e4),
            tooltip=get_tooltip("autosegment", "min_object_size"),
        )
        settings.layout().addLayout(layout)

        # Add extra settings for volumetric segmentation: gap_closing and min_extent.
        if self.volumetric:
            self.gap_closing = 2
            self.gap_closing_param, layout = self._add_int_param(
                "gap_closing",
                self.gap_closing,
                min_val=0,
                max_val=10,
                tooltip=get_tooltip("autosegment", "gap_closing"),
            )
            settings.layout().addLayout(layout)

            self.min_extent = 2
            self.min_extent_param, layout = self._add_int_param(
                "min_extent",
                self.min_extent,
                min_val=0,
                max_val=10,
                tooltip=get_tooltip("autosegment", "min_extent"),
            )
            settings.layout().addLayout(layout)

    def _ais_settings(self):
        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())

        # Create the UI element for center_distance_threshold.
        self.center_distance_thresh = 0.5
        self.center_distance_thresh_param, layout = self._add_float_param(
            "center_distance_thresh",
            self.center_distance_thresh,
            tooltip=get_tooltip("autosegment", "center_distance_thresh"),
        )
        settings.layout().addLayout(layout)

        # Create the UI element for boundary_distance_threshold.
        self.boundary_distance_thresh = 0.5
        self.boundary_distance_thresh_param, layout = self._add_float_param(
            "boundary_distance_thresh",
            self.boundary_distance_thresh,
            tooltip=get_tooltip("autosegment", "boundary_distance_thresh"),
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
            "pred_iou_thresh",
            self.pred_iou_thresh,
            tooltip=get_tooltip("autosegment", "pred_iou_thresh"),
        )
        settings.layout().addLayout(layout)

        # Create the UI element for stability score thresh.
        self.stability_score_thresh = 0.95
        self.stability_score_thresh_param, layout = self._add_float_param(
            "stability_score_thresh",
            self.stability_score_thresh,
            tooltip=get_tooltip("autosegment", "stability_score_thresh"),
        )
        settings.layout().addLayout(layout)

        # Create the UI element for box nms thresh.
        self.box_nms_thresh = 0.7
        self.box_nms_thresh_param, layout = self._add_float_param(
            "box_nms_thresh",
            self.box_nms_thresh,
            tooltip=get_tooltip("autosegment", "box_nms_thresh"),
        )
        settings.layout().addLayout(layout)

        # Add min_object_size.
        self._add_common_settings(settings)

        return settings

    def _create_settings(self):
        setting_values = (
            self._ais_settings() if self.with_decoder else self._amg_settings()
        )
        settings = _make_collapsible(
            setting_values, title="Automatic Segmentation Settings", tooltip=get_tooltip("autosegment", "settings"),
        )
        return settings

    def _empty_segmentation_warning(self):
        msg = "The automatic segmentation result does not contain any objects."
        msg += "Setting a smaller value for 'min_object_size' may help."
        if not self.with_decoder:
            msg += "Setting smaller values for 'pred_iou_thresh' and 'stability_score_thresh' may also help."
        val_results = {"message_type": "error", "message": msg}
        return _generate_message(
            val_results["message_type"], val_results["message"]
        )

    def _run_segmentation_2d(self, kwargs, i=None):
        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker
        def seg_impl():
            def pbar_init(total, description):
                pbar_signals.pbar_total.emit(total)
                pbar_signals.pbar_description.emit(description)

            seg = _instance_segmentation_impl(
                self.min_object_size,
                i=i,
                pbar_init=pbar_init,
                pbar_update=lambda update: pbar_signals.pbar_update.emit(
                    update
                ),
                **kwargs,
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
            if state.is_sam2:
                from micro_sam.precompute_state import _has_autoseg_state
                embeddings_are_precomputed = _has_autoseg_state(
                    state.embedding_path, "amg", state_count=n_slices,
                )
            else:
                embeddings_are_precomputed = (state.autoseg_state is not None) and (
                    len(state.autoseg_state) > n_slices
                )
            if not embeddings_are_precomputed:
                return False
        return True

    def _run_segmentation_3d(self, kwargs):
        allow_segment_3d = self._allow_segment_3d()
        if not allow_segment_3d:
            val_results = {
                "message_type": "error",
                "message": "Volumetric segmentation with AMG is only supported if you have a GPU.",
            }
            return _generate_message(
                val_results["message_type"], val_results["message"]
            )

        pbar, pbar_signals = _create_pbar_for_threadworker()

        # @thread_worker
        def seg_impl():
            segmentation = np.zeros_like(
                self._viewer.layers["auto_segmentation"].data
            )
            offset = 0

            def pbar_init(total, description):
                pbar_signals.pbar_total.emit(total)
                pbar_signals.pbar_description.emit(description)

            pbar_init(segmentation.shape[0], "Segment volume")

            # Further optimization: parallelize if state is precomputed for all slices
            for i in range(segmentation.shape[0]):
                seg = _instance_segmentation_impl(
                    self.min_object_size, i=i, **kwargs
                )
                seg_max = seg.max()
                if seg_max == 0:
                    continue
                seg[seg != 0] += offset
                offset = seg_max + offset
                segmentation[i] = seg
                pbar_signals.pbar_update.emit(1)

            pbar_signals.pbar_reset.emit()
            segmentation = merge_instance_segmentation_3d(
                segmentation,
                beta=0.5,
                gap_closing=self.gap_closing,
                min_z_extent=self.min_extent,
                verbose=True,
                pbar_init=pbar_init,
                pbar_update=lambda update: pbar_signals.pbar_update.emit(1),
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


class AutoSegmentWidget(_WidgetBase):
    """Automatic segmentation widget for SAM2 with 'amg', 'sparse' and 'dense' modes.

    Subclasses set `_is_tracking = True` to hide the z-tiling controls (tracking segments per frame
    in 2d, so z-tiling does not apply).

    When a UniSAM2 decoder is loaded (`AnnotatorState.decoder`), only the decoder-based 'sparse'
    (flow, LM data) and 'dense' (multicut, EM data) modes are offered - these operate on the
    foreground and directed-distance predictions of the decoder via
    `micro_sam.v2.automatic_segmentation` and supersede grid-based AMG. The 'amg' mode (grid-based
    automatic mask generation via `micro_sam.v2.instance_segmentation`, no decoder required) is only
    offered as a fallback when no decoder is available. A mode dropdown sits next to the 'Apply to
    Volume' switch and the post-processing parameters refresh on mode change.

    Disk-backed caching of the state is opted into via the 'cache automatic segmentation state'
    checkbox in the embedding settings (read here through the embedding widget); when off, the state
    is kept in memory only.

    Args:
        viewer: The napari viewer.
        with_decoder: Whether the loaded model has a UniSAM2 decoder for automatic segmentation.
        volumetric: Whether the data is volumetric (3d).
        parent: The parent Qt widget.
    """

    _is_tracking = False

    def __init__(self, viewer, with_decoder, volumetric, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self.with_decoder = with_decoder
        self.volumetric = volumetric
        # With a decoder we default to (and only offer) the decoder-based modes; 'amg' is the
        # fallback (and only mode) when no decoder is available.
        self.mode = "sparse" if with_decoder else "amg"
        self.settings = None
        # The flow computation backend is always the (faster) cpp implementation.
        self.backend = "cpp"
        # z block / halo for 3d decoder inference: the volume is decoded in z chunks to bound memory.
        # These only matter for volumetric decoder modes; set 'tile_z' >= the slice count for no z-tiling.
        from micro_sam.v2.util import DEFAULT_TILE_Z, DEFAULT_HALO_Z
        self.tile_z, self.halo_z = DEFAULT_TILE_Z, DEFAULT_HALO_Z
        # Cache of the (initialized) segmentation generator so changing post-processing parameters
        # only re-runs 'generate', not the expensive UniSAM2 inference. Keyed by the inputs.
        self._segmenter = None
        self._segmenter_key = None
        self._create_widget()

    def _create_widget(self):
        # Top row: the 'Apply to Volume' switch (3d only) next to the mode dropdown.
        top_row = QtWidgets.QHBoxLayout()
        if self.volumetric:
            self.apply_to_volume = False
            self.apply_to_volume_checkbox = self._add_boolean_param(
                "apply_to_volume",
                self.apply_to_volume,
                title="Apply to Volume",
                tooltip=get_tooltip("autosegment", "apply_to_volume"),
            )
            top_row.addWidget(self.apply_to_volume_checkbox)

        # With a UniSAM2 decoder we only offer the decoder-based 'sparse' (flow, LM) and 'dense'
        # (multicut, EM) modes; 'amg' (grid-based, no decoder) is the fallback when none is loaded.
        mode_choices = ["sparse", "dense"] if self.with_decoder else ["amg"]
        self.mode_dropdown, mode_layout = self._add_choice_param(
            "mode",
            self.mode,
            mode_choices,
            title="mode:",
            update=self._on_mode_changed,
            tooltip=get_tooltip("autosegment", "mode"),
        )
        top_row.addLayout(mode_layout)
        self.layout().addLayout(top_row)

        # Advanced post-processing settings, shown inline and refreshed on mode change.
        self.settings = self._make_settings_widget()
        self.layout().addWidget(self.settings)

        # Run button.
        self.run_button = QtWidgets.QPushButton("Automatic Segmentation")
        self.run_button.clicked.connect(self.__call__)
        self.run_button.setToolTip(get_tooltip("autosegment", "run_button"))
        self.layout().addWidget(self.run_button)

    def _reset_segmentation_mode(self, with_decoder):
        # If the decoder availability is unchanged there is nothing to rebuild.
        if with_decoder == self.with_decoder:
            return
        self.with_decoder = with_decoder

        # The mode dropdown (built from 'with_decoder') must be rebuilt when the loaded model changes:
        # with a decoder we offer only the decoder-based 'sparse'/'dense' modes, and without one only
        # the 'amg' fallback - otherwise a finetuned model would keep showing the wrong options.
        mode_choices = ["sparse", "dense"] if with_decoder else ["amg"]
        self.mode_dropdown.blockSignals(True)
        self.mode_dropdown.clear()
        self.mode_dropdown.addItems(mode_choices)
        self.mode_dropdown.setCurrentText("sparse" if with_decoder else "amg")
        self.mode_dropdown.blockSignals(False)

        # Drop the cached segmenter, since the loaded model (and so its predictions) changed, and
        # refresh the settings panel to match the new default mode.
        self._segmenter = None
        self._segmenter_key = None
        self._on_mode_changed()

    def _on_mode_changed(self, index=None):
        self.mode = self.mode_dropdown.currentText()
        new_settings = self._make_settings_widget()
        self.layout().replaceWidget(self.settings, new_settings)
        self.settings.deleteLater()
        self.settings = new_settings

    def _make_settings_widget(self):
        # All hyperparameters (except mode and apply-to-volume) live in one collapsible
        # 'Advanced Settings' panel. The z block / halo (3d decoder modes) sit at the very top.
        advanced = QtWidgets.QWidget()
        advanced.setLayout(QtWidgets.QVBoxLayout())
        advanced.layout().setContentsMargins(0, 0, 0, 0)
        self._add_z_tiling_params(advanced)
        if self.mode == "amg":
            self._amg_settings(advanced)
        elif self.mode == "dense":
            self._dense_settings(advanced)
        else:
            self._sparse_settings(advanced)

        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())
        settings.layout().setContentsMargins(0, 0, 0, 0)
        advanced_tooltip = get_tooltip("autosegment", "advanced_settings")
        settings.layout().addWidget(
            _make_collapsible(advanced, title="Advanced Settings", tooltip=advanced_tooltip)
        )
        return settings

    def _add_density_threshold(self, settings):
        self.density_threshold_param, layout = self._add_float_param(
            "density_threshold", self.density_threshold, min_val=0.0, max_val=100.0, step=1.0,
            tooltip=get_tooltip("autosegment", "density_threshold"),
        )
        settings.layout().addLayout(layout)

    def _add_z_tiling_params(self, settings):
        # 3d decoder inference decodes the volume in z blocks (with a halo for context) to bound
        # memory. 'tile_z' and 'halo_z' sit side by side at the top of the settings. Only for
        # volumetric decoder segmentation - not 2d, not tracking (per-frame 2d) and not amg
        # (slice-by-slice, no z decoder pass).
        if not self.volumetric or self._is_tracking or self.mode == "amg":
            return
        row = QtWidgets.QHBoxLayout()
        self.tile_z_param, _ = self._add_int_param(
            "tile_z", self.tile_z, min_val=1, max_val=512, title="tile_z:",
            tooltip=get_tooltip("autosegment", "tile_z"), layout=row,
        )
        self.halo_z_param, _ = self._add_int_param(
            "halo_z", self.halo_z, min_val=0, max_val=128, title="overlap_z:",
            tooltip=get_tooltip("autosegment", "halo_z"), layout=row,
        )
        settings.layout().addLayout(row)

    def _add_flow_integration_params(self, settings, n_iter, dt=0.5, sigma=1.0):
        self.n_iter = n_iter
        self.n_iter_param, layout = self._add_int_param(
            "n_iter", self.n_iter, min_val=1, max_val=1000, tooltip=get_tooltip("autosegment", "n_iter"),
        )
        settings.layout().addLayout(layout)

        self.dt = dt
        self.dt_param, layout = self._add_float_param(
            "dt", self.dt, min_val=0.0, max_val=5.0, step=0.1, tooltip=get_tooltip("autosegment", "dt"),
        )
        settings.layout().addLayout(layout)

        self.sigma = sigma
        self.sigma_param, layout = self._add_float_param(
            "sigma", self.sigma, min_val=0.0, max_val=10.0, step=0.1, tooltip=get_tooltip("autosegment", "sigma"),
        )
        settings.layout().addLayout(layout)

        # Default to 8 threads for the post-processing flow/multicut backends (a sensible default that
        # does not oversubscribe; the user can raise it up to the spinbox maximum).
        self.n_threads = min(8, mp.cpu_count())
        self.n_threads_param, layout = self._add_int_param(
            "n_threads", self.n_threads, min_val=1, max_val=64, tooltip=get_tooltip("autosegment", "n_threads"),
        )
        settings.layout().addLayout(layout)

    def _amg_settings(self, settings):
        # Grid-based SAM2 AMG parameters (no decoder required). points_per_side / pred_iou_thresh /
        # stability_score_thresh control the (expensive) mask generation; min_object_size is applied
        # in the (cheap) post-processing.
        self.points_per_side = 32
        self.points_per_side_param, layout = self._add_int_param(
            "points_per_side", self.points_per_side, min_val=1, max_val=256,
            tooltip=get_tooltip("autosegment", "points_per_side"),
        )
        settings.layout().addLayout(layout)

        self.pred_iou_thresh = 0.8
        self.pred_iou_thresh_param, layout = self._add_float_param(
            "pred_iou_thresh", self.pred_iou_thresh, min_val=0.0, max_val=1.0, step=0.05,
            tooltip=get_tooltip("autosegment", "pred_iou_thresh"),
        )
        settings.layout().addLayout(layout)

        self.stability_score_thresh = 0.9
        self.stability_score_thresh_param, layout = self._add_float_param(
            "stability_score_thresh", self.stability_score_thresh, min_val=0.0, max_val=1.0, step=0.05,
            tooltip=get_tooltip("autosegment", "stability_score_thresh"),
        )
        settings.layout().addLayout(layout)

        self.min_object_size = 50
        self.min_object_size_param, layout = self._add_int_param(
            "min_object_size", self.min_object_size, min_val=0, max_val=int(1e4),
            tooltip=get_tooltip("autosegment", "min_object_size"),
        )
        settings.layout().addLayout(layout)

    def _sparse_settings(self, settings):
        # Flow-based instance segmentation parameters (LM data).
        from micro_sam.v2.postprocessing import DEFAULT_POSTPROCESSING
        defaults = DEFAULT_POSTPROCESSING["sparse"]

        self.foreground_threshold = defaults["foreground_threshold"]
        self.foreground_threshold_param, layout = self._add_float_param(
            "foreground_threshold", self.foreground_threshold, min_val=0.0, max_val=1.0, step=0.05,
            tooltip=get_tooltip("autosegment", "foreground_threshold"),
        )
        settings.layout().addLayout(layout)

        self.density_threshold = defaults["density_threshold"]
        self._add_density_threshold(settings)

        self.min_object_size = defaults["min_size"]
        self.min_object_size_param, layout = self._add_int_param(
            "min_object_size", self.min_object_size, min_val=0, max_val=int(1e4),
            tooltip=get_tooltip("autosegment", "min_object_size"),
        )
        settings.layout().addLayout(layout)

        self._add_flow_integration_params(
            settings,
            n_iter=defaults["n_iter"],
            dt=defaults["dt"],
            sigma=defaults["sigma"],
        )

    def _dense_settings(self, settings):
        # Multicut-based instance segmentation parameters (EM data, 2d and 3d).
        from micro_sam.v2.postprocessing import DEFAULT_POSTPROCESSING
        defaults = DEFAULT_POSTPROCESSING["dense"]

        self.beta = defaults["beta"]
        self.beta_param, layout = self._add_float_param(
            "beta", self.beta, min_val=0.0, max_val=1.0, step=0.05,
            tooltip=get_tooltip("autosegment", "beta"),
        )
        settings.layout().addLayout(layout)

        self.density_threshold = defaults["density_threshold"]
        self._add_density_threshold(settings)

        self._add_flow_integration_params(
            settings,
            n_iter=defaults["n_iter"],
            dt=defaults["dt"],
            sigma=defaults["sigma"],
        )

    def _postproc_kwargs(self):
        if self.mode == "dense":
            return dict(
                beta=self.beta, density_threshold=self.density_threshold, n_iter=self.n_iter,
                dt=self.dt, sigma=self.sigma, n_threads=self.n_threads, backend=self.backend,
            )
        return dict(
            foreground_threshold=self.foreground_threshold, density_threshold=self.density_threshold,
            min_size=self.min_object_size, n_iter=self.n_iter, dt=self.dt, sigma=self.sigma,
            n_threads=self.n_threads, backend=self.backend,
        )

    def _get_tiling(self):
        # In-plane (xy) tiling for automatic segmentation, taken from the embedding widget (where the
        # embeddings' tiling is configured). Returns (None, None) when tiling is off. z-tiling is not
        # handled here - it is a decoder-inference concern driven by this widget's 'tile_z'/'halo_z'.
        state = AnnotatorState()
        embed_widget = state.widgets.get("embeddings")
        if embed_widget is None or getattr(embed_widget, "tiling", "no") != "yes":
            return None, None
        return _process_tiling_inputs(
            embed_widget.tile_x, embed_widget.tile_y, embed_widget.halo_x, embed_widget.halo_y,
        )

    def _z_tiling(self, n_slices):
        # The z block / halo for 3d decoder inference. 'tile_z' >= the slice count means no z-tiling
        # (the whole volume is decoded in one z block).
        z_block = self.tile_z if 0 < self.tile_z < n_slices else n_slices
        z_halo = self.halo_z if z_block < n_slices else 0
        return z_block, z_halo

    def _state_save_path(self, state):
        # The state cache is opted into via the embedding settings' 'cache automatic segmentation state'
        # checkbox; when on it persists next to the embeddings, else in-memory only ('_segmenter' cache).
        embed_widget = state.widgets.get("embeddings")
        return state.embedding_path if getattr(embed_widget, "cache_state", False) else None

    def _run_unisam2(self, state, run_raw, ndim, z, pbar_init=None, pbar_update=None):
        from micro_sam.precompute_state import cache_autoseg_state

        device = next(state.decoder.parameters()).device
        save_path = self._state_save_path(state)

        # All decoder auto-seg cases reuse the precomputed embeddings and run the decoder on them (no
        # encoder re-run). The tiling is taken from the embeddings (tiled embeddings have a top-level
        # 'input_size' of None). This covers 2d and 3d, tiled and untiled.
        tile_shape, halo = None, None
        z_block, z_halo = None, None
        if not self.volumetric or ndim == 3:
            # Plain 2d image, or the whole 3d volume: the precomputed embeddings match directly.
            image_embeddings = state.image_embeddings
            is_tiled = image_embeddings["input_size"] is None
            if ndim == 3:  # the decoder pass is z-chunked using the auto-seg z block / halo controls.
                z_block, z_halo = self._z_tiling(int(run_raw.shape[0]))
        else:
            # A single slice of a 3d volume: reuse that slice's features (no re-encode). For untiled
            # embeddings, build the slice's 2d embedding; for tiled embeddings, pass the tiled 3d
            # embeddings + slice index 'z' and let the segmenter reconstruct each tile's slice.
            emb3d = state.image_embeddings
            if emb3d is not None and emb3d.get("input_size") is not None:
                image_embeddings = {
                    "features": np.asarray(emb3d["features"][z:z + 1]),
                    "input_size": emb3d["input_size"], "original_size": emb3d["original_size"],
                }
                is_tiled = False
            else:
                image_embeddings, is_tiled = emb3d, True

        # The in-memory cache avoids re-running the model when only the post-processing parameters
        # change; 'cache_autoseg_state' additionally persists the decoder predictions in the
        # embedding Zarr so a later run / session reuses them. The whole volume is
        # cached under one key ('state'); a single segmented slice under 'state-{z}'.
        cache_key = (state.data_signature, "unisam2", ndim, z, tile_shape, halo, z_block, z_halo,
                     image_embeddings is not None)
        if self._segmenter is None or self._segmenter_key != cache_key:
            self._segmenter = cache_autoseg_state(
                "ais", state.decoder, run_raw, image_embeddings, save_path, ndim=ndim,
                model_type=getattr(state.predictor, "model_type", None),
                i=z, state_index=(None if ndim == 3 else z), is_tiled=is_tiled,
                tile_shape=tile_shape, halo=halo, device=device, z_block=z_block, z_halo=z_halo,
                pbar_init=pbar_init, pbar_update=pbar_update, verbose=False,
            )
            self._segmenter_key = cache_key

        return self._segmenter.generate(mode=self.mode, **self._postproc_kwargs())

    def _run_amg(self, state, run_raw, ndim, z, pbar_init=None, pbar_update=None):
        from micro_sam.v2.instance_segmentation import get_amg_segmenter, automatic_3d_segmentation
        from micro_sam.precompute_state import cache_autoseg_state

        # The SAM2 model: 'state.predictor' is the image predictor (2d) wrapping the model, or the
        # video predictor itself (3d); both can drive the grid-based mask generator.
        model = getattr(state.predictor, "model", state.predictor)
        model_type = getattr(state.predictor, "model_type", None)
        save_path = self._state_save_path(state)

        generate_kwargs = dict(min_object_size=self.min_object_size, with_background=True)
        amg_params = dict(
            points_per_side=self.points_per_side, pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )

        if ndim == 3:  # Segment slice-by-slice and stitch across z. Tiling is in-plane (None if off).
            tile_shape, halo = self._get_tiling()
            segmenter = get_amg_segmenter(model, is_tiled=tile_shape is not None, model_type=model_type, **amg_params)
            # Reuse the precomputed 3d embeddings per slice (tiled or not) so AMG does not re-encode,
            # and cache each slice's grid-prediction state in the embedding Zarr.
            return automatic_3d_segmentation(
                run_raw, segmenter, tile_shape=tile_shape, halo=halo,
                image_embeddings=state.image_embeddings, state_save_path=save_path,
                pbar_init=pbar_init, pbar_update=pbar_update, **generate_kwargs,
            )

        # For plain 2d the annotator has already precomputed the embeddings, so we reuse them (the
        # tiling is taken from the embeddings). For a single slice of a volume the embeddings are
        # video-style, so we compute the slice embedding - tiling stays None when it is not configured.
        if self.volumetric:
            tile_shape, halo = self._get_tiling()
            image_embeddings, is_tiled = None, tile_shape is not None
        else:
            tile_shape, halo, image_embeddings = None, None, state.image_embeddings
            is_tiled = image_embeddings["input_size"] is None

        # The in-memory cache lets changing the post-processing parameters re-run only the cheap
        # 'generate'; the on-disk cache (via 'cache_autoseg_state') persists the state across sessions.
        cache_key = (state.data_signature, "amg", z, tile_shape, halo, image_embeddings is not None,
                     self.points_per_side, self.pred_iou_thresh, self.stability_score_thresh)
        if self._segmenter is None or self._segmenter_key != cache_key:
            if is_tiled:  # The tiled segmenter reports per-tile progress.
                self._segmenter = cache_autoseg_state(
                    "amg", model, run_raw, image_embeddings, save_path, model_type=model_type,
                    state_index=z, is_tiled=True, tile_shape=tile_shape, halo=halo,
                    pbar_init=pbar_init, pbar_update=pbar_update, verbose=False, **amg_params,
                )
            else:  # A single 2d image is one step.
                if pbar_init is not None:
                    pbar_init(1, "Automatic segmentation")
                self._segmenter = cache_autoseg_state(
                    "amg", model, run_raw, image_embeddings, save_path, model_type=model_type,
                    state_index=z, is_tiled=False, verbose=False, **amg_params,
                )
                if pbar_update is not None:
                    pbar_update(1)
            self._segmenter_key = cache_key

        return self._segmenter.generate(**generate_kwargs)

    def __call__(self):
        state = AnnotatorState()
        if self.mode != "amg" and (not self.with_decoder or state.decoder is None):
            return _generate_message(
                "error",
                "The 'sparse' and 'dense' modes require a finetuned UniSAM2 model with a decoder. "
                "Load one via the 'custom weights' path in the embedding widget, or use the 'amg' mode.",
            )
        if _validate_layers(self._viewer, automatic_segmentation=True):
            return
        # The (2d) decoder modes reuse the precomputed image embeddings, so they must exist and match
        # the current image. If embeddings were reset (e.g. after an image change) this prompts the
        # user to recompute them instead of segmenting with stale embeddings.
        if _validate_embeddings(self._viewer):
            return

        # Get the raw image and determine the run dimensionality.
        image_name = state.get_image_name(self._viewer)
        raw = np.asarray(self._viewer.layers[image_name].data)

        apply_to_volume = self.volumetric and getattr(self, "apply_to_volume", False)
        z = None
        if apply_to_volume:
            run_raw, ndim = raw, 3
        elif self.volumetric:  # segment only the current slice
            z = int(self._viewer.dims.point[0])
            run_raw, ndim = raw[z], 2
        else:
            run_raw, ndim = raw, 2

        # Show a progress bar in the napari activity dock (and the status-bar wheel) that advances
        # with the actual work: per tile for tiled runs, per slice for 3d, and as a single step for a
        # plain 2d image. Thread workers are disabled in this tool (see top of module), so the run is
        # synchronous; we drive the bar via callbacks the backends call between units and pump the Qt
        # event loop with 'processEvents' on each update so it repaints live. It is always closed in
        # the 'finally' block. (3d decoder inference runs through a thread pool and is reported as a
        # single step, since it cannot update the napari bar live.)
        pbar, pbar_signals = _create_pbar_for_threadworker()

        def pbar_init(total, description):
            pbar_signals.pbar_total.emit(total)
            pbar_signals.pbar_description.emit(description)
            QtWidgets.QApplication.processEvents()

        def pbar_update(update=1):
            pbar_signals.pbar_update.emit(update)
            QtWidgets.QApplication.processEvents()

        pbar_signals.pbar_description.emit(f"Running automatic segmentation ({self.mode})")
        QtWidgets.QApplication.processEvents()
        try:
            if self.mode == "amg":
                seg = self._run_amg(state, run_raw, ndim, z, pbar_init=pbar_init, pbar_update=pbar_update)
            else:
                seg = self._run_unisam2(state, run_raw, ndim, z, pbar_init=pbar_init, pbar_update=pbar_update)
        finally:
            pbar_signals.pbar_stop.emit()

        if z is None:
            self._viewer.layers["auto_segmentation"].data = seg
        else:
            self._viewer.layers["auto_segmentation"].data[z] = seg
        self._viewer.layers["auto_segmentation"].refresh()
        _select_layer(self._viewer, "auto_segmentation")


class AutoTrackWidget(AutoSegmentWidget):
    _is_tracking = True

    def _create_widget(self):
        top_row = QtWidgets.QHBoxLayout()
        self.apply_to_volume = False
        self.apply_to_volume_checkbox = self._add_boolean_param(
            "apply_to_volume",
            self.apply_to_volume,
            title="Track Timeseries",
            tooltip=get_tooltip("autotrack", "run_tracking"),
        )
        top_row.addWidget(self.apply_to_volume_checkbox)

        mode_choices = ["sparse", "dense"] if self.with_decoder else ["amg"]
        self.mode_dropdown, mode_layout = self._add_choice_param(
            "mode",
            self.mode,
            mode_choices,
            title="segmentation mode:",
            update=self._on_mode_changed,
            tooltip=get_tooltip("autosegment", "mode"),
        )
        top_row.addLayout(mode_layout)
        self.layout().addLayout(top_row)

        self.settings = self._make_settings_widget()
        self.layout().addWidget(self.settings)

        self.run_button = QtWidgets.QPushButton("Automatic Tracking")
        self.run_button.clicked.connect(self.__call__)
        self.run_button.setToolTip(get_tooltip("autotrack", "run_button"))
        self.layout().addWidget(self.run_button)

    def _empty_tracking_warning(self):
        return _generate_message(
            "error",
            "The automatic tracking result does not contain any objects. "
            "Try adjusting the automatic tracking settings.",
        )

    def _run_frame_segmentation(self, state, frame, frame_id, pbar_init=None, pbar_update=None):
        # Forward the progress callbacks so a frame advances the overall bar per tile (tiled) or by
        # one step (untiled). The caller passes a no-op 'pbar_init' so a frame cannot reset the total.
        if self.mode == "amg":
            return self._run_amg(state, frame, 2, frame_id, pbar_init=pbar_init, pbar_update=pbar_update)
        return self._run_unisam2(state, frame, 2, frame_id, pbar_init=pbar_init, pbar_update=pbar_update)

    def _n_inplane_tiles(self, state, raw):
        # In-plane tiles per frame for the current run (1 if not tiled). The auto-tracking bar
        # advances per tile, so its total is this times the number of frames.
        from bioimage_cpp.utils import Blocking

        if self.mode == "amg":
            tile_shape, _ = self._get_tiling()
        else:  # decoder modes reuse the precomputed embeddings; tiled ones have no top-level input_size.
            emb = state.image_embeddings
            tile_shape = None
            if emb is not None and emb.get("input_size") is None:
                tile_shape = tuple(int(s) for s in emb["features"].attrs["tile_shape"])
        if tile_shape is None:
            return 1
        return Blocking([0, 0], list(raw.shape[1:3]), list(tile_shape)).number_of_blocks

    def _track_timeseries(self, state, raw):
        segmentation = np.zeros_like(self._viewer.layers["auto_segmentation"].data)
        pbar, pbar_signals = _create_pbar_for_threadworker()
        offset = 0

        def tracking_pbar_init(total, description):
            pbar_signals.pbar_total.emit(total)
            pbar_signals.pbar_description.emit(description)
            QtWidgets.QApplication.processEvents()

        def tracking_pbar_update(update=1):
            pbar_signals.pbar_update.emit(update)
            QtWidgets.QApplication.processEvents()

        # Swallow each frame's 'pbar_init' so it cannot reset the overall total; the per-tile (or
        # per-step) 'pbar_update' calls drive the bar instead.
        def frame_pbar_init(total, description):
            pass

        n_tiles = self._n_inplane_tiles(state, raw)
        try:
            # One determinate bar over the actual work: n_tiles x n_frames. The per-frame segmentation
            # advances it per tile (tiled) or once per frame (untiled), so a tiled run no longer looks
            # like it is doing only n_frames steps.
            pbar_signals.pbar_total.emit(n_tiles * len(raw))
            QtWidgets.QApplication.processEvents()
            for frame_id, frame in enumerate(raw):
                pbar_signals.pbar_description.emit(
                    f"Automatic segmentation ({self.mode}): frame {frame_id + 1}/{len(raw)}"
                )
                QtWidgets.QApplication.processEvents()
                seg = self._run_frame_segmentation(
                    state, frame, frame_id, pbar_init=frame_pbar_init, pbar_update=tracking_pbar_update,
                )
                seg_max = int(seg.max())
                if seg_max != 0:
                    seg[seg != 0] += offset
                    offset += seg_max
                    segmentation[frame_id] = seg

            if offset == 0:
                self._viewer.layers["auto_segmentation"].data = segmentation
                self._viewer.layers["auto_segmentation"].refresh()
                return self._empty_tracking_warning()

            pbar_signals.pbar_reset.emit()
            segmentation, lineages = track_across_frames(
                raw,
                segmentation,
                verbose=True,
                pbar_init=tracking_pbar_init,
                pbar_update=tracking_pbar_update,
            )
        finally:
            pbar_signals.pbar_stop.emit()

        state.lineage = lineages
        self._viewer.layers["auto_segmentation"].data = segmentation
        self._viewer.layers["auto_segmentation"].refresh()
        _select_layer(self._viewer, "auto_segmentation")

    def __call__(self):
        state = AnnotatorState()
        if not (self.volumetric and getattr(self, "apply_to_volume", False)):
            return super().__call__()
        if self.mode != "amg" and (not self.with_decoder or state.decoder is None):
            return _generate_message(
                "error",
                "The 'sparse' and 'dense' modes require a finetuned UniSAM2 model with a decoder. "
                "Load one via the 'custom weights' path in the embedding widget, or use the 'amg' mode.",
            )
        if _validate_layers(self._viewer, automatic_segmentation=True):
            return None
        if state.committed_lineages:
            return _generate_message(
                "error",
                "Automatic tracking can only be called if you have not committed interactive tracking results yet.",
            )

        image_name = state.get_image_name(self._viewer)
        raw = np.asarray(self._viewer.layers[image_name].data)
        if raw.ndim != 3:
            return _generate_message("error", "Automatic tracking expects a 2d timeseries.")
        return self._track_timeseries(state, raw)
