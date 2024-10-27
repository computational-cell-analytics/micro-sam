"""Implements the widgets used in the annotation plugins.
"""

import os
import pickle
from pathlib import Path
from typing import Optional
import multiprocessing as mp

import h5py
import json
import zarr
import z5py
import napari
import numpy as np

import elf.parallel

from qtpy import QtWidgets
from qtpy.QtCore import QObject, Signal
from superqt import QCollapsible
from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container, create_widget
# We have disabled the thread workers for now because they result in a
# massive slowdown in napari >= 0.5.
# See also https://forum.image.sc/t/napari-thread-worker-leads-to-massive-slowdown/103786
# from napari.qt.threading import thread_worker
from napari.utils import progress

from ._state import AnnotatorState
from . import util as vutil
from ._tooltips import get_tooltip
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


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear_volume(viewer: "napari.viewer.Viewer", all_slices: bool = True) -> None:
    """Widget for clearing the current annotations in 3D.

    Args:
        viewer: The napari viewer.
        all_slices: Choose whether to clear the annotations for all or only the current slice.
    """
    if all_slices:
        vutil.clear_annotations(viewer)
    else:
        i = int(viewer.cursor.position[0])
        vutil.clear_annotations_slice(viewer, i=i)


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
        i = int(viewer.cursor.position[0])
        vutil.clear_annotations_slice(viewer, i=i)


def _commit_impl(viewer, layer, preserve_committed):
    # Check if we have a z_range. If yes, use it to set a bounding box.
    state = AnnotatorState()
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
    if preserve_committed:
        prev_seg = viewer.layers["committed_objects"].data[bb]
        mask[prev_seg != 0] = 0

    # Write the current object to committed objects.
    seg[mask] += id_offset
    viewer.layers["committed_objects"].data[bb][mask] = seg[mask]
    viewer.layers["committed_objects"].refresh()

    return id_offset, seg, mask, bb


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

    # Write metadata about the model that's being used etc.
    # Only if it's not written to the file yet.
    if "data_signature" not in f.attrs:
        state = AnnotatorState()
        embeds = state.widgets["embeddings"]
        tile_shape, halo = _process_tiling_inputs(embeds.tile_x, embeds.tile_y, embeds.halo_x, embeds.halo_y)
        signature = util._get_embedding_signature(
            input_=None,  # We don't need this because we pass the data signature.
            predictor=state.predictor,
            tile_shape=tile_shape,
            halo=halo,
            data_signature=state.data_signature,
        )
        for key, val in signature.items():
            f.attrs[key] = val

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
        # TODO write the settings for the auto segmentation widget.
        return

    def write_prompts(object_id, prompts, point_prompts):
        g = f.create_group(f"prompts/{object_id}")
        if prompts is not None and len(prompts) > 0:
            data = np.array(prompts)
            g.create_dataset("prompts", data=data, chunks=data.shape)
        if point_prompts is not None and len(point_prompts) > 0:
            g.create_dataset("point_prompts", data=point_prompts, chunks=point_prompts.shape)

    # TODO write the settings for the segmentation widget if necessary.
    # Commit the prompts for all the objects in the commit.
    object_ids = np.unique(seg[mask])
    if len(object_ids) == 1:  # We only have a single object.
        write_prompts(object_ids[0], viewer.layers["prompts"].data, viewer.layers["point_prompts"].data)
    else:
        # TODO this logic has to be updated to be compatible with the new batched prompting
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
    preserve_committed: bool = True,
    commit_path: Optional[Path] = None,
) -> None:
    """Widget for committing the segmented objects from automatic or interactive segmentation.

    Args:
        viewer: The napari viewer.
        layer: Select the layer to commit. Can be either 'current_object' to commit interacitve segmentation results.
            Or 'auto_segmentation' to commit automatic segmentation results.
        preserve_committed: If active already committted objects are not over-written by new commits.
        commit_path: Select a file path where the committed results and prompts will be saved.
            This feature is still experimental.
    """
    _, seg, mask, bb = _commit_impl(viewer, layer, preserve_committed)

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
    preserve_committed: bool = True,
    commit_path: Optional[Path] = None,
) -> None:
    """Widget for committing the objects from interactive tracking.

    Args:
        viewer: The napari viewer.
        layer: Select the layer to commit. Can be either 'current_object' to commit interacitve segmentation results.
            Or 'auto_segmentation' to commit automatic segmentation results.
        preserve_committed: If active already committted objects are not over-written by new commits.
        commit_path: Select a file path where the committed results and prompts will be saved.
            This feature is still experimental.
    """
    # Commit the segmentation layer.
    id_offset, seg, mask, bb = _commit_impl(viewer, layer, preserve_committed)

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
def settings_widget(
    cache_directory: Optional[Path] = util.get_cache_directory(),
) -> None:
    """Widget to update global micro_sam settings.

    Args:
        cache_directory: Select the path for the micro_sam cache directory. `$HOME/.cache/micro_sam`.
    """
    os.environ["MICROSAM_CACHEDIR"] = str(cache_directory)
    print(f"micro-sam cache directory set to: {cache_directory}")


def _generate_message(message_type, message) -> bool:
    """
    Displays a message dialog based on the provided message type.

    Args:
        message_type (str): The type of message to display. Valid options are:
            - "error": Displays a critical error message with an "Ok" button.
            - "info": Displays an informational message in a separate dialog box.
                 The user can dismiss it by either clicking "Ok" or closing the dialog.
        message (str): The message content to be displayed in the dialog.

    Returns:
        bool: A flag indicating whether the user aborted the operation based on the
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


def _validate_prompts(viewer: "napari.viewer.Viewer") -> bool:
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
    if _validate_prompts(viewer):
        return None

    shape = viewer.layers["current_object"].data.shape

    # get the current box and point prompts
    boxes, masks = vutil.shape_layer_to_prompts(viewer.layers["prompts"], shape)
    points, labels = vutil.point_layer_to_prompts(viewer.layers["point_prompts"], with_stop_annotation=False)

    predictor = AnnotatorState().predictor
    image_embeddings = AnnotatorState().image_embeddings
    seg = vutil.prompt_segmentation(
        predictor, points, labels, boxes, masks, shape, image_embeddings=image_embeddings,
        multiple_box_prompts=True, batched=batched, previous_segmentation=viewer.layers["current_object"].data,
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
    if _validate_prompts(viewer):
        return None

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
    """Segment object for the current prompts.

    Args:
        viewer: The napari viewer.
    """
    if _validate_embeddings(viewer):
        return None
    if _validate_prompts(viewer):
        return None
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
        section1_layout.addLayout(self._create_model_section())
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
        image_shape = self.image_selection.get_value().data.shape
        state.image_shape = image_shape

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

    def _update_model(self):
        print("Computed embeddings for", self.model_type)
        state = AnnotatorState()
        # Update the widget itself. This is necessary because we may have loaded
        # some settings from the embedding file and have to reflect them in the widget.
        vutil._sync_embedding_widget(
            self,
            model_type=self.model_type,
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
                state.widgets["autosegment"], self.model_type, self.custom_weights, update_decoder=with_decoder
            )
            # Load the AMG/AIS state if we have a 3d segmentation plugin.
            if state.widgets["autosegment"].volumetric and with_decoder:
                state.amg_state = vutil._load_is_state(state.embedding_path)
            elif state.widgets["autosegment"].volumetric and not with_decoder:
                state.amg_state = vutil._load_amg_state(state.embedding_path)

        # Set the default settings for this model in the nd-segmentation widget if it is part of
        # the currently used plugin.
        if "segment_nd" in state.widgets:
            vutil._sync_ndsegment_widget(state.widgets["segment_nd"], self.model_type, self.custom_weights)

    def _create_model_section(self):
        self.model_type = util._DEFAULT_MODEL

        self.model_options = list(util.models().urls.keys())
        # Filter out the decoders from the model list.
        self.model_options = [model for model in self.model_options if not model.endswith("decoder")]

        layout = QtWidgets.QVBoxLayout()
        self.model_dropdown, layout = self._add_choice_param(
            "model_type", self.model_type, self.model_options, title="Model:", layout=layout,
            tooltip=get_tooltip("embedding", "model")
        )
        return layout

    def _create_settings_widget(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        # Create UI for the device.
        self.device = "auto"
        device_options = ["auto"] + util._available_devices()

        self.device_dropdown, layout = self._add_choice_param("device", self.device, device_options,
                                                              tooltip=get_tooltip("embedding", "device"))
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

        # Create UI for prefering the decoder.
        self.prefer_decoder = True
        widget = self._add_boolean_param(
            "prefer_decoder", self.prefer_decoder, title="Prefer Segmentation Decoder",
            tooltip=get_tooltip("embedding", "prefer_decoder")
        )
        setting_values.layout().addWidget(widget)

        settings = _make_collapsible(setting_values, title="Embedding Settings")
        return settings

    def _validate_inputs(self):
        """
        Validates the inputs for the annotation process and returns a dictionary
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

        # Check if we have an existing embedding path.
        # If yes we check the data signature of these embeddings against the selected image
        # and we ask the user if they want to load these embeddings.
        if self.embeddings_save_path and os.listdir(self.embeddings_save_path):
            try:
                f = zarr.open(self.embeddings_save_path, "a")

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

    def __call__(self, skip_validate=False):
        # Validate user inputs.
        if not skip_validate and self._validate_inputs():
            return

        # Get the image.
        image = self.image_selection.get_value()

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

            state.initialize_predictor(
                image_data, model_type=self.model_type, save_path=save_path, ndim=ndim,
                device=self.device, checkpoint_path=self.custom_weights, tile_shape=tile_shape, halo=halo,
                prefer_decoder=self.prefer_decoder, pbar_init=pbar_init,
                pbar_update=lambda update: pbar_signals.pbar_update.emit(update),
            )
            pbar_signals.pbar_stop.emit()

        compute_image_embedding()
        self._update_model()
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
        self.projection = "points"
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
            pbar_signals.pbar_stop.emit()

            state.z_range = (z_min, z_max)
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
        if _validate_prompts(self._viewer):
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


def _instance_segmentation_impl(with_background, min_object_size, i=None, pbar_init=None, pbar_update=None, **kwargs):
    state = AnnotatorState()
    _handle_amg_state(state, i, pbar_init, pbar_update)

    seg = state.amg.generate(**kwargs)
    if len(seg) == 0:
        shape = state.image_shape
        seg = np.zeros(shape[-2:], dtype="uint32")
    else:
        seg = instance_segmentation.mask_data_to_segmentation(
            seg, with_background=with_background, min_object_size=min_object_size
        )
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

        # Create the UI element for with background.
        self.with_background = True
        settings.layout().addWidget(self._add_boolean_param(
            "with_background", self.with_background,
            tooltip=get_tooltip("autosegment", "with_background")
            ))

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

        # Add min_object_size and with_background
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

        # Add min_object_size and with_background
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
                self.with_background, self.min_object_size, i=i,
                pbar_init=pbar_init,
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
                seg = _instance_segmentation_impl(self.with_background, self.min_object_size, i=i, **kwargs)
                seg_max = seg.max()
                if seg_max == 0:
                    continue
                seg[seg != 0] += offset
                offset = seg_max + offset
                segmentation[i] = seg
                pbar_signals.pbar_update.emit(1)

            pbar_signals.pbar_reset.emit()
            segmentation = merge_instance_segmentation_3d(
                segmentation, beta=0.5, with_background=self.with_background,
                gap_closing=self.gap_closing, min_z_extent=self.min_extent,
                verbose=True, pbar_init=pbar_init,
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
            i = int(self._viewer.cursor.position[0])
            worker = self._run_segmentation_2d(kwargs, i=i)
        else:
            worker = self._run_segmentation_2d(kwargs)
        _select_layer(self._viewer, "auto_segmentation")
        return worker
