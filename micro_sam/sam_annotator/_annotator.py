import os
from datetime import datetime
from typing import List, Optional

import napari
import numpy as np
from joblib import dump, hash as joblib_hash, load
from magicgui.widgets import CheckBox, ComboBox, Container, FileEdit, FunctionGui, Label, PushButton, SpinBox, Widget
from napari.utils.notifications import show_info
from qtpy import QtWidgets

from . import _widgets as widgets
from . import util as vutil
from ._state import AnnotatorState
from ._tooltips import get_tooltip
from ..__version__ import __version__ as micro_sam_version

# Placeholder shapes used to seed the annotator layers before a real image is loaded.
# Only the dimensionality matters. The tool resets the values to the image shape on load.
PLACEHOLDER_SHAPE = {2: (256, 256), 3: (16, 256, 256)}


class _AnnotatorBase(QtWidgets.QScrollArea):
    """Base class for micro_sam annotation plugins.

    Implements the logic for the 2d, 3d and tracking annotator.
    The annotators differ in their data dimensionality and the widgets.
    """

    def _require_layers(self, layer_choices: Optional[List[str]] = None):

        # Check whether the image is initialized already. And use the image shape and scale for the layers.
        state = AnnotatorState()
        shape = self._shape if state.image_shape is None else state.image_shape

        # Add the label layers for the current object, the automatic segmentation and the committed segmentation.
        dummy_data = np.zeros(shape, dtype="uint32")
        image_scale = state.image_scale

        # Before adding new layers, we always check whether a layer with this name already exists or not.
        if "current_object" not in self._viewer.layers:
            if (
                layer_choices and "current_object" in layer_choices
            ):  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer("current_object")
            self._viewer.add_labels(data=dummy_data, name="current_object")
            if image_scale is not None:
                self._viewer.layers["current_object"].scale = image_scale

        if "auto_segmentation" not in self._viewer.layers:
            if (
                layer_choices and "auto_segmentation" in layer_choices
            ):  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer(
                    "auto_segmentation"
                )
            self._viewer.add_labels(data=dummy_data, name="auto_segmentation")
            if image_scale is not None:
                self._viewer.layers["auto_segmentation"].scale = image_scale

        if "committed_objects" not in self._viewer.layers:
            if (
                layer_choices and "committed_objects" in layer_choices
            ):  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer(
                    "committed_objects"
                )
            self._viewer.add_labels(data=dummy_data, name="committed_objects")
            # Randomize colors so it is easy to see when object committed.
            self._viewer.layers["committed_objects"].new_colormap()
            if image_scale is not None:
                self._viewer.layers["committed_objects"].scale = image_scale

        # Add the point layer for point prompts.
        self._point_labels = ["positive", "negative"]
        if "point_prompts" in self._viewer.layers:
            self._point_prompt_layer = self._viewer.layers["point_prompts"]
        else:
            self._point_prompt_layer = self._viewer.add_points(
                name="point_prompts",
                property_choices={"label": self._point_labels},
                border_color="label",
                border_color_cycle=vutil.LABEL_COLOR_CYCLE,
                symbol="o",
                face_color="transparent",
                border_width=0.5,
                size=12,
                ndim=self._ndim,
            )
            self._point_prompt_layer.border_color_mode = "cycle"

        if "prompts" in self._viewer.layers:
            self._shape_prompt_layer = self._viewer.layers["prompts"]
        else:
            # Add the shape layer for box and other shape prompts.
            self._shape_prompt_layer = self._viewer.add_shapes(
                face_color="transparent",
                edge_color="label",
                edge_color_cycle=vutil.LABEL_COLOR_CYCLE,
                edge_width=4,
                name="prompts",
                ndim=self._ndim,
                property_choices={"label": self._point_labels},
            )
            self._shape_prompt_layer.edge_color_mode = "cycle"

        # Migrate a pre-existing prompt layer and keep boxes / dense mask prompts green. Open
        # paths retain their positive / negative property and use the same colors as point prompts.
        if "label" not in self._shape_prompt_layer.properties:
            properties = dict(self._shape_prompt_layer.properties)
            properties["label"] = np.full(len(self._shape_prompt_layer.data), "positive", dtype=object)
            self._shape_prompt_layer.properties = properties
            current_properties = self._shape_prompt_layer.current_properties
            current_properties["label"] = np.array(["positive"])
            self._shape_prompt_layer.current_properties = current_properties
            self._shape_prompt_layer.edge_color_cycle = vutil.LABEL_COLOR_CYCLE
            self._shape_prompt_layer.edge_color = "label"
            self._shape_prompt_layer.edge_color_mode = "cycle"
        if not self._shape_prompt_layer.metadata.get("micro_sam_prompt_labels_configured", False):
            self._shape_prompt_layer.events.data.connect(vutil.normalize_prompt_shape_labels)
            self._shape_prompt_layer.events.mode.connect(vutil.sync_prompt_shape_current_color)
            self._shape_prompt_layer.events.current_properties.connect(vutil.sync_prompt_shape_current_color)
            self._shape_prompt_layer.metadata["micro_sam_prompt_labels_configured"] = True
        vutil.normalize_prompt_shape_labels(self._shape_prompt_layer)
        vutil.sync_prompt_shape_current_color(self._shape_prompt_layer)

    # Child classes have to implement this function and create a dictionary with the widgets.
    def _get_widgets(self):
        raise NotImplementedError(
            "The child classes of _AnnotatorBase have to implement _get_widgets."
        )

    def _create_embedding_widget(self):
        return widgets.EmbeddingWidget()

    def _create_widgets(self):
        # Create the embedding widget and connect all events related to it.
        self._embedding_widget = self._create_embedding_widget()
        # Connect events for the image selection box.
        self._viewer.layers.events.inserted.connect(
            self._embedding_widget.image_selection.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._embedding_widget.image_selection.reset_choices
        )
        # Connect the run button with the function to update the image.
        self._embedding_widget.run_button.clicked.connect(self._update_image)

        # Create the prompt widget. (The same for all plugins.)
        # Child plugins decide whether to expose it as a separate group (e.g. tracking) or to
        # embed it into another widget (e.g. the interactive segmentation widget).
        shape_prompt_layer = self._viewer.layers["prompts"]
        linked_layers = [shape_prompt_layer] if "label" in shape_prompt_layer.current_properties else None
        self._prompt_widget = widgets.create_prompt_menu(
            self._point_prompt_layer, self._point_labels, linked_layers=linked_layers
        )

        # Create the dictionary for the widgets and get the widgets of the child plugin.
        self._widgets = {"embeddings": self._embedding_widget}
        self._widgets.update(self._get_widgets())

    def _create_keybindings(self):
        @self._viewer.bind_key("s", overwrite=True)
        def _segment(viewer):
            self._widgets["segment"](viewer)

        # Note: we also need to over-write the keybindings for specific layers.
        # See https://github.com/napari/napari/issues/7302 for details.
        # Here, we need to over-write the 's' keybinding for both of the prompt layers.
        prompt_layer = self._viewer.layers["prompts"]
        point_prompt_layer = self._viewer.layers["point_prompts"]

        @prompt_layer.bind_key("s", overwrite=True)
        def _segment_prompts(event):
            self._widgets["segment"](self._viewer)

        @point_prompt_layer.bind_key("s", overwrite=True)
        def _segment_point_prompts(event):
            self._widgets["segment"](self._viewer)

        @prompt_layer.bind_key("t", overwrite=True)
        def _toggle_shape_prompt_label(event=None):
            vutil.toggle_label(self._point_prompt_layer, self._shape_prompt_layer)

        @point_prompt_layer.bind_key("t", overwrite=True)
        def _toggle_point_prompt_label(event=None):
            vutil.toggle_label(self._point_prompt_layer, self._shape_prompt_layer)

        @self._viewer.bind_key("c", overwrite=True)
        def _commit(viewer):
            self._widgets["commit"](viewer)

        @self._viewer.bind_key("t", overwrite=True)
        def _toggle_label(event=None):
            vutil.toggle_label(self._point_prompt_layer, self._shape_prompt_layer)

        @self._viewer.bind_key("Shift-C", overwrite=True)
        def _clear_annotations(viewer):
            self._widgets["clear"](viewer)

    # We could implement a better way of initializing the segmentation result,
    # so that instead of just passing a numpy array an existing layer from the napari
    # viewer can be chosen.
    # See https://github.com/computational-cell-analytics/micro-sam/issues/335
    def __init__(self, viewer: "napari.viewer.Viewer", ndim: int) -> None:
        """Create the annotator GUI.

        Args:
            viewer: The napari viewer.
            ndim: The number of spatial dimension of the image data (2 or 3).
        """
        super().__init__()
        self._viewer = viewer

        # Guard against re-entrant image-selection handling while we replace the image
        # layer during normalization (the replacement itself fires selection events).
        self._suppress_selection_rebuild = False

        # Add the layers for prompts and segmented obejcts.
        # Initialize with a dummy shape, which is reset to the correct shape once an image is set.
        self._ndim = ndim
        self._shape = PLACEHOLDER_SHAPE[ndim]
        self._require_layers()

        # Create all the widgets and populate the layout.
        self._create_widgets()
        AnnotatorState().widgets = self._widgets

        # Add the key bindings in common between all annotators.
        self._create_keybindings()

        # Build the scroll area content from the current set of widgets.
        self._populate_widget_layout()

    def _populate_widget_layout(self):
        # Build the scroll area content from the current set of widgets in 'self._widgets'.
        # This can be called again to rebuild the layout, e.g. when the image dimensionality
        # changes and the dimension-specific widgets have to be replaced.
        annotator_widget = QtWidgets.QWidget()
        annotator_widget.setLayout(QtWidgets.QVBoxLayout())
        for widget in self._widgets.values():
            widget_frame = QtWidgets.QGroupBox()
            widget_layout = QtWidgets.QVBoxLayout()
            if isinstance(widget, (Container, FunctionGui, Widget)):
                # This is a magicgui type and we need to get the native qt widget.
                widget_layout.addWidget(widget.native)
            else:
                # This is a qt type and we add the widget directly.
                widget_layout.addWidget(widget)
            widget_frame.setLayout(widget_layout)
            annotator_widget.layout().addWidget(widget_frame)

        self._annotator_widget = annotator_widget
        # Allow widget to resize within scroll area.
        self.setWidgetResizable(True)
        # Replacing the inner widget deletes the previous one (and its now-orphaned children).
        self.setWidget(self._annotator_widget)

    def _maybe_normalize_image_layer(self, image_layer, ndim=None):
        """Normalize the selected image layer in place so all layers stay aligned.

        Squeezes singleton axes and maps the channel axis to RGB (see
        ``util.prepare_annotation_image``). When this changes the data shape or the rgb
        flag, the image layer is replaced with the normalized version. Replacement passes
        the normalized array by reference (napari does not copy), so it does not duplicate
        the image buffer. Returns the (possibly new) layer and its spatial dimensionality.

        Args:
            image_layer: The selected napari image layer.
            ndim: Optional image-dimensionality override forwarded to ``prepare_annotation_image``
                (``None`` = auto-detect, else 2 or 3).
        """
        # Re-derive from the original (pre-normalization) data when available, so toggling the ndim
        # override reinterprets the raw image instead of an already-reduced one.
        source = image_layer.metadata.get("micro_sam_original_data", image_layer.data)
        data, detected_ndim, rgb = vutil.prepare_annotation_image(source, ndim=ndim)

        # Nothing changed: keep the existing layer.
        if data.shape == tuple(image_layer.data.shape) and bool(image_layer.rgb) == rgb:
            return image_layer, detected_ndim

        name = image_layer.name
        scale = image_layer.scale
        # Carry over the scale only when it still matches the normalized dimensionality.
        keep_scale = len(scale) == (data.ndim - 1 if rgb else data.ndim)

        self._suppress_selection_rebuild = True
        try:
            del self._viewer.layers[name]
            new_layer = self._viewer.add_image(data, name=name, rgb=rgb)
            if keep_scale:
                new_layer.scale = scale
            # Remember the original so a later ndim-override change re-derives from it.
            new_layer.metadata["micro_sam_original_data"] = source
        finally:
            self._suppress_selection_rebuild = False

        return new_layer, detected_ndim

    def _rebuild_for_ndim(self, ndim, force=False):
        # Rebuild the layers and dimension-specific widgets for a new dimensionality.
        # This supports loading an image whose dimensionality differs from the one the
        # annotator was created with, e.g. opening the plugin and then loading a 3D image.
        # 'force=True' rebuilds even when the dimensionality is unchanged - used on an image
        # change to reset the widgets (checkboxes, ...) and layers to a fresh-open state.
        if ndim == self._ndim and not force:
            return
        self._ndim = ndim
        self._shape = PLACEHOLDER_SHAPE[ndim]

        # Remove the existing micro_sam layers so they are recreated with the new ndim and shape.
        layer_names = ("current_object", "auto_segmentation", "committed_objects", "point_prompts", "prompts")
        for layer_name in layer_names:
            if layer_name in self._viewer.layers:
                del self._viewer.layers[layer_name]
        self._require_layers()

        # The prompt widget is bound to the point prompt layer, so it is recreated alongside it.
        shape_prompt_layer = self._viewer.layers["prompts"]
        linked_layers = [shape_prompt_layer] if "label" in shape_prompt_layer.current_properties else None
        self._prompt_widget = widgets.create_prompt_menu(
            self._point_prompt_layer, self._point_labels, linked_layers=linked_layers
        )

        # Rebuild the dimension-specific widgets, keeping the shared embedding widget.
        self._widgets = {"embeddings": self._embedding_widget}
        self._widgets.update(self._get_widgets())
        AnnotatorState().widgets = self._widgets

        # Rebuild the layout and rebind the keybindings to the new widgets and layers.
        self._populate_widget_layout()
        self._create_keybindings()

    def _update_image(self, segmentation_result=None):
        state = AnnotatorState()

        # Whether embeddings already exist and avoid clearing objects in layers.
        if state.skip_recomputing_embeddings:
            return

        # This is encountered when there is no image layer available / selected.
        # In this case, we need not update the image shape or check for changes.
        # NOTE: On code-level, this happens when '__init__' method is called by '_AnnotatorBase',
        #       where one of the first steps is to '_create_widgets', which reaches here.
        if state.image_shape is None:
            return

        # Update the image shape if it has changed.
        if state.image_shape != self._shape:
            if len(state.image_shape) != self._ndim:
                raise RuntimeError(
                    f"The dim of the annotator {self._ndim} does not match the image data of shape {state.image_shape}."
                )
            self._shape = state.image_shape

        # Before we reset the layers, we ensure all expected layers exist.
        self._require_layers()

        # Update the image scale.
        scale = state.image_scale

        # Reset all layers.
        self._viewer.layers["current_object"].data = np.zeros(
            self._shape, dtype="uint32"
        )
        self._viewer.layers["current_object"].scale = scale
        self._viewer.layers["auto_segmentation"].data = np.zeros(
            self._shape, dtype="uint32"
        )
        self._viewer.layers["auto_segmentation"].scale = scale

        if segmentation_result is None or segmentation_result is False:
            self._viewer.layers["committed_objects"].data = np.zeros(
                self._shape, dtype="uint32"
            )
        else:
            assert segmentation_result.shape == self._shape
            self._viewer.layers["committed_objects"].data = segmentation_result
        self._viewer.layers["committed_objects"].scale = scale

        self._viewer.layers["point_prompts"].scale = scale
        self._viewer.layers["prompts"].scale = scale

        vutil.clear_annotations(self._viewer, clear_segmentations=False)


class _ClassifierBase(QtWidgets.QScrollArea):
    """Base class for the pixel and object classifier plugins.

    Holds the shared GUI scaffolding (layers, label widget, settings/train widgets) and the
    train/predict/clear/load/save/spec logic. Subclasses declare the state-attribute names and
    implement the per-tool feature computation, label accumulation, training and prediction
    projection via a small set of hooks. Both tools have an 'annotations' and a 'prediction' label
    layer; the dimensionality follows the image (2d YX or 3d ZYX).
    """

    # Per-tool configuration, set by subclasses.
    rf_attr = None  # state attribute holding the trained classifier
    features_attr = None  # state attribute caching the features
    aux_attr = None  # state attribute caching the per-tool aux data (grid_shape | seg_ids)
    label_widget_title = "Label names:"
    max_components = 256  # PCA upper bound (256 pixel channels, 257 object features)
    tool_key = None  # "pixel" | "object", selects the tool-specific tooltips
    supports_apply_to_volume = True  # if False the tool always runs over the full image/volume

    #
    # Hooks the subclasses implement.
    #

    def _compute_features(self):
        """Return (features, aux) for the current image, caching on the state; (None, None) on failure."""
        raise NotImplementedError

    def _compute_training_labels(self, aux):
        """Return the per-feature-row training labels (0 = unlabeled), or None on failure."""
        raise NotImplementedError

    def _train(self, features, labels, previous_features, previous_labels, n_components, random_state):
        """Train and return the classifier."""
        raise NotImplementedError

    def _project_prediction(self, prediction, aux):
        """Map the flat per-row prediction back to an image-shaped array, or None on failure."""
        raise NotImplementedError

    def _extra_classification_sections(self):
        """Extra widgets/layouts shown above the settings dropdown (e.g. the segmentation selector)."""
        return []

    #
    # Shared state accessors and helpers.
    #

    def _invalidate_features(self, *args):
        state = AnnotatorState()
        setattr(state, self.features_attr, None)
        setattr(state, self.aux_attr, None)

    def _get_rf(self):
        return getattr(AnnotatorState(), self.rf_attr)

    def _set_rf(self, rf):
        setattr(AnnotatorState(), self.rf_attr, rf)

    def _current_slice(self):
        # The z index currently shown in the viewer (for 3d per-slice operations).
        return int(self._viewer.dims.current_step[0])

    def _get_cached_upsampler(self):
        # Load the AnyUp model once and cache it on the state (small model, reused across runs).
        from ..pixel_classification import get_anyup_upsampler
        state = AnnotatorState()
        if state.anyup_upsampler is None:
            device = getattr(state.predictor, "device", None)
            state.anyup_upsampler = get_anyup_upsampler(device=device)
        return state.anyup_upsampler

    def _resolve_anyup(self):
        # Returns (image, upsampler, ok). image/upsampler are None unless AnyUp is enabled.
        image, upsampler = None, None
        if self._get_use_anyup():
            try:
                upsampler = self._get_cached_upsampler()
            except ImportError as e:
                widgets._generate_message("error", str(e))
                return None, None, False
            image = self._viewer.layers["image"].data
        return image, upsampler, True

    def accumulate_batch_features(self):
        """Add the current image's labeled features to the running batch training set.

        Uses the per-tool feature and label hooks so it works for both classifiers. No-op when the
        features cannot be computed or when the current image has no annotations.
        """
        features, aux = self._compute_features()
        if features is None:
            return
        labels = self._compute_training_labels(aux)
        if labels is None:
            return
        valid = labels != 0
        if valid.sum() == 0:
            return
        state = AnnotatorState()
        new_features, new_labels = features[valid], labels[valid]
        if state.previous_features is None:
            state.previous_features, state.previous_labels = new_features, new_labels
        else:
            state.previous_features = np.concatenate([state.previous_features, new_features], axis=0)
            state.previous_labels = np.concatenate([state.previous_labels, new_labels], axis=0)

    #
    # Layers.
    #

    def _require_layers(self, layer_choices: Optional[List[str]] = None):
        # The dimensionality comes from the embedding widget ('state.ndim', RGB-aware) so the label
        # layers always match the image: 2d (YX) for 2d data, 3d (ZYX) for 3d data.
        state = AnnotatorState()
        if state.image_shape is None:
            ndim, shape = self._ndim, self._shape
        else:
            ndim = len(state.image_shape) if state.ndim is None else state.ndim
            shape = tuple(state.image_shape)[:ndim]

        dummy_data = np.zeros(shape, dtype="uint32")
        image_scale = None if state.image_scale is None else tuple(state.image_scale)[:ndim]

        # Drop any existing label layer whose dimensionality no longer matches the image. Reassigning
        # data of a different ndim corrupts the napari layer transforms, so we rebuild from scratch.
        for name in ("annotations", "prediction"):
            if name in self._viewer.layers and self._viewer.layers[name].data.ndim != len(shape):
                del self._viewer.layers[name]

        if "annotations" not in self._viewer.layers:
            if layer_choices and "annotations" in layer_choices:
                widgets._validation_window_for_missing_layer("annotations")
            annotation_layer = self._viewer.add_labels(data=dummy_data, name="annotations")
            if image_scale is not None:
                self._viewer.layers["annotations"].scale = image_scale
            # Reduce the brush size and set the default mode to "paint" brush mode.
            annotation_layer.brush_size = 3
            annotation_layer.mode = "paint"
            # Start painting with label id 1 (id 0 is the unlabeled background). napari already
            # defaults 'selected_label' to 1, so assigning 1 fires no change-event and the layer
            # controls spinbox stays at its displayed 0. Toggle to force the event so it shows 1.
            annotation_layer.selected_label = 2
            annotation_layer.selected_label = 1

        if "prediction" not in self._viewer.layers:
            if layer_choices and "prediction" in layer_choices:
                widgets._validation_window_for_missing_layer("prediction")
            self._viewer.add_labels(data=dummy_data, name="prediction")
            if image_scale is not None:
                self._viewer.layers["prediction"].scale = image_scale

        # Move 'annotations' to the top of the layer stack so scribbles are always visible above the
        # prediction, and make it the active layer so the controls (incl. the label id) show it and
        # the user can paint right away rather than on the last-added 'prediction' layer.
        self._viewer.layers.move(self._viewer.layers.index("annotations"), len(self._viewer.layers))
        self._viewer.layers.selection.active = self._viewer.layers["annotations"]

    def _update_image(self, segmentation_result=None):
        state = AnnotatorState()

        # Whether embeddings already exist and avoid clearing objects in layers.
        if state.skip_recomputing_embeddings:
            return

        if state.image_shape is None:
            return

        # Use the dimensionality determined by the embedding widget ('state.ndim', RGB-aware) so the label
        # layers always match the image: 2d (YX) for 2d data, 3d (ZYX) for 3d data. '_require_layers'
        # rebuilds any layer whose dimensionality no longer matches before we reset its data.
        self._ndim = len(state.image_shape) if state.ndim is None else state.ndim
        self._shape = tuple(state.image_shape)[:self._ndim]

        # The 'Apply to Volume' checkbox only makes sense for 3d data (and only for tools that have it).
        if self._apply_to_volume is not None:
            self._apply_to_volume.visible = self._ndim == 3

        # The features depend on the image, so they have to be recomputed for a new image.
        self._invalidate_features()

        # Before we reset the layers, we ensure all expected layers exist at the correct ndim.
        self._require_layers()

        scale = None if state.image_scale is None else tuple(state.image_scale)[:self._ndim]
        self._viewer.layers["annotations"].data = np.zeros(self._shape, dtype="uint32")
        self._viewer.layers["prediction"].data = np.zeros(self._shape, dtype="uint32")
        if scale is not None:
            self._viewer.layers["annotations"].scale = scale
            self._viewer.layers["prediction"].scale = scale

    #
    # Label-name widget.
    #

    def _create_label_widget(self):
        self._label_form = QtWidgets.QFormLayout()
        scroll_area = QtWidgets.QScrollArea()
        inner = QtWidgets.QWidget()
        inner.setLayout(self._label_form)
        scroll_area.setWidget(inner)
        scroll_area.setWidgetResizable(True)

        layout = QtWidgets.QVBoxLayout()
        header = QtWidgets.QLabel(self.label_widget_title)
        header.setToolTip(get_tooltip("classification", "label_names"))
        layout.addWidget(header)
        layout.addWidget(scroll_area)
        return layout

    def _make_label_role_widget(self, lbl):
        # Build the left-hand widget for a label row: a color swatch matching the annotation layer
        # color for this id, followed by the exact id ("ID <n>"). The id is stored as the object name
        # so it can be read back reliably when removing vanished rows.
        container = QtWidgets.QWidget()
        container.setObjectName(str(int(lbl)))
        row_layout = QtWidgets.QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(14, 14)
        color = self._viewer.layers["annotations"].get_color(int(lbl))
        r, g, b = (int(round(255 * c)) for c in color[:3])
        swatch.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #888;")

        row_layout.addWidget(swatch)
        row_layout.addWidget(QtWidgets.QLabel(f"ID {int(lbl)}"))
        row_layout.addStretch(1)
        return container

    def _refresh_label_widget(self):
        state = AnnotatorState()

        # Get the current label ids.
        ids = np.unique(self._viewer.layers["annotations"].data)[1:]
        if state.previous_labels is not None:
            ids = np.union1d(ids, np.unique(state.previous_labels))

        # Add new rows.
        for lbl in ids:
            if lbl in self._label_names:
                continue
            line = QtWidgets.QLineEdit(self._label_names.get(lbl, ""))
            line.setToolTip(get_tooltip("classification", "label_name_row"))
            self._label_names[lbl] = ""
            self._label_form.addRow(self._make_label_role_widget(lbl), line)
            line.textChanged.connect(lambda txt, lbl=lbl: self._label_names.__setitem__(lbl, txt))

        # Remove rows whose label vanished. 'removeRow' deletes the row's widgets itself.
        for row in reversed(range(self._label_form.rowCount())):
            lbl_id = int(self._label_form.itemAt(row, QtWidgets.QFormLayout.LabelRole).widget().objectName())
            if lbl_id not in ids:
                self._label_form.removeRow(row)
                self._label_names.pop(lbl_id, None)

    #
    # Settings and train widgets.
    #

    def _create_train_widget(self):
        # The 'Train and Predict' button is kept at the top level, outside the settings dropdown.
        # A single 'Apply to Volume' checkbox governs both 'Train and Predict' and 'Clear Annotations'
        # (shown only for 3d data, see '_update_image'): when checked they act on the whole volume,
        # when unchecked (the default) only on the current slice. Tools that do not support it
        # ('supports_apply_to_volume' False) omit the checkbox and always run over the full image/volume.
        train_button = PushButton(text="Train and Predict [Shift + T]")
        train_button.native.setToolTip(get_tooltip("classification", "train_button"))
        clear_button = PushButton(text="Clear Annotations [Shift + C]")
        clear_button.native.setToolTip(get_tooltip("classification", "clear_button"))

        apply_to_volume = None
        if self.supports_apply_to_volume:
            apply_to_volume = CheckBox(value=False, text="Apply to Volume")
            apply_to_volume.native.setToolTip(get_tooltip("classification", "apply_to_volume"))

        def _volume_value():
            return True if apply_to_volume is None else apply_to_volume.value

        train_button.clicked.connect(lambda: self._run_train_and_predict(_volume_value()))
        clear_button.clicked.connect(lambda: self._clear_annotations(_volume_value()))

        @self._viewer.bind_key("Shift-T", overwrite=True)
        def _train_and_predict(event=None):
            self._run_train_and_predict(_volume_value())

        @self._viewer.bind_key("Shift-C", overwrite=True)
        def _clear(event=None):
            self._clear_annotations(_volume_value())

        # The two buttons sit side-by-side and expand to share the row width equally. QSizePolicy.Policy
        # is nested in Qt6 and top-level in Qt5.
        button_row = Container(layout="horizontal", widgets=[train_button, clear_button], labels=False)
        button_row.native.layout().setContentsMargins(0, 0, 0, 0)
        size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
        for button in (train_button, clear_button):
            button.native.setSizePolicy(size_policy.Expanding, size_policy.Fixed)
        widgets_ = ([apply_to_volume] if apply_to_volume is not None else []) + [button_row]
        container = Container(widgets=widgets_, labels=False)
        return container, apply_to_volume

    def _create_classifier_io_widget(self):
        # Optional PCA dimensionality reduction. Off by default (all features used); checking it reveals
        # a number box for the count of top PCA components to reduce the features to before training.
        use_top_features = CheckBox(value=False, text="Choose top feature channels")
        use_top_features.native.setToolTip(get_tooltip("classification", f"use_top_features_{self.tool_key}"))

        # Optional AnyUp upsampling. When checked, the SAM/SAM2 embedding is upsampled with AnyUp using
        # the original image as guidance instead of plain interpolation. Toggling it changes the
        # features, so the cached features are cleared on change to force a recompute.
        use_anyup = CheckBox(value=False, text="Upsample with AnyUp")
        use_anyup.native.setToolTip(get_tooltip("classification", "use_anyup"))

        checkbox_row = Container(layout="horizontal", widgets=[use_top_features, use_anyup], labels=False)
        checkbox_row.native.layout().setContentsMargins(0, 0, 0, 0)
        checkbox_row.native.layout().addStretch(1)

        top_features_tooltip = get_tooltip("classification", f"top_features_{self.tool_key}")
        top_features = SpinBox(value=10, min=1, max=self.max_components, step=1)
        top_features.native.setToolTip(top_features_tooltip)
        self._top_features_spinbox = top_features
        top_features_label = Label(value="top feature channels:")
        top_features_label.native.setToolTip(top_features_tooltip)
        top_features_row = Container(
            layout="horizontal", widgets=[top_features_label, top_features], labels=False,
        )
        top_features_row.native.layout().setContentsMargins(0, 0, 0, 0)
        top_features_row.native.layout().addStretch(1)
        top_features_row.visible = False
        use_top_features.changed.connect(lambda checked: setattr(top_features_row, "visible", checked))
        use_anyup.changed.connect(self._invalidate_features)

        # Random seed. 'fixed' trains the random forest with a fixed seed so the prediction is
        # reproducible. 'random' leaves it unseeded so results vary slightly between runs. The exact
        # seed value does not matter, so this is a simple two-way choice rather than a numeric field.
        random_seed = ComboBox(value="fixed", choices=["fixed", "random"])
        random_seed_tooltip = get_tooltip("classification", "random_seed")
        random_seed.native.setToolTip(random_seed_tooltip)
        # Let the dropdown expand to fill the row width (no trailing stretch), matching the path widgets.
        size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
        random_seed.native.setSizePolicy(size_policy.Expanding, size_policy.Fixed)
        random_seed_label = Label(value="random seed:")
        random_seed_label.native.setToolTip(random_seed_tooltip)
        random_seed_row = Container(
            layout="horizontal", widgets=[random_seed_label, random_seed], labels=False,
        )
        random_seed_row.native.layout().setContentsMargins(0, 0, 0, 0)

        def get_n_components():
            return int(top_features.value) if use_top_features.value else 0

        def get_use_anyup():
            return bool(use_anyup.value)

        def get_random_state():
            return 0 if random_seed.value == "fixed" else None

        def set_options(use_top_features_val, n_top_features, use_anyup_val, random_seed_val=None):
            # Restore the option controls from a loaded classifier's spec.
            use_top_features.value = bool(use_top_features_val)
            if n_top_features:
                top_features.value = int(n_top_features)
            use_anyup.value = bool(use_anyup_val)
            if random_seed_val is not None:
                random_seed.value = random_seed_val

        self._get_n_components = get_n_components
        self._get_use_anyup = get_use_anyup
        self._get_random_state = get_random_state
        self._set_options = set_options

        # Classifier load and export. Load takes a stored model file. Export chooses a destination
        # folder (the current working directory by default) and saves the model there with an
        # auto-generated name.
        load_path = FileEdit(label="load classifier path:", mode="r", filter="*.joblib")
        load_path.line_edit.native.setPlaceholderText("/path/to/stored_model.joblib")
        load_path.native.setToolTip(get_tooltip("classification", "load_path"))
        load_button = PushButton(text="Load classifier")
        load_button.native.setToolTip(get_tooltip("classification", "load_button"))
        load_button.clicked.connect(lambda: self._load_rf(load_path.value))

        export_dir = FileEdit(label="export classifier folder:", mode="d", value=os.getcwd())
        export_dir.native.setToolTip(get_tooltip("classification", "export_dir"))
        export_button = PushButton(text="Export classifier")
        export_button.native.setToolTip(get_tooltip("classification", "export_button"))
        export_button.clicked.connect(lambda: self._save_rf(export_dir.value))

        rows = [checkbox_row, top_features_row, random_seed_row, load_path, load_button, export_dir, export_button]
        return Container(widgets=rows, labels=False)

    def _create_widgets(self):
        # Create the embedding widget and connect all events related to it.
        self._embedding_widget = widgets.ClassificationEmbeddingWidget()
        self._viewer.layers.events.inserted.connect(self._embedding_widget.image_selection.reset_choices)
        self._viewer.layers.events.removed.connect(self._embedding_widget.image_selection.reset_choices)
        self._embedding_widget.run_button.clicked.connect(self._update_image)

        self._train_and_predict_widget, self._apply_to_volume = self._create_train_widget()
        if self._apply_to_volume is not None:
            self._apply_to_volume.visible = False
        self._classifier_io_widget = self._create_classifier_io_widget()

        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())
        settings.layout().addWidget(self._classifier_io_widget.native)
        collapsible = widgets._make_collapsible(
            settings, title="Classification Settings", tooltip=get_tooltip("classification", "settings"),
        )

        # Any tool-specific sections (e.g. the object classifier's segmentation selector) sit above
        # the settings dropdown, with the train/predict button below it.
        classification_section = QtWidgets.QWidget()
        classification_section.setLayout(QtWidgets.QVBoxLayout())
        for extra in self._extra_classification_sections():
            if isinstance(extra, QtWidgets.QLayout):
                extra_container = QtWidgets.QWidget()
                extra_container.setLayout(extra)
                classification_section.layout().addWidget(extra_container)
            else:
                classification_section.layout().addWidget(extra)
        classification_section.layout().addWidget(collapsible)
        classification_section.layout().addWidget(self._train_and_predict_widget.native)

        self._label_widget = self._create_label_widget()

        self._widgets = {
            "embeddings": self._embedding_widget,
            "classification": classification_section,
            "label_widget": self._label_widget,
        }

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        """Create the classifier GUI.

        Args:
            viewer: The napari viewer.
        """
        super().__init__()
        self._viewer = viewer
        self._annotator_widget = QtWidgets.QWidget()
        self._annotator_widget.setLayout(QtWidgets.QVBoxLayout())

        # Add the layers for annotations and prediction. Initialize with a dummy shape, which is reset
        # to the correct shape once an image is set.
        self._shape = (256, 256)
        self._ndim = len(self._shape)
        self._require_layers()

        # Create all the widgets and add them to the layout.
        self._label_names = {}  # The names for the labels.
        self._create_widgets()

        for widget_name, widget in self._widgets.items():
            widget_frame = QtWidgets.QGroupBox()
            widget_layout = QtWidgets.QVBoxLayout()
            if isinstance(widget, (Container, FunctionGui, Widget)):
                # This is a magicgui type and we need to get the native qt widget.
                widget_layout.addWidget(widget.native)
            elif isinstance(widget, QtWidgets.QLayout):
                widget_layout.addLayout(widget)
            else:
                # This is a qt type and we add the widget directly.
                widget_layout.addWidget(widget)
            widget_frame.setLayout(widget_layout)
            self._annotator_widget.layout().addWidget(widget_frame)

        # Connect the label layer and the refresh function.
        self._refresh_label_widget()

        # Set the expected annotator class and widgets on the state.
        state = AnnotatorState()
        state.annotator = self
        state.widgets = self._widgets

        self.setWidgetResizable(True)
        self.setWidget(self._annotator_widget)

    #
    # Train / predict / clear / load / save / spec.
    #

    def _update_feature_cap(self, n_features):
        """Clamp the PCA top-features control to the actual embedding feature dimension."""
        spinbox = getattr(self, "_top_features_spinbox", None)
        if spinbox is None:
            return
        spinbox.max = int(n_features)
        if spinbox.value > n_features:
            spinbox.value = int(n_features)

    def _run_train_and_predict(self, apply_to_volume=True):
        state = AnnotatorState()
        self._require_layers()

        features, aux = self._compute_features()
        if features is None:
            return None
        self._update_feature_cap(features.shape[1])
        labels = self._compute_training_labels(aux)
        if labels is None:
            return None

        previous_features, previous_labels = state.previous_features, state.previous_labels
        if (labels == 0).all() and (previous_labels is None):
            return widgets._generate_message("error", "You have not provided any annotations.")

        n_components = self._get_n_components()
        random_state = self._get_random_state()
        rf = self._train(features, labels, previous_features, previous_labels, n_components, random_state)
        self._set_rf(rf)

        self._predict_and_show(rf, features, aux, apply_to_volume=apply_to_volume)

    def _predict_and_show(self, rf, features, aux, apply_to_volume=True):
        try:
            pred = rf.predict(features)
        except ValueError:
            return widgets._generate_message(
                "error", "The loaded classifier does not match the current embeddings. Use the same model type."
            )
        prediction = self._project_prediction(pred, aux)
        if prediction is None:
            return None
        layer = self._viewer.layers["prediction"]
        if apply_to_volume or prediction.ndim < 3:
            layer.data = prediction
        else:
            data = layer.data if layer.data.shape == prediction.shape else np.zeros_like(prediction)
            z = self._current_slice()
            data[z] = prediction[z]
            layer.data = data
        self._refresh_label_widget()

    def _clear_annotations(self, apply_to_volume=True):
        # Remove the annotation scribbles and the prediction: the whole volume, or only the current
        # slice for 3d data when 'apply_to_volume' is False.
        if "annotations" not in self._viewer.layers:
            return widgets._generate_message("error", "There is no annotations layer to clear.")
        whole = apply_to_volume or self._viewer.layers["annotations"].data.ndim < 3
        for name in ("annotations", "prediction"):
            if name not in self._viewer.layers:
                continue
            layer = self._viewer.layers[name]
            if whole:
                layer.data = np.zeros_like(layer.data)
            else:
                data = layer.data
                data[self._current_slice()] = 0
                layer.data = data
            layer.refresh()

    def _load_rf(self, model_path):
        model_path = str(model_path)
        if not model_path or not os.path.exists(model_path):
            return widgets._generate_message("error", "You have to provide a valid path to load the classifier.")

        # Stored as {'rf': ..., 'model_spec': ...}; older files are a bare classifier (no spec).
        obj = load(model_path)
        if isinstance(obj, dict) and "rf" in obj:
            rf, spec = obj["rf"], obj.get("model_spec", {})
        else:
            rf, spec = obj, {}
        self._set_rf(rf)
        if spec:
            self._restore_from_spec(spec)

        # Predict on the current image if embeddings are available.
        features, aux = self._compute_features()
        if features is None:
            return None
        self._require_layers()
        self._predict_and_show(rf, features, aux)

    def _resolve_export_path(self, export_dir, rf):
        # Auto-generated name: <image>_<nclasses>classes_<date>_<time>_<hash>.joblib in the chosen folder.
        state = AnnotatorState()
        name = state.image_name or (self._viewer.layers["image"].name if "image" in self._viewer.layers else "image")
        name = os.path.splitext(os.path.basename(str(name)))[0]
        n_classes = len(rf.classes_)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{name}_{n_classes}classes_{stamp}_{joblib_hash(rf)[:8]}.joblib"
        base_dir = str(export_dir).strip() or os.getcwd()
        return os.path.join(base_dir, fname)

    def _save_rf(self, export_dir):
        rf = self._get_rf()
        if rf is None:
            return widgets._generate_message("error", "You have not trained or loaded a classifier yet.")
        out_path = self._resolve_export_path(export_dir, rf)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Store the classifier together with its specs, so a load can restore the full config.
        dump({"rf": rf, "model_spec": self._classifier_spec(rf)}, out_path)
        show_info(f"Exported classifier to {out_path}")

    def _gather_class_names(self, rf):
        # User-provided class names keyed by class id, for the classes the classifier knows. None if none.
        names = self._label_names or {}
        class_names = {int(k): v for k, v in names.items() if v and int(k) in rf.classes_}
        return class_names or None

    def _classifier_spec(self, rf):
        # Specs stored alongside the classifier so a load can restore the full config.
        state = AnnotatorState()
        ew = state.widgets.get("embeddings")
        tiling_on = getattr(ew, "tiling", "no") == "yes"
        n_components = self._get_n_components()
        use_anyup = bool(self._get_use_anyup())
        random_seed = "random" if self._get_random_state() is None else "fixed"
        # The GUI sets 'ew.model_type' only after it computes embeddings. Use the predictor's
        # model_type (always set) instead, so a CLI-launched session still records the model.
        model_type = getattr(ew, "model_type", None) or getattr(state.predictor, "model_type", None)
        return {
            "micro_sam_version": micro_sam_version,
            "model_family": getattr(ew, "model_family", None),
            "model_size": getattr(ew, "model_size", None),
            "model_type": model_type,
            "custom_weights": getattr(ew, "custom_weights", None) or None,
            "tiling": getattr(ew, "tiling", "no"),
            "tile_shape": [ew.tile_x, ew.tile_y] if tiling_on else None,
            "halo": [ew.halo_x, ew.halo_y] if tiling_on else None,
            "use_top_features": n_components > 0,
            "n_top_features": n_components if n_components > 0 else None,
            "upsampling": "anyup" if use_anyup else "interpolation",
            "random_seed": random_seed,
            "ndim": getattr(self, "_ndim", None),
            "class_ids": [int(c) for c in rf.classes_],
            "class_names": self._gather_class_names(rf),
        }

    def _restore_from_spec(self, spec):
        # Push a loaded classifier's stored specs back into the widgets, so the session matches the
        # config the classifier was trained with. Warn (but don't recompute) if the current embeddings
        # were computed with a different model.
        state = AnnotatorState()
        ew = state.widgets.get("embeddings")

        # Model family / size are set directly from the stored strings (the classifier widget's family
        # names don't match '_sync_embedding_widget's model_type -> family inference). Family first, since
        # the size options are rebuilt when it changes.
        family, size = spec.get("model_family"), spec.get("model_size")
        if ew is not None and family is not None:
            # The classification widget routes the family to the primary or advanced selector. Other
            # widgets set the family dropdown directly.
            setter = getattr(ew, "set_model_family_size", None)
            if setter is not None:
                setter(family, size)
            else:
                ew.model_family_dropdown.setCurrentText(family)
                if size is not None:
                    ew.model_size_dropdown.setCurrentText(size)

        # Tiling, tile and halo params and custom weights via the shared sync helper (these field names match).
        # 'ew.model_type' can be unset until the GUI computes embeddings, so read it via getattr.
        if ew is not None:
            vutil._sync_embedding_widget(
                ew, model_type=spec.get("model_type") or getattr(ew, "model_type", None),
                save_path=None, checkpoint_path=spec.get("custom_weights"),
                device=None, tile_shape=spec.get("tile_shape"), halo=spec.get("halo"),
            )

        # Top-feature selection, AnyUp/interpolation upsampling and the random-seed mode.
        self._set_options(
            spec.get("use_top_features", False), spec.get("n_top_features"),
            spec.get("upsampling") == "anyup", spec.get("random_seed"),
        )

        # Class names, restored into the label widget.
        class_names = spec.get("class_names") or {}
        if class_names:
            self._label_names.update({int(k): v for k, v in class_names.items()})
            self._refresh_label_widget()

        # Warn if the current embeddings were computed with a different model than the classifier expects.
        stored_model = spec.get("model_type")
        current_model = getattr(state.predictor, "model_type", None) if state.predictor is not None else None
        if stored_model is not None and current_model is not None and stored_model != current_model:
            show_info(
                f"The loaded classifier was trained with '{stored_model}', but the current embeddings use "
                f"'{current_model}'. Recompute the embeddings with the restored settings before predicting."
            )
