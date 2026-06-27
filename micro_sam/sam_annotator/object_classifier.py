import os
from datetime import datetime
from joblib import dump, hash as joblib_hash, load
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Tuple, Union

import imageio.v3 as imageio
import napari
import numpy as np
import torch

from magicgui import magicgui
from magicgui.widgets import (
    CheckBox, ComboBox, Widget, Container, FileEdit, FunctionGui, Label, PushButton, SpinBox
)
from napari.utils.notifications import show_info
from qtpy import QtWidgets

from skimage.measure import regionprops_table
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import util
from ..__version__ import __version__ as micro_sam_version
from ..v2.util import DEFAULT_MODEL
from ..object_classification import compute_object_features, project_prediction_to_segmentation
from ._state import AnnotatorState
from . import _widgets as widgets
from .util import _sync_embedding_widget

# Object features are the object area plus the per-channel mean of the 256-channel SAM/SAM2 image
# embedding, i.e. 257 features. PCA can reduce to at most this many components.
OBJECT_FEATURES = 257
INTERNAL_LABEL_LAYER_NAMES = {"annotations", "prediction"}

#
# Utility functionality.
# Some of this could be refactored to general purpose functionality that can also
# be used for inference with the trained classifier.
#


def _accumulate_labels(segmentation, annotations):

    def majority_label(mask, annotation):
        ids, counts = np.unique(annotation[mask], return_counts=True)
        if len(ids) == 1 and ids[0] == 0:
            return 0
        if ids[0] == 0:
            ids, counts = ids[1:], counts[1:]
        return ids[np.argmax(counts)]

    all_features = regionprops_table(
        segmentation, intensity_image=annotations, properties=("label",),
        extra_properties=[majority_label],
    )
    return all_features["majority_label"].astype("int")


def _train_rf(features, labels, previous_features=None, previous_labels=None, n_components=None, **rf_kwargs):
    assert len(features) == len(labels)
    valid = labels != 0
    X, y = features[valid], labels[valid]

    if previous_features is not None:
        assert previous_labels is not None and len(previous_features) == len(previous_labels)
        X = np.concatenate([previous_features, X], axis=0)
        y = np.concatenate([previous_labels, y], axis=0)

    rf = RandomForestClassifier(**rf_kwargs)

    # Optionally reduce the features to the top-n PCA components. n_components is clamped to the
    # number of features and samples; if it covers all features we skip PCA and use the plain RF.
    # Object features mix area (large magnitude) with embedding means (small), so we standardize
    # them before PCA to keep area from dominating the components.
    n_features = X.shape[1]
    k = min(int(n_components), n_features, len(X)) if n_components else 0
    if 0 < k < n_features:
        model = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=k)), ("rf", rf)])
    else:
        model = rf

    model.fit(X, y)
    return model


def _get_cached_upsampler():
    # Load the AnyUp model once and cache it on the state (small model, reused across runs).
    from ..pixel_classification import get_anyup_upsampler
    state = AnnotatorState()
    if state.anyup_upsampler is None:
        device = getattr(state.predictor, "device", None)
        state.anyup_upsampler = get_anyup_upsampler(device=device)
    return state.anyup_upsampler


def _compute_object_features_if_needed(viewer):
    # Returns (features, seg_ids) for the current image+segmentation, computing/caching if needed.
    state = AnnotatorState()
    if state.object_features is None:
        if widgets._validate_embeddings(viewer):
            return None, None
        segmentation_layer = _get_selected_segmentation_layer()
        if segmentation_layer is None:
            return None, None
        segmentation = segmentation_layer.data
        use_anyup = getattr(state.annotator, "_get_use_anyup", None)
        image, upsampler = None, None
        if use_anyup is not None and use_anyup():
            try:
                upsampler = _get_cached_upsampler()
            except ImportError as e:
                widgets._generate_message("error", str(e))
                return None, None
            image = viewer.layers["image"].data
        seg_ids, features = compute_object_features(
            state.image_embeddings, segmentation, image=image, upsampler=upsampler,
        )
        state.seg_ids, state.object_features = seg_ids, features
    return state.object_features, state.seg_ids


def _predict_and_show(viewer, rf, features, seg_ids, apply_to_volume=True):
    state = AnnotatorState()
    segmentation_layer = _get_selected_segmentation_layer()
    if segmentation_layer is None:
        return None
    segmentation = segmentation_layer.data
    try:
        pred = rf.predict(features)
    except ValueError:
        return widgets._generate_message(
            "error", "The loaded classifier does not match the current embeddings. Use the same model type."
        )
    prediction = project_prediction_to_segmentation(segmentation, pred, seg_ids)
    layer = viewer.layers["prediction"]
    if apply_to_volume or prediction.ndim < 3:
        layer.data = prediction
    else:
        data = layer.data if layer.data.shape == prediction.shape else np.zeros_like(prediction)
        z = _current_slice(viewer)
        data[z] = prediction[z]
        layer.data = data
    state.annotator._refresh_label_widget()


def _run_train_and_predict(viewer, apply_to_volume=True):
    # Get the object features and the annotations.
    state = AnnotatorState()
    state.annotator._require_layers()
    annotations = viewer.layers["annotations"].data
    segmentation_layer = _get_selected_segmentation_layer()
    if segmentation_layer is None:
        return None
    segmentation = segmentation_layer.data

    features, seg_ids = _compute_object_features_if_needed(viewer)
    if features is None:
        return None

    previous_features, previous_labels = state.previous_features, state.previous_labels
    labels = _accumulate_labels(segmentation, annotations)
    if (labels == 0).all() and (previous_labels is None):
        return widgets._generate_message("error", "You have not provided any annotations.")

    # Optionally reduce to the top-n PCA feature channels, read from the settings widget.
    get_n_components = getattr(state.annotator, "_get_n_components", None)
    n_components = get_n_components() if get_n_components is not None else 0

    # Run RF training and store it in the state.
    state.object_rf = _train_rf(
        features, labels, previous_features=previous_features, previous_labels=previous_labels,
        n_estimators=200, max_depth=10, n_jobs=cpu_count(), n_components=n_components,
    )

    # Run and set the prediction.
    _predict_and_show(viewer, state.object_rf, features, seg_ids, apply_to_volume=apply_to_volume)


def _restore_from_spec(spec):
    # Push a loaded classifier's stored specs back into the widgets, so the session matches the
    # config the classifier was trained with. Warn (but don't recompute) if the current embeddings
    # were computed with a different model.
    state = AnnotatorState()
    ann = state.annotator
    ew = state.widgets.get("embeddings")

    # Model family / size are set directly from the stored strings (the classifier widget's family
    # names don't match '_sync_embedding_widget's model_type -> family inference). Family first, since
    # the size options are rebuilt when it changes.
    family, size = spec.get("model_family"), spec.get("model_size")
    if ew is not None and family is not None:
        ew.model_family_dropdown.setCurrentText(family)
        if size is not None:
            ew.model_size_dropdown.setCurrentText(size)

    # Tiling, tile/halo params and custom weights via the shared sync helper (these field names do
    # match). 'model_type' is only used by the helper for the tiling/custom-weights side here.
    if ew is not None:
        _sync_embedding_widget(
            ew, model_type=spec.get("model_type") or ew.model_type,
            save_path=None, checkpoint_path=spec.get("custom_weights"),
            device=None, tile_shape=spec.get("tile_shape"), halo=spec.get("halo"),
        )

    # Top-feature selection and AnyUp/interpolation upsampling.
    if getattr(ann, "_set_options", None) is not None:
        ann._set_options(
            spec.get("use_top_features", False), spec.get("n_top_features"),
            spec.get("upsampling") == "anyup",
        )

    # Class names, restored into the label widget.
    class_names = spec.get("class_names") or {}
    if class_names and getattr(ann, "_label_names", None) is not None:
        ann._label_names.update({int(k): v for k, v in class_names.items()})
        ann._refresh_label_widget()

    # Warn if the current embeddings were computed with a different model than the classifier expects.
    stored_model = spec.get("model_type")
    current_model = getattr(state.predictor, "model_type", None) if state.predictor is not None else None
    if stored_model is not None and current_model is not None and stored_model != current_model:
        show_info(
            f"Loaded classifier was trained with '{stored_model}', but the current embeddings use "
            f"'{current_model}'. Recompute the embeddings with the restored settings before predicting."
        )


def _load_rf(viewer, model_path):
    state = AnnotatorState()
    model_path = str(model_path)
    if not model_path or not os.path.exists(model_path):
        return widgets._generate_message("error", "You have to provide a valid path to load the classifier.")

    # Stored as {'rf': ..., 'model_spec': ...}; older files are a bare classifier (no spec).
    obj = load(model_path)
    if isinstance(obj, dict) and "rf" in obj:
        state.object_rf, spec = obj["rf"], obj.get("model_spec", {})
    else:
        state.object_rf, spec = obj, {}
    if spec:
        _restore_from_spec(spec)

    # Predict on the current image if embeddings are available.
    features, seg_ids = _compute_object_features_if_needed(viewer)
    if features is None:
        return None
    state.annotator._require_layers()
    _predict_and_show(viewer, state.object_rf, features, seg_ids)


def _get_selected_segmentation_layer():
    state = AnnotatorState()
    segmentation_layer = None if state.segmentation_selection is None else state.segmentation_selection.get_value()
    if segmentation_layer is None:
        widgets._generate_message("error", "You have to select a segmentation labels layer.")
        return None
    return segmentation_layer


def _resolve_export_path(viewer, export_dir, rf):
    # The classifier is saved into the chosen folder (defaulting to the current working directory)
    # with a descriptive auto-generated name: <image>_<nclasses>classes_<date>_<time>_<hash>.joblib.
    state = AnnotatorState()
    name = state.image_name or (viewer.layers["image"].name if "image" in viewer.layers else "image")
    name = os.path.splitext(os.path.basename(str(name)))[0]
    n_classes = len(rf.classes_)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{name}_{n_classes}classes_{stamp}_{joblib_hash(rf)[:8]}.joblib"
    base_dir = str(export_dir).strip() or os.getcwd()
    return os.path.join(base_dir, fname)


def _gather_class_names(rf):
    # User-provided class names keyed by class id, for the classes the classifier knows. None if none.
    state = AnnotatorState()
    names = getattr(state.annotator, "_label_names", None) or {}
    class_names = {int(k): v for k, v in names.items() if v and int(k) in rf.classes_}
    return class_names or None


def _classifier_spec(rf):
    # Specs stored alongside the classifier so a load can restore the full config.
    state = AnnotatorState()
    ann = state.annotator
    ew = state.widgets.get("embeddings")
    tiling_on = getattr(ew, "tiling", "no") == "yes"
    n_components = ann._get_n_components() if getattr(ann, "_get_n_components", None) else 0
    use_anyup = bool(ann._get_use_anyup()) if getattr(ann, "_get_use_anyup", None) else False
    return {
        "micro_sam_version": micro_sam_version,
        "model_family": getattr(ew, "model_family", None),
        "model_size": getattr(ew, "model_size", None),
        "model_type": getattr(ew, "model_type", None),
        "custom_weights": getattr(ew, "custom_weights", None) or None,
        "tiling": getattr(ew, "tiling", "no"),
        "tile_shape": [ew.tile_x, ew.tile_y] if tiling_on else None,
        "halo": [ew.halo_x, ew.halo_y] if tiling_on else None,
        "use_top_features": n_components > 0,
        "n_top_features": n_components if n_components > 0 else None,
        "upsampling": "anyup" if use_anyup else "interpolation",
        "ndim": getattr(ann, "_ndim", None),
        "class_ids": [int(c) for c in rf.classes_],
        "class_names": _gather_class_names(rf),
    }


def _save_rf(viewer, export_dir):
    state = AnnotatorState()
    if state.object_rf is None:
        return widgets._generate_message("error", "You have not trained or loaded a classifier yet.")
    out_path = _resolve_export_path(viewer, export_dir, state.object_rf)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Store the classifier together with its specs, so a load can restore the full config.
    dump({"rf": state.object_rf, "model_spec": _classifier_spec(state.object_rf)}, out_path)
    show_info(f"Exported classifier to {out_path}")


def _current_slice(viewer):
    # The z index currently shown in the viewer (for 3d per-slice operations).
    return int(viewer.dims.current_step[0])


def _clear_annotations(viewer, apply_to_volume=True):
    # Remove the annotation scribbles and the prediction: the whole volume, or only the current
    # slice for 3d data when 'apply_to_volume' is False.
    if "annotations" not in viewer.layers:
        return widgets._generate_message("error", "There is no annotations layer to clear.")
    whole = apply_to_volume or viewer.layers["annotations"].data.ndim < 3
    for name in ("annotations", "prediction"):
        if name not in viewer.layers:
            continue
        layer = viewer.layers[name]
        if whole:
            layer.data = np.zeros_like(layer.data)
        else:
            data = layer.data
            data[_current_slice(viewer)] = 0
            layer.data = data
        layer.refresh()


def _create_train_widget(viewer):
    # The 'Train and predict' button is kept at the top level, outside the settings dropdown.
    # A single 'Apply to Volume' checkbox governs both 'Train and predict' and 'Clear Annotations'
    # (shown only for 3d data, see '_update_image'): when checked they act on the whole volume,
    # when unchecked (the default) only on the current slice.
    train_button = PushButton(text="Train and predict [Shift + T]")
    clear_button = PushButton(text="Clear Annotations [C]")
    apply_to_volume = CheckBox(value=False, text="Apply to Volume")
    apply_to_volume.native.setToolTip(
        "Apply 'Train and predict' and 'Clear Annotations' to the whole volume. When unchecked, "
        "they act only on the current slice (training always uses all annotations). Only relevant for 3d data."
    )
    train_button.clicked.connect(lambda: _run_train_and_predict(viewer, apply_to_volume.value))
    clear_button.clicked.connect(lambda: _clear_annotations(viewer, apply_to_volume.value))

    @viewer.bind_key("Shift-T", overwrite=True)
    def _train_and_predict(event=None):
        _run_train_and_predict(viewer, apply_to_volume.value)

    @viewer.bind_key("c", overwrite=True)
    def _clear(event=None):
        _clear_annotations(viewer, apply_to_volume.value)

    # The shared "Apply to Volume" checkbox sits on top, left-aligned with the settings dropdown
    # above and the buttons below; the two buttons sit side-by-side, packed to the left.
    button_row = Container(layout="horizontal", widgets=[train_button, clear_button], labels=False)
    button_row.native.layout().setContentsMargins(0, 0, 0, 0)
    # Let both buttons expand to share the row width equally (instead of staying at their
    # minimum size). QSizePolicy.Policy is nested in Qt6 and top-level in Qt5.
    size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
    for button in (train_button, clear_button):
        button.native.setSizePolicy(size_policy.Expanding, size_policy.Fixed)
    container = Container(widgets=[apply_to_volume, button_row], labels=False)
    return container, apply_to_volume


def _create_classifier_io_widget(viewer):
    # Optional PCA dimensionality reduction. The checkbox is off by default, in which case all
    # object features are used and PCA is never applied. Checking it reveals a number box for the
    # count of top PCA components to reduce the features to before training. Object features are
    # the area plus the 256 per-channel embedding means, i.e. 257, which is the maximum.
    use_top_features = CheckBox(value=False, text="Choose top feature channels")
    use_top_features.native.setToolTip(
        "Reduce the object features to their most informative components via PCA before training. "
        "When unchecked, all features are used and no PCA is applied."
    )

    # Optional AnyUp upsampling. When checked, the SAM/SAM2 embedding is upsampled with AnyUp using
    # the original image as guidance instead of plain interpolation. Toggling it changes the
    # features, so the cached features are cleared on change to force a recompute.
    use_anyup = CheckBox(value=False, text="Upsample with AnyUp")
    use_anyup.native.setToolTip(
        "Use AnyUp to upsample the embedding with the original image as guidance, for sharper "
        "features near object boundaries. When unchecked, plain interpolation is used."
    )

    # The two option checkboxes sit side by side (PCA on the left, AnyUp on the right). A trailing
    # stretch packs them flush left; zeroing the nested-container margins lines them up with the
    # path fields below (which otherwise get double-indented by the nested container).
    checkbox_row = Container(layout="horizontal", widgets=[use_top_features, use_anyup], labels=False)
    checkbox_row.native.layout().setContentsMargins(0, 0, 0, 0)
    checkbox_row.native.layout().addStretch(1)

    # The PCA component count appears on its own row below, revealed when the checkbox is ticked.
    top_features = SpinBox(value=10, min=1, max=OBJECT_FEATURES, step=1)
    top_features.native.setToolTip(
        f"Number of top PCA components to reduce the object features to, between 1 and {OBJECT_FEATURES} "
        "(object area plus the 256 per-channel embedding means)."
    )
    top_features_row = Container(
        layout="horizontal", widgets=[Label(value="top feature channels:"), top_features], labels=False,
    )
    top_features_row.native.layout().setContentsMargins(0, 0, 0, 0)
    top_features_row.native.layout().addStretch(1)
    top_features_row.visible = False
    use_top_features.changed.connect(lambda checked: setattr(top_features_row, "visible", checked))

    def get_n_components():
        return int(top_features.value) if use_top_features.value else 0

    def _invalidate_features(*args):
        state = AnnotatorState()
        state.object_features, state.seg_ids = None, None
    use_anyup.changed.connect(_invalidate_features)

    def get_use_anyup():
        return bool(use_anyup.value)

    def set_options(use_top_features_val, n_top_features, use_anyup_val):
        # Restore the option controls from a loaded classifier's spec.
        use_top_features.value = bool(use_top_features_val)
        if n_top_features:
            top_features.value = int(n_top_features)
        use_anyup.value = bool(use_anyup_val)

    # Classifier load/export, with separate paths (load a stored model, retrain, then export
    # the current one elsewhere). The path fields follow the same path-selector style as the
    # custom weights in the embedding widget.
    load_path = FileEdit(label="load classifier path:", mode="r", filter="*.joblib")
    load_path.line_edit.native.setPlaceholderText("/path/to/stored_model.joblib")
    load_path.native.setToolTip(
        "Path to a stored classifier (.joblib) to load and apply to the current image."
    )
    load_button = PushButton(text="Load classifier")
    load_button.native.setToolTip("Load the selected classifier and predict on the current image.")
    load_button.clicked.connect(lambda: _load_rf(viewer, load_path.value))

    # Export saves to the chosen folder (pre-filled with the current working directory) with an
    # auto-generated name: <image>_<nclasses>classes_<date>_<time>_<hash>.joblib.
    export_dir = FileEdit(label="export classifier folder:", mode="d", value=os.getcwd())
    export_dir.native.setToolTip(
        "Folder where the trained classifier is saved. The file name is generated automatically as "
        "<image>_<nclasses>classes_<date>_<time>_<hash>.joblib. Defaults to the current working directory."
    )
    export_button = PushButton(text="Export classifier")
    export_button.native.setToolTip("Save the current classifier into the selected folder.")
    export_button.clicked.connect(lambda: _save_rf(viewer, export_dir.value))

    container = Container(
        widgets=[checkbox_row, top_features_row, load_path, load_button, export_dir, export_button], labels=False,
    )
    return container, get_n_components, get_use_anyup, set_options

#
# Object classifier implementation.
#


# TODO add a gui element that shows the current label ids, how many objects are labeled, and that
# enables naming them so that the user can keep track of what has been labeled
class ObjectClassifier(QtWidgets.QScrollArea):

    def _require_layers(self, layer_choices: Optional[List[str]] = None):
        # Check whether the image is initialized already. And use the image shape and scale for the layers.
        # The dimensionality comes from the embedding widget ('state.ndim', RGB-aware) so the label layers
        # always match the image: 2d (YX) for 2d data, 3d (ZYX) for 3d data.
        state = AnnotatorState()
        if state.image_shape is None:
            ndim, shape = self._ndim, self._shape
        else:
            ndim = len(state.image_shape) if state.ndim is None else state.ndim
            shape = tuple(state.image_shape)[:ndim]

        # Add the label layers for the current object, the automatic segmentation and the committed segmentation.
        dummy_data = np.zeros(shape, dtype="uint32")
        image_scale = None if state.image_scale is None else tuple(state.image_scale)[:ndim]

        # Drop any existing label layer whose dimensionality no longer matches the image. Reassigning data
        # of a different ndim corrupts the napari layer transforms, so we rebuild such layers from scratch.
        for name in ("annotations", "prediction"):
            if name in self._viewer.layers and self._viewer.layers[name].data.ndim != len(shape):
                del self._viewer.layers[name]

        # Before adding new layers, we always check whether a layer with this name already exists or not.
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

        # Move 'annotations' to the top of the layer stack so scribbles are always visible above
        # the segmentation and prediction, and make it the active layer so the controls (incl. the
        # label id) show it and the user can paint right away rather than on another layer.
        self._viewer.layers.move(self._viewer.layers.index("annotations"), len(self._viewer.layers))
        self._viewer.layers.selection.active = self._viewer.layers["annotations"]

    def _invalidate_object_features(self, *args):
        state = AnnotatorState()
        state.object_features, state.seg_ids = None, None

    def _reset_segmentation_layer_choices(self, *args):
        previous_selection = self.segmentation_selection.value
        self.segmentation_selection.reset_choices()
        choices = self.segmentation_selection.choices
        if any(layer is previous_selection for layer in choices):
            self.segmentation_selection.value = previous_selection
        else:
            self._select_default_segmentation_layer()
        if self.segmentation_selection.value is not previous_selection:
            self._invalidate_object_features()

    def _is_segmentation_layer(self, layer):
        return isinstance(layer, napari.layers.Labels) and layer.name.lower() not in INTERNAL_LABEL_LAYER_NAMES

    def _find_default_segmentation_layer(self):
        candidates = [layer for layer in self._viewer.layers if self._is_segmentation_layer(layer)]
        if not candidates:
            return None

        for layer in candidates:
            if layer.name == "segmentation":
                return layer

        for layer in candidates:
            if "seg" in layer.name.lower():
                return layer

        return candidates[0]

    def _select_default_segmentation_layer(self):
        default_layer = self._find_default_segmentation_layer()
        if default_layer is not None:
            self.segmentation_selection.value = default_layer

    def _segmentation_layer_choices(self):
        return [(layer.name, layer) for layer in self._viewer.layers if self._is_segmentation_layer(layer)]

    def _create_segmentation_layer_section(self):
        segmentation_selection = QtWidgets.QVBoxLayout()
        segmentation_layer_widget = QtWidgets.QLabel("Segmentation:")
        segmentation_selection.addWidget(segmentation_layer_widget)
        self.segmentation_selection = ComboBox(choices=lambda _: self._segmentation_layer_choices())
        self._select_default_segmentation_layer()
        self.segmentation_selection.changed.connect(self._invalidate_object_features)
        state = AnnotatorState()
        state.segmentation_selection = self.segmentation_selection
        segmentation_selection.addWidget(self.segmentation_selection.native)
        return segmentation_selection

    def _create_label_widget(self):
        self._label_form = QtWidgets.QFormLayout()
        scroll_area = QtWidgets.QScrollArea()
        inner = QtWidgets.QWidget()
        inner.setLayout(self._label_form)
        scroll_area.setWidget(inner)
        scroll_area.setWidgetResizable(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Object label names:"))
        layout.addWidget(scroll_area)

        return layout

    def _make_label_role_widget(self, lbl):
        # Build the left-hand widget for a label row: a color swatch matching the annotation
        # layer color for this id, followed by the exact id ("ID <n>"). The id is stored as the
        # object name so it can be read back reliably when removing vanished rows.
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
            self._label_names[lbl] = ""
            self._label_form.addRow(self._make_label_role_widget(lbl), line)
            line.textChanged.connect(lambda txt, lbl=lbl: self._label_names.__setitem__(lbl, txt))

        # Remove rows whose label vanished. 'removeRow' deletes the row's widgets itself.
        for row in reversed(range(self._label_form.rowCount())):
            lbl_id = int(self._label_form.itemAt(row, QtWidgets.QFormLayout.LabelRole).widget().objectName())
            if lbl_id not in ids:
                self._label_form.removeRow(row)
                self._label_names.pop(lbl_id, None)

    def _create_widgets(self):
        # Create the embedding widget and connect all events related to it.
        self._embedding_widget = widgets.ClassificationEmbeddingWidget()
        # Connect events for the image selection box.
        self._viewer.layers.events.inserted.connect(self._embedding_widget.image_selection.reset_choices)
        self._viewer.layers.events.removed.connect(self._embedding_widget.image_selection.reset_choices)
        # Connect the run button with the function to update the image.
        self._embedding_widget.run_button.clicked.connect(self._update_image)

        # The segmentation selection stays visible at top level; only classifier options live in the
        # "Classification Settings" dropdown. The 'Apply to Volume' checkboxes are shown only for
        # 3d data (toggled in '_update_image').
        self._train_and_predict_widget, self._apply_to_volume = _create_train_widget(self._viewer)
        self._apply_to_volume.visible = False
        self._seg_selection_widget = self._create_segmentation_layer_section()
        self._viewer.layers.events.inserted.connect(self._reset_segmentation_layer_choices)
        self._viewer.layers.events.removed.connect(self._reset_segmentation_layer_choices)
        self._viewer.layers.events.reordered.connect(self._reset_segmentation_layer_choices)
        io = _create_classifier_io_widget(self._viewer)
        self._classifier_io_widget, self._get_n_components, self._get_use_anyup, self._set_options = io

        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())
        settings.layout().addWidget(self._classifier_io_widget.native)
        collapsible = widgets._make_collapsible(settings, title="Classification Settings")

        classification_section = QtWidgets.QWidget()
        classification_section.setLayout(QtWidgets.QVBoxLayout())
        seg_container = QtWidgets.QWidget()
        seg_container.setLayout(self._seg_selection_widget)
        classification_section.layout().addWidget(seg_container)
        classification_section.layout().addWidget(collapsible)
        classification_section.layout().addWidget(self._train_and_predict_widget.native)

        # A separate section: the object label names.
        self._label_widget = self._create_label_widget()

        self._widgets = {
            "embeddings": self._embedding_widget,
            "classification": classification_section,
            "label_widget": self._label_widget,
        }

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        """Create the GUI for the object classifier.

        Args:
            viewer: The napari viewer.
        """
        super().__init__()
        self._viewer = viewer
        self._annotator_widget = QtWidgets.QWidget()
        self._annotator_widget.setLayout(QtWidgets.QVBoxLayout())

        # Add the layers for prompts and segmented obejcts.
        # Initialize with a dummy shape, which is reset to the correct shape once an image is set.
        self._shape = (256, 256)
        self._ndim = len(self._shape)
        self._require_layers()

        # Create all the widgets and add them to the layout.
        self._label_names = {}  # The names for the object labels.
        self._create_widgets()

        # We could refactor this.
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

        # Set the expected annotator class to the state.
        state = AnnotatorState()
        state.annotator = self

        # Add the widgets to the state.
        state.widgets = self._widgets

        # Add the widget to the scroll area.
        self.setWidgetResizable(True)  # Allow widget to resize within scroll area.
        self.setWidget(self._annotator_widget)

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

        # The 'Apply to Volume' checkbox only makes sense for 3d data.
        self._apply_to_volume.visible = self._ndim == 3

        # Before we reset the layers, we ensure all expected layers exist at the correct ndim.
        self._require_layers()

        # Update the image scale.
        scale = None if state.image_scale is None else tuple(state.image_scale)[:self._ndim]

        # Reset all layers.
        self._viewer.layers["annotations"].data = np.zeros(self._shape, dtype="uint32")
        self._viewer.layers["prediction"].data = np.zeros(self._shape, dtype="uint32")
        if scale is not None:
            self._viewer.layers["annotations"].scale = scale
            self._viewer.layers["prediction"].scale = scale


def object_classifier(
    image: np.ndarray,
    segmentation: np.ndarray,
    embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
    model_type: str = DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    ndim: Optional[int] = None,
) -> Optional["napari.viewer.Viewer"]:
    """Start the object classifier for a given image and segmentation.

    Args:
        image: The image data.
        segmentation: The segmentation data.
        embedding_path: Filepath where to save the embeddings
            or the precompted image embeddings computed by `precompute_image_embeddings`.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
            By default, does not return the napari viewer.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        ndim: The dimensionality of the data. If not given will be derived from the data.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    if ndim is None:
        ndim = image.ndim - 1 if image.shape[-1] == 3 and image.ndim in (3, 4) else image.ndim

    state = AnnotatorState()
    state.image_shape = image.shape[:ndim]
    state.ndim = ndim

    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, precompute_amg_state=False,
        ndim=ndim, checkpoint_path=checkpoint_path, device=device,
        skip_load=False, use_cli=True,
    )

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    viewer.add_labels(segmentation, name="segmentation")

    annotator = ObjectClassifier(viewer)

    # Trigger layer update of the annotator so that layers have the correct shape.
    # And initialize the 'committed_objects' with the segmentation result if it was given.
    annotator._update_image()

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator, name="(Object Classifier) Segment Anything for Microscopy")
    _sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=model_type if checkpoint_path is None else state.predictor.model_type,
        save_path=embedding_path,
        checkpoint_path=checkpoint_path,
        device=device,
        tile_shape=tile_shape,
        halo=halo,
    )

    if return_viewer:
        return viewer

    napari.run()


def image_series_object_classifier(
    images: List[np.ndarray],
    segmentations: List[np.ndarray],
    output_folder: str,
    embedding_paths: Optional[List[Union[str, util.ImageEmbeddings]]] = None,
    model_type: str = DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    ndim: Optional[int] = None,
) -> None:
    """Start the object classifier for a list of images and segmentations.

    This function will save the all features and labels for annotated objects,
    to enable training a random forest on multiple images.

    Args:
        images: The input images.
        segmentations: The input segmentations.
        output_folder: The folder where segmentation results, trained random forest
            and the features, labels aggregated during training will be saved.
        embedding_paths: Filepaths where to save the embeddings
            or the precompted image embeddings computed by `precompute_image_embeddings`.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        ndim: The dimensionality of the data. If not given will be derived from the data.
    """
    # TODO precompute the embeddings if not computed, can re-use 'precompute' from image series annotator.
    # TODO support file paths as inputs
    # TODO option to skip segmented
    if len(images) != len(segmentations):
        raise ValueError(
            f"Expect the same number of images and segmentations, got {len(images)}, {len(segmentations)}."
        )

    end_msg = "You have annotated the last image. Do you wish to close napari?"

    # Initialize the object classifier on the fist image / segmentation.
    viewer = object_classifier(
        image=images[0], segmentation=segmentations[0],
        embedding_path=None if embedding_paths is None else embedding_paths[0],
        model_type=model_type, tile_shape=tile_shape, halo=halo,
        return_viewer=True, checkpoint_path=checkpoint_path,
        device=device, ndim=ndim,
    )

    os.makedirs(output_folder, exist_ok=True)
    next_image_id = 0

    def _save_prediction(image, pred, image_id):
        fname = f"{Path(image).stem}_prediction.tif" if isinstance(image, str) else f"prediction_{image_id}.tif"
        save_path = os.path.join(output_folder, fname)
        imageio.imwrite(save_path, pred, compression="zlib")

    # TODO handle cases where rf for the image was not trained, raise a message, enable contnuing
    # Add functionality for going to the next image.
    @magicgui(call_button="Next Image [N]")
    def next_image(*args):
        nonlocal next_image_id

        # Get the state and the current segmentation (note that next image id has not yet been increased)
        state = AnnotatorState()
        segmentation = segmentations[next_image_id]

        # Keep track of the previous features and labels.
        labels = _accumulate_labels(segmentation, viewer.layers["annotations"].data)
        valid = labels != 0
        if valid.sum() > 0:
            features, labels = state.object_features[valid], labels[valid]
            if state.previous_features is None:
                state.previous_features, state.previous_labels = features, labels
            else:
                state.previous_features = np.concatenate([state.previous_features, features], axis=0)
                state.previous_labels = np.concatenate([state.previous_labels, labels], axis=0)
            # Save the accumulated features and labels.
            np.save(os.path.join(output_folder, "features.npy"), state.previous_features)
            np.save(os.path.join(output_folder, "labels.npy"), state.previous_labels)

        # Save the current prediction and RF (with its specs, matching the single-image export).
        _save_prediction(images[next_image_id], viewer.layers["prediction"].data, next_image_id)
        dump(
            {"rf": state.object_rf, "model_spec": _classifier_spec(state.object_rf)},
            os.path.join(output_folder, "rf.joblib"),
        )

        # Go to the next image.
        next_image_id += 1

        # Check if we are done.
        if next_image_id == len(images):
            # Inform the user via dialog.
            abort = widgets._generate_message("info", end_msg)
            if not abort:
                viewer.close()
            return

        # Get the next image, segmentation and embedding_path.
        image = images[next_image_id]
        segmentation = segmentations[next_image_id]
        embedding_path = None if embedding_paths is None else embedding_paths[next_image_id]

        # Set the new image in the viewer, state and annotator.
        viewer.layers["image"].data = image
        viewer.layers["segmentation"].data = segmentation

        state.initialize_predictor(
            image, model_type=model_type, ndim=ndim,
            save_path=embedding_path,
            tile_shape=tile_shape, halo=halo,
            predictor=state.predictor, device=device,
        )
        state.image_shape = image.shape if image.ndim == ndim else image.shape[:-1]
        state.ndim = ndim
        state.annotator._update_image()

        # Clear the object features and seg-ids from the state.
        state.object_features = None
        state.seg_ids = None

    viewer.window.add_dock_widget(next_image)

    @viewer.bind_key("n", overwrite=True)
    def _next_image(viewer):
        next_image(viewer)

    napari.run()


# TODO: folder annotator
# TODO: main function
