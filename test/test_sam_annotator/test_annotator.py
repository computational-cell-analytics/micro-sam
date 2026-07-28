import platform

import numpy as np
import pytest
from skimage.data import binary_blobs

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator.annotator import annotator, detect_ndim, detect_ndim_from_viewer, Annotator
from micro_sam._test_util import check_layer_initialization


def test_progress_bar_initial_description(monkeypatch):
    """A progress description supplied at creation is visible before the backend reports a total."""
    from micro_sam.sam_annotator import _widgets

    captured = {}

    class FakeProgress:
        def update(self, value):
            pass

        def set_description(self, description):
            pass

        def close(self):
            pass

        def reset(self):
            pass

    def fake_progress(**kwargs):
        captured["kwargs"] = kwargs
        return FakeProgress()

    monkeypatch.setattr(_widgets, "progress", fake_progress)
    _widgets._create_pbar_for_threadworker("Preparing image embeddings")
    assert captured["kwargs"] == {"desc": "Preparing image embeddings"}


class TestDetectNdim:
    """Test the detect_ndim helper function."""

    def test_2d_grayscale(self):
        """Test 2D grayscale image detection."""
        image = np.zeros((512, 512), dtype=np.uint8)
        assert detect_ndim(image) == 2

    def test_2d_rgb(self):
        """Test 2D RGB image detection."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        assert detect_ndim(image) == 2

    def test_3d_grayscale(self):
        """Test 3D grayscale volume detection."""
        image = np.zeros((10, 512, 512), dtype=np.uint8)
        assert detect_ndim(image) == 3

    def test_3d_rgb(self):
        """Test 3D RGB volume detection."""
        image = np.zeros((10, 512, 512, 3), dtype=np.uint8)
        assert detect_ndim(image) == 3

    def test_ambiguous_shape_assumes_3d(self):
        """Test that shape (3, 512, 512) is interpreted as 3D grayscale."""
        image = np.zeros((3, 512, 512), dtype=np.uint8)
        # By default, assumes 3D grayscale rather than RGB 2D
        assert detect_ndim(image) == 3

    def test_invalid_1d_shape(self):
        """Test that 1D arrays raise ValueError."""
        image = np.zeros(512, dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid image shape"):
            detect_ndim(image)

    def test_invalid_4d_non_rgb(self):
        """Test that 4D arrays without RGB channel raise ValueError."""
        image = np.zeros((10, 10, 512, 512), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid 4D shape"):
            detect_ndim(image)

    def test_invalid_5d_shape(self):
        """Test that 5D arrays raise ValueError."""
        image = np.zeros((2, 3, 10, 512, 512), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid image shape"):
            detect_ndim(image)


class TestAnnotatorApiNdimOverride:
    """The annotator() API / CLI threads the ndim override into normalization (no GUI needed: these
    raise at the first line, before any model compute)."""

    def test_forcing_3d_on_2d_image_raises(self):
        with pytest.raises(ValueError, match="3D volume"):
            annotator(np.zeros((64, 64), dtype="uint8"), ndim=3)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="Invalid ndim override"):
            annotator(np.zeros((64, 64), dtype="uint8"), ndim=5)


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
class TestDetectNdimFromViewer:
    """Test detecting ndim from image layers loaded in the viewer."""

    def test_empty_viewer_defaults_to_2d(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        assert detect_ndim_from_viewer(viewer) == 2
        viewer.close()

    def test_detects_2d_image(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(binary_blobs(128), name="image")
        assert detect_ndim_from_viewer(viewer) == 2
        viewer.close()

    def test_detects_3d_image(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(np.stack(4 * [binary_blobs(128)]), name="volume")
        assert detect_ndim_from_viewer(viewer) == 3
        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
class TestAnnotatorClass:
    """Test the unified Annotator class."""

    def test_annotator_2d(self, make_napari_viewer_proxy):
        image = binary_blobs(512)
        model_type = DEFAULT_MODEL

        viewer = make_napari_viewer_proxy()
        # test generating image embedding, then adding micro-sam dock widgets to the GUI
        viewer = annotator(
            image,
            model_type=model_type,
            viewer=viewer,
            return_viewer=True,
        )

        check_layer_initialization(viewer, image.shape)
        viewer.close()  # must close the viewer at the end of tests

    def test_widget_no_image_defaults_to_2d(self, make_napari_viewer_proxy):
        # Reproduces opening the plugin from the napari Plugins menu with no image loaded.
        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer)
        assert widget._ndim == 2
        assert "scribble_prompts" not in viewer.layers
        assert viewer.layers["prompts"].ndim == 2
        assert viewer.layers["prompts"].current_properties["label"][0] == "positive"
        viewer.layers["prompts"].mode = "add_polyline"
        toggle_binding = next(
            callback for key, callback in viewer.layers["prompts"].keymap.items() if str(key) == "T"
        )
        toggle_binding(viewer.layers["prompts"])
        assert widget._prompt_widget[0].value == "negative"
        assert viewer.layers["point_prompts"].current_properties["label"][0] == "negative"
        assert viewer.layers["prompts"].current_properties["label"][0] == "negative"
        assert viewer.layers["prompts"].current_edge_color == "red"
        toggle_binding(viewer.layers["prompts"])
        assert widget._prompt_widget[0].value == "positive"
        widget._prompt_widget[0].value = "negative"
        assert viewer.layers["point_prompts"].current_properties["label"][0] == "negative"
        assert viewer.layers["prompts"].current_properties["label"][0] == "negative"
        assert viewer.layers["prompts"].current_edge_color == "red"
        viewer.layers["prompts"].add_rectangles(np.array([[0, 0], [8, 8]]))
        viewer.layers["prompts"].add_paths(np.array([[1, 1], [7, 7]]))
        np.testing.assert_array_equal(viewer.layers["prompts"].properties["label"], ["positive", "negative"])
        # Selecting a scribble means working in the shape layer, so the menu relabels it.
        viewer.layers.selection.active = viewer.layers["prompts"]
        viewer.layers["prompts"].selected_data = {1}
        widget._prompt_widget[0].value = "positive"
        np.testing.assert_array_equal(viewer.layers["prompts"].properties["label"], ["positive", "positive"])

        # From the point layer the same menu change leaves the scribble alone.
        viewer.layers["prompts"].selected_data = {1}
        viewer.layers.selection.active = viewer.layers["point_prompts"]
        widget._prompt_widget[0].value = "negative"
        np.testing.assert_array_equal(viewer.layers["prompts"].properties["label"], ["positive", "positive"])
        assert viewer.layers["point_prompts"].current_properties["label"][0] == "negative"
        viewer.close()

    def test_point_layer_toggle_leaves_a_drawn_scribble_alone(self, make_napari_viewer_proxy):
        """Flipping polarity for the next point must not relabel a scribble drawn earlier."""
        viewer = make_napari_viewer_proxy()
        Annotator(viewer)
        shapes, points = viewer.layers["prompts"], viewer.layers["point_prompts"]

        # Draw a positive scribble, as the user does before moving to the point layer.
        shapes.mode = "add_path"
        shapes.add_paths(np.array([[1.0, 1.0], [7.0, 7.0]]))
        assert shapes.properties["label"][0] == "positive"

        points.add(np.array([[4.0, 4.0]]))
        toggle = next(cb for key, cb in points.keymap.items() if str(key) == "T")
        toggle(points)

        # Both layers switch their drawing default, so the shared prompt menu stays truthful.
        assert points.current_properties["label"][0] == "negative"
        assert shapes.current_properties["label"][0] == "negative"
        # The scribble that was already drawn keeps its own label.
        np.testing.assert_array_equal(shapes.properties["label"], ["positive"])
        viewer.close()

    def test_shape_layer_toggle_leaves_a_placed_point_alone(self, make_napari_viewer_proxy):
        """The mirror of the case above: switching back to the shape layer must not relabel the point."""
        viewer = make_napari_viewer_proxy()
        Annotator(viewer)
        shapes, points = viewer.layers["prompts"], viewer.layers["point_prompts"]

        points.add(np.array([[4.0, 4.0]]))
        points.selected_data = {0}
        assert points.properties["label"][0] == "positive"

        toggle = next(cb for key, cb in shapes.keymap.items() if str(key) == "T")
        toggle(shapes)

        assert shapes.current_properties["label"][0] == "negative"
        assert points.current_properties["label"][0] == "negative"
        np.testing.assert_array_equal(points.properties["label"], ["positive"])
        viewer.close()

    def test_tracking_point_layer_toggle_leaves_a_drawn_scribble_alone(self, make_napari_viewer_proxy):
        """The tracking annotator shares the rule: the toggle only relabels the layer in use."""
        from micro_sam.sam_annotator.annotator_tracking import AnnotatorTracking

        viewer = make_napari_viewer_proxy()
        viewer.add_image(np.stack(4 * [binary_blobs(64)]), name="timeseries")
        AnnotatorTracking(viewer)
        shapes, points = viewer.layers["prompts"], viewer.layers["point_prompts"]

        shapes.mode = "add_path"
        shapes.add_paths(np.array([[0.0, 1.0, 1.0], [0.0, 7.0, 7.0]]))
        assert shapes.properties["label"][0] == "positive"

        toggle = next(cb for key, cb in points.keymap.items() if str(key) == "T")
        toggle(points)

        assert points.current_properties["label"][0] == "negative"
        np.testing.assert_array_equal(shapes.properties["label"], ["positive"])
        viewer.close()

    def test_widget_detects_ndim_from_loaded_image(self, make_napari_viewer_proxy):
        # When an image is loaded before opening the widget, ndim is detected from it.
        viewer = make_napari_viewer_proxy()
        viewer.add_image(np.stack(4 * [binary_blobs(128)]), name="volume")
        widget = Annotator(viewer)
        assert widget._ndim == 3
        viewer.close()

    def test_widget_rebuilds_when_3d_image_loaded_after_open(self, make_napari_viewer_proxy):
        # Open the widget without an image (defaults to 2D), then load a 3D image.
        # Selecting it as input rebuilds the annotator for 3D.
        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer)
        assert widget._ndim == 2

        viewer.add_image(np.stack(4 * [binary_blobs(128)]), name="volume")
        widget._embedding_widget.image_selection.reset_choices()
        assert widget._ndim == 3
        # The prompt layers must be recreated with the new dimensionality.
        assert viewer.layers["point_prompts"].ndim == 3
        assert viewer.layers["prompts"].ndim == 3
        viewer.close()

    def test_annotator_3d(self, make_napari_viewer_proxy):
        image = np.stack(4 * [binary_blobs(512)])
        model_type = DEFAULT_MODEL

        viewer = make_napari_viewer_proxy()
        # test generating image embedding, then adding micro-sam dock widgets to the GUI
        viewer = annotator(
            image,
            model_type=model_type,
            viewer=viewer,
            return_viewer=True
        )

        check_layer_initialization(viewer, image.shape)
        viewer.close()  # must close the viewer at the end of tests

    def test_tiling_defaults_and_not_force_enabled(self, make_napari_viewer_proxy):
        # Regression for the tiling-criterion bugs: the embedding widget uses the centralized tiling
        # defaults, a small image (below the threshold) is not auto-tiled, and syncing the widget
        # after an embedding compute must not force the tiling dropdown to "yes".
        from micro_sam.sam_annotator._state import AnnotatorState
        from micro_sam.v2.util import DEFAULT_TILE_SHAPE, DEFAULT_HALO

        viewer = make_napari_viewer_proxy()
        viewer.add_image(binary_blobs(256), name="image")
        widget = Annotator(viewer)
        ew = widget._embedding_widget
        ew.image_selection.reset_choices()

        # Centralized defaults are used, and a 256x256 image is below the tiling threshold.
        assert (ew.tile_x, ew.tile_y) == DEFAULT_TILE_SHAPE
        assert (ew.halo_x, ew.halo_y) == DEFAULT_HALO
        assert ew.tiling == "no"

        # Syncing after a compute with tiling off must keep tiling off (used to be forced to "yes").
        # '_validate_model_type_and_custom_weights' sets 'model_type', as the first step of '__call__'.
        ew._validate_model_type_and_custom_weights()
        state = AnnotatorState()
        ew._update_model(state)
        assert ew.tiling == "no"
        assert ew.tiling_dropdown.currentText() == "no"

        # When the user enabled tiling, the choice and the values used must be retained across a sync.
        ew.tiling_dropdown.setCurrentText("yes")
        ew.tile_x_param.setValue(640)
        ew._update_model(state)
        assert ew.tiling_dropdown.currentText() == "yes"
        assert ew.tile_x == 640

        viewer.close()

    def test_reset_inputs_keeps_optional_paths_unset(self, qapp):
        """Clearing inputs restores safe defaults without creating a blank custom checkpoint path."""
        from micro_sam.sam_annotator._widgets import EmbeddingWidget

        ew = EmbeddingWidget(ndim_choice=True)

        # Optional paths start unset, and whitespace entered in a path field is unset as well.
        assert ew.custom_weights is None
        assert ew.custom_weights_param.text() == ""
        ew.custom_weights_param.setText(" ")
        assert ew.custom_weights is None

        # Batching starts at the value the VRAM table recommends for the default model and device,
        # which is one without a GPU. Read from the table rather than from the widget's own helper,
        # which would agree with it trivially.
        from micro_sam.util import _get_default_device
        from micro_sam.v2.util import recommend_batch_size

        device = _get_default_device() if ew.device == "auto" else ew.device
        recommended = recommend_batch_size(ew.model_type, device)
        assert ew.batch_size == recommended
        assert ew.batch_size_param.value() == recommended

        # Reproduce the input reset used when a different image layer is selected.
        ew.custom_weights_param.setText("/tmp/custom-weights.pt")
        ew.batch_size_param.setValue(recommended + 1)
        assert ew.custom_weights == "/tmp/custom-weights.pt"
        assert ew.batch_size == recommended + 1
        ew._reset_inputs_to_defaults()
        assert ew.custom_weights is None
        assert ew.custom_weights_param.text() == ""
        assert ew.batch_size == recommended
        assert ew.batch_size_param.value() == recommended

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_batched_checkbox_hidden_when_tiled(self, make_napari_viewer_proxy, ndim):
        # Regression for 3c: batched (multi-object) prompting is unsupported with tiling, so the
        # 'Batched' checkbox must be hidden while the embeddings are tiled (and shown otherwise).
        # This holds for both the 2d and 3d segmentation annotator.
        from micro_sam.sam_annotator._state import AnnotatorState

        image = binary_blobs(256) if ndim == 2 else np.stack(4 * [binary_blobs(256)])
        shape = (256, 256) if ndim == 2 else (4, 256, 256)

        viewer = make_napari_viewer_proxy()
        viewer.add_image(image, name="image")
        widget = Annotator(viewer)
        assert widget._ndim == ndim
        interactive = widget._widgets["interactive"]
        state = AnnotatorState()

        # Non-tiled embeddings -> batched control shown.
        state.image_embeddings = {"input_size": (256, 256), "original_size": shape, "features": None}
        interactive._update_batched_visibility()
        assert not interactive.batched_checkbox.isHidden()

        # Tiled embeddings (top-level input_size is None) -> hidden and reset to single-object.
        interactive.batched_checkbox.setChecked(True)
        state.image_embeddings = {"input_size": None, "original_size": shape, "features": None}
        interactive._update_batched_visibility()
        assert interactive.batched_checkbox.isHidden()
        assert not interactive.batched

        # Back to non-tiled -> shown again.
        state.image_embeddings = {"input_size": (256, 256), "original_size": shape, "features": None}
        interactive._update_batched_visibility()
        assert not interactive.batched_checkbox.isHidden()

        viewer.close()

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_batched_checkbox_disabled_with_scribbles(self, make_napari_viewer_proxy, ndim):
        """Batched mode is unavailable exactly while the prompt layer contains a scribble."""
        from micro_sam.sam_annotator._state import AnnotatorState

        image = binary_blobs(256) if ndim == 2 else np.stack(4 * [binary_blobs(256)])
        shape = (256, 256) if ndim == 2 else (4, 256, 256)
        scribble = (
            np.array([[1, 1], [7, 7]])
            if ndim == 2 else np.array([[1, 1, 1], [1, 7, 7]])
        )

        viewer = make_napari_viewer_proxy()
        viewer.add_image(image, name="image")
        widget = Annotator(viewer)
        interactive = widget._widgets["interactive"]
        prompt_layer = viewer.layers["prompts"]

        # Start in the normal, non-tiled state and enable batched mode.
        AnnotatorState().image_embeddings = {
            "input_size": (256, 256), "original_size": shape, "features": None,
        }
        interactive._update_batched_visibility()
        assert interactive.batched_checkbox.isEnabled()
        normal_tooltip = interactive.batched_checkbox.toolTip()
        interactive.batched_checkbox.setChecked(True)

        # Adding a scribble immediately resets and disables batched mode.
        prompt_layer.add_paths(scribble)
        assert not interactive.batched_checkbox.isChecked()
        assert not interactive.batched_checkbox.isEnabled()
        assert not interactive.batched
        assert "unavailable while scribble prompts are present" in interactive.batched_checkbox.toolTip()
        if ndim == 3:
            assert not interactive._segment_widget.batched

        # Removing the last scribble restores normal batched availability.
        prompt_layer.selected_data = {0}
        prompt_layer.remove_selected()
        assert interactive.batched_checkbox.isEnabled()
        assert interactive.batched_checkbox.toolTip() == normal_tooltip

        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
class TestNdimOverride:
    """Multi-channel handling via the 'image dimensions' (ndim) override dropdown."""

    def test_dropdown_only_in_segmentation_annotator(self, make_napari_viewer_proxy):
        from micro_sam.sam_annotator.annotator_tracking import AnnotatorTracking

        viewer = make_napari_viewer_proxy()
        seg = Annotator(viewer)
        assert seg._embedding_widget.ndim_choice is True
        assert hasattr(seg._embedding_widget, "image_ndim_dropdown")
        viewer.close()

        viewer = make_napari_viewer_proxy()
        track = AnnotatorTracking(viewer)
        assert track._embedding_widget.ndim_choice is False
        assert not hasattr(track._embedding_widget, "image_ndim_dropdown")
        viewer.close()

    def test_channels_first_forced_2d(self, make_napari_viewer_proxy):
        # A channels-first (C, H, W) array is auto-detected as a volume. Forcing '2d' reads it as a
        # 2d multi-channel image (mapped to RGB) and rebuilds the annotator for 2d.
        viewer = make_napari_viewer_proxy()
        viewer.add_image(np.zeros((4, 64, 64), dtype="uint8"), name="image")
        widget = Annotator(viewer)
        assert widget._ndim == 3  # auto: channels-first -> volume

        widget._embedding_widget.image_ndim_dropdown.setCurrentText("2d")
        assert widget._ndim == 2
        assert viewer.layers["image"].rgb is True
        assert tuple(viewer.layers["image"].data.shape) == (64, 64, 3)
        assert viewer.layers["point_prompts"].ndim == 2
        assert viewer.layers["prompts"].ndim == 2
        viewer.close()

    def test_channels_last_two_channel_auto(self, make_napari_viewer_proxy):
        # A channels-last 2-channel image is auto-detected as 2d and padded to RGB.
        viewer = make_napari_viewer_proxy()
        viewer.add_image(np.zeros((64, 64, 2), dtype="uint8"), name="image")
        widget = Annotator(viewer)
        assert widget._ndim == 2
        assert viewer.layers["image"].rgb is True
        assert tuple(viewer.layers["image"].data.shape) == (64, 64, 3)
        viewer.close()

    def test_ndim_override_round_trip_re_derives_from_original(self, make_napari_viewer_proxy):
        # (3, 64, 64): auto -> 3-slice volume; force '2d' -> RGB; force '3d' -> back to the original
        # 3-slice volume (re-derived from the stored original, not the reduced RGB layer).
        viewer = make_napari_viewer_proxy()
        viewer.add_image(np.zeros((3, 64, 64), dtype="uint8"), name="image")
        widget = Annotator(viewer)
        assert widget._ndim == 3

        widget._embedding_widget.image_ndim_dropdown.setCurrentText("2d")
        assert widget._ndim == 2
        assert tuple(viewer.layers["image"].data.shape) == (64, 64, 3)
        assert viewer.layers["image"].rgb is True

        widget._embedding_widget.image_ndim_dropdown.setCurrentText("3d")
        assert widget._ndim == 3
        assert tuple(viewer.layers["image"].data.shape) == (3, 64, 64)
        assert viewer.layers["image"].rgb is False
        viewer.close()

    def test_force_3d_on_2d_image_warns_and_reverts_to_auto(self, make_napari_viewer_proxy, monkeypatch):
        # Forcing '3d' on a genuinely 2D image is invalid: a modal warning is shown and the dropdown
        # reverts to 'auto' (the image stays 2D). The modal is patched so it does not block the test.
        from qtpy import QtWidgets

        calls = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: calls.append(a)))

        viewer = make_napari_viewer_proxy()
        viewer.add_image(np.zeros((64, 64), dtype="uint8"), name="image")
        widget = Annotator(viewer)
        assert widget._ndim == 2

        widget._embedding_widget.image_ndim_dropdown.setCurrentText("3d")
        assert len(calls) == 1  # a warning dialog was shown
        assert widget._embedding_widget.image_ndim_dropdown.currentText() == "auto"  # reverted
        assert widget._embedding_widget._ndim_override() is None
        assert widget._ndim == 2  # image is still 2D
        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
class TestZTilingControls:
    """The z block / halo controls live on the (volumetric) auto-seg widget, not the embedding widget."""

    def test_z_tiling_not_on_embedding_widget(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=3)
        ew = widget._embedding_widget
        # The embedding widget keeps only the in-plane tile/halo, not the z block/halo.
        assert hasattr(ew, "tile_x") and hasattr(ew, "halo_x")
        assert not hasattr(ew, "tile_z_param") and not hasattr(ew, "halo_z_param")
        viewer.close()

    def test_z_tiling_on_volumetric_autoseg_widget(self, make_napari_viewer_proxy):
        from micro_sam.v2.util import DEFAULT_TILE_Z, DEFAULT_HALO_Z

        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=3)
        autoseg = widget._widgets["autosegment"]
        assert autoseg.volumetric is True
        assert (autoseg.tile_z, autoseg.halo_z) == (DEFAULT_TILE_Z, DEFAULT_HALO_Z)
        # 'tile_z' >= the slice count disables z-tiling (whole volume in one block).
        assert autoseg._z_tiling(n_slices=2) == (2, 0)
        # A deeper volume gets the configured z block + halo.
        assert autoseg._z_tiling(n_slices=100) == (DEFAULT_TILE_Z, DEFAULT_HALO_Z)
        viewer.close()

    def test_z_tiling_hidden_for_2d_autoseg(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=2)
        autoseg = widget._widgets["autosegment"]
        assert autoseg.volumetric is False
        # No z-tiling spinboxes are built for a 2d segmentation widget.
        assert not hasattr(autoseg, "tile_z_param")
        viewer.close()


class TestAutoSegVolumeDispatch:
    """'Apply to volume' decides the run dimensionality of automatic segmentation on a 3d volume:
    off -> only the current slice (2d); on -> the whole volume (3d), segmented slice by slice.
    This is what makes the state caching happen per-slice, on demand."""

    def _dispatch(self, monkeypatch, *, apply_to_volume, current_slice=2):
        from types import SimpleNamespace
        from micro_sam.sam_annotator import _widgets
        from micro_sam.sam_annotator._widgets import AutoSegmentWidget

        volume = np.zeros((5, 16, 16), dtype="float32")
        auto_layer = SimpleNamespace(data=np.zeros((5, 16, 16), dtype="uint32"), refresh=lambda: None)
        viewer = SimpleNamespace(
            layers={"image": SimpleNamespace(data=volume), "auto_segmentation": auto_layer},
            dims=SimpleNamespace(point=(current_slice, 0, 0)),
        )

        calls = {}

        def fake_run_amg(state, run_raw, ndim, z, pbar_init=None, pbar_update=None):
            calls.update(run_raw_shape=tuple(run_raw.shape), ndim=ndim, z=z)
            return np.zeros(run_raw.shape if ndim == 3 else run_raw.shape[-2:], dtype="uint32")

        # Duck-typed stand-in so we exercise the '__call__' dispatch without instantiating a QWidget.
        widget = SimpleNamespace(
            _viewer=viewer, mode="amg", volumetric=True, apply_to_volume=apply_to_volume,
            _run_amg=fake_run_amg,
        )

        def fake_pbar():
            signal = lambda: SimpleNamespace(emit=lambda *a, **k: None)  # noqa
            signals = SimpleNamespace(
                pbar_total=signal(), pbar_description=signal(), pbar_update=signal(), pbar_stop=signal(),
            )
            return SimpleNamespace(), signals

        monkeypatch.setattr(_widgets, "AnnotatorState", lambda: SimpleNamespace(get_image_name=lambda v: "image"))
        monkeypatch.setattr(_widgets, "_validate_layers", lambda *a, **k: False)
        monkeypatch.setattr(_widgets, "_validate_embeddings", lambda *a, **k: False)
        monkeypatch.setattr(_widgets, "_select_layer", lambda *a, **k: None)
        monkeypatch.setattr(_widgets, "_create_pbar_for_threadworker", fake_pbar)
        monkeypatch.setattr(
            _widgets, "QtWidgets",
            SimpleNamespace(QApplication=SimpleNamespace(processEvents=lambda *a, **k: None)),
        )

        AutoSegmentWidget.__call__(widget)
        return calls

    def test_apply_to_volume_off_runs_current_slice_only(self, monkeypatch):
        calls = self._dispatch(monkeypatch, apply_to_volume=False, current_slice=2)
        assert calls["ndim"] == 2  # a single 2d slice, not the whole volume
        assert calls["z"] == 2  # the currently viewed slice
        assert calls["run_raw_shape"] == (16, 16)

    def test_apply_to_volume_on_runs_whole_volume(self, monkeypatch):
        calls = self._dispatch(monkeypatch, apply_to_volume=True)
        assert calls["ndim"] == 3  # the whole volume, segmented slice by slice
        assert calls["z"] is None
        assert calls["run_raw_shape"] == (5, 16, 16)


class TestAutoSegStatePersistence:
    """By default (caching off) auto-seg persists no state, so the embedding zarr gets no
    'autoseg_state' group; it is written only when the user opts in. '_state_save_path'
    is the gate: None means in-memory only, a path means persist into the embedding zarr."""

    def _state_save_path(self, cache_state, embedding_path="/tmp/e.zarr", with_widget=True):
        from types import SimpleNamespace
        from micro_sam.sam_annotator._widgets import AutoSegmentWidget

        widgets = {"embeddings": SimpleNamespace(cache_state=cache_state)} if with_widget else {}
        state = SimpleNamespace(widgets=widgets, embedding_path=embedding_path)
        return AutoSegmentWidget._state_save_path(SimpleNamespace(), state)

    def test_no_persist_by_default(self):
        assert self._state_save_path(cache_state=False) is None  # default: in-memory only, no zarr group

    def test_persist_when_opted_in(self):
        assert self._state_save_path(cache_state=True) == "/tmp/e.zarr"

    def test_no_persist_without_embedding_widget(self):
        assert self._state_save_path(cache_state=True, with_widget=False) is None

    def test_enabling_persistence_invalidates_the_in_memory_cache(self, monkeypatch):
        """Turning disk caching on after an in-memory run must route through the cache helper again."""
        from types import MethodType, SimpleNamespace

        import micro_sam.precompute_state as precompute_state
        from micro_sam.sam_annotator._widgets import AutoSegmentWidget

        calls = []

        class _Segmenter:
            def generate(self, **kwargs):
                return np.zeros((8, 8), dtype="uint32")

        def fake_cache_autoseg_state(*args, **kwargs):
            calls.append(args[4])  # save_path
            return _Segmenter()

        monkeypatch.setattr(precompute_state, "cache_autoseg_state", fake_cache_autoseg_state)

        embedding_widget = SimpleNamespace(cache_state=False)
        state = SimpleNamespace(
            predictor=SimpleNamespace(model=object(), model_type="hvit_t"),
            image_embeddings={"input_size": (8, 8)},
            embedding_path="/tmp/embeddings.zarr",
            data_signature="data",
            widgets={"embeddings": embedding_widget},
        )
        widget = SimpleNamespace(
            volumetric=False, min_object_size=0, points_per_side=32,
            pred_iou_thresh=0.8, stability_score_thresh=0.9,
            _segmenter=None, _segmenter_key=None,
        )
        widget._state_save_path = MethodType(AutoSegmentWidget._state_save_path, widget)

        raw = np.zeros((8, 8), dtype="uint8")
        AutoSegmentWidget._run_amg(widget, state, raw, ndim=2, z=None)
        embedding_widget.cache_state = True
        AutoSegmentWidget._run_amg(widget, state, raw, ndim=2, z=None)

        assert calls == [None, "/tmp/embeddings.zarr"]


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
class TestAutoSegDefaultMode:
    """The default automatic-segmentation mode reflects the default model's decoder availability."""

    def test_default_model_has_decoder_predicate(self):
        from micro_sam.v2.util import has_registered_decoder, DEFAULT_MODEL
        assert has_registered_decoder(DEFAULT_MODEL) is True  # the Microscopy default has a decoder
        assert has_registered_decoder("hvit_t") is False  # a plain backbone does not

    def test_autoseg_defaults_to_sparse_for_decoder_model(self, make_napari_viewer_proxy):
        # Regression: with the Microscopy default model (which has a decoder) the auto-seg widget must
        # start in a decoder mode ('sparse'), not 'amg', even before embeddings are computed.
        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=2)
        autoseg = widget._widgets["autosegment"]
        assert autoseg.with_decoder is True
        assert autoseg.mode == "sparse"
        assert autoseg.mode_dropdown.currentText() == "sparse"
        viewer.close()

    def test_autoseg_settings_use_v2_defaults(self):
        from micro_sam.v2.postprocessing import DEFAULT_POSTPROCESSING
        from micro_sam.sam_annotator._widgets import AutoSegmentWidget

        class _FakeLayout:
            def addLayout(self, layout):
                pass

        class _FakeSettings:
            def __init__(self):
                self._layout = _FakeLayout()

            def layout(self):
                return self._layout

        class _FakeAutoSegmentWidget:
            def _add_float_param(self, *args, **kwargs):
                return None, None

            def _add_int_param(self, *args, **kwargs):
                return None, None

            def _add_density_threshold(self, settings):
                pass

            def _add_flow_integration_params(self, settings, n_iter, dt=0.5, sigma=1.0):
                self.n_iter = n_iter
                self.dt = dt
                self.sigma = sigma

        autoseg = _FakeAutoSegmentWidget()
        AutoSegmentWidget._sparse_settings(autoseg, _FakeSettings())
        defaults = DEFAULT_POSTPROCESSING["sparse"]
        assert autoseg.foreground_threshold == defaults["foreground_threshold"]
        assert autoseg.density_threshold == defaults["density_threshold"]
        assert autoseg.min_object_size == defaults["min_size"]
        assert autoseg.sigma == defaults["sigma"]
        assert autoseg.n_iter == defaults["n_iter"]
        assert autoseg.dt == defaults["dt"]

        autoseg = _FakeAutoSegmentWidget()
        AutoSegmentWidget._dense_settings(autoseg, _FakeSettings())
        defaults = DEFAULT_POSTPROCESSING["dense"]
        assert autoseg.beta == defaults["beta"]
        assert autoseg.density_threshold == defaults["density_threshold"]
        assert autoseg.sigma == defaults["sigma"]
        assert autoseg.n_iter == defaults["n_iter"]
        assert autoseg.dt == defaults["dt"]

    def test_embedding_recompute_clears_cached_prediction(self):
        from types import SimpleNamespace

        from micro_sam.sam_annotator._widgets import EmbeddingWidget

        autosegment = SimpleNamespace(_segmenter=object(), _segmenter_key=object())
        state = SimpleNamespace(widgets={"autosegment": autosegment})
        EmbeddingWidget._clear_autosegment_cache(state)
        assert autosegment._segmenter is None
        assert autosegment._segmenter_key is None
