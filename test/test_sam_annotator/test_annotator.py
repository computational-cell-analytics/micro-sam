import platform

import numpy as np
import pytest
import torch
from skimage.data import binary_blobs

from micro_sam.v2.util import DEFAULT_MODEL, DEFAULT_TILE_SHAPE, DEFAULT_HALO
from micro_sam.sam_annotator.annotator import annotator, detect_ndim, detect_ndim_from_viewer, Annotator
from micro_sam._test_util import check_layer_initialization


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize(
    "shape, ndim, options, expected_tile, expected_halo",
    [
        ((2048, 2048), 2, [], DEFAULT_TILE_SHAPE, DEFAULT_HALO),
        ((2048, 2048, 3), 2, [], DEFAULT_TILE_SHAPE, DEFAULT_HALO),
        ((3, 2048, 1024), 2, [], DEFAULT_TILE_SHAPE, DEFAULT_HALO),
        ((768, 768), 2, [], None, None),
        ((768, 769), 2, [], DEFAULT_TILE_SHAPE, DEFAULT_HALO),
        ((4, 128, 2048), 3, [], DEFAULT_TILE_SHAPE, DEFAULT_HALO),
        ((2048, 32, 32), 3, [], None, None),
        ((2048, 2048), 2, ["--tile_shape", "420", "420", "--overlap", "64", "64"], (420, 420), (64, 64)),
        ((2048, 2048), 2, ["--overlap", "64", "64"], DEFAULT_TILE_SHAPE, (64, 64)),
    ],
)
def test_cli_annotator_default_tiling(
    make_napari_viewer_proxy, monkeypatch, shape, ndim, options, expected_tile, expected_halo,
):
    """CLI startup must resolve tiling before computing embeddings and show the same settings."""
    from click.testing import CliRunner
    from micro_sam._cli import cli
    from micro_sam import util
    from micro_sam.sam_annotator._state import AnnotatorState
    import napari

    image = np.zeros(shape, dtype="uint8")
    viewer = make_napari_viewer_proxy()
    captured = {}
    monkeypatch.setattr(util, "load_image_data", lambda *args, **kwargs: image)
    monkeypatch.setattr(napari, "Viewer", lambda: viewer)
    monkeypatch.setattr(napari, "run", lambda: None)
    monkeypatch.setattr(
        AnnotatorState, "initialize_predictor", lambda self, image, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(Annotator, "_update_image", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        cli, ["annotator", "segmentation", "-i", "image.tif", "--ndim", str(ndim), *options],
    )
    assert result.exit_code == 0, result.output or repr(result.exception)
    assert captured["tile_shape"] == expected_tile
    assert captured["halo"] == expected_halo
    widget = AnnotatorState().widgets["embeddings"]
    assert widget.tiling == ("no" if expected_tile is None else "yes")
    if expected_tile is not None:
        assert (widget.tile_x, widget.tile_y) == expected_tile
        assert (widget.halo_x, widget.halo_y) == expected_halo
    viewer.close()


def _run_annotator_cli(make_napari_viewer_proxy, monkeypatch, tool, image, options=()):
    """Invoke one of the 'micro_sam annotator' commands and capture how it computed the embeddings.

    Returns the kwargs passed to 'initialize_predictor' and the embedding widget the tool synced,
    so a test can check that the tool computed the embeddings with the settings it displays.
    """
    from click.testing import CliRunner
    from micro_sam._cli import cli
    from micro_sam import util
    from micro_sam.sam_annotator._state import AnnotatorState
    from micro_sam.sam_annotator.annotator_tracking import AnnotatorTracking
    from micro_sam.sam_annotator.object_classifier import ObjectClassifier
    from micro_sam.sam_annotator.pixel_classifier import PixelClassifier
    import napari

    viewer = make_napari_viewer_proxy()
    captured = {}
    monkeypatch.setattr(util, "load_image_data", lambda *args, **kwargs: image)
    monkeypatch.setattr(napari, "Viewer", lambda: viewer)
    monkeypatch.setattr(napari, "run", lambda: None)
    monkeypatch.setattr(
        AnnotatorState, "initialize_predictor", lambda self, image, **kwargs: captured.update(kwargs),
    )
    for annotator_class in (Annotator, AnnotatorTracking, ObjectClassifier, PixelClassifier):
        monkeypatch.setattr(annotator_class, "_update_image", lambda *args, **kwargs: None)

    result = CliRunner().invoke(cli, ["annotator", tool, "-i", "image.tif", *options])
    assert result.exit_code == 0, result.output or repr(result.exception)
    return captured, AnnotatorState().widgets["embeddings"], viewer


# The tracking annotator is always 3d, the other tools take the dimensionality from the CLI.
@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize(
    "tool, options",
    [
        ("segmentation", ["--ndim", "2"]),
        ("pixel-classification", ["--ndim", "2"]),
        ("object-classification", ["--ndim", "2"]),
        ("tracking", []),
    ],
)
def test_cli_default_tiling_is_the_same_for_all_tools(make_napari_viewer_proxy, monkeypatch, tool, options):
    """Every annotator must compute the embeddings with the same default tiling for a large image."""
    image = np.zeros((2, 2048, 2048) if tool == "tracking" else (2048, 2048), dtype="uint8")
    captured, widget, viewer = _run_annotator_cli(make_napari_viewer_proxy, monkeypatch, tool, image, options)

    assert captured["tile_shape"] == DEFAULT_TILE_SHAPE
    assert captured["halo"] == DEFAULT_HALO
    # The tool must also show the settings it used, so the GUI does not claim a different tiling.
    assert widget.tiling == "yes"
    assert (widget.tile_x, widget.tile_y) == DEFAULT_TILE_SHAPE
    assert (widget.halo_x, widget.halo_y) == DEFAULT_HALO
    viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize(
    "tool, options",
    [
        ("segmentation", ["--ndim", "2"]),
        ("pixel-classification", ["--ndim", "2"]),
        ("object-classification", ["--ndim", "2"]),
    ],
)
def test_cli_reuses_cached_embeddings(make_napari_viewer_proxy, monkeypatch, tmp_path, tool, options):
    """Passing an embedding path must reuse the cache instead of recomputing it with another tiling."""
    from types import SimpleNamespace
    from micro_sam.util import _open_embeddings, _write_embedding_signature

    image = np.zeros((2048, 2048), dtype="uint8")
    predictor = SimpleNamespace(model_type="hvit_t_cells", model_name="hvit_t_cells", _hash=None, device="cpu")

    # Embeddings computed earlier (e.g. by another tool) with a tiling that is not the default.
    embedding_path = str(tmp_path / "embeddings.zarr")
    f = _open_embeddings(embedding_path, mode="a")
    _write_embedding_signature(f, image, predictor, (420, 420), (64, 64), input_size=None, original_size=None)
    getattr(f, "file", f).close()

    captured, widget, viewer = _run_annotator_cli(
        make_napari_viewer_proxy, monkeypatch, tool, image, [*options, "-e", embedding_path],
    )
    assert captured["save_path"] == embedding_path
    assert captured["tile_shape"] == (420, 420)
    assert captured["halo"] == (64, 64)
    assert (widget.tile_x, widget.tile_y) == (420, 420)
    assert (widget.halo_x, widget.halo_y) == (64, 64)
    viewer.close()


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

    def _dispatch(self, monkeypatch, *, apply_to_volume, current_slice=2, device="cuda"):
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
        progress_events = []

        def fake_run_apg(state, run_raw, ndim, z, pbar_init=None, pbar_update=None):
            calls.update(run_raw_shape=tuple(run_raw.shape), ndim=ndim, z=z)
            for total, description in [(2, "Prompt batches"), (1, "Merge masks")]:
                pbar_init(total, description)
                pbar_update(total)
            return np.zeros(run_raw.shape if ndim == 3 else run_raw.shape[-2:], dtype="uint32")

        # Duck-typed stand-in so we exercise the '__call__' dispatch without instantiating a QWidget.
        widget = SimpleNamespace(
            _viewer=viewer, mode="apg", volumetric=True, apply_to_volume=apply_to_volume,
            with_decoder=True, _is_tracking=False, _run_apg=fake_run_apg,
        )

        def fake_pbar():
            signal = lambda name: SimpleNamespace(emit=lambda *args: progress_events.append((name, args)))  # noqa
            signals = SimpleNamespace(
                pbar_total=signal("total"), pbar_description=signal("description"),
                pbar_update=signal("update"), pbar_stop=signal("stop"), pbar_reset=signal("reset"),
            )
            return SimpleNamespace(), signals

        # A decoder on 'device'; APG on a whole volume is only offered on an accelerator.
        decoder = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device(device))]))
        monkeypatch.setattr(
            _widgets, "AnnotatorState",
            lambda: SimpleNamespace(get_image_name=lambda v: "image", decoder=decoder, image_embeddings=None),
        )
        monkeypatch.setattr(_widgets, "_validate_layers", lambda *a, **k: False)
        monkeypatch.setattr(_widgets, "_validate_embeddings", lambda *a, **k: False)
        monkeypatch.setattr(_widgets, "_select_layer", lambda *a, **k: None)
        monkeypatch.setattr(_widgets, "_create_pbar_for_threadworker", fake_pbar)
        monkeypatch.setattr(
            _widgets, "QtWidgets",
            SimpleNamespace(QApplication=SimpleNamespace(processEvents=lambda *a, **k: None)),
        )

        AutoSegmentWidget.__call__(widget)
        if calls:
            assert [event for event in progress_events if event[0] != "description"] == [
                ("reset", ()), ("total", (2,)), ("update", (2,)),
                ("reset", ()), ("total", (1,)), ("update", (1,)), ("stop", ()),
            ]
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

    @pytest.mark.parametrize("apply_to_volume", [False, True])
    def test_apg_on_a_volume_is_refused_on_the_cpu(self, monkeypatch, apply_to_volume):
        """Volumetric data rules APG out on the CPU as a whole - the current slice included, since
        it is the data kind and not the individual run that decides."""
        from micro_sam.sam_annotator import _widgets

        messages = []
        monkeypatch.setattr(_widgets, "_generate_message", lambda kind, msg: messages.append((kind, msg)))
        calls = self._dispatch(monkeypatch, apply_to_volume=apply_to_volume, device="cpu")

        assert calls == {}  # the run never starts
        assert len(messages) == 1
        kind, msg = messages[0]
        assert kind == "error"
        assert "CPU" in msg


class TestApgAvailability:
    """APG prompts the video predictor, so the annotator offers it for volumetric data and for
    tracking only on an accelerator. A plain 2d image runs it anywhere. The restriction is the
    annotator's; the backend runs wherever it is pointed (see test_v2_automatic_prompt_generation)."""

    def _state(self, device, embeddings=None):
        from types import SimpleNamespace

        decoder = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device(device))]))
        return SimpleNamespace(decoder=decoder, image_embeddings=embeddings or {"input_size": (8, 8)})

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_a_plain_2d_image_runs_anywhere(self, device):
        from micro_sam.sam_annotator._widgets import _apg_error

        assert _apg_error(self._state(device), volumetric=False) is None

    @pytest.mark.parametrize("device", ["cuda", "cuda:1", "mps"])
    def test_accelerators_run_volumetric_apg(self, device):
        from micro_sam.sam_annotator._widgets import _apg_error

        # An indexed device ('cuda:1') must be recognized just like the bare one.
        assert _apg_error(self._state(device), volumetric=True) is None
        assert _apg_error(self._state(device), volumetric=True, is_tracking=True) is None

    def test_volumetric_data_is_refused_on_the_cpu(self):
        """For the whole widget: a single slice of a volume is refused too, not just the whole volume."""
        from micro_sam.sam_annotator._widgets import _apg_error

        error = _apg_error(self._state("cpu"), volumetric=True)
        assert error is not None
        assert "CPU" in error and "volumetric data" in error
        assert "'sparse' or 'dense'" in error  # it says what to use instead

    def test_tracking_is_refused_on_the_cpu(self):
        from micro_sam.sam_annotator._widgets import _apg_error

        error = _apg_error(self._state("cpu"), volumetric=True, is_tracking=True)
        assert error is not None
        assert "CPU" in error and "tracking" in error

    @pytest.mark.parametrize("device", ["cuda", "mps"])
    def test_tiled_embeddings_are_accepted_for_a_volume_on_an_accelerator(self, device):
        """A tiled volume is segmented block by block by the tiled APG generator."""
        from micro_sam.sam_annotator._widgets import _apg_error

        # Tiled embeddings have no top-level 'input_size'.
        tiled = self._state(device, embeddings={"input_size": None})
        assert _apg_error(tiled, volumetric=True) is None

    def test_tiled_embeddings_are_refused_for_a_volume_on_the_cpu(self):
        from micro_sam.sam_annotator._widgets import _apg_error

        error = _apg_error(self._state("cpu", embeddings={"input_size": None}), volumetric=True)
        assert error is not None and "CPU" in error


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
class TestApgModeIsHiddenOnCpu:
    """A mode that cannot run is not offered: on the CPU the dropdown drops APG for volumetric data
    and for tracking. The device is only known once the model is loaded, so the list is rebuilt by
    the sync that runs after 'Compute Embeddings'."""

    def _load_decoder(self, device):
        """Put a decoder on 'device' into the state, as computing the embeddings would."""
        from types import SimpleNamespace
        from micro_sam.sam_annotator._state import AnnotatorState

        state = AnnotatorState()
        state.decoder = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=torch.device(device))])
        )
        return state

    def _choices(self, widget):
        return [widget.mode_dropdown.itemText(i) for i in range(widget.mode_dropdown.count())]

    def test_a_2d_annotator_keeps_apg_on_the_cpu(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        autoseg = Annotator(viewer, ndim=2)._widgets["autosegment"]
        self._load_decoder("cpu")
        autoseg._reset_segmentation_mode(True)

        assert self._choices(autoseg) == ["sparse", "apg", "dense"]
        assert autoseg.mode == "sparse"
        viewer.close()

    def test_a_3d_annotator_drops_apg_on_the_cpu(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        autoseg = Annotator(viewer, ndim=3)._widgets["autosegment"]
        # Before the model is loaded the device is unknown, so APG is still offered.
        assert "apg" in self._choices(autoseg)

        self._load_decoder("cpu")
        autoseg._reset_segmentation_mode(True)

        assert self._choices(autoseg) == ["sparse", "dense"]
        assert autoseg.mode == "sparse"  # the default falls back to the first mode that can run
        assert autoseg.run_button.isEnabled() is True  # sparse and dense still work on the CPU
        viewer.close()

    def test_a_3d_annotator_keeps_apg_on_an_accelerator(self, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        autoseg = Annotator(viewer, ndim=3)._widgets["autosegment"]
        self._load_decoder("cuda")
        autoseg._reset_segmentation_mode(True)

        assert self._choices(autoseg) == ["apg", "sparse", "dense"]
        assert autoseg.mode == "apg"
        viewer.close()

    def test_the_tracking_widget_drops_apg_on_the_cpu(self, qtbot):
        from micro_sam.sam_annotator._widgets import AutoTrackWidget

        widget = AutoTrackWidget(viewer=None, with_decoder=True, volumetric=True)
        qtbot.addWidget(widget)
        self._load_decoder("cpu")
        widget._reset_segmentation_mode(True)

        assert self._choices(widget) == ["sparse", "dense"]
        assert widget.mode == "sparse"

    def test_switching_the_device_puts_apg_back(self, make_napari_viewer_proxy):
        """Recomputing on a GPU after a CPU run must offer APG again, even though the decoder
        availability did not change."""
        viewer = make_napari_viewer_proxy()
        autoseg = Annotator(viewer, ndim=3)._widgets["autosegment"]

        self._load_decoder("cpu")
        autoseg._reset_segmentation_mode(True)
        assert "apg" not in self._choices(autoseg)

        self._load_decoder("cuda")
        autoseg._reset_segmentation_mode(True)
        assert self._choices(autoseg) == ["apg", "sparse", "dense"]
        assert autoseg.mode == "apg"
        viewer.close()


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

    def test_enabling_decoder_persistence_writes_the_in_memory_state(self, monkeypatch):
        from types import MethodType, SimpleNamespace

        import micro_sam.precompute_state as precompute_state
        import micro_sam.v2.instance_segmentation as instance_segmentation
        from micro_sam.sam_annotator._widgets import AutoSegmentWidget

        calls = {"cache": [], "save": []}
        prediction = np.ones((4, 8, 8), dtype="float32")

        class Decoder:
            def parameters(self):
                yield SimpleNamespace(device="cpu")

        class Segmenter:
            def __init__(self):
                self.state = {"prediction": prediction}

            def get_state(self):
                return self.state

            def set_state(self, state):
                self.state = state

            def generate(self, **kwargs):
                return np.zeros((8, 8), dtype="uint32")

            def clear_state(self):
                pass

        def fake_cache(*args, **kwargs):
            calls["cache"].append(args[4])
            return Segmenter()

        def fake_save(state, save_path, **kwargs):
            calls["save"].append((state, save_path, kwargs))

        monkeypatch.setattr(precompute_state, "cache_autoseg_state", fake_cache)
        monkeypatch.setattr(precompute_state, "save_ais_state", fake_save)
        monkeypatch.setattr(instance_segmentation, "get_unisam2_segmentation_generator", lambda *a, **k: Segmenter())

        embedding_widget = SimpleNamespace(cache_state=False)
        state = SimpleNamespace(
            predictor=SimpleNamespace(model_type="hvit_t_cells"), decoder=Decoder(),
            image_embeddings={"input_size": (8, 8)}, inference_devices=None,
            embedding_path="/tmp/embeddings.zarr", data_signature="data",
            widgets={"embeddings": embedding_widget},
        )
        widget = SimpleNamespace(
            volumetric=False, mode="sparse", _segmenter=None, _segmenter_key=None,
            _decoder_state=None, _decoder_state_key=None, _decoder_state_save_path=None,
            _proposals=None, _proposals_key=None, _postproc_kwargs=lambda: {},
        )
        for name in (
            "_state_save_path", "_release_segmenter", "_decoder_key", "_cached_decoder_state",
            "_store_decoder_state", "_persist_decoder_state",
        ):
            setattr(widget, name, MethodType(getattr(AutoSegmentWidget, name), widget))

        raw = np.zeros((8, 8), dtype="uint8")
        AutoSegmentWidget._run_unisam2(widget, state, raw, ndim=2, z=None)
        embedding_widget.cache_state = True
        AutoSegmentWidget._run_unisam2(widget, state, raw, ndim=2, z=None)

        assert calls["cache"] == [None]
        assert len(calls["save"]) == 1
        saved_state, save_path, kwargs = calls["save"][0]
        assert saved_state["prediction"] is prediction
        assert save_path == "/tmp/embeddings.zarr"
        assert kwargs == {"state_index": None, "model_type": "hvit_t_cells"}
        assert widget._decoder_state_save_path == "/tmp/embeddings.zarr"


@pytest.mark.parametrize("z", [None, 3])
def test_apg_widget_reuses_the_decoder_state(monkeypatch, z):
    from types import MethodType, SimpleNamespace

    import micro_sam.precompute_state as precompute_state
    import micro_sam.v2.instance_segmentation as instance_segmentation
    from micro_sam.sam_annotator._widgets import AutoSegmentWidget

    calls = {"cache": 0, "factory": 0}

    class Decoder:
        def parameters(self):
            yield SimpleNamespace(device="cpu")

    class DecoderSegmenter:
        def get_state(self):
            return {"prediction": np.ones((4, 8, 8), dtype="float32")}

    class PromptGenerator:
        def __init__(self):
            self.propose_calls = 0

        def set_state(self, state):
            self.state = state

        def propose(self, **kwargs):
            self.propose_calls += 1
            self.propose_kwargs = kwargs
            return ["proposal"]

        def select(self, proposals, **kwargs):
            self.proposals = proposals
            self.select_kwargs = kwargs
            return np.ones((8, 8), dtype="uint32")

    prompt_generator = PromptGenerator()

    def fake_cache(*args, **kwargs):
        calls["cache"] += 1
        assert args[0] == "ais"
        assert kwargs["is_tiled"] is False
        assert kwargs["i"] == z
        return DecoderSegmenter()

    def fake_factory(**kwargs):
        calls["factory"] += 1
        assert kwargs["segmentation_mode"] == "apg"
        return prompt_generator

    monkeypatch.setattr(precompute_state, "cache_autoseg_state", fake_cache)
    monkeypatch.setattr(instance_segmentation, "get_instance_segmentation_generator", fake_factory)

    image_embeddings = {
        "features": np.zeros((1, 4, 8, 8), dtype="float32"),
        "input_size": 1024,
        "original_size": (8, 8),
    }
    state = SimpleNamespace(
        predictor=SimpleNamespace(model=object(), model_type="hvit_t_cells"),
        decoder=Decoder(), image_embeddings=image_embeddings, inference_devices=None,
        data_signature="image", widgets={}, embedding_path=None,
    )
    widget = SimpleNamespace(
        _segmenter=None, _segmenter_key=None, _decoder_state=None, _decoder_state_key=None,
        _proposals=None, _proposals_key=None,
        _state_save_path=lambda state: "/tmp/embeddings.zarr",
        _apg_propose_kwargs=lambda: {"candidate_threshold": 1.5},
        _apg_select_kwargs=lambda: {"score_threshold": 0.6},
    )
    for name in (
        "_release_segmenter", "_decoder_key", "_cached_decoder_state", "_store_decoder_state",
        "_persist_decoder_state",
    ):
        setattr(widget, name, MethodType(getattr(AutoSegmentWidget, name), widget))

    stages, updates = [], []

    def pbar_init(total, description):
        stages.append((total, description))

    result = AutoSegmentWidget._run_apg(
        widget, state, np.zeros((8, 8)), ndim=2, z=z,
        pbar_init=pbar_init, pbar_update=updates.append,
    )
    second_result = AutoSegmentWidget._run_apg(
        widget, state, np.zeros((8, 8)), ndim=2, z=z,
        pbar_init=pbar_init, pbar_update=updates.append,
    )

    assert calls == {"cache": 1, "factory": 1}
    assert prompt_generator.state["image_embeddings"] is image_embeddings
    assert prompt_generator.state.get("i") == z
    assert prompt_generator.propose_calls == 1
    assert prompt_generator.propose_kwargs["pbar_init"] is pbar_init
    assert prompt_generator.propose_kwargs["pbar_update"] == updates.append
    assert stages == [(1, "APG: merging masks")] * 2
    assert updates == [1, 1]
    assert prompt_generator.select_kwargs == {"score_threshold": 0.6}
    assert np.array_equal(result, second_result)

    # Rebuilding APG after a mode switch reuses the decoder state but regenerates the prompts.
    widget._segmenter = None
    widget._segmenter_key = None
    widget._proposals = None
    widget._proposals_key = None
    third_result = AutoSegmentWidget._run_apg(
        widget, state, np.zeros((8, 8)), ndim=2, z=z,
        pbar_init=pbar_init, pbar_update=updates.append,
    )

    assert calls == {"cache": 1, "factory": 2}
    assert prompt_generator.propose_calls == 2
    assert np.array_equal(result, third_result)


@pytest.mark.parametrize("z", [None, 3])
def test_tiled_apg_widget_hands_the_raw_image_and_embeddings_to_the_generator(monkeypatch, z):
    from types import MethodType, SimpleNamespace

    import micro_sam.precompute_state as precompute_state
    import micro_sam.v2.instance_segmentation as instance_segmentation
    from micro_sam.sam_annotator._widgets import AutoSegmentWidget

    calls = {"factory": 0, "initialize": 0, "generate": 0}

    class Decoder:
        def parameters(self):
            yield SimpleNamespace(device="cpu")

    class TiledPromptGenerator:
        def initialize(self, image, **kwargs):
            calls["initialize"] += 1
            self.image = image
            self.initialize_kwargs = kwargs

        def generate(self, **kwargs):
            calls["generate"] += 1
            self.generate_kwargs = kwargs
            return np.ones((8, 8), dtype="uint32")

        def clear_state(self):
            pass

    prompt_generator = TiledPromptGenerator()

    def fail_cache(*args, **kwargs):
        pytest.fail("Tiled APG must not use the whole-image decoder state.")

    def fake_factory(**kwargs):
        calls["factory"] += 1
        assert kwargs["is_tiled"] is True
        assert kwargs["segmentation_mode"] == "apg"
        return prompt_generator

    monkeypatch.setattr(precompute_state, "cache_autoseg_state", fail_cache)
    monkeypatch.setattr(instance_segmentation, "get_instance_segmentation_generator", fake_factory)

    features = SimpleNamespace(attrs={"tile_shape": (4, 4), "halo": (1, 1)})
    image_embeddings = {"features": features, "input_size": None}
    state = SimpleNamespace(
        predictor=SimpleNamespace(model=object(), model_type="hvit_t_cells"),
        decoder=Decoder(), image_embeddings=image_embeddings, inference_devices=None,
        data_signature="image", widgets={}, embedding_path=None,
    )
    widget = SimpleNamespace(
        _segmenter=None, _segmenter_key=None, _decoder_state=None, _decoder_state_key=None,
        _decoder_state_save_path=None, _proposals=None, _proposals_key=None,
        _state_save_path=lambda state: "/tmp/embeddings.zarr",
        _apg_kwargs=lambda ndim: {"candidate_threshold": 1.5, "score_threshold": 0.6},
    )
    widget._release_segmenter = MethodType(AutoSegmentWidget._release_segmenter, widget)

    def pbar_init(total, desc):
        pass

    def pbar_update(n):
        pass

    raw = np.zeros((8, 8), dtype="uint8")
    result = AutoSegmentWidget._run_apg(
        widget, state, raw, ndim=2, z=z, pbar_init=pbar_init, pbar_update=pbar_update,
    )
    second_result = AutoSegmentWidget._run_apg(
        widget, state, raw, ndim=2, z=z, pbar_init=pbar_init, pbar_update=pbar_update,
    )

    assert calls == {"factory": 1, "initialize": 1, "generate": 2}
    assert prompt_generator.image is raw
    # The blocks read the embeddings instead of encoding, a slice of a volume by its index.
    assert prompt_generator.initialize_kwargs == {
        "ndim": 2, "tile_shape": (4, 4), "halo": (1, 1), "verbose": False,
        "image_embeddings": image_embeddings, "i": z,
    }
    assert prompt_generator.generate_kwargs == {
        "candidate_threshold": 1.5, "score_threshold": 0.6,
        "pbar_init": pbar_init, "pbar_update": pbar_update,
    }
    assert np.array_equal(result, second_result)


@pytest.mark.parametrize("tile_z, halo_z, expected_z", [(4, 2, (4, 2)), (32, 2, (10, 0))])
def test_tiled_apg_widget_blocks_a_volume_in_z(monkeypatch, tile_z, halo_z, expected_z):
    """A tiled volume is segmented in (z, y, x) blocks: the widget's z tiling in front of the in-plane
    tiling of the embeddings, with no z halo once a single z block spans the volume."""
    from types import MethodType, SimpleNamespace

    import micro_sam.precompute_state as precompute_state
    import micro_sam.v2.instance_segmentation as instance_segmentation
    from micro_sam.sam_annotator._widgets import AutoSegmentWidget

    class TiledPromptGenerator:
        def initialize(self, image, **kwargs):
            self.image = image
            self.initialize_kwargs = kwargs

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return np.ones(self.image.shape, dtype="uint32")

    prompt_generator = TiledPromptGenerator()

    def fake_factory(**kwargs):
        assert kwargs["is_tiled"] is True and kwargs["ndim"] == 3
        return prompt_generator

    def fail_cache(*args, **kwargs):
        pytest.fail("Tiled APG must not use the whole-volume decoder state.")

    monkeypatch.setattr(precompute_state, "cache_autoseg_state", fail_cache)
    monkeypatch.setattr(instance_segmentation, "get_instance_segmentation_generator", fake_factory)

    features = SimpleNamespace(attrs={"tile_shape": (4, 4), "halo": (1, 1)})
    state = SimpleNamespace(
        predictor=SimpleNamespace(model=object(), model_type="hvit_t_cells"),
        decoder=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="mps")])),
        image_embeddings={"features": features, "input_size": None}, inference_devices=None,
        data_signature="volume", widgets={}, embedding_path=None,
    )
    widget = SimpleNamespace(
        _segmenter=None, _segmenter_key=None, _proposals=None, _proposals_key=None,
        tile_z=tile_z, halo_z=halo_z,
        _state_save_path=lambda state: None,
        _apg_kwargs=lambda ndim: {"candidate_threshold": (1.5, 2.0), "n_objects_per_pass": 16},
    )
    widget._release_segmenter = MethodType(AutoSegmentWidget._release_segmenter, widget)
    widget._z_tiling = MethodType(AutoSegmentWidget._z_tiling, widget)

    raw = np.zeros((10, 8, 8), dtype="uint8")
    result = AutoSegmentWidget._run_apg(widget, state, raw, ndim=3, z=None)

    assert prompt_generator.image is raw
    assert prompt_generator.initialize_kwargs == {
        "ndim": 3, "tile_shape": (expected_z[0], 4, 4), "halo": (expected_z[1], 1, 1), "verbose": False,
        "image_embeddings": state.image_embeddings, "i": None,
    }
    assert prompt_generator.generate_kwargs["n_objects_per_pass"] == 16
    assert result.shape == raw.shape


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
class TestAutoSegDefaultMode:
    """The default automatic-segmentation mode reflects the default model's decoder availability."""

    def test_default_model_has_decoder_predicate(self):
        from micro_sam.v2.util import has_registered_decoder, DEFAULT_MODEL
        assert has_registered_decoder(DEFAULT_MODEL) is True  # the Microscopy default has a decoder
        assert has_registered_decoder("hvit_t") is False  # a plain backbone does not

    @pytest.mark.parametrize("device, expected", [("cpu", "sparse"), ("cuda", "apg"), ("mps", "apg")])
    def test_autoseg_default_for_decoder_model(self, make_napari_viewer_proxy, monkeypatch, device, expected):
        from micro_sam.sam_annotator import _widgets
        monkeypatch.setattr(_widgets.util, "get_device", lambda requested=None: requested or device)
        # Before embeddings are computed, choose the default for the available hardware.
        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=2)
        autoseg = widget._widgets["autosegment"]
        assert autoseg.with_decoder is True
        assert autoseg.mode == expected
        assert autoseg.mode_dropdown.currentText() == expected
        choices = [autoseg.mode_dropdown.itemText(i) for i in range(autoseg.mode_dropdown.count())]
        assert choices == (["sparse", "apg", "dense"] if device == "cpu" else ["apg", "sparse", "dense"])
        parameter = "density_threshold_param" if expected == "sparse" else "candidate_threshold_param"
        assert hasattr(autoseg, parameter)
        viewer.close()

    def test_autoseg_is_disabled_without_a_decoder(self, make_napari_viewer_proxy, monkeypatch):
        # Every mode runs off the decoder predictions - AMG is not offered in the annotator - so a
        # model without a decoder disables the run button instead of falling back to another mode.
        from micro_sam.sam_annotator._widgets import AutoSegmentWidget
        from micro_sam.sam_annotator import _widgets
        monkeypatch.setattr(_widgets.util, "get_device", lambda requested=None: requested or "cuda")

        viewer = make_napari_viewer_proxy()
        autoseg = AutoSegmentWidget(viewer, with_decoder=False, volumetric=False)
        choices = [autoseg.mode_dropdown.itemText(i) for i in range(autoseg.mode_dropdown.count())]
        assert choices == ["apg", "sparse", "dense"]  # never 'amg'
        assert autoseg.mode == "apg"
        assert autoseg.run_button.isEnabled() is False

        autoseg._reset_segmentation_mode(True)
        assert autoseg.mode == "apg"
        assert autoseg.run_button.isEnabled() is True

        autoseg._reset_segmentation_mode(False)
        assert autoseg.run_button.isEnabled() is False
        viewer.close()

    def test_running_without_a_decoder_reports_the_reason(self, make_napari_viewer_proxy, monkeypatch):
        # Clicking run anyway (the button can be enabled from a stale decoder state) explains why
        # nothing happens rather than falling back to AMG.
        from micro_sam.sam_annotator import _widgets
        from micro_sam.sam_annotator._state import AnnotatorState

        messages = []
        monkeypatch.setattr(_widgets, "_generate_message", lambda kind, msg: messages.append((kind, msg)))

        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=2)
        autoseg = widget._widgets["autosegment"]
        assert AnnotatorState().decoder is None  # no embeddings computed yet
        autoseg()

        assert len(messages) == 1
        kind, msg = messages[0]
        assert kind == "error"
        assert "decoder" in msg
        viewer.close()

    def test_apg_controls_use_backend_defaults(self, make_napari_viewer_proxy):
        from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION, default_prompt_generation

        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=2)
        autoseg = widget._widgets["autosegment"]
        autoseg.mode_dropdown.setCurrentText("apg")

        defaults = default_prompt_generation(DEFAULT_MODEL, is_volume=False)
        assert autoseg.mode == "apg"
        assert autoseg.candidate_threshold_param.value() == defaults["candidate_threshold"]
        assert autoseg.foreground_threshold_param.value() == defaults["foreground_threshold"]
        assert autoseg.min_candidate_size_param.value() == defaults["min_candidate_size"]
        assert autoseg.score_threshold_param.value() == defaults["score_threshold"]
        assert autoseg.max_overlap_param.value() == defaults["max_overlap"]
        assert autoseg.min_object_size_param.value() == defaults["min_size"]
        assert autoseg.multimasking_checkbox.isChecked() == DEFAULT_PROMPT_GENERATION["multimasking"]
        assert autoseg.refine_with_box_prompts_checkbox.isChecked() == (
            DEFAULT_PROMPT_GENERATION["refinement"] == "boxes"
        )
        viewer.close()

    def test_volumetric_apg_controls_build_generate_kwargs(self, make_napari_viewer_proxy):
        from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION, default_prompt_generation

        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=3)
        autoseg = widget._widgets["autosegment"]
        autoseg.mode_dropdown.setCurrentText("apg")

        # A volume run uses the 3d defaults, the same the automatic segmentation CLI and API use.
        autoseg.apply_to_volume_checkbox.setChecked(True)
        defaults_3d = default_prompt_generation(DEFAULT_MODEL, is_volume=True)
        kwargs = autoseg._apg_kwargs(ndim=3)
        for key in ("candidate_threshold", "min_candidate_size", "score_threshold", "max_overlap", "min_size", "sigma"):
            assert kwargs[key] == pytest.approx(defaults_3d[key]), key
        assert kwargs["n_objects_per_pass"] == DEFAULT_PROMPT_GENERATION["n_objects_per_pass"]
        assert kwargs["early_stop_patience"] == DEFAULT_PROMPT_GENERATION["early_stop_patience"]

        autoseg.early_stop_patience_param.setValue(3)
        assert autoseg._apg_kwargs(ndim=3)["early_stop_patience"] == 3
        viewer.close()

    def test_apply_to_volume_switches_between_slice_and_volume_apg_parameters(self, make_napari_viewer_proxy):
        """A slice run and a volume run start from their own defaults and keep their own edits."""
        from micro_sam.v2.automatic_prompt_generation import default_prompt_generation

        viewer = make_napari_viewer_proxy()
        autoseg = Annotator(viewer, ndim=3)._widgets["autosegment"]
        autoseg.mode_dropdown.setCurrentText("apg")
        defaults_2d = default_prompt_generation(DEFAULT_MODEL, is_volume=False)
        defaults_3d = default_prompt_generation(DEFAULT_MODEL, is_volume=True)

        def shown():
            return (
                autoseg.candidate_threshold_param.value(), autoseg.min_object_size_param.value(),
                autoseg.sigma_param.value(), autoseg.candidate_threshold_high_row.isVisibleTo(autoseg),
            )

        slice_defaults = (defaults_2d["candidate_threshold"], defaults_2d["min_size"], defaults_2d["sigma"], False)
        volume_defaults = (defaults_3d["candidate_threshold"][0], defaults_3d["min_size"], defaults_3d["sigma"], True)
        assert shown() == pytest.approx(slice_defaults)

        autoseg.apply_to_volume_checkbox.setChecked(True)
        assert shown() == pytest.approx(volume_defaults)
        autoseg.min_object_size_param.setValue(77)

        autoseg.apply_to_volume_checkbox.setChecked(False)
        assert shown() == pytest.approx(slice_defaults)
        assert autoseg._apg_kwargs(ndim=2)["min_size"] == defaults_2d["min_size"]

        autoseg.apply_to_volume_checkbox.setChecked(True)  # the volume edit is still there
        assert shown() == pytest.approx((volume_defaults[0], 77, volume_defaults[2], True))
        assert autoseg._apg_kwargs(ndim=3)["min_size"] == 77
        viewer.close()

    def test_apg_kwargs_split_matches_propose_and_select(self, make_napari_viewer_proxy):
        """The 2d run calls 'propose' and 'select' separately, so each must get exactly its own
        arguments and the two together must still cover everything 'generate' was given."""
        import inspect

        from micro_sam.v2.automatic_prompt_generation import AutomaticPromptGenerator

        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=2)
        autoseg = widget._widgets["autosegment"]
        autoseg.mode_dropdown.setCurrentText("apg")

        propose_kwargs = autoseg._apg_propose_kwargs()
        select_kwargs = autoseg._apg_select_kwargs()

        propose_params = set(inspect.signature(AutomaticPromptGenerator.propose).parameters) - {"self"}
        select_params = set(inspect.signature(AutomaticPromptGenerator.select).parameters) - {"self", "proposals"}
        assert set(propose_kwargs) <= propose_params
        assert set(select_kwargs) == select_params

        # Together they are what the single-call form would have passed.
        assert set(propose_kwargs) | set(select_kwargs) == set(autoseg._apg_kwargs(ndim=2))
        viewer.close()

    def test_commit_settings_cover_every_mode(self, make_napari_viewer_proxy):
        """Committing an automatic segmentation to a commit path records the widget's settings, so
        every mode must report parameters it actually has, in a json-serializable form."""
        import json

        from micro_sam.sam_annotator._state import AnnotatorState
        from micro_sam.sam_annotator._widgets import _get_auto_segmentation_options

        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer, ndim=3)
        autoseg = widget._widgets["autosegment"]
        state = AnnotatorState()

        for mode in ("apg", "sparse", "dense"):
            autoseg.mode_dropdown.setCurrentText(mode)
            for apply_to_volume in (False, True):
                autoseg.apply_to_volume = apply_to_volume
                options = _get_auto_segmentation_options(state, np.array([1, 2]))
                assert options["object_ids"] == [1, 2]
                assert options["mode"] == mode
                assert options["apply_to_volume"] is apply_to_volume
                # The z block / halo are part of a volumetric decoder run.
                assert options["tile_z"] == autoseg.tile_z
                assert options["halo_z"] == autoseg.halo_z
                json.dumps(options)  # It is written to the zarr attributes of the commit path.

        assert _get_auto_segmentation_options(state, [])["mode"] == "dense"
        viewer.close()

    def test_autoseg_settings_use_v2_defaults(self):
        from micro_sam.v2.postprocessing import default_postprocessing
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
        defaults = default_postprocessing(DEFAULT_MODEL, "sparse")
        assert autoseg.foreground_threshold == defaults["foreground_threshold"]
        assert autoseg.density_threshold == defaults["density_threshold"]
        assert autoseg.min_object_size == defaults["min_size"]
        assert autoseg.sigma == defaults["sigma"]
        assert autoseg.n_iter == defaults["n_iter"]
        assert autoseg.dt == defaults["dt"]

        autoseg = _FakeAutoSegmentWidget()
        AutoSegmentWidget._dense_settings(autoseg, _FakeSettings())
        defaults = default_postprocessing(DEFAULT_MODEL, "dense")
        assert autoseg.beta == defaults["beta"]
        assert autoseg.density_threshold == defaults["density_threshold"]
        assert autoseg.sigma == defaults["sigma"]
        assert autoseg.n_iter == defaults["n_iter"]
        assert autoseg.dt == defaults["dt"]

    def test_embedding_recompute_clears_cached_prediction(self):
        from types import SimpleNamespace

        from micro_sam.sam_annotator._widgets import EmbeddingWidget

        from types import MethodType

        from micro_sam.sam_annotator._widgets import AutoSegmentWidget

        autosegment = SimpleNamespace(
            _segmenter=None, _segmenter_key=object(),
            _decoder_state={"prediction": 1}, _decoder_state_key=object(),
            _decoder_state_save_path="/tmp/embeddings.zarr",
        )
        autosegment._release_segmenter = MethodType(AutoSegmentWidget._release_segmenter, autosegment)
        autosegment._drop_decoder_state = MethodType(AutoSegmentWidget._drop_decoder_state, autosegment)
        state = SimpleNamespace(widgets={"autosegment": autosegment})
        EmbeddingWidget._clear_autosegment_cache(state)
        assert autosegment._segmenter is None
        assert autosegment._segmenter_key is None
        # The prediction belongs to the embeddings that were just replaced.
        assert autosegment._decoder_state is None
        assert autosegment._decoder_state_key is None
        assert autosegment._decoder_state_save_path is None


@pytest.mark.parametrize("mode", ["apg", "sparse"])
def test_tracking_progress_handles_apg_stages(monkeypatch, mode):
    from types import SimpleNamespace
    from micro_sam.sam_annotator import _widgets

    bar = SimpleNamespace(n=0, total=0, closed=False)
    descriptions = []

    def signal(callback):
        return SimpleNamespace(emit=callback)

    def update(n):
        bar.n += n
        assert bar.n <= bar.total

    signals = SimpleNamespace(
        pbar_reset=signal(lambda: setattr(bar, "n", 0)),
        pbar_total=signal(lambda n: setattr(bar, "total", n)),
        pbar_update=signal(update),
        pbar_description=signal(descriptions.append),
        pbar_stop=signal(lambda: setattr(bar, "closed", True)),
    )
    monkeypatch.setattr(_widgets, "_create_pbar_for_threadworker", lambda: (bar, signals))
    raw = np.zeros((2, 8, 8), dtype="uint8")
    layer = SimpleNamespace(data=np.zeros_like(raw, dtype="uint32"), refresh=lambda: None)

    def segment(state, frame, frame_id, pbar_init, pbar_update):
        stages = [(3, "APG: prompting batches"), (1, "APG: merging masks")] if mode == "apg" else [(1, "Decoder")]
        for total, description in stages:
            pbar_init(total, description)
            for _ in range(total):
                pbar_update(1)
        return np.zeros(frame.shape, dtype="uint32")

    widget = SimpleNamespace(
        mode=mode, _viewer=SimpleNamespace(layers={"auto_segmentation": layer}),
        _n_inplane_tiles=lambda state, raw: 1, _run_frame_segmentation=segment,
        _empty_tracking_warning=lambda: None,
    )
    _widgets.AutoTrackWidget._track_timeseries(widget, SimpleNamespace(), raw)
    assert bar.closed
    if mode == "apg":
        assert "Frame 2/2 - APG: prompting batches" in descriptions
        assert bar.n == bar.total == 1
    else:
        assert bar.n == bar.total == 2


@pytest.mark.parametrize("accelerator", ["cuda", "mps"])
def test_2d_autoseg_default_follows_loaded_device(qtbot, monkeypatch, accelerator):
    from types import SimpleNamespace
    from micro_sam.sam_annotator import _widgets
    from micro_sam.sam_annotator._state import AnnotatorState

    # An accelerator is available, but the user can explicitly load the model on CPU.
    monkeypatch.setattr(_widgets.util, "get_device", lambda requested=None: requested or accelerator)
    widget = _widgets.AutoSegmentWidget(viewer=None, with_decoder=True, volumetric=False)
    qtbot.addWidget(widget)
    assert widget.mode == "apg"

    for device, expected in [("cpu", "sparse"), (accelerator, "apg"), ("cpu", "sparse")]:
        AnnotatorState().decoder = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=torch.device(device))]),
        )
        widget._reset_segmentation_mode(True)
        assert widget.mode == widget.mode_dropdown.currentText() == expected
        assert widget.run_button.isEnabled()
        assert "apg" in widget._mode_choices()

        # Re-syncing the same device must preserve an explicit user choice.
        widget.mode_dropdown.setCurrentText("apg" if device == "cpu" else "dense")
        chosen = widget.mode
        settings = widget.settings
        widget._reset_segmentation_mode(True)
        assert widget.mode == chosen
        assert widget.settings is settings
