import platform

import numpy as np
import pytest
from skimage.data import binary_blobs

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator.annotator import annotator, detect_ndim, detect_ndim_from_viewer, Annotator
from micro_sam._test_util import check_layer_initialization


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
        # Selecting it as input should rebuild the annotator for 3D.
        viewer = make_napari_viewer_proxy()
        widget = Annotator(viewer)
        assert widget._ndim == 2

        viewer.add_image(np.stack(4 * [binary_blobs(128)]), name="volume")
        widget._embedding_widget.image_selection.reset_choices()
        assert widget._ndim == 3
        # The prompt layers must be recreated with the new dimensionality.
        assert viewer.layers["point_prompts"].ndim == 3
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
        # A channels-first (C, H, W) array is auto-detected as a volume; forcing '2d' reads it as a
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
