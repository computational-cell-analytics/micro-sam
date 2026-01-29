import numpy as np
import pytest

from micro_sam.sam_annotator.annotator import (
    Annotator,
    _detect_ndim,
    annotator,
)


class TestDetectNdim:
    """Test the _detect_ndim helper function."""

    def test_2d_grayscale(self):
        """Test 2D grayscale image detection."""
        image = np.zeros((512, 512), dtype=np.uint8)
        assert _detect_ndim(image) == 2

    def test_2d_rgb(self):
        """Test 2D RGB image detection."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        assert _detect_ndim(image) == 2

    def test_3d_grayscale(self):
        """Test 3D grayscale volume detection."""
        image = np.zeros((10, 512, 512), dtype=np.uint8)
        assert _detect_ndim(image) == 3

    def test_3d_rgb(self):
        """Test 3D RGB volume detection."""
        image = np.zeros((10, 512, 512, 3), dtype=np.uint8)
        assert _detect_ndim(image) == 3

    def test_ambiguous_shape_assumes_3d(self):
        """Test that shape (3, 512, 512) is interpreted as 3D grayscale."""
        image = np.zeros((3, 512, 512), dtype=np.uint8)
        # By default, assumes 3D grayscale rather than RGB 2D
        assert _detect_ndim(image) == 3

    def test_invalid_1d_shape(self):
        """Test that 1D arrays raise ValueError."""
        image = np.zeros(512, dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid image shape"):
            _detect_ndim(image)

    def test_invalid_4d_non_rgb(self):
        """Test that 4D arrays without RGB channel raise ValueError."""
        image = np.zeros((10, 10, 512, 512), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid 4D shape"):
            _detect_ndim(image)

    def test_invalid_5d_shape(self):
        """Test that 5D arrays raise ValueError."""
        image = np.zeros((2, 3, 10, 512, 512), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid image shape"):
            _detect_ndim(image)


@pytest.mark.gui
class TestAnnotatorClass:
    """Test the unified Annotator class."""

    def test_annotator_explicit_ndim_2d(self, make_napari_viewer_proxy):
        """Test Annotator with explicit ndim=2."""
        from micro_sam.sam_annotator._state import AnnotatorState

        viewer = make_napari_viewer_proxy()
        image = np.zeros((512, 512), dtype=np.uint8)

        # Set up state
        state = AnnotatorState()
        state.image_shape = image.shape
        state.reset_state()

        viewer.add_image(image, name="image")
        annotator_instance = Annotator(viewer, ndim=2, reset_state=False)

        assert annotator_instance._ndim == 2
        assert "segment" in annotator_instance._widgets
        assert "clear" in annotator_instance._widgets
        assert "segment_nd" not in annotator_instance._widgets

    def test_annotator_explicit_ndim_3d(self, make_napari_viewer_proxy):
        """Test Annotator with explicit ndim=3."""
        from micro_sam.sam_annotator._state import AnnotatorState

        viewer = make_napari_viewer_proxy()
        image = np.zeros((10, 512, 512), dtype=np.uint8)

        # Set up state
        state = AnnotatorState()
        state.image_shape = image.shape
        state.reset_state()

        viewer.add_image(image, name="image")
        annotator_instance = Annotator(viewer, ndim=3, reset_state=False)

        assert annotator_instance._ndim == 3
        assert "segment" in annotator_instance._widgets
        assert "clear" in annotator_instance._widgets
        assert "segment_nd" in annotator_instance._widgets

    def test_annotator_auto_detect_2d(self, make_napari_viewer_proxy):
        """Test Annotator with auto-detection for 2D."""
        from micro_sam.sam_annotator._state import AnnotatorState

        viewer = make_napari_viewer_proxy()
        image = np.zeros((512, 512), dtype=np.uint8)

        # Set up state
        state = AnnotatorState()
        state.image_shape = image.shape
        state.reset_state()

        viewer.add_image(image, name="image")
        annotator_instance = Annotator(viewer, ndim=None, reset_state=False)

        assert annotator_instance._ndim == 2

    def test_annotator_auto_detect_3d(self, make_napari_viewer_proxy):
        """Test Annotator with auto-detection for 3D."""
        from micro_sam.sam_annotator._state import AnnotatorState

        viewer = make_napari_viewer_proxy()
        image = np.zeros((10, 512, 512), dtype=np.uint8)

        # Set up state
        state = AnnotatorState()
        state.image_shape = image.shape
        state.reset_state()

        viewer.add_image(image, name="image")
        annotator_instance = Annotator(viewer, ndim=None, reset_state=False)

        assert annotator_instance._ndim == 3

    def test_annotator_invalid_ndim(self, make_napari_viewer_proxy):
        """Test Annotator with invalid ndim."""
        from micro_sam.sam_annotator._state import AnnotatorState

        viewer = make_napari_viewer_proxy()
        image = np.zeros((512, 512), dtype=np.uint8)

        # Set up state
        state = AnnotatorState()
        state.image_shape = image.shape
        state.reset_state()

        viewer.add_image(image, name="image")

        with pytest.raises(ValueError, match="Invalid ndim"):
            Annotator(viewer, ndim=4, reset_state=False)

    def test_annotator_no_image_layer(self, make_napari_viewer_proxy):
        """Test Annotator raises error when no image layer exists."""
        from micro_sam.sam_annotator._state import AnnotatorState

        viewer = make_napari_viewer_proxy()

        # Set up state without image
        state = AnnotatorState()
        state.image_shape = (512, 512)
        state.reset_state()

        with pytest.raises(ValueError, match="no image layer found"):
            Annotator(viewer, ndim=None, reset_state=False)


class TestAnnotatorFunction:
    """Test the annotator() function."""

    def test_annotator_ndim_validation(self):
        """Test that annotator() validates ndim matches image shape."""
        image_2d = np.zeros((512, 512), dtype=np.uint8)

        # Should raise error when ndim doesn't match
        with pytest.raises(ValueError, match="does not match detected ndim"):
            annotator(image_2d, ndim=3, return_viewer=True)

    def test_annotator_auto_detect_2d(self):
        """Test annotator() auto-detects 2D images."""
        image = np.zeros((512, 512), dtype=np.uint8)

        # Should auto-detect ndim=2
        viewer = annotator(image, ndim=None, return_viewer=True)
        assert viewer is not None
        viewer.close()

    def test_annotator_auto_detect_3d(self):
        """Test annotator() auto-detects 3D images."""
        image = np.zeros((10, 512, 512), dtype=np.uint8)

        # Should auto-detect ndim=3
        viewer = annotator(image, ndim=None, return_viewer=True)
        assert viewer is not None
        viewer.close()

    def test_annotator_explicit_2d(self):
        """Test annotator() with explicit ndim=2."""
        image = np.zeros((512, 512), dtype=np.uint8)

        viewer = annotator(image, ndim=2, return_viewer=True)
        assert viewer is not None
        viewer.close()

    def test_annotator_explicit_3d(self):
        """Test annotator() with explicit ndim=3."""
        image = np.zeros((10, 512, 512), dtype=np.uint8)

        viewer = annotator(image, ndim=3, return_viewer=True)
        assert viewer is not None
        viewer.close()

    def test_annotator_2d_rgb(self):
        """Test annotator() with 2D RGB image."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)

        viewer = annotator(image, ndim=2, return_viewer=True)
        assert viewer is not None
        viewer.close()

    def test_annotator_3d_rgb(self):
        """Test annotator() with 3D RGB volume."""
        image = np.zeros((10, 512, 512, 3), dtype=np.uint8)

        viewer = annotator(image, ndim=3, return_viewer=True)
        assert viewer is not None
        viewer.close()

    def test_annotator_invalid_ndim_value(self):
        """Test annotator() rejects invalid ndim values."""
        image = np.zeros((512, 512), dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid ndim"):
            annotator(image, ndim=4, return_viewer=True)
