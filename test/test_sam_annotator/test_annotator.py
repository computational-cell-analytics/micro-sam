import platform

import numpy as np
import pytest
from skimage.data import binary_blobs

import micro_sam.util as util
from micro_sam.sam_annotator.annotator import annotator, detect_ndim
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


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
class TestAnnotatorClass:
    """Test the unified Annotator class."""

    def test_annotator_2d(self, make_napari_viewer_proxy):
        image = binary_blobs(512)
        model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

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

    def test_annotator_3d(self, make_napari_viewer_proxy):
        image = np.stack(4 * [binary_blobs(512)])
        model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

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
