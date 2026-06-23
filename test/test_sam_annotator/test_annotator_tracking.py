import platform

import numpy as np
import pytest
from skimage.data import binary_blobs

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator import annotator_tracking
from micro_sam.sam_annotator.annotator_tracking import _validate_tracking_model_type
from micro_sam._test_util import check_layer_initialization


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_annotator_tracking(make_napari_viewer_proxy):
    """Integration test for annotator_tracking.
    """

    image = np.stack(4 * [binary_blobs(512)])

    viewer = make_napari_viewer_proxy()
    # test generating image embedding, then adding micro-sam dock widgets to the GUI
    viewer = annotator_tracking(
        image,
        model_type=DEFAULT_MODEL,
        viewer=viewer,
        return_viewer=True
    )

    check_layer_initialization(viewer, image.shape)
    viewer.close()  # must close the viewer at the end of tests


def test_tracking_rejects_sam1_models():
    with pytest.raises(ValueError, match="only supports micro-sam2/SAM2"):
        _validate_tracking_model_type("vit_b_lm")
