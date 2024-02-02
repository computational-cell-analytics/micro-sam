import platform

import numpy as np
import pytest
from skimage.data import binary_blobs

import micro_sam.util as util
from micro_sam._test_util import check_layer_initialization
from micro_sam.sam_annotator import annotator_3d


@pytest.mark.gui
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_annotator_3d(make_napari_viewer_proxy):
    """Integration test for annotator_3d.
    """

    image = np.stack(4 * [binary_blobs(512)])
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    viewer = make_napari_viewer_proxy()
    # test generating image embedding, then adding micro-sam dock widgets to the GUI
    viewer = annotator_3d(
        image,
        model_type=model_type,
        viewer=viewer,
        return_viewer=True
    )

    check_layer_initialization(viewer, image.shape)
    viewer.close()  # must close the viewer at the end of tests
