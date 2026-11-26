import platform

import pytest
import numpy as np
from skimage.data import binary_blobs

import micro_sam.util as util
from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam._test_util import check_layer_initialization
from micro_sam.sam_annotator.annotator_timeseries import annotator_timeseries


@pytest.mark.gui
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_annotator_timeseries(make_napari_viewer_proxy, monkeypatch):
    """Integration test for annotator_timeseries.
    """

    image = np.stack(3 * [binary_blobs(512)])
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    viewer = make_napari_viewer_proxy()
    viewer = annotator_timeseries(image, model_type=model_type, viewer=viewer, return_viewer=True)
    check_layer_initialization(viewer, image.shape)

    # Each frame is segmented on its own and gets new ids, without merging across frames.
    frame = np.zeros(image.shape[1:], dtype="uint32")
    frame[:10, :10] = 1
    monkeypatch.setattr(widgets, "_instance_segmentation_impl", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(widgets, "merge_instance_segmentation_3d", None)
    AnnotatorState().widgets["autosegment"]()

    segmentation = viewer.layers["auto_segmentation"].data
    assert [int(segmentation[i].max()) for i in range(3)] == [1, 2, 3]
    viewer.close()  # must close the viewer at the end of tests
