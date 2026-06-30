import os
import platform
import tempfile

import numpy as np
import imageio.v3 as imageio
import pytest
from skimage.data import binary_blobs

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.annotator_tracking import AnnotatorTracking, image_series_tracking_annotator


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_image_series_tracking_navigation(make_napari_viewer_proxy):
    """Drive the tracking series harness: each item is a video, advancing saves the tracks and loads
    the next video.
    """
    videos = [np.stack(3 * [binary_blobs(256)]).astype("float32") for _ in range(2)]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = os.path.join(tmpdir, "tracking_results")
        viewer = make_napari_viewer_proxy()
        viewer = image_series_tracking_annotator(
            videos, output_folder, model_type=DEFAULT_MODEL, viewer=viewer, return_viewer=True,
        )

        state = AnnotatorState()
        assert isinstance(state.annotator, AnnotatorTracking)
        assert "committed_objects" in viewer.layers
        # Only the Next control is registered.
        assert "series_next" in state.widgets
        assert "series_prev" not in state.widgets

        # Commit a fake tracking result for the first video and advance.
        tracks = np.ones((3, 256, 256), dtype="uint32")
        viewer.layers["committed_objects"].data = tracks
        state.widgets["series_next"]()

        assert os.path.exists(os.path.join(output_folder, "tracks_00000.tif"))
        np.testing.assert_array_equal(imageio.imread(os.path.join(output_folder, "tracks_00000.tif")), tracks)

        # We advanced to the second video with a cleared committed layer.
        assert viewer.layers["committed_objects"].data.sum() == 0
        np.testing.assert_array_equal(viewer.layers["image"].data, videos[1])

        viewer.close()
