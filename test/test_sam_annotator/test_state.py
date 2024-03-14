import unittest

from skimage.data import binary_blobs
from magicgui.widgets import Container
from micro_sam.util import VIT_T_SUPPORT


class TestState(unittest.TestCase):
    model_type = "vit_t" if VIT_T_SUPPORT else "vit_b"

    def test_state_for_interactive_segmentation(self):
        from micro_sam.sam_annotator._state import AnnotatorState
        image = binary_blobs(512)

        state = AnnotatorState()
        state.initialize_predictor(image, self.model_type, ndim=2)
        state.image_shape = image.shape
        self.assertTrue(state.initialized_for_interactive_segmentation())

    def test_state_for_tracking(self):
        from micro_sam.sam_annotator._state import AnnotatorState

        state = AnnotatorState()
        state.current_track_id = 1
        state.lineage = {1: {}}
        state.committed_lineages = []
        state.tracking_widget = Container()
        self.assertTrue(state.initialized_for_tracking())


if __name__ == "__main__":
    unittest.main()
