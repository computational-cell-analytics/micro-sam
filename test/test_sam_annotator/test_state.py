import unittest
import micro_sam.util as util

from skimage.data import binary_blobs
from magicgui.widgets import Container


class TestState(unittest.TestCase):
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    def test_state_for_interactive_segmentation(self):
        from micro_sam.sam_annotator._state import AnnotatorState
        image = binary_blobs(512)
        predictor = util.get_sam_model(model_type=self.model_type)
        image_embeddings = util.precompute_image_embeddings(predictor, image)

        state = AnnotatorState()
        state.image_embeddings = image_embeddings
        state.predictor = predictor
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
