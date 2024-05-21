import json
import os
import platform

from mobile_sam.predictor import SamPredictor as MobileSamPredictor
from segment_anything.predictor import SamPredictor
import numpy as np
import pytest
import torch
import zarr

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator._widgets import EmbeddingWidget
from micro_sam.util import _compute_data_signature


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment.
# tmp_path is a regular pytest fixture.
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_embedding_widget(make_napari_viewer, tmp_path):
    """Test embedding widget for micro-sam napari plugin."""
    # Setup
    viewer = make_napari_viewer()
    layer = viewer.open_sample("napari", "camera")[0]
    my_widget = EmbeddingWidget()

    # Set the widget parameters
    my_widget.image = layer
    my_widget.model_type = "vit_t"
    my_widget.device = "cpu"
    my_widget.embeddings_save_path = tmp_path

    # Run image embedding widget.
    worker = my_widget(skip_validate=True)
    worker.await_workers()  # blocks until thread worker is finished the embedding

    # Check in-memory state for predictor and embeddings.
    assert isinstance(AnnotatorState().predictor, (SamPredictor, MobileSamPredictor))
    assert AnnotatorState().image_embeddings is not None
    assert "features" in AnnotatorState().image_embeddings.keys()
    assert "input_size" in AnnotatorState().image_embeddings.keys()
    assert "original_size" in AnnotatorState().image_embeddings.keys()
    assert isinstance(AnnotatorState().image_embeddings["features"], (torch.Tensor, np.ndarray))
    assert AnnotatorState().image_embeddings["original_size"] == layer.data.shape

    # Check saved embedding results are what we expect to have.
    temp_path_files = os.listdir(tmp_path)
    temp_path_files.sort()
    assert temp_path_files == [".zattrs", ".zgroup", "features"]
    with open(os.path.join(tmp_path, ".zattrs")) as f:
        content = f.read()
    zarr_dict = json.loads(content)
    assert zarr_dict.get("original_size") == list(layer.data.shape)
    assert zarr_dict.get("data_signature") == _compute_data_signature(layer.data)
    assert zarr.open(os.path.join(tmp_path, "features")).shape == (1, 256, 64, 64)

    # Close the viewer at the end of the test.
    viewer.close()
