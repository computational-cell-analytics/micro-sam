import json
import os
import platform

from mobile_sam.predictor import SamPredictor as MobileSamPredictor
from segment_anything.predictor import SamPredictor
import pytest
import torch
import zarr

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator._widgets import embedding
from micro_sam.util import _compute_data_signature


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment.
# tmp_path is a regular pytest fixture.
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_embedding_widget(make_napari_viewer, tmp_path):
    """Test embedding widget for micro-sam napari plugin."""
    # setup
    viewer = make_napari_viewer()
    layer = viewer.open_sample('napari', 'camera')[0]
    my_widget = embedding()
    # run image embedding widget
    worker = my_widget(image=layer, model="vit_t", device="cpu", save_path=tmp_path)
    worker.await_workers()  # blocks until thread worker is finished the embedding
    # Check in-memory state - predictor
    assert isinstance(AnnotatorState().predictor, (SamPredictor, MobileSamPredictor))
    # Check in-memory state - image embeddings
    assert AnnotatorState().image_embeddings is not None
    assert 'features' in AnnotatorState().image_embeddings.keys()
    assert 'input_size' in AnnotatorState().image_embeddings.keys()
    assert 'original_size' in AnnotatorState().image_embeddings.keys()
    assert isinstance(AnnotatorState().image_embeddings["features"], torch.Tensor)
    assert AnnotatorState().image_embeddings["original_size"] == layer.data.shape
    # Check saved embedding results are what we expect to have
    temp_path_files = os.listdir(tmp_path)
    temp_path_files.sort()
    assert temp_path_files == ['.zattrs', '.zgroup', 'features']
    with open(os.path.join(tmp_path, ".zattrs")) as f:
        content = f.read()
    zarr_dict = json.loads(content)
    assert zarr_dict.get("original_size") == list(layer.data.shape)
    assert zarr_dict.get("data_signature") == _compute_data_signature(layer.data)
    assert zarr.open(os.path.join(tmp_path, "features")).shape == (1, 256, 64, 64)
    viewer.close()  # must close the viewer at the end of tests
