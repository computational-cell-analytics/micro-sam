import json
import os

import zarr

from micro_sam.sam_annotator._widgets import embedding_widget, Model
from micro_sam.util import _compute_data_signature


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment.
# tmp_path is a regular pytest fixture.
def test_embedding_widget(make_napari_viewer, tmp_path):
    """Test embedding widget for micro-sam napari plugin."""
    # setup
    viewer = make_napari_viewer()
    layer = viewer.open_sample('napari', 'camera')[0]
    my_widget = embedding_widget()
    # run image embedding widget
    worker = my_widget(layer, model=Model.vit_t, device="cpu", save_path=tmp_path)
    worker.await_workers()  # blocks until thread worker is finished the embedding
    # Open embedding results and check they are as expected
    assert os.listdir(tmp_path) == ['.zattrs', '.zgroup', 'features']
    with open(os.path.join(tmp_path, ".zattrs")) as f:
        content = f.read()
    zarr_dict = json.loads(content)
    assert zarr_dict.get("original_size") == list(layer.data.shape)
    assert zarr_dict.get("data_signature") == _compute_data_signature(layer.data)
    assert zarr.open(os.path.join(tmp_path, "features")).shape == (1, 256, 64, 64)
    viewer.close()  # must close the viewer at the end of tests
