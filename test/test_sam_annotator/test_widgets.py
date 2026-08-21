import json
import os
import platform
import warnings

try:
    # Avoid import warnigns from mobile_sam
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mobile_sam.predictor import SamPredictor as MobileSamPredictor
except ImportError:
    from segment_anything.predictor import SamPredictor as MobileSamPredictor
from segment_anything.predictor import SamPredictor
import numpy as np
import pytest
import torch
import zarr

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator._widgets import _WidgetBase, EmbeddingWidget
from micro_sam.util import _compute_data_signature


@pytest.mark.skipif(platform.system() == "Windows", reason="GUI test does not work on Windows.")
def test_string_and_path_param_empty_value_semantics(qtbot):
    widget = _WidgetBase()
    qtbot.addWidget(widget)

    widget.name = "sam_model"
    name_param, layout = widget._add_string_param("name", widget.name)
    widget.layout().addLayout(layout)
    name_param.setText("")
    assert widget.name == ""

    widget.checkpoint_path = "/path/to/checkpoint.pt"
    path_param, layout = widget._add_path_param("checkpoint_path", widget.checkpoint_path, "file")
    widget.layout().addLayout(layout)
    path_param.setText("")
    assert widget.checkpoint_path is None


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment.
# tmp_path is a regular pytest fixture.
@pytest.mark.skipif(platform.system() in ("Windows", "Linux", "Darwin"), reason="Gui test is not working on windows.")
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
    my_widget(skip_validate=True)

    # Previous version when we used a thread-worker
    # worker = my_widget(skip_validate=True)
    # worker.await_workers()  # blocks until thread worker is finished the embedding

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


@pytest.mark.gui
def test_batch_size_visibility_follows_device_and_model(qtbot):
    """The batch size control is only shown where it has an effect (GPU, and not a VFM encoder)."""
    from micro_sam.sam_annotator._widgets import ClassificationEmbeddingWidget

    widget = ClassificationEmbeddingWidget()
    qtbot.addWidget(widget)

    widget.device = "cpu"
    widget.model_type = "hvit_t_cells"
    widget._refresh_batch_size()
    assert widget._batch_size_widget.isHidden()

    widget.device = "cuda"
    widget._refresh_batch_size()
    assert not widget._batch_size_widget.isHidden()

    # The VFM encoders offered by the classifiers compute their embeddings unbatched.
    widget.model_type = "vit_b_dinov2"
    widget._refresh_batch_size()
    assert widget._batch_size_widget.isHidden()

    widget.model_type = "vit_b_lm"
    widget._refresh_batch_size()
    assert not widget._batch_size_widget.isHidden()


@pytest.mark.gui
def test_hidden_batch_size_is_not_applied(qtbot, monkeypatch):
    """A GPU batch size must not stay in effect after switching to a device that cannot use it."""
    from micro_sam.sam_annotator._widgets import ClassificationEmbeddingWidget

    widget = ClassificationEmbeddingWidget()
    qtbot.addWidget(widget)

    widget.model_type = "hvit_t_cells"
    widget.device = "cuda"
    widget.batch_size_param.setValue(32)
    assert widget.batch_size == 32
    assert widget._effective_batch_size() == 32

    widget.device = "cpu"
    widget._refresh_batch_size()
    assert widget._batch_size_widget.isHidden()
    assert widget._effective_batch_size() == 1
    # The remembered GPU preference survives, so switching back restores it.
    assert widget.batch_size == 32
    widget.device = "cuda"
    assert widget._effective_batch_size() == 32

    # 'auto' follows whatever it resolves to.
    widget.device = "auto"
    monkeypatch.setattr("micro_sam.util._get_default_device", lambda: "cpu")
    assert widget._effective_batch_size() == 1
    monkeypatch.setattr("micro_sam.util._get_default_device", lambda: "cuda")
    assert widget._effective_batch_size() == 32
