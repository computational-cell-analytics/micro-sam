"""The segmentation and tracking annotators support only SAM2. The classification tools also support SAM1."""

import numpy as np
import pytest

from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator.annotator import annotator
from micro_sam.sam_annotator.batch_annotator import SegmentationBatchTask
from micro_sam.sam_annotator.annotator_tracking import annotator_tracking


def _make_widget(widget_class, model_type, is_timeseries=False):
    # Skip '__init__' so that the test builds no Qt widgets. The model check reads only these attributes.
    widget = widget_class.__new__(widget_class)
    widget.model_type = model_type
    widget.is_timeseries = is_timeseries
    return widget


def test_segmentation_annotator_rejects_sam1():
    with pytest.raises(ValueError, match="segmentation annotator does not support the model 'vit_b_lm'"):
        annotator(np.zeros((32, 32), dtype="uint8"), model_type="vit_b_lm")


def test_tracking_annotator_rejects_sam1():
    with pytest.raises(ValueError, match="tracking annotator does not support the model 'vit_b_lm'"):
        annotator_tracking(np.zeros((2, 32, 32), dtype="uint8"), model_type="vit_b_lm")


def test_batch_segmentation_rejects_sam1():
    with pytest.raises(ValueError, match="segmentation annotator does not support the model 'vit_b_lm'"):
        SegmentationBatchTask(
            ndim=2, model_type="vit_b_lm", embedding_path=None, tile_shape=None, halo=None,
            precompute_autoseg_state=False, checkpoint_path=None, device=None, prefer_decoder=True,
        )


@pytest.mark.parametrize("is_timeseries, tool", [(False, "segmentation"), (True, "tracking")])
def test_embedding_widget_rejects_sam1(monkeypatch, is_timeseries, tool):
    messages = []
    monkeypatch.setattr(widgets, "_generate_message", lambda kind, msg: messages.append((kind, msg)) or True)

    # Loading saved SAM1 embeddings also sets this model type, so this covers that case too.
    widget = _make_widget(widgets.EmbeddingWidget, "vit_b_lm", is_timeseries=is_timeseries)
    assert widget._validate_model_support()
    expected = f"The {tool} annotator does not support the model 'vit_b_lm'. Use a (µ)SAM2 model ('hvit_*')."
    assert messages == [("error", expected)]

    widget = _make_widget(widgets.EmbeddingWidget, "hvit_t_cells", is_timeseries=is_timeseries)
    assert not widget._validate_model_support()
    assert len(messages) == 1


@pytest.mark.parametrize("model_type", ["vit_b_lm", "vit_b_dinov2", "hvit_t_cells"])
def test_classification_widget_accepts_other_models(monkeypatch, model_type):
    monkeypatch.setattr(widgets, "_generate_message", lambda kind, msg: pytest.fail(msg))
    widget = _make_widget(widgets.ClassificationEmbeddingWidget, model_type)
    assert not widget._validate_model_support()
