import types
import unittest

import pytest
import numpy as np
from skimage.data import binary_blobs

from magicgui.widgets import Container

import torch

from micro_sam.v2.util import DEFAULT_MODEL


class TestState(unittest.TestCase):
    model_type = DEFAULT_MODEL

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
        state.widgets = {"tracking": Container()}
        self.assertTrue(state.initialized_for_tracking())


def test_autoseg_names_have_no_legacy_aliases():
    """Only the canonical automatic segmenter and cached-state names are exposed."""
    from micro_sam.sam_annotator._state import AnnotatorState

    state = AnnotatorState()
    assert hasattr(state, "automatic_segmenter")
    assert hasattr(state, "autoseg_state")  # canonical name
    assert not hasattr(state, "amg")
    assert not hasattr(state, "amg_state")  # old aliases gone
    assert not hasattr(state, "auto_state")


def test_blank_model_paths_are_normalized(monkeypatch):
    """Blank API paths must fall back to the registered model instead of being loaded as files."""
    import micro_sam.sam_annotator._state as state_module
    import micro_sam.v2.util as v2_util

    captured = {}

    def fake_get_sam_model(model_type, ndim, device, checkpoint_path, decoder_path, use_cli):
        captured["checkpoint_path"] = checkpoint_path
        captured["decoder_path"] = decoder_path
        return object(), {}

    def fake_precompute_image_embeddings(**kwargs):
        captured["batch_size"] = kwargs["batch_size"]
        return {
            "features": np.zeros((1, 1, 1, 1), dtype="float32"),
            "input_size": (8, 8),
            "original_size": (8, 8),
        }

    monkeypatch.setattr(state_module, "_get_sam_model", fake_get_sam_model)
    monkeypatch.setattr(v2_util, "precompute_image_embeddings", fake_precompute_image_embeddings)

    state = state_module.AnnotatorState()
    state.initialize_predictor(
        np.zeros((8, 8), dtype="uint8"),
        model_type="hvit_t",
        ndim=2,
        checkpoint_path=" ",
        decoder_path="\t",
        prefer_decoder=False,
        batch_size=4,
    )

    assert captured == {"checkpoint_path": None, "decoder_path": None, "batch_size": 4}


def run_initialize_predictor_with_embedding_fn(monkeypatch, embedding_fn, device):
    """Drive 'initialize_predictor' far enough to record how the embedding function was called."""
    import micro_sam.util as util
    import micro_sam.sam_annotator._state as state_module

    monkeypatch.setattr(util, "get_embedding_function", lambda model_type: embedding_fn)
    state = state_module.AnnotatorState()
    state.initialize_predictor(
        np.zeros((8, 8), dtype="uint8"), model_type="hvit_t", ndim=2,
        device=device, predictor=object(), prefer_decoder=False,
    )


@pytest.mark.parametrize(
    "device,expected",
    [(None, None), ("auto", None), ("cuda", "cuda"), ("cuda:1", "cuda:1"), ("cpu", "cpu")],
)
def test_annotator_pins_inference_to_the_selected_device(monkeypatch, device, expected):
    """An explicit device must constrain inference; only None / 'auto' may fan out over all GPUs."""
    captured = {}

    def fake_embedding_fn(devices=None, **kwargs):
        captured["devices"] = devices
        return {
            "features": np.zeros((1, 1, 1, 1), dtype="float32"),
            "input_size": (8, 8),
            "original_size": (8, 8),
        }

    run_initialize_predictor_with_embedding_fn(monkeypatch, fake_embedding_fn, device)
    assert captured["devices"] == expected


def test_annotator_expands_sam2_volume_tiles_for_embedding_api(monkeypatch):
    import micro_sam.util as util
    import micro_sam.sam_annotator._state as state_module
    import micro_sam.v2.prompt_based_segmentation as prompt_based_segmentation

    captured = {}

    def fake_embedding_fn(**kwargs):
        captured.update(kwargs)
        return {"features": object(), "input_size": None}

    monkeypatch.setattr(util, "get_embedding_function", lambda model_type: fake_embedding_fn)
    monkeypatch.setattr(util, "make_temp_embedding_path", lambda: "temporary.zarr")
    monkeypatch.setattr(util, "_open_embeddings", lambda *args, **kwargs: types.SimpleNamespace(
        attrs={"data_signature": "signature"},
    ))
    monkeypatch.setattr(prompt_based_segmentation, "TiledPromptableSegmentation3D", lambda **kwargs: object())

    state = state_module.AnnotatorState()
    state.initialize_predictor(
        np.zeros((2, 8, 8), dtype="uint8"), model_type="hvit_t", ndim=3,
        predictor=object(), prefer_decoder=False, tile_shape=(4, 4), halo=(1, 1),
    )

    assert captured["tile_shape"] == (2, 4, 4)
    assert captured["halo"] == (0, 1, 1)


@pytest.mark.parametrize(
    "device,expected", [("auto", ["cuda:0", "cuda:1"]), ("cuda", ["cuda:1"]), ("cuda:0", ["cuda:0"])]
)
def test_selected_device_reaches_the_batched_backend(monkeypatch, device, expected):
    """The device the annotator forwards has to resolve to the GPUs the user selected, and no others."""
    from micro_sam.v2 import batched_inference

    captured = {}

    def fake_embedding_fn(devices=None, **kwargs):
        captured["devices"] = devices
        return {
            "features": np.zeros((1, 1, 1, 1), dtype="float32"),
            "input_size": (8, 8),
            "original_size": (8, 8),
        }

    run_initialize_predictor_with_embedding_fn(monkeypatch, fake_embedding_fn, device)

    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(batched_inference, "_model_device", lambda model: torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    resolved = batched_inference._resolve_devices(model, captured["devices"])

    assert [str(resolved_device) for resolved_device in resolved] == expected


def test_embedding_function_without_devices_is_called_unchanged(monkeypatch):
    """The SAM1 embedding function has no 'devices' parameter, so it must not receive one."""
    captured = {}

    def fake_embedding_fn(**kwargs):
        captured.update(kwargs)
        return {
            "features": np.zeros((1, 1, 1, 1), dtype="float32"),
            "input_size": (8, 8),
            "original_size": (8, 8),
        }

    run_initialize_predictor_with_embedding_fn(monkeypatch, fake_embedding_fn, "cuda:1")
    assert "devices" not in captured


if __name__ == "__main__":
    unittest.main()
