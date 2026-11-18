"""Tests for the start weights of a micro-sam2 training: all checkpoint formats, on one GPU and with DDP."""
import os
import shutil

import pytest

import torch

MODEL_TYPE = "hvit_t_cells"


@pytest.fixture(scope="module")
def registry_states():
    from micro_sam.util import load_checkpoint
    from micro_sam.v2.util import _download_finetuned_sam2_model

    model_path, _, decoder_path = _download_finetuned_sam2_model(MODEL_TYPE)
    model_state = load_checkpoint(model_path)["model"]
    decoder_state = load_checkpoint(decoder_path)
    decoder_state = decoder_state.get("model_state", decoder_state)
    return model_path, decoder_path, model_state, decoder_state


@pytest.fixture(scope="module")
def checkpoints(registry_states, tmp_path_factory):
    """Write the registry weights in each checkpoint format that a training can start from."""
    model_path, decoder_path, model_state, decoder_state = registry_states
    folder = tmp_path_factory.mktemp("checkpoints")
    decoder_only = {k: v for k, v in decoder_state.items() if not k.startswith("encoder.")}

    paths = {"model_file": model_path}
    contents = {
        # The 'best.pt' of interactive training. DDP and torch.compile add a prefix to the keys.
        "interactive": {"model_state": model_state},
        "ddp": {"model_state": {"module." + k: v for k, v in model_state.items()}},
        "compiled": {"model_state": {"_orig_mod." + k: v for k, v in model_state.items()}},
        # The 'best.pt' of joint training and of automatic training.
        "joint": {"model_state": model_state, "decoder_state": decoder_only},
        "automatic": {"model_state": decoder_state},
    }
    for name, content in contents.items():
        paths[name] = os.path.join(folder, f"{name}.pt")
        torch.save(content, paths[name])

    # A model file with its decoder file next to it, as the registry and the export provide them.
    paths["pair"] = os.path.join(folder, "my_model.pt")
    shutil.copy(model_path, paths["pair"])
    shutil.copy(decoder_path, os.path.join(folder, "my_model_decoder.pt"))
    return paths


def _decoder_settings(decoder_state):
    output_channels, initial_features = decoder_state["out_conv.weight"].shape[:2]
    return int(output_channels), int(initial_features)


def _assert_same(state, expected, prefix_map=("", "")):
    old, new = prefix_map
    for key, value in expected.items():
        if key.startswith(old):
            assert torch.equal(state[new + key[len(old):]].cpu(), value.cpu()), key


@pytest.mark.slow
@pytest.mark.parametrize("name", ["model_file", "interactive", "ddp", "compiled", "joint", "pair"])
def test_interactive_training_starts_from_checkpoint(checkpoints, registry_states, name):
    pytest.importorskip("training.model.sam2")
    from micro_sam.v2.training.util import get_sam2_train_model

    # The backbone comes from the checkpoint, not from the model type.
    model = get_sam2_train_model(model_type="hvit_b", device="cpu", checkpoint_path=checkpoints[name])
    _assert_same(model.state_dict(), registry_states[2])


@pytest.mark.slow
def test_interactive_training_starts_from_peft_checkpoint(tmp_path):
    pytest.importorskip("training.model.sam2")
    from micro_sam.v2.models.peft_sam2 import LoRASurgery
    from micro_sam.v2.training.util import get_sam2_train_model

    peft_kwargs = {"rank": 2, "peft_module": LoRASurgery}
    model = get_sam2_train_model(model_type="hvit_t", device="cpu", peft_kwargs=peft_kwargs)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "w_b_linear" in name:
                parameter.fill_(0.5)
    checkpoint_path = os.path.join(tmp_path, "best.pt")
    torch.save({"model_state": model.state_dict(), "peft_kwargs": model.peft_config}, checkpoint_path)

    # Without 'peft_kwargs' the training continues with the PEFT layers of the checkpoint.
    restarted = get_sam2_train_model(model_type="hvit_t", device="cpu", checkpoint_path=checkpoint_path)
    assert restarted.peft_config == model.peft_config
    _assert_same(restarted.state_dict(), model.state_dict())

    with pytest.raises(ValueError, match="PEFT layers"):
        get_sam2_train_model(
            model_type="hvit_t", device="cpu", checkpoint_path=checkpoint_path,
            peft_kwargs={"rank": 4, "peft_module": LoRASurgery},
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, has_decoder",
    [("joint", True), ("automatic", True), ("pair", True), ("ddp", False), ("interactive", False)],
)
def test_automatic_training_starts_from_checkpoint(checkpoints, registry_states, name, has_decoder):
    from micro_sam.v2.training.training import _build_unisam2_model

    _, _, model_state, decoder_state = registry_states
    output_channels, initial_features = _decoder_settings(decoder_state)
    model = _build_unisam2_model(
        "hvit_t", "cpu", output_channels=output_channels, initial_features=initial_features,
        checkpoint_path=checkpoints[name],
    )
    state = model.state_dict()
    _assert_same(state, model_state, prefix_map=("image_encoder.", "encoder.inner."))

    decoder_only = {k: v for k, v in decoder_state.items() if not k.startswith("encoder.")}
    if has_decoder:
        _assert_same(state, decoder_only)
    else:  # Without a decoder in the checkpoint, the decoder starts from random weights.
        assert not torch.equal(state["out_conv.weight"], decoder_only["out_conv.weight"])


@pytest.mark.slow
def test_semantic_training_starts_from_checkpoint(checkpoints, registry_states):
    from micro_sam.v2.training.training import _build_semantic_sam2_model

    _, _, model_state, decoder_state = registry_states
    _, initial_features = _decoder_settings(decoder_state)
    model = _build_semantic_sam2_model(
        "hvit_t", "cpu", num_classes=3, initial_features=initial_features, checkpoint_path=checkpoints["joint"],
    )
    state = model.state_dict()
    _assert_same(state, model_state, prefix_map=("image_encoder.", "encoder.inner."))
    # The decoder comes from the checkpoint. The output head for the classes is new.
    _assert_same(state, {k: v for k, v in decoder_state.items() if not k.startswith(("encoder.", "out_conv."))})
    assert state["out_conv.weight"].shape[0] == 3


def test_strip_wrapper_prefixes():
    from micro_sam.v2.util import strip_wrapper_prefixes

    state = {"module._orig_mod.a": 1, "_orig_mod.b": 2, "c": 3}
    assert strip_wrapper_prefixes(state) == {"a": 1, "b": 2, "c": 3}
