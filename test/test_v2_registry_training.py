import pytest

import torch

from torch_em.util import get_constructor_arguments

from micro_sam.v2 import util as registry_util
from micro_sam.v2.training import util as training_util, training
from micro_sam.v2.models.util import UniSAM2, SemanticSAM2, SAM2EncoderAdapter


class DecoderModel(torch.nn.Module):
    def __init__(self, channels=5, features=2):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Conv3d(3, features, 1)
        self.out_conv = torch.nn.Conv3d(features, channels, 1)


@pytest.mark.parametrize("model_class, channel_argument", [(UniSAM2, "output_channels"), (SemanticSAM2, "num_classes")])
def test_subclass_metadata_preserves_constructor_arguments(model_class, channel_argument):
    encoder = torch.nn.Linear(2, 2)
    model = model_class(
        encoder=encoder, img_size=64, initial_features=4, encoder_checkpoint=None,
        resize_input=False, **{channel_argument: 5},
    )
    arguments = get_constructor_arguments(model)
    assert arguments["encoder"] is encoder
    assert arguments[channel_argument] == 5
    assert "backbone" not in arguments
    assert "encoder_checkpoint" not in arguments

    restored = model_class(**arguments)
    assert isinstance(restored.encoder, SAM2EncoderAdapter)
    assert restored.encoder.inner is encoder
    assert restored.img_size == 64
    assert restored.initial_features == 4
    assert restored.resize_input is False
    restored.load_state_dict(model.state_dict())


@pytest.fixture
def registered_decoder(tmp_path, monkeypatch):
    model = DecoderModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    checkpoint = tmp_path / "decoder.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    monkeypatch.setattr(
        registry_util, "_download_finetuned_sam2_model", lambda model_type: ("encoder.pt", "hash", checkpoint)
    )
    return model


def test_registered_decoder_loading_preserves_shared_encoder(registered_decoder):
    model = DecoderModel()
    encoder = model.encoder
    encoder_state = {key: value.clone() for key, value in encoder.state_dict().items()}
    training._init_sam2_decoder(model, "hvit_t_cells")

    assert model.encoder is encoder
    for key, value in encoder_state.items():
        torch.testing.assert_close(encoder.state_dict()[key], value)
    for key, value in registered_decoder.state_dict().items():
        if not key.startswith("encoder."):
            torch.testing.assert_close(model.state_dict()[key], value)


def test_semantic_initialization_preserves_output_head(registered_decoder):
    model = DecoderModel(channels=3)
    head_state = {key: value.clone() for key, value in model.out_conv.state_dict().items()}
    training._init_sam2_decoder(model, "hvit_t_cells", skip_output_head=True)

    for key, value in registered_decoder.decoder.state_dict().items():
        torch.testing.assert_close(model.decoder.state_dict()[key], value)
    for key, value in head_state.items():
        torch.testing.assert_close(model.out_conv.state_dict()[key], value)


@pytest.mark.parametrize("options", [{"channels": 4}, {"features": 3}])
def test_registered_decoder_rejects_incompatible_architecture(registered_decoder, options):
    with pytest.raises(ValueError, match="initial_features.*with_boundaries"):
        training._init_sam2_decoder(DecoderModel(**options), "hvit_t_cells")


def test_automatic_builder_loads_registered_decoder(monkeypatch, registered_decoder):
    from micro_sam.v2.models import util as models_util

    monkeypatch.setattr(models_util, "UniSAM2", lambda **kwargs: DecoderModel())
    model = training._build_unisam2_model("hvit_t_cells", device="cpu", output_channels=5, initial_features=2)
    for key, value in registered_decoder.state_dict().items():
        if not key.startswith("encoder."):
            torch.testing.assert_close(model.state_dict()[key], value)


@pytest.mark.parametrize("checkpoint_path", [None, "custom.pt"])
def test_interactive_training_resolves_registry_checkpoint(monkeypatch, checkpoint_path):
    from sam2 import build_sam

    captured = {}
    fetched = []

    def fetch(model_type):
        fetched.append(model_type)
        return "registered.pt", "hash", "decoder.pt"

    def build(**kwargs):
        captured.update(kwargs)
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(registry_util, "_download_finetuned_sam2_model", fetch)
    monkeypatch.setattr(training_util, "_download_finetuned_sam2_model", fetch)
    monkeypatch.setattr(training_util, "sam2_train_class", lambda: torch.nn.Linear)
    monkeypatch.setattr(build_sam, "build_sam2", build)
    training_util.get_sam2_train_model(model_type="hvit_t_cells", device="cpu", checkpoint_path=checkpoint_path)

    assert captured["ckpt_path"] == ("registered.pt" if checkpoint_path is None else checkpoint_path)
    assert captured["config_file"] == registry_util.CFG_PATHS["hvit_t"]
    assert fetched == (["hvit_t_cells"] if checkpoint_path is None else [])
