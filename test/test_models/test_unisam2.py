import pytest
import torch

from micro_sam.v2.models.util import UniSAM2


@pytest.fixture(scope="module")
def encoder():
    from micro_sam.v2.util import get_sam2_model

    return get_sam2_model(model_type="hvit_t", input_type="images", device="cpu").image_encoder


def test_unisam2_builds_the_decoder_at_the_requested_width(encoder):
    default = UniSAM2(encoder=encoder, output_channels=4, device="cpu")
    narrow = UniSAM2(encoder=encoder, output_channels=4, device="cpu", initial_features=32)

    assert default.out_conv.in_channels == 64
    assert narrow.out_conv.in_channels == 32
    # The same modules at half the width: identical keys, halved decoder channels.
    default_state, narrow_state = default.state_dict(), narrow.state_dict()
    assert set(default_state) == set(narrow_state)
    assert tuple(narrow_state["base.block.0.block.1.weight"].shape) == (256, 256, 3, 3, 3)
    assert tuple(default_state["base.block.0.block.1.weight"].shape) == (512, 256, 3, 3, 3)
    assert tuple(narrow_state["decoder_head.block.0.block.1.weight"].shape) == (32, 64, 3, 3, 3)
    with torch.no_grad():
        out = narrow(torch.rand(1, 3, 1, 256, 256))
    assert tuple(out.shape) == (1, 4, 1, 256, 256)


def test_unisam2_loads_a_narrow_state_dict_strictly(encoder):
    narrow = UniSAM2(encoder=encoder, output_channels=4, device="cpu", initial_features=32)
    state = {key: torch.zeros_like(value) for key, value in narrow.state_dict().items()}
    rebuilt = UniSAM2(encoder=encoder, output_channels=4, device="cpu", initial_features=32)
    rebuilt.load_state_dict(state)
    assert all(torch.equal(value, torch.zeros_like(value)) for key, value in rebuilt.state_dict().items()
               if key.startswith(("out_conv", "base", "decoder")))


def test_unisam2_fifth_channel_is_a_probability(encoder):
    model = UniSAM2(encoder=encoder, output_channels=5, device="cpu", initial_features=32)
    assert model.out_conv.out_channels == 5 and model.out_channels == 5
    with torch.no_grad():
        out = model(torch.rand(1, 3, 1, 256, 256))
    assert tuple(out.shape) == (1, 5, 1, 256, 256)
    assert out[:, 0].min() >= 0 and out[:, 4].min() >= 0 and out[:, 4].max() <= 1
    assert out[:, 1:4].min() >= -1 and out[:, 1:4].max() <= 1


def test_get_unisam2_model_reads_the_channel_count_off_the_checkpoint(encoder, tmp_path):
    from micro_sam.v2.instance_segmentation import get_unisam2_model

    model = UniSAM2(encoder=encoder, output_channels=5, device="cpu", initial_features=32)
    torch.save(model.state_dict(), tmp_path / "five.pt")
    four = UniSAM2(encoder=encoder, output_channels=4, device="cpu", initial_features=32)
    torch.save({"unetr_state": four.state_dict()}, tmp_path / "joint.pt")

    loaded = get_unisam2_model(tmp_path / "five.pt", device="cpu", encoder=encoder)
    assert loaded.out_channels == 5 and loaded.out_conv.in_channels == 32
    assert get_unisam2_model(tmp_path / "joint.pt", device="cpu", encoder=encoder).out_channels == 4
