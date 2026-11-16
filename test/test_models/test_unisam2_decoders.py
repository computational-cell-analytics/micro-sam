import os

import pytest
import torch

from micro_sam.v2.models import decoders
from micro_sam.v2.models.util import UniSAM2


@pytest.fixture(scope="module")
def encoder():
    from micro_sam.v2.util import get_sam2_model

    return get_sam2_model(model_type="hvit_t", input_type="images", device="cpu").image_encoder


@pytest.mark.parametrize("decoder_type", ["unetr_skip", "hiera_u", "cagd_3d", "c3f2"])
@pytest.mark.parametrize("n_z", [1, 2])
def test_multi_scale_decoders_predict_at_input_shape(encoder, decoder_type, n_z):
    model = UniSAM2(encoder=encoder, output_channels=5, initial_features=8, decoder_type=decoder_type)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, n_z, 128, 96))
    assert tuple(out.shape) == (1, 5, n_z, 128, 96)
    # The activation: sigmoid foreground and boundary, tanh distances.
    assert out[:, 0].min() >= 0 and out[:, 4].max() <= 1 and out[:, 1:4].abs().max() <= 1


def test_multi_scale_adapter_takes_the_sam2_projections():
    model = UniSAM2(encoder="hvit_t", output_channels=5, initial_features=8, decoder_type="hiera_u")
    from micro_sam.v2.util import get_sam2_model

    sam2 = get_sam2_model(model_type="hvit_t", input_type="images", device="cpu")
    assert torch.equal(model.encoder.conv_s0.weight, sam2.sam_mask_decoder.conv_s0.weight)
    assert torch.equal(model.encoder.conv_s1.weight, sam2.sam_mask_decoder.conv_s1.weight)


SMALL_DECODERS = {
    "hiera_u": lambda: decoders.HieraUDecoder(out_channels=5, widths=(32, 24, 16, 16, 8), stem_widths=(8, 16)),
    "cagd_3d": lambda: decoders.CAGD3DDecoder(out_channels=5, widths=(16, 32, 80, 128), stem_widths=(8, 16)),
    "c3f2": lambda: decoders.C3F2Decoder(out_channels=5, widths=(64, 32, 16), head_width=16, stem_widths=(8, 16)),
}


@pytest.mark.parametrize("decoder_type", sorted(SMALL_DECODERS))
def test_decoder_decodes_a_volume_as_its_stacked_slices_at_init(decoder_type):
    torch.manual_seed(0)
    decoder = SMALL_DECODERS[decoder_type]().double()
    decoder.eval()
    n_z, size = 3, 64
    features = {
        "s16": torch.randn(n_z, 256, size // 16, size // 16, dtype=torch.float64),
        "s8": torch.randn(n_z, 64, size // 8, size // 8, dtype=torch.float64),
        "s4": torch.randn(n_z, 32, size // 4, size // 4, dtype=torch.float64),
    }
    image = torch.randn(n_z, 3, size, size, dtype=torch.float64)
    with torch.no_grad():
        volume = decoder(features, image, n_z)
        slices = torch.cat([
            decoder({key: value[i:i + 1] for key, value in features.items()}, image[i:i + 1], 1) for i in range(n_z)
        ])
    assert torch.allclose(volume, slices, atol=1e-12, rtol=0)

    # Once the z branches are trained, the slices interact.
    with torch.no_grad():
        for module in decoder.modules():
            if isinstance(module, decoders.ZResidual):
                module.pointwise.weight.normal_()
            if isinstance(module, decoders.ZAttention):
                module.proj.weight.normal_()
        volume = decoder(features, image, n_z)
    assert not torch.allclose(volume, slices, atol=1e-6, rtol=0)


def test_z_attention_mixes_slices():
    torch.manual_seed(0)
    attention = decoders.ZAttention(16, n_heads=2).double()
    x = torch.randn(2 * 4, 16, 5, 6, dtype=torch.float64)
    assert torch.equal(attention(x, 4), x)  # zero-initialised output projection
    with torch.no_grad():
        attention.proj.weight.normal_()
    y = attention(x, 4)
    # Changing one slice changes the output of the other slices of the same volume, but not of the other volume.
    x2 = x.clone()
    x2[1] += torch.randn_like(x2[1])  # not a constant: the LayerNorm would remove it
    y2 = attention(x2, 4)
    assert not torch.allclose(y[0], y2[0])
    assert torch.allclose(y[4:], y2[4:], atol=1e-12, rtol=0)


def test_cagd_3d_deep_supervision(encoder):
    model = UniSAM2(encoder=encoder, output_channels=5, initial_features=8, decoder_type="cagd_3d",
                    decoder_kwargs={"widths": (16, 32, 80, 128), "stem_widths": (8, 16)})
    x = torch.rand(1, 3, 2, 128, 96)
    model.train()
    out = model(x)
    assert len(model.auxiliary_outputs) == 3
    for aux in model.auxiliary_outputs:
        assert aux.shape == out.shape
        assert aux[:, 0].min() >= 0 and aux[:, 4].max() <= 1 and aux[:, 1:4].abs().max() <= 1
    sum(aux.mean() for aux in model.auxiliary_outputs).backward()
    assert model.seg_decoder.aux_heads[0].weight.grad is not None
    model.eval()
    with torch.no_grad():
        model(x)
    assert model.auxiliary_outputs is None


@pytest.mark.parametrize("decoder_type", ["unetr", "unetr_skip", "hiera_u", "cagd_3d", "c3f2"])
def test_decoder_checkpoint_round_trip(encoder, decoder_type, tmp_path):
    from micro_sam.v2.instance_segmentation import get_unisam2_model

    torch.manual_seed(0)
    model = UniSAM2(encoder=encoder, output_channels=5, initial_features=8, decoder_type=decoder_type)
    path = os.path.join(tmp_path, "checkpoint.pt")
    torch.save({"model_state": model.state_dict(), "decoder_config": model.decoder_config}, path)
    loaded = get_unisam2_model(path, encoder=encoder)
    assert loaded.decoder_type == decoder_type
    model.eval()
    x = torch.rand(1, 3, 1, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_invalid_decoder_arguments(encoder):
    with pytest.raises(ValueError):
        UniSAM2(encoder=encoder, output_channels=5, decoder_type="cagd")
    with pytest.raises(ValueError):
        UniSAM2(encoder=encoder, output_channels=5, decoder_type="unetr_skip", decoder_kwargs={"widths": (8,)})
    with pytest.raises(ValueError):
        decoders.ZAttention(48, n_heads=5)
