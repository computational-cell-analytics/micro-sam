import os
import platform

import pytest
import numpy as np
import imageio.v3 as imageio

import torch

pytestmark = pytest.mark.skipif(platform.system() == "Windows", reason="The model export does not work on Windows.")

MODEL_TYPE = "hvit_t_cells"


@pytest.fixture(scope="module")
def registry_files():
    from micro_sam.v2.util import _download_finetuned_sam2_model
    checkpoint_path, _, decoder_path = _download_finetuned_sam2_model(MODEL_TYPE)
    return checkpoint_path, decoder_path


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    from micro_sam.sample_data import fetch_livecell_example_data
    return imageio.imread(fetch_livecell_example_data(tmp_path_factory.mktemp("sample_data")))


def _predict(package, inputs):
    import bioimageio.core
    from bioimageio.core.digest_spec import create_sample_for_model

    model = bioimageio.core.load_description(package)
    sample = create_sample_for_model(model=model, inputs=inputs)
    with bioimageio.core.create_prediction_pipeline(model, devices=["cpu"]) as pp:
        prediction = pp.predict_sample_without_blocking(sample)
    return np.asarray(prediction.members["masks"].data)


@pytest.mark.slow
def test_weights_hold_the_decoder_once(registry_files, tmp_path):
    from micro_sam.util import load_checkpoint
    from micro_sam.v2.bioimageio.model_export import _get_weights

    weight_path, backbone, with_decoder = _get_weights(MODEL_TYPE, None, None, tmp_path)
    assert backbone == "hvit_t" and with_decoder

    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    model_state = load_checkpoint(registry_files[0])["model"]
    assert weights["model_state"].keys() == model_state.keys()
    assert all(torch.equal(weights["model_state"][k], v) for k, v in model_state.items())
    assert not any(k.startswith("encoder.inner.") for k in weights["decoder_state"])
    assert "out_conv.weight" in weights["decoder_state"]


@pytest.mark.slow
def test_weights_reject_a_decoder_with_another_encoder(registry_files, tmp_path):
    from micro_sam.util import load_checkpoint
    from micro_sam.v2.bioimageio.model_export import _get_weights

    decoder_state = load_checkpoint(registry_files[1])
    key = next(k for k in decoder_state if k.startswith("encoder.inner."))
    decoder_state[key] = decoder_state[key] + 1
    decoder_path = os.path.join(tmp_path, "decoder.pt")
    torch.save(decoder_state, decoder_path)
    with pytest.raises(ValueError, match="image encoder"):
        _get_weights(MODEL_TYPE, registry_files[0], decoder_path, tmp_path)


@pytest.mark.slow
def test_export_does_interactive_and_automatic_segmentation(image, tmp_path):
    from micro_sam.v2.bioimageio import export_sam2_model

    package = os.path.join(tmp_path, "model.zip")
    export_sam2_model(image, None, MODEL_TYPE, "micro-sam2 test", package)
    assert os.path.exists(package)

    image_tensor = image.astype("float32")[None, None]
    masks = _predict(package, {"image": image_tensor})
    assert masks.ndim == 5 and masks.shape[1] > 10 and masks.shape[-2:] == image.shape

    boxes = np.array([[[100, 100, 160, 150], [300, 200, 360, 260]]], dtype="int64")
    masks = _predict(package, {"image": image_tensor, "box_prompts": boxes})
    assert masks.shape == (1, 2, 1) + image.shape
    assert all(mask.sum() > 0 for mask in masks[0, :, 0])


@pytest.mark.slow
def test_export_without_decoder_needs_prompts(registry_files, image, tmp_path):
    from micro_sam.v2.bioimageio import export_sam2_model

    labels = np.zeros(image.shape, dtype="uint32")
    labels[100:150, 100:160], labels[200:260, 300:360] = 1, 2
    package = os.path.join(tmp_path, "model.zip")
    with pytest.raises(ValueError, match="label_image"):
        export_sam2_model(image, None, "my_model", "micro-sam2 test", package, checkpoint_path=registry_files[0])
    export_sam2_model(image, labels, "my_model", "micro-sam2 test", package, checkpoint_path=registry_files[0])

    with pytest.raises(Exception, match="no decoder"):
        _predict(package, {"image": image.astype("float32")[None, None]})
