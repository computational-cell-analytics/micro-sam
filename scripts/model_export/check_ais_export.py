import tempfile
from pathlib import Path

import numpy as np
import bioimageio.core
import imageio.v3 as imageio

import torch

from micro_sam import util
from micro_sam.bioimageio import export_sam_model
from micro_sam.sample_data import fetch_tracking_example_data
from micro_sam.bioimageio.predictor_adaptor import PredictorAdaptor


AIS_MODEL_TYPE = "vit_t_lm"
BASE_MODEL_TYPE = "vit_t"


def load_ais_model():
    registry = util.models()
    model_path = registry.fetch(AIS_MODEL_TYPE)
    decoder_path = registry.fetch(f"{AIS_MODEL_TYPE}_decoder")
    model = PredictorAdaptor(BASE_MODEL_TYPE)
    model.load_state_dict({
        "model_state": torch.load(model_path, map_location="cpu", weights_only=True),
        "decoder_state": torch.load(decoder_path, map_location="cpu", weights_only=True),
    })
    model.eval()
    return model


def run_ais(model, image):
    model.sam.reset_image()
    input_tensor = torch.from_numpy(util._to_image(image).transpose(2, 0, 1)[None])
    masks, scores, embeddings = model(image=input_tensor)
    assert masks.ndim == 5
    assert masks.shape[-2:] == image.shape[-2:]
    assert scores.shape == (1, masks.shape[1], 1)
    assert embeddings.shape == (1, 256, 64, 64)
    return masks


def make_export_labels(masks, shape):
    labels = np.zeros(shape, dtype="uint32")
    next_id = 1
    for mask in masks[0, :, 0].numpy().astype(bool):
        available = np.logical_and(mask, labels == 0)
        if not np.any(available):
            continue
        labels[available] = next_id
        next_id += 1
        if next_id == 3:
            return labels

    raise RuntimeError("AIS returned fewer than two non-overlapping masks.")


def make_empty_ais_case():
    image = np.zeros((256, 256), dtype="uint8")
    labels = np.zeros(image.shape, dtype="uint32")
    for seg_id, (y, x) in enumerate(((64, 64), (176, 176)), start=1):
        image[y:y + 2, x:x + 2] = 255
        labels[y:y + 2, x:x + 2] = seg_id

    return image, labels


def validate_package(image, labels, model_type, name, output_path):
    export_sam_model(image=image, label_image=labels, model_type=model_type, name=name, output_path=output_path)
    summary = bioimageio.core.test_model(output_path, devices=["cpu"])
    if summary.status != "passed":
        raise RuntimeError(summary.format())
    print(f"Validated {name}")


def check_ais_export():
    cache_dir = Path(util.get_cache_directory()) / "sample_data"
    tracking_dir = Path(fetch_tracking_example_data(cache_dir))
    ctc_image = imageio.imread(tracking_dir / "t000.tif")

    model = load_ais_model()
    ctc_masks = run_ais(model, ctc_image)
    assert ctc_masks.shape[1] > 0
    ctc_labels = make_export_labels(ctc_masks, ctc_image.shape)

    empty_image, empty_labels = make_empty_ais_case()
    empty_masks = run_ais(model, empty_image)
    assert empty_masks.shape[1] == 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        validate_package(ctc_image, ctc_labels, BASE_MODEL_TYPE, "decoder-free-ctc", output_dir / "decoder-free.zip")
        validate_package(ctc_image, ctc_labels, AIS_MODEL_TYPE, "ais-ctc", output_dir / "ais-ctc.zip")
        validate_package(empty_image, empty_labels, AIS_MODEL_TYPE, "ais-empty", output_dir / "ais-empty.zip")


def main():
    check_ais_export()


if __name__ == "__main__":
    main()
