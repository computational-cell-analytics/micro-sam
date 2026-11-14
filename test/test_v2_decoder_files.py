import os
import shutil

import pytest
import numpy as np
import imageio.v3 as imageio

import torch

MODEL_TYPE = "hvit_t_cells"


def test_find_decoder_checkpoint(tmp_path):
    from micro_sam.v2.instance_segmentation import find_decoder_checkpoint

    model_path = os.path.join(tmp_path, "hvit_t_cells.pt")
    open(model_path, "w").close()
    assert find_decoder_checkpoint(model_path) is None

    for decoder_name in ("hvit_t_cells_decoder", "hvit_t_cells_decoder.pt"):
        decoder_path = os.path.join(tmp_path, decoder_name)
        open(decoder_path, "w").close()
        assert find_decoder_checkpoint(model_path) == decoder_path
        os.remove(decoder_path)

    # A model file without an extension, as in the model cache.
    model_path = os.path.join(tmp_path, "hvit_t_cells")
    open(model_path, "w").close()
    open(model_path + "_decoder", "w").close()
    assert find_decoder_checkpoint(model_path) == model_path + "_decoder"


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """The registry files as a model file and a decoder file, a joint checkpoint, and a model file alone."""
    from micro_sam.util import load_checkpoint
    from micro_sam.v2.util import _download_finetuned_sam2_model

    model_path, _, decoder_path = _download_finetuned_sam2_model(MODEL_TYPE)
    folder = tmp_path_factory.mktemp("checkpoints")

    pair = os.path.join(folder, "pair")
    os.makedirs(pair)
    shutil.copyfile(model_path, os.path.join(pair, "hvit_t_cells.pt"))
    shutil.copyfile(decoder_path, os.path.join(pair, "hvit_t_cells_decoder.pt"))

    decoder_state = load_checkpoint(decoder_path)
    joint = os.path.join(folder, "best.pt")
    torch.save({
        "model_state": load_checkpoint(model_path)["model"],
        "decoder_state": {k: v for k, v in decoder_state.items() if not k.startswith("encoder.inner.")},
    }, joint)

    alone = os.path.join(folder, "alone")
    os.makedirs(alone)
    shutil.copyfile(model_path, os.path.join(alone, "hvit_t_cells.pt"))
    return {
        "pair": os.path.join(pair, "hvit_t_cells.pt"), "joint": joint, "alone": os.path.join(alone, "hvit_t_cells.pt"),
    }


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    from micro_sam.sample_data import fetch_livecell_example_data
    return imageio.imread(fetch_livecell_example_data(tmp_path_factory.mktemp("sample_data")))[:256, :256]


@pytest.mark.slow
@pytest.mark.parametrize("source", ["pair", "joint"])
def test_automatic_segmentation_with_decoder_files(checkpoints, image, source):
    from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

    predictor, segmenter = get_predictor_and_segmenter(MODEL_TYPE, device="cpu")
    expected = automatic_instance_segmentation(predictor, segmenter, image, ndim=2, verbose=False)

    predictor, segmenter = get_predictor_and_segmenter("hvit_t", checkpoint=checkpoints[source], device="cpu")
    segmentation = automatic_instance_segmentation(predictor, segmenter, image, ndim=2, verbose=False)
    assert len(np.unique(expected)) > 2
    assert np.array_equal(segmentation, expected)


@pytest.mark.slow
def test_model_file_without_decoder(checkpoints):
    from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter
    from micro_sam.v2.instance_segmentation import NoDecoderError

    with pytest.raises(NoDecoderError):
        get_predictor_and_segmenter("hvit_t", checkpoint=checkpoints["alone"], device="cpu")


@pytest.mark.slow
@pytest.mark.parametrize("source,has_decoder", [("pair", True), ("joint", True), ("alone", False)])
def test_annotator_state_loads_decoder_files(checkpoints, image, source, has_decoder):
    from micro_sam.sam_annotator._state import AnnotatorState

    state = AnnotatorState()
    state.reset_state()
    try:
        state.initialize_predictor(
            image, model_type="hvit_t", ndim=2, device="cpu", checkpoint_path=checkpoints[source],
        )
        assert (state.decoder is not None) == has_decoder
    finally:
        state.reset_state()


@pytest.mark.slow
def test_annotator_state_takes_a_decoder_path(checkpoints, image):
    from micro_sam.sam_annotator._state import AnnotatorState

    pair_folder = os.path.dirname(checkpoints["pair"])
    state = AnnotatorState()
    state.reset_state()
    try:
        state.initialize_predictor(
            image, model_type="hvit_t", ndim=2, device="cpu", checkpoint_path=checkpoints["alone"],
            decoder_path=os.path.join(pair_folder, "hvit_t_cells_decoder.pt"),
        )
        assert state.decoder is not None
    finally:
        state.reset_state()
