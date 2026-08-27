import types

import numpy as np
import pytest
import torch

from bioimage_cpp.utils import Blocking

from micro_sam.v2.instance_segmentation import (
    UniSAM2InstanceSegmentation, get_instance_segmentation_generator,
)
from micro_sam.v2.automatic_prompt_generation import (
    AutomaticPromptGenerator, TiledAutomaticPromptGenerator, derive_point_prompts, merge_by_score,
    interior_points, derive_refinement_prompts, mask_to_logits, _parse_refinement,
    postmerge_refinement_gate_features, _lowres_feature_context, REFINEMENT_STATS_3D,
)
from micro_sam.v2.normalization import to_image
from micro_sam.v2.multimask_selection import (
    GroupwiseMLP, MASK_TOKEN_FEATURE_NAMES, MASK_TOKEN_LOWRES_FEATURE_NAMES,
    MULTIMASK_FEATURE_NAMES, REFINEMENT_GATE_FEATURE_NAMES, combine_selector_features_torch,
    POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES, extract_multimask_features_torch,
    load_feature_scorer, refinement_gate_features_torch,
)


def test_apg_declares_no_postprocessing_mode():
    # The front end reads this to decide whether 'generate' takes the AIS 'mode' argument.
    assert AutomaticPromptGenerator._has_postprocessing_mode is False
    assert TiledAutomaticPromptGenerator._has_postprocessing_mode is False
    assert getattr(UniSAM2InstanceSegmentation, "_has_postprocessing_mode", True) is True


@pytest.mark.parametrize(
    "schema,expected",
    [("lowres_v1", 19), ("token_v1", 258), ("token_lowres_v1", 275)],
)
def test_compact_selector_feature_schemas_are_three_mask_only(schema, expected):
    lowres = torch.arange(2 * 3 * 19, dtype=torch.float32).reshape(2, 3, 19)
    scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    tokens = torch.arange(2 * 3 * 256, dtype=torch.float32).reshape(2, 3, 256)
    features = combine_selector_features_torch(schema, lowres, scores, tokens)
    assert features.shape == (2, 3, expected)
    if schema == "token_v1":
        assert tuple(MASK_TOKEN_FEATURE_NAMES) and torch.equal(features[:, :, 0], scores)
        assert torch.equal(features[0, :, 1], torch.arange(3, dtype=torch.float32))
    if schema == "token_lowres_v1":
        assert len(MASK_TOKEN_LOWRES_FEATURE_NAMES) == expected
        assert torch.equal(features[:, :, :19], lowres)

    with pytest.raises(ValueError, match="three multimask alternatives"):
        combine_selector_features_torch(schema, lowres[:, :2], scores[:, :2], tokens[:, :2])


def test_lowres_feature_context_uses_padded_resize_coordinates():
    class Transforms:
        resolution = 16

        def transform_coords(self, coords, normalize, orig_hw):
            assert normalize and orig_hw == (4, 8)
            return coords * (self.resolution / max(orig_hw))

    predictor = types.SimpleNamespace(
        model=types.SimpleNamespace(image_size=16), _orig_hw=[(4, 8)], _transforms=Transforms(),
    )
    foreground = np.arange(32, dtype="float32").reshape(4, 8)
    resized, points = _lowres_feature_context(
        predictor, foreground, np.array([[4.0, 2.0]], dtype="float32"), (4, 4), torch.device("cpu"),
    )
    expected = torch.nn.functional.interpolate(
        torch.as_tensor(foreground)[None, None], size=(8, 16), mode="bilinear",
        align_corners=False, antialias=True,
    )
    expected = torch.nn.functional.pad(expected, (0, 0, 0, 8))
    expected = torch.nn.functional.interpolate(
        expected, size=(4, 4), mode="bilinear", align_corners=False, antialias=True,
    )[0, 0]
    assert torch.allclose(resized, expected)
    assert torch.allclose(points, torch.tensor([[2.0, 1.0]]))


def test_factory_rejects_incomplete_apg_arguments():
    with pytest.raises(ValueError, match="decoder"):
        get_instance_segmentation_generator(segmentation_mode="apg")
    with pytest.raises(ValueError, match="model"):
        get_instance_segmentation_generator(segmentation_mode="apg", decoder=object())
    with pytest.raises(ValueError, match="Invalid segmentation_mode"):
        get_instance_segmentation_generator(segmentation_mode="unknown")


@pytest.mark.parametrize("is_tiled,expected", [(False, AutomaticPromptGenerator),
                                               (True, TiledAutomaticPromptGenerator)])
def test_factory_returns_the_apg_classes(monkeypatch, is_tiled, expected):
    predictor = types.SimpleNamespace(model=types.SimpleNamespace(model_type="hvit_b"))
    monkeypatch.setattr("micro_sam.v2.util.get_sam2_image_predictor", lambda model: predictor)

    decoder = object()
    segmenter = get_instance_segmentation_generator(
        model=predictor.model, decoder=decoder, segmentation_mode="apg", is_tiled=is_tiled,
    )
    assert type(segmenter) is expected
    assert segmenter._model is decoder
    assert segmenter._predictor is predictor
    # The embedding cache is keyed on these, which a SAM2 image predictor does not carry by itself.
    assert predictor.model_type == "hvit_b"
    assert predictor.model_name == "hvit_b"


def test_apg_encodes_multichannel_images_with_per_channel_normalization():
    class Predictor:
        device = "cpu"
        model = types.SimpleNamespace(image_size=1024)
        _features = {"high_res_feats": []}
        _orig_hw = [(2, 3)]

        def reset_predictor(self):
            pass

        def set_image(self, image):
            self.image = image

        def get_image_embedding(self):
            return torch.zeros((1, 1, 1, 1))

    values = np.arange(6, dtype="float32").reshape(2, 3)
    image = np.stack([values, 1000.0 + 100.0 * values, 10.0 - values], axis=-1)
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._predictor = Predictor()

    segmenter._encode(image)

    assert np.array_equal(segmenter._predictor.image, to_image(image))
    assert np.array_equal(segmenter._predictor.image.min(axis=(0, 1)), [0, 0, 0])
    assert np.array_equal(segmenter._predictor.image.max(axis=(0, 1)), [255, 255, 255])


def test_derive_point_prompts_returns_xy_points_inside_the_candidates():
    foreground = np.zeros((32, 32), dtype="float32")
    foreground[4:12, 20:28] = 1.0
    # A flow that points into the blob from every side, so the density converges inside it.
    distances = np.zeros((2, 32, 32), dtype="float32")
    ys, xs = np.mgrid[0:32, 0:32]
    distances[0] = (ys - 8.0) * foreground
    distances[1] = (xs - 24.0) * foreground

    prompts = derive_point_prompts(
        foreground, distances, candidate_threshold=1.0, foreground_threshold=0.5, min_candidate_size=1,
    )
    assert prompts is not None
    points = prompts["points"]
    assert points.ndim == 3 and points.shape[1:] == (1, 2)
    assert (prompts["point_labels"] == 1).all()
    for x, y in points[:, 0, :]:
        # XY order, and the point has to lie in the blob rather than beside it.
        assert foreground[int(y), int(x)] > 0.5


def test_derive_point_prompts_returns_none_without_candidates():
    foreground = np.zeros((16, 16), dtype="float32")
    distances = np.zeros((2, 16, 16), dtype="float32")
    assert derive_point_prompts(foreground, distances, candidate_threshold=1.0) is None


def test_interior_points_lie_in_their_own_component():
    labels = np.zeros((32, 32), dtype="uint32")
    labels[4:14, 4:20] = 1  # A solid block, whose deepest point is its middle.
    labels[np.arange(20, 28), np.arange(20, 28)] = 2  # A one pixel wide diagonal.
    labels[0, 30] = 3  # A single pixel on the image border.

    points = interior_points(labels)
    assert len(points) == 3
    # The v1 helper places the thin ones outside the component they were derived for.
    for label_id, point in enumerate(points, start=1):
        assert labels[tuple(point)] == label_id
    # Ten rows high, so five is as deep as it gets, and the first such pixel wins.
    assert tuple(points[0]) == (8, 8)


def test_interior_points_skips_missing_labels():
    labels = np.zeros((16, 16), dtype="uint32")
    labels[2:6, 2:6] = 1
    labels[10:14, 10:14] = 3  # Label 2 is absent, as it is once the size filter has run.

    points = interior_points(labels)
    assert len(points) == 2
    assert labels[tuple(points[0])] == 1
    assert labels[tuple(points[1])] == 3


def test_merge_by_score_truncates_to_the_unclaimed_pixels():
    shape = (16, 16)
    high = np.zeros(shape, dtype=bool)
    high[2:10, 2:10] = True
    low = np.zeros(shape, dtype=bool)
    low[8:14, 8:14] = True  # overlaps the better-scoring mask in a 2x2 corner
    records = [
        {"segmentation": low, "predicted_iou": 0.5, "stability_score": 0.5},
        {"segmentation": high, "predicted_iou": 0.9, "stability_score": 0.9},
    ]

    segmentation = merge_by_score(records, shape, max_overlap=0.3, min_size=1)
    assert set(np.unique(segmentation)) == {0, 1, 2}
    # The better-scoring mask is painted whole and keeps the contested corner.
    assert int((segmentation == 1).sum()) == int(high.sum())
    assert int((segmentation == 2).sum()) == int(low.sum()) - 4


def test_merge_by_score_reports_why_each_record_was_dropped():
    shape = (16, 16)
    high = np.zeros(shape, dtype=bool)
    high[2:12, 2:12] = True
    inside = np.zeros(shape, dtype=bool)
    inside[3:9, 3:9] = True  # entirely inside the better-scoring mask
    tiny = np.zeros(shape, dtype=bool)
    tiny[14, 14] = True
    records = [
        {"segmentation": inside, "predicted_iou": 0.5, "stability_score": 0.5},
        {"segmentation": high, "predicted_iou": 0.9, "stability_score": 0.9},
        {"segmentation": tiny, "predicted_iou": 0.8, "stability_score": 0.8},
    ]

    segmentation, reasons = merge_by_score(
        records, shape, max_overlap=0.3, min_size=4, return_reasons=True
    )
    # The reasons are in the order the records were given, not in merge order.
    assert reasons == ["duplicate", "kept", "too small"]
    assert set(np.unique(segmentation)) == {0, 1}


def test_merge_by_score_reasons_do_not_change_the_segmentation():
    shape = (16, 16)
    first = np.zeros(shape, dtype=bool)
    first[2:10, 2:10] = True
    second = np.zeros(shape, dtype=bool)
    second[8:14, 8:14] = True
    records = [
        {"segmentation": second, "predicted_iou": 0.5, "stability_score": 0.5},
        {"segmentation": first, "predicted_iou": 0.9, "stability_score": 0.9},
    ]

    plain = merge_by_score(records, shape, max_overlap=0.3, min_size=1)
    with_extras, matches, reasons = merge_by_score(
        records, shape, max_overlap=0.3, min_size=1, return_matches=True, return_reasons=True
    )
    assert np.array_equal(plain, with_extras)
    assert matches == {1: 1, 2: 0}
    assert reasons == ["kept", "kept"]


def test_merge_by_score_rejects_a_candidate_that_is_mostly_claimed():
    shape = (16, 16)
    high = np.zeros(shape, dtype=bool)
    high[2:12, 2:12] = True
    inside = np.zeros(shape, dtype=bool)
    inside[3:9, 3:9] = True  # entirely inside the better-scoring mask
    records = [
        {"segmentation": inside, "predicted_iou": 0.5, "stability_score": 0.5},
        {"segmentation": high, "predicted_iou": 0.9, "stability_score": 0.9},
    ]
    segmentation = merge_by_score(records, shape, max_overlap=0.3, min_size=1)
    assert set(np.unique(segmentation)) == {0, 1}


def test_multimask_features_include_prompt_and_triplet_evidence():
    masks = np.zeros((2, 3, 8, 8), dtype=bool)
    masks[0, 0, 1:4, 1:4] = True
    masks[0, 1, 1:6, 1:6] = True
    masks[0, 2, 0:8, 0:8] = True
    masks[1, :, 5:8, 5:8] = True
    scores = np.array([[0.9, 0.8, 0.7], [0.7, 0.8, 0.9]], dtype="float32")
    stability = np.full((2, 3), 0.8, dtype="float32")
    points = np.array([[2, 2], [6, 6]], dtype="float32")
    foreground = np.zeros((8, 8), dtype="float32")
    foreground[1:7, 1:7] = 1.0

    features = extract_multimask_features_torch(
        torch.as_tensor(masks), torch.as_tensor(scores), torch.as_tensor(stability),
        points, foreground, 0.7,
    )

    assert features.shape == (2, 3, len(MULTIMASK_FEATURE_NAMES))
    assert torch.isfinite(features).all()
    # All alternatives contain their own seed. Only the largest first-prompt alternative contains
    # the other prompt, and its foreground precision is lower because it covers the whole image.
    assert (features[:, :, 12] == 1).all()
    assert torch.equal(features[0, :, 11], torch.tensor([0.0, 0.0, 1.0]))
    assert features[0, 2, 14] < features[0, 1, 14]

    gate = refinement_gate_features_torch(features, torch.as_tensor(scores), torch.tensor([1, 2]))
    assert gate.shape == (2, len(REFINEMENT_GATE_FEATURE_NAMES))
    assert gate[0, -3] == pytest.approx(-0.1)
    assert gate[0, -1] == 1.0

@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for GPU feature parity.",
))])
def test_torch_multimask_features_are_device_stable(device):
    rng = np.random.default_rng(17)
    masks = rng.random((4, 3, 13, 15)) > 0.65
    # Exercise an empty alternative, tied scores/ranks, repeated seeds and clipped boundary points.
    masks[0, 2] = False
    scores = np.array([
        [0.8, 0.8, 0.4], [0.6, 0.7, 0.9], [0.9, 0.5, 0.7], [0.3, 0.4, 0.5],
    ], dtype="float32")
    stability = rng.random((4, 3), dtype="float32")
    points = np.array([[-2, 2], [7, 6], [7, 6], [20, 12]], dtype="float32")
    context = np.concatenate((points, np.array([[3, 4], [12, 8]], dtype="float32")))
    foreground = rng.random((13, 15), dtype="float32")
    indices = np.arange(4)

    expected = extract_multimask_features_torch(
        torch.as_tensor(masks), torch.as_tensor(scores), torch.as_tensor(stability),
        points, foreground, 0.7, context, indices,
    )
    actual = extract_multimask_features_torch(
        torch.as_tensor(masks, device=device), torch.as_tensor(scores, device=device),
        torch.as_tensor(stability, device=device), points, foreground, 0.7, context, indices,
    ).cpu()
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    selected = torch.tensor([0, 2, 1, 2])
    expected_gate = refinement_gate_features_torch(expected, torch.as_tensor(scores), selected)
    actual_gate = refinement_gate_features_torch(
        actual.to(device), torch.as_tensor(scores, device=device), selected.to(device),
    ).cpu()
    assert torch.allclose(actual_gate, expected_gate, rtol=1e-5, atol=1e-5)


def test_pointwise_mlp_artifact_roundtrip(tmp_path):
    names = REFINEMENT_GATE_FEATURE_NAMES
    module = torch.nn.Sequential(
        torch.nn.Linear(len(names), 16), torch.nn.ReLU(), torch.nn.Linear(16, 1),
    )
    state = {
        "kind": "mlp", "feature_version": 1, "feature_names": list(names),
        "hidden_sizes": [16], "dropout": 0.0,
        "mean": np.zeros(len(names), dtype="float32"),
        "scale": np.ones(len(names), dtype="float32"),
        "state_dict": module.state_dict(), "metadata": {},
    }
    path = tmp_path / "gate.pt"
    torch.save(state, path)
    scorer = load_feature_scorer(path)

    assert scorer.predict(np.zeros((2, len(names)), dtype="float32")).shape == (2,)


def test_signed_postmerge_gate_artifact_preserves_negative_predictions(tmp_path):
    names = POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES
    module = torch.nn.Sequential(torch.nn.Linear(len(names), 1))
    torch.nn.init.zeros_(module[0].weight)
    torch.nn.init.constant_(module[0].bias, -0.25)
    path = tmp_path / "signed-gate.pt"
    torch.save({
        "kind": "mlp", "feature_version": 1, "feature_names": list(names),
        "hidden_sizes": [], "dropout": 0.0,
        "mean": np.zeros(len(names), dtype="float32"),
        "scale": np.ones(len(names), dtype="float32"),
        "state_dict": module.state_dict(),
        "metadata": {"gate_stage": "postmerge", "output_activation": "identity"},
    }, path)
    scorer = load_feature_scorer(path)

    prediction = scorer.predict(np.zeros((2, len(names)), dtype="float32"))
    assert scorer.gate_stage == "postmerge"
    assert np.allclose(prediction, -0.25)


@pytest.mark.parametrize(
    "names,stage,error",
    [
        (REFINEMENT_GATE_FEATURE_NAMES, "during-merge", "Unsupported refinement gate stage"),
        (REFINEMENT_GATE_FEATURE_NAMES, "postmerge", "Pre-merge refinement gate features"),
        (POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES, "premerge", "Post-merge refinement gate features"),
        (MULTIMASK_FEATURE_NAMES, "postmerge", "refinement-gate feature schema"),
    ],
)
def test_refinement_gate_artifacts_validate_their_stage_and_schema(tmp_path, names, stage, error):
    module = torch.nn.Sequential(torch.nn.Linear(len(names), 1))
    path = tmp_path / "invalid-gate.pt"
    torch.save({
        "kind": "mlp", "feature_version": 1, "feature_names": list(names),
        "hidden_sizes": [], "dropout": 0.0,
        "mean": np.zeros(len(names), dtype="float32"),
        "scale": np.ones(len(names), dtype="float32"),
        "state_dict": module.state_dict(), "metadata": {"gate_stage": stage},
    }, path)

    with pytest.raises(ValueError, match=error):
        load_feature_scorer(path)


def test_installing_a_custom_refinement_gate_validates_its_stage():
    segmenter = object.__new__(AutomaticPromptGenerator)
    invalid_gate = types.SimpleNamespace(gate_stage="during-merge")

    with pytest.raises(ValueError, match="Unsupported refinement gate stage"):
        segmenter.set_multimask_models(refinement_gate=invalid_gate)


def test_groupwise_mlp_artifact_roundtrip_and_permutation_equivariance(tmp_path):
    torch.manual_seed(17)
    module = GroupwiseMLP(len(MULTIMASK_FEATURE_NAMES), hidden_size=32, dropout=0.0)
    state = {
        "kind": "groupwise_mlp", "feature_version": 1,
        "feature_names": list(MULTIMASK_FEATURE_NAMES), "n_alternatives": 3,
        "hidden_size": 32, "dropout": 0.0,
        "mean": np.zeros(len(MULTIMASK_FEATURE_NAMES), dtype="float32"),
        "scale": np.ones(len(MULTIMASK_FEATURE_NAMES), dtype="float32"),
        "state_dict": module.state_dict(), "metadata": {},
    }
    path = tmp_path / "groupwise.pt"
    torch.save(state, path)
    scorer = load_feature_scorer(path)
    features = np.random.default_rng(4).normal(size=(5, 3, len(MULTIMASK_FEATURE_NAMES))).astype("float32")
    prediction = scorer.predict_grouped(features)
    permutation = np.array([2, 0, 1])
    permuted = scorer.predict_grouped(features[:, permutation])

    assert prediction.shape == (5, 3)
    assert np.allclose(permuted, prediction[:, permutation])


def test_groupwise_mlp_artifact_supports_singleton_groups(tmp_path):
    module = GroupwiseMLP(len(MULTIMASK_FEATURE_NAMES), hidden_size=16, dropout=0.0)
    state = {
        "kind": "groupwise_mlp", "feature_version": 1,
        "feature_names": list(MULTIMASK_FEATURE_NAMES), "n_alternatives": 1,
        "hidden_size": 16, "dropout": 0.0,
        "mean": np.zeros(len(MULTIMASK_FEATURE_NAMES), dtype="float32"),
        "scale": np.ones(len(MULTIMASK_FEATURE_NAMES), dtype="float32"),
        "state_dict": module.state_dict(), "metadata": {},
    }
    path = tmp_path / "single-groupwise.pt"
    torch.save(state, path)
    scorer = load_feature_scorer(path)
    features = np.random.default_rng(5).normal(
        size=(4, 1, len(MULTIMASK_FEATURE_NAMES)),
    ).astype("float32")

    assert scorer.predict_grouped(features).shape == (4, 1)


def test_grouped_merge_accepts_at_most_one_alternative_per_prompt():
    shape = (16, 16)
    first = np.zeros(shape, dtype=bool)
    first[2:8, 2:8] = True
    second = np.zeros(shape, dtype=bool)
    second[2:10, 2:10] = True
    independent = np.zeros(shape, dtype=bool)
    independent[10:15, 10:15] = True
    records = [
        {"segmentation": first, "predicted_iou": 0.9, "stability_score": 1.0,
         "merge_score": 0.9, "multimask_group": 0},
        {"segmentation": second, "predicted_iou": 0.8, "stability_score": 1.0,
         "merge_score": 0.8, "multimask_group": 0},
        {"segmentation": independent, "predicted_iou": 0.7, "stability_score": 1.0,
         "merge_score": 0.7, "multimask_group": 1},
    ]

    segmentation, reasons = merge_by_score(records, shape, max_overlap=0.3, min_size=1, return_reasons=True)

    assert set(np.unique(segmentation)) == {0, 1, 2}
    assert reasons == ["kept", "alternative not selected", "kept"]


def test_grouped_merge_tries_a_lower_alternative_after_rejection():
    shape = (16, 16)
    claimed = np.zeros(shape, dtype=bool)
    claimed[1:10, 1:10] = True
    rejected = claimed.copy()
    fallback = np.zeros(shape, dtype=bool)
    fallback[10:15, 10:15] = True
    records = [
        {"segmentation": claimed, "predicted_iou": 0.95, "stability_score": 1.0, "merge_score": 0.95},
        {"segmentation": rejected, "predicted_iou": 0.9, "stability_score": 1.0,
         "merge_score": 0.9, "multimask_group": 3},
        {"segmentation": fallback, "predicted_iou": 0.8, "stability_score": 1.0,
         "merge_score": 0.8, "multimask_group": 3},
    ]

    segmentation, matches, reasons = merge_by_score(
        records, shape, max_overlap=0.3, min_size=1, return_matches=True, return_reasons=True,
    )

    assert reasons == ["kept", "duplicate", "kept"]
    assert matches == {1: 0, 2: 2}
    assert segmentation[12, 12] == 2


def _make_tiled_generator(shape, tile_shape, halo, monkeypatch, predictor=None):
    """A tiled generator with the tiling set up, but no model, predictor or embeddings."""
    segmenter = object.__new__(TiledAutomaticPromptGenerator)
    segmenter._tiling = Blocking([0, 0], list(shape), list(tile_shape))
    segmenter._halo = list(halo)
    segmenter._i = None
    segmenter._predictor = object() if predictor is None else predictor
    segmenter._image_embeddings = object()
    segmenter._prediction = np.zeros((4, *shape), dtype="float32")
    segmenter._last_generation_stats = {}
    segmenter.visited_tiles = []

    def set_precomputed(predictor_, image_embeddings, i=None, tile_id=None):
        """Record the visit and resize the fake predictor to the tile, as the real one does."""
        segmenter.visited_tiles.append(tile_id)
        if hasattr(predictor_, "shape"):
            predictor_.shape = tuple(box.stop - box.start for box in segmenter._tile_bounding_box(tile_id))

    monkeypatch.setattr("micro_sam.v2.automatic_prompt_generation.set_precomputed", set_precomputed)
    return segmenter


def test_generator_prepares_a_video_embedding_slice(monkeypatch):
    calls = []
    feature = torch.zeros((1, 4, 8, 8), dtype=torch.float32, requires_grad=True)
    predictor = types.SimpleNamespace(
        model=types.SimpleNamespace(image_size=1024),
        _features=None,
        _orig_hw=None,
    )

    def set_slice(image_predictor, image_embeddings, i):
        calls.append((image_embeddings, i))
        image_predictor._features = {
            "image_embed": feature,
            "high_res_feats": [np.zeros((1, 2, 16, 16), dtype="float32")],
        }
        image_predictor._orig_hw = [(64, 64)]

    predictor.get_image_embedding = lambda: predictor._features["image_embed"]
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation._set_image_predictor_from_3d_embeddings", set_slice,
    )

    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._predictor = predictor
    video_embeddings = {"features": object(), "fpn": object()}
    image_embeddings = segmenter._prepare_image_embeddings(video_embeddings, i=3)

    assert calls == [(video_embeddings, 3)]
    assert isinstance(image_embeddings["features"], np.ndarray)
    assert image_embeddings["features"].shape == (1, 4, 8, 8)
    assert predictor._features["image_embed"] is feature
    assert predictor._features["image_embed"].requires_grad
    assert image_embeddings["original_size"] == [(64, 64)]


def test_tiled_refinement_sets_a_video_embedding_slice(monkeypatch):
    calls = []

    class Feature:
        attrs = {"original_size": (32, 32)}

    image_embeddings = {
        "features": {"0": Feature()},
        "fpn": {"0": {"0": "fpn-0", "1": "fpn-1"}},
        "pos_enc": {"0": {"0": "pos-0", "1": "pos-1"}},
    }
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation._set_image_predictor_from_backbone",
        lambda *args: calls.append(args),
    )

    segmenter = object.__new__(TiledAutomaticPromptGenerator)
    segmenter._predictor = object()
    segmenter._image_embeddings = image_embeddings
    segmenter._i = 2
    segmenter._set_region(0)

    assert calls == [(
        segmenter._predictor, ["fpn-0", "fpn-1"], ["pos-0", "pos-1"],
        image_embeddings["features"]["0"], (32, 32), 2,
    )]


def test_tiled_generator_restores_decoder_state():
    class Features:
        attrs = {"shape": (64, 64), "tile_shape": (32, 32), "halo": (8, 8)}

    prediction = np.ones((4, 64, 64), dtype="float32")
    image_embeddings = {"features": Features(), "input_size": None}
    segmenter = object.__new__(TiledAutomaticPromptGenerator)
    segmenter._prediction = None
    segmenter._is_initialized = False
    segmenter._image_embeddings = None
    segmenter._i = None
    segmenter._owns_image_embeddings = False
    segmenter._tiling = None
    segmenter._halo = None

    segmenter.set_state({"prediction": prediction, "image_embeddings": image_embeddings, "i": 3})

    assert segmenter._prediction is prediction
    assert segmenter._image_embeddings is image_embeddings
    assert segmenter._i == 3
    assert segmenter._tiling.number_of_blocks == 4
    assert segmenter._halo == [8, 8]


def test_tiles_for_points_assigns_every_prompt_to_exactly_one_tile(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch)

    # XY points, one per quadrant, plus one in the halo overlap of the first tile.
    points = np.array([[[5, 5]], [[40, 5]], [[5, 40]], [[40, 40]], [[35, 5]]], dtype="float32")
    assignment = segmenter._tiles_for_points(points)

    assert sum(len(v) for v in assignment.values()) == len(points)
    assert sorted(index for indices in assignment.values() for index in indices) == list(range(5))
    # (x=35, y=5) sits in the first tile's halo but the second tile's inner block, so only that tile
    # prompts it.
    assert assignment[0] == [0]
    assert assignment[1] == [1, 4]
    assert assignment[2] == [2]
    assert assignment[3] == [3]


def test_tiled_apply_and_select_maps_prompts_and_masks_between_frames(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch)

    tile_ids = []
    real_tile_bounding_box = segmenter._tile_bounding_box

    def tile_bounding_box(tile_id):
        """Track which tile is being prompted, so the fake can build a correctly shaped mask."""
        tile_bounding_box.current = real_tile_bounding_box(tile_id)
        tile_ids.append(tile_id)
        return tile_bounding_box.current

    def apply_prompts(prompts, multimasking, batch_size):
        """One 6x6 record per prompt, centred on the (tile-local) prompt point."""
        box = tile_bounding_box.current
        tile_shape_ = tuple(s.stop - s.start for s in box)
        records = []
        for x, y in prompts["points"][:, 0, :]:
            mask = np.zeros(tile_shape_, dtype=bool)
            mask[int(y) - 3:int(y) + 3, int(x) - 3:int(x) + 3] = True
            records.append({
                "segmentation": mask, "predicted_iou": 0.9, "stability_score": 0.9,
                "point": (float(x), float(y)),
            })
        return records

    segmenter._tile_bounding_box = tile_bounding_box
    segmenter._apply_prompts = apply_prompts

    # Two prompts in different tiles, in XY.
    points = np.array([[[10, 10]], [[50, 50]]], dtype="float32")
    prompts = {"points": points, "point_labels": np.ones((2, 1), dtype="int32")}
    proposals = segmenter._apply(prompts, multimasking=False, batch_size=8)
    # The records return to the full image's frame, although the prompting is tile-local.
    assert sorted(record["point"] for proposal in proposals for record in proposal["records"]) \
        == [(10.0, 10.0), (50.0, 50.0)]
    segmentation, context = segmenter._merge(
        proposals, shape, score_threshold=0.0, max_overlap=0.3, min_size=1,
    )
    assert context is None  # Only a refinement asks for one.

    assert segmentation.shape == shape
    assert segmentation.dtype == np.dtype("uint32")
    # Two instances, each sitting at its prompt in the full image's frame rather than at a tile offset.
    assert len(np.unique(segmentation)) == 3
    assert segmentation[10, 10] != 0
    assert segmentation[50, 50] != 0
    assert segmentation[10, 10] != segmentation[50, 50]
    assert tile_ids == sorted(set(tile_ids))  # every tile with prompts is visited once, in order
    # The tile a proposal came from is carried, so a refinement knows where to re-prompt it.
    assert [proposal["tile_id"] for proposal in proposals] == tile_ids


def test_tiled_generator_cannot_serialize_or_restore_without_embeddings(monkeypatch):
    segmenter = _make_tiled_generator((64, 64), (32, 32), (8, 8), monkeypatch)
    with pytest.raises(NotImplementedError):
        segmenter.get_state()
    with pytest.raises(ValueError, match="image_embeddings"):
        segmenter.set_state({})


def test_reinitializing_generator_releases_owned_volume_embeddings(monkeypatch):
    closed, removed, precompute_paths, propagators = [], [], [], []

    class Embeddings(dict):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def close(self):
            closed.append(self.path)

    class Propagator:
        def __init__(self):
            self.was_reset = False

        def reset_predictor(self):
            self.was_reset = True

    paths = iter(["first.zarr", "second.zarr"])

    def precompute(predictor, image, **kwargs):
        path = kwargs["save_path"]
        precompute_paths.append(path)
        return Embeddings(path)

    def build_propagator(self, volume, image_embeddings):
        propagator = Propagator()
        propagators.append(propagator)
        return propagator

    monkeypatch.setattr("micro_sam.v2.automatic_prompt_generation.make_temp_embedding_path", lambda: next(paths))
    monkeypatch.setattr("micro_sam.v2.automatic_prompt_generation.precompute_image_embeddings", precompute)
    monkeypatch.setattr("micro_sam.v2.automatic_prompt_generation.set_precomputed", lambda *args: None)
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation.UniSAM2InstanceSegmentation.initialize",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(AutomaticPromptGenerator, "_build_propagator", build_propagator)
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation.shutil.rmtree",
        lambda path, **kwargs: removed.append(path),
    )

    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._predictor = types.SimpleNamespace(reset_predictor=lambda: None)
    segmenter._video_predictor = types.SimpleNamespace()
    segmenter._prediction = None
    segmenter._is_initialized = False
    segmenter._image_embeddings = None
    segmenter._owns_image_embeddings = False
    segmenter._volume = None
    segmenter._propagator = None
    segmenter._temporary_embedding_path = None
    volume = np.zeros((2, 16, 16), dtype="uint8")

    segmenter.initialize(volume, ndim=3)
    segmenter.initialize(volume, ndim=3)
    assert precompute_paths == ["first.zarr", "second.zarr"]
    assert closed == ["first.zarr"]
    assert removed == ["first.zarr"]
    assert propagators[0].was_reset

    external_embeddings = Embeddings("external.zarr")
    segmenter.initialize(volume[0], ndim=2, image_embeddings=external_embeddings)
    assert closed == ["first.zarr", "second.zarr"]
    assert removed == ["first.zarr", "second.zarr"]
    assert propagators[1].was_reset
    assert segmenter._volume is None
    assert segmenter._propagator is None
    segmenter.clear_state()
    assert "external.zarr" not in closed

    segmenter.initialize(volume, ndim=3, save_path="user.zarr")
    segmenter.clear_state()
    assert closed[-1] == "user.zarr"
    assert "user.zarr" not in removed


def test_reinitializing_tiled_generator_removes_the_previous_temporary_store(monkeypatch):
    class Features:
        attrs = {"shape": (32, 32), "tile_shape": (16, 16), "halo": (4, 4)}

    paths = iter(["first.zarr", "second.zarr"])
    removed = []
    precompute_paths = []

    def precompute(predictor, image, **kwargs):
        precompute_paths.append(kwargs["save_path"])
        return {"features": Features()}

    monkeypatch.setattr("micro_sam.v2.automatic_prompt_generation.make_temp_embedding_path", lambda: next(paths))
    monkeypatch.setattr("micro_sam.v2.automatic_prompt_generation.precompute_image_embeddings", precompute)
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation.TiledUniSAM2InstanceSegmentation.initialize",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation.shutil.rmtree",
        lambda path, **kwargs: removed.append(path),
    )

    segmenter = object.__new__(TiledAutomaticPromptGenerator)
    segmenter._predictor = types.SimpleNamespace(reset_predictor=lambda: None)
    segmenter._prediction = None
    segmenter._is_initialized = False
    segmenter._image_embeddings = None
    segmenter._volume = None
    segmenter._propagator = None
    segmenter._temporary_embedding_path = None
    segmenter._tiling = None
    segmenter._halo = None
    image = np.zeros((32, 32), dtype="uint8")

    segmenter.initialize(image, tile_shape=(16, 16), halo=(4, 4))
    segmenter.initialize(image, tile_shape=(16, 16), halo=(4, 4))

    assert precompute_paths == ["first.zarr", "second.zarr"]
    assert removed == ["first.zarr"]
    assert segmenter._temporary_embedding_path == "second.zarr"

    segmenter.initialize(image, tile_shape=(16, 16), halo=(4, 4), save_path="user.zarr")

    assert precompute_paths == ["first.zarr", "second.zarr", "user.zarr"]
    assert removed == ["first.zarr", "second.zarr"]
    assert segmenter._temporary_embedding_path is None

    segmenter.clear_state()
    assert removed == ["first.zarr", "second.zarr"]


def test_parse_refinement_resolves_the_mode_and_its_defaults():
    components, resolved = _parse_refinement("points+boxes", {"n_positives": 5, "policy": "keep-if-better"})
    assert components == ("points", "boxes")
    assert resolved["n_positives"] == 5
    assert resolved["policy"] == "keep-if-better"
    assert resolved["n_negatives"] == 6  # the measured default fills in
    assert resolved["min_consistency"] == 0.7
    assert resolved["box_extension"] == 0


def test_parse_refinement_rejects_invalid_modes_and_kwargs():
    with pytest.raises(ValueError, match="combination"):
        _parse_refinement("points+blobs", None)
    with pytest.raises(ValueError, match="repetition"):
        _parse_refinement("points+points", None)
    with pytest.raises(ValueError, match="dense-only"):
        _parse_refinement("masks", None)
    with pytest.raises(ValueError, match="policy"):
        _parse_refinement("points", {"policy": "always"})
    # A kwarg of a component that is not part of the mode is as invalid as an unknown one.
    with pytest.raises(ValueError, match="box_extension"):
        _parse_refinement("points", {"box_extension": 2})
    with pytest.raises(ValueError, match="n_positive"):
        _parse_refinement("boxes", {"n_positives": 3})


def _two_instance_segmentation():
    segmentation = np.zeros((32, 32), dtype="uint32")
    segmentation[4:12, 4:12] = 1
    segmentation[4:12, 20:28] = 2
    return segmentation


def test_refinement_prompts_group_suppressed_prompts_onto_their_instance():
    segmentation = _two_instance_segmentation()
    # Three prompts inside instance 1 (one survived, two were suppressed), one inside instance 2,
    # and one on the background, which belongs to nobody.
    points = np.array([[6, 6], [10, 6], [6, 10], [24, 6], [16, 16]], dtype="float32")
    prompts = derive_refinement_prompts(
        segmentation, points, {1: (6.0, 6.0), 2: (24.0, 6.0)}, n_positives=3, n_negatives=0,
    )
    positives = prompts[1]["points"][prompts[1]["point_labels"] == 1]
    assert sorted(map(tuple, positives.tolist())) == [(6.0, 6.0), (6.0, 10.0), (10.0, 6.0)]
    # The background prompt is in nobody's positives.
    all_points = np.concatenate([prompt["points"] for prompt in prompts.values()])
    assert not (all_points == np.array([16.0, 16.0])).all(axis=1).any()


def test_refinement_prompts_always_keep_the_surviving_prompt():
    segmentation = _two_instance_segmentation()
    points = np.array([[6, 6], [10, 6], [6, 10], [10, 10], [24, 6]], dtype="float32")
    prompts = derive_refinement_prompts(
        segmentation, points, {1: (10.0, 10.0), 2: (24.0, 6.0)}, n_positives=2, n_negatives=0,
    )
    positives = prompts[1]["points"][prompts[1]["point_labels"] == 1]
    assert len(positives) == 2
    assert (10.0, 10.0) in map(tuple, positives.tolist())
    # Farthest-point subsampling: of the remaining candidates, (6, 6) is farthest from (10, 10).
    assert (6.0, 6.0) in map(tuple, positives.tolist())


def test_refinement_prompts_take_the_nearest_other_prompts_as_negatives():
    segmentation = np.zeros((32, 64), dtype="uint32")
    segmentation[4:12, 4:12] = 1
    segmentation[4:12, 20:28] = 2
    segmentation[4:12, 50:58] = 3
    points = np.array([[6, 6], [24, 6], [54, 6]], dtype="float32")
    surviving = {1: (6.0, 6.0), 2: (24.0, 6.0), 3: (54.0, 6.0)}

    prompts = derive_refinement_prompts(segmentation, points, surviving, n_positives=1, n_negatives=1)
    negatives = prompts[1]["points"][prompts[1]["point_labels"] == 0]
    # Instance 2's prompt is much closer to instance 1 than instance 3's.
    assert negatives.tolist() == [[24.0, 6.0]]

    # A distance cap excludes even the nearest one when it is too far away.
    prompts = derive_refinement_prompts(
        segmentation, points, surviving, n_positives=1, n_negatives=1, max_negative_distance=5.0,
    )
    assert (prompts[1]["point_labels"] == 0).sum() == 0


def test_postmerge_gate_features_capture_visible_masks_and_assembled_negatives():
    segmentation = _two_instance_segmentation()
    # The second source mask has lost one of its eight columns in the final visible segmentation.
    # Post-merge gates derive both fractions from that final result, without merge-internal claim maps.
    segmentation[4:12, 27] = 0
    records = [
        {
            "segmentation": np.ones((8, 8), dtype=bool),
            "bounding_box": (slice(4, 12), slice(4, 12)),
            "predicted_iou": 0.9, "stability_score": 0.8, "selection_score": 0.85,
            "merge_score": 0.85, "multimask_index": 2, "point": (6.0, 6.0),
        },
        {
            "segmentation": np.ones((8, 8), dtype=bool),
            "bounding_box": (slice(4, 12), slice(20, 28)),
            "predicted_iou": 0.8, "stability_score": 0.9, "selection_score": 0.75,
            "merge_score": 0.75, "multimask_index": 1, "point": (24.0, 6.0),
        },
    ]
    context = {
        "proposals": records, "records": records, "matches": {1: 0, 2: 1},
        "score_filter": "selection_score", "score_threshold": 0.7,
    }
    prompts = derive_refinement_prompts(
        segmentation, np.array([[6, 6], [10, 6], [24, 6]], dtype="float32"),
        {1: (6.0, 6.0), 2: (24.0, 6.0)}, n_positives=1, n_negatives=1,
    )
    foreground = np.ones(segmentation.shape, dtype="float32")
    features, instance_ids = postmerge_refinement_gate_features(
        segmentation, context, prompts, foreground, foreground_threshold=0.5,
    )

    assert instance_ids.tolist() == [1, 2]
    assert features.shape == (2, len(POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES))
    assert np.isfinite(features).all()
    columns = {name: index for index, name in enumerate(POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES)}
    assert np.allclose(features[:, columns["visible_fraction"]], [1.0, 0.875])
    assert np.allclose(features[:, columns["negative_prompt_count"]], 1.0)
    assert features[1, columns["claimed_fraction"]] == pytest.approx(0.125)
    assert features[0, columns["selection_minus_predicted_iou"]] == pytest.approx(-0.05)


def test_mask_to_logits_preserves_aspect_ratio_and_padding():
    mask = np.zeros((64, 128), dtype=bool)
    mask[16:32, 64:96] = True
    logits = mask_to_logits(mask)
    assert logits.shape == (1, 256, 256)
    assert logits.dtype == np.dtype("float32")
    # The image frame scales both axes by two and pads the lower half.
    binary = logits[0] > 0
    rows, columns = np.nonzero(binary)
    assert 28 <= rows.min() <= 36 and 60 <= rows.max() <= 68
    assert 124 <= columns.min() <= 132 and 188 <= columns.max() <= 196
    assert not binary[128:].any()
    # Logits are symmetric and finite, so the prompt encoder sees a proper probability.
    assert np.isfinite(logits).all()
    assert np.isclose(logits.max(), -logits.min())


def _make_refinement_generator(segmentation, records, matches):
    """A generator wired for `_reprompt_instances`, with no model behind it."""
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._predictor = types.SimpleNamespace(device="cpu", mask_threshold=0.0)
    segmenter._prediction = np.zeros((4, *segmentation.shape), dtype="float32")
    segmenter._last_generation_stats = {}
    segmenter._context = {"proposals": records, "records": records, "matches": matches}
    return segmenter


def test_replace_policy_repaints_from_the_second_round_and_restores_empty_masks():
    segmentation = _two_instance_segmentation()
    records = [
        {"predicted_iou": 0.9, "stability_score": 1.0, "point": (6.0, 6.0)},
        {"predicted_iou": 0.8, "stability_score": 1.0, "point": (24.0, 6.0)},
    ]
    segmenter = _make_refinement_generator(segmentation, records, {1: 0, 2: 1})

    grown = np.zeros_like(segmentation, dtype=bool)
    grown[2:14, 2:14] = True
    empty = np.zeros_like(segmentation, dtype=bool)
    predictions = iter([[(grown, 0.5), (empty, 0.99)]])
    segmenter._predict_refinement_batch = lambda *args, **kwargs: next(predictions)

    refined = segmenter._reprompt_instances(
        segmentation, segmenter._context, ("boxes",),
        _parse_refinement(
            "boxes", {"policy": "replace", "min_consistency": None, "max_foreign_overlap": None},
        )[1], batch_size=8,
    )
    # Instance 1 is repainted from its (lower-scoring) second-round mask, instance 2 is restored.
    assert (refined == 1).sum() == grown.sum()
    assert np.array_equal(refined == 2, segmentation == 2)
    assert segmenter._last_generation_stats["refined_instances"] == 2
    assert segmenter._last_generation_stats["replaced_instances"] == 1


def test_keep_if_better_policy_keeps_the_first_round_unless_the_score_improves():
    segmentation = _two_instance_segmentation()
    records = [
        {"predicted_iou": 0.9, "stability_score": 1.0, "point": (6.0, 6.0)},
        {"predicted_iou": 0.8, "stability_score": 1.0, "point": (24.0, 6.0)},
    ]
    segmenter = _make_refinement_generator(segmentation, records, {1: 0, 2: 1})

    worse = np.zeros_like(segmentation, dtype=bool)
    worse[4:12, 4:12] = True
    worse[12:16, 4:12] = True
    better = np.zeros_like(segmentation, dtype=bool)
    better[4:14, 20:28] = True
    predictions = iter([[(worse, 0.5), (better, 0.95)]])
    segmenter._predict_refinement_batch = lambda *args, **kwargs: next(predictions)

    refined = segmenter._reprompt_instances(
        segmentation, segmenter._context, ("boxes",),
        _parse_refinement(
            "boxes", {"policy": "keep-if-better", "min_consistency": None, "max_foreign_overlap": None},
        )[1], batch_size=8,
    )
    # 0.5 < 0.9 keeps the first round; 0.95 > 0.8 takes the second.
    assert np.array_equal(refined == 1, segmentation == 1)
    assert (refined == 2).sum() == better.sum()
    assert segmenter._last_generation_stats["replaced_instances"] == 1


def test_higher_scoring_instances_win_contested_pixels():
    segmentation = _two_instance_segmentation()
    records = [
        {"predicted_iou": 0.6, "stability_score": 1.0, "point": (6.0, 6.0)},
        {"predicted_iou": 0.7, "stability_score": 1.0, "point": (24.0, 6.0)},
    ]
    segmenter = _make_refinement_generator(segmentation, records, {1: 0, 2: 1})

    left = np.zeros_like(segmentation, dtype=bool)
    left[4:12, 4:18] = True
    right = np.zeros_like(segmentation, dtype=bool)
    right[4:12, 14:28] = True  # contests columns 14-17
    predictions = iter([[(left, 0.5), (right, 0.9)]])
    segmenter._predict_refinement_batch = lambda *args, **kwargs: next(predictions)

    refined = segmenter._reprompt_instances(
        segmentation, segmenter._context, ("boxes",),
        _parse_refinement(
            "boxes", {"policy": "replace", "min_consistency": None, "max_foreign_overlap": None},
        )[1], batch_size=8,
    )
    assert (refined[4:12, 14:18] == 2).all()


class _RecordingPredictor:
    """Captures the prompts of a refinement batch and answers with fixed logits."""

    device = "cpu"
    mask_threshold = 0.0

    def __init__(self, shape):
        self.shape = shape
        self.calls = []

    def _prep_prompts(self, points, labels, boxes, mask_logits, normalize):
        self.calls.append({"points": points, "labels": labels, "boxes": boxes, "mask_logits": mask_logits})
        coords = None if points is None else torch.as_tensor(points)
        point_labels = None if labels is None else torch.as_tensor(labels)
        box = None if boxes is None else torch.as_tensor(boxes)
        masks = None if mask_logits is None else torch.as_tensor(mask_logits)
        return masks, coords, point_labels, box

    def _predict(self, coords, labels, boxes, mask_input, multimask_output, return_logits):
        n = len(coords) if coords is not None else len(boxes)
        logits = torch.full((n, 1, *self.shape), -10.0)
        logits[:, :, 4:12, 4:12] = 10.0
        return logits, torch.full((n, 1), 0.9), None


def test_refinement_batches_pad_points_with_the_ignore_label(monkeypatch):
    segmentation = _two_instance_segmentation()
    records = [
        {"predicted_iou": 0.9, "stability_score": 1.0, "point": (6.0, 6.0)},
        {"predicted_iou": 0.8, "stability_score": 1.0, "point": (24.0, 6.0)},
    ]
    segmenter = _make_refinement_generator(segmentation, records, {1: 0, 2: 1})
    predictor = _RecordingPredictor(segmentation.shape)
    segmenter._predictor = predictor

    # Instance 1 has two positives and one negative, instance 2 a single positive: padded to 3.
    point_prompts = {
        1: {"points": np.array([[6, 6], [10, 10], [24, 6]], dtype="float32"),
            "point_labels": np.array([1, 1, 0], dtype="int32")},
        2: {"points": np.array([[24, 6]], dtype="float32"), "point_labels": np.array([1], dtype="int32")},
    }
    batch = [(1, (slice(4, 12), slice(4, 12))), (2, (slice(4, 12), slice(20, 28)))]
    kwargs = _parse_refinement("points+boxes+masks", {"box_extension": 2})[1]
    predictions = segmenter._predict_refinement_batch(
        segmentation, batch, ("points", "boxes", "masks"), point_prompts, kwargs,
    )

    assert len(predictions) == 2
    call = predictor.calls[0]
    assert call["points"].shape == (2, 3, 2)
    assert call["labels"].tolist() == [[1, 1, 0], [1, -1, -1]]
    assert np.array_equal(call["points"][1, 1:], np.zeros((2, 2), dtype="float32"))
    # The boxes are XYXY, grown by the extension and clipped to the image.
    assert call["boxes"].tolist() == [[2.0, 2.0, 14.0, 14.0], [18.0, 2.0, 30.0, 14.0]]
    assert call["mask_logits"].shape == (2, 1, 256, 256)


def test_select_without_refinement_matches_the_plain_merge():
    shape = (32, 32)
    first = np.zeros(shape, dtype=bool)
    first[4:12, 4:12] = True
    second = np.zeros(shape, dtype=bool)
    second[4:12, 20:28] = True
    weak = np.zeros(shape, dtype=bool)
    weak[20:28, 4:12] = True
    proposals = [
        {"segmentation": first, "predicted_iou": 0.9, "stability_score": 1.0, "point": (6.0, 6.0)},
        {"segmentation": second, "predicted_iou": 0.8, "stability_score": 1.0, "point": (24.0, 6.0)},
        {"segmentation": weak, "predicted_iou": 0.3, "stability_score": 1.0, "point": (6.0, 24.0)},
    ]
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._prediction = np.zeros((4, *shape), dtype="float32")
    segmenter._last_generation_stats = {}

    segmentation = segmenter.select(proposals, score_threshold=0.6, max_overlap=0.15, min_size=1)
    expected = merge_by_score(proposals[:2], shape, max_overlap=0.15, min_size=1)
    assert np.array_equal(segmentation, expected)
    # No refinement, so nothing needs the model and no statistics are recorded.
    assert segmenter._last_generation_stats == {}


def test_select_can_filter_by_learned_score_or_skip_the_initial_filter():
    shape = (24, 24)
    learned = np.zeros(shape, dtype=bool)
    learned[2:8, 2:8] = True
    raw = np.zeros(shape, dtype=bool)
    raw[14:20, 14:20] = True
    proposals = [
        {"segmentation": learned, "predicted_iou": 0.5, "selection_score": 0.9,
         "stability_score": 1.0, "merge_score": 0.9},
        {"segmentation": raw, "predicted_iou": 0.9, "selection_score": 0.4,
         "stability_score": 1.0, "merge_score": 0.4},
    ]
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._prediction = np.zeros((4, *shape), dtype="float32")
    segmenter._last_generation_stats = {}

    predicted = segmenter.select(proposals, score_threshold=0.6, score_filter="predicted_iou", min_size=1)
    selected = segmenter.select(proposals, score_threshold=0.6, score_filter="selection_score", min_size=1)
    unfiltered = segmenter.select(proposals, score_filter="none", min_size=1)

    assert predicted[16, 16] != 0 and predicted[4, 4] == 0
    assert selected[4, 4] != 0 and selected[16, 16] == 0
    assert unfiltered[4, 4] != 0 and unfiltered[16, 16] != 0
    with pytest.raises(ValueError, match="Invalid score filter"):
        segmenter.select(proposals, score_filter="utility", min_size=1)


def test_select_validates_the_refinement_before_touching_the_model():
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._prediction = np.zeros((4, 16, 16), dtype="float32")
    with pytest.raises(ValueError, match="refinement mode"):
        segmenter.select([], refinement="blobs")
    with pytest.raises(ValueError, match="refinement_kwargs"):
        segmenter.select([], refinement="points", refinement_kwargs={"unknown": 1})


def test_select_with_point_refinement_reprompts_each_instance(monkeypatch):
    shape = (32, 32)
    first = np.zeros(shape, dtype=bool)
    first[4:12, 4:12] = True
    second = np.zeros(shape, dtype=bool)
    second[4:12, 20:28] = True
    proposals = [
        {"segmentation": first, "predicted_iou": 0.9, "stability_score": 1.0, "point": (6.0, 6.0)},
        {"segmentation": second, "predicted_iou": 0.8, "stability_score": 1.0, "point": (24.0, 6.0)},
    ]
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._prediction = np.zeros((4, *shape), dtype="float32")
    segmenter._predictor = types.SimpleNamespace(device="cpu", mask_threshold=0.0)
    segmenter._last_generation_stats = {}

    seen = {}

    def predict_refinement_batch(segmentation, batch, components, point_prompts, refinement_kwargs):
        seen["components"] = components
        seen["prompts"] = point_prompts
        return [(segmentation == instance_id, 0.95) for instance_id, _ in batch]

    segmenter._predict_refinement_batch = predict_refinement_batch

    segmentation = segmenter.select(
        proposals, score_threshold=0.6, max_overlap=0.15, min_size=1,
        refinement="points", refinement_kwargs={"n_negatives": 1},
    )
    assert seen["components"] == ("points",)
    # Every instance re-prompts with its own positive and the other instance's prompt as negative.
    assert seen["prompts"][1]["point_labels"].tolist() == [1, 0]
    assert seen["prompts"][2]["point_labels"].tolist() == [1, 0]
    assert set(np.unique(segmentation)) == {0, 1, 2}
    assert segmenter._last_generation_stats["refined_instances"] == 2
    assert segmenter._last_generation_stats["merge_reasons"] == {"kept": 2}


def test_consistency_gate_keeps_the_first_round_when_the_masks_disagree():
    segmentation = _two_instance_segmentation()
    records = [
        {"predicted_iou": 0.9, "stability_score": 1.0, "point": (6.0, 6.0)},
        {"predicted_iou": 0.8, "stability_score": 1.0, "point": (24.0, 6.0)},
    ]
    segmenter = _make_refinement_generator(segmentation, records, {1: 0, 2: 1})

    # Instance 1's re-prompt lands somewhere else entirely; instance 2's only polishes the boundary.
    elsewhere = np.zeros_like(segmentation, dtype=bool)
    elsewhere[20:28, 4:12] = True
    polished = np.zeros_like(segmentation, dtype=bool)
    polished[4:12, 20:27] = True
    predictions = iter([[(elsewhere, 0.99), (polished, 0.99)]])
    segmenter._predict_refinement_batch = lambda *args, **kwargs: next(predictions)

    refined = segmenter._reprompt_instances(
        segmentation, segmenter._context, ("boxes",),
        _parse_refinement("boxes", {"min_consistency": 0.7, "max_foreign_overlap": None})[1], batch_size=8,
    )
    assert np.array_equal(refined == 1, segmentation == 1)  # gated, first round kept
    assert (refined == 2).sum() == polished.sum()  # consistent, second round adopted
    assert segmenter._last_generation_stats["gated_consistency"] == 1
    assert segmenter._last_generation_stats["replaced_instances"] == 1


def test_foreign_overlap_gate_rejects_growth_into_neighbours():
    segmentation = _two_instance_segmentation()
    records = [
        {"predicted_iou": 0.9, "stability_score": 1.0, "point": (6.0, 6.0)},
        {"predicted_iou": 0.8, "stability_score": 1.0, "point": (24.0, 6.0)},
    ]
    segmenter = _make_refinement_generator(segmentation, records, {1: 0, 2: 1})

    # Instance 1's re-prompt swallows instance 2; instance 2's stays inside itself.
    swallowing = np.zeros_like(segmentation, dtype=bool)
    swallowing[4:12, 4:28] = True
    inside = np.zeros_like(segmentation, dtype=bool)
    inside[5:11, 21:27] = True
    predictions = iter([[(swallowing, 0.99), (inside, 0.99)]])
    segmenter._predict_refinement_batch = lambda *args, **kwargs: next(predictions)

    refined = segmenter._reprompt_instances(
        segmentation, segmenter._context, ("boxes",),
        _parse_refinement("boxes", {"max_foreign_overlap": 0.1, "min_consistency": None})[1], batch_size=8,
    )
    assert np.array_equal(refined == 1, segmentation == 1)
    assert (refined == 2).sum() == inside.sum()
    assert segmenter._last_generation_stats["gated_foreign"] == 1


def test_interior_negative_source_uses_neighbour_interior_points():
    segmentation = _two_instance_segmentation()
    points = np.array([[6, 6], [24, 6]], dtype="float32")
    surviving = {1: (6.0, 6.0), 2: (24.0, 6.0)}

    prompts = derive_refinement_prompts(
        segmentation, points, surviving, n_positives=1, n_negatives=1, negative_source="interior",
    )
    expected = interior_points(segmentation)[:, ::-1].astype("float32")  # per instance, XY
    negatives_1 = prompts[1]["points"][prompts[1]["point_labels"] == 0]
    negatives_2 = prompts[2]["points"][prompts[2]["point_labels"] == 0]
    assert np.array_equal(negatives_1[0], expected[1])  # instance 2's interior point
    assert np.array_equal(negatives_2[0], expected[0])  # instance 1's interior point


def test_min_negative_distance_excludes_candidates_near_the_own_mask():
    # Instance 2 borders instance 1, so its prompt sits two pixels from instance 1's mask;
    # instance 3 and its prompt sit far away.
    segmentation = np.zeros((32, 32), dtype="uint32")
    segmentation[4:12, 4:12] = 1
    segmentation[4:12, 13:20] = 2
    segmentation[24:30, 4:12] = 3
    points = np.array([[6, 6], [13, 6], [6, 26]], dtype="float32")
    surviving = {1: (6.0, 6.0), 2: (13.0, 6.0), 3: (6.0, 26.0)}

    near = derive_refinement_prompts(segmentation, points, surviving, n_positives=1, n_negatives=1)
    assert near[1]["points"][near[1]["point_labels"] == 0].tolist() == [[13.0, 6.0]]

    filtered = derive_refinement_prompts(
        segmentation, points, surviving, n_positives=1, n_negatives=1, min_negative_distance=5.0,
    )
    # The nearby prompt is excluded, so the far one is selected instead.
    assert filtered[1]["points"][filtered[1]["point_labels"] == 0].tolist() == [[6.0, 26.0]]


def test_refinement_prompts_report_the_grouped_supply():
    segmentation = _two_instance_segmentation()
    points = np.array([[6, 6], [10, 6], [6, 10], [24, 6]], dtype="float32")
    prompts = derive_refinement_prompts(
        segmentation, points, {1: (6.0, 6.0), 2: (24.0, 6.0)}, n_positives=2, n_negatives=0,
    )
    # The supply counts all grouped prompts beyond the anchor, before any subsampling.
    assert prompts[1]["n_grouped"] == 2
    assert prompts[2]["n_grouped"] == 0


def test_merge_by_score_reports_reasons_without_changing_the_result():
    shape = (16, 16)
    high = np.zeros(shape, dtype=bool)
    high[2:10, 2:10] = True
    overlapping = np.zeros(shape, dtype=bool)
    overlapping[6:14, 6:14] = True  # 4x4 of its 8x8 pixels are claimed by the better mask
    tiny = np.zeros(shape, dtype=bool)
    tiny[15, 15] = True
    records = [
        {"segmentation": overlapping, "predicted_iou": 0.5, "stability_score": 0.5},
        {"segmentation": high, "predicted_iou": 0.9, "stability_score": 0.9},
        {"segmentation": tiny, "predicted_iou": 0.8, "stability_score": 0.8},
    ]

    plain = merge_by_score(records, shape, max_overlap=0.1, min_size=4)
    segmentation, reasons = merge_by_score(
        records, shape, max_overlap=0.1, min_size=4, return_reasons=True,
    )
    assert np.array_equal(plain, segmentation)
    assert reasons == ["duplicate", "kept", "too small"]


def test_parse_refinement_covers_the_new_components_and_couplings():
    components, resolved = _parse_refinement("points+boxes+masks", {"box_extension": 4})
    assert components == ("points", "boxes", "masks")
    assert resolved["box_extension"] == 4
    with pytest.raises(ValueError, match="dense-only"):
        _parse_refinement("masks", None)
    with pytest.raises(ValueError, match="negative_source"):
        _parse_refinement("points", {"negative_source": "centroids"})
    with pytest.raises(ValueError, match="Invalid refinement mode"):
        _parse_refinement("recover", None)
    with pytest.raises(ValueError, match="recover_max_claimed"):
        _parse_refinement("points+boxes", {"recover_max_claimed": 0.4})
    with pytest.raises(ValueError, match="min_grouped_for_points"):
        _parse_refinement("points", {"min_grouped_for_points": 2})


def _tile_record(box, point, predicted_iou=0.9, stability_score=1.0):
    """A record whose mask fills 'box', a (y_slice, x_slice) in the frame it was predicted in."""
    shape = tuple(side.stop - side.start for side in box)
    return {
        "segmentation": np.ones(shape, dtype=bool), "bounding_box": box,
        "predicted_iou": predicted_iou, "stability_score": stability_score, "point": point,
    }


def _tile_proposal(segmenter, tile_id, records):
    """One tile's proposal, as `_apply` returns it."""
    return {
        "tile_id": tile_id, "bounding_box": segmenter._tile_bounding_box(tile_id), "records": records,
    }


class _BlockPredictor:
    """Answers every prompt with a block around its anchor, at the shape of the region that is set.

    The anchor is the prompt's box centre, or its first positive point when there is no box, so a
    prompt translated into the wrong frame comes back as a visibly displaced mask.
    """

    mask_threshold = 0.0

    def __init__(self, shape, device="cpu"):
        self.shape = shape
        self.device = torch.device(device)
        self.calls = []

    def _prep_prompts(self, points, labels, boxes, mask_logits, normalize):
        self.calls.append({"points": points, "labels": labels, "boxes": boxes, "mask_logits": mask_logits})
        coords = None if points is None else torch.as_tensor(points, device=self.device)
        point_labels = None if labels is None else torch.as_tensor(labels, device=self.device)
        box = None if boxes is None else torch.as_tensor(boxes, device=self.device)
        masks = None if mask_logits is None else torch.as_tensor(mask_logits, device=self.device)
        return masks, coords, point_labels, box

    def _anchors(self, coords, labels, boxes):
        if boxes is not None:
            return [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for x0, y0, x1, y1 in boxes.tolist()]
        anchors = []
        for row, row_labels in zip(coords.tolist(), labels.tolist()):
            positive = [point for point, label in zip(row, row_labels) if label == 1]
            anchors.append(tuple(positive[0]))
        return anchors

    def _predict(self, coords, labels, boxes, mask_input, multimask_output, return_logits):
        anchors = self._anchors(coords, labels, boxes)
        n_masks = 3 if multimask_output else 1
        logits = torch.full((len(anchors), n_masks, *self.shape), -10.0, device=self.device)
        for row, (x, y) in enumerate(anchors):
            # Two rows taller than an 8x8 first-round mask, so a replacement is visible but consistent.
            y0, y1 = max(0, int(y) - 5), min(self.shape[0], int(y) + 5)
            x0, x1 = max(0, int(x) - 4), min(self.shape[1], int(x) + 4)
            logits[row, :, y0:y1, x0:x1] = 10.0
        # Descending, so the argmax over the mask dimension is deterministic.
        scores = torch.tensor([0.9, 0.7, 0.5][:n_masks], device=self.device).repeat(len(anchors), 1)
        return logits, scores, None


def _make_plain_generator(shape, predictor):
    """A non-tiled generator wired for `select`, with no model behind it."""
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._predictor = predictor
    segmenter._prediction = np.zeros((4, *shape), dtype="float32")
    segmenter._last_generation_stats = {}
    return segmenter


def test_apply_prompts_can_eagerly_score_or_defer_multimasks():
    class AlternativeIndexScorer:
        def predict_grouped_tensor(self, features):
            return features[:, :, 8]

        def predict(self, features):
            return np.asarray(features)[:, 8]

    shape = (32, 32)
    segmenter = _make_plain_generator(shape, _BlockPredictor(shape))
    segmenter._microscopy_multimask_scorer = AlternativeIndexScorer()
    segmenter._refinement_gate_model = None
    prompts = {
        "points": np.array([[[8.0, 8.0]], [[24.0, 24.0]]], dtype="float32"),
        "point_labels": np.ones((2, 1), dtype="int32"),
    }
    foreground = np.ones(shape, dtype="float32")

    eager = segmenter._apply_prompts(
        prompts, multimasking=True, batch_size=8, multimask_scorer="microscopy",
        multimask_selection="eager", foreground=foreground,
    )
    deferred = segmenter._apply_prompts(
        prompts, multimasking=True, batch_size=8, multimask_scorer="microscopy",
        multimask_selection="deferred", foreground=foreground,
    )

    assert len(eager) == 2 and {record["multimask_index"] for record in eager} == {2}
    assert len(deferred) == 6
    assert {record["multimask_group"] for record in deferred} == {0, 1}
    assert all(record["merge_score"] == record["multimask_index"] for record in deferred)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the device transfer test.")
def test_apply_prompts_moves_cpu_selector_scores_to_decoder_device():
    class CpuScorer:
        def predict_grouped_tensor(self, features):
            assert features.device.type == "cuda"
            return features[:, :, 8].cpu()

    shape = (32, 32)
    segmenter = _make_plain_generator(shape, _BlockPredictor(shape, device="cuda"))
    segmenter._microscopy_multimask_scorer = CpuScorer()
    segmenter._refinement_gate_model = None
    prompts = {
        "points": np.array([[[8.0, 8.0]], [[24.0, 24.0]]], dtype="float32"),
        "point_labels": np.ones((2, 1), dtype="int32"),
    }

    records = segmenter._apply_prompts(
        prompts, multimasking=True, batch_size=8, multimask_scorer="microscopy",
        foreground=np.ones(shape, dtype="float32"),
    )

    assert len(records) == 2
    assert {record["multimask_index"] for record in records} == {2}


def test_apply_prompts_can_score_the_dedicated_single_mask():
    class SingletonScorer:
        def predict_grouped_tensor(self, features):
            assert features.shape[1] == 1
            return torch.full(features.shape[:2], 0.75, device=features.device)

    shape = (32, 32)
    segmenter = _make_plain_generator(shape, _BlockPredictor(shape))
    segmenter._microscopy_multimask_scorer = SingletonScorer()
    segmenter._refinement_gate_model = None
    prompts = {
        "points": np.array([[[8.0, 8.0]], [[24.0, 24.0]]], dtype="float32"),
        "point_labels": np.ones((2, 1), dtype="int32"),
    }
    records = segmenter._apply_prompts(
        prompts, multimasking=False, batch_size=8, multimask_scorer="microscopy",
        foreground=np.ones(shape, dtype="float32"), return_multimask_features=True,
    )

    assert len(records) == 2
    assert all(record["selection_score"] == pytest.approx(0.75) for record in records)
    assert all(record["merge_score"] == pytest.approx(0.75) for record in records)
    assert all(record["multimask_index"] == 0 for record in records)
    assert all(record["multimask_features"].shape == (19,) for record in records)


def test_single_mask_allows_learned_scoring_but_not_deferred_selection():
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._microscopy_multimask_scorer = object()

    segmenter._validate_multimask_options(False, "microscopy", "eager", is_volume=False)
    with pytest.raises(ValueError, match="Deferred multimask selection"):
        segmenter._validate_multimask_options(False, "microscopy", "deferred", is_volume=False)


def test_uncertainty_gate_refines_only_selected_instances():
    shape = (32, 32)
    segmentation = np.zeros(shape, dtype="uint32")
    segmentation[4:12, 4:12] = 1
    segmentation[20:28, 20:28] = 2
    records = [
        _tile_record((slice(4, 12), slice(4, 12)), (8.0, 8.0), predicted_iou=0.9),
        _tile_record((slice(20, 28), slice(20, 28)), (24.0, 24.0), predicted_iou=0.8),
    ]
    records[0]["uncertainty_score"] = 0.2
    records[1]["uncertainty_score"] = 0.8
    context = {
        "proposals": records, "records": records, "matches": {1: 0, 2: 1},
        "score_threshold": 0.5,
    }
    segmenter = _make_plain_generator(shape, types.SimpleNamespace(device="cpu"))
    calls = []

    def predict(segmentation_, batch, components, point_prompts, refinement_kwargs):
        calls.extend(instance_id for instance_id, _ in batch)
        return [(segmentation_ == instance_id, 0.95) for instance_id, _ in batch]

    segmenter._predict_refinement_batch = predict
    _, kwargs = _parse_refinement(
        "boxes", {"gate": "uncertainty", "gate_threshold": 0.5, "min_consistency": None,
                  "max_foreign_overlap": None},
    )
    refined = segmenter._reprompt_instances(segmentation, context, ("boxes",), kwargs, batch_size=8)

    assert calls == [2]
    assert np.array_equal(refined, segmentation)
    assert segmenter._last_generation_stats["refinement_eligible_instances"] == 2
    assert segmenter._last_generation_stats["uncertainty_selected_instances"] == 1
    assert segmenter._last_generation_stats["refined_instances"] == 1


def test_postmerge_uncertainty_gate_scores_after_prompt_assembly():
    shape = (32, 32)
    segmentation = _two_instance_segmentation()
    records = [
        _tile_record((slice(4, 12), slice(4, 12)), (8.0, 8.0), predicted_iou=0.9),
        _tile_record((slice(4, 12), slice(20, 28)), (24.0, 8.0), predicted_iou=0.8),
    ]
    context = {
        "proposals": records, "records": records, "matches": {1: 0, 2: 1},
        "score_threshold": 0.6, "score_filter": "predicted_iou",
    }
    segmenter = _make_plain_generator(shape, types.SimpleNamespace(device="cpu"))

    class Gate:
        gate_stage = "postmerge"

        def predict_tensor(self, features):
            assert features.shape == (2, len(POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES))
            # Signed utility: only the second instance is predicted to benefit.
            return torch.tensor([-0.1, 0.2])

    segmenter._refinement_gate_model = Gate()
    calls = []

    def predict(segmentation_, batch, components, point_prompts, refinement_kwargs):
        calls.extend(instance_id for instance_id, _ in batch)
        return [(segmentation_ == instance_id, 0.95) for instance_id, _ in batch]

    segmenter._predict_refinement_batch = predict
    _, kwargs = _parse_refinement(
        "points+boxes", {
            "gate": "uncertainty", "gate_threshold": 0.0,
            "min_consistency": None, "max_foreign_overlap": None,
        },
    )
    refined = segmenter._reprompt_instances(
        segmentation, context, ("points", "boxes"), kwargs, batch_size=8,
    )

    assert calls == [2]
    assert np.array_equal(refined, segmentation)
    assert records[0]["uncertainty_score"] == pytest.approx(-0.1)
    assert records[1]["uncertainty_score"] == pytest.approx(0.2)


def _equivalence_records():
    """Two 8x8 instances prompted at their centres, plus two sub-threshold prompts inside them."""
    return [
        _tile_record((slice(4, 12), slice(4, 12)), (8.0, 8.0), predicted_iou=0.9),
        _tile_record((slice(4, 12), slice(20, 28)), (24.0, 8.0), predicted_iou=0.8),
        _tile_record((slice(5, 9), slice(5, 9)), (6.0, 6.0), predicted_iou=0.3),
        _tile_record((slice(5, 9), slice(21, 25)), (22.0, 6.0), predicted_iou=0.3),
    ]


@pytest.mark.parametrize("refinement", ["boxes", "points", "points+boxes", "points+boxes+masks"])
@pytest.mark.parametrize("policy", ["replace", "keep-if-better"])
@pytest.mark.parametrize("multimasking", [False, True])
def test_single_tile_refinement_equals_the_non_tiled_result(monkeypatch, refinement, policy, multimasking):
    """One tile covering the image collapses every tiled step, so both paths must agree exactly."""
    shape = (32, 32)
    kwargs = {"policy": policy, "multimasking": multimasking}

    plain = _make_plain_generator(shape, _BlockPredictor(shape))
    plain_result = plain.select(
        _equivalence_records(), score_threshold=0.5, max_overlap=0.3, min_size=1,
        refinement=refinement, refinement_kwargs=kwargs, batch_size=8,
    )

    tiled = _make_tiled_generator(shape, shape, (0, 0), monkeypatch, predictor=_BlockPredictor(shape))
    tiled_result = tiled.select(
        [_tile_proposal(tiled, 0, _equivalence_records())], score_threshold=0.5, max_overlap=0.3,
        min_size=1, refinement=refinement, refinement_kwargs=kwargs, batch_size=8,
    )

    assert plain_result.max() > 0
    assert np.array_equal(plain_result, tiled_result)
    shared = set(plain._last_generation_stats) & set(tiled._last_generation_stats)
    assert {key: plain._last_generation_stats[key] for key in shared} \
        == {key: tiled._last_generation_stats[key] for key in shared}
    assert tiled._last_generation_stats["dropped_negatives"] == 0
    assert tiled._last_generation_stats["stitch_dropped_instances"] == 0


def _two_tile_proposals(segmenter):
    """One instance in tile 0 and one in tile 3, each prompted at its own centre."""
    return [
        _tile_proposal(segmenter, 0, [_tile_record((slice(4, 12), slice(4, 12)), (8.0, 8.0))]),
        # Tile 3's outer block starts at (24, 24), so its instance sits at (44:52, 44:52) globally.
        _tile_proposal(segmenter, 3, [_tile_record((slice(20, 28), slice(20, 28)), (48.0, 48.0))]),
    ]


def test_tiled_merge_builds_a_global_refinement_context(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch)

    proposals = _two_tile_proposals(segmenter)
    # A sub-threshold prompt inside the first instance: part of the point pool, not of the merge.
    proposals[0]["records"].append(
        _tile_record((slice(6, 10), slice(6, 10)), (7.0, 7.0), predicted_iou=0.3)
    )
    segmentation, context = segmenter._merge(
        proposals, shape, score_threshold=0.5, max_overlap=0.3, min_size=1, return_context=True,
    )

    present = set(np.unique(segmentation)) - {0}
    assert set(context["matches"]) == present and len(present) == 2
    # The ids are unique across tiles and every one of them points back at the record that made it.
    for instance_id, record_index in context["matches"].items():
        record = context["records"][record_index]
        x, y = record["point"]
        assert segmentation[int(y), int(x)] == instance_id
    assert len(context["records"]) == len(context["record_tiles"]) == 2
    assert sorted(context["record_tiles"].values()) == [0, 3]
    # 'proposals' keeps the sub-threshold prompt, which the point pool needs.
    assert len(context["proposals"]) == 3
    assert context["score_threshold"] == 0.5


def test_tiled_merge_prunes_an_instance_the_stitch_overwrote(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch)

    # Both tiles predict the same object in their shared halo: tile 0 is stitched first and keeps it.
    proposals = [
        _tile_proposal(segmenter, 0, [_tile_record((slice(4, 12), slice(28, 36)), (31.0, 8.0))]),
        _tile_proposal(segmenter, 1, [_tile_record((slice(4, 12), slice(4, 12)), (33.0, 8.0))]),
    ]
    segmentation, context = segmenter._merge(
        proposals, shape, score_threshold=0.5, max_overlap=0.3, min_size=1, return_context=True,
    )

    assert set(np.unique(segmentation)) - {0} == {1}
    assert set(context["matches"]) == {1}
    assert segmenter._last_generation_stats["stitch_dropped_instances"] == 1
    # The overwritten record is still in the context, it just no longer owns an instance.
    assert len(context["records"]) == 2


def test_tiled_merge_counts_records_not_tiles(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch)

    proposals = _two_tile_proposals(segmenter)
    proposals[0]["records"].append(
        _tile_record((slice(6, 10), slice(6, 10)), (7.0, 7.0), predicted_iou=0.3)
    )
    segmenter._merge(
        proposals, shape, score_threshold=0.5, max_overlap=0.3, min_size=1, return_context=True,
    )
    assert segmenter._last_generation_stats["proposed_candidates"] == 3
    assert segmenter._last_generation_stats["scored_candidates"] == 2
    assert segmenter._last_generation_stats["merge_reasons"] == {"kept": 2}

    # Without a refinement the merge records nothing, as the non-tiled one does not either.
    segmenter._last_generation_stats = {}
    segmenter._merge(proposals, shape, score_threshold=0.5, max_overlap=0.3, min_size=1)
    assert segmenter._last_generation_stats == {}


def _refine_two_tiles(segmenter, refinement, refinement_kwargs, proposals=None):
    """Merge two tiles and refine them, returning the refined segmentation."""
    return segmenter.select(
        _two_tile_proposals(segmenter) if proposals is None else proposals,
        score_threshold=0.5, max_overlap=0.3, min_size=1,
        refinement=refinement, refinement_kwargs=refinement_kwargs, batch_size=8,
    )


def test_tiled_refinement_visits_each_owning_tile_once_in_order(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    predictor = _BlockPredictor(shape)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch, predictor=predictor)

    _refine_two_tiles(segmenter, "boxes", {"policy": "replace"})
    # Only the two tiles that own an instance are set up, once each and in order.
    assert segmenter.visited_tiles == [0, 3]


def test_tiled_postmerge_gate_scores_the_stitched_segmentation(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(
        shape, tile_shape, halo, monkeypatch,
        predictor=types.SimpleNamespace(device="cpu", mask_threshold=0.0),
    )

    class Gate:
        gate_stage = "postmerge"
        feature_names = POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES

        def predict_tensor(self, features):
            self.features = np.asarray(features)
            return torch.tensor([-0.1, 0.2])

    gate = Gate()
    segmenter._refinement_gate_model = gate
    calls = []
    proposals = [
        _tile_proposal(
            segmenter, 0,
            [_tile_record((slice(4, 12), slice(28, 36)), (31.0, 8.0))],
        ),
        # Tile 1 starts at x=24. Its 28:40 global mask loses 28:36 to the instance tile 0
        # stitched first, leaving one third of its source area visible at 36:40.
        _tile_proposal(
            segmenter, 1,
            [_tile_record((slice(4, 12), slice(4, 16)), (38.0, 8.0), predicted_iou=0.8)],
        ),
    ]

    def predict(crop, batch, components, point_prompts, refinement_kwargs):
        calls.extend(instance_id for instance_id, _ in batch)
        return [(crop == instance_id, 0.95) for instance_id, _ in batch]

    segmenter._predict_refinement_batch = predict
    refined = _refine_two_tiles(
        segmenter, "points+boxes", {
            "gate": "uncertainty", "gate_threshold": 0.0,
            "min_consistency": None, "max_foreign_overlap": None,
        }, proposals=proposals,
    )

    columns = {name: index for index, name in enumerate(POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES)}
    assert gate.features.shape == (2, len(POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES))
    assert np.allclose(gate.features[:, columns["claimed_fraction"]], [0.0, 2.0 / 3.0])
    assert np.allclose(gate.features[:, columns["visible_fraction"]], [1.0, 1.0 / 3.0])
    assert calls == [2]
    assert set(np.unique(refined)) == {0, 1, 2}
    assert (refined[4:12, 28:36] == 1).all() and (refined[4:12, 36:40] == 2).all()


def test_tiled_refinement_prompts_are_block_local_and_boxes_clipped(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    predictor = _BlockPredictor(shape)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch, predictor=predictor)

    _refine_two_tiles(segmenter, "points+boxes", {"policy": "replace", "min_consistency": None})
    # Tile 3's instance is at (44:52, 44:52) globally and its block starts at (24, 24).
    boxes = predictor.calls[1]["boxes"]
    assert np.array_equal(boxes, np.array([[20, 20, 28, 28]], dtype="float32"))
    assert np.array_equal(predictor.calls[1]["points"][0, 0], np.array([24.0, 24.0], dtype="float32"))

    # A box extension is clipped to the block, not to the image.
    predictor.calls.clear()
    segmenter._last_generation_stats = {}
    _refine_two_tiles(segmenter, "boxes", {"policy": "replace", "min_consistency": None, "box_extension": 30})
    block_shape = tuple(box.stop - box.start for box in segmenter._tile_bounding_box(3))
    assert np.array_equal(
        predictor.calls[1]["boxes"], np.array([[0, 0, block_shape[1], block_shape[0]]], dtype="float32")
    )


def test_tiled_refinement_drops_negatives_outside_the_owning_block(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    predictor = _BlockPredictor(shape)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch, predictor=predictor)

    _refine_two_tiles(segmenter, "points", {"policy": "replace", "min_consistency": None, "n_negatives": 1})
    # Each instance's only neighbour is in the other corner of the image, well outside its block.
    for call in predictor.calls:
        assert set(np.unique(call["labels"])) == {1}
    assert segmenter._last_generation_stats["dropped_negatives"] == 2


def test_tiled_refinement_keeps_an_instance_whose_second_round_is_empty(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(
        shape, tile_shape, halo, monkeypatch, predictor=types.SimpleNamespace(device="cpu", mask_threshold=0.0)
    )

    def predict(crop, batch, components, point_prompts, refinement_kwargs):
        return [(np.zeros(crop.shape, dtype=bool), 0.99) for _ in batch]

    segmenter._predict_refinement_batch = predict
    refined = _refine_two_tiles(segmenter, "boxes", {"policy": "replace"})
    # Both instances keep their first-round mask instead of vanishing, as they do untiled.
    assert set(np.unique(refined)) - {0} == {1, 2}
    assert refined[4:12, 4:12].all() and refined[44:52, 44:52].all()
    assert segmenter._last_generation_stats["replaced_instances"] == 0


def test_tiled_refinement_applies_the_consistency_gate(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(
        shape, tile_shape, halo, monkeypatch, predictor=types.SimpleNamespace(device="cpu", mask_threshold=0.0)
    )

    def predict(crop, batch, components, point_prompts, refinement_kwargs):
        # A mask in the opposite corner of the block, which cannot be a polished first round.
        mask = np.zeros(crop.shape, dtype=bool)
        mask[-10:, -10:] = True
        return [(mask, 0.99) for _ in batch]

    segmenter._predict_refinement_batch = predict
    refined = _refine_two_tiles(segmenter, "boxes", {"policy": "replace"})
    assert np.array_equal(refined[4:12, 4:12], np.ones((8, 8), dtype="uint32"))
    assert segmenter._last_generation_stats["gated_consistency"] == 2
    assert segmenter._last_generation_stats["replaced_instances"] == 0


def test_tiled_refinement_gates_growth_into_a_neighbouring_tiles_instance(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(
        shape, tile_shape, halo, monkeypatch, predictor=types.SimpleNamespace(device="cpu", mask_threshold=0.0)
    )

    # The second instance is owned by tile 1 but lies inside tile 0's block, so tile 0's crop sees it.
    proposals = [
        _tile_proposal(segmenter, 0, [_tile_record((slice(4, 12), slice(4, 12)), (8.0, 8.0))]),
        _tile_proposal(segmenter, 1, [_tile_record((slice(4, 12), slice(9, 15)), (36.0, 8.0), 0.8)]),
    ]

    def predict(crop, batch, components, point_prompts, refinement_kwargs):
        mask = np.zeros(crop.shape, dtype=bool)
        mask[4:12, 4:39] = True  # Grows from the first instance across the second one.
        return [(mask, 0.99) for _ in batch]

    segmenter._predict_refinement_batch = predict
    # The consistency gate would veto such a grown mask first, so only the foreign gate is left on.
    refined = _refine_two_tiles(
        segmenter, "boxes", {"policy": "replace", "min_consistency": None}, proposals=proposals,
    )
    assert segmenter._last_generation_stats["gated_foreign"] == 1
    assert np.array_equal(refined[4:12, 4:12], np.ones((8, 8), dtype="uint32"))


def test_tiled_refinement_repaints_in_global_score_order(monkeypatch):
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    segmenter = _make_tiled_generator(
        shape, tile_shape, halo, monkeypatch, predictor=types.SimpleNamespace(device="cpu", mask_threshold=0.0)
    )

    proposals = [
        _tile_proposal(segmenter, 0, [_tile_record((slice(4, 12), slice(20, 30)), (25.0, 8.0))]),
        # Tile 1's block starts at column 24, so this instance is at columns 34:44 globally.
        _tile_proposal(segmenter, 1, [_tile_record((slice(4, 12), slice(10, 20)), (39.0, 8.0), 0.8)]),
    ]

    masks = iter([(slice(4, 12), slice(20, 36), 0.5), (slice(4, 12), slice(6, 20), 0.9)])

    def predict(crop, batch, components, point_prompts, refinement_kwargs):
        rows, columns, score = next(masks)
        mask = np.zeros(crop.shape, dtype=bool)
        mask[rows, columns] = True
        return [(mask, score)]

    segmenter._predict_refinement_batch = predict
    refined = _refine_two_tiles(
        segmenter, "boxes",
        {"policy": "replace", "min_consistency": None, "max_foreign_overlap": None}, proposals=proposals,
    )
    # Columns 30-35 are contested; the second instance scores higher, although its tile comes later.
    assert (refined[4:12, 30:36] == 2).all()


def test_refinement_regions_default_to_the_whole_image():
    segmenter = object.__new__(AutomaticPromptGenerator)
    assert segmenter._region_of({}, 0) is None
    assert segmenter._region_box(None) == (slice(None), slice(None))
    assert segmenter._set_region(None) is None


def test_tiled_apply_feeds_the_refinement_end_to_end(monkeypatch):
    """`_apply` to `select` with a refinement, the chain the tiled generator actually runs."""
    shape, tile_shape, halo = (64, 64), (32, 32), (8, 8)
    predictor = _BlockPredictor(shape)
    segmenter = _make_tiled_generator(shape, tile_shape, halo, monkeypatch, predictor=predictor)

    real_tile_bounding_box = segmenter._tile_bounding_box

    def apply_prompts(prompts, multimasking, batch_size):
        """One 8x8 record per prompt, centred on the (tile-local) prompt point."""
        records = []
        for x, y in prompts["points"][:, 0, :]:
            box = (slice(int(y) - 4, int(y) + 4), slice(int(x) - 4, int(x) + 4))
            records.append(_tile_record(box, (float(x), float(y))))
        return records

    segmenter._apply_prompts = apply_prompts
    points = np.array([[[10, 10]], [[50, 50]]], dtype="float32")
    proposals = segmenter._apply(
        {"points": points, "point_labels": np.ones((2, 1), dtype="int32")}, multimasking=False, batch_size=8,
    )
    segmentation = segmenter.select(
        proposals, score_threshold=0.5, max_overlap=0.3, min_size=1,
        refinement="points+boxes", refinement_kwargs={"policy": "replace"}, batch_size=8,
    )

    assert set(np.unique(segmentation)) - {0} == {1, 2}
    # Both instances are refined in their own tile, so both grow by the predictor's two extra rows.
    assert segmenter._last_generation_stats["replaced_instances"] == 2
    for instance_id, (y, x) in zip((1, 2), ((10, 10), (50, 50))):
        assert segmentation[y, x] == instance_id
        rows = np.nonzero(segmentation == instance_id)[0]
        assert rows.max() - rows.min() + 1 == 10
    # The refinement runs in the tile that produced the instance, not in a neighbouring one.
    assert segmenter.visited_tiles == [0, 3, 0, 3]
    assert real_tile_bounding_box(3)[0].start == 24


# --- volumetric refinement -------------------------------------------------------------------------


class _VolumePredictor:
    """Answers each prompt batch with the masks a test supplies, and records what it was asked."""

    device = "cpu"
    mask_threshold = 0.0

    def __init__(self, responses):
        # One (masks, scores) pair per '_predict' call, in call order.
        self.responses = list(responses)
        self.calls = []

    def _prep_prompts(self, points, labels, boxes, mask_logits, normalize):
        self.calls.append({"points": points, "labels": labels, "boxes": boxes, "mask_logits": mask_logits})
        return (
            None if mask_logits is None else torch.as_tensor(mask_logits),
            None if points is None else torch.as_tensor(points),
            None if labels is None else torch.as_tensor(labels),
            None if boxes is None else torch.as_tensor(boxes),
        )

    def _predict(self, coords, labels, boxes, mask_input, multimask_output, return_logits):
        masks, scores = self.responses.pop(0)
        # +-10 logits, so the stability score is exactly 1 and the combined score is the given one.
        logits = torch.where(torch.as_tensor(np.asarray(masks))[:, None], 10.0, -10.0)
        return logits, torch.as_tensor(scores, dtype=torch.float32)[:, None], None

    @property
    def refinement_calls(self):
        """The calls that carry a box or a mask cue, which the anchor scoring never does."""
        return [call for call in self.calls if call["boxes"] is not None or call["mask_logits"] is not None]


class _RecordingPropagator:
    """Records the conditioning pushed for each object, and answers a pass with fixed masks."""

    def __init__(self, video_segments=None):
        self.pushed = []
        self.video_segments = video_segments or {}

    def reset_tracking(self):
        self.pushed.append(("reset",))

    def reset_predictor(self):
        pass

    def add_point_prompts(self, frame_ids, points, point_labels, object_id=None, **kwargs):
        self.pushed.append((
            "points", int(frame_ids), int(object_id),
            np.asarray(points).tolist(), np.asarray(point_labels).tolist(),
        ))

    def add_box_prompts(self, frame_ids, boxes=None, object_id=None):
        self.pushed.append(("box", int(frame_ids), int(object_id), np.asarray(boxes[0]).tolist()))

    def add_prompt_set(self, frame_id, points=None, point_labels=None, box=None, object_id=1,
                       clear_old_points=True):
        self.pushed.append((
            "set", int(frame_id), int(object_id),
            None if points is None else np.asarray(points).tolist(),
            None if point_labels is None else np.asarray(point_labels).tolist(),
            None if box is None else np.asarray(box).tolist(),
        ))

    def add_mask_prompts(self, frame_ids, masks=None, object_id=None, refine=True):
        self.pushed.append(("mask", int(frame_ids), int(object_id), int(np.asarray(masks[0]).sum()), refine))

    def propagate_prompts(self, early_stop_patience=None):
        return self.video_segments


def _volume_generator(monkeypatch, shape, predictor, propagator=None):
    """An initialized volumetric generator whose slice features are handed out by the monkeypatch."""
    frames_seen = []
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation._set_image_predictor_from_3d_embeddings",
        lambda predictor, embeddings, frame: frames_seen.append(frame),
    )
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._prediction = np.zeros((4, *shape), dtype="float32")
    segmenter._predictor = predictor
    segmenter._propagator = propagator
    segmenter._volume = np.zeros(shape, dtype="uint8")
    segmenter._image_embeddings = {}
    segmenter._is_initialized = True
    segmenter._last_generation_stats = {key: 0 for key in REFINEMENT_STATS_3D}
    return segmenter, frames_seen


def _mask(shape, rows, columns):
    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    return mask


def _two_anchor_prompts():
    """Two candidates on frame 0 and one on frame 2, as `derive_volume_prompts` returns them."""
    return {
        "points": np.array([[[6, 6]], [[24, 6]], [[24, 24]]], dtype="float32"),
        "point_labels": np.ones((3, 1), dtype="int32"),
        "frames": np.array([0, 0, 2], dtype="int64"),
    }


def _score_volume(segmenter, refinement=None, refinement_kwargs=None, prompts=None, **kwargs):
    components = resolved = None
    if refinement is not None:
        components, resolved = _parse_refinement(refinement, refinement_kwargs, is_volume=True)
    return segmenter._score_candidates(
        prompts or _two_anchor_prompts(), multimasking=False, batch_size=64,
        score_threshold=kwargs.get("score_threshold", 0.6), max_overlap=kwargs.get("max_overlap", 0.15),
        components=components, refinement_kwargs=resolved,
    )


def test_parse_refinement_resolves_the_volume_surface():
    # Learned uncertainty gates are 2d-only; volumes add their propagation conditioning strategy.
    _, image = _parse_refinement("points+boxes", None)
    _, volume = _parse_refinement("points+boxes", None, is_volume=True)
    assert set(image) - set(volume) == {"gate", "gate_threshold"}
    assert set(volume) - set(image) == {"conditioning"}
    assert volume["conditioning"] == "prompts"
    # Two values were measured separately in 3d and differ from 2d; the rest are shared.
    assert (volume["n_negatives"], volume["min_consistency"]) == (4, 0.85)
    assert (image["n_negatives"], image["min_consistency"]) == (6, 0.7)
    assert {key: volume[key] for key in ("n_positives", "policy", "box_extension", "negative_source")} == {
        key: image[key] for key in ("n_positives", "policy", "box_extension", "negative_source")
    }

    with pytest.raises(ValueError, match="gate"):
        _parse_refinement("points+boxes", {"gate": "uncertainty"}, is_volume=True)
    with pytest.raises(ValueError, match="gate_threshold"):
        _parse_refinement("points+boxes", {"gate_threshold": 0.5}, is_volume=True)
    with pytest.raises(ValueError, match="Invalid conditioning"):
        _parse_refinement("points+boxes", {"conditioning": "logits"}, is_volume=True)
    with pytest.raises(ValueError, match="dense-only"):
        _parse_refinement("masks", None, is_volume=True)


def test_volume_scoring_without_refinement_carries_only_the_propagation_prompt(monkeypatch):
    # The guard on the unrefined path: a candidate is what it always was, so its propagation is too.
    shape = (32, 32)
    predictor = _VolumePredictor([
        ([_mask(shape, slice(4, 12), slice(4, 12)), _mask(shape, slice(4, 12), slice(20, 28))], [0.9, 0.8]),
        ([_mask(shape, slice(20, 28), slice(20, 28))], [0.7]),
    ])
    segmenter, frames_seen = _volume_generator(monkeypatch, (3, *shape), predictor)
    candidates = _score_volume(segmenter)

    assert frames_seen == [0, 2]
    assert [candidate["frame"] for candidate in candidates] == [0, 0, 2]
    assert all(set(candidate) == {"frame", "point", "score", "stability"} for candidate in candidates)
    # No second round means no extra forward: exactly one scoring call per anchor slice.
    assert predictor.refinement_calls == []
    assert len(predictor.calls) == 2


def test_volume_refinement_reprompts_every_candidate_on_its_anchor_slice(monkeypatch):
    shape = (32, 32)
    first = [_mask(shape, slice(4, 12), slice(4, 12)), _mask(shape, slice(4, 12), slice(20, 28))]
    # A polished boundary: one row and column wider, so the consistency gate passes.
    refined = [_mask(shape, slice(4, 13), slice(4, 13)), _mask(shape, slice(4, 13), slice(20, 29))]
    predictor = _VolumePredictor([
        (first, [0.9, 0.8]),
        (refined, [0.95, 0.85]),
        ([_mask(shape, slice(20, 28), slice(20, 28))], [0.7]),
        ([_mask(shape, slice(20, 29), slice(20, 29))], [0.75]),
    ])
    segmenter, frames_seen = _volume_generator(monkeypatch, (3, *shape), predictor)
    # The gate is pinned: this test is about the re-prompt, and the refined masks below sit at IoU
    # 0.79 of the first round, which the measured 3d default of 0.85 would veto.
    candidates = _score_volume(
        segmenter, refinement="points+boxes", refinement_kwargs={"min_consistency": 0.7},
    )

    # One pass over each anchor slice: the features are read once, then scored and refined on.
    assert frames_seen == [0, 2]
    assert [len(call["points"]) for call in predictor.refinement_calls] == [2, 1]
    assert segmenter._last_generation_stats["refined_candidates"] == 3
    assert segmenter._last_generation_stats["replaced_candidates"] == 3
    assert segmenter._last_generation_stats["gated_consistency"] == 0
    assert segmenter._last_generation_stats["gated_foreign"] == 0

    # The second round's score is what orders the 3d merge, and it goes in as one combined value.
    assert candidates[0]["score"] == pytest.approx(0.95, abs=1e-6)
    assert candidates[0]["stability"] == 1.0
    # The conditioning is the re-prompt the second round was itself conditioned on - the instance's
    # box, and its own point as the positive with the neighbour's as a negative - so the video
    # predictor's decoder rebuilds the mask from the prompt the gates accepted.
    conditioning = candidates[0]["conditioning"]
    assert conditioning["box"] == (4, 4, 12, 12)
    assert conditioning["point_labels"].tolist() == [1, 0]
    assert conditioning["points"].tolist() == [[6.0, 6.0], [24.0, 6.0]]


def test_volume_consistency_gate_keeps_the_first_round_prompt(monkeypatch):
    shape = (32, 32)
    predictor = _VolumePredictor([
        ([_mask(shape, slice(4, 12), slice(4, 12))], [0.9]),
        # Somewhere else entirely: a reshape, not a polish.
        ([_mask(shape, slice(18, 26), slice(18, 26))], [0.99]),
    ])
    segmenter, _ = _volume_generator(monkeypatch, (1, *shape), predictor)
    prompts = {
        "points": np.array([[[6, 6]]], dtype="float32"),
        "point_labels": np.ones((1, 1), dtype="int32"),
        "frames": np.array([0], dtype="int64"),
    }
    candidates = _score_volume(segmenter, refinement="points+boxes", prompts=prompts)

    assert segmenter._last_generation_stats["gated_consistency"] == 1
    assert segmenter._last_generation_stats["replaced_candidates"] == 0
    # Rejected, so it propagates from its first-round point at its first-round score.
    assert "conditioning" not in candidates[0]
    assert candidates[0]["score"] == pytest.approx(0.9, abs=1e-6)


def test_volume_foreign_overlap_gate_rejects_growth_into_a_neighbour(monkeypatch):
    shape = (32, 32)
    first = [_mask(shape, slice(4, 12), slice(4, 12)), _mask(shape, slice(4, 12), slice(20, 28))]
    # The first instance swallows the second one's territory.
    refined = [_mask(shape, slice(4, 12), slice(4, 28)), _mask(shape, slice(4, 12), slice(20, 28))]
    predictor = _VolumePredictor([(first, [0.9, 0.8]), (refined, [0.95, 0.85])])
    segmenter, _ = _volume_generator(monkeypatch, (1, *shape), predictor)
    prompts = {
        "points": np.array([[[6, 6]], [[24, 6]]], dtype="float32"),
        "point_labels": np.ones((2, 1), dtype="int32"),
        "frames": np.array([0, 0], dtype="int64"),
    }
    candidates = _score_volume(
        segmenter, refinement="points+boxes", refinement_kwargs={"min_consistency": None}, prompts=prompts,
    )

    assert segmenter._last_generation_stats["gated_foreign"] == 1
    assert "conditioning" not in candidates[0]
    assert "conditioning" in candidates[1]


def test_keep_if_better_policy_keeps_the_first_round_on_a_volume(monkeypatch):
    shape = (32, 32)
    predictor = _VolumePredictor([
        ([_mask(shape, slice(4, 12), slice(4, 12))], [0.9]),
        ([_mask(shape, slice(4, 13), slice(4, 13))], [0.5]),
    ])
    segmenter, _ = _volume_generator(monkeypatch, (1, *shape), predictor)
    prompts = {
        "points": np.array([[[6, 6]]], dtype="float32"),
        "point_labels": np.ones((1, 1), dtype="int32"),
        "frames": np.array([0], dtype="int64"),
    }
    candidates = _score_volume(
        segmenter, refinement="points+boxes", refinement_kwargs={"policy": "keep-if-better"}, prompts=prompts,
    )

    assert segmenter._last_generation_stats["replaced_candidates"] == 0
    assert "conditioning" not in candidates[0]


def test_unrefined_candidate_is_propagated_from_its_single_point():
    propagator = _RecordingPropagator()
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._propagator = propagator
    segmenter._condition_pass({"frame": 3, "point": (7.0, 2.0)}, object_id=1)
    # The propagator takes YX, and nothing else is pushed.
    assert propagator.pushed == [("points", 3, 1, [[2.0, 7.0]], [1])]


@pytest.mark.parametrize("mode, expected", [
    # A push is a decoder step on the anchor frame, so the strategy decides how many it gets: one
    # per point after the box, one for all the points after the box, or one for everything.
    ("prompts", ["box", "points"]),
    ("prompts-grouped", ["box", "set"]),
    ("prompts-joint", ["set"]),
])
def test_prompt_conditioning_pushes_what_its_strategy_asks_for(mode, expected):
    propagator = _RecordingPropagator()
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._propagator = propagator
    candidate = {
        "frame": 1, "point": (6.0, 6.0),
        "conditioning": {
            "mode": mode,
            "box": (4, 5, 12, 13),
            "points": np.array([[6, 6], [24, 6]], dtype="float32"),
            "point_labels": np.array([1, 0], dtype="int32"),
        },
    }
    segmenter._condition_pass(candidate, object_id=2)

    assert [entry[0] for entry in propagator.pushed] == expected
    # The propagator takes YX for the box and the points, whichever strategy sent them.
    box_push = next(e for e in propagator.pushed if e[0] in ("box", "set"))
    if box_push[0] == "box":
        assert box_push[3] == [5.0, 4.0, 13.0, 12.0]
    else:
        assert box_push[5] == [5.0, 4.0, 13.0, 12.0]
    point_push = next(e for e in propagator.pushed if e[0] in ("points", "set") and e[3] is not None)
    assert point_push[3] == [[6.0, 6.0], [6.0, 24.0]]


def test_mask_conditioning_hands_over_an_already_refined_mask():
    propagator = _RecordingPropagator()
    segmenter = object.__new__(AutomaticPromptGenerator)
    segmenter._propagator = propagator
    mask = _mask((32, 32), slice(4, 12), slice(4, 12))
    segmenter._condition_pass({"frame": 0, "point": (6.0, 6.0), "conditioning": {"mask": mask}}, object_id=1)

    # Refined against this slice already, so the propagator must not refine it a second time.
    assert propagator.pushed == [("mask", 0, 1, 64, False)]


def test_mask_conditioning_is_selected_by_the_kwarg(monkeypatch):
    shape = (32, 32)
    predictor = _VolumePredictor([
        ([_mask(shape, slice(4, 12), slice(4, 12))], [0.9]),
        ([_mask(shape, slice(4, 13), slice(4, 13))], [0.95]),
    ])
    segmenter, _ = _volume_generator(monkeypatch, (1, *shape), predictor)
    prompts = {
        "points": np.array([[[6, 6]]], dtype="float32"),
        "point_labels": np.ones((1, 1), dtype="int32"),
        "frames": np.array([0], dtype="int64"),
    }
    candidates = _score_volume(
        segmenter, refinement="points+boxes",
        refinement_kwargs={"conditioning": "mask", "min_consistency": 0.7}, prompts=prompts,
    )
    conditioning = candidates[0]["conditioning"]
    assert set(conditioning) == {"mask"}
    assert int(conditioning["mask"].sum()) == 81


def test_volume_generate_runs_the_refinement_end_to_end(monkeypatch):
    shape = (32, 32)
    first = [_mask(shape, slice(4, 12), slice(4, 12)), _mask(shape, slice(4, 12), slice(20, 28))]
    refined = [_mask(shape, slice(4, 13), slice(4, 13)), _mask(shape, slice(4, 13), slice(20, 29))]
    predictor = _VolumePredictor([(first, [0.9, 0.8]), (refined, [0.95, 0.85])])
    # One propagated slice per object, enough for the merge to paint something.
    video_segments = {0: {1: first[0][None], 2: first[1][None]}}
    propagator = _RecordingPropagator(video_segments)
    segmenter, _ = _volume_generator(monkeypatch, (2, *shape), predictor, propagator)
    segmenter._last_generation_stats = {}
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation.derive_volume_prompts",
        lambda *args, **kwargs: {
            "points": np.array([[[6, 6]], [[24, 6]]], dtype="float32"),
            "point_labels": np.ones((2, 1), dtype="int32"),
            "frames": np.array([0, 0], dtype="int64"),
        },
    )

    segmentation = segmenter.generate(
        refinement="points+boxes", refinement_kwargs={"min_consistency": 0.7}, min_size=1,
    )

    assert segmentation.shape == (2, *shape)
    assert sorted(np.unique(segmentation)) == [0, 1, 2]
    # Every refinement counter is reported, so a run never leaves the column absent.
    assert set(REFINEMENT_STATS_3D) <= set(segmenter._last_generation_stats)
    assert segmenter._last_generation_stats["replaced_candidates"] == 2
    # Both objects reached the propagator box-first then points, in one pass, which is what the
    # default 'prompts' strategy asks for.
    assert [entry[0] for entry in propagator.pushed] == ["reset", "box", "points", "box", "points"]


def test_volume_refinement_is_off_by_default():
    # The pipeline default stays None in both dimensions; the mode is an explicit opt-in.
    from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION
    assert DEFAULT_PROMPT_GENERATION["refinement"] is None
    assert DEFAULT_PROMPT_GENERATION["refinement_kwargs"] is None


def test_volume_refinement_survives_an_anchor_slice_that_keeps_nothing(monkeypatch):
    # Every record below 'min_size', so the slice's merge keeps nothing and there is no instance to
    # re-prompt. The refinement has to fall through rather than index an empty segmentation.
    shape = (32, 32)
    predictor = _VolumePredictor([([_mask(shape, slice(4, 6), slice(4, 6))], [0.9])])
    segmenter, _ = _volume_generator(monkeypatch, (1, *shape), predictor)
    prompts = {
        "points": np.array([[[5, 5]]], dtype="float32"),
        "point_labels": np.ones((1, 1), dtype="int32"),
        "frames": np.array([0], dtype="int64"),
    }
    candidates = _score_volume(
        segmenter, refinement="points+boxes",
        refinement_kwargs={"negative_source": "interior"}, prompts=prompts,
    )
    assert candidates == []
    assert segmenter._last_generation_stats["refined_candidates"] == 0


def test_refined_conditioning_reaches_the_real_propagator(monkeypatch):
    """The refinement and the propagator, wired together rather than each against a stub.

    Both sides were unit tested in isolation and a shape neither of them exercised - one candidate on
    a frame, so one positive and no negatives - still broke: the (y, x) to (x, y) reversal left a
    negative stride on the size-1 axis, which torch refuses. So this runs the real
    'PromptableSegmentation3D.add_prompt_set' against a predictor that makes torch's own check.
    """
    from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D

    class TensorPredictor:
        """Converts what it is handed, which is the check that matters here."""

        def __init__(self):
            self.calls = []

        def add_new_points_or_box(self, inference_state, frame_idx, obj_id, clear_old_points=False,
                                  points=None, labels=None, box=None):
            if points is not None:
                torch.tensor(points, dtype=torch.float32)
            if labels is not None:
                torch.tensor(labels, dtype=torch.int32)
            if box is not None:
                torch.tensor(box, dtype=torch.float32)
            self.calls.append(obj_id)

    shape = (32, 32)
    # Two frames: the first has two candidates, the second only one - the case that broke.
    predictor = _VolumePredictor([
        ([_mask(shape, slice(4, 12), slice(4, 12)), _mask(shape, slice(4, 12), slice(20, 28))], [0.9, 0.8]),
        ([_mask(shape, slice(4, 13), slice(4, 13)), _mask(shape, slice(4, 13), slice(20, 29))], [0.95, 0.85]),
        ([_mask(shape, slice(20, 28), slice(20, 28))], [0.7]),
        ([_mask(shape, slice(20, 29), slice(20, 29))], [0.75]),
    ])
    segmenter, _ = _volume_generator(monkeypatch, (3, *shape), predictor)
    candidates = _score_volume(
        segmenter, refinement="points+boxes", refinement_kwargs={"min_consistency": 0.7},
    )

    # One candidate alone on its frame gets a single positive and no negatives.
    single = [c for c in candidates if c["frame"] == 2]
    assert len(single) == 1
    assert single[0]["conditioning"]["points"].shape == (1, 2)

    propagator = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    propagator.predictor = TensorPredictor()
    propagator.volume = np.zeros((3, *shape), dtype="uint8")
    propagator.inference_state = {}
    propagator._pushed_points, propagator._pushed_boxes, propagator._pushed_masks = {}, {}, {}
    propagator._prompt_history = []
    propagator._prompt_signatures = set()
    segmenter._propagator = propagator

    for object_id, candidate in enumerate(candidates, start=1):
        segmenter._condition_pass(candidate, object_id)
    # Every object reached the predictor, and every array it was handed converted - which is the
    # check: the default strategy pushes the box and then each point, so the single-candidate object
    # sends the one-point array that used to carry a negative stride.
    assert sorted(set(propagator.predictor.calls)) == [1, 2, 3]


@pytest.mark.parametrize("refinement", [
    "points", "boxes", "points+boxes", "points+boxes+masks",
])
@pytest.mark.parametrize("conditioning", ["prompts", "prompts-grouped", "prompts-joint", "mask"])
def test_every_conditioning_a_mode_produces_is_pushable(monkeypatch, refinement, conditioning):
    """The producer/consumer contract, over the whole cross product.

    A reversed one-point array previously slipped through tests of the two halves and torch refused
    it. This pushes every candidate of every mode through the real propagator and lets torch check it.
    """
    from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D

    class TensorPredictor:
        # 'add_mask_prompts' resizes into the predictor's frame before it pushes.
        image_size = 512

        def __init__(self):
            self.pushed = 0

        def add_new_points_or_box(self, inference_state, frame_idx, obj_id, clear_old_points=False,
                                  points=None, labels=None, box=None):
            for array, dtype in ((points, torch.float32), (labels, torch.int32), (box, torch.float32)):
                if array is not None:
                    torch.tensor(array, dtype=dtype)
            self.pushed += 1

        def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
            self.pushed += 1

    shape = (32, 32)
    kept = _mask(shape, slice(4, 20), slice(4, 20))
    duplicate = _mask(shape, slice(4, 20), slice(16, 32))
    single = _mask(shape, slice(20, 28), slice(20, 28))
    # Two candidates on frame 0 and one alone on frame 2: the single-point case. There are enough
    # responses for every mode's scoring and re-prompting.
    predictor = _VolumePredictor([
        ([kept, duplicate], [0.9, 0.8]), ([kept, duplicate], [0.95, 0.85]), ([duplicate], [0.8]),
        ([single], [0.7]), ([single], [0.75]), ([single], [0.7]),
    ])
    segmenter, _ = _volume_generator(monkeypatch, (3, *shape), predictor)
    prompts = {
        "points": np.array([[[8, 8]], [[24, 8]], [[24, 24]]], dtype="float32"),
        "point_labels": np.ones((3, 1), dtype="int32"),
        "frames": np.array([0, 0, 2], dtype="int64"),
    }
    candidates = _score_volume(
        segmenter, refinement=refinement, refinement_kwargs={"conditioning": conditioning},
        prompts=prompts,
    )

    propagator = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    propagator.predictor = TensorPredictor()
    propagator.volume = np.zeros((3, *shape), dtype="uint8")
    propagator.inference_state = {}
    propagator._pushed_points, propagator._pushed_boxes, propagator._pushed_masks = {}, {}, {}
    propagator._prompt_history = []
    propagator._prompt_signatures = set()
    segmenter._propagator = propagator

    assert candidates, "the fixture should produce at least one candidate for every mode"
    for object_id, candidate in enumerate(candidates, start=1):
        segmenter._condition_pass(candidate, object_id)
    assert propagator.predictor.pushed >= len(candidates)
