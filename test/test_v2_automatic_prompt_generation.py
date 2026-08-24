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
    interior_points,
)


def test_apg_declares_no_postprocessing_mode():
    # The front end reads this to decide whether 'generate' takes the AIS 'mode' argument.
    assert AutomaticPromptGenerator._has_postprocessing_mode is False
    assert TiledAutomaticPromptGenerator._has_postprocessing_mode is False
    assert getattr(UniSAM2InstanceSegmentation, "_has_postprocessing_mode", True) is True


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


def _make_tiled_generator(shape, tile_shape, halo, monkeypatch):
    """A tiled generator with the tiling set up, but no model, predictor or embeddings."""
    segmenter = object.__new__(TiledAutomaticPromptGenerator)
    segmenter._tiling = Blocking([0, 0], list(shape), list(tile_shape))
    segmenter._halo = list(halo)
    segmenter._i = None
    segmenter._predictor = object()
    segmenter._image_embeddings = object()
    segmenter._prediction = np.zeros((4, *shape), dtype="float32")
    monkeypatch.setattr("micro_sam.v2.automatic_prompt_generation.set_precomputed", lambda *a, **k: None)
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


def test_tiled_generator_sets_a_video_embedding_slice(monkeypatch):
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
    segmenter._set_tile_embeddings(tile_id=0)

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


def test_tiled_generator_restores_a_volumetric_state(monkeypatch):
    class Features:
        attrs = {"shape": (64, 64), "tile_shape": (32, 32), "halo": (8, 8)}

    created = []
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation.TiledPromptableSegmentation3D",
        lambda *args, **kwargs: created.append((args, kwargs)) or "propagator",
    )
    prediction = np.ones((4, 3, 64, 64), dtype="float32")
    image_embeddings = {"features": Features(), "input_size": None}
    volume = np.zeros((3, 64, 64), dtype="uint8")
    video_predictor = object()
    segmenter = object.__new__(TiledAutomaticPromptGenerator)
    segmenter._prediction = None
    segmenter._is_initialized = False
    segmenter._video_predictor = video_predictor
    segmenter._image_embeddings = None
    segmenter._volume = None
    segmenter._propagator = None
    segmenter._offload_to_cpu = True
    segmenter._max_cached_frames = 3
    segmenter._inference_device = "cuda:1"
    segmenter._i = 7
    segmenter._owns_image_embeddings = False
    segmenter._tiling = None
    segmenter._halo = None

    segmenter.set_state({
        "prediction": prediction,
        "image_embeddings": image_embeddings,
        "volume": volume,
    })

    assert segmenter._prediction is prediction
    assert segmenter._image_embeddings is image_embeddings
    assert segmenter._volume is volume
    assert segmenter._propagator == "propagator"
    assert segmenter._i is None
    assert created == [(
        (video_predictor, volume, image_embeddings),
        {"devices": "cuda:1", "offload_state_to_cpu": True, "max_cached_frames": 3},
    )]


@pytest.mark.parametrize("devices,expected", [(None, "cuda:1"), (["cuda:2"], ["cuda:2"])])
def test_tiled_generator_forwards_devices_to_volume_propagation(monkeypatch, devices, expected):
    class Features:
        attrs = {"shape": (3, 32, 32), "tile_shape": (16, 16), "halo": (4, 4)}

    decoder_calls = []
    propagator_calls = []
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation.TiledUniSAM2InstanceSegmentation.initialize",
        lambda *args, **kwargs: decoder_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "micro_sam.v2.automatic_prompt_generation.TiledPromptableSegmentation3D",
        lambda *args, **kwargs: propagator_calls.append(kwargs) or "propagator",
    )
    segmenter = object.__new__(TiledAutomaticPromptGenerator)
    segmenter._video_predictor = object()
    segmenter._inference_device = "cuda:1"
    segmenter._image_embeddings = None
    segmenter._volume = None
    segmenter._propagator = None
    segmenter._temporary_embedding_path = None
    segmenter._offload_to_cpu = None
    segmenter._max_cached_frames = None
    segmenter._tiling = None
    segmenter._halo = None
    image_embeddings = {"features": Features()}
    volume = np.zeros((3, 32, 32), dtype="uint8")

    segmenter.initialize(volume, ndim=3, image_embeddings=image_embeddings, devices=devices)

    assert decoder_calls[0]["devices"] is devices
    assert propagator_calls[0]["devices"] == expected


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


def test_multicut_preserves_a_mask_that_crosses_from_its_anchor_tile_halo(monkeypatch):
    segmenter = _make_tiled_generator((4, 8), (4, 4), (0, 2), monkeypatch)
    mask = np.zeros((1, 4, 6), dtype=bool)
    mask[:, 1:3, 2:6] = True
    proposals = [{
        "tile_id": 0,
        "bounding_box": (slice(0, 1), slice(0, 4), slice(0, 6)),
        "records": [{
            "segmentation": mask,
            "predicted_iou": 0.9,
            "stability_score": 0.9,
        }],
    }]

    segmentation = segmenter._merge_via_multicut(
        proposals, (1, 4, 8), score_threshold=0.5, max_overlap=0.5, min_size=1,
    )

    # The prompt belongs to tile 0 (x < 4), but its propagated mask reaches two pixels into tile 1.
    assert np.all(segmentation[:, 1:3, 2:6] == 1)
    assert np.all(segmentation[:, :, 6:] == 0)


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
            records.append({"segmentation": mask, "predicted_iou": 0.9, "stability_score": 0.9})
        return records

    segmenter._tile_bounding_box = tile_bounding_box
    segmenter._apply_prompts = apply_prompts

    # Two prompts in different tiles, in XY.
    points = np.array([[[10, 10]], [[50, 50]]], dtype="float32")
    prompts = {"points": points, "point_labels": np.ones((2, 1), dtype="int32")}
    proposals = segmenter._apply(prompts, multimasking=False, batch_size=8)
    segmentation = segmenter._merge(proposals, shape, score_threshold=0.0, max_overlap=0.3, min_size=1)

    assert segmentation.shape == shape
    assert segmentation.dtype == np.dtype("uint32")
    # Two instances, each sitting at its prompt in the full image's frame rather than at a tile offset.
    assert len(np.unique(segmentation)) == 3
    assert segmentation[10, 10] != 0
    assert segmentation[50, 50] != 0
    assert segmentation[10, 10] != segmentation[50, 50]
    assert tile_ids == sorted(set(tile_ids))  # every tile with prompts is visited once, in order


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
