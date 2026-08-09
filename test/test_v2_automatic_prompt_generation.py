import types

import numpy as np
import pytest

from bioimage_cpp.utils import Blocking

from micro_sam.v2.instance_segmentation import (
    UniSAM2InstanceSegmentation, get_instance_segmentation_generator,
)
from micro_sam.v2.automatic_prompt_generation import (
    AutomaticPromptGenerator, TiledAutomaticPromptGenerator, derive_point_prompts, merge_by_score,
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
    segmenter._predictor = object()
    segmenter._image_embeddings = object()
    segmenter._prediction = np.zeros((4, *shape), dtype="float32")
    monkeypatch.setattr("micro_sam.v2.automatic_prompt_generation.set_precomputed", lambda *a, **k: None)
    return segmenter


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


def test_tiled_apply_and_merge_maps_prompts_and_masks_between_frames(monkeypatch):
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
    segmentation = segmenter._apply_and_merge(
        prompts, shape, multimasking=False, batch_size=8, score_threshold=0.0,
        max_overlap=0.3, min_size=1,
    )

    assert segmentation.shape == shape
    assert segmentation.dtype == np.dtype("uint32")
    # Two instances, each sitting at its prompt in the full image's frame rather than at a tile offset.
    assert len(np.unique(segmentation)) == 3
    assert segmentation[10, 10] != 0
    assert segmentation[50, 50] != 0
    assert segmentation[10, 10] != segmentation[50, 50]
    assert tile_ids == sorted(set(tile_ids))  # every tile with prompts is visited once, in order


def test_tiled_generator_cannot_serialize_its_state(monkeypatch):
    segmenter = _make_tiled_generator((64, 64), (32, 32), (8, 8), monkeypatch)
    with pytest.raises(NotImplementedError):
        segmenter.get_state()
    with pytest.raises(NotImplementedError):
        segmenter.set_state({})
