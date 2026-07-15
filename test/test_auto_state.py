"""Tests for the SAM2 automatic-segmentation state caching ('auto_state').

These are model-free: the segmenters are constructed directly (bypassing the model download and
encoder pass) and populated with hand-built state, so the tests exercise `get_state`/`set_state`,
the serialization helpers and the staleness guards without loading a SAM2 checkpoint.
"""

import os

import numpy as np
import torch

from sam2.utils.amg import mask_to_rle_pytorch
from bioimage_cpp.utils import Blocking

from micro_sam.v2.instance_segmentation import (
    _LazyRLEMask, AutomaticMaskGenerationSegmenter, TiledAutomaticMaskGenerationSegmenter,
)
from micro_sam.v2.automatic_segmentation import UniSAM2InstanceSegmentation
from micro_sam.precompute_state import (
    _auto_state_path, _save_amg_state_v2, _load_amg_state_v2,
    _save_ais_state_v2, _load_ais_state_v2, _ais_state_matches, _signature_matches,
)

DEFAULT_AMG_PARAMS = {
    "points_per_side": 32, "pred_iou_thresh": 0.8, "stability_score_thresh": 0.9, "model_type": "hvit_t",
}


def _rle_mask(binary):
    """A single AMG mask dict with an uncompressed RLE 'segmentation', as SAM2's AMG produces."""
    rle = mask_to_rle_pytorch(torch.from_numpy(binary[None]).to(torch.bool))[0]
    return {"segmentation": rle, "area": int(binary.sum())}


def _make_amg_segmenter(masks, original_size, params=None):
    segmenter = object.__new__(AutomaticMaskGenerationSegmenter)
    segmenter._masks = [_LazyRLEMask(mask) for mask in masks]
    segmenter._original_size = original_size
    segmenter._amg_params = dict(params or DEFAULT_AMG_PARAMS)
    segmenter._is_initialized = True
    return segmenter


def _make_tiled_amg_segmenter(masks_per_tile, original_size, tile_shape, halo, params=None):
    segmenter = object.__new__(TiledAutomaticMaskGenerationSegmenter)
    segmenter._masks = [[_LazyRLEMask(mask) for mask in tile] for tile in masks_per_tile]
    segmenter._original_size = original_size
    segmenter._tile_shape = tile_shape
    segmenter._halo = halo
    segmenter._amg_params = dict(params or DEFAULT_AMG_PARAMS)
    segmenter._tiling = Blocking([0, 0], list(original_size), list(tile_shape))
    segmenter._is_initialized = True
    return segmenter


def test_auto_state_path():
    path, key = _auto_state_path("/emb", "amg", None)
    assert path.endswith("auto_state_amg.pickle") and key is None
    path, key = _auto_state_path("/emb", "amg", 3)
    assert path.endswith(os.path.join("auto_state_amg", "state-3.pkl")) and key is None
    path, key = _auto_state_path("/emb", "ais", None)
    assert path.endswith("auto_state_ais.h5") and key == "state"
    path, key = _auto_state_path("/emb", "ais", 2)
    assert path.endswith("auto_state_ais.h5") and key == "state-2"


def test_amg_get_set_state_roundtrip():
    m1 = np.zeros((16, 16), bool)
    m1[2:6, 2:6] = True
    m2 = np.zeros((16, 16), bool)
    m2[9:13, 9:13] = True
    segmenter = _make_amg_segmenter([_rle_mask(m1), _rle_mask(m2)], (16, 16))

    state = segmenter.get_state()
    # The masks are stored as compact RLE dicts (not decoded arrays) and the params are recorded.
    assert isinstance(state["masks"][0], dict) and isinstance(state["masks"][0]["segmentation"], dict)
    assert state["params"] == DEFAULT_AMG_PARAMS
    assert state["original_size"] == (16, 16)

    restored = object.__new__(AutomaticMaskGenerationSegmenter)
    restored.set_state(state)
    assert restored._is_initialized
    assert isinstance(restored._masks[0], _LazyRLEMask)

    out_original = segmenter.generate(min_object_size=0, with_background=False)
    out_restored = restored.generate(min_object_size=0, with_background=False)
    assert np.array_equal(out_original, out_restored)
    assert out_restored.max() == 2  # both objects survive the round-trip


def test_amg_empty_state_roundtrip():
    segmenter = _make_amg_segmenter([], (12, 12))
    restored = object.__new__(AutomaticMaskGenerationSegmenter)
    restored.set_state(segmenter.get_state())
    out = restored.generate()
    assert out.shape == (12, 12) and out.max() == 0


def test_tiled_amg_get_set_state_roundtrip():
    # Two tiles side by side: full (16, 32), tile (16, 16), no halo.
    tile1 = np.zeros((16, 16), bool)
    tile1[2:6, 2:6] = True
    tile2 = np.zeros((16, 16), bool)
    tile2[8:12, 8:12] = True
    segmenter = _make_tiled_amg_segmenter(
        [[_rle_mask(tile1)], [_rle_mask(tile2)]], (16, 32), (16, 16), (0, 0),
    )

    state = segmenter.get_state()
    assert state["tile_shape"] == (16, 16) and state["halo"] == (0, 0)

    restored = object.__new__(TiledAutomaticMaskGenerationSegmenter)
    restored.set_state(state)
    assert restored._tiling.number_of_blocks == segmenter._tiling.number_of_blocks
    assert restored._tile_shape == (16, 16) and restored._halo == (0, 0)

    out_original = segmenter.generate(min_object_size=0, with_background=False)
    out_restored = restored.generate(min_object_size=0, with_background=False)
    assert np.array_equal(out_original, out_restored)


def test_ais_get_set_state_roundtrip():
    prediction = np.random.RandomState(0).rand(4, 16, 16).astype("float32")
    segmenter = UniSAM2InstanceSegmentation(model=None)
    segmenter._prediction = prediction
    segmenter._is_initialized = True

    restored = UniSAM2InstanceSegmentation(model=None)
    restored.set_state(segmenter.get_state())
    assert restored._is_initialized
    assert np.array_equal(restored._prediction, prediction)


def test_amg_serialization_and_param_match(tmp_path):
    m = np.zeros((16, 16), bool)
    m[3:7, 3:7] = True
    segmenter = _make_amg_segmenter([_rle_mask(m)], (16, 16))

    path, _ = _auto_state_path(str(tmp_path), "amg", None)
    _save_amg_state_v2(segmenter, path)
    loaded = _load_amg_state_v2(path)

    assert loaded["params"] == segmenter._amg_params and len(loaded["masks"]) == 1
    # The param dict is what 'cache_amg_state_v2' compares to decide whether to reuse the state.
    assert loaded.get("params") == segmenter._amg_params
    changed = _make_amg_segmenter([], (16, 16), params={**DEFAULT_AMG_PARAMS, "points_per_side": 64})
    assert loaded.get("params") != changed._amg_params


def test_ais_serialization_and_staleness_guard(tmp_path):
    prediction = np.arange(4 * 4 * 4, dtype="float32").reshape(4, 4, 4)
    segmenter = UniSAM2InstanceSegmentation(model=None)
    segmenter._prediction = prediction
    segmenter._is_initialized = True

    path, key = _auto_state_path(str(tmp_path), "ais", None)
    _save_ais_state_v2(segmenter, path, key, "hvit_t_cells")
    loaded = _load_ais_state_v2(path, key)

    assert np.array_equal(loaded["prediction"], prediction)
    assert loaded["model_type"] == "hvit_t_cells"
    assert _ais_state_matches(loaded, "hvit_t_cells")          # same model -> reuse
    assert not _ais_state_matches(loaded, "hvit_t_other")      # different model -> recompute
    assert _ais_state_matches(loaded, None)                    # unknown request -> reuse
    assert _ais_state_matches({"prediction": prediction}, "hvit_t_cells")  # legacy (no signature) -> reuse


def test_signature_matches():
    assert _signature_matches("a", "a")
    assert not _signature_matches("a", "b")   # both known and different -> stale
    assert _signature_matches(None, "a")      # cached unknown (legacy) -> reuse
    assert _signature_matches("a", None)      # requested unknown -> reuse
    assert _signature_matches(None, None)


def test_amg_embedding_signature_roundtrip(tmp_path):
    m = np.zeros((16, 16), bool)
    m[3:7, 3:7] = True
    segmenter = _make_amg_segmenter([_rle_mask(m)], (16, 16))
    path, _ = _auto_state_path(str(tmp_path), "amg", None)
    _save_amg_state_v2(segmenter, path, embedding_signature="sig-A")
    loaded = _load_amg_state_v2(path)
    assert loaded["embedding_signature"] == "sig-A"
    assert _signature_matches(loaded.get("embedding_signature"), "sig-A")
    assert not _signature_matches(loaded.get("embedding_signature"), "sig-B")  # embeddings changed -> stale


def test_ais_embedding_signature_roundtrip(tmp_path):
    segmenter = UniSAM2InstanceSegmentation(model=None)
    segmenter._prediction = np.zeros((4, 4, 4), dtype="float32")
    segmenter._is_initialized = True
    path, key = _auto_state_path(str(tmp_path), "ais", None)
    _save_ais_state_v2(segmenter, path, key, "hvit_t_cells", embedding_signature="sig-A")
    loaded = _load_ais_state_v2(path, key)
    assert loaded["embedding_signature"] == "sig-A"
    assert not _signature_matches(loaded.get("embedding_signature"), "sig-B")  # embeddings changed -> stale
