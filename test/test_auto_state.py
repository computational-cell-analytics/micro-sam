"""Tests for the SAM2 automatic-segmentation state caching ('autoseg_state').

These are model-free: the segmenters are constructed directly (bypassing the model download and
encoder pass) and populated with hand-built state, so the tests exercise `get_state`/`set_state`,
the serialization helpers and the staleness guards without loading a SAM2 checkpoint.
"""

import numpy as np
import pytest
import torch
from bioimage_cpp.utils import Blocking
from sam2.utils.amg import mask_to_rle_pytorch

from micro_sam.precompute_state import (
    AUTOSEG_STATE_ATTRIBUTE,
    AUTOSEG_STATE_GROUP,
    _ais_state_matches,
    _autoseg_state_key,
    _has_autoseg_state,
    _load_ais_state_v2,
    _load_amg_state_v2,
    _load_autoseg_state,
    _save_ais_state_v2,
    _save_amg_state_v2,
    _save_autoseg_state,
    _signature_matches,
    cache_autoseg_state,
)
from micro_sam.util import _open_embeddings, _create_dataset_with_data
from micro_sam.v2.automatic_segmentation import UniSAM2InstanceSegmentation
from micro_sam.v2.instance_segmentation import (
    AutomaticMaskGenerationSegmenter,
    TiledAutomaticMaskGenerationSegmenter,
    _LazyRLEMask,
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


def test_autoseg_state_key():
    assert _autoseg_state_key(None) == "state"
    assert _autoseg_state_key(3) == "state-3"


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

    save_path = str(tmp_path / "embeddings.zarr")
    key = _autoseg_state_key(None)
    _save_amg_state_v2(segmenter, save_path, key)
    loaded = _load_amg_state_v2(save_path, key)

    assert loaded["params"] == segmenter._amg_params and len(loaded["masks"]) == 1
    # The param dict is what 'cache_autoseg_state' compares to decide whether to reuse the state.
    assert loaded.get("params") == segmenter._amg_params
    changed = _make_amg_segmenter([], (16, 16), params={**DEFAULT_AMG_PARAMS, "points_per_side": 64})
    assert loaded.get("params") != changed._amg_params


def test_ais_serialization_and_staleness_guard(tmp_path):
    prediction = np.arange(4 * 4 * 4, dtype="float32").reshape(4, 4, 4)
    segmenter = UniSAM2InstanceSegmentation(model=None)
    segmenter._prediction = prediction
    segmenter._is_initialized = True

    save_path = str(tmp_path / "embeddings.zarr")
    key = _autoseg_state_key(None)
    _save_ais_state_v2(segmenter, save_path, key, "hvit_t_cells")
    loaded = _load_ais_state_v2(save_path, key)

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
    save_path = str(tmp_path / "embeddings.zarr")
    key = _autoseg_state_key(None)
    _save_amg_state_v2(segmenter, save_path, key, embedding_signature="sig-A")
    loaded = _load_amg_state_v2(save_path, key)
    assert loaded["embedding_signature"] == "sig-A"
    assert _signature_matches(loaded.get("embedding_signature"), "sig-A")
    assert not _signature_matches(loaded.get("embedding_signature"), "sig-B")  # embeddings changed -> stale


def test_ais_embedding_signature_roundtrip(tmp_path):
    segmenter = UniSAM2InstanceSegmentation(model=None)
    segmenter._prediction = np.zeros((4, 4, 4), dtype="float32")
    segmenter._is_initialized = True
    save_path = str(tmp_path / "embeddings.zarr")
    key = _autoseg_state_key(None)
    _save_ais_state_v2(segmenter, save_path, key, "hvit_t_cells", embedding_signature="sig-A")
    loaded = _load_ais_state_v2(save_path, key)
    assert loaded["embedding_signature"] == "sig-A"
    assert not _signature_matches(loaded.get("embedding_signature"), "sig-B")  # embeddings changed -> stale


def test_states_share_embedding_zarr_and_record_metadata(tmp_path):
    save_path = str(tmp_path / "embeddings.zarr")
    amg = _make_amg_segmenter([], (16, 16))
    ais = UniSAM2InstanceSegmentation(model=None)
    ais._prediction = np.zeros((4, 16, 16), dtype="float32")
    ais._is_initialized = True

    _save_amg_state_v2(amg, save_path, "state-0")
    _save_amg_state_v2(amg, save_path, "state-1")
    _save_ais_state_v2(ais, save_path, "state", "hvit_t_cells")

    embeddings = _open_embeddings(save_path, mode="r")
    assert set(embeddings.attrs[AUTOSEG_STATE_ATTRIBUTE]) == {"amg", "ais"}
    state_root = embeddings[AUTOSEG_STATE_GROUP]
    assert state_root["amg"].attrs["state_count"] == 2
    assert state_root["ais"].attrs["state_count"] == 1
    assert "state-0" in state_root["amg"] and "state" in state_root["ais"]

    # Cache completeness is checked from group metadata, without deserializing masks or predictions.
    assert _has_autoseg_state(save_path, "amg", state_count=2)
    assert not _has_autoseg_state(save_path, "amg", state_count=3)
    assert _has_autoseg_state(save_path, "ais")


def test_state_stored_inside_embedding_zarr_representations(tmp_path):
    """The state lives in the SAME zarr container as the embeddings (no sidecar file). AMG is saved as
    a pickle bitstream (a 1-D uint8 dataset); AIS as an individual float32 array. This is the storage
    contract behind the h5 -> zarr migration."""
    save_path = str(tmp_path / "embeddings.zarr")

    # Mimic a precomputed embedding container: a 'features' dataset plus an identifying attr.
    embeddings = _open_embeddings(save_path, mode="a")
    _create_dataset_with_data(embeddings, "features", data=np.zeros((1, 256, 8, 8), dtype="float32"))
    embeddings.attrs["original_size"] = [[64, 64]]

    m = np.zeros((16, 16), bool)
    m[3:7, 3:7] = True
    _save_amg_state_v2(_make_amg_segmenter([_rle_mask(m)], (16, 16)), save_path, "state")

    ais = UniSAM2InstanceSegmentation(model=None)
    ais._prediction = np.zeros((4, 16, 16), dtype="float32")
    ais._is_initialized = True
    _save_ais_state_v2(ais, save_path, "state", "hvit_t_cells")

    reopened = _open_embeddings(save_path, mode="r")
    # One container holds the embeddings and both state modes (same filepath, no sidecar).
    assert "features" in reopened
    assert "original_size" in reopened.attrs  # embeddings stay identifiable after writing the state
    root = reopened[AUTOSEG_STATE_GROUP]

    amg_ds = root["amg"]["state"]
    assert amg_ds.dtype == np.uint8 and len(amg_ds.shape) == 1  # a pickle bitstream, not decoded arrays

    ais_ds = root["ais"]["state"]["prediction"]
    assert ais_ds.dtype == np.float32 and tuple(ais_ds.shape) == (4, 16, 16)  # an individual array
    assert ais_ds.chunks[0] == 1  # per-channel chunks so a read does not inflate one big chunk


def test_cache_autoseg_state_ais_is_on_demand(tmp_path, monkeypatch):
    """The lazy contract via the mode dispatcher: a matching cached AIS state is loaded on demand (no
    decoder rerun), and a stale one is recomputed. Mirrors how image embeddings are reused when cached."""
    prediction = np.arange(4 * 8 * 8, dtype="float32").reshape(4, 8, 8)
    seed = UniSAM2InstanceSegmentation(model=None)
    seed._prediction = prediction
    seed._is_initialized = True

    save_path = str(tmp_path / "embeddings.zarr")
    key = _autoseg_state_key(None)
    _save_ais_state_v2(seed, save_path, key, "hvit_t_cells")

    # Track whether the (expensive) decoder pass runs; it must not run on a cache hit.
    initialize_calls = []

    def fake_initialize(self, *args, **kwargs):
        initialize_calls.append(True)
        self._prediction = np.zeros((4, 8, 8), dtype="float32")
        self._is_initialized = True

    monkeypatch.setattr(UniSAM2InstanceSegmentation, "initialize", fake_initialize)
    raw = np.zeros((8, 8), dtype="uint8")

    # Cache hit: the state is loaded from disk and no decoder is run ('decoder=None' would crash the
    # real 'initialize', so a clean return with the cached prediction proves it was reused as-is).
    segmenter = cache_autoseg_state(
        "ais", None, raw, None, save_path, ndim=2, model_type="hvit_t_cells", verbose=False,
    )
    assert initialize_calls == []
    assert np.array_equal(segmenter.get_state()["prediction"], prediction)

    # Stale cache (different model): recomputed on demand instead of silently reusing the wrong state.
    cache_autoseg_state(
        "ais", None, raw, None, save_path, ndim=2, model_type="hvit_t_other", verbose=False,
    )
    assert initialize_calls == [True]


def test_amg_state_loads_per_slice_on_demand(tmp_path):
    """Each slice's AMG state is a separate on-disk key, loaded one at a time. A volume's state is
    streamed per slice (like the lazy per-slice embeddings), not materialized as one whole array."""
    m0 = np.zeros((16, 16), bool)
    m0[1:5, 1:5] = True
    m1 = np.zeros((16, 16), bool)
    m1[10:14, 10:14] = True
    _save_amg_state_v2(_make_amg_segmenter([_rle_mask(m0)], (16, 16)), str(tmp_path / "e.zarr"), "state-0")
    _save_amg_state_v2(_make_amg_segmenter([_rle_mask(m1)], (16, 16)), str(tmp_path / "e.zarr"), "state-1")
    save_path = str(tmp_path / "e.zarr")

    # Loading one slice returns only that slice's state, without touching the sibling slices.
    state0 = _load_amg_state_v2(save_path, _autoseg_state_key(0))
    restored = object.__new__(AutomaticMaskGenerationSegmenter)
    restored.set_state(state0)
    out = restored.generate(min_object_size=0, with_background=False)
    assert out[3, 3] == 1 and out[12, 12] == 0  # slice-0 object present, slice-1 object not loaded

    # A slice that was never cached loads as None (so it is computed on demand), not an error.
    assert _load_amg_state_v2(save_path, _autoseg_state_key(5)) is None


def test_cache_autoseg_state_routes_by_mode(monkeypatch):
    """The 'cache_autoseg_state' dispatcher forwards to the AMG / AIS implementation by mode, and
    rejects unknown modes."""
    import micro_sam.precompute_state as ps

    calls = []
    monkeypatch.setattr(ps, "_cache_amg_state_v2", lambda *a, **k: calls.append(("amg", a, k)) or "amg-seg")
    monkeypatch.setattr(ps, "_cache_ais_state_v2", lambda *a, **k: calls.append(("ais", a, k)) or "ais-seg")

    assert ps.cache_autoseg_state("amg", "MODEL", "RAW", None, "sp", points_per_side=8) == "amg-seg"
    assert ps.cache_autoseg_state("ais", "DECODER", "RAW", None, "sp", ndim=3) == "ais-seg"
    assert calls[0] == ("amg", ("MODEL", "RAW", None, "sp"), {"points_per_side": 8})
    assert calls[1] == ("ais", ("DECODER", "RAW", None, "sp"), {"ndim": 3})

    with pytest.raises(ValueError, match="Invalid automatic-segmentation state mode"):
        ps.cache_autoseg_state("bogus", None, None, None, None)


def test_save_load_autoseg_state_dispatch(tmp_path):
    """The save/load dispatchers route to the AMG (pickle bitstream) and AIS (array) implementations."""
    save_path = str(tmp_path / "e.zarr")
    m = np.zeros((16, 16), bool)
    m[3:7, 3:7] = True
    _save_autoseg_state("amg", _make_amg_segmenter([_rle_mask(m)], (16, 16)), save_path, "state")

    ais = UniSAM2InstanceSegmentation(model=None)
    ais._prediction = np.zeros((4, 16, 16), dtype="float32")
    ais._is_initialized = True
    _save_autoseg_state("ais", ais, save_path, "state", model_type="hvit_t_cells")

    amg_loaded = _load_autoseg_state("amg", save_path, "state")
    ais_loaded = _load_autoseg_state("ais", save_path, "state")
    assert len(amg_loaded["masks"]) == 1  # AMG masks came back
    assert tuple(ais_loaded["prediction"].shape) == (4, 16, 16)  # AIS array came back
    assert _load_autoseg_state("amg", save_path, "state-99") is None  # missing key -> None

    with pytest.raises(ValueError, match="Invalid automatic-segmentation state mode"):
        _load_autoseg_state("bogus", save_path, "state")
