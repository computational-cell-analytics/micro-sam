from contextlib import nullcontext
from pathlib import Path
import types
import inspect

import numpy as np
import torch
import pytest

from micro_sam.v2.instance_segmentation import (
    _block_shape_and_halo, _set_image_predictor_from_backbone, _decode_3d_feature_batch,
    _get_decoder_autocast, UniSAM2InstanceSegmentation, TiledUniSAM2InstanceSegmentation,
    TiledAutomaticMaskGenerationSegmenter, amg_3d_segmentation,
    get_instance_segmentation_generator, get_decoder,
)
from micro_sam.v2.postprocessing import DEFAULT_POSTPROCESSING, run_multicut
from micro_sam.v2.util import DEFAULT_TILE_Z, DEFAULT_HALO_Z, ImageEmbeddings


def _run_decoder_3d(model, image_embeddings, device="cpu"):
    """Run the UniSAM2 decoder on 3d embeddings through the class (the only inference entry point)."""
    return UniSAM2InstanceSegmentation(model, device=device)._run_decoder_3d(image_embeddings)


def _run_decoder_2d(model, image_embeddings, device="cpu"):
    """Run the UniSAM2 decoder on 2d embeddings through the class (the only inference entry point)."""
    return UniSAM2InstanceSegmentation(model, device=device)._run_decoder_2d(image_embeddings)


def test_run_multicut_uses_dense_defaults():
    signature = inspect.signature(run_multicut)
    for name, value in DEFAULT_POSTPROCESSING["dense"].items():
        assert signature.parameters[name].default == value


class _FakeSAM2Model:
    """Mimics SAM2's `_prepare_backbone_features` (flatten + permute, no encoder), for model-free tests."""

    num_feature_levels = 3
    directly_add_no_mem_embed = False
    no_mem_embed = None

    def parameters(self):
        yield torch.zeros(1)  # so `next(model.parameters()).device` works

    def _prepare_backbone_features(self, backbone_out):
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        pos = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in pos]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos = [x.flatten(2).permute(2, 0, 1) for x in pos]
        return backbone_out, vision_feats, vision_pos, feat_sizes


class _FakePredictor:
    def __init__(self):
        self.model = _FakeSAM2Model()
        self._bb_feat_sizes = [(8, 8), (4, 4), (2, 2)]  # high -> low resolution
        self._features = None
        self._orig_hw = None
        self._is_image_set = False


class _FakeEmbeddingFile:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _make_tiled_amg_segmenter():
    segmenter = object.__new__(TiledAutomaticMaskGenerationSegmenter)
    segmenter._mask_generator = types.SimpleNamespace(predictor=object())

    def no_masks(shape):
        return []

    segmenter._generate_masks_for_shape = no_masks
    segmenter._is_initialized = False
    return segmenter


def _mock_tiled_amg_embeddings(monkeypatch, tmp_path, fail_precompute=False):
    temporary_paths = []
    embedding_files = []
    save_paths = []

    def make_temp_path():
        path = tmp_path / f"implicit-{len(temporary_paths)}.zarr"
        temporary_paths.append(path)
        return str(path)

    def fake_precompute(predictor, image, **kwargs):
        path = Path(kwargs["save_path"])
        path.mkdir(parents=True, exist_ok=True)
        save_paths.append(path)
        if fail_precompute:
            raise RuntimeError("encoder failed")
        embedding_file = _FakeEmbeddingFile()
        embedding_files.append(embedding_file)
        features = types.SimpleNamespace(
            attrs={"tile_shape": kwargs["tile_shape"], "halo": kwargs["halo"], "shape": image.shape[:2]},
            file=embedding_file,
        )
        return ImageEmbeddings({"features": features}, store=embedding_file)

    monkeypatch.setattr("micro_sam.v2.instance_segmentation.make_temp_embedding_path", make_temp_path)
    monkeypatch.setattr("micro_sam.v2.instance_segmentation.precompute_image_embeddings", fake_precompute)
    monkeypatch.setattr("micro_sam.v2.instance_segmentation.set_precomputed", lambda *args, **kwargs: None)
    return temporary_paths, embedding_files, save_paths


def test_tiled_amg_removes_implicit_embedding_store(monkeypatch, tmp_path):
    segmenter = _make_tiled_amg_segmenter()
    temporary_paths, embedding_files, save_paths = _mock_tiled_amg_embeddings(monkeypatch, tmp_path)

    segmenter.initialize(
        np.zeros((4, 4), dtype="uint8"),
        tile_shape=(4, 4), halo=(0, 0), verbose=False,
    )

    assert len(temporary_paths) == 1
    assert save_paths == temporary_paths
    assert not temporary_paths[0].exists()
    assert embedding_files[0].closed
    assert segmenter._is_initialized


def test_tiled_amg_removes_implicit_store_on_mask_error(monkeypatch, tmp_path):
    segmenter = _make_tiled_amg_segmenter()
    temporary_paths, embedding_files, _ = _mock_tiled_amg_embeddings(monkeypatch, tmp_path)

    def boom(shape):
        raise RuntimeError("mask generation failed")

    segmenter._generate_masks_for_shape = boom
    with pytest.raises(RuntimeError, match="mask generation failed"):
        segmenter.initialize(
            np.zeros((4, 4), dtype="uint8"),
            tile_shape=(4, 4), halo=(0, 0), verbose=False,
        )

    assert not temporary_paths[0].exists()
    assert embedding_files[0].closed
    assert not segmenter._is_initialized


def test_tiled_amg_removes_implicit_store_on_encoder_error(monkeypatch, tmp_path):
    segmenter = _make_tiled_amg_segmenter()
    temporary_paths, embedding_files, _ = _mock_tiled_amg_embeddings(
        monkeypatch, tmp_path, fail_precompute=True,
    )

    with pytest.raises(RuntimeError, match="encoder failed"):
        segmenter.initialize(
            np.zeros((4, 4), dtype="uint8"), tile_shape=(4, 4), halo=(0, 0), verbose=False,
        )

    assert not temporary_paths[0].exists()
    assert embedding_files == []
    assert not segmenter._is_initialized


def test_tiled_amg_keeps_user_embedding_store(monkeypatch, tmp_path):
    segmenter = _make_tiled_amg_segmenter()
    temporary_paths, embedding_files, save_paths = _mock_tiled_amg_embeddings(monkeypatch, tmp_path)
    user_path = tmp_path / "user-embeddings.zarr"

    segmenter.initialize(
        np.zeros((4, 4), dtype="uint8"),
        save_path=str(user_path), tile_shape=(4, 4), halo=(0, 0), verbose=False,
    )

    assert temporary_paths == []
    assert save_paths == [user_path]
    assert user_path.exists()
    assert embedding_files[0].closed


def test_tiled_3d_amg_does_not_accumulate_implicit_stores(monkeypatch, tmp_path):
    segmenter = _make_tiled_amg_segmenter()
    temporary_paths, embedding_files, _ = _mock_tiled_amg_embeddings(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "micro_sam.v2.instance_segmentation.merge_instance_segmentation_3d",
        lambda segmentation, **kwargs: segmentation,
    )

    result = amg_3d_segmentation(
        np.zeros((3, 4, 4), dtype="uint8"), segmenter,
        tile_shape=(4, 4), halo=(0, 0), verbose=False,
    )

    assert result.shape == (3, 4, 4)
    assert len(temporary_paths) == 3
    assert all(not path.exists() for path in temporary_paths)
    assert all(embedding_file.closed for embedding_file in embedding_files)


def test_set_image_predictor_from_backbone_reconstructs_features():
    # The reconstruction must pick slice i's per-level features, in the right order, reshaped back to
    # (1, C, H, W). The lowest-res level is stored as 'features', 'pos_enc' once for the volume.
    z, c = 4, 3
    sizes = [(8, 8), (4, 4), (2, 2)]
    fpn = [np.random.rand(z, 1, c, h, w).astype("float32") for (h, w) in sizes[:-1]]
    features = np.random.rand(z, 1, c, *sizes[-1]).astype("float32")
    pos_enc = [np.zeros((1, 1, c, h, w), dtype="float32") for (h, w) in sizes]

    predictor = _FakePredictor()
    _set_image_predictor_from_backbone(predictor, fpn, pos_enc, features, original_size=(64, 64), i=2)

    assert predictor._is_image_set is True
    assert predictor._orig_hw == [(64, 64)]
    image_embed = predictor._features["image_embed"]
    high_res = predictor._features["high_res_feats"]
    assert tuple(image_embed.shape) == (1, c, 2, 2)
    assert [tuple(f.shape) for f in high_res] == [(1, c, 8, 8), (1, c, 4, 4)]
    # The flatten/permute/reshape round-trips, so the features equal slice i of the input levels.
    assert np.allclose(image_embed.numpy(), features[2])
    assert np.allclose(high_res[0].numpy(), fpn[0][2])
    assert np.allclose(high_res[1].numpy(), fpn[1][2])


class _FakeUNETR:
    """A stand-in for UNETR3D that mimics its per-slice encoder loop, for model-free tests.

    `UniSAM2InstanceSegmentation._run_decoder_3d` swaps in a stub encoder and calls `model(dummy)`. This fake
    reproduces UNETR3D.forward's ``[self.encoder(x[:, :, i])[0] for i in range(Z)]`` loop, records the
    per-slice features the stub returned, and emits a (1, 4, Z, H, W) prediction of the dummy size.
    A plain (non-nn.Module) class so the encoder attribute can be freely swapped and restored.
    """

    def __init__(self, img_size=8, output_dtype=torch.float32):
        self.encoder = types.SimpleNamespace(img_size=img_size)
        self.seen = []       # features the stub returned on the most recent call
        self.output_dtype = output_dtype
        self.call_z = []     # z size of each decoder call (one per z block)
        self.call_hw = []

    def __call__(self, x):  # x: (1, 3, Z, H, W) dummy
        z = x.shape[2]
        self.seen = [self.encoder(x[:, :, i])[0] for i in range(z)]
        self.call_z.append(z)
        h, w = x.shape[-2:]
        self.call_hw.append((h, w))
        return torch.zeros((1, 4, z, h, w), dtype=self.output_dtype)


def test_decoder_3d_stub_returns_per_slice_features_in_order():
    # The stub must hand the decoder each slice's feature, in order, as a (1, C, h, w) tensor.
    # z <= DEFAULT_TILE_Z so it is a single decoder pass.
    z, c, h, w = DEFAULT_TILE_Z, 2, 4, 4
    feats = np.arange(z * c * h * w, dtype="float32").reshape(z, c, h, w)
    model = _FakeUNETR(img_size=8)
    out = _run_decoder_3d(model, {"features": feats, "original_size": (8, 8)})
    assert out.shape == (4, z, 8, 8)
    assert model.call_z == [z]  # single pass
    assert len(model.seen) == z
    for i, feat in enumerate(model.seen):
        assert tuple(feat.shape) == (1, c, h, w)
        assert np.allclose(feat[0].cpu().numpy(), feats[i])


def test_decoder_3d_squeezes_5d_features():
    # Regression: tiled / save-path features are (Z, 1, C, h, w); the singleton batch axis must be
    # squeezed so the decoder still gets (1, C, h, w) per slice (previously raised 'got 5').
    z, c, h, w = 3, 2, 4, 4
    feats5 = np.arange(z * 1 * c * h * w, dtype="float32").reshape(z, 1, c, h, w)
    model = _FakeUNETR(img_size=8)
    out = _run_decoder_3d(model, {"features": feats5, "original_size": (8, 8)})
    assert out.shape == (4, z, 8, 8)
    for i, feat in enumerate(model.seen):
        assert tuple(feat.shape) == (1, c, h, w)
        assert np.allclose(feat[0].cpu().numpy(), feats5[i, 0])


def test_decoder_2d_squeezes_5d_features():
    # Regression: a single slice from save-path 3d embeddings is (1, 1, C, h, w); the singleton batch
    # axis must be squeezed so the 2d decoder gets (1, C, h, w). This is the auto-tracking per-frame
    # and interactive single-slice path, which crashed ('got 5') when embeddings came from a cache.
    c, h, w = 2, 4, 4
    feats5 = np.arange(1 * 1 * c * h * w, dtype="float32").reshape(1, 1, c, h, w)
    model = _FakeUNETR(img_size=8)
    out = _run_decoder_2d(model, {"features": feats5, "original_size": (8, 8)})
    assert out.shape == (4, 8, 8)
    assert len(model.seen) == 1
    assert tuple(model.seen[0].shape) == (1, c, h, w)
    assert np.allclose(model.seen[0][0].cpu().numpy(), feats5[0, 0])


def test_decoder_2d_uses_original_non_square_shape():
    features = np.zeros((1, 2, 4, 4), dtype="float32")
    model = _FakeUNETR(img_size=8)
    out = _run_decoder_2d(model, {"features": features, "original_size": (4, 8)})

    assert out.shape == (4, 4, 8)
    assert model.call_hw == [(4, 8)]


def test_decoder_predictions_are_cached_as_float32():
    features = np.zeros((1, 2, 4, 4), dtype="float32")
    model = _FakeUNETR(img_size=8, output_dtype=torch.float16)
    out = _run_decoder_2d(model, {"features": features, "original_size": (8, 8)})

    assert out.dtype == np.float32


@pytest.mark.parametrize("device_type", ("cuda", "mps"))
def test_decoder_autocast_uses_fp16_on_accelerators(device_type, monkeypatch):
    calls = []

    def fake_autocast(device_type, dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    with _get_decoder_autocast(torch.device(device_type)):
        pass

    assert calls == [(device_type, torch.float16)]


def test_decoder_autocast_leaves_cpu_unchanged(monkeypatch):
    def fail_autocast(*args, **kwargs):
        pytest.fail("CPU decoder inference must not enable autocast.")

    monkeypatch.setattr(torch, "autocast", fail_autocast)
    with _get_decoder_autocast(torch.device("cpu")):
        pass


def test_full_inference_uses_decoder_autocast(monkeypatch):
    autocast_devices = []

    def fake_autocast(device):
        autocast_devices.append(device)
        return nullcontext()

    def fake_predict_with_halo_pipelined(**kwargs):
        prediction_function = kwargs["prediction_function"]
        prediction_function(lambda inputs: inputs, torch.zeros(1))
        return kwargs["output"]

    monkeypatch.setattr(
        "micro_sam.v2.instance_segmentation._get_decoder_autocast",
        fake_autocast,
    )
    monkeypatch.setattr(
        "torch_em.util.prediction.predict_with_halo_pipelined",
        fake_predict_with_halo_pipelined,
    )

    model = _FakeUNETR(img_size=8)
    segmenter = UniSAM2InstanceSegmentation(model, device="cpu")
    output = segmenter._run_full_inference(
        np.zeros((8, 8), dtype="float32"),
        ndim=2,
        batch_size=1,
        devices="cpu",
    )

    assert output.shape == (4, 8, 8)
    assert autocast_devices == [torch.device("cpu")]


def test_full_inference_normalizes_each_volume_slice_independently(monkeypatch):
    from micro_sam.v2.normalization import normalize_raw

    crop = np.array(
        [[[[0.0, 1.0], [2.0, 3.0]], [[100.0, 110.0], [120.0, 130.0]]]],
        dtype="float32",
    )
    preprocessed = {}

    def fake_predict_with_halo_pipelined(**kwargs):
        preprocessed["crop"] = kwargs["preprocess"](crop)
        return kwargs["output"]

    monkeypatch.setattr(
        "torch_em.util.prediction.predict_with_halo_pipelined",
        fake_predict_with_halo_pipelined,
    )

    model = _FakeUNETR(img_size=8)
    segmenter = UniSAM2InstanceSegmentation(model, device="cpu")
    output = segmenter._run_full_inference(
        np.zeros((2, 2, 2), dtype="float32"),
        ndim=3,
        batch_size=1,
        devices="cpu",
    )

    expected = np.concatenate([normalize_raw(crop, axis=(-2, -1))] * 3, axis=0)
    assert output.shape == (4, 2, 2, 2)
    assert np.allclose(preprocessed["crop"], expected)
    assert np.allclose(preprocessed["crop"].min(axis=(-2, -1)), 0.0)
    assert np.allclose(preprocessed["crop"].max(axis=(-2, -1)), 1.0)


def _stage_3d_ais(
    monkeypatch, embedding_path, initialize=None, calls=None, segmenter=None, device=None, devices=None,
):
    """Drive `automatic_instance_segmentation` for 3d AIS with fakes, capturing the temp-store calls."""
    from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation

    calls = {"removed": []} if calls is None else calls
    embedding_model = object()
    embedding_file = _FakeEmbeddingFile()
    embeddings = ImageEmbeddings({
        "features": types.SimpleNamespace(file=embedding_file),
        "input_size": 8,
        "original_size": (8, 8),
    }, store=embedding_file)
    calls["embedding_file"] = embedding_file
    temp_path = "ephemeral-embeddings.zarr"

    def fake_precompute(predictor, raw, **kwargs):
        calls["precompute"] = (predictor, raw, kwargs)
        return embeddings

    monkeypatch.setattr("micro_sam.v2.util.precompute_image_embeddings", fake_precompute)
    monkeypatch.setattr("micro_sam.util.make_temp_embedding_path", lambda: temp_path)
    monkeypatch.setattr("shutil.rmtree", lambda path, **kwargs: calls["removed"].append(path))

    segmenter = segmenter or UniSAM2InstanceSegmentation(_FakeUNETR(img_size=8), device="cpu")
    segmenter.initialize = initialize or (lambda raw, ndim, **kwargs: calls.__setitem__("initialize", (raw, kwargs)))
    segmenter.generate = lambda mode, **kwargs: np.ones((2, 8, 8), dtype="uint32")
    raw = np.zeros((2, 8, 8), dtype="uint8")
    result = automatic_instance_segmentation(
        predictor=types.SimpleNamespace(model=embedding_model),
        segmenter=segmenter, input_path=raw, ndim=3, embedding_path=embedding_path, verbose=False,
        device=device, devices=devices,
    )
    return calls, embeddings, embedding_model, raw, temp_path, result


def test_automatic_3d_ais_stages_embeddings_without_save_path(monkeypatch):
    calls, embeddings, embedding_model, raw, temp_path, result = _stage_3d_ais(monkeypatch, embedding_path=None)

    predictor, precompute_raw, precompute_kwargs = calls["precompute"]
    assert predictor is embedding_model
    assert precompute_raw is raw
    # The front end owns the ephemeral store: it passes the temp path and removes it after decoding.
    assert precompute_kwargs["save_path"] == temp_path
    assert precompute_kwargs["lazy_loading"] is True
    assert calls["initialize"][1]["image_embeddings"] is embeddings
    assert calls["removed"] == [temp_path]
    assert calls["embedding_file"].closed
    assert (result == 1).all()


def test_automatic_3d_ais_keeps_user_embedding_path(monkeypatch):
    calls, _, _, _, _, _ = _stage_3d_ais(monkeypatch, embedding_path="user.zarr")
    # A caller-provided path is used as-is and never removed.
    assert calls["precompute"][2]["save_path"] == "user.zarr"
    assert calls["removed"] == []
    assert calls["embedding_file"].closed


def test_automatic_3d_ais_removes_temp_store_on_error(monkeypatch):
    calls = {"removed": []}

    def boom(raw, ndim, **kwargs):
        raise RuntimeError("decoder failed")

    with pytest.raises(RuntimeError, match="decoder failed"):
        _stage_3d_ais(monkeypatch, embedding_path=None, initialize=boom, calls=calls)
    # The temp store is removed even when decoding raises.
    assert calls["removed"] == ["ephemeral-embeddings.zarr"]
    assert calls["embedding_file"].closed


def test_inference_device_intent():
    # The 'devices=None' fallback: default pins to the model device. The front end's auto (None)
    # intent fans out (None); an explicit device/list is used as given.
    pin = UniSAM2InstanceSegmentation(_FakeUNETR(img_size=8), device="cuda:1")
    assert pin._inference_devices(None) == "cuda:1"
    fan_out = UniSAM2InstanceSegmentation(_FakeUNETR(img_size=8), device="cuda:1", inference_device=None)
    assert fan_out._inference_devices(None) is None
    explicit = UniSAM2InstanceSegmentation(_FakeUNETR(img_size=8), device="cuda:0", inference_device="cuda:1")
    assert explicit._inference_devices(None) == "cuda:1"
    assert fan_out._inference_devices(["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]


@pytest.mark.parametrize(
    "configured_device,device,devices,expected",
    [
        (None, None, None, None),
        ("cuda:1", None, None, "cuda:1"),
        (None, "cuda:1", None, "cuda:1"),
        ("cuda:0", "cuda:0", ["cuda:1"], ["cuda:1"]),
    ],
)
def test_automatic_ais_uses_same_effective_devices_for_encoder_and_decoder(
    monkeypatch, configured_device, device, devices, expected,
):
    segmenter = UniSAM2InstanceSegmentation(
        _FakeUNETR(img_size=8), device="cuda:0", inference_device=configured_device,
    )
    calls, _, _, _, _, _ = _stage_3d_ais(
        monkeypatch,
        embedding_path=None,
        segmenter=segmenter,
        device=device,
        devices=devices,
    )

    assert calls["precompute"][2]["devices"] == expected
    assert calls["initialize"][1]["devices"] == expected


def test_decoder_3d_zchunks_deep_volume():
    # Regression for the z-tiling fix: a deep volume (small in-plane, not tiled in-plane) must be
    # decoded in bounded z blocks, not all at once, so peak memory stays bounded.
    z, c, h, w = 10, 2, 4, 4

    feats = np.zeros((z, c, h, w), dtype="float32")
    model = _FakeUNETR(img_size=8)
    out = _run_decoder_3d(model, {"features": feats, "original_size": (8, 8)})
    assert out.shape == (4, z, 8, 8)
    assert len(model.call_z) > 1  # chunked along z, not a single whole-stack pass
    # Every decoder call stays within one z block plus the halo on each side.
    assert max(model.call_z) <= DEFAULT_TILE_Z + 2 * DEFAULT_HALO_Z
    # The inner z blocks fully tile the stack (ceil(10 / 4) = 3 blocks).
    assert len(model.call_z) == int(np.ceil(z / DEFAULT_TILE_Z))


def test_block_shape_3d_no_tiling_chunks_z():
    # Deep volume, no in-plane tiling -> chunk along z with the default z block + halo, full in-plane.
    # Regression: this used to be the whole volume as one block, which blew up memory.
    block, halo = _block_shape_and_halo((100, 512, 512), ndim=3, tile_shape=None, halo=None)
    assert block == (DEFAULT_TILE_Z, 512, 512)
    assert halo == (DEFAULT_HALO_Z, 0, 0)


def test_block_shape_3d_no_tiling_shallow_volume():
    # Fewer slices than the default z block -> single z block, no z halo.
    block, halo = _block_shape_and_halo((3, 256, 256), ndim=3, tile_shape=None, halo=None)
    assert block == (3, 256, 256)
    assert halo == (0, 0, 0)


def test_block_shape_2d_no_tiling_single_block():
    block, halo = _block_shape_and_halo((256, 256), ndim=2, tile_shape=None, halo=None)
    assert block == (1, 256, 256)
    assert halo == (0, 0, 0)


def test_block_shape_3d_tiling_uses_tile():
    block, halo = _block_shape_and_halo((50, 1024, 1024), ndim=3, tile_shape=(4, 512, 512), halo=(2, 64, 64))
    assert block == (4, 512, 512)
    assert halo == (2, 64, 64)


def test_block_shape_3d_in_plane_tiling_keeps_default_z_chunking():
    # The CLI and the annotator only ever pass an in-plane (y, x) tile. It must be combined with the
    # default z block instead of being used as a (z, y, x) block shape.
    block, halo = _block_shape_and_halo((50, 1024, 1024), ndim=3, tile_shape=(512, 512), halo=(64, 64))
    assert block == (DEFAULT_TILE_Z, 512, 512)
    assert halo == (DEFAULT_HALO_Z, 64, 64)


def test_block_shape_3d_in_plane_tiling_shallow_volume():
    # Fewer slices than the default z block -> single z block, no z halo.
    block, halo = _block_shape_and_halo((3, 1024, 1024), ndim=3, tile_shape=(512, 512), halo=(64, 64))
    assert block == (3, 512, 512)
    assert halo == (0, 64, 64)


def test_block_shape_3d_in_plane_tiling_without_halo():
    block, halo = _block_shape_and_halo((50, 1024, 1024), ndim=3, tile_shape=(512, 512), halo=None)
    assert block == (DEFAULT_TILE_Z, 512, 512)
    assert halo == (DEFAULT_HALO_Z, 0, 0)


@pytest.mark.parametrize("tile_shape,halo", [((512, 512), (64, 64)), ((4, 512, 512), (2, 64, 64))])
def test_block_shape_3d_matches_predict_with_halo_arity(tile_shape, halo):
    # predict_with_halo asserts len(block_shape) == len(halo) == ndim.
    block, block_halo = _block_shape_and_halo((50, 1024, 1024), ndim=3, tile_shape=tile_shape, halo=halo)
    assert len(block) == len(block_halo) == 3


def test_block_shape_2d_tiling_uses_tile():
    block, halo = _block_shape_and_halo((1024, 1024), ndim=2, tile_shape=(512, 512), halo=(64, 64))
    assert block == (1, 512, 512)
    assert halo == (0, 64, 64)


def test_factory_dispatch_ais():
    # A decoder + 'ais' resolves to the (tiled) decoder-based segmenter. Construction is model-free.
    decoder = object()
    seg = get_instance_segmentation_generator(decoder=decoder, segmentation_mode="ais")
    assert isinstance(seg, UniSAM2InstanceSegmentation)
    assert not isinstance(seg, TiledUniSAM2InstanceSegmentation)
    seg_tiled = get_instance_segmentation_generator(decoder=decoder, is_tiled=True, segmentation_mode="ais")
    assert isinstance(seg_tiled, TiledUniSAM2InstanceSegmentation)


def test_factory_defaults_to_ais_with_decoder():
    # With a decoder and no explicit mode, the factory defaults to AIS.
    assert isinstance(get_instance_segmentation_generator(decoder=object()), UniSAM2InstanceSegmentation)


def test_factory_amg_dispatch(monkeypatch):
    # 'amg' (and the no-decoder default) route to get_amg_segmenter with the SAM2 model. Stub it so
    # the dispatch is tested without building a real SAM2 mask generator.
    sentinel = object()
    calls = {}

    def fake_get_amg(model, is_tiled=False, **kwargs):
        calls["model"], calls["is_tiled"] = model, is_tiled
        return sentinel

    monkeypatch.setattr("micro_sam.v2.instance_segmentation.get_amg_segmenter", fake_get_amg)
    model = object()
    assert get_instance_segmentation_generator(model=model, is_tiled=True, segmentation_mode="amg") is sentinel
    assert calls == {"model": model, "is_tiled": True}
    # No decoder and no mode also defaults to AMG.
    assert get_instance_segmentation_generator(model=model) is sentinel


def test_factory_invalid_and_missing_args():
    with pytest.raises(ValueError):
        get_instance_segmentation_generator(decoder=object(), segmentation_mode="bogus")
    with pytest.raises(ValueError):  # 'ais' requires a decoder
        get_instance_segmentation_generator(segmentation_mode="ais")
    with pytest.raises(ValueError):  # 'amg' requires a model
        get_instance_segmentation_generator(segmentation_mode="amg")


def test_get_decoder_requires_a_source():
    # A base backbone with neither a checkpoint nor a registered decoder cannot resolve a decoder.
    with pytest.raises(ValueError):
        get_decoder("hvit_t")


class _AttrArray(np.ndarray):
    """A numpy array carrying a zarr-like `.attrs` dict, to mimic a per-tile embedding dataset."""

    def __new__(cls, data, attrs):
        obj = np.asarray(data).view(cls)
        obj.attrs = attrs
        return obj


class _FakeFeatsGroup:
    """Mimic a zarr features group: per-tile datasets keyed by str(tile_id), plus tiling attrs."""

    def __init__(self, tiles, attrs):
        self._tiles = tiles
        self.attrs = attrs

    def __getitem__(self, key):
        return self._tiles[key]


def test_tiled_decoder_2d_stitches_all_tiles():
    # A (8, 8) image tiled into four (4, 4) tiles (no halo): the tiled AIS decoder must decode each
    # tile through the class (_run_decoder_2d) and stitch them into a (4, 8, 8) prediction.
    c, h, w = 2, 2, 2
    tile_shape, shape = (4, 4), (8, 8)
    tiles = {
        str(tid): _AttrArray(np.zeros((1, c, h, w), dtype="float32"), {"original_size": tile_shape})
        for tid in range(4)
    }
    feats_group = _FakeFeatsGroup(tiles, {"shape": shape, "tile_shape": tile_shape, "halo": (0, 0)})

    model = _FakeUNETR(img_size=8)
    segmenter = TiledUniSAM2InstanceSegmentation(model, device="cpu")
    out = segmenter._run_decoder_tiled_2d({"features": feats_group})
    assert out.shape == (4, *shape)
    assert len(model.call_z) == 4  # one decoder pass per tile


@pytest.mark.parametrize("devices,expected", [(None, "cuda:1"), ("cpu", "cpu")])
def test_configured_device_is_used_when_no_devices_are_given(devices, expected, monkeypatch):
    # Regression: the batched decoder paths only forwarded 'devices', so the device the segmenter was
    # constructed with was silently replaced by the model's device (or by all visible GPUs).
    import micro_sam.v2.batched_inference as batched_inference

    forwarded = {}

    def fake_decode(model, image_embeddings, **kwargs):
        forwarded["devices"] = kwargs["devices"]
        return np.zeros((4, 1, 8, 8), dtype="float32")

    monkeypatch.setattr(batched_inference, "_decode_volume_embeddings", fake_decode)
    segmenter = UniSAM2InstanceSegmentation(_FakeUNETR(img_size=8), device="cuda:1")
    segmenter._run_decoder_3d({"features": np.zeros((1, 2, 4, 4), dtype="float32")}, devices=devices)
    assert forwarded["devices"] == expected


class _RecordingOutput:
    """Stands in for the decoder output tensor and records the conversions applied to it."""

    def __init__(self, array, calls):
        self._array = array
        self._calls = calls

    def detach(self):
        self._calls.append("detach")
        return self

    def cpu(self):
        self._calls.append("cpu")
        return self

    def float(self):
        self._calls.append("float")
        return self

    def numpy(self):
        return self._array


class _RecordingModel:
    def __init__(self, output):
        self.encoder = types.SimpleNamespace(img_size=8)
        self._output = output

    def __call__(self, x):
        return self._output


def test_decoder_output_is_moved_to_cpu_before_the_float_cast():
    # Casting on the device would hold an fp32 copy of the whole output next to the fp16 one,
    # cancelling out the memory that fp16 inference saves.
    calls = []
    array = np.zeros((1, 4, 1, 8, 8), dtype="float32")
    model = _RecordingModel(_RecordingOutput(array, calls))

    out = _decode_3d_feature_batch(model, torch.zeros((1, 1, 2, 4, 4)), (8, 8), "cpu")

    assert out is array
    assert calls == ["detach", "cpu", "float"]
