import types

import numpy as np
import torch

from micro_sam.v2.automatic_segmentation import (
    _block_shape_and_halo, run_unisam2_decoder_on_3d_embeddings, run_unisam2_decoder_on_embeddings,
)
from micro_sam.v2.instance_segmentation import _set_image_predictor_from_backbone
from micro_sam.v2.util import DEFAULT_TILE_Z, DEFAULT_HALO_Z


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


def test_set_image_predictor_from_backbone_reconstructs_features():
    # The reconstruction must pick slice i's per-level features, in the right order, reshaped back to
    # (1, C, H, W): image_embed = lowest-res level, high_res_feats = the higher-res levels.
    z, c = 4, 3
    sizes = [(8, 8), (4, 4), (2, 2)]
    fpn = [np.random.rand(z, 1, c, h, w).astype("float32") for (h, w) in sizes]
    pos_enc = [np.zeros((z, 1, c, h, w), dtype="float32") for (h, w) in sizes]

    predictor = _FakePredictor()
    _set_image_predictor_from_backbone(predictor, fpn, pos_enc, original_size=(64, 64), i=2)

    assert predictor._is_image_set is True
    assert predictor._orig_hw == [(64, 64)]
    image_embed = predictor._features["image_embed"]
    high_res = predictor._features["high_res_feats"]
    assert tuple(image_embed.shape) == (1, c, 2, 2)
    assert [tuple(f.shape) for f in high_res] == [(1, c, 8, 8), (1, c, 4, 4)]
    # The flatten/permute/reshape round-trips, so the features equal slice i of the input levels.
    assert np.allclose(image_embed.numpy(), fpn[2][2])
    assert np.allclose(high_res[0].numpy(), fpn[0][2])
    assert np.allclose(high_res[1].numpy(), fpn[1][2])


class _FakeUNETR:
    """A stand-in for UNETR3D that mimics its per-slice encoder loop, for model-free tests.

    `run_unisam2_decoder_on_3d_embeddings` swaps in a stub encoder and calls `model(dummy)`. This fake
    reproduces UNETR3D.forward's ``[self.encoder(x[:, :, i])[0] for i in range(Z)]`` loop, records the
    per-slice features the stub returned, and emits a (1, 4, Z, H, W) prediction of the dummy size.
    A plain (non-nn.Module) class so the encoder attribute can be freely swapped and restored.
    """

    def __init__(self, img_size=8):
        self.encoder = types.SimpleNamespace(img_size=img_size)
        self.seen = []       # features the stub returned on the most recent call
        self.call_z = []     # z size of each decoder call (one per z block)

    def __call__(self, x):  # x: (1, 3, Z, H, W) dummy
        z = x.shape[2]
        self.seen = [self.encoder(x[:, :, i])[0] for i in range(z)]
        self.call_z.append(z)
        h, w = x.shape[-2:]
        return torch.zeros((1, 4, z, h, w))


def test_decoder_3d_stub_returns_per_slice_features_in_order():
    # The stub must hand the decoder each slice's feature, in order, as a (1, C, h, w) tensor.
    # z <= DEFAULT_TILE_Z so it is a single decoder pass.
    z, c, h, w = DEFAULT_TILE_Z, 2, 4, 4
    feats = np.arange(z * c * h * w, dtype="float32").reshape(z, c, h, w)
    model = _FakeUNETR(img_size=8)
    out = run_unisam2_decoder_on_3d_embeddings(model, {"features": feats, "original_size": (8, 8)}, device="cpu")
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
    out = run_unisam2_decoder_on_3d_embeddings(model, {"features": feats5, "original_size": (8, 8)}, device="cpu")
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
    out = run_unisam2_decoder_on_embeddings(model, {"features": feats5, "original_size": (8, 8)}, device="cpu")
    assert out.shape == (4, 8, 8)
    assert len(model.seen) == 1
    assert tuple(model.seen[0].shape) == (1, c, h, w)
    assert np.allclose(model.seen[0][0].cpu().numpy(), feats5[0, 0])


def test_decoder_3d_zchunks_deep_volume():
    # Regression for the z-tiling fix: a deep volume (small in-plane, not tiled in-plane) must be
    # decoded in bounded z blocks, not all at once, so peak memory stays bounded.
    z, c, h, w = 10, 2, 4, 4
    feats = np.zeros((z, c, h, w), dtype="float32")
    model = _FakeUNETR(img_size=8)
    out = run_unisam2_decoder_on_3d_embeddings(model, {"features": feats, "original_size": (8, 8)}, device="cpu")
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


def test_block_shape_2d_tiling_uses_tile():
    block, halo = _block_shape_and_halo((1024, 1024), ndim=2, tile_shape=(512, 512), halo=(64, 64))
    assert block == (1, 512, 512)
    assert halo == (0, 64, 64)
