"""Automatic instance segmentation with the UniSAM2 model.

The UniSAM2 model (a UNETR decoder on top of the SAM2 image encoder, see
`micro_sam.v2.models.util.UniSAM2`) predicts a foreground probability map and three directed
distance channels. These predictions are converted into an instance segmentation by one of two
strategies in `micro_sam.v2.postprocessing`:

- ``sparse`` -> `flow_instance_segmentation`: CellPose-style flow following. Suitable for light
  microscopy (LM) data, works in 2d and 3d.
- ``dense`` -> `run_multicut`: slice-wise oversegmentation + graph multicut. Suitable for electron
  microscopy (EM) data with large, densely-packed objects. Works for 2d and 3d (2d is run as a
  single-slice volume).

Inference supports in-plane (xy) tiling with a halo for both 2d and 3d data; when no tile shape is
given the whole image is processed in a single block.
"""

import os
import sys
import types
import contextlib
from typing import Optional, Union

import numpy as np
import torch

from bioimage_cpp.utils import Blocking

from .util import _DEFAULT_MODEL, DEFAULT_TILE_Z, DEFAULT_HALO_Z
from .postprocessing import flow_instance_segmentation, run_multicut


def _alias_legacy_namespace():
    """Alias the modules that were moved out of the old ``micro_sam2`` namespace.

    Older UniSAM2 checkpoints were pickled while this package lived under ``micro_sam2``, so
    ``torch.load`` needs the moved modules to be importable under their original paths.

    NOTE: This is currently around to debug stuff. This will be gone very soon!
    """
    import micro_sam.v2.datasets.sampler as datasets_sampler
    import micro_sam.v2.datasets.wrapper as datasets_wrapper
    import micro_sam.v2.transforms.labels as transforms_labels
    import micro_sam.v2.transforms.raw as transforms_raw

    root = sys.modules.setdefault("micro_sam2", types.ModuleType("micro_sam2"))
    root.__path__ = []
    datasets = sys.modules.setdefault("micro_sam2.datasets", types.ModuleType("micro_sam2.datasets"))
    datasets.__path__ = []
    transforms = sys.modules.setdefault("micro_sam2.transforms", types.ModuleType("micro_sam2.transforms"))
    transforms.__path__ = []

    sys.modules["micro_sam2.datasets.sampler"] = datasets_sampler
    sys.modules["micro_sam2.datasets.wrapper"] = datasets_wrapper
    sys.modules["micro_sam2.transforms.labels"] = transforms_labels
    sys.modules["micro_sam2.transforms.raw"] = transforms_raw
    root.datasets, root.transforms = datasets, transforms
    datasets.sampler, datasets.wrapper = datasets_sampler, datasets_wrapper
    transforms.labels, transforms.raw = transforms_labels, transforms_raw


def get_unisam2_model(
    checkpoint_path: Union[str, "os.PathLike"],
    device: Optional[Union[str, torch.device]] = None,
    encoder: str = _DEFAULT_MODEL,
    output_channels: int = 4,
) -> torch.nn.Module:
    """Load a UniSAM2 model for automatic segmentation from a checkpoint.

    Args:
        checkpoint_path: Path to the UniSAM2 checkpoint.
        device: The device to load the model onto.
        encoder: The SAM2 encoder backbone to build, e.g. 'hvit_t'.
        output_channels: The number of output channels (foreground + directed distances).

    Returns:
        The UniSAM2 model in eval mode.
    """
    from .models.util import UniSAM2

    _alias_legacy_namespace()

    state = torch.load(checkpoint_path, weights_only=False, map_location=device or "cpu")
    # The standalone trainer saves the full model under 'model_state'; the joint trainer saves it
    # under 'unetr_state' or 'decoder_state'. We also accept a raw state dict.
    if isinstance(state, dict):
        model_state = state.get("model_state", state.get("unetr_state", state.get("decoder_state", state)))
    else:
        model_state = state

    # The standalone UniSAM2 trainer builds the model with a string encoder, so the SAM2 image
    # encoder lives directly under 'encoder.*'. The joint trainer instead passes the SAM2 image
    # encoder module, which gets wrapped in 'SAM2EncoderAdapter' and so lives under 'encoder.inner.*'.
    # Detect the latter and rebuild the matching structure by passing a SAM2 image encoder module.
    needs_adapter = isinstance(encoder, str) and any(k.startswith("encoder.inner.") for k in model_state)
    if needs_adapter:
        from .util import get_sam2_model
        sam2_model = get_sam2_model(model_type=encoder, input_type="images", device=device or "cpu")
        encoder = sam2_model.image_encoder

    model = UniSAM2(encoder=encoder, output_channels=output_channels)
    model.load_state_dict(model_state)

    if device is not None:
        model.to(device)
    model.eval()
    return model


def _resize_spatial(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """Resize the trailing (Y, X) of a (B, C, Z, Y, X) tensor to `size`, leaving Z unchanged."""
    b, c, z, y, x_dim = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * z, c, y, x_dim)
    x = torch.nn.functional.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)
    return x.reshape(b, z, c, size[0], size[1]).permute(0, 2, 1, 3, 4)


class _SquareResizeWrapper(torch.nn.Module):
    """Run UniSAM2 with SAM2's square (anisotropic) resize convention.

    SAM2 resizes inputs to a fixed square `img_size` (not aspect-preserving), and the UniSAM2 decoder
    was trained on these square features. The UNETR3D forward instead applies a SAM-style
    aspect-preserving resize + crop, which disagrees for non-square inputs. Square-resizing each block
    to `img_size` here makes the inner preprocess/postprocess spatially no-ops, so the full-inference
    path matches the square convention (and the precomputed-embeddings path); the prediction is then
    resized back to the block size.
    """

    def __init__(self, model: torch.nn.Module, img_size: int) -> None:
        super().__init__()
        self.model = model
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[-2:]
        x = _resize_spatial(x, (self.img_size, self.img_size))
        out = self.model(x)
        return _resize_spatial(out, spatial)


@contextlib.contextmanager
def _bridge_halo_progress(pbar_update):
    """Bridge `predict_with_halo`'s internal per-block tqdm to an external `pbar_update` callback.

    `predict_with_halo` wraps its thread-pool map in a tqdm (a napari progress bar inside napari)
    that micro-sam cannot otherwise drive, so 3d/tiled auto-segmentation showed no real progress.
    This temporarily swaps that module-level tqdm for a thin iterator that fires `pbar_update` once
    per completed block. It is a no-op fallback if `pbar_update` is None.
    """
    if pbar_update is None:
        yield
        return

    import torch_em.util.prediction as prediction_module
    original_tqdm = prediction_module.tqdm

    class _ProgressBridge:
        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            for item in self._iterable:
                yield item
                pbar_update(1)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, n=1):
            pbar_update(n)

        def set_description(self, *args, **kwargs):
            pass

        def close(self):
            pass

    prediction_module.tqdm = _ProgressBridge
    try:
        yield
    finally:
        prediction_module.tqdm = original_tqdm


def _n_blocks(spatial_shape, ndim, block_shape):
    """Number of blocks `predict_with_halo` iterates over, for a determinate progress total."""
    blocked = spatial_shape if ndim == 3 else (1, *spatial_shape)
    return int(np.prod([int(np.ceil(s / b)) for s, b in zip(blocked, block_shape)]))


def _block_shape_and_halo(spatial_shape, ndim, tile_shape, halo):
    """Compute the (z, y, x) block shape and halo for `predict_with_halo`.

    For 3d data a volume is always chunked along z - using the explicit z tile when tiling is on,
    or the default z block when in-plane tiling is off (the whole in-plane plane per chunk). The
    model is trained on small z crops, so running every slice at once is both out-of-distribution
    and a memory blow-up (the cause of the 3d 'killed' reports). 2d data is a single (1, y, x) block.

    Args:
        spatial_shape: The spatial image shape, (Y, X) for 2d or (Z, Y, X) for 3d.
        ndim: The number of spatial dimensions (2 or 3).
        tile_shape: The in-plane/3d tile shape, or None for no tiling.
        halo: The tile halo, or None for no overlap.

    Returns:
        The (block_shape, block_halo) tuples in (z, y, x) order for `predict_with_halo`.
    """
    is_3d = ndim == 3
    if tile_shape is None and is_3d:
        n_slices = spatial_shape[0]
        z_block = min(DEFAULT_TILE_Z, n_slices)
        block_shape = (z_block, spatial_shape[1], spatial_shape[2])
        block_halo = (DEFAULT_HALO_Z if z_block < n_slices else 0, 0, 0)
    elif tile_shape is None:
        block_shape = (1, *spatial_shape)
        block_halo = (0, 0, 0)
    elif is_3d:
        block_shape = tuple(tile_shape)  # (z, y, x)
        block_halo = (0, 0, 0) if halo is None else tuple(halo)
    else:
        block_shape = (1, *tile_shape)  # (1, y, x)
        block_halo = (0, *((0, 0) if halo is None else halo))
    return block_shape, block_halo


def run_unisam2_inference(
    model: torch.nn.Module,
    raw: np.ndarray,
    ndim: int,
    device: Optional[Union[str, torch.device]] = None,
    tile_shape: Optional[tuple] = None,
    halo: Optional[tuple] = None,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
) -> np.ndarray:
    """Run the UniSAM2 model to predict foreground and directed distances.

    Inference is tiled with a halo. For 3d data the tiling is fully 3d (the tile shape and halo
    include the z axis); for 2d it is in-plane. When `tile_shape` is None the whole image is
    processed as a single block (no tiling).

    Args:
        model: The UniSAM2 model.
        raw: The input image, shape (Y, X) for 2d or (Z, Y, X) for 3d.
        ndim: The number of spatial dimensions (2 or 3).
        device: The device to run inference on.
        tile_shape: The tile shape for tiled prediction - (y, x) for 2d and (z, y, x) for 3d.
            None disables tiling.
        halo: The halo for the overlap between tiles - (y, x) for 2d and (z, y, x) for 3d.
            None means no overlap.
        pbar_init: Callback to initialize an external progress bar, called with the block count.
        pbar_update: Callback to update an external progress bar, called once per completed block.

    Returns:
        The model predictions stacked along the channel axis, shape (4, *spatial).
        Channel 0 is the foreground probability, channels 1-3 the directed distances.
    """
    from torch_em.util.prediction import predict_with_halo
    from torch_em.transform.raw import normalize

    def _preprocess(crop):
        return np.concatenate([normalize(crop)] * 3, axis=0)

    is_3d = ndim == 3

    block_shape, block_halo = _block_shape_and_halo(tuple(raw.shape), ndim, tile_shape, halo)
    if pbar_init is not None:
        desc = "Automatic segmentation (volume)" if is_3d else "Automatic segmentation"
        pbar_init(_n_blocks(tuple(raw.shape), ndim, block_shape), desc)

    if is_3d:
        input_ = raw[np.newaxis].astype("float32")
        output = np.zeros((4, *raw.shape), dtype="float32")
    else:
        input_ = raw[np.newaxis, np.newaxis].astype("float32")
        output = np.zeros((4, 1, *raw.shape), dtype="float32")

    # Wrap the model so each block is square-resized to 'img_size' before the forward pass, matching
    # SAM2's square resize (and the precomputed-embeddings path) instead of UNETR3D's aspect-preserving
    # resize + crop, which would misalign non-square blocks (the whole image when untiled, or edge tiles).
    img_size = getattr(getattr(model, "encoder", None), "img_size", 1024)
    square_model = _SquareResizeWrapper(model, img_size)

    with _bridge_halo_progress(pbar_update):
        output = predict_with_halo(
            input_=input_,
            model=square_model,
            block_shape=block_shape,
            halo=block_halo,
            preprocess=_preprocess,
            gpu_ids=[device] if device is not None else None,
            output=output,
            with_channels=True,
        )
    if not is_3d:
        output = output[:, 0]
    return output


def segment_from_predictions(prediction: np.ndarray, mode: str = "sparse", **kwargs) -> np.ndarray:
    """Convert UniSAM2 predictions into an instance segmentation.

    Args:
        prediction: The UniSAM2 predictions, shape (4, *spatial). Channel 0 is the foreground
            probability and channels 1-3 are the directed distances.
        mode: The segmentation mode. 'sparse' uses flow-based segmentation (LM data, 2d and 3d),
            'dense' uses multicut-based segmentation (EM data, 2d and 3d).
        kwargs: Additional parameters forwarded to the postprocessing function
            (`flow_instance_segmentation` for 'sparse', `run_multicut` for 'dense').

    Returns:
        The instance segmentation, uint32 array with the spatial shape of the prediction.
    """
    foreground = prediction[0]
    if mode == "dense":
        boundary_map = foreground.max() - foreground
        denom = boundary_map.max()
        if denom > 0:
            boundary_map = boundary_map / denom
        # Multicut uses the in-plane (y, x) distance channels.
        distances = np.stack([prediction[2], prediction[3]])
        # run_multicut expects volumetric inputs. For 2d data we run it on a single-slice volume
        # (no z-edges), which yields a 2d multicut, and squeeze the result back to 2d.
        if boundary_map.ndim == 2:
            seg = run_multicut(boundary_map[None], distances[:, None], **kwargs)[0]
        else:
            seg = run_multicut(boundary_map, distances, **kwargs)
    else:
        seg = flow_instance_segmentation(foreground, prediction[1:], **kwargs)
    return seg.astype("uint32")


def automatic_instance_segmentation(
    model: torch.nn.Module,
    raw: np.ndarray,
    ndim: int,
    mode: str = "sparse",
    device: Optional[Union[str, torch.device]] = None,
    inference_kwargs: Optional[dict] = None,
    **postproc_kwargs,
) -> np.ndarray:
    """Run UniSAM2 inference and convert the predictions into an instance segmentation.

    Args:
        model: The UniSAM2 model.
        raw: The input image, shape (Y, X) for 2d or (Z, Y, X) for 3d.
        ndim: The number of spatial dimensions (2 or 3).
        mode: The segmentation mode, 'sparse' (flow) or 'dense' (multicut).
        device: The device to run inference on.
        inference_kwargs: Additional keyword arguments for `run_unisam2_inference`.
        postproc_kwargs: Additional keyword arguments for the postprocessing.

    Returns:
        The instance segmentation, uint32 array.
    """
    prediction = run_unisam2_inference(model, raw, ndim, device=device, **(inference_kwargs or {}))
    return segment_from_predictions(prediction, mode=mode, **postproc_kwargs)


class _StubEncoder(torch.nn.Module):
    """Encoder replacement that returns precomputed features, bypassing the SAM2 image encoder.

    `UNETR3D.forward` calls `encoder(slice)[0]` to get the per-slice features and has no encoder
    skip connections, so returning the precomputed `vision_features` here reproduces the full
    forward pass without re-running the encoder.
    """

    def __init__(self, feature: torch.Tensor, img_size: int = 1024) -> None:
        super().__init__()
        self.feature = feature
        self.img_size = img_size

    def forward(self, x):  # noqa
        return [self.feature]


class _StubEncoder3D(torch.nn.Module):
    """Encoder replacement for 3d that returns precomputed per-slice features in call order.

    `UNETR3D.forward` runs ``torch.stack([self.encoder(x[:, :, i])[0] for i in range(Z)], dim=2)``,
    i.e. it calls the encoder once per z slice in order, so returning the i-th precomputed slice
    feature on the i-th call reproduces the full forward without re-running the encoder.
    """

    def __init__(self, features: torch.Tensor, img_size: int = 1024) -> None:
        super().__init__()
        self.features = features  # (Z, C, h, w)
        self.img_size = img_size
        self._idx = 0

    def forward(self, x):  # noqa
        feature = self.features[self._idx:self._idx + 1]
        self._idx += 1
        return [feature]


def _decode_3d_feature_block(model, feature, original_size, device):
    """Run the UNETR3D decoder on a ``(z, C, h, w)`` feature block via the stub encoder.

    Temporarily swaps the model's encoder for `_StubEncoder3D` (which returns the precomputed per-slice
    features in order) and runs the decoder on an original-size dummy, so the model resizes the
    prediction back to ``(z, H, W)`` itself.

    Returns:
        The prediction for this z block, shape ``(4, z, H, W)``.
    """
    img_size = getattr(model.encoder, "img_size", 1024)
    real_encoder = model.encoder
    model.encoder = _StubEncoder3D(feature, img_size)
    try:
        dummy = torch.zeros((1, 3, feature.shape[0], *original_size), device=device)
        output = model(dummy)  # (1, 4, z, H, W)
    finally:
        model.encoder = real_encoder
    return output[0].detach().cpu().numpy()  # (4, z, H, W)


@torch.no_grad()
def run_unisam2_decoder_on_3d_embeddings(
    model: torch.nn.Module, image_embeddings: dict, device: Optional[Union[str, torch.device]] = None,
    z_block: Optional[int] = None, z_halo: Optional[int] = None,
) -> np.ndarray:
    """Run only the UniSAM2 decoder on precomputed 3d embeddings (no encoder pass).

    Reuses the per-slice ``vision_features`` (shape ``(Z, C, h, w)``) produced for the volume by
    `micro_sam.v2.util.precompute_image_embeddings` - the same embeddings used for interactive 3d
    segmentation. Because those come from the video predictor's aspect-preserving resize + pad, which
    is exactly what `UNETR3D`'s preprocess/postprocess assume, the model resizes the prediction back
    to the original ``(Z, H, W)`` itself (no manual resize, unlike the 2d square-stretch path).

    The decoder pass is chunked along z (with a halo for 3d-conv context), so a deep volume with a
    small in-plane size - which is not tiled in-plane - does not decode the whole stack at once and
    run out of memory.

    Args:
        model: The UniSAM2 model (a `UNETR3D`).
        image_embeddings: Precomputed 3d image embeddings (with `features` of shape ``(Z, C, h, w)``).
        device: The device to run inference on.
        z_block: Number of slices to decode per z block (defaults to `DEFAULT_TILE_Z`). A value `>=`
            the slice count decodes the whole stack in one pass (no z-tiling).
        z_halo: Number of overlapping slices between z blocks, used as context and discarded when
            stitching (defaults to `DEFAULT_HALO_Z`).

    Returns:
        The predictions stacked along the channel axis, shape ``(4, Z, H, W)``.
    """
    z_block = DEFAULT_TILE_Z if z_block is None else int(z_block)
    z_halo = DEFAULT_HALO_Z if z_halo is None else int(z_halo)

    features = np.asarray(image_embeddings["features"])
    # Per-slice features are (Z, C, h, w); the tiled / save-path layout keeps a singleton batch axis
    # (Z, 1, C, h, w) - squeeze it so the stub returns a (1, C, h, w) feature per slice.
    if features.ndim == 5 and features.shape[1] == 1:
        features = features[:, 0]
    if features.ndim != 4:
        raise ValueError(
            f"Decoder-from-embeddings (3d) requires 3d embeddings (features with ndim 4 or 5), got {features.ndim}."
        )
    n_slices = features.shape[0]
    feature = torch.as_tensor(features, device=device).float()
    original_size = tuple(int(s) for s in np.array(image_embeddings["original_size"]).reshape(-1)[:2])

    # Whole stack fits in one z block: a single decoder pass.
    if n_slices <= z_block:
        return _decode_3d_feature_block(model, feature, original_size, device)

    # Otherwise decode in z blocks with a halo and stitch the inner range, bounding peak memory.
    output = np.zeros((4, n_slices, *original_size), dtype="float32")
    for z0 in range(0, n_slices, z_block):
        z1 = min(z0 + z_block, n_slices)
        c0, c1 = max(0, z0 - z_halo), min(n_slices, z1 + z_halo)
        pred = _decode_3d_feature_block(model, feature[c0:c1], original_size, device)  # (4, c1-c0, H, W)
        inner = z0 - c0
        output[:, z0:z1] = pred[:, inner:inner + (z1 - z0)]
    return output


@torch.no_grad()
def run_unisam2_decoder_on_embeddings(
    model: torch.nn.Module, image_embeddings: dict, device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """Run only the UniSAM2 decoder on precomputed image embeddings (no encoder pass).

    Reuses 2d embeddings produced by `micro_sam.v2.util.precompute_image_embeddings` (the same ones
    used for interactive segmentation and AMG). The encoder is temporarily replaced by a stub that
    returns the precomputed `vision_features`, so the rest of the model (the UNETR decoder) runs
    exactly as in the full forward pass. Only supported for 2d embeddings.

    Args:
        model: The UniSAM2 model.
        image_embeddings: Precomputed 2d image embeddings (with `features` of shape (1, C, h, w)).
        device: The device to run inference on.

    Returns:
        The predictions stacked along the channel axis, shape (4, Y, X).
    """
    features = np.asarray(image_embeddings["features"])
    if features.ndim != 4:
        raise ValueError(
            f"Decoder-from-embeddings requires 2d image embeddings (features with ndim 4), got {features.ndim}."
        )
    feature = torch.as_tensor(features, device=device).float()
    original_size = tuple(int(s) for s in np.array(image_embeddings["original_size"]).reshape(-1)[:2])

    img_size = getattr(model.encoder, "img_size", 1024)
    real_encoder = model.encoder
    model.encoder = _StubEncoder(feature, img_size)
    try:
        # The precomputed SAM2 features come from a square 'img_size x img_size' resize of the image
        # (SAM2 resizes to a fixed square, not aspect-preserving). The UNETR3D postprocessing instead
        # assumes the SAM-style aspect-preserving resize + crop, which only matches for square images
        # and otherwise misaligns the prediction. So we run the decoder on a square dummy (making the
        # crop a no-op) and resize the square prediction back to the original image size ourselves.
        dummy = torch.zeros((1, 3, 1, img_size, img_size), device=device)
        output = model(dummy)
    finally:
        model.encoder = real_encoder

    prediction = output[0, :, 0]  # (4, img_size, img_size)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(0), size=original_size, mode="bilinear", align_corners=False,
    )[0]
    return prediction.detach().cpu().numpy()


@torch.no_grad()
def run_unisam2_decoder_on_tiled_embeddings(
    model: torch.nn.Module,
    image_embeddings: dict,
    device: Optional[Union[str, torch.device]] = None,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
) -> np.ndarray:
    """Run the UniSAM2 decoder on precomputed tiled 2d embeddings and stitch the tiles.

    For each tile the decoder is run on the precomputed per-tile features (no encoder pass), and the
    inner block of the per-tile prediction (the halo is used only as context) is written into the
    full output - the same halo stitching as the micro-sam v1 tiled decoder.

    Args:
        model: The UniSAM2 model.
        image_embeddings: Precomputed tiled 2d image embeddings (with per-tile `features`/`high_res_feats`
            groups and `shape`/`tile_shape`/`halo` attrs), see `precompute_image_embeddings`.
        device: The device to run inference on.
        pbar_init: Callback to initialize an external progress bar, called with the number of tiles.
        pbar_update: Callback to update an external progress bar, called once per tile.

    Returns:
        The predictions stacked along the channel axis, shape (4, Y, X).
    """
    feats_group = image_embeddings["features"]
    shape = tuple(int(s) for s in feats_group.attrs["shape"])
    tile_shape = tuple(int(s) for s in feats_group.attrs["tile_shape"])
    halo = tuple(int(s) for s in feats_group.attrs["halo"])
    tiling = Blocking([0, 0], list(shape), list(tile_shape))

    if pbar_init is not None:
        pbar_init(tiling.number_of_blocks, "Automatic segmentation (tiles)")

    output = np.zeros((4, *shape), dtype="float32")
    for tile_id in range(tiling.number_of_blocks):
        tile_features = feats_group[str(tile_id)]
        # The UNETR decoder only needs the vision features, so we pass them as a single-image embedding.
        tile_embeddings = {"features": np.asarray(tile_features), "original_size": tile_features.attrs["original_size"]}
        tile_prediction = run_unisam2_decoder_on_embeddings(model, tile_embeddings, device=device)

        block = tiling.get_block_with_halo(tile_id, halo=list(halo))
        local_bb = tuple(slice(b, e) for b, e in zip(block.inner_block_local.begin, block.inner_block_local.end))
        inner_bb = tuple(slice(b, e) for b, e in zip(block.inner_block.begin, block.inner_block.end))
        output[(slice(None),) + inner_bb] = tile_prediction[(slice(None),) + local_bb]
        if pbar_update is not None:
            pbar_update(1)

    return output


@torch.no_grad()
def run_unisam2_decoder_on_tiled_3d_embeddings(
    model: torch.nn.Module,
    image_embeddings: dict,
    device: Optional[Union[str, torch.device]] = None,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
    z_block: Optional[int] = None,
    z_halo: Optional[int] = None,
) -> np.ndarray:
    """Run the UniSAM2 decoder on precomputed tiled 3d embeddings and stitch the tiles in-plane.

    Like `run_unisam2_decoder_on_tiled_embeddings`, but each tile holds the per-slice features for a
    full ``(Z, tile_y, tile_x)`` sub-volume, so the decoder is run in 3d per tile (itself z-chunked,
    see `run_unisam2_decoder_on_3d_embeddings`) and the inner block is stitched in-plane. No encoder
    pass is run.

    Args:
        model: The UniSAM2 model.
        image_embeddings: Precomputed tiled 3d embeddings (per-tile `features` groups with
            ``shape`` (Z, Y, X) / ``tile_shape`` / ``halo`` attrs), see `precompute_image_embeddings`.
        device: The device to run inference on.
        pbar_init: Callback to initialize an external progress bar, called with the number of tiles.
        pbar_update: Callback to update an external progress bar, called once per tile.
        z_block: Number of slices to decode per z block within each tile (defaults to `DEFAULT_TILE_Z`).
        z_halo: Number of overlapping slices between z blocks (defaults to `DEFAULT_HALO_Z`).

    Returns:
        The predictions stacked along the channel axis, shape ``(4, Z, Y, X)``.
    """
    feats_group = image_embeddings["features"]
    shape = tuple(int(s) for s in feats_group.attrs["shape"])  # (Z, Y, X)
    tile_shape = tuple(int(s) for s in feats_group.attrs["tile_shape"])  # (y, x)
    halo = tuple(int(s) for s in feats_group.attrs["halo"])  # (y, x)
    tiling = Blocking([0, 0], list(shape[1:]), list(tile_shape))  # in-plane only; z is full per tile

    if pbar_init is not None:
        pbar_init(tiling.number_of_blocks, "Automatic segmentation (tiles)")

    output = np.zeros((4, *shape), dtype="float32")
    for tile_id in range(tiling.number_of_blocks):
        tile_features = feats_group[str(tile_id)]  # (Z, C, h, w)
        tile_embeddings = {"features": np.asarray(tile_features), "original_size": tile_features.attrs["original_size"]}
        tile_prediction = run_unisam2_decoder_on_3d_embeddings(
            model, tile_embeddings, device=device, z_block=z_block, z_halo=z_halo,
        )  # (4, Z, ty, tx)

        block = tiling.get_block_with_halo(tile_id, halo=list(halo))
        local_bb = tuple(slice(b, e) for b, e in zip(block.inner_block_local.begin, block.inner_block_local.end))
        inner_bb = tuple(slice(b, e) for b, e in zip(block.inner_block.begin, block.inner_block.end))
        # Full channel + full z, inner block in-plane.
        output[(slice(None), slice(None)) + inner_bb] = tile_prediction[(slice(None), slice(None)) + local_bb]
        if pbar_update is not None:
            pbar_update(1)

    return output


class UniSAM2InstanceSegmentation:
    """Generates an instance segmentation with the UniSAM2 model.

    Mirrors the `initialize` / `generate` interface of the micro-sam v1
    `InstanceSegmentationWithDecoder`. Use it as follows:
    ```python
    segmenter = UniSAM2InstanceSegmentation(model)
    segmenter.initialize(image, ndim=2)  # Run the UniSAM2 inference.
    masks = segmenter.generate(mode="sparse", foreground_threshold=0.6)  # Post-process.
    ```

    Args:
        model: The UniSAM2 model.
        device: The device to run inference on.
    """

    def __init__(self, model: torch.nn.Module, device: Optional[Union[str, torch.device]] = None) -> None:
        self._model = model
        self._device = device
        self._prediction = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the segmenter has already been initialized."""
        return self._is_initialized

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        ndim: int,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        tile_shape: Optional[tuple] = None,
        halo: Optional[tuple] = None,
        pbar_init: Optional[callable] = None,
        pbar_update: Optional[callable] = None,
        z_block: Optional[int] = None,
        z_halo: Optional[int] = None,
    ) -> None:
        """Run the UniSAM2 inference and store the foreground and distance predictions.

        Args:
            image: The input image, shape (Y, X) for 2d or (Z, Y, X) for 3d.
            ndim: The number of spatial dimensions (2 or 3).
            image_embeddings: Optional precomputed image embeddings. If given, only the decoder is run
                on them (no encoder pass), reusing the embeddings shared with interactive / AMG - for
                both 2d and 3d. See `precompute_image_embeddings`.
            i: Index for the image data. Unused here, kept for interface compatibility.
            tile_shape: Unused for the non-tiled segmenter (no tiling); kept so the interface matches
                the tiled segmenter.
            halo: Unused for the non-tiled segmenter; kept for interface compatibility.
            pbar_init: Callback to initialize an external progress bar. The decoder-on-embeddings path
                is a single step; the full-inference path reports per block (per z chunk for 3d).
            pbar_update: Callback to update an external progress bar.
            z_block: Number of slices per z block for the 3d decoder pass (defaults to `DEFAULT_TILE_Z`).
            z_halo: Overlapping slices between z blocks (defaults to `DEFAULT_HALO_Z`).
        """
        if image_embeddings is not None:
            # Decoder-only on precomputed embeddings (2d or 3d) is a single step.
            if pbar_init is not None:
                pbar_init(1, "Automatic segmentation")
            if ndim == 3:
                self._prediction = run_unisam2_decoder_on_3d_embeddings(
                    self._model, image_embeddings, device=self._device, z_block=z_block, z_halo=z_halo,
                )
            else:
                self._prediction = run_unisam2_decoder_on_embeddings(
                    self._model, image_embeddings, device=self._device
                )
            if pbar_update is not None:
                pbar_update(1)
        else:
            # Full inference reports per block via 'run_unisam2_inference' (per z chunk for 3d).
            self._prediction = run_unisam2_inference(
                self._model, image, ndim, device=self._device,
                pbar_init=pbar_init, pbar_update=pbar_update,
            )
        self._is_initialized = True

    def generate(self, mode: str = "sparse", **kwargs) -> np.ndarray:
        """Convert the stored predictions into an instance segmentation.

        Args:
            mode: The segmentation mode, 'sparse' (flow) or 'dense' (multicut).
            kwargs: Additional parameters forwarded to the postprocessing.

        Returns:
            The instance segmentation, uint32 array.
        """
        if not self._is_initialized:
            raise RuntimeError("The segmenter has not been initialized. Call 'initialize' first.")
        return segment_from_predictions(self._prediction, mode=mode, **kwargs)


class TiledUniSAM2InstanceSegmentation(UniSAM2InstanceSegmentation):
    """Generates a tiled instance segmentation with the UniSAM2 model.

    Like `UniSAM2InstanceSegmentation`, but the model inference is tiled in-plane (xy) with a halo.

    Args:
        model: The UniSAM2 model.
        device: The device to run inference on.
    """

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        ndim: int,
        tile_shape: Optional[tuple] = None,
        halo: Optional[tuple] = None,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        pbar_init: Optional[callable] = None,
        pbar_update: Optional[callable] = None,
        z_block: Optional[int] = None,
        z_halo: Optional[int] = None,
    ) -> None:
        """Run the tiled UniSAM2 inference and store the foreground and distance predictions.

        Args:
            image: The input image, shape (Y, X) for 2d or (Z, Y, X) for 3d.
            ndim: The number of spatial dimensions (2 or 3).
            tile_shape: The tile shape for tiled prediction - (y, x) for 2d, (z, y, x) for 3d.
            halo: The halo for the overlap between tiles - (y, x) for 2d, (z, y, x) for 3d.
            image_embeddings: Optional precomputed tiled image embeddings. If given, the decoder is run
                per tile on them and stitched (no encoder pass), reusing the embeddings shared with
                interactive / AMG - for both 2d and 3d. See `precompute_image_embeddings`.
            i: Index for the image data. Unused here, kept for interface compatibility.
            pbar_init: Callback to initialize an external progress bar, called with the number of
                tiles (2d) or the number of blocks / z chunks (3d).
            pbar_update: Callback to update an external progress bar, called once per tile / block.
            z_block: Number of slices per z block for the per-tile 3d decoder (defaults to `DEFAULT_TILE_Z`).
            z_halo: Overlapping slices between z blocks (defaults to `DEFAULT_HALO_Z`).
        """
        if image_embeddings is not None and ndim == 3:
            self._prediction = run_unisam2_decoder_on_tiled_3d_embeddings(
                self._model, image_embeddings, device=self._device,
                pbar_init=pbar_init, pbar_update=pbar_update, z_block=z_block, z_halo=z_halo,
            )
        elif image_embeddings is not None and ndim == 2:
            self._prediction = run_unisam2_decoder_on_tiled_embeddings(
                self._model, image_embeddings, device=self._device,
                pbar_init=pbar_init, pbar_update=pbar_update,
            )
        else:
            # 3d tiled inference runs through 'predict_with_halo'; its per-block tqdm is bridged to
            # the external progress bar inside 'run_unisam2_inference' so progress is reported per block.
            self._prediction = run_unisam2_inference(
                self._model, image, ndim, device=self._device, tile_shape=tile_shape, halo=halo,
                pbar_init=pbar_init, pbar_update=pbar_update,
            )
        self._is_initialized = True


def get_unisam2_segmentation_generator(
    model: torch.nn.Module,
    is_tiled: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> UniSAM2InstanceSegmentation:
    """Get the UniSAM2 automatic instance segmentation generator.

    Args:
        model: The UniSAM2 model.
        is_tiled: Whether to use tiled inference.
        device: The device to run inference on.

    Returns:
        The instance segmentation generator, either `TiledUniSAM2InstanceSegmentation` (if tiled)
        or `UniSAM2InstanceSegmentation`.
    """
    if is_tiled:
        return TiledUniSAM2InstanceSegmentation(model, device=device)
    return UniSAM2InstanceSegmentation(model, device=device)
