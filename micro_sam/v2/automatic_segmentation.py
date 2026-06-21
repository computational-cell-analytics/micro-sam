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
from typing import Optional, Union

import numpy as np
import torch

from bioimage_cpp.utils import Blocking

from .util import _DEFAULT_MODEL
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
    model = UniSAM2(encoder=encoder, output_channels=output_channels)

    state = torch.load(checkpoint_path, weights_only=False, map_location=device or "cpu")
    # The standalone trainer saves the full model under 'model_state'; the joint trainer saves it
    # under 'unetr_state'. We also accept a raw state dict.
    if isinstance(state, dict):
        model_state = state.get("model_state", state.get("unetr_state", state))
    else:
        model_state = state
    model.load_state_dict(model_state)

    if device is not None:
        model.to(device)
    model.eval()
    return model


def run_unisam2_inference(
    model: torch.nn.Module,
    raw: np.ndarray,
    ndim: int,
    device: Optional[Union[str, torch.device]] = None,
    tile_shape: Optional[tuple] = None,
    halo: Optional[tuple] = None,
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

    Returns:
        The model predictions stacked along the channel axis, shape (4, *spatial).
        Channel 0 is the foreground probability, channels 1-3 the directed distances.
    """
    from torch_em.util.prediction import predict_with_halo
    from torch_em.transform.raw import normalize

    def _preprocess(crop):
        return np.concatenate([normalize(crop)] * 3, axis=0)

    is_3d = ndim == 3

    # Build the block shape and halo in (z, y, x) order for predict_with_halo. For 3d the tile
    # shape and halo are full 3d; for 2d they are in-plane (y, x). Without a tile shape the whole
    # image is a single block.
    if tile_shape is None:
        block_shape = tuple(raw.shape) if is_3d else (1, *raw.shape)
        block_halo = (0, 0, 0)
    elif is_3d:
        block_shape = tuple(tile_shape)  # (z, y, x)
        block_halo = (0, 0, 0) if halo is None else tuple(halo)
    else:
        block_shape = (1, *tile_shape)  # (1, y, x)
        block_halo = (0, *((0, 0) if halo is None else halo))

    if is_3d:
        input_ = raw[np.newaxis].astype("float32")
        output = np.zeros((4, *raw.shape), dtype="float32")
    else:
        input_ = raw[np.newaxis, np.newaxis].astype("float32")
        output = np.zeros((4, 1, *raw.shape), dtype="float32")

    output = predict_with_halo(
        input_=input_,
        model=model,
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

    real_encoder = model.encoder
    model.encoder = _StubEncoder(feature, getattr(real_encoder, "img_size", 1024))
    try:
        # The stub ignores the input content, so a zero image of the original shape is enough for
        # the model's preprocess/postprocess to resize the prediction back to the original size.
        dummy = torch.zeros((1, 3, 1, *original_size), device=device)
        output = model(dummy)
    finally:
        model.encoder = real_encoder
    return output[0, :, 0].detach().cpu().numpy()


@torch.no_grad()
def run_unisam2_decoder_on_tiled_embeddings(
    model: torch.nn.Module, image_embeddings: dict, device: Optional[Union[str, torch.device]] = None,
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

    Returns:
        The predictions stacked along the channel axis, shape (4, Y, X).
    """
    feats_group = image_embeddings["features"]
    shape = tuple(int(s) for s in feats_group.attrs["shape"])
    tile_shape = tuple(int(s) for s in feats_group.attrs["tile_shape"])
    halo = tuple(int(s) for s in feats_group.attrs["halo"])
    tiling = Blocking([0, 0], list(shape), list(tile_shape))

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
    ) -> None:
        """Run the UniSAM2 inference and store the foreground and distance predictions.

        Args:
            image: The input image, shape (Y, X) for 2d or (Z, Y, X) for 3d.
            ndim: The number of spatial dimensions (2 or 3).
            image_embeddings: Optional precomputed 2d image embeddings. If given (and 2d), only the
                decoder is run on them, reusing the embeddings shared with interactive / AMG instead
                of re-running the encoder. Ignored for 3d. See `precompute_image_embeddings`.
            i: Index for the image data. Unused here, kept for interface compatibility.
            tile_shape: Unused for the non-tiled segmenter (no tiling); kept so the interface matches
                the tiled segmenter.
            halo: Unused for the non-tiled segmenter; kept for interface compatibility.
        """
        if image_embeddings is not None and ndim == 2:
            self._prediction = run_unisam2_decoder_on_embeddings(
                self._model, image_embeddings, device=self._device
            )
        else:
            self._prediction = run_unisam2_inference(self._model, image, ndim, device=self._device)
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
    ) -> None:
        """Run the tiled UniSAM2 inference and store the foreground and distance predictions.

        Args:
            image: The input image, shape (Y, X) for 2d or (Z, Y, X) for 3d.
            ndim: The number of spatial dimensions (2 or 3).
            tile_shape: The tile shape for tiled prediction - (y, x) for 2d, (z, y, x) for 3d.
            halo: The halo for the overlap between tiles - (y, x) for 2d, (z, y, x) for 3d.
            image_embeddings: Optional precomputed tiled 2d image embeddings. If given (and 2d), the
                decoder is run per tile on them and stitched, reusing the embeddings shared with
                interactive / AMG. Ignored for 3d. See `precompute_image_embeddings`.
            i: Index for the image data. Unused here, kept for interface compatibility.
        """
        if image_embeddings is not None and ndim == 2:
            self._prediction = run_unisam2_decoder_on_tiled_embeddings(
                self._model, image_embeddings, device=self._device
            )
        else:
            self._prediction = run_unisam2_inference(
                self._model, image, ndim, device=self._device, tile_shape=tile_shape, halo=halo,
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
