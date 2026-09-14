"""Automatic semantic segmentation with a SAM2 based UNETR.

The models predict one class id per voxel. The class layout comes from the label transform that the model
was trained with. :func:`micro_sam.v2.transforms.labels.semantic_labels` derives the three class layout of
background, object boundary and object interior from an instance segmentation.
"""

import os
from typing import Optional, Tuple, Union

import numpy as np

import torch

from torch_em.util.prediction import predict_with_halo

from micro_sam.util import get_device
from micro_sam.v2.models.util import SemanticSAM2
from micro_sam.v2.normalization import normalize_raw
from micro_sam.v2.util import autocast_dtype, DEFAULT_TILE_SHAPE, DEFAULT_HALO, DEFAULT_TILE_Z, DEFAULT_HALO_Z


def get_semantic_model(
    model_type: str,
    num_classes: int = 3,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    device: Optional[Union[str, torch.device]] = None,
    initial_features: int = 32,
) -> torch.nn.Module:
    """Get a SemanticSAM2 model for inference.

    Args:
        model_type: The SAM2 encoder variant the model was trained with.
        num_classes: The number of semantic classes, the background class included.
        checkpoint_path: The 'best.pt' or 'latest.pt' file of a training run. The model keeps the pretrained
            SAM2 encoder and an untrained decoder if this is None.
        device: The device to load the model on. Auto-selects if None.
        initial_features: Width of the convolutional decoder the model was trained with.

    Returns:
        The model in evaluation mode.
    """
    device = get_device(device)
    model = SemanticSAM2(
        encoder=model_type, num_classes=num_classes, initial_features=initial_features, device=device
    )
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
    model.eval()
    return model


def semantic_segmentation(
    model: torch.nn.Module,
    raw: np.ndarray,
    tile_shape: Optional[Tuple[int, ...]] = None,
    halo: Optional[Tuple[int, ...]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Run semantic segmentation on an image or a volume.

    The raw data is normalized once over the whole input and is then predicted in tiles. The model resizes
    every tile to its encoder size, so the tile shape plus twice the halo sets the scale at which the model
    sees the objects. Keep that sum close to the training patch shape.

    Args:
        model: The semantic segmentation model, see :func:`get_semantic_model`.
        raw: The image of shape (Y, X), or the volume of shape (Z, Y, X).
        tile_shape: The inner shape of a prediction tile. Uses the micro-sam tiling defaults if None.
        halo: The halo added to every side of a tile. Uses the micro-sam tiling defaults if None.
        verbose: Whether to show a progress bar.

    Returns:
        The class ids, with the shape of the raw data.
    """
    if raw.ndim not in (2, 3):
        raise ValueError(f"The raw data has {raw.ndim} dimensions. Expected an image (Y, X) or a volume (Z, Y, X).")

    is_3d = raw.ndim == 3
    if tile_shape is None:
        tile_shape = (DEFAULT_TILE_Z,) + DEFAULT_TILE_SHAPE if is_3d else DEFAULT_TILE_SHAPE
    if halo is None:
        halo = (DEFAULT_HALO_Z,) + DEFAULT_HALO if is_3d else DEFAULT_HALO

    # 2d normalizes over the image and 3d over every z slice, which is what the training transforms do.
    normalized = normalize_raw(raw, axis=(1, 2) if is_3d else (0, 1))

    device = next(model.parameters()).device
    dtype = autocast_dtype(device)

    def prediction_function(net, inp):
        # The tiles arrive as (B, 1, Z, Y, X) or (B, 1, Y, X), and the encoder expects three channels.
        x = torch.cat([inp] * 3, dim=1)
        if not is_3d:
            x = x.unsqueeze(2)  # the model always expects a z axis
        if dtype is None:
            prediction = net(x)
        else:
            with torch.autocast(device_type=device.type, dtype=dtype):
                prediction = net(x)
        prediction = prediction.float()
        return prediction if is_3d else prediction.squeeze(2)

    logits = predict_with_halo(
        input_=normalized,
        model=model,
        gpu_ids=[device],
        block_shape=tile_shape,
        halo=halo,
        preprocess=None,
        prediction_function=prediction_function,
        disable_tqdm=not verbose,
        tqdm_desc="Run semantic segmentation",
    )
    return np.argmax(logits, axis=0).astype("uint8")
