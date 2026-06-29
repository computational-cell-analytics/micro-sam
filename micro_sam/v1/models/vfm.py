"""Vision Foundation Model (DINO, UNI) encoders for the classification tools.

The classification tools operate directly on dense image-encoder features, so they can use backbones
beyond SAM. This module produces embeddings in the same `ImageEmbeddings` format that
`compute_pixel_features` and `compute_object_features` consume.

Three loading backends are supported:
- `torch_hub`: DINOv2 (fetches model code + weights at runtime, no auth).
- `hf`: DINOv3 via the `transformers` library (gated, user provides HuggingFace access).
- `timm`: the MahmoodLab histopathology models UNI / UNI2-h via `timm` (gated, user provides access).

The gated models (DINOv3, UNI, UNI2-h) are never hosted or distributed by us: each user provides their
own HuggingFace access by accepting the license on the model page and authenticating in the terminal
(via `huggingface-cli login` or the `HF_TOKEN` environment variable), which the loading library picks
up automatically.

Supports 2D and 3D images, with or without tiling. Each tile / slice is encoded independently and the
embeddings are stored in the same zarr layout that SAM uses, so the classification feature computation
(`compute_pixel_features` / `compute_object_features`) consumes them unchanged.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from bioimage_cpp.utils import Blocking

from ... import util

# Registry of supported Vision Foundation Model encoders, keyed by the micro-sam `model_type`.
# 'backend' selects how the model is loaded: 'torch_hub' (DINOv2, auto-download) builds 'repo'/'entrypoint'
# via torch.hub; 'hf' (DINOv3, gated) loads the HuggingFace 'repo' via transformers.AutoModel; 'timm'
# (UNI/UNI2-h, gated) loads the HuggingFace 'repo' via timm.create_model.
VFM_MODELS: Dict[str, Dict] = {
    # DINOv2 (patch size 14); weights auto-download via torch.hub, no authentication.
    "dino_v2_vits": {"backend": "torch_hub", "repo": "facebookresearch/dinov2", "entrypoint": "dinov2_vits14", "patch_size": 14, "embed_dim": 384},  # noqa
    "dino_v2_vitb": {"backend": "torch_hub", "repo": "facebookresearch/dinov2", "entrypoint": "dinov2_vitb14", "patch_size": 14, "embed_dim": 768},  # noqa
    "dino_v2_vitl": {"backend": "torch_hub", "repo": "facebookresearch/dinov2", "entrypoint": "dinov2_vitl14", "patch_size": 14, "embed_dim": 1024},  # noqa
    "dino_v2_vitg": {"backend": "torch_hub", "repo": "facebookresearch/dinov2", "entrypoint": "dinov2_vitg14", "patch_size": 14, "embed_dim": 1536},  # noqa
    # DINOv3 (patch size 16); loaded from HuggingFace via transformers. Gated: user provides HF access.
    "dino_v3_vits": {"backend": "hf", "repo": "facebook/dinov3-vits16-pretrain-lvd1689m", "patch_size": 16, "embed_dim": 384},  # noqa
    "dino_v3_vitb": {"backend": "hf", "repo": "facebook/dinov3-vitb16-pretrain-lvd1689m", "patch_size": 16, "embed_dim": 768},  # noqa
    "dino_v3_vitl": {"backend": "hf", "repo": "facebook/dinov3-vitl16-pretrain-lvd1689m", "patch_size": 16, "embed_dim": 1024},  # noqa
    # UNI / UNI2-h (MahmoodLab histopathology); loaded from HuggingFace via timm. Gated: user provides HF access.
    "uni": {"backend": "timm", "repo": "hf-hub:MahmoodLab/uni", "patch_size": 16, "embed_dim": 1024, "timm_variant": "uni"},  # noqa
    "uni2_h": {"backend": "timm", "repo": "hf-hub:MahmoodLab/UNI2-h", "patch_size": 14, "embed_dim": 1536, "timm_variant": "uni2_h"},  # noqa
}

# Map each model to a human-readable size label for the GUI dropdowns.
VFM_SIZE_LABELS = {
    "dino_v2_vits": "small", "dino_v2_vitb": "base", "dino_v2_vitl": "large", "dino_v2_vitg": "giant",
    "dino_v3_vits": "small", "dino_v3_vitb": "base", "dino_v3_vitl": "large",
    "uni": "large", "uni2_h": "huge",  # UNI is ViT-L, UNI2-h is ViT-H.
}

# ImageNet statistics the web-pretrained VFM models expect, applied after mapping the image to [0, 1].
VFM_MEAN = (0.485, 0.456, 0.406)
VFM_STD = (0.229, 0.224, 0.225)

# Default longest-side input size; snapped down to a multiple of the patch size per model.
DEFAULT_VFM_IMG_SIZE = 1024


def is_vfm_model(model_type: str) -> bool:
    """Whether a model_type string refers to a registered VFM encoder (DINO / UNI)."""
    return isinstance(model_type, str) and model_type in VFM_MODELS


def get_vfm_model_names() -> Tuple[str, ...]:
    """The model_type strings of all supported VFM encoders."""
    return tuple(VFM_MODELS.keys())


class VFMEncoder(torch.nn.Module):
    """Wraps a Vision Foundation Model and produces dense (C, H, W) features for the classification tools.

    Works across the RGB backends: torch.hub (DINOv2), transformers (DINOv3) and timm (UNI / UNI2-h).
    The image is mapped to RGB (via `util._to_image`, matching what SAM sees), resized so its longest
    side equals `img_size` and padded to a square multiple of the patch size, then encoded. As with the
    SAM1 image encoder, the square feature map has its content in the top-left sub-rectangle, which the
    classification feature computation crops back to the image aspect ratio. All backends return the
    identical (C, h, w) feature contract, so downstream consumption is backend-agnostic.

    Args:
        model: The loaded model (a torch.hub, transformers or timm module).
        spec: The `VFM_MODELS` entry for this model.
        img_size: The longest-side input size, snapped down to a multiple of the patch size.
        device: The device to run the encoder on.
    """

    def __init__(self, model, spec: Dict, img_size: int, device):
        super().__init__()
        self.model = model.to(device).eval()
        self.device = device
        self.patch_size = spec["patch_size"]
        self.embed_dim = spec["embed_dim"]
        self.backend = spec["backend"]
        # The encoder needs a square input whose side is a multiple of the patch size.
        self.img_size = (img_size // self.patch_size) * self.patch_size
        # Set by 'get_vfm_model' so downstream code can read the model identity like a SAM predictor.
        self.model_type = None
        self.model_name = None

    @staticmethod
    def _resize_longest_side(height: int, width: int, long_side: int) -> Tuple[int, int]:
        scale = long_side / max(height, width)
        return int(round(height * scale)), int(round(width * scale))

    def _to_unit_rgb(self, image: np.ndarray) -> torch.Tensor:
        """Map an image to a (1, 3, H, W) tensor in [0, 1], preserving colour for true RGB.

        A genuine 8-bit RGB image (e.g. H&E) is scaled straight by 1/255 so the stain colour is kept,
        matching the standard DINO / UNI preprocessing. Grayscale, 2-channel or non-8-bit inputs go
        through SAM's per-channel min-max mapping (`util._to_image`), which normalizes arbitrary intensity
        ranges - and is colour-neutral for grayscale, since the replicated channels are scaled identically.
        """
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8:
            rgb = image  # true 8-bit RGB: keep the colour, just scale to [0, 1]
        else:
            rgb = util._to_image(image)  # (H, W, 3) uint8, per-channel min-max normalized
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).float().permute(2, 0, 1)
        return tensor.unsqueeze(0) / 255.0

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self._to_unit_rgb(image)
        mean = torch.tensor(VFM_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(VFM_STD, device=self.device).view(1, 3, 1, 1)

        new_h, new_w = self._resize_longest_side(x.shape[-2], x.shape[-1], self.img_size)
        x = F.interpolate(x, (new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        x = (x - mean) / std
        # Pad bottom/right to a square of side 'img_size' (a multiple of the patch size).
        x = F.pad(x, (0, self.img_size - new_w, 0, self.img_size - new_h))
        return x, (new_h, new_w)

    @torch.no_grad()
    def encode(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Encode an image into a dense feature map.

        Args:
            image: The input image, (H, W) or (H, W, C).

        Returns:
            The feature map of shape (C, H/patch, W/patch).
            The resized (pre-padding) input size, (h, w).
        """
        x, input_size = self._preprocess(image)
        param = next(self.model.parameters(), None)
        if param is not None:
            x = x.to(dtype=param.dtype)

        h, w = self.img_size // self.patch_size, self.img_size // self.patch_size
        if self.backend == "torch_hub":  # DINOv2 exposes a dense-feature helper directly.
            features = self.model.get_intermediate_layers(x, n=1, reshape=True)[0]  # (1, C, h, w)
        else:
            # 'hf' (transformers DINOv3) and 'timm' (UNI/UNI2-h) both return a token sequence of the
            # form [prefix tokens..., patch tokens]. The patch tokens are the trailing h*w entries, so
            # slicing from the end is robust to the (model-dependent) number of cls/register tokens.
            # Reshape to the (1, C, h, w) grid the torch.hub path returns, keeping the feature contract
            # identical across all backends.
            if self.backend == "hf":
                tokens = self.model(pixel_values=x).last_hidden_state  # (1, T, C)
            else:  # timm
                tokens = self.model.forward_features(x)  # (1, T, C)
            features = tokens[:, -(h * w):, :].reshape(1, h, w, -1).permute(0, 3, 1, 2)
        return features.float().squeeze(0).cpu().numpy().astype("float32"), input_size


def get_vfm_model(
    model_type: str,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    img_size: int = DEFAULT_VFM_IMG_SIZE,
) -> VFMEncoder:
    """Load a DINO / UNI encoder for the classification tools.

    Args:
        model_type: One of the keys in `VFM_MODELS`, e.g. 'dino_v2_vitb' or 'dino_v3_vitb'.
        device: The device to load the model on. By default the best available device.
        checkpoint_path: For DINOv3 (HuggingFace), an optional local model directory to load instead of
            the gated HuggingFace repo. Ignored for DINOv2, whose weights download automatically.
        img_size: The longest-side input size; snapped down to a multiple of the patch size.

    Returns:
        The VFM encoder.
    """
    if model_type not in VFM_MODELS:
        raise ValueError(
            f"Unknown VFM model '{model_type}'. Available models: {sorted(VFM_MODELS)}."
        )
    spec = VFM_MODELS[model_type]
    device = util.get_device(device)

    # xFormers' memory-efficient attention is CUDA-only; on CPU some backbones (e.g. the torch.hub DINOv2
    # code) would try it and crash, so disable it there. The DINOv2-derived code reads this env var at
    # import, so it must be set before the backbone is loaded. No-op if xFormers is not installed.
    if str(device) == "cpu":
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    if spec["backend"] == "torch_hub":  # DINOv2: code + weights auto-download, no authentication.
        model = torch.hub.load(spec["repo"], spec["entrypoint"])
    elif spec["backend"] == "hf":  # DINOv3: HuggingFace via transformers, user's own HF access.
        model = _load_hf_model(spec["repo"], checkpoint_path)
    else:  # 'timm': UNI / UNI2-h from HuggingFace via timm, user's own HF access.
        model = _load_timm_model(spec)

    encoder = VFMEncoder(model, spec, img_size=img_size, device=device)
    encoder.model_type = model_type
    encoder.model_name = model_type
    return encoder


def _load_timm_model(spec: Dict):
    """Load a UNI / UNI2-h model from HuggingFace via timm, using the user's own HF access.

    timm resolves the HuggingFace token from the environment (`HF_TOKEN`) or the `huggingface-cli login`
    cache, so the user manages their own access; we never handle their token.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "Loading UNI / UNI2-h requires the 'timm' package. Install it with 'pip install timm'."
        ) from e

    # Shared kwargs; 'dynamic_img_size' lets the ViT accept our (square, patch-multiple) inputs.
    kwargs = {"pretrained": True, "num_classes": 0, "dynamic_img_size": True, "init_values": 1e-5}
    if spec.get("timm_variant") == "uni2_h":  # UNI2-h needs its full custom ViT-H config (see model card).
        kwargs.update({
            "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2, "no_embed_class": True, "reg_tokens": 8,
            "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU,
        })

    try:
        return timm.create_model(spec["repo"], **kwargs)
    except Exception as e:
        repo = spec["repo"].split("hf-hub:")[-1]
        raise RuntimeError(
            f"Could not load '{spec['repo']}' via timm; see the chained error above for the exact "
            f"reason. The UNI / UNI2-h weights are gated by MahmoodLab: request access at "
            f"https://huggingface.co/{repo} (it requires an institutional-email HuggingFace account) "
            "and authenticate in your terminal via 'huggingface-cli login' or the 'HF_TOKEN' "
            "environment variable."
        ) from e


def _load_hf_model(repo: str, checkpoint_path=None):
    """Load a gated DINOv3 model from HuggingFace via transformers, with the user's own HF access.

    `transformers` resolves the HuggingFace token from the environment (`HF_TOKEN`) or the
    `huggingface-cli login` cache, so the user manages their own access; we never handle their token.
    """
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "Loading DINOv3 requires the 'transformers' package (>=4.56). Install it with "
            "'pip install transformers'."
        ) from e

    source = str(checkpoint_path) if checkpoint_path else repo
    try:
        return AutoModel.from_pretrained(source)
    except Exception as e:
        # The chained error (e) carries the exact reason: license not requested, access still pending
        # the repo authors' review, missing/invalid token, etc. We point the user at it rather than
        # assuming a single cause.
        raise RuntimeError(
            f"Could not load the DINOv3 model '{source}' from HuggingFace; see the chained error above "
            f"for the exact reason. The weights are gated by Meta: request access at "
            f"https://huggingface.co/{repo} (approval by the repo authors may still be pending) and "
            "authenticate in your terminal via 'huggingface-cli login' or the 'HF_TOKEN' environment "
            "variable."
        ) from e


def _handle_pbar(verbose, pbar_init, pbar_update):
    """Return (pbar_init, pbar_update, pbar_close): use the given callbacks, else a local tqdm bar."""
    if pbar_init is not None and pbar_update is not None:
        return pbar_init, pbar_update, (lambda: None)

    pbar = None

    def init(total, description):
        nonlocal pbar
        pbar = tqdm(total=total, desc=description, disable=not verbose)

    def update(n):
        if pbar is not None:
            pbar.update(n)

    def close():
        if pbar is not None:
            pbar.close()

    return init, update, close


def _write_signature(f, encoder, input_, mode):
    """Write the root-level embedding signature read by 'AnnotatorState.initialize_predictor'."""
    f.attrs["data_signature"] = util._compute_data_signature(input_)
    f.attrs["model_type"] = encoder.model_type
    f.attrs["model_name"] = encoder.model_type
    f.attrs["patch_size"] = encoder.patch_size
    f.attrs["embed_dim"] = encoder.embed_dim
    f.attrs["micro_sam_version"] = util.__version__
    f.attrs["vfm_mode"] = mode


def _load_cached_embeddings(f, encoder, input_):
    """Return cached embeddings from an opened zarr group if they match the image and model; else None."""
    if "features" not in f or f.attrs.get("vfm_mode") is None:
        return None
    matches = (
        f.attrs.get("data_signature") == util._compute_data_signature(input_)
        and f.attrs.get("model_name") == encoder.model_type
    )
    if not matches:
        return None
    if f.attrs["vfm_mode"] in ("2d", "3d"):  # non-tiled: 'features' is an array, sizes in root attrs
        return {
            "features": f["features"][:],
            "input_size": list(f.attrs["input_size"]),
            "original_size": list(f.attrs["original_size"]),
        }
    return {"features": f["features"], "input_size": None, "original_size": None}  # tiled: group of tiles


def _compute_vfm_2d(encoder, input_, f, save_path, pbar_init, pbar_update):
    from ...util import _create_dataset_with_data
    pbar_init(1, "Compute Image Embeddings 2D")
    features, input_size = encoder.encode(input_)
    features = features[None]  # (1, C, h, w), matching the SAM1 2D layout
    pbar_update(1)

    image_embeddings = {"features": features, "input_size": list(input_size), "original_size": list(input_.shape[:2])}
    if save_path is not None:
        _create_dataset_with_data(f, "features", data=features)
        f.attrs["input_size"] = list(input_size)
        f.attrs["original_size"] = list(input_.shape[:2])
        _write_signature(f, encoder, input_, "2d")
    return image_embeddings


def _compute_vfm_3d(encoder, input_, f, save_path, pbar_init, pbar_update):
    from ...util import _create_dataset_with_data
    n_slices = input_.shape[0]
    pbar_init(n_slices, "Compute Image Embeddings 3D")
    planes, input_size = [], None
    for z in range(n_slices):
        feats, input_size = encoder.encode(input_[z])
        planes.append(feats[None])  # (1, C, h, w)
        pbar_update(1)
    features = np.stack(planes)  # (Z, 1, C, h, w)

    original_size = list(input_[0].shape[:2])
    image_embeddings = {"features": features, "input_size": list(input_size), "original_size": original_size}
    if save_path is not None:
        _create_dataset_with_data(f, "features", data=features, chunks=(1, 1) + features.shape[2:])
        f.attrs["input_size"] = list(input_size)
        f.attrs["original_size"] = original_size
        _write_signature(f, encoder, input_, "3d")
    return image_embeddings


def _compute_vfm_tiled(encoder, input_, tile_shape, halo, f, pbar_init, pbar_update, is_3d):
    from ...util import _create_dataset_with_data
    spatial_shape = input_.shape[1:] if is_3d else input_.shape[:2]
    tiling = Blocking([0, 0], list(spatial_shape), list(tile_shape))
    n_tiles = tiling.number_of_blocks
    n_slices = input_.shape[0] if is_3d else 1

    features = f.require_group("features")
    features.attrs["shape"] = list(spatial_shape)
    features.attrs["tile_shape"] = list(tile_shape)
    features.attrs["halo"] = list(halo)

    pbar_init(n_tiles * n_slices, f"Compute Image Embeddings {'3D' if is_3d else '2D'} tiled")
    datasets = {}
    for z in range(n_slices):
        for tile_id in range(n_tiles):
            block = tiling.get_block_with_halo(tile_id, list(halo))
            bb = tuple(slice(beg, end) for beg, end in zip(block.outer_block.begin, block.outer_block.end))
            tile_input = input_[(z,) + bb] if is_3d else input_[bb]
            feats, input_size = encoder.encode(tile_input)  # (C, h, w)

            ds_name = str(tile_id)
            if is_3d:
                if ds_name not in datasets:
                    shape = (n_slices, 1) + feats.shape
                    ds = _create_dataset_with_data(
                        features, ds_name, data=np.zeros(shape, dtype="float32"), chunks=(1, 1) + feats.shape
                    )
                    ds.attrs["original_size"] = list(tile_input.shape[:2])
                    ds.attrs["input_size"] = list(input_size)
                    datasets[ds_name] = ds
                datasets[ds_name][z] = feats[None]
            else:
                ds = _create_dataset_with_data(features, ds_name, data=feats[None])  # (1, C, h, w)
                ds.attrs["original_size"] = list(tile_input.shape[:2])
                ds.attrs["input_size"] = list(input_size)
            pbar_update(1)

    _write_signature(f, encoder, input_, "tiled-3d" if is_3d else "tiled-2d")
    return {"features": features, "input_size": None, "original_size": None}


def precompute_vfm_embeddings(
    predictor: VFMEncoder,
    input_: np.ndarray,
    save_path: Optional[Union[str, os.PathLike]] = None,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    lazy_loading: bool = False,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
    **kwargs,
) -> util.ImageEmbeddings:
    """Compute VFM image embeddings in the `ImageEmbeddings` format used by the classification tools.

    This mirrors `micro_sam.v1.util.precompute_image_embeddings` (same signature and return contract) so
    it can be used as a drop-in for VFM (DINO / UNI) encoders. Supports 2D and 3D images, with or without
    tiling.

    Args:
        predictor: The VFM encoder (named 'predictor' to match the SAM precompute signature).
        input_: The input data, 2D (H, W[, C]) or 3D (Z, H, W[, C]).
        save_path: Optional path to a zarr container to cache the embeddings.
        ndim: The dimensionality of the data. If not given, deduced from the input.
        tile_shape: Shape of tiles for tiled embedding computation. By default no tiling.
        halo: Overlap of the tiles for tiled embedding computation. By default no tiling.
        verbose: Whether to be verbose.
        lazy_loading: Unused; accepted for signature compatibility.
        pbar_init: Callback to initialize an external progress bar (steps, description).
        pbar_update: Callback to update an external progress bar.

    Returns:
        The image embeddings.
    """
    import zarr
    ndim = input_.ndim if ndim is None else ndim
    if ndim not in (2, 3):
        raise ValueError(f"Invalid dimensionality {ndim}, expect 2 or 3 dimensional data.")

    # Open / create the zarr container and return cached embeddings if they match; otherwise truncate.
    if save_path is None:
        f = zarr.group()
    elif os.path.exists(save_path):
        f = zarr.open(save_path, mode="a")
        cached = _load_cached_embeddings(f, predictor, input_)
        if cached is not None:
            return cached
        f = zarr.open(save_path, mode="w")
    else:
        f = zarr.open(save_path, mode="a")

    pbar_init, pbar_update, pbar_close = _handle_pbar(verbose, pbar_init, pbar_update)
    is_3d = ndim == 3
    if tile_shape is not None:
        embeddings = _compute_vfm_tiled(predictor, input_, tile_shape, halo, f, pbar_init, pbar_update, is_3d)
    elif is_3d:
        embeddings = _compute_vfm_3d(predictor, input_, f, save_path, pbar_init, pbar_update)
    else:
        embeddings = _compute_vfm_2d(predictor, input_, f, save_path, pbar_init, pbar_update)
    pbar_close()
    return embeddings
