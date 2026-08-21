import os
import sys
import pooch
import shutil
import warnings
import contextlib
from pathlib import Path
from typing import Any, Union, Literal, Optional, Sequence, Tuple

import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import numpy as np

import torch

from micro_sam.v2.models._video_predictor import _build_sam2_video_predictor
from micro_sam.v2.normalization import IMAGE_PREPROCESSING, VIDEO_PREPROCESSING, to_image
from micro_sam.util import (
    get_device, get_cache_directory, microsam_cachedir, _open_embeddings,
    _configure_mps_memory, device_type, make_temp_embedding_path, BF16_MIN_CAPABILITY,
)


Device = Optional[Union[str, torch.device]]
Devices = Optional[Union[str, torch.device, Sequence[Union[str, torch.device]]]]

# Precision preference on cuda: bf16, then fp16, then fp32. See `micro_sam.util.BF16_MIN_CAPABILITY`.
FP16_MIN_CAPABILITY = (7, 0)


def _precision_device(device: Device = None) -> torch.device:
    """The device a precision decision is made for, unvalidated so a missing backend answers fp32."""
    return torch.device(get_device()) if device is None else torch.device(device)


def autocast_dtype(device: Device = None) -> Optional[torch.dtype]:
    """The dtype SAM2 and UniSAM2 inference runs in on a device.

    Args:
        device: The device the forward pass runs on. Defaults to the best available one.

    Returns:
        The half precision dtype the device runs natively, or None where inference stays in fp32.
    """
    device = _precision_device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    capability = torch.cuda.get_device_capability(device)
    if capability >= BF16_MIN_CAPABILITY:
        return torch.bfloat16
    if capability >= FP16_MIN_CAPABILITY:
        return torch.float16
    return None


def autocast(device: Device = None):
    """The autocast context inference runs in on a device.

    Args:
        device: The device the forward pass runs on. Defaults to the best available one.

    Returns:
        The autocast context, or a null context for fp32.
    """
    device = _precision_device(device)
    dtype = autocast_dtype(device)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def precision_name(device: Device = None) -> str:
    """The precision name of a device, for the embedding cache signature.

    Args:
        device: The device the forward pass runs on.

    Returns:
        One of 'bf16', 'fp16' or 'fp32'.
    """
    return {torch.bfloat16: "bf16", torch.float16: "fp16", None: "fp32"}[autocast_dtype(device)]


def to_float32(value: Any) -> Any:
    """Cast the floating point tensors of a nested structure to fp32.

    Autocast returns half precision, but the embedding cache is fp32 and numpy has no bfloat16.

    Args:
        value: A tensor, or a dict, list or tuple containing tensors.

    Returns:
        The same structure with its floating point tensors in fp32.
    """
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: to_float32(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_float32(item) for item in value)
    return value


def encode_image(predictor, image: np.ndarray) -> None:
    """Run the image encoder in the device precision, keeping the cached features in fp32.

    Args:
        predictor: The SAM2 image predictor.
        image: The image to encode.
    """
    with autocast(predictor.device):
        predictor.set_image(image)
    predictor._features = to_float32(predictor._features)


class ImageEmbeddings(dict):
    """Embedding result with explicit lifetime management for its backing store."""

    def __init__(self, embeddings, store=None, temporary_path=None):
        super().__init__(embeddings)
        self._store = store
        self._temporary_path = temporary_path
        self._closed = False

    @property
    def closed(self):
        """Whether this embedding resource is closed."""
        return self._closed

    @property
    def temporary_path(self):
        """The owned ephemeral path, or None for memory-backed or persistent embeddings."""
        return self._temporary_path

    def close(self):
        """Close the backing store and remove an owned ephemeral path."""
        if self._closed:
            return
        self._closed = True
        try:
            store = getattr(self._store, "file", self._store)
            close = getattr(store, "close", None)
            if close is not None:
                close()
        finally:
            self._store = None
            if self._temporary_path is not None:
                shutil.rmtree(self._temporary_path, ignore_errors=True)
                self._temporary_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


# NOTE: The model config is expected to be fetched from the module's relative path location.
sys.path.append(str(Path(sam2.__file__).parents[0]))


_DEFAULT_MODEL = "hvit_t"

# Only SAM2.1 is supported.
CFG_PATHS = {
    "hvit_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "hvit_s": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "hvit_b": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "hvit_l": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

SUPPORTED_MODELS = ["hvit_t", "hvit_s", "hvit_b", "hvit_l"]

URLS = {
    "hvit_t": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "hvit_s": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "hvit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "hvit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

HASHES = {
    "hvit_t": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    "hvit_s": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    "hvit_b": "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
    "hvit_l": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
}


# Default in-plane tiling for large images. The tool enables tiling when an in-plane axis exceeds
# DEFAULT_TILING_THRESHOLD. The SAM input patch per axis is then DEFAULT_TILE_SHAPE + 2 * DEFAULT_HALO,
# which is kept equal to the threshold (512 + 2 * 128 = 768).
DEFAULT_TILING_THRESHOLD = 768
DEFAULT_TILE_SHAPE = (512, 512)
DEFAULT_HALO = (128, 128)

# Default z block / halo for volumetric (3d) tiling. Each decoder pass spans the inner block plus the
# halo on each side, i.e. DEFAULT_TILE_Z + 2 * DEFAULT_HALO_Z = 8 slices, matching the UniSAM2 8-slice
# training crop (so the z-convolutions see the z-context they were trained on. Do not enlarge this
# beyond the training crop). Set the z tile >= the slice count to disable z-tiling.
DEFAULT_TILE_Z = 4
DEFAULT_HALO_Z = 2

# Encoder batch size per free-VRAM band in GiB, then per backbone, measured end to end. A device
# uses the largest band it reaches, see BAND_TOLERANCE.
VRAM_BATCH_SIZES = {
    4: {"hvit_t": 2, "hvit_s": 2, "hvit_b": 2, "hvit_l": 1},
    6: {"hvit_t": 2, "hvit_s": 2, "hvit_b": 2, "hvit_l": 2},
    8: {"hvit_t": 4, "hvit_s": 4, "hvit_b": 4, "hvit_l": 4},
    10: {"hvit_t": 4, "hvit_s": 4, "hvit_b": 4, "hvit_l": 4},
    12: {"hvit_t": 4, "hvit_s": 4, "hvit_b": 4, "hvit_l": 4},
    16: {"hvit_t": 8, "hvit_s": 8, "hvit_b": 8, "hvit_l": 8},
    24: {"hvit_t": 8, "hvit_s": 8, "hvit_b": 8, "hvit_l": 8},
    32: {"hvit_t": 16, "hvit_s": 16, "hvit_b": 16, "hvit_l": 16},
    40: {"hvit_t": 16, "hvit_s": 16, "hvit_b": 16, "hvit_l": 16},
    48: {"hvit_t": 16, "hvit_s": 16, "hvit_b": 16, "hvit_l": 16},
    80: {"hvit_t": 32, "hvit_s": 32, "hvit_b": 32, "hvit_l": 32},
}

# The backbone assumed for an unknown model name, the most memory-hungry one in the table.
FALLBACK_BACKBONE = "hvit_l"

# Bands are keyed by nominal card size, but a card never reports all of it as free (CUDA context,
# ECC reserve): an 80 GB A100 has 79.25 GiB. Without this slack every such card falls a band short.
BAND_TOLERANCE = 0.95


def _band_for(free_gib):
    """The largest tabulated VRAM band the device reaches, or None if it reaches none."""
    reached = [band for band in VRAM_BATCH_SIZES if free_gib >= band * BAND_TOLERANCE]
    return max(reached) if reached else None


def _backbone_of(model_type):
    """Map a model name onto its backbone, e.g. 'hvit_t_cells' -> 'hvit_t'."""
    backbone = str(model_type)[:6]
    return backbone if backbone in SUPPORTED_MODELS else FALLBACK_BACKBONE


def _free_vram_gib(device):
    """The free VRAM of a CUDA device in GiB, or None for a non-CUDA device."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.mem_get_info(device)[0] / 1024 ** 3


def recommend_batch_size(model_type, device, n_jobs=None):
    """Look up the encoder batch size for a model on a device.

    Args:
        model_type: The model name, e.g. 'hvit_t' or 'hvit_t_cells'. Unknown names are treated as the
            most memory-hungry backbone.
        device: The device the encoder runs on. Non-CUDA devices always get a batch size of one, as
            does a device with less free VRAM than the smallest tabulated band.
        n_jobs: The total number of tiles / slices, to avoid a batch larger than the work.

    Returns:
        The batch size to use, at least one.
    """
    free_gib = _free_vram_gib(device)
    if free_gib is None:
        return 1

    band = _band_for(free_gib)
    if band is None:
        # The smallest entry is calibrated for the smallest band, not below it, so a device that
        # reaches no band stays at one instead of relying on the OOM backoff to recover.
        return 1

    batch_size = VRAM_BATCH_SIZES[band][_backbone_of(model_type)]
    if n_jobs is not None:
        batch_size = min(batch_size, max(1, int(n_jobs)))
    return int(batch_size)


def needs_default_tiling(shape):
    """Whether to enable default in-plane tiling for a given image shape.

    Args:
        shape: The image shape without any channel axis. Either 2d (y, x) or 3d (z, y, x);
            for 3d only the in-plane (y, x) axes are considered, not the leading z axis.

    Returns:
        Whether tiling should be enabled by default.
    """
    if len(shape) == 2:
        return shape[0] > DEFAULT_TILING_THRESHOLD or shape[1] > DEFAULT_TILING_THRESHOLD
    elif len(shape) == 3:
        return shape[1] > DEFAULT_TILING_THRESHOLD or shape[2] > DEFAULT_TILING_THRESHOLD
    return False


# Finetuned SAM2 models (the micro-sam "model download console" for SAM2). These are exported into
# the two-file micro-sam layout - an interactive predictor checkpoint ('<name>') and a UniSAM2
# decoder checkpoint ('<name>_decoder') - by 'scripts/model_export/export_sam2_cells_model.py'.
# This mirrors the v1 'micro_sam.v1.util.models' registry (encoder + '_decoder' entries) so the
# backend (download, loading, GUI, automatic segmentation) works the same way as for SAM v1.
# The base SAM2 backbone is read off the first 6 characters of the name (e.g. 'hvit_t_cells' ->
# 'hvit_t'), so no explicit backbone mapping is needed. The user-facing GUI name is defined in the
# annotator widgets.
FINETUNED_MODELS = [
    # Microscopy generalist: joint SAM2 + UniSAM2 model with the 'hvit_t' backbone.
    "hvit_t_cells",
]

# The default model for the annotation tools (GUI + CLI + Python API). This is the single source of
# truth for the default. The GUI derives its synthetic 'vit_<size><suffix>' selector string from it.
DEFAULT_MODEL = "hvit_t_cells"

FINETUNED_URLS = {
    "hvit_t_cells": "https://owncloud.gwdg.de/index.php/s/iqNv2cjhPMGOo9J/download",
    "hvit_t_cells_decoder": "https://owncloud.gwdg.de/index.php/s/VlXsFg16Qh2SsiA/download",
}

FINETUNED_HASHES = {
    "hvit_t_cells": "xxh128:0d1873746eda30f2c1b1fd3edd9a82d0",
    "hvit_t_cells_decoder": "xxh128:301163dbb748519da1e03057789f1ccf",
}


def models():
    """Return the finetuned SAM2 models registry.

    Mirrors `micro_sam.v1.util.models`: finetuned SAM2 checkpoints and their UniSAM2 decoders are
    registered with their xxh128 hashes and download URLs and fetched via pooch. The base SAM2
    backbones (hvit_t/s/b/l) are downloaded separately, see `_get_checkpoint`.

    Returns:
        The pooch registry for the finetuned SAM2 models.
    """
    registry = pooch.create(
        path=os.path.join(microsam_cachedir(), "models"),
        base_url="",
        registry=FINETUNED_HASHES,
        urls=FINETUNED_URLS,
    )
    return registry


def get_model_names():
    """Return the names of the finetuned SAM2 models available in the download console."""
    return list(FINETUNED_MODELS)


def has_registered_decoder(model_type):
    """Whether a finetuned SAM2 model has a registered UniSAM2 decoder (for automatic segmentation).

    A cheap registry lookup (no download), used e.g. to decide the default automatic-segmentation mode
    before the model / decoder has actually been loaded.

    Args:
        model_type: The SAM2 model name, e.g. 'hvit_t_cells'.

    Returns:
        Whether a '<model_type>_decoder' is registered.
    """
    return f"{model_type}_decoder" in FINETUNED_HASHES


def _download_finetuned_sam2_model(model_type, progress_bar_factory=None):
    """Download a finetuned SAM2 model and (if available) its UniSAM2 decoder.

    Mirrors `micro_sam.v1.util._download_sam_model`.

    Args:
        model_type: The finetuned model name, e.g. 'hvit_t_cells'.
        progress_bar_factory: Optional callable creating a progress bar for the download.

    Returns:
        A tuple of (checkpoint_path, model_hash, decoder_path). 'decoder_path' is None if the model
        does not have a registered decoder.
    """
    model_registry = models()

    progress_bar = True
    if not os.path.exists(os.path.join(get_cache_directory(), model_type)) and progress_bar_factory is not None:
        progress_bar = progress_bar_factory(model_type)

    checkpoint_path = model_registry.fetch(model_type, progressbar=progress_bar)
    if not isinstance(progress_bar, bool):
        progress_bar.close()

    model_hash = model_registry.registry[model_type]

    decoder_name = f"{model_type}_decoder"
    decoder_path = model_registry.fetch(
        decoder_name, progressbar=True
    ) if decoder_name in model_registry.registry else None

    return checkpoint_path, model_hash, decoder_path


def _get_device(device=None):
    if device is None or device == "auto":
        device = get_device()
    else:
        _configure_mps_memory(device)

    if device_type(device) == "cuda":
        # NOTE: Adapt global variables to work with flash attentions.
        sam2.modeling.sam.transformer.OLD_GPU = True
        sam2.modeling.sam.transformer.USE_FLASH_ATTN = True
        sam2.modeling.sam.transformer.MATH_KERNEL_ON = True

    return device


def _get_checkpoint(model_type=_DEFAULT_MODEL):
    save_directory = os.path.expanduser(pooch.os_cache("micro_sam/v2/models"))

    fname = f"{model_type}_sam2.1"
    pooch.retrieve(
        url=URLS[model_type],
        known_hash=HASHES[model_type],
        fname=fname,
        path=save_directory,
        progressbar=True
    )

    checkpoint_path = os.path.join(save_directory, fname)
    return checkpoint_path


def _build_sam2_backbone(model_cfg, checkpoint_path, device, input_type):
    """Build the SAM2 image model or the SAM2 video predictor from a config and a checkpoint."""
    if input_type == "images":
        return build_sam2(
            config_file=model_cfg, ckpt_path=checkpoint_path, device=device, mode="eval", apply_postprocessing=False,
        )
    return _build_sam2_video_predictor(config_file=model_cfg, ckpt_path=checkpoint_path, device=device)


def _load_peft_finetuned_sam2(model_cfg, model_type, input_type, finetuned_checkpoint, device, peft_kwargs, state=None):
    """Build a SAM2 model with parameter efficient finetuning applied and load finetuned weights.

    A PEFT-finetuned checkpoint contains the injected PEFT parameters (e.g. LoRA layers), so the
    base backbone architecture cannot load it directly. This mirrors the SAM v1 loading path in
    `micro_sam.v1.util.get_sam_model`: the base backbone is built, the same PEFT surgery used during
    training is re-applied, and only then are the finetuned weights loaded on top.

    Args:
        model_cfg: The SAM2 model config path.
        model_type: The SAM2 model name (the base backbone is derived from the first 6 characters).
        input_type: Whether the inputs are images or videos.
        finetuned_checkpoint: Path to the PEFT-finetuned checkpoint.
        device: The pytorch device.
        peft_kwargs: Keyword arguments for `micro_sam.v2.models.peft_sam2.PEFT_Sam2`.
        state: An already-loaded checkpoint (to avoid reloading it). Loaded from `finetuned_checkpoint` if None.

    Returns:
        The PEFT SAM2 model with the finetuned weights loaded.
    """
    from micro_sam.v2.models.peft_sam2 import PEFT_Sam2

    # Build from the base backbone; the finetuned weights (with the PEFT parameters) are loaded after
    # the surgery so that the checkpoint keys match the wrapped architecture.
    base_checkpoint = _get_checkpoint(model_type=model_type[:6])
    model = _build_sam2_backbone(model_cfg, base_checkpoint, device, input_type)
    model = PEFT_Sam2(model, **peft_kwargs).sam

    if state is None:
        state = torch.load(finetuned_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:  # Exported micro-sam / native SAM2 layout.
        model_state = state["model"]
    elif isinstance(state, dict) and "model_state" in state:  # Raw torch-em trainer checkpoint.
        model_state = state["model_state"]
    else:
        model_state = state
    # Strip a DistributedDataParallel 'module.' prefix if the checkpoint was saved under DDP.
    model_state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in model_state.items()}

    try:
        model.load_state_dict(model_state)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load the finetuned PEFT weights. This usually means the given 'peft_kwargs' do not "
            "match the ones used at training time (e.g. a different rank or PEFT method)."
        ) from e
    model.to(device)
    model.eval()
    return model


def get_sam2_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    input_type: Literal["images", "videos"] = "images",
    peft_kwargs: Optional[dict] = None,
):
    """Get the Segment Anything 2 (SAM2) model for interactive segmentation of images and videos.

    Args:
        model_type: The choice of size for the vision transformer, eg. `hvit_t`. The default is `hvit_t` model.
        device: The pytorch device.
        checkpoint_path: Filepath to the pretrained model weights.
        input_type: Whether the inputs are images or videos.
        peft_kwargs: Keyword arguments for `micro_sam.v2.models.peft_sam2.PEFT_Sam2`. If given, the model
            is loaded as a PEFT-finetuned model, i.e. the base backbone is built, the PEFT surgery is
            re-applied, and the finetuned weights are loaded on top. If not given, a PEFT config saved in
            a user-provided `checkpoint_path` (see `get_sam2_train_model`) is auto-detected and applied.

    Returns:
        The SAM2 model.
    """
    # The base SAM2 backbone is the first 6 characters of the name, e.g. 'hvit_t_cells' -> 'hvit_t'.
    # Finetuned micro-sam weights come from the registry rather than the base SAM2 download.
    is_finetuned = model_type in FINETUNED_MODELS
    model_cfg = CFG_PATHS[model_type[:6]]

    device = _get_device(device)

    if input_type not in ("images", "videos"):
        raise ValueError(f"'{input_type}' is not a valid input type.")

    # Only a user-provided checkpoint can carry a saved PEFT config; the base / registered downloads
    # never do, so we avoid loading them twice for the common (non-PEFT) path.
    user_provided_checkpoint = checkpoint_path is not None

    if checkpoint_path is None:
        if is_finetuned:
            checkpoint_path, _, _ = _download_finetuned_sam2_model(model_type)
        else:
            checkpoint_path = _get_checkpoint(model_type=model_type)

    # If the caller did not pass peft_kwargs, auto-detect a PEFT config saved in the checkpoint.
    saved_state = None
    if not peft_kwargs and user_provided_checkpoint:
        from micro_sam.models.peft import deserialize_peft_kwargs
        from micro_sam.v2.models.peft_sam2 import PEFT_MODULES
        saved_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(saved_state, dict) and saved_state.get("peft_kwargs") is not None:
            peft_kwargs = deserialize_peft_kwargs(saved_state["peft_kwargs"], PEFT_MODULES)
        else:
            saved_state = None  # Not a PEFT checkpoint; let the build function load it normally.

    if peft_kwargs and isinstance(peft_kwargs, dict):
        # We do not quantize at inference; a QLoRA-trained model is loaded in full precision (as in
        # `micro_sam.v1.util.get_sam_model`). Copy first so the caller's dict is not mutated.
        peft_kwargs = {k: v for k, v in peft_kwargs.items() if k != "quantize"}
        model = _load_peft_finetuned_sam2(
            model_cfg, model_type, input_type, checkpoint_path, device, peft_kwargs, state=saved_state,
        )
    else:
        model = _build_sam2_backbone(model_cfg, checkpoint_path, device, input_type)

    # Both predictor wrappers and direct model use need this metadata for embedding signatures.
    model.model_type = model_type
    model.model_name = model_type  # TODO: What is this exactly?

    return model


def export_custom_qlora_sam2_model(
    checkpoint_path: Optional[Union[str, os.PathLike]],
    finetuned_path: Union[str, os.PathLike],
    model_type: str,
    save_path: Union[str, os.PathLike],
) -> None:
    """Export a QLoRA-finetuned SAM2 model to a full-precision LoRA-style checkpoint.

    QLoRA freezes the 4-bit image encoder while the LoRA adapters and non-encoder SAM2 components are
    trained. The export keeps these finetuned tensors and reconstructs only the frozen encoder tensors
    from the pristine full-precision model, renaming LoRA-wrapped keys as needed. The exported checkpoint
    can then be loaded with the LoRA backbone by passing the corresponding `peft_kwargs` to
    `get_sam2_model` (or auto-detected if stored).

    Args:
        checkpoint_path: Path to the base SAM2 backbone the model was finetuned from (None -> default download).
        finetuned_path: Path to the QLoRA-finetuned checkpoint.
        model_type: The SAM2 model type, e.g. 'hvit_t'.
        save_path: Where to save the exported checkpoint.
    """
    # Step 1: The base (full-precision) SAM2 model that finetuning started from.
    sam = get_sam2_model(model_type=model_type, checkpoint_path=checkpoint_path, device="cpu")

    # Step 2: Load the QLoRA-finetuned checkpoint.
    ft_state = torch.load(finetuned_path, map_location="cpu", weights_only=False)
    if isinstance(ft_state, dict) and "model_state" in ft_state:
        ft_model_state = ft_state["model_state"]
    elif isinstance(ft_state, dict) and "model" in ft_state:
        ft_model_state = ft_state["model"]
    else:
        ft_model_state = ft_state
    ft_model_state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in ft_model_state.items()}

    # Step 3: Keep the trained non-encoder parameters and full-precision LoRA layers, recording which
    # blocks have LoRA on the attention and/or feed forward layers.
    updated_model_state = {k: v for k, v in ft_model_state.items() if not k.startswith("image_encoder.")}
    modified_attn_layers = set()
    modified_mlp_layers = set()
    for k, v in ft_model_state.items():
        if not k.startswith("image_encoder."):
            continue
        layer_id = int(k.split("blocks.")[1].split(".")[0]) if "blocks." in k else None
        if k.find("qkv.w_a_linear") != -1 or k.find("qkv.w_b_linear") != -1:
            modified_attn_layers.add(layer_id)
            updated_model_state[k] = v
        if k.find("mlp.w_a_linear") != -1 or k.find("mlp.w_b_linear") != -1:
            modified_mlp_layers.add(layer_id)
            updated_model_state[k] = v

    # Step 4: Reconstruct the frozen image encoder from the base model, renaming the LoRA-wrapped keys so
    # the frozen base qkv lives under 'qkv.qkv_proj' and the frozen base MLP under 'mlp.mlp_layer'.
    for k, v in sam.state_dict().items():
        if not k.startswith("image_encoder."):
            continue
        layer_id = int(k.split("blocks.")[1].split(".")[0]) if "blocks." in k else None
        if k.find("attn.qkv.") != -1:
            if layer_id in modified_attn_layers:
                k = k.replace("qkv", "qkv.qkv_proj")
        elif k.find("mlp") != -1 and k.find("image_encoder") != -1:
            if layer_id in modified_mlp_layers:
                k = k.replace("mlp.", "mlp.mlp_layer.")
        updated_model_state[k] = v

    # Step 5: Replace the model state (retaining other checkpoint entries, e.g. a stored peft config).
    if isinstance(ft_state, dict) and "model_state" in ft_state:
        ft_state["model_state"] = updated_model_state
        out_state = ft_state
    elif isinstance(ft_state, dict) and "model" in ft_state:
        ft_state["model"] = updated_model_state
        out_state = ft_state
    else:
        out_state = {"model": updated_model_state, "model_type": model_type}

    # Step 6: Store the exported checkpoint.
    torch.save(out_state, save_path)


class _PrecisionImagePredictor(SAM2ImagePredictor):
    """A SAM2 image predictor whose prompt encoder and mask decoder run in the device precision."""

    def _predict(self, *args, **kwargs):
        with autocast(self.device):
            return to_float32(super()._predict(*args, **kwargs))


def configure_image_predictor(predictor):
    """Configure a SAM2 image predictor to always use resize-longest."""
    from micro_sam.v2.transforms.resize import ResizeLongestSideTransforms

    old = predictor._transforms
    predictor._transforms = ResizeLongestSideTransforms(
        resolution=predictor.model.image_size,
        mask_threshold=predictor.mask_threshold,
        max_hole_area=getattr(old, "max_hole_area", 0.0),
        max_sprinkle_area=getattr(old, "max_sprinkle_area", 0.0),
    )
    return predictor


def get_sam2_image_predictor(model, **kwargs):
    """Build a SAM2 image predictor with resize-longest preprocessing and the device precision."""
    return configure_image_predictor(_PrecisionImagePredictor(model, **kwargs))


def _predictor_device(predictor) -> torch.device:
    """The device a predictor runs on, whether it exposes it as an attribute or as a method."""
    device = getattr(predictor, "device", None)
    if callable(device):
        device = device()
    if device is None:
        device = next(predictor.parameters()).device
    return torch.device(device)


def _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo, preprocessing):
    """Validate saved embeddings against the requested configuration.

    Returns True if the saved embeddings are stale and should be recomputed (the model, tiling or
    preprocessing changed), False if they can be loaded. Raises if they belong to different image
    data (data signature mismatch). `preprocessing` is the policy expected for the current path
    (2d image or 3d / video, see `micro_sam.v2.normalization`).
    """
    # We can have an empty zarr file that was already created to save the embeddings in. A
    # feature-bearing file without the completion metadata is a partial cache: reject it unless it
    # explicitly records the current preprocessing policy. Untagged caches can use the former video
    # resize implementation and cannot be safely resumed.
    if "input_size" not in f.attrs:
        normalization = f.attrs.get("normalization")
        return "features" in f and normalization != preprocessing

    # Creates all the metadata that is stored along with the embeddings.
    # TODO: This is currently paired with `micro_sam`-level metadata. Should we get separate for `micro_sam.v2`?
    from micro_sam.util import _get_embedding_signature
    signature = _get_embedding_signature(input_, predictor, tile_shape, halo)
    signature["normalization"] = preprocessing
    signature["precision"] = precision_name(_predictor_device(predictor))

    stale = False
    for key, val in signature.items():
        # Missing current preprocessing metadata means stale. We still tolerate other legacy signature fields.
        if key not in f.attrs:
            if key in ("normalization", "precision"):
                stale = True
            continue
        if f.attrs[key] == val:
            continue
        # Different image data: surface as an error rather than silently overwriting it.
        if key == "data_signature":
            raise RuntimeError(
                f"Embeddings file {save_path} is invalid due to mismatch in {key}: "
                f"{f.attrs.get(key)} != {val}. Please recompute embeddings in a new file."
            )
        # A version bump alone does not invalidate the embeddings.
        if key == "micro_sam_version":
            warnings.warn(
                f"The signature for {key} in embeddings file {save_path} has a mismatch: "
                f"{f.attrs.get(key)} != {val}. This key was recently added, so your embeddings are likely correct. "
                "But please recompute them if model predictions don't look as expected."
            )
            continue
        # Model, tiling or normalization changed: the saved embeddings are stale and must be recomputed.
        stale = True
    return stale


def _write_embedding_signature(f, input_, predictor, tile_shape, halo, input_size, original_size, preprocessing):
    """Write the common embedding metadata plus the SAM2 preprocessing policy and precision."""
    from micro_sam.util import _write_embedding_signature as _write_common_signature

    _write_common_signature(f, input_, predictor, tile_shape, halo, input_size, original_size)
    f.attrs["normalization"] = preprocessing
    # The encoder precision changes the stored values, so another one is stale.
    f.attrs["precision"] = precision_name(_predictor_device(predictor))


def _compute_2d(input_, predictor, f, save_path, pbar_init, pbar_update):
    # Check if the embeddings are already cached.
    if save_path is not None and "original_size" in f.attrs:
        # In this case we load the embeddings.
        features = f["features"][:]
        original_size = f.attrs["original_size"]
        input_size = f.attrs["input_size"]
        # The high-resolution features are stored as a list of datasets and are needed by the decoder.
        high_res_features = _load_list_datasets(f, "high_res_feats", lazy_loading=False)
        image_embeddings = {
            "features": features,
            "high_res_feats": high_res_features,
            "input_size": input_size,
            "original_size": original_size,
        }
        # Also set the embeddings.
        set_precomputed(predictor, image_embeddings)
        return image_embeddings

    pbar_init(1, "Compute Image Embeddings 2D")
    # Otherwise we have to compute the embeddings.
    predictor.reset_predictor()

    encode_image(predictor, to_image(input_))
    features = predictor.get_image_embedding().cpu().numpy()
    high_res_features = predictor._features.get("high_res_feats")
    original_size = predictor._orig_hw
    # SAM2ImagePredictor exposes the model resolution via its underlying model, unlike the
    # video predictor which subclasses SAM2Base and has 'image_size' directly.
    input_size = predictor.model.image_size
    pbar_update(1)

    # Save the embeddings if we have a save_path.
    if save_path is not None:
        from micro_sam.util import _create_dataset_with_data
        _create_dataset_with_data(f, "features", data=features)
        # Store the high-resolution features (a list of tensors) needed by the SAM2 decoder.
        high_res_group = f.require_group("high_res_feats")
        for i, feat in enumerate(high_res_features):
            _create_dataset_with_data(high_res_group, str(i), data=feat.cpu().numpy())
        _write_embedding_signature(
            f, input_, predictor, tile_shape=None, halo=None, input_size=input_size,
            original_size=original_size, preprocessing=IMAGE_PREPROCESSING,
        )

    image_embeddings = {
        "features": features,
        "high_res_feats": high_res_features,
        "input_size": input_size,
        "original_size": original_size,
    }
    return image_embeddings


def _load_list_datasets(group, prefix_name, lazy_loading):
    if prefix_name not in group:
        return []

    subgroup = group[prefix_name]
    out = []
    i = 0
    while str(i) in subgroup:
        ds = subgroup[str(i)]
        out.append(ds if lazy_loading else ds[:])
        i += 1
    return out


def precompute_image_embeddings(
    predictor,
    input_: np.ndarray,
    save_path: Optional[Union[str, os.PathLike]] = None,
    lazy_loading: bool = False,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
):
    """Compute the image embeddings (output of the encoder) for the input.

    If 'save_path' is given the embeddings will be loaded/saved in a zarr container.

    Args:
        predictor: The SAM2 image or video predictor.
        input_: The input image or volume.
        save_path: Optional zarr path for persistent embedding storage.
        lazy_loading: Stream embeddings from zarr instead of materializing them in memory. Without
            `save_path`, the returned `ImageEmbeddings` owns an ephemeral on-disk store; use it as a
            context manager or call `close()` when finished.
        ndim: The number of spatial dimensions. By default this is inferred from the input.
        tile_shape: Optional in-plane tile shape.
        halo: Optional in-plane tile halo.
        verbose: Whether to show progress.
        batch_size: The batch size used when running inference for multiple slices and / or tiles.
            By default it is looked up independently on each CUDA device from its free VRAM (see
            `recommend_batch_size`). Pass an integer to run every device at that batch size.
        devices: Device or devices used for embedding inference. If None and the predictor is on
            CUDA, all visible CUDA devices are used.
        num_prefetch_workers: Number of threads used to read and preprocess input jobs.
        num_write_workers: Number of threads used to write the embeddings. Only has an effect for
            volumes and tiled images, which are written incrementally.
        pbar_init: Optional callback to initialize external progress.
        pbar_update: Optional callback to update external progress.

    Returns:
        An `ImageEmbeddings` resource. Call `close()` when finished or use it as a context manager.
    """
    ndim = input_.ndim if ndim is None else ndim
    preprocessing = IMAGE_PREPROCESSING if ndim == 2 else VIDEO_PREPROCESSING
    if ndim == 2:
        configure_image_predictor(predictor)

    is_streamed = ndim == 3 or tile_shape is not None
    temporary_path = None
    if lazy_loading and is_streamed and save_path is None:
        # Without a save path the zarr is held in memory, which defeats lazy loading. Back it by an
        # ephemeral on-disk store so the tiles / slices are streamed from disk instead.
        temporary_path = save_path = make_temp_embedding_path()

    # Handle the embedding save_path.
    # We don't have a save path, open in memory zarr file to hold tiled embeddings.
    if save_path is None:
        f = _open_embeddings(None)

    # We have a save path and it already exists. Embeddings will be loaded from it,
    # check that the saved embeddings in there match the parameters of the function call.
    elif os.path.exists(save_path):
        f = _open_embeddings(save_path, mode="a")
        if _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo, preprocessing):
            # Close the old handle before truncating the store.
            getattr(f, "file", f).close()
            f = _open_embeddings(save_path, mode="w")

    # We have a save path and it does not exist yet. Create the zarr file to which the
    # embeddings will then be saved.
    else:
        f = _open_embeddings(save_path, mode="a")

    # Persist the policy before writing any feature data, so an interrupted partial cache can only
    # be resumed under the same preprocessing policy.
    if save_path is not None:
        f.attrs["normalization"] = preprocessing

    from micro_sam.util import handle_pbar
    from micro_sam.v2.batched_inference import _compute_3d, _compute_tiled_2d, _compute_tiled_3d

    _, pbar_init, pbar_update, pbar_close = handle_pbar(verbose, pbar_init, pbar_update)

    resource = ImageEmbeddings({}, store=f, temporary_path=temporary_path)
    if ndim == 2 and tile_shape is None:
        embeddings = _compute_2d(input_, predictor, f, save_path, pbar_init, pbar_update)
    elif ndim == 2 and tile_shape is not None:
        if halo is None:
            raise ValueError("To compute tiled embeddings the parameter halo has to be passed.")
        embeddings = _compute_tiled_2d(
            input_, predictor, tile_shape, halo, f, save_path, pbar_init, pbar_update,
            batch_size=batch_size, devices=devices, num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
        )
    elif ndim == 3 and tile_shape is None:
        embeddings = _compute_3d(
            input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update,
            batch_size=batch_size, devices=devices, num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
        )
    elif ndim == 3 and tile_shape is not None:
        if halo is None:
            raise ValueError("To compute tiled embeddings the parameter halo has to be passed.")
        embeddings = _compute_tiled_3d(
            input_, predictor, tile_shape, halo, f, save_path, pbar_init, pbar_update,
            batch_size=batch_size, devices=devices, num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
        )
    else:
        raise ValueError(f"Invalid dimensionality {input_.ndim}, expect 2 or 3 dim data.")

    pbar_close()
    uses_store = tile_shape is not None or (ndim == 3 and lazy_loading)
    if not uses_store:
        resource.close()
    resource.update(embeddings)
    return resource


def _shared_pos_enc(level):
    """Read the positional encoding shared by every slice. Unlike 'features' and 'fpn', it is stored once.

    Args:
        level: One stored positional-encoding level, shaped (1, 1, C, H, W).

    Returns:
        The encoding for any slice, shaped (1, C, H, W).
    """
    return level[0]


def _backbone_fpn(fpn_levels, features):
    """Rebuild the encoder's FPN levels, whose last one is stored as 'features' rather than twice.

    Args:
        fpn_levels: The stored FPN levels for one slice, all but the last.
        features: That slice's 'features', which is the missing last level.

    Returns:
        The full list of FPN levels, in the order the encoder produced them.
    """
    return list(fpn_levels) + [features]


def _to_device_tensor(data, device):
    """Convert embedding data (numpy array or torch tensor on any device) to a float tensor.

    Freshly computed embeddings may still be tensors on MPS/CUDA, which 'np.asarray' cannot
    convert; move those to the target device directly instead of via numpy.

    Args:
        data: The embedding data, either a numpy array or a torch tensor on any device.
        device: The target device for the returned tensor.

    Returns:
        The data as a float tensor on the given device.
    """
    if torch.is_tensor(data):
        return data.detach().to(device).float()
    return torch.as_tensor(np.asarray(data), device=device).float()


def set_precomputed(predictor, image_embeddings, i: Optional[int] = None, tile_id: Optional[int] = None):
    """Set precomputed image embeddings on a SAM2 image predictor.

    Only 2d embeddings are set this way. A volume's embeddings are not: the video predictor reads
    them per frame while it tracks, see `CustomVideoPredictor.init_state`.

    Args:
        predictor: The SAM2 image predictor to set the embeddings on.
        image_embeddings: The precomputed embeddings, as returned by `precompute_image_embeddings`.
        i: The slice index, which 2d embeddings do not have. Passing one raises, so that a volumetric
            call routed here by mistake fails instead of segmenting the wrong thing.
        tile_id: The tile to set, for tiled embeddings. That tile's features are read from the store
            and set as if they were those of a whole image.

    Returns:
        The predictor, with the embeddings set on it.
    """
    if tile_id is not None:
        tile_features = image_embeddings["features"][str(tile_id)]
        # The SAM2 image predictor also needs the high-resolution features (used by the decoder),
        # which are stored per tile under 'high_res_feats/{tile_id}/{level}'.
        high_res_feats = _load_list_datasets(image_embeddings["high_res_feats"], str(tile_id), lazy_loading=False)
        tile_image_embeddings = {
            "features": tile_features,
            "high_res_feats": high_res_feats,
            "input_size": tile_features.attrs["input_size"],
            "original_size": tile_features.attrs["original_size"],
        }
        return set_precomputed(predictor, tile_image_embeddings, i=i)

    try:
        device = predictor.device()  # Works for video predictor.
    except TypeError:
        device = predictor.device  # Otherwise, for image predictor.

    features = image_embeddings["features"]
    if features.ndim != 4:
        raise ValueError(
            f"Expected 2d embeddings, whose features have 4 dimensions, got {features.ndim}. The "
            "embeddings of a volume are read by the video predictor's 'init_state' instead."
        )
    if i is not None:
        raise ValueError("The data is 2D so an index is not needed.")

    # Convert to tensors on the predictor device, as 'predictor.set_image' would for the decoder.
    image_embed = _to_device_tensor(features, device)
    high_res_feats = [_to_device_tensor(feat, device) for feat in image_embeddings["high_res_feats"]]
    predictor._features = {"image_embed": image_embed, "high_res_feats": high_res_feats}
    predictor._is_image_set = True
    predictor._orig_hw = image_embeddings["original_size"]
    return predictor
