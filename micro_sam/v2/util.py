import os
import sys
import pooch
import warnings
from pathlib import Path
from typing import Union, Literal, Optional, Tuple

import numpy as np

import torch

from micro_sam.util import (
    get_device, get_cache_directory, microsam_cachedir, _open_embeddings, _create_dataset_without_data,
    _configure_mps_memory,
)
from micro_sam.v2.models._video_predictor import _build_sam2_video_predictor
from micro_sam.v2.normalization import RAW_NORMALIZATION, to_image

import sam2
from sam2.build_sam import build_sam2


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


# Default in-plane tiling for large images. Tiling is enabled when an in-plane axis exceeds
# DEFAULT_TILING_THRESHOLD; the SAM input patch per axis is then DEFAULT_TILE_SHAPE + 2 * DEFAULT_HALO,
# which is kept equal to the threshold (512 + 2 * 128 = 768).
DEFAULT_TILING_THRESHOLD = 768
DEFAULT_TILE_SHAPE = (512, 512)
DEFAULT_HALO = (128, 128)

# Default z block / halo for volumetric (3d) tiling. Each decoder pass spans the inner block plus the
# halo on each side, i.e. DEFAULT_TILE_Z + 2 * DEFAULT_HALO_Z = 8 slices, matching the UniSAM2 8-slice
# training crop (so the z-convolutions see the z-context they were trained on; do not enlarge this
# beyond the training crop). Set the z tile >= the slice count to disable z-tiling.
DEFAULT_TILE_Z = 4
DEFAULT_HALO_Z = 2


def needs_default_tiling(shape):
    """Whether default in-plane tiling should be enabled for a given image shape.

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
# truth for the default; the GUI derives its synthetic 'vit_<size><suffix>' selector string from it.
DEFAULT_MODEL = "hvit_t_cells"

FINETUNED_URLS = {
    "hvit_t_cells": "https://owncloud.gwdg.de/index.php/s/PJRPRXC3BNOLJ6X/download",
    "hvit_t_cells_decoder": "https://owncloud.gwdg.de/index.php/s/URqdbdzJiUtUiq1/download",
}

FINETUNED_HASHES = {
    "hvit_t_cells": "xxh128:385a8521cbadad2536b2e7950c394f80",
    "hvit_t_cells_decoder": "xxh128:842add10a67e4c7827d97f033e62a6f5",
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

    if device == "cuda":
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


def get_sam2_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    input_type: Literal["images", "videos"] = "images",
):
    """Get the Segment Anything 2 (SAM2) model for interactive segmentation of images and videos.

    Args:
        model_type: The choice of size for the vision transformer, eg. `hvit_t`. The default is `hvit_t` model.
        device: The pytorch device.
        checkpoint_path: Filepath to the pretrained model weights.
        input_type: Whether the inputs are images or videos.

    Returns:
        The SAM2 model.
    """
    # The base SAM2 backbone is the first 6 characters of the name, e.g. 'hvit_t_cells' -> 'hvit_t';
    # finetuned micro-sam weights come from the registry rather than the base SAM2 download.
    is_finetuned = model_type in FINETUNED_MODELS
    model_cfg = CFG_PATHS[model_type[:6]]

    device = _get_device(device)

    if input_type == "images":
        _build_segment_anything_2 = build_sam2
    elif input_type == "videos":
        _build_segment_anything_2 = _build_sam2_video_predictor
    else:
        raise ValueError(f"'{input_type}' is not a valid input type.")

    if checkpoint_path is None:
        if is_finetuned:
            checkpoint_path, _, _ = _download_finetuned_sam2_model(model_type)
        else:
            checkpoint_path = _get_checkpoint(model_type=model_type)

    model = _build_segment_anything_2(
        config_file=model_cfg,
        ckpt_path=checkpoint_path,
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )

    if input_type == "videos":
        model.model_type = model_type
        model.model_name = model_type  # TODO: What is this exactly?

    return model


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
    """Build a SAM2 image predictor with resize-longest preprocessing."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return configure_image_predictor(SAM2ImagePredictor(model, **kwargs))


def _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo):
    """Validate saved embeddings against the requested configuration.

    Returns True if the saved embeddings are stale and should be recomputed (the model, tiling or
    normalization changed), False if they can be loaded. Raises if they belong to different image
    data (data signature mismatch).
    """
    # We may have an empty zarr file that was already created to save the embeddings in. A
    # feature-bearing file without the completion metadata is a partial cache: only resume it when
    # it already records the current normalization policy. Otherwise it may contain legacy min-max
    # features, which must not be mixed with newly computed percentile-normalized features.
    if "input_size" not in f.attrs:
        return "features" in f and f.attrs.get("normalization") != RAW_NORMALIZATION

    # Creates all the metadta that is stored along with the embeddings.
    # TODO: This is currently paired with `micro_sam`-level metadata. Should we get separate for `micro_sam.v2`?
    from micro_sam.util import _get_embedding_signature
    signature = _get_embedding_signature(input_, predictor, tile_shape, halo)
    signature["normalization"] = RAW_NORMALIZATION

    stale = False
    for key, val in signature.items():
        # Embeddings without normalization metadata used the former min-max policy.
        if key not in f.attrs:
            stale = stale or key == "normalization"
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


def _write_embedding_signature(f, input_, predictor, tile_shape, halo, input_size, original_size):
    """Write the common embedding metadata plus the SAM2 normalization policy."""
    from micro_sam.util import _write_embedding_signature as _write_common_signature

    _write_common_signature(f, input_, predictor, tile_shape, halo, input_size, original_size)
    f.attrs["normalization"] = RAW_NORMALIZATION


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

    predictor.set_image(to_image(input_))
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
            f, input_, predictor, tile_shape=None, halo=None, input_size=input_size, original_size=original_size,
        )

    image_embeddings = {
        "features": features,
        "high_res_feats": high_res_features,
        "input_size": input_size,
        "original_size": original_size,
    }
    return image_embeddings


def _compute_tiled_2d(input_, predictor, tile_shape, halo, f, save_path, pbar_init, pbar_update):
    from micro_sam.util import _create_dataset_with_data
    from bioimage_cpp.utils import Blocking

    features = f.require_group("features")
    high_res_group = f.require_group("high_res_feats")

    # If the tiled embeddings are already cached we just return the open groups.
    if save_path is not None and "shape" in features.attrs:
        return {"features": features, "high_res_feats": high_res_group, "input_size": None, "original_size": None}

    tiling = Blocking([0, 0], list(input_.shape[:2]), list(tile_shape))
    n_tiles = tiling.number_of_blocks

    features.attrs["shape"] = list(input_.shape[:2])
    features.attrs["tile_shape"] = list(tile_shape)
    features.attrs["halo"] = list(halo)

    pbar_init(n_tiles, "Compute Image Embeddings 2D tiled")
    predictor.reset_predictor()
    for tile_id in range(n_tiles):
        block = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
        bb = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))
        predictor.set_image(to_image(input_[bb]))

        tile_features = predictor.get_image_embedding().cpu().numpy()
        high_res_features = [feat.cpu().numpy() for feat in predictor._features["high_res_feats"]]
        ds = _create_dataset_with_data(features, str(tile_id), data=tile_features)
        ds.attrs["input_size"] = predictor.model.image_size
        # Store the original size in the predictor's nested '[[h, w]]' layout (one entry per image),
        # so 'set_precomputed' restores '_orig_hw[-1] == (h, w)' for non-square tiles too.
        ds.attrs["original_size"] = [list(predictor._orig_hw[0])]

        tile_high_res = high_res_group.require_group(str(tile_id))
        for level, feat in enumerate(high_res_features):
            _create_dataset_with_data(tile_high_res, str(level), data=feat)

        predictor.reset_predictor()
        pbar_update(1)

    if save_path is not None:
        _write_embedding_signature(
            f, input_, predictor, tile_shape=tile_shape, halo=halo, input_size=None, original_size=None,
        )

    return {"features": features, "high_res_feats": high_res_group, "input_size": None, "original_size": None}


def _compute_tiled_3d(input_, predictor, tile_shape, halo, f, save_path, pbar_init, pbar_update):
    from bioimage_cpp.utils import Blocking

    features = f.require_group("features")
    pos_enc_group = f.require_group("pos_enc")
    fpn_group = f.require_group("fpn")

    # If the tiled embeddings are already cached we just return the open groups.
    if save_path is not None and "shape" in features.attrs:
        return {
            "features": features, "pos_enc": pos_enc_group, "fpn": fpn_group,
            "input_size": None, "original_size": None,
        }

    # The volume is tiled in-plane (xy); each tile is its own (Z, tile_y, tile_x) sub-volume.
    tiling = Blocking([0, 0], list(input_.shape[1:]), list(tile_shape))
    n_tiles = tiling.number_of_blocks
    n_slices = input_.shape[0]

    features.attrs["shape"] = list(input_.shape)
    features.attrs["tile_shape"] = list(tile_shape)
    features.attrs["halo"] = list(halo)

    # Progress is reported per actual patch (tile-column x z slice), not per tile, since each tile
    # encodes all z slices and that inner loop is the bulk of the work.
    pbar_init(n_tiles * n_slices, "Compute Image Embeddings 3D tiled")
    for tile_id in range(n_tiles):
        block = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
        bb = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))
        sub_volume = np.asarray(input_[:, bb[0], bb[1]])

        # Compute the per-slice video-style features for this tile-column (as in '_compute_3d').
        inference_state = predictor.init_state(
            volume=sub_volume, volume_embeddings=None, device=predictor.device, ignore_caching_features=True,
        )

        # Stream one slice at a time into per-tile datasets, moving each slice's features off-device
        # right away, so peak memory is a single slice rather than the whole tile-column. Buffering all
        # slices (and then stacking them) held ~200 MB/slice of high-res features on-device and OOMed on
        # deep volumes even when saving to disk. Datasets are created lazily from the first slice's
        # shapes with per-slice '(1, ...)' chunking, matching '_compute_3d' and the on-disk layout the
        # lazy tiled consumer expects (features/<tile>, pos_enc/<tile>/<level>, fpn/<tile>/<level>).
        feature_ds, pos_enc_dsets, fpn_dsets = None, None, None
        tile_pos_group = pos_enc_group.require_group(str(tile_id))
        tile_fpn_group = fpn_group.require_group(str(tile_id))
        input_size, original_size = None, None
        for z in range(n_slices):
            vision_feats, pos_encs, fpns, original_sizes, input_sizes = _compute_embeddings_batched_3d(
                inference_state, predictor, [z], [to_image(sub_volume[z])], pbar_update=pbar_update,
            )
            curr_feat = vision_feats[0]
            if curr_feat.ndim == 3:
                curr_feat = curr_feat.unsqueeze(0)
            curr_pos, curr_fpn = pos_encs[0], fpns[0]

            if feature_ds is None:
                feature_ds = _create_dataset_without_data(
                    features, str(tile_id), shape=(n_slices,) + tuple(curr_feat.shape),
                    dtype="float32", chunks=(1,) + tuple(curr_feat.shape),
                )
                pos_enc_dsets = [
                    _create_dataset_without_data(
                        tile_pos_group, str(level), shape=(n_slices,) + tuple(t.shape),
                        dtype="float32", chunks=(1,) + tuple(t.shape),
                    ) for level, t in enumerate(curr_pos)
                ]
                fpn_dsets = [
                    _create_dataset_without_data(
                        tile_fpn_group, str(level), shape=(n_slices,) + tuple(t.shape),
                        dtype="float32", chunks=(1,) + tuple(t.shape),
                    ) for level, t in enumerate(curr_fpn)
                ]

            feature_ds[z] = curr_feat.detach().cpu().numpy()
            for level, t in enumerate(curr_pos):
                pos_enc_dsets[level][z] = t.detach().cpu().numpy()
            for level, t in enumerate(curr_fpn):
                fpn_dsets[level][z] = t.detach().cpu().numpy()
            input_size, original_size = input_sizes[-1], original_sizes[-1]

        feature_ds.attrs["input_size"] = input_size
        feature_ds.attrs["original_size"] = list(original_size)

    if save_path is not None:
        _write_embedding_signature(
            f, input_, predictor, tile_shape=tile_shape, halo=halo, input_size=None, original_size=None,
        )

    return {
        "features": features, "pos_enc": pos_enc_group, "fpn": fpn_group,
        "input_size": None, "original_size": None,
    }


def _create_list_dataset_without_data(group, prefix_name, tensors, dtype, z_slices):
    subgroup = group.require_group(prefix_name)

    ds_list = []
    for i, curr_tensor in enumerate(tensors):
        curr_shape = tuple(curr_tensor.shape)
        shape = (z_slices,) + curr_shape
        chunks = (1,) + curr_shape
        name = str(i)

        if name in subgroup:
            ds = subgroup[name]
            if ds.shape != shape:
                raise RuntimeError(f"Invalid shape for {prefix_name}/{name}: expected {shape}, got {ds.shape}")
            if getattr(ds, "chunks", None) is not None and ds.chunks != chunks:
                raise RuntimeError(f"Invalid chunks for {prefix_name}/{name}: expected {chunks}, got {ds.chunks}")
        else:
            ds = _create_dataset_without_data(subgroup, name, shape=shape, dtype=dtype, chunks=chunks)

        ds_list.append(ds)

    return ds_list


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


@torch.no_grad
def _compute_embeddings_batched_3d(inference_state, predictor, batched_z, batched_images, pbar_update=None):
    batched_vision_features, batched_pos_enc, batched_backbone_fpn, original_sizes, input_sizes = [], [], [], [], []

    for image, z_id in zip(batched_images, batched_z):
        # Run the image encoder to extract relevant features
        predictor._get_image_feature(inference_state, frame_idx=z_id, batch_size=1)

        # Let's extract the current 'cached_features' outputs
        _, curr_backbone_out = inference_state["cached_features"][z_id]

        # Store the vision transformer outputs and other stuff.
        batched_vision_features.append(curr_backbone_out["vision_features"])
        batched_pos_enc.append(curr_backbone_out["vision_pos_enc"])
        batched_backbone_fpn.append(curr_backbone_out["backbone_fpn"])
        original_sizes.append(image.shape[:2])
        input_sizes.append(predictor.image_size)

        if pbar_update is not None:  # Advance per slice so a tile-column reports per-patch progress.
            pbar_update(1)

    return batched_vision_features, batched_pos_enc, batched_backbone_fpn, original_sizes, input_sizes


def _compute_3d(input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update, batch_size):
    # Check if the embeddings are already fully cached.
    if save_path is not None and "original_size" in f.attrs:
        # In this case we load the embeddings.
        features = f["features"] if lazy_loading else f["features"][:]
        pos_enc = _load_list_datasets(f, "pos_enc", lazy_loading)
        fpn = _load_list_datasets(f, "fpn", lazy_loading)
        original_size = f.attrs["original_size"]
        input_size = f.attrs["input_size"]
        image_embeddings = {
            "features": features,
            "pos_enc": pos_enc,
            "fpn": fpn,
            "input_size": input_size,
            "original_size": original_size,
        }
        return image_embeddings

    # Otherwise we have to compute the embeddings.

    # First check if we have a save path or not and set things up accordingly.
    if save_path is None:
        features, pos_encs, fpns = [], [], []
        save_features = False
        partial_features = False
    else:
        save_features = True
        embed_shape = (1, 256, 64, 64)
        shape = (input_.shape[0],) + embed_shape
        chunks = (1,) + embed_shape
        if "features" in f:
            partial_features = True
            features = f["features"]
            if features.shape != shape or features.chunks != chunks:
                raise RuntimeError("Invalid partial features")
        else:
            partial_features = False
            from micro_sam.util import _create_dataset_without_data
            features = _create_dataset_without_data(f, "features", shape=shape, chunks=chunks, dtype="float32")

    # We create the 'inference_state' object which keeps all important components in memory.
    # Pass the predictor's device so encoder inputs match the model when it is not the default device.
    inference_state = predictor.init_state(
        volume=input_,
        volume_embeddings=None,  # NOTE: It's a mandatory argument, but with the argument below, passing 'None' doesn't matter.  # noqa
        device=predictor.device,
        ignore_caching_features=True
    )

    # Initialize the pbar and batches.
    n_slices = input_.shape[0]
    pbar_init(n_slices, "Compute Image Embeddings 3D")
    n_batches = int(np.ceil(n_slices / batch_size))
    pos_enc_dsets, fpn_dsets = None, None

    for batch_id in range(n_batches):
        z_start = batch_id * batch_size
        z_stop = min(z_start + batch_size, n_slices)

        batched_images, batched_z = [], []
        for z in range(z_start, z_stop):
            # Skip feature computation in case of partial features in non-zero slice.
            if partial_features and np.count_nonzero(features[z]) != 0:
                continue

            tile_input = to_image(input_[z])
            batched_images.append(tile_input)
            batched_z.append(z)

        (
            batched_vision_features, batched_pos_enc, batched_backbone_fpn, original_sizes, input_sizes
        ) = _compute_embeddings_batched_3d(inference_state, predictor, batched_z, batched_images)

        for z, curr_vision_feats, curr_pos_enc, curr_back_fpn in zip(
            batched_z, batched_vision_features, batched_pos_enc, batched_backbone_fpn
        ):
            if curr_vision_feats.ndim == 3:
                curr_vision_feats = curr_vision_feats.unsqueeze(0)

            if save_features:
                features[z] = curr_vision_feats.detach().cpu().numpy()
                if pos_enc_dsets is None:
                    pos_enc_dsets = _create_list_dataset_without_data(
                        f, "pos_enc", curr_pos_enc, dtype="float32", z_slices=n_slices
                    )
                for i, t in enumerate(curr_pos_enc):
                    pos_enc_dsets[i][z] = t.detach().cpu().numpy()

                if fpn_dsets is None:
                    fpn_dsets = _create_list_dataset_without_data(
                        f, "fpn", curr_back_fpn, dtype="float32", z_slices=n_slices
                    )
                for i, t in enumerate(curr_back_fpn):
                    fpn_dsets[i][z] = t.detach().cpu().numpy()

            else:
                features.append(curr_vision_feats)
                pos_encs.append(curr_pos_enc)
                fpns.append(curr_back_fpn)

            pbar_update(1)

    if save_features:
        _write_embedding_signature(
            f, input_, predictor, tile_shape=None, halo=None,
            input_size=input_sizes[-1], original_size=original_sizes[-1],
        )
    else:
        # Concatenate across the z axis for 'vision_features'.
        features = torch.cat(features).cpu().numpy()

        # Concatenate across the z axis for other features too.
        depth = 3  # Corresponds to the depth of both FPN and Positional Embeddings.
        pos_encs = [torch.stack([p[i] for p in pos_encs]) for i in range(depth)]
        fpns = [torch.stack([p[i] for p in fpns]) for i in range(depth)]

    pos_enc = _load_list_datasets(f, "pos_enc", lazy_loading) if save_features else pos_encs
    fpn = _load_list_datasets(f, "fpn", lazy_loading) if save_features else fpns

    image_embeddings = {
        "features": features,
        "pos_enc": pos_enc,
        "fpn": fpn,
        "input_size": input_sizes[-1],
        "original_size": original_sizes[-1],
    }
    return image_embeddings


def precompute_image_embeddings(
    predictor,
    input_: np.ndarray,
    save_path: Optional[Union[str, os.PathLike]] = None,
    lazy_loading: bool = False,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    batch_size: int = 1,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
):
    """Compute the image embeddings (output of the encoder) for the input.

    If 'save_path' is given the embeddings will be loaded/saved in a zarr container.

    Args:
        ...

    Returns:
        The image embeddings.
    """
    ndim = input_.ndim if ndim is None else ndim
    if ndim == 2:
        configure_image_predictor(predictor)

    # Handle the embedding save_path.
    # We don't have a save path, open in memory zarr file to hold tiled embeddings.
    if save_path is None:
        f = _open_embeddings(None)

    # We have a save path and it already exists. Embeddings will be loaded from it,
    # check that the saved embeddings in there match the parameters of the function call.
    elif os.path.exists(save_path):
        f = _open_embeddings(save_path, mode="a")
        if _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo):
            # Stale embeddings: truncate and recompute, overwriting them.
            f = _open_embeddings(save_path, mode="w")

    # We have a save path and it does not exist yet. Create the zarr file to which the
    # embeddings will then be saved.
    else:
        f = _open_embeddings(save_path, mode="a")

    # Persist the policy before writing any feature data. If embedding computation is interrupted,
    # the partial cache can then be resumed only under the same normalization policy.
    if save_path is not None:
        f.attrs["normalization"] = RAW_NORMALIZATION

    from micro_sam.util import handle_pbar
    _, pbar_init, pbar_update, pbar_close = handle_pbar(verbose, pbar_init, pbar_update)

    if ndim == 2 and tile_shape is None:
        embeddings = _compute_2d(input_, predictor, f, save_path, pbar_init, pbar_update)
    elif ndim == 2 and tile_shape is not None:
        if halo is None:
            raise ValueError("To compute tiled embeddings the parameter halo has to be passed.")
        embeddings = _compute_tiled_2d(input_, predictor, tile_shape, halo, f, save_path, pbar_init, pbar_update)
    elif ndim == 3 and tile_shape is None:
        embeddings = _compute_3d(input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update, batch_size)
    elif ndim == 3 and tile_shape is not None:
        if halo is None:
            raise ValueError("To compute tiled embeddings the parameter halo has to be passed.")
        embeddings = _compute_tiled_3d(input_, predictor, tile_shape, halo, f, save_path, pbar_init, pbar_update)
    else:
        raise ValueError(f"Invalid dimensionality {input_.ndim}, expect 2 or 3 dim data.")

    pbar_close()
    return embeddings


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


def set_precomputed(
    predictor,
    image_embeddings,
    i: Optional[int] = None,
    tile_id: Optional[int] = None,
    input_: Optional[np.ndarray] = None,
):
    """Set the precomputed image embeddings for a predictor.

    Args:
        ...

    Returns:
        ...
    """
    if tile_id is not None:
        tile_features = image_embeddings["features"][str(tile_id)]
        if "pos_enc" in image_embeddings:
            # 3D tiled embeddings: the per-tile positional encodings and FPN outputs (stored under
            # 'pos_enc/{tile_id}/{level}' and 'fpn/{tile_id}/{level}') are needed to set up the video
            # inference state for this tile-column. 'input_' must be the tile sub-volume.
            pos_enc = _load_list_datasets(image_embeddings["pos_enc"], str(tile_id), lazy_loading=False)
            fpn = _load_list_datasets(image_embeddings["fpn"], str(tile_id), lazy_loading=False)
            tile_image_embeddings = {
                "features": np.asarray(tile_features),
                "pos_enc": pos_enc,
                "fpn": fpn,
                "input_size": tile_features.attrs["input_size"],
                "original_size": tile_features.attrs["original_size"],
            }
            return set_precomputed(predictor, tile_image_embeddings, i=i, input_=input_)

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
    assert features.ndim in (4, 5), f"{features.ndim}"
    if features.ndim == 5:
        if i is None:
            raise ValueError("The data is 3D so an index i is needed.")

        if input_ is None:
            raise AssertionError("For 3D inputs, you must provide the original multi-dimensional array.")

        # Prepare the inference state
        inference_state = predictor.init_state(
            volume=input_, volume_embeddings=image_embeddings, ignore_caching_features=True,
        )

        # Get other visual features, eg. positional embeddings and FPN outputs to prepare 'backbone_out'.
        pos_list = image_embeddings["pos_enc"]
        fpn_list = image_embeddings["fpn"]

        # There's an easy assumption made here. The first dimension of 'features' corresponds to n-slices.
        running_features = {}
        for slice_id in range(features.shape[0]):
            image = inference_state["images"][slice_id].to(device).float().unsqueeze(0)
            vision_features = _to_device_tensor(features[slice_id], device)
            vision_pos_enc = [_to_device_tensor(t[slice_id], device) for t in pos_list]
            backbone_fpn = [_to_device_tensor(t[slice_id], device) for t in fpn_list]
            backbone_out = {
                "vision_features": vision_features, "vision_pos_enc": vision_pos_enc, "backbone_fpn": backbone_fpn,
            }
            running_features[slice_id] = (image, backbone_out)

        inference_state["cached_features"] = running_features
        return predictor, inference_state

    elif features.ndim == 4:
        if i is not None:
            raise ValueError("The data is 2D so an index is not needed.")

        # Convert to tensors on the predictor device, as 'predictor.set_image' would for the decoder.
        image_embed = _to_device_tensor(features, device)
        high_res_feats = [_to_device_tensor(feat, device) for feat in image_embeddings["high_res_feats"]]
        predictor._features = {"image_embed": image_embed, "high_res_feats": high_res_feats}
        predictor._is_image_set = True
        predictor._orig_hw = image_embeddings["original_size"]
        return predictor
