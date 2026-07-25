"""Precompute and cache image embeddings for image data (SAM1, SAM2 or VFM encoders).
"""

import os
import pickle
import inspect
from glob import glob
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from segment_anything.predictor import SamPredictor

import torch

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from . import util
from .v1 import instance_segmentation


def cache_amg_state(
    predictor: SamPredictor,
    raw: np.ndarray,
    image_embeddings: util.ImageEmbeddings,
    save_path: Union[str, os.PathLike],
    verbose: bool = True,
    i: Optional[int] = None,
    **kwargs,
) -> instance_segmentation.AutoSegBase:
    """Compute and cache or load the state for the automatic mask generator.

    Args:
        predictor: The Segment Anything predictor.
        raw: The image data.
        image_embeddings: The image embeddings.
        save_path: The embedding save path. The AMG state will be stored in 'save_path/amg_state.pickle'.
        verbose: Whether to run the computation verbose. By default, set to 'True'.
        i: The index for which to cache the state.
        kwargs: The keyword arguments for the amg class.

    Returns:
        The automatic mask generator class with the cached state.
    """
    is_tiled = image_embeddings["input_size"] is None
    amg = instance_segmentation.get_instance_segmentation_generator(predictor, is_tiled=is_tiled, **kwargs)

    # If i is given we compute the state for a given slice/frame.
    # And we have to save the state for slices/frames separately.
    if i is None:
        save_path_amg = os.path.join(save_path, "amg_state.pickle")
    else:
        os.makedirs(os.path.join(save_path, "amg_state"), exist_ok=True)
        save_path_amg = os.path.join(save_path, "amg_state", f"state-{i}.pkl")

    if os.path.exists(save_path_amg):
        if verbose:
            print("Load the AMG state from", save_path_amg)
        with open(save_path_amg, "rb") as f:
            amg_state = pickle.load(f)
        amg.set_state(amg_state)
        return amg

    if verbose:
        print("Precomputing the state for instance segmentation.")

    amg.initialize(raw if i is None else raw[i], image_embeddings=image_embeddings, verbose=verbose, i=i)
    amg_state = amg.get_state()

    # put all state onto the cpu so that the state can be deserialized without a gpu
    new_crop_list = []
    for mask_data in amg_state["crop_list"]:
        for k, v in mask_data.items():
            if torch.is_tensor(v):
                mask_data[k] = v.cpu()
        new_crop_list.append(mask_data)
    amg_state["crop_list"] = new_crop_list

    with open(save_path_amg, "wb") as f:
        pickle.dump(amg_state, f)

    return amg


def cache_is_state(
    predictor: SamPredictor,
    decoder: torch.nn.Module,
    raw: np.ndarray,
    image_embeddings: util.ImageEmbeddings,
    save_path: Union[str, os.PathLike],
    verbose: bool = True,
    i: Optional[int] = None,
    skip_load: bool = False,
    **kwargs,
) -> Optional[instance_segmentation.AutoSegBase]:
    """Compute and cache or load the state for the decoder-based instance segmentation.

    Args:
        predictor: The Segment Anything predictor.
        decoder: The instance segmentation decoder.
        raw: The image data.
        image_embeddings: The image embeddings.
        save_path: The embedding save path. The state will be stored in 'save_path/is_state.h5'.
        verbose: Whether to run the computation verbose. By default, set to 'True'.
        i: The index for which to cache the state.
        skip_load: Skip loading the state if it is precomputed. By default, set to 'False'.
        kwargs: The keyword arguments for the instance segmentation class.

    Returns:
        The instance segmentation class with the cached state.
    """
    is_tiled = image_embeddings["input_size"] is None
    amg = instance_segmentation.get_instance_segmentation_generator(
        predictor, is_tiled=is_tiled, decoder=decoder, **kwargs
    )

    # If i is given we compute the state for a given slice/frame.
    # And we have to save the state for slices/frames separately.
    save_path = os.path.join(save_path, "is_state.h5")
    save_key = "state" if i is None else f"state-{i}"

    with h5py.File(save_path, "a") as f:
        if save_key in f:
            if skip_load:  # Skip loading to speed this up for cases where we don't need the return val.
                return

            if verbose:
                print("Load instance segmentation state from", save_path, ":", save_key)
            g = f[save_key]
            state = {
                "foreground": g["foreground"][:],
                "boundary_distances": g["boundary_distances"][:],
                "center_distances": g["center_distances"][:],
            }
            amg.set_state(state)
            return amg

    if verbose:
        print("Precomputing the state for instance segmentation.")

    amg.initialize(raw, image_embeddings=image_embeddings, verbose=verbose, i=i)
    state = amg.get_state()

    with h5py.File(save_path, "a") as f:
        g = f.create_group(save_key)
        g.create_dataset("foreground", data=state["foreground"], compression="gzip")
        g.create_dataset("boundary_distances", data=state["boundary_distances"], compression="gzip")
        g.create_dataset("center_distances", data=state["center_distances"], compression="gzip")

    return amg


AUTOSEG_STATE_GROUP = "autoseg_state"
AUTOSEG_STATE_ATTRIBUTE = "autoseg_state"
AUTOSEG_STATE_VERSION = 1


def _autoseg_state_key(i):
    """Return the Zarr key for a whole-image/volume state or a per-slice state."""
    return "state" if i is None else f"state-{i}"


def _get_autoseg_state_group(embeddings, mode, create=False):
    """Return the mode-specific state group from an open embedding Zarr."""
    if mode not in ("amg", "ais"):
        raise ValueError(f"Invalid automatic segmentation mode: {mode}")

    if create:
        root = embeddings.require_group(AUTOSEG_STATE_GROUP)
        group = root.require_group(mode)
        group.attrs["version"] = AUTOSEG_STATE_VERSION
        return group

    if AUTOSEG_STATE_GROUP not in embeddings:
        return None
    root = embeddings[AUTOSEG_STATE_GROUP]
    return root.get(mode, None)


def _record_autoseg_state(embeddings, mode, group):
    """Record which automatic-segmentation states are present in the embedding metadata."""
    modes = embeddings.attrs.get(AUTOSEG_STATE_ATTRIBUTE, [])
    if isinstance(modes, str):
        modes = [modes]
    modes = sorted(set(modes) | {mode})
    embeddings.attrs[AUTOSEG_STATE_ATTRIBUTE] = modes
    group.attrs["state_count"] = len(group)


# Embedding metadata that defines the identity of the embeddings the state was computed from. If any
# of these change (image, tiling, model, normalization) the cached state is stale and must not be
# reused. 'micro_sam_version' is deliberately excluded (a version bump alone does not invalidate it).
EMBEDDING_SIGNATURE_KEYS = ("data_signature", "tile_shape", "halo", "normalization", "model_name", "model_hash")


def _embedding_signature(save_path):
    """A stable signature of the embeddings at `save_path`, or None if it cannot be read.

    Stamped into the cached automatic-segmentation state so it is not reused against embeddings that
    were recomputed with different settings (e.g. tiling toggled) but the same image data.
    """
    if save_path is None:
        return None
    try:
        attrs = dict(util._open_embeddings(save_path, mode="r").attrs)
    except Exception:
        return None
    return "|".join(f"{key}={attrs.get(key)}" for key in EMBEDDING_SIGNATURE_KEYS)


def _signature_matches(cached, requested):
    """Whether a cached signature is compatible: equal, or unknown on either side (lenient)."""
    return cached is None or requested is None or cached == requested


def _save_amg_state_v2(segmenter, save_path, key, embedding_signature=None):
    """Serialize an AMG state as a byte array inside the embedding Zarr."""
    state = segmenter.get_state()
    payload = np.frombuffer(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), dtype="uint8")

    embeddings = util._open_embeddings(save_path, mode="a")
    group = _get_autoseg_state_group(embeddings, "amg", create=True)
    if key in group:
        del group[key]
    chunks = (min(len(payload), 1024 ** 2),)
    dataset = util._create_dataset_with_data(group, key, data=payload, chunks=chunks)
    if embedding_signature is not None:
        dataset.attrs["embedding_signature"] = embedding_signature
    _record_autoseg_state(embeddings, "amg", group)


def _load_amg_state_v2(save_path, key):
    """Load one AMG state from the embedding Zarr, without touching other slice states."""
    embeddings = util._open_embeddings(save_path, mode="r")
    group = _get_autoseg_state_group(embeddings, "amg")
    if group is None or key not in group:
        return None
    dataset = group[key]
    state = pickle.loads(np.asarray(dataset[:], dtype="uint8").tobytes())
    state["embedding_signature"] = dataset.attrs.get("embedding_signature", None)
    return state


def _save_ais_state_v2(segmenter, save_path, key, model_type, embedding_signature=None):
    """Store decoder predictions as an array inside the embedding Zarr."""
    prediction = segmenter.get_state()["prediction"]
    embeddings = util._open_embeddings(save_path, mode="a")
    group = _get_autoseg_state_group(embeddings, "ais", create=True)
    if key in group:
        del group[key]
    state_group = group.create_group(key)

    # Keep channel and z chunks independent so writing and later reading volumetric predictions
    # does not require materializing one very large compressed chunk.
    chunks = list(prediction.shape)
    chunks[0] = 1
    if prediction.ndim == 4:
        chunks[1] = 1
    chunks[-2] = min(chunks[-2], 512)
    chunks[-1] = min(chunks[-1], 512)
    util._create_dataset_with_data(state_group, "prediction", data=prediction, chunks=tuple(chunks))

    # Record which model produced the prediction so it is not reused with a different decoder.
    if model_type is not None:
        state_group.attrs["model_type"] = model_type
    if embedding_signature is not None:
        state_group.attrs["embedding_signature"] = embedding_signature
    _record_autoseg_state(embeddings, "ais", group)


def _load_ais_state_v2(save_path, key):
    """Load one AIS prediction from the embedding Zarr on demand."""
    embeddings = util._open_embeddings(save_path, mode="r")
    group = _get_autoseg_state_group(embeddings, "ais")
    if group is None or key not in group:
        return None
    state_group = group[key]
    return {
        "prediction": state_group["prediction"][:],
        "model_type": state_group.attrs.get("model_type", None),
        "embedding_signature": state_group.attrs.get("embedding_signature", None),
    }


def _has_autoseg_state(save_path, mode, state_count=1):
    """Check cache completeness from Zarr metadata without loading state arrays or bitstreams."""
    if save_path is None or not os.path.exists(save_path):
        return False
    # Any failure to read the metadata (unreadable / partial zarr, backend error) means we cannot
    # confirm a usable cache, so we treat it as absent rather than propagate into the GUI gating.
    try:
        embeddings = util._open_embeddings(save_path, mode="r")
        group = _get_autoseg_state_group(embeddings, mode)
        if group is None:
            return False
        return int(group.attrs.get("state_count", len(group))) >= state_count
    except Exception:
        return False


def _ais_state_matches(state, model_type):
    """Whether the tool can reuse a cached AIS state for `model_type`.

    The AIS prediction depends only on the decoder and the embeddings, so the only staleness risk is
    reusing it with a different decoder. We reuse the cached state unless both the stored and the
    requested `model_type` are known and differ (a state written without a signature is reused).
    """
    cached = state.get("model_type")
    return cached is None or model_type is None or cached == model_type


def cache_autoseg_state(mode, model_or_decoder, raw, image_embeddings, save_path, **kwargs):
    """Compute, cache or load the SAM2 automatic-segmentation state for one image / slice / volume.

    A single entry point over the two modes; both reuse a matching cached state instead of recomputing.
        - 'amg': grid-based mask generation. `model_or_decoder` is the SAM2 model; extra kwargs are the
          AMG parameters (see `_cache_amg_state_v2`).
        - 'ais': UniSAM2 decoder prediction. `model_or_decoder` is the decoder; `ndim` is required
          (see `_cache_ais_state_v2`).

    Args:
        mode: The automatic-segmentation mode, 'amg' or 'ais'.
        model_or_decoder: The SAM2 model (AMG) or the UniSAM2 decoder (AIS).
        raw: The image data.
        image_embeddings: The (optionally precomputed) image embeddings.
        save_path: The embedding save path used to store the state, or None to skip caching.
        kwargs: Mode-specific keyword arguments forwarded to the underlying cache function.

    Returns:
        The segmenter with the (cached or freshly computed) state set.
    """
    if mode == "amg":
        return _cache_amg_state_v2(model_or_decoder, raw, image_embeddings, save_path, **kwargs)
    if mode == "ais":
        return _cache_ais_state_v2(model_or_decoder, raw, image_embeddings, save_path, **kwargs)
    raise ValueError(f"Invalid automatic-segmentation state mode: {mode!r} (expected 'amg' or 'ais').")


def _save_autoseg_state(mode, segmenter, save_path, key, **kwargs):
    """Save an autoseg state into the embedding Zarr: 'amg' as pickled masks, 'ais' as a decoder array."""
    if mode == "amg":
        return _save_amg_state_v2(segmenter, save_path, key, **kwargs)
    if mode == "ais":
        return _save_ais_state_v2(segmenter, save_path, key, **kwargs)
    raise ValueError(f"Invalid automatic-segmentation state mode: {mode!r} (expected 'amg' or 'ais').")


def _load_autoseg_state(mode, save_path, key):
    """Load one autoseg state (AMG masks or AIS prediction) from the embedding Zarr, or None if absent."""
    if mode == "amg":
        return _load_amg_state_v2(save_path, key)
    if mode == "ais":
        return _load_ais_state_v2(save_path, key)
    raise ValueError(f"Invalid automatic-segmentation state mode: {mode!r} (expected 'amg' or 'ais').")


def _cache_amg_state_v2(
    model: torch.nn.Module,
    raw: np.ndarray,
    image_embeddings: Optional[dict],
    save_path: Optional[Union[str, os.PathLike]],
    model_type: Optional[str] = None,
    i: Optional[int] = None,
    state_index: Optional[int] = None,
    is_tiled: Optional[bool] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.9,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
    **kwargs,
):
    """Compute and cache, or load, the SAM2 grid-based (AMG) automatic-segmentation state.

    The SAM2 counterpart of `cache_amg_state`. The state (the predicted masks) is stored in the
    embedding Zarr under 'autoseg_state/amg'. A cached state is reused only if it was
    computed with the same AMG parameters; otherwise it is recomputed and overwritten. Pass
    'save_path=None' to compute in memory without caching.

    Args:
        model: The SAM2 model, loaded via `micro_sam.v2.util.get_sam2_model`.
        raw: The image data.
        image_embeddings: The (optionally precomputed) image embeddings. When None the segmenter
            computes them from `raw`.
        save_path: The embedding save path used to store the state, or None to skip caching.
        model_type: The SAM2 model type, e.g. 'hvit_t'. Recorded in the cached state.
        i: The slice index passed to the segmenter's `initialize` (for reusing volume embeddings).
        state_index: The index used for the on-disk state (defaults to `i`). Set this to identify a
            slice on disk when `i`/`image_embeddings` do not (e.g. a prebuilt single-slice embedding).
        is_tiled: Whether to use the tiled segmenter. By default inferred from the embeddings.
        tile_shape: The tile shape for the tiled segmenter.
        halo: The tile overlap for the tiled segmenter.
        points_per_side: The number of grid points sampled per image side.
        pred_iou_thresh: The predicted-IoU filter threshold.
        stability_score_thresh: The stability-score filter threshold.
        verbose: Whether to run verbose.
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        kwargs: Additional keyword arguments for the AMG segmenter.

    Returns:
        The AMG segmenter with the (cached or freshly computed) state set.
    """
    from .v2.instance_segmentation import get_amg_segmenter

    if is_tiled is None:
        is_tiled = image_embeddings is not None and image_embeddings.get("input_size") is None

    segmenter = get_amg_segmenter(
        model, is_tiled=is_tiled, model_type=model_type, points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, **kwargs,
    )

    key_index = i if state_index is None else state_index
    key, signature = None, None
    if save_path is not None:
        key = _autoseg_state_key(key_index)
        signature = _embedding_signature(save_path)
        state = _load_amg_state_v2(save_path, key)
        if state is not None:
            matches = state.get("params") == segmenter._amg_params
            matches = matches and _signature_matches(state.get("embedding_signature"), signature)
            if matches:
                if verbose:
                    print("Load the AMG state from", save_path, ":", key)
                segmenter.set_state(state)
                return segmenter

    if verbose:
        print("Precomputing the state for automatic mask generation.")

    init_kwargs = {"tile_shape": tile_shape, "halo": halo} if is_tiled else {}
    segmenter.initialize(
        raw, image_embeddings=image_embeddings, i=i, verbose=verbose,
        pbar_init=pbar_init, pbar_update=pbar_update, **init_kwargs,
    )
    if key is not None:
        _save_amg_state_v2(segmenter, save_path, key, embedding_signature=signature)
    return segmenter


def _cache_amg_slice(segmenter, save_path, i, init_fn, embedding_signature=None):
    """Load slice `i`'s AMG state from `save_path` if present and matching, else init and save.

    Used by `micro_sam.v2.instance_segmentation.amg_3d_segmentation` to cache the per-slice
    grid-prediction state of a volume. `init_fn(i)` runs the (expensive) `initialize` for the slice.
    """
    key = _autoseg_state_key(i)
    state = _load_amg_state_v2(save_path, key)
    if state is not None:
        matches = state.get("params") == segmenter._amg_params
        matches = matches and _signature_matches(state.get("embedding_signature"), embedding_signature)
        if matches:
            segmenter.set_state(state)
            return
    init_fn(i)
    _save_amg_state_v2(segmenter, save_path, key, embedding_signature=embedding_signature)


def _cache_amg_volume_state(
    model: torch.nn.Module,
    get_slice: callable,
    n_slices: int,
    image_embeddings: Optional[dict],
    save_path: Union[str, os.PathLike],
    model_type: Optional[str] = None,
    is_tiled: Optional[bool] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
    **amg_kwargs,
):
    """Cache the per-slice AMG grid-prediction state for a whole volume with one shared segmenter.

    Builds the AMG segmenter and reads the embedding signature once, then caches each slice via
    `_cache_amg_slice`, reusing the shared 3d embeddings. This avoids rebuilding the segmenter and
    re-reading the signature on every slice, which a per-slice `cache_autoseg_state` loop would do.

    Args:
        model: The SAM2 model.
        get_slice: Callable mapping a slice index to the 2d image for that slice. The image is only
            used if `image_embeddings` is None; with embeddings the slice is taken from them.
        n_slices: The number of slices in the volume.
        image_embeddings: The precomputed 3d (video-style) embeddings for the volume.
        save_path: The embedding save path used to store the per-slice state.
        model_type: The SAM2 model type, recorded in the cached state.
        is_tiled: Whether to use the tiled segmenter. By default inferred from the embeddings.
        tile_shape: The tile shape for the tiled segmenter.
        halo: The tile overlap for the tiled segmenter.
        verbose: Whether to report progress. By default, set to 'True'.
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        amg_kwargs: Additional keyword arguments for the AMG segmenter.

    Returns:
        The AMG segmenter, with the last slice's state set.
    """
    from .v2.instance_segmentation import get_amg_segmenter

    if is_tiled is None:
        is_tiled = image_embeddings is not None and image_embeddings.get("input_size") is None

    segmenter = get_amg_segmenter(model, is_tiled=is_tiled, model_type=model_type, **amg_kwargs)
    signature = _embedding_signature(save_path)
    init_kwargs = {"tile_shape": tile_shape, "halo": halo} if is_tiled else {}

    def init_slice(i):
        segmenter.initialize(get_slice(i), image_embeddings=image_embeddings, i=i, verbose=False, **init_kwargs)

    _, pbar_init, pbar_update, pbar_close = util.handle_pbar(verbose, pbar_init, pbar_update)
    pbar_init(n_slices, "Precompute automatic segmentation state")
    for i in range(n_slices):
        _cache_amg_slice(segmenter, save_path, i, init_slice, embedding_signature=signature)
        pbar_update(1)
    pbar_close()
    return segmenter


def _cache_ais_state_v2(
    decoder: torch.nn.Module,
    raw: np.ndarray,
    image_embeddings: Optional[dict],
    save_path: Optional[Union[str, os.PathLike]],
    ndim: int,
    model_type: Optional[str] = None,
    i: Optional[int] = None,
    state_index: Optional[int] = None,
    is_tiled: Optional[bool] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    device: Optional[str] = None,
    z_block: Optional[int] = None,
    z_halo: Optional[int] = None,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
):
    """Compute and cache, or load, the SAM2 decoder-based (AIS) automatic-segmentation state.

    The SAM2 counterpart of `cache_is_state`, using the UniSAM2 decoder. The state (the foreground
    and directed-distance predictions) is stored in the embedding Zarr under
    'autoseg_state/ais', using the key 'state' (whole image / volume) or 'state-{i}'
    (a slice). It is independent of the post-processing parameters, so it is always reusable. Pass
    'save_path=None' to skip caching.

    Args:
        decoder: The UniSAM2 model, loaded via `micro_sam.v2.instance_segmentation.get_unisam2_model`.
        raw: The image data.
        image_embeddings: The (optionally precomputed) image embeddings. When given only the decoder
            is run on them (no encoder pass).
        save_path: The embedding save path used to store the state, or None to skip caching.
        ndim: The number of spatial dimensions (2 or 3).
        model_type: The SAM2 model type. Recorded in the cached state so it is not reused with a
            different decoder.
        i: The slice index passed to the segmenter's `initialize`.
        state_index: The index used for the on-disk state (defaults to `i`).
        is_tiled: Whether to use the tiled segmenter. By default inferred from the embeddings.
        tile_shape: The tile shape for the tiled segmenter.
        halo: The tile overlap for the tiled segmenter.
        device: The device to run inference on.
        z_block: Number of slices to decode per z block for volumes.
        z_halo: Number of overlapping slices between z blocks for volumes.
        verbose: Whether to run verbose.
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.

    Returns:
        The AIS segmenter with the (cached or freshly computed) state set.
    """
    from .v2.instance_segmentation import get_unisam2_segmentation_generator

    if is_tiled is None:
        is_tiled = image_embeddings is not None and image_embeddings.get("input_size") is None

    segmenter = get_unisam2_segmentation_generator(decoder, is_tiled=is_tiled, device=device)

    key_index = i if state_index is None else state_index
    key, signature = None, None
    if save_path is not None:
        key = _autoseg_state_key(key_index)
        signature = _embedding_signature(save_path)
        state = _load_ais_state_v2(save_path, key)
        matches = state is not None and _ais_state_matches(state, model_type)
        matches = matches and _signature_matches(state.get("embedding_signature"), signature)
        if matches:
            if verbose:
                print("Load instance segmentation state from", save_path, ":", key)
            segmenter.set_state(state)
            return segmenter

    if verbose:
        print("Precomputing the state for automatic instance segmentation.")

    segmenter.initialize(
        raw, ndim, image_embeddings=image_embeddings, i=i, tile_shape=tile_shape, halo=halo,
        z_block=z_block, z_halo=z_halo, pbar_init=pbar_init, pbar_update=pbar_update,
    )
    if key is not None:
        _save_ais_state_v2(segmenter, save_path, key, model_type, embedding_signature=signature)
    return segmenter


def _resolve_unisam2_decoder(model_type, checkpoint_path, device):
    """Return a UniSAM2 decoder for the SAM2 model if one is available, else None (fall back to AMG).

    Mirrors `micro_sam.v2.instance_segmentation.get_decoder`: a decoder from a custom
    `checkpoint_path`, or the registered decoder of a finetuned model (e.g. 'hvit_t_cells'). Any
    failure (e.g. an interactive-only checkpoint without a decoder) returns None.
    """
    from .v2.util import FINETUNED_MODELS, has_registered_decoder, _download_finetuned_sam2_model
    from .v2.instance_segmentation import get_unisam2_model

    encoder = model_type[:6]
    if checkpoint_path is not None:
        decoder_source = checkpoint_path
    elif model_type in FINETUNED_MODELS and has_registered_decoder(model_type):
        _, _, decoder_source = _download_finetuned_sam2_model(model_type)
    else:
        return None
    try:
        return get_unisam2_model(decoder_source, device=util.get_device(device), encoder=encoder)
    except Exception as e:
        print(f"Could not load a UniSAM2 decoder from '{decoder_source}': {e}")
        return None


def _cache_autoseg_state_for_file(
    predictor, decoder, model_type, image_data, embeddings, save_path, ndim, verbose,
):
    """Cache the SAM2 automatic-segmentation state for one file: AIS if a decoder is given, else AMG."""
    if decoder is not None:  # AIS segments the whole image / volume in one pass.
        device = next(decoder.parameters()).device
        cache_autoseg_state(
            "ais", decoder, image_data, embeddings, save_path, ndim=ndim, model_type=model_type,
            device=device, verbose=verbose,
        )
    elif ndim == 2:  # AMG on a single 2d image.
        model = getattr(predictor, "model", predictor)
        cache_autoseg_state("amg", model, image_data, embeddings, save_path, model_type=model_type, verbose=verbose)
    else:  # AMG on a volume: cache the per-slice grid state, reusing the 3d embeddings and one segmenter.
        model = getattr(predictor, "model", predictor)
        _cache_amg_volume_state(
            model, lambda i: image_data[i], image_data.shape[0], embeddings, save_path,
            model_type=model_type, verbose=verbose,
        )


def precompute_state(
    input_path: Union[os.PathLike, str],
    output_path: Union[os.PathLike, str],
    pattern: Optional[str] = None,
    model_type: str = "hvit_t",
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    key: Optional[str] = None,
    ndim: Optional[int] = None,
    precompute_autoseg_state: bool = False,
    prefer_decoder: bool = True,
    batch_size: Optional[int] = 1,
    devices: Optional[Union[str, Sequence[str]]] = None,
) -> None:
    """Precompute and cache the image embeddings (and, optionally, the automatic-segmentation state).

    The embeddings are saved in the same zarr format the annotators use, so the output can be loaded
    directly by the `micro_sam.annotator` CLI and the napari GUI by passing the same path as the
    embedding path (with a matching model and image).

    Args:
        input_path: The input image file(s). Can either be a single image file (e.g. tif or png),
            a container file (e.g. hdf5 or zarr) or a folder with image files.
            In case of a container file the argument `key` must be given. In case of a folder
            the `pattern` argument must be given to subselect files.
        output_path: The output path where the embeddings will be saved. For a single input this is the path
            to the embeddings zarr; for a folder of inputs this is the directory the embeddings are saved in.
        pattern: Glob pattern to select files in a folder. The embeddings will be computed
            for each of these files. To select all files in a folder pass "*".
        model_type: The model to use. Supports SAM1 ('vit_*'), SAM2 ('hvit_*') and VFM (DINO / UNI)
            encoders. By default the `hvit_t` model is used.
        checkpoint_path: Path to a checkpoint for a custom model.
        key: The key to the input file. This is needed for container files (e.g. hdf5 or zarr)
            or to load several images as 3d volume. Provide a glob pattern, e.g. "*.tif", for this case.
        ndim: The dimensionality of the data. By default, computed from the input data.
        precompute_autoseg_state: Whether to also precompute the automatic-segmentation state in the
            embeddings (a longer start-up in exchange for a faster first automatic segmentation).
            Supported for SAM2 ('hvit_*') models.
        prefer_decoder: Whether to use the decoder-based state (AIS) when the SAM2 model has a decoder,
            instead of grid-based mask generation (AMG).
        batch_size: The number of tiles / slices per model call. Pass None to select a throughput-efficient
            value per device. Ignored by the model families that do not support batching (VFM encoders).
        devices: The device or devices to compute the embeddings on. By default all visible CUDA devices
            are used. Only supported for SAM2 ('hvit_*') models.
    """
    # Imported lazily to avoid a circular import ('_state' imports from this module).
    from micro_sam.sam_annotator._state import _get_sam_model

    is_sam2 = model_type.startswith("hvit")
    if precompute_autoseg_state and not is_sam2:
        raise ValueError(
            "Precomputing the automatic-segmentation state via 'precompute_state' is only supported for "
            "SAM2 ('hvit_*') models. For SAM1 use the annotator command with "
            "'--precompute_autoseg_state'."
        )

    # Dispatch to the embedding function for the model family (SAM1 / SAM2 / VFM). All share the same
    # interface, so the per-family predictor from '_get_sam_model' feeds directly into it.
    compute_embeddings = util.get_embedding_function(model_type)

    # Only SAM2 supports multi-device inference and only SAM1 / SAM2 support batching, so forward
    # these settings just to the families whose embedding function accepts them.
    supported = inspect.signature(compute_embeddings).parameters
    compute_kwargs = {
        name: value for name, value in (("batch_size", batch_size), ("devices", devices))
        if name in supported
    }

    # Resolve the UniSAM2 decoder once (AIS); when none is available the state is cached with AMG.
    decoder = None
    if precompute_autoseg_state and prefer_decoder:
        decoder = _resolve_unisam2_decoder(model_type, checkpoint_path, device=None)

    # Determine the input files and matching output embedding paths.
    single = pattern is None
    if single:
        input_files, output_paths = [input_path], [output_path]
    else:
        input_files = sorted(glob(os.path.join(input_path, pattern)))
        if len(input_files) == 0:
            raise ValueError(f"Could not find any files matching the pattern '{pattern}' in '{input_path}'.")
        os.makedirs(output_path, exist_ok=True)
        output_paths = [os.path.join(output_path, os.path.basename(f)) for f in input_files]

    predictor, current_ndim = None, None
    for input_file, out_path in tqdm(
        zip(input_files, output_paths), total=len(input_files), desc="Precompute embeddings", disable=single
    ):
        image_data = input_file if isinstance(input_file, np.ndarray) else util.load_image_data(input_file, key)
        file_ndim = image_data.ndim if ndim is None else ndim

        # Build the predictor for the data dimensionality (for SAM2 a 2d image vs. 3d video predictor).
        # We reuse the annotator's model loader so the embeddings match what the GUI / CLI expect.
        if predictor is None or file_ndim != current_ndim:
            predictor, _ = _get_sam_model(
                model_type=model_type, ndim=file_ndim, device=None,
                checkpoint_path=checkpoint_path, decoder_path=None, use_cli=True,
            )
            current_ndim = file_ndim

        save_path = str(Path(out_path).with_suffix(".zarr"))
        embeddings = compute_embeddings(
            predictor=predictor, input_=image_data, save_path=save_path, ndim=file_ndim, verbose=single,
            **compute_kwargs,
        )

        if precompute_autoseg_state:
            _cache_autoseg_state_for_file(
                predictor, decoder, model_type, image_data, embeddings, save_path, file_ndim, verbose=single,
            )
