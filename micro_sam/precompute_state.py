"""Precompute and cache image embeddings for image data (SAM1, SAM2 or VFM encoders).
"""

import os
import pickle
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np

import torch

from segment_anything.predictor import SamPredictor

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
) -> instance_segmentation.AMGBase:
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
) -> Optional[instance_segmentation.AMGBase]:
    """Compute and cache or load the state for the automatic mask generator.

    Args:
        predictor: The Segment Anything predictor.
        decoder: The instance segmentation decoder.
        raw: The image data.
        image_embeddings: The image embeddings.
        save_path: The embedding save path. The AMG state will be stored in 'save_path/amg_state.pickle'.
        verbose: Whether to run the computation verbose. By default, set to 'True'.
        i: The index for which to cache the state.
        skip_load: Skip loading the state if it is precomputed. By default, set to 'False'.
        kwargs: The keyword arguments for the amg class.

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


def _auto_state_path(save_path, mode, i):
    """Resolve the on-disk path (and h5 key) for the cached SAM2 automatic-segmentation state.

    'mode' is 'amg' (grid masks, pickled) or 'ais' (decoder predictions, h5). 'i' selects a
    per-slice entry for a volume, or the whole image / volume when None.
    """
    if mode == "amg":
        if i is None:
            return os.path.join(save_path, "auto_state_amg.pickle"), None
        return os.path.join(save_path, "auto_state_amg", f"state-{i}.pkl"), None
    return os.path.join(save_path, "auto_state_ais.h5"), ("state" if i is None else f"state-{i}")


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


def _save_amg_state_v2(segmenter, path, embedding_signature=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = segmenter.get_state()
    state["embedding_signature"] = embedding_signature
    with open(path, "wb") as f:
        pickle.dump(state, f)


def _load_amg_state_v2(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_ais_state_v2(segmenter, path, key, model_type, embedding_signature=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "a") as f:
        if key in f:
            del f[key]
        g = f.create_group(key)
        g.create_dataset("prediction", data=segmenter.get_state()["prediction"], compression="gzip")
        # Record which model produced the prediction so it is not reused with a different decoder.
        if model_type is not None:
            g.attrs["model_type"] = model_type
        if embedding_signature is not None:
            g.attrs["embedding_signature"] = embedding_signature


def _load_ais_state_v2(path, key):
    with h5py.File(path, "r") as f:
        if key not in f:
            return None
        g = f[key]
        return {
            "prediction": g["prediction"][:],
            "model_type": g.attrs.get("model_type", None),
            "embedding_signature": g.attrs.get("embedding_signature", None),
        }


def _ais_state_matches(state, model_type):
    """Whether a cached AIS state may be reused for `model_type`.

    The AIS prediction depends only on the decoder and the embeddings, so the only staleness risk is
    reusing it with a different decoder. We reuse the cached state unless both the stored and the
    requested `model_type` are known and differ (a state written without a signature is reused).
    """
    cached = state.get("model_type")
    return cached is None or model_type is None or cached == model_type


def cache_amg_state_v2(
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

    The SAM2 counterpart of `cache_amg_state`. The state (the predicted masks) is stored next to
    the embeddings at 'save_path/auto_state_amg.pickle' (or 'auto_state_amg/state-{i}.pkl' for a
    slice). A cached state is reused only if it was computed with the same AMG parameters, otherwise
    it is recomputed and overwritten. Pass 'save_path=None' to compute in memory without caching.

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
    path, signature = None, None
    if save_path is not None:
        path, _ = _auto_state_path(save_path, "amg", key_index)
        signature = _embedding_signature(save_path)
        if os.path.exists(path):
            state = _load_amg_state_v2(path)
            matches = state.get("params") == segmenter._amg_params
            matches = matches and _signature_matches(state.get("embedding_signature"), signature)
            if matches:
                if verbose:
                    print("Load the AMG state from", path)
                segmenter.set_state(state)
                return segmenter

    if verbose:
        print("Precomputing the state for automatic mask generation.")

    init_kwargs = {"tile_shape": tile_shape, "halo": halo} if is_tiled else {}
    segmenter.initialize(
        raw, image_embeddings=image_embeddings, i=i, verbose=verbose,
        pbar_init=pbar_init, pbar_update=pbar_update, **init_kwargs,
    )
    if path is not None:
        _save_amg_state_v2(segmenter, path, embedding_signature=signature)
    return segmenter


def _cache_amg_slice(segmenter, save_path, i, init_fn, embedding_signature=None):
    """Load slice `i`'s AMG state from `save_path` if present and matching, else init and save.

    Used by `micro_sam.v2.instance_segmentation.automatic_3d_segmentation` to cache the per-slice
    grid-prediction state of a volume. `init_fn(i)` runs the (expensive) `initialize` for the slice.
    """
    path, _ = _auto_state_path(save_path, "amg", i)
    if os.path.exists(path):
        state = _load_amg_state_v2(path)
        matches = state.get("params") == segmenter._amg_params
        matches = matches and _signature_matches(state.get("embedding_signature"), embedding_signature)
        if matches:
            segmenter.set_state(state)
            return
    init_fn(i)
    _save_amg_state_v2(segmenter, path, embedding_signature=embedding_signature)


def cache_ais_state_v2(
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
    and directed-distance predictions) is stored next to the embeddings at 'save_path/auto_state_ais.h5'
    under the key 'state' (whole image / volume) or 'state-{i}' (a slice). It is independent of the
    post-processing parameters, so it is always reusable. Pass 'save_path=None' to skip caching.

    Args:
        decoder: The UniSAM2 model, loaded via `micro_sam.v2.automatic_segmentation.get_unisam2_model`.
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
    from .v2.automatic_segmentation import get_unisam2_segmentation_generator

    if is_tiled is None:
        is_tiled = image_embeddings is not None and image_embeddings.get("input_size") is None

    segmenter = get_unisam2_segmentation_generator(decoder, is_tiled=is_tiled, device=device)

    key_index = i if state_index is None else state_index
    path, key, signature = None, None, None
    if save_path is not None:
        path, key = _auto_state_path(save_path, "ais", key_index)
        signature = _embedding_signature(save_path)
        state = _load_ais_state_v2(path, key) if os.path.exists(path) else None
        matches = state is not None and _ais_state_matches(state, model_type)
        matches = matches and _signature_matches(state.get("embedding_signature"), signature)
        if matches:
            if verbose:
                print("Load instance segmentation state from", path, ":", key)
            segmenter.set_state(state)
            return segmenter

    if verbose:
        print("Precomputing the state for automatic instance segmentation.")

    segmenter.initialize(
        raw, ndim, image_embeddings=image_embeddings, i=i, tile_shape=tile_shape, halo=halo,
        z_block=z_block, z_halo=z_halo, pbar_init=pbar_init, pbar_update=pbar_update,
    )
    if path is not None:
        _save_ais_state_v2(segmenter, path, key, model_type, embedding_signature=signature)
    return segmenter


def _resolve_unisam2_decoder(model_type, checkpoint_path, device):
    """Return a UniSAM2 decoder for the SAM2 model if one is available, else None (fall back to AMG).

    Mirrors `micro_sam.v2.automatic_segmentation.get_segmenter`: a decoder from a custom
    `checkpoint_path`, or the registered decoder of a finetuned model (e.g. 'hvit_t_cells'). Any
    failure (e.g. an interactive-only checkpoint without a decoder) returns None.
    """
    from .v2.util import FINETUNED_MODELS, has_registered_decoder, _download_finetuned_sam2_model
    from .v2.automatic_segmentation import get_unisam2_model

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


def _cache_auto_state_for_file(predictor, decoder, model_type, image_data, embeddings, save_path, ndim, verbose):
    """Cache the SAM2 automatic-segmentation state for one file: AIS if a decoder is given, else AMG."""
    if decoder is not None:  # AIS segments the whole image / volume in one pass.
        device = next(decoder.parameters()).device
        cache_ais_state_v2(
            decoder, image_data, embeddings, save_path, ndim=ndim, model_type=model_type,
            device=device, verbose=verbose,
        )
    elif ndim == 2:  # AMG on a single 2d image.
        model = getattr(predictor, "model", predictor)
        cache_amg_state_v2(model, image_data, embeddings, save_path, model_type=model_type, verbose=verbose)
    else:  # AMG on a volume: cache the per-slice grid state, reusing the 3d embeddings.
        model = getattr(predictor, "model", predictor)
        n = image_data.shape[0]
        for i in tqdm(range(n), total=n, desc="Precompute auto state", disable=not verbose):
            cache_amg_state_v2(model, image_data[i], embeddings, save_path, model_type=model_type, i=i, verbose=False)


def precompute_state(
    input_path: Union[os.PathLike, str],
    output_path: Union[os.PathLike, str],
    pattern: Optional[str] = None,
    model_type: str = "hvit_t",
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    key: Optional[str] = None,
    ndim: Optional[int] = None,
    precompute_auto_state: bool = False,
    prefer_decoder: bool = True,
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
        precompute_auto_state: Whether to also precompute the automatic-segmentation state next to the
            embeddings (a longer start-up in exchange for a faster first automatic segmentation).
            Supported for SAM2 ('hvit_*') models.
        prefer_decoder: Whether to use the decoder-based state (AIS) when the SAM2 model has a decoder,
            instead of grid-based mask generation (AMG).
    """
    # Imported lazily to avoid a circular import ('_state' imports from this module).
    from micro_sam.sam_annotator._state import _get_sam_model

    is_sam2 = model_type.startswith("h")
    if precompute_auto_state and not is_sam2:
        raise ValueError(
            "Precomputing the automatic-segmentation state via 'precompute_state' is only supported for "
            "SAM2 ('hvit_*') models. For SAM1 use the annotator command with '--precompute_amg_state'."
        )

    # Dispatch to the embedding function for the model family (SAM1 / SAM2 / VFM). All share the same
    # interface, so the per-family predictor from '_get_sam_model' feeds directly into it.
    compute_embeddings = util.get_embedding_function(model_type)

    # Resolve the UniSAM2 decoder once (AIS); when none is available the state is cached with AMG.
    decoder = None
    if precompute_auto_state and prefer_decoder:
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
            predictor=predictor, input_=image_data, save_path=save_path, ndim=file_ndim, verbose=single
        )

        if precompute_auto_state:
            _cache_auto_state_for_file(
                predictor, decoder, model_type, image_data, embeddings, save_path, file_ndim, verbose=single,
            )
