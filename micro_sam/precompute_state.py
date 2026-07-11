"""Precompute and cache image embeddings for image data (SAM1, SAM2 or VFM encoders).
"""

import os
import pickle
from glob import glob
from pathlib import Path
from functools import partial
from typing import Optional, Tuple, Union, List

import h5py
import numpy as np

import torch
import torch.nn as nn

from segment_anything.predictor import SamPredictor

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from . import util
from .v1.util import precompute_image_embeddings
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


def _precompute_state_for_file(
    predictor, input_path, output_path, key, ndim, tile_shape, halo, precompute_amg_state, decoder, verbose
):
    if isinstance(input_path, np.ndarray):
        image_data = input_path
    else:
        image_data = util.load_image_data(input_path, key)

    # Precompute the image embeddings.
    output_path = Path(output_path).with_suffix(".zarr")
    embeddings = precompute_image_embeddings(
        predictor, image_data, output_path, ndim=ndim, tile_shape=tile_shape, halo=halo, verbose=verbose
    )

    # Precompute the state for automatic instance segmnetaiton (AMG or AIS).
    if precompute_amg_state:
        if decoder is None:
            cache_function = partial(
                cache_amg_state, predictor=predictor, image_embeddings=embeddings, save_path=output_path
            )
        else:
            cache_function = partial(
                cache_is_state, predictor=predictor, decoder=decoder,
                image_embeddings=embeddings, save_path=output_path
            )

        if ndim is None:
            ndim = image_data.ndim

        if ndim == 2:
            cache_function(raw=image_data, verbose=verbose)
        else:
            n = image_data.shape[0]
            for i in tqdm(range(n), total=n, desc="Precompute instance segmentation state", disable=not verbose):
                cache_function(raw=image_data, i=i, verbose=False)


def _precompute_state_for_files(
    predictor: SamPredictor,
    input_files: Union[List[Union[os.PathLike, str]], List[np.ndarray]],
    output_path: Union[os.PathLike, str],
    key: Optional[str] = None,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    precompute_amg_state: bool = False,
    decoder: Optional["nn.Module"] = None,
):
    os.makedirs(output_path, exist_ok=True)
    idx = 0
    for file_path in tqdm(input_files, total=len(input_files), desc="Precompute state for files"):

        if isinstance(file_path, np.ndarray):
            out_path = os.path.join(output_path, f"embedding_{idx:05}.tif")
        else:
            out_path = os.path.join(output_path, os.path.basename(file_path))

        _precompute_state_for_file(
            predictor, file_path, out_path,
            key=key, ndim=ndim, tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state, decoder=decoder,
            verbose=False,
        )
        idx += 1


def precompute_state(
    input_path: Union[os.PathLike, str],
    output_path: Union[os.PathLike, str],
    pattern: Optional[str] = None,
    model_type: str = "hvit_t",
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    key: Optional[str] = None,
    ndim: Optional[int] = None,
) -> None:
    """Precompute and cache the image embeddings for the input image(s).

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
    """
    # Imported lazily to avoid a circular import ('_state' imports from this module).
    from micro_sam.sam_annotator._state import _get_sam_model

    # Dispatch to the embedding function for the model family (SAM1 / SAM2 / VFM). All share the same
    # interface, so the per-family predictor from '_get_sam_model' feeds directly into it.
    compute_embeddings = util.get_embedding_function(model_type)

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
        compute_embeddings(
            predictor=predictor, input_=image_data, save_path=save_path, ndim=file_ndim, verbose=single
        )
