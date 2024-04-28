"""Precompute image embeddings and automatic mask generator state for image data.
"""

import os
import pickle

from functools import partial
from glob import glob
from pathlib import Path
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

from . import instance_segmentation, util


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
        predictor: The segment anything predictor.
        raw: The image data.
        image_embeddings: The image embeddings.
        save_path: The embedding save path. The AMG state will be stored in 'save_path/amg_state.pickle'.
        verbose: Whether to run the computation verbose.
        i: The index for which to cache the state.
        kwargs: The keyword arguments for the amg class.

    Returns:
        The automatic mask generator class with the cached state.
    """
    is_tiled = image_embeddings["input_size"] is None
    amg = instance_segmentation.get_amg(predictor, is_tiled, **kwargs)

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

    amg.initialize(raw, image_embeddings=image_embeddings, verbose=verbose, i=i)
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
        predictor: The segment anything predictor.
        decoder: The instance segmentation decoder.
        raw: The image data.
        image_embeddings: The image embeddings.
        save_path: The embedding save path. The AMG state will be stored in 'save_path/amg_state.pickle'.
        verbose: Whether to run the computation verbose.
        i: The index for which to cache the state.
        skip_load: Skip loading the state if it is precomputed.
        kwargs: The keyword arguments for the amg class.

    Returns:
        The instance segmentation class with the cached state.
    """
    is_tiled = image_embeddings["input_size"] is None
    amg = instance_segmentation.get_amg(predictor, is_tiled, decoder=decoder, **kwargs)

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
    predictor, input_path, output_path, key, ndim, tile_shape, halo, precompute_amg_state, decoder,
):
    if isinstance(input_path, np.ndarray):
        image_data = input_path
    else:
        image_data = util.load_image_data(input_path, key)

    # Precompute the image embeddings.
    output_path = Path(output_path).with_suffix(".zarr")
    embeddings = util.precompute_image_embeddings(
        predictor, image_data, output_path, ndim=ndim, tile_shape=tile_shape, halo=halo,
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
            cache_function(raw=image_data, verbose=True)
        else:
            n = image_data.shape[0]
            for i in tqdm(range(n), total=n, desc="Precompute instance segmentation state"):
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
    for i, file_path in enumerate(tqdm(input_files, total=len(input_files), desc="Precompute state for files")):

        if isinstance(file_path, np.ndarray):
            out_path = os.path.join(output_path, f"embedding_{i:05}.tif")
        else:
            out_path = os.path.join(output_path, os.path.basename(file_path))

        _precompute_state_for_file(
            predictor, file_path, out_path,
            key=key, ndim=ndim, tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state, decoder=decoder,
        )


def precompute_state(
    input_path: Union[os.PathLike, str],
    output_path: Union[os.PathLike, str],
    pattern: Optional[str] = None,
    model_type: str = util._DEFAULT_MODEL,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    key: Optional[str] = None,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    precompute_amg_state: bool = False,
) -> None:
    """Precompute the image embeddings and other optional state for the input image(s).

    Args:
        input_path: The input image file(s). Can either be a single image file (e.g. tif or png),
            a container file (e.g. hdf5 or zarr) or a folder with images files.
            In case of a container file the argument `key` must be given. In case of a folder
            it can be given to provide a glob pattern to subselect files from the folder.
        output_path: The output path were the embeddings and other state will be saved.
        pattern: Glob pattern to select files in a folder. The embeddings will be computed
            for each of these files. To select all files in a folder pass "*".
        model_type: The SegmentAnything model to use. Will use the standard vit_h model by default.
        checkpoint_path: Path to a checkpoint for a custom model.
        key: The key to the input file. This is needed for contaner files (e.g. hdf5 or zarr)
            or to load several images as 3d volume. Provide a glob pattern, e.g. "*.tif", for this case.
        ndim: The dimensionality of the data.
        tile_shape: Shape of tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction.
        precompute_amg_state: Whether to precompute the state for automatic instance segmentation
            in addition to the image embeddings.
    """
    predictor, state = util.get_sam_model(
        model_type=model_type, checkpoint_path=checkpoint_path, return_state=True,
    )
    if "decoder_state" in state:
        decoder = instance_segmentation.get_decoder(predictor.model.image_encoder, state["decoder_state"])
    else:
        decoder = None

    # Check if we precompute the state for a single file or for a folder with image files.
    if pattern is None:
        _precompute_state_for_file(
            predictor, input_path, output_path, key,
            ndim=ndim, tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state,
            decoder=decoder,
        )
    else:
        input_files = glob(os.path.join(input_path, pattern))
        _precompute_state_for_files(
            predictor, input_files, output_path, key=key,
            ndim=ndim, tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state,
            decoder=decoder,
        )


def main():
    """@private"""
    import argparse

    available_models = list(util.get_model_names())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(description="Compute the embeddings for an image.")
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="The filepath to the image data. Supports all data types that can be read by imageio (e.g. tif, png, ...) "
        "or elf.io.open_file (e.g. hdf5, zarr, mrc). For the latter you also need to pass the 'key' parameter."
    )
    parser.add_argument(
        "-e", "--embedding_path", required=True, help="The path where the embeddings will be saved."
    )

    parser.add_argument(
        "--pattern", help="Pattern / wildcard for selecting files in a folder. To select all files use '*'."
    )
    parser.add_argument(
        "-k", "--key",
        help="The key for opening data with elf.io.open_file. This is the internal path for a hdf5 or zarr container, "
        "for an image stack it is a wild-card, e.g. '*.png' and for mrc it is 'data'."
    )

    parser.add_argument(
        "-m", "--model_type", default=util._DEFAULT_MODEL,
        help=f"The segment anything model that will be used, one of {available_models}."
    )
    parser.add_argument(
        "-c", "--checkpoint", default=None,
        help="Checkpoint from which the SAM model will be loaded loaded."
    )
    parser.add_argument(
        "--tile_shape", nargs="+", type=int, help="The tile shape for using tiled prediction.", default=None
    )
    parser.add_argument(
        "--halo", nargs="+", type=int, help="The halo for using tiled prediction.", default=None
    )
    parser.add_argument(
        "-n", "--ndim", type=int, default=None,
        help="The number of spatial dimensions in the data. "
        "Please specify this if your data has a channel dimension."
    )
    parser.add_argument(
        "-p", "--precompute_amg_state", action="store_true",
        help="Whether to precompute the state for automatic instance segmentation."
    )

    args = parser.parse_args()
    precompute_state(
        args.input_path, args.embedding_path,
        model_type=args.model_type, checkpoint_path=args.checkpoint,
        pattern=args.pattern, key=args.key,
        tile_shape=args.tile_shape, halo=args.halo, ndim=args.ndim,
        precompute_amg_state=args.precompute_amg_state,
    )


if __name__ == "__main__":
    main()
