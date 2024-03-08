"""Precompute image embeddings and automatic mask generator state for image data.
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
    **kwargs,
) -> instance_segmentation.AMGBase:
    """Compute and cache or load the state for the automatic mask generator.

    Args:
        predictor: The segment anything predictor.
        decoder: The instance segmentation decoder.
        raw: The image data.
        image_embeddings: The image embeddings.
        save_path: The embedding save path. The AMG state will be stored in 'save_path/amg_state.pickle'.
        verbose: Whether to run the computation verbose.
        i: The index for which to cache the state.
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
    predictor, input_path, output_path, key, ndim,
    tile_shape, halo, precompute_amg_state,
    decoder=None,
):
    image_data = util.load_image_data(input_path, key)
    output_path = Path(output_path).with_suffix(".zarr")
    embeddings = util.precompute_image_embeddings(
        predictor, image_data, output_path, ndim=ndim, tile_shape=tile_shape, halo=halo,
    )
    if precompute_amg_state:
        if decoder is None:
            cache_amg_state(predictor, image_data, embeddings, output_path, verbose=True)
        else:
            cache_is_state(predictor, decoder, image_data, embeddings, output_path, verbose=True)


def _precompute_state_for_files(
    predictor, input_files, output_path, ndim, tile_shape, halo, precompute_amg_state,
    decoder=None,
):
    os.makedirs(output_path, exist_ok=True)
    for file_path in tqdm(input_files, desc="Precompute state for files."):
        out_path = os.path.join(output_path, os.path.basename(file_path))
        _precompute_state_for_file(
            predictor, file_path, out_path,
            key=None, ndim=ndim, tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state, decoder=decoder,
        )


def precompute_state(
    input_path: Union[os.PathLike, str],
    output_path: Union[os.PathLike, str],
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
        model_type: The SegmentAnything model to use. Will use the standard vit_h model by default.
        checkpoint_path: Path to a checkpoint for a custom model.
        key: The key to the input file. This is needed for contaner files (e.g. hdf5 or zarr)
            and can be used to provide a glob pattern if the input is a folder with image files.
        ndim: The dimensionality of the data.
        tile_shape: Shape of tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction.
        precompute_amg_state: Whether to precompute the state for automatic instance segmentation
            in addition to the image embeddings.
    """
    predictor = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint_path)
    # check if we precompute the state for a single file or for a folder with image files
    if os.path.isdir(input_path) and Path(input_path).suffix not in (".n5", ".zarr"):
        pattern = "*" if key is None else key
        input_files = glob(os.path.join(input_path, pattern))
        _precompute_state_for_files(
            predictor, input_files, output_path,
            ndim=ndim, tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state,
        )
    else:
        _precompute_state_for_file(
            predictor, input_path, output_path, key,
            ndim=ndim, tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state,
        )


def main():
    """@private"""
    import argparse

    parser = argparse.ArgumentParser(description="Compute the embeddings for an image.")
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-m", "--model_type", default=util._DEFAULT_MODEL)
    parser.add_argument("-c", "--checkpoint_path", default=None)
    parser.add_argument("-k", "--key")
    parser.add_argument(
        "--tile_shape", nargs="+", type=int, help="The tile shape for using tiled prediction", default=None
    )
    parser.add_argument(
        "--halo", nargs="+", type=int, help="The halo for using tiled prediction", default=None
    )
    parser.add_argument("-n", "--ndim", type=int)
    parser.add_argument("-p", "--precompute_amg_state", action="store_true")

    args = parser.parse_args()
    precompute_state(
        args.input_path, args.output_path, args.model_type, args.checkpoint_path,
        key=args.key, tile_shape=args.tile_shape, halo=args.halo, ndim=args.ndim,
        precompute_amg_state=args.precompute_amg_state,
    )


if __name__ == "__main__":
    main()
