import os
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import imageio.v3 as imageio

from . import util
from .instance_segmentation import (
    get_amg, get_decoder, mask_data_to_segmentation, InstanceSegmentationWithDecoder, AMGBase
)
from .multi_dimensional_segmentation import automatic_3d_segmentation


def get_predictor_and_segmenter(
    model_type: str,
    checkpoint: Optional[Union[os.PathLike, str]] = None,
    device: str = None,
    amg: Optional[bool] = None,
    is_tiled: bool = False,
    **kwargs,
) -> Tuple[util.SamPredictor, Union[AMGBase, InstanceSegmentationWithDecoder]]:
    """Get the Segment Anything model and class for automatic instance segmentation.

    Args:
        model_type: The Segment Anything model choice.
        checkpoint: The filepath to the stored model checkpoints.
        device: The torch device.
        amg: Whether to perform automatic segmentation in AMG mode.
            Otherwise AIS will be used, which requires a special segmentation decoder.
            If not specified AIS will be used if it is available and otherwise AMG will be used.
        is_tiled: Whether to return segmenter for performing segmentation in tiling window style.
        kwargs: Keyword arguments for the automatic instance segmentation class.

    Returns:
        The Segment Anything model.
        The automatic instance segmentation class.
    """
    # Get the device
    device = util.get_device(device=device)

    # Get the predictor and state for Segment Anything models.
    predictor, state = util.get_sam_model(
        model_type=model_type, device=device, checkpoint_path=checkpoint, return_state=True,
    )

    if amg is None:
        amg = "decoder_state" not in state
    if amg:
        decoder = None
    else:
        if "decoder_state" not in state:
            raise RuntimeError("You have passed amg=False, but your model does not contain a segmentation decoder.")
        decoder_state = state["decoder_state"]
        decoder = get_decoder(image_encoder=predictor.model.image_encoder, decoder_state=decoder_state, device=device)

    segmenter = get_amg(
        predictor=predictor, is_tiled=is_tiled, decoder=decoder, **kwargs
    )

    return predictor, segmenter


def automatic_instance_segmentation(
    predictor: util.SamPredictor,
    segmenter: Union[AMGBase, InstanceSegmentationWithDecoder],
    input_path: Union[Union[os.PathLike, str], np.ndarray],
    output_path: Optional[Union[os.PathLike, str]] = None,
    embedding_path: Optional[Union[os.PathLike, str]] = None,
    key: Optional[str] = None,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    **generate_kwargs
) -> np.ndarray:
    """Run automatic segmentation for the input image.

    Args:
        predictor: The Segment Anything model.
        segmenter: The automatic instance segmentation class.
        input_path: input_path: The input image file(s). Can either be a single image file (e.g. tif or png),
            or a container file (e.g. hdf5 or zarr).
        output_path: The output path where the instance segmentations will be saved.
        embedding_path: The path where the embeddings are cached already / will be saved.
        key: The key to the input file. This is needed for container files (eg. hdf5 or zarr)
            or to load several images as 3d volume. Provide a glob patterm, eg. "*.tif", for this case.
        ndim: The dimensionality of the data. By default the dimensionality of the data will be used.
            If you have RGB data you have to specify this explicitly, e.g. pass ndim=2 for 2d segmentation of RGB.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction.
        verbose: Verbosity flag.
        generate_kwargs: optional keyword arguments for the generate function of the AMG or AIS class.

    Returns:
        The segmentation result.
    """
    # Load the input image file.
    if isinstance(input_path, np.ndarray):
        image_data = input_path
    else:
        image_data = util.load_image_data(input_path, key)

    ndim = image_data.ndim if ndim is None else ndim

    if ndim == 2:
        if (image_data.ndim != 2) and (image_data.ndim != 3 and image_data.shape[-1] != 3):
            raise ValueError(f"The inputs does not match the shape expectation of 2d inputs: {image_data.shape}")

        # Precompute the image embeddings.
        image_embeddings = util.precompute_image_embeddings(
            predictor=predictor,
            input_=image_data,
            save_path=embedding_path,
            ndim=ndim,
            tile_shape=tile_shape,
            halo=halo,
            verbose=verbose,
        )

        segmenter.initialize(image=image_data, image_embeddings=image_embeddings)
        masks = segmenter.generate(**generate_kwargs)

        if len(masks) == 0:  # instance segmentation can have no masks, hence we just save empty labels
            if isinstance(segmenter, InstanceSegmentationWithDecoder):
                this_shape = segmenter._foreground.shape
            elif isinstance(segmenter, AMGBase):
                this_shape = segmenter._original_size
            else:
                this_shape = image_data.shape[-2:]

            instances = np.zeros(this_shape, dtype="uint32")
        else:
            instances = mask_data_to_segmentation(masks, with_background=True, min_object_size=0)
    else:
        if (image_data.ndim != 3) and (image_data.ndim != 4 and image_data.shape[-1] != 3):
            raise ValueError(f"The inputs does not match the shape expectation of 3d inputs: {image_data.shape}")

        instances = automatic_3d_segmentation(
            volume=image_data,
            predictor=predictor,
            segmentor=segmenter,
            embedding_path=embedding_path,
            tile_shape=tile_shape,
            halo=halo,
            verbose=verbose,
            **generate_kwargs
        )

    if output_path is not None:
        # Save the instance segmentation
        output_path = Path(output_path).with_suffix(".tif")
        imageio.imwrite(output_path, instances, compression="zlib")

    return instances


def main():
    """@private"""
    import argparse

    available_models = list(util.get_model_names())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(description="Run automatic segmentation for an image.")
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="The filepath to the image data. Supports all data types that can be read by imageio (e.g. tif, png, ...) "
        "or elf.io.open_file (e.g. hdf5, zarr, mrc). For the latter you also need to pass the 'key' parameter."
    )
    parser.add_argument(
        "-o", "--output_path", required=True,
        help="The filepath to store the instance segmentation. The current support stores segmentation in a 'tif' file."
    )
    parser.add_argument(
        "-e", "--embedding_path", default=None, type=str, help="The path where the embeddings will be saved."
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
        help="The number of spatial dimensions in the data. Please specify this if your data has a channel dimension."
    )
    parser.add_argument(
        "--amg", action="store_true", help="Whether to use automatic mask generation with the model."
    )
    parser.add_argument(
        "-d", "--device", default=None,
        help="The device to use for the predictor. Can be one of 'cuda', 'cpu' or 'mps' (only MAC)."
        "By default the most performant available device will be selected."
    )

    args, parameter_args = parser.parse_known_args()

    def _convert_argval(value):
        # The values for the parsed arguments need to be in the expected input structure as provided.
        # i.e. integers and floats should be in their original types.
        try:
            return int(value)
        except ValueError:
            return float(value)

    # NOTE: the script below allows the possibility to catch additional parsed arguments which correspond to
    # the automatic segmentation post-processing parameters (eg. 'center_distance_threshold' in AIS)
    generate_kwargs = {
        parameter_args[i].lstrip("--"): _convert_argval(parameter_args[i + 1]) for i in range(0, len(parameter_args), 2)
    }

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        amg=args.amg,
        is_tiled=args.tile_shape is not None,
    )

    automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=args.input_path,
        output_path=args.output_path,
        embedding_path=args.embedding_path,
        key=args.key,
        ndim=args.ndim,
        tile_shape=args.tile_shape,
        halo=args.halo,
        **generate_kwargs,
    )


if __name__ == "__main__":
    main()
