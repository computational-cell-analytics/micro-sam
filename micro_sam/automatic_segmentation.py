import os
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import imageio.v3 as imageio

from . import util
from .instance_segmentation import (
    get_amg, get_decoder, mask_data_to_segmentation, InstanceSegmentationWithDecoder, AMGBase
)
from .multi_dimensional_segmentation import automatic_3d_segmentation


def automatic_instance_segmentation(
    input_path: Union[os.PathLike, str],
    output_path: Union[os.PathLike, str],
    embedding_path: Optional[Union[os.PathLike, str]] = None,
    model_type: str = util._DEFAULT_MODEL,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    key: Optional[str] = None,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    use_amg: bool = False,
    **generate_kwargs
) -> None:
    """Run automatic segmentation for the input image.

    Args:
        input_path: input_path: The input image file(s). Can either be a single image file (e.g. tif or png),
            or a container file (e.g. hdf5 or zarr).
        output_path: The output path where the instance segmentations will be saved.
        embedding_path: The path where the embeddings are cached already / will be saved.
        model_type: The SegmentAnything model to use. Will use the standard vit_l model by default.
        checkpoint_path: Path to a checkpoint for a custom model.
        key: The key to the input file. This is needed for container files (eg. hdf5 or zarr)
            or to load several images as 3d volume. Provide a glob patterm, eg. "*.tif", for this case.
        ndim: The dimensionality of the data.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction.
        use_amg: Whether to use Automatic Mask Generation (AMG) as the automatic segmentation method.
    """
    predictor, state = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint_path, return_state=True)

    if "decoder_state" in state and not use_amg:  # AIS
        decoder = get_decoder(predictor.model.image_encoder, state["decoder_state"])
        segmenter = get_amg(predictor=predictor, decoder=decoder, is_tiled=tile_shape is not None)
    else:  # AMG
        segmenter = get_amg(predictor=predictor, is_tiled=tile_shape is not None)

    # Load the input image file.
    if isinstance(input_path, np.ndarray):
        image_data = input_path
    else:
        image_data = util.load_image_data(input_path, key)

    if ndim == 3 or image_data.ndim == 3:
        instances = automatic_3d_segmentation(
            volume=image_data,
            predictor=predictor,
            segmentor=segmenter,
            embedding_path=embedding_path,
            tile_shape=tile_shape,
            halo=halo,
            **generate_kwargs
        )
    else:
        # Precompute the image embeddings.
        image_embeddings = util.precompute_image_embeddings(
            predictor=predictor,
            input_=image_data,
            save_path=embedding_path,
            ndim=ndim,
            tile_shape=tile_shape,
            halo=halo,
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

    # Save the instance segmentation
    output_path = Path(output_path).with_suffix(".tif")
    imageio.imwrite(output_path, instances, compression="zlib")


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

    automatic_instance_segmentation(
        input_path=args.input_path,
        output_path=args.output_path,
        embedding_path=args.embedding_path,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        key=args.key,
        ndim=args.ndim,
        tile_shape=args.tile_shape,
        halo=args.halo,
        use_amg=args.amg,
        **generate_kwargs,
    )


if __name__ == "__main__":
    main()
