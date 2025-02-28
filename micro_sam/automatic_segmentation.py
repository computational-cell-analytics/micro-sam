import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import imageio.v3 as imageio

from torch_em.data.datasets.util import split_kwargs

from . import util
from .instance_segmentation import (
    get_amg, get_decoder, mask_data_to_segmentation, InstanceSegmentationWithDecoder,
    AMGBase, AutomaticMaskGenerator, TiledAutomaticMaskGenerator
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
        kwargs: Keyword arguments for the automatic mask generation class.

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
            raise RuntimeError("You have passed 'amg=False', but your model does not contain a segmentation decoder.")
        decoder_state = state["decoder_state"]
        decoder = get_decoder(image_encoder=predictor.model.image_encoder, decoder_state=decoder_state, device=device)

    segmenter = get_amg(predictor=predictor, is_tiled=is_tiled, decoder=decoder, **kwargs)

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
    return_embeddings: bool = False,
    annotate: bool = False,
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
        return_embeddings: Whether to return the precomputed image embeddings.
        annotate: Whether to activate the annotator for continue annotation process.
        generate_kwargs: optional keyword arguments for the generate function of the AMG or AIS class.

    Returns:
        The segmentation result.
    """
    # Avoid overwriting already stored segmentations.
    if output_path is not None:
        output_path = Path(output_path).with_suffix(".tif")
        if os.path.exists(output_path):
            print(f"The segmentation results are already stored at '{os.path.abspath(output_path)}'.")
            return

    # Load the input image file.
    if isinstance(input_path, np.ndarray):
        image_data = input_path
    else:
        image_data = util.load_image_data(input_path, key)

    ndim = image_data.ndim if ndim is None else ndim

    # We perform additional post-processing for AMG-only.
    # Otherwise, we ignore additional post-processing for AIS.
    if isinstance(segmenter, InstanceSegmentationWithDecoder):
        generate_kwargs["output_mode"] = None

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

        # If we run AIS with tiling then we use the same tile shape for the watershed postprocessing.
        if isinstance(segmenter, InstanceSegmentationWithDecoder) and tile_shape is not None:
            generate_kwargs.update({"tile_shape": tile_shape, "halo": halo})

        segmenter.initialize(image=image_data, image_embeddings=image_embeddings, verbose=verbose)
        masks = segmenter.generate(**generate_kwargs)

        if isinstance(masks, list):
            # whether the predictions from 'generate' are list of dict,
            # which contains additional info req. for post-processing, eg. area per object.
            if len(masks) == 0:
                instances = np.zeros(image_data.shape[:2], dtype="uint32")
            else:
                instances = mask_data_to_segmentation(masks, with_background=True, min_object_size=0)
        else:
            # if (raw) predictions provided, store them as it is w/o further post-processing.
            instances = masks

    else:
        if (image_data.ndim != 3) and (image_data.ndim != 4 and image_data.shape[-1] != 3):
            raise ValueError(f"The inputs does not match the shape expectation of 3d inputs: {image_data.shape}")

        outputs = automatic_3d_segmentation(
            volume=image_data,
            predictor=predictor,
            segmentor=segmenter,
            embedding_path=embedding_path,
            tile_shape=tile_shape,
            halo=halo,
            verbose=verbose,
            return_embeddings=return_embeddings,
            **generate_kwargs
        )

        if return_embeddings:
            instances, image_embeddings = outputs
        else:
            instances = outputs

    # Allow opening the automatic segmentation in the annotator for further annotation, if desired.
    if annotate:
        from micro_sam.sam_annotator import annotator_2d, annotator_3d
        annotator_function = annotator_2d if ndim == 2 else annotator_3d

        viewer = annotator_function(
            image=image_data,
            model_type=predictor.model_name,
            embedding_path=embedding_path,
            segmentation_result=instances,  # Initializes the automatic segmentation to the annotator.
            tile_shape=tile_shape,
            halo=halo,
            return_viewer=True,  # Returns the viewer, which allows the user to store the updated segmentations.
        )

        # Start the GUI here
        import napari
        napari.run()

        # We extract the segmentation in "committed_objects" layer, where the user either:
        # a) Performed interactive segmentation / corrections and committed them, OR
        # b) Did not do anything and closed the annotator, i.e. keeps the segmentations as it is.
        instances = viewer.layers["committed_objects"].data

    # Save the instance segmentation, if 'output_path' provided.
    if output_path is not None:
        imageio.imwrite(output_path, instances, compression="zlib")
        print(f"The segmentation results are stored at '{os.path.abspath(output_path)}'.")

    if return_embeddings:
        return instances, image_embeddings
    else:
        return instances


def _get_inputs_from_paths(paths, pattern):
    "Function to get all filepaths in a directory."

    if isinstance(paths, str):
        paths = [paths]

    fpaths = []
    for path in paths:
        if os.path.isdir(path):  # if the path is a directory, fetch all inputs provided with a pattern.
            assert pattern is not None, \
                f"You must provide a pattern to search for files in the directory: '{os.path.abspath(path)}'."
            fpaths.extend(glob(os.path.join(path, pattern)))

        else:  # Otherwise, it is just one filepath.
            fpaths.append(path)

    return fpaths


def _has_extension(fpath: Union[os.PathLike, str]) -> bool:
    "Returns whether the provided path has an extension or not."
    return bool(os.path.splitext(fpath)[1])


def main():
    """@private"""
    import argparse

    available_models = list(util.get_model_names())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(description="Run automatic segmentation for an image.")
    parser.add_argument(
        "-i", "--input_path", required=True, type=str, nargs="+",
        help="The filepath to the image data. Supports all data types that can be read by imageio (e.g. tif, png, ...) "
        "or elf.io.open_file (e.g. hdf5, zarr, mrc). For the latter you also need to pass the 'key' parameter."
    )
    parser.add_argument(
        "-o", "--output_path", required=True, type=str,
        help="The filepath to store the instance segmentation. The current support stores segmentation in a 'tif' file."
    )
    parser.add_argument(
        "-e", "--embedding_path", default=None, type=str, help="The path where the embeddings will be saved."
    )
    parser.add_argument(
        "--pattern", type=str, help="Pattern / wildcard for selecting files in a folder. To select all files use '*'."
    )
    parser.add_argument(
        "-k", "--key", default=None, type=str,
        help="The key for opening data with elf.io.open_file. This is the internal path for a hdf5 or zarr container, "
        "for an image stack it is a wild-card, e.g. '*.png' and for mrc it is 'data'."
    )
    parser.add_argument(
        "-m", "--model_type", default=util._DEFAULT_MODEL, type=str,
        help=f"The segment anything model that will be used, one of {available_models}."
    )
    parser.add_argument(
        "-c", "--checkpoint", default=None, type=str, help="Checkpoint from which the SAM model will be loaded."
    )
    parser.add_argument(
        "--tile_shape", nargs="+", type=int, help="The tile shape for using tiled prediction.", default=None
    )
    parser.add_argument(
        "--halo", nargs="+", type=int, help="The halo for using tiled prediction.", default=None
    )
    parser.add_argument(
        "-n", "--ndim", default=None, type=int,
        help="The number of spatial dimensions in the data. Please specify this if your data has a channel dimension."
    )
    parser.add_argument(
        "--mode", default="auto", type=str,
        help="The choice of automatic segmentation with the Segment Anything models. Either 'auto', 'amg' or 'ais'."
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Whether to continue annotation after the automatic segmentation is generated."
    )
    parser.add_argument(
        "-d", "--device", default=None, type=str,
        help="The device to use for the predictor. Can be one of 'cuda', 'cpu' or 'mps' (only MAC)."
        "By default the most performant available device will be selected."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to allow verbosity of outputs."
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
    extra_kwargs = {
        parameter_args[i].lstrip("--"): _convert_argval(parameter_args[i + 1]) for i in range(0, len(parameter_args), 2)
    }

    # Separate extra arguments as per where they should be passed in the automatic segmentation class.
    # This is done to ensure the extra arguments are allocated to the desired location.
    # eg. for AMG, 'points_per_side' is expected by '__init__',
    # and 'stability_score_thresh' is expected in 'generate' method.
    amg_class = AutomaticMaskGenerator if args.tile_shape is None else TiledAutomaticMaskGenerator
    amg_kwargs, generate_kwargs = split_kwargs(amg_class, **extra_kwargs)

    # Validate for the expected automatic segmentation mode.
    # By default, it is set to 'auto', i.e. searches for the decoder state to prioritize AIS for finetuned models.
    # Otherwise, runs AMG for all models in any case.
    amg = None
    if args.mode != "auto":
        assert args.mode in ["ais", "amg"], \
            f"'{args.mode}' is not a valid automatic segmentation mode. Please choose either 'amg' or 'ais'."
        amg = (args.mode == "amg")

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        amg=amg,
        is_tiled=args.tile_shape is not None,
        **amg_kwargs,
    )

    # Get the filepaths to input images.
    # Check wiether the inputs are as expected, otherwise assort them.
    input_path = args.input_path
    pattern = args.pattern
    input_paths = _get_inputs_from_paths(input_path, pattern)

    assert len(input_paths) > 0, "We internally could not extract any image data."

    # Get other paths (i.e. for storing outputs and embedding path)
    output_path = args.output_path
    embedding_path = args.embedding_path
    has_one_input = len(input_paths) == 1

    # Run automatic segmentation per image.
    for path in tqdm(input_paths, desc="Run automatic segmentation"):
        if has_one_input:  # if we have one image only.
            _output_fpath = str(Path(output_path).with_suffix(".tif"))
            _embedding_fpath = embedding_path

        else:  # if we have multiple image, we need to make the other target filepaths compatible.
            # Let's check for 'embedding_path'.
            _embedding_fpath = embedding_path
            if embedding_path:
                if _has_extension(embedding_path):  # in this case, use filename as addl. suffix to provided path.
                    _embedding_fpath = str(Path(embedding_path).with_suffix(".zarr"))
                    _embedding_fpath = _embedding_fpath.replace(".zarr", f"_{Path(path).stem}.zarr")
                else:   # otherwise, for directory, use image filename for multiple images.
                    os.makedirs(embedding_path, exist_ok=True)
                    _embedding_fpath = os.path.join(embedding_path, Path(os.path.basename(path)).with_suffix(".zarr"))

            # Next, let's check for output file to store segmentation.
            if _has_extension(output_path):  # in this case, use filename as addl. suffix to provided path.
                _output_fpath = str(Path(output_path).with_suffix(".tif"))
                _output_fpath = _output_fpath.replace(".tif", f"_{Path(path).stem}.tif")
            else:  # otherwise, for directory, use image filename for multiple images.
                os.makedirs(output_path, exist_ok=True)
                _output_fpath = os.path.join(output_path, Path(os.path.basename(path)).with_suffix(".tif"))

        automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=path,
            output_path=_output_fpath,
            embedding_path=_embedding_fpath,
            key=args.key,
            ndim=args.ndim,
            tile_shape=args.tile_shape,
            halo=args.halo,
            annotate=args.annotate,
            verbose=args.verbose,
            **generate_kwargs,
        )


if __name__ == "__main__":
    main()
