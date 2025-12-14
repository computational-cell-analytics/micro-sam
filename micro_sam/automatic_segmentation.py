import os
import warnings
from glob import glob
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import imageio.v3 as imageio

from torch_em.data.datasets.util import split_kwargs

from . import util
from .instance_segmentation import (
    get_amg, get_decoder,
    InstanceSegmentationWithDecoder, AMGBase, AutomaticMaskGenerator, TiledAutomaticMaskGenerator
)
from .multi_dimensional_segmentation import automatic_3d_segmentation, automatic_tracking_implementation


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
        device: The torch device. By default, automatically chooses the best available device.
        amg: Whether to perform automatic segmentation in AMG mode.
            Otherwise AIS will be used, which requires a special segmentation decoder.
            If not specified AIS will be used if it is available and otherwise AMG will be used.
        is_tiled: Whether to return segmenter for performing segmentation in tiling window style.
            By default, set to 'False'.
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


def _add_suffix_to_output_path(output_path: Union[str, os.PathLike], suffix: str) -> str:
    fpath = Path(output_path).resolve()
    fext = fpath.suffix if fpath.suffix else ".tif"
    return str(fpath.with_name(f"{fpath.stem}{suffix}{fext}"))


def automatic_tracking(
    predictor: util.SamPredictor,
    segmenter: Union[AMGBase, InstanceSegmentationWithDecoder],
    input_path: Union[Union[os.PathLike, str], np.ndarray],
    output_path: Optional[Union[os.PathLike, str]] = None,
    embedding_path: Optional[Union[os.PathLike, str]] = None,
    key: Optional[str] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    return_embeddings: bool = False,
    annotate: bool = False,
    batch_size: int = 1,
    **generate_kwargs
) -> Tuple[np.ndarray, List[Dict]]:
    """Run automatic tracking for the input timeseries.

    Args:
        predictor: The Segment Anything model.
        segmenter: The automatic instance segmentation class.
        input_path: input_path: The input image file(s). Can either be a single image file (e.g. tif or png),
            or a container file (e.g. hdf5 or zarr).
        output_path: The folder where the tracking outputs will be saved in CTC format.
        embedding_path: The path where the embeddings are cached already / will be saved.
        key: The key to the input file. This is needed for container files (eg. hdf5 or zarr)
            or to load several images as 3d volume. Provide a glob patterm, eg. "*.tif", for this case.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction. By default prediction is run without tiling.
        verbose: Verbosity flag. By default, set to 'True'.
        return_embeddings: Whether to return the precomputed image embeddings.
            By default, does not return the embeddings.
        annotate: Whether to activate the annotator for continue annotation process.
            By default, does not activate the annotator.
        batch_size: The batch size to compute image embeddings over tiles / z-planes.
            By default, does it sequentially, i.e. one after the other.
        generate_kwargs: optional keyword arguments for the generate function of the AMG or AIS class.

    Returns:
        The tracking result as a timeseries, where each object is labeled by its track id.
        The lineages representing cell divisions, stored as a dictionary.
    """
    # Load the input image file.
    if isinstance(input_path, np.ndarray):
        image_data = input_path
    else:
        image_data = util.load_image_data(input_path, key)

    if (image_data.ndim != 3) and (image_data.ndim != 4 and image_data.shape[-1] != 3):
        raise ValueError(f"The inputs does not match the shape expectation of 3d inputs: {image_data.shape}")

    gap_closing, min_time_extent = generate_kwargs.get("gap_closing"), generate_kwargs.get("min_time_extent")
    segmentation, lineage, image_embeddings = automatic_tracking_implementation(
        image_data,
        predictor,
        segmenter,
        embedding_path=embedding_path,
        gap_closing=gap_closing,
        min_time_extent=min_time_extent,
        tile_shape=tile_shape,
        halo=halo,
        verbose=verbose,
        batch_size=batch_size,
        return_embeddings=True,
        output_folder=output_path,
        **generate_kwargs,
    )

    if annotate:
        # TODO We need to support initialization of the tracking annotator with the tracking result for this.
        raise NotImplementedError("Annotation after running the automated tracking is currently not supported.")

    if return_embeddings:
        return segmentation, lineage, image_embeddings
    else:
        return segmentation, lineage


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
    batch_size: int = 1,
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
        halo: Overlap of the tiles for tiled prediction. By default prediction is run without tiling.
        verbose: Verbosity flag. By default, set to 'True'.
        return_embeddings: Whether to return the precomputed image embeddings.
            By default, does not return the embeddings.
        annotate: Whether to activate the annotator for continue annotation process.
            By default, does not activate the annotator.
        batch_size: The batch size to compute image embeddings over tiles / z-planes.
            By default, does it sequentially, i.e. one after the other.
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
            batch_size=batch_size,
        )
        initialize_kwargs = dict(image=image_data, image_embeddings=image_embeddings, verbose=verbose)

        # If we run AIS with tiling then we use the same tile shape for the watershed postprocessing.
        # In this case, we also add the batch size to the initialize kwargs,
        # so that the segmentation decoder can be applied in a batched fashion.
        if isinstance(segmenter, InstanceSegmentationWithDecoder) and tile_shape is not None:
            generate_kwargs.update({"tile_shape": tile_shape, "halo": halo})
            initialize_kwargs["batch_size"] = batch_size

        segmenter.initialize(**initialize_kwargs)
        instances = segmenter.generate(**generate_kwargs)

    else:
        if (image_data.ndim != 3) and (image_data.ndim != 4 and image_data.shape[-1] != 3):
            raise ValueError(f"The inputs does not match the shape expectation of 3d inputs: {image_data.shape}")

        instances, image_embeddings = automatic_3d_segmentation(
            volume=image_data,
            predictor=predictor,
            segmentor=segmenter,
            embedding_path=embedding_path,
            tile_shape=tile_shape,
            halo=halo,
            verbose=verbose,
            return_embeddings=True,
            batch_size=batch_size,
            **generate_kwargs
        )

    # Before starting to annotate, if at all desired, store the automatic segmentations in the first stage.
    if output_path is not None:
        _output_path = _add_suffix_to_output_path(output_path, "_automatic") if annotate else output_path
        imageio.imwrite(_output_path, instances, compression="zlib")
        if verbose:
            print(f"The automatic segmentation results are stored at '{os.path.abspath(_output_path)}'.")

    # Allow opening the automatic segmentation in the annotator for further annotation, if desired.
    if annotate:
        from micro_sam.sam_annotator import annotator_2d, annotator_3d
        annotator_function = annotator_2d if ndim == 2 else annotator_3d

        viewer = annotator_function(
            image=image_data,
            model_type=predictor.model_name,
            embedding_path=image_embeddings,  # Providing the precomputed image embeddings.
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
            if verbose:
                print(f"The final segmentation results are stored at '{os.path.abspath(output_path)}'.")

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
        if os.path.isfile(path):  # It is just one filepath.
            fpaths.append(path)
        else:  # Otherwise, if the path is a directory, fetch all inputs provided with a pattern.
            assert pattern is not None, \
                f"You must provide a pattern to search for files in the directory: '{os.path.abspath(path)}'."
            fpaths.extend(glob(os.path.join(path, pattern)))

    return fpaths


def main():
    """@private"""
    import argparse

    available_models = list(util.get_model_names())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(
        description="Run automatic segmentation or tracking for 2d, 3d or timeseries data.\n"
        "Either a single input file or multiple input files are supported. You can specify multiple files "
        "by either providing multiple filepaths to the '--i/--input_paths' argument, or by providing an argument "
        "to '--pattern' to use a wildcard pattern ('*') for selecting multiple files.\n"
        "NOTE: for automatic 3d segmentation or tracking the data has to be stored as volume / timeseries, "
        "stacking individual tif images is not supported.\n"
        "Segmentation is performed using one of the two modes supported by micro_sam: \n"
        "automatic instance segmentation (AIS) or automatic mask generation (AMG).\n"
        "In addition to the options listed below, "
        "you can also passed additional arguments for these two segmentation modes:\n"
        "For AIS: '--center_distance_threshold', '--boundary_distance_threshold' and other arguments of `InstanceSegmentationWithDecoder.generate`."  # noqa
        "For AMG: '--pred_iou_thresh', '--stability_score_thresh' and other arguments of `AutomaticMaskGenerator.generate`."  # noqa
    )
    parser.add_argument(
        "-i", "--input_path", required=True, type=str, nargs="+",
        help="The filepath(s) to the image data. Supports all data types that can be read by imageio (e.g. tif, png, ...) "  # noqa
        "or elf.io.open_file (e.g. hdf5, zarr, mrc). For the latter you also need to pass the 'key' parameter."
    )
    parser.add_argument(
        "-o", "--output_path", required=True, type=str,
        help="The filepath to store the results. If multiple inputs are provied, "
        "this should be a folder. For a single image, you should provide the path to a tif file for the output segmentation."  # noqa
        "NOTE: Segmentation results are stored as tif files, tracking results in the CTC fil format ."
    )
    parser.add_argument(
        "-e", "--embedding_path", default=None, type=str,
        help="An optional path where the embeddings will be saved. If multiple inputs are provided, "
        "this should be a folder. Otherwise you can store embeddings in single zarr file."
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
        "--batch_size", type=int, default=1,
        help="The batch size for computing image embeddings over tiles or z-plane. "
        "By default, computes the image embeddings for one tile / z-plane at a time."
    )
    parser.add_argument(
        "--tracking", action="store_true", help="Run automatic tracking instead of instance segmentation. "
        "NOTE: It is only supported for timeseries inputs."
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

    # Get the filepaths to input images (and other paths to store stuff, eg. segmentations and embeddings)
    # Check whether the inputs are as expected, otherwise assort them.
    input_paths = _get_inputs_from_paths(args.input_path, args.pattern)
    assert len(input_paths) > 0, "'micro-sam' could not extract any image data internally."

    output_path = args.output_path
    embedding_path = args.embedding_path
    has_one_input = len(input_paths) == 1

    instance_seg_function = automatic_tracking if args.tracking else partial(
        automatic_instance_segmentation, ndim=args.ndim
    )

    # Run automatic segmentation per image.
    for input_path in tqdm(input_paths, desc="Run automatic " + ("tracking" if args.tracking else "segmentation")):
        if has_one_input:  # When we have only one image / volume.
            _embedding_fpath = embedding_path  # Either folder or zarr file, would work for both.

            output_fdir = os.path.splitext(output_path)[0]
            os.makedirs(output_fdir, exist_ok=True)

            # For tracking, we ensure that the output path is a folder,
            # i.e. does not have an extension. We throw a warning if the user provided an extension.
            if args.tracking:
                if os.path.splitext(output_path)[-1]:
                    warnings.warn(
                        f"The output folder has an extension '{os.path.splitext(output_path)[-1]}'. "
                        "We remove it and treat it as a folder to store tracking outputs in CTC format."
                    )
                _output_fpath = output_fdir
            else:  # Otherwise, we can store outputs for user directly in the provided filepath, ensuring extension .tif
                _output_fpath = f"{output_fdir}.tif"

        else:  # When we have multiple images.
            # Get the input filename, without the extension.
            input_name = str(Path(input_path).stem)

            # Let's check the 'embedding_path'.
            if embedding_path is None:  # For computing embeddings on-the-fly, we don't care about the path logic.
                _embedding_fpath = embedding_path
            else:  # Otherwise, store each embeddings inside a folder.
                embedding_folder = os.path.splitext(embedding_path)[0]  # Treat the provided embedding path as folder.
                os.makedirs(embedding_folder, exist_ok=True)
                _embedding_fpath = os.path.join(embedding_folder, f"{input_name}.zarr")  # Create each embedding file.

            # Get the output folder name.
            output_folder = os.path.splitext(output_path)[0]
            os.makedirs(output_folder, exist_ok=True)

            # Next, let's check for output file to store segmentation (or tracks).
            if args.tracking:  # For tracking, we store CTC outputs in subfolders, with input_name as folder.
                _output_fpath = os.path.join(output_folder, input_name)
            else:  # Otherwise, store each result inside a folder.
                _output_fpath = os.path.join(output_folder, f"{input_name}.tif")

        instance_seg_function(
            predictor=predictor,
            segmenter=segmenter,
            input_path=input_path,
            output_path=_output_fpath,
            embedding_path=_embedding_fpath,
            key=args.key,
            tile_shape=args.tile_shape,
            halo=args.halo,
            annotate=args.annotate,
            verbose=args.verbose,
            batch_size=args.batch_size,
            **generate_kwargs,
        )
