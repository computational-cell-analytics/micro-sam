from typing import Optional, Tuple, Union

import napari
import numpy as np
import torch

from .. import util
from .annotator import Annotator, annotator
from .util import _initialize_parser


class Annotator2d(Annotator):
    """2D annotator (backward compatibility wrapper).

    This class is a thin wrapper around the unified Annotator class,
    maintaining backward compatibility with existing code.
    """

    def __init__(
        self, viewer: "napari.viewer.Viewer", reset_state: bool = True
    ) -> None:
        super().__init__(viewer, ndim=2, reset_state=reset_state)


def annotator_2d(
    image: np.ndarray,
    embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    prefer_decoder: bool = True,
) -> Optional["napari.viewer.Viewer"]:
    """Start the 2d annotation tool for a given image.

    Args:
        image: The image data.
        embedding_path: Filepath where to save the embeddings
            or the precompted image embeddings computed by `precompute_image_embeddings`.
        segmentation_result: An initial segmentation to load.
            This can be used to correct segmentations with Segment Anything or to save and load progress.
            The segmentation will be loaded as the 'committed_objects' layer.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
            By default, does not return the napari viewer.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster. By default, set to 'False'.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        prefer_decoder: Whether to use decoder based instance segmentation if
            the model used has an additional decoder for instance segmentation.
            By default, set to 'True'.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    return annotator(
        image=image,
        ndim=2,
        embedding_path=embedding_path,
        segmentation_result=segmentation_result,
        model_type=model_type,
        tile_shape=tile_shape,
        halo=halo,
        return_viewer=return_viewer,
        viewer=viewer,
        precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path,
        device=device,
        prefer_decoder=prefer_decoder,
    )


def main():
    """@private"""
    parser = _initialize_parser(
        description="Run interactive segmentation for an image."
    )
    args = parser.parse_args()
    image = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(
            args.segmentation_result, key=args.segmentation_key
        )

    annotator_2d(
        image,
        embedding_path=args.embedding_path,
        segmentation_result=segmentation_result,
        model_type=args.model_type,
        tile_shape=args.tile_shape,
        halo=args.halo,
        precompute_amg_state=args.precompute_amg_state,
        checkpoint_path=args.checkpoint,
        device=args.device,
        prefer_decoder=args.prefer_decoder,
    )
