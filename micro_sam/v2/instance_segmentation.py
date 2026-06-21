"""Automatic mask generation (AMG) for the SAM2 model.

This implements the grid-based automatic mask generation (AMG) of SAM2 with the same
`initialize` / `generate` interface as the micro-sam v1 `AutomaticMaskGenerator`. The expensive
grid prediction happens in `initialize`, the cheap conversion to an instance segmentation in
`generate`. AMG is supported for 2d images and for 3d volumes; the 3d segmentation runs AMG
slice-by-slice and stitches the per-slice results across z with the multi-dimensional segmentation
stitching (`micro_sam.v1.multi_dimensional_segmentation.merge_instance_segmentation_3d`).

For large images a tiled backend (`TiledAutomaticMaskGenerationSegmenter`) splits the image into
tiles with a halo, runs AMG per tile and stitches the per-tile results, matching the tiled
interface used elsewhere in micro-sam (and by the GUI).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from bioimage_cpp.utils import Blocking

from torch_em.transform.raw import normalize

from micro_sam.util import mask_data_to_segmentation
from micro_sam.v1.inference import _merge_segmentations
from micro_sam.v1.multi_dimensional_segmentation import merge_instance_segmentation_3d
from micro_sam.v2.util import precompute_image_embeddings, set_precomputed


def _to_amg_input(image: np.ndarray, ensure_8bit: bool = True) -> np.ndarray:
    """Convert an image into the HWC uint8 representation expected by SAM2's mask generator.

    Args:
        image: The input image, either grayscale (Y, X) or RGB (Y, X, 3).
        ensure_8bit: Whether to rescale images whose values exceed 255 to the uint8 range.

    Returns:
        The image as a HWC uint8 array.
    """
    if ensure_8bit and image.max() > 255:
        image = normalize(image) * 255
    if image.ndim == 2:  # Convert single channel images to RGB images.
        image = np.stack([image] * 3, axis=-1)
    return image.astype("uint8")


class AutomaticMaskGenerationSegmenter:
    """Generates an instance segmentation for the SAM2 model using grid-based prompting (AMG).

    Wraps the native `sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator` and exposes the
    same `initialize` / `generate` interface as the micro-sam v1 `AutomaticMaskGenerator`, so it
    can be used both for single 2d images and, via `automatic_3d_segmentation`, for 3d volumes.

    The parameters that control the (expensive) mask prediction, e.g. `points_per_side` and the
    quality thresholds, are passed to the constructor. The (cheap) conversion of the predicted
    masks into an instance segmentation is controlled via `generate`.

    The image embeddings are computed and cached via `micro_sam.v2.util.precompute_image_embeddings`
    (or taken from precomputed embeddings passed to `initialize`, e.g. by the GUI) and set on the
    predictor with `set_precomputed`, so the grid prediction reuses them instead of recomputing.

    Use this class as follows:
    ```python
    segmenter = AutomaticMaskGenerationSegmenter(model)
    segmenter.initialize(image)  # Run the grid prediction, this is the expensive computation.
    masks = segmenter.generate(min_object_size=50)  # Convert to an instance segmentation, fast.
    ```

    Args:
        model: The SAM2 model, loaded via `micro_sam.v2.util.get_sam2_model`.
        model_type: The SAM2 model type, e.g. 'hvit_t'. Used to tag cached embeddings; only needed
            when caching embeddings via `save_path`.
        points_per_side: The number of points sampled along one side of the image. By default '32'.
        points_per_batch: The number of points run simultaneously by the model. By default '64'.
        pred_iou_thresh: Filter threshold in [0, 1] using the model's predicted mask quality.
            By default '0.8'.
        stability_score_thresh: Filter threshold in [0, 1] using the stability of the mask under
            changes to the binarization cutoff. By default '0.9'. This is lower than SAM2's native
            default of '0.95' because the embeddings here are computed with percentile normalization
            (the micro-sam SAM2 convention), under which masks score marginally lower in stability.
        ensure_8bit: Whether to rescale images whose values exceed 255 to the uint8 range.
            By default 'True'.
        kwargs: Additional keyword arguments forwarded to `SAM2AutomaticMaskGenerator`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: Optional[str] = None,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.9,
        ensure_8bit: bool = True,
        **kwargs,
    ) -> None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            output_mode="binary_mask",
            **kwargs,
        )
        # The embedding signature written by 'precompute_image_embeddings' reads 'model_type' and
        # 'model_name' off the predictor. The video predictor gets these in 'get_sam2_model', but the
        # image predictor used here does not, so we set them (matching the GUI, see _state.py).
        predictor = self._mask_generator.predictor
        predictor.model_type = model_type or getattr(model, "model_type", None) or "hvit"
        predictor.model_name = model_type or getattr(model, "model_name", None) or predictor.model_type
        self._ensure_8bit = ensure_8bit
        self._masks = None
        self._original_size = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the segmenter has already been initialized."""
        return self._is_initialized

    def _generate_from_precomputed(self) -> List[Dict[str, Any]]:
        """Run the grid-based mask prediction reusing the embeddings already set on the predictor.

        The embeddings are expected to be set on `self._mask_generator.predictor` (via
        `precompute_image_embeddings` or `set_precomputed`). We temporarily neutralize the
        predictor's `set_image`, which the native mask generator calls once per crop, so that it
        reuses the precomputed embeddings instead of recomputing them. This is only valid for the
        single-crop case (`crop_n_layers=0`, the default).
        """
        predictor = self._mask_generator.predictor
        dummy = np.zeros((*self._original_size, 3), dtype="uint8")
        original_set_image = predictor.set_image
        predictor.set_image = lambda *args, **kwargs: None
        try:
            masks = self._mask_generator.generate(dummy)
        finally:
            predictor.set_image = original_set_image
        return masks

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: bool = False,
        pbar_init: Optional[callable] = None,
        pbar_update: Optional[callable] = None,
        **kwargs,
    ) -> None:
        """Run the grid-based mask prediction and store the resulting masks.

        The image embeddings are computed (and cached if `save_path` is given) via
        `micro_sam.v2.util.precompute_image_embeddings`, or taken from `image_embeddings` if
        provided (e.g. by the GUI), and set on the predictor with `set_precomputed`. The grid
        prediction then reuses these embeddings instead of recomputing them.

        Args:
            image: The input image, grayscale (Y, X) or RGB (Y, X, 3). When `image_embeddings` is
                given the image content is not used (only the precomputed embeddings are).
            image_embeddings: Optional precomputed image embeddings. See `precompute_image_embeddings`.
            i: Index for the image data. Only relevant for externally precomputed embeddings; for
                a single 2d image (the per-slice case) it must be None.
            save_path: Optional path to cache the computed embeddings in a zarr container.
            verbose: Verbosity flag. By default 'False'.
            pbar_init: Callback to initialize an external progress bar.
            pbar_update: Callback to update an external progress bar.
            kwargs: Additional arguments, ignored. Kept for interface compatibility.
        """
        predictor = self._mask_generator.predictor
        if image_embeddings is None:
            # Computes (or loads from save_path) the embeddings and sets them on the predictor.
            precompute_image_embeddings(
                predictor, image, save_path=save_path, ndim=2, verbose=verbose,
                pbar_init=pbar_init, pbar_update=pbar_update,
            )
        else:
            set_precomputed(predictor, image_embeddings, i=i)

        self._original_size = tuple(int(s) for s in predictor._orig_hw[0])
        self._masks = self._generate_from_precomputed()
        self._is_initialized = True

    def generate(
        self,
        min_object_size: int = 0,
        max_object_size: Optional[int] = None,
        with_background: bool = True,
        output_mode: str = "instance_segmentation",
    ) -> Union[List[Dict[str, Any]], np.ndarray]:
        """Convert the predicted masks into an instance segmentation.

        Args:
            min_object_size: The minimal size of an object in pixels. By default '0'.
            max_object_size: The maximal size of an object in pixels. By default 'None'.
            with_background: Whether to remove the largest object, which often covers the
                background for AMG. By default 'True'.
            output_mode: The form masks are returned in. Either 'instance_segmentation' to return
                a single label array, or 'binary_mask' to return the list of mask dictionaries.
                By default 'instance_segmentation'.

        Returns:
            The instance segmentation as a uint32 array, or the list of mask dictionaries.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "AutomaticMaskGenerationSegmenter has not been initialized. Call initialize first."
            )
        if output_mode != "instance_segmentation":
            return self._masks
        if len(self._masks) == 0:  # No masks were found.
            return np.zeros(self._original_size, dtype="uint32")
        return mask_data_to_segmentation(
            masks=self._masks,
            shape=self._original_size,
            min_object_size=min_object_size,
            max_object_size=max_object_size,
            with_background=with_background,
        )


class TiledAutomaticMaskGenerationSegmenter(AutomaticMaskGenerationSegmenter):
    """Generates a tiled instance segmentation for the SAM2 model using grid-based prompting (AMG).

    Implements the same functionality as `AutomaticMaskGenerationSegmenter`, but the image is split
    into tiles with a halo, AMG is run on each tile, and the per-tile instance segmentations are
    stitched back into a single segmentation. This is the tiled backend used for large images (and
    by the GUI). The `initialize` / `generate` interface matches the micro-sam v1
    `TiledAutomaticMaskGenerator`.

    Args:
        model: The SAM2 model, loaded via `micro_sam.v2.util.get_sam2_model`.
        kwargs: Additional keyword arguments for `AutomaticMaskGenerationSegmenter`.
    """

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        tile_shape: Optional[Tuple[int, int]] = None,
        halo: Optional[Tuple[int, int]] = None,
        i: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Run the grid-based mask prediction tile-by-tile and store the per-tile masks.

        Args:
            image: The input image, grayscale (Y, X) or RGB (Y, X, 3).
            tile_shape: The tile shape for the tiled prediction, (y, x).
            halo: The overlap between the tiles, (y, x).
            i: Index of the slice to segment if `image` has three spatial dimensions.
            verbose: Verbosity flag. By default 'False'.
            kwargs: Additional arguments, ignored. Kept for interface compatibility.
        """
        if tile_shape is None or halo is None:
            raise ValueError("Both 'tile_shape' and 'halo' have to be passed for the tiled segmenter.")

        if image.ndim == 3 and image.shape[-1] != 3 and i is not None:
            image = image[i]
        image = _to_amg_input(image, ensure_8bit=self._ensure_8bit)
        self._original_size = image.shape[:2]

        self._tiling = Blocking([0, 0], list(self._original_size), list(tile_shape))
        self._halo = tuple(halo)

        self._masks = []
        n_tiles = self._tiling.number_of_blocks
        for tile_id in tqdm(range(n_tiles), desc="Compute masks for tile", disable=not verbose):
            block = self._tiling.get_block_with_halo(tile_id, list(self._halo)).outer_block
            bb = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))
            self._masks.append(self._mask_generator.generate(image[bb]))

        self._is_initialized = True

    def generate(
        self,
        min_object_size: int = 0,
        max_object_size: Optional[int] = None,
        with_background: bool = True,
        output_mode: str = "instance_segmentation",
    ) -> np.ndarray:
        """Convert the per-tile masks into an instance segmentation and stitch the tiles.

        Args:
            min_object_size: The minimal size of an object in pixels. By default '0'.
            max_object_size: The maximal size of an object in pixels. By default 'None'.
            with_background: Whether to remove the largest object per tile. By default 'True'.
            output_mode: The form masks are returned in. Only 'instance_segmentation' is supported
                for the tiled segmenter. By default 'instance_segmentation'.

        Returns:
            The stitched instance segmentation as a uint32 array.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "TiledAutomaticMaskGenerationSegmenter has not been initialized. Call initialize first."
            )
        if output_mode != "instance_segmentation":
            raise ValueError("The tiled segmenter only supports output_mode='instance_segmentation'.")

        segmentation = np.zeros(self._original_size, dtype="uint32")
        offset = 0
        for tile_id, masks in enumerate(self._masks):
            block = self._tiling.get_block_with_halo(tile_id, list(self._halo)).outer_block
            bb = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))
            tile_shape = tuple(end - begin for begin, end in zip(block.begin, block.end))

            if len(masks) == 0:
                continue
            this_seg = mask_data_to_segmentation(
                masks=masks,
                shape=tile_shape,
                min_object_size=min_object_size,
                max_object_size=max_object_size,
                with_background=with_background,
            )

            # Offset the per-tile instance ids so that they stay unique across tiles.
            max_id = int(this_seg.max())
            if max_id == 0:
                continue
            this_seg[this_seg != 0] += offset
            offset += max_id

            if tile_id == 0:
                segmentation[bb] = this_seg
            else:
                segmentation[bb] = _merge_segmentations(this_seg, segmentation[bb])

        return segmentation


def get_amg_segmenter(
    model: torch.nn.Module, is_tiled: bool = False, **kwargs
) -> AutomaticMaskGenerationSegmenter:
    """Get the SAM2 automatic mask generation segmenter.

    Args:
        model: The SAM2 model, loaded via `micro_sam.v2.util.get_sam2_model`.
        is_tiled: Whether to use the tiled segmenter for large images. By default 'False'.
        kwargs: Additional keyword arguments for the segmenter.

    Returns:
        The automatic mask generation segmenter, either `TiledAutomaticMaskGenerationSegmenter`
        (if tiled) or `AutomaticMaskGenerationSegmenter`.
    """
    if is_tiled:
        return TiledAutomaticMaskGenerationSegmenter(model, **kwargs)
    return AutomaticMaskGenerationSegmenter(model, **kwargs)


def automatic_3d_segmentation(
    volume: np.ndarray,
    segmenter: AutomaticMaskGenerationSegmenter,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """Automatically segment objects in a volume with the SAM2 mask generator.

    Segments each slice individually in 2d with AMG and then merges the slices across z based on
    the overlap of objects between adjacent slices, using the multi-dimensional segmentation
    stitching (`merge_instance_segmentation_3d`). When `tile_shape` and `halo` are given, the
    per-slice AMG is run with the tiled segmenter.

    Args:
        volume: The input volume, shape (Z, Y, X).
        segmenter: The automatic mask generation segmenter. Use a
            `TiledAutomaticMaskGenerationSegmenter` together with `tile_shape` and `halo`.
        with_background: Whether the segmentation has background. By default 'True'.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing operation.
            The value is the number of iterations for the closing. By default 'None'.
        min_z_extent: Require a minimal extent in z for the segmented objects. This can help to
            prevent segmentation artifacts. By default 'None'.
        tile_shape: The tile shape for the tiled per-slice prediction, (y, x). By default 'None'.
        halo: The overlap between the tiles, (y, x). By default 'None'.
        verbose: Verbosity flag. By default 'True'.
        kwargs: Keyword arguments for the 'generate' method of the segmenter.

    Returns:
        The 3d instance segmentation, uint32 array of shape (Z, Y, X).
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3d volume of shape (Z, Y, X), got shape {volume.shape}.")

    init_kwargs = {}
    if tile_shape is not None and halo is not None:
        init_kwargs = {"tile_shape": tile_shape, "halo": halo}

    segmentation = np.zeros(volume.shape, dtype="uint32")
    offset = 0
    for i in tqdm(range(volume.shape[0]), desc="Segment slices", disable=not verbose):
        segmenter.initialize(volume[i], verbose=False, **init_kwargs)
        seg = segmenter.generate(**kwargs)

        # Offset the per-slice instance ids so that they are unique across the whole volume.
        max_z = int(seg.max())
        if max_z == 0:
            continue
        seg[seg != 0] += offset
        offset += max_z
        segmentation[i] = seg

    segmentation = merge_instance_segmentation_3d(
        segmentation,
        beta=0.5,
        with_background=with_background,
        gap_closing=gap_closing,
        min_z_extent=min_z_extent,
        verbose=verbose,
    )
    return segmentation
