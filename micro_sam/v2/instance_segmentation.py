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

from bioimage_cpp.utils import Blocking

from micro_sam.util import mask_data_to_segmentation
from micro_sam.v1.inference import _merge_segmentations
from micro_sam.v1.multi_dimensional_segmentation import merge_instance_segmentation_3d
from micro_sam.v2.util import precompute_image_embeddings, set_precomputed, _load_list_datasets


def _set_image_predictor_from_backbone(predictor, fpn_list, pos_enc_list, original_size, i):
    """Set a SAM2 image predictor's ``_features`` for slice ``i`` from stored backbone outputs.

    ``fpn_list`` / ``pos_enc_list`` are the per-level backbone FPN outputs and positional encodings,
    each indexable as ``level[i] -> (1, C, H, W)``. This reconstructs the image predictor's features
    exactly as ``SAM2ImagePredictor.set_image`` does (``_prepare_backbone_features`` + reshape) but
    without re-running the (expensive) image encoder.
    """
    model = predictor.model
    device = next(model.parameters()).device

    def _slice(level):
        t = torch.as_tensor(np.asarray(level[i]), device=device).float()
        return t if t.ndim == 4 else t.unsqueeze(0)  # ensure (B, C, H, W)

    backbone_out = {
        "backbone_fpn": [_slice(level) for level in fpn_list],
        "vision_pos_enc": [_slice(level) for level in pos_enc_list],
    }
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])
    ][::-1]
    predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    predictor._orig_hw = [tuple(int(s) for s in np.array(original_size).reshape(-1)[:2])]
    predictor._is_image_set = True


def _set_image_predictor_from_3d_embeddings(predictor, image_embeddings, i):
    """Set a SAM2 image predictor's features for slice ``i`` from precomputed 3d (video-style) embeddings.

    The 3d embeddings produced for the volume (the same ones used for interactive 3d and the decoder)
    store the per-slice backbone FPN outputs (``fpn``) and positional encodings (``pos_enc``), so the
    per-slice AMG reuses them instead of re-encoding each slice.
    """
    _set_image_predictor_from_backbone(
        predictor, image_embeddings["fpn"], image_embeddings["pos_enc"], image_embeddings["original_size"], i,
    )


class _LazyRLEMask(dict):
    """A mask dict whose 'segmentation' is stored as an (uncompressed) RLE and decoded on access.

    The SAM2 mask generator can emit hundreds-thousands of masks; holding them all as full-resolution
    binary arrays is what makes AMG run out of memory (it gets OS-killed even for 2d). Storing them as
    compact RLE and decoding to a binary mask only when 'segmentation' is read (e.g. inside
    `mask_data_to_segmentation`'s loop) keeps the peak at a single full-resolution mask at a time.
    All other fields (area, bbox, ...) are plain dict entries and are read without decoding.
    """

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if key == "segmentation" and isinstance(value, dict):
            from sam2.utils.amg import rle_to_mask
            return rle_to_mask(value)
        return value


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
            default of '0.95' because the embeddings here come from micro-sam's percentile-normalized
            inputs, under which masks score marginally lower in stability.
        kwargs: Additional keyword arguments forwarded to `SAM2AutomaticMaskGenerator`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: Optional[str] = None,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 32,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.9,
        **kwargs,
    ) -> None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        # 'output_mode="uncompressed_rle"' stores each mask as a compact RLE instead of a full-
        # resolution binary array; we decode them lazily, one at a time, via '_LazyRLEMask' (see
        # '_generate_masks_for_shape'). Together with the lower 'points_per_batch' (which bounds the
        # number of masks upscaled to full resolution at once during prediction) this keeps AMG from
        # running out of memory on large images, where the old binary-mask storage got OS-killed.
        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            output_mode="uncompressed_rle",
            **kwargs,
        )
        # Use the shared resize-longest transform for AMG.
        from micro_sam.v2.util import configure_image_predictor
        configure_image_predictor(self._mask_generator.predictor)
        # The embedding signature written by 'precompute_image_embeddings' reads 'model_type' and
        # 'model_name' off the predictor. The video predictor gets these in 'get_sam2_model', but the
        # image predictor used here does not, so we set them (matching the GUI, see _state.py).
        predictor = self._mask_generator.predictor
        predictor.model_type = model_type or getattr(model, "model_type", None) or "hvit"
        predictor.model_name = model_type or getattr(model, "model_name", None) or predictor.model_type
        self._masks = None
        self._original_size = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the segmenter has already been initialized."""
        return self._is_initialized

    def _generate_masks_for_shape(self, shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Run the grid-based mask prediction reusing the embeddings already set on the predictor.

        The embeddings are expected to be set on `self._mask_generator.predictor` (via
        `precompute_image_embeddings` or `set_precomputed`). We temporarily neutralize the
        predictor's `set_image`, which the native mask generator calls once per crop, so that it
        reuses the precomputed embeddings instead of recomputing them. This is only valid for the
        single-crop case (`crop_n_layers=0`, the default). `shape` is the (Y, X) size the mask
        generator should assume (the full image, or a single tile).
        """
        predictor = self._mask_generator.predictor
        dummy = np.zeros((*shape, 3), dtype="uint8")
        original_set_image = predictor.set_image
        predictor.set_image = lambda *args, **kwargs: None
        try:
            masks = self._mask_generator.generate(dummy)
        finally:
            predictor.set_image = original_set_image
        # Wrap the RLE masks so 'segmentation' decodes to a binary mask only when read (one at a
        # time), instead of materialising every mask at full resolution. See '_LazyRLEMask'.
        return [_LazyRLEMask(mask) for mask in masks]

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
        elif "fpn" in image_embeddings and i is not None:
            # Reuse a slice of the precomputed 3d (video-style) embeddings: reconstruct the image
            # predictor's features for slice 'i' without re-running the encoder.
            _set_image_predictor_from_3d_embeddings(predictor, image_embeddings, i)
        else:
            set_precomputed(predictor, image_embeddings, i=i)

        self._original_size = tuple(int(s) for s in predictor._orig_hw[0])
        self._masks = self._generate_masks_for_shape(self._original_size)
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
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: bool = False,
        pbar_init: Optional[callable] = None,
        pbar_update: Optional[callable] = None,
        **kwargs,
    ) -> None:
        """Run the grid-based mask prediction tile-by-tile and store the per-tile masks.

        The tiled image embeddings are computed (and cached if `save_path` is given) via
        `precompute_image_embeddings`, or taken from `image_embeddings` if provided (e.g. by the
        GUI). The per-tile embeddings are then set on the predictor with `set_precomputed` and the
        grid prediction reuses them. When embeddings are provided the tiling is taken from them.

        Args:
            image: The input image, grayscale (Y, X) or RGB (Y, X, 3). Content unused when
                `image_embeddings` is given.
            tile_shape: The tile shape for the tiled prediction, (y, x). Taken from the embeddings
                when `image_embeddings` is given.
            halo: The overlap between the tiles, (y, x). Taken from the embeddings when given.
            image_embeddings: Optional precomputed tiled image embeddings.
            i: Index of the slice to segment if `image` has three spatial dimensions.
            save_path: Optional path to cache the computed embeddings in a zarr container.
            verbose: Verbosity flag. By default 'False'.
            kwargs: Additional arguments, ignored. Kept for interface compatibility.
        """
        predictor = self._mask_generator.predictor
        # Reuse a slice of precomputed 3d (video-style) tiled embeddings: reconstruct the image
        # predictor's features per tile for slice 'i' without re-running the encoder.
        if image_embeddings is not None and "fpn" in image_embeddings and i is not None:
            self._initialize_slice_from_3d_embeddings(image_embeddings, i)
            return

        if image_embeddings is None:
            if tile_shape is None or halo is None:
                raise ValueError("Both 'tile_shape' and 'halo' have to be passed for the tiled segmenter.")
            if image.ndim == 3 and image.shape[-1] != 3 and i is not None:
                image = image[i]
            image_embeddings = precompute_image_embeddings(
                predictor, image, save_path=save_path, ndim=2, tile_shape=tile_shape, halo=halo, verbose=verbose,
            )

        feats = image_embeddings["features"]
        tile_shape = tuple(int(s) for s in feats.attrs["tile_shape"])
        halo = tuple(int(s) for s in feats.attrs["halo"])
        self._original_size = tuple(int(s) for s in feats.attrs["shape"])
        self._tiling = Blocking([0, 0], list(self._original_size), list(tile_shape))
        self._halo = tuple(halo)

        from micro_sam.util import handle_pbar
        _, pbar_init, pbar_update, pbar_close = handle_pbar(verbose, pbar_init, pbar_update)

        self._masks = []
        n_tiles = self._tiling.number_of_blocks
        pbar_init(n_tiles, "Automatic segmentation (tiles)")
        for tile_id in range(n_tiles):
            block = self._tiling.get_block_with_halo(tile_id, list(self._halo)).outer_block
            set_precomputed(predictor, image_embeddings, tile_id=tile_id)
            tile_size = tuple(end - begin for begin, end in zip(block.begin, block.end))
            self._masks.append(self._generate_masks_for_shape(tile_size))
            pbar_update(1)
        pbar_close()

        self._is_initialized = True

    def _initialize_slice_from_3d_embeddings(self, image_embeddings, i):
        """Run the tiled AMG for slice ``i`` reusing precomputed 3d (video-style) tiled embeddings.

        Each tile's image features for slice ``i`` are reconstructed from the per-tile ``fpn``/``pos_enc``
        (no encoder pass), the grid prediction is run per tile, and the per-tile masks are stitched
        in-plane by `generate`. The tiling is taken from the embeddings; the z axis is not tiled.
        """
        predictor = self._mask_generator.predictor
        feats = image_embeddings["features"]
        full_shape = tuple(int(s) for s in feats.attrs["shape"])  # (Z, Y, X)
        tile_shape = tuple(int(s) for s in feats.attrs["tile_shape"])
        halo = tuple(int(s) for s in feats.attrs["halo"])
        self._original_size = full_shape[1:]  # in-plane (Y, X); z is not tiled
        self._tiling = Blocking([0, 0], list(self._original_size), list(tile_shape))
        self._halo = halo

        self._masks = []
        for tile_id in range(self._tiling.number_of_blocks):
            block = self._tiling.get_block_with_halo(tile_id, list(self._halo)).outer_block
            fpn_tile = _load_list_datasets(image_embeddings["fpn"], str(tile_id), lazy_loading=False)
            pos_tile = _load_list_datasets(image_embeddings["pos_enc"], str(tile_id), lazy_loading=False)
            original_size = feats[str(tile_id)].attrs["original_size"]
            _set_image_predictor_from_backbone(predictor, fpn_tile, pos_tile, original_size, i)
            tile_size = tuple(end - begin for begin, end in zip(block.begin, block.end))
            self._masks.append(self._generate_masks_for_shape(tile_size))

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
    image_embeddings: Optional[dict] = None,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
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
        image_embeddings: Optional precomputed 3d (video-style) embeddings for the volume. When given
            (and not tiled), each slice's AMG reuses the precomputed features instead of re-encoding.
        verbose: Verbosity flag. By default 'True'.
        pbar_init: Callback to initialize an external progress bar, called with the number of slices.
        pbar_update: Callback to update an external progress bar, called once per segmented slice.
        kwargs: Keyword arguments for the 'generate' method of the segmenter.

    Returns:
        The 3d instance segmentation, uint32 array of shape (Z, Y, X).
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3d volume of shape (Z, Y, X), got shape {volume.shape}.")

    init_kwargs = {}
    if tile_shape is not None and halo is not None:
        init_kwargs = {"tile_shape": tile_shape, "halo": halo}
    # Reuse the precomputed 3d embeddings per slice (no re-encode) for both the tiled and non-tiled
    # paths; the segmenter reconstructs each slice's features from them.
    reuse_embeddings = image_embeddings is not None

    from micro_sam.util import handle_pbar
    _, pbar_init, pbar_update, pbar_close = handle_pbar(verbose, pbar_init, pbar_update)
    pbar_init(volume.shape[0], "Automatic segmentation (slices)")

    segmentation = np.zeros(volume.shape, dtype="uint32")
    offset = 0
    for i in range(volume.shape[0]):
        if reuse_embeddings:
            segmenter.initialize(volume[i], image_embeddings=image_embeddings, i=i, verbose=False, **init_kwargs)
        else:
            segmenter.initialize(volume[i], verbose=False, **init_kwargs)
        seg = segmenter.generate(**kwargs)

        # Offset the per-slice instance ids so that they are unique across the whole volume.
        max_z = int(seg.max())
        if max_z != 0:
            seg[seg != 0] += offset
            offset += max_z
            segmentation[i] = seg
        pbar_update(1)
    pbar_close()

    segmentation = merge_instance_segmentation_3d(
        segmentation,
        beta=0.5,
        with_background=with_background,
        gap_closing=gap_closing,
        min_z_extent=min_z_extent,
        verbose=verbose,
    )
    return segmentation
