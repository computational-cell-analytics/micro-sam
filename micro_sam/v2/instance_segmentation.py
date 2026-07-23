"""Automatic instance segmentation backends for the SAM2 model.

This module holds all the backend engines for automatic segmentation with SAM2, mirroring the
micro-sam v1 `instance_segmentation` module (which holds AMG / AIS / APG). Two engines are provided:

- AMG (`AutomaticMaskGenerationSegmenter`, `TiledAutomaticMaskGenerationSegmenter`): grid-based
  automatic mask generation, no decoder required. The expensive grid prediction happens in
  `initialize`, the cheap conversion to an instance segmentation in `generate`. Supported for 2d
  images and, via `amg_3d_segmentation`, for 3d volumes (run slice-by-slice and stitched
  across z with `micro_sam.v1.multi_dimensional_segmentation.merge_instance_segmentation_3d`).

- AIS (`UniSAM2InstanceSegmentation`, `TiledUniSAM2InstanceSegmentation`): decoder-based instance
  segmentation with the UniSAM2 model (a UNETR decoder on top of the SAM2 image encoder, see
  `micro_sam.v2.models.util.UniSAM2`). The decoder predicts a foreground probability map and three
  directed distance channels, which `generate` converts into an instance segmentation via
  `micro_sam.v2.postprocessing` ('sparse' -> flow following for LM data, 'dense' -> multicut for EM
  data). All UniSAM2 inference is encapsulated in these classes.

Both engines share the `initialize` / `generate` / `get_state` / `set_state` interface of the v1
`AutoSegBase`, support in-plane (xy) tiling with a halo, and are selected via
`get_instance_segmentation_generator`.
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from bioimage_cpp.utils import Blocking

from micro_sam.util import mask_data_to_segmentation
from micro_sam.v1.inference import _merge_segmentations
from micro_sam.v1.instance_segmentation import AutoSegBase
from micro_sam.v1.multi_dimensional_segmentation import merge_instance_segmentation_3d
from micro_sam.v2.postprocessing import flow_instance_segmentation, run_multicut
from micro_sam.v2.util import (
    precompute_image_embeddings, set_precomputed, _load_list_datasets,
    _DEFAULT_MODEL, DEFAULT_TILE_Z, DEFAULT_HALO_Z, Devices,
)

DEFAULT_SEGMENTATION_MODE_WITH_DECODER = "ais"


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


class AutomaticMaskGenerationSegmenter(AutoSegBase):
    """Generates an instance segmentation for the SAM2 model using grid-based prompting (AMG).

    Wraps the native `sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator` and exposes the
    same `initialize` / `generate` interface as the micro-sam v1 `AutomaticMaskGenerator`, so it
    can be used both for single 2d images and, via `amg_3d_segmentation`, for 3d volumes.

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
        # The parameters that are baked into the predicted masks during 'initialize'. They are stored
        # in the cached state so a reused state can be validated against the requested parameters.
        self._amg_params = {
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "model_type": predictor.model_type,
        }
        self._masks = None
        self._original_size = None
        self._is_initialized = False

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

    def get_state(self) -> Dict[str, Any]:
        """Return the cached mask-generation state so it can be serialized and later restored.

        The state holds the predicted masks (as compact RLE dicts), the image size and the
        parameters the masks were generated with (used to validate a reused state). Restore it
        with `set_state` to skip the expensive grid prediction in `initialize`.
        """
        if not self._is_initialized:
            raise RuntimeError("Cannot get the state before the segmenter has been initialized.")
        return {
            "masks": [dict(mask) for mask in self._masks],
            "original_size": self._original_size,
            "params": dict(self._amg_params),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the state produced by `get_state`, marking the segmenter initialized.

        The masks are re-wrapped in `_LazyRLEMask` so `generate` decodes them lazily as before.
        """
        self._masks = [_LazyRLEMask(mask) for mask in state["masks"]]
        self._original_size = tuple(int(s) for s in state["original_size"])
        self._is_initialized = True

    def clear_state(self) -> None:
        """Clear the cached masks."""
        self._masks = None
        self._original_size = None
        self._is_initialized = False


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
                predictor, image, save_path=save_path, ndim=2, tile_shape=tile_shape, halo=halo,
                verbose=verbose, lazy_loading=True,
            )

        feats = image_embeddings["features"]
        tile_shape = tuple(int(s) for s in feats.attrs["tile_shape"])
        halo = tuple(int(s) for s in feats.attrs["halo"])
        self._original_size = tuple(int(s) for s in feats.attrs["shape"])
        self._tile_shape = tile_shape
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
        self._tile_shape = tile_shape
        self._tiling = Blocking([0, 0], list(self._original_size), list(tile_shape))
        self._halo = halo

        self._masks = []
        for tile_id in range(self._tiling.number_of_blocks):
            block = self._tiling.get_block_with_halo(tile_id, list(self._halo)).outer_block
            # Keep the per-tile datasets lazy so '_set_image_predictor_from_backbone' reads only slice
            # 'i' from disk, instead of pulling the tile's whole z-stack into RAM on every slice.
            fpn_tile = _load_list_datasets(image_embeddings["fpn"], str(tile_id), lazy_loading=True)
            pos_tile = _load_list_datasets(image_embeddings["pos_enc"], str(tile_id), lazy_loading=True)
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

    def get_state(self) -> Dict[str, Any]:
        """Return the cached per-tile mask state, plus the tiling needed to restore it."""
        if not self._is_initialized:
            raise RuntimeError("Cannot get the state before the segmenter has been initialized.")
        return {
            "masks": [[dict(mask) for mask in tile_masks] for tile_masks in self._masks],
            "original_size": self._original_size,
            "tile_shape": self._tile_shape,
            "halo": self._halo,
            "params": dict(self._amg_params),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the per-tile masks and rebuild the tiling from `get_state`."""
        self._masks = [[_LazyRLEMask(mask) for mask in tile_masks] for tile_masks in state["masks"]]
        self._original_size = tuple(int(s) for s in state["original_size"])
        self._tile_shape = tuple(int(s) for s in state["tile_shape"])
        self._halo = tuple(int(s) for s in state["halo"])
        self._tiling = Blocking([0, 0], list(self._original_size), list(self._tile_shape))
        self._is_initialized = True


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


def amg_3d_segmentation(
    volume: np.ndarray,
    segmenter: AutomaticMaskGenerationSegmenter,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    image_embeddings: Optional[dict] = None,
    state_save_path: Optional[str] = None,
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
        state_save_path: Optional path to the embedding Zarr in which to cache the per-slice
            grid-prediction state. When set, a slice reuses its cached state instead of re-running
            the grid prediction; freshly computed slices are written back.
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

    def init_slice(i):
        if reuse_embeddings:
            segmenter.initialize(volume[i], image_embeddings=image_embeddings, i=i, verbose=False, **init_kwargs)
        else:
            segmenter.initialize(volume[i], verbose=False, **init_kwargs)

    if state_save_path is not None:
        from micro_sam.precompute_state import _cache_amg_slice, _embedding_signature
        state_signature = _embedding_signature(state_save_path)

    segmentation = np.zeros(volume.shape, dtype="uint32")
    offset = 0
    for i in range(volume.shape[0]):
        if state_save_path is not None:
            _cache_amg_slice(segmenter, state_save_path, i, init_slice, embedding_signature=state_signature)
        else:
            init_slice(i)
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


#
# UniSAM2 decoder-based instance segmentation (AIS)
#


def get_unisam2_model(checkpoint_path, device=None, encoder=_DEFAULT_MODEL, output_channels=4):
    """Load a UniSAM2 model for automatic segmentation from a checkpoint.

    Args:
        checkpoint_path: Path to the UniSAM2 checkpoint.
        device: The device to load the model onto.
        encoder: The SAM2 encoder to build the decoder on. Either the backbone name to build from
            scratch, e.g. 'hvit_t', or a prebuilt SAM2 image-encoder module to reuse (which avoids
            rebuilding / downloading the base backbone). Its weights are (re)defined by the checkpoint.
        output_channels: The number of output channels (foreground + directed distances).

    Returns:
        The UniSAM2 model in eval mode.
    """
    from micro_sam.v2.models.util import UniSAM2

    state = torch.load(checkpoint_path, weights_only=False, map_location=device or "cpu")
    # The standalone trainer saves the full model under 'model_state'; the joint trainer saves it
    # under 'unetr_state' or 'decoder_state'. We also accept a raw state dict.
    if isinstance(state, dict):
        model_state = state.get("model_state", state.get("unetr_state", state.get("decoder_state", state)))
    else:
        model_state = state

    # The standalone UniSAM2 trainer builds the model with a string encoder, so the SAM2 image
    # encoder lives directly under 'encoder.*'. The joint trainer instead passes the SAM2 image
    # encoder module, which gets wrapped in 'SAM2EncoderAdapter' and so lives under 'encoder.inner.*'.
    # Detect the latter and rebuild the matching structure by passing a SAM2 image encoder module.
    needs_adapter = isinstance(encoder, str) and any(k.startswith("encoder.inner.") for k in model_state)
    if needs_adapter:
        from micro_sam.v2.util import get_sam2_model
        sam2_model = get_sam2_model(model_type=encoder, input_type="images", device=device or "cpu")
        encoder = sam2_model.image_encoder

    model = UniSAM2(encoder=encoder, output_channels=output_channels)
    model.load_state_dict(model_state)

    if device is not None:
        model.to(device)
    model.eval()
    return model


def get_decoder(model_type, checkpoint=None, device=None, encoder=None):
    """Resolve and load the UniSAM2 decoder for a SAM2 model.

    The decoder is provided either by an explicit `checkpoint` or by a finetuned model with a
    registered decoder (e.g. 'hvit_t_cells'). Mirrors the micro-sam v1 `get_decoder`.

    Args:
        model_type: The SAM2 model. A finetuned model with a registered decoder, or a base backbone
            combined with `checkpoint`.
        checkpoint: Optional path to a decoder checkpoint to build the UniSAM2 decoder from.
        device: The device to load the decoder onto.
        encoder: Optional prebuilt SAM2 image-encoder module to build the decoder on, reused instead
            of rebuilding the base backbone (its weights are redefined by the checkpoint's strict
            load). By default the base backbone name (first 6 chars of `model_type`) is used.

    Returns:
        The UniSAM2 decoder model.
    """
    from micro_sam.v2.util import FINETUNED_MODELS, has_registered_decoder, _download_finetuned_sam2_model

    # Reuse a prebuilt encoder module if given, else build the decoder on the base backbone name
    # (first 6 characters, e.g. 'hvit_t_cells' -> 'hvit_t').
    if encoder is None:
        encoder = model_type[:6]
    if checkpoint is not None:
        decoder_source = checkpoint
    elif model_type in FINETUNED_MODELS and has_registered_decoder(model_type):
        _, _, decoder_source = _download_finetuned_sam2_model(model_type)
    else:
        raise ValueError(
            f"Automatic segmentation with SAM2 requires a finetuned model with a registered decoder "
            f"or a decoder checkpoint. '{model_type}' provides neither."
        )
    return get_unisam2_model(decoder_source, device=device, encoder=encoder)


def _resize_spatial(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """Resize the trailing (Y, X) of a (B, C, Z, Y, X) tensor to `size`, leaving Z unchanged."""
    b, c, z, y, x_dim = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * z, c, y, x_dim)
    x = torch.nn.functional.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)
    return x.reshape(b, z, c, size[0], size[1]).permute(0, 2, 1, 3, 4)


class ResizeLongestSideWrapper(torch.nn.Module):
    """Run UniSAM2 with resize-longest and bottom/right padding."""

    def __init__(self, model: torch.nn.Module, img_size: int) -> None:
        super().__init__()
        self.model = model
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from micro_sam.v2.transforms.resize import resize_longest_side_and_pad_tensor

        spatial = x.shape[-2:]
        x, resized = resize_longest_side_and_pad_tensor(x, self.img_size)
        out = self.model(x)
        out = out[..., :resized[0], :resized[1]]
        return _resize_spatial(out, spatial)


@contextlib.contextmanager
def _bridge_halo_progress(pbar_update):
    """Bridge `predict_with_halo`'s internal per-block tqdm to an external `pbar_update` callback.

    `predict_with_halo` wraps its thread-pool map in a tqdm (a napari progress bar inside napari)
    that micro-sam cannot otherwise drive, so 3d/tiled auto-segmentation showed no real progress.
    This temporarily swaps that module-level tqdm for a thin iterator that fires `pbar_update` once
    per completed block. It is a no-op fallback if `pbar_update` is None.
    """
    if pbar_update is None:
        yield
        return

    import torch_em.util.prediction as prediction_module
    original_tqdm = prediction_module.tqdm

    class _ProgressBridge:
        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            for item in self._iterable:
                yield item
                pbar_update(1)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, n=1):
            pbar_update(n)

        def set_description(self, *args, **kwargs):
            pass

        def close(self):
            pass

    prediction_module.tqdm = _ProgressBridge
    try:
        yield
    finally:
        prediction_module.tqdm = original_tqdm


def _n_blocks(spatial_shape, ndim, block_shape):
    """Number of blocks `predict_with_halo` iterates over, for a determinate progress total."""
    blocked = spatial_shape if ndim == 3 else (1, *spatial_shape)
    return int(np.prod([int(np.ceil(s / b)) for s, b in zip(blocked, block_shape)]))


def _block_shape_and_halo(spatial_shape, ndim, tile_shape, halo):
    """Compute the (z, y, x) block shape and halo for `predict_with_halo`.

    For 3d data a volume is always chunked along z - using the explicit z tile when tiling is on,
    or the default z block when in-plane tiling is off (the whole in-plane plane per chunk). The
    model is trained on small z crops, so running every slice at once is both out-of-distribution
    and a memory blow-up (the cause of the 3d 'killed' reports). 2d data is a single (1, y, x) block.

    Args:
        spatial_shape: The spatial image shape, (Y, X) for 2d or (Z, Y, X) for 3d.
        ndim: The number of spatial dimensions (2 or 3).
        tile_shape: The tile shape, or None for no tiling. For 3d data either an in-plane (y, x)
            tile, which keeps the default z chunking, or an explicit (z, y, x) tile.
        halo: The tile halo, or None for no overlap.

    Returns:
        The (block_shape, block_halo) tuples in (z, y, x) order for `predict_with_halo`.
    """
    is_3d = ndim == 3
    if tile_shape is None and is_3d:
        n_slices = spatial_shape[0]
        z_block = min(DEFAULT_TILE_Z, n_slices)
        block_shape = (z_block, spatial_shape[1], spatial_shape[2])
        block_halo = (DEFAULT_HALO_Z if z_block < n_slices else 0, 0, 0)
    elif tile_shape is None:
        block_shape = (1, *spatial_shape)
        block_halo = (0, 0, 0)
    elif is_3d:
        # Tiling is in-plane, so the CLI and the annotator pass a 2-entry (y, x) tile. Prepend the
        # default z block, keeping z chunked exactly as it is without tiling. A 3-entry (z, y, x)
        # tile is used as given.
        if len(tile_shape) == 2:
            n_slices = spatial_shape[0]
            z_block = min(DEFAULT_TILE_Z, n_slices)
            block_shape = (z_block, *tile_shape)
            z_halo = DEFAULT_HALO_Z if z_block < n_slices else 0
            block_halo = (z_halo, *((0, 0) if halo is None else tuple(halo)[-2:]))
        else:
            block_shape = tuple(tile_shape)  # (z, y, x)
            block_halo = (0, 0, 0) if halo is None else tuple(halo)
    else:
        block_shape = (1, *tile_shape)  # (1, y, x)
        block_halo = (0, *((0, 0) if halo is None else halo))
    return block_shape, block_halo


class _StubEncoder(torch.nn.Module):
    """Encoder replacement that returns precomputed per-slice features in call order.

    For a single feature volume, `features` has shape (Z, C, H, W). Batched decoder inference uses
    (B, Z, C, H, W), in which case each encoder call returns the corresponding slice for all batch
    elements. This reproduces the encoder outputs expected by `UNETR3D.forward` without re-encoding.
    """

    def __init__(self, features: torch.Tensor, img_size: int = 1024) -> None:
        super().__init__()
        self.features = features
        self.img_size = img_size
        self._idx = 0

    def forward(self, x):  # noqa
        if self.features.ndim == 5:
            feature = self.features[:, self._idx]
        else:
            feature = self.features[self._idx:self._idx + 1]
        self._idx += 1
        return [feature]


def _get_decoder_autocast(device):
    """Use FP16 decoder autocast on supported accelerator backends."""
    device_type = torch.device(device).type
    if device_type in ("cuda", "mps"):
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    return contextlib.nullcontext()


def _decode_3d_feature_batch(model, features, original_size, device):
    """Decode precomputed feature volumes with shape (B, Z, C, H, W).

    CUDA and MPS decoder inference use FP16 autocast. This matches UniSAM2 training on CUDA and keeps
    the largest default annotator tile within a 10 GB MIG partition. Cached predictions remain float32.
    """
    if features.ndim != 5:
        raise ValueError(f"Expected batched features with shape (B, Z, C, H, W), got {features.shape}.")
    device = features.device if device is None else torch.device(device)
    img_size = getattr(model.encoder, "img_size", 1024)
    real_encoder = model.encoder
    model.encoder = _StubEncoder(features, img_size)
    try:
        dummy = torch.zeros((features.shape[0], 3, features.shape[1], *original_size), device=device)
        with _get_decoder_autocast(device):
            output = model(dummy)
    finally:
        model.encoder = real_encoder
    # Move before casting: an fp32 copy of the whole output on the accelerator would cancel out the
    # memory that fp16 inference saves (and can trigger the OOM backoff).
    return output.detach().cpu().float().numpy()


def _decode_3d_feature_block(model, feature, original_size, device):
    """Decode one precomputed feature volume with shape (Z, C, H, W)."""
    return _decode_3d_feature_batch(model, feature.unsqueeze(0), original_size, device)[0]


def _segment_from_predictions(prediction: np.ndarray, mode: str = "sparse", **kwargs) -> np.ndarray:
    """Convert UniSAM2 predictions into an instance segmentation.

    Args:
        prediction: The UniSAM2 predictions, shape (4, *spatial). Channel 0 is the foreground
            probability and channels 1-3 are the directed distances.
        mode: The segmentation mode. 'sparse' uses flow-based segmentation (LM data, 2d and 3d),
            'dense' uses multicut-based segmentation (EM data, 2d and 3d).
        kwargs: Additional parameters forwarded to the postprocessing function
            (`flow_instance_segmentation` for 'sparse', `run_multicut` for 'dense').

    Returns:
        The instance segmentation, uint32 array with the spatial shape of the prediction.
    """
    foreground = prediction[0]
    if mode == "dense":
        boundary_map = foreground.max() - foreground
        denom = boundary_map.max()
        if denom > 0:
            boundary_map = boundary_map / denom
        # Multicut uses the in-plane (y, x) distance channels.
        distances = np.stack([prediction[2], prediction[3]])
        # run_multicut expects volumetric inputs. For 2d data we run it on a single-slice volume
        # (no z-edges), which yields a 2d multicut, and squeeze the result back to 2d.
        if boundary_map.ndim == 2:
            seg = run_multicut(boundary_map[None], distances[:, None], **kwargs)[0]
        else:
            seg = run_multicut(boundary_map, distances, **kwargs)
    else:
        seg = flow_instance_segmentation(foreground, prediction[1:], **kwargs)
    return seg.astype("uint32")


class UniSAM2InstanceSegmentation(AutoSegBase):
    """Generates an instance segmentation with the UniSAM2 decoder (AIS).

    A concrete `AutoSegBase` (like `AutomaticMaskGenerationSegmenter`), but predicts the segmentation
    with the UniSAM2 decoder instead of grid prompts. All UniSAM2 inference is encapsulated here (the
    only way to run it). Use it as follows:
    ```python
    segmenter = UniSAM2InstanceSegmentation(model)
    segmenter.initialize(image, ndim=2)  # Run the UniSAM2 inference.
    masks = segmenter.generate(mode="sparse", foreground_threshold=0.6)  # Post-process.
    ```

    Args:
        model: The UniSAM2 model (see `get_unisam2_model` / `get_decoder`).
        device: The device to run inference on.
    """

    def __init__(self, model: torch.nn.Module, device: Optional[Union[str, torch.device]] = None) -> None:
        self._model = model
        self._device = device
        self._prediction = None
        self._is_initialized = False

    def _inference_devices(self, devices: Devices) -> Devices:
        """Fall back to the configured device, so it is not overridden by the multi-GPU default."""
        return self._device if devices is None else devices

    def _run_full_inference(
        self, raw, ndim, tile_shape=None, halo=None, pbar_init=None, pbar_update=None,
        batch_size=None, devices: Devices = None, num_prefetch_workers=4, num_write_workers=1,
    ):
        """Run queued, batched UniSAM2 encoder and decoder inference.

        Tiled reads and preprocessing, GPU inference, and output writes overlap through the torch-em
        prediction pipeline. If `batch_size` is None, candidate sizes are benchmarked and a
        throughput-efficient batch is selected independently on every CUDA device.
        """
        from torch_em.util.prediction import predict_with_halo_pipelined
        from micro_sam.v2.normalization import normalize_raw
        from micro_sam.v2.batched_inference import (
            _compute_auto_batch_sizes, _prepare_models, _release_model_replicas, _resolve_devices,
        )

        def _preprocess(crop):
            return np.concatenate([normalize_raw(crop, axis=(-2, -1))] * 3, axis=0)

        def _predict(this_model, inputs):
            with _get_decoder_autocast(inputs.device):
                return this_model(inputs)

        def _predict_probe(this_model, inputs):
            return _predict(this_model, inputs.clamp(0.0, 1.0))

        is_3d = ndim == 3
        block_shape, block_halo = _block_shape_and_halo(tuple(raw.shape), ndim, tile_shape, halo)
        n_blocks = _n_blocks(tuple(raw.shape), ndim, block_shape)
        if pbar_init is not None:
            desc = "Automatic segmentation (volume)" if is_3d else "Automatic segmentation"
            pbar_init(n_blocks, desc)

        if is_3d:
            input_ = raw[np.newaxis].astype("float32")
            output = np.zeros((4, *raw.shape), dtype="float32")
        else:
            input_ = raw[np.newaxis, np.newaxis].astype("float32")
            output = np.zeros((4, 1, *raw.shape), dtype="float32")

        img_size = getattr(getattr(self._model, "encoder", None), "img_size", 1024)
        resize_model = ResizeLongestSideWrapper(self._model, img_size)
        resolved_devices = _resolve_devices(resize_model, self._inference_devices(devices))

        if batch_size is None:
            model_devices = _prepare_models(resize_model, resolved_devices)
            patch_shape = tuple(block + 2 * overlap for block, overlap in zip(block_shape, block_halo))
            try:
                batch_sizes = _compute_auto_batch_sizes(
                    model_devices=model_devices,
                    n_jobs=n_blocks,
                    patch_shape=patch_shape,
                    in_channels=3,
                    # The probe's synthetic input bypasses `_preprocess`; clamp it into the [0, 1]
                    # range the model asserts (values are irrelevant to the memory measurement).
                    prediction_function=_predict_probe,
                )
                batch_size = min(batch_sizes)
            finally:
                _release_model_replicas(model_devices)
        elif int(batch_size) < 1:
            raise ValueError(f"batch_size must be positive or None, got {batch_size}.")

        with _bridge_halo_progress(pbar_update):
            output = predict_with_halo_pipelined(
                input_=input_,
                model=resize_model,
                block_shape=block_shape,
                halo=block_halo,
                preprocess=_preprocess,
                prediction_function=_predict,
                gpu_ids=resolved_devices,
                output=output,
                with_channels=True,
                batch_size=int(batch_size),
                num_prefetch_workers=num_prefetch_workers,
                num_write_workers=num_write_workers,
                disable_tqdm=pbar_update is not None,
            )
        if not is_3d:
            output = output[:, 0]
        return output

    @torch.no_grad()
    def _run_decoder_2d(self, image_embeddings):
        """Run only the UniSAM2 decoder on precomputed 2d image embeddings (no encoder pass).

        Reuses resize-longest 2d embeddings produced by `precompute_image_embeddings`. Returns the
        predictions stacked along the channel axis, shape (4, Y, X).
        """
        features = np.asarray(image_embeddings["features"])
        # A single slice taken from save-path 3d embeddings keeps a singleton batch axis, i.e.
        # (1, 1, C, h, w); squeeze it back to the (1, C, h, w) a 2d embedding has (the in-memory layout).
        if features.ndim == 5 and features.shape[1] == 1:
            features = features[:, 0]
        if features.ndim != 4:
            raise ValueError(
                f"Decoder-from-embeddings requires 2d image embeddings (features with ndim 4), got {features.ndim}."
            )
        feature = torch.as_tensor(features, device=self._device).float()
        original_size = tuple(int(s) for s in np.array(image_embeddings["original_size"]).reshape(-1)[:2])

        output = _decode_3d_feature_batch(self._model, feature.unsqueeze(1), original_size, self._device)
        return output[0, :, 0]

    @torch.no_grad()
    def _run_decoder_3d(
        self, image_embeddings, z_block=None, z_halo=None, pbar_init=None, pbar_update=None,
        batch_size=None, devices: Devices = None, num_prefetch_workers=4, num_write_workers=1,
    ):
        """Decode queued z blocks from precomputed 3d embeddings."""
        from micro_sam.v2.batched_inference import _decode_volume_embeddings

        return _decode_volume_embeddings(
            model=self._model,
            image_embeddings=image_embeddings,
            z_block=z_block,
            z_halo=z_halo,
            pbar_init=pbar_init,
            pbar_update=pbar_update,
            batch_size=batch_size,
            devices=self._inference_devices(devices),
            num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
        )

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        ndim: int,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        tile_shape: Optional[tuple] = None,
        halo: Optional[tuple] = None,
        pbar_init: Optional[callable] = None,
        pbar_update: Optional[callable] = None,
        z_block: Optional[int] = None,
        z_halo: Optional[int] = None,
        batch_size: Optional[int] = 1,
        devices: Devices = None,
        num_prefetch_workers: int = 4,
        num_write_workers: int = 1,
    ) -> None:
        """Run the UniSAM2 inference and store foreground and distance predictions.

        Args:
            image: The input image, shape (Y, X) for 2d or (Z, Y, X) for 3d.
            ndim: The number of spatial dimensions (2 or 3).
            image_embeddings: Optional precomputed image embeddings. If given, only the decoder runs.
            i: Index for the image data. Unused here, kept for interface compatibility.
            tile_shape: Unused for the non-tiled segmenter.
            halo: Unused for the non-tiled segmenter.
            pbar_init: Callback to initialize an external progress bar.
            pbar_update: Callback to update an external progress bar.
            z_block: Number of slices per decoder z block.
            z_halo: Overlapping decoder slices used as context.
            batch_size: The batch size used when running inference for multiple z-blocks and / or tiles.
                Defaults to one; pass None for throughput-based automatic selection.
            devices: Inference device or devices. None uses all visible GPUs when the model is on CUDA.
            num_prefetch_workers: Number of input reading and preprocessing threads.
            num_write_workers: Number of output writing threads.
        """
        if image_embeddings is not None and ndim == 3:
            self._prediction = self._run_decoder_3d(
                image_embeddings,
                z_block=z_block,
                z_halo=z_halo,
                pbar_init=pbar_init,
                pbar_update=pbar_update,
                batch_size=batch_size,
                devices=devices,
                num_prefetch_workers=num_prefetch_workers,
                num_write_workers=num_write_workers,
            )
        elif image_embeddings is not None:
            if pbar_init is not None:
                pbar_init(1, "Automatic segmentation")
            self._prediction = self._run_decoder_2d(image_embeddings)
            if pbar_update is not None:
                pbar_update(1)
        else:
            self._prediction = self._run_full_inference(
                image,
                ndim,
                pbar_init=pbar_init,
                pbar_update=pbar_update,
                batch_size=batch_size,
                devices=devices,
                num_prefetch_workers=num_prefetch_workers,
                num_write_workers=num_write_workers,
            )
        self._is_initialized = True

    def generate(self, mode: str = "sparse", **kwargs) -> np.ndarray:
        """Convert the stored predictions into an instance segmentation.

        Args:
            mode: The segmentation mode, 'sparse' (flow) or 'dense' (multicut).
            kwargs: Additional parameters forwarded to the postprocessing.

        Returns:
            The instance segmentation, uint32 array.
        """
        if not self._is_initialized:
            raise RuntimeError("The segmenter has not been initialized. Call 'initialize' first.")
        return _segment_from_predictions(self._prediction, mode=mode, **kwargs)

    def get_state(self) -> dict:
        """Return the cached decoder predictions so they can be serialized and later restored.

        The state holds the (4, *spatial) foreground + directed-distance predictions. Restore it
        with `set_state` to skip the expensive decoder inference in `initialize`. It is independent
        of the post-processing parameters (those are applied in `generate`), so it is always reusable.
        """
        if not self._is_initialized:
            raise RuntimeError("Cannot get the state before the segmenter has been initialized.")
        return {"prediction": self._prediction}

    def set_state(self, state: dict) -> None:
        """Restore the state produced by `get_state`, marking the segmenter initialized."""
        self._prediction = np.asarray(state["prediction"])
        self._is_initialized = True

    def clear_state(self) -> None:
        """Clear the cached decoder predictions."""
        self._prediction = None
        self._is_initialized = False


class TiledUniSAM2InstanceSegmentation(UniSAM2InstanceSegmentation):
    """Generates a tiled instance segmentation with the UniSAM2 decoder (AIS).

    Like `UniSAM2InstanceSegmentation`, but the model inference is tiled in-plane (xy) with a halo.

    Args:
        model: The UniSAM2 model (see `get_unisam2_model` / `get_decoder`).
        device: The device to run inference on.
    """

    @torch.no_grad()
    def _run_decoder_tiled_2d(
        self, image_embeddings, pbar_init=None, pbar_update=None,
        batch_size=None, devices: Devices = None, num_prefetch_workers=4, num_write_workers=1,
    ):
        """Decode and stitch queued 2d embedding tiles."""
        from micro_sam.v2.batched_inference import _decode_tiled_2d_embeddings

        return _decode_tiled_2d_embeddings(
            model=self._model,
            image_embeddings=image_embeddings,
            pbar_init=pbar_init,
            pbar_update=pbar_update,
            batch_size=batch_size,
            devices=self._inference_devices(devices),
            num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
        )

    @torch.no_grad()
    def _run_decoder_tiled_3d(
        self, image_embeddings, pbar_init=None, pbar_update=None, z_block=None, z_halo=None,
        batch_size=None, devices: Devices = None, num_prefetch_workers=4, num_write_workers=1,
    ):
        """Decode batches spanning tile columns and z blocks, then stitch them."""
        from micro_sam.v2.batched_inference import _decode_tiled_3d_embeddings

        return _decode_tiled_3d_embeddings(
            model=self._model,
            image_embeddings=image_embeddings,
            pbar_init=pbar_init,
            pbar_update=pbar_update,
            z_block=z_block,
            z_halo=z_halo,
            batch_size=batch_size,
            devices=self._inference_devices(devices),
            num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
        )

    @torch.no_grad()
    def _run_decoder_tiled_3d_slice(
        self, image_embeddings, i, pbar_init=None, pbar_update=None,
        batch_size=None, devices: Devices = None, num_prefetch_workers=4, num_write_workers=1,
    ):
        """Decode one volume slice across all embedding tiles in batches."""
        from micro_sam.v2.batched_inference import _decode_tiled_3d_slice

        return _decode_tiled_3d_slice(
            model=self._model,
            image_embeddings=image_embeddings,
            index=i,
            pbar_init=pbar_init,
            pbar_update=pbar_update,
            batch_size=batch_size,
            devices=self._inference_devices(devices),
            num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
        )

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        ndim: int,
        tile_shape: Optional[tuple] = None,
        halo: Optional[tuple] = None,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        pbar_init: Optional[callable] = None,
        pbar_update: Optional[callable] = None,
        z_block: Optional[int] = None,
        z_halo: Optional[int] = None,
        batch_size: Optional[int] = 1,
        devices: Devices = None,
        num_prefetch_workers: int = 4,
        num_write_workers: int = 1,
    ) -> None:
        """Run tiled UniSAM2 inference and store foreground and distance predictions.

        `batch_size=None` benchmarks candidate sizes and selects a throughput-efficient batch
        independently on each selected CUDA device.
        Reads and preprocessing are queued through `num_prefetch_workers`, output writes through
        `num_write_workers`.
        """
        decoder_kwargs = {
            "pbar_init": pbar_init,
            "pbar_update": pbar_update,
            "batch_size": batch_size,
            "devices": devices,
            "num_prefetch_workers": num_prefetch_workers,
            "num_write_workers": num_write_workers,
        }
        if image_embeddings is not None and ndim == 2 and "fpn" in image_embeddings and i is not None:
            self._prediction = self._run_decoder_tiled_3d_slice(
                image_embeddings, i, **decoder_kwargs,
            )
        elif image_embeddings is not None and ndim == 3:
            self._prediction = self._run_decoder_tiled_3d(
                image_embeddings, z_block=z_block, z_halo=z_halo, **decoder_kwargs,
            )
        elif image_embeddings is not None and ndim == 2:
            self._prediction = self._run_decoder_tiled_2d(image_embeddings, **decoder_kwargs)
        else:
            self._prediction = self._run_full_inference(
                image,
                ndim,
                tile_shape=tile_shape,
                halo=halo,
                pbar_init=pbar_init,
                pbar_update=pbar_update,
                batch_size=batch_size,
                devices=devices,
                num_prefetch_workers=num_prefetch_workers,
                num_write_workers=num_write_workers,
            )
        self._is_initialized = True


def get_unisam2_segmentation_generator(
    model: torch.nn.Module,
    is_tiled: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> UniSAM2InstanceSegmentation:
    """Get the UniSAM2 decoder-based (AIS) instance segmentation generator.

    Args:
        model: The UniSAM2 model.
        is_tiled: Whether to use tiled inference.
        device: The device to run inference on.

    Returns:
        The instance segmentation generator, either `TiledUniSAM2InstanceSegmentation` (if tiled)
        or `UniSAM2InstanceSegmentation`.
    """
    if is_tiled:
        return TiledUniSAM2InstanceSegmentation(model, device=device)
    return UniSAM2InstanceSegmentation(model, device=device)


def get_instance_segmentation_generator(
    model: Optional[torch.nn.Module] = None,
    decoder: Optional[torch.nn.Module] = None,
    is_tiled: bool = False,
    segmentation_mode: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> AutoSegBase:
    """Get the automatic instance segmentation generator (AMG or AIS), mirroring the v1 factory.

    Args:
        model: The SAM2 model, required for the 'amg' mode.
        decoder: The UniSAM2 decoder model, required for the 'ais' mode.
        is_tiled: Whether to use the tiled segmenter.
        segmentation_mode: One of 'amg' or 'ais'. By default 'ais' is used if a decoder is given,
            otherwise 'amg'.
        device: The device to run inference on ('ais' only).
        kwargs: Additional keyword arguments for the AMG segmenter.

    Returns:
        The segmentation generator instance.
    """
    if segmentation_mode is None:
        segmentation_mode = DEFAULT_SEGMENTATION_MODE_WITH_DECODER if decoder is not None else "amg"

    if segmentation_mode.lower() == "amg":
        if model is None:
            raise ValueError("The 'amg' segmentation mode requires a SAM2 'model'.")
        return get_amg_segmenter(model, is_tiled=is_tiled, **kwargs)
    elif segmentation_mode.lower() == "ais":
        if decoder is None:
            raise ValueError("The 'ais' segmentation mode requires a UniSAM2 'decoder'.")
        return get_unisam2_segmentation_generator(decoder, is_tiled=is_tiled, device=device)
    else:
        raise ValueError(f"Invalid segmentation_mode: {segmentation_mode}. Choose 'amg' or 'ais'.")
