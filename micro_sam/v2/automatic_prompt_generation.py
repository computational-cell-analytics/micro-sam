"""Automatic prompt generation (APG) for UniSAM2: propose seeds generously, let the model judge them.

The flow post-processing thresholds the decoder's convergence density to get instances. APG proposes
candidates *below* that threshold, prompts the interactive branch with each one, and keeps the masks it
scores highly, so the model does the discrimination that thresholding cannot. Worth +0.085 mSA on livecell.

The parameter surface is deliberately small. Levers that were tried and dropped are recorded in
`finetuning/v2/evaluation/APG.md`, which is the place to look before adding one back.
"""

import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.ndimage import find_objects

import torch

from bioimage_cpp.utils import Blocking
from bioimage_cpp.segmentation import label

from .normalization import normalize_raw
from ..v1.inference import _merge_segmentations
from ..v1.instance_segmentation import _get_centers
from ..util import make_temp_embedding_path, _ensure_rgb
from .util import precompute_image_embeddings, set_precomputed
from .postprocessing import DEFAULT_POSTPROCESSING, _compute_flow_density
from .instance_segmentation import (
    TiledUniSAM2InstanceSegmentation, UniSAM2InstanceSegmentation, USE_MODEL_DEVICE, Devices
)

# Only enters the merge order, never a cutoff, so it is a constant rather than a parameter.
STABILITY_SCORE_OFFSET = 1.0

DEFAULT_PROMPT_GENERATION = {
    # Below the flow post-processing's 'density_threshold': the model rejects the surplus candidates.
    # 1.5 is the modal per-dataset optimum over twelve datasets, on six of them.
    "candidate_threshold": 1.5,
    "min_candidate_size": 4,
    "score_threshold": 0.6,
    # The one axis that transfers: optimal on eleven of twelve datasets.
    "max_overlap": 0.15,
    "multimasking": True,
    # Off by default until the box stage is swept beyond livecell.
    "refine_with_box_prompts": False,
    "box_extension": 0,
    # Shared with the sparse post-processing, but tuned there for one clean peak per object rather than
    # for candidate recall, so they are exposed rather than pinned.
    "foreground_threshold": DEFAULT_POSTPROCESSING["sparse"]["foreground_threshold"],
    "n_iter": DEFAULT_POSTPROCESSING["sparse"]["n_iter"],
    "dt": DEFAULT_POSTPROCESSING["sparse"]["dt"],
    "sigma": DEFAULT_POSTPROCESSING["sparse"]["sigma"],
    "min_size": DEFAULT_POSTPROCESSING["sparse"]["min_size"],
}


def derive_point_prompts(
    foreground: np.ndarray,
    directed_distances: np.ndarray,
    candidate_threshold: float = DEFAULT_PROMPT_GENERATION["candidate_threshold"],
    foreground_threshold: float = DEFAULT_PROMPT_GENERATION["foreground_threshold"],
    n_iter: int = DEFAULT_PROMPT_GENERATION["n_iter"],
    dt: float = DEFAULT_PROMPT_GENERATION["dt"],
    sigma: float = DEFAULT_PROMPT_GENERATION["sigma"],
    min_candidate_size: int = DEFAULT_PROMPT_GENERATION["min_candidate_size"],
    backend: str = "cpp",
    n_threads: int = 1,
) -> Optional[Dict[str, np.ndarray]]:
    """Derive one positive point prompt per convergence-density component.

    The v1 counterpart intersects thresholded centre and boundary distances; UniSAM2 predicts directed
    distances instead, so the candidates come from the flow-convergence density that the sparse
    post-processing also seeds from, thresholded lower.

    Args:
        foreground: Foreground probability map, shape (Y, X).
        directed_distances: Distance channels stacked along axis 0. A leading z-channel is dropped, so
            `prediction[1:]` can be passed regardless of dimensionality.
        candidate_threshold: Density threshold for proposing candidates. Lower proposes more. The density
            of a component scales with the object's area, so this is coupled to object size.
        foreground_threshold: Foreground binarisation threshold, which bounds the pixels that can be
            proposed from.
        n_iter: Number of flow-integration steps. Together with 'dt' this is the distance a pixel is
            advected, which has to be enough to reach the object's centre.
        dt: Integration step size. Mostly only the product with 'n_iter' matters.
        sigma: Gaussian sigma for smoothing the convergence-density map.
        min_candidate_size: Discard components smaller than this, which are noise rather than objects.
        backend: Flow computation backend, ``"python"`` or ``"cpp"``.
        n_threads: Number of threads for the cpp backend.

    Returns:
        The prompts as {'points': (N, 1, 2) in XY, 'point_labels': (N, 1)}, or None if none were found.
    """
    if directed_distances.shape[0] > foreground.ndim:
        directed_distances = directed_distances[-foreground.ndim:]

    fg_mask = foreground > foreground_threshold
    density = _compute_flow_density(
        directed_distances, fg_mask, n_iter=int(n_iter), dt=dt, sigma=sigma,
        backend=backend, n_threads=n_threads,
    )
    candidates = label(density > candidate_threshold)

    if min_candidate_size > 0:
        ids, sizes = np.unique(candidates, return_counts=True)
        discard = ids[(sizes < min_candidate_size) & (ids > 0)]
        if discard.size:
            candidates[np.isin(candidates, discard)] = 0
    if candidates.max() == 0:
        return None

    # The interior point rather than the centroid: a curved or elongated object's centroid can lie
    # outside it, and a prompt outside the object segments the wrong thing.
    centers = _get_centers(candidates)
    if len(centers) == 0:
        return None

    return {
        "points": np.ascontiguousarray(centers[:, ::-1], dtype="float32")[:, None, :],  # SAM2 wants XY
        "point_labels": np.ones((len(centers), 1), dtype="int32"),
    }


def merge_by_score(
    records: List[Dict[str, Any]], shape: tuple, max_overlap: float = 0.3, min_size: int = 50,
) -> np.ndarray:
    """Merge prediction records in descending score order, each claiming only unclaimed pixels.

    Linear in the number of candidates, where `micro_sam.util.apply_nms` is quadratic, and marginally
    better on livecell: truncating a later mask preserves the better-scoring instance's boundary.

    Args:
        records: The prediction records, as produced by `AutomaticPromptGenerator._apply_prompts`.
        shape: The spatial shape of the output.
        max_overlap: Reject a candidate when more than this fraction of it is already claimed. This is
            the duplicate suppression of the merge.
        min_size: Minimum object size to keep.

    Returns:
        The instance segmentation, uint32 array.
    """
    out = np.zeros(shape, dtype="uint32")
    scores = np.array([record["predicted_iou"] * record["stability_score"] for record in records])
    next_id = 1
    for index in np.argsort(-scores):
        mask = records[index]["segmentation"]
        mask = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        area = int(mask.sum())
        if area < min_size:
            continue
        if int((mask & (out > 0)).sum()) / area > max_overlap:
            continue
        fresh = mask & (out == 0)
        if int(fresh.sum()) < min_size:
            continue
        out[fresh] = next_id
        next_id += 1
    return out


def refine_with_boxes(
    predictor, segmentation: np.ndarray, batch_size: int = 64, box_extension: int = 0,
) -> np.ndarray:
    """Re-prompt every instance with its bounding box and repaint the result.

    A box is much less ambiguous than a point. Derive the boxes from the predicted masks, not from the
    candidate regions: a candidate region is a fragment, so its box says the object is fragment-sized.

    Args:
        predictor: The SAM2 image predictor. The image must already be set on it.
        segmentation: The instance segmentation to refine.
        batch_size: Number of boxes per forward pass.
        box_extension: Number of pixels every box is grown by. Confluent data prefers 0, because a grown
            box reaches into the neighbouring object.

    Returns:
        The refined instance segmentation, uint32 array with the shape of the input.
    """
    shape = segmentation.shape
    boxes, ids = [], []
    for index, slices in enumerate(find_objects(segmentation)):
        if slices is None:
            continue
        y_slice, x_slice = slices
        boxes.append([
            max(0, x_slice.start - box_extension), max(0, y_slice.start - box_extension),
            min(shape[1], x_slice.stop + box_extension), min(shape[0], y_slice.stop + box_extension),
        ])
        ids.append(index + 1)
    if not boxes:
        return segmentation

    boxes, ids = np.array(boxes, dtype="float32"), np.array(ids, dtype="uint32")

    masks, scores = [], []
    for start in range(0, len(boxes), batch_size):
        batch = boxes[start:start + batch_size]
        mask, score, _ = predictor.predict(box=batch, multimask_output=False)
        masks.append(np.asarray(mask).reshape(len(batch), *shape).astype(bool))
        scores.append(np.asarray(score).reshape(-1))
    masks, scores = np.concatenate(masks), np.concatenate(scores)

    # Ascending score, so that the most confident instance is painted last and wins contested pixels.
    refined = np.zeros(shape, dtype="uint32")
    for index in np.argsort(scores):
        refined[masks[index]] = ids[index]
    return refined


class AutomaticPromptGenerator(UniSAM2InstanceSegmentation):
    """Generates an instance segmentation automatically, from prompts derived from the UniSAM2 decoder.

    Used in the same way as `UniSAM2InstanceSegmentation`, and the counterpart of
    `micro_sam.v1.instance_segmentation.AutomaticPromptGenerator`:
    ```python
    segmenter = AutomaticPromptGenerator(model, predictor)
    segmenter.initialize(image, ndim=2)  # Encode the image, then run the decoder on the encoding.
    masks = segmenter.generate(score_threshold=0.6)  # Prompt, then merge the masks.
    ```

    Only 2d images are supported.

    Args:
        model: The UniSAM2 model (see `get_unisam2_model` / `get_decoder`).
        predictor: The SAM2 image predictor for the interactive branch of the same model.
        device: The device the model lives on (used for the non-tiled 2d decoder).
        inference_device: The device intent used as the `devices=None` fallback. Defaults to the
            model device (single GPU); pass None to fan out over all visible GPUs, or a device / list.
    """

    # Read by `automatic_instance_segmentation` to decide whether to pass the AIS 'mode' argument.
    _has_postprocessing_mode = False

    def __init__(
        self,
        model: torch.nn.Module,
        predictor,
        device: Optional[Union[str, torch.device]] = None,
        inference_device: Devices = USE_MODEL_DEVICE,
    ) -> None:
        super().__init__(model, device=device, inference_device=inference_device)
        self._predictor = predictor
        self._image_embeddings = None
        # The embedding cache is keyed on these, which a SAM2 image predictor does not carry itself.
        sam2_model = getattr(predictor, "model", None)
        if getattr(predictor, "model_type", None) is None:
            predictor.model_type = getattr(sam2_model, "model_type", None) or "hvit"
        if getattr(predictor, "model_name", None) is None:
            predictor.model_name = getattr(sam2_model, "model_name", None) or predictor.model_type

    def _encode(self, image: np.ndarray) -> dict:
        """Run the image encoder once and return the embeddings that both branches use."""
        self._predictor.reset_predictor()
        self._predictor.set_image(_ensure_rgb(normalize_raw(image, output_dtype="uint8")))
        return {
            "features": self._predictor.get_image_embedding().cpu().numpy(),
            "high_res_feats": self._predictor._features["high_res_feats"],
            "input_size": self._predictor.model.image_size,
            "original_size": self._predictor._orig_hw,
        }

    def initialize(
        self, image: np.ndarray, ndim: int = 2, image_embeddings: Optional[dict] = None, **kwargs
    ) -> None:
        """Encode the image, run the decoder on that encoding and leave the predictor ready to be prompted.

        Both branches of a joint checkpoint share their image encoder weights, so one pass serves both
        and `generate` can be called repeatedly without any further encoding.

        Args:
            image: The input image, shape (Y, X) or (Y, X, C).
            ndim: The number of spatial dimensions. Must be 2.
            image_embeddings: Optional precomputed image embeddings. If given, the encoder does not run.
            kwargs: Additional arguments for `UniSAM2InstanceSegmentation.initialize`.
        """
        if ndim != 2:
            raise ValueError(f"Automatic prompt generation supports 2d images only, got ndim={ndim}.")

        if image_embeddings is None:
            image_embeddings = self._encode(image)
        else:
            set_precomputed(self._predictor, image_embeddings)
        super().initialize(image, ndim=ndim, image_embeddings=image_embeddings, **kwargs)
        self._image_embeddings = image_embeddings

    def get_state(self) -> dict:
        """Return the decoder predictions and the image embeddings, so that both branches can be restored.

        `generate` also prompts the interactive branch, which needs the encoding of the same image.
        """
        state = super().get_state()
        state["image_embeddings"] = self._image_embeddings
        return state

    def set_state(self, state: dict) -> None:
        """Restore the decoder predictions and the encoding of the image they belong to.

        The state must hold either 'image_embeddings' or 'image'. Without one, `generate` would prompt
        whatever image the predictor still holds.

        Args:
            state: The state, as returned by `get_state`, or a dict with 'prediction' and 'image'.
        """
        image_embeddings = state.get("image_embeddings")
        if image_embeddings is None:
            if "image" not in state:
                raise ValueError("The state must hold either 'image_embeddings' or 'image'.")
            image_embeddings = self._encode(state["image"])
        else:
            set_precomputed(self._predictor, image_embeddings)
        super().set_state(state)
        self._image_embeddings = image_embeddings

    def clear_state(self) -> None:
        """Clear the decoder predictions and the image that is set on the predictor."""
        super().clear_state()
        self._image_embeddings = None
        self._predictor.reset_predictor()

    @torch.no_grad()
    def generate(
        self,
        candidate_threshold: float = DEFAULT_PROMPT_GENERATION["candidate_threshold"],
        foreground_threshold: float = DEFAULT_PROMPT_GENERATION["foreground_threshold"],
        n_iter: int = DEFAULT_PROMPT_GENERATION["n_iter"],
        dt: float = DEFAULT_PROMPT_GENERATION["dt"],
        sigma: float = DEFAULT_PROMPT_GENERATION["sigma"],
        min_candidate_size: int = DEFAULT_PROMPT_GENERATION["min_candidate_size"],
        score_threshold: float = DEFAULT_PROMPT_GENERATION["score_threshold"],
        max_overlap: float = DEFAULT_PROMPT_GENERATION["max_overlap"],
        min_size: int = DEFAULT_PROMPT_GENERATION["min_size"],
        refine_with_box_prompts: bool = DEFAULT_PROMPT_GENERATION["refine_with_box_prompts"],
        box_extension: int = DEFAULT_PROMPT_GENERATION["box_extension"],
        multimasking: bool = DEFAULT_PROMPT_GENERATION["multimasking"],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Derive prompts from the stored predictions, apply them and merge the masks.

        Args:
            candidate_threshold: Density threshold for proposing candidates. Lower proposes more.
            foreground_threshold: Foreground binarisation threshold. Here it only limits which pixels can
                be proposed from, since the masks come from the interactive branch, so it trades candidate
                recall rather than boundary quality.
            n_iter: Number of flow-integration steps for the candidate density.
            dt: Integration step size. Mostly only the product with 'n_iter' matters, which is the
                distance a pixel is advected.
            sigma: Gaussian sigma for smoothing the candidate density. Less smoothing leaves more peaks,
                which costs precision for the sparse post-processing but buys candidate recall here.
            min_candidate_size: Discard density components smaller than this.
            score_threshold: Discard candidates whose predicted IoU is below this.
            max_overlap: Reject a candidate when more than this fraction of it is already claimed.
            min_size: Minimum object size in the result.
            refine_with_box_prompts: Whether to re-prompt every merged instance with its bounding box,
                see `refine_with_boxes`.
            box_extension: Number of pixels every box of that second stage is grown by.
            multimasking: Whether to predict several masks per point and keep the best scoring one. A
                single point is ambiguous between one object and a cluster, so this is on by default.
            batch_size: Number of prompts per forward pass.

        Returns:
            The instance segmentation, uint32 array with the spatial shape of the prediction.
        """
        if not self._is_initialized:
            raise RuntimeError("The segmenter has not been initialized. Call 'initialize' first.")

        shape = self._prediction[0].shape
        prompts = derive_point_prompts(
            self._prediction[0], self._prediction[1:], candidate_threshold=candidate_threshold,
            foreground_threshold=foreground_threshold, n_iter=n_iter, dt=dt, sigma=sigma,
            min_candidate_size=min_candidate_size,
        )
        if prompts is None:
            return np.zeros(shape, dtype="uint32")

        segmentation = self._apply_and_merge(
            prompts, shape, multimasking=multimasking, batch_size=batch_size,
            score_threshold=score_threshold, max_overlap=max_overlap, min_size=min_size,
        )
        if refine_with_box_prompts and segmentation.max() > 0:
            segmentation = self._refine_boxes(segmentation, batch_size, box_extension)
        return segmentation

    def _apply_and_merge(
        self, prompts: dict, shape: tuple, multimasking: bool, batch_size: int,
        score_threshold: float, max_overlap: float, min_size: int,
    ) -> np.ndarray:
        """Turn the prompts into masks and merge them into an instance segmentation."""
        records = self._apply_prompts(prompts, multimasking=multimasking, batch_size=batch_size)
        records = [record for record in records if record["predicted_iou"] >= score_threshold]
        if not records:
            return np.zeros(shape, dtype="uint32")
        return merge_by_score(records, shape, max_overlap=max_overlap, min_size=min_size)

    def _refine_boxes(self, segmentation: np.ndarray, batch_size: int, box_extension: int) -> np.ndarray:
        """Re-prompt every instance with its bounding box, see `refine_with_boxes`."""
        return refine_with_boxes(
            self._predictor, segmentation, batch_size=batch_size, box_extension=box_extension,
        )

    def _apply_prompts(self, prompts, multimasking: bool, batch_size: int) -> List[Dict[str, Any]]:
        """Prompt the interactive branch in batches, returning records for the merge."""
        from sam2.utils.amg import calculate_stability_score

        points, point_labels = prompts["points"], prompts["point_labels"]
        mask_threshold = getattr(self._predictor, "mask_threshold", 0.0)

        records = []
        for start in range(0, len(points), batch_size):
            stop = start + batch_size
            batch_points, batch_labels = points[start:stop], point_labels[start:stop]
            masks, scores, _ = self._predictor.predict(
                point_coords=batch_points, point_labels=batch_labels,
                multimask_output=multimasking, return_logits=True,
            )
            # 'predict' squeezes the prompt axis away for a single prompt, so it is restored here before
            # the multimask proposals of each prompt are reduced to the best-scoring one.
            n_prompts = len(batch_points)
            logits = np.asarray(masks).reshape(n_prompts, -1, *masks.shape[-2:])
            scores = np.asarray(scores).reshape(n_prompts, -1)
            best = scores.argmax(axis=1)
            index = np.arange(n_prompts)
            logits, scores = logits[index, best], scores[index, best]

            logits = torch.from_numpy(np.ascontiguousarray(logits))
            stability = calculate_stability_score(logits, mask_threshold, STABILITY_SCORE_OFFSET)
            for mask, score, stable in zip(logits > mask_threshold, scores, stability):
                if not mask.any():
                    continue
                records.append({
                    "segmentation": mask,
                    "predicted_iou": float(score),
                    "stability_score": float(stable),
                })
        return records


class TiledAutomaticPromptGenerator(AutomaticPromptGenerator, TiledUniSAM2InstanceSegmentation):
    """Generates an instance segmentation with automatically generated prompts, for tiled inference.

    Like `AutomaticPromptGenerator`, but both branches run tile by tile, which keeps the encoder at its
    native resolution instead of downscaling the whole image to its input size.

    The prompts are derived once from the stitched prediction, so a candidate spanning a tile border is
    proposed once. Each is assigned to the tile whose inner block holds its point and prompted within
    that tile's halo, so no object is segmented twice and no mask is cut off at a nearby border.

    Args:
        model: The UniSAM2 model (see `get_unisam2_model` / `get_decoder`).
        predictor: The SAM2 image predictor for the interactive branch of the same model.
        device: The device the model lives on.
        inference_device: The device intent used as the `devices=None` fallback.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        predictor,
        device: Optional[Union[str, torch.device]] = None,
        inference_device: Devices = USE_MODEL_DEVICE,
    ) -> None:
        super().__init__(model, predictor, device=device, inference_device=inference_device)
        self._tiling = None
        self._halo = None
        self._temporary_embedding_path = None

    def initialize(
        self,
        image: np.ndarray,
        ndim: int = 2,
        image_embeddings: Optional[dict] = None,
        tile_shape: Optional[tuple] = None,
        halo: Optional[tuple] = None,
        save_path: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Compute the tiled embeddings, run the decoder on them and keep them for the prompting.

        The same tiled embeddings serve both branches. Unlike the non-tiled generator they are needed
        again in `generate`, so they are held until `clear_state`.

        Args:
            image: The input image, shape (Y, X) or (Y, X, C).
            ndim: The number of spatial dimensions. Must be 2.
            image_embeddings: Optional precomputed tiled image embeddings. The tiling is taken from
                them when they are given.
            tile_shape: The tile shape, (y, x). Required when no embeddings are given.
            halo: The overlap between the tiles, (y, x). Required when no embeddings are given.
            save_path: Optional path to cache the computed embeddings in a zarr container. Without one
                an ephemeral store is used, which `clear_state` removes.
            verbose: Whether to print progress while the embeddings are computed.
            kwargs: Additional arguments for `TiledUniSAM2InstanceSegmentation.initialize`.
        """
        if ndim != 2:
            raise ValueError(f"Automatic prompt generation supports 2d images only, got ndim={ndim}.")

        if image_embeddings is None:
            if tile_shape is None or halo is None:
                raise ValueError("Both 'tile_shape' and 'halo' have to be passed for the tiled generator.")
            path = save_path
            if path is None:
                self._temporary_embedding_path = make_temp_embedding_path()
                path = self._temporary_embedding_path
            image_embeddings = precompute_image_embeddings(
                self._predictor, image, save_path=path, ndim=2, tile_shape=tile_shape, halo=halo,
                verbose=verbose, lazy_loading=True,
            )

        TiledUniSAM2InstanceSegmentation.initialize(
            self, image, ndim=2, image_embeddings=image_embeddings, **kwargs
        )
        self._image_embeddings = image_embeddings

        # From the embeddings, not the arguments, so the prompting cannot disagree with the encoding.
        features = image_embeddings["features"]
        self._tiling = Blocking(
            [0, 0], [int(s) for s in features.attrs["shape"]], [int(s) for s in features.attrs["tile_shape"]]
        )
        self._halo = [int(s) for s in features.attrs["halo"]]

    def _tile_bounding_box(self, tile_id: int) -> tuple:
        """The outer (halo-extended) block of a tile, as a slice tuple."""
        block = self._tiling.get_block_with_halo(tile_id, list(self._halo)).outer_block
        return tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))

    def _tiles_for_points(self, points: np.ndarray) -> Dict[int, List[int]]:
        """Group prompt indices by the tile whose inner block holds their point.

        The inner blocks do not overlap, so every candidate is prompted exactly once.
        """
        assignment = {}
        for index, (x, y) in enumerate(points[:, 0, :]):
            tile_id = self._tiling.coordinates_to_block_id([int(y), int(x)])
            assignment.setdefault(tile_id, []).append(index)
        return assignment

    def _apply_and_merge(
        self, prompts: dict, shape: tuple, multimasking: bool, batch_size: int,
        score_threshold: float, max_overlap: float, min_size: int,
    ) -> np.ndarray:
        """Prompt each tile with the candidates that belong to it, then stitch the per-tile merges."""
        points, point_labels = prompts["points"], prompts["point_labels"]
        segmentation = np.zeros(shape, dtype="uint32")
        offset = 0

        for tile_id, indices in sorted(self._tiles_for_points(points).items()):
            bounding_box = self._tile_bounding_box(tile_id)
            tile_shape = tuple(box.stop - box.start for box in bounding_box)
            # The prompts are in the full image's frame, the tile's embeddings in the tile's.
            origin = np.array([bounding_box[1].start, bounding_box[0].start], dtype="float32")

            set_precomputed(self._predictor, self._image_embeddings, tile_id=tile_id)
            records = self._apply_prompts(
                {"points": points[indices] - origin, "point_labels": point_labels[indices]},
                multimasking=multimasking, batch_size=batch_size,
            )
            records = [record for record in records if record["predicted_iou"] >= score_threshold]
            if not records:
                continue

            tile_segmentation = merge_by_score(
                records, tile_shape, max_overlap=max_overlap, min_size=min_size,
            )
            max_id = int(tile_segmentation.max())
            if max_id == 0:
                continue
            # Keep the instance ids unique across tiles before the halo overlaps are resolved.
            tile_segmentation[tile_segmentation != 0] += offset
            offset += max_id
            segmentation[bounding_box] = _merge_segmentations(
                tile_segmentation, segmentation[bounding_box]
            )
        return segmentation

    def _refine_boxes(self, segmentation: np.ndarray, batch_size: int, box_extension: int) -> np.ndarray:
        """Re-prompt every instance with its bounding box, in the tile that holds its interior point.

        Refined once, by the tile whose inner block holds its point, so two tiles cannot both claim it.
        """
        ids = np.unique(segmentation)
        ids = ids[ids != 0]
        if ids.size == 0:
            return segmentation

        centers = _get_centers(segmentation)
        if len(centers) != len(ids):
            raise RuntimeError(f"Got {len(centers)} interior points for {len(ids)} instances.")
        assignment = {}
        for label_id, (y, x) in zip(ids, centers):
            tile_id = self._tiling.coordinates_to_block_id([int(y), int(x)])
            assignment.setdefault(tile_id, []).append(label_id)

        refined = np.zeros_like(segmentation)
        for tile_id, label_ids in sorted(assignment.items()):
            bounding_box = self._tile_bounding_box(tile_id)
            crop = segmentation[bounding_box]
            crop = np.where(np.isin(crop, label_ids), crop, 0).astype("uint32")

            set_precomputed(self._predictor, self._image_embeddings, tile_id=tile_id)
            tile_refined = refine_with_boxes(
                self._predictor, crop, batch_size=batch_size, box_extension=box_extension,
            )
            # Refined masks keep their ids; an earlier tile wins a contested pixel, as in the merge.
            target = refined[bounding_box]
            refined[bounding_box] = np.where(target == 0, tile_refined, target)
        return refined

    def get_state(self) -> dict:
        """@private"""
        raise NotImplementedError(
            "The tiled prompt generator cannot serialize its state, because it holds tiled embeddings."
        )

    def set_state(self, state: dict) -> None:
        """@private"""
        raise NotImplementedError(
            "The tiled prompt generator cannot restore its state, because it holds tiled embeddings."
        )

    def clear_state(self) -> None:
        """Clear the decoder predictions and the tiled embeddings, removing an ephemeral store."""
        super().clear_state()
        self._tiling = None
        self._halo = None
        if self._temporary_embedding_path is not None:
            shutil.rmtree(self._temporary_embedding_path, ignore_errors=True)
            self._temporary_embedding_path = None
