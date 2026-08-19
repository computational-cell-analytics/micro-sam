"""Automatic prompt generation (APG) for UniSAM2: propose seeds generously, let the model judge them.

The flow post-processing thresholds the decoder's convergence density to get instances. APG proposes
candidates *below* that threshold, prompts the interactive branch with each one, and keeps the masks it
scores highly, so the model does the discrimination that thresholding cannot. Worth +0.085 mSA on livecell.

A volume works the same way, with the SAM2 video predictor in place of the image predictor: the flow
is integrated in 3d so a density component is a whole object, each is scored in 2d on the slice it
converges on, and the survivors are propagated through the volume. Worth +0.06 mSA over AIS on ten
volumetric LM datasets and +0.27 on four EM ones.

The parameter surface is deliberately small: everything here was measured to matter on at least one
dataset, and a good many other levers were tried and dropped. For volumes those were: re-prompting
every instance with the prompts the merge grouped onto it (+0.001), negative prompts from the
adjacent instances (+0.001), conditioning the anchor slice with a box or the 2d mask instead of the
point (+0.001 and -0.005), and sampling further positive prompts from an instance's own mask
(-0.034, because a prompt on a propagated slice turns it into a conditioning frame and replaces the
mask there with a single-point one). Selection is not what limits the result: an oracle that hands
every object its best-matching propagated mask scores 0.006 above the merge.
"""

import shutil
import contextlib
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm
from scipy.ndimage import find_objects, distance_transform_edt

import torch

from bioimage_cpp.utils import Blocking
from bioimage_cpp.segmentation import label

from .normalization import normalize_raw
from ..v1.inference import _merge_segmentations
from ..util import make_temp_embedding_path, _ensure_rgb
from .postprocessing import DEFAULT_POSTPROCESSING, _compute_flow_density
from .prompt_based_segmentation import PromptableSegmentation3D, _crop_to_original_shape
from .util import precompute_image_embeddings, set_precomputed, get_sam2_image_predictor
from .instance_segmentation import (
    TiledUniSAM2InstanceSegmentation, UniSAM2InstanceSegmentation, USE_MODEL_DEVICE, Devices,
    _set_image_predictor_from_backbone, _set_image_predictor_from_3d_embeddings,
)

# Only enters the merge order, never a cutoff, so it is a constant.
STABILITY_SCORE_OFFSET = 1.0

DEFAULT_PROMPT_GENERATION = {
    # Below the flow post-processing's 'density_threshold': the model rejects the surplus candidates.
    "candidate_threshold": 1.5,
    "min_candidate_size": 4,
    "score_threshold": 0.6,
    # The one axis that transfers: optimal on eleven of twelve datasets.
    "max_overlap": 0.15,
    "multimasking": True,
    # Off by default until the box stage is swept beyond livecell.
    "refine_with_box_prompts": False,
    "box_extension": 0,
    # Volumes only. A ladder separates the peaks that a single threshold merges into one component.
    "candidate_threshold_3d": (1.5, 10.0),
    # Volumes only. Trades device memory against the pass count.
    "n_objects_per_pass": 16,
    # Volumes only. None propagates through the whole volume.
    "early_stop_patience": None,
    # Shared with the sparse post-processing, but tuned there for one peak per object, not for recall.
    "foreground_threshold": DEFAULT_POSTPROCESSING["sparse"]["foreground_threshold"],
    "n_iter": DEFAULT_POSTPROCESSING["sparse"]["n_iter"],
    "dt": DEFAULT_POSTPROCESSING["sparse"]["dt"],
    "sigma": DEFAULT_POSTPROCESSING["sparse"]["sigma"],
    "min_size": DEFAULT_POSTPROCESSING["sparse"]["min_size"],
    # Throughput only, the density is the same either way.
    "n_threads": 8,
}


def sam2_autocast(device):
    """Run the SAM2 branch in half precision, as the UniSAM2 decoder already does.

    Worth 1.15x end to end on livecell, for -0.0004 mSA, which is inside the noise of the merge.
    Only on cuda: MPS has no tensor cores, so half precision buys no throughput there and only costs
    the conversions. Measured on an M-series Mac, fp16 makes the prompting 1.35x slower and `generate`
    1.26x slower than fp32, which is also the more accurate of the two.

    Args:
        device: The device the branch runs on. Half precision is used on cuda only.

    Returns:
        The autocast context, or a null context where half precision does not apply.
    """
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def interior_points(labels: np.ndarray) -> np.ndarray:
    """The deepest interior point of every labelled component, in ascending label order.

    The v1 counterpart transforms a `find_boundaries` image, whose 'outer' mode marks a thin
    component's own pixels as boundary. Its maximum then falls on the bounding box's first pixel,
    which need not lie in the component at all.

    Args:
        labels: A labelled image or volume.

    Returns:
        The points as (N, ndim), one per component, in the array's own axis order.
    """
    points = []
    for index, bounding_box in enumerate(find_objects(labels), start=1):
        if bounding_box is None:
            continue
        # Padded, so a component reaching the border is measured from that border too.
        distances = distance_transform_edt(np.pad(labels[bounding_box] == index, 1))
        coordinates = np.unravel_index(int(np.argmax(distances)), distances.shape)
        points.append(tuple(int(c) + box.start - 1 for c, box in zip(coordinates, bounding_box)))
    return np.array(points, dtype="int64").reshape(-1, labels.ndim)


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
    n_threads: int = DEFAULT_PROMPT_GENERATION["n_threads"],
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

    # The interior point rather than the centroid, which can lie outside a curved object.
    centers = interior_points(candidates)
    if len(centers) == 0:
        return None

    return {
        "points": np.ascontiguousarray(centers[:, ::-1], dtype="float32")[:, None, :],  # SAM2 wants XY
        "point_labels": np.ones((len(centers), 1), dtype="int32"),
    }


def derive_volume_prompts(
    foreground: np.ndarray,
    directed_distances: np.ndarray,
    candidate_threshold: float = DEFAULT_PROMPT_GENERATION["candidate_threshold"],
    foreground_threshold: float = DEFAULT_PROMPT_GENERATION["foreground_threshold"],
    n_iter: int = DEFAULT_PROMPT_GENERATION["n_iter"],
    dt: float = DEFAULT_PROMPT_GENERATION["dt"],
    sigma: float = DEFAULT_PROMPT_GENERATION["sigma"],
    spacing: Optional[tuple] = None,
    min_candidate_size: int = DEFAULT_PROMPT_GENERATION["min_candidate_size"],
    backend: str = "cpp",
    n_threads: int = DEFAULT_PROMPT_GENERATION["n_threads"],
) -> Optional[Dict[str, np.ndarray]]:
    """Derive one positive point prompt, on one slice, per volumetric convergence-density component.

    The volumetric counterpart of `derive_point_prompts`. The flow is integrated in 3d, so a component
    of the resulting density is a whole object rather than one of its cross-sections, and each object
    is prompted once instead of once per slice. The prompt is placed on the slice where the component
    converges most, which is the slice the video predictor propagates away from in both directions.

    Args:
        foreground: Foreground probability map, shape (Z, Y, X).
        directed_distances: Distance channels stacked along axis 0, shape (3, Z, Y, X).
        candidate_threshold: Density threshold for proposing candidates, or several of them. Lower
            proposes more, but also merges the peaks of touching objects into one component, so a
            ladder recovers what a single threshold cannot separate.
        foreground_threshold: Foreground binarisation threshold, which bounds the voxels that can be
            proposed from.
        n_iter: Number of flow-integration steps. Together with 'dt' this is the distance a voxel is
            advected, which has to be enough to reach the object's centre.
        dt: Integration step size. Mostly only the product with 'n_iter' matters.
        sigma: Gaussian sigma for smoothing the convergence-density map.
        spacing: Anisotropic voxel spacing, e.g. (4, 1, 1), for physically isotropic smoothing.
        min_candidate_size: Discard components smaller than this, which are noise rather than objects.
        backend: Flow computation backend, ``"python"`` or ``"cpp"``.
        n_threads: Number of threads for the cpp backend.

    Returns:
        The prompts as {'points': (N, 1, 2) in XY, 'point_labels': (N, 1), 'frames': (N,) slice
        indices}, or None if no candidate was found.
    """
    if foreground.ndim != 3:
        raise ValueError(f"Volumetric prompt generation expects a (Z, Y, X) foreground map, got {foreground.shape}.")
    if directed_distances.shape[0] != 3:
        raise ValueError(f"Expected 3 distance channels, got {directed_distances.shape[0]}.")

    fg_mask = foreground > foreground_threshold
    density = _compute_flow_density(
        directed_distances, fg_mask, n_iter=int(n_iter), dt=dt, sigma=sigma, spacing=spacing,
        backend=backend, n_threads=n_threads,
    )

    points, frames, seen = [], [], set()
    # Descending, so that the peaks a lower threshold merges into one component are proposed first.
    for threshold in sorted(np.atleast_1d(np.asarray(candidate_threshold, dtype="float32")), reverse=True):
        candidates = label(density > threshold)
        if min_candidate_size > 0:
            ids, sizes = np.unique(candidates, return_counts=True)
            discard = ids[(sizes < min_candidate_size) & (ids > 0)]
            if discard.size:
                candidates[np.isin(candidates, discard)] = 0

        for index, bounding_box in enumerate(find_objects(candidates)):
            if bounding_box is None:
                continue
            component = candidates[bounding_box] == (index + 1)
            component_density = np.where(component, density[bounding_box], 0.0)
            # The most converged slice, i.e. the one closest to the object's centre.
            z = int(np.argmax(component_density.sum(axis=(1, 2))))
            plane = component_density[z]
            # The density peak: it is inside the component and the least ambiguous prompt.
            y, x = np.unravel_index(int(np.argmax(plane)), plane.shape)
            anchor = (bounding_box[0].start + z, bounding_box[1].start + y, bounding_box[2].start + x)
            # A component that no lower threshold has merged peaks at the same voxel at every level.
            if anchor in seen:
                continue
            seen.add(anchor)
            frames.append(anchor[0])
            points.append((anchor[2], anchor[1]))  # SAM2 wants XY.

    if not points:
        return None

    return {
        "points": np.array(points, dtype="float32")[:, None, :],
        "point_labels": np.ones((len(points), 1), dtype="int32"),
        "frames": np.array(frames, dtype="int64"),
    }


def merge_by_score(
    records: List[Dict[str, Any]], shape: tuple, max_overlap: float = 0.3, min_size: int = 50,
    return_matches: bool = False, return_reasons: bool = False,
) -> Union[np.ndarray, tuple]:
    """Merge prediction records in descending score order, each claiming only unclaimed pixels.

    Linear in the number of candidates, where `micro_sam.util.apply_nms` is quadratic, and marginally
    better on livecell: truncating a later mask preserves the better-scoring instance's boundary.

    A record carries a 'bounding_box' (a tuple of slices) and its mask is only the crop inside that
    box, so both the merge and the records themselves scale with the objects rather than with the
    image. A volumetric mask per object would otherwise make the 3d generator run out of memory.

    Args:
        records: The prediction records, as produced by `AutomaticPromptGenerator._apply_prompts`.
        shape: The spatial shape of the output.
        max_overlap: Reject a candidate when more than this fraction of it is already claimed. This is
            the duplicate suppression of the merge.
        min_size: Minimum object size to keep.
        return_matches: Whether to also return which record made each instance.
        return_reasons: Whether to also return why each record was kept or dropped. A candidate is
            'too small', a 'duplicate' when a better-scoring mask already claims more than
            'max_overlap' of it, 'truncated below min size' when too few of its pixels are free, or
            'kept'. This is what the merge does, reported rather than recomputed.

    Returns:
        The instance segmentation, uint32 array. If `return_matches`, additionally a mapping from
        every instance id to the index of the record that made it. If `return_reasons`, additionally
        the reason per record, in the order the records were given.
    """
    out = np.zeros(shape, dtype="uint32")
    scores = np.array([record["predicted_iou"] * record["stability_score"] for record in records])
    full_box = tuple(slice(None) for _ in shape)
    matches = {}
    reasons = ["" for _ in records]
    next_id = 1
    for index in np.argsort(-scores):
        record = records[index]
        mask = record["segmentation"]
        mask = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        area = int(mask.sum())
        if area < min_size:
            reasons[index] = "too small"
            continue
        # A view, so painting the fresh pixels below writes straight into the output.
        target = out[record.get("bounding_box", full_box)]
        claimed = target[mask]
        if int(np.count_nonzero(claimed)) / area > max_overlap:
            reasons[index] = "duplicate"
            continue
        fresh = mask & (target == 0)
        if int(fresh.sum()) < min_size:
            reasons[index] = "truncated below min size"
            continue
        target[fresh] = next_id
        reasons[index] = "kept"
        matches[next_id] = int(index)
        next_id += 1

    result = (out,)
    if return_matches:
        result += (matches,)
    if return_reasons:
        result += (reasons,)
    return result[0] if len(result) == 1 else result


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


def _records_shape(records: List[Dict[str, Any]]) -> tuple:
    """The smallest canvas that holds every record, which is all a merge of cropped masks needs."""
    boxes = [record["bounding_box"] for record in records]
    return tuple(max(box[axis].stop for box in boxes) for axis in range(len(boxes[0])))


def _volume_records(
    video_segments: Dict[int, Dict[int, np.ndarray]], candidates: List[dict], shape: tuple,
) -> List[Dict[str, Any]]:
    """Assemble one volumetric mask record per propagated object, cropped to its bounding box.

    The propagation yields its masks slice by slice. Holding one full-volume mask per object would
    scale with the object count times the volume, so each object is kept as its bounding-box crop,
    which scales with the objects themselves. `merge_by_score` paints from those crops.

    Args:
        video_segments: The propagation result, {frame: {object_id: mask}}, as returned by
            `PromptableSegmentation3D.propagate_prompts`.
        candidates: The candidates of this pass, in the order their object ids were assigned.
        shape: The shape of the volume, (Z, Y, X).

    Returns:
        The records, carrying the anchor-slice scores of their candidate for the merge.
    """
    per_object = {}
    for frame, masks in sorted(video_segments.items()):
        for object_id, mask in masks.items():
            mask = _crop_to_original_shape(np.asarray(mask).squeeze(), shape[-2:])
            rows, columns = np.nonzero(mask)
            if len(rows) == 0:
                continue
            y0, y1 = int(rows.min()), int(rows.max()) + 1
            x0, x1 = int(columns.min()), int(columns.max()) + 1
            per_object.setdefault(object_id, []).append(
                (int(frame), y0, y1, x0, x1, mask[y0:y1, x0:x1].copy())
            )

    records = []
    for object_id, entries in per_object.items():
        candidate = candidates[object_id - 1]
        # The frames are in ascending order, so the first and last entry bound the object in z.
        z0, z1 = entries[0][0], entries[-1][0] + 1
        y0, x0 = min(entry[1] for entry in entries), min(entry[3] for entry in entries)
        y1, x1 = max(entry[2] for entry in entries), max(entry[4] for entry in entries)

        local = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=bool)
        for frame, from_y, to_y, from_x, to_x, mask in entries:
            local[frame - z0, from_y - y0:to_y - y0, from_x - x0:to_x - x0] = mask

        records.append({
            "segmentation": local,
            "bounding_box": (slice(z0, z1), slice(y0, y1), slice(x0, x1)),
            "predicted_iou": candidate["score"],
            "stability_score": candidate["stability"],
        })
    return records


class AutomaticPromptGenerator(UniSAM2InstanceSegmentation):
    """Generates an instance segmentation automatically, from prompts derived from the UniSAM2 decoder.

    Used in the same way as `UniSAM2InstanceSegmentation`, and the counterpart of
    `micro_sam.v1.instance_segmentation.AutomaticPromptGenerator`:
    ```python
    segmenter = AutomaticPromptGenerator(model, predictor)
    segmenter.initialize(image, ndim=2)  # Encode the image, then run the decoder on the encoding.
    masks = segmenter.generate(score_threshold=0.6)  # Prompt, then merge the masks.
    ```

    Volumes are segmented by passing the SAM2 video predictor instead of an image predictor:
    ```python
    segmenter = AutomaticPromptGenerator(model, video_predictor)
    segmenter.initialize(volume, ndim=3)
    masks = segmenter.generate()  # Prompt one slice per object, propagate, then merge the masks.
    ```
    The decoder finds the objects and the interactive branch segments them either way. What differs is
    that an object's mask comes from propagating one prompt through the volume rather than from a
    single forward pass, and that the candidates are scored in 2d on the slice they are prompted on,
    which drops the weak and the duplicate ones before the propagation - which is where the cost is.
    The decoder and the video predictor read the same 3d embeddings, so the volume is encoded once.

    Args:
        model: The UniSAM2 model (see `get_unisam2_model` / `get_decoder`).
        predictor: The SAM2 image predictor for the interactive branch of the same model, or its
            video predictor, which is what a volume is prompted with.
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
        # The image predictor is built on the video predictor's own weights: no second backbone.
        self._video_predictor = predictor if hasattr(predictor, "propagate_in_video") else None
        if self._video_predictor is None:
            self._predictor = predictor
        else:
            self._predictor = get_sam2_image_predictor(predictor)
        predictor = self._predictor

        self._image_embeddings = None
        self._volume = None
        self._propagator = None
        self._offload_to_cpu = None
        self._max_cached_frames = None
        self._temporary_embedding_path = None
        self._i = None
        # The embedding cache is keyed on these, which a SAM2 image predictor does not carry itself.
        sam2_model = getattr(predictor, "model", None)
        if getattr(predictor, "model_type", None) is None:
            predictor.model_type = getattr(sam2_model, "model_type", None) or "hvit"
        if getattr(predictor, "model_name", None) is None:
            predictor.model_name = getattr(sam2_model, "model_name", None) or predictor.model_type

    def _encode(self, image: np.ndarray) -> dict:
        """Run the image encoder once and return the embeddings that both branches use."""
        self._predictor.reset_predictor()
        with sam2_autocast(self._predictor.device):
            self._predictor.set_image(_ensure_rgb(normalize_raw(image, output_dtype="uint8")))
        return {
            "features": self._predictor.get_image_embedding().cpu().numpy(),
            "high_res_feats": self._predictor._features["high_res_feats"],
            "input_size": self._predictor.model.image_size,
            "original_size": self._predictor._orig_hw,
        }

    def _prepare_image_embeddings(self, image_embeddings: dict, i: Optional[int]) -> dict:
        """Prepare image embeddings for the interactive branch."""
        if i is None:
            set_precomputed(self._predictor, image_embeddings)
            return image_embeddings

        if "fpn" not in image_embeddings:
            raise ValueError("A slice index requires video-style embeddings with FPN features.")
        _set_image_predictor_from_3d_embeddings(self._predictor, image_embeddings, i)
        return {
            "features": self._predictor.get_image_embedding(),
            "high_res_feats": self._predictor._features["high_res_feats"],
            "input_size": self._predictor.model.image_size,
            "original_size": self._predictor._orig_hw,
        }

    def initialize(
        self,
        image: np.ndarray,
        ndim: int = 2,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: bool = False,
        offload_to_cpu: Optional[bool] = None,
        cache_all_slices: bool = False,
        lazy_embeddings: bool = True,
        **kwargs,
    ) -> None:
        """Encode the input, run the decoder on that encoding and leave the predictor ready to be prompted.

        Both branches of a joint checkpoint share their image encoder weights, so one pass serves both
        and `generate` can be called repeatedly without any further encoding.

        Args:
            image: The input image, shape (Y, X) or (Y, X, C), or the input volume, shape (Z, Y, X).
            ndim: The number of spatial dimensions, 2 or 3. A volume requires a video predictor.
            image_embeddings: Optional precomputed image embeddings. If given, the encoder does not run.
            i: The slice index for video-style embeddings. By default the embeddings contain one image.
            save_path: Optional path to cache the embeddings of a volume in a zarr container. Without
                one an ephemeral store is used, which `clear_state` removes.
            verbose: Whether to print progress while the embeddings of a volume are computed.
            offload_to_cpu: Whether a volume's tracking state is held on the host rather than on the
                device. False propagates the same masks 6-11% faster, for the memory of one pass of
                objects across the volume, which is about 17 MB per slice at the default pass size.
                None leaves the choice to `PromptableSegmentation3D`, which offloads on cuda.
            cache_all_slices: Whether every slice's features stay on the device while a volume is
                propagated. A pass walks the whole volume, so a cache shorter than it is never hit
                and every slice is fetched again on every pass - which is most of the run on a deep
                volume. Costs about 90 MB of device memory per slice.
            lazy_embeddings: Whether a volume's embeddings are read from their store slice by slice.
                False holds them in host memory instead, which makes the fetch above much cheaper
                without spending device memory.
            kwargs: Additional arguments for `UniSAM2InstanceSegmentation.initialize`.
        """
        if ndim == 3:
            self._i = None
            self._initialize_volume(
                image, image_embeddings, save_path, verbose, offload_to_cpu, cache_all_slices,
                lazy_embeddings, **kwargs
            )
            return
        if ndim != 2:
            raise ValueError(f"Automatic prompt generation supports 2d and 3d inputs, got ndim={ndim}.")

        if image_embeddings is None:
            image_embeddings = self._encode(image)
        else:
            image_embeddings = self._prepare_image_embeddings(image_embeddings, i)
        super().initialize(image, ndim=ndim, image_embeddings=image_embeddings, **kwargs)
        self._image_embeddings = image_embeddings
        self._i = None

    def _build_propagator(self, volume, image_embeddings) -> PromptableSegmentation3D:
        """The video predictor this volume is propagated with, holding its state where asked to."""
        return PromptableSegmentation3D(
            self._video_predictor, volume, image_embeddings,
            offload_state_to_cpu=self._offload_to_cpu, max_cached_frames=self._max_cached_frames,
        )

    def _initialize_volume(
        self, volume, image_embeddings, save_path, verbose, offload_to_cpu, cache_all_slices,
        lazy_embeddings, **kwargs
    ) -> None:
        """Encode the volume, run the decoder on that encoding and initialize the video predictor."""
        if self._video_predictor is None:
            raise ValueError(
                "Volumetric prompt generation prompts the SAM2 video predictor, so it has to be "
                "constructed with one instead of an image predictor."
            )
        if volume.ndim != 3:
            raise ValueError(f"Volumetric prompt generation expects a (Z, Y, X) volume, got shape {volume.shape}.")

        if image_embeddings is None:
            path = save_path
            if path is None:
                self._temporary_embedding_path = make_temp_embedding_path()
                path = self._temporary_embedding_path
            image_embeddings = precompute_image_embeddings(
                self._video_predictor, volume, save_path=path, ndim=3, verbose=verbose,
                lazy_loading=lazy_embeddings,
            )

        UniSAM2InstanceSegmentation.initialize(self, volume, ndim=3, image_embeddings=image_embeddings, **kwargs)
        self._image_embeddings = image_embeddings
        self._volume = volume
        self._offload_to_cpu = offload_to_cpu
        self._max_cached_frames = int(volume.shape[0]) if cache_all_slices else None
        self._propagator = self._build_propagator(volume, image_embeddings)

    def get_state(self) -> dict:
        """Return the decoder predictions and the image embeddings, so that both branches can be restored.

        `generate` also prompts the interactive branch, which needs the encoding of the same input. For
        a volume the volume itself is part of the state, because the video predictor is initialized on it.
        """
        state = super().get_state()
        state["image_embeddings"] = self._image_embeddings
        if self._volume is not None:
            state["volume"] = self._volume
        return state

    def set_state(self, state: dict) -> None:
        """Restore the decoder predictions and the encoding of the input they belong to.

        The state must hold either 'image_embeddings' or 'image'. Without one, `generate` would prompt
        whatever image the predictor still holds. A volumetric state must also hold 'volume'.

        Args:
            state: The state, as returned by `get_state`, or a dict with 'prediction' and 'image'.
        """
        image_embeddings = state.get("image_embeddings")
        volume = state.get("volume")
        i = state.get("i")

        if volume is not None:
            if image_embeddings is None:
                raise ValueError("A volumetric state must hold the 'image_embeddings' of its volume.")
            super().set_state(state)
            self._volume = volume
            self._propagator = self._build_propagator(volume, image_embeddings)
            self._i = None
        else:
            if image_embeddings is None:
                if "image" not in state:
                    raise ValueError("The state must hold either 'image_embeddings' or 'image'.")
                image_embeddings = self._encode(state["image"])
            else:
                image_embeddings = self._prepare_image_embeddings(image_embeddings, i)
            super().set_state(state)
            self._i = None
        self._image_embeddings = image_embeddings

    def clear_state(self) -> None:
        """Clear the decoder predictions and the input that is set on the predictor."""
        super().clear_state()
        self._image_embeddings = None
        self._i = None
        self._predictor.reset_predictor()
        self._volume = None
        if self._propagator is not None:
            self._propagator.reset_predictor()
            self._propagator = None
        if self._temporary_embedding_path is not None:
            shutil.rmtree(self._temporary_embedding_path, ignore_errors=True)
            self._temporary_embedding_path = None

    @torch.no_grad()
    def generate(
        self,
        candidate_threshold: Optional[Union[float, Sequence[float]]] = None,
        foreground_threshold: float = DEFAULT_PROMPT_GENERATION["foreground_threshold"],
        n_iter: int = DEFAULT_PROMPT_GENERATION["n_iter"],
        dt: float = DEFAULT_PROMPT_GENERATION["dt"],
        sigma: float = DEFAULT_PROMPT_GENERATION["sigma"],
        spacing: Optional[tuple] = None,
        min_candidate_size: int = DEFAULT_PROMPT_GENERATION["min_candidate_size"],
        score_threshold: float = DEFAULT_PROMPT_GENERATION["score_threshold"],
        max_overlap: float = DEFAULT_PROMPT_GENERATION["max_overlap"],
        min_size: int = DEFAULT_PROMPT_GENERATION["min_size"],
        refine_with_box_prompts: bool = DEFAULT_PROMPT_GENERATION["refine_with_box_prompts"],
        box_extension: int = DEFAULT_PROMPT_GENERATION["box_extension"],
        multimasking: bool = DEFAULT_PROMPT_GENERATION["multimasking"],
        n_objects_per_pass: int = DEFAULT_PROMPT_GENERATION["n_objects_per_pass"],
        early_stop_patience: Optional[int] = DEFAULT_PROMPT_GENERATION["early_stop_patience"],
        batch_size: int = 64,
        n_threads: int = DEFAULT_PROMPT_GENERATION["n_threads"],
        verbose: bool = False,
    ) -> np.ndarray:
        """Derive prompts from the stored predictions, apply them and merge the masks.

        Args:
            candidate_threshold: Density threshold for proposing candidates, or several of them for
                a volume. By default 1.5 for an image and (1.5, 10.0) for a volume, see
                `derive_volume_prompts`.
            foreground_threshold: Foreground binarisation threshold. Here it only limits which pixels can
                be proposed from, since the masks come from the interactive branch, so it trades candidate
                recall rather than boundary quality.
            n_iter: Number of flow-integration steps for the candidate density.
            dt: Integration step size. Mostly only the product with 'n_iter' matters, which is the
                distance a pixel is advected.
            sigma: Gaussian sigma for smoothing the candidate density. Less smoothing leaves more peaks,
                which costs precision for the sparse post-processing but buys candidate recall here.
            spacing: Anisotropic voxel spacing of a volume, e.g. (4, 1, 1).
            min_candidate_size: Discard density components smaller than this.
            score_threshold: Discard candidates whose predicted IoU is below this.
            max_overlap: Reject a candidate when more than this fraction of it is already claimed. For a
                volume this applies on the slice a candidate is prompted on and again on the 3d merge.
            min_size: Minimum object size in the result.
            refine_with_box_prompts: Whether to re-prompt every merged instance with its bounding box,
                see `refine_with_boxes`. Images only.
            box_extension: Number of pixels every box of that second stage is grown by.
            multimasking: Whether to predict several masks per point and keep the best scoring one. A
                single point is ambiguous between one object and a cluster, so this is on by default.
            n_objects_per_pass: Number of objects propagated together through a volume. The video
                predictor runs them as one batch, so this trades device memory against the pass count.
            early_stop_patience: Stop a propagation pass after this many consecutive slices in which
                every object of the pass is empty. None propagates through the whole volume.
            batch_size: Number of prompts per forward pass.
            n_threads: Number of threads for the flow integration the candidates come from.
            verbose: Whether to show progress over the propagation passes of a volume.

        Returns:
            The instance segmentation, uint32 array with the spatial shape of the prediction.
        """
        if not self._is_initialized:
            raise RuntimeError("The segmenter has not been initialized. Call 'initialize' first.")

        shape = self._prediction[0].shape
        # The prediction carries the dimensionality it was run at: (4, Y, X) or (4, Z, Y, X).
        is_volume = self._prediction.ndim == 4
        if candidate_threshold is None:
            candidate_threshold = DEFAULT_PROMPT_GENERATION[
                "candidate_threshold_3d" if is_volume else "candidate_threshold"
            ]
        if is_volume:
            prompts = derive_volume_prompts(
                self._prediction[0], self._prediction[1:], candidate_threshold=candidate_threshold,
                foreground_threshold=foreground_threshold, n_iter=n_iter, dt=dt, sigma=sigma,
                spacing=spacing, min_candidate_size=min_candidate_size, n_threads=n_threads,
            )
            if prompts is None:
                return np.zeros(shape, dtype="uint32")
            candidates = self._score_candidates(
                prompts, multimasking=multimasking, batch_size=batch_size,
                score_threshold=score_threshold, max_overlap=max_overlap,
            )
            records = self._propagate_candidates(
                candidates, n_objects_per_pass=n_objects_per_pass,
                early_stop_patience=early_stop_patience, verbose=verbose,
            )
            return merge_by_score(records, shape, max_overlap=max_overlap, min_size=min_size)

        proposals = self.propose(
            candidate_threshold=candidate_threshold, foreground_threshold=foreground_threshold,
            n_iter=n_iter, dt=dt, sigma=sigma, min_candidate_size=min_candidate_size,
            multimasking=multimasking, batch_size=batch_size, n_threads=n_threads,
        )
        return self.select(
            proposals, score_threshold=score_threshold, max_overlap=max_overlap, min_size=min_size,
            refine_with_box_prompts=refine_with_box_prompts, box_extension=box_extension, batch_size=batch_size,
        )

    @torch.no_grad()
    def propose(
        self,
        candidate_threshold: Optional[float] = DEFAULT_PROMPT_GENERATION["candidate_threshold"],
        foreground_threshold: float = DEFAULT_PROMPT_GENERATION["foreground_threshold"],
        n_iter: int = DEFAULT_PROMPT_GENERATION["n_iter"],
        dt: float = DEFAULT_PROMPT_GENERATION["dt"],
        sigma: float = DEFAULT_PROMPT_GENERATION["sigma"],
        min_candidate_size: int = DEFAULT_PROMPT_GENERATION["min_candidate_size"],
        multimasking: bool = DEFAULT_PROMPT_GENERATION["multimasking"],
        batch_size: int = 64,
        n_threads: int = DEFAULT_PROMPT_GENERATION["n_threads"],
    ) -> list:
        """Derive the prompts and turn them into scored mask proposals, without selecting any of them.

        This is the half of `generate` that needs the model. Splitting it off lets a parameter sweep
        reuse one set of proposals across every value of 'score_threshold', 'max_overlap' and
        'min_size', which only filter and merge. Images only: a volume gates its propagation on the
        score, so its proposals depend on those parameters too.

        Args:
            candidate_threshold: Density threshold for proposing candidates, see `derive_point_prompts`.
            foreground_threshold: Foreground binarisation threshold.
            n_iter: Number of flow-integration steps for the candidate density.
            dt: Integration step size.
            sigma: Gaussian sigma for smoothing the candidate density.
            min_candidate_size: Discard density components smaller than this.
            multimasking: Whether to predict several masks per point and keep the best scoring one.
            batch_size: Number of prompts per forward pass.
            n_threads: Number of threads for the flow integration the candidates come from.

        Returns:
            The proposals, to be passed to `select`. Their layout is an implementation detail of the
            generator that produced them.
        """
        if not self._is_initialized:
            raise RuntimeError("The segmenter has not been initialized. Call 'initialize' first.")
        if self._prediction.ndim == 4:
            raise ValueError("Proposals can only be reused for an image, because a volume gates its propagation.")

        prompts = derive_point_prompts(
            self._prediction[0], self._prediction[1:], candidate_threshold=candidate_threshold,
            foreground_threshold=foreground_threshold, n_iter=n_iter, dt=dt, sigma=sigma,
            min_candidate_size=min_candidate_size, n_threads=n_threads,
        )
        if prompts is None:
            return []
        return self._apply(prompts, multimasking=multimasking, batch_size=batch_size)

    def select(
        self,
        proposals: list,
        score_threshold: float = DEFAULT_PROMPT_GENERATION["score_threshold"],
        max_overlap: float = DEFAULT_PROMPT_GENERATION["max_overlap"],
        min_size: int = DEFAULT_PROMPT_GENERATION["min_size"],
        refine_with_box_prompts: bool = DEFAULT_PROMPT_GENERATION["refine_with_box_prompts"],
        box_extension: int = DEFAULT_PROMPT_GENERATION["box_extension"],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Merge the proposals of `propose` into an instance segmentation.

        Args:
            proposals: The proposals, as returned by `propose` on the same generator.
            score_threshold: Discard proposals whose predicted IoU is below this.
            max_overlap: Reject a proposal when more than this fraction of it is already claimed.
            min_size: Minimum object size in the result.
            refine_with_box_prompts: Whether to re-prompt every merged instance with its bounding box.
            box_extension: Number of pixels every box of that second stage is grown by.
            batch_size: Number of boxes per forward pass of the refinement.

        Returns:
            The instance segmentation, uint32 array with the spatial shape of the prediction.
        """
        shape = self._prediction[0].shape
        if not proposals:
            return np.zeros(shape, dtype="uint32")

        segmentation = self._merge(
            proposals, shape, score_threshold=score_threshold, max_overlap=max_overlap, min_size=min_size,
        )
        if refine_with_box_prompts and segmentation.max() > 0:
            segmentation = self._refine_boxes(segmentation, batch_size, box_extension)
        return segmentation

    def _apply(self, prompts: dict, multimasking: bool, batch_size: int) -> list:
        """Turn the prompts into mask proposals."""
        return self._apply_prompts(prompts, multimasking=multimasking, batch_size=batch_size)

    def _merge(
        self, proposals: list, shape: tuple, score_threshold: float, max_overlap: float, min_size: int,
    ) -> np.ndarray:
        """Merge the mask proposals into an instance segmentation."""
        records = [record for record in proposals if record["predicted_iou"] >= score_threshold]
        if not records:
            return np.zeros(shape, dtype="uint32")
        return merge_by_score(records, shape, max_overlap=max_overlap, min_size=min_size)

    def _refine_boxes(self, segmentation: np.ndarray, batch_size: int, box_extension: int) -> np.ndarray:
        """Re-prompt every instance with its bounding box, see `refine_with_boxes`."""
        with sam2_autocast(self._predictor.device):
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
            n_prompts = len(batch_points)
            # Reduced on the device, so only the kept mask is transferred rather than every proposal.
            mask_input, coords, labels, _ = self._predictor._prep_prompts(
                batch_points, batch_labels, None, None, True,
            )
            with sam2_autocast(self._predictor.device):
                logits, scores, _ = self._predictor._predict(
                    coords, labels, None, mask_input, multimasking, return_logits=True,
                )
            logits = logits.reshape(n_prompts, -1, *logits.shape[-2:])
            scores = scores.reshape(n_prompts, -1)
            index = torch.arange(n_prompts, device=scores.device)
            best = scores.argmax(dim=1)
            logits, scores = logits[index, best], scores[index, best]

            stability = calculate_stability_score(logits, mask_threshold, STABILITY_SCORE_OFFSET)
            binary = logits > mask_threshold
            # Two reductions on the device: an np.nonzero per mask costs more than the rest of the loop.
            rows_any = binary.any(dim=2).cpu().numpy()
            columns_any = binary.any(dim=1).cpu().numpy()
            masks = binary.cpu().numpy()
            scores = scores.float().cpu().numpy()
            stability = stability.float().cpu().numpy()
            for offset, (mask, row_any, column_any, score, stable) in enumerate(
                zip(masks, rows_any, columns_any, scores, stability)
            ):
                if not row_any.any():
                    continue
                y0, y1 = int(row_any.argmax()), len(row_any) - int(row_any[::-1].argmax())
                x0, x1 = int(column_any.argmax()), len(column_any) - int(column_any[::-1].argmax())
                records.append({
                    # The crop rather than the full mask: the merge is linear in the mask's size.
                    "segmentation": mask[y0:y1, x0:x1].copy(),
                    "bounding_box": (slice(y0, y1), slice(x0, x1)),
                    "predicted_iou": float(score),
                    "stability_score": float(stable),
                    # Empty masks are dropped, so the record order does not track the prompts.
                    "prompt_index": start + offset,
                })
        return records

    def _score_candidates(
        self, prompts: dict, multimasking: bool, batch_size: int, score_threshold: float,
        max_overlap: float,
    ) -> List[dict]:
        """Prompt every candidate in 2d on its anchor slice, and keep the strong, non-duplicate ones.

        This runs before the propagation, which is where the cost is: a candidate the model scores
        poorly, or that a better-scoring one already covers on that slice, is never propagated. It
        also gives every surviving candidate the predicted IoU that orders the volumetric merge.

        Returns:
            The surviving candidates, each with the prompt it will be propagated with.
        """
        points, point_labels, frames = prompts["points"], prompts["point_labels"], prompts["frames"]
        candidates = []
        for frame in np.unique(frames):
            indices = np.where(frames == frame)[0]
            # Reads the slice's features out of the volume's embeddings, so nothing is re-encoded.
            _set_image_predictor_from_3d_embeddings(self._predictor, self._image_embeddings, int(frame))
            records = self._apply_prompts(
                {"points": points[indices], "point_labels": point_labels[indices]},
                multimasking=multimasking, batch_size=batch_size,
            )
            records = [record for record in records if record["predicted_iou"] >= score_threshold]
            if not records:
                continue

            _, kept = merge_by_score(
                records, _records_shape(records), max_overlap=max_overlap,
                min_size=DEFAULT_PROMPT_GENERATION["min_size"], return_matches=True,
            )
            for record_index in kept.values():
                record = records[record_index]
                x, y = points[indices[record["prompt_index"]], 0]
                candidates.append({
                    "frame": int(frame),
                    "point": (float(x), float(y)),
                    "score": record["predicted_iou"],
                    "stability": record["stability_score"],
                })
        return candidates

    def _propagate_candidates(
        self, candidates: List[dict], n_objects_per_pass: int, early_stop_patience: Optional[int],
        verbose: bool,
    ) -> List[Dict[str, Any]]:
        """Turn every candidate into a volumetric mask by propagating its prompts through the volume.

        Every object of a pass is anchored on the same slice, because the video predictor propagates
        them together from the earliest slice any of them is conditioned on: an object anchored later
        than that would be tracked before it is conditioned, and never propagated backwards at all.
        """
        by_anchor = {}
        for candidate in candidates:
            by_anchor.setdefault(candidate["frame"], []).append(candidate)
        passes = [
            group[start:start + n_objects_per_pass]
            for _, group in sorted(by_anchor.items())
            for start in range(0, len(group), n_objects_per_pass)
        ]

        records = []
        for batch in tqdm(passes, desc="Propagate prompts", disable=not verbose):
            # Only the objects: re-reading the volume's features each pass costs more than the propagation.
            self._propagator.reset_tracking()
            for object_id, candidate in enumerate(batch, start=1):
                x, y = candidate["point"]
                self._propagator.add_point_prompts(
                    frame_ids=candidate["frame"],
                    points=np.array([[y, x]], dtype="float32"),  # The propagator takes YX.
                    point_labels=np.array([1], dtype="int32"),
                    object_id=object_id,
                )
            video_segments = self._propagator.propagate_prompts(early_stop_patience=early_stop_patience)
            records.extend(_volume_records(video_segments, batch, self._volume.shape))

        self._propagator.reset_predictor()
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

    def initialize(
        self,
        image: np.ndarray,
        ndim: int = 2,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
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
            i: The slice index for tiled video-style embeddings. By default the embeddings contain one image.
            tile_shape: The tile shape, (y, x). Required when no embeddings are given.
            halo: The overlap between the tiles, (y, x). Required when no embeddings are given.
            save_path: Optional path to cache the computed embeddings in a zarr container. Without one
                an ephemeral store is used, which `clear_state` removes.
            verbose: Whether to print progress while the embeddings are computed.
            kwargs: Additional arguments for `TiledUniSAM2InstanceSegmentation.initialize`.
        """
        if ndim != 2:
            raise ValueError(f"Tiled prompt generation supports 2d images only, got ndim={ndim}.")
        if image_embeddings is None and (tile_shape is None or halo is None):
            raise ValueError("Both 'tile_shape' and 'halo' have to be passed for the tiled generator.")

        if self._temporary_embedding_path is not None:
            self.clear_state()

        if image_embeddings is None:
            path = save_path
            if path is None:
                self._temporary_embedding_path = make_temp_embedding_path()
                path = self._temporary_embedding_path
            image_embeddings = precompute_image_embeddings(
                self._predictor, image, save_path=path, ndim=2, tile_shape=tile_shape, halo=halo,
                verbose=verbose, lazy_loading=True,
            )

        TiledUniSAM2InstanceSegmentation.initialize(
            self, image, ndim=2, image_embeddings=image_embeddings, i=i, **kwargs
        )
        self._image_embeddings = image_embeddings
        self._i = i

        # From the embeddings, not the arguments, so the prompting cannot disagree with the encoding.
        features = image_embeddings["features"]
        self._tiling = Blocking(
            [0, 0], [int(s) for s in features.attrs["shape"][-2:]],
            [int(s) for s in features.attrs["tile_shape"]],
        )
        self._halo = [int(s) for s in features.attrs["halo"]]

    def _set_tile_embeddings(self, tile_id: int) -> None:
        """Set the embeddings for one tile on the image predictor."""
        if self._i is None:
            set_precomputed(self._predictor, self._image_embeddings, tile_id=tile_id)
            return

        name = str(tile_id)
        features = self._image_embeddings["features"][name]
        fpn_group = self._image_embeddings["fpn"][name]
        pos_enc_group = self._image_embeddings["pos_enc"][name]
        fpn = [fpn_group[str(level)] for level in range(len(fpn_group))]
        pos_enc = [pos_enc_group[str(level)] for level in range(len(pos_enc_group))]
        _set_image_predictor_from_backbone(
            self._predictor, fpn, pos_enc, features, features.attrs["original_size"], self._i,
        )

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

    def _apply(self, prompts: dict, multimasking: bool, batch_size: int) -> list:
        """Prompt each tile with the candidates that belong to it, keeping the tiles apart."""
        points, point_labels = prompts["points"], prompts["point_labels"]

        proposals = []
        for tile_id, indices in sorted(self._tiles_for_points(points).items()):
            bounding_box = self._tile_bounding_box(tile_id)
            # The prompts are in the full image's frame, the tile's embeddings in the tile's.
            origin = np.array([bounding_box[1].start, bounding_box[0].start], dtype="float32")

            self._set_tile_embeddings(tile_id)
            records = self._apply_prompts(
                {"points": points[indices] - origin, "point_labels": point_labels[indices]},
                multimasking=multimasking, batch_size=batch_size,
            )
            if records:
                proposals.append({"bounding_box": bounding_box, "records": records})
        return proposals

    def _merge(
        self, proposals: list, shape: tuple, score_threshold: float, max_overlap: float, min_size: int,
    ) -> np.ndarray:
        """Stitch the per-tile merges into one segmentation, resolving the halo overlaps."""
        segmentation = np.zeros(shape, dtype="uint32")
        offset = 0

        for proposal in proposals:
            bounding_box = proposal["bounding_box"]
            tile_shape = tuple(box.stop - box.start for box in bounding_box)
            records = [record for record in proposal["records"] if record["predicted_iou"] >= score_threshold]
            if not records:
                continue

            tile_segmentation = merge_by_score(records, tile_shape, max_overlap=max_overlap, min_size=min_size)
            max_id = int(tile_segmentation.max())
            if max_id == 0:
                continue
            # Keep the instance ids unique across tiles before the halo overlaps are resolved.
            tile_segmentation[tile_segmentation != 0] += offset
            offset += max_id
            segmentation[bounding_box] = _merge_segmentations(tile_segmentation, segmentation[bounding_box])
        return segmentation

    def _refine_boxes(self, segmentation: np.ndarray, batch_size: int, box_extension: int) -> np.ndarray:
        """Re-prompt every instance with its bounding box, in the tile that holds its interior point.

        Refined once, by the tile whose inner block holds its point, so two tiles cannot both claim it.
        """
        ids = np.unique(segmentation)
        ids = ids[ids != 0]
        if ids.size == 0:
            return segmentation

        centers = interior_points(segmentation)
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

            self._set_tile_embeddings(tile_id)
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
