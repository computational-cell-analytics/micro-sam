import gc
import queue
import ctypes
import hashlib
import platform
from copy import copy
from concurrent import futures
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import torch

from micro_sam.v2.util import Devices
from micro_sam.util import device_type
from micro_sam.v1.prompt_based_segmentation import _process_box, _compute_logits_from_mask
from micro_sam.v2.transforms.resize import resize_longest_side_and_pad_spatial_numpy, ResizeLongestSideTransforms


def _trim_cpu_heap():
    """Return freed CPU heap to the OS on glibc-based Linux. It does nothing elsewhere.

    'gc.collect' drops the Python references and frees the tensors, but glibc keeps the freed pages
    in its arenas rather than returning them to the OS, so RSS stays high on native CPU systems (the
    low-RAM target). 'malloc_trim' releases that free heap, so clearing the embedding cache actually
    lowers RSS.
    """
    if platform.system() != "Linux":
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _free_device_memory():
    """Return freed tensor memory to the allocator / OS after clearing cached embeddings.

    Covers GPU (empty the CUDA / MPS caching allocator) and native CPU systems (trim the glibc heap
    so RSS actually drops, not just the Python-level references).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    _trim_cpu_heap()


def _crop_to_original_shape(mask, shape):
    """Crop a SAM2 video-predictor mask back to the original slice shape.

    The video predictor pads non-square frames to a square of side max(H, W) (padding appended at the
    bottom/right) and returns masks at that padded size. The image content occupies the top-left
    [0:H, 0:W] region, so cropping recovers the original (H, W) mask. For square volumes this is a no-op.
    """
    return mask[:shape[0], :shape[1]]


def _tile_index_for(tiling, halo, y, x):
    """Return the id of the tile whose inner (halo-free) block contains the point (y, x)."""
    for tile_id in range(tiling.number_of_blocks):
        inner = tiling.get_block_with_halo(tile_id, list(halo)).inner_block
        if inner.begin[0] <= y < inner.end[0] and inner.begin[1] <= x < inner.end[1]:
            return tile_id
    return 0


def _inner_block_slices(tiling, halo, tile_id):
    """Return the (local, global) inner-block slices for placing a tile result into the full array."""
    block = tiling.get_block_with_halo(tile_id, list(halo))
    local = tuple(slice(b, e) for b, e in zip(block.inner_block_local.begin, block.inner_block_local.end))
    glob = tuple(slice(b, e) for b, e in zip(block.inner_block.begin, block.inner_block.end))
    return local, glob


def _box_to_tiles(tiling, halo, box):
    """Assign a box to every tile whose inner block it overlaps, clipped to the tile's outer block.

    A single box can span several tiles (e.g. at a 4-tile junction); each tile segments the box's
    portion that falls in it and the results are unioned. The box and the returned clipped boxes are
    in (y0, x0, y1, x1) order.

    Returns:
        Mapping from tile id to the clipped box (in global coordinates).
    """
    y0, x0, y1, x1 = box
    assignments = {}
    for tile_id in range(tiling.number_of_blocks):
        block = tiling.get_block_with_halo(tile_id, list(halo))
        inner, outer = block.inner_block, block.outer_block
        # Skip tiles whose inner (halo-free) block the box does not overlap.
        if y1 <= inner.begin[0] or y0 >= inner.end[0] or x1 <= inner.begin[1] or x0 >= inner.end[1]:
            continue
        assignments[tile_id] = np.array([
            max(y0, outer.begin[0]), max(x0, outer.begin[1]),
            min(y1, outer.end[0]), min(x1, outer.end[1]),
        ])
    return assignments


def _crop_mask_to_tile(tiling, halo, tile_id, mask):
    """Crop a full-plane 2d mask to a tile's outer (halo-included) block, so it stays aligned with
    the tile's sub-volume."""
    outer = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
    y0, x0 = int(outer.begin[0]), int(outer.begin[1])
    y1, x1 = int(outer.end[0]), int(outer.end[1])
    return np.asarray(mask)[y0:y1, x0:x1]


def _validate_pre_refined_masks(volume_shape, frame_ids, masks, object_id):
    """Validate full-resolution masks before they are added to the persistent video state."""
    if masks is None:
        return []
    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        masks = [masks]
    else:
        masks = list(masks)
    if len(masks) == 0:
        return []

    def broadcast_ids(value, default, name):
        value = default if value is None else value
        if isinstance(value, (list, tuple, np.ndarray)):
            values = np.asarray(value)
            if values.ndim == 0:
                values = [values.item()]
            elif values.ndim == 1:
                values = values.tolist()
            else:
                raise ValueError(f"'{name}' must be a scalar or one-dimensional sequence.")
            if len(values) == 1:
                values = values * len(masks)
            if len(values) != len(masks):
                raise ValueError(f"Expected 1 or {len(masks)} {name}, got {len(values)}.")
        else:
            values = [value] * len(masks)

        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values):
            raise TypeError(f"'{name}' must contain integer IDs.")
        return [int(value) for value in values]

    frame_ids = broadcast_ids(frame_ids, None, "frame_ids")
    object_ids = broadcast_ids(object_id, 1, "object_id")
    depth, height, width = (int(size) for size in volume_shape)

    prepared = []
    keys = set()
    for frame_id, mask, obj_id in zip(frame_ids, masks, object_ids):
        if not 0 <= frame_id < depth:
            raise ValueError(f"Frame ID {frame_id} is outside the valid range [0, {depth - 1}].")
        if obj_id <= 0:
            raise ValueError(f"Object ID must be positive, got {obj_id}.")

        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.shape != (height, width):
            raise ValueError(
                f"Each pre-refined mask must have shape {(height, width)}, got {mask.shape}."
            )
        if not np.issubdtype(mask.dtype, np.bool_) and not np.issubdtype(mask.dtype, np.number):
            raise TypeError("Pre-refined masks must have a boolean or numeric dtype.")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("Pre-refined masks must be binary (contain only 0 and 1).")
        mask = mask.astype(bool, copy=False)
        if not mask.any():
            raise ValueError("Pre-refined masks must contain at least one foreground pixel.")

        key = (obj_id, frame_id)
        if key in keys:
            raise ValueError(
                f"Only one pre-refined mask can be supplied for object {obj_id} on frame {frame_id}."
            )
        keys.add(key)
        prepared.append((frame_id, mask, obj_id))

    return prepared


def _clone_image_predictor(predictor, model: torch.nn.Module):
    """Copy an image predictor onto a replica model, with per-image state cleared."""
    from micro_sam.v2.util import configure_image_predictor

    replica = copy(predictor)
    replica.__dict__.pop("_micro_sam_predictor_devices", None)  # do not share the source cache
    replica.model = model
    replica.reset_predictor()
    return configure_image_predictor(replica)


def _get_image_predictor_devices(predictor, devices: Devices) -> List[Tuple]:
    """Create and cache one image-predictor replica per selected device."""
    from micro_sam.v2.batched_inference import _prepare_models, _resolve_devices

    resolved_devices = _resolve_devices(predictor.model, devices)
    cache_key = tuple(str(device) for device in resolved_devices)
    cached = getattr(predictor, "_micro_sam_predictor_devices", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    predictor_devices = [
        (predictor if model is predictor.model else _clone_image_predictor(predictor, model), device)
        for model, device in _prepare_models(predictor.model, resolved_devices)
    ]
    predictor._micro_sam_predictor_devices = (cache_key, predictor_devices)
    return predictor_devices


def promptable_segmentation_2d(
    predictor,
    image: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    boxes: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    batched: Optional[bool] = None,
):
    """@private"""
    from micro_sam.v2.util import configure_image_predictor
    configure_image_predictor(predictor)

    if image is not None:
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            # Make channel-first RGB images channel-last. Grayscale and channel-last inputs are
            # handled by 'to_image' below.
            image = image.transpose(1, 2, 0)

        # Set the predictor state.
        from micro_sam.v2.normalization import to_image
        predictor.set_image(to_image(image))

    have_points = points is not None and len(points) > 0
    have_boxes = boxes is not None and len(boxes) > 0
    if have_points:
        assert len(points) == len(labels)

    # If no prompts are provided, return 'None'.
    if not have_points and not have_boxes:
        return

    # Batched multi-object segmentation: each positive point and each box defines a separate object.
    if batched:
        return _batched_promptable_segmentation_2d(predictor, points, labels, boxes, masks)

    # SAM2 concatenates boxes and points along the prompt axis, so several boxes cannot share one
    # point batch. Reject the combination, as in SAM v1, instead of failing inside the predictor.
    if have_points and have_boxes and len(boxes) > 1:
        print("Point prompts can only be combined with a single box/shape prompt. Skipping segmentation.")
        return None

    # A napari polygon or ellipse yields a filled mask prompt (from 'shape_layer_to_prompts'); when
    # any is present, route to the mask-aware path that feeds them to SAM2 as low-res logit prompts.
    if masks is not None and any(m is not None for m in masks):
        return _promptable_segmentation_2d_with_masks(predictor, points, labels, boxes, masks)

    kwargs = {}
    if have_points:
        kwargs["point_coords"] = points[:, ::-1].copy()  # Ensure contiguous array convention so that PyTorch likes it.
        kwargs["point_labels"] = labels
    if have_boxes:
        shape = predictor._orig_hw[0]
        kwargs["box"] = np.array([_process_box(b, shape) for b in boxes])

    # Run interactive segmentation.
    masks, scores, logits = predictor.predict(
        multimask_output=False,  # NOTE: Hard-coded to 'False' atm.
        **kwargs
    )

    # Get the count of points / boxes.
    n_points = len(points) if have_points else 0
    n_boxes = len(boxes) if have_boxes else 0

    if n_points > 1 or n_boxes > 1:  # Has more than one object, expected instance segmentation.
        out = np.zeros(masks.shape[-2:])
        for i, curr_mask in enumerate(masks, start=1):
            out[curr_mask.squeeze() > 0] = i
    else:
        out = masks.squeeze()

    # HACK: Hard-code the expected data type for labels for napari labels layer: uint8
    out = out.astype("uint8")

    return out


def _promptable_segmentation_2d_with_masks(predictor, points, labels, boxes, masks):
    """Single-image promptable segmentation that uses polygon/ellipse mask prompts.

    A napari polygon or ellipse yields both a bounding box and a filled mask (from
    'shape_layer_to_prompts'); the mask is passed to SAM2 as a low-res logit prompt ('mask_input')
    alongside its box, mirroring the SAM v1 behaviour. Objects are segmented one at a time (as in v1)
    so a mask prompt only conditions its own object.
    """
    shape = predictor._orig_hw[0]
    have_points = points is not None and len(points) > 0
    have_boxes = boxes is not None and len(boxes) > 0

    def _predict_one(box=None, mask=None, extra_points=None, extra_labels=None):
        kwargs = {"multimask_output": False}
        if box is not None:
            kwargs["box"] = np.array([_process_box(box, shape)])
        if mask is not None:
            kwargs["mask_input"] = _compute_logits_from_mask(mask)
        if extra_points is not None and len(extra_points) > 0:
            kwargs["point_coords"] = extra_points[:, ::-1].copy()
            kwargs["point_labels"] = extra_labels
        out_masks, _, _ = predictor.predict(**kwargs)
        return out_masks.squeeze()

    # Points combined with a single shape (v1 convention: only one box allowed alongside points,
    # which the caller has already checked).
    if have_points and have_boxes:
        seg = _predict_one(box=boxes[0], mask=masks[0], extra_points=points, extra_labels=labels)
        return (seg > 0).astype("uint8")

    # One object per shape (each box, with its mask if it is a polygon/ellipse).
    out = np.zeros(tuple(shape), dtype="uint8")
    for seg_id, (box, mask) in enumerate(zip(boxes, masks), 1):
        seg = _predict_one(box=box, mask=mask)
        out[seg > 0] = seg_id
    return out


def _batched_promptable_segmentation_2d(predictor, points, labels, boxes, masks=None):
    """Batched 2D segmentation where each positive point and each box defines a separate object.

    Negative points are shared as negative prompts for every object. A box that comes from a
    polygon/ellipse (its entry in 'masks' is not None) also carries a soft mask-logit cue. This
    matches the batched convention of the SAM v1 annotator and the 3D unified segment widget.
    """
    shape = predictor._orig_hw[0]

    points = np.zeros((0, 2)) if points is None else np.asarray(points)
    labels = np.zeros((0,), dtype=int) if labels is None else np.asarray(labels)
    positive_points = points[labels == 1]
    negative_points = points[labels != 1]
    n_neg = len(negative_points)

    seg = np.zeros(tuple(shape), dtype="uint32")
    object_id = 0

    def _assign(masks):
        nonlocal object_id
        masks = np.asarray(masks)
        if masks.ndim == 3:  # A single object is returned as (C, H, W); add the object dimension.
            masks = masks[None]
        for curr_mask in masks:
            object_id += 1
            seg[curr_mask[0] > 0] = object_id

    # One object per positive point, each combined with the shared negative points.
    if len(positive_points) > 0:
        obj_points, obj_labels = [], []
        for pos in positive_points:
            pts = np.concatenate([pos[None], negative_points], axis=0) if n_neg else pos[None]
            lbs = np.concatenate([[1], np.zeros(n_neg, dtype=int)]) if n_neg else np.array([1])
            obj_points.append(pts)
            obj_labels.append(lbs)
        # Reverse the last axis to convert (row, col) to the (x, y) convention SAM2 expects.
        batched_points = np.stack(obj_points)[..., ::-1].copy()
        batched_labels = np.stack(obj_labels)
        # Keep 'masks': the box branch below needs the mask prompts, not this prediction.
        point_masks, _, _ = predictor.predict(
            point_coords=batched_points, point_labels=batched_labels, multimask_output=False,
        )
        _assign(point_masks)

    # One object per box, each combined with the shared negative points.
    if boxes is not None and len(boxes) > 0:
        have_box_masks = masks is not None and any(m is not None for m in masks)
        if have_box_masks:
            # A batched predict takes a single shared 'mask_input', so segment per object to give each
            # box its own soft mask cue (a box without a mask, e.g. a rectangle, uses the box alone).
            for bidx, box in enumerate(boxes):
                kwargs = {"box": np.array([_process_box(box, shape)]), "multimask_output": False}
                if masks[bidx] is not None:
                    kwargs["mask_input"] = _compute_logits_from_mask(np.asarray(masks[bidx]))
                if n_neg:
                    kwargs["point_coords"] = negative_points[None][..., ::-1].copy()
                    kwargs["point_labels"] = np.zeros((1, n_neg), dtype=int)
                out_masks, _, _ = predictor.predict(**kwargs)
                _assign(out_masks)
        else:
            processed_boxes = np.array([_process_box(b, shape) for b in boxes])
            kwargs = {"box": processed_boxes, "multimask_output": False}
            if n_neg:
                neg = np.repeat(negative_points[None], len(processed_boxes), axis=0)[..., ::-1].copy()
                kwargs["point_coords"] = neg
                kwargs["point_labels"] = np.zeros((len(processed_boxes), n_neg), dtype=int)
            out_masks, _, _ = predictor.predict(**kwargs)
            _assign(out_masks)

    if object_id == 0:
        return None

    return seg


def tiled_promptable_segmentation_2d(
    predictor,
    image_embeddings: dict,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    boxes: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    batched: Optional[bool] = None,
    devices: Devices = None,
):
    """Tiled 2d promptable segmentation for the SAM2 image predictor.

    Routes the prompts to the tile-column they fall in, sets that tile's precomputed embeddings on
    the predictor (via `set_precomputed`), runs `promptable_segmentation_2d` on the tile, and
    stitches the per-tile mask into the full image. Points are in (y, x) order, as passed by the
    annotator. Same return convention as `promptable_segmentation_2d`.

    Independent active tiles run concurrently on persistent predictor replicas. By default all
    visible CUDA devices are used; pass `devices` to select or restrict them.
    """
    from bioimage_cpp.utils import Blocking
    from micro_sam.v2.util import set_precomputed

    feats = image_embeddings["features"]
    shape = tuple(int(s) for s in feats.attrs["shape"])
    tile_shape = tuple(int(s) for s in feats.attrs["tile_shape"])
    halo = tuple(int(s) for s in feats.attrs["halo"])
    tiling = Blocking([0, 0], list(shape), list(tile_shape))

    have_points = points is not None and len(points) > 0
    have_boxes = boxes is not None and len(boxes) > 0
    if not have_points and not have_boxes:
        return None

    # Group the prompts by the tile each falls in, so the tool segments an object across every tile
    # that holds its prompts. Points are (y, x). Boxes are (y0, x0, y1, x1). A polygon or ellipse
    # box carries a filled mask prompt. The tool crops it to each tile's outer block so it stays aligned.
    tile_points, tile_labels, tile_boxes, tile_masks = {}, {}, {}, {}
    if have_points:
        for point, label in zip(np.asarray(points), np.asarray(labels)):
            tid = _tile_index_for(tiling, halo, int(round(point[0])), int(round(point[1])))
            tile_points.setdefault(tid, []).append(point)
            tile_labels.setdefault(tid, []).append(label)
    if have_boxes:
        for bidx, box in enumerate(boxes):
            box_mask = masks[bidx] if masks is not None else None
            # A box can span several tiles. Segment its clipped portion in each (boxes are y0,x0,y1,x1).
            for tid, clipped in _box_to_tiles(tiling, halo, np.asarray(box)).items():
                tile_boxes.setdefault(tid, []).append(clipped)
                tile_masks.setdefault(tid, []).append(
                    None if box_mask is None else _crop_mask_to_tile(tiling, halo, tid, box_mask)
                )

    tile_jobs = {}
    for tile_id in sorted(set(tile_points) | set(tile_boxes)):
        tpoints = np.asarray(tile_points.get(tile_id, [])).reshape(-1, 2)
        tlabels = np.asarray(tile_labels.get(tile_id, []), dtype=int)
        tboxes = tile_boxes.get(tile_id, [])
        # Only segment tiles that have a positive cue (a positive point or a box); a tile with only
        # negative points has nothing to segment there.
        if not ((tlabels == 1).any() or len(tboxes) > 0):
            continue

        outer = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
        y0, x0 = int(outer.begin[0]), int(outer.begin[1])
        local_points = (tpoints - np.array([y0, x0])) if len(tpoints) else tpoints
        local_boxes = [b - np.array([y0, x0, y0, x0]) for b in tboxes] if tboxes else None
        local_masks = tile_masks.get(tile_id) if tboxes else None
        tile_jobs[tile_id] = local_points, tlabels, local_boxes, local_masks

    predictor_devices = _get_image_predictor_devices(predictor, devices)
    # Spread the active tiles round-robin. Their ids are sparse, so mapping by 'tile_id % n_devices'
    # would run e.g. the tiles 0, 2, 4 on one device and leave the others idle.
    groups = {}
    for position, tile_id in enumerate(sorted(tile_jobs)):
        worker_id = position % len(predictor_devices)
        groups.setdefault(worker_id, []).append(tile_id)

    def run_group(worker_id, tile_ids):
        local_predictor, _ = predictor_devices[worker_id]
        results = []
        for tile_id in tile_ids:
            local_points, local_labels, local_boxes, local_masks = tile_jobs[tile_id]
            set_precomputed(local_predictor, image_embeddings, tile_id=tile_id)
            tile_seg = promptable_segmentation_2d(
                local_predictor, image=None, points=local_points, labels=local_labels,
                boxes=local_boxes, masks=local_masks, batched=batched,
            )
            results.append((tile_id, tile_seg))
        return results

    if len(groups) < 2:
        results = run_group(*next(iter(groups.items()))) if groups else []
    else:
        results = []
        with futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
            tasks = [pool.submit(run_group, worker_id, tile_ids) for worker_id, tile_ids in groups.items()]
            for task in tasks:
                results.extend(task.result())

    out = np.zeros(shape, dtype="uint32")
    found = False
    for tile_id, tile_seg in sorted(results):
        if tile_seg is None:
            continue

        local, glob = _inner_block_slices(tiling, halo, tile_id)
        region = tile_seg[local]
        # Union the per-tile result into the output, preserving object ids (an object spanning tiles
        # keeps the same id and is merged across the tile boundary).
        sub = out[glob]
        positive = region > 0
        sub[positive] = region[positive]
        found = True

    return out if found else None


class PromptableSegmentation3D:
    """Promptable segmentation class for volumetric data.
    """
    def __init__(
        self, predictor, volume, volume_embeddings, device=None,
        offload_video_to_cpu=None, offload_state_to_cpu=None,
    ):
        from micro_sam.v2.util import _get_device
        self.predictor = predictor
        self.volume = volume
        self.volume_embeddings = volume_embeddings
        # 'device=None' uses the predictor's auto-detected device.
        self.device = device
        # Offloading frames/state to CPU bounds GPU memory for large volumes on CUDA. On MPS it is off
        # by default: unified memory saves nothing, and SAM2's CPU->MPS 'non_blocking' transfer of the
        # consolidated masks races, giving intermittent garbage/NaN masks (patchy interactive results).
        is_mps = device_type(_get_device(device)) == "mps"
        self.offload_video_to_cpu = (not is_mps) if offload_video_to_cpu is None else offload_video_to_cpu
        self.offload_state_to_cpu = (not is_mps) if offload_state_to_cpu is None else offload_state_to_cpu

        if self.volume.ndim != 3:
            raise AssertionError(f"The dimensionality of the volume must be 3, got '{self.volume.ndim}'")

        self.init_predictor()

        # Track prompts already pushed to the persistent SAM2 state, keyed by (object_id, frame_id),
        # so a re-run adds only newly placed prompts on top of the existing state (true incremental
        # refinement) instead of re-adding duplicates. Cleared on 'reset_predictor'.
        self._pushed_points = {}  # (object_id, frame_id) -> set of (y, x, label)
        self._pushed_boxes = {}  # (object_id, frame_id) -> set of box corner tuples
        self._pushed_masks = {}  # (object_id, frame_id) -> set of mask content digests
        self._pushed_pre_refined_masks = {}  # (object_id, frame_id) -> set of mask content digests
        # Signature of the prompt set of the previous round, see 'sync_prompt_state'.
        self._prompt_signatures = set()
        self._image_style_trafo = None  # lazily built resize-longest transform for per-slice mask refinement

    def init_predictor(self):
        # Initialize the inference state.
        self.inference_state = self.predictor.init_state(
            volume=self.volume, volume_embeddings=self.volume_embeddings, device=self.device,
            offload_video_to_cpu=self.offload_video_to_cpu, offload_state_to_cpu=self.offload_state_to_cpu,
        )

    def _clear_pushed_prompts(self):
        # The dedup bookkeeping describes the SAM2 state, so it must never outlive it.
        self._pushed_points = {}
        self._pushed_boxes = {}
        self._pushed_masks = {}
        self._pushed_pre_refined_masks = {}
        self._prompt_signatures = set()

    def sync_prompt_state(self, signatures):
        """Discard the persistent state when prompts were removed or changed since the last round.

        The dedup in 'add_*_prompts' only detects added prompts. Additive refinement (the current
        prompts are a superset of the previous ones) therefore keeps the state and pushes just the
        new prompts, while a removal, relabel or move replays everything from scratch.

        Args:
            signatures: Hashable descriptors of every prompt currently in the annotation layers.
        """
        signatures = set(signatures)
        if not self._prompt_signatures.issubset(signatures):
            self.reset_predictor()
        self._prompt_signatures = signatures

    def reset_predictor(self):
        # Reset the state after finishing the segmentation round.
        self.predictor.reset_state(self.inference_state)
        self._clear_pushed_prompts()
        # Drop the per-frame embedding cache (up to MAX_CACHED_FRAMES slices of high-res features) so
        # committing / clearing frees its RAM. SAM2's 'reset_state' clears the tracking outputs but not
        # this cache. The embeddings are disk-backed, so the next prompt re-reads the needed frame
        # lazily via '_get_image_feature' - the cache stays empty until then.
        self.inference_state["cached_features"] = {}
        _free_device_memory()

    def get_progress_total(self, z_range=None):
        """Return the number of slice propagation steps for the requested z range."""
        if z_range is None:
            return int(self.volume.shape[0])
        return int(z_range[1] - z_range[0] + 1)

    def _broadcast(self, value, n):
        """Broadcast a scalar frame/object id to a length-'n' list (or validate a length-1 or -'n'
        sequence). Guards against the earlier bug where a single frame id zipped against several
        points silently dropped all but the first point."""
        if isinstance(value, (list, tuple, np.ndarray)):
            values = [int(v) for v in value]
            if len(values) == 1:
                values = values * n
            if len(values) != n:
                raise ValueError(f"Expected 1 or {n} ids, got {len(values)}.")
            return values
        return [int(value)] * n

    def add_point_prompts(
        self,
        frame_ids: Union[int, List[int]],
        points: np.ndarray,
        point_labels: np.ndarray,
        object_id: Optional[Union[List[int], int]] = None,
        multiple_objects: bool = False,  # Enables multi-object segmentation.
    ):
        """Add point prompts (in (y, x) order) to the persistent SAM2 state, one object at a time.

        Several points can be added in a single call. A point already pushed in an earlier call (same
        object, frame, rounded position and label) is skipped, so re-running with the full prompt set
        only pushes newly placed points. New points are appended ('clear_old_points=False') so they
        correct the running segmentation instead of replacing it.
        """
        if multiple_objects and object_id is not None:
            raise ValueError("Cannot enable multi-object segmentation and also pass a fixed object id.")
        if points is None or len(points) == 0:
            return

        points = np.asarray(points)
        point_labels = np.asarray(point_labels)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("'points' must have shape (N, 2).")
        if len(points) != len(point_labels):
            raise AssertionError("The number of points and labels do not match.")

        n = len(points)
        frame_ids = self._broadcast(frame_ids, n)
        object_ids = self._broadcast(1 if object_id is None else object_id, n)

        for frame_id, (y, x), label, obj_id in zip(frame_ids, points, point_labels, object_ids):
            signature = (int(round(float(y))), int(round(float(x))), int(label))
            seen = self._pushed_points.setdefault((obj_id, frame_id), set())
            if signature in seen:
                continue
            seen.add(signature)
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_id,
                obj_id=obj_id,
                clear_old_points=False,
                points=np.array([[x, y]]),  # SAM2 expects (x, y).
                labels=np.array([label]),
            )

    def add_box_prompts(
        self,
        frame_ids: Union[int, List[int]],
        boxes: Optional[np.ndarray] = None,
        object_id: Optional[Union[int, List[int]]] = None,
    ):
        """Add box prompts (in (y0, x0, y1, x1) order) to the persistent SAM2 state.

        A box is pushed at most once per (object, frame); re-running with the same box is a no-op.
        SAM2 requires the box before any point on the same object/frame, so a box clears existing
        points ('clear_old_points=True'); we re-add any already-pushed points afterwards so a box and
        its correction points combine regardless of the order they were drawn in.
        """
        if boxes is None or len(boxes) == 0:
            return

        boxes = [np.asarray(b) for b in boxes]
        n = len(boxes)
        frame_ids = self._broadcast(frame_ids, n)
        object_ids = self._broadcast(1 if object_id is None else object_id, n)

        for frame_id, box, obj_id in zip(frame_ids, boxes, object_ids):
            key = (obj_id, frame_id)
            signature = tuple(np.round(box).astype(int).tolist())
            seen = self._pushed_boxes.setdefault(key, set())
            if signature in seen:
                continue
            seen.add(signature)
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_id,
                obj_id=obj_id,
                clear_old_points=True,
                box=np.array([_process_box(box, self.volume.shape[-2:])]),
            )
            for y, x, label in self._pushed_points.get(key, set()):
                self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_id,
                    obj_id=obj_id,
                    clear_old_points=False,
                    points=np.array([[x, y]]),
                    labels=np.array([label]),
                )

    def _prepare_mask(self, mask):
        """Bring a full-resolution 2d boolean mask into the padded-square frame the video predictor
        sees, so SAM2's own (direct) resize inside 'add_new_mask' is a no-op and the mask stays
        aligned with the resize-longest-side + pad the frames use."""
        mask = np.asarray(mask).astype(bool)
        target = self.predictor.image_size
        if mask.shape[-2:] == (target, target):
            return mask
        prepared, _ = resize_longest_side_and_pad_spatial_numpy(mask, target, is_label=True)
        return prepared.astype(bool)

    def add_mask_prompts(
        self,
        frame_ids: Union[int, List[int]],
        masks: Optional[List[np.ndarray]] = None,
        object_id: Optional[Union[int, List[int]]] = None,
    ):
        """Add mask prompts (full-resolution 2d boolean masks) to the persistent SAM2 state.

        A napari polygon or ellipse is filled into a mask (from 'shape_layer_to_prompts'). We first
        refine the drawn shape into the object on its seed frame (box + soft mask-logit cue, as in
        the per-slice path), then seed propagation with the refined mask - so the seed slice matches
        the per-slice result instead of reproducing the raw outline. SAM2's video predictor conditions
        a frame on either a mask or points/box (not both), so a mask prompt does not combine with
        points on the same object/frame. A mask already pushed (same object, frame, content) is
        skipped so re-runs only add newly drawn masks.
        """
        if masks is None or len(masks) == 0:
            return

        masks = [np.asarray(m) for m in masks]
        n = len(masks)
        frame_ids = self._broadcast(frame_ids, n)
        object_ids = self._broadcast(1 if object_id is None else object_id, n)

        for frame_id, mask, obj_id in zip(frame_ids, masks, object_ids):
            key = (obj_id, frame_id)
            # A stable digest, so the signature does not change between processes.
            signature = hashlib.sha1(np.ascontiguousarray(mask).tobytes()).hexdigest()
            seen = self._pushed_masks.setdefault(key, set())
            if signature in seen:
                continue
            seen.add(signature)

            # Refine the drawn shape into the object on the seed frame, then seed propagation with the
            # refined mask. The box is the shape's bounding box (nonzero extent of the filled mask).
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                continue
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype="float32")  # (x0, y0, x1, y1)
            refined = self._image_style_predict(frame_id, box=box, mask=mask)

            self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=frame_id,
                obj_id=obj_id,
                mask=self._prepare_mask(refined),
            )

    def add_pre_refined_masks(
        self,
        frame_ids: Union[int, List[int]],
        masks: Optional[List[np.ndarray]] = None,
        object_id: Optional[Union[int, List[int]]] = None,
    ):
        """Condition propagation directly on already-refined full-resolution binary masks.

        Unlike :meth:`add_mask_prompts`, this method does not run the image-style SAM2 decoder.
        The masks are assumed to be final seed-frame segmentations and are passed directly to the
        video predictor's mask-conditioning path.
        """
        prepared = _validate_pre_refined_masks(self.volume.shape, frame_ids, masks, object_id)
        if not prepared:
            return

        pending = []
        for frame_id, mask, obj_id in prepared:
            signature = hashlib.sha1(np.ascontiguousarray(mask).tobytes()).hexdigest()
            if signature not in self._pushed_pre_refined_masks.get((obj_id, frame_id), set()):
                pending.append((frame_id, mask, obj_id, signature))

        try:
            for frame_id, mask, obj_id, signature in pending:
                self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_id,
                    obj_id=obj_id,
                    mask=self._prepare_mask(mask),
                )
                self._pushed_pre_refined_masks.setdefault((obj_id, frame_id), set()).add(signature)
        except Exception:
            # A batch can fail after earlier masks have changed the persistent SAM2 state. Reset all
            # state so bookkeeping and predictor conditioning cannot disagree on the next run.
            self.reset_predictor()
            raise

    def _propagate_in_direction(
        self, reverse, update_progress=None, early_stop_patience=None, z_range=None, seen_frames=None
    ):
        """Run SAM2 propagation in one temporal direction, optionally stopping early.

        Each step of the SAM2 video predictor runs the full memory attention and mask decoder
        for one frame, which is the dominant cost of volumetric segmentation (especially on CPU).
        Early stopping reads the masks we already compute and breaks out of the propagation once
        the object has clearly left the volume, so we do not keep running the network on frames
        that no longer contain the object.

        Args:
            reverse: Propagate backwards in time (towards lower slice indices) if True.
            update_progress: Optional callback invoked with the number of newly processed frames.
            early_stop_patience: If given, stop this direction after this many consecutive frames
                in which every tracked object's mask is empty (i.e. the object has left the volume).
                'None' disables early stopping and propagates to the end of the volume.
            z_range: If given, an inclusive '(z_min, z_max)' bound on the slice indices propagation
                may cover. Propagation stops at the range edge in this direction. 'None' propagates
                to the end of the volume.
            seen_frames: Optional set of already-counted frame indices, shared across both directions,
                so a frame processed in both (the conditioning frame) advances 'update_progress' once.

        Returns:
            Mapping from frame index to per-object boolean masks, in the order frames were yielded.
        """
        video_segments = {}
        consecutive_empty = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state, reverse=reverse,
        ):
            # Hard z-range bound: stop once propagation would leave the user-selected slice range.
            if z_range is not None and not (z_range[0] <= out_frame_idx <= z_range[1]):
                break

            per_object = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }
            video_segments[out_frame_idx] = per_object
            # Count each slice at most once across both directions (the conditioning frame is yielded
            # by both), so the progress bar reaches its total exactly on a full pass without overshoot.
            if update_progress is not None and (seen_frames is None or out_frame_idx not in seen_frames):
                update_progress(1)
            if seen_frames is not None:
                seen_frames.add(out_frame_idx)

            # Early stopping: once every tracked object is absent for 'early_stop_patience'
            # consecutive frames, the object left the volume and there is nothing more to track.
            # A single empty frame is not enough (SAM2 can momentarily drop and recover a mask), so
            # we require a run of empty frames before breaking.
            if early_stop_patience is not None:
                frame_is_empty = not any(mask.any() for mask in per_object.values())
                consecutive_empty = consecutive_empty + 1 if frame_is_empty else 0
                if consecutive_empty >= early_stop_patience:
                    break

            # Hard z-range bound: we have just stored the edge slice, so stop before the predictor
            # steps outside the range (and pays for a frame we would discard).
            if z_range is not None and out_frame_idx == (z_range[0] if reverse else z_range[1]):
                break

        return video_segments

    def propagate_prompts(self, update_progress=None, early_stop_patience=None, z_range=None):
        # First, we propagate the masklets forward in time using the input prompts in selected frames.
        # 'update_progress' is an optional callback that is called with the number of newly processed
        # frames, so callers (e.g. the napari annotator) can report propagation progress to the user.
        # 'early_stop_patience' bounds the propagation by stopping a direction once the object is
        # absent for that many consecutive frames (see '_propagate_in_direction'). 'z_range' is an
        # inclusive '(z_min, z_max)' hard bound on the slices that propagation can cover.
        # Shared across both directions so the conditioning frame (yielded by both) is counted once.
        seen_frames = set()
        forward_video_segments = self._propagate_in_direction(
            reverse=False, update_progress=update_progress, early_stop_patience=early_stop_patience,
            z_range=z_range, seen_frames=seen_frames,
        )

        # Next, we do the propagation reverse in time.
        reverse_video_segments = {}
        if len(forward_video_segments) < self.volume.shape[0]:  # Perform reverse propagation only if necessary
            reverse_video_segments = self._propagate_in_direction(
                reverse=True, update_progress=update_progress, early_stop_patience=early_stop_patience,
                z_range=z_range, seen_frames=seen_frames,
            )
            # NOTE: The order is reversed to stitch the reverse propagation with forward.
            reverse_video_segments = dict(reversed(list(reverse_video_segments.items())))

        # Now stitch the segmented slices together.
        video_segments = {**reverse_video_segments, **forward_video_segments}
        return video_segments

    def _image_features_for_frame(self, frame_idx):
        """Build (image_embed, high_res_feats) for one slice from its precomputed features.

        Mirrors 'SAM2ImagePredictor.set_image', including the no-memory embedding that marks a
        memory-free image prediction. Couples to SAM2 predictor internals ('_get_image_feature',
        'no_mem_embed'), consistent with how the video-predictor subclass already uses them.
        """
        predictor = self.predictor
        _, _, vision_feats, _, feat_sizes = predictor._get_image_feature(self.inference_state, int(frame_idx), 1)
        if predictor.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + predictor.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *fs) for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        return feats[-1], feats[:-1]

    def _image_style_predict(self, frame_idx, box=None, mask=None, points=None, labels=None):
        """Image-predictor-style single-slice prediction using this slice's precomputed features.

        Runs the SAM2 prompt encoder + mask decoder with a box + soft mask-logit cue (and any
        correction points), so the decoder refines the prompt into the object. This reproduces the
        2d behaviour, unlike the video predictor's 'add_new_mask', which hard-conditions on the drawn
        mask and returns it unchanged. Returns a full-resolution boolean segmentation for the slice.
        """
        predictor = self.predictor
        device = self.inference_state["device"]
        orig_hw = tuple(int(s) for s in self.volume.shape[-2:])
        image_size = predictor.image_size
        scale = float(image_size) / max(orig_hw)  # resize-longest maps original coords into the model frame

        image_embed, high_res_feats = self._image_features_for_frame(frame_idx)

        # Box (labels 2, 3) plus any correction points, in (x, y), scaled into the model frame.
        coords, labs = [], []
        if box is not None:
            box = np.asarray(box, dtype="float32").reshape(2, 2) * scale
            coords.append(torch.as_tensor(box, dtype=torch.float, device=device))
            labs.append(torch.tensor([2, 3], dtype=torch.int, device=device))
        if points is not None and len(points) > 0:
            pts = np.asarray(points, dtype="float32") * scale
            coords.append(torch.as_tensor(pts, dtype=torch.float, device=device))
            labs.append(torch.as_tensor(np.asarray(labels), dtype=torch.int, device=device))
        concat_points = (torch.cat(coords)[None], torch.cat(labs)[None]) if coords else None

        # Soft low-res mask logits (1, 256, 256) from the filled shape, matching the frame's resize+pad.
        mask_input = None
        if mask is not None:
            logits = _compute_logits_from_mask(np.asarray(mask))
            mask_input = torch.as_tensor(logits, dtype=torch.float, device=device)[None]

        sparse, dense = predictor.sam_prompt_encoder(points=concat_points, boxes=None, masks=mask_input)
        low_res_masks, _, _, _ = predictor.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        if self._image_style_trafo is None:
            self._image_style_trafo = ResizeLongestSideTransforms(resolution=image_size, mask_threshold=0.0)
        seg = self._image_style_trafo.postprocess_masks(low_res_masks, orig_hw)
        return (seg.squeeze() > 0.0).cpu().numpy().astype("uint32")

    def _refine_slice_from_mask(self, frame_idx, boxes=None, masks=None, points=None, labels=None):
        """Per-slice refinement entry: pair the filled mask with its own box (the shape layer's
        'boxes'/'masks' are index-aligned) and refine it into the object via '_image_style_predict'."""
        idx = next(i for i, m in enumerate(masks) if m is not None)
        box = boxes[idx] if boxes is not None and idx < len(boxes) else None
        return self._image_style_predict(frame_idx, box=box, mask=masks[idx], points=points, labels=labels)

    def segment_slice(
        self,
        frame_idx: int,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[List] = None,
        masks: Optional[List] = None,
        object_id: int = 1,
    ):
        """Segment a single slice using SAM2 video predictor.

        Args:
            frame_idx: Slice index to segment.
            points: Point prompts (N, 2) array.
            labels: Point labels (N,) array.
            boxes: List of box prompts.
            masks: List of mask prompts (can be None).
            object_id: Object ID to use for the segmentation (default: 1).

        Returns:
            Segmentation mask for the slice (2D array), or None if no valid prompts provided.
        """
        # Validate prompts
        have_points = points is not None and len(points) > 0
        have_boxes = boxes is not None and len(boxes) > 0
        have_masks = masks is not None and any(m is not None for m in masks)

        if not have_points and not have_boxes and not have_masks:
            return None

        try:
            if have_masks:
                # A lasso, polygon or ellipse yields a filled mask. Refine it into the object the way the
                # 2d image predictor does (box + soft mask-logit cue through the mask decoder) rather
                # than 'add_new_mask', which hard-conditions on the drawn shape and returns it verbatim.
                seg = self._refine_slice_from_mask(
                    frame_idx, boxes=boxes, masks=masks,
                    points=points if have_points else None, labels=labels if have_points else None,
                )
            else:
                # Prepare prompts
                box = boxes[0] if have_boxes else None

                # Add prompts to the specific frame
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    points=points if have_points else None,
                    labels=labels if have_points else None,
                    box=box,
                )

                # Extract the mask from logits
                # out_mask_logits shape: (num_objects, 1, H, W)
                mask_logits = out_mask_logits[0]  # Get first object
                seg = (mask_logits.squeeze() > 0.0).cpu().numpy()

                # Crop back to the original slice shape (the video predictor pads non-square frames).
                seg = _crop_to_original_shape(seg, self.volume.shape[-2:]).astype("uint32")

        finally:
            # Reset the state to clear this object's prompts, along with the bookkeeping that
            # describes it, so the next segmentation starts fresh.
            self.predictor.reset_state(self.inference_state)
            self._clear_pushed_prompts()

        return seg

    def predict(self, update_progress=None, early_stop_patience=None, z_range=None):
        # First, we propagate prompts.
        video_segments = self.propagate_prompts(
            update_progress=update_progress, early_stop_patience=early_stop_patience, z_range=z_range,
        )

        # Next, let's merge the segmented objects per frame back together as instances per slice.
        # We allocate the full-volume output and index it by the slice id so that frames skipped by
        # early stopping (which are absent from 'video_segments') stay as background instead of
        # shifting the remaining slices out of alignment with the volume.
        shape = self.volume.shape[-2:]
        segmentation = np.zeros((self.volume.shape[0],) + tuple(shape), dtype="uint64")
        for slice_idx, instances in video_segments.items():
            per_slice_seg = segmentation[slice_idx]
            for instance_idx, instance_mask in instances.items():
                mask = _crop_to_original_shape(instance_mask.squeeze(), shape)
                per_slice_seg[mask] = instance_idx

        return segmentation


class TiledPromptableSegmentation3D:
    """Tiled promptable segmentation for volumetric data.

    Routes each prompt to the in-plane tile-column it falls in, runs a per-tile
    `PromptableSegmentation3D` on the tile's sub-volume (reusing the precomputed tiled embeddings),
    and stitches the per-tile results into the full volume. Exposes the same interface as
    `PromptableSegmentation3D`, so it is a drop-in replacement when the embeddings are tiled.

    Args:
        predictor: The SAM2 video predictor.
        volume: The input volume, shape (Z, Y, X).
        volume_embeddings: The precomputed tiled 3d embeddings (with per-tile `features`/`pos_enc`/
            `fpn` groups and `shape`/`tile_shape`/`halo` attrs). See `precompute_image_embeddings`.
        devices: Devices used for independent tile columns. By default all visible CUDA devices
            are used when the predictor is on CUDA. Pass a single device to pin inference to it.
    """

    def __init__(self, predictor, volume, volume_embeddings, devices: Devices = None, **kwargs):
        from bioimage_cpp.utils import Blocking
        from micro_sam.v2.batched_inference import _prepare_models, _resolve_devices

        resolved_devices = _resolve_devices(predictor, devices)
        self._predictor_devices = _prepare_models(predictor, resolved_devices)

        self.predictor = predictor
        self.volume = volume
        self.volume_embeddings = volume_embeddings
        self._kwargs = kwargs

        feats = volume_embeddings["features"]
        self.shape = tuple(int(s) for s in feats.attrs["shape"])
        self.tile_shape = tuple(int(s) for s in feats.attrs["tile_shape"])
        self.halo = tuple(int(s) for s in feats.attrs["halo"])
        self.tiling = Blocking([0, 0], list(self.shape[1:]), list(self.tile_shape))

        # Per-tile state, built lazily for the tiles that actually receive prompts.
        self._segmenters = {}
        # Device assigned to each active tile, see '_worker_id'.
        self._tile_workers = {}
        # Signature of the prompt set of the previous round, see 'sync_prompt_state'.
        self._prompt_signatures = set()

    def init_predictor(self):
        # Per-tile inference states are created lazily in '_get_segmenter'.
        pass

    def sync_prompt_state(self, signatures):
        """Discard the per-tile states when prompts were removed or changed since the last round.

        See `PromptableSegmentation3D.sync_prompt_state`. The prompt set is compared once for the
        whole volume, because a moved prompt can change which tile it belongs to.

        Args:
            signatures: Hashable descriptors of every prompt currently in the annotation layers.
        """
        signatures = set(signatures)
        if not self._prompt_signatures.issubset(signatures):
            self.reset_predictor()
        self._prompt_signatures = signatures

    def reset_predictor(self):
        # Drop the per-tile segmenters (each with its own inference state + embedding cache) so
        # committing or clearing frees their RAM. The tool rebuilds them lazily for the next prompt.
        for segmenter in self._segmenters.values():
            segmenter.reset_predictor()
        self._segmenters = {}
        self._tile_workers = {}
        self._prompt_signatures = set()
        _free_device_memory()

    def get_progress_total(self, z_range=None):
        """Return tile-slice propagation steps for the currently active tiles."""
        z_depth = self.shape[0] if z_range is None else z_range[1] - z_range[0] + 1
        return int(z_depth * len(self._segmenters))

    def _tile_index(self, y, x):
        """Return the id of the tile whose inner (halo-free) block contains the point (y, x)."""
        for tile_id in range(self.tiling.number_of_blocks):
            inner = self.tiling.get_block_with_halo(tile_id, list(self.halo)).inner_block
            if inner.begin[0] <= y < inner.end[0] and inner.begin[1] <= x < inner.end[1]:
                return tile_id
        return 0

    def _worker_id(self, tile_id):
        """Return the device a tile is bound to, assigned round-robin when the tile is first used.

        The tile state (inference state + embedding cache) lives on one device, so the affinity is
        kept for the tile's lifetime. Assigning by 'tile_id % n_devices' instead would leave devices
        idle, because the active tile ids are sparse (e.g. the tiles 0, 2, 4 share one residue).
        """
        if tile_id not in self._tile_workers:
            self._tile_workers[tile_id] = len(self._tile_workers) % len(self._predictor_devices)
        return self._tile_workers[tile_id]

    def _outer_offset(self, tile_id):
        """Return the (y0, x0) origin of the tile's outer (halo-included) block."""
        outer = self.tiling.get_block_with_halo(tile_id, list(self.halo)).outer_block
        return int(outer.begin[0]), int(outer.begin[1])

    def _get_segmenter(self, tile_id):
        if tile_id not in self._segmenters:
            from micro_sam.v2.util import _load_list_datasets

            outer = self.tiling.get_block_with_halo(tile_id, list(self.halo)).outer_block
            bb = (slice(int(outer.begin[0]), int(outer.end[0])), slice(int(outer.begin[1]), int(outer.end[1])))
            sub_volume = np.ascontiguousarray(self.volume[:, bb[0], bb[1]])

            feats = self.volume_embeddings["features"]
            tile_dataset = feats[str(tile_id)]
            # Keep the per-tile datasets lazy so the video predictor streams this tile-column one
            # frame at a time from disk, instead of materialising the whole column (~124 MB/slice).
            tile_embeddings = {
                "features": tile_dataset,
                "pos_enc": _load_list_datasets(self.volume_embeddings["pos_enc"], str(tile_id), lazy_loading=True),
                "fpn": _load_list_datasets(self.volume_embeddings["fpn"], str(tile_id), lazy_loading=True),
                "input_size": tile_dataset.attrs["input_size"],
                "original_size": tile_dataset.attrs["original_size"],
            }
            predictor, device = self._predictor_devices[self._worker_id(tile_id)]
            self._segmenters[tile_id] = PromptableSegmentation3D(
                predictor, sub_volume, tile_embeddings, device=device, **self._kwargs
            )
        return self._segmenters[tile_id]

    def _inner_slices(self, tile_id):
        """Return the (local, global) inner-block slices for placing a tile result into the volume."""
        block = self.tiling.get_block_with_halo(tile_id, list(self.halo))
        local = tuple(slice(b, e) for b, e in zip(block.inner_block_local.begin, block.inner_block_local.end))
        glob = tuple(slice(b, e) for b, e in zip(block.inner_block.begin, block.inner_block.end))
        return local, glob

    def _run_tile_jobs(
        self, tile_ids: List[int], function: Callable, update_progress: Optional[Callable[[int], None]] = None,
    ) -> List[Tuple]:
        """Run tile jobs concurrently across devices and serially within each device."""
        groups = {}
        for tile_id in tile_ids:
            groups.setdefault(self._worker_id(tile_id), []).append(tile_id)

        def run_group(group, worker_update=None):
            results = []
            for tile_id in group:
                if worker_update is None:
                    result = function(tile_id)
                else:
                    result = function(tile_id, worker_update)
                results.append((tile_id, result))
            return results

        if len(groups) < 2:
            if not groups:
                return []
            group = next(iter(groups.values()))
            return run_group(group, update_progress)

        progress_queue = queue.Queue()

        def forward_progress():
            increment = 0
            while True:
                try:
                    increment += int(progress_queue.get_nowait())
                except queue.Empty:
                    break
            if increment:
                update_progress(increment)

        results = []
        with futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
            worker_update = progress_queue.put if update_progress is not None else None
            tasks = [pool.submit(run_group, group, worker_update) for group in groups.values()]
            pending = set(tasks)
            while pending:
                done, pending = futures.wait(pending, timeout=0.05, return_when=futures.FIRST_COMPLETED)
                forward_progress()
                for task in done:
                    results.extend(task.result())
        forward_progress()
        return sorted(results, key=lambda result: result[0])

    def segment_slice(self, frame_idx, points=None, labels=None, boxes=None, masks=None, object_id=1):
        """Segment a single slice. Points are (x, y), boxes (x0, y0, x1, y1), as passed by the annotator.

        Groups the prompts by the tile they fall in, segments every tile with a positive cue, and
        unions the per-tile masks - so an object spanning tiles is segmented on both sides.
        """
        have_points = points is not None and len(points) > 0
        have_boxes = boxes is not None and len(boxes) > 0
        have_masks = masks is not None and any(m is not None for m in masks)
        if not have_points and not have_boxes and not have_masks:
            return None

        tile_points, tile_labels, tile_boxes, tile_masks = {}, {}, {}, {}
        if have_points:
            for point, label in zip(np.asarray(points), np.asarray(labels)):
                tid = self._tile_index(int(round(point[1])), int(round(point[0])))  # (y, x) from (x, y)
                tile_points.setdefault(tid, []).append(point)
                tile_labels.setdefault(tid, []).append(label)
        if have_boxes:
            for bidx, box in enumerate(boxes):
                box = np.asarray(box)  # (x0, y0, x1, y1)
                box_yx = np.array([box[1], box[0], box[3], box[2]])
                box_mask = masks[bidx] if masks is not None else None
                for tid, clipped in _box_to_tiles(self.tiling, self.halo, box_yx).items():
                    tile_boxes.setdefault(tid, []).append(np.array([clipped[1], clipped[0], clipped[3], clipped[2]]))
                    tile_masks.setdefault(tid, []).append(
                        None if box_mask is None else _crop_mask_to_tile(self.tiling, self.halo, tid, box_mask)
                    )

        tile_jobs = {}
        for tile_id in sorted(set(tile_points) | set(tile_boxes)):
            tpoints = np.asarray(tile_points.get(tile_id, [])).reshape(-1, 2)
            tlabels = np.asarray(tile_labels.get(tile_id, []), dtype=int)
            tboxes = tile_boxes.get(tile_id, [])
            if not ((tlabels == 1).any() or len(tboxes) > 0):
                continue
            y0, x0 = self._outer_offset(tile_id)
            local_points = (tpoints - np.array([x0, y0])) if len(tpoints) else None
            local_boxes = [b - np.array([x0, y0, x0, y0]) for b in tboxes] if tboxes else None
            local_masks = tile_masks.get(tile_id) if tboxes else None
            tile_jobs[tile_id] = (
                local_points,
                tlabels if len(tlabels) else None,
                local_boxes,
                local_masks,
            )

        def segment_tile(tile_id):
            local_points, local_labels, local_boxes, local_masks = tile_jobs[tile_id]
            return self._get_segmenter(tile_id).segment_slice(
                frame_idx, points=local_points, labels=local_labels,
                boxes=local_boxes, masks=local_masks, object_id=object_id,
            )

        out = np.zeros(self.shape[1:], dtype="uint32")
        found = False
        for tile_id, tile_seg in self._run_tile_jobs(sorted(tile_jobs), segment_tile):
            if tile_seg is None:
                continue
            local, glob = self._inner_slices(tile_id)
            region = tile_seg[local]
            sub = out[glob]
            positive = region > 0
            sub[positive] = region[positive]
            found = True

        return out if found else None

    def add_point_prompts(self, frame_ids, points, point_labels, object_id=None, multiple_objects=False):
        """Add point prompts. Points are in (y, x) order. Each is routed to the tile it falls in, so
        an object with prompts in several tiles is added to each of those tiles."""
        if points is None or len(points) == 0:
            return
        if object_id is None:
            object_id = 1

        tile_points, tile_labels = {}, {}
        for point, label in zip(np.asarray(points), np.asarray(point_labels)):
            tid = self._tile_index(int(round(point[0])), int(round(point[1])))
            tile_points.setdefault(tid, []).append(point)
            tile_labels.setdefault(tid, []).append(label)

        for tile_id, tpoints in tile_points.items():
            y0, x0 = self._outer_offset(tile_id)
            local_points = np.asarray(tpoints) - np.array([y0, x0])
            self._get_segmenter(tile_id).add_point_prompts(
                frame_ids=frame_ids, points=local_points, point_labels=np.asarray(tile_labels[tile_id]),
                object_id=object_id, multiple_objects=multiple_objects,
            )

    def add_box_prompts(self, frame_ids, boxes=None, object_id=None):
        """Add box prompts (y0, x0, y1, x1). A box spanning several tiles is added, clipped, to each."""
        if boxes is None or len(boxes) == 0:
            return
        # One object id per box. Default to a single object (id 1) when not batched.
        if object_id is None:
            object_id = [1] * len(boxes)
        elif not isinstance(object_id, list):
            object_id = [object_id] * len(boxes)
        # Group the (clipped) boxes and their object ids by the tile each falls in.
        tile_boxes, tile_ids = {}, {}
        for box, obj_id in zip(boxes, object_id):
            for tid, clipped in _box_to_tiles(self.tiling, self.halo, np.asarray(box)).items():
                tile_boxes.setdefault(tid, []).append(clipped)
                tile_ids.setdefault(tid, []).append(obj_id)
        for tile_id, tboxes in tile_boxes.items():
            y0, x0 = self._outer_offset(tile_id)
            local_boxes = [b - np.array([y0, x0, y0, x0]) for b in tboxes]
            self._get_segmenter(tile_id).add_box_prompts(
                frame_ids=frame_ids, boxes=np.array(local_boxes), object_id=tile_ids[tile_id],
            )

    def add_mask_prompts(self, frame_ids, masks=None, object_id=None):
        """Add mask prompts. Each mask is routed to the tiles its filled region overlaps, cropped to
        each tile's outer block, and added there (so a mask spanning tiles is added on both sides)."""
        if masks is None or len(masks) == 0:
            return
        masks = [np.asarray(m) for m in masks]
        # One object id per mask. Default to a single object (id 1) when not batched.
        if object_id is None:
            object_id = [1] * len(masks)
        elif not isinstance(object_id, list):
            object_id = [object_id] * len(masks)

        for mask, obj_id in zip(masks, object_id):
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                continue
            box_yx = np.array([ys.min(), xs.min(), ys.max() + 1, xs.max() + 1])
            for tid in _box_to_tiles(self.tiling, self.halo, box_yx):
                self._get_segmenter(tid).add_mask_prompts(
                    frame_ids=frame_ids, masks=[_crop_mask_to_tile(self.tiling, self.halo, tid, mask)],
                    object_id=obj_id,
                )

    def add_pre_refined_masks(self, frame_ids, masks=None, object_id=None):
        """Condition active tile columns directly on already-refined full-resolution masks.

        A tile is active only when foreground intersects its halo-free inner block. The full-plane
        mask is cropped to that tile's outer block before it is forwarded with the original object
        and frame IDs.
        """
        prepared = _validate_pre_refined_masks(self.shape, frame_ids, masks, object_id)
        if not prepared:
            return

        tile_jobs = []
        for frame_id, mask, obj_id in prepared:
            for tile_id in range(self.tiling.number_of_blocks):
                block = self.tiling.get_block_with_halo(tile_id, list(self.halo))
                inner = block.inner_block
                inner_slice = (
                    slice(int(inner.begin[0]), int(inner.end[0])),
                    slice(int(inner.begin[1]), int(inner.end[1])),
                )
                if not mask[inner_slice].any():
                    continue
                tile_jobs.append((
                    tile_id,
                    frame_id,
                    _crop_mask_to_tile(self.tiling, self.halo, tile_id, mask),
                    obj_id,
                ))

        try:
            for tile_id, frame_id, local_mask, obj_id in tile_jobs:
                self._get_segmenter(tile_id).add_pre_refined_masks(
                    frame_ids=frame_id, masks=[local_mask], object_id=obj_id,
                )
        except Exception:
            # Avoid retaining only a prefix of the tile-conditioning jobs.
            self.reset_predictor()
            raise

    def predict(self, update_progress=None, early_stop_patience=None, z_range=None):
        """Propagate the prompts in every active tile and stitch the results into the full volume.

        Object ids are preserved across tiles (the inner blocks are disjoint), so an object that was
        prompted in several tiles keeps one id and is merged across the tile boundaries.
        """
        segmentation = np.zeros(self.shape, dtype="uint64")

        def predict_tile(tile_id, tile_update=None):
            return self._segmenters[tile_id].predict(
                update_progress=tile_update, early_stop_patience=early_stop_patience, z_range=z_range,
            )

        for tile_id, tile_seg in self._run_tile_jobs(
            sorted(self._segmenters), predict_tile, update_progress=update_progress,
        ):
            local, glob = self._inner_slices(tile_id)
            inner = tile_seg[(slice(None),) + local]
            region = segmentation[(slice(None),) + glob]
            positive = inner != 0
            region[positive] = inner[positive]
        return segmentation
