"""Functions for prompt-based segmentation with Segment Anything.
"""

import warnings
from typing import Optional, Tuple

import numpy as np

from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries

from segment_anything.predictor import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

import torch

from bioimage_cpp.utils import Blocking
from bioimage_cpp.filters import gaussian_smoothing
from bioimage_cpp.distance import distance_transform

from .. import util
from .util import set_precomputed


#
# helper functions for translating mask inputs into other prompts
#


# compute the bounding box from a mask. SAM expects the following input:
# box (np.ndarray or None): A length 4 array given a box prompt to the model, in XYXY format.
def _compute_box_from_mask(mask, original_size=None, box_extension=0):
    coords = np.where(mask == 1)
    min_y, min_x = coords[0].min(), coords[1].min()
    max_y, max_x = coords[0].max(), coords[1].max()
    box = np.array([min_y, min_x, max_y + 1, max_x + 1])
    return _process_box(box, mask.shape, original_size=original_size, box_extension=box_extension)


# sample points from a mask. SAM expects the following point inputs:
def _compute_points_from_mask(mask, original_size, box_extension, use_single_point=False):
    box = _compute_box_from_mask(mask, box_extension=box_extension)

    # get slice and offset in python coordinate convention
    bb = (slice(box[1], box[3]), slice(box[0], box[2]))
    offset = np.array([box[1], box[0]])

    # crop the mask and compute distances
    cropped_mask = mask[bb]
    object_boundaries = find_boundaries(cropped_mask, mode="outer")
    distances = gaussian_smoothing(distance_transform(object_boundaries == 0), sigma=1.0)
    inner_distances = distances.copy()
    cropped_mask = cropped_mask.astype("bool")
    inner_distances[~cropped_mask] = 0.0
    if use_single_point:
        center = inner_distances.argmax()
        center = np.unravel_index(center, inner_distances.shape)
        point_coords = (center + offset)[None]
        point_labels = np.ones(1, dtype="uint8")
        return point_coords[:, ::-1], point_labels

    outer_distances = distances.copy()
    outer_distances[cropped_mask] = 0.0

    # sample positives and negatives from the distance maxima
    inner_maxima = peak_local_max(inner_distances, exclude_border=False, min_distance=3)
    outer_maxima = peak_local_max(outer_distances, exclude_border=False, min_distance=5)

    # derive the positive (=inner maxima) and negative (=outer maxima) points
    point_coords = np.concatenate([inner_maxima, outer_maxima]).astype("float64")
    point_coords += offset

    if original_size is not None:
        scale_factor = np.array([
            original_size[0] / float(mask.shape[0]), original_size[1] / float(mask.shape[1])
        ])[None]
        point_coords *= scale_factor

    # get the point labels
    point_labels = np.concatenate(
        [np.ones(len(inner_maxima), dtype="uint8"), np.zeros(len(outer_maxima), dtype="uint8")]
    )
    return point_coords[:, ::-1], point_labels


def _compute_logits_from_mask(mask, eps=1e-3):

    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    # resize to the expected mask shape of SAM (256x256)
    assert mask.ndim == 2
    expected_shape = (256, 256)

    # Resize the *binary* mask (instead of the inverse-sigmoid logits) to SAM's expected
    # mask shape and re-binarize afterwards. This keeps small objects from being washed out
    # by the antialiased downscaling that ResizeLongestSide applies, which otherwise makes
    # the mask prompt too weak for small objects in large (and non-square) images.
    binary_mask = (mask == 1).astype("float32")

    if binary_mask.shape != expected_shape:
        trafo = ResizeLongestSide(expected_shape[0])
        binary_mask = trafo.apply_image_torch(torch.from_numpy(binary_mask[None, None]))
        binary_mask = binary_mask.numpy().squeeze()

        if binary_mask.shape != expected_shape:  # shape is not square -> pad the other side
            h, w = binary_mask.shape
            padh = expected_shape[0] - h
            padw = expected_shape[1] - w
            # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
            pad_width = ((0, padh), (0, padw))
            binary_mask = np.pad(binary_mask, pad_width, mode="constant", constant_values=0)

    logits = np.where(binary_mask > 0.5, inv_sigmoid(1 - eps), inv_sigmoid(eps)).astype("float32")
    logits = logits[None]
    assert logits.shape == (1, 256, 256), f"{logits.shape}"
    return logits


#
# other helper functions
#


def _process_box(box, shape, original_size=None, box_extension=0):
    if box_extension == 0:  # no extension
        extension_y, extension_x = 0, 0
    elif box_extension >= 1:  # extension by a fixed factor
        extension_y, extension_x = box_extension, box_extension
    else:  # extension by fraction of the box len
        len_y, len_x = box[2] - box[0], box[3] - box[1]
        extension_y, extension_x = box_extension * len_y, box_extension * len_x

    box = np.array([
        max(box[1] - extension_x, 0), max(box[0] - extension_y, 0),
        min(box[3] + extension_x, shape[1]), min(box[2] + extension_y, shape[0]),
    ])

    if original_size is not None:
        trafo = ResizeLongestSide(max(original_size))
        box = trafo.apply_boxes(box[None], (256, 256)).squeeze()

    # round up the bounding box values
    box = np.round(box).astype(int)

    return box


# Select the correct tile based on average of points
# and bring the points to the coordinate system of the tile.
# Discard points that are not in the tile and warn if this happens.
def _points_to_tile(prompts, shape, tile_shape, halo):
    points, labels = prompts

    tiling = Blocking([0, 0], shape, tile_shape)
    center = np.mean(points, axis=0).round().astype("int").tolist()
    tile_id = tiling.coordinates_to_block_id(center)

    tile = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
    offset = tile.begin
    this_tile_shape = tile.shape

    points_in_tile = points - np.array(offset)
    labels_in_tile = labels

    valid_point_mask = (points_in_tile >= 0).all(axis=1)
    valid_point_mask = np.logical_and(
        valid_point_mask,
        np.logical_and(
            points_in_tile[:, 0] < this_tile_shape[0], points_in_tile[:, 1] < this_tile_shape[1]
        )
    )
    if not valid_point_mask.all():
        points_in_tile = points_in_tile[valid_point_mask]
        labels_in_tile = labels_in_tile[valid_point_mask]
        warnings.warn(
            f"{(~valid_point_mask).sum()} points were not in the tile and are dropped"
        )

    return tile_id, tile, (points_in_tile, labels_in_tile)


def _box_to_tile(box, shape, tile_shape, halo):
    tiling = Blocking([0, 0], shape, tile_shape)
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]).round().astype("int").tolist()
    tile_id = tiling.coordinates_to_block_id(center)

    tile = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
    offset = tile.begin
    this_tile_shape = tile.shape

    box_in_tile = np.array(
        [
            max(box[0] - offset[0], 0), max(box[1] - offset[1], 0),
            min(box[2] - offset[0], this_tile_shape[0]), min(box[3] - offset[1], this_tile_shape[1])
        ]
    )

    return tile_id, tile, box_in_tile


def _mask_to_tile(mask, shape, tile_shape, halo):
    tiling = Blocking([0, 0], shape, tile_shape)

    coords = np.where(mask)
    center = np.array([np.mean(coords[0]), np.mean(coords[1])]).round().astype("int").tolist()
    tile_id = tiling.coordinates_to_block_id(center)

    tile = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
    bb = tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end))

    mask_in_tile = mask[bb]
    return tile_id, tile, mask_in_tile


def _initialize_predictor(predictor, image_embeddings, i, prompts, to_tile):
    tile = None

    # Set the precomputed state for tiled prediction.
    if image_embeddings is not None and image_embeddings["input_size"] is None:
        features = image_embeddings["features"]
        shape, tile_shape, halo = features.attrs["shape"], features.attrs["tile_shape"], features.attrs["halo"]
        tile_id, tile, prompts = to_tile(prompts, shape, tile_shape, halo)
        set_precomputed(predictor, image_embeddings, i, tile_id=tile_id)

    # Set the precomputed state for normal prediction.
    elif image_embeddings is not None:
        shape = image_embeddings["original_size"]
        set_precomputed(predictor, image_embeddings, i)

    else:
        shape = predictor.original_size

    return predictor, tile, prompts, shape


def _tile_to_full_mask(mask, shape, tile):
    full_mask = np.zeros(mask.shape[0:1] + tuple(shape), dtype=mask.dtype)
    bb = tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end))
    full_mask[(slice(None),) + bb] = mask
    return full_mask


def _prepare_tiled_mask_prompt_jobs(mask, shape, tile_shape, halo, box=None, points=None, labels=None):
    """Prepare local prompts for refining one mask across all relevant embedding tiles."""
    shape = tuple(shape)
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"The mask shape must match the tiled embedding shape {shape}, got {mask.shape}.")
    mask = mask.astype(bool, copy=False)

    if box is not None:
        box = np.asarray(box)
        if box.shape != (4,):
            raise ValueError(f"The box must have shape (4,), got {box.shape}.")

    if points is None:
        if labels is not None:
            raise ValueError("Point labels were passed without point prompts.")
        points = np.empty((0, 2), dtype="float64")
        labels = np.empty((0,), dtype="int64")
    else:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"The points must have shape (N, 2), got {points.shape}.")
        if labels is None:
            raise ValueError("If points are passed you also need to pass labels.")
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(points):
            raise ValueError("The point labels must be a one-dimensional array matching the points.")
        if len(points) > 0:
            valid = (
                (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &
                (points[:, 1] >= 0) & (points[:, 1] < shape[1])
            )
            if not valid.all():
                raise ValueError("All point prompts must lie inside the tiled image.")

    tiling = Blocking([0, 0], shape, tile_shape)
    active_tile_ids = set()
    for tile_id in range(tiling.number_of_blocks):
        block = tiling.get_block_with_halo(tile_id, list(halo))
        inner = block.inner_block
        inner_bb = tuple(slice(beg, end) for beg, end in zip(inner.begin, inner.end))
        if mask[inner_bb].any():
            active_tile_ids.add(tile_id)

        if box is not None:
            y0, x0, y1, x1 = box
            intersects_inner = not (
                y1 <= inner.begin[0] or y0 >= inner.end[0] or
                x1 <= inner.begin[1] or x0 >= inner.end[1]
            )
            if intersects_inner:
                active_tile_ids.add(tile_id)

    # Positive points may deliberately expand the target beyond its existing mask or box. Activate
    # only the tile that owns the point; once active, a tile receives every sparse cue in its halo.
    for point, label in zip(points, labels):
        if label == 1:
            coordinate = np.round(point).astype("int").tolist()
            coordinate = [min(coord, sh - 1) for coord, sh in zip(coordinate, shape)]
            active_tile_ids.add(tiling.coordinates_to_block_id(coordinate))

    jobs = {}
    for tile_id in sorted(active_tile_ids):
        block = tiling.get_block_with_halo(tile_id, list(halo))
        outer = block.outer_block
        outer_bb = tuple(slice(beg, end) for beg, end in zip(outer.begin, outer.end))
        local_mask = mask[outer_bb]

        local_box = None
        if box is not None:
            y0, x0, y1, x1 = box
            clipped_box = np.array([
                max(y0, outer.begin[0]), max(x0, outer.begin[1]),
                min(y1, outer.end[0]), min(x1, outer.end[1]),
            ])
            if clipped_box[0] < clipped_box[2] and clipped_box[1] < clipped_box[3]:
                local_box = clipped_box - np.array([
                    outer.begin[0], outer.begin[1], outer.begin[0], outer.begin[1]
                ])

        if len(points) == 0:
            local_points, local_labels = None, None
        else:
            point_mask = (
                (points[:, 0] >= outer.begin[0]) & (points[:, 0] < outer.end[0]) &
                (points[:, 1] >= outer.begin[1]) & (points[:, 1] < outer.end[1])
            )
            local_points = points[point_mask] - np.asarray(outer.begin)
            local_labels = labels[point_mask]
            if len(local_points) == 0:
                local_points, local_labels = None, None

        has_positive_point = local_labels is not None and np.any(local_labels == 1)
        if not (local_mask.any() or local_box is not None or has_positive_point):
            continue

        jobs[tile_id] = {
            "block": block,
            "mask": local_mask,
            "box": local_box,
            "points": local_points,
            "labels": local_labels,
        }

    return tiling, jobs


def _stitch_tiled_mask_predictions(predictions, shape):
    """Stitch tile predictions through disjoint inner blocks."""
    output = np.zeros((1,) + tuple(shape), dtype=bool)
    for block, prediction in predictions:
        prediction = np.asarray(prediction)
        if prediction.ndim == 2:
            prediction = prediction[None]
        if prediction.shape[0] != 1:
            raise ValueError(f"Expected one prediction per tile, got shape {prediction.shape}.")

        local = tuple(
            slice(beg, end) for beg, end in zip(block.inner_block_local.begin, block.inner_block_local.end)
        )
        glob = tuple(slice(beg, end) for beg, end in zip(block.inner_block.begin, block.inner_block.end))
        output[(slice(None),) + glob] = prediction[(slice(None),) + local].astype(bool, copy=False)
    return output


#
# functions for prompted segmentation:
# - segment_from_points: use point prompts as input
# - segment_from_mask: use binary mask as input, support conversion to mask, box and point prompts
# - segment_from_box: use box prompt as input
# - segment_from_box_and_points: use box and point prompts as input
#


def segment_from_points(
    predictor: SamPredictor,
    points: np.ndarray,
    labels: np.ndarray,
    image_embeddings: Optional[util.ImageEmbeddings] = None,
    i: Optional[int] = None,
    multimask_output: bool = False,
    return_all: bool = False,
    use_best_multimask: Optional[bool] = None,
):
    """Segmentation from point prompts.

    Args:
        predictor: The segment anything predictor.
        points: The point prompts given in the image coordinate system.
        labels: The labels (positive or negative) associated with the points.
        image_embeddings: Optional precomputed image embeddings.
            Has to be passed if the predictor is not yet initialized.
        i: Index for the image data. Required if the input data has three spatial dimensions
            or a time dimension and two spatial dimensions.
        multimask_output: Whether to return multiple or just a single mask. By default, set to 'False'.
        return_all: Whether to return the score and logits in addition to the mask. By default, set to 'False'.
        use_best_multimask: Whether to use multimask output and then choose the best mask.
            By default this is used for a single positive point and not otherwise.

    Returns:
        The binary segmentation mask.
    """
    predictor, tile, prompts, shape = _initialize_predictor(
        predictor, image_embeddings, i, (points, labels), _points_to_tile
    )
    points, labels = prompts

    if use_best_multimask is None:
        use_best_multimask = len(points) == 1 and labels[0] == 1
    multimask_output_ = multimask_output or use_best_multimask

    # predict the mask
    mask, scores, logits = predictor.predict(
        point_coords=points[:, ::-1],  # SAM has reversed XY conventions
        point_labels=labels,
        multimask_output=multimask_output_,
    )

    if use_best_multimask:
        best_mask_id = np.argmax(scores)
        mask = mask[best_mask_id][None]

    if tile is not None:
        mask = _tile_to_full_mask(mask, shape, tile)

    if return_all:
        return mask, scores, logits
    else:
        return mask


def segment_from_mask(
    predictor: SamPredictor,
    mask: np.ndarray,
    image_embeddings: Optional[util.ImageEmbeddings] = None,
    i: Optional[int] = None,
    use_box: bool = True,
    use_mask: bool = True,
    use_points: bool = False,
    original_size: Optional[Tuple[int, ...]] = None,
    multimask_output: bool = False,
    return_all: bool = False,
    return_logits: bool = False,
    box_extension: float = 0.0,
    box: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    use_single_point: bool = False,
):
    """Segmentation from a mask prompt.

    Args:
        predictor: The segment anything predictor.
        mask: The mask used to derive prompts.
        image_embeddings: Optional precomputed image embeddings.
            Has to be passed if the predictor is not yet initialized.
        i: Index for the image data. Required if the input data has three spatial dimensions
            or a time dimension and two spatial dimensions.
        use_box: Whether to derive the bounding box prompt from the mask. By default, set to 'True'.
        use_mask: Whether to use the mask itself as prompt. By default, set to 'True'.
        use_points: Whether to derive point prompts from the mask. By default, set to 'False'.
        original_size: Full image shape. Use this if the mask that is being passed
            downsampled compared to the original image.
        multimask_output: Whether to return multiple or just a single mask. By default, set to 'False'.
        return_all: Whether to return the score and logits in addition to the mask. By default, set to 'False'.
        box_extension: Relative factor used to enlarge the bounding box prompt.
            By default, does not enlarge the bounding box.
        box: Precomputed bounding box.
        points: Precomputed point prompts.
        labels: Positive/negative labels corresponding to the point prompts.
        use_single_point: Whether to derive just a single point from the mask.
            In case use_points is true.

    Returns:
        The binary segmentation mask.
    """
    prompts = (mask, box, points, labels)

    def _to_tile(prompts, shape, tile_shape, halo):
        mask, box, points, labels = prompts
        tile_id, tile, mask = _mask_to_tile(mask, shape, tile_shape, halo)
        if points is not None:
            tile_id_points, tile, point_prompts = _points_to_tile((points, labels), shape, tile_shape, halo)
            if tile_id_points != tile_id:
                raise RuntimeError(f"Inconsistent tile ids for mask and point prompts: {tile_id_points} != {tile_id}.")
            points, labels = point_prompts
        if box is not None:
            tile_id_box, tile, box = _box_to_tile(box, shape, tile_shape, halo)
            if tile_id_box != tile_id:
                raise RuntimeError(f"Inconsistent tile ids for mask and box prompts: {tile_id_box} != {tile_id}.")
        return tile_id, tile, (mask, box, points, labels)

    predictor, tile, prompts, shape = _initialize_predictor(predictor, image_embeddings, i, prompts, _to_tile)
    mask, box, points, labels = prompts

    if points is not None:
        if labels is None:
            raise ValueError("If points are passed you also need to pass labels.")
        point_coords, point_labels = points, labels

    elif use_points and mask.sum() != 0:
        point_coords, point_labels = _compute_points_from_mask(
            mask, original_size=original_size, box_extension=box_extension,
            use_single_point=use_single_point,
        )

    else:
        point_coords, point_labels = None, None

    if box is None:
        box = _compute_box_from_mask(
            mask, original_size=original_size, box_extension=box_extension
        ) if use_box and mask.sum() != 0 else None
    else:
        box = _process_box(box, mask.shape, original_size=original_size, box_extension=box_extension)

    logits = _compute_logits_from_mask(mask) if use_mask else None

    mask, scores, logits = predictor.predict(
        point_coords=point_coords, point_labels=point_labels,
        mask_input=logits, box=box,
        multimask_output=multimask_output, return_logits=return_logits
    )

    if tile is not None:
        mask = _tile_to_full_mask(mask, shape, tile)

    if return_all:
        return mask, scores, logits
    else:
        return mask


def segment_from_mask_tiled(
    predictor: SamPredictor,
    mask: np.ndarray,
    image_embeddings: util.ImageEmbeddings,
    i: Optional[int] = None,
    box: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
):
    """Refine one binary mask target across all relevant tiled image embeddings.

    Unlike :func:`segment_from_mask`, which routes all prompts to one tile, this function activates
    every tile whose inner block intersects the dense mask or the optional enclosing box. A positive
    point also activates its owning tile, so corrections can expand the target beyond the source
    mask. Each prediction is restricted to the tile's disjoint inner block during stitching.

    The box uses ``(y0, x0, y1, x1)`` coordinates and points use ``(y, x)`` coordinates, consistent
    with the other tiled prompt routing helpers.

    Args:
        predictor: The segment anything predictor.
        mask: The full-size binary mask prompt.
        image_embeddings: Precomputed tiled image embeddings.
        i: Index for a 3D image or time series.
        box: Optional full-image enclosing box.
        points: Optional full-image point corrections.
        labels: Positive/negative labels corresponding to the point corrections.

    Returns:
        The full-size binary prediction with shape ``(1, H, W)``.
    """
    if image_embeddings is None or image_embeddings.get("input_size") is not None:
        raise ValueError("segment_from_mask_tiled requires tiled image embeddings.")

    features = image_embeddings["features"]
    shape = tuple(features.attrs["shape"])
    tile_shape = tuple(features.attrs["tile_shape"])
    halo = tuple(features.attrs["halo"])
    _, jobs = _prepare_tiled_mask_prompt_jobs(
        mask, shape, tile_shape, halo, box=box, points=points, labels=labels
    )

    predictions = []
    for tile_id, job in jobs.items():
        set_precomputed(predictor, image_embeddings, i=i, tile_id=tile_id)
        # ``segment_from_mask`` passes explicit point coordinates directly to SAM, which expects
        # (x, y). Keep this public tiled helper consistent with napari and the other prompt helpers
        # by routing in (y, x) above and swapping only at the untiled predictor boundary.
        local_points = None if job["points"] is None else job["points"][:, ::-1]
        prediction = segment_from_mask(
            predictor,
            job["mask"],
            use_box=False,
            use_mask=True,
            use_points=False,
            box=job["box"],
            points=local_points,
            labels=job["labels"],
        )
        predictions.append((job["block"], prediction))

    return _stitch_tiled_mask_predictions(predictions, shape)


def segment_from_box(
    predictor: SamPredictor,
    box: np.ndarray,
    image_embeddings: Optional[util.ImageEmbeddings] = None,
    i: Optional[int] = None,
    multimask_output: bool = False,
    return_all: bool = False,
    box_extension: float = 0.0,
):
    """Segmentation from a box prompt.

    Args:
        predictor: The segment anything predictor.
        box: The box prompt.
        image_embeddings: Optional precomputed image embeddings.
            Has to be passed if the predictor is not yet initialized.
        i: Index for the image data. Required if the input data has three spatial dimensions
            or a time dimension and two spatial dimensions.
        multimask_output: Whether to return multiple or just a single mask. By default, set to 'False'.
        return_all: Whether to return the score and logits in addition to the mask. By default, set to 'False'.
        box_extension: Relative factor used to enlarge the bounding box prompt.
            By default, does not enlarge the bounding box.

    Returns:
        The binary segmentation mask.
    """
    predictor, tile, box, shape = _initialize_predictor(
        predictor, image_embeddings, i, box, _box_to_tile
    )
    mask, scores, logits = predictor.predict(
        box=_process_box(box, shape, box_extension=box_extension), multimask_output=multimask_output
    )

    if tile is not None:
        mask = _tile_to_full_mask(mask, shape, tile)

    if return_all:
        return mask, scores, logits
    else:
        return mask


def segment_from_box_and_points(
    predictor: SamPredictor,
    box: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    image_embeddings: Optional[util.ImageEmbeddings] = None,
    i: Optional[int] = None,
    multimask_output: bool = False,
    return_all: bool = False,
):
    """Segmentation from a box prompt and point prompts.

    Args:
        predictor: The segment anything predictor.
        box: The box prompt.
        points: The point prompts, given in the image coordinates system.
        labels: The point labels, either positive or negative.
        image_embeddings: Optional precomputed image embeddings.
            Has to be passed if the predictor is not yet initialized.
        i: Index for the image data. Required if the input data has three spatial dimensions
            or a time dimension and two spatial dimensions.
        multimask_output: Whether to return multiple or just a single mask. By default, set to 'False'.
        return_all: Whether to return the score and logits in addition to the mask. By default, set to 'False'.

    Returns:
        The binary segmentation mask.
    """
    def box_and_points_to_tile(prompts, shape, tile_shape, halo):
        box, points, labels = prompts
        tile_id, tile, point_prompts = _points_to_tile((points, labels), shape, tile_shape, halo)
        points, labels = point_prompts
        tile_id_box, tile, box = _box_to_tile(box, shape, tile_shape, halo)
        if tile_id_box != tile_id:
            raise RuntimeError(f"Inconsistent tile ids for box and point annotations: {tile_id_box} != {tile_id}.")
        return tile_id, tile, (box, points, labels)

    predictor, tile, prompts, shape = _initialize_predictor(
        predictor, image_embeddings, i, (box, points, labels), box_and_points_to_tile
    )
    box, points, labels = prompts

    mask, scores, logits = predictor.predict(
        point_coords=points[:, ::-1],  # SAM has reversed XY conventions
        point_labels=labels,
        box=_process_box(box, shape),
        multimask_output=multimask_output
    )

    if tile is not None:
        mask = _tile_to_full_mask(mask, shape, tile)

    if return_all:
        return mask, scores, logits
    else:
        return mask
