"""
Functions for prompt-based segmentation with Segment Anything.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
from nifty.tools import blocking
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt

from segment_anything.predictor import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from . import util


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
    distances = gaussian(distance_transform_edt(object_boundaries == 0))
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
        [
            np.ones(len(inner_maxima), dtype="uint8"),
            np.zeros(len(outer_maxima), dtype="uint8"),
        ]
    )
    return point_coords[:, ::-1], point_labels


def _compute_logits_from_mask(mask, eps=1e-3):

    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

    # resize to the expected mask shape of SAM (256x256)
    assert logits.ndim == 2
    expected_shape = (256, 256)

    if logits.shape == expected_shape:  # shape matches, do nothing
        pass

    elif logits.shape[0] == logits.shape[1]:  # shape is square
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image_torch(torch.from_numpy(logits[None, None]))
        logits = logits.numpy().squeeze()

    else:  # shape is not square
        # resize the longest side to expected shape
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image_torch(torch.from_numpy(logits[None, None]))
        logits = logits.numpy().squeeze()

        # pad the other side
        h, w = logits.shape
        padh = expected_shape[0] - h
        padw = expected_shape[1] - w
        # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
        pad_width = ((0, padh), (0, padw))
        logits = np.pad(logits, pad_width, mode="constant", constant_values=0)

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

    tiling = blocking([0, 0], shape, tile_shape)
    center = np.mean(points, axis=0).round().astype("int").tolist()
    tile_id = tiling.coordinatesToBlockId(center)

    tile = tiling.getBlockWithHalo(tile_id, list(halo)).outerBlock
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
    tiling = blocking([0, 0], shape, tile_shape)
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]).round().astype("int").tolist()
    tile_id = tiling.coordinatesToBlockId(center)

    tile = tiling.getBlockWithHalo(tile_id, list(halo)).outerBlock
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
    tiling = blocking([0, 0], shape, tile_shape)

    coords = np.where(mask)
    center = np.array([np.mean(coords[0]), np.mean(coords[1])]).round().astype("int").tolist()
    tile_id = tiling.coordinatesToBlockId(center)

    tile = tiling.getBlockWithHalo(tile_id, list(halo)).outerBlock
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
        util.set_precomputed(predictor, image_embeddings, i, tile_id=tile_id)

    # Set the precomputed state for normal prediction.
    elif image_embeddings is not None:
        shape = image_embeddings["original_size"]
        util.set_precomputed(predictor, image_embeddings, i)

    else:
        shape = predictor.original_size

    return predictor, tile, prompts, shape


def _tile_to_full_mask(mask, shape, tile):
    full_mask = np.zeros(mask.shape[0:1] + tuple(shape), dtype=mask.dtype)
    bb = tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end))
    full_mask[(slice(None),) + bb] = mask
    return full_mask


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
        multimask_output: Whether to return multiple or just a single mask.
        return_all: Whether to return the score and logits in addition to the mask.
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
        use_box: Whether to derive the bounding box prompt from the mask.
        use_mask: Whether to use the mask itself as prompt.
        use_points: Whether to derive point prompts from the mask.
        original_size: Full image shape. Use this if the mask that is being passed
            downsampled compared to the original image.
        multimask_output: Whether to return multiple or just a single mask.
        return_all: Whether to return the score and logits in addition to the mask.
        box_extension: Relative factor used to enlarge the bounding box prompt.
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

    elif use_points:
        point_coords, point_labels = _compute_points_from_mask(
            mask, original_size=original_size, box_extension=box_extension,
            use_single_point=use_single_point,
        )

    else:
        point_coords, point_labels = None, None

    if box is None:
        box = _compute_box_from_mask(
            mask, original_size=original_size, box_extension=box_extension
        ) if use_box else None
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
        multimask_output: Whether to return multiple or just a single mask.
        return_all: Whether to return the score and logits in addition to the mask.
        box_extension: Relative factor used to enlarge the bounding box prompt.

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
        multimask_output: Whether to return multiple or just a single mask.
        return_all: Whether to return the score and logits in addition to the mask.

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
