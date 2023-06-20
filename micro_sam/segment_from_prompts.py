import warnings

import numpy as np
from nifty.tools import blocking
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from scipy.ndimage import distance_transform_edt

from segment_anything.utils.transforms import ResizeLongestSide
from . import util


#
# helper functions for translating mask inputs into other prompts
#


# compute the bounding box from a mask. SAM expects the following input:
# box (np.ndarray or None): A length 4 array given a box prompt to the model, in XYXY format.
def _compute_box_from_mask(mask, original_size=None, box_extension=0):
    coords = np.where(mask == 1)
    box = np.array([
        max(coords[1].min() - box_extension, 0), max(coords[0].min() - box_extension, 0),
        min(coords[1].max() + 1 + box_extension, mask.shape[1]),
        min(coords[0].max() + 1 + box_extension, mask.shape[0]),
    ])
    # TODO how do we deal with aspect ratios???
    if original_size is not None:
        trafo = ResizeLongestSide(max(original_size))
        box = trafo.apply_boxes(box[None], (256, 256)).squeeze()
    return box


# sample points from a mask. SAM expects the following point inputs:
def _compute_points_from_mask(mask, original_size):
    box = _compute_box_from_mask(mask, box_extension=5)

    # get slice and offset in python coordinate convention
    bb = (slice(box[1], box[3]), slice(box[0], box[2]))
    offset = np.array([box[1], box[0]])

    # crop the mask and compute distances
    cropped_mask = mask[bb]
    inner_distances = gaussian(distance_transform_edt(cropped_mask == 1))
    outer_distances = gaussian(distance_transform_edt(cropped_mask == 0))

    # sample positives and negatives from the distance maxima
    inner_maxima = peak_local_max(inner_distances, exclude_border=False)
    outer_maxima = peak_local_max(outer_distances, exclude_border=False)

    # derive the positive (=inner maxima) and negative (=outer maxima) points
    point_coords = np.concatenate([inner_maxima, outer_maxima]).astype("float64")
    point_coords += offset

    if original_size is not None:
        scale_factor = np.array([
            float(mask.shape[0]) / original_size[0], float(mask.shape[1]) / original_size[1]
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
    assert logits.ndim == 2

    # resize to the expected shape of SAM (256x256) if needed
    if logits.shape != (256, 256):  # shapes match already
        trafo = ResizeLongestSide(256)
        logits = trafo.apply_image(logits[..., None])

        # this transformation resizes the longest side to 256
        # if the input is not square we need to pad the other side to 256
        if logits.shape != (256, 256):
            raise NotImplementedError("Not implemented for non-square images")

            # I don't know yet how to correctly resize the logits for non-square images.
            # I have tried:
            # - padding the shorter size to 256
            # - just resizing to 256
            # but both approaches fail (the mask is not predicted correctly, probably because it is misanligned)

            # assert sum(sh == 256 for sh in logits.shape) == 1
            # pad_dim = 1 if logits.shape[0] == 256 else 0
            # pad_width = (0, 256 - logits.shape[pad_dim])
            # pad_width = (pad_width, (0, 0)) if pad_dim == 0 else ((0, 0), pad_width)
            # pad_value = logits.min()
            # logits = np.pad(logits, pad_width, mode="constant", constant_values=pad_value)

            # from skimage.transform import resize
            # logits = resize(logits, (256, 256))

            # import napari
            # v = napari.Viewer()
            # v.add_image(logits)
            # scale = tuple(float(sh) / lsh for sh, lsh in zip(mask.shape, logits.shape))
            # v.add_image(logits, scale=scale, name="logits_rescaled")
            # v.add_labels(mask + 1)
            # napari.run()

    logits = logits[None]

    assert logits.shape == (1, 256, 256), f"{logits.shape}"
    return logits


#
# other helper functions
#


def _process_box(box, original_size=None):
    box_processed = box[[1, 0, 3, 2]]
    # TODO how do we deal with aspect ratios???
    if original_size is not None:
        trafo = ResizeLongestSide(max(original_size))
        box_processed = trafo.apply_boxes(box[None], (256, 256)).squeeze()
    return box_processed


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
    tile, shape = None, None

    # set uthe precomputed state for tiled prediction
    if image_embeddings is not None and image_embeddings["input_size"] is None:
        features = image_embeddings["features"]
        shape, tile_shape, halo = features.attrs["shape"], features.attrs["tile_shape"], features.attrs["halo"]
        tile_id, tile, prompts = to_tile(prompts, shape, tile_shape, halo)
        features = features[tile_id]
        tile_image_embeddings = {
            "features": features,
            "input_size": features.attrs["input_size"],
            "original_size": features.attrs["original_size"]
        }
        util.set_precomputed(predictor, tile_image_embeddings, i)

    # set the precomputed state for normal prediction
    elif image_embeddings is not None:
        util.set_precomputed(predictor, image_embeddings, i)

    return predictor, tile, prompts, shape


def _tile_to_full_mask(mask, shape, tile, return_all, multimask_output):
    assert not return_all and not multimask_output
    full_mask = np.zeros((1,) + tuple(shape), dtype=mask.dtype)
    bb = tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end))
    full_mask[0][bb] = mask[0]
    return full_mask


#
# functions for prompted:
# - segment_from_points: use point prompts as input
# - segment_from_mask: use binary mask as input, support conversion to mask, box and point prompts
# - segment_from_box: use box prompt as input
# - segment_from_box_and_points: use box and point prompts as input
#


def segment_from_points(
    predictor, points, labels,
    image_embeddings=None,
    i=None, multimask_output=False, return_all=False,
):
    predictor, tile, prompts, shape = _initialize_predictor(
        predictor, image_embeddings, i, (points, labels), _points_to_tile
    )
    points, labels = prompts

    # predict the mask
    mask, scores, logits = predictor.predict(
        point_coords=points[:, ::-1],  # SAM has reversed XY conventions
        point_labels=labels,
        multimask_output=multimask_output,
    )

    if tile is not None:
        return _tile_to_full_mask(mask, shape, tile, return_all, multimask_output)
    if return_all:
        return mask, scores, logits
    else:
        return mask


# use original_size if the mask is downscaled w.r.t. the original image size
def segment_from_mask(
    predictor, mask,
    image_embeddings=None, i=None,
    use_mask=True, use_box=True, use_points=False,
    original_size=None, multimask_output=False,
    return_all=False, return_logits=False,
    box_extension=0,
):
    predictor, tile, mask, shape = _initialize_predictor(
        predictor, image_embeddings, i, mask, _mask_to_tile
    )

    if use_points:
        point_coords, point_labels = _compute_points_from_mask(mask, original_size=original_size)
    else:
        point_coords, point_labels = None, None
    box = _compute_box_from_mask(mask, original_size=original_size, box_extension=box_extension) if use_box else None
    logits = _compute_logits_from_mask(mask) if use_mask else None
    mask, scores, logits = predictor.predict(
        point_coords=point_coords, point_labels=point_labels,
        mask_input=logits, box=box,
        multimask_output=multimask_output, return_logits=return_logits
    )

    if tile is not None:
        return _tile_to_full_mask(mask, shape, tile, return_all, multimask_output)
    if return_all:
        return mask, scores, logits
    else:
        return mask


def segment_from_box(
    predictor, box,
    image_embeddings=None, i=None, original_size=None,
    multimask_output=False, return_all=False,
):
    predictor, tile, box, shape = _initialize_predictor(
        predictor, image_embeddings, i, box, _box_to_tile
    )
    mask, scores, logits = predictor.predict(box=_process_box(box, original_size), multimask_output=multimask_output)

    if tile is not None:
        return _tile_to_full_mask(mask, shape, tile, return_all, multimask_output)
    if return_all:
        return mask, scores, logits
    else:
        return mask


def segment_from_box_and_points(
    predictor, box, points, labels,
    image_embeddings=None, i=None, original_size=None,
    multimask_output=False, return_all=False,
):
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
        box=_process_box(box, original_size),
        multimask_output=multimask_output
    )

    if tile is not None:
        return _tile_to_full_mask(mask, shape, tile, return_all, multimask_output)
    if return_all:
        return mask, scores, logits
    else:
        return mask
