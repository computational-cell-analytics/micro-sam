import numpy as np

from segment_anything.utils.transforms import ResizeLongestSide
from . import util


# compute the bounding box. SAM expects the following input:
# box (np.ndarray or None): A length 4 array given a box prompt to the model, in XYXY format.
def _compute_box(mask, original_size=None):
    coords = np.where(mask == 1)
    box = np.array([
        coords[1].min(), coords[0].min(),
        coords[1].max() + 1, coords[0].max() + 1,
    ])
    # TODO how do we deal with aspect ratios???
    if original_size is not None:
        trafo = ResizeLongestSide(max(original_size))
        box = trafo.apply_boxes(box[None], (256, 256)).squeeze()
    return box


def _process_box(box, original_size=None):
    box_processed = box[[1, 0, 3, 2]]
    # TODO how do we deal with aspect ratios???
    if original_size is not None:
        trafo = ResizeLongestSide(max(original_size))
        box_processed = trafo.apply_boxes(box[None], (256, 256)).squeeze()
    return box_processed


def _compute_logits(mask, eps=1e-3):

    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

    # resize to the expected shape of SAM (256x256) if needed
    if logits.shape == (256, 256):  # shapes match already
        logits = logits[None]
    else:  # shapes don't match, need to resize
        trafo = ResizeLongestSide(256)
        logits = trafo.apply_image(logits[..., None])[None]

    assert logits.shape == (1, 256, 256), f"{logits.shape}"
    return logits


def segment_from_points(
    predictor, points, labels,
    image_embeddings=None,
    i=None, multimask_output=False, return_all=False,
):
    # set the precomputed state
    if image_embeddings is not None:
        util.set_precomputed(predictor, image_embeddings, i)

    # predict the mask
    mask, scores, logits = predictor.predict(
        point_coords=points[:, ::-1],  # SAM has reversed XY conventions
        point_labels=labels,
        multimask_output=multimask_output,
    )
    if return_all:
        return mask, scores, logits
    else:
        return mask


# use original_size if the mask is downscaled w.r.t. the original image size
def segment_from_mask(
    predictor, mask,
    image_embeddings=None, i=None,
    use_mask=True, use_box=True,
    original_size=None, multimask_output=False, return_all=False,
):
    if image_embeddings is not None:
        util.set_precomputed(predictor, image_embeddings, i)
    box = _compute_box(mask, original_size=original_size) if use_box else None
    logits = _compute_logits(mask) if use_mask else None
    mask, scores, logits = predictor.predict(mask_input=logits, box=box, multimask_output=multimask_output)
    if return_all:
        return mask, scores, logits
    else:
        return mask


def segment_from_box(
    predictor, box,
    image_embeddings=None, i=None, original_size=None,
    multimask_output=False, return_all=False,
):
    if image_embeddings is not None:
        util.set_precomputed(predictor, image_embeddings, i)
    mask, scores, logits = predictor.predict(box=_process_box(box, original_size), multimask_output=multimask_output)
    if return_all:
        return mask, scores, logits
    else:
        return mask
