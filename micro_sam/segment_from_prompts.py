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
