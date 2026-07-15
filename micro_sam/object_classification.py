import os
from joblib import load
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from bioimage_cpp.utils import Blocking, take_dict

from skimage.measure import regionprops_table
from skimage.transform import resize

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from .import util
from .v1.util import precompute_image_embeddings


def _anyup_object_resize(embeds_chw, image_region, target_hw, upsampler):
    # Upsample a (C, h, w) embedding to (target_h, target_w, C) with AnyUp, using the image region
    # as guidance. The encoders pad the image to a square, so we square-pad the image here to match
    # how the segmentation is padded.
    import torch
    import torch.nn.functional as F
    from .pixel_classification import _to_anyup_image

    # AnyUp's internal convolutions need at least 2 px per spatial dim. Fall back to plain
    # interpolation for degenerate regions.
    if min(embeds_chw.shape[-2:]) < 2 or min(target_hw) < 2 or min(image_region.shape[:2]) < 2:
        return resize(
            embeds_chw.transpose(1, 2, 0), tuple(target_hw) + (embeds_chw.shape[0],), preserve_range=True
        ).astype("float32")

    device = next(upsampler.parameters()).device
    image_tensor = _to_anyup_image(image_region, device)  # (1, 3, H, W)
    h, w = image_tensor.shape[-2:]
    image_tensor = F.pad(image_tensor, (0, max(h - w, 0), 0, max(w - h, 0)))
    feature_tensor = torch.from_numpy(np.ascontiguousarray(embeds_chw)).to(device).float().unsqueeze(0)
    from .pixel_classification import ANYUP_Q_CHUNK_SIZE
    with torch.no_grad():
        upsampled = upsampler(
            image_tensor, feature_tensor, output_size=tuple(target_hw), q_chunk_size=ANYUP_Q_CHUNK_SIZE
        )
    return upsampled[0].permute(1, 2, 0).cpu().numpy().astype("float32")


def _compute_object_features_impl(
    embeddings, segmentation, resize_embedding_shape, image_region=None, upsampler=None, pbar_update=None,
):
    # Keep the raw (C, h, w) embedding for AnyUp, which needs the channel axis first.
    embeddings_chw = embeddings

    # Bring the segmentation to a square shape matching the (square) embedding. The encoders pad the
    # image to a square, so we zero-pad the segmentation to match.
    shape = segmentation.shape
    if shape[0] == shape[1]:
        segmentation_rescaled = segmentation
    elif shape[0] > shape[1]:
        segmentation_rescaled = np.pad(segmentation, ((0, 0), (0, shape[0] - shape[1])))
    else:
        segmentation_rescaled = np.pad(segmentation, ((0, shape[1] - shape[0]), (0, 0)))
    assert segmentation_rescaled.shape[0] == segmentation_rescaled.shape[1]
    shape = segmentation_rescaled.shape

    # Resize the segmentation and embeddings to be of the same size.

    # We first resize the embedding, to an intermediate shape (passed as parameter).
    # The motivation for this is to avoid loosing smaller segmented objects when resizing the segmentation
    # to the original embedding shape. On the other hand, we avoid resizing the embeddings to the full segmentation
    # shape for efficiency reasons.
    resize_hw = tuple(min(rsh, sh) for rsh, sh in zip(resize_embedding_shape, shape))
    if upsampler is not None:
        # AnyUp upsamples the embedding using the image region instead of plain interpolation.
        embeddings = _anyup_object_resize(embeddings_chw, image_region, resize_hw, upsampler)
    else:
        embeddings = embeddings_chw.transpose(1, 2, 0)  # put the channel axis last
        embeddings = resize(
            embeddings, resize_hw + (embeddings.shape[-1],), preserve_range=True
        ).astype(embeddings.dtype)

    segmentation_rescaled = resize(
        segmentation_rescaled, embeddings.shape[:2], order=0, anti_aliasing=False, preserve_range=True
    ).astype(segmentation.dtype)

    # Which features do we use?
    all_features = regionprops_table(
        segmentation_rescaled, intensity_image=embeddings, properties=("label", "area", "mean_intensity"),
    )
    seg_ids = all_features["label"]
    features = pd.DataFrame(all_features)[
        ["area"] + [f"mean_intensity-{i}" for i in range(embeddings.shape[-1])]
    ].values

    if pbar_update is not None:
        pbar_update(1)
    return seg_ids, features


def _create_seg_and_embed_generator(segmentation, image_embeddings, is_tiled, is_3d, image=None):
    assert is_tiled or is_3d

    if is_tiled:
        tile_embeds = image_embeddings["features"]
        tile_shape, halo = tile_embeds.attrs["tile_shape"], tile_embeds.attrs["halo"]
        tiling = Blocking([0, 0], tile_embeds.attrs["shape"], tile_shape)
        length = tiling.number_of_blocks * segmentation.shape[0] if is_3d else tiling.number_of_blocks
    else:
        tiling = None
        length = segmentation.shape[0]

    # The generators yield (segmentation, embeddings, image_region) per slice / tile. The image
    # region is None unless an image is given (only needed for AnyUp upsampling).
    if is_3d and is_tiled:  # 3d data with tiling
        def generator():
            for z in range(segmentation.shape[0]):
                seg_z = segmentation[z]
                image_z = None if image is None else image[z]
                for block_id in range(tiling.number_of_blocks):
                    block = tiling.get_block_with_halo(block_id, halo)

                    # Get the embeddings and segmentation for this block and slice.
                    embeds = tile_embeds[str(block_id)][z].squeeze()

                    bb = tuple(slice(beg, end) for beg, end in zip(block.outer_block.begin, block.outer_block.end))
                    seg = seg_z[bb]
                    image_region = None if image_z is None else image_z[bb]

                    yield seg, embeds, image_region

    elif is_3d:  # 3d data no tiling
        def generator():
            for z in range(length):
                seg = segmentation[z]
                embeds = image_embeddings["features"][z].squeeze()
                image_region = None if image is None else image[z]
                yield seg, embeds, image_region

    else:  # 2d data with tiling
        def generator():
            for block_id in range(length):
                block = tiling.get_block_with_halo(block_id, halo)

                # Get the embeddings and segmentation for this block.
                embeds = tile_embeds[str(block_id)][:].squeeze()
                bb = tuple(slice(beg, end) for beg, end in zip(block.outer_block.begin, block.outer_block.end))
                seg = segmentation[bb]
                image_region = None if image is None else image[bb]

                yield seg, embeds, image_region

    return generator, length


def compute_object_features(
    image_embeddings: util.ImageEmbeddings,
    segmentation: np.ndarray,
    resize_embedding_shape: Tuple[int, int] = (256, 256),
    verbose: bool = True,
    image: Optional[np.ndarray] = None,
    upsampler=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute object features based on SAM embeddings.

    Args:
        image_embeddings: The precomputed image embeddings.
        segmentation: The segmentation for which to compute the features.
        resize_embedding_shape: Shape for intermediate resizing of the embeddings.
        verbose: Whether to print a progressbar for the computation.
        image: The original image, required when `upsampler` is given (AnyUp uses it as guidance).
        upsampler: An optional AnyUp model (see `pixel_classification.get_anyup_upsampler`). When
            given, the embedding is upsampled with AnyUp using the image instead of plain interpolation.

    Returns:
        The segmentation ids.
        The object features.
    """
    if upsampler is not None and image is None:
        raise ValueError("An 'image' is required when an AnyUp 'upsampler' is given.")

    is_tiled = image_embeddings["input_size"] is None
    is_3d = segmentation.ndim == 3

    # If we have simple embeddings, i.e. 2d without tiling, then we can directly compute the features.
    if not is_tiled and not is_3d:
        embeddings = image_embeddings["features"].squeeze()
        # AnyUp is the slow part, so show a progress bar (which also drives napari's activity dots)
        # for the single upsampling call.
        anyup_pbar = tqdm(
            total=1, desc="Upsampling with AnyUp", disable=(not verbose) or (upsampler is None)
        )
        result = _compute_object_features_impl(
            embeddings, segmentation, resize_embedding_shape, image_region=image, upsampler=upsampler,
            pbar_update=anyup_pbar.update if upsampler is not None else None,
        )
        anyup_pbar.close()
        return result

    # Otherwise, we compute the features by iterating over slices and/or tiles,
    # compute the features for each slice / tile and accumulate them.

    # First, we compute the segmentation ids and initialize the required data structures.
    seg_ids = np.unique(segmentation).tolist()
    if seg_ids[0] == 0:
        seg_ids = seg_ids[1:]
    visited = {seg_id: False for seg_id in seg_ids}

    # Then, we create a generator for iterating over the slices and / or tile.
    # This generator returns the respective segmentation and embeddings.
    seg_embed_generator, n_gen = _create_seg_and_embed_generator(
        segmentation, image_embeddings, is_tiled=is_tiled, is_3d=is_3d, image=image
    )

    # Feature vector = object area + per-channel embedding mean, so the width follows the embedding
    # channel count (256 for SAM1/SAM2, larger for e.g. DINO encoders).
    n_channels = int(next(seg_embed_generator())[1].shape[0])
    features = np.zeros((len(seg_ids), n_channels + 1), dtype="float32")

    # With AnyUp, label the bar accordingly since the upsampling is the slow part.
    desc = "Upsampling with AnyUp" if upsampler is not None else "Compute object features"
    for seg, embeds, image_region in tqdm(
        seg_embed_generator(), total=n_gen, disable=not verbose, desc=desc
    ):
        # Compute this seg ids and features.
        this_seg_ids, this_features = _compute_object_features_impl(
            embeds, seg, resize_embedding_shape, image_region=image_region, upsampler=upsampler
        )
        this_seg_ids = this_seg_ids.tolist()

        # Find which of the seg ids are new (= processed for the first time).
        # And the seg ids that were already visited.
        new_idx = np.array([seg_ids.index(seg_id) for seg_id in this_seg_ids if not visited[seg_id]], dtype="int")
        visited_idx = np.array([seg_ids.index(seg_id) for seg_id in this_seg_ids if visited[seg_id]], dtype="int")

        # Get the corresponding feature indices.
        this_new_idx = np.array(
            [this_seg_ids.index(seg_id) for seg_id in this_seg_ids if not visited[seg_id]], dtype="int"
        )
        this_visited_idx = np.array(
            [this_seg_ids.index(seg_id) for seg_id in this_seg_ids if visited[seg_id]], dtype="int"
        )

        # New features can be written directly.
        features[new_idx] = this_features[this_new_idx]

        # Features that were already visited can be merged.
        if len(visited_idx) > 0:
            # Get the sizes, which are needed for computing the mean.
            prev_size = features[visited_idx, 0:1]
            this_size = this_features[this_visited_idx, 0:1]

            # The sizes themselve are merged by addition.
            features[visited_idx, 0] += this_features[this_visited_idx, 0]

            # Mean values are merged via weighted sum.
            features[visited_idx, 1:] = (
                prev_size * features[visited_idx, 1:] + this_size * this_features[this_visited_idx, 1:]
            ) / (prev_size + this_size)

        # Set all seg ids from this block to visited.
        visited.update({seg_id: True for seg_id in this_seg_ids})

    return np.array(seg_ids), features


def project_prediction_to_segmentation(
    segmentation: np.ndarray,
    object_prediction: np.ndarray,
    seg_ids: np.ndarray
) -> np.ndarray:
    """Project object level prediction to the corresponding segmentation to obtain a pixel level prediction.

    Args:
        segmentation: The segmentation from which the object prediction is derived.
        object_prediction: The object prediction.
        seg_ids: The segmentation ids matching the object prediction.

    Returns:
        The pixel level object prediction, corresponding to a semantic segmentation.
    """
    assert len(object_prediction) == len(seg_ids)

    # bioimage_cpp.take_dict only accepts these integer label dtypes. Napari label layers may use
    # smaller dtypes such as uint8, so cast only for the relabeling call.
    if segmentation.dtype not in (np.uint32, np.uint64, np.int32, np.int64):
        if segmentation.dtype == bool or np.issubdtype(segmentation.dtype, np.unsignedinteger):
            segmentation_for_relabeling = segmentation.astype("uint32", copy=False)
        elif np.issubdtype(segmentation.dtype, np.signedinteger):
            segmentation_for_relabeling = segmentation.astype("int32", copy=False)
        else:
            raise TypeError(f"The segmentation must have an integer dtype, got dtype={segmentation.dtype}.")
    else:
        segmentation_for_relabeling = segmentation

    prediction = {seg_id: class_pred for seg_id, class_pred in zip(seg_ids, object_prediction)}
    # Find missing segmentation ids. This will include the background id, but may include other ids of small objects.
    # Such objects may get removed in the resizing operations.
    missing_ids = np.setdiff1d(np.unique(segmentation), seg_ids)
    prediction.update({missing_id: 0 for missing_id in missing_ids})
    return take_dict(prediction, segmentation_for_relabeling)


# TODO handle images / segmentations as file paths
# TODO think about the function signature, specially how exactly we pass model and optional embedding path.
# TODO halo and tile shape
# TODO add heuristic for ndim
def run_prediction_with_object_classifier(
    images: Sequence[Union[str, os.PathLike, np.ndarray]],
    segmentations: Sequence[Union[str, os.PathLike, np.ndarray]],
    predictor,
    rf_path: Union[str, os.PathLike],
    image_key: Optional[str] = None,
    segmentation_key: Optional[str] = None,
    project_prediction: bool = True,
    ndim: Optional[int] = None,
    upsampler=None,
    model_type: Optional[str] = None,
) -> List[np.ndarray]:
    """Run prediction with a pretrained object classifier on a series of images.

    Args:
        images: The images, either given as a list of numpy array or filepaths.
        segmentations: The segmentations, either given as a list of numpy array or filepaths.
        predictor:
        rf_path:
        image_key:
        segmentation_key:
        project_prediction:
        ndim:
        upsampler: An optional AnyUp model (see `pixel_classification.get_anyup_upsampler`) to
            upsample the embeddings with the image as guidance instead of plain interpolation.
            Use the same setting that the classifier was trained with.

    Returns:
        The predictions.
    """
    assert len(images) == len(segmentations)
    # Stored as {'rf': ..., 'model_spec': ...}; older files are a bare classifier.
    obj = load(rf_path)
    rf = obj["rf"] if isinstance(obj, dict) and "rf" in obj else obj
    compute_embeddings = util.get_embedding_function(model_type) if model_type is not None \
        else precompute_image_embeddings
    predictions = []
    for image, segmentation in tqdm(
        zip(images, segmentations), total=len(images), desc="Run prediction with object classifier"
    ):
        if isinstance(image, (str, os.PathLike)):
            image = util.load_image_data(image, key=image_key)
        if isinstance(segmentation, (str, os.PathLike)):
            segmentation = util.load_image_data(segmentation, key=segmentation_key)
        embeddings = compute_embeddings(predictor, image, verbose=False, ndim=ndim)
        seg_ids, features = compute_object_features(
            embeddings, segmentation, verbose=False, image=image if upsampler is not None else None,
            upsampler=upsampler,
        )
        prediction = rf.predict(features)
        if project_prediction:
            prediction = project_prediction_to_segmentation(segmentation, prediction, seg_ids)
        predictions.append(prediction)
    return predictions
