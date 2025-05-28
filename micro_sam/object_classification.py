import os
from joblib import load
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from nifty.tools import blocking, takeDict
from skimage.measure import regionprops_table
from skimage.transform import resize

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from .import util


def _compute_object_features_impl(embeddings, segmentation, resize_embedding_shape):
    # Get the embeddings and put the channel axis last.
    embeddings = embeddings.transpose(1, 2, 0)

    # Pad the segmentation to be of square shape.
    shape = segmentation.shape
    if shape[0] == shape[1]:
        segmentation_rescaled = segmentation
    elif shape[0] > shape[1]:
        segmentation_rescaled = np.pad(segmentation, ((0, 0), (0, shape[0] - shape[1])))
    elif shape[1] > shape[0]:
        segmentation_rescaled = np.pad(segmentation, ((0, shape[1] - shape[0]), (0, 0)))
    assert segmentation_rescaled.shape[0] == segmentation_rescaled.shape[1]
    shape = segmentation_rescaled.shape

    # Resize the segmentation and embeddings to be of the same size.

    # We first resize the embedding, to an intermediate shape (passed as parameter).
    # The motivation for this is to avoid loosing smaller segmented objects when resizing the segmentation
    # to the original embedding shape. On the other hand, we avoid resizing the embeddings to the full segmentation
    # shape for efficiency reasons.
    resize_shape = tuple(min(rsh, sh) for rsh, sh in zip(resize_embedding_shape, shape)) + (embeddings.shape[-1],)
    embeddings = resize(embeddings, resize_shape, preserve_range=True).astype(embeddings.dtype)

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

    return seg_ids, features


def _create_seg_and_embed_generator(segmentation, image_embeddings, is_tiled, is_3d):
    assert is_tiled or is_3d

    if is_tiled:
        tile_embeds = image_embeddings["features"]
        tile_shape, halo = tile_embeds.attrs["tile_shape"], tile_embeds.attrs["halo"]
        tiling = blocking([0, 0], tile_embeds.attrs["shape"], tile_shape)
        length = tiling.numberOfBlocks * segmentation.shape[0] if is_3d else tiling.numberOfBlocks
    else:
        tiling = None
        length = segmentation.shape[0]

    if is_3d and is_tiled:  # 3d data with tiling
        def generator():
            for z in range(segmentation.shape[0]):
                seg_z = segmentation[z]
                for block_id in range(tiling.numberOfBlocks):
                    block = tiling.getBlockWithHalo(block_id, halo)

                    # Get the embeddings and segmentation for this block and slice.
                    embeds = tile_embeds[str(block_id)][z].squeeze()

                    bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
                    seg = seg_z[bb]

                    yield seg, embeds

    elif is_3d:  # 3d data no tiling
        def generator():
            for z in range(length):
                seg = segmentation[z]
                embeds = image_embeddings["features"][z].squeeze()
                yield seg, embeds

    else:  # 2d data with tiling
        def generator():
            for block_id in range(length):
                block = tiling.getBlockWithHalo(block_id, halo)

                # Get the embeddings and segmentation for this block.
                embeds = tile_embeds[str(block_id)][:].squeeze()
                bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
                seg = segmentation[bb]

                yield seg, embeds

    return generator, length


def compute_object_features(
    image_embeddings: util.ImageEmbeddings,
    segmentation: np.ndarray,
    resize_embedding_shape: Tuple[int, int] = (256, 256),
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute object features based on SAM embeddings.

    Args:
        image_embeddings: The precomputed image embeddings.
        segmentation: The segmentation for which to compute the features.
        resize_embedding_shape: Shape for intermediate resizing of the embeddings.
        verbose: Whether to print a progressbar for the computation.

    Returns:
        The segmentation ids.
        The object features.
    """
    is_tiled = image_embeddings["input_size"] is None
    is_3d = segmentation.ndim == 3

    # If we have simple embeddings, i.e. 2d without tiling, then we can directly compute the features.
    if not is_tiled and not is_3d:
        embeddings = image_embeddings["features"].squeeze()
        return _compute_object_features_impl(embeddings, segmentation, resize_embedding_shape)

    # Otherwise, we compute the features by iterating over slices and/or tiles,
    # compute the features for each slice / tile and accumulate them.

    # Fist, we compute the segmentation ids and initialize the required data structures.
    seg_ids = np.unique(segmentation).tolist()
    if seg_ids[0] == 0:
        seg_ids = seg_ids[1:]
    visited = {seg_id: False for seg_id in seg_ids}

    n_features = 257  # Don't hard-code?
    features = np.zeros((len(seg_ids), n_features), dtype="float32")

    # Then, we create a generator for iterating over the slices and / or tile.
    # This generator returns the respective segmentation and embeddings.
    seg_embed_generator, n_gen = _create_seg_and_embed_generator(
        segmentation, image_embeddings, is_tiled=is_tiled, is_3d=is_3d
    )

    for seg, embeds in tqdm(
        seg_embed_generator(), total=n_gen, disable=not verbose, desc="Compute object features"
    ):
        # Compute this seg ids and features.
        this_seg_ids, this_features = _compute_object_features_impl(embeds, seg, resize_embedding_shape)
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
            # Get ths sizes, which are needed for computing the mean.
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
    prediction = {seg_id: class_pred for seg_id, class_pred in zip(seg_ids, object_prediction)}
    prediction[0] = 0
    return takeDict(prediction, segmentation)


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
) -> List[np.ndarray]:
    """Run prediction with a pretrained object classifier on a series of images.

    Args:
        images: The images, either given as a list of numpy array or filepaths.
        segmentations: The segmentaitons, either given as a list of numpy array or filepaths.
        predictor:
        rf_path:
        image_key:
        segmentation_key:
        project_prediction:
        ndim:

    Returns:
        The predictions.
    """
    assert len(images) == len(segmentations)
    rf = load(rf_path)
    predictions = []
    for image, segmentation in tqdm(
        zip(images, segmentations), total=len(images), desc="Run prediction with object classifier"
    ):
        embeddings = util.precompute_image_embeddings(predictor, image, verbose=False, ndim=ndim)
        seg_ids, features = compute_object_features(embeddings, segmentation, verbose=False)
        prediction = rf.predict(features)
        if project_prediction:
            prediction = project_prediction_to_segmentation(segmentation, prediction, seg_ids)
        predictions.append(prediction)
    return predictions
