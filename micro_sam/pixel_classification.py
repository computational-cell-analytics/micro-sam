import os
from joblib import dump, load
from multiprocessing import cpu_count
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from bioimage_cpp.utils import Blocking

from sklearn.ensemble import RandomForestClassifier

from skimage.transform import resize

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from . import util
from .v1.util import precompute_image_embeddings

# Default in-plane grid size (longest side) for the per-pixel feature grid.
# Non-tiled images use 'grid_size'; tiled images use the larger 'max_grid_size', since tiling
# yields more genuine embedding detail (n_tiles x the per-tile resolution).
DEFAULT_GRID_SIZE = 256
DEFAULT_MAX_GRID_SIZE = 512


def _grid_shape(image_hw: Tuple[int, int], target_long: int) -> Tuple[Tuple[int, int], float]:
    """Downsample an in-plane shape so the longest side equals 'target_long', preserving aspect."""
    height, width = image_hw
    scale = target_long / max(height, width)
    grid = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    return grid, scale


def _aspect_crop(embeddings: np.ndarray, block_hw: Tuple[int, int]) -> np.ndarray:
    """Crop a square (C, H, W) embedding to the aspect ratio of the (non-square) block it covers.

    SAM pads the input to a square before encoding, so the embedding is square even for
    non-square images. We crop away the padded region to recover the image-aligned part.
    """
    emb_h, emb_w = embeddings.shape[-2:]
    block_h, block_w = block_hw
    if block_h == block_w:
        return embeddings
    if block_h > block_w:
        return embeddings[:, :, :int(round((block_w / block_h) * emb_w))]
    return embeddings[:, :int(round((block_h / block_w) * emb_h)), :]


def _resize_to_grid(embeddings: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a (C, H, W) embedding to a (target_h, target_w, C) feature image."""
    feature_image = embeddings.transpose(1, 2, 0)
    resize_shape = (target_hw[0], target_hw[1], feature_image.shape[-1])
    return resize(feature_image, resize_shape, preserve_range=True).astype("float32")


def _block_to_grid(embeddings: np.ndarray, block_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> np.ndarray:
    """Crop a block embedding to its aspect ratio and resize it to the target grid shape."""
    return _resize_to_grid(_aspect_crop(embeddings, block_hw), target_hw)


def _compute_tiled_feature_image(features, image_hw, max_grid_size, z=None):
    """Assemble a downsampled (GH, GW, C) feature image for a single 2d (tiled) plane.

    For 3d data 'z' selects the slice of each tile's embedding; for 2d data 'z' is None.
    """
    tile_shape, halo, shape = features.attrs["tile_shape"], features.attrs["halo"], features.attrs["shape"]
    tiling = Blocking([0, 0], list(shape), list(tile_shape))
    grid, scale = _grid_shape(image_hw, max_grid_size)

    feature_image = None
    for block_id in range(tiling.number_of_blocks):
        tile_embeds = features[str(block_id)]
        embeds = np.asarray(tile_embeds[:] if z is None else tile_embeds[z]).squeeze()
        if feature_image is None:
            feature_image = np.zeros(grid + (embeds.shape[0],), dtype="float32")

        block = tiling.get_block_with_halo(block_id, list(halo))
        outer, inner_local = block.outer_block, block.inner_block_local
        outer_hw = (outer.end[0] - outer.begin[0], outer.end[1] - outer.begin[1])

        # Crop the (square-padded) tile embedding to the outer block aspect, then slice the inner block.
        embeds = _aspect_crop(embeds, outer_hw)
        tile_scale = (embeds.shape[-2] / outer_hw[0], embeds.shape[-1] / outer_hw[1])
        iy0, iy1 = int(round(inner_local.begin[0] * tile_scale[0])), int(round(inner_local.end[0] * tile_scale[0]))
        ix0, ix1 = int(round(inner_local.begin[1] * tile_scale[1])), int(round(inner_local.end[1] * tile_scale[1]))
        inner_embeds = embeds[:, iy0:iy1, ix0:ix1]

        # Map the inner block to its position in the global grid and place the resized features there.
        gy0, gy1 = outer.begin[0] + inner_local.begin[0], outer.begin[0] + inner_local.end[0]
        gx0, gx1 = outer.begin[1] + inner_local.begin[1], outer.begin[1] + inner_local.end[1]
        by0, by1 = int(round(gy0 * scale)), int(round(gy1 * scale))
        bx0, bx1 = int(round(gx0 * scale)), int(round(gx1 * scale))
        feature_image[by0:by1, bx0:bx1] = _resize_to_grid(inner_embeds, (by1 - by0, bx1 - bx0))

    return feature_image, grid


def compute_pixel_features(
    image_embeddings: util.ImageEmbeddings,
    image_shape: Tuple[int, ...],
    grid_size: int = DEFAULT_GRID_SIZE,
    max_grid_size: int = DEFAULT_MAX_GRID_SIZE,
    verbose: bool = True,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Compute per-pixel features from SAM embeddings for pixel classification.

    Each spatial location of the SAM embedding becomes a feature vector. The features are
    computed on a downsampled in-plane grid (preserving the image aspect ratio) to keep them
    tractable, and the grid prediction is later projected back to the full image resolution.

    Args:
        image_embeddings: The precomputed image embeddings.
        image_shape: The spatial shape of the image, (H, W) for 2d or (Z, H, W) for 3d.
        grid_size: In-plane grid size (longest side) for non-tiled images.
        max_grid_size: In-plane grid size (longest side) for tiled images.
        verbose: Whether to print a progressbar for the computation.

    Returns:
        The per-pixel features, of shape (N, C) flattened over the grid in row-major order.
        The grid shape, (gh, gw) for 2d or (Z, gh, gw) for 3d.
    """
    is_tiled = image_embeddings["input_size"] is None
    is_3d = len(image_shape) == 3
    features = image_embeddings["features"]

    if is_3d:
        depth = image_shape[0]
        image_hw = (image_shape[1], image_shape[2])
        planes = []
        for z in tqdm(range(depth), total=depth, disable=not verbose, desc="Compute pixel features"):
            if is_tiled:
                plane, grid = _compute_tiled_feature_image(features, image_hw, max_grid_size, z=z)
            else:
                embeds = np.asarray(features[z]).squeeze()
                grid, _ = _grid_shape(image_hw, grid_size)
                plane = _block_to_grid(embeds, image_hw, grid)
            planes.append(plane)
        feature_image = np.stack(planes)
        grid_shape = (depth,) + grid
    else:
        image_hw = (image_shape[0], image_shape[1])
        if is_tiled:
            feature_image, grid = _compute_tiled_feature_image(features, image_hw, max_grid_size)
        else:
            embeds = np.asarray(features).squeeze()
            grid, _ = _grid_shape(image_hw, grid_size)
            feature_image = _block_to_grid(embeds, image_hw, grid)
        grid_shape = grid

    return feature_image.reshape(-1, feature_image.shape[-1]), grid_shape


def accumulate_pixel_labels(annotation: np.ndarray, grid_shape: Tuple[int, ...]) -> np.ndarray:
    """Resize a full-resolution annotation onto the feature grid and flatten it.

    Args:
        annotation: The pixel annotations (scribbles), of shape (H, W) or (Z, H, W).
        grid_shape: The feature grid shape returned by `compute_pixel_features`.

    Returns:
        The labels flattened over the grid, aligned row-wise with the features.
    """
    def resize_plane(plane, target_hw):
        return resize(
            plane, target_hw, order=0, anti_aliasing=False, preserve_range=True
        ).astype(annotation.dtype)

    if len(grid_shape) == 2:
        return resize_plane(annotation, grid_shape).reshape(-1)

    depth, grid_h, grid_w = grid_shape
    labels = np.empty((depth, grid_h, grid_w), dtype=annotation.dtype)
    for z in range(depth):
        labels[z] = resize_plane(annotation[z], (grid_h, grid_w))
    return labels.reshape(-1)


def project_prediction_to_image(
    grid_prediction: np.ndarray, grid_shape: Tuple[int, ...], image_shape: Tuple[int, ...]
) -> np.ndarray:
    """Project a per-grid-pixel prediction back to the full image resolution.

    Args:
        grid_prediction: The flat prediction over the grid (one entry per grid pixel).
        grid_shape: The feature grid shape returned by `compute_pixel_features`.
        image_shape: The full image spatial shape, (H, W) or (Z, H, W).

    Returns:
        The pixel level prediction at the image resolution, a semantic segmentation.
    """
    grid = np.asarray(grid_prediction).reshape(grid_shape)

    def resize_plane(plane, target_hw):
        return resize(plane, target_hw, order=0, anti_aliasing=False, preserve_range=True).astype(grid.dtype)

    if len(grid_shape) == 2:
        return resize_plane(grid, image_shape)

    prediction = np.empty(image_shape, dtype=grid.dtype)
    for z in range(image_shape[0]):
        prediction[z] = resize_plane(grid[z], image_shape[1:])
    return prediction


def train_pixel_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    previous_features: Optional[np.ndarray] = None,
    previous_labels: Optional[np.ndarray] = None,
    n_estimators: int = 200,
    max_depth: int = 10,
    n_jobs: Optional[int] = None,
    **rf_kwargs,
) -> RandomForestClassifier:
    """Train a random forest on per-pixel features and labels.

    Pixels with label 0 are treated as unlabeled and excluded from training. This is the
    shared training core used by both the interactive annotator and the batch training
    function `run_training_with_pixel_classifier`.

    Args:
        features: The per-pixel features, of shape (N, C).
        labels: The per-pixel labels, of shape (N,). Label 0 is treated as unlabeled.
        previous_features: Features accumulated from previously annotated images, to train on jointly.
        previous_labels: Labels matching `previous_features`.
        n_estimators: The number of trees in the random forest.
        max_depth: The maximum depth of each tree.
        n_jobs: The number of parallel jobs for training. By default uses all available cores.
        rf_kwargs: Additional keyword arguments for the `RandomForestClassifier`.

    Returns:
        The trained random forest.
    """
    assert len(features) == len(labels)
    valid = labels != 0
    X, y = features[valid], labels[valid]

    if previous_features is not None:
        assert previous_labels is not None and len(previous_features) == len(previous_labels)
        X = np.concatenate([previous_features, X], axis=0)
        y = np.concatenate([previous_labels, y], axis=0)

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        n_jobs=cpu_count() if n_jobs is None else n_jobs, **rf_kwargs,
    )
    rf.fit(X, y)
    return rf


# TODO think about the function signature, specially how exactly we pass model and optional embedding path.
# TODO halo and tile shape.
def run_training_with_pixel_classifier(
    images: Sequence[Union[str, os.PathLike, np.ndarray]],
    annotations: Sequence[Union[str, os.PathLike, np.ndarray]],
    predictor,
    rf_path: Union[str, os.PathLike],
    image_key: Optional[str] = None,
    annotation_key: Optional[str] = None,
    ndim: Optional[int] = None,
    n_estimators: int = 200,
    max_depth: int = 10,
    n_jobs: Optional[int] = None,
    **rf_kwargs,
) -> RandomForestClassifier:
    """Train a pixel classifier on a series of images and (sparse) annotations.

    Object features are computed from the SAM embeddings for each image, the annotations
    are mapped onto the corresponding feature grid, and a random forest is trained on all
    annotated pixels (pixels with label 0 are treated as unlabeled and ignored).

    Args:
        images: The images, either given as a list of numpy arrays or filepaths.
        annotations: The pixel annotations (scribbles), either as numpy arrays or filepaths.
            Pixels with value 0 are treated as unlabeled. Must match the order of `images`.
        predictor: The Segment Anything predictor used to compute the embeddings.
        rf_path: The filepath where the trained random forest will be saved.
        image_key: The key for the image data, for filepath inputs (e.g. an internal dataset path).
        annotation_key: The key for the annotation data, for filepath inputs.
        ndim: The dimensionality of the data. If not given will be derived from the data.
        n_estimators: The number of trees in the random forest.
        max_depth: The maximum depth of each tree.
        n_jobs: The number of parallel jobs for training. By default uses all available cores.
        rf_kwargs: Additional keyword arguments for the `RandomForestClassifier`.

    Returns:
        The trained random forest.
    """
    if len(images) != len(annotations):
        raise ValueError(
            f"Expect the same number of images and annotations, got {len(images)}, {len(annotations)}."
        )

    all_features, all_labels = [], []
    for image, annotation in tqdm(
        zip(images, annotations), total=len(images), desc="Compute features for pixel classifier training"
    ):
        if isinstance(image, (str, os.PathLike)):
            image = util.load_image_data(image, key=image_key)
        if isinstance(annotation, (str, os.PathLike)):
            annotation = util.load_image_data(annotation, key=annotation_key)

        this_ndim = ndim if ndim is not None else (image.ndim - 1 if image.shape[-1] == 3 else image.ndim)
        image_shape = image.shape[:this_ndim]
        embeddings = precompute_image_embeddings(predictor, image, verbose=False, ndim=this_ndim)
        features, grid_shape = compute_pixel_features(embeddings, image_shape, verbose=False)
        labels = accumulate_pixel_labels(annotation, grid_shape)

        valid = labels != 0
        if valid.any():
            all_features.append(features[valid])
            all_labels.append(labels[valid])

    if not all_features:
        raise ValueError("None of the provided annotations contain labeled pixels.")

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    rf = train_pixel_classifier(
        features, labels, n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, **rf_kwargs,
    )

    dump(rf, rf_path)
    return rf


# TODO handle images as file paths.
# TODO think about the function signature, specially how exactly we pass model and optional embedding path.
# TODO halo and tile shape.
# TODO add heuristic for ndim.
def run_prediction_with_pixel_classifier(
    images: Sequence[Union[str, os.PathLike, np.ndarray]],
    predictor,
    rf_path: Union[str, os.PathLike],
    image_key: Optional[str] = None,
    ndim: Optional[int] = None,
) -> List[np.ndarray]:
    """Run prediction with a pretrained pixel classifier on a series of images.

    Args:
        images: The images, either given as a list of numpy arrays or filepaths.
        predictor: The Segment Anything predictor used to compute the embeddings.
        rf_path: The filepath to the trained random forest.
        image_key: The key for the image data, for filepath inputs (e.g. an internal dataset path).
        ndim: The dimensionality of the data. If not given will be derived from the data.

    Returns:
        The pixel level predictions.
    """
    rf = load(rf_path)
    predictions = []
    for image in tqdm(images, total=len(images), desc="Run prediction with pixel classifier"):
        if isinstance(image, (str, os.PathLike)):
            image = util.load_image_data(image, key=image_key)
        this_ndim = ndim if ndim is not None else (image.ndim - 1 if image.shape[-1] == 3 else image.ndim)
        image_shape = image.shape[:this_ndim]
        embeddings = precompute_image_embeddings(predictor, image, verbose=False, ndim=this_ndim)
        features, grid_shape = compute_pixel_features(embeddings, image_shape, verbose=False)
        prediction = rf.predict(features)
        predictions.append(project_prediction_to_image(prediction, grid_shape, image_shape))
    return predictions
