"""Classical (hand-crafted) image features to add on top of the foundation model features.

The pixel features are an ilastik-like multi-scale filter bank, computed with `bioimage_cpp` on the
image resized to the pixel classification feature grid. The object features are morphology and
intensity statistics, computed with `bioimage_py` on the full resolution segmentation, with `scipy` and
`skimage` only for the statistics these packages do not provide.
"""

import warnings
from multiprocessing import cpu_count
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops_table

import bioimage_cpp.filters as bic_filters
from bioimage_cpp.transformation import resample

from bioimage_py.morphology import morphology, regionprops


PIXEL_FEATURES = (
    "gaussian_smoothing",
    "laplacian_of_gaussian",
    "gaussian_gradient_magnitude",
    "difference_of_gaussians",
    "structure_tensor_eigenvalues",
    "hessian_of_gaussian_eigenvalues",
)
"""The supported classical pixel features. Each one is computed per channel and per sigma."""

OBJECT_FEATURES = (
    "size",
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "extent",
    "equivalent_diameter",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "solidity",
)
"""The supported classical object features. The intensity features are computed per channel."""

DEFAULT_SIGMAS = (0.5, 1.0, 2.0, 4.0)
"""The default filter scales, in pixels of the pixel classification feature grid."""

# The ratio between the two scales of the difference of Gaussians.
DOG_RATIO = 1.6

INTENSITY_FEATURES = {
    "intensity_mean": ndimage.mean,
    "intensity_std": ndimage.standard_deviation,
    "intensity_min": ndimage.minimum,
    "intensity_max": ndimage.maximum,
}
SHAPE_FEATURES = ("extent", "equivalent_diameter", "axis_major_length", "axis_minor_length", "eccentricity")


def _validate(features: Sequence[str], supported: Sequence[str], kind: str) -> list:
    features = list(features)
    if not features:
        raise ValueError(f"Expect at least one classical {kind} feature, choose from {list(supported)}.")
    invalid = [name for name in features if name not in supported]
    if invalid:
        raise ValueError(f"Invalid classical {kind} features {invalid}, choose from {list(supported)}.")
    # Use the canonical order, so that the feature columns do not depend on the order of the selection.
    return [name for name in supported if name in features]


def _to_channels(image: np.ndarray, ndim: int) -> np.ndarray:
    """Bring the image to float32 with a trailing channel axis, normalized to [0, 1] per channel."""
    image = np.asarray(image)
    if image.ndim == ndim:
        image = image[..., None]
    elif image.ndim != ndim + 1:
        raise ValueError(f"Expect an image with {ndim} spatial dimensions and optional channels, got {image.shape}.")
    image = image.astype("float32")
    axes = tuple(range(ndim))
    minval, maxval = image.min(axis=axes), image.max(axis=axes)
    return (image - minval) / np.maximum(maxval - minval, 1e-7)


def _resize_plane(plane: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    # Anti-aliased resize with pixel-center alignment, matching 'skimage.transform.resize'.
    scale = [s / t for s, t in zip(plane.shape, target_hw)]
    matrix = np.array([[scale[0], 0.0, (scale[0] - 1) / 2], [0.0, scale[1], (scale[1] - 1) / 2]])
    return resample(
        np.ascontiguousarray(plane), matrix, bounding_box=tuple(slice(0, t) for t in target_hw)
    )


def _filter_plane(plane: np.ndarray, features: Sequence[str], sigmas: Sequence[float]) -> list:
    responses = []
    for sigma in sigmas:
        for name in features:
            if name == "gaussian_smoothing":
                responses.append(bic_filters.gaussian_smoothing(plane, sigma))
            elif name == "laplacian_of_gaussian":
                responses.append(bic_filters.laplacian_of_gaussian(plane, sigma))
            elif name == "gaussian_gradient_magnitude":
                responses.append(bic_filters.gaussian_gradient_magnitude(plane, sigma))
            elif name == "difference_of_gaussians":
                smoothed = bic_filters.gaussian_smoothing(plane, sigma)
                responses.append(smoothed - bic_filters.gaussian_smoothing(plane, DOG_RATIO * sigma))
            elif name == "structure_tensor_eigenvalues":
                eigenvalues = bic_filters.structure_tensor_eigenvalues(plane, sigma, sigma)
                responses.extend(np.moveaxis(eigenvalues, -1, 0))
            else:
                eigenvalues = bic_filters.hessian_of_gaussian_eigenvalues(plane, sigma)
                responses.extend(np.moveaxis(eigenvalues, -1, 0))
    return responses


def compute_classical_pixel_features(
    image: np.ndarray,
    grid_shape: Tuple[int, ...],
    features: Sequence[str] = PIXEL_FEATURES,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> np.ndarray:
    """Compute classical filter features on the pixel classification feature grid.

    The image is normalized to [0, 1] per channel and resized to the grid, then the filters are applied
    per channel, per sigma and, for 3d data, per slice. The result is aligned row-wise with the features
    of `pixel_classification.compute_pixel_features`.

    Args:
        image: The image, of shape (H, W) or (Z, H, W), with an optional trailing channel axis.
        grid_shape: The feature grid shape, (gh, gw) for 2d or (Z, gh, gw) for 3d.
        features: The classical pixel features to compute, a subset of `PIXEL_FEATURES`.
        sigmas: The filter scales, in pixels of the feature grid.

    Returns:
        The per-pixel features, of shape (N, F) flattened over the grid in row-major order.
    """
    features = _validate(features, PIXEL_FEATURES, "pixel")
    image = _to_channels(image, len(grid_shape))
    if len(grid_shape) == 2:
        image = image[None]
        grid_shape = (1,) + tuple(grid_shape)
    if image.shape[0] != grid_shape[0]:
        raise ValueError(f"The image has {image.shape[0]} slices, but the feature grid has {grid_shape[0]}.")

    planes = []
    for z in range(grid_shape[0]):
        responses = []
        for c in range(image.shape[-1]):
            plane = _resize_plane(image[z, ..., c], grid_shape[1:])
            responses.extend(_filter_plane(plane, features, sigmas))
        planes.append(np.stack(responses, axis=-1))
    return np.stack(planes).reshape(-1, planes[0].shape[-1]).astype("float32")


def compute_classical_object_features(
    image: np.ndarray,
    segmentation: np.ndarray,
    seg_ids: Optional[np.ndarray] = None,
    features: Sequence[str] = OBJECT_FEATURES,
) -> np.ndarray:
    """Compute classical morphology and intensity features per object.

    Args:
        image: The image, with the spatial shape of the segmentation and an optional trailing channel axis.
        segmentation: The segmentation, of shape (H, W) or (Z, H, W). The background id 0 is ignored.
        seg_ids: The ids of the objects to compute the features for, e.g. the ids returned by
            `object_classification.compute_object_features`. By default all objects in the segmentation.
        features: The classical object features to compute, a subset of `OBJECT_FEATURES`.

    Returns:
        The object features, of shape (len(seg_ids), F), in the order of `seg_ids`.
    """
    features = _validate(features, OBJECT_FEATURES, "object")
    segmentation = np.asarray(segmentation)
    if segmentation.dtype not in (np.uint32, np.uint64, np.int32, np.int64):
        segmentation = segmentation.astype("uint32")
    image = _to_channels(image, segmentation.ndim)
    if image.shape[:-1] != segmentation.shape:
        raise ValueError(f"The image shape {image.shape} does not match the segmentation shape {segmentation.shape}.")

    if seg_ids is None:
        seg_ids = np.unique(segmentation)
        seg_ids = seg_ids[seg_ids != 0]
    seg_ids = np.asarray(seg_ids)

    columns = {}
    shape_features = [name for name in features if name in SHAPE_FEATURES]
    if "size" in features or shape_features:
        table = morphology(segmentation)
        if "size" in features:
            columns["size"] = table.set_index("label")["size"]
        if shape_features:
            props = regionprops(segmentation, table, num_workers=cpu_count()).set_index("label")
            major, minor = props["axis_major_length"], props["axis_minor_length"]
            eccentricity = np.sqrt(1.0 - (minor / major.where(major > 0)) ** 2).fillna(0.0)
            shape_columns = {
                "extent": props["extent"], "equivalent_diameter": props["equivalent_diameter_area"],
                "axis_major_length": major, "axis_minor_length": minor, "eccentricity": eccentricity,
            }
            columns.update({name: shape_columns[name] for name in shape_features})

    if "solidity" in features:
        # Flat or tiny objects have no convex hull, we treat them as fully convex.
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            props = regionprops_table(segmentation, properties=("label", "solidity"))
        solidity = np.where(np.isfinite(props["solidity"]), props["solidity"], 1.0)
        columns["solidity"] = pd.Series(solidity, index=props["label"])

    ordered = {}
    for name in features:
        if name in INTENSITY_FEATURES:
            for c in range(image.shape[-1]):
                values = INTENSITY_FEATURES[name](image[..., c], labels=segmentation, index=seg_ids)
                ordered[f"{name}_{c}"] = pd.Series(np.asarray(values), index=seg_ids)
        else:
            ordered[name] = columns[name]

    # Objects that are missing from the segmentation get zero features.
    result = pd.DataFrame(ordered).reindex(seg_ids).fillna(0.0)
    return result.to_numpy(dtype="float32")
