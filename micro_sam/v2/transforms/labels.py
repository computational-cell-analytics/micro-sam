from typing import Optional, Tuple

import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.measure import label as connected_components

import vigra


def _axondeepseg_pre_label_transform(y):
    """Extract axon instances from AxonDeepSeg semantic labels via connected components.

    Runs before the sampler so MinInstanceSampler can count actual axon instances
    rather than just binary foreground (0/1).
    """
    return connected_components(y == 2).astype("uint32")


def _em_cell_label_trafo(y, label_trafo):
    y = label_trafo(y)

    # Prepare the true background.
    instances = y[0]

    bd = find_boundaries(instances.astype("uint32"), mode="outer").astype("uint8")
    fg = (instances > 0).astype("uint8")
    expected_fg = (fg & ~bd).astype("uint8")

    expected_y = np.concatenate([expected_fg[None], y[2:]], axis=0)

    return expected_y


def _plantseg_label_trafo(y, data, label_trafo):
    # Let's reject the samples first.
    if data == "root":
        y[y == 1] = 0
    elif data == "ovules":
        y[y == -1] = 0
    else:
        raise ValueError

    y = label_trafo(y)

    return y


class DirectedPerObjectBoundaryDistanceTransform:
    eps = 1e-7

    def __init__(
        self,
        min_size: int = 0,
        foreground: bool = True,
        instances: bool = False,
        apply_label: bool = True,
        sampling: Optional[Tuple[float, ...]] = None,
    ):
        self.min_size = min_size
        self.distance_fill_value = 1
        self.foreground = foreground
        self.instances = instances
        self.apply_label = apply_label
        self.sampling = sampling

    def compute_normalized_directed_distances(self, labels, label_id, boundaries, bb, distances):
        """@private
        """
        cropped_mask = labels[bb] == label_id
        inv_mask = ~cropped_mask

        cropped_boundary_mask = boundaries[bb]

        kwargs = {} if self.sampling is None else {"pixel_pitch": self.sampling}
        this_distances = vigra.filters.vectorDistanceTransform(cropped_boundary_mask, **kwargs)
        this_distances[inv_mask] = 0

        spatial_axes = tuple(range(labels.ndim))
        this_distances /= (np.abs(this_distances).max(axis=spatial_axes, keepdims=True) + self.eps)

        distances[bb][cropped_mask] = this_distances[cropped_mask]
        return distances

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Compute the per object distance transform.

        Args:
            labels: The segmentation

        Returns:
            The distances.
        """
        is_2d = (labels.ndim == 2)

        if labels.ndim == 2:
            labels = labels[None]

        if self.apply_label:
            labels = connected_components(labels).astype("uint32")
        else:  # Otherwise just relabel the segmentation.
            labels = vigra.analysis.relabelConsecutive(labels)[0].astype("uint32")

        # Filter out small objects if min_size is specified.
        if self.min_size > 0:
            ids, sizes = np.unique(labels, return_counts=True)
            discard_ids = ids[sizes < self.min_size]
            labels[np.isin(labels, discard_ids)] = 0
            labels = vigra.analysis.relabelConsecutive(labels)[0].astype("uint32")

        # Compute the boundaries.
        boundaries = find_boundaries(labels, mode="inner").astype("uint32")

        # Compute region properties to derive bounding boxes and centers.
        ndim = labels.ndim
        props = regionprops(labels)
        bounding_boxes = {
            prop.label: tuple(slice(prop.bbox[i], prop.bbox[i + ndim]) for i in range(ndim)) for prop in props
        }

        # Compute how many distance channels we have.
        n_channels = 3

        # Compute the per object distances.
        distances = np.full(labels.shape + (n_channels,), self.distance_fill_value, dtype="float32")
        for prop in props:
            label_id = prop.label
            distances = self.compute_normalized_directed_distances(
                labels, label_id, boundaries, bounding_boxes[label_id], distances
            )

        # Bring the distance channel to the first dimension.
        to_channel_first = (ndim,) + tuple(range(ndim))
        distances = distances.transpose(to_channel_first)

        # Add the foreground mask as first channel if specified.
        if self.foreground:
            binary_labels = (labels > 0).astype("float32")
            distances = np.concatenate([binary_labels[None], distances], axis=0)

        if self.instances:
            distances = np.concatenate([labels[None], distances], axis=0)

        if is_2d:
            assert distances.ndim == 4
            assert distances.shape[1] == 1
            distances = distances.squeeze(1)

        return distances
