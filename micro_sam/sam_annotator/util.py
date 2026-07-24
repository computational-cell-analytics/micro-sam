import os
import pickle
import warnings
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import napari
import numpy as np
from skimage import draw
from scipy.ndimage import shift

from .. import util
from ..v1 import prompt_based_segmentation
from .. import _model_settings as model_settings
from ..v1.multi_dimensional_segmentation import _validate_projection

# Green and Red
LABEL_COLOR_CYCLE = ["#00FF00", "#FF0000"]
"""@private"""

SCRIBBLE_SHAPE_TYPES = ("path", "line")
"""Napari shape types interpreted as sparse scribble prompts."""

SCRIBBLE_DRAW_MODES = ("add_path", "add_polyline", "add_line")
"""Napari Shapes modes that create open scribble prompts."""


#
# Misc helper functions
#


def _channels_to_rgb(image: np.ndarray) -> np.ndarray:
    """Map a 2D image's trailing channel axis to exactly 3 channels.

    A single channel is replicated, two channels are padded with a zero channel, three channels are
    left as-is, and more than three channels are reduced to the first three (with a warning).
    """
    n_channels = image.shape[-1]
    if n_channels == 3:
        return image
    if n_channels == 1:
        return np.concatenate([image] * 3, axis=-1)
    if n_channels == 2:
        zero_channel = np.zeros(image.shape[:-1] + (1,), dtype=image.dtype)
        return np.concatenate([image, zero_channel], axis=-1)
    warnings.warn(f"You provided an input with {n_channels} channels. Only the first three will be used.")
    return image[..., :3]


def prepare_annotation_image(image: np.ndarray, ndim: Optional[int] = None) -> Tuple[np.ndarray, int, bool]:
    """Normalize an image for annotation: squeeze singletons and map 2D channels to RGB.

    Singleton axes (commonly exposed by formats like CZI) are squeezed out across all axes.
    For a 2D image, the trailing channel axis is mapped to 3 channels: a 2-channel input is
    padded with a zero channel and a 4-channel input is reduced to the first three (with a
    warning). A 3D volume with a channel axis (3D+C) is not supported.

    Args:
        image: The input image data.
        ndim: Optional override for the spatial dimensionality (2 or 3). With ``None`` (the default)
            the dimensionality is auto-detected from the shape (a trailing axis of size 3 -> RGB 2D,
            of size 2 or 4 -> channels mapped to RGB 2D, otherwise a 3D volume). With ``2`` a 3D array
            is read as a 2D multi-channel image, taking the smallest axis as the channel axis (so a
            channels-first ``(C, H, W)`` array also works) and mapping the channels to RGB. With ``3``
            a 3D array is read as a ``(Z, H, W)`` volume.

    Returns:
        A tuple of the normalized image, its spatial dimensionality (2 or 3), and whether
        it has a trailing RGB channel axis.

    Raises:
        ValueError: If the (overridden) dimensionality cannot be applied to the image shape, or the
            squeezed image is neither a 2D image nor a grayscale 3D volume.
    """
    if ndim not in (None, 2, 3):
        raise ValueError(f"Invalid ndim override: {ndim}. Expected None, 2 or 3.")

    image = np.squeeze(image)

    # Forced 2D: read a 3D array as a 2D multi-channel image. The channel axis is taken to be the
    # smallest axis (so both channels-first (C, H, W) and channels-last (H, W, C) work); it is moved
    # to the trailing position and mapped to 3 channels.
    if ndim == 2:
        if image.ndim == 2:
            return image, 2, False
        if image.ndim == 3:
            channel_axis = int(np.argmin(image.shape))
            image = np.moveaxis(image, channel_axis, -1)
            return _channels_to_rgb(image), 2, True
        raise ValueError(f"Cannot interpret shape {image.shape} as a 2D image.")

    # Forced 3D: read a 3D array as a (Z, H, W) volume. A channel axis (4D, or 3D+C) is not supported.
    if ndim == 3:
        if image.ndim == 3:
            return image, 3, False
        raise ValueError(
            f"Cannot interpret shape {image.shape} as a 3D volume (3D data with channels is not supported yet)."
        )

    # Auto-detect. A 4D array is either a 3D volume with a channel axis (Z, H, W, C) or a volumetric
    # time series (T, Z, H, W). Neither is supported: the v2 3D path assumes a grayscale (Z, H, W)
    # volume, so a channel axis would otherwise produce wrong-shaped masks.
    if image.ndim == 4:
        if image.shape[-1] in (2, 3, 4):
            raise ValueError(
                f"3D volumes with a channel axis are not supported yet, got shape {image.shape}."
            )
        raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D image data (3D+t is not supported).")

    # 2D image with a 2- or 4-channel trailing axis: map it to 3 channels. A trailing axis of any
    # other size is left alone, so a size-3 axis stays RGB and anything else is treated as a volume.
    if image.ndim == 3 and image.shape[-1] in (2, 4):
        image = _channels_to_rgb(image)

    # Map the (possibly normalized) shape to a spatial dimensionality and rgb flag.
    if image.ndim == 2:
        return image, 2, False
    elif image.ndim == 3:
        if image.shape[-1] == 3:
            return image, 2, True
        return image, 3, False
    raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D image data.")


def set_prompt_label(layer, new_label):
    """Set the current prompt label and relabel selected shapes consistently.

    Napari Points applies ``current_properties`` changes to selected points in all modes. Shapes,
    however, only applies them to selected shapes in select or pan/zoom mode. Explicitly updating
    the selected open shapes here keeps changing the prompt menu or pressing ``T`` consistent for
    both layer types, including immediately after drawing a path, polyline or line.
    """
    if isinstance(layer, napari.layers.Shapes):
        # Napari may briefly retain a selected shape index after the corresponding geometry and
        # feature row have been removed. Setting current_properties in that state tries to update a
        # missing pandas row. Drop these stale indices before applying the new drawing label.
        valid_selection = {
            index for index in layer.selected_data if 0 <= index < len(layer.data)
        }
        if valid_selection != layer.selected_data:
            layer.selected_data = valid_selection

    current_properties = layer.current_properties
    current_properties["label"] = np.array([new_label])
    layer.current_properties = current_properties

    if isinstance(layer, napari.layers.Shapes) and layer.selected_data:
        properties = dict(layer.properties)
        labels = np.asarray(properties.get("label", []), dtype=object).copy()
        shape_types = list(layer.shape_type)
        if len(labels) == len(shape_types):
            for index in layer.selected_data:
                if shape_types[index] in SCRIBBLE_SHAPE_TYPES:
                    labels[index] = new_label
            properties["label"] = labels
            layer.properties = properties

            # Keep the drawing default on the requested label. Updating the feature table above
            # may infer a current value from the edited selection.
            current_properties = layer.current_properties
            current_properties["label"] = np.array([new_label])
            layer.current_properties = current_properties

    layer.refresh()
    if isinstance(layer, napari.layers.Shapes):
        # During layer reset/teardown napari can briefly clear shape geometry before shrinking the
        # feature table. Refreshing mapped colors in that transient state raises because the color
        # array and ShapeList have different lengths. The subsequent data/features event will
        # refresh once they are aligned again.
        n_shapes = len(layer.data)
        if any(len(values) != n_shapes for values in layer.properties.values()):
            return
    layer.refresh_colors()


def toggle_label(prompts, *linked_prompts):
    """Toggle the positive/negative label for one or more prompt layers."""
    # Use the first layer as the source of truth, then keep all linked prompt layers in sync.
    current_label = prompts.current_properties["label"][0]
    new_label = "negative" if current_label == "positive" else "positive"
    for layer in (prompts,) + linked_prompts:
        set_prompt_label(layer, new_label)


def normalize_prompt_shape_labels(layer_or_event):
    """Keep closed shape prompts positive while preserving labels for open scribbles.

    The shared Shapes layer uses its ``label`` property for edge coloring. Boxes and dense mask
    shapes do not support negative semantics, so they are normalized to positive after creation.
    The function then restores the current drawing defaults. A box that you draw does not change
    the label for the next point or scribble. In the tracking annotator it also keeps the current
    ``track_id``.
    """
    layer = layer_or_event if hasattr(layer_or_event, "shape_type") else layer_or_event.source
    shape_types = list(layer.shape_type)
    labels = np.asarray(layer.properties.get("label", []), dtype=object)
    if len(shape_types) != len(labels):
        return

    normalized = labels.copy()
    for index, shape_type in enumerate(shape_types):
        if shape_type not in SCRIBBLE_SHAPE_TYPES:
            normalized[index] = "positive"
    if np.array_equal(labels, normalized):
        return

    # Save the drawing defaults first. An assignment to 'layer.properties' resets the current
    # value of every column. This would otherwise drop the current 'track_id' on the tracking layer.
    current_properties = dict(layer.current_properties)
    properties = dict(layer.properties)
    properties["label"] = normalized
    layer.properties = properties
    layer.current_properties = current_properties
    layer.refresh_colors()


def sync_prompt_shape_current_color(layer_or_event):
    """Synchronize the Shapes drawing color with the current scribble label and tool."""
    layer = layer_or_event if hasattr(layer_or_event, "shape_type") else layer_or_event.source
    label = layer.current_properties.get("label", np.array(["positive"]))[0]
    is_scribble_tool = layer.mode in SCRIBBLE_DRAW_MODES
    color_index = 1 if is_scribble_tool and label == "negative" else 0
    layer.current_edge_color = LABEL_COLOR_CYCLE[color_index]


def clear_annotations(viewer: napari.Viewer, clear_segmentations=True) -> None:
    """@private"""
    viewer.layers["point_prompts"].data = []
    viewer.layers["point_prompts"].refresh()
    if "prompts" in viewer.layers:
        # Select all prompts and then remove them.
        # This is how it worked before napari 0.5.
        # viewer.layers["prompts"].data = []
        viewer.layers["prompts"].selected_data = set(range(len(viewer.layers["prompts"].data)))
        viewer.layers["prompts"].remove_selected()
        viewer.layers["prompts"].refresh()
    if not clear_segmentations:
        return
    viewer.layers["current_object"].data = np.zeros(viewer.layers["current_object"].data.shape, dtype="uint32")
    viewer.layers["current_object"].refresh()


def clear_annotations_slice(viewer: napari.Viewer, i: int, clear_segmentations=True) -> None:
    """@private"""
    point_prompts = viewer.layers["point_prompts"].data
    point_prompts = point_prompts[point_prompts[:, 0] != i]
    viewer.layers["point_prompts"].data = point_prompts
    viewer.layers["point_prompts"].refresh()
    if "prompts" in viewer.layers:
        prompt_layer = viewer.layers["prompts"]
        prompt_layer.selected_data = {
            index for index, prompt in enumerate(prompt_layer.data) if (prompt[:, 0] == i).all()
        }
        prompt_layer.remove_selected()
        prompt_layer.refresh()
    if not clear_segmentations:
        return
    viewer.layers["current_object"].data[i] = 0
    viewer.layers["current_object"].refresh()


#
# Helper functions to extract prompts from napari layers.
#


def point_layer_to_prompts(
    layer: napari.layers.Points, i=None, track_id=None, with_stop_annotation=True, exclude_states=None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract point prompts for SAM from a napari point layer.

    Args:
        layer: The point layer from which to extract the prompts.
        i: Index for the data (required for 3d or timeseries data).
        track_id: Id of the current track (required for tracking data).
        with_stop_annotation: Whether a single negative point will be interpreted
            as stop annotation or just returned as normal prompt.
        exclude_states: Track-states to drop (e.g. ('division',)); such points mark a lineage
            event rather than a segmentation prompt and must not be fed to the predictor.

    Returns:
        The point coordinates for the prompts.
        The labels (positive or negative / 1 or 0) for the prompts.
    """

    points = layer.data
    labels = layer.properties["label"]
    assert len(points) == len(labels)

    # Drop points tagged with an excluded track-state (division markers are not prompts).
    keep = np.ones(len(points), dtype=bool)
    if exclude_states is not None and "state" in layer.properties:
        keep = ~np.isin(np.asarray(layer.properties["state"]), list(exclude_states))

    if i is None:
        assert points.shape[1] == 2, f"{points.shape}"
        this_points, this_labels = points[keep], labels[keep]
    else:
        assert points.shape[1] == 3, f"{points.shape}"
        mask = (np.round(points[:, 0]) == i) & keep
        this_points = points[mask][:, 1:]
        this_labels = labels[mask]
    assert len(this_points) == len(this_labels)

    if track_id is not None:
        assert i is not None
        track_ids = np.array(list(map(int, layer.properties["track_id"])))[mask]
        track_id_mask = track_ids == track_id
        this_labels, this_points = this_labels[track_id_mask], this_points[track_id_mask]
    assert len(this_points) == len(this_labels)

    this_labels = np.array([1 if label == "positive" else 0 for label in this_labels])
    # a single point with a negative label is interpreted as 'stop' signal
    # in this case we return None
    if with_stop_annotation and (len(this_points) == 1 and this_labels[0] == 0):
        return None

    return this_points, this_labels


def _scribble_geometry(vertices, image_shape, spacing):
    """Return normalized stroke geometry and bend information for adaptive sampling."""
    vertices = np.asarray(vertices, dtype="float64")
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) == 0:
        raise ValueError("A scribble must have shape (N, 2) with at least one vertex.")

    image_shape = np.asarray(image_shape, dtype="float64")
    if image_shape.shape != (2,) or np.any(image_shape <= 0):
        raise ValueError(f"Invalid 2D image shape: {tuple(image_shape)}.")

    # SAM2 embeds prompts in a square 1024-pixel input frame. Measuring the stroke there makes the
    # sampling density independent of the source image resolution and aspect ratio.
    model_vertices = vertices * (1024.0 / image_shape)

    # Remove consecutive duplicate vertices before measuring length or curvature. Freehand paths
    # may contain these when the mouse briefly stops moving.
    if len(model_vertices) > 1:
        keep = np.concatenate([[True], np.linalg.norm(np.diff(model_vertices, axis=0), axis=1) > 0])
        model_vertices = model_vertices[keep]

    segment_lengths = np.linalg.norm(np.diff(model_vertices, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])

    # Ramer-Douglas-Peucker simplification removes freehand jitter before curvature is measured.
    # The retained inner vertices are meaningful bends that we try to preserve in the samples.
    bend_indices = np.empty((0,), dtype="int64")
    bend_angles = np.empty((0,), dtype="float64")
    if len(model_vertices) > 2 and total_length > 0:
        tolerance = spacing / 4.0
        retained = {0, len(model_vertices) - 1}
        pending = [(0, len(model_vertices) - 1)]
        while pending:
            start, stop = pending.pop()
            if stop <= start + 1:
                continue
            start_point, stop_point = model_vertices[start], model_vertices[stop]
            direction = stop_point - start_point
            relative = model_vertices[start + 1:stop] - start_point
            length_squared = float(np.dot(direction, direction))
            if length_squared == 0:
                distances = np.linalg.norm(relative, axis=1)
            else:
                fractions = np.clip(relative @ direction / length_squared, 0.0, 1.0)
                projections = start_point + fractions[:, None] * direction
                distances = np.linalg.norm(model_vertices[start + 1:stop] - projections, axis=1)
            split = int(np.argmax(distances)) + start + 1
            if distances[split - start - 1] > tolerance:
                retained.add(split)
                pending.extend([(start, split), (split, stop)])

        retained = np.asarray(sorted(retained), dtype="int64")
        simplified = model_vertices[retained]
        if len(simplified) > 2:
            incoming = simplified[1:-1] - simplified[:-2]
            outgoing = simplified[2:] - simplified[1:-1]
            cosine = np.sum(incoming * outgoing, axis=1) / (
                np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
            )
            bend_indices = retained[1:-1]
            bend_angles = np.arccos(np.clip(cosine, -1.0, 1.0))

    return model_vertices, image_shape, cumulative, total_length, bend_indices, bend_angles


def _scribble_sample_count(vertices, image_shape, spacing):
    """Determine the sample demand from a stroke's length and simplified curvature."""
    geometry = _scribble_geometry(vertices, image_shape, spacing)
    total_length, bend_angles = geometry[3], geometry[5]
    if total_length == 0:
        return 1, total_length

    length_samples = max(2, int(np.ceil(total_length / spacing)) + 1)
    # Add one sample per 90 degrees of accumulated, de-jittered turning. This gives compact curved
    # strokes more representation without letting every raw freehand vertex consume the budget.
    curvature_samples = int(np.ceil(bend_angles.sum() / (np.pi / 2.0))) if len(bend_angles) else 0
    return length_samples + curvature_samples, total_length


def _allocate_scribble_samples(desired_counts, lengths, max_points):
    """Share a global sample budget fairly, favouring strokes with greater demand."""
    desired_counts = np.asarray(desired_counts, dtype="int64")
    lengths = np.asarray(lengths, dtype="float64")
    if len(desired_counts) == 0:
        return desired_counts

    # Every stroke must influence the prediction. If there are more strokes than the configured
    # budget, expand it just enough to retain one representative point from each instead of
    # silently dropping annotations.
    budget = max(int(max_points), len(desired_counts))
    if desired_counts.sum() <= budget:
        return desired_counts

    allocated = np.ones_like(desired_counts)
    remaining = budget - len(allocated)

    # Preserve both endpoints where the budget permits, prioritizing longer strokes if it does not.
    endpoint_candidates = np.flatnonzero(desired_counts > 1)
    endpoint_candidates = endpoint_candidates[np.argsort(-lengths[endpoint_candidates], kind="stable")]
    n_endpoints = min(remaining, len(endpoint_candidates))
    allocated[endpoint_candidates[:n_endpoints]] += 1
    remaining -= n_endpoints
    if remaining == 0:
        return allocated

    # Allocate the rest proportionally to each stroke's unmet length/curvature demand. Largest
    # remainders make the result deterministic while using the complete budget.
    demand = desired_counts - allocated
    quotas = remaining * demand / demand.sum()
    additions = np.floor(quotas).astype("int64")
    allocated += additions
    remaining -= int(additions.sum())
    if remaining:
        residual = quotas - additions
        candidates = np.flatnonzero(demand > additions)
        candidates = candidates[np.argsort(-residual[candidates], kind="stable")]
        allocated[candidates[:remaining]] += 1
    return allocated


def _resample_scribble(vertices, image_shape, spacing, n_points):
    """Resample one stroke while preserving endpoints and its strongest bends."""
    model_vertices, image_shape, cumulative, total_length, bend_indices, bend_angles = (
        _scribble_geometry(vertices, image_shape, spacing)
    )
    if total_length == 0:
        return model_vertices[:1] * (image_shape / 1024.0)
    if n_points == 1:
        sample_distances = np.array([total_length / 2.0])
    else:
        n_bends = min(max(0, n_points - 2), len(bend_indices))
        if n_bends:
            strongest = np.argsort(-bend_angles, kind="stable")[:n_bends]
            mandatory = np.concatenate([[0.0], cumulative[bend_indices[strongest]], [total_length]])
            mandatory = np.unique(mandatory)
        else:
            mandatory = np.array([0.0, total_length])

        n_extra = n_points - len(mandatory)
        if n_extra > 0:
            interval_lengths = np.diff(mandatory)
            quotas = n_extra * interval_lengths / total_length
            per_interval = np.floor(quotas).astype("int64")
            remainder = n_extra - int(per_interval.sum())
            if remainder:
                order = np.argsort(-(quotas - per_interval), kind="stable")
                per_interval[order[:remainder]] += 1

            extra_distances = []
            for start, stop, count in zip(mandatory[:-1], mandatory[1:], per_interval):
                if count:
                    extra_distances.extend(np.linspace(start, stop, count + 2)[1:-1])
            sample_distances = np.sort(np.concatenate([mandatory, extra_distances]))
        else:
            sample_distances = mandatory

    segment_ids = np.searchsorted(cumulative, sample_distances, side="right") - 1
    segment_ids = np.clip(segment_ids, 0, len(model_vertices) - 2)

    local_lengths = sample_distances - cumulative[segment_ids]
    segment_lengths = np.diff(cumulative)
    fractions = np.divide(
        local_lengths,
        segment_lengths[segment_ids],
        out=np.zeros_like(local_lengths),
        where=segment_lengths[segment_ids] > 0,
    )
    sampled_model = model_vertices[segment_ids] + fractions[:, None] * (
        model_vertices[segment_ids + 1] - model_vertices[segment_ids]
    )
    return sampled_model * (image_shape / 1024.0)


def scribble_layer_to_prompts(
    layer: napari.layers.Shapes,
    image_shape: Tuple[int, int],
    i=None,
    spacing: float = 32.0,
    max_points_per_stroke: Optional[int] = None,
    max_points: int = 64,
    deduplication_distance: float = 4.0,
    track_id=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert positive/negative open strokes into sparse SAM point prompts.

    Napari stores both its freehand path and click-defined polyline tools as ``path`` shapes; a
    two-vertex stroke is stored as ``line``. Each accepted stroke is resampled uniformly by arc
    length in SAM's normalized 1024-pixel input space. A global prompt budget is shared according
    to stroke length and simplified curvature, and nearby same-label samples from overlapping
    strokes are collapsed. This keeps prompt density stable across source resolutions without
    undersampling every long stroke at the same per-stroke cutoff.

    Args:
        layer: The shared ``prompts`` Shapes layer. Non-scribble shapes are ignored.
        image_shape: The ``(height, width)`` of the image or volume slice.
        i: Slice index for a 3D layer. Must be omitted for a 2D layer.
        spacing: Approximate spacing between samples in SAM's 1024-pixel input space.
        max_points_per_stroke: Optional compatibility cap for each stroke. By default there is no
            per-stroke cap and ``max_points`` governs all strokes together.
        max_points: Target maximum number of samples across all accepted scribbles. If there are
            more strokes than this limit, it expands to retain one representative per stroke.
        deduplication_distance: Distance in normalized model pixels below which samples from
            overlapping strokes with the same label are considered duplicates.
        track_id: Id of the current track. Required for tracking data. When given, the function
            converts only the strokes whose ``track_id`` property matches.

    Returns:
        Sampled coordinates in ``(y, x)`` order and SAM labels (positive ``1``, negative ``0``).
    """
    if spacing <= 0:
        raise ValueError("'spacing' must be positive.")
    if max_points_per_stroke is not None and max_points_per_stroke <= 0:
        raise ValueError("'max_points_per_stroke' must be positive.")
    if max_points <= 0:
        raise ValueError("'max_points' must be positive.")
    if deduplication_distance < 0:
        raise ValueError("'deduplication_distance' must be non-negative.")

    shape_data = layer.data
    shape_types = layer.shape_type
    stroke_labels = layer.properties.get("label", [])
    if not (len(shape_data) == len(shape_types) == len(stroke_labels)):
        raise AssertionError("Scribble shapes, shape types and labels must have matching lengths.")

    if track_id is None:
        stroke_track_ids = [None] * len(shape_data)
    else:
        stroke_track_ids = list(map(int, layer.properties["track_id"]))
        if len(stroke_track_ids) != len(shape_data):
            raise AssertionError("Scribble shapes and track ids must have matching lengths.")

    strokes = []
    for vertices, shape_type, stroke_label, stroke_track_id in zip(
        shape_data, shape_types, stroke_labels, stroke_track_ids
    ):
        if track_id is not None and stroke_track_id != track_id:
            continue
        if shape_type in ("rectangle", "ellipse", "polygon"):
            continue
        if shape_type not in SCRIBBLE_SHAPE_TYPES:
            warnings.warn(
                f"Shape type {shape_type} is not a scribble and will be ignored. "
                "Use path, polyline or line in the 'prompts' layer.",
                stacklevel=2,
            )
            continue
        if stroke_label not in ("positive", "negative"):
            warnings.warn(
                f"Unknown scribble label {stroke_label!r}; the stroke will be ignored.", stacklevel=2
            )
            continue

        vertices = np.asarray(vertices)
        if i is None:
            if vertices.ndim != 2 or vertices.shape[1] != 2:
                raise ValueError("2D scribble vertices must have shape (N, 2).")
            vertices_yx = vertices
        else:
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError("3D scribble vertices must have shape (N, 3).")
            stroke_slices = np.round(vertices[:, 0]).astype(int)
            if not np.all(stroke_slices == i):
                continue
            vertices_yx = vertices[:, 1:]

        desired_count, length = _scribble_sample_count(vertices_yx, image_shape, spacing)
        if max_points_per_stroke is not None:
            desired_count = min(desired_count, max_points_per_stroke)
        strokes.append((vertices_yx, stroke_label, desired_count, length))

    if not strokes:
        return np.empty((0, 2), dtype="float64"), np.empty((0,), dtype="int64")

    sample_counts = _allocate_scribble_samples(
        [stroke[2] for stroke in strokes], [stroke[3] for stroke in strokes], max_points
    )
    image_shape_array = np.asarray(image_shape, dtype="float64")
    points, labels = [], []
    previous_model_points = {0: [], 1: []}
    for (vertices_yx, stroke_label, _, _), sample_count in zip(strokes, sample_counts):
        sampled = _resample_scribble(
            vertices_yx, image_shape=image_shape, spacing=spacing, n_points=sample_count
        )
        sam_label = 1 if stroke_label == "positive" else 0
        current_model_points = []
        for point in sampled:
            model_point = point * (1024.0 / image_shape_array)
            # Deduplicate only against earlier strokes so both endpoints of a very short individual
            # stroke survive. Opposite-label conflicts are intentionally retained.
            previous = previous_model_points[sam_label]
            if previous and (
                np.min(np.linalg.norm(np.asarray(previous) - model_point, axis=1)) <= deduplication_distance
            ):
                continue
            if current_model_points and np.min(
                np.linalg.norm(np.asarray(current_model_points) - model_point, axis=1)
            ) == 0:
                continue
            current_model_points.append(model_point)
            points.append(point)
            labels.append(sam_label)
        previous_model_points[sam_label].extend(current_model_points)

    return np.asarray(points), np.asarray(labels, dtype="int64")


def get_scribble_slices(layer: napari.layers.Shapes, track_id=None) -> np.ndarray:
    """Return the sorted z-slices that contain open scribble shapes in a 3D Shapes layer.

    When ``track_id`` is given, the function considers only scribbles whose ``track_id`` property matches.
    """
    shape_data = layer.data
    shape_types = layer.shape_type
    if len(shape_data) != len(shape_types):
        raise AssertionError("Scribble shapes and shape types must have matching lengths.")

    if track_id is None:
        stroke_track_ids = [None] * len(shape_data)
    else:
        stroke_track_ids = list(map(int, layer.properties["track_id"]))

    slices = []
    for vertices, shape_type, stroke_track_id in zip(shape_data, shape_types, stroke_track_ids):
        if shape_type not in SCRIBBLE_SHAPE_TYPES:
            continue
        if track_id is not None and stroke_track_id != track_id:
            continue
        vertices = np.asarray(vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            continue
        stroke_slices = np.round(vertices[:, 0]).astype(int)
        if not np.all(stroke_slices == stroke_slices[0]):
            warnings.warn("A 3D scribble must stay on one z-slice and will be ignored.", stacklevel=2)
            continue
        slices.append(stroke_slices[0])

    return np.unique(slices).astype("int64") if slices else np.empty((0,), dtype="int64")


def merge_point_prompts(*prompt_sets):
    """Merge ``(points, labels)`` prompt tuples, preserving an empty ``(0, 2)`` convention."""
    point_arrays, label_arrays = [], []
    for points, labels in prompt_sets:
        points = np.asarray(points).reshape(-1, 2)
        labels = np.asarray(labels).reshape(-1)
        if len(points) != len(labels):
            raise AssertionError("The number of prompt coordinates and labels must match.")
        if len(points):
            point_arrays.append(points)
            label_arrays.append(labels)
    if not point_arrays:
        return np.empty((0, 2), dtype="float64"), np.empty((0,), dtype="int64")
    return np.concatenate(point_arrays), np.concatenate(label_arrays)


def shape_layer_to_prompts(
    layer: napari.layers.Shapes, shape: Tuple[int, int], i=None, track_id=None
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
    """Extract prompts for SAM from a napari shape layer.

    Extracts the bounding box for 'rectangle' shapes and the bounding box and corresponding mask
    for 'ellipse' and 'polygon' shapes.

    Args:
        prompt_layer: The napari shape layer.
        shape: The image shape.
        i: Index for the data (required for 3d or timeseries data).
        track_id: Id of the current track (required for tracking data).

    Returns:
        The box prompts.
        The mask prompts.
    """

    def _to_prompts(shape_data, shape_types):
        boxes, masks = [], []

        for data, type_ in zip(shape_data, shape_types):

            if type_ == "rectangle":
                boxes.append(data)
                masks.append(None)

            elif type_ == "ellipse":
                boxes.append(data)
                center = np.mean(data, axis=0)
                radius_r = ((data[2] - data[1]) / 2)[0]
                radius_c = ((data[1] - data[0]) / 2)[1]
                rr, cc = draw.ellipse(center[0], center[1], radius_r, radius_c, shape=shape)
                mask = np.zeros(shape, dtype=bool)
                mask[rr, cc] = 1
                masks.append(mask)

            elif type_ == "polygon":
                boxes.append(data)
                rr, cc = draw.polygon(data[:, 0], data[:, 1], shape=shape)
                mask = np.zeros(shape, dtype=bool)
                mask[rr, cc] = 1
                masks.append(mask)

            elif type_ in SCRIBBLE_SHAPE_TYPES:
                continue

            else:
                warnings.warn(f"Shape type {type_} is not supported and will be ignored.")

        # map to correct box format
        boxes = [
            np.array([box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()]) for box in boxes
        ]
        return boxes, masks

    shape_data, shape_types = layer.data, layer.shape_type
    assert len(shape_data) == len(shape_types)
    if len(shape_data) == 0:
        return [], []

    if i is not None:
        if track_id is None:
            prompt_selection = [j for j, data in enumerate(shape_data) if (data[:, 0] == i).all()]
        else:
            track_ids = np.array(list(map(int, layer.properties["track_id"])))
            prompt_selection = [
                j for j, (data, this_track_id) in enumerate(zip(shape_data, track_ids))
                if ((data[:, 0] == i).all() and this_track_id == track_id)
            ]

        shape_data = [shape_data[j][:, 1:] for j in prompt_selection]
        shape_types = [shape_types[j] for j in prompt_selection]

    boxes, masks = _to_prompts(shape_data, shape_types)
    return boxes, masks


def prompt_layer_to_state(prompt_layer: napari.layers.Points, i: int) -> str:
    """Get the state of the track from a point layer for a given timeframe.

    Only relevant for annotator_tracking.

    Args:
        prompt_layer: The napari layer.
        i: Timeframe of the data.

    Returns:
        The state of this frame (either "division" or "track").
    """
    state = prompt_layer.properties["state"]

    points = prompt_layer.data
    assert points.shape[1] == 3, f"{points.shape}"
    mask = points[:, 0] == i
    this_points = points[mask][:, 1:]
    this_state = state[mask]
    assert len(this_points) == len(this_state)

    # we set the state to 'division' if at least one point in this frame has a division label
    if any(st == "division" for st in this_state):
        return "division"
    else:
        return "track"


def prompt_layers_to_state(point_layer: napari.layers.Points, box_layer: napari.layers.Shapes, i: int) -> str:
    """Get the state of the track from a point layer and shape layer for a given timeframe.

    Only relevant for annotator_tracking.

    Args:
        point_layer: The napari point layer.
        box_layer: The napari box layer.
        i: Timeframe of the data.

    Returns:
        The state of this frame (either "division" or "track").
    """
    state = point_layer.properties["state"]

    points = point_layer.data
    assert points.shape[1] == 3, f"{points.shape}"
    mask = points[:, 0] == i
    if mask.sum() > 0:
        this_state = state[mask].tolist()
    else:
        this_state = []

    box_states = box_layer.properties["state"]
    this_box_states = [
        state for box, state in zip(box_layer.data, box_states)
        if (box[:, 0] == i).all()
    ]
    this_state.extend(this_box_states)

    # we set the state to 'division' if at least one point in this frame has a division label
    if any(st == "division" for st in this_state):
        return "division"
    else:
        return "track"


#
# Helper functions to run (multi-dimensional) segmentation on napari layers.
#


def segment_slices_with_prompts(
    predictor, point_prompts, box_prompts, image_embeddings, shape, track_id=None, update_progress=None,
):
    """@private"""
    assert len(shape) == 3
    image_shape = shape[1:]
    seg = np.zeros(shape, dtype="uint32")

    z_values = np.round(point_prompts.data[:, 0])
    z_values_boxes = np.concatenate([box[:1, 0] for box in box_prompts.data]) if box_prompts.data else\
        np.zeros(0, dtype="int")
    z_values_scribbles = get_scribble_slices(box_prompts) if track_id is None else np.zeros(0, dtype="int")

    if track_id is not None:
        track_ids_points = np.array(list(map(int, point_prompts.properties["track_id"])))
        assert len(track_ids_points) == len(z_values)
        z_values = z_values[track_ids_points == track_id]

        if len(z_values_boxes) > 0:
            track_ids_boxes = np.array(list(map(int, box_prompts.properties["track_id"])))
            assert len(track_ids_boxes) == len(z_values_boxes), f"{len(track_ids_boxes)}, {len(z_values_boxes)}"
            z_values_boxes = z_values_boxes[track_ids_boxes == track_id]

    slices = np.unique(np.concatenate([z_values, z_values_boxes, z_values_scribbles])).astype("int")
    stop_lower, stop_upper = False, False

    if update_progress is None:
        def update_progress(*args):
            pass

    for i in slices:
        scribble_points, scribble_labels = scribble_layer_to_prompts(
            box_prompts, image_shape=image_shape, i=i
        )
        have_scribbles = len(scribble_points) > 0
        points_i = point_layer_to_prompts(
            point_prompts, i, track_id, with_stop_annotation=not have_scribbles
        )

        # do we end the segmentation at the outer slices?
        if points_i is None:

            if i == slices[0]:  # The bottom slice is a stop slice.
                stop_lower = True
                seg[i] = 0
            elif i == slices[-1]:  # The top sloce is a stop slice.
                stop_upper = True
                seg[i] = 0
            else:  # We have a stop annotation somewhere in the middle. Ignore this.
                # Remove this slice from the annotated slices, so that it is segmented via
                # projection in the next step.
                slices = np.setdiff1d(slices, i)
                print(f"You have provided a stop annotation (single red point) in slice {i},")
                print("but you have annotated slices above or below it. This stop annotation will")
                print(f"be ignored and the slice {i} will be segmented normally.")

            update_progress(1)
            continue

        boxes, masks = shape_layer_to_prompts(box_prompts, image_shape, i=i, track_id=track_id)
        points, labels = merge_point_prompts(points_i, (scribble_points, scribble_labels))
        if have_scribbles and not boxes and not np.any(labels == 1):
            warnings.warn(
                f"Ignoring negative-only scribbles on slice {i}: add a positive point, scribble, "
                "box or mask prompt on the same slice.",
                stacklevel=2,
            )
            slices = np.setdiff1d(slices, i)
            update_progress(1)
            continue

        seg_i = prompt_segmentation(
            predictor, points, labels, boxes, masks, image_shape, multiple_box_prompts=False,
            image_embeddings=image_embeddings, i=i
        )
        if seg_i is None:
            print(f"The prompts at slice or frame {i} are invalid and the segmentation was skipped.")
            print("This will lead to a wrong segmentation across slices or frames.")
            print(f"Please correct the prompts in {i} and rerun the segmentation.")
            continue

        seg[i] = seg_i
        update_progress(1)

    return seg, slices, stop_lower, stop_upper


# For advanced batching: match prompts to already segmented objects and continue segmentation.
def _match_prompts(previous_segmentation, points, boxes, seg_ids):
    # Create a mapping between ids and prompts.
    batched_prompts = {}
    # seg_boundaries = find_boundaries(previous_segmentation, mode="inner")
    # indices = distance_transform_edt(seg_boundaries, return_distance=False, return_index=True)
    return batched_prompts


def _batched_interactive_segmentation(predictor, points, labels, boxes, image_embeddings, i, previous_segmentation):
    prev_seg = previous_segmentation if i is None else previous_segmentation[i]
    seg = np.zeros(prev_seg.shape, dtype="uint32")

    # seg_ids = np.unique(previous_segmentation)
    # assert seg_ids[0] == 0

    batched_points, batched_labels = [], []
    negative_points, negative_labels = [], []
    for j in range(len(points)):
        if labels[j] == 1:  # positive point
            batched_points.append(points[j:j+1])
            batched_labels.append(labels[j:j+1])
        else:  # negative points
            negative_points.append(points[j:j+1])
            negative_labels.append(labels[j:j+1])

    batched_prompts = [(None, point, label) for point, label in zip(batched_points, batched_labels)]
    batched_prompts.extend([(box, None, None) for box in boxes])
    batched_prompts = {i: prompt for i, prompt in enumerate(batched_prompts, 1)}

    # For advanced batching: match prompts to already segmented objects and continue segmentation.
    # (This is left here as a reference for how this can be implemented.
    #  I have not decided yet if this is actually a good idea or not.)
    # # If we have no objects: this is the first call for a batched segmentation.
    # # We treat each positive point or box as a separate object.
    # if len(seg_ids) == 1:
    #     # Create a list of all prompts.
    #     batched_prompts = [(None, point, label) for point, label in zip(batched_points, batched_labels)]
    #     batched_prompts.extend([(box, None, None) for box in boxes])
    #     batched_prompts = {i: prompt for i, prompt in enumerate(batched_prompts, 1)}

    # # Otherwise we match the prompts to existing objects.
    # else:
    #     batched_prompts = _match_prompts(prev_seg, batched_points, boxes, seg_ids)

    for seg_id, prompt in batched_prompts.items():
        box, point, label = prompt
        if len(negative_points) > 0:
            if point is None:
                point, label = negative_points, negative_labels
            else:
                point = np.concatenate([point] + negative_points)
                label = np.concatenate([label] + negative_labels)

        if (box is not None) and (point is not None):
            prediction = prompt_based_segmentation.segment_from_box_and_points(
                predictor, box, point, label, image_embeddings=image_embeddings, i=i
            ).squeeze()
        elif (box is not None) and (point is None):
            prediction = prompt_based_segmentation.segment_from_box(
                predictor, box, image_embeddings=image_embeddings, i=i
            ).squeeze()
        else:
            prediction = prompt_based_segmentation.segment_from_points(
                predictor, point, label, image_embeddings=image_embeddings, i=i
            ).squeeze()

        seg[prediction] = seg_id

    return seg


def prompt_segmentation(
    predictor, points, labels, boxes, masks, shape, multiple_box_prompts,
    image_embeddings=None, i=None, box_extension=0, batched=None, previous_segmentation=None,
):
    """@private"""
    assert len(points) == len(labels)
    have_points = len(points) > 0
    have_boxes = len(boxes) > 0

    # No prompts were given, return None.
    if not have_points and not have_boxes:
        return

    # Batched interactive segmentation.
    elif batched:
        assert previous_segmentation is not None
        seg = _batched_interactive_segmentation(
            predictor, points, labels, boxes, image_embeddings, i, previous_segmentation
        )

    # Box and point prompts were given.
    elif have_points and have_boxes:
        if len(boxes) > 1:
            print("You have provided point prompts and more than one box prompt.")
            print("This setting is currently not supported.")
            print("When providing both points and prompts you can only segment one object at a time.")
            return
        mask = masks[0]
        if mask is None:
            seg = prompt_based_segmentation.segment_from_box_and_points(
                predictor, boxes[0], points, labels, image_embeddings=image_embeddings, i=i
            ).squeeze()
        else:
            seg = prompt_based_segmentation.segment_from_mask(
                predictor, mask, box=boxes[0], points=points, labels=labels, image_embeddings=image_embeddings, i=i
            ).squeeze()

    # Only point prompts were given.
    elif have_points and not have_boxes:
        seg = prompt_based_segmentation.segment_from_points(
            predictor, points, labels, image_embeddings=image_embeddings, i=i
        ).squeeze()

    # Only box prompts were given.
    elif not have_points and have_boxes:
        seg = np.zeros(shape, dtype="uint32")

        if len(boxes) > 1 and not multiple_box_prompts:
            print("You have provided more than one box annotation. This is not yet supported in the 3d annotator.")
            print("You can only segment one object at a time in 3d.")
            return

        # Batch this?
        for seg_id, (box, mask) in enumerate(zip(boxes, masks), 1):
            if mask is None:
                prediction = prompt_based_segmentation.segment_from_box(
                    predictor, box, image_embeddings=image_embeddings, i=i
                ).squeeze()
            else:
                prediction = prompt_based_segmentation.segment_from_mask(
                    predictor, mask, box=box, image_embeddings=image_embeddings, i=i,
                    box_extension=box_extension,
                ).squeeze()
            seg[prediction] = seg_id

    return seg


def _compute_movement(seg, t0, t1):

    def compute_center(t):
        # computation with center of mass
        center = np.where(seg[t] == 1)
        center = np.array([np.mean(center[0]), np.mean(center[1])])
        return center

    center0 = compute_center(t0)
    center1 = compute_center(t1)

    move = center0 - center1
    return move.astype("float64")


def _shift_object(mask, motion_model):
    mask_shifted = np.zeros_like(mask)
    shift(mask, motion_model, output=mask_shifted, order=0, prefilter=False)
    return mask_shifted


def track_from_prompts(
    point_prompts, box_prompts, seg, predictor, slices, image_embeddings,
    stop_upper, threshold, projection, motion_smoothing=0.5, box_extension=0, update_progress=None,
):
    """@private
    """
    use_box, use_mask, use_points, use_single_point = _validate_projection(projection)

    if update_progress is None:
        def update_progress(*args):
            pass

    # shift the segmentation based on the motion model and update the motion model
    def _update_motion_model(seg, t, t0, motion_model):
        if t in (t0, t0 + 1):  # this is the first or second frame, we don't have a motion yet
            pass
        elif t == t0 + 2:  # this the third frame, we initialize the motion model
            current_move = _compute_movement(seg, t - 1, t - 2)
            motion_model = current_move
        else:  # we already have a motion model and update it
            current_move = _compute_movement(seg, t - 1, t - 2)
            alpha = motion_smoothing
            motion_model = alpha * motion_model + (1 - alpha) * current_move

        return motion_model

    has_division = False
    motion_model = None
    verbose = False

    t0 = int(slices.min())
    t = t0 + 1
    while True:

        # update the motion model
        motion_model = _update_motion_model(seg, t, t0, motion_model)

        # use the segmentation from prompts if we are in a slice with prompts
        if t in slices:
            seg_prev = None
            seg_t = seg[t]
            # currently using the box layer doesn't work for keeping track of the track state
            # track_state = prompt_layers_to_state(point_prompts, box_prompts, t)
            track_state = prompt_layer_to_state(point_prompts, t)

        # otherwise project the mask (under the motion model) and segment the next slice from the mask
        else:
            if verbose:
                print(f"Tracking object in frame {t} with movement {motion_model}")

            seg_prev = seg[t - 1]
            # shift the segmentation according to the motion model
            if motion_model is not None:
                seg_prev = _shift_object(seg_prev, motion_model)

            seg_t = prompt_based_segmentation.segment_from_mask(
                predictor, seg_prev, image_embeddings=image_embeddings, i=t,
                use_mask=use_mask, use_box=use_box, use_points=use_points,
                box_extension=box_extension, use_single_point=use_single_point,
            )
            track_state = "track"

            # are we beyond the last slice with prompt?
            # if no: we continue tracking because we know we need to connect to a future frame
            # if yes: we only continue tracking if overlaps are above the threshold
            if t < slices[-1]:
                seg_prev = None

            update_progress(1)

        if (threshold is not None) and (seg_prev is not None):
            iou = util.compute_iou(seg_prev, seg_t)
            if iou < threshold:
                msg = f"Segmentation stopped at frame {t} due to IOU {iou} < {threshold}."
                print(msg)
                break

        # stop if we have a division
        if track_state == "division":
            has_division = True
            break

        seg[t] = seg_t
        t += 1

        # stop tracking if we have stop upper set (i.e. single negative point was set to indicate stop track)
        if t == slices[-1] and stop_upper:
            break

        # stop if we are at the last slce
        if t == seg.shape[0]:
            break

    return seg, has_division


def _sync_embedding_widget(widget, model_type, save_path, checkpoint_path, device, tile_shape, halo):

    # VFM families (DINO / UNI / SAM3) live in the classification widget's advanced tier. Let the widget
    # place the selection. The SAM family/size parsing below is skipped for these names: it parses the
    # size positionally (vit_<size>), which does not apply and even index-errors on short names ('sam3').
    from ..models.vfm import is_vfm_model
    is_vfm = is_vfm_model(model_type)
    if is_vfm and hasattr(widget, "set_model_family_size"):
        family, size = widget._family_and_size_for_model(model_type)
        widget.set_model_family_size(family, size)

    if not is_vfm:
        # Update the index for model family, eg. 'Natural Images (SAM)', 'Light Microscopy', etc.
        supported_dropdown_maps = {
            "lm": "Light Microscopy",
            "em_organelles": "Electron Microscopy",
            "medical_imaging": "Medical Imaging",
            "histopathology": "Histopathology",
        }

        if model_type.startswith("hvit"):  # SAM2 models, eg. 'hvit_t'.
            # Finetuned SAM2 families carry a suffix (e.g. 'hvit_t_cells' -> 'Microscopy'); the plain
            # backbones ('hvit_t', ...) are natural-image models.
            if model_type.endswith("_cells"):
                model_family = "Microscopy"
            else:
                model_family = "Natural Images"
        else:
            model_family = "Natural Images (SAM)"  # No suffix match: stick to 'Natural Images (SAM)'.
            for k, v in supported_dropdown_maps.items():
                if model_type.endswith(k):
                    model_family = v
                    break

        index = widget.model_family_dropdown.findText(model_family)
        if index >= 0:
            widget.model_family_dropdown.setCurrentIndex(index)

        # Update the index for model size, eg. 'base', 'tiny', etc.
        size_map = {"t": "tiny", "s": "small", "b": "base", "l": "large", "h": "huge"}
        size_idx = 5 if model_type.startswith("hvit") else 4
        model_size = size_map.get(model_type[size_idx])

        if model_size is not None:
            index = widget.model_size_dropdown.findText(model_size)
            if index >= 0:
                widget.model_size_dropdown.setCurrentIndex(index)

    if save_path is not None and isinstance(save_path, str):
        widget.embeddings_save_path_param.setText(str(save_path))

    if checkpoint_path is not None:
        widget.custom_weights_param.setText(str(checkpoint_path))

    if device is not None:
        widget.device = device
        index = widget.device_dropdown.findText(device)
        widget.device_dropdown.setCurrentIndex(index)

    if tile_shape is not None:
        widget.tile_x_param.setValue(tile_shape[0])
        widget.tile_y_param.setValue(tile_shape[1])
        # Enable tiling so the loaded tile shape is used and shown.
        widget.tiling_dropdown.setCurrentText("yes")

    if halo is not None:
        widget.halo_x_param.setValue(halo[0])
        widget.halo_y_param.setValue(halo[1])


# Read parameters from checkpoint path if it is given instead.
def _sync_autosegment_widget(widget, model_type, checkpoint_path, update_decoder=None):
    if update_decoder is not None:
        widget._reset_segmentation_mode(update_decoder)

    # Apply per-model default settings for the v1 generators if the widget exposes them
    # (e.g. the automatic tracking widget). The new dense/sparse segmentation widget does not
    # expose these parameters, since its backend is deferred, so the updates are skipped there.
    if getattr(widget, "with_decoder", False):
        settings = model_settings.AIS_SETTINGS.get(model_type, {})
        params = ("center_distance_thresh", "boundary_distance_thresh")
    else:
        settings = model_settings.AMG_SETTINGS.get(model_type, {})
        params = ("pred_iou_thresh", "stability_score_thresh", "min_object_size")

    for param in params:
        if param in settings and hasattr(widget, f"{param}_param"):
            getattr(widget, f"{param}_param").setValue(settings[param])


# Read parameters from checkpoint path if it is given instead.
def _sync_ndsegment_widget(widget, model_type, checkpoint_path):
    settings = model_settings.ND_SEGMENT_SETTINGS.get(model_type, {})

    if "projection_mode" in settings:
        projection_mode = settings["projection_mode"]
        widget.projection = projection_mode
        index = widget.projection_dropdown.findText(projection_mode)
        if index > 0:
            widget.projection_dropdown.setCurrentIndex(index)

    params = ("iou_threshold", "box_extension")
    for param in params:
        if param in settings:
            getattr(widget, f"{param}_param").setValue(settings[param])


def _load_amg_state(embedding_path):
    if embedding_path is None or not os.path.exists(embedding_path):
        return {"cache_folder": None}

    cache_folder = os.path.join(embedding_path, "amg_state")
    os.makedirs(cache_folder, exist_ok=True)
    amg_state = {"cache_folder": cache_folder}

    state_paths = glob(os.path.join(cache_folder, "*.pkl"))
    for path in state_paths:
        with open(path, "rb") as f:
            state = pickle.load(f)
        i = int(Path(path).stem.split("-")[-1])
        amg_state[i] = state
    return amg_state


def _load_is_state(embedding_path):
    if embedding_path is None or not os.path.exists(embedding_path):
        return {"cache_path": None}

    cache_path = os.path.join(embedding_path, "is_state.h5")
    is_state = {"cache_path": cache_path}

    with h5py.File(cache_path, "a") as f:
        for name, g in f.items():
            i = int(name.split("-")[-1])
            state = {
                "foreground": g["foreground"][:],
                "boundary_distances": g["boundary_distances"][:],
                "center_distances": g["center_distances"][:],
            }
            is_state[i] = state

    return is_state


def _autoseg_state_descriptor(embedding_path, mode):
    """Descriptor of the SAM2 automatic-segmentation state cache in the embedding Zarr.

    Returns the embedding path and mode ('amg' or 'ais'); the state itself is loaded on demand by
    `micro_sam.precompute_state.cache_autoseg_state`. The SAM2 automatic segmentation widget
    reads/writes the cache directly, so this only records where it lives.
    """
    if embedding_path is None or not os.path.exists(embedding_path):
        return {"embedding_path": None, "mode": mode}
    return {"embedding_path": embedding_path, "mode": mode}
