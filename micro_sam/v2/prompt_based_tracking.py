"""Promptable tracking of one object in a 3d timeseries, (T, Z, Y, X), with the SAM2 video predictor.

The image encoder of SAM2 sees one frame at a time. So the embeddings of a (T, Z, Y, X) timeseries are the
embeddings of its T * Z planes. The video predictor can then follow an object along any sequence of these planes.
`PromptableTracking3D` uses two such sequences:
- The planes of one timepoint: the tracker propagates the object through z, which gives its volume.
- One plane per timepoint: the tracker propagates the object through time.
The plane of the next timepoint is the largest plane of the object in the current timepoint. So the sequence
through time follows an object that moves in z.
"""

from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D, _crop_to_original_shape


class _PlaneSubset:
    """The planes of a stored per-plane array at the given indices. The index list can change at any time."""

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __getitem__(self, index):
        return self.data[self.indices[index]]


def _subset_embeddings(embeddings, indices):
    """Get a view of volume embeddings whose frame i is the plane indices[i]."""
    subset = dict(embeddings)
    subset["features"] = _PlaneSubset(embeddings["features"], indices)
    subset["fpn"] = [_PlaneSubset(level, indices) for level in embeddings["fpn"]]
    return subset


def _largest_plane(volume_mask):
    return int(np.argmax(volume_mask.sum(axis=(1, 2))))


class PromptableTracking3D:
    """Promptable tracking of one object in a 3d timeseries.

    The prompts of a timepoint give the volume of the object in it. The tracker propagates the object forward in
    time from the first prompted timepoint. A later prompted timepoint corrects the track.

    Args:
        predictor: The SAM2 video predictor.
        timeseries: The timeseries with the shape (T, Z, Y, X). The tracker only reads its shape.
        embeddings: The volume embeddings of the timeseries with its first two axes flattened. So the plane
            (t, z) is the frame t * Z + z. Compute them with
            `precompute_image_embeddings(predictor, timeseries.reshape(-1, Y, X), ndim=3)`.
        device: The torch device. None uses the device of the predictor.
        z_patience: The number of consecutive planes without the object that stops the propagation through z.
    """

    def __init__(self, predictor, timeseries, embeddings, device=None, z_patience=2):
        if timeseries.ndim != 4:
            raise ValueError(f"The timeseries has the shape {timeseries.shape}. It must have the shape (T, Z, Y, X).")
        if embeddings.get("input_size") is None:
            raise ValueError(
                "The 3d tracking does not support tiled embeddings. Compute the embeddings without tiling."
            )
        self.predictor = predictor
        self.shape = tuple(int(s) for s in timeseries.shape)
        self.z_patience = z_patience
        n_timepoints, n_planes, height, width = self.shape

        # The segmenters read the shape of their volume only, so these arrays hold no data.
        self._volume_planes = list(range(n_planes))
        self._volume_segmenter = PromptableSegmentation3D(
            predictor, np.broadcast_to(np.uint8(0), (n_planes, height, width)),
            _subset_embeddings(embeddings, self._volume_planes), device=device,
        )
        self._time_planes = [t * n_planes for t in range(n_timepoints)]
        self._time_segmenter = PromptableSegmentation3D(
            predictor, np.broadcast_to(np.uint8(0), (n_timepoints, height, width)),
            _subset_embeddings(embeddings, self._time_planes), device=device,
        )
        self._points = {}  # (t, z) -> list of (y, x, label)
        self._boxes = {}  # (t, z) -> list of boxes

    def add_point_prompts(self, t: int, z: int, points: np.ndarray, point_labels: np.ndarray):
        """Add point prompts, (N, 2) in (y, x) order, on the plane z of the timepoint t."""
        prompts = self._points.setdefault((int(t), int(z)), [])
        for (y, x), label in zip(points, point_labels):
            prompts.append((float(y), float(x), int(label)))

    def add_box_prompts(self, t: int, z: int, boxes: np.ndarray):
        """Add box prompts, in (y0, x0, y1, x1) order, on the plane z of the timepoint t."""
        self._boxes.setdefault((int(t), int(z)), []).extend(np.asarray(box, dtype="float32") for box in boxes)

    def remove_prompts(self, t: int):
        """Remove the prompts on all planes of the timepoint t."""
        self._points = {key: value for key, value in self._points.items() if key[0] != t}
        self._boxes = {key: value for key, value in self._boxes.items() if key[0] != t}

    def reset_predictor(self):
        """Remove all prompts and free the device memory of the propagation."""
        self._points, self._boxes = {}, {}
        self._volume_segmenter.reset_predictor()
        self._time_segmenter.reset_predictor()

    def prompted_timepoints(self) -> List[int]:
        """Get the sorted timepoints that have a prompt."""
        return sorted({t for t, _ in (*self._points, *self._boxes)})

    def _enter_timepoint(self, t):
        self._volume_segmenter.release()
        self._volume_planes[:] = range(t * self.shape[1], (t + 1) * self.shape[1])

    def segment_timepoint(self, t: int) -> np.ndarray:
        """Segment the object at the timepoint t from the prompts on its planes, without tracking.

        Returns:
            The boolean mask of the object with the shape (Z, Y, X).
        """
        self._enter_timepoint(t)
        segmenter = self._volume_segmenter
        for (prompt_t, z), boxes in self._boxes.items():
            if prompt_t == t:
                segmenter.add_box_prompts(frame_ids=z, boxes=boxes)
        for (prompt_t, z), points in self._points.items():
            if prompt_t == t:
                points = np.array(points)
                segmenter.add_point_prompts(frame_ids=z, points=points[:, :2], point_labels=points[:, 2])
        return segmenter.predict(early_stop_patience=self.z_patience) == 1

    def _volume_from_mask(self, t, z, mask):
        """Segment the volume of the object at a timepoint from its mask on one plane."""
        self._enter_timepoint(t)
        self._volume_segmenter.add_mask_prompts(frame_ids=z, masks=[mask], refine=False)
        return self._volume_segmenter.predict(early_stop_patience=self.z_patience) == 1

    def get_progress_total(self, t_range: Optional[Tuple[int, int]] = None) -> int:
        """Get the number of timepoints that `predict` processes: from the first prompted one to the end."""
        t_min, t_max = (0, self.shape[0] - 1) if t_range is None else t_range
        prompted = [t for t in self.prompted_timepoints() if t_min <= t <= t_max]
        return t_max - min(prompted) + 1 if prompted else 0

    def propagate(self, t_range: Optional[Tuple[int, int]] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Track the object forward in time from its first prompted timepoint.

        Args:
            t_range: The inclusive bound (t_min, t_max) on the timepoints. None uses all timepoints.

        Yields:
            The timepoint and the boolean mask of the object in it with the shape (Z, Y, X).
        """
        n_timepoints, n_planes, height, width = self.shape
        t_min, t_max = (0, n_timepoints - 1) if t_range is None else t_range

        time_segmenter = self._time_segmenter
        time_segmenter.release()
        prompted_volumes = {}
        for t in self.prompted_timepoints():
            if not t_min <= t <= t_max:
                continue
            volume_mask = self.segment_timepoint(t)
            if not volume_mask.any():
                continue
            z = _largest_plane(volume_mask)
            prompted_volumes[t] = volume_mask
            self._time_planes[t] = t * n_planes + z
            time_segmenter.add_mask_prompts(frame_ids=t, masks=[volume_mask[z]], refine=False)
        if not prompted_volumes:
            return

        t_start = min(prompted_volumes)
        for t, _, logits in self.predictor.propagate_in_video(
            time_segmenter.inference_state, start_frame_idx=t_start, max_frame_num_to_track=t_max - t_start,
        ):
            z = self._time_planes[t] - t * n_planes
            if t in prompted_volumes:
                volume_mask = prompted_volumes[t]
            else:
                mask = _crop_to_original_shape((logits[0, 0] > 0).cpu().numpy(), (height, width))
                if mask.any():
                    volume_mask = self._volume_from_mask(t, z, mask)
                else:
                    volume_mask = np.zeros((n_planes, height, width), dtype=bool)
            yield t, volume_mask

            # The predictor reads the plane of the next timepoint after this point.
            if volume_mask.any():
                z = _largest_plane(volume_mask)
            if t + 1 < n_timepoints and t + 1 not in prompted_volumes:
                self._time_planes[t + 1] = (t + 1) * n_planes + z

    def predict(
        self, t_range: Optional[Tuple[int, int]] = None, update_progress: Optional[Callable] = None,
    ) -> Dict[int, np.ndarray]:
        """Track the object and return its mask per timepoint.

        Args:
            t_range: The inclusive bound (t_min, t_max) on the timepoints. None uses all timepoints.
            update_progress: The callback for the progress. It gets the number of new timepoints.

        Returns:
            The dictionary that maps each timepoint to the boolean mask of the object with the shape (Z, Y, X).
        """
        track = {}
        for t, volume_mask in self.propagate(t_range=t_range):
            track[t] = volume_mask
            if update_progress is not None:
                update_progress(1)
        return track
