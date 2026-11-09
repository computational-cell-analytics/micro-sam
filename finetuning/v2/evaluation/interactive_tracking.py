"""A headless copy of the micro-sam2 tracking annotator and a simulated user that drives it.

`TrackingAnnotator` keeps the state of the tracking annotator ('micro_sam.sam_annotator.annotator_tracking') without
napari: the point and box prompts with their track id and track state, the 'current_object' and 'committed_objects'
layers and the lineage. Its actions copy the annotator's:
- `segment` is 'Segment Object' with 'Apply to All Frames' ('UnifiedSegmentWidget._run_tracking'): propagate the
  prompts of the current track forward in time, bounded by a stop annotation or a division.
- a 'division' point ends the mother track at its frame and seeds two daughter tracks ('_update_lineage').
- `commit` is 'Commit' ('commit_track' / '_commit_impl'): move the lineage to the committed objects.
The predictor and the interactive segmenter come from 'AnnotatorState.initialize_predictor', as in the annotator.

`SimulatedUser` annotates every lineage of the ground-truth like a human would: a box (or point) on the first frame
of a track, corrections on the worst frame, a stop where the track ends, and a division where it divides.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from micro_sam.v2.util import resolve_default_tiling
from micro_sam.sam_annotator._state import AnnotatorState


@dataclass
class PointPrompt:
    t: int
    y: float
    x: float
    label: int  # 1 positive, 0 negative.
    state: str  # 'track' or 'division'.
    track_id: int


@dataclass
class BoxPrompt:
    t: int
    box: np.ndarray  # (y0, x0, y1, x1).
    track_id: int


def commit_objects(committed, committed_lineages, current_object, lineage):
    """Commit a lineage like 'commit_track' with the default 'pixels' preserve mode.

    The track ids of the lineage are offset by the largest committed id, and committed pixels are not overwritten.
    """
    id_offset = int(committed.max())
    mask = (current_object != 0) & (committed == 0)
    committed[mask] = current_object[mask] + id_offset
    committed_lineages.append(
        {parent + id_offset: [child + id_offset for child in children] for parent, children in lineage.items()}
    )


class TrackingAnnotator:
    """The state and the actions of the tracking annotator, without napari.

    Args:
        image: The timeseries, (T, Y, X).
        model_type: The SAM2 model, e.g. 'hvit_l_cells'.
        embedding_path: Where the embeddings are cached. None caches them in a temporary folder.
        device: The torch device.
    """

    def __init__(self, image, model_type, embedding_path=None, device=None):
        state = AnnotatorState()
        state.reset_state()
        state.image_shape = image.shape
        tile_shape, halo = resolve_default_tiling(image.shape, None, None, embedding_path)
        state.initialize_predictor(
            image, model_type=model_type, save_path=embedding_path, halo=halo, tile_shape=tile_shape,
            prefer_decoder=True, ndim=3, device=device, use_cli=True,
        )
        self.segmenter = state.interactive_segmenter
        self.shape = image.shape
        self.committed_objects = np.zeros(self.shape, dtype="uint32")
        self.committed_lineages = []
        self.reset_track_state()

    def reset_track_state(self):
        """Clear the prompts, the current object and the lineage ('_reset_tracking_state' and 'clear_annotations')."""
        self.current_track_id = 1
        self.lineage = {1: []}
        self.current_object = np.zeros(self.shape, dtype="uint32")
        self.points, self.boxes = [], []
        # Every change of the current object, as (tag, kind, track id, flat indices or first frame). The caller sets
        # 'log_tag', so that the changes can be replayed in part (see 'replay').
        self.log, self.log_tag = [], 0

    def add_point(self, t, yx, positive=True, state="track", track_id=None):
        """Place a point prompt for a track, by default the current one."""
        track_id = self.current_track_id if track_id is None else track_id
        self.points.append(PointPrompt(int(t), float(yx[0]), float(yx[1]), int(positive), state, int(track_id)))

    def add_box(self, t, box, track_id=None):
        """Place a box prompt, (y0, x0, y1, x1), for a track, by default the current one."""
        track_id = self.current_track_id if track_id is None else track_id
        self.boxes.append(BoxPrompt(int(t), np.asarray(box, dtype="float64"), int(track_id)))

    def remove_prompts(self, t, track_id=None):
        """Delete the point and box prompts of a track on a frame, as a user deletes them in the layers.

        The division markers stay.
        """
        track_id = self.current_track_id if track_id is None else track_id
        self.points = [p for p in self.points if not (p.t == t and p.track_id == track_id and p.state != "division")]
        self.boxes = [b for b in self.boxes if not (b.t == t and b.track_id == track_id)]

    def add_division(self, t, yx, track_id=None):
        """Mark the division of a track on its last frame, with a point in the 'division' state."""
        self.add_point(t, yx, positive=True, state="division", track_id=track_id)

    def _frame_points(self, t, track_id):
        """Get the point prompts of a track on a frame without the division markers, and whether they are a stop."""
        points = [p for p in self.points if p.t == t and p.track_id == track_id and p.state != "division"]
        has_box = any(b.t == t and b.track_id == track_id for b in self.boxes)
        # A lone negative point beside no box is a stop annotation ('collect_frame_prompts').
        is_stop = len(points) == 1 and points[0].label == 0 and not has_box
        return points, is_stop

    def _division_frame(self, track_id):
        frames = [p.t for p in self.points if p.state == "division" and p.track_id == track_id]
        return min(frames) if frames else None

    def _mother_division_frame(self, track_id):
        for mother, daughters in self.lineage.items():
            if track_id in daughters:
                return self._division_frame(mother)
        return None

    def _update_lineage(self, mother):
        """Seed two daughter tracks of a dividing track, once."""
        if mother not in self.lineage or len(self.lineage[mother]) > 0:
            return
        daughter1 = max(self.lineage.keys()) + 1
        daughter2 = daughter1 + 1
        self.lineage[mother] = [daughter1, daughter2]
        self.lineage[daughter1] = []
        self.lineage[daughter2] = []

    def _propagate_track(self, track_id, division_frame):
        """Propagate the prompts of one track forward in time ('propagate_track')."""
        self.segmenter.reset_predictor()

        prompted_frames, stop_frames = [], []
        have_positive_cue = False
        for t in sorted({p.t for p in self.points}):
            points, is_stop = self._frame_points(t, track_id)
            if is_stop:
                stop_frames.append(t)
                continue
            if not points:
                continue
            labels = np.array([p.label for p in points])
            have_positive_cue = have_positive_cue or bool(np.any(labels == 1))
            self.segmenter.add_point_prompts(
                frame_ids=t, points=np.array([[p.y, p.x] for p in points]), point_labels=labels,
            )
            prompted_frames.append(t)

        for t in sorted({b.t for b in self.boxes}):
            boxes = [b.box for b in self.boxes if b.t == t and b.track_id == track_id]
            for box in boxes:
                self.segmenter.add_box_prompts(frame_ids=t, boxes=[box])
            if boxes:
                have_positive_cue = True
                prompted_frames.append(t)

        if not prompted_frames or not have_positive_cue:
            return None

        annotated = sorted(set(prompted_frames) | set(stop_frames))
        stop_upper = annotated[-1] in stop_frames
        z_lo = min(prompted_frames)
        z_hi = max(prompted_frames) if stop_upper else self.shape[0] - 1
        if division_frame is not None:
            z_hi = min(z_hi, division_frame)
        if z_hi < z_lo:
            return None
        return self.segmenter.predict(early_stop_patience=None, z_range=(z_lo, z_hi))

    def segment(self):
        """Run 'Segment Object' for the current track over all frames ('_run_tracking')."""
        track_id = self.current_track_id
        division_frame = self._division_frame(track_id)
        segmented = division_frame is not None and bool(
            np.any(self.current_object[:division_frame + 1] == track_id)
        )
        if segmented:
            # A division of a segmented track only erases it after the division frame.
            after = self.current_object[division_frame + 1:]
            after[after == track_id] = 0
            self.log.append((self.log_tag, "truncate", track_id, division_frame + 1))
            self._update_lineage(track_id)
            return

        seg = self._propagate_track(track_id, division_frame)
        if seg is not None:
            mother_division = self._mother_division_frame(track_id)
            if mother_division is not None:
                seg = seg.copy()
                seg[:mother_division + 1] = 0
            self.current_object[self.current_object == track_id] = 0
            self.current_object[seg == 1] = track_id
            self.log.append((self.log_tag, "write", track_id, np.flatnonzero(seg == 1)))
        if division_frame is not None:
            self._update_lineage(track_id)

    @staticmethod
    def replay(log, shape, max_tag):
        """Rebuild the current object from the logged changes whose tag is at most 'max_tag'."""
        current = np.zeros(shape, dtype="uint32")
        for tag, kind, track_id, value in log:
            if tag > max_tag:
                continue
            if kind == "write":
                current[current == track_id] = 0
                current.flat[value] = track_id
            else:
                after = current[value:]
                after[after == track_id] = 0
        return current

    def commit(self):
        """Run 'Commit' for the current lineage and reset the tracking state."""
        commit_objects(self.committed_objects, self.committed_lineages, self.current_object, self.lineage)
        self.segmenter.reset_predictor()
        self.reset_track_state()


class V1TrackingAnnotator:
    """The state and the actions of the micro-sam v1 tracking annotator, without napari.

    Copies the tracking of micro-sam v1 (before 'Refactor tracking annotator', #1251): the frames with prompts of the
    current track are segmented from their prompts ('segment_slices_with_prompts'), then the track is propagated
    frame by frame from the first of them with the mask of the previous frame and a motion model
    ('track_from_prompts'), with the defaults of its widget. A frame with a 'division' point stops the track before
    it, so a user marks a division on the first frame of the daughters. It has the interface of `TrackingAnnotator`.

    Args:
        image: The timeseries, (T, Y, X).
        model_type: The micro-sam v1 model, e.g. 'vit_l_lm'.
        embedding_path: Where the embeddings are cached.
        device: The torch device.
    """

    # The defaults of the v1 tracking widget.
    projection = "single_point"
    iou_threshold = 0.5
    box_extension = 0.05
    motion_smoothing = 0.5

    def __init__(self, image, model_type="vit_l_lm", embedding_path=None, device=None):
        from micro_sam.v1.util import get_sam_model, precompute_image_embeddings
        self.predictor = get_sam_model(model_type=model_type, device=device)
        self.image_embeddings = precompute_image_embeddings(
            self.predictor, image, save_path=embedding_path, ndim=3, verbose=True,
        )
        self.shape = image.shape
        self.committed_objects = np.zeros(self.shape, dtype="uint32")
        self.committed_lineages = []
        self.reset_track_state()

    reset_track_state = TrackingAnnotator.reset_track_state
    add_point = TrackingAnnotator.add_point
    add_box = TrackingAnnotator.add_box
    remove_prompts = TrackingAnnotator.remove_prompts
    _update_lineage = TrackingAnnotator._update_lineage
    replay = TrackingAnnotator.replay

    def add_division(self, t, yx, track_id=None):
        """Mark the division of a track whose last frame is t: v1 takes the marker on the next frame."""
        self.add_point(t + 1, yx, positive=True, state="division", track_id=track_id)

    def _frame_prompts(self, t, track_id):
        """The point prompts (y, x), labels and boxes of a track on a frame; division points are prompts in v1."""
        points = [p for p in self.points if p.t == t and p.track_id == track_id]
        boxes = [b.box for b in self.boxes if b.t == t and b.track_id == track_id]
        return np.array([[p.y, p.x] for p in points]).reshape(-1, 2), np.array([p.label for p in points]), boxes

    def _segment_prompted_frames(self, track_id):
        """Segment the frames with prompts of a track ('segment_slices_with_prompts')."""
        from micro_sam.v1 import prompt_based_segmentation as pbs
        seg = np.zeros(self.shape, dtype="uint32")
        frames = sorted({p.t for p in self.points if p.track_id == track_id} |
                        {b.t for b in self.boxes if b.track_id == track_id})
        stop_upper = False
        for t in list(frames):
            points, labels, boxes = self._frame_prompts(t, track_id)
            if len(points) == 1 and labels[0] == 0 and not boxes:  # A lone negative point is a stop annotation.
                if t == frames[-1]:
                    stop_upper = True
                elif t != frames[0]:
                    frames.remove(t)
                continue
            emb = dict(image_embeddings=self.image_embeddings, i=t)
            if len(points) and boxes:
                pred = pbs.segment_from_box_and_points(self.predictor, boxes[0], points, labels, **emb)
            elif boxes:
                pred = pbs.segment_from_box(self.predictor, boxes[0], **emb)
            else:
                pred = pbs.segment_from_points(self.predictor, points, labels, **emb)
            seg[t] = np.asarray(pred).squeeze() > 0
        return seg, np.array(frames, dtype="int"), stop_upper

    def _is_division(self, t):
        """'prompt_layer_to_state': a frame is in the division state if any point on it is."""
        return any(p.t == t and p.state == "division" for p in self.points)

    def _track(self, seg, frames, stop_upper):
        """Propagate the track from its first prompted frame ('track_from_prompts')."""
        from scipy.ndimage import shift
        from micro_sam.util import compute_iou
        from micro_sam.v1 import prompt_based_segmentation as pbs
        from micro_sam.v1.multi_dimensional_segmentation import _validate_projection
        use_box, use_mask, use_points, use_single_point = _validate_projection(self.projection)

        def center(t):
            ys, xs = np.where(seg[t] == 1)
            return np.array([ys.mean(), xs.mean()])

        motion_model, has_division = None, False
        t0 = int(frames.min())
        t = t0 + 1
        while t < seg.shape[0]:
            if t == t0 + 2:
                motion_model = center(t - 1) - center(t - 2)
            elif t > t0 + 2:
                motion_model = self.motion_smoothing * motion_model + (1 - self.motion_smoothing) * (
                    center(t - 1) - center(t - 2))
            if t in frames:
                seg_prev, seg_t = None, seg[t]
                state = "division" if self._is_division(t) else "track"
            else:
                seg_prev = seg[t - 1]
                if not seg_prev.any():  # The track is lost: SAM has no mask to continue from.
                    break
                if motion_model is not None and not np.isnan(motion_model).any():
                    shifted = np.zeros_like(seg_prev)
                    shift(seg_prev, motion_model, output=shifted, order=0, prefilter=False)
                    seg_prev = shifted
                seg_t = pbs.segment_from_mask(
                    self.predictor, seg_prev, image_embeddings=self.image_embeddings, i=t, use_mask=use_mask,
                    use_box=use_box, use_points=use_points, box_extension=self.box_extension,
                    use_single_point=use_single_point,
                )
                seg_t = (np.asarray(seg_t).squeeze() > 0).astype(seg.dtype)
                state = "track"
                if t < frames[-1]:
                    seg_prev = None
            if seg_prev is not None and compute_iou(seg_prev, seg_t) < self.iou_threshold:
                break
            if state == "division":
                has_division = True
                break
            seg[t] = seg_t
            t += 1
            if t == frames[-1] and stop_upper:
                break
        return seg, has_division

    def segment(self):
        """Run 'Track Object' of the v1 annotator for the current track."""
        track_id = self.current_track_id
        seg, frames, stop_upper = self._segment_prompted_frames(track_id)
        if len(frames) == 0:
            return
        seg, has_division = self._track(seg, frames, stop_upper)
        if has_division and len(self.lineage.get(track_id, [])) == 0:
            self._update_lineage(track_id)
        self.current_object[self.current_object == track_id] = 0
        self.current_object[seg == 1] = track_id
        self.log.append((self.log_tag, "write", track_id, np.flatnonzero(seg == 1)))

    def commit(self):
        commit_objects(self.committed_objects, self.committed_lineages, self.current_object, self.lineage)
        self.reset_track_state()


def center_of(mask):
    """Get the point of a mask farthest from its boundary, in its largest component, as (y, x)."""
    components, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum_labels(mask, components, index=np.arange(1, n + 1))
        mask = components == (np.argmax(sizes) + 1)
    distances = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    return np.unravel_index(np.argmax(distances), mask.shape)


def box_of(mask):
    """Get the bounding box of a mask, (y0, x0, y1, x1)."""
    ys, xs = np.where(mask)
    return np.array([ys.min(), xs.min(), ys.max() + 1, xs.max() + 1])


def digest(array):
    """Get a hash of an array, to compare a large result without keeping a copy."""
    return hashlib.blake2b(np.ascontiguousarray(array).data).hexdigest()


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


class SimulatedUser:
    """A simulated user who annotates every ground-truth lineage with the tracking annotator.

    For each track of a lineage, from the root down to the daughters:
    1. A box (or a point) on the first frame of the track, then 'Segment Object'.
    2. Where the track ends without dividing before the last frame, and the propagation goes on: a positive point on
       its last frame and a stop (a lone negative point) on the next one.
    3. Up to 'n_corrections' corrections on the frame with the lowest IoU while it is below 'accept_iou'. With
       'points', a positive point in the largest missed region and a negative point in the largest false region, at
       their centers. With 'box', the prompts of the track on that frame are deleted and its box is drawn again.
    4. Where the track divides: a division point on its last frame, then 'Segment Object' seeds the daughters.
    Then 'Commit'. The corrections come before the division, because the annotator only truncates a segmented
    mother once its division is marked and does not propagate it again.

    The changes of the annotator are tagged with the number of the correction of the track, so the result for a
    smaller correction budget follows from the same run by a replay (see `results_per_budget`).

    Args:
        annotator: The `TrackingAnnotator`.
        gt_masks: The dense ground-truth track masks, (T, Y, X).
        gt_tracks: The ground-truth lineage table, with the columns 'track', 'start', 'end', 'parent'.
        first_prompt: 'box' or 'point'.
        n_corrections: The largest number of corrections per track.
        correction: 'points' or 'box', see above.
        accept_iou: The IoU on every frame of a track above which the user accepts it.
    """

    def __init__(
        self, annotator, gt_masks, gt_tracks, first_prompt="box", n_corrections=4, accept_iou=0.9, correction="box",
    ):
        if correction not in ("points", "box"):
            raise ValueError(f"Invalid correction '{correction}', choose 'points' or 'box'.")
        self.correction = correction
        self.annotator = annotator
        self.gt_masks = gt_masks
        self.tracks = gt_tracks.set_index("track")
        self.first_prompt = first_prompt
        self.n_corrections = n_corrections
        self.accept_iou = accept_iou
        self.children = {track: [] for track in self.tracks.index}
        for track, parent in self.tracks["parent"].items():
            if parent in self.children:
                self.children[parent].append(track)
        self.n_frames = gt_masks.shape[0]
        self.lineage_records = []

    def roots(self):
        parents = self.tracks["parent"]
        roots = [track for track in self.tracks.index if parents[track] not in self.children]
        return sorted(roots, key=lambda track: (self.tracks.loc[track, "start"], track))

    def _gt_frames(self, gt_track):
        start, end = int(self.tracks.loc[gt_track, "start"]), int(self.tracks.loc[gt_track, "end"])
        return [t for t in range(start, end + 1) if np.any(self.gt_masks[t] == gt_track)]

    def _track_mask(self, track_id):
        return self.annotator.current_object == track_id

    def _correct(self, track_id, gt_track, frames):
        """Place one correction on the worst frame of a track; return the number of prompts, 0 if the user accepts."""
        pred = self._track_mask(track_id)
        if self.correction == "box":
            # A frame with the box of the track already has the box a redraw would give.
            boxed = {b.t for b in self.annotator.boxes if b.track_id == track_id}
            frames = [t for t in frames if t not in boxed]
            if not frames:
                return 0
        scores = [iou(pred[t], self.gt_masks[t] == gt_track) for t in frames]
        if min(scores) >= self.accept_iou:
            return 0
        t = frames[int(np.argmin(scores))]
        gt, mask = self.gt_masks[t] == gt_track, pred[t]
        if self.correction == "box":
            self.annotator.remove_prompts(t)
            self.annotator.add_box(t, box_of(gt))
            self.annotator.segment()
            return 1
        missed, false = gt & ~mask, mask & ~gt
        n_prompts = 0
        # A lone negative point would be a stop, so a positive point always comes with it.
        positive = missed if missed.any() else gt
        self.annotator.add_point(t, center_of(positive), positive=True)
        n_prompts += 1
        if false.any():
            self.annotator.add_point(t, center_of(false), positive=False)
            n_prompts += 1
        self.annotator.segment()
        return n_prompts

    def annotate_track(self, track_id, gt_track):
        """Annotate one track and return its record: the masks after each correction and the prompts placed."""
        annotator = self.annotator
        annotator.current_track_id = track_id
        annotator.log_tag = 0
        frames = self._gt_frames(gt_track)
        start, end = frames[0], frames[-1]
        first_mask = self.gt_masks[start] == gt_track
        if self.first_prompt == "box":
            annotator.add_box(start, box_of(first_mask))
        else:
            annotator.add_point(start, center_of(first_mask), positive=True)
        annotator.segment()
        n_prompts = [1]

        children = self.children[gt_track]
        if not children and end < self.n_frames - 1 and self._track_mask(track_id)[end + 1:].any():
            if end != start:
                annotator.add_point(end, center_of(self.gt_masks[end] == gt_track), positive=True)
            annotator.add_point(end + 1, center_of(self.gt_masks[end] == gt_track), positive=False)
            annotator.segment()
            n_prompts[0] += 1 if end == start else 2

        for correction in range(1, self.n_corrections + 1):
            annotator.log_tag = correction
            placed = self._correct(track_id, gt_track, frames)
            if placed == 0:
                break
            n_prompts.append(placed)

        division = None
        annotator.log_tag = 0
        if children:
            annotator.add_division(end, center_of(self.gt_masks[end] == gt_track))
            annotator.segment()
            n_prompts[0] += 1
            division = end
        return {"track_id": track_id, "gt_track": gt_track, "n_prompts": n_prompts, "division": division}

    def annotate_lineage(self, root):
        """Annotate the lineage of a root track, commit it and keep the record of each track."""
        queue = [(root, 1)]
        records = []
        while queue:
            gt_track, track_id = queue.pop(0)
            if not self._gt_frames(gt_track):
                continue
            record = self.annotate_track(track_id, gt_track)
            records.append(record)
            if record["division"] is not None:
                daughters = self.annotator.lineage[track_id]
                # A ground-truth track with more than two children keeps only the first two.
                for gt_child, daughter in zip(sorted(self.children[gt_track]), daughters):
                    queue.append((gt_child, daughter))
        self.lineage_records.append({
            "lineage": {k: list(v) for k, v in self.annotator.lineage.items()}, "records": records,
            "log": self.annotator.log, "final": digest(self.annotator.current_object),
        })
        self.annotator.commit()

    def run(self, progress=None, lineages=None):
        """Annotate all lineages, or only the roots lineages[0] to lineages[1] (exclusive) of `roots`.

        The lineages are independent until they are committed, so a range of them can be annotated on its own and the
        records of all ranges merged in the order of `roots` (see `results_per_budget`).
        """
        roots = self.roots()
        if lineages is not None:
            roots = roots[lineages[0]:lineages[1]]
        for root in roots if progress is None else progress(roots):
            self.annotate_lineage(root)

    def results_per_budget(self, budgets):
        """Yield the committed result for each correction budget by a replay of the logged annotator changes.

        Each track is annotated to the end before the next one starts, so dropping the changes of the corrections
        beyond a budget gives what a user with that budget would get. For the full budget this is the annotator's own
        result, which is checked. The budgets are built one at a time to bound the memory.

        Yields:
            The budget, the committed objects, the committed lineages and the number of prompts.
        """
        shape = self.gt_masks.shape
        for budget in budgets:
            committed = np.zeros(shape, dtype="uint32")
            committed_lineages, n_prompts = [], 0
            for entry in self.lineage_records:
                n_prompts += sum(sum(record["n_prompts"][:budget + 1]) for record in entry["records"])
                current = TrackingAnnotator.replay(entry["log"], shape, budget)
                if budget >= self.n_corrections and digest(current) != entry["final"]:
                    raise RuntimeError("The replayed lineage differs from the annotator's result.")
                commit_objects(committed, committed_lineages, current, entry["lineage"])
            yield budget, committed, committed_lineages, n_prompts
