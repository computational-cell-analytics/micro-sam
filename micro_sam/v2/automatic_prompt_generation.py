"""Automatic prompt generation (APG) for UniSAM2: propose seeds generously, let the model judge them.

The flow post-processing thresholds the decoder's convergence density to get instances. APG proposes
candidates *below* that threshold, prompts the interactive branch with each one, and keeps the masks it
scores highly, so the model does the discrimination that thresholding cannot. Worth +0.085 mSA on livecell.

A volume works the same way, with the SAM2 video predictor in place of the image predictor: the flow
is integrated in 3d so a density component is a whole object, each is scored in 2d on the slice it
converges on, and the survivors are propagated through the volume. Worth +0.06 mSA over AIS on ten
volumetric LM datasets and +0.27 on four EM ones.

The parameter surface is deliberately small: everything here was measured to matter on at least one
dataset, and a good many other levers were tried and dropped. For volumes those were: re-prompting
every instance with the prompts the merge grouped onto it (+0.001), negative prompts from the
adjacent instances (+0.001), conditioning the anchor slice with a box or the 2d mask instead of the
point (+0.001 and -0.005), and sampling further positive prompts from an instance's own mask
(-0.034, because a prompt on a propagated slice turns it into a conditioning frame and replaces the
mask there with a single-point one). Selection is not what limits the result: an oracle that hands
every object its best-matching propagated mask scores 0.006 above the merge.

The first three of those are exactly the ingredients 2d found only pay in combination, and combining
them on the anchor slice - which the -0.034 above is also the reason they cannot be combined anywhere
else - is worth +2.3% mSA on 32-slice crops for +1.4% runtime, so the optional 'refinement' does that.
It is off by default in both dimensions, being short of the +5% the optimization gates ask for. One
thing measured there is worth knowing before touching this file: pushing the same prompt to the video
predictor in one call rather than several costs 1.75% mSA, because each push re-runs the mask decoder
on the conditioning frame. See finetuning/v2/evaluation/optimization/notes/APG_3D_OPTIMIZATION.md, experiment 6.
"""

import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm
from scipy.ndimage import find_objects, distance_transform_edt

import torch
import torch.nn.functional as F

from sam2.utils.amg import calculate_stability_score

from bioimage_cpp.utils import Blocking
from bioimage_cpp.segmentation import label

from .normalization import to_image
from .transforms.resize import resize_longest_side_and_pad_tensor
from .multimask_selection import (
    POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES, combine_selector_features_torch, extract_multimask_features_torch,
    refinement_gate_features_torch, refinement_gate_stage, selector_input_schema,
)
from ..util import make_temp_embedding_path
from .postprocessing import _compute_flow_density
from .prompt_based_segmentation import PromptableSegmentation3D, _crop_to_original_shape
from .util import (
    DEFAULT_MODEL, autocast, encode_image, precompute_image_embeddings, set_precomputed,
    get_sam2_image_predictor,
)
from .instance_segmentation import (
    TiledUniSAM2InstanceSegmentation, UniSAM2InstanceSegmentation, USE_MODEL_DEVICE, Devices,
    _set_image_predictor_from_backbone, _set_image_predictor_from_3d_embeddings,
)

# Only enters the merge order, never a cutoff, so it is a constant.
STABILITY_SCORE_OFFSET = 1.0

# Per (model_type, mode) defaults from the registry parameter search, same methodology as
# `micro_sam.v2.postprocessing.DEFAULT_POSTPROCESSING`. 2D and 3D are swept independently (a volume
# prompts once per object rather than once per slice), so they get separate tables.
DEFAULT_PROMPT_GENERATION_2D = {
    "hvit_t": {
        "foreground_threshold": 0.7, "candidate_threshold": 3.0, "n_iter": 50, "sigma": 0.5,
        "min_candidate_size": 4, "score_threshold": 0.6, "max_overlap": 0.3, "min_size": 50,
    },
    "hvit_s": {
        "foreground_threshold": 0.7, "candidate_threshold": 1.0, "n_iter": 50, "sigma": 2.0,
        "min_candidate_size": 4, "score_threshold": 0.5, "max_overlap": 0.15, "min_size": 50,
    },
    "hvit_b": {
        "foreground_threshold": 0.7, "candidate_threshold": 3.0, "n_iter": 50, "sigma": 0.5,
        "min_candidate_size": 4, "score_threshold": 0.5, "max_overlap": 0.15, "min_size": 50,
    },
    "hvit_l": {
        "foreground_threshold": 0.7, "candidate_threshold": 2.25, "n_iter": 50, "sigma": 1.0,
        "min_candidate_size": 4, "score_threshold": 0.6, "max_overlap": 0.3, "min_size": 50,
    },
}

DEFAULT_PROMPT_GENERATION_3D = {
    "hvit_t": {
        "candidate_threshold": (1.5, 10.0), "sigma": 1.0, "min_candidate_size": 1,
        "score_threshold": 0.6, "max_overlap": 0.15, "min_size": 100,
    },
    "hvit_s": {
        "candidate_threshold": (0.5, 5.0), "sigma": 1.0, "min_candidate_size": 1,
        "score_threshold": 0.6, "max_overlap": 0.5, "min_size": 100,
    },
    "hvit_b": {
        "candidate_threshold": (1.0, 5.0), "sigma": 1.0, "min_candidate_size": 4,
        "score_threshold": 0.6, "max_overlap": 0.5, "min_size": 100,
    },
    "hvit_l": {
        "candidate_threshold": (1.5, 10.0), "sigma": 0.25, "min_candidate_size": 1,
        "score_threshold": 0.6, "max_overlap": 0.5, "min_size": 100,
    },
}


def default_prompt_generation(model_type: str = DEFAULT_MODEL, is_volume: bool = False) -> dict:
    """The default APG parameters for one model type and dimensionality.

    Args:
        model_type: The SAM2 backbone, e.g. 'hvit_t', or a finetuned model built on one (only the
            backbone prefix is used to look up the table). Must be one of the 4 registry backbones.
        is_volume: Whether to use the 3D table (`derive_volume_prompts`) or the 2D one
            (`derive_point_prompts`). A volume prompts once per object rather than once per slice, so
            the two are swept and tuned independently.

    Returns:
        The default parameter dict for that model type and dimensionality.
    """
    table = DEFAULT_PROMPT_GENERATION_3D if is_volume else DEFAULT_PROMPT_GENERATION_2D
    backbone = model_type[:6]
    if backbone not in table:
        raise ValueError(
            f"No default prompt generation parameters for model type '{model_type}'. "
            f"Choose one built on a backbone in {sorted(table)}."
        )
    return table[backbone]


DEFAULT_PROMPT_GENERATION = {
    # Which record score the 2D pre-merge eligibility threshold applies to. Learned scoring is an
    # explicit opt-in; the historical predicted-IoU filter remains the default.
    "score_filter": "predicted_iou",
    "multimasking": True,
    # Images only. The default is the exact historical predicted-IoU argmax. A microscopy scorer is
    # installed explicitly with `set_multimask_models` before selecting it here.
    "multimask_scorer": "predicted_iou",
    # Images only. Deferred selection retains the alternatives until the score-ordered merge and
    # accepts at most one alternative from each prompt.
    "multimask_selection": "eager",
    # Off by default: no refinement mode has passed the optimization gates yet, see 'REFINEMENT_COMPONENTS'.
    "refinement": None,
    "refinement_kwargs": None,
    # Volumes only. Trades device memory against the pass count.
    "n_objects_per_pass": 16,
    # Volumes only. None propagates through the whole volume, which costs one frame step per slice
    # even once every object of a pass has ended. Measured on 32-slice crops of the five 3d
    # benchmark datasets, stopping after two empty slices left the segmentation bit-identical on all
    # of them while skipping 34% of GoNuclear's frame steps and 12% of C. elegans's, so it is on by
    # default; the annotator has used this value since the volume widget was written. Raise it where
    # objects are expected to disappear and reappear, since SAM2 can drop a mask and recover it.
    "early_stop_patience": 2,
    # Number of image prompts (or refinement boxes) evaluated per forward pass.
    "batch_size": 64,
    # These are constant across all registry backbones and dimensionalities.
    "foreground_threshold": 0.7,
    "n_iter": 50,
    # Shared with the sparse post-processing's 'dt', tuned there for one peak per object rather than
    # for recall, but 0.5 for every registry backbone either way, so it stays a flat constant here.
    "dt": 0.5,
    # Throughput only, the density is the same either way.
    "n_threads": 8,
}

# The components a refinement mode can be assembled from, and the keyword arguments each accepts.
# A mode is a '+'-joined combination, e.g. 'points', 'boxes' or 'points+boxes': every component
# contributes its prompt to one joint re-prompt per instance, so 'points+boxes' conditions on both.
REFINEMENT_COMPONENTS = ("points", "boxes", "masks")
REFINEMENT_KWARGS = {
    "shared": (
        "policy", "multimasking", "min_consistency", "max_foreign_overlap", "gate", "gate_threshold",
    ),
    "points": (
        "n_positives", "n_negatives", "max_negative_distance", "negative_source",
        "min_negative_distance",
    ),
    "boxes": ("box_extension",),
    "masks": (),
}
DEFAULT_REFINEMENT = {
    # The defaults are the measured optimum of the recommended mode, 'points+boxes': +4.2% macro mSA
    # on the tuned subset and +4.9% on the held-out one, for about +35-50% runtime. See
    # finetuning/v2/evaluation/optimization/notes/APG_2D_OPTIMIZATION.md; the pipeline default stays
    # 'refinement=None'.
    # 'replace' repaints every instance from its second-round mask; 'keep-if-better' keeps the
    # first-round mask unless the second round scores higher — which for a box-anchored re-prompt
    # it nearly always does, so the geometric gates below are what actually arbitrates.
    "policy": "replace",
    # Off by default: a multi-prompt re-prompt is not ambiguous the way a single point is.
    "multimasking": False,
    # Accept a second-round mask only if its IoU with the first-round mask reaches this: the
    # re-prompt may polish the boundary but not reshape the instance. 0.85 over-gates, 0.5 barely
    # fires; None accepts any overlap.
    "min_consistency": 0.7,
    # Keep the first-round mask when more than this fraction of the second-round mask lies on
    # *other* first-round instances, which is a re-prompt growing into a neighbour. None allows any.
    "max_foreign_overlap": 0.15,
    # `all` preserves the established opt-in refinement. `uncertainty` evaluates the installed
    # refinement gate and only re-prompts records whose predicted utility reaches the threshold.
    "gate": "all",
    "gate_threshold": 0.0,
    # The surviving prompt only: grouped extra positives measurably hurt (p1 > p2 > p3 on both
    # subsets). The suppressed prompts' productive role is as the neighbours' negative pool.
    "n_positives": 1,
    # Negative points per instance, taken from the prompts of the nearest other instances. The
    # response peaks at 6-8 and collapses beyond 12; 8 trades LiveCELL for DynamicNuclearNet.
    "n_negatives": 6,
    # Only prompts within this distance (in pixels) of an instance's bounding box can be its
    # negatives. None takes the nearest ones regardless of distance.
    "max_negative_distance": None,
    # Where an instance's negatives come from: the other instances' first-round 'prompts', or the
    # deepest 'interior' point of each other instance's mask, which sits away from shared borders.
    "negative_source": "prompts",
    # Exclude negatives closer than this (in pixels) to the instance's own first-round mask: a
    # negative touching the instance's true extent cuts into the object instead of bounding it.
    "min_negative_distance": 0,
    # Number of pixels every box prompt is grown by. Confluent data prefers 0, because a grown box
    # reaches into the neighbouring object.
    "box_extension": 0,
}

# A volume refines its anchor slices, before the propagation. A prompt on an already propagated slice
# would turn it into a conditioning frame and replace the mask there with a single-point one (-0.034
# mSA, see the module docstring), so a finished track cannot be touched; what the second round
# produces is the conditioning the propagation starts from rather than a finished mask. The
# components mean the same as in 2d. The learned uncertainty gate is image-only; 'conditioning' is
# the one volume addition, see `DEFAULT_REFINEMENT_3D`.
# How an accepted re-prompt is pushed onto the anchor frame, see 'DEFAULT_REFINEMENT_3D'.
CONDITIONING_MODES = ("prompts", "prompts-grouped", "prompts-joint", "mask")
REFINEMENT_KWARGS_3D = {
    "shared": tuple(
        key for key in REFINEMENT_KWARGS["shared"] if key not in ("gate", "gate_threshold")
    ) + ("conditioning",),
    "points": REFINEMENT_KWARGS["points"],
    "boxes": REFINEMENT_KWARGS["boxes"],
    "masks": REFINEMENT_KWARGS["masks"],
}
DEFAULT_REFINEMENT_3D = {
    key: value for key, value in DEFAULT_REFINEMENT.items() if key not in ("gate", "gate_threshold")
}
# The counters a volume's refinement reports, all of them accumulated over the anchor slices. Zeroed
# together when a refinement runs, so a mode that cannot produce one still reports it as 0 rather
# than leaving the column absent for that run only.
REFINEMENT_STATS_3D = (
    "refined_candidates", "replaced_candidates", "gated_consistency", "gated_foreign",
    "refinement_negatives",
)
DEFAULT_REFINEMENT_3D.update({
    # Measured on the 3d benchmark, and different from 2d on both axes. Negatives peak at four rather
    # than six to eight: every one of a volume's negatives comes from a candidate anchored on the same
    # slice, so each is a real in-plane neighbour and four already bound the object. Zero falls below
    # the baseline and twelve back to +0.3%.
    "n_negatives": 4,
    # Tighter than 2d's 0.7, where 0.85 over-gated. In 2d a wrongly accepted re-prompt costs one
    # instance's mask; here it becomes the conditioning frame of a whole track, so refusing more of
    # them pays. 0.95 over-gates in 3d too, vetoing 374 of 925 second rounds. With this gate tight,
    # 'max_foreign_overlap' never fires, so it is kept at the 2d value only because it costs nothing.
    "min_consistency": 0.85,
    # How an accepted second round reaches the propagation, which is not the formality it looks
    # like: every push of a prompt re-runs SAM2's mask decoder on the anchor frame and feeds the
    # previous prediction back in, so pushing the same prompt in more steps refines the anchor
    # iteratively rather than repeating work.
    #   'prompts'         the box, then one push per point. The most steps, and the best measured
    #                     quality by a wide margin: +1.87% macro against +0.11% for a single push.
    #   'prompts-grouped' the box, then one push carrying every point. Two steps.
    #   'prompts-joint'   one push carrying the box and every point. One step, so the propagation
    #                     starts from a single forward and the iterative refinement is gone.
    #   'mask'            the refined mask itself, via 'add_new_mask'. The anchor frame is then
    #                     exactly what the gates accepted, but the decoder never sees the prompt.
    #                     Conditioning an anchor with an *unrefined* 2d mask measured -0.005 mSA.
    "conditioning": "prompts",
})


def _parse_refinement(
    refinement: str, refinement_kwargs: Optional[Dict[str, Any]], is_volume: bool = False,
) -> tuple:
    """Parse a refinement mode into its components and resolve its keyword arguments.

    Args:
        refinement: The mode, a '+'-joined combination of `REFINEMENT_COMPONENTS`.
        refinement_kwargs: The mode's keyword arguments. Only keys that one of the mode's
            components (or every mode) accepts are allowed.
        is_volume: Whether the mode is resolved for a volume, which accepts and defaults its
            keyword arguments differently, see `REFINEMENT_KWARGS_3D` and `DEFAULT_REFINEMENT_3D`.

    Returns:
        The components as a tuple, and the resolved keyword arguments with the defaults filled in.
    """
    accepted = REFINEMENT_KWARGS_3D if is_volume else REFINEMENT_KWARGS
    defaults = DEFAULT_REFINEMENT_3D if is_volume else DEFAULT_REFINEMENT
    components = tuple(refinement.split("+"))
    unknown = [component for component in components if component not in REFINEMENT_COMPONENTS]
    if unknown or len(set(components)) != len(components):
        raise ValueError(
            f"Invalid refinement mode {refinement!r}: expected a '+'-joined combination of "
            f"{', '.join(REFINEMENT_COMPONENTS)} without repetition."
        )
    if components == ("masks",):
        raise ValueError(
            "A mask prompt can only condition a re-prompt, not drive one alone: SAM2 is not trained "
            "for dense-only prompting. Combine it, e.g. 'points+masks' or 'boxes+masks'."
        )

    allowed = set(accepted["shared"])
    for component in components:
        allowed.update(accepted[component])
    refinement_kwargs = refinement_kwargs or {}
    unknown = sorted(set(refinement_kwargs) - allowed)
    if unknown:
        raise ValueError(
            f"Invalid refinement_kwargs for mode {refinement!r}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )

    resolved = {key: defaults[key] for key in allowed}
    resolved.update(refinement_kwargs)
    if resolved["policy"] not in ("replace", "keep-if-better"):
        raise ValueError(f"Invalid refinement policy {resolved['policy']!r}: expected 'replace' or 'keep-if-better'.")
    if resolved.get("gate", "all") not in ("all", "uncertainty"):
        raise ValueError(
            f"Invalid refinement gate {resolved['gate']!r}: expected 'all' or 'uncertainty'."
        )
    if not np.isfinite(resolved.get("gate_threshold", 0.0)):
        raise ValueError("The refinement gate threshold must be finite.")
    if resolved.get("negative_source", "prompts") not in ("prompts", "interior"):
        raise ValueError(
            f"Invalid negative_source {resolved['negative_source']!r}: expected 'prompts' or 'interior'."
        )
    if resolved.get("conditioning", "prompts") not in CONDITIONING_MODES:
        raise ValueError(
            f"Invalid conditioning {resolved['conditioning']!r}: expected one of "
            f"{', '.join(CONDITIONING_MODES)}."
        )
    return components, resolved


def mask_to_logits(mask: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Turn a binary mask into the low-resolution logit prompt SAM2 expects.

    The image predictor preserves the aspect ratio and pads the bottom or right side. The mask uses
    the same frame, so it stays aligned with the image features and other prompts.

    Args:
        mask: The binary mask, shape (Y, X).
        eps: Probability assigned to the background, from which the logit magnitude follows.

    Returns:
        The logits, shape (1, 256, 256), float32.
    """
    binary = torch.from_numpy(np.asarray(mask, dtype="float32"))[None, None]
    resized, _ = resize_longest_side_and_pad_tensor(binary, target_length=256)
    logit = float(np.log((1.0 - eps) / eps))
    return np.where(resized[0].numpy() > 0.5, logit, -logit).astype("float32")


def _assign_points_to_instances(segmentation: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The instance id under every point, 0 where a point lies on the background.

    Args:
        segmentation: The instance segmentation.
        points: The points as (N, 2) in XY, following the prompt convention.

    Returns:
        The instance ids as (N,), int64.
    """
    if len(points) == 0:
        return np.zeros(0, dtype="int64")
    coordinates = np.round(points[:, ::-1]).astype("int64")  # XY to YX.
    coordinates = np.clip(coordinates, 0, np.array(segmentation.shape) - 1)
    return segmentation[tuple(coordinates.T)].astype("int64")


def _subsample_positives(anchor: np.ndarray, candidates: np.ndarray, n_positives: int) -> np.ndarray:
    """The anchor plus up to 'n_positives - 1' candidates, chosen greedily for spatial coverage.

    Farthest-point subsampling: each step adds the candidate farthest from everything selected so
    far, so the kept positives spread over the instance instead of clustering. Deterministic, with
    ties broken by candidate order.

    Args:
        anchor: The point that is always kept, shape (2,).
        candidates: The candidate points, shape (M, 2).
        n_positives: The total number of positives to keep, including the anchor.

    Returns:
        The selected points, shape (min(n_positives, M + 1), 2).
    """
    selected = [np.asarray(anchor, dtype="float32")]
    remaining = [np.asarray(candidate, dtype="float32") for candidate in candidates]
    while remaining and len(selected) < n_positives:
        distances = [min(float(np.linalg.norm(candidate - point)) for point in selected) for candidate in remaining]
        selected.append(remaining.pop(int(np.argmax(distances))))
    return np.stack(selected)


def _select_negatives(
    candidates: np.ndarray, bounding_box: tuple, n_negatives: int, max_negative_distance: Optional[float],
) -> np.ndarray:
    """The candidates nearest to the instance, which are the ones the model could confuse with it.

    Nearness is the distance to the instance's bounding box, which is zero inside it; ties are
    broken by the distance to the box's center and then by candidate order, so the selection is
    deterministic.

    Args:
        candidates: The candidate points as (M, 2) in XY.
        bounding_box: The instance's bounding box as a (y_slice, x_slice) tuple.
        n_negatives: Number of negatives to select.
        max_negative_distance: Discard candidates farther than this from the bounding box.

    Returns:
        The selected points, shape (min(n_negatives, M'), 2).
    """
    if len(candidates) == 0 or n_negatives <= 0:
        return np.zeros((0, 2), dtype="float32")
    y_slice, x_slice = bounding_box
    low = np.array([x_slice.start, y_slice.start], dtype="float32")
    high = np.array([x_slice.stop - 1, y_slice.stop - 1], dtype="float32")

    outside = np.maximum(np.maximum(low - candidates, candidates - high), 0.0)
    distances = np.linalg.norm(outside, axis=1)
    center_distances = np.linalg.norm(candidates - (low + high) / 2.0, axis=1)
    order = np.lexsort((np.arange(len(candidates)), center_distances, distances))
    if max_negative_distance is not None:
        order = order[distances[order] <= max_negative_distance]
    return candidates[order[:n_negatives]].astype("float32")


def _distances_to_mask(
    segmentation: np.ndarray, instance_id: int, bounding_box: tuple, candidates: np.ndarray, reach: float,
) -> np.ndarray:
    """The Euclidean distance of every candidate point to the instance's mask, exact up to 'reach'.

    The transform runs on the instance's bounding box grown by 'reach', so a candidate outside that
    crop is farther than 'reach' and reported as infinity, which the caller's threshold treats the
    same way.

    Args:
        segmentation: The instance segmentation.
        instance_id: The instance whose mask is measured to.
        bounding_box: The instance's bounding box, as a (y_slice, x_slice) tuple.
        candidates: The candidate points as (M, 2) in XY.
        reach: The distance up to which the measurement has to be exact.

    Returns:
        The distances as (M,), float32, with infinity beyond 'reach'.
    """
    margin = int(np.ceil(reach)) + 1
    y_slice, x_slice = bounding_box
    y0, y1 = max(0, y_slice.start - margin), min(segmentation.shape[0], y_slice.stop + margin)
    x0, x1 = max(0, x_slice.start - margin), min(segmentation.shape[1], x_slice.stop + margin)
    distances_in_crop = distance_transform_edt(segmentation[y0:y1, x0:x1] != instance_id)

    distances = np.full(len(candidates), np.inf, dtype="float32")
    for position, (x, y) in enumerate(np.round(candidates).astype("int64")):
        if y0 <= y < y1 and x0 <= x < x1:
            distances[position] = distances_in_crop[y - y0, x - x0]
    return distances


def derive_refinement_prompts(
    segmentation: np.ndarray,
    points: np.ndarray,
    surviving_points: Dict[int, tuple],
    n_positives: int = DEFAULT_REFINEMENT["n_positives"],
    n_negatives: int = DEFAULT_REFINEMENT["n_negatives"],
    max_negative_distance: Optional[float] = DEFAULT_REFINEMENT["max_negative_distance"],
    negative_source: str = DEFAULT_REFINEMENT["negative_source"],
    min_negative_distance: float = DEFAULT_REFINEMENT["min_negative_distance"],
) -> Dict[int, Dict[str, np.ndarray]]:
    """Group the first round's prompts onto the instances they landed in and derive re-prompts.

    An instance's positives are the prompt that made it plus the suppressed prompts inside it: each
    marks a spot the decoder proposed an object at, so together they cover the instance in a way a
    single point cannot. Its negatives are nearby points belonging to other instances, which tell
    the model where the instance ends.

    Args:
        segmentation: The instance segmentation of the first round.
        points: All first-round prompt points as (N, 2) in XY, including the suppressed ones.
        surviving_points: The prompt that made each instance, as {instance_id: (x, y)}.
        n_positives: Number of positive points per instance, including the surviving prompt. The
            surplus is subsampled farthest-point-first, for spatial coverage.
        n_negatives: Number of negative points per instance. The surplus is dropped farthest-first,
            measured to the instance's bounding box.
        max_negative_distance: Only points within this distance of an instance's bounding box can
            be its negatives. None takes the nearest ones regardless of distance.
        negative_source: Where the negatives come from: the other instances' first-round 'prompts',
            or the deepest 'interior' point of each other instance's mask, which sits away from any
            border the two instances share.
        min_negative_distance: Exclude negatives closer than this to the instance's own mask. A
            negative that touches the instance's true extent cuts into the object instead of
            bounding it, which is the suspected failure on densely packed data.

    Returns:
        The prompts per instance, as {instance_id: {'points': (M, 2) XY, 'point_labels': (M,),
        'n_grouped': int}}, where 'n_grouped' counts the suppressed prompts grouped onto the
        instance before any subsampling — the supply signal an adaptive re-prompt keys on.
    """
    points = np.asarray(points, dtype="float32").reshape(-1, 2)
    assignment = _assign_points_to_instances(segmentation, points)

    if negative_source == "interior":
        # One deep interior point per instance, ordered by ascending instance id; converted to XY.
        interior = interior_points(segmentation)[:, ::-1].astype("float32")
        present = np.array(sorted(surviving_points), dtype="int64")
        negative_points, negative_owners = interior, present
    elif negative_source == "prompts":
        foreign = assignment > 0
        negative_points, negative_owners = points[foreign], assignment[foreign]
    else:
        raise ValueError(f"Invalid negative_source {negative_source!r}: expected 'prompts' or 'interior'.")

    prompts = {}
    for index, bounding_box in enumerate(find_objects(segmentation), start=1):
        if bounding_box is None:
            continue
        anchor = np.asarray(surviving_points[index], dtype="float32")
        grouped = points[assignment == index]
        # The anchor is one of the grouped points; keep it once, as the seed of the subsampling.
        grouped = grouped[np.linalg.norm(grouped - anchor, axis=1) > 0]
        positives = _subsample_positives(anchor, grouped, n_positives)

        candidates = negative_points[negative_owners != index]
        if min_negative_distance > 0 and len(candidates) and n_negatives > 0:
            distances = _distances_to_mask(segmentation, index, bounding_box, candidates, min_negative_distance)
            candidates = candidates[distances >= min_negative_distance]
        negatives = _select_negatives(candidates, bounding_box, n_negatives, max_negative_distance)

        prompts[index] = {
            "points": np.concatenate([positives, negatives]).astype("float32"),
            "point_labels": np.concatenate([
                np.ones(len(positives), dtype="int32"), np.zeros(len(negatives), dtype="int32"),
            ]),
            "n_grouped": int(len(grouped)),
        }
    return prompts


def postmerge_refinement_gate_features(
    segmentation: np.ndarray,
    context: Dict[str, Any],
    point_prompts: Optional[Dict[int, Dict[str, np.ndarray]]],
    foreground: np.ndarray,
    foreground_threshold: float,
) -> tuple:
    """Describe accepted first-pass instances after merging and prompt assembly.

    These features deliberately use the exact visible mask and the exact positive/negative points
    that a ``points+boxes`` refinement would consume. Unlike the historical gate, they therefore
    capture truncation, neighborhood and negative-prompt evidence that does not exist until after
    the first-pass merge. Rows are returned in ascending instance-id order.
    """
    segmentation = np.asarray(segmentation)
    foreground = np.asarray(foreground, dtype="float32")
    if foreground.shape != segmentation.shape:
        raise ValueError(
            f"Expected foreground shape {segmentation.shape}, got {foreground.shape}."
        )
    boxes = find_objects(segmentation)
    instance_ids = np.asarray(sorted(context["matches"]), dtype="int64")
    if not len(instance_ids):
        return np.empty((0, len(POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES)), dtype="float32"), instance_ids

    points = np.stack([
        np.asarray(context["records"][context["matches"][int(instance_id)]]["point"], dtype="float32")
        for instance_id in instance_ids
    ])
    if len(points) > 1:
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest_instance = distances.min(axis=1)
    else:
        nearest_instance = np.full(1, float(max(segmentation.shape)), dtype="float32")

    rows = []
    for row_index, instance_id_value in enumerate(instance_ids):
        instance_id = int(instance_id_value)
        record_index = context["matches"][instance_id]
        record = context["records"][record_index]
        bounding_box = boxes[instance_id - 1]
        if bounding_box is None:
            raise RuntimeError(f"Merged instance {instance_id} has no bounding box.")
        visible = segmentation[bounding_box] == instance_id
        visible_area = float(visible.sum())
        source_area = float(np.asarray(record["segmentation"], dtype=bool).sum())
        height, width = visible.shape
        box_area = float(height * width)
        foreground_crop = foreground[bounding_box]
        denominator = max(visible_area, 1.0)
        foreground_mean = float(foreground_crop[visible].sum() / denominator)
        foreground_precision = float(
            np.count_nonzero(visible & (foreground_crop > foreground_threshold)) / denominator
        )
        border_contacts = sum((
            bounding_box[0].start == 0,
            bounding_box[0].stop == segmentation.shape[0],
            bounding_box[1].start == 0,
            bounding_box[1].stop == segmentation.shape[1],
        ))

        prompt = None if point_prompts is None else point_prompts.get(instance_id)
        if prompt is None:
            positive_count, negative_count, grouped_count = 1, 0, 0
            negative_distances = np.empty(0, dtype="float32")
        else:
            prompt_labels = np.asarray(prompt["point_labels"])
            prompt_points = np.asarray(prompt["points"], dtype="float32")
            positive_count = int(np.count_nonzero(prompt_labels == 1))
            negative_count = int(np.count_nonzero(prompt_labels == 0))
            grouped_count = int(prompt.get("n_grouped", 0))
            negative_distances = np.linalg.norm(
                prompt_points[prompt_labels == 0] - np.asarray(record["point"], dtype="float32"), axis=1,
            )
        distance_default = float(max(segmentation.shape))
        nearest_negative = (
            float(negative_distances.min()) if len(negative_distances) else distance_default
        )
        mean_negative = (
            float(negative_distances.mean()) if len(negative_distances) else distance_default
        )
        predicted_iou = float(record["predicted_iou"])
        stability = float(record["stability_score"])
        selection_score = float(record.get("selection_score", predicted_iou))
        merge_score = float(record.get("merge_score", predicted_iou * stability))
        score_filter = context.get("score_filter", "predicted_iou")
        score_filter_margin = (
            0.0 if score_filter == "none"
            else float(record.get(score_filter, predicted_iou)) - float(context["score_threshold"])
        )
        claimed_fraction = float(np.clip(
            1.0 - visible_area / max(source_area, 1.0), 0.0, 1.0,
        ))
        rows.append((
            predicted_iou,
            stability,
            predicted_iou * stability,
            selection_score,
            selection_score - predicted_iou,
            merge_score,
            score_filter_margin,
            float(record.get("multimask_index", 0)),
            float(np.log1p(source_area)),
            float(np.log1p(visible_area)),
            visible_area / max(source_area, 1.0),
            float(np.log1p(box_area)),
            visible_area / max(box_area, 1.0),
            float(np.log(max(width, 1) / max(height, 1))),
            float(border_contacts / 4.0),
            foreground_mean,
            foreground_precision,
            claimed_fraction,
            float(np.log1p(len(instance_ids))),
            float(np.log1p(nearest_instance[row_index])),
            float(grouped_count),
            float(positive_count),
            float(negative_count),
            float(np.log1p(nearest_negative)),
            float(np.log1p(mean_negative)),
        ))
    features = np.asarray(rows, dtype="float32")
    if features.shape[1] != len(POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES):
        raise RuntimeError("Post-merge refinement features do not match their declared schema.")
    if not np.isfinite(features).all():
        raise RuntimeError("Post-merge refinement features contain a non-finite value.")
    return features, instance_ids


def _shift_box(bounding_box: tuple, offset: tuple) -> tuple:
    """Translate a bounding box by a per-axis offset, returning it unchanged for a zero offset."""
    if not any(offset):
        return bounding_box
    return tuple(slice(box.start + shift, box.stop + shift) for box, shift in zip(bounding_box, offset))


def _localize_prompts(prompt: Dict[str, Any], origin: tuple, extent: tuple) -> tuple:
    """Translate one instance's re-prompt into a region's frame, dropping the negatives outside it.

    The instance's own mask lies inside the region it is re-prompted in, so its positives are only
    clamped onto the border pixel they round to. A negative comes from a neighbouring instance and
    can lie outside the region — beyond a tile's halo — and has to go, because the predictor
    normalises the prompt against the region's own shape.

    Args:
        prompt: The instance's prompt, as `derive_refinement_prompts` returns it.
        origin: The region's (y, x) origin in the full image.
        extent: The region's (y, x) shape.

    Returns:
        The translated prompt, and the number of negatives dropped.
    """
    origin_xy = np.array([origin[1], origin[0]], dtype="float32")
    high_xy = np.array([extent[1] - 1, extent[0] - 1], dtype="float32")
    points = np.asarray(prompt["points"], dtype="float32") - origin_xy
    labels = np.asarray(prompt["point_labels"])

    # A point that rounds onto one of the region's pixels is inside it. Positives are never dropped.
    inside = np.all((points >= -0.5) & (points <= high_xy + 0.5), axis=1)
    keep = inside | (labels == 1)
    localized = {
        "points": np.clip(points[keep], 0.0, high_xy),
        "point_labels": labels[keep],
        "n_grouped": prompt["n_grouped"],
    }
    return localized, int(np.count_nonzero(~keep))


def _prompt_box(bounding_box: tuple, shape: tuple, box_extension: int) -> tuple:
    """An instance's bounding box as the XYXY box prompt SAM2 takes, grown and clipped to 'shape'.

    Args:
        bounding_box: The instance's bounding box, as a (y_slice, x_slice) tuple.
        shape: The (Y, X) shape the grown box is clipped to.
        box_extension: Number of pixels the box is grown by on every side.

    Returns:
        The box as (x0, y0, x1, y1).
    """
    y_slice, x_slice = bounding_box
    return (
        max(0, x_slice.start - box_extension), max(0, y_slice.start - box_extension),
        min(shape[1], x_slice.stop + box_extension), min(shape[0], y_slice.stop + box_extension),
    )


def _predict_three_lowres(predictor, coords, labels, boxes, mask_input):
    """Run SAM2's ordinary three-mask branch without postprocessing all three masks."""
    concat_points = None if coords is None else (coords, labels)
    if boxes is not None:
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device).repeat(boxes.size(0), 1)
        concat_points = (
            box_coords, box_labels,
        ) if concat_points is None else (
            torch.cat((box_coords, concat_points[0]), dim=1),
            torch.cat((box_labels, concat_points[1]), dim=1),
        )
    sparse, dense = predictor.model.sam_prompt_encoder(
        points=concat_points, boxes=None, masks=mask_input,
    )
    batched = concat_points is not None and concat_points[0].shape[0] > 1
    image_index = -1
    high_res = [
        feature[image_index].unsqueeze(0) for feature in predictor._features["high_res_feats"]
    ]
    lowres, scores, mask_tokens, _ = predictor.model.sam_mask_decoder.predict_masks(
        image_embeddings=predictor._features["image_embed"][image_index].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        repeat_image=batched,
        high_res_features=high_res,
    )
    # SAM2's public multimask branch is exactly tokens/masks 1:4. Token 0 remains outside this path.
    lowres, scores, mask_tokens = lowres[:, 1:], scores[:, 1:], mask_tokens[:, 1:]
    if lowres.shape[1] != 3 or mask_tokens.shape[1] != 3:
        raise RuntimeError(
            f"Expected SAM2's three multimask outputs, got {lowres.shape[1]} masks and "
            f"{mask_tokens.shape[1]} tokens."
        )
    return lowres.clamp(-32.0, 32.0), scores, mask_tokens


def _lowres_feature_context(predictor, foreground, context_points, lowres_shape, device):
    """Map APG foreground and prompt coordinates into SAM2's square low-resolution frame."""
    resolution = int(predictor.model.image_size)
    foreground = torch.as_tensor(foreground, dtype=torch.float32, device=device)[None, None]
    foreground, _ = resize_longest_side_and_pad_tensor(foreground, target_length=resolution)
    foreground = F.interpolate(
        foreground, size=lowres_shape, mode="bilinear", align_corners=False, antialias=True,
    )[0, 0]
    original_size = tuple(int(value) for value in predictor._orig_hw[-1])
    points = predictor._transforms.transform_coords(
        torch.as_tensor(context_points, dtype=torch.float32, device=device),
        normalize=True, orig_hw=original_size,
    )
    scale = torch.tensor(
        [lowres_shape[1] / resolution, lowres_shape[0] / resolution],
        dtype=torch.float32, device=device,
    )
    return foreground, points * scale


def interior_points(labels: np.ndarray) -> np.ndarray:
    """The deepest interior point of every labelled component, in ascending label order.

    The v1 counterpart transforms a `find_boundaries` image, whose 'outer' mode marks a thin
    component's own pixels as boundary. Its maximum then falls on the bounding box's first pixel,
    which need not lie in the component at all.

    Args:
        labels: A labelled image or volume.

    Returns:
        The points as (N, ndim), one per component, in the array's own axis order.
    """
    points = []
    for index, bounding_box in enumerate(find_objects(labels), start=1):
        if bounding_box is None:
            continue
        # Padded, so a component reaching the border is measured from that border too.
        distances = distance_transform_edt(np.pad(labels[bounding_box] == index, 1))
        coordinates = np.unravel_index(int(np.argmax(distances)), distances.shape)
        points.append(tuple(int(c) + box.start - 1 for c, box in zip(coordinates, bounding_box)))
    return np.array(points, dtype="int64").reshape(-1, labels.ndim)


def derive_point_prompts(
    foreground: np.ndarray,
    directed_distances: np.ndarray,
    model_type: str = DEFAULT_MODEL,
    candidate_threshold: Optional[float] = None,
    foreground_threshold: Optional[float] = None,
    n_iter: Optional[int] = None,
    dt: Optional[float] = None,
    sigma: Optional[float] = None,
    min_candidate_size: Optional[int] = None,
    n_threads: int = DEFAULT_PROMPT_GENERATION["n_threads"],
) -> Optional[Dict[str, np.ndarray]]:
    """Derive one positive point prompt per convergence-density component.

    The v1 counterpart intersects thresholded centre and boundary distances; UniSAM2 predicts directed
    distances instead, so the candidates come from the flow-convergence density that the sparse
    post-processing also seeds from, thresholded lower.

    Args:
        foreground: Foreground probability map, shape (Y, X).
        directed_distances: Distance channels stacked along axis 0. A leading z-channel is dropped, so
            `prediction[1:]` can be passed regardless of dimensionality.
        model_type: The SAM2 backbone the predictions came from, e.g. 'hvit_t'. Selects the default
            for any of the tunable parameters below left as None, see `default_prompt_generation`.
        candidate_threshold: Density threshold for proposing candidates. Lower proposes more. The density
            of a component scales with the object's area, so this is coupled to object size.
        foreground_threshold: Foreground binarisation threshold, which bounds the pixels that can be
            proposed from.
        n_iter: Number of flow-integration steps. Together with 'dt' this is the distance a pixel is
            advected, which has to be enough to reach the object's centre.
        dt: Integration step size. Mostly only the product with 'n_iter' matters.
        sigma: Gaussian sigma for smoothing the convergence-density map.
        min_candidate_size: Discard components smaller than this, which are noise rather than objects.
        n_threads: Number of threads for the flow computation.

    Returns:
        The prompts as {'points': (N, 1, 2) in XY, 'point_labels': (N, 1)}, or None if none were found.
    """
    defaults = default_prompt_generation(model_type, is_volume=False)
    if candidate_threshold is None:
        candidate_threshold = defaults["candidate_threshold"]
    if foreground_threshold is None:
        foreground_threshold = defaults["foreground_threshold"]
    if n_iter is None:
        n_iter = defaults["n_iter"]
    if dt is None:
        dt = DEFAULT_PROMPT_GENERATION["dt"]
    if sigma is None:
        sigma = defaults["sigma"]
    if min_candidate_size is None:
        min_candidate_size = defaults["min_candidate_size"]

    if directed_distances.shape[0] > foreground.ndim:
        directed_distances = directed_distances[-foreground.ndim:]

    fg_mask = foreground > foreground_threshold
    density = _compute_flow_density(
        directed_distances, fg_mask, n_iter=int(n_iter), dt=dt, sigma=sigma, n_threads=n_threads,
    )
    candidates = label(density > candidate_threshold)

    if min_candidate_size > 0:
        ids, sizes = np.unique(candidates, return_counts=True)
        discard = ids[(sizes < min_candidate_size) & (ids > 0)]
        if discard.size:
            candidates[np.isin(candidates, discard)] = 0
    if candidates.max() == 0:
        return None

    # The interior point rather than the centroid, which can lie outside a curved object.
    centers = interior_points(candidates)
    if len(centers) == 0:
        return None

    return {
        "points": np.ascontiguousarray(centers[:, ::-1], dtype="float32")[:, None, :],  # SAM2 wants XY
        "point_labels": np.ones((len(centers), 1), dtype="int32"),
    }


def derive_volume_prompts(
    foreground: np.ndarray,
    directed_distances: np.ndarray,
    model_type: str = DEFAULT_MODEL,
    candidate_threshold: Optional[Union[float, Sequence[float]]] = None,
    foreground_threshold: Optional[float] = None,
    n_iter: Optional[int] = None,
    dt: Optional[float] = None,
    sigma: Optional[float] = None,
    spacing: Optional[tuple] = None,
    min_candidate_size: Optional[int] = None,
    n_threads: int = DEFAULT_PROMPT_GENERATION["n_threads"],
) -> Optional[Dict[str, np.ndarray]]:
    """Derive one positive point prompt, on one slice, per volumetric convergence-density component.

    The volumetric counterpart of `derive_point_prompts`. The flow is integrated in 3d, so a component
    of the resulting density is a whole object rather than one of its cross-sections, and each object
    is prompted once instead of once per slice. The prompt is placed on the slice where the component
    converges most, which is the slice the video predictor propagates away from in both directions.

    Args:
        foreground: Foreground probability map, shape (Z, Y, X).
        directed_distances: Distance channels stacked along axis 0, shape (3, Z, Y, X).
        model_type: The SAM2 backbone the predictions came from, e.g. 'hvit_t'. Selects the default
            for any of the tunable parameters below left as None, see `default_prompt_generation`.
        candidate_threshold: Density threshold for proposing candidates, or several of them. Lower
            proposes more, but also merges the peaks of touching objects into one component, so a
            ladder recovers what a single threshold cannot separate.
        foreground_threshold: Foreground binarisation threshold, which bounds the voxels that can be
            proposed from.
        n_iter: Number of flow-integration steps. Together with 'dt' this is the distance a voxel is
            advected, which has to be enough to reach the object's centre.
        dt: Integration step size. Mostly only the product with 'n_iter' matters.
        sigma: Gaussian sigma for smoothing the convergence-density map.
        spacing: Anisotropic voxel spacing, e.g. (4, 1, 1), for physically isotropic smoothing.
        min_candidate_size: Discard components smaller than this, which are noise rather than objects.
        n_threads: Number of threads for the flow computation.

    Returns:
        The prompts as {'points': (N, 1, 2) in XY, 'point_labels': (N, 1), 'frames': (N,) slice
        indices}, or None if no candidate was found.
    """
    if foreground.ndim != 3:
        raise ValueError(f"Volumetric prompt generation expects a (Z, Y, X) foreground map, got {foreground.shape}.")
    if directed_distances.shape[0] != 3:
        raise ValueError(f"Expected 3 distance channels, got {directed_distances.shape[0]}.")

    defaults = default_prompt_generation(model_type, is_volume=True)
    if candidate_threshold is None:
        candidate_threshold = defaults["candidate_threshold"]
    if foreground_threshold is None:
        # 0.7 for every registry backbone in the 2D table (not swept for volumes), so it stays a flat
        # constant here rather than going through `default_prompt_generation`.
        foreground_threshold = 0.7
    if n_iter is None:
        # 50 for every registry backbone in the 2D table, same reasoning as 'foreground_threshold'.
        n_iter = 50
    if dt is None:
        dt = DEFAULT_PROMPT_GENERATION["dt"]
    if sigma is None:
        sigma = defaults["sigma"]
    if min_candidate_size is None:
        min_candidate_size = defaults["min_candidate_size"]

    fg_mask = foreground > foreground_threshold
    density = _compute_flow_density(
        directed_distances, fg_mask, n_iter=int(n_iter), dt=dt, sigma=sigma, spacing=spacing,
        n_threads=n_threads,
    )

    points, frames, seen = [], [], set()
    # Descending, so that the peaks a lower threshold merges into one component are proposed first.
    for threshold in sorted(np.atleast_1d(np.asarray(candidate_threshold, dtype="float32")), reverse=True):
        candidates = label(density > threshold)
        if min_candidate_size > 0:
            ids, sizes = np.unique(candidates, return_counts=True)
            discard = ids[(sizes < min_candidate_size) & (ids > 0)]
            if discard.size:
                candidates[np.isin(candidates, discard)] = 0

        for index, bounding_box in enumerate(find_objects(candidates)):
            if bounding_box is None:
                continue
            component = candidates[bounding_box] == (index + 1)
            component_density = np.where(component, density[bounding_box], 0.0)
            # The most converged slice, i.e. the one closest to the object's centre.
            z = int(np.argmax(component_density.sum(axis=(1, 2))))
            plane = component_density[z]
            # The density peak: it is inside the component and the least ambiguous prompt.
            y, x = np.unravel_index(int(np.argmax(plane)), plane.shape)
            anchor = (bounding_box[0].start + z, bounding_box[1].start + y, bounding_box[2].start + x)
            # A component that no lower threshold has merged peaks at the same voxel at every level.
            if anchor in seen:
                continue
            seen.add(anchor)
            frames.append(anchor[0])
            points.append((anchor[2], anchor[1]))  # SAM2 wants XY.

    if not points:
        return None

    return {
        "points": np.array(points, dtype="float32")[:, None, :],
        "point_labels": np.ones((len(points), 1), dtype="int32"),
        "frames": np.array(frames, dtype="int64"),
    }


def merge_by_score(
    records: List[Dict[str, Any]], shape: tuple, max_overlap: float = 0.3, min_size: int = 50,
    return_matches: bool = False, return_reasons: bool = False,
) -> Union[np.ndarray, tuple]:
    """Merge prediction records in descending score order, each claiming only unclaimed pixels.

    Linear in the number of candidates, where `micro_sam.util.apply_nms` is quadratic, and marginally
    better on livecell: truncating a later mask preserves the better-scoring instance's boundary.

    A record carries a 'bounding_box' (a tuple of slices) and its mask is only the crop inside that
    box, so both the merge and the records themselves scale with the objects rather than with the
    image. A volumetric mask per object would otherwise make the 3d generator run out of memory.

    Args:
        records: The prediction records, as produced by `AutomaticPromptGenerator._apply_prompts`.
        shape: The spatial shape of the output.
        max_overlap: Reject a candidate when more than this fraction of it is already claimed. This is
            the duplicate suppression of the merge.
        min_size: Minimum object size to keep.
        return_matches: Whether to also return which record made each instance.
        return_reasons: Whether to also return why each record was kept or dropped. A candidate is
            'too small', a 'duplicate' when a better-scoring mask already claims more than
            'max_overlap' of it, 'truncated below min size' when too few of its pixels are free, or
            'kept'. This is what the merge does, reported rather than recomputed.
    Returns:
        The instance segmentation, uint32 array. If `return_matches`, additionally a mapping from
        every instance id to the index of the record that made it. If `return_reasons`, additionally
        the reason per record, in the order the records were given.
    """
    out = np.zeros(shape, dtype="uint32")
    scores = np.array([
        record.get("merge_score", record["predicted_iou"] * record["stability_score"])
        for record in records
    ])
    if not np.isfinite(scores).all():
        raise ValueError("Every merge score must be finite.")
    full_box = tuple(slice(None) for _ in shape)
    matches = {}
    reasons = ["" for _ in records]
    accepted_groups = set()
    next_id = 1
    for index in sorted(range(len(records)), key=lambda candidate: (-scores[candidate], candidate)):
        record = records[index]
        group = record.get("multimask_group")
        if group is not None and group in accepted_groups:
            reasons[index] = "alternative not selected"
            continue
        mask = record["segmentation"]
        mask = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        area = int(mask.sum())
        if area < min_size:
            reasons[index] = "too small"
            continue
        # A view, so painting the fresh pixels below writes straight into the output.
        target = out[record.get("bounding_box", full_box)]
        claimed = target[mask]
        if int(np.count_nonzero(claimed)) / area > max_overlap:
            reasons[index] = "duplicate"
            continue
        fresh = mask & (target == 0)
        if int(fresh.sum()) < min_size:
            reasons[index] = "truncated below min size"
            continue
        target[fresh] = next_id
        reasons[index] = "kept"
        matches[next_id] = int(index)
        if group is not None:
            accepted_groups.add(group)
        next_id += 1

    result = (out,)
    if return_matches:
        result += (matches,)
    if return_reasons:
        result += (reasons,)
    return result[0] if len(result) == 1 else result


def _records_shape(records: List[Dict[str, Any]]) -> tuple:
    """The smallest canvas that holds every record, which is all a merge of cropped masks needs."""
    boxes = [record["bounding_box"] for record in records]
    return tuple(max(box[axis].stop for box in boxes) for axis in range(len(boxes[0])))


def _volume_records(
    video_segments: Dict[int, Dict[int, np.ndarray]], candidates: List[dict], shape: tuple,
) -> List[Dict[str, Any]]:
    """Assemble one volumetric mask record per propagated object, cropped to its bounding box.

    The propagation yields its masks slice by slice. Holding one full-volume mask per object would
    scale with the object count times the volume, so each object is kept as its bounding-box crop,
    which scales with the objects themselves. `merge_by_score` paints from those crops.

    Args:
        video_segments: The propagation result, {frame: {object_id: mask}}, as returned by
            `PromptableSegmentation3D.propagate_prompts`.
        candidates: The candidates of this pass, in the order their object ids were assigned.
        shape: The shape of the volume, (Z, Y, X).

    Returns:
        The records, carrying the anchor-slice scores of their candidate for the merge.
    """
    per_object = {}
    for frame, masks in sorted(video_segments.items()):
        for object_id, mask in masks.items():
            mask = _crop_to_original_shape(np.asarray(mask).squeeze(), shape[-2:])
            rows, columns = np.nonzero(mask)
            if len(rows) == 0:
                continue
            y0, y1 = int(rows.min()), int(rows.max()) + 1
            x0, x1 = int(columns.min()), int(columns.max()) + 1
            per_object.setdefault(object_id, []).append(
                (int(frame), y0, y1, x0, x1, mask[y0:y1, x0:x1].copy())
            )

    records = []
    for object_id, entries in per_object.items():
        candidate = candidates[object_id - 1]
        # The frames are in ascending order, so the first and last entry bound the object in z.
        z0, z1 = entries[0][0], entries[-1][0] + 1
        y0, x0 = min(entry[1] for entry in entries), min(entry[3] for entry in entries)
        y1, x1 = max(entry[2] for entry in entries), max(entry[4] for entry in entries)

        local = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=bool)
        for frame, from_y, to_y, from_x, to_x, mask in entries:
            local[frame - z0, from_y - y0:to_y - y0, from_x - x0:to_x - x0] = mask

        records.append({
            "segmentation": local,
            "bounding_box": (slice(z0, z1), slice(y0, y1), slice(x0, x1)),
            "predicted_iou": candidate["score"],
            "stability_score": candidate["stability"],
        })
    return records


class AutomaticPromptGenerator(UniSAM2InstanceSegmentation):
    """Generates an instance segmentation automatically, from prompts derived from the UniSAM2 decoder.

    Used in the same way as `UniSAM2InstanceSegmentation`, and the counterpart of
    `micro_sam.v1.instance_segmentation.AutomaticPromptGenerator`:
    ```python
    segmenter = AutomaticPromptGenerator(model, predictor)
    segmenter.initialize(image, ndim=2)  # Encode the image, then run the decoder on the encoding.
    masks = segmenter.generate(score_threshold=0.6)  # Prompt, then merge the masks.
    ```

    Volumes are segmented by passing the SAM2 video predictor instead of an image predictor:
    ```python
    segmenter = AutomaticPromptGenerator(model, video_predictor)
    segmenter.initialize(volume, ndim=3)
    masks = segmenter.generate()  # Prompt one slice per object, propagate, then merge the masks.
    ```
    The decoder finds the objects and the interactive branch segments them either way. What differs is
    that an object's mask comes from propagating one prompt through the volume rather than from a
    single forward pass, and that the candidates are scored in 2d on the slice they are prompted on,
    which drops the weak and the duplicate ones before the propagation - which is where the cost is.
    The decoder and the video predictor read the same 3d embeddings, so the volume is encoded once.

    Args:
        model: The UniSAM2 model (see `get_unisam2_model` / `get_decoder`).
        predictor: The SAM2 image predictor for the interactive branch of the same model, or its
            video predictor, which is what a volume is prompted with.
        device: The device the model lives on (used for the non-tiled 2d decoder).
        inference_device: The device intent used as the `devices=None` fallback. Defaults to the
            model device (single GPU); pass None to fan out over all visible GPUs, or a device / list.
    """

    # Read by `automatic_instance_segmentation` to decide whether to pass the AIS 'mode' argument.
    _has_postprocessing_mode = False
    _model_type = DEFAULT_MODEL

    def __init__(
        self,
        model: torch.nn.Module,
        predictor,
        device: Optional[Union[str, torch.device]] = None,
        inference_device: Devices = USE_MODEL_DEVICE,
    ) -> None:
        super().__init__(model, device=device, inference_device=inference_device)
        # The image predictor is built on the video predictor's own weights: no second backbone.
        self._video_predictor = predictor if hasattr(predictor, "propagate_in_video") else None
        if self._video_predictor is None:
            self._predictor = predictor
        else:
            self._predictor = get_sam2_image_predictor(predictor)
        predictor = self._predictor

        self._image_embeddings = None
        self._volume = None
        self._propagator = None
        self._offload_to_cpu = None
        self._max_cached_frames = None
        self._temporary_embedding_path = None
        self._i = None
        self._owns_image_embeddings = False
        self._last_generation_stats = {}
        self._microscopy_multimask_scorer = None
        self._refinement_gate_model = None
        # The embedding cache is keyed on these, which a SAM2 image predictor does not carry itself.
        sam2_model = getattr(predictor, "model", None)
        if getattr(predictor, "model_type", None) is None:
            predictor.model_type = getattr(sam2_model, "model_type", None) or "hvit"
        if getattr(predictor, "model_name", None) is None:
            predictor.model_name = getattr(sam2_model, "model_name", None) or predictor.model_type

    def set_multimask_models(self, scorer=None, refinement_gate=None) -> None:
        """Install fitted feature models used by the optional 2D APG optimization modes.

        Both objects implement ``predict(features)`` and are intentionally injected rather than
        loaded from an implicit global path. This keeps checkpoints and evaluation artifacts
        attributable. The normal predicted-IoU path does not require either model.
        """
        if refinement_gate is not None:
            refinement_gate_stage(refinement_gate)
        self._microscopy_multimask_scorer = scorer
        self._refinement_gate_model = refinement_gate

    def _validate_multimask_options(
        self, multimasking: bool, multimask_scorer: str, multimask_selection: str, is_volume: bool,
    ) -> None:
        if multimask_scorer not in ("predicted_iou", "microscopy"):
            raise ValueError(
                f"Invalid multimask scorer {multimask_scorer!r}: expected 'predicted_iou' or 'microscopy'."
            )
        if multimask_selection not in ("eager", "deferred"):
            raise ValueError(
                f"Invalid multimask selection {multimask_selection!r}: expected 'eager' or 'deferred'."
            )
        changed = multimask_scorer != "predicted_iou" or multimask_selection != "eager"
        if multimask_selection == "deferred" and not multimasking:
            raise ValueError("Deferred multimask selection requires multimasking=True.")
        if is_volume and changed:
            raise ValueError("Microscopy multimask scoring and deferred selection currently support 2d only.")
        if multimask_scorer == "microscopy" and self._microscopy_multimask_scorer is None:
            raise RuntimeError(
                "multimask_scorer='microscopy' requires a fitted scorer; call set_multimask_models first."
            )

    def _encode(self, image: np.ndarray) -> dict:
        """Run the image encoder once and return the embeddings that both branches use."""
        self._predictor.reset_predictor()
        # Preserve each microscopy channel's contrast before SAM2 sees the image.
        encode_image(self._predictor, to_image(image))
        return {
            "features": self._predictor.get_image_embedding().cpu().numpy(),
            "high_res_feats": self._predictor._features["high_res_feats"],
            "input_size": self._predictor.model.image_size,
            "original_size": self._predictor._orig_hw,
        }

    def _prepare_image_embeddings(self, image_embeddings: dict, i: Optional[int]) -> dict:
        """Prepare image embeddings for the interactive branch."""
        if i is None:
            set_precomputed(self._predictor, image_embeddings)
            return image_embeddings

        if "fpn" not in image_embeddings:
            raise ValueError("A slice index requires video-style embeddings with FPN features.")
        _set_image_predictor_from_3d_embeddings(self._predictor, image_embeddings, i)
        return {
            "features": self._predictor.get_image_embedding().detach().cpu().numpy(),
            "high_res_feats": self._predictor._features["high_res_feats"],
            "input_size": self._predictor.model.image_size,
            "original_size": self._predictor._orig_hw,
        }

    def initialize(
        self,
        image: np.ndarray,
        ndim: int = 2,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        save_path: Optional[str] = None,
        verbose: bool = False,
        offload_to_cpu: Optional[bool] = None,
        cache_all_slices: bool = False,
        lazy_embeddings: bool = True,
        **kwargs,
    ) -> None:
        """Encode the input, run the decoder on that encoding and leave the predictor ready to be prompted.

        Both branches of a joint checkpoint share their image encoder weights, so one pass serves both
        and `generate` can be called repeatedly without any further encoding.

        Args:
            image: The input image, shape (Y, X) or (Y, X, C), or the input volume, shape (Z, Y, X).
            ndim: The number of spatial dimensions, 2 or 3. A volume requires a video predictor.
            image_embeddings: Optional precomputed image embeddings. If given, the encoder does not run.
            i: The slice index for video-style embeddings. By default the embeddings contain one image.
            save_path: Optional path to cache the embeddings of a volume in a zarr container. Without
                one an ephemeral store is used, which `clear_state` removes.
            verbose: Whether to print progress while the embeddings of a volume are computed.
            offload_to_cpu: Whether a volume's tracking state is held on the host rather than on the
                device. False propagates the same masks 6-11% faster, for the memory of one pass of
                objects across the volume, which is about 17 MB per slice at the default pass size.
                None leaves the choice to `PromptableSegmentation3D`, which offloads on cuda.
            cache_all_slices: Whether every slice's features stay on the device while a volume is
                propagated. A pass walks the whole volume, so a cache shorter than it is never hit
                and every slice is fetched again on every pass - which is most of the run on a deep
                volume. Costs about 90 MB of device memory per slice.
            lazy_embeddings: Whether a volume's embeddings are read from their store slice by slice.
                False holds them in host memory instead, which makes the fetch above much cheaper
                without spending device memory.
            kwargs: Additional arguments for `UniSAM2InstanceSegmentation.initialize`.
        """
        if ndim not in (2, 3):
            raise ValueError(f"Automatic prompt generation supports 2d and 3d inputs, got ndim={ndim}.")

        # A volume leaves a propagator and possibly a temporary embedding store behind. Release both
        # before replacing its state, including when the next input is a 2d image.
        if self._volume is not None or self._temporary_embedding_path is not None:
            self.clear_state()

        if ndim == 3:
            self._i = None
            self._initialize_volume(
                image, image_embeddings, save_path, verbose, offload_to_cpu, cache_all_slices,
                lazy_embeddings, **kwargs
            )
            return

        owns_image_embeddings = image_embeddings is None
        if image_embeddings is None:
            image_embeddings = self._encode(image)
        else:
            image_embeddings = self._prepare_image_embeddings(image_embeddings, i)
        super().initialize(image, ndim=ndim, image_embeddings=image_embeddings, **kwargs)
        self._image_embeddings = image_embeddings
        self._i = None
        self._owns_image_embeddings = owns_image_embeddings

    def _build_propagator(self, volume, image_embeddings) -> PromptableSegmentation3D:
        """The video predictor this volume is propagated with, holding its state where asked to."""
        return PromptableSegmentation3D(
            self._video_predictor, volume, image_embeddings,
            offload_state_to_cpu=self._offload_to_cpu, max_cached_frames=self._max_cached_frames,
        )

    def _initialize_volume(
        self, volume, image_embeddings, save_path, verbose, offload_to_cpu, cache_all_slices,
        lazy_embeddings, **kwargs
    ) -> None:
        """Encode the volume, run the decoder on that encoding and initialize the video predictor."""
        if self._video_predictor is None:
            raise ValueError(
                "Volumetric prompt generation prompts the SAM2 video predictor, so it has to be "
                "constructed with one instead of an image predictor."
            )
        if volume.ndim != 3:
            raise ValueError(f"Volumetric prompt generation expects a (Z, Y, X) volume, got shape {volume.shape}.")

        owns_image_embeddings = image_embeddings is None
        if image_embeddings is None:
            path = save_path
            if path is None:
                self._temporary_embedding_path = make_temp_embedding_path()
                path = self._temporary_embedding_path
            image_embeddings = precompute_image_embeddings(
                self._video_predictor, volume, save_path=path, ndim=3, verbose=verbose,
                lazy_loading=lazy_embeddings,
            )

        UniSAM2InstanceSegmentation.initialize(self, volume, ndim=3, image_embeddings=image_embeddings, **kwargs)
        self._image_embeddings = image_embeddings
        self._owns_image_embeddings = owns_image_embeddings
        self._volume = volume
        self._offload_to_cpu = offload_to_cpu
        self._max_cached_frames = int(volume.shape[0]) if cache_all_slices else None
        self._propagator = self._build_propagator(volume, image_embeddings)

    def get_state(self) -> dict:
        """Return the decoder predictions and the image embeddings, so that both branches can be restored.

        `generate` also prompts the interactive branch, which needs the encoding of the same input. For
        a volume the volume itself is part of the state, because the video predictor is initialized on it.
        """
        state = super().get_state()
        state["image_embeddings"] = self._image_embeddings
        if self._volume is not None:
            state["volume"] = self._volume
        return state

    def set_state(self, state: dict) -> None:
        """Restore the decoder predictions and the encoding of the input they belong to.

        The state must hold either 'image_embeddings' or 'image'. Without one, `generate` would prompt
        whatever image the predictor still holds. A volumetric state must also hold 'volume'.

        Args:
            state: The state, as returned by `get_state`, or a dict with 'prediction' and 'image'.
        """
        image_embeddings = state.get("image_embeddings")
        volume = state.get("volume")
        i = state.get("i")

        if volume is not None:
            if image_embeddings is None:
                raise ValueError("A volumetric state must hold the 'image_embeddings' of its volume.")
            super().set_state(state)
            self._volume = volume
            self._propagator = self._build_propagator(volume, image_embeddings)
            self._i = None
        else:
            if image_embeddings is None:
                if "image" not in state:
                    raise ValueError("The state must hold either 'image_embeddings' or 'image'.")
                image_embeddings = self._encode(state["image"])
            else:
                image_embeddings = self._prepare_image_embeddings(image_embeddings, i)
            super().set_state(state)
            self._i = None
        self._image_embeddings = image_embeddings
        self._owns_image_embeddings = False

    def clear_state(self) -> None:
        """Clear the decoder predictions and the input that is set on the predictor."""
        owned_embeddings = self._image_embeddings if getattr(self, "_owns_image_embeddings", False) else None
        super().clear_state()
        self._image_embeddings = None
        self._i = None
        self._owns_image_embeddings = False
        self._predictor.reset_predictor()
        self._volume = None
        if self._propagator is not None:
            self._propagator.reset_predictor()
            self._propagator = None
        try:
            close = getattr(owned_embeddings, "close", None)
            if close is not None:
                close()
        finally:
            if self._temporary_embedding_path is not None:
                shutil.rmtree(self._temporary_embedding_path, ignore_errors=True)
                self._temporary_embedding_path = None

    @torch.no_grad()
    def generate(
        self,
        candidate_threshold: Optional[Union[float, Sequence[float]]] = None,
        foreground_threshold: Optional[float] = None,
        n_iter: Optional[int] = None,
        dt: Optional[float] = None,
        sigma: Optional[float] = None,
        spacing: Optional[tuple] = None,
        min_candidate_size: Optional[int] = None,
        score_threshold: Optional[float] = None,
        score_filter: str = DEFAULT_PROMPT_GENERATION["score_filter"],
        max_overlap: Optional[float] = None,
        min_size: Optional[int] = None,
        refinement: Optional[str] = DEFAULT_PROMPT_GENERATION["refinement"],
        refinement_kwargs: Optional[Dict[str, Any]] = DEFAULT_PROMPT_GENERATION["refinement_kwargs"],
        multimasking: bool = DEFAULT_PROMPT_GENERATION["multimasking"],
        multimask_scorer: str = DEFAULT_PROMPT_GENERATION["multimask_scorer"],
        multimask_selection: str = DEFAULT_PROMPT_GENERATION["multimask_selection"],
        n_objects_per_pass: int = DEFAULT_PROMPT_GENERATION["n_objects_per_pass"],
        early_stop_patience: Optional[int] = DEFAULT_PROMPT_GENERATION["early_stop_patience"],
        batch_size: int = DEFAULT_PROMPT_GENERATION["batch_size"],
        n_threads: int = DEFAULT_PROMPT_GENERATION["n_threads"],
        verbose: bool = False,
    ) -> np.ndarray:
        """Derive prompts from the stored predictions, apply them and merge the masks.

        Args:
            candidate_threshold: Density threshold for proposing candidates, or several of them for
                a volume. By default resolved per model type and dimensionality, see
                `default_prompt_generation`.
            foreground_threshold: Foreground binarisation threshold. Here it only limits which pixels can
                be proposed from, since the masks come from the interactive branch, so it trades candidate
                recall rather than boundary quality.
            n_iter: Number of flow-integration steps for the candidate density.
            dt: Integration step size. Mostly only the product with 'n_iter' matters, which is the
                distance a pixel is advected.
            sigma: Gaussian sigma for smoothing the candidate density. Less smoothing leaves more peaks,
                which costs precision for the sparse post-processing but buys candidate recall here.
            spacing: Anisotropic voxel spacing of a volume, e.g. (4, 1, 1).
            min_candidate_size: Discard density components smaller than this.
            score_threshold: Discard candidates whose selected filter score is below this.
            score_filter: Filter image proposals by 'predicted_iou' (the default), the installed
                model's 'selection_score', or 'none'. Volumes support predicted IoU only.
            max_overlap: Reject a candidate when more than this fraction of it is already claimed. For a
                volume this applies on the slice a candidate is prompted on and again on the 3d merge.
            min_size: Minimum object size in the result.
            refinement: Optional second round, a '+'-joined combination of 'points' (the first
                round's prompts grouped onto each merged instance, see `derive_refinement_prompts`),
                'boxes' (its bounding box) and 'masks' (its mask as a logit prompt). None (the default)
                runs no second round. For a volume the round runs on each candidate's anchor slice,
                before the propagation, and produces the conditioning that propagation starts from
                rather than a finished mask, see `_refine_anchors`.
            refinement_kwargs: Keyword arguments of that second round, validated against the mode's
                components; see `DEFAULT_REFINEMENT`, or `DEFAULT_REFINEMENT_3D` for a volume, for
                the accepted keys and their defaults.
            multimasking: Whether to predict several masks per point and keep the best scoring one. A
                single point is ambiguous between one object and a cluster, so this is on by default.
            multimask_scorer: How to score the alternatives of an image prompt: SAM2's generic
                'predicted_iou' or an installed microscopy feature scorer.
            multimask_selection: Whether to choose one alternative 'eager'ly or defer the choice to
                the grouped score-ordered merge. Images only.
            n_objects_per_pass: Number of objects propagated together through a volume. The video
                predictor runs them as one batch, so this trades device memory against the pass count.
            early_stop_patience: Stop a propagation pass after this many consecutive slices in which
                every object of the pass is empty. None propagates through the whole volume. Two by
                default: propagating past the end of every object of a pass only reproduces empty
                masks, so this saves work without changing the result. Raise it where an object may
                disappear and reappear, because a stop that fires mid-object truncates it.
            batch_size: Number of prompts per forward pass.
            n_threads: Number of threads for the flow integration the candidates come from.
            verbose: Whether to show progress over the propagation passes of a volume.

        Returns:
            The instance segmentation, uint32 array with the spatial shape of the prediction.
        """
        if not self._is_initialized:
            raise RuntimeError("The segmenter has not been initialized. Call 'initialize' first.")

        self._last_generation_stats = {}
        shape = self._prediction[0].shape
        # The prediction carries the dimensionality it was run at: (4, Y, X) or (4, Z, Y, X).
        is_volume = self._prediction.ndim == 4
        defaults = default_prompt_generation(self._model_type, is_volume=is_volume)
        self._validate_multimask_options(
            multimasking, multimask_scorer, multimask_selection, is_volume=is_volume,
        )
        if candidate_threshold is None:
            candidate_threshold = defaults["candidate_threshold"]
        if score_threshold is None:
            score_threshold = defaults["score_threshold"]
        if max_overlap is None:
            max_overlap = defaults["max_overlap"]
        if min_size is None:
            min_size = defaults["min_size"]

        if is_volume:
            if score_filter != "predicted_iou":
                raise ValueError("Volumes currently support score_filter='predicted_iou' only.")
            components = resolved = None
            if refinement is not None:
                components, resolved = _parse_refinement(refinement, refinement_kwargs, is_volume=True)
            prompts = derive_volume_prompts(
                self._prediction[0], self._prediction[1:], model_type=self._model_type,
                candidate_threshold=candidate_threshold, foreground_threshold=foreground_threshold,
                n_iter=n_iter, dt=dt, sigma=sigma, spacing=spacing,
                min_candidate_size=min_candidate_size, n_threads=n_threads,
            )
            if prompts is None:
                self._last_generation_stats = {
                    "proposed_candidates": 0,
                    "scored_candidates": 0,
                    "unique_anchor_slices": 0,
                    "propagation_passes": 0,
                    "propagated_frame_steps": 0,
                    "early_stopped_frame_steps": 0,
                }
                if components is not None:
                    self._last_generation_stats.update({key: 0 for key in REFINEMENT_STATS_3D})
                return np.zeros(shape, dtype="uint32")
            self._last_generation_stats["proposed_candidates"] = len(prompts["points"])
            if components is not None:
                self._last_generation_stats.update({key: 0 for key in REFINEMENT_STATS_3D})
            # The refinement's forwards are not wrapped by '_apply_prompts', which has its own.
            if components is None:
                candidates = self._score_candidates(
                    prompts, multimasking=multimasking, batch_size=batch_size,
                    score_threshold=score_threshold, max_overlap=max_overlap,
                    components=components, refinement_kwargs=resolved,
                )
            else:
                with autocast(self._predictor.device):
                    candidates = self._score_candidates(
                        prompts, multimasking=multimasking, batch_size=batch_size,
                        score_threshold=score_threshold, max_overlap=max_overlap,
                        components=components, refinement_kwargs=resolved,
                    )
            self._last_generation_stats["scored_candidates"] = len(candidates)
            records = self._propagate_candidates(
                candidates, n_objects_per_pass=n_objects_per_pass,
                early_stop_patience=early_stop_patience, verbose=verbose,
            )
            return merge_by_score(records, shape, max_overlap=max_overlap, min_size=min_size)

        proposals = self.propose(
            candidate_threshold=candidate_threshold, foreground_threshold=foreground_threshold,
            n_iter=n_iter, dt=dt, sigma=sigma, min_candidate_size=min_candidate_size,
            multimasking=multimasking, multimask_scorer=multimask_scorer,
            multimask_selection=multimask_selection, batch_size=batch_size, n_threads=n_threads,
            compute_multimask_uncertainty=(
                refinement is not None
                and (refinement_kwargs or {}).get("gate", DEFAULT_REFINEMENT["gate"]) == "uncertainty"
                and refinement_gate_stage(self._refinement_gate_model) == "premerge"
            ),
        )
        return self.select(
            proposals, score_threshold=score_threshold, score_filter=score_filter,
            max_overlap=max_overlap, min_size=min_size,
            refinement=refinement, refinement_kwargs=refinement_kwargs, batch_size=batch_size,
        )

    @torch.no_grad()
    def propose(
        self,
        candidate_threshold: Optional[float] = None,
        foreground_threshold: Optional[float] = None,
        n_iter: Optional[int] = None,
        dt: Optional[float] = None,
        sigma: Optional[float] = None,
        min_candidate_size: Optional[int] = None,
        multimasking: bool = DEFAULT_PROMPT_GENERATION["multimasking"],
        multimask_scorer: str = DEFAULT_PROMPT_GENERATION["multimask_scorer"],
        multimask_selection: str = DEFAULT_PROMPT_GENERATION["multimask_selection"],
        batch_size: int = DEFAULT_PROMPT_GENERATION["batch_size"],
        n_threads: int = DEFAULT_PROMPT_GENERATION["n_threads"],
        compute_multimask_uncertainty: bool = False,
        return_multimask_features: bool = False,
        multimask_feature_schema: Optional[str] = None,
    ) -> list:
        """Derive the prompts and turn them into scored mask proposals, without selecting any of them.

        This is the half of `generate` that needs the model. Splitting it off lets a parameter sweep
        reuse one set of proposals across every value of 'score_threshold', 'max_overlap' and
        'min_size', which only filter and merge. Images only: a volume gates its propagation on the
        score, so its proposals depend on those parameters too.

        Args:
            candidate_threshold: Density threshold for proposing candidates, see `derive_point_prompts`.
            foreground_threshold: Foreground binarisation threshold.
            n_iter: Number of flow-integration steps for the candidate density.
            dt: Integration step size.
            sigma: Gaussian sigma for smoothing the candidate density.
            min_candidate_size: Discard density components smaller than this.
            multimasking: Whether to predict several masks per point and keep the best scoring one.
            multimask_scorer: 'predicted_iou' or an installed 'microscopy' feature scorer.
            multimask_selection: Choose one alternative 'eager'ly or retain a grouped 'deferred' set.
            batch_size: Number of prompts per forward pass.
            n_threads: Number of threads for the flow integration the candidates come from.
            compute_multimask_uncertainty: Attach refinement-gate scores to the selected records.
            return_multimask_features: Attach the selector feature vector for training or diagnostics.
            multimask_feature_schema: Internal extraction override for compact scorer training. None
                takes the installed scorer's schema, or the historical dense schema without one.

        Returns:
            The proposals, to be passed to `select`. Their layout is an implementation detail of the
            generator that produced them.
        """
        if not self._is_initialized:
            raise RuntimeError("The segmenter has not been initialized. Call 'initialize' first.")
        if self._prediction.ndim == 4:
            raise ValueError("Proposals can only be reused for an image, because a volume gates its propagation.")
        self._validate_multimask_options(
            multimasking, multimask_scorer, multimask_selection, is_volume=False,
        )
        if compute_multimask_uncertainty and not multimasking:
            raise ValueError("Uncertainty-gated refinement requires multimasking=True.")
        if compute_multimask_uncertainty and self._refinement_gate_model is None:
            raise RuntimeError(
                "Computing multimask uncertainty requires a fitted refinement gate; "
                "call set_multimask_models first."
            )

        defaults = default_prompt_generation(self._model_type, is_volume=False)
        if foreground_threshold is None:
            foreground_threshold = defaults["foreground_threshold"]
        prompts = derive_point_prompts(
            self._prediction[0], self._prediction[1:], model_type=self._model_type,
            candidate_threshold=candidate_threshold, foreground_threshold=foreground_threshold,
            n_iter=n_iter, dt=dt, sigma=sigma, min_candidate_size=min_candidate_size, n_threads=n_threads,
        )
        if prompts is None:
            return []
        return self._apply(
            prompts, multimasking=multimasking, batch_size=batch_size,
            multimask_scorer=multimask_scorer, multimask_selection=multimask_selection,
            compute_multimask_uncertainty=compute_multimask_uncertainty,
            return_multimask_features=return_multimask_features,
            multimask_feature_schema=multimask_feature_schema,
            foreground_threshold=foreground_threshold,
        )

    def select(
        self,
        proposals: list,
        score_threshold: Optional[float] = None,
        score_filter: str = DEFAULT_PROMPT_GENERATION["score_filter"],
        max_overlap: Optional[float] = None,
        min_size: Optional[int] = None,
        refinement: Optional[str] = DEFAULT_PROMPT_GENERATION["refinement"],
        refinement_kwargs: Optional[Dict[str, Any]] = DEFAULT_PROMPT_GENERATION["refinement_kwargs"],
        batch_size: int = DEFAULT_PROMPT_GENERATION["batch_size"],
    ) -> np.ndarray:
        """Merge the proposals of `propose` into an instance segmentation.

        Args:
            proposals: The proposals, as returned by `propose` on the same generator.
            score_threshold: Discard proposals whose selected filter score is below this.
            score_filter: The proposal field used by the threshold: 'predicted_iou',
                'selection_score', or 'none'.
            max_overlap: Reject a proposal when more than this fraction of it is already claimed.
            min_size: Minimum object size in the result.
            refinement: Optional second round, a '+'-joined combination of 'points', 'boxes',
                and 'masks'; see `generate`.
            refinement_kwargs: Keyword arguments of that second round, validated against the mode's
                components; see `DEFAULT_REFINEMENT` for the accepted keys and their defaults.
            batch_size: Number of prompts per forward pass of the refinement.

        Returns:
            The instance segmentation, uint32 array with the spatial shape of the prediction.
        """
        defaults = default_prompt_generation(self._model_type, is_volume=False)
        if score_threshold is None:
            score_threshold = defaults["score_threshold"]
        if max_overlap is None:
            max_overlap = defaults["max_overlap"]
        if min_size is None:
            min_size = defaults["min_size"]

        components = resolved = None
        if score_filter not in ("predicted_iou", "selection_score", "none"):
            raise ValueError(
                f"Invalid score filter {score_filter!r}: expected 'predicted_iou', "
                "'selection_score' or 'none'."
            )
        if refinement is not None:
            components, resolved = _parse_refinement(refinement, refinement_kwargs)

        shape = self._prediction[0].shape
        if not proposals:
            return np.zeros(shape, dtype="uint32")

        segmentation, context = self._merge(
            proposals, shape, score_threshold=score_threshold, max_overlap=max_overlap, min_size=min_size,
            return_context=components is not None, score_filter=score_filter,
        )
        if components is not None and segmentation.max() > 0:
            segmentation = self._refine(segmentation, context, components, resolved, batch_size)
        return segmentation

    def _region_of(self, context: dict, record_index: int):
        """The region a record's instance is re-prompted in, keyed however the generator likes.

        A refinement runs region by region, because a tiled generator has to point its predictor at
        one tile at a time. Here the whole image is one region, so every record shares the key None.
        """
        return None

    def _region_box(self, key) -> tuple:
        """The region's bounding box, as a slice tuple into the segmentation."""
        return (slice(None), slice(None))

    def _set_region(self, key) -> None:
        """Point the predictor at the region. Its image is already set for a single one."""

    def _apply(
        self, prompts: dict, multimasking: bool, batch_size: int, multimask_scorer: str = "predicted_iou",
        multimask_selection: str = "eager", compute_multimask_uncertainty: bool = False,
        return_multimask_features: bool = False,
        multimask_feature_schema: Optional[str] = None,
        foreground_threshold: float = DEFAULT_PROMPT_GENERATION["foreground_threshold"],
    ) -> list:
        """Turn the prompts into mask proposals."""
        kwargs = {"multimasking": multimasking, "batch_size": batch_size}
        if (
            multimask_scorer != "predicted_iou"
            or multimask_selection != "eager"
            or compute_multimask_uncertainty
            or return_multimask_features
            or multimask_feature_schema is not None
        ):
            kwargs.update({
                "multimask_scorer": multimask_scorer, "multimask_selection": multimask_selection,
                "compute_multimask_uncertainty": compute_multimask_uncertainty,
                "return_multimask_features": return_multimask_features,
                "multimask_feature_schema": multimask_feature_schema,
                "foreground": self._prediction[0], "foreground_threshold": foreground_threshold,
            })
        return self._apply_prompts(prompts, **kwargs)

    def _merge(
        self, proposals: list, shape: tuple, score_threshold: float, max_overlap: float, min_size: int,
        return_context: bool = False, score_filter: str = "predicted_iou",
    ) -> tuple:
        """Merge the mask proposals into an instance segmentation.

        Returns:
            The segmentation, and the refinement context: the score-filtered records and the mapping
            from every instance id to the record that made it. Without `return_context` the context
            is None, so the matches are not computed for nothing.
        """
        if score_filter == "none":
            records = list(proposals)
        else:
            missing = [index for index, record in enumerate(proposals) if score_filter not in record]
            if missing:
                raise ValueError(
                    f"Cannot filter by {score_filter!r}: {len(missing)} proposal records lack that score."
                )
            records = [record for record in proposals if record[score_filter] >= score_threshold]
        if not records:
            return np.zeros(shape, dtype="uint32"), None
        if not return_context:
            return merge_by_score(records, shape, max_overlap=max_overlap, min_size=min_size), None
        segmentation, matches, reasons = merge_by_score(
            records, shape, max_overlap=max_overlap, min_size=min_size,
            return_matches=True, return_reasons=True,
        )
        self._last_generation_stats.update({
            "proposed_candidates": len(proposals),
            "scored_candidates": len(records),
            "merge_reasons": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
        })
        return segmentation, {
            "proposals": proposals, "records": records, "matches": matches,
            "score_threshold": score_threshold, "score_filter": score_filter,
        }

    def _refine(
        self, segmentation: np.ndarray, context: dict, components: tuple, refinement_kwargs: dict,
        batch_size: int,
    ) -> np.ndarray:
        """Re-prompt the accepted instances with the requested refinement components."""
        with autocast(self._predictor.device):
            return self._reprompt_instances(
                segmentation, context, components, refinement_kwargs, batch_size,
            )

    def _reprompt_instances(
        self, segmentation: np.ndarray, context: dict, components: tuple, refinement_kwargs: dict,
        batch_size: int,
    ) -> np.ndarray:
        """Predict one second-round mask per instance from the assembled prompts and repaint.

        Every requested component contributes to one joint re-prompt per instance, evaluated in
        batches. The 'policy' decides what the second round is allowed to do: 'replace' repaints
        every instance from its new mask (the most confident painted last, so it wins contested
        pixels), 'keep-if-better' keeps the first-round mask unless the new one scores higher. The
        geometric gates then veto independently of either policy: a second-round mask that is
        inconsistent with the first round ('min_consistency') or grows into a neighbour
        ('max_foreign_overlap') is discarded, because the model's own score cannot arbitrate across
        prompt types. An instance whose re-prompt comes back empty keeps its first-round mask.

        The instances are grouped by the region they are re-prompted in and every region is set up
        once, since a tiled generator pays for each switch. Within a region everything runs on its
        crop of the segmentation, which is the frame the predictor works in; only the repaint at
        the end is global, so the score order arbitrates across regions as well as within them.
        """
        shape = segmentation.shape
        all_instances = [
            (index + 1, bounding_box)
            for index, bounding_box in enumerate(find_objects(segmentation))
            if bounding_box is not None
        ]
        instances = all_instances
        unselected = []
        gate_requested = refinement_kwargs.get("gate", "all") == "uncertainty"
        gate_model = getattr(self, "_refinement_gate_model", None)
        gate_stage = refinement_gate_stage(gate_model)
        if gate_requested and gate_stage == "premerge":
            threshold = float(refinement_kwargs["gate_threshold"])
            instances = []
            for instance in all_instances:
                instance_id, _ = instance
                record = context["records"][context["matches"][instance_id]]
                if "uncertainty_score" not in record:
                    raise RuntimeError(
                        "Uncertainty-gated refinement requires proposals carrying uncertainty scores. "
                        "Generate them with a fitted refinement gate model."
                    )
                (instances if record["uncertainty_score"] >= threshold else unselected).append(instance)

        point_prompts = None
        if "points" in components:
            all_points_list, seen_groups = [], set()
            for record_index, record in enumerate(context["proposals"]):
                group = record.get("multimask_group", ("record", record_index))
                if group in seen_groups:
                    continue
                seen_groups.add(group)
                all_points_list.append(record["point"])
            all_points = np.array(all_points_list, dtype="float32")
            surviving_points = {
                instance_id: context["records"][record_index]["point"]
                for instance_id, record_index in context["matches"].items()
            }
            point_prompts = derive_refinement_prompts(
                segmentation, all_points, surviving_points,
                n_positives=refinement_kwargs["n_positives"], n_negatives=refinement_kwargs["n_negatives"],
                max_negative_distance=refinement_kwargs["max_negative_distance"],
                negative_source=refinement_kwargs["negative_source"],
                min_negative_distance=refinement_kwargs["min_negative_distance"],
            )

        if gate_requested and gate_stage == "postmerge":
            if gate_model is None:
                raise RuntimeError(
                    "Post-merge uncertainty-gated refinement requires a fitted refinement gate; "
                    "call set_multimask_models first."
                )
            first_record = context["records"][next(iter(context["matches"].values()))]
            feature_foreground_threshold = float(first_record.get(
                "foreground_threshold", DEFAULT_PROMPT_GENERATION["foreground_threshold"],
            ))
            gate_features, gate_instance_ids = postmerge_refinement_gate_features(
                segmentation, context, point_prompts, self._prediction[0], feature_foreground_threshold,
            )
            if hasattr(gate_model, "predict_tensor"):
                gate_scores = gate_model.predict_tensor(gate_features).cpu().numpy()
            else:
                gate_scores = np.asarray(gate_model.predict(gate_features), dtype="float32")
            if gate_scores.shape != (len(gate_instance_ids),) or not np.isfinite(gate_scores).all():
                raise RuntimeError("The post-merge refinement gate returned invalid scores.")
            threshold = float(refinement_kwargs["gate_threshold"])
            by_id = {int(instance_id): float(score) for instance_id, score in zip(gate_instance_ids, gate_scores)}
            instances, unselected = [], []
            for instance in all_instances:
                instance_id, _ = instance
                record = context["records"][context["matches"][instance_id]]
                record["uncertainty_score"] = by_id[instance_id]
                (instances if by_id[instance_id] >= threshold else unselected).append(instance)

        self._last_generation_stats.update({
            "refinement_eligible_instances": len(all_instances),
            "uncertainty_selected_instances": len(instances),
        })
        if not instances:
            self._last_generation_stats.update({
                "refined_instances": 0, "replaced_instances": 0,
                "dropped_negatives": 0,
                "gated_consistency": 0, "gated_foreign": 0,
            })
            return segmentation

        # Every instance needs the record that made it, for its first-round score.
        groups = {}
        for instance_id, bounding_box in instances:
            if instance_id not in context["matches"]:
                raise RuntimeError(
                    f"Instance {instance_id} is in the segmentation but not in the merge context. The "
                    "refinement cannot score it against its first round."
                )
            key = self._region_of(context, context["matches"][instance_id])
            groups.setdefault(key, []).append((instance_id, bounding_box))

        min_consistency = refinement_kwargs["min_consistency"]
        max_foreign_overlap = refinement_kwargs["max_foreign_overlap"]
        keep_if_better = refinement_kwargs["policy"] == "keep-if-better"
        chosen, replaced, dropped = [], 0, 0
        for instance_id, bounding_box in unselected:
            record = context["records"][context["matches"][instance_id]]
            chosen.append((
                record.get("merge_score", record["predicted_iou"] * record["stability_score"]),
                instance_id, bounding_box, segmentation[bounding_box] == instance_id,
            ))
        gated = {"gated_consistency": 0, "gated_foreign": 0}
        for key in sorted(groups):
            self._set_region(key)
            region_box = self._region_box(key)
            crop = segmentation[region_box]
            origin = tuple(box.start or 0 for box in region_box)

            region_prompts = point_prompts
            if point_prompts is not None and (any(origin) or crop.shape != shape):
                region_prompts = {}
                for instance_id, _ in groups[key]:
                    region_prompts[instance_id], region_dropped = _localize_prompts(
                        point_prompts[instance_id], origin, crop.shape
                    )
                    dropped += region_dropped
            region_instances = [
                (instance_id, _shift_box(bounding_box, tuple(-shift for shift in origin)))
                for instance_id, bounding_box in groups[key]
            ]

            for start in range(0, len(region_instances), batch_size):
                batch = region_instances[start:start + batch_size]
                predictions = self._predict_refinement_batch(
                    crop, batch, components, region_prompts, refinement_kwargs,
                )
                for (instance_id, bounding_box), (mask, score) in zip(batch, predictions):
                    record = context["records"][context["matches"][instance_id]]
                    first_round_score = record["predicted_iou"] * record["stability_score"]
                    first_round_merge_score = record.get("merge_score", first_round_score)
                    take_second = mask.any() and (not keep_if_better or score > first_round_score)
                    if take_second and min_consistency is not None:
                        first_round_mask = crop == instance_id
                        union = int(np.count_nonzero(mask | first_round_mask))
                        iou = int(np.count_nonzero(mask & first_round_mask)) / union if union else 0.0
                        if iou < min_consistency:
                            take_second = False
                            gated["gated_consistency"] += 1
                    if take_second and max_foreign_overlap is not None:
                        on_mask = crop[mask]
                        foreign = int(np.count_nonzero((on_mask != 0) & (on_mask != instance_id)))
                        if foreign / int(mask.sum()) > max_foreign_overlap:
                            take_second = False
                            gated["gated_foreign"] += 1
                    if take_second:
                        replaced += 1
                        rows, columns = np.nonzero(mask)
                        box = (slice(int(rows.min()), int(rows.max()) + 1),
                               slice(int(columns.min()), int(columns.max()) + 1))
                        chosen.append((score, instance_id, _shift_box(box, origin), mask[box]))
                    else:
                        chosen.append((
                            first_round_merge_score, instance_id, _shift_box(bounding_box, origin),
                            crop[bounding_box] == instance_id,
                        ))

        self._last_generation_stats.update({
            "refined_instances": len(instances), "replaced_instances": replaced,
            "dropped_negatives": dropped, **gated,
        })
        # Ascending score, so that the most confident instance is painted last and wins contested pixels.
        refined = np.zeros(shape, dtype="uint32")
        for score, instance_id, bounding_box, mask in sorted(chosen, key=lambda entry: (entry[0], entry[1])):
            refined[bounding_box][mask] = instance_id
        return refined

    def _predict_refinement_batch(
        self, segmentation: np.ndarray, batch: list, components: tuple,
        point_prompts: Optional[dict], refinement_kwargs: dict,
    ) -> list:
        """Return one (mask, combined score) pair per instance in ``batch``."""
        shape = segmentation.shape
        box_extension = refinement_kwargs.get("box_extension", 0)
        points = labels = boxes = mask_logits = None

        if "points" in components:
            per_instance = [point_prompts[instance_id] for instance_id, _ in batch]
            width = max(len(prompt["points"]) for prompt in per_instance)
            points = np.zeros((len(batch), width, 2), dtype="float32")
            labels = np.full((len(batch), width), -1, dtype="int32")
            for row, prompt in enumerate(per_instance):
                points[row, :len(prompt["points"])] = prompt["points"]
                labels[row, :len(prompt["points"])] = prompt["point_labels"]
        if "boxes" in components:
            boxes = np.array([
                _prompt_box(bounding_box, shape, box_extension) for _, bounding_box in batch
            ], dtype="float32")
        if "masks" in components:
            mask_logits = np.stack([mask_to_logits(segmentation == instance_id) for instance_id, _ in batch])

        return self._predict_prompt_batch(
            points, labels, boxes, mask_logits, refinement_kwargs["multimasking"],
        )

    def _predict_prompt_batch(
        self,
        points: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        boxes: Optional[np.ndarray],
        mask_logits: Optional[np.ndarray],
        multimasking: bool,
    ) -> list:
        """One batched forward pass over assembled prompts, returning (mask, combined score) pairs.

        The point counts may differ between the batch's prompts, so the coordinate arrays are padded
        to the longest with the label -1, which the SAM2 prompt encoder treats as 'not a point'.
        """
        n_prompts = len(points) if points is not None else len(boxes if boxes is not None else mask_logits)
        mask_input, coords, point_labels, box_input = self._predictor._prep_prompts(
            points, labels, boxes, mask_logits, True,
        )
        logits, scores, _ = self._predictor._predict(
            coords, point_labels, box_input, mask_input, multimasking, return_logits=True,
        )
        logits = logits.reshape(n_prompts, -1, *logits.shape[-2:])
        scores = scores.reshape(n_prompts, -1)
        index = torch.arange(n_prompts, device=scores.device)
        best = scores.argmax(dim=1)
        logits, scores = logits[index, best], scores[index, best]

        mask_threshold = getattr(self._predictor, "mask_threshold", 0.0)
        stability = calculate_stability_score(logits, mask_threshold, STABILITY_SCORE_OFFSET)
        masks = (logits > mask_threshold).cpu().numpy()
        combined = (scores.float() * stability.float()).cpu().numpy()
        return [(mask, float(score)) for mask, score in zip(masks, combined)]

    def _apply_prompts(
        self, prompts, multimasking: bool, batch_size: int, multimask_scorer: str = "predicted_iou",
        multimask_selection: str = "eager", compute_multimask_uncertainty: bool = False,
        return_multimask_features: bool = False,
        multimask_feature_schema: Optional[str] = None,
        foreground: Optional[np.ndarray] = None,
        foreground_threshold: float = DEFAULT_PROMPT_GENERATION["foreground_threshold"],
    ) -> List[Dict[str, Any]]:
        """Prompt in batches and return eager records or grouped multimask alternatives."""
        points, point_labels = prompts["points"], prompts["point_labels"]
        mask_threshold = getattr(self._predictor, "mask_threshold", 0.0)
        if multimask_feature_schema is None:
            multimask_feature_schema = (
                selector_input_schema(self._microscopy_multimask_scorer)
                if multimask_scorer == "microscopy" else "dense_v1"
            )
        compact_features = multimask_feature_schema != "dense_v1"
        if compact_features and not multimasking:
            raise ValueError("Low-resolution and mask-token selector schemas require multimasking=True.")
        advanced = bool(
            multimask_scorer != "predicted_iou"
            or multimask_selection == "deferred"
            or compute_multimask_uncertainty
            or return_multimask_features
            or compact_features
        )
        if advanced and foreground is None:
            raise ValueError("Multimask feature scoring requires the APG foreground prediction.")
        feature_foreground = feature_context_points = None
        lowres_foreground = lowres_context_points = None
        if advanced and not compact_features:
            feature_foreground = torch.as_tensor(
                foreground, dtype=torch.float32, device=self._predictor.device,
            )
            feature_context_points = torch.as_tensor(
                points[:, 0], dtype=torch.float32, device=self._predictor.device,
            )

        records = []
        feature_seconds = scorer_seconds = transfer_seconds = record_seconds = 0.0
        alternatives_returned = 0
        changed_from_iou = torch.zeros((), dtype=torch.int64, device=self._predictor.device)
        for start in range(0, len(points), batch_size):
            stop = start + batch_size
            batch_points, batch_labels = points[start:stop], point_labels[start:stop]
            n_prompts = len(batch_points)
            # Reduced on the device, so only the kept mask is transferred rather than every proposal.
            mask_input, coords, labels, _ = self._predictor._prep_prompts(
                batch_points, batch_labels, None, None, True,
            )
            with autocast(self._predictor.device):
                if compact_features:
                    lowres_logits, scores, mask_tokens = _predict_three_lowres(
                        self._predictor, coords, labels, None, mask_input,
                    )
                    logits = None
                else:
                    logits, scores, _ = self._predictor._predict(
                        coords, labels, None, mask_input, multimasking, return_logits=True,
                    )
                    logits = logits.reshape(n_prompts, -1, *logits.shape[-2:])
                    lowres_logits = mask_tokens = None
            scores = scores.reshape(n_prompts, -1)
            if not advanced:
                # Preserve the historical fast path exactly: select on the device, then transfer
                # only the kept mask and calculate its stability.
                index = torch.arange(n_prompts, device=scores.device)
                best = scores.argmax(dim=1)
                selected_logits, selected_scores = logits[index, best], scores[index, best]
                stability = calculate_stability_score(
                    selected_logits, mask_threshold, STABILITY_SCORE_OFFSET
                )
                binary = selected_logits > mask_threshold
                selection_scores = None
                selected = np.zeros(n_prompts, dtype="int64")
                alternative_indices = np.asarray(best.cpu(), dtype="int64")
                gate_scores = None
            else:
                source_logits = lowres_logits if compact_features else logits
                n_alternatives = source_logits.shape[1]
                stability = calculate_stability_score(
                    source_logits.reshape(n_prompts * n_alternatives, *source_logits.shape[-2:]),
                    mask_threshold, STABILITY_SCORE_OFFSET,
                ).reshape(n_prompts, n_alternatives)
                feature_binary = source_logits > mask_threshold
                cuda_timing = scores.device.type == "cuda"
                if cuda_timing:
                    feature_started, feature_finished = torch.cuda.Event(True), torch.cuda.Event(True)
                    feature_started.record()
                else:
                    feature_started = time.perf_counter()
                prompt_indices = torch.arange(start, start + n_prompts, device=scores.device)
                if compact_features:
                    if lowres_foreground is None:
                        lowres_foreground, lowres_context_points = _lowres_feature_context(
                            self._predictor, foreground, points[:, 0], source_logits.shape[-2:], scores.device,
                        )
                    lowres_mask_features = extract_multimask_features_torch(
                        feature_binary, scores, stability, lowres_context_points[start:stop],
                        lowres_foreground, foreground_threshold,
                        context_points=lowres_context_points, prompt_indices=prompt_indices,
                    )
                    features_tensor = combine_selector_features_torch(
                        multimask_feature_schema, lowres_mask_features, scores, mask_tokens,
                    )
                    gate_base_features = lowres_mask_features
                else:
                    features_tensor = extract_multimask_features_torch(
                        feature_binary, scores, stability, batch_points[:, 0], feature_foreground,
                        foreground_threshold, context_points=feature_context_points,
                        prompt_indices=prompt_indices,
                    )
                    gate_base_features = features_tensor
                if cuda_timing:
                    feature_finished.record()
                    scorer_started, scorer_finished = torch.cuda.Event(True), torch.cuda.Event(True)
                    scorer_started.record()
                else:
                    feature_seconds += time.perf_counter() - feature_started
                    scorer_started = time.perf_counter()
                if multimask_scorer == "predicted_iou":
                    selection_scores_tensor = scores.to(torch.float32)
                elif hasattr(self._microscopy_multimask_scorer, "predict_grouped_tensor"):
                    selection_scores_tensor = self._microscopy_multimask_scorer.predict_grouped_tensor(
                        features_tensor,
                    )
                elif hasattr(self._microscopy_multimask_scorer, "predict_tensor"):
                    selection_scores_tensor = self._microscopy_multimask_scorer.predict_tensor(
                        features_tensor.reshape(-1, features_tensor.shape[-1]),
                    ).reshape(n_prompts, n_alternatives)
                else:
                    selection_scores_tensor = torch.as_tensor(
                        np.asarray(self._microscopy_multimask_scorer.predict(
                            features_tensor.cpu().numpy().reshape(-1, features_tensor.shape[-1]),
                        ), dtype="float32").reshape(n_prompts, n_alternatives),
                        dtype=torch.float32,
                        device=scores.device,
                    )
                selected_tensor = selection_scores_tensor.argmax(dim=1)
                raw_best = scores.argmax(dim=1)
                changed_from_iou += torch.count_nonzero(selected_tensor != raw_best)
                if compute_multimask_uncertainty:
                    gate_columns = []
                    for alternative_index in range(n_alternatives):
                        chosen = torch.full(
                            (n_prompts,), alternative_index, dtype=torch.int64, device=scores.device,
                        )
                        gate_features = refinement_gate_features_torch(
                            gate_base_features, selection_scores_tensor, chosen,
                        )
                        if hasattr(self._refinement_gate_model, "predict_tensor"):
                            gate_prediction = self._refinement_gate_model.predict_tensor(gate_features)
                        else:
                            gate_prediction = torch.as_tensor(
                                self._refinement_gate_model.predict(gate_features.cpu().numpy()),
                                dtype=torch.float32, device=scores.device,
                            )
                        gate_columns.append(gate_prediction)
                    gate_scores_tensor = torch.stack(gate_columns, dim=1)
                else:
                    gate_scores_tensor = None
                if cuda_timing:
                    scorer_finished.record()
                else:
                    scorer_seconds += time.perf_counter() - scorer_started

                if multimask_selection == "eager":
                    row_index = torch.arange(n_prompts, device=scores.device)
                    if compact_features:
                        kept_logits = source_logits[row_index, selected_tensor][:, None]
                    else:
                        kept_masks = feature_binary[row_index, selected_tensor][:, None]
                    kept_scores = scores[row_index, selected_tensor][:, None]
                    kept_stability = stability[row_index, selected_tensor][:, None]
                else:
                    if compact_features:
                        kept_logits = source_logits
                    else:
                        kept_masks = feature_binary
                    kept_scores, kept_stability = scores, stability

                if compact_features:
                    kept_masks = self._predictor._transforms.postprocess_masks(
                        kept_logits, self._predictor._orig_hw[-1],
                    ) > mask_threshold

                # The baseline already reduces mask extents on the GPU. Keeping the same strategy
                # here avoids scanning the much larger eager/deferred mask arrays again on CPU.
                rows_any_tensor = kept_masks.any(dim=3)
                columns_any_tensor = kept_masks.any(dim=2)

                transfer_started = time.perf_counter()
                masks_np = kept_masks.cpu().numpy()
                rows_any = rows_any_tensor.cpu().numpy()
                columns_any = columns_any_tensor.cpu().numpy()
                scores_np = kept_scores.float().cpu().numpy()
                stability_np = kept_stability.float().cpu().numpy()
                retain_features = return_multimask_features or multimask_selection == "deferred"
                features = features_tensor.cpu().numpy() if retain_features else None
                selection_scores = selection_scores_tensor.cpu().numpy()
                selected = selected_tensor.cpu().numpy()
                gate_scores = gate_scores_tensor.cpu().numpy() if gate_scores_tensor is not None else None
                transfer_seconds += time.perf_counter() - transfer_started
                if features is not None and not np.isfinite(features).all():
                    raise RuntimeError("The Torch multimask feature extractor produced a non-finite value.")
                if not np.isfinite(selection_scores).all():
                    raise RuntimeError("The multimask scorer produced a non-finite value.")
                if gate_scores is not None and not np.isfinite(gate_scores).all():
                    raise RuntimeError("The refinement gate produced a non-finite value.")
                if cuda_timing:
                    feature_seconds += feature_started.elapsed_time(feature_finished) / 1000.0
                    scorer_seconds += scorer_started.elapsed_time(scorer_finished) / 1000.0
                alternative_indices = (
                    selected if multimask_selection == "eager"
                    else np.arange(n_alternatives, dtype="int64")
                )
                binary = None

            # Two reductions on the device: an np.nonzero per mask costs more than the rest of the loop.
            if not advanced:
                rows_any = binary.any(dim=2).cpu().numpy()[:, None]
                columns_any = binary.any(dim=1).cpu().numpy()[:, None]
                masks_np = binary.cpu().numpy()[:, None]
                scores_np = selected_scores.float().cpu().numpy()[:, None]
                stability_np = stability.float().cpu().numpy()[:, None]
            records_started = time.perf_counter()
            for offset in range(n_prompts):
                choices = range(masks_np.shape[1])
                for local_alternative in choices:
                    mask = masks_np[offset, local_alternative]
                    row_any, column_any = rows_any[offset, local_alternative], columns_any[offset, local_alternative]
                    if not row_any.any():
                        continue
                    y0, y1 = int(row_any.argmax()), len(row_any) - int(row_any[::-1].argmax())
                    x0, x1 = int(column_any.argmax()), len(column_any) - int(column_any[::-1].argmax())
                    alternative_index = int(
                        alternative_indices[offset] if np.ndim(alternative_indices) else alternative_indices
                    ) if masks_np.shape[1] == 1 else int(alternative_indices[local_alternative])
                    if advanced:
                        raw_score = float(scores_np[offset, local_alternative])
                        stable = float(stability_np[offset, local_alternative])
                        selection_score = float(selection_scores[offset, alternative_index])
                    else:
                        raw_score = float(scores_np[offset, 0])
                        stable = float(stability_np[offset, 0])
                        selection_score = raw_score
                    record = {
                        "segmentation": mask[y0:y1, x0:x1].copy(),
                        "bounding_box": (slice(y0, y1), slice(x0, x1)),
                        "predicted_iou": raw_score,
                        "stability_score": stable,
                        "prompt_index": start + offset,
                        "point": (float(batch_points[offset, 0, 0]), float(batch_points[offset, 0, 1])),
                        "foreground_threshold": float(foreground_threshold),
                        "multimask_index": alternative_index,
                        "selection_score": selection_score,
                        "merge_score": (
                            selection_score if multimask_scorer == "microscopy" else raw_score * stable
                        ),
                    }
                    if return_multimask_features or multimask_selection == "deferred":
                        record["multimask_features"] = features[offset, alternative_index].copy()
                    if multimask_selection == "deferred" and masks_np.shape[1] > 1:
                        record["multimask_group"] = start + offset
                    if gate_scores is not None:
                        record["uncertainty_score"] = float(gate_scores[offset, alternative_index])
                    records.append(record)
                    alternatives_returned += 1
            if advanced:
                record_seconds += time.perf_counter() - records_started
        if advanced:
            self._last_generation_stats.update({
                "multimask_alternatives": alternatives_returned,
                "multimask_changed_from_iou": int(changed_from_iou.cpu()),
                "multimask_feature_schema": multimask_feature_schema,
                "multimask_feature_seconds": feature_seconds,
                "multimask_scorer_seconds": scorer_seconds,
                "multimask_transfer_seconds": transfer_seconds,
                "multimask_record_seconds": record_seconds,
            })
        return records

    def _score_candidates(
        self, prompts: dict, multimasking: bool, batch_size: int, score_threshold: float,
        max_overlap: float, components: Optional[tuple] = None,
        refinement_kwargs: Optional[dict] = None,
    ) -> List[dict]:
        """Prompt every candidate in 2d on its anchor slice, and keep the strong, non-duplicate ones.

        This runs before the propagation, which is where the cost is: a candidate the model scores
        poorly, or that a better-scoring one already covers on that slice, is never propagated. It
        also gives every surviving candidate the predicted IoU that orders the volumetric merge.

        A refinement runs here too, slice by slice, while the predictor is still pointed at that
        slice: re-reading a slice's features costs more than the re-prompt itself does.

        Args:
            prompts: The volumetric prompts, as `derive_volume_prompts` returns them.
            multimasking: Whether to predict several masks per point and keep the best scoring one.
            batch_size: Number of prompts per forward pass.
            score_threshold: Discard candidates whose predicted IoU is below this.
            max_overlap: Reject a candidate when more than this fraction of it is already claimed.
            components: The refinement components, or None to run no second round.
            refinement_kwargs: The resolved refinement keyword arguments.

        Returns:
            The surviving candidates, each with the prompt it will be propagated with.
        """
        points, point_labels, frames = prompts["points"], prompts["point_labels"], prompts["frames"]
        slice_shape = self._prediction[0].shape[-2:]
        min_size = default_prompt_generation(self._model_type, is_volume=False)["min_size"]
        candidates = []
        for frame in np.unique(frames):
            indices = np.where(frames == frame)[0]
            # Reads the slice's features out of the volume's embeddings, so nothing is re-encoded.
            _set_image_predictor_from_3d_embeddings(self._predictor, self._image_embeddings, int(frame))
            records = self._apply_prompts(
                {"points": points[indices], "point_labels": point_labels[indices]},
                multimasking=multimasking, batch_size=batch_size,
            )
            records = [record for record in records if record["predicted_iou"] >= score_threshold]
            if not records:
                continue

            if components is None:
                _, kept = merge_by_score(
                    records, _records_shape(records), max_overlap=max_overlap,
                    min_size=min_size, return_matches=True,
                )
                candidates.extend(
                    self._anchor_candidate(int(frame), records[record_index])
                    for record_index in kept.values()
                )
                continue

            # On the slice's own canvas rather than the records' bounding one, because a refinement
            # looks instance ids up by position: for the foreign-overlap gate and for the negatives.
            segmentation, matches = merge_by_score(
                records, slice_shape, max_overlap=max_overlap,
                min_size=min_size,
                return_matches=True,
            )
            context = {
                "frame": int(frame), "segmentation": segmentation, "records": records,
                "matches": matches, "points": points[indices][:, 0, :],
            }
            candidates.extend(self._refine_anchors(context, components, refinement_kwargs, batch_size))
        return candidates

    def _anchor_candidate(self, frame: int, record: dict) -> dict:
        """One propagation candidate: the prompt that made the anchor mask, and the merge's score."""
        return {
            "frame": int(frame),
            "point": record["point"],
            "score": record["predicted_iou"],
            "stability": record["stability_score"],
        }

    def _refine_anchors(
        self, context: dict, components: tuple, refinement_kwargs: dict, batch_size: int,
    ) -> List[dict]:
        """Re-prompt every scored candidate of one anchor slice, and return what to propagate.

        The volumetric counterpart of `_reprompt_instances`, and the only place a volume can refine
        at all: a prompt on an already propagated slice would turn it into a conditioning frame and
        replace the mask there with a single-point one. So the second round runs on the anchor slice,
        before the propagation, and what it produces is not a mask - the propagation still makes that
        - but the conditioning the propagation starts from and the score that orders the 3d merge.

        The prompts, the batched forward and both gates are the 2d ones, applied in-plane on the
        anchor slice: `derive_refinement_prompts` groups that slice's suppressed prompts onto their
        candidate and takes the neighbouring ones as negatives, 'min_consistency' keeps a re-prompt
        from reshaping the anchor rather than polishing it, and 'max_foreign_overlap' keeps it out of
        a neighbour. A candidate whose re-prompt comes back empty or gated keeps its first round.

        Returns:
            The slice's candidates in first-round score order. An accepted re-prompt carries its
            conditioning under 'conditioning'; a rejected one has no such key and is propagated from
            its first-round point, exactly as without a refinement.
        """
        segmentation, records = context["segmentation"], context["records"]
        frame, frame_points = context["frame"], context["points"]
        candidates = {
            instance_id: self._anchor_candidate(frame, records[record_index])
            for instance_id, record_index in context["matches"].items()
        }

        point_prompts = None
        if "points" in components:
            point_prompts = derive_refinement_prompts(
                segmentation, frame_points,
                {instance_id: candidate["point"] for instance_id, candidate in candidates.items()},
                n_positives=refinement_kwargs["n_positives"],
                n_negatives=refinement_kwargs["n_negatives"],
                max_negative_distance=refinement_kwargs["max_negative_distance"],
                negative_source=refinement_kwargs["negative_source"],
                min_negative_distance=refinement_kwargs["min_negative_distance"],
            )

        boxes = {
            index + 1: bounding_box
            for index, bounding_box in enumerate(find_objects(segmentation))
            if bounding_box is not None
        }
        instances = [(instance_id, boxes[instance_id]) for instance_id in candidates]
        min_consistency = refinement_kwargs["min_consistency"]
        max_foreign_overlap = refinement_kwargs["max_foreign_overlap"]
        keep_if_better = refinement_kwargs["policy"] == "keep-if-better"
        stats = self._last_generation_stats
        stats["refined_candidates"] += len(instances)
        if point_prompts is not None:
            # A volume can only take negatives from the candidates anchored on this same slice, so it
            # reaches 'n_negatives' far less often than an image does - which is what would explain a
            # flat response to that parameter. Reported, rather than assumed either way.
            stats["refinement_negatives"] += sum(
                int(np.count_nonzero(point_prompts[instance_id]["point_labels"] == 0))
                for instance_id, _ in instances
            )

        for start in range(0, len(instances), batch_size):
            batch = instances[start:start + batch_size]
            predictions = self._predict_refinement_batch(
                segmentation, batch, components, point_prompts, refinement_kwargs,
            )
            for (instance_id, bounding_box), (mask, score) in zip(batch, predictions):
                candidate = candidates[instance_id]
                first_round_score = candidate["score"] * candidate["stability"]
                take_second = mask.any() and (not keep_if_better or score > first_round_score)
                if take_second and min_consistency is not None:
                    first_round_mask = segmentation == instance_id
                    union = int(np.count_nonzero(mask | first_round_mask))
                    iou = int(np.count_nonzero(mask & first_round_mask)) / union if union else 0.0
                    if iou < min_consistency:
                        take_second = False
                        stats["gated_consistency"] += 1
                if take_second and max_foreign_overlap is not None:
                    on_mask = segmentation[mask]
                    foreign = int(np.count_nonzero((on_mask != 0) & (on_mask != instance_id)))
                    if foreign / int(mask.sum()) > max_foreign_overlap:
                        take_second = False
                        stats["gated_foreign"] += 1
                if not take_second:
                    continue
                stats["replaced_candidates"] += 1
                # The merge only reads the product of the two, and the second round reports it as one
                # combined score, so it goes in whole rather than being split back up arbitrarily.
                candidate["score"], candidate["stability"] = float(score), 1.0
                candidate["conditioning"] = self._anchor_conditioning(
                    mask, bounding_box, segmentation.shape, components,
                    point_prompts, instance_id, refinement_kwargs,
                )
        return list(candidates.values())

    def _anchor_conditioning(
        self, mask: np.ndarray, bounding_box: tuple, shape: tuple, components: tuple,
        point_prompts: Optional[dict], instance_id: int, refinement_kwargs: dict,
    ) -> dict:
        """What the propagation conditions the anchor frame on, once a second round is accepted.

        'conditioning="mask"' hands the video predictor the refined mask, so the anchor frame is
        exactly the mask the gates accepted, but its decoder never sees the prompt that produced it.
        'prompts' pushes that prompt instead - the same box and points the second round used - and
        lets the decoder rebuild the mask, which is what the interactive branch does everywhere else.
        """
        if refinement_kwargs["conditioning"] == "mask":
            return {"mask": mask}
        conditioning = {"mode": refinement_kwargs["conditioning"]}
        if "boxes" in components:
            conditioning["box"] = _prompt_box(bounding_box, shape, refinement_kwargs["box_extension"])
        if "points" in components:
            conditioning["points"] = point_prompts[instance_id]["points"]
            conditioning["point_labels"] = point_prompts[instance_id]["point_labels"]
        return conditioning

    def _propagate_candidates(
        self, candidates: List[dict], n_objects_per_pass: int, early_stop_patience: Optional[int],
        verbose: bool,
    ) -> List[Dict[str, Any]]:
        """Turn every candidate into a volumetric mask by propagating its prompts through the volume.

        Every object of a pass is anchored on the same slice, because the video predictor propagates
        them together from the earliest slice any of them is conditioned on: an object anchored later
        than that would be tracked before it is conditioned, and never propagated backwards at all.
        """
        by_anchor = {}
        for candidate in candidates:
            by_anchor.setdefault(candidate["frame"], []).append(candidate)
        passes = [
            group[start:start + n_objects_per_pass]
            for _, group in sorted(by_anchor.items())
            for start in range(0, len(group), n_objects_per_pass)
        ]
        self._last_generation_stats.update({
            "unique_anchor_slices": len(by_anchor),
            "propagation_passes": len(passes),
        })

        records = []
        propagated_frame_steps = 0
        for batch in tqdm(passes, desc="Propagate prompts", disable=not verbose):
            # Only the objects: re-reading the volume's features each pass costs more than the propagation.
            self._propagator.reset_tracking()
            for object_id, candidate in enumerate(batch, start=1):
                self._condition_pass(candidate, object_id)
            video_segments = self._propagator.propagate_prompts(early_stop_patience=early_stop_patience)
            propagated_frame_steps += len(video_segments)
            records.extend(_volume_records(video_segments, batch, self._volume.shape))

        possible_frame_steps = len(passes) * int(self._volume.shape[0])
        self._last_generation_stats.update({
            "propagated_frame_steps": propagated_frame_steps,
            "early_stopped_frame_steps": possible_frame_steps - propagated_frame_steps,
        })
        self._propagator.reset_predictor()
        return records

    def _condition_pass(self, candidate: dict, object_id: int) -> None:
        """Push one candidate's conditioning onto the propagator, as the object with this id.

        Without a refinement a candidate is conditioned on the single point the decoder proposed it
        at. A refined one carries either the mask its second round produced, which conditions the
        anchor frame directly, or that round's box and points - in as many pushes as its
        'conditioning' asks for, because a push is a decoder step on the anchor frame rather than
        bookkeeping, and more of them refine the anchor further. See `DEFAULT_REFINEMENT_3D`.
        """
        frame = candidate["frame"]
        conditioning = candidate.get("conditioning")
        if conditioning is None:
            x, y = candidate["point"]
            self._propagator.add_point_prompts(
                frame_ids=frame,
                points=np.array([[y, x]], dtype="float32"),  # The propagator takes YX.
                point_labels=np.array([1], dtype="int32"),
                object_id=object_id,
            )
            return

        mask = conditioning.get("mask")
        if mask is not None:
            # Already refined against this slice, so the propagator must not refine it a second time.
            self._propagator.add_mask_prompts(
                frame_ids=frame, masks=[mask], object_id=object_id, refine=False,
            )
            return

        # Every producer of a conditioning dict sets this; a missing one is a bug in the producer,
        # so it raises rather than silently picking a strategy.
        mode = conditioning["mode"]
        box = conditioning.get("box")
        points = conditioning.get("points")
        # The propagator takes YX for both.
        points_yx = None if points is None else np.asarray(points, dtype="float32")[:, ::-1]
        labels = conditioning.get("point_labels")
        box_yx = None if box is None else np.array([box[1], box[0], box[3], box[2]], dtype="float32")

        if mode == "prompts-joint":
            self._propagator.add_prompt_set(
                frame_id=frame, points=points_yx, point_labels=labels, box=box_yx,
                object_id=object_id,
            )
            return

        # A box has to come first, and it clears whatever the object carried on the frame.
        if box_yx is not None:
            self._propagator.add_box_prompts(
                frame_ids=frame, boxes=[box_yx], object_id=object_id,
            )
        if points_yx is None or not len(points_yx):
            return
        if mode == "prompts-grouped":
            self._propagator.add_prompt_set(
                frame_id=frame, points=points_yx, point_labels=labels, object_id=object_id,
                # Appends to the box rather than replacing it.
                clear_old_points=box_yx is None,
            )
            return
        self._propagator.add_point_prompts(
            frame_ids=frame, points=points_yx,
            point_labels=np.asarray(labels, dtype="int32"), object_id=object_id,
        )


class TiledAutomaticPromptGenerator(AutomaticPromptGenerator, TiledUniSAM2InstanceSegmentation):
    """Generates an instance segmentation with automatically generated prompts, for tiled inference.

    Like `AutomaticPromptGenerator`, but both branches run tile by tile, which keeps the encoder at its
    native resolution instead of downscaling the whole image to its input size.

    The prompts are derived once from the stitched prediction, so a candidate spanning a tile border is
    proposed once. Each is assigned to the tile whose inner block holds its point and prompted within
    that tile's halo, so no object is segmented twice and no mask is cut off at a nearby border.

    A refinement runs the same way: every instance is re-prompted in the tile that produced it, with
    its prompts translated into that tile's frame, while the acceptance gates and the final repaint
    stay global. The negatives an instance takes from its neighbours are chosen across the whole
    image, so a neighbour beyond the halo is dropped from the re-prompt and counted in the
    'dropped_negatives' statistic — a large count means the halo is too small for 'n_negatives'.

    Args:
        model: The UniSAM2 model (see `get_unisam2_model` / `get_decoder`).
        predictor: The SAM2 image predictor for the interactive branch of the same model.
        device: The device the model lives on.
        inference_device: The device intent used as the `devices=None` fallback.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        predictor,
        device: Optional[Union[str, torch.device]] = None,
        inference_device: Devices = USE_MODEL_DEVICE,
    ) -> None:
        super().__init__(model, predictor, device=device, inference_device=inference_device)
        self._tiling = None
        self._halo = None

    def initialize(
        self,
        image: np.ndarray,
        ndim: int = 2,
        image_embeddings: Optional[dict] = None,
        i: Optional[int] = None,
        tile_shape: Optional[tuple] = None,
        halo: Optional[tuple] = None,
        save_path: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Compute the tiled embeddings, run the decoder on them and keep them for the prompting.

        The same tiled embeddings serve both branches. Unlike the non-tiled generator they are needed
        again in `generate`, so they are held until `clear_state`.

        Args:
            image: The input image, shape (Y, X) or (Y, X, C).
            ndim: The number of spatial dimensions. Must be 2.
            image_embeddings: Optional precomputed tiled image embeddings. The tiling is taken from
                them when they are given.
            i: The slice index for tiled video-style embeddings. By default the embeddings contain one image.
            tile_shape: The tile shape, (y, x). Required when no embeddings are given.
            halo: The overlap between the tiles, (y, x). Required when no embeddings are given.
            save_path: Optional path to cache the computed embeddings in a zarr container. Without one
                an ephemeral store is used, which `clear_state` removes.
            verbose: Whether to print progress while the embeddings are computed.
            kwargs: Additional arguments for `TiledUniSAM2InstanceSegmentation.initialize`.
        """
        if ndim != 2:
            raise ValueError(f"Tiled prompt generation supports 2d images only, got ndim={ndim}.")
        if image_embeddings is None and (tile_shape is None or halo is None):
            raise ValueError("Both 'tile_shape' and 'halo' have to be passed for the tiled generator.")

        if self._temporary_embedding_path is not None:
            self.clear_state()

        owns_image_embeddings = image_embeddings is None
        if image_embeddings is None:
            path = save_path
            if path is None:
                self._temporary_embedding_path = make_temp_embedding_path()
                path = self._temporary_embedding_path
            image_embeddings = precompute_image_embeddings(
                self._predictor, image, save_path=path, ndim=2, tile_shape=tile_shape, halo=halo,
                verbose=verbose, lazy_loading=True,
            )

        TiledUniSAM2InstanceSegmentation.initialize(
            self, image, ndim=2, image_embeddings=image_embeddings, i=i, **kwargs
        )
        self._image_embeddings = image_embeddings
        self._i = i
        self._owns_image_embeddings = owns_image_embeddings
        self._set_tiling(image_embeddings)

    def _set_tiling(self, image_embeddings: dict) -> None:
        # From the embeddings, not the arguments, so the prompting cannot disagree with the encoding.
        features = image_embeddings["features"]
        self._tiling = Blocking(
            [0, 0], [int(s) for s in features.attrs["shape"][-2:]],
            [int(s) for s in features.attrs["tile_shape"]],
        )
        self._halo = [int(s) for s in features.attrs["halo"]]

    def _set_tile_embeddings(self, tile_id: int) -> None:
        """Set the embeddings for one tile on the image predictor."""
        if self._i is None:
            set_precomputed(self._predictor, self._image_embeddings, tile_id=tile_id)
            return

        name = str(tile_id)
        features = self._image_embeddings["features"][name]
        fpn_group = self._image_embeddings["fpn"][name]
        pos_enc_group = self._image_embeddings["pos_enc"][name]
        fpn = [fpn_group[str(level)] for level in range(len(fpn_group))]
        pos_enc = [pos_enc_group[str(level)] for level in range(len(pos_enc_group))]
        _set_image_predictor_from_backbone(
            self._predictor, fpn, pos_enc, features, features.attrs["original_size"], self._i,
        )

    def _tile_bounding_box(self, tile_id: int) -> tuple:
        """The outer (halo-extended) block of a tile, as a slice tuple."""
        block = self._tiling.get_block_with_halo(tile_id, list(self._halo)).outer_block
        return tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))

    def _tiles_for_points(self, points: np.ndarray) -> Dict[int, List[int]]:
        """Group prompt indices by the tile whose inner block holds their point.

        The inner blocks do not overlap, so every candidate is prompted exactly once.
        """
        assignment = {}
        for index, (x, y) in enumerate(points[:, 0, :]):
            tile_id = self._tiling.coordinates_to_block_id([int(y), int(x)])
            assignment.setdefault(tile_id, []).append(index)
        return assignment

    def _apply(
        self, prompts: dict, multimasking: bool, batch_size: int, multimask_scorer: str = "predicted_iou",
        multimask_selection: str = "eager", compute_multimask_uncertainty: bool = False,
        return_multimask_features: bool = False,
        multimask_feature_schema: Optional[str] = None,
        foreground_threshold: float = DEFAULT_PROMPT_GENERATION["foreground_threshold"],
    ) -> list:
        """Prompt each tile with the candidates that belong to it, keeping the tiles apart."""
        points, point_labels = prompts["points"], prompts["point_labels"]

        proposals = []
        for tile_id, indices in sorted(self._tiles_for_points(points).items()):
            bounding_box = self._tile_bounding_box(tile_id)
            # The prompts are in the full image's frame, the tile's embeddings in the tile's.
            origin = np.array([bounding_box[1].start, bounding_box[0].start], dtype="float32")

            self._set_tile_embeddings(tile_id)
            local_prompts = {"points": points[indices] - origin, "point_labels": point_labels[indices]}
            kwargs = {"multimasking": multimasking, "batch_size": batch_size}
            if (
                multimask_scorer != "predicted_iou"
                or multimask_selection != "eager"
                or compute_multimask_uncertainty
                or return_multimask_features
                or multimask_feature_schema is not None
            ):
                kwargs.update({
                    "multimask_scorer": multimask_scorer, "multimask_selection": multimask_selection,
                    "compute_multimask_uncertainty": compute_multimask_uncertainty,
                    "return_multimask_features": return_multimask_features,
                    "multimask_feature_schema": multimask_feature_schema,
                    "foreground": self._prediction[0][bounding_box],
                    "foreground_threshold": foreground_threshold,
                })
            records = self._apply_prompts(local_prompts, **kwargs)
            for record in records:
                # Back into the full image's frame, so the records agree with the non-tiled ones.
                record["point"] = (record["point"][0] + float(origin[0]), record["point"][1] + float(origin[1]))
                if "multimask_group" in record:
                    record["multimask_group"] = (tile_id, record["multimask_group"])
            if records:
                proposals.append({"tile_id": tile_id, "bounding_box": bounding_box, "records": records})
        return proposals

    def _merge(
        self, proposals: list, shape: tuple, score_threshold: float, max_overlap: float, min_size: int,
        return_context: bool = False, score_filter: str = "predicted_iou",
    ) -> tuple:
        """Stitch the per-tile merges into one segmentation, resolving the halo overlaps.

        Each tile is merged on its own and its instance ids are offset to stay unique across the
        image, so the stitch only decides which tile owns a contested pixel — it never renames an
        id. The refinement context therefore carries across it: the per-tile matches are shifted by
        the same offset as the ids, and the instances a neighbouring tile overwrote entirely are
        pruned, since an id that is no longer in the segmentation has nothing left to refine.

        Post-merge refinement features derive visibility loss from the stitched segmentation, so the
        context does not need per-tile claim maps.
        """
        segmentation = np.zeros(shape, dtype="uint32")
        offset = 0

        all_records, records, reasons, matches, record_tiles = [], [], [], {}, {}
        for proposal in proposals:
            bounding_box = proposal["bounding_box"]
            tile_shape = tuple(box.stop - box.start for box in bounding_box)
            # Flattened before the tile can be skipped below, so the record indices cannot shear.
            all_records.extend(proposal["records"])
            record_offset = len(records)
            if score_filter == "none":
                tile_records = list(proposal["records"])
            else:
                missing = [record for record in proposal["records"] if score_filter not in record]
                if missing:
                    raise ValueError(
                        f"Cannot filter by {score_filter!r}: {len(missing)} tile records lack that score."
                    )
                tile_records = [
                    record for record in proposal["records"] if record[score_filter] >= score_threshold
                ]
            records.extend(tile_records)
            if return_context:
                record_tiles.update({
                    record_offset + index: proposal["tile_id"] for index in range(len(tile_records))
                })
            if not tile_records:
                continue

            if return_context:
                tile_segmentation, tile_matches, tile_reasons = merge_by_score(
                    tile_records, tile_shape, max_overlap=max_overlap, min_size=min_size,
                    return_matches=True, return_reasons=True,
                )
                reasons.extend(tile_reasons)
            else:
                tile_segmentation = merge_by_score(
                    tile_records, tile_shape, max_overlap=max_overlap, min_size=min_size
                )
            max_id = int(tile_segmentation.max())
            if max_id == 0:
                continue
            if return_context:
                matches.update({
                    instance_id + offset: record_index + record_offset
                    for instance_id, record_index in tile_matches.items()
                })
            # Keep the instance ids unique across tiles before the halo overlaps are resolved.
            tile_segmentation[tile_segmentation != 0] += offset
            offset += max_id
            # An earlier tile keeps every pixel it claimed, which is the halo resolution.
            previous = segmentation[bounding_box]
            segmentation[bounding_box] = np.where(previous != 0, previous, tile_segmentation)

        if not return_context:
            return segmentation, None

        present = {int(instance_id) for instance_id in np.unique(segmentation)} - {0}
        stitch_dropped = len(matches) - len(present)
        matches = {
            instance_id: record_index for instance_id, record_index in matches.items()
            if instance_id in present
        }
        if set(matches) != present:
            raise RuntimeError(
                f"The stitched segmentation has {len(present - set(matches))} instances that no tile "
                "merge accounts for, so the refinement context would be incomplete."
            )
        self._last_generation_stats.update({
            "proposed_candidates": len(all_records),
            "scored_candidates": len(records),
            "merge_reasons": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
            "stitch_dropped_instances": stitch_dropped,
        })
        return segmentation, {
            "proposals": all_records, "records": records, "matches": matches,
            "record_tiles": record_tiles, "score_threshold": score_threshold,
            "score_filter": score_filter,
        }

    def _region_of(self, context: dict, record_index: int):
        """The tile that produced a record, which is the tile its instance is re-prompted in.

        That tile's embeddings made the first-round mask, so the mask, its bounding box and every
        prompt grouped onto it lie inside the tile's halo-extended block, and the stitch can only
        take pixels away. Assigning by the instance's interior point instead would carry no such
        guarantee, and would truncate an instance that the point's tile does not fully cover.
        """
        return context["record_tiles"][record_index]

    def _region_box(self, key) -> tuple:
        """@private"""
        return self._tile_bounding_box(key)

    def _set_region(self, key) -> None:
        """@private"""
        set_precomputed(self._predictor, self._image_embeddings, tile_id=key)

    def get_state(self) -> dict:
        """@private"""
        raise NotImplementedError(
            "The tiled prompt generator cannot serialize its state, because it holds tiled embeddings."
        )

    def set_state(self, state: dict) -> None:
        """Restore a stitched decoder prediction and the tiled embeddings used for prompting."""
        image_embeddings = state.get("image_embeddings")
        if image_embeddings is None:
            raise ValueError("A tiled prompt-generator state must hold its 'image_embeddings'.")

        TiledUniSAM2InstanceSegmentation.set_state(self, state)
        self._image_embeddings = image_embeddings
        self._i = state.get("i")
        self._owns_image_embeddings = False
        self._set_tiling(image_embeddings)

    def clear_state(self) -> None:
        """Clear the decoder predictions and the tiled embeddings, removing an ephemeral store."""
        super().clear_state()
        self._tiling = None
        self._halo = None
