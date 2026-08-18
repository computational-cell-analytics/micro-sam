"""Time the stages of automatic prompt generation, to see where its cost actually is.

Splits a run into the stages that can be optimised independently: encoding the input, running the
decoder on that encoding, deriving the candidates from the decoder output (CPU flow integration),
prompting the interactive branch with them, and merging the resulting masks. Reports the mean of
several samples, since the first one pays for CUDA warm-up and for loading the data.

Usage:
    python profile_apg.py -d livecell -m hvit_t -n 5
    python profile_apg.py -d gonuclear -m hvit_t -n 2 --ndim 3
"""

import time
import hashlib
import argparse
from contextlib import contextmanager

import numpy as np
import pandas as pd

import torch

from micro_sam.v2.automatic_prompt_generation import STABILITY_SCORE_OFFSET

from common import DATA_ROOT, DATASET_SPACING, export_joint_checkpoint
from baselines_common import _load_data


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Accumulates wall time per named stage."""

    def __init__(self):
        self.totals = {}

    @contextmanager
    def stage(self, name):
        synchronize()
        start = time.perf_counter()
        yield
        synchronize()
        self.totals.setdefault(name, 0.0)
        self.totals[name] += time.perf_counter() - start


def build_segmenter(model_type, ndim, device):
    """Build the prompt generator the same way the evaluation does."""
    from micro_sam.v2.util import get_sam2_model
    from micro_sam.v2.instance_segmentation import get_unisam2_model, get_instance_segmentation_generator

    interactive_path, decoder_path = export_joint_checkpoint(model_type, "best")
    decoder = get_unisam2_model(decoder_path, device=device, encoder=model_type)
    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=interactive_path,
        **({"input_type": "videos"} if ndim == 3 else {}),
    )
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=ndim,
    )


def segmentation_fingerprint(segmentation):
    """A hash of the object sizes, which ignores how the instances happen to be numbered."""
    ids, sizes = np.unique(segmentation, return_counts=True)
    return hashlib.sha256(np.sort(sizes[ids != 0]).tobytes()).hexdigest()[:12]


def profile_prompt_stage(segmenter, prompts, timer, batch_size=64):
    """Break the prompting stage down, to see what inside it costs what.

    Mirrors `AutomaticPromptGenerator._apply_prompts` step by step. Kept separate from it so that the
    timed code stays readable there.
    """
    from sam2.utils.amg import calculate_stability_score

    points, point_labels = prompts["points"], prompts["point_labels"]
    predictor = segmenter._predictor
    mask_threshold = getattr(predictor, "mask_threshold", 0.0)

    for start in range(0, len(points), batch_size):
        batch_points, batch_labels = points[start:start + batch_size], point_labels[start:start + batch_size]
        n_prompts = len(batch_points)
        with timer.stage("prep_prompts"):
            mask_input, coords, labels, _ = predictor._prep_prompts(batch_points, batch_labels, None, None, True)
        with timer.stage("model_forward"):
            logits, scores, low_res = predictor._predict(
                coords, labels, None, mask_input, True, return_logits=True,
            )
        with timer.stage("reduce_on_device"):
            logits = logits.reshape(n_prompts, -1, *logits.shape[-2:])
            scores = scores.reshape(n_prompts, -1)
            index = torch.arange(n_prompts, device=scores.device)
            best = scores.argmax(dim=1)
            logits, scores = logits[index, best], scores[index, best]
            stability = calculate_stability_score(logits, mask_threshold, STABILITY_SCORE_OFFSET)
            binary = logits > mask_threshold
        with timer.stage("transfer"):
            masks = binary.cpu().numpy()
            scores = scores.float().cpu().numpy()
            stability = stability.float().cpu().numpy()
        with timer.stage("bounding_boxes"):
            for mask in masks:
                rows, columns = np.nonzero(mask)
                if len(rows):
                    _ = mask[int(rows.min()):int(rows.max()) + 1, int(columns.min()):int(columns.max()) + 1].copy()
        # The upscale inside the forward pass runs on every multimask proposal, so time one alone.
        with timer.stage("upscale_one_proposal"):
            predictor._transforms.postprocess_masks(low_res[:, :1], predictor._orig_hw[-1])


@contextmanager
def autocast_context(choice, device):
    """Run the SAM2 branch under reduced precision, which is how SAM2's own inference examples run it."""
    if choice == "none" or not str(device).startswith("cuda"):
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=getattr(torch, choice)):
            yield


def profile_2d(segmenter, raw, timer, params):
    """Time an image, splitting initialize into encoder + decoder and generate into its stages."""
    sam2_precision = params["autocast"]
    segmenter.clear_state()
    with timer.stage("encode"):
        with autocast_context(sam2_precision, params["device"]):
            embeddings = segmenter._encode(raw)
    with timer.stage("decoder"):
        segmenter.initialize(raw, ndim=2, image_embeddings=embeddings)

    from micro_sam.v2.automatic_prompt_generation import derive_point_prompts
    with timer.stage("derive_candidates"):
        prompts = derive_point_prompts(
            segmenter._prediction[0], segmenter._prediction[1:],
            candidate_threshold=params["candidate_threshold"],
            foreground_threshold=params["foreground_threshold"],
            n_iter=params["n_iter"], sigma=params["sigma"],
            min_candidate_size=params["min_candidate_size"],
        )
    n_candidates = 0 if prompts is None else len(prompts["points"])

    with timer.stage("prompt"):
        with autocast_context(sam2_precision, params["device"]):
            proposals = [] if prompts is None else segmenter._apply(
                prompts, multimasking=True, batch_size=params["batch_size"]
            )
    if params.get("breakdown") and prompts is not None:
        profile_prompt_stage(segmenter, prompts, timer)
    with timer.stage("merge"):
        segmentation = segmenter.select(
            proposals, score_threshold=params["score_threshold"], max_overlap=params["max_overlap"],
            min_size=params["min_size"], refine_with_box_prompts=False,
        )
    with timer.stage("refine_boxes"):
        if segmentation.max() > 0:
            with autocast_context(sam2_precision, params["device"]):
                segmenter._refine_boxes(segmentation, params["batch_size"], 0)
    return n_candidates, int(segmentation.max()), segmentation_fingerprint(segmentation)


def profile_3d(segmenter, volume, timer, params, spacing):
    """Time a volume, splitting generate into candidates, 2d scoring and propagation."""
    from micro_sam.v2.automatic_prompt_generation import derive_volume_prompts, merge_by_score

    segmenter.clear_state()
    with timer.stage("encode+decoder"):
        segmenter.initialize(volume, ndim=3)

    with timer.stage("derive_candidates"):
        prompts = derive_volume_prompts(
            segmenter._prediction[0], segmenter._prediction[1:],
            candidate_threshold=params["candidate_threshold_3d"],
            foreground_threshold=params["foreground_threshold"],
            n_iter=params["n_iter"], sigma=params["sigma"], spacing=spacing,
            min_candidate_size=params["min_candidate_size"],
        )
    n_candidates = 0 if prompts is None else len(prompts["points"])
    if prompts is None:
        return 0, 0, "empty"

    with timer.stage("score_candidates"):
        candidates = segmenter._score_candidates(
            prompts, multimasking=True, batch_size=64,
            score_threshold=params["score_threshold"], max_overlap=params["max_overlap"],
        )
    with timer.stage("propagate"):
        records = segmenter._propagate_candidates(
            candidates, n_objects_per_pass=params["n_objects_per_pass"],
            early_stop_patience=None, verbose=False,
        )
    with timer.stage("merge"):
        segmentation = merge_by_score(
            records, segmenter._prediction[0].shape, max_overlap=params["max_overlap"],
            min_size=params["min_size"],
        )
    return n_candidates, int(segmentation.max()), segmentation_fingerprint(segmentation)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", default="livecell", help="Dataset to profile on.")
    parser.add_argument("-m", "--model_type", default="hvit_t", help="SAM2 backbone of the joint model.")
    parser.add_argument("-n", "--n_samples", type=int, default=5, help="Samples to time.")
    parser.add_argument("--ndim", type=int, default=2, choices=[2, 3], help="Spatial dimensions.")
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    parser.add_argument("--breakdown", action="store_true", help="Also break the 2d prompting stage down.")
    parser.add_argument("--autocast", default="none", choices=["none", "bfloat16", "float16"], help="SAM2 precision.")
    parser.add_argument("--batch_size", type=int, default=64, help="Prompts per forward pass.")
    parser.add_argument("--n_objects_per_pass", type=int, default=None, help="Objects propagated together.")
    args = parser.parse_args()

    from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION
    params = dict(DEFAULT_PROMPT_GENERATION)
    params["breakdown"] = args.breakdown
    params["batch_size"] = args.batch_size
    params["autocast"] = args.autocast
    params["device"] = args.device
    if args.n_objects_per_pass is not None:
        params["n_objects_per_pass"] = args.n_objects_per_pass

    segmenter = build_segmenter(args.model_type, args.ndim, args.device)
    spacing = DATASET_SPACING.get(args.dataset_name)

    timer, warmup, counts = Timer(), Timer(), []
    loader = _load_data(args.dataset_name, DATA_ROOT, ndim=args.ndim)
    for index, (raw, _, _) in enumerate(loader):
        if index > args.n_samples:
            break
        # The first sample pays for CUDA warm-up, so it is timed separately and dropped.
        into = warmup if index == 0 else timer
        if args.ndim == 2:
            counted = profile_2d(segmenter, raw, into, params)
        else:
            with autocast_context(args.autocast, args.device):
                counted = profile_3d(segmenter, raw, into, params, spacing)
        if index > 0:
            counts.append(counted)
        print(f"sample {index}: {counted[0]} candidates -> {counted[1]} instances, fingerprint {counted[2]}")

    n = len(counts)
    if n == 0:
        raise RuntimeError("No samples were timed.")
    total = sum(timer.totals.values())
    rows = [
        {"stage": stage, "sec_per_sample": seconds / n, "percent": 100.0 * seconds / total}
        for stage, seconds in timer.totals.items()
    ]
    table = pd.DataFrame(rows).sort_values("percent", ascending=False)
    print()
    print(f"{args.dataset_name} ({args.model_type}, {args.ndim}d), autocast={args.autocast}, "
          f"batch_size={args.batch_size}, n_objects_per_pass={params['n_objects_per_pass']}, "
          f"mean over {n} samples")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"total: {total / n:.3f} s/sample")
    print(f"candidates: mean {np.mean([c[0] for c in counts]):.0f}")
    print(f"instances: mean {np.mean([c[1] for c in counts]):.0f}")


if __name__ == "__main__":
    main()
