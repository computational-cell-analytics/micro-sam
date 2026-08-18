"""Measure what 2d automatic prompt generation costs per stage, and what a speed-up costs in score.

Every variant runs the very same code path the evaluation runs, and is timed by wrapping the real
functions rather than by mirroring them, so the numbers cannot drift from what the library does. Each
image is scored as well as timed: a variant is only useful if it leaves mSA where the baseline had it,
so the report pairs the two per image and gives the mean delta over the same samples.

Usage:
    python benchmark_apg_2d.py -n 25 --variants baseline threads8
    python benchmark_apg_2d.py -n 25 --variants baseline bfloat16 --save results.csv
"""

import os
import json
import time
import argparse
import functools
from contextlib import contextmanager, nullcontext

import pandas as pd
from tqdm import tqdm

import torch

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.v2 import automatic_prompt_generation as apg
from micro_sam.v1.evaluation.livecell import _get_livecell_paths

from common import DATA_ROOT, drop_excluded_livecell, export_joint_checkpoint
from baselines_common import load_evaluation_sample_2d

# What a variant may change. Only knobs that leave the candidates alone belong here: a variant that
# proposed different ones would be a different method, not a faster one. 'baseline' is the generator
# as it was before this benchmark, so every measured change is paired against the same reference.
VARIANTS = {
    "baseline": {"flow_threads": 1, "interior_points": "v1", "half_precision": False},
    "threads8": {"flow_threads": 8, "interior_points": "v1", "half_precision": False},
    "points": {"flow_threads": 8, "half_precision": False},
    "shipped": {"flow_threads": 8},
    "batch256": {"flow_threads": 8, "batch_size": 256},
    "tf32": {"flow_threads": 8, "tf32": True},
    "bfloat16": {"flow_threads": 8, "half_precision": False, "autocast": "bfloat16"},
    "bfloat16_weights": {"flow_threads": 8, "half_precision": False, "autocast": "bfloat16",
                         "weights": "bfloat16"},
    "float16_weights": {"flow_threads": 8, "weights": "float16"},
}


@contextmanager
def autocast_context(choice, device):
    """Run the SAM2 branch in reduced precision, which is how SAM2's own inference examples run it."""
    if choice is None or not str(device).startswith("cuda"):
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=getattr(torch, choice)):
            yield


@contextmanager
def weight_context(segmenter, dtype):
    """Hold the SAM2 backbone's weights in reduced precision, restoring the float32 ones afterwards.

    Autocast casts what it runs through a matmul or a convolution and leaves everything else in
    float32. The mask decoder is bandwidth-bound rather than compute-bound, so this asks the stronger
    question: what if the whole backbone were half precision?
    """
    if dtype is None:
        yield
        return
    model = segmenter._predictor.model
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    model.to(getattr(torch, dtype))
    try:
        yield
    finally:
        model.to(torch.float32)
        model.load_state_dict(saved)


@contextmanager
def tf32_context(enabled):
    """Allow TF32 matmuls for one variant, restoring the process default afterwards."""
    if enabled is None:
        yield
        return
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class Timer:
    """Accumulates wall time per named stage, synchronizing so device work lands in its own stage."""

    def __init__(self):
        self.totals = {}

    @contextmanager
    def stage(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - start

    def wrap(self, name, function):
        """Return 'function' with its time counted towards 'name'."""
        @functools.wraps(function)
        def timed(*args, **kwargs):
            with self.stage(name):
                return function(*args, **kwargs)
        return timed


@contextmanager
def half_precision_context(enabled):
    """Turn the generator's own half-precision autocast off, so a variant can measure without it."""
    if enabled:
        yield
        return
    original = apg.sam2_autocast
    apg.sam2_autocast = lambda device: nullcontext()
    try:
        yield
    finally:
        apg.sam2_autocast = original


@contextmanager
def instrument(timer, segmenter, flow_threads, points_version):
    """Time the inner stages by wrapping the functions the generator actually calls.

    The wrapping is undone on exit, so one process can time several variants in a row. The flow
    density is given its thread count here, and 'points_version' puts the whole-image transform of
    `micro_sam.v1.instance_segmentation._get_centers` back, which is what the generator used to do.
    """
    from micro_sam.v1.instance_segmentation import _get_centers

    originals = {name: getattr(apg, name) for name in ("_compute_flow_density", "label", "interior_points")}
    encode, apply_prompts = segmenter._encode, segmenter._apply

    def density(*args, **kwargs):
        kwargs["n_threads"] = flow_threads
        with timer.stage("flow_density"):
            return originals["_compute_flow_density"](*args, **kwargs)

    points = _get_centers if points_version == "v1" else originals["interior_points"]
    apg._compute_flow_density = density
    apg.label = timer.wrap("connected_components", originals["label"])
    apg.interior_points = timer.wrap("interior_points", points)
    segmenter._encode = timer.wrap("encode", encode)
    segmenter._apply = timer.wrap("prompt", apply_prompts)
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(apg, name, function)
        segmenter._encode, segmenter._apply = encode, apply_prompts


def build_segmenter(model_type, device, joint_checkpoint="best"):
    """Build the prompt generator exactly the way evaluate_automatic_baselines builds it."""
    from micro_sam.v2.util import get_sam2_model
    from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator

    from common import load_unisam2_model

    interactive_path, decoder_path = export_joint_checkpoint(model_type, joint_checkpoint)
    decoder = load_unisam2_model(decoder_path, device, encoder=model_type)
    model = get_sam2_model(model_type=model_type, device=device, checkpoint_path=interactive_path)
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device,
    )


def load_livecell_subset(n_per_cell_type):
    """Load the stratified livecell test subset, cropped and filtered as the evaluation loads it."""
    image_paths, gt_paths = _get_livecell_paths(
        input_folder=os.path.join(DATA_ROOT, "livecell"), split="test", n_val_per_cell_type=n_per_cell_type,
    )
    image_paths, gt_paths = drop_excluded_livecell(image_paths, gt_paths)
    samples = []
    for image_path, gt_path in tqdm(sorted(zip(image_paths, gt_paths)), desc="Load livecell"):
        image, gt = load_evaluation_sample_2d(image_path, gt_path, None, None, "livecell")
        if gt.max() == 0:
            continue
        samples.append((os.path.basename(image_path), image, gt))
    return samples


def run_variant(segmenter, samples, variant, params, device):
    """Segment every sample with one variant, timing the stages and scoring the result."""
    settings = VARIANTS[variant]
    autocast = settings.get("autocast")
    batch_size = settings.get("batch_size", 64)
    flow_threads = settings.get("flow_threads", 1)
    points_version = settings.get("interior_points", "v2")

    propose_keys = ("candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size")
    select_keys = ("score_threshold", "max_overlap", "min_size", "refine_with_box_prompts", "box_extension")
    propose_params = {key: params[key] for key in propose_keys if key in params}
    select_params = {key: params[key] for key in select_keys if key in params}

    timer, rows = Timer(), []
    with tf32_context(settings.get("tf32")), weight_context(segmenter, settings.get("weights")), \
            half_precision_context(settings.get("half_precision", True)), \
            instrument(timer, segmenter, flow_threads, points_version):
        for name, image, gt in tqdm(samples, desc=f"APG {variant}"):
            start = time.perf_counter()
            with autocast_context(autocast, device):
                with timer.stage("initialize"):
                    segmenter.initialize(image, ndim=2)
                with timer.stage("propose"):
                    proposals = segmenter.propose(batch_size=batch_size, **propose_params)
            with timer.stage("select"):
                segmentation = segmenter.select(proposals, batch_size=batch_size, **select_params)
            elapsed = time.perf_counter() - start
            msa, accuracies = mean_segmentation_accuracy(segmentation, gt, return_accuracies=True)
            rows.append({
                "variant": variant, "image": name, "seconds": elapsed, "msa": float(msa),
                "sa50": float(accuracies[0]), "sa75": float(accuracies[5]),
                "n_proposals": len(proposals), "n_instances": int(segmentation.max()),
            })
            segmenter.clear_state()
    return pd.DataFrame(rows), timer.totals


def report(results, stage_totals, n_samples):
    """Print the per-stage table of every variant and the paired score delta against the baseline."""
    baseline = results[results["variant"] == "baseline"].set_index("image")

    print()
    print("Stages, seconds per image")
    stages = pd.DataFrame(stage_totals).T / n_samples
    print(stages.to_string(float_format=lambda v: f"{v:.4f}"))

    print()
    print(f"Scores and total runtime over {n_samples} images")
    rows = []
    for variant, frame in results.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "msa": frame["msa"].mean(),
            "sa50": frame["sa50"].mean(),
            "sa75": frame["sa75"].mean(),
            "sec_per_image": frame["seconds"].mean(),
        }
        if not baseline.empty and variant != "baseline":
            paired = frame.set_index("image")["msa"] - baseline["msa"]
            row["msa_delta"] = paired.mean()
            row["msa_delta_max_abs"] = paired.abs().max()
            row["speedup"] = baseline["seconds"].mean() / frame["seconds"].mean()
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n_per_cell_type", type=int, default=25, help="LIVECell test images per cell type.")
    parser.add_argument("-m", "--model_type", default="hvit_t", help="SAM2 backbone of the joint model.")
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    parser.add_argument("--variants", nargs="+", default=["baseline"], help=f"Any of {sorted(VARIANTS)}.")
    parser.add_argument("--apg_params", default=None, help="JSON overrides for the generation parameters.")
    parser.add_argument("--save", default=None, help="Write the per-image results to this CSV.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed images run before every variant.")
    args = parser.parse_args()

    unknown = [variant for variant in args.variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}, choose from {sorted(VARIANTS)}.")

    params = json.loads(args.apg_params) if args.apg_params else {}
    samples = load_livecell_subset(args.n_per_cell_type)
    print(f"Benchmarking on {len(samples)} livecell test images.")
    segmenter = build_segmenter(args.model_type, args.device)

    results, stage_totals = [], {}
    for variant in args.variants:
        if args.warmup:
            run_variant(segmenter, samples[:args.warmup], variant, params, args.device)
        frame, totals = run_variant(segmenter, samples, variant, params, args.device)
        results.append(frame)
        stage_totals[variant] = totals
    results = pd.concat(results, ignore_index=True)

    report(results, stage_totals, len(samples))
    if args.save:
        results.to_csv(args.save, index=False)
        print(f"Wrote the per-image results to '{args.save}'.")


if __name__ == "__main__":
    main()
