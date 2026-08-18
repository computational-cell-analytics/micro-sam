"""Measure what automatic prompt generation costs per stage, and what a speed-up costs in score.

Every variant runs the code path the evaluation runs. The stages are timed by wrapping the functions
that the generator calls, not by mirroring them, so the numbers cannot drift from what the library
does. Each sample is scored as well as timed. A variant is only useful if it leaves mSA where the
baseline had it, so the report pairs the runtime with the mean score delta over the same samples.

Usage:
    python evaluate_apg.py -d livecell --variants baseline threads8
    python evaluate_apg.py -d gonuclear --ndim 3 -n 3 --variants shipped nopp32 --save results.csv
"""

import os
import json
import time
import argparse
import functools
from contextlib import ExitStack, contextmanager, nullcontext

import pandas as pd
from tqdm import tqdm

import torch

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.v2.util import get_sam2_model
from micro_sam.v2 import automatic_prompt_generation as apg
from micro_sam.v1.evaluation.livecell import _get_livecell_paths
from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION
from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator

from common import (
    DATA_ROOT, DATASET_SPACING, drop_excluded_livecell, export_joint_checkpoint, load_unisam2_model,
)
from baselines_common import _load_data, load_evaluation_sample_2d

# What a variant may change. Only knobs that leave the candidates alone belong here: a variant that
# proposed different ones would be a different method, not a faster one. 'baseline' is the generator
# as it was before this benchmark, so every measured change is paired against the same reference.
# The last three apply to volumes only.
VARIANTS = {
    "baseline": {"n_threads": 1, "half_precision": False},
    "threads8": {"n_threads": 8, "half_precision": False},
    "shipped": {},
    "batch256": {"batch_size": 256},
    "tf32": {"tf32": True},
    "bfloat16": {"half_precision": False, "autocast": "bfloat16"},
    "bfloat16_weights": {"half_precision": False, "autocast": "bfloat16", "weights": "bfloat16"},
    "float16_weights": {"weights": "float16"},
    "nopp4": {"n_objects_per_pass": 4},
    "nopp32": {"n_objects_per_pass": 32},
    "on_device": {"offload_to_cpu": False},
}

# The generation parameters, taken from the library defaults so a run measures what ships.
GENERATE_KEYS = (
    "candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size",
    "score_threshold", "max_overlap", "min_size", "refine_with_box_prompts", "box_extension",
    "multimasking", "n_objects_per_pass", "early_stop_patience", "n_threads",
)

# Variant settings that `generate` takes directly. The others change how the model runs, not what it does.
GENERATE_OVERRIDES = ("n_threads", "batch_size", "n_objects_per_pass")

# The stage names, and the functions whose runtime they collect. A module entry is patched on the
# `automatic_prompt_generation` module, a segmenter entry on the instance.
MODULE_STAGES = {
    2: {"flow_density": "_compute_flow_density", "connected_components": "label",
        "interior_points": "interior_points"},
    3: {"derive_candidates": "derive_volume_prompts", "merge": "merge_by_score"},
}
SEGMENTER_STAGES = {
    2: {"encode": "_encode", "prompt": "_apply", "merge": "_merge", "refine_boxes": "_refine_boxes"},
    3: {"score_candidates": "_score_candidates", "propagate": "_propagate_candidates"},
}


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
def autocast_context(choice, device):
    """Run the SAM2 branch in reduced precision, which is how SAM2's own inference examples run it."""
    if choice is None or not str(device).startswith("cuda"):
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=getattr(torch, choice)):
            yield


@contextmanager
def weight_context(segmenter, dtype):
    """Hold the SAM2 backbone's weights in reduced precision, and restore the float32 ones afterwards.

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
    """Allow TF32 matmuls for one variant, and restore the process default afterwards."""
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
def instrument(timer, segmenter, ndim):
    """Time the inner stages by wrapping the functions that the generator actually calls.

    The wrapping is undone on exit, so one process can time several variants in a row.
    """
    targets = [(apg, MODULE_STAGES[ndim]), (segmenter, SEGMENTER_STAGES[ndim])]
    originals = [{stage: getattr(obj, name) for stage, name in names.items()} for obj, names in targets]
    for (obj, names), saved in zip(targets, originals):
        for stage, name in names.items():
            setattr(obj, name, timer.wrap(stage, saved[stage]))
    try:
        yield
    finally:
        for (obj, names), saved in zip(targets, originals):
            for stage, name in names.items():
                setattr(obj, name, saved[stage])


def build_segmenter(model_type, ndim, device, joint_checkpoint="best"):
    """Build the prompt generator the way the evaluation builds it."""
    interactive_path, decoder_path = export_joint_checkpoint(model_type, joint_checkpoint)
    decoder = load_unisam2_model(decoder_path, device, encoder=model_type)
    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=interactive_path,
        **({"input_type": "videos"} if ndim == 3 else {}),
    )
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=ndim,
    )


def load_samples(dataset_name, ndim, n_samples=None, n_per_cell_type=25):
    """Load the samples the evaluation scores, as (name, raw, ground truth) triples.

    The livecell test split is subsampled per cell type, which is how the evaluation stratifies it.
    Every other dataset is loaded in the order the evaluation loads it. Samples without any
    annotated object are dropped, since they cannot be scored.
    """
    if dataset_name == "livecell":
        image_paths, gt_paths = _get_livecell_paths(
            input_folder=os.path.join(DATA_ROOT, "livecell"), split="test", n_val_per_cell_type=n_per_cell_type,
        )
        image_paths, gt_paths = drop_excluded_livecell(image_paths, gt_paths)
        pairs = sorted(zip(image_paths, gt_paths))
        loader = (
            (os.path.basename(image_path),) + load_evaluation_sample_2d(image_path, gt_path, None, None, "livecell")
            for image_path, gt_path in pairs
        )
        total = len(pairs)
    else:
        loader = (
            (f"sample{index:03d}", raw, gt)
            for index, (raw, gt, _) in enumerate(_load_data(dataset_name, DATA_ROOT, ndim=ndim))
        )
        total = n_samples

    samples = []
    for name, raw, gt in tqdm(loader, total=total, desc=f"Load {dataset_name}"):
        if gt.max() > 0:
            samples.append((name, raw, gt))
        if n_samples is not None and len(samples) >= n_samples:
            break
    return samples


def resolve_params(overrides, ndim):
    """The generation parameters for one run, with 'overrides' applied on top of the library defaults."""
    params = {key: DEFAULT_PROMPT_GENERATION[key] for key in GENERATE_KEYS}
    params.update(overrides)
    if ndim == 3:
        # A candidate's density scales with the object's size, so a volume has its own threshold.
        default_3d = DEFAULT_PROMPT_GENERATION["candidate_threshold_3d"]
        params["candidate_threshold"] = overrides.get("candidate_threshold_3d", default_3d)
    params.pop("candidate_threshold_3d", None)
    return params


def run_variant(segmenter, samples, variant, params, ndim, device, spacing):
    """Segment every sample with one variant, timing the stages and scoring the result."""
    settings = VARIANTS[variant]
    generate_params = dict(params, batch_size=64)
    generate_params.update({key: settings[key] for key in GENERATE_OVERRIDES if key in settings})
    initialize_params = {"offload_to_cpu": settings["offload_to_cpu"]} if "offload_to_cpu" in settings else {}

    timer, rows = Timer(), []
    with ExitStack() as stack:
        stack.enter_context(tf32_context(settings.get("tf32")))
        stack.enter_context(weight_context(segmenter, settings.get("weights")))
        stack.enter_context(half_precision_context(settings.get("half_precision", True)))
        stack.enter_context(instrument(timer, segmenter, ndim))
        for name, raw, gt in tqdm(samples, desc=f"APG {variant} {ndim}d"):
            start = time.perf_counter()
            with autocast_context(settings.get("autocast"), device):
                with timer.stage("initialize"):
                    segmenter.initialize(raw, ndim=ndim, **initialize_params)
                segmentation = segmenter.generate(spacing=spacing, **generate_params)
            elapsed = time.perf_counter() - start
            msa, accuracies = mean_segmentation_accuracy(segmentation, gt, return_accuracies=True)
            rows.append({
                "variant": variant, "sample": name, "seconds": elapsed, "msa": float(msa),
                "sa50": float(accuracies[0]), "sa75": float(accuracies[5]),
                "n_instances": int(segmentation.max()),
            })
            segmenter.clear_state()

    totals = timer.totals
    # 'encode' runs inside 'initialize', so report the remainder of it as the decoder pass.
    if "encode" in totals:
        totals["decoder"] = totals.pop("initialize") - totals["encode"]
    return pd.DataFrame(rows), totals


def report(results, stage_totals, n_samples, dataset_name, ndim):
    """Print the per-stage table of every variant and the paired score delta against the baseline."""
    baseline = results[results["variant"] == results["variant"].iloc[0]].set_index("sample")

    print()
    print(f"{dataset_name} ({ndim}d), stages in seconds per sample")
    stages = pd.DataFrame(stage_totals).T / n_samples
    print(stages.to_string(float_format=lambda value: f"{value:.4f}"))

    print()
    print(f"Scores and total runtime over {n_samples} samples")
    rows = []
    for variant, frame in results.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "msa": frame["msa"].mean(),
            "sa50": frame["sa50"].mean(),
            "sa75": frame["sa75"].mean(),
            "sec_per_sample": frame["seconds"].mean(),
        }
        if not baseline.empty and variant != baseline["variant"].iloc[0]:
            paired = frame.set_index("sample")["msa"] - baseline["msa"]
            row["msa_delta"] = paired.mean()
            row["msa_delta_max_abs"] = paired.abs().max()
            row["speedup"] = baseline["seconds"].mean() / frame["seconds"].mean()
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", default="livecell", help="Dataset to benchmark on.")
    parser.add_argument("-m", "--model_type", default="hvit_t", help="SAM2 backbone of the joint model.")
    parser.add_argument("-n", "--n_samples", type=int, default=None, help="Samples to time, all of them by default.")
    parser.add_argument("--ndim", type=int, default=2, choices=[2, 3], help="Spatial dimensions.")
    parser.add_argument("--n_per_cell_type", type=int, default=25, help="LIVECell test images per cell type.")
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    parser.add_argument("--joint_checkpoint", default="best", help="Checkpoint of the joint model.")
    parser.add_argument("--variants", nargs="+", default=["shipped"], help=f"Any of {sorted(VARIANTS)}.")
    parser.add_argument("--apg_params", default=None, help="JSON overrides for the generation parameters.")
    parser.add_argument("--save", default=None, help="Write the per-sample results to this CSV.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed samples run before every variant.")
    args = parser.parse_args()

    unknown = [variant for variant in args.variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}, choose from {sorted(VARIANTS)}.")

    params = resolve_params(json.loads(args.apg_params) if args.apg_params else {}, args.ndim)
    samples = load_samples(args.dataset_name, args.ndim, args.n_samples, args.n_per_cell_type)
    if not samples:
        raise RuntimeError(f"No annotated samples were loaded for '{args.dataset_name}'.")
    print(f"Benchmarking on {len(samples)} {args.dataset_name} samples.")

    segmenter = build_segmenter(args.model_type, args.ndim, args.device, args.joint_checkpoint)
    spacing = DATASET_SPACING.get(args.dataset_name)

    results, stage_totals = [], {}
    for variant in args.variants:
        # The first samples pay for CUDA warm-up, so they run untimed.
        if args.warmup:
            run_variant(segmenter, samples[:args.warmup], variant, params, args.ndim, args.device, spacing)
        frame, totals = run_variant(segmenter, samples, variant, params, args.ndim, args.device, spacing)
        results.append(frame)
        stage_totals[variant] = totals
    results = pd.concat(results, ignore_index=True)

    report(results, stage_totals, len(samples), args.dataset_name, args.ndim)
    if args.save:
        results.to_csv(args.save, index=False)
        print(f"Wrote the per-sample results to '{args.save}'.")


if __name__ == "__main__":
    main()
