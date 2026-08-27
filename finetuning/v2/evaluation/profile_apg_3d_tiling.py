"""Stage-by-stage profile of tiled 3d APG, to find what is worth parallelizing.

Splits one volume's runtime into the encoder embeddings, the decoder pass, the prompt derivation,
the candidate scoring, the propagation and the tile merge, and samples every GPU's utilization
while it runs. The propagation dominates on a single device (see the APG notes), so the point of
the breakdown is to see what is left once it is spread over several GPUs.

Usage:
    python profile_apg_3d_tiling.py -m hvit_t_cells --z_crop 32
    python profile_apg_3d_tiling.py -d cremi --z_crop 32 --xy_crop 768 --devices cuda:0  # one GPU
"""

import os
import time
import json
import shutil
import argparse
import threading
import subprocess
from contextlib import contextmanager

import numpy as np

import torch

from micro_sam.v2 import automatic_prompt_generation as apg_module
from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D, TiledPromptableSegmentation3D
from micro_sam.v2.util import get_sam2_model, precompute_image_embeddings, FINETUNED_MODELS
from micro_sam.v2.instance_segmentation import get_decoder, get_instance_segmentation_generator

from common import DATA_ROOT, VOLUME_SPEED_OPTIONS, load_data, resolve_params

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results", "apg_3d_profile")


class GpuSampler:
    """Samples every GPU's utilization and memory in the background while a stage runs."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        query = "utilization.gpu,memory.used"
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
            except (subprocess.SubprocessError, OSError):
                continue
            row = [[int(v) for v in line.split(",")] for line in out.strip().splitlines() if line.strip()]
            if row:
                self.samples.append((time.time(), row))

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summarize(self, start, end):
        """Mean utilization and peak memory per GPU over a time window."""
        window = [row for stamp, row in self.samples if start <= stamp <= end]
        if not window:
            return None
        array = np.array(window, dtype="float32")  # (n_samples, n_gpus, 2)
        return {
            "util_mean": array[..., 0].mean(axis=0).round(1).tolist(),
            "memory_peak_mb": array[..., 1].max(axis=0).astype(int).tolist(),
        }


class Timings:
    """Wall-clock windows of the named stages, in the order they ran."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.stages = []
        self.totals = {}

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        yield
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        self.stages.append({"name": name, "seconds": end - start, "gpu": self.sampler.summarize(start, end)})
        print(f"[stage] {name}: {end - start:.1f}s")

    def wrap(self, owner, name):
        """Time every call of a method, summed under its name."""
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            with self.stage(name):
                result = original(*args, **kwargs)
            return result

        setattr(owner, name, timed)
        return original

    def accumulate(self, owner, name):
        """Sum the wall time and call count of a hot method, without a line per call."""
        original = getattr(owner, name)
        total = self.totals.setdefault(name, {"calls": 0, "seconds": 0.0})

        def counted(*args, **kwargs):
            start = time.time()
            result = original(*args, **kwargs)
            total["calls"] += 1
            total["seconds"] += time.time() - start
            return result

        setattr(owner, name, counted)
        return original


def instrument_merge(tally):
    """Tally why the volumetric merge kept or dropped every propagated candidate.

    The merge already reports its reasons; asking for them costs nothing and says how much of the
    propagation was spent on candidates that lose, which is what candidate pruning would recover.
    """
    original = apg_module.merge_by_score

    def counted(records, shape, **kwargs):
        result = original(records, shape, return_reasons=True, **kwargs)
        head, reasons = result[:-1], result[-1]
        # Only the volumetric merge, not the per-slice duplicate suppression, which asks for matches.
        if not kwargs.get("return_matches"):
            for reason in reasons:
                tally[reason] = tally.get(reason, 0) + 1
        return head[0] if len(head) == 1 else head

    apg_module.merge_by_score = counted


def instrument_propagation(timings, job_times):
    """Break the propagation into its parts, and time every job so the imbalance is visible.

    Only sees the propagation that runs in this process. With the worker pool the passes run in the
    worker processes, so the parts stay at zero and the stage total is what there is to read.
    """
    for name in ("_get_segmenter", "add_point_prompts", "propagate_tile", "release_tile"):
        timings.accumulate(TiledPromptableSegmentation3D, name)
    timings.accumulate(PromptableSegmentation3D, "reset_tracking")
    timings.accumulate(apg_module, "_volume_records")

    original = apg_module.propagate_passes

    def timed_job(propagator, tile_id, passes, *args, **kwargs):
        start = time.time()
        result = original(propagator, tile_id, passes, *args, **kwargs)
        job_times.append({"tile_id": int(tile_id), "passes": len(passes), "seconds": time.time() - start})
        return result

    apg_module.propagate_passes = timed_job


def profile_volume(model, raw, tile_shape, halo, params, timings, breakdown, tile_times, embedding_path):
    """Run one volume, timing every stage of the generator."""
    model.clear_state()
    with timings.stage("encoder_embeddings"):
        embeddings = precompute_image_embeddings(
            model._video_predictor, raw, save_path=embedding_path, ndim=3,
            tile_shape=tile_shape, halo=halo, verbose=False, lazy_loading=True,
        )
    with timings.stage("decoder"):
        model.initialize(raw, ndim=3, image_embeddings=embeddings, **VOLUME_SPEED_OPTIONS)

    if breakdown:
        instrument_propagation(timings, tile_times)
    timings.wrap(apg_module, "derive_volume_prompts")
    timings.wrap(model, "_score_candidates")
    timings.wrap(model, "_propagate_candidates")
    timings.wrap(model, "_merge")
    with timings.stage("generate_total"):
        segmentation = model.generate(**params).astype("uint32")
    return segmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset_name", default="gonuclear", help="The 3d dataset to segment.")
    parser.add_argument("-m", "--model_type", default="hvit_t_cells", choices=FINETUNED_MODELS)
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=(384, 384), help="In-plane tile shape (y, x).")
    parser.add_argument("--halo", type=int, nargs=2, default=(64, 64), help="In-plane tile halo (y, x).")
    parser.add_argument("--sample_index", type=int, default=0, help="Profile only this one sample, by index.")
    parser.add_argument("--z_crop", type=int, default=None, help="Center-crop the volume to this many z slices.")
    parser.add_argument("--xy_crop", type=int, default=None, help="Center-crop the volume in y and x.")
    parser.add_argument("--devices", nargs="*", default=None, help="Devices. Every visible GPU by default.")
    parser.add_argument("--tag", default="baseline", help="Name this profile in the saved json.")
    parser.add_argument("--breakdown", action="store_true", help="Also break the propagation into its parts.")
    parser.add_argument("--embedding_dir", default=None, help="Directory the embeddings are cached in.")
    parser.add_argument(
        "--n_worker_processes", type=int, default=None,
        help="Processes the propagation runs in. One per device by default; 0 keeps it in this process.",
    )
    parser.add_argument("--save_segmentation", default=None, help="Optional npy path for the segmentation.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # None fans out over every visible GPU, which is what an evaluation on a whole node wants.
    inference_device = args.devices or None
    model = get_sam2_model(model_type=args.model_type, device=device, input_type="videos")
    decoder = get_decoder(model_type=args.model_type, device=device, encoder=model.image_encoder)
    model = get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=3, is_tiled=True,
        inference_device=inference_device, n_worker_processes=args.n_worker_processes,
    )
    params = resolve_params(ndim=3)

    crop_shape = (args.z_crop or 10**6, args.xy_crop or 10**6, args.xy_crop or 10**6)
    samples = load_data(args.dataset_name, args.input_path, ndim=3, crop_shape=crop_shape)
    raw, labels = None, None
    for index, (sample_raw, sample_labels, _) in enumerate(samples):
        if index == args.sample_index:
            raw, labels = sample_raw, sample_labels
            break
    if raw is None:
        raise ValueError(f"Sample {args.sample_index} not found in '{args.dataset_name}'.")

    print(f"Volume shape {raw.shape}, {int(labels.max())} ground-truth objects, devices {args.devices}.")
    embedding_path = None
    if args.embedding_dir is not None:
        os.makedirs(args.embedding_dir, exist_ok=True)
        embedding_path = os.path.join(args.embedding_dir, f"{args.dataset_name}_{args.model_type}.zarr")
        shutil.rmtree(embedding_path, ignore_errors=True)

    sampler = GpuSampler()
    sampler.start()
    timings = Timings(sampler)
    start = time.time()
    tile_times, merge_tally = [], {}
    instrument_merge(merge_tally)
    segmentation = profile_volume(
        model, raw, tuple(args.tile_shape), tuple(args.halo), params, timings, args.breakdown, tile_times,
        embedding_path,
    )
    total = time.time() - start
    sampler.stop()

    print(f"Total {total:.1f}s, {int(segmentation.max())} predicted objects.")
    for stage in timings.stages:
        share = 100 * stage["seconds"] / total
        gpu = stage["gpu"]
        util = "" if gpu is None else f" util {gpu['util_mean']} mem {gpu['memory_peak_mb']}"
        print(f"{stage['name']}: {stage['seconds']:.1f}s ({share:.1f}%){util}")
    for name, total in timings.totals.items():
        print(f"{name}: {total['seconds']:.1f}s over {total['calls']} calls")
    for job in sorted(tile_times, key=lambda entry: -entry["seconds"]):
        print(f"tile {job['tile_id']}: {job['seconds']:.1f}s for {job['passes']} passes")
    propagated = sum(merge_tally.values())
    for reason, count in sorted(merge_tally.items(), key=lambda item: -item[1]):
        print(f"merge '{reason}': {count} of {propagated} propagated candidates ({100 * count / propagated:.1f}%)")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    tag = f"{args.dataset_name}_{args.model_type}_{args.tag}"
    with open(os.path.join(RESULTS_ROOT, f"{tag}.json"), "w") as f:
        json.dump({
            "tag": args.tag, "dataset": args.dataset_name, "model": args.model_type,
            "volume_shape": list(raw.shape), "tile_shape": list(args.tile_shape), "halo": list(args.halo),
            "devices": args.devices, "total_seconds": total, "n_objects": int(segmentation.max()),
            "stages": timings.stages, "totals": timings.totals, "tiles": tile_times,
            "merge_reasons": merge_tally,
        }, f, indent=2)
    if args.save_segmentation is not None:
        np.save(args.save_segmentation, segmentation)


if __name__ == "__main__":
    main()
