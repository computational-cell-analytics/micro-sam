"""Isolate the propagation of tiled 3d APG and measure how it scales with the objects of a pass.

The propagation is ~89% of a volume's runtime, so it is the only stage worth tuning further. This
runs it on its own, on one tile, with a controlled number of objects, so that a change can be read
off in seconds rather than by re-running a whole volume. It also reports where the time goes: the
CUDA time of the forwards, the host time spent launching them, and the blocking transfer of every
frame's masks back to the host.

Usage:
    python benchmark_propagation.py --objects 1 2 4 8 16
    python benchmark_propagation.py --objects 13 --torch_profile
    python benchmark_propagation.py --objects 16 --variant low_res_output
"""

import os
import time
import json
import argparse

import numpy as np

import torch

from micro_sam.v2.util import get_sam2_model, precompute_image_embeddings, FINETUNED_MODELS
from micro_sam.v2.models._video_predictor import CustomVideoPredictor
from micro_sam.v2.prompt_based_segmentation import TiledPromptableSegmentation3D

from common import DATA_ROOT, load_data

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results", "propagation_benchmark")


VARIANTS = ("baseline", "low_res_output", "cache_prepared")


def apply_variant(name):
    """Patch out one part of a pass so its share of the time can be read off.

    'low_res_output' skips the upsample of every frame's masks to the volume's resolution, so the
    host also receives a sixteenth of the bytes - it prices the output path, and its masks are wrong.
    'cache_prepared' keeps the flattened per-frame features rather than rebuilding them on every pass
    over the same slice.
    """
    if name == "baseline":
        return

    if name == "cache_prepared":
        original = CustomVideoPredictor._get_image_feature
        cache = {}

        def cached(self, inference_state, frame_idx, batch_size):
            key = (id(inference_state), int(frame_idx), int(batch_size))
            if key not in cache:
                cache[key] = original(self, inference_state, frame_idx, batch_size)
            return cache[key]

        CustomVideoPredictor._get_image_feature = cached
        return

    def low_res(self, inference_state, any_res_masks):
        return any_res_masks, any_res_masks

    CustomVideoPredictor._get_orig_video_res_output = low_res


def load_volume(dataset_name, input_path, z_crop, xy_crop):
    """The first scorable volume of a dataset, centre-cropped."""
    crop_shape = (z_crop, xy_crop, xy_crop)
    for raw, labels, _ in load_data(dataset_name, input_path, ndim=3, crop_shape=crop_shape):
        if labels.max() > 0:
            return raw
    raise ValueError(f"No annotated sample in '{dataset_name}'.")


def build_propagator(model, raw, embedding_path, tile_shape, halo, device, cache_all_slices=False):
    """One tiled propagator over the volume, reusing a cached embedding store when there is one."""
    embeddings = precompute_image_embeddings(
        model, raw, save_path=embedding_path, ndim=3, tile_shape=tile_shape, halo=halo,
        verbose=False, lazy_loading=True, devices=device,
    )
    return TiledPromptableSegmentation3D(
        model, raw, embeddings, devices=device, offload_state_to_cpu=False,
        max_cached_frames=int(raw.shape[0]) if cache_all_slices else None,
    )


def prompt_points(propagator, tile_id, n_objects):
    """`n_objects` points spread over a tile's inner block, in the volume's (y, x) frame."""
    inner = propagator.tiling.get_block_with_halo(tile_id, list(propagator.halo)).inner_block
    y0, x0 = int(inner.begin[0]), int(inner.begin[1])
    y1, x1 = int(inner.end[0]), int(inner.end[1])
    side = int(np.ceil(np.sqrt(n_objects)))
    ys = np.linspace(y0 + (y1 - y0) / (2 * side), y1 - (y1 - y0) / (2 * side), side)
    xs = np.linspace(x0 + (x1 - x0) / (2 * side), x1 - (x1 - x0) / (2 * side), side)
    return [(float(y), float(x)) for y in ys for x in xs][:n_objects]


def run_pass(propagator, tile_id, points, anchor):
    """One propagation pass: prompt every object on the anchor slice, then track both directions."""
    propagator.reset_tile_tracking(tile_id)
    for object_id, (y, x) in enumerate(points, start=1):
        propagator.add_point_prompts(
            frame_ids=anchor, points=np.array([[y, x]], dtype="float32"),
            point_labels=np.array([1], dtype="int32"), object_id=object_id,
        )
    return propagator.propagate_tile(tile_id)[0]


def time_pass(propagator, tile_id, n_objects, anchor, repeats):
    """Median seconds of a pass, and the cold pass before it that fills the feature cache.

    The cold pass reads every slice's features from the store, decompresses them and uploads them,
    which a warm pass finds cached. The gap between the two is what a prefetch could hide, and the
    schedule pays it once per job rather than once per volume, so it is worth pricing.
    """
    points = prompt_points(propagator, tile_id, n_objects)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    run_pass(propagator, tile_id, points, anchor)
    torch.cuda.synchronize()
    cold = time.time() - start

    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.time()
        segments = run_pass(propagator, tile_id, points, anchor)
        torch.cuda.synchronize()
        timings.append(time.time() - start)
    peak = torch.cuda.max_memory_allocated() / 1e9
    return float(np.median(timings)), len(segments), cold, peak


def profile_pass(propagator, tile_id, n_objects, anchor):
    """The top operators of one pass, by CUDA time and by host time."""
    from torch.profiler import profile, ProfilerActivity

    points = prompt_points(propagator, tile_id, n_objects)
    run_pass(propagator, tile_id, points, anchor)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run_pass(propagator, tile_id, points, anchor)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=14))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=14))
    totals = prof.key_averages()
    cuda = sum(event.self_device_time_total for event in totals) / 1e6
    cpu = sum(event.self_cpu_time_total for event in totals) / 1e6
    print(f"Totals: {cuda:.2f}s on device, {cpu:.2f}s on host.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset_name", default="gonuclear", help="The 3d dataset to segment.")
    parser.add_argument("-m", "--model_type", default="hvit_t_cells", choices=FINETUNED_MODELS)
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("--objects", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="Objects per pass.")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=(384, 384), help="In-plane tile shape.")
    parser.add_argument("--halo", type=int, nargs=2, default=(64, 64), help="In-plane tile halo.")
    parser.add_argument("--tile_id", type=int, default=0, help="The tile to propagate.")
    parser.add_argument("--z_crop", type=int, default=32, help="Slices of the volume to keep.")
    parser.add_argument("--xy_crop", type=int, default=1024, help="In-plane crop of the volume.")
    parser.add_argument("--anchor", type=int, default=None, help="Anchor slice. The middle by default.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed passes per object count.")
    parser.add_argument("--device", default="cuda:0", help="The device to propagate on.")
    parser.add_argument("--embedding_dir", default=os.environ.get("TMPDIR", "/tmp"), help="Embedding cache.")
    parser.add_argument("--torch_profile", action="store_true", help="Print the operator breakdown.")
    parser.add_argument(
        "--cache_all_slices", action="store_true",
        help="Keep every slice's features on the device instead of the free-VRAM heuristic.",
    )
    parser.add_argument("--tag", default="baseline", help="Name this benchmark in the saved json.")
    parser.add_argument("--variant", default="baseline", choices=VARIANTS, help="Price one part of a pass.")
    parser.add_argument(
        "--num_maskmem", type=int, default=None,
        help="Memory frames the attention reads, SAM2's 7 by default. Fewer is faster and less exact.",
    )
    args = parser.parse_args()

    apply_variant(args.variant)
    raw = load_volume(args.dataset_name, args.input_path, args.z_crop, args.xy_crop)
    anchor = args.anchor if args.anchor is not None else raw.shape[0] // 2
    print(f"Volume {raw.shape}, tile {args.tile_id}, anchor slice {anchor}, device {args.device}.")

    os.makedirs(args.embedding_dir, exist_ok=True)
    embedding_path = os.path.join(
        args.embedding_dir,
        f"bench_{args.dataset_name}_{args.model_type}_z{args.z_crop}_xy{args.xy_crop}.zarr",
    )
    model = get_sam2_model(model_type=args.model_type, device=args.device, input_type="videos")
    if args.num_maskmem is not None:
        # The temporal embeddings are indexed by recency, so a smaller window uses a valid prefix.
        model.num_maskmem = args.num_maskmem
    propagator = build_propagator(
        model, raw, embedding_path, tuple(args.tile_shape), tuple(args.halo), args.device,
        cache_all_slices=args.cache_all_slices,
    )

    if args.torch_profile:
        profile_pass(propagator, args.tile_id, max(args.objects), anchor)
        return

    n_frames = raw.shape[0]
    rows = []
    for n_objects in args.objects:
        seconds, tracked, cold, peak = time_pass(propagator, args.tile_id, n_objects, anchor, args.repeats)
        per_frame = 1000 * seconds / max(1, tracked)
        per_object_frame = per_frame / n_objects
        rows.append({
            "objects": n_objects, "seconds": seconds, "frames": tracked, "cold_seconds": cold,
            "peak_memory_gb": peak,
            "ms_per_frame": per_frame, "ms_per_object_frame": per_object_frame,
        })
        print(
            f"{n_objects:3d} objects: {seconds:6.3f}s over {tracked} frames "
            f"({per_frame:6.2f} ms/frame, {per_object_frame:5.2f} ms/object-frame); "
            f"cold {cold:6.3f}s, so {cold - seconds:6.3f}s of it is filling the feature cache; "
            f"peak {peak:.1f} GB"
        )

    if len(rows) > 1:
        scaling = rows[-1]["seconds"] / rows[0]["seconds"]
        objects = rows[-1]["objects"] / rows[0]["objects"]
        print(f"{objects:.0f}x the objects costs {scaling:.2f}x the time (1.0 = free, {objects:.0f} = serial).")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    save_path = os.path.join(RESULTS_ROOT, f"{args.dataset_name}_{args.model_type}_{args.tag}.json")
    with open(save_path, "w") as f:
        json.dump({
            "tag": args.tag, "variant": args.variant,
            "volume_shape": list(raw.shape),
            "n_frames": n_frames, "rows": rows,
        }, f, indent=2)


if __name__ == "__main__":
    main()
