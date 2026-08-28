"""Segment the full gonuclear sample 0 volume with tiled 3d APG at wave=1 and wave=4, saved to h5.

Every GPU setup writes into the same h5 file, under its own group, so the raw volume and the
ground truth are stored once instead of once per setup.

Usage:
    python run_gonuclear_waves_h5.py --devices cuda:0 --group 1gpu -o gonuclear_waves.h5
    python run_gonuclear_waves_h5.py --devices cuda:0 cuda:1 cuda:2 cuda:3 --group 4gpu -o gonuclear_waves.h5
"""

import os
import time
import argparse

import h5py

import torch

from micro_sam.v2.util import get_sam2_model, FINETUNED_MODELS
from micro_sam.v2.instance_segmentation import get_decoder, get_instance_segmentation_generator

from common import DATA_ROOT, VOLUME_SPEED_OPTIONS, load_data, resolve_params


def build_segmenter(model_type, device, devices, n_worker_processes):
    model = get_sam2_model(model_type=model_type, device=device, input_type="videos")
    decoder = get_decoder(model_type=model_type, device=device, encoder=model.image_encoder)
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=3, is_tiled=True,
        inference_device=devices, n_worker_processes=n_worker_processes,
    )


def segment_volume(model, raw, tile_shape, halo, waves, max_size_factor=None, embedding_path=None):
    model.clear_state()
    model.initialize(
        raw, ndim=3, tile_shape=tile_shape, halo=halo, save_path=embedding_path, **VOLUME_SPEED_OPTIONS
    )
    overrides = {"propagation_waves": waves, "max_size_factor": max_size_factor}
    params = resolve_params(overrides, ndim=3)
    start = time.time()
    seg = model.generate(**params).astype("uint32")
    elapsed = time.time() - start
    print(f"wave={waves}: segmented in {elapsed:.1f}s, {int(seg.max())} predicted objects.")
    return seg, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_type", default="hvit_t_cells", choices=FINETUNED_MODELS)
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=(384, 384), help="In-plane tile shape (y, x).")
    parser.add_argument("--halo", type=int, nargs=2, default=(64, 64), help="In-plane tile halo (y, x).")
    parser.add_argument("--sample_index", type=int, default=0, help="Which gonuclear test sample to segment.")
    parser.add_argument("--devices", nargs="*", default=None, help="Devices to spread the work over.")
    parser.add_argument("--n_worker_processes", type=int, default=None, help="Propagation worker processes.")
    parser.add_argument("--embedding_dir", default=None, help="Cache the embeddings here instead of the default.")
    parser.add_argument("-o", "--output_path", required=True, help="h5 path to write raw/gt/seg into.")
    parser.add_argument("--group", required=True, help="Group name for this GPU setup, e.g. '1gpu' or '4gpu'.")
    parser.add_argument(
        "--max_size_factor", type=float, default=None,
        help="Reject a candidate this many times larger than the median candidate size. None disables it.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segmenter(args.model_type, device, args.devices or None, args.n_worker_processes)

    # (10**6,) * 3 always exceeds the volume, so this loads the sample at full size, not a center crop.
    samples = load_data("gonuclear", args.input_path, ndim=3, crop_shape=(10**6, 10**6, 10**6))
    raw, labels = None, None
    for index, (sample_raw, sample_labels, _) in enumerate(samples):
        if index == args.sample_index:
            raw, labels = sample_raw, sample_labels
            break
    if raw is None:
        raise ValueError(f"Sample {args.sample_index} not found in gonuclear.")
    print(f"Sample {args.sample_index}: volume shape {raw.shape}, {int(labels.max())} ground-truth objects.")

    embedding_path = None
    if args.embedding_dir is not None:
        os.makedirs(args.embedding_dir, exist_ok=True)
        embedding_path = os.path.join(args.embedding_dir, f"gonuclear_{args.model_type}_sample{args.sample_index}.zarr")

    with h5py.File(args.output_path, "a") as f:
        if "raw" not in f:
            f.create_dataset("raw", data=raw, compression="gzip")
        if "gt" not in f:
            f.create_dataset("gt", data=labels, compression="gzip")
        group = f.require_group(f"seg/{args.group}")
        group.attrs["devices"] = args.devices or "all"
        group.attrs["n_worker_processes"] = args.n_worker_processes if args.n_worker_processes is not None else -1
        for waves in (4,):
            seg, elapsed = segment_volume(
                model, raw, tuple(args.tile_shape), tuple(args.halo), waves, args.max_size_factor, embedding_path,
            )
            dataset = group.create_dataset(f"wave{waves}", data=seg, compression="gzip")
            dataset.attrs["seconds"] = elapsed
            dataset.attrs["n_objects"] = int(seg.max())

    print(f"Wrote '{args.output_path}'.")


if __name__ == "__main__":
    main()
