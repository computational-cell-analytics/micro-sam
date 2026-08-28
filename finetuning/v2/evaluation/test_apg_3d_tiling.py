"""Sanity check for tiled 3d APG: segment a full 3d volume instead of the usual small center crop.

Every other 3d APG evaluation (see `apg3d_gonuclear.sh`) runs on `CROP_SHAPE_3D`, an (8, 512, 512)
center crop, because the untiled generator holds one video-predictor state for the whole volume and
a dataset like gonuclear does not fit. This script is the first check that the tiled generator
(XYZ blocks, stitched by `bioimage_py`'s halo-overlap multicut, see `TiledAutomaticPromptGenerator`)
covers the whole volume instead.

Uses the model registry ingested from owncloud (`micro_sam.v2.util.FINETUNED_MODELS`, e.g.
'hvit_t_cells'), not the local joint training checkpoints.

Usage:
    python test_apg_3d_tiling.py -m hvit_t_cells
    python test_apg_3d_tiling.py -d embedseg -m hvit_t_cells
    python test_apg_3d_tiling.py -m hvit_s_cells --tile_shape 256 256 --halo 48 48
    python test_apg_3d_tiling.py -m hvit_t_cells --z_block 64 --z_halo 8  # also block z
    python test_apg_3d_tiling.py -m hvit_t_cells --devices cuda:0  # pin to one GPU
"""

import os
import time
import argparse

import pandas as pd

import torch

from micro_sam.v2.util import get_sam2_model, FINETUNED_MODELS
from micro_sam.v2.instance_segmentation import get_decoder, get_instance_segmentation_generator
from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION

from common import DATA_ROOT, VOLUME_SPEED_OPTIONS, load_data, resolve_params, run_dataset_evaluation

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results", "apg_3d_tiling")


def build_segmenter(model_type, device, devices, workers_per_device=1):
    """Build the tiled 3d prompt generator from the owncloud-ingested registry, no local checkpoints."""
    model = get_sam2_model(model_type=model_type, device=device, input_type="videos")
    decoder = get_decoder(model_type=model_type, device=device, encoder=model.image_encoder)
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=3, is_tiled=True,
        inference_device=devices, workers_per_device=workers_per_device,
    )


def segment_volume(model, raw, block_shape, halo, params):
    """Segment one volume with the tiled generator, from a clean state."""
    model.clear_state()
    model.initialize(raw, ndim=3, tile_shape=block_shape, halo=halo, **VOLUME_SPEED_OPTIONS)
    return model.generate(**params).astype("uint32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset_name", default="gonuclear", help="The 3d dataset to segment.")
    parser.add_argument("-m", "--model_type", default="hvit_t_cells", choices=FINETUNED_MODELS)
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=(384, 384), help="In-plane block shape (y, x).")
    parser.add_argument("--halo", type=int, nargs=2, default=(64, 64), help="In-plane block halo (y, x).")
    parser.add_argument(
        "--z_block", type=int, default=None, help="Z block size. Whole depth (no z split) by default.",
    )
    parser.add_argument("--z_halo", type=int, default=0, help="Z halo. 0 by default (no z split).")
    parser.add_argument(
        "--devices", nargs="*", default=None,
        help="Devices to spread the work over. Every visible GPU by default.",
    )
    parser.add_argument(
        "--workers_per_device", type=int, default=1,
        help="Independent tile/block workers to run concurrently per device. >1 helps when a "
             "single worker leaves a device's compute or memory underused.",
    )
    parser.add_argument("--sample_index", type=int, default=None, help="Score only this one sample, by index.")
    parser.add_argument(
        "--max_size_factor", type=float, default=None,
        help="Reject a candidate this many times larger than the median size it is merged against. "
             "Catches SAM2 propagation drift (a track that grows onto background). None disables it.",
    )
    parser.add_argument(
        "--propagation_waves", type=int, default=None,
        help="Rounds to propagate candidates in, highest scoring first, pruning duplicates a "
             "higher round already covers. A candidate near a tile/block's halo is never pruned. "
             "None keeps the library default (one round, no pruning).",
    )
    parser.add_argument(
        "--n_threads", type=int, default=None,
        help="CPU threads for one tile/block's own candidate proposal work. None keeps the library "
             "default (8); raise it to use more of a node's cores across concurrent tile/block workers.",
    )
    parser.add_argument(
        "--z_crop", type=int, default=None,
        help="Center-crop the volume to this many z slices, for a fast turnaround. Full depth if unset.",
    )
    parser.add_argument(
        "--xy_crop", type=int, default=None,
        help="Center-crop the volume to this many pixels in y and x. Full in-plane size if unset.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # None fans out over every visible GPU: the decoder and every tile/block's propagation use them.
    model = build_segmenter(args.model_type, device, args.devices or None, args.workers_per_device)
    overrides = {}
    if args.max_size_factor is not None:
        overrides["max_size_factor"] = args.max_size_factor
    if args.propagation_waves is not None:
        overrides["propagation_waves"] = args.propagation_waves
    if args.n_threads is not None:
        overrides["n_threads"] = args.n_threads
    params = resolve_params(overrides, ndim=3)

    tag = f"{args.dataset_name}_{args.model_type}"

    if args.z_crop is not None:
        tag = f"{tag}_z{args.z_crop}"
    if args.xy_crop is not None:
        tag = f"{tag}_xy{args.xy_crop}"
    if args.z_block is not None:
        tag = f"{tag}_zblock{args.z_block}_zhalo{args.z_halo}"
    if args.workers_per_device != 1:
        tag = f"{tag}_workers{args.workers_per_device}"
    # Only when it is not the library default, so a default run still reads the results it already wrote.
    if params["max_size_factor"] != DEFAULT_PROMPT_GENERATION["max_size_factor"]:
        tag = f"{tag}_maxsize{params['max_size_factor']}"
    if params["propagation_waves"] != DEFAULT_PROMPT_GENERATION["propagation_waves"]:
        tag = f"{tag}_waves{params['propagation_waves']}"
    if params["n_threads"] != DEFAULT_PROMPT_GENERATION["n_threads"]:
        tag = f"{tag}_threads{params['n_threads']}"

    # (10**6,) * 3 always exceeds the volume, so an unset axis keeps that axis whole (a center crop).
    crop_shape = (args.z_crop or 10**6, args.xy_crop or 10**6, args.xy_crop or 10**6)
    samples = load_data(args.dataset_name, args.input_path, ndim=3, crop_shape=crop_shape)
    for index, (raw, labels, _) in enumerate(samples):
        if labels.max() == 0 or (args.sample_index is not None and index != args.sample_index):
            continue
        save_path = os.path.join(RESULTS_ROOT, f"{tag}_sample{index}.csv")
        if os.path.exists(save_path):
            print(f"Sample {index}: already scored at '{save_path}', skipping.")
            print(pd.read_csv(save_path))
            continue

        print(f"Sample {index}: volume shape {raw.shape}, {int(labels.max())} ground-truth objects.")

        z_block = args.z_block or raw.shape[0]
        block_shape = (z_block, *args.tile_shape)
        halo = (args.z_halo, *args.halo)

        start = time.time()
        seg = segment_volume(model, raw, block_shape, halo, params)
        elapsed = time.time() - start
        print(f"Sample {index}: segmented in {elapsed:.1f}s, {int(seg.max())} predicted objects.")

        results = run_dataset_evaluation([labels], [seg], args.dataset_name, save_path=save_path)
        print(results)


if __name__ == "__main__":
    main()
