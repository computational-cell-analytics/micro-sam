"""Sanity check for tiled 3d APG: segment a full 3d volume instead of the usual small center crop.

Every other 3d APG evaluation (see `apg3d_gonuclear.sh`) runs on `CROP_SHAPE_3D`, an (8, 512, 512)
center crop, because the untiled generator holds one video-predictor state for the whole volume and
a dataset like gonuclear does not fit. This script is the first check that the tiled generator
(in-plane tiling, built and torn down one tile at a time, see `TiledAutomaticPromptGenerator`)
covers the whole volume instead.

Uses the model registry ingested from owncloud (`micro_sam.v2.util.FINETUNED_MODELS`, e.g.
'hvit_t_cells'), not the local joint training checkpoints.

Usage:
    python test_apg_3d_tiling.py -m hvit_t_cells
    python test_apg_3d_tiling.py -d embedseg -m hvit_t_cells
    python test_apg_3d_tiling.py -m hvit_s_cells --tile_shape 256 256 --halo 48 48
"""

import os
import time
import argparse

import pandas as pd

import torch

from micro_sam.v2.util import get_sam2_model, FINETUNED_MODELS
from micro_sam.v2.instance_segmentation import get_decoder, get_instance_segmentation_generator

from common import DATA_ROOT, VOLUME_SPEED_OPTIONS, load_data, resolve_params, run_dataset_evaluation

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results", "apg_3d_tiling")


def build_segmenter(model_type, device):
    """Build the tiled 3d prompt generator from the owncloud-ingested registry, no local checkpoints."""
    model = get_sam2_model(model_type=model_type, device=device, input_type="videos")
    decoder = get_decoder(model_type=model_type, device=device, encoder=model.image_encoder)
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=3, is_tiled=True,
    )


def segment_volume(model, raw, tile_shape, halo, params):
    """Segment one volume with the tiled generator, from a clean state."""
    model.clear_state()
    model.initialize(raw, ndim=3, tile_shape=tile_shape, halo=halo, **VOLUME_SPEED_OPTIONS)
    return model.generate(**params).astype("uint32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset_name", default="gonuclear", help="The 3d dataset to segment.")
    parser.add_argument("-m", "--model_type", default="hvit_t_cells", choices=FINETUNED_MODELS)
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=(384, 384), help="In-plane tile shape (y, x).")
    parser.add_argument("--halo", type=int, nargs=2, default=(64, 64), help="In-plane tile halo (y, x).")
    parser.add_argument("--sample_index", type=int, default=None, help="Score only this one sample, by index.")
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
    model = build_segmenter(args.model_type, device)
    params = resolve_params(ndim=3)

    tag = f"{args.dataset_name}_{args.model_type}"
    if args.z_crop is not None:
        tag = f"{tag}_z{args.z_crop}"
    if args.xy_crop is not None:
        tag = f"{tag}_xy{args.xy_crop}"

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

        start = time.time()
        seg = segment_volume(model, raw, tuple(args.tile_shape), tuple(args.halo), params)
        elapsed = time.time() - start
        print(f"Sample {index}: segmented in {elapsed:.1f}s, {int(seg.max())} predicted objects.")

        results = run_dataset_evaluation([labels], [seg], args.dataset_name, save_path=save_path)
        print(results)


if __name__ == "__main__":
    main()
