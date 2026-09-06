"""Check the tiled APG refinement against the non-tiled one, with a real model.

The unit tests pin the tiled refinement against a fake predictor. This script closes the loop on an
actual checkpoint, in two steps:

1. Single-tile agreement. A tiling whose one tile covers the whole image collapses every tiled step
   — the merge runs once, the id offset is zero, nothing is pruned and the prompt origin is (0, 0) —
   so the tiled result has to reproduce the non-tiled one. It is not asserted bit-for-bit, because
   the two paths reach their embeddings differently ('set_image' versus a tiled zarr and
   'set_precomputed'), and those last bits can differ.

2. Multi-tile smoke run. A genuine tiling, with and without the refinement, reporting the statistics
   that only tiling produces. A large 'dropped_negatives' means the halo is too small for the
   configured 'n_negatives', and the refinement is then working with fewer negatives than it asked
   for.

Example:
    python development/check_apg_tiled_refinement.py --image crop.tif --labels crop_labels.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning" / "v2" / "evaluation"))


def _load(path, key):
    """Read an image from a tif/png, or from a dataset of an hdf5 container."""
    if Path(path).suffix in (".h5", ".hdf5", ".n5", ".zarr"):
        import h5py
        with h5py.File(path, "r") as f:
            return f[key][:]
    import imageio.v3 as imageio
    return imageio.imread(path)


def _load_model(model_type, checkpoint, device):
    """Both halves of a joint checkpoint: the interactive branch and the decoder."""
    from common import export_joint_checkpoint, load_unisam2_model
    from micro_sam.v2.util import get_sam2_model

    interactive_path, decoder_path = export_joint_checkpoint(model_type, checkpoint)
    model = get_sam2_model(model_type=model_type, device=device, checkpoint_path=interactive_path)
    return model, load_unisam2_model(decoder_path, device, encoder=model.image_encoder)


def _build(model, decoder, device, is_tiled):
    """The APG segmenter for one tiling mode. Both share the model, so only one lands on the device."""
    from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator

    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=2, is_tiled=is_tiled,
    )


def _score(segmentation, labels):
    """Mean segmentation accuracy against the ground truth, or None without one."""
    if labels is None:
        return None
    from elf.evaluation import mean_segmentation_accuracy
    return mean_segmentation_accuracy(segmentation, labels)


def _report(name, segmentation, labels, stats):
    msa = _score(segmentation, labels)
    n_instances = len(np.unique(segmentation)) - 1
    print(f"  {name:<28} {n_instances:>5} instances" + ("" if msa is None else f"   mSA {msa:.4f}"))
    interesting = (
        "replaced_instances", "gated_consistency", "gated_foreign", "dropped_negatives",
        "stitch_dropped_instances",
    )
    reported = {key: stats[key] for key in interesting if key in stats}
    if reported:
        print(f"  {'':<28} {reported}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="The input image, a tif/png or an hdf5 container.")
    parser.add_argument("--key", default="raw", help="The dataset name, for an hdf5 input.")
    parser.add_argument("--labels", help="Optional ground truth, for the mSA of the smoke run.")
    parser.add_argument("--labels-key", default="labels", help="The dataset name, for hdf5 ground truth.")
    parser.add_argument("--model-type", default="hvit_t", help="The SAM2 backbone of the joint model.")
    parser.add_argument("--checkpoint", default="best", help="The joint trainer checkpoint to export.")
    parser.add_argument("--device", default="cuda", help="The torch device.")
    parser.add_argument("--refinement", default="points+boxes", help="The refinement mode to check.")
    parser.add_argument("--tile-shape", type=int, nargs=2, default=(256, 256), help="Tiling of the smoke run.")
    parser.add_argument("--halo", type=int, nargs=2, default=(32, 32), help="Halo of the smoke run.")
    args = parser.parse_args()

    image = _load(args.image, args.key)
    labels = None if args.labels is None else _load(args.labels, args.labels_key)
    print(f"Image {image.shape}, refinement {args.refinement!r}")

    generate_kwargs = {"refinement": args.refinement}

    model, decoder = _load_model(args.model_type, args.checkpoint, args.device)

    print("\nSingle tile, which has to reproduce the non-tiled result:")
    plain = _build(model, decoder, args.device, is_tiled=False)
    plain.initialize(image, ndim=2)
    plain_proposals = plain.propose()
    plain_plain = plain.select(plain_proposals)
    plain._last_generation_stats = {}
    plain_refined = plain.select(plain_proposals, **generate_kwargs)
    _report("non-tiled", plain_refined, labels, plain._last_generation_stats)
    plain.clear_state()

    tiled = _build(model, decoder, args.device, is_tiled=True)
    # One tile covering the image. Its outer block is clipped to the image, so the halo is irrelevant.
    tiled.initialize(image, ndim=2, tile_shape=tuple(image.shape[:2]), halo=(0, 0))
    tiled_plain = tiled.generate()
    tiled._last_generation_stats = {}
    tiled_refined = tiled.generate(**generate_kwargs)
    _report("tiled, one tile", tiled_refined, labels, tiled._last_generation_stats)
    tiled.clear_state()

    # The two paths reach their embeddings differently, so the first round already diverges a little.
    # What has to hold is that the refinement adds no divergence of its own on top of it.
    before = float((plain_plain == tiled_plain).mean())
    after = float((plain_refined == tiled_refined).mean())
    print(f"  pixel agreement before the refinement: {before:.6f}, after it: {after:.6f}")
    print(f"  identical after the refinement: {np.array_equal(plain_refined, tiled_refined)}")
    if after < before - 0.01:
        print("  PROBLEM: the refinement pulls the two paths apart, which tiling alone must not do.")

    print(f"\nSmoke run with tiles {tuple(args.tile_shape)} and halo {tuple(args.halo)}:")
    tiled.initialize(image, ndim=2, tile_shape=tuple(args.tile_shape), halo=tuple(args.halo))
    for name, kwargs in (("tiled, no refinement", {}), ("tiled, refined", generate_kwargs)):
        tiled._last_generation_stats = {}
        segmentation = tiled.generate(**kwargs)
        _report(name, segmentation, labels, tiled._last_generation_stats)
    tiled.clear_state()


if __name__ == "__main__":
    main()
