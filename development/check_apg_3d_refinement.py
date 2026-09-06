"""Check the volumetric APG refinement against a real model and a real crop.

The unit tests pin the mechanism against a fake predictor: which prompts are assembled, which gate
fires, what reaches the propagator. This script closes the loop on an actual checkpoint, where the
questions are different ones:

1. Does the refinement change anything? A refined run whose segmentation equals the baseline's means
   the conditioning never reached the propagation, however many instances the counters claim were
   replaced.
2. What does it cost? The whole case for refining the anchor slice rather than re-propagating is that
   it adds one 2d forward per anchor slice and no propagation, so the generation time should barely
   move. The reported frame steps are the check that it really did not: they are a function of the
   passes and the depth, not of the timing noise.
3. Do 'prompts' and 'mask' conditioning differ? They seed the same anchor frame two different ways,
   so if they agree bit for bit, one of the two paths is not doing what it says.

Example:
    python development/check_apg_3d_refinement.py --image crop.h5 --key raw \
        --labels crop.h5 --labels-key labels --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning" / "v2" / "evaluation"))

# The modes worth a look on a real volume, as (refinement, refinement_kwargs).
CHECKS = {
    "none": (None, {}),
    "points": ("points", {}),
    "boxes": ("boxes", {}),
    "points+boxes": ("points+boxes", {}),
    "points+boxes/mask": ("points+boxes", {"conditioning": "mask"}),
    "points+boxes/ungated": ("points+boxes", {"min_consistency": None, "max_foreign_overlap": None}),
    "recover": ("recover", {}),
    "points+boxes+recover": ("points+boxes+recover", {}),
}
DEFAULT_MODES = ("none", "boxes", "points+boxes", "points+boxes/mask", "recover")


def _load(path, key):
    """Read a volume from an hdf5/zarr dataset, or from a tif stack."""
    if Path(path).suffix in (".h5", ".hdf5", ".n5", ".zarr"):
        import h5py
        with h5py.File(path, "r") as f:
            return f[key][:]
    import imageio.v3 as imageio
    return imageio.imread(path)


def _load_model(model_type, checkpoint, device):
    """Both halves of a joint checkpoint. A volume is prompted with the video predictor."""
    from common import export_joint_checkpoint, load_unisam2_model
    from micro_sam.v2.util import get_sam2_model

    interactive_path, decoder_path = export_joint_checkpoint(model_type, checkpoint)
    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=interactive_path, input_type="videos",
    )
    return model, load_unisam2_model(decoder_path, device, encoder=model.image_encoder)


def _score(segmentation, labels):
    """Mean segmentation accuracy against the ground truth, or None without one."""
    if labels is None:
        return None
    from elf.evaluation import mean_segmentation_accuracy
    return mean_segmentation_accuracy(segmentation, labels)


def _report(name, segmentation, labels, stats, seconds, baseline):
    msa = _score(segmentation, labels)
    n_instances = len(np.unique(segmentation)) - 1
    line = f"  {name:<24} {n_instances:>5} objects  {seconds:>7.1f} s"
    if msa is not None:
        line += f"  mSA {msa:.6f}"
    if baseline is not None:
        agreement = float((segmentation == baseline).mean())
        line += f"  agreement {agreement:.6f}"
    print(line)
    interesting = (
        "scored_candidates", "propagation_passes", "propagated_frame_steps", "refined_candidates",
        "replaced_candidates", "gated_consistency", "gated_foreign", "recovery_candidates",
        "recovered_candidates",
    )
    print(f"  {'':<24} " + "  ".join(f"{key}={stats[key]}" for key in interesting if key in stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="The input volume, an hdf5 container or a tif stack.")
    parser.add_argument("--key", default="raw", help="The dataset name, for an hdf5 input.")
    parser.add_argument("--labels", help="Optional ground truth, for the mSA of each mode.")
    parser.add_argument("--labels-key", default="labels", help="The dataset name, for hdf5 ground truth.")
    parser.add_argument("--model-type", default="hvit_t", help="The SAM2 backbone of the joint model.")
    parser.add_argument("--checkpoint", default="best", help="The joint trainer checkpoint to export.")
    parser.add_argument("--device", default="cuda", help="The torch device.")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=list(CHECKS))
    parser.add_argument("--spacing", type=float, nargs=3, help="Anisotropic voxel spacing, e.g. 4 1 1.")
    args = parser.parse_args()

    from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator

    volume = _load(args.image, args.key)
    labels = None if args.labels is None else _load(args.labels, args.labels_key)
    print(f"Volume {volume.shape}, modes {args.modes}")

    model, decoder = _load_model(args.model_type, args.checkpoint, args.device)
    segmenter = get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=args.device, ndim=3,
    )
    # The volume is encoded once and every mode reads the same embeddings, as the benchmark does.
    segmenter.initialize(volume, ndim=3, offload_to_cpu=False)

    results = {}
    baseline = None
    for name in args.modes:
        refinement, refinement_kwargs = CHECKS[name]
        segmenter._last_generation_stats = {}
        start = time.perf_counter()
        segmentation = segmenter.generate(
            refinement=refinement, refinement_kwargs=refinement_kwargs or None,
            spacing=None if args.spacing is None else tuple(args.spacing),
        )
        seconds = time.perf_counter() - start
        _report(name, segmentation, labels, segmenter._last_generation_stats, seconds, baseline)
        results[name] = segmentation
        if name == "none":
            baseline = segmentation
    segmenter.clear_state()

    print()
    if baseline is not None:
        for name, segmentation in results.items():
            if name != "none" and np.array_equal(segmentation, baseline):
                print(f"  PROBLEM: {name!r} equals the unrefined result, so its conditioning never landed.")
    prompts, mask = results.get("points+boxes"), results.get("points+boxes/mask")
    if prompts is not None and mask is not None and np.array_equal(prompts, mask):
        print("  PROBLEM: 'prompts' and 'mask' conditioning agree bit for bit, which they should not.")


if __name__ == "__main__":
    main()
