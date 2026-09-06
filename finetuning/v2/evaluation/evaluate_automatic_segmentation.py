"""Evaluation of micro-sam2 automatic segmentation, for 2d and 3d, LM and EM.

Two modes share one pipeline:
  ais: the jointly finetuned UniSAM2 decoder, turned into instances by the flow / multicut postprocessing.
  apg: the same decoder proposes candidates, which the interactive branch turns into masks.

The parameters are tuned separately, once per dataset and mode, by `parameter_search.py`. This
script only reads what that sweep found (see `common.read_tuned_params`) and scores the test split
with it. A dataset with no data held out from the evaluation keeps the library defaults, because a
sweep there would select its parameters on the very samples the reported score is measured on. See
`common.VAL_SPLITS` for what counts as held out.

Usage examples:
    python evaluate_automatic_segmentation.py -d livecell -m hvit_b -e <exp> --mode ais
    python evaluate_automatic_segmentation.py -d gonuclear -m hvit_b -e <exp> --mode apg
    python evaluate_automatic_segmentation.py -d cremi -m hvit_b -e <exp> --mode ais --skip_tuning
"""

import os
import json
import hashlib
import argparse
import warnings

import pandas as pd
from tqdm import tqdm

import torch

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASET_SPACING, GT_MIN_SIZE_2D, MODEL_TYPES, MODES,
    VOLUME_SPEED_OPTIONS, build_model, check_data_download, drop_severed_objects, genuine_misses,
    has_val_split, load_apg_overrides, load_data, n_samples, postprocess_unisam2, predict_unisam2,
    read_tuned_params, resolve_checkpoint_identity, run_dataset_evaluation,
)


def segment(model, mode, raw, ndim, dataset_name, model_type, params, device, spacing=None, devices=None):
    """Segment one sample with the tuned parameters of a mode."""
    if mode == "apg":
        model.clear_state()
        model.initialize(raw, ndim=ndim, **(VOLUME_SPEED_OPTIONS if ndim == 3 else {}))
        volume_params = {"spacing": spacing} if ndim == 3 else {}
        return model.generate(**{**volume_params, **params}).astype("uint32")

    prediction = predict_unisam2(model, raw, ndim=ndim, device=device, devices=devices)
    return postprocess_unisam2(prediction, dataset_name, model_type=model_type, params=params)


def run_evaluation(
    model, mode, dataset_name, data_root, experiment_folder, model_type, params, device,
    crop_shape=None, checkpoint_id=None, devices=None, tuned=None, result_tag=None, config_name=None,
    artifacts=None,
):
    """Score the test split with the given parameters and write the result CSV.

    The parameters land in the result file next to the metrics, so a number can always be traced back
    to the run that produced it.

    Args:
        model: The model of the mode, from `common.build_model`.
        mode: The segmentation mode, one of MODES.
        dataset_name: The dataset to score.
        data_root: The root the data lives in.
        experiment_folder: The folder the results are written to.
        model_type: The SAM2 backbone, which names the result file.
        params: The parameters to segment with, or None for the library defaults.
        device: The torch device.
        crop_shape: The 3d center crop.
        checkpoint_id: The checksum of all model weights used by the mode.
        devices: The devices inference spreads over. All visible GPUs by default.
        tuned: Whether 'params' came from the tuning sweep. Names the result file 'tuned' or
            'default'; by default inferred from whether there are parameters at all.
        result_tag: Optional tag appended to the result file name, so that a run with explicit
            parameter overrides or learned artifacts does not collide with the plain evaluation.
        config_name: The name of the configuration the overrides came from, stored in the results.
        artifacts: Paths of learned artifacts installed on the model, stored as checksums.

    Returns:
        The results as a DataFrame.
    """
    if tuned is None:
        tuned = bool(params)
    tag = "tuned" if tuned else "default"
    if result_tag:
        tag = f"{tag}_{result_tag}"
    legacy_path = os.path.join(
        experiment_folder, "results", f"{dataset_name}_micro_sam2_{model_type}_{mode}_{tag}.csv"
    )
    save_path = legacy_path if checkpoint_id is None else legacy_path[:-4] + f"_ckpt-{checkpoint_id}.csv"
    if os.path.exists(save_path):
        print(f"Results already stored at '{save_path}'.")
        return pd.read_csv(save_path)
    if checkpoint_id is not None and os.path.exists(legacy_path):
        warnings.warn(
            f"Using legacy evaluation result '{legacy_path}'. It has no checkpoint checksum, so its "
            "weights cannot be verified."
        )
        return pd.read_csv(legacy_path)

    ndim = 3 if dataset_name in DATASETS_3D else 2
    spacing = DATASET_SPACING.get(dataset_name)
    border_min_size = GT_MIN_SIZE_2D.get(dataset_name, 0) if ndim == 2 else 0
    total = n_samples(dataset_name, data_root)
    samples = load_data(dataset_name, data_root, ndim, crop_shape=crop_shape)

    all_gt, all_seg, misses = [], [], []
    for raw, labels, valid_roi in tqdm(samples, total=total, desc=f"{mode}-{model_type}"):
        if labels.max() == 0:  # Nothing to score without ground-truth.
            continue
        seg = segment(
            model, mode, raw, ndim, dataset_name, model_type, params or {}, device, spacing=spacing,
            devices=devices,
        )
        if valid_roi is not None:
            seg[~valid_roi] = 0
        if ndim == 2:
            # The ground truth has no severed objects either, so predicting one is not a false positive.
            seg = drop_severed_objects(seg, border_min_size)
        else:
            misses.append(genuine_misses(labels, seg))
        all_gt.append(labels)
        all_seg.append(seg)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = run_dataset_evaluation(all_gt, all_seg, dataset_name, save_path)
    if misses:
        # The aggregate metric hides which objects went missing.
        results["unmatched"] = sum(count[0] for count in misses)
        results["genuine_misses"] = sum(count[1] for count in misses)
    results["parameters"] = json.dumps(params, sort_keys=True, default=str) if params else "default"
    if config_name is not None:
        results["config_name"] = config_name
    if artifacts:
        checksums = {
            name: hashlib.sha256(open(path, "rb").read()).hexdigest() for name, path in sorted(artifacts.items())
        }
        results["artifacts"] = json.dumps(checksums, sort_keys=True)
    results.to_csv(save_path, index=False)
    print(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", required=True, choices=sorted(DATASETS_2D + DATASETS_3D))
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t", choices=MODEL_TYPES)
    parser.add_argument("--mode", type=str, default="ais", choices=MODES, help="The segmentation mode to evaluate.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Weights instead of the joint export.")
    parser.add_argument("--joint_checkpoint", type=str, default="best", help="Joint checkpoint name, without '.pt'.")
    parser.add_argument(
        "--interactive_checkpoint", type=str, default=None,
        help="Standalone interactive weights for --mode apg, bypassing the joint checkpoint entirely. "
             "Requires -c/--checkpoint for the decoder half too.",
    )
    parser.add_argument("--skip_tuning", action="store_true", help="Evaluate with the library defaults.")
    parser.add_argument("--tuning_root", type=str, default=None, help="Where parameter_search.py wrote its sweeps.")
    parser.add_argument("--crop_3d", type=int, nargs=3, default=None, help="Override the 3d crop (Z Y X).")
    parser.add_argument(
        "--propagation_waves", type=int, default=None,
        help="Volumes only. Rounds the candidates are propagated in, overriding whatever was tuned.",
    )
    parser.add_argument("--devices", nargs="*", default=None, help="Inference devices. All visible GPUs by default.")
    parser.add_argument(
        "--apg_params", type=str, default=None,
        help="APG only. A benchmark-style JSON configuration whose 'params_2d' are layered over the tuned "
             "parameters (or the defaults with --skip_tuning).",
    )
    parser.add_argument(
        "--multimask_scorer_artifact", type=str, default=None,
        help="APG 2d only. Fitted feature scorer used by multimask_scorer='microscopy'.",
    )
    parser.add_argument(
        "--refinement_gate_artifact", type=str, default=None,
        help="APG 2d only. Fitted utility scorer used by refinement_kwargs.gate='uncertainty'.",
    )
    parser.add_argument(
        "--result_tag", type=str, default=None,
        help="Tag appended to the result file name. Defaults to the --apg_params configuration name.",
    )
    args = parser.parse_args()

    check_data_download(args.dataset_name, args.input_path)
    learned = (args.apg_params, args.multimask_scorer_artifact, args.refinement_gate_artifact)
    if any(option is not None for option in learned) and args.mode != "apg":
        parser.error("--apg_params and the learned artifacts apply to --mode apg only.")
    if (args.multimask_scorer_artifact or args.refinement_gate_artifact) and args.dataset_name in DATASETS_3D:
        parser.error("The learned multimask scorer and refinement gate support 2d datasets only.")

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ndim = 3 if args.dataset_name in DATASETS_3D else 2
    crop_shape = tuple(args.crop_3d) if args.crop_3d else None
    checkpoint_id, joint_checksum = resolve_checkpoint_identity(
        args.mode, args.model_type, args.joint_checkpoint, args.checkpoint,
        interactive_checkpoint_path=args.interactive_checkpoint,
    )
    model = build_model(
        args.mode, args.model_type, device, ndim,
        joint_checkpoint=args.joint_checkpoint, checkpoint_path=args.checkpoint,
        joint_checksum=joint_checksum, interactive_checkpoint_path=args.interactive_checkpoint,
        devices=args.devices or None,
    )
    artifacts = {
        name: path for name, path in (
            ("multimask_scorer", args.multimask_scorer_artifact),
            ("refinement_gate", args.refinement_gate_artifact),
        ) if path is not None
    }
    if artifacts:
        from micro_sam.v2.multimask_selection import load_feature_scorer
        model.set_multimask_models(
            scorer=(
                load_feature_scorer(args.multimask_scorer_artifact, device=device)
                if args.multimask_scorer_artifact else None
            ),
            refinement_gate=(
                load_feature_scorer(args.refinement_gate_artifact, device=device)
                if args.refinement_gate_artifact else None
            ),
        )

    params = None
    tuned = False
    if not args.skip_tuning:
        if has_val_split(args.dataset_name):
            tuning_root = args.tuning_root or os.path.join(args.experiment_folder, "tuning")
            tuning_root = os.path.join(tuning_root, args.mode)
            try:
                params = read_tuned_params(tuning_root, args.dataset_name, args.model_type, checkpoint_id)
                tuned = True
            except FileNotFoundError:
                warnings.warn(
                    f"No tuned parameters for '{args.dataset_name}' under '{tuning_root}'. Run "
                    "parameter_search.py first, or pass --skip_tuning. Using the library defaults."
                )
        else:
            print(f"'{args.dataset_name}' has no data held out from the evaluation, so the defaults are used.")

    if args.propagation_waves is not None:
        # On top of the tuned set, or on its own: a volume takes it either way, and the defaults for
        # everything else are what 'generate' applies when a key is missing.
        params = {**(params or {}), "propagation_waves": args.propagation_waves}

    config_name, result_tag = None, args.result_tag
    if args.apg_params is not None:
        config_name, overrides = load_apg_overrides(args.apg_params)
        params = {**(params or {}), **overrides}
        if result_tag is None:
            result_tag = config_name

    run_evaluation(
        model, args.mode, args.dataset_name, args.input_path, args.experiment_folder, args.model_type,
        params, device, crop_shape=crop_shape, checkpoint_id=checkpoint_id,
        devices=args.devices or None, tuned=tuned, result_tag=result_tag, config_name=config_name,
        artifacts=artifacts or None,
    )


if __name__ == "__main__":
    main()
