"""Evaluation of SAM2 interactive segmentation, for 2d and 3d, LM and EM.

Runs the jointly finetuned micro-sam2 branch by default, and the pretrained SAM2 backbone with
'--weights pretrained'. Both go through the same inference, so the finetuning is the only difference
between the two numbers, which is what makes them comparable.

The model is prompted iteratively: the first prompt is a box or a point on
every ground-truth object, and every later iteration adds a correction click derived from the error
of the previous one. One CSV is written per iteration, so the whole correction curve is reported.

There is nothing to tune here. The settings that change the numbers are fixed to the combination the
model was trained for: the previous mask logits are fed back on every 2d correction click (SAM2 is
trained with them, see 'SAM2Train._iter_correct_pt_sampling'), and the volumetric path propagates
through the video predictor. Only the test split is scored.

Usage examples:
    python evaluate_interactive_segmentation.py -d livecell -m hvit_b -e <exp> -p box
    python evaluate_interactive_segmentation.py -d livecell -m hvit_b -e <exp> -p point --no-use_masks
    python evaluate_interactive_segmentation.py -d embedseg -m hvit_b -e <exp> -p box
    python evaluate_interactive_segmentation.py -d livecell -m hvit_t -e <exp> --weights pretrained
"""

import os
import shutil
import argparse
import warnings

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

import torch

from micro_sam.v2.normalization import normalize_raw
from micro_sam.v2.evaluation.inference import run_interactive_segmentation_2d, run_interactive_segmentation_3d

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D, MODEL_TYPES, CROP_SHAPE_3D, CHECKPOINT_PATHS,
    check_data_download, checkpoint_checksum, export_joint_checkpoint, get_joint_checkpoint,
    interactive_result_name, interactive_run_tag, load_data, n_samples, run_dataset_evaluation,
)


def resolve_weights(weights: str, model_type: str, joint_checkpoint: str, checkpoint=None):
    """The checkpoint to prompt with, and the name its results are filed under.

    New results include a content checksum. The plain tag is also returned so finished evaluation
    runs from before checksum-based naming remain usable.

    Args:
        weights: 'joint' for the finetuned branch, 'pretrained' for the SAM2 backbone.
        model_type: The SAM2 backbone, e.g. 'hvit_t'.
        joint_checkpoint: The joint trainer checkpoint, without the '.pt' suffix.
        checkpoint: An explicit checkpoint path, which overrides both.

    Returns:
        The checkpoint path, checksum-qualified result tag and legacy result tag.
    """
    if weights == "pretrained":
        path = checkpoint or CHECKPOINT_PATHS[model_type]
        legacy_tag = "sam2"
        return path, f"{legacy_tag}_ckpt-{checkpoint_checksum(path)}", legacy_tag

    legacy_tag = "micro_sam2" if joint_checkpoint == "best" else f"micro_sam2_{joint_checkpoint}"
    if checkpoint is not None:
        return checkpoint, f"{legacy_tag}_ckpt-{checkpoint_checksum(checkpoint)}", legacy_tag

    # The joint checkpoint bundles both branches, so it is split on first use.
    source_checksum = checkpoint_checksum(get_joint_checkpoint(model_type, joint_checkpoint))
    path = export_joint_checkpoint(model_type, joint_checkpoint, source_checksum=source_checksum)[0]
    return path, f"{legacy_tag}_ckpt-{source_checksum}", legacy_tag


def completed_result_paths(save_paths, legacy_paths):
    """Return a complete exact or legacy result set, preferring checksum-qualified results."""
    if all(os.path.exists(path) for path in save_paths):
        return save_paths
    if all(os.path.exists(path) for path in legacy_paths):
        warnings.warn(
            f"Using legacy evaluation results in '{os.path.dirname(legacy_paths[0])}'. They have no "
            "checkpoint checksum, so their weights cannot be verified."
        )
        return legacy_paths
    return None


def to_uint8(raw):
    """Convert raw input to uint8, which is what the SAM2 preprocessing expects."""
    return normalize_raw(raw, output_dtype="uint8")


def write_2d_inputs(dataset_name, data_root, input_dir, gt_dir, min_size=0):
    """Write the cropped images and labels the 2d inference reads, and return their paths.

    The inference works off files so that a preempted job resumes per image rather than per dataset.
    """
    image_paths, gt_paths = [], []
    total = n_samples(dataset_name, data_root)
    samples = tqdm(load_data(dataset_name, data_root, 2, min_size), total=total, desc="save-crops")
    for sample_id, (raw, labels, _) in enumerate(samples):
        if labels.max() == 0:  # Inference skips these, so they must not be scored either.
            continue

        image_path = os.path.join(input_dir, f"{sample_id:05d}.tif")
        gt_path = os.path.join(gt_dir, f"{sample_id:05d}.tif")
        imageio.imwrite(image_path, to_uint8(raw), compression="zlib")
        imageio.imwrite(gt_path, labels.astype("uint32"), compression="zlib")
        image_paths.append(image_path)
        gt_paths.append(gt_path)
    return image_paths, gt_paths


def run_interactive_evaluation_2d(
    dataset_name, data_root, experiment_folder, device, model_type, checkpoint_path, tag, legacy_tag,
    start_with_box=True, n_iterations=8, use_masks=True, mask_threshold=0.0, min_size=0,
):
    """Run iterative prompting on the 2d test split and write one result CSV per iteration."""
    prompt = "box" if start_with_box else "point"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, interactive_result_name(
            dataset_name, tag, model_type, prompt, iteration,
            ndim=2, use_masks=use_masks, mask_threshold=mask_threshold, min_size=min_size,
        ))
        for iteration in range(n_iterations)
    ]
    legacy_paths = [
        os.path.join(results_dir, interactive_result_name(
            dataset_name, legacy_tag, model_type, prompt, iteration,
            ndim=2, use_masks=use_masks, mask_threshold=mask_threshold, min_size=min_size,
        ))
        for iteration in range(n_iterations)
    ]
    if completed_result_paths(save_paths, legacy_paths) is not None:
        print(f"Results already stored at '{results_dir}'.")
        return

    # Keyed by every setting that changes the predictions, so concurrent runs of the same model and
    # dataset never share a tree and delete each other's predictions on cleanup. '/tmp' is a small
    # RAM-backed tmpfs on the compute nodes, so it is avoided here.
    prediction_root = os.path.join(
        experiment_folder, "predictions", f"{tag}_{model_type}", dataset_name,
        f"{prompt}{interactive_run_tag(2, use_masks, mask_threshold, min_size)}",
    )
    input_dir = os.path.join(prediction_root, "inputs", "images")
    gt_dir = os.path.join(prediction_root, "inputs", "labels")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    image_paths, gt_paths = write_2d_inputs(dataset_name, data_root, input_dir, gt_dir, min_size)

    prediction_dir = run_interactive_segmentation_2d(
        image_paths=image_paths,
        gt_paths=gt_paths,
        image_key=None,
        gt_key=None,
        prediction_dir=prediction_root,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        start_with_box_prompt=start_with_box,
        device=device,
        n_iterations=n_iterations,
        use_masks=use_masks,
        ensure_8bit=False,
        mask_threshold=mask_threshold,
    )

    os.makedirs(results_dir, exist_ok=True)
    for iteration, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        pred_dir = os.path.join(prediction_dir, f"iteration{iteration:02d}")
        pred_paths = [os.path.join(pred_dir, os.path.basename(path)) for path in image_paths]
        results = run_dataset_evaluation(gt_paths, pred_paths, dataset_name, save_path)
        print(f"Iteration {iteration:02d}: {results}")

    shutil.rmtree(prediction_root, ignore_errors=True)


def run_interactive_evaluation_3d(
    dataset_name, data_root, experiment_folder, device, model_type, checkpoint_path, tag, legacy_tag,
    start_with_box=True, n_iterations=8, min_size=0, crop_shape=CROP_SHAPE_3D,
):
    """Run iterative prompting on the 3d test split and write one result CSV per iteration.

    The prompts of a volume are placed on the middle slice of each object and propagated through the
    video predictor, so a correction click acts on the whole object rather than on one slice.
    """
    prompt = "box" if start_with_box else "point"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, interactive_result_name(
            dataset_name, tag, model_type, prompt, iteration, ndim=3, min_size=min_size,
        ))
        for iteration in range(n_iterations)
    ]
    legacy_paths = [
        os.path.join(results_dir, interactive_result_name(
            dataset_name, legacy_tag, model_type, prompt, iteration, ndim=3, min_size=min_size,
        ))
        for iteration in range(n_iterations)
    ]
    if completed_result_paths(save_paths, legacy_paths) is not None:
        print(f"Results already stored at '{results_dir}'.")
        return

    # Keyed by model type and dataset, since the cached predictions are reused across runs and the
    # per-sample names are otherwise identical for every dataset. 'min_size' changes the ground truth
    # and therefore the prompts, so it has to key the cache too.
    prediction_root = os.path.join(
        experiment_folder, "predictions", f"{tag}_{model_type}",
        dataset_name if not min_size else f"{dataset_name}_min{min_size}",
    )
    total = n_samples(dataset_name, data_root)
    samples = load_data(dataset_name, data_root, 3, min_size=min_size, crop_shape=crop_shape)

    all_gt, all_valid_rois = [], []
    pred_paths_per_iter = [[] for _ in range(n_iterations)]
    for sample_id, (raw, labels, valid_roi) in enumerate(tqdm(samples, total=total, desc=f"{tag}-3d")):
        if labels.max() == 0:  # Nothing to prompt, and nothing to score.
            continue

        sample_prediction_dir = run_interactive_segmentation_3d(
            raw=np.stack([to_uint8(frame) for frame in raw]),
            labels=labels,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            start_with_box_prompt=start_with_box,
            prediction_dir=os.path.join(prediction_root, f"sample_{sample_id:05d}"),
            prediction_fname=f"{sample_id:05d}.tif",
            device=device,
            n_iterations=n_iterations,
        )
        all_gt.append(labels)
        all_valid_rois.append(valid_roi)
        for iteration in range(n_iterations):
            pred_paths_per_iter[iteration].append(
                os.path.join(sample_prediction_dir, f"iteration{iteration}", f"{sample_id:05d}.tif")
            )

    os.makedirs(results_dir, exist_ok=True)
    for iteration, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        predictions = []
        for pred_path, valid_roi in zip(pred_paths_per_iter[iteration], all_valid_rois):
            prediction = imageio.imread(pred_path)
            if valid_roi is not None:
                prediction[~valid_roi] = 0
            predictions.append(prediction)
        results = run_dataset_evaluation(all_gt, predictions, dataset_name, save_path)
        print(f"Iteration {iteration:02d}: {results}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", required=True, choices=sorted(DATASETS_2D + DATASETS_3D))
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t", choices=MODEL_TYPES)
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=("box", "point"))
    parser.add_argument("-iter", "--n_iterations", type=int, default=8, help="Iterative prompting rounds.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Weights instead of the joint export.")
    parser.add_argument("--joint_checkpoint", type=str, default="best", help="Joint checkpoint name, without '.pt'.")
    parser.add_argument("--weights", type=str, default="joint", choices=("joint", "pretrained"),
                        help="Prompt the finetuned joint branch, or the pretrained SAM2 backbone.")
    parser.add_argument("--ndim", type=int, default=None, choices=(2, 3), help="Defaults to the dataset's own.")
    parser.add_argument(
        "--min_size", type=int, default=0,
        help="Drop ground-truth objects below this many pixels, from both prompting and scoring. "
             "Cropping leaves unrecoverable slivers at the crop faces."
    )
    parser.add_argument(
        "--mask_threshold", type=float, default=0.0,
        help="Threshold on the predicted mask logits (SAM2 default 0.0). 2d only."
    )
    parser.add_argument(
        "--use_masks", action=argparse.BooleanOptionalAction, default=True,
        help="Feed the previous logits masks back as mask prompts. 2d only, on by default."
    )
    parser.add_argument("--crop_3d", type=int, nargs=3, default=None, help="Override the 3d crop (Z Y X).")
    args = parser.parse_args()

    check_data_download(args.dataset_name, args.input_path)

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ndim = args.ndim or (3 if args.dataset_name in DATASETS_3D else 2)
    checkpoint, tag, legacy_tag = resolve_weights(
        args.weights, args.model_type, args.joint_checkpoint, args.checkpoint
    )

    if ndim == 2:
        run_interactive_evaluation_2d(
            args.dataset_name, args.input_path, args.experiment_folder, device, args.model_type,
            checkpoint, tag, legacy_tag,
            start_with_box=(args.prompt_choice == "box"), n_iterations=args.n_iterations,
            use_masks=args.use_masks, mask_threshold=args.mask_threshold, min_size=args.min_size,
        )
    else:
        run_interactive_evaluation_3d(
            args.dataset_name, args.input_path, args.experiment_folder, device, args.model_type,
            checkpoint, tag, legacy_tag,
            start_with_box=(args.prompt_choice == "box"), n_iterations=args.n_iterations,
            min_size=args.min_size, crop_shape=tuple(args.crop_3d) if args.crop_3d else CROP_SHAPE_3D,
        )


if __name__ == "__main__":
    main()
