"""Evaluation of automatic and interactive 3D segmentation with micro_sam2.

Volumes are center-cropped to (8, 512, 512) before inference.

Usage examples:
    # Interactive 3D on lucchi, start with box prompt
    python evaluate_3d.py -d lucchi -i <data_root> -e <experiment_folder> --mode interactive -p box

    # Interactive 3D on embedseg, start with point, use logits masks
    python evaluate_3d.py -d embedseg -i <data_root> -e <experiment_folder> --mode interactive -p point --use_masks

    # Automatic segmentation on lucchi
    python evaluate_3d.py -d lucchi -i <data_root> -e <experiment_folder> --mode automatic
"""

import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from micro_sam import util

from micro_sam.v2.evaluation import inference, evaluation
from micro_sam.evaluation.evaluation import run_evaluation

from common import (
    CHECKPOINT_PATHS, DATA_ROOT, DATASETS_3D, get_data_paths, load_volume,
    UNISAM2_CHECKPOINT, load_unisam2_model, predict_unisam2, postprocess_unisam2,
)

CROP_SHAPE_3D = (8, 512, 512)


def run_interactive_evaluation_3d(
    dataset_name,
    data_root,
    model_type,
    backbone,
    experiment_folder,
    prompt_choice="box",
    n_iterations=8,
    use_masks=False,
    device=None,
    checkpoint_path=None,
    min_size=10,
    cleanup_predictions=False,
):
    """Run interactive 3D segmentation with iterative prompting and evaluate results."""
    if device is None:
        device = util.get_device()

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS[backbone][model_type]

    start_with_box = (prompt_choice == "box")

    raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root)

    fname_list, label_list = [], []
    for raw_path, label_path in zip(raw_paths, label_paths):
        raw, labels, _ = load_volume(
            raw_path=raw_path,
            label_path=label_path,
            raw_key=raw_key,
            label_key=label_key,
            dataset_name=dataset_name,
            crop_shape=CROP_SHAPE_3D,
        )

        fname = Path(raw_path).stem
        print(f"Processing '{fname}': raw {raw.shape}, {len(set(labels.flat)) - 1} instances")

        prediction_root = inference.run_interactive_segmentation_3d(
            raw=raw,
            labels=labels,
            model_type=model_type,
            backbone=backbone,
            checkpoint_path=checkpoint_path,
            start_with_box_prompt=start_with_box,
            prediction_dir=experiment_folder,
            prediction_fname=fname,
            device=device,
            min_size=min_size,
            n_iterations=n_iterations,
            use_masks=use_masks,
        )

        fname_list.append(fname)
        label_list.append(labels)

    evaluation.run_evaluation_for_iterative_prompting(
        labels=label_list,
        prediction_fnames=fname_list,
        prediction_dir=prediction_root,
        experiment_folder=experiment_folder,
        start_with_box_prompt=start_with_box,
        n_iterations=n_iterations,
        min_size=min_size,
        use_masks=use_masks,
    )

    if cleanup_predictions:
        shutil.rmtree(prediction_root)


def run_automatic_evaluation_3d(
    dataset_name,
    data_root,
    experiment_folder,
    device,
    checkpoint_path=None,
    crop_shape=CROP_SHAPE_3D,
):
    """Run automatic segmentation (directed distances) and evaluate on 3D volumes."""
    if checkpoint_path is None:
        checkpoint_path = UNISAM2_CHECKPOINT

    save_path = os.path.join(experiment_folder, "results", "automatic.csv")
    if os.path.exists(save_path):
        print(f"Results already stored at '{save_path}'.")
        return

    model = load_unisam2_model(checkpoint_path, device)
    raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root)

    all_gt, all_seg = [], []
    for raw_path, label_path in tqdm(
        zip(raw_paths, label_paths), total=len(raw_paths), desc="automatic 3D"
    ):
        raw, labels, valid_roi = load_volume(
            raw_path=raw_path,
            label_path=label_path,
            raw_key=raw_key,
            label_key=label_key,
            dataset_name=dataset_name,
            crop_shape=crop_shape,
        )
        fname = Path(raw_path).stem
        print(f"  {fname}: raw {raw.shape}, {len(np.unique(labels)) - 1} instances")

        out = predict_unisam2(model, raw, ndim=3, device=device)
        seg = postprocess_unisam2(out, dataset_name)
        if valid_roi is not None:
            seg[~valid_roi] = 0

        all_gt.append(labels)
        all_seg.append(seg)

    os.makedirs(os.path.join(experiment_folder, "results"), exist_ok=True)
    results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg, save_path=save_path)
    print(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate micro_sam2 for 3D segmentation.")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, choices=DATASETS_3D)
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT)
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t",
                        help="SAM2 model size (interactive mode only).")
    parser.add_argument("-b", "--backbone", type=str, default="sam2.1",
                        help="SAM2 backbone version (interactive mode only).")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=["box", "point"])
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None)
    parser.add_argument("--automatic_checkpoint", type=str, default=None)
    parser.add_argument("-iter", "--n_iterations", type=int, default=8)
    parser.add_argument("--use_masks", action="store_true", help="Use logits masks across iterations.")
    parser.add_argument("--cleanup_predictions", action="store_true",
                        help="Delete stored predictions after CSV is saved.")
    parser.add_argument(
        "--mode", type=str, default="all", choices=["all", "interactive", "automatic"],
        help="Which evaluations to run: 'all' runs interactive + automatic, or pick one.",
    )
    args = parser.parse_args()

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_all = (args.mode == "all")

    if run_all or args.mode == "interactive":
        run_interactive_evaluation_3d(
            dataset_name=args.dataset_name,
            data_root=args.input_path,
            model_type=args.model_type,
            backbone=args.backbone,
            experiment_folder=args.experiment_folder,
            prompt_choice=args.prompt_choice,
            n_iterations=args.n_iterations,
            use_masks=args.use_masks,
            checkpoint_path=args.checkpoint_path,
            cleanup_predictions=args.cleanup_predictions,
        )

    if run_all or args.mode == "automatic":
        run_automatic_evaluation_3d(
            dataset_name=args.dataset_name,
            data_root=args.input_path,
            experiment_folder=args.experiment_folder,
            device=device,
            checkpoint_path=args.automatic_checkpoint,
        )


if __name__ == "__main__":
    main()
