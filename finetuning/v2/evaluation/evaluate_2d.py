"""Evaluation of automatic and interactive 2D segmentation with micro_sam2.

Usage examples:
    # AMG on livecell
    python evaluate_2d.py -d livecell -i <data_root> -e <experiment_folder> --mode amg

    # Interactive (iterative prompting, start with box) on livecell
    python evaluate_2d.py -d livecell -i <data_root> -e <experiment_folder> --mode interactive -p box

    # Interactive with logits mask propagation, start with point
    python evaluate_2d.py -d livecell -i <data_root> -e <experiment_folder> --mode interactive -p point --use_masks
"""

import os
import shutil
from glob import glob
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import label as connected_components

import torch
from elf.evaluation import mean_segmentation_accuracy
from elf.io import open_file

from micro_sam.evaluation.evaluation import run_evaluation

from micro_sam.v2.evaluation import inference

from common import (
    CHECKPOINT_PATHS, DATA_ROOT, DATASETS_2D, get_data_paths,
    UNISAM2_CHECKPOINT, load_unisam2_model, predict_unisam2, postprocess_unisam2,
)

CROP_SHAPE_2D = (512, 512)


def _center_crop_roi_2d(shape, crop_shape):
    roi = []
    for s, c in zip(shape[:2], crop_shape):
        c = min(c, s)
        start = (s - c) // 2
        roi.append(slice(start, start + c))
    return tuple(roi)


def _read_2d(path, key):
    """Read a 2D (or 2D+channel) array from an image file or from an H5/zarr file using key."""
    if key is not None:
        arr = open_file(path, mode="r")[key][:]
    else:
        arr = np.asarray(imageio.imread(path))
    # Transpose channel-first (C, H, W) arrays to channel-last (H, W, C).
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
        arr = arr.transpose(1, 2, 0)
    return arr


def _save_2d_crops(
    image_paths, gt_paths, experiment_folder, crop_shape=CROP_SHAPE_2D,
    raw_key=None, label_key=None,
):
    """Save center-cropped 2D images and GT labels; return paths to the crops.

    Crops are stored under <experiment_folder>/data_crops/ and are skipped on
    subsequent calls when the files already exist.
    """
    crop_dir = os.path.join(experiment_folder, "data_crops")
    img_dir = os.path.join(crop_dir, "images")
    gt_dir = os.path.join(crop_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    cropped_image_paths, cropped_gt_paths = [], []
    for img_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Saving 2D crops"
    ):
        fname = Path(img_path).stem + ".tif"
        out_img = os.path.join(img_dir, fname)
        out_gt = os.path.join(gt_dir, fname)

        if not os.path.exists(out_img):
            image = _read_2d(img_path, raw_key)
            roi = _center_crop_roi_2d(image.shape, crop_shape)
            imageio.imwrite(out_img, image[roi], compression=5)

        if not os.path.exists(out_gt):
            gt = _read_2d(gt_path, label_key)
            roi = _center_crop_roi_2d(gt.shape, crop_shape)
            imageio.imwrite(out_gt, gt[roi], compression=5)

        cropped_image_paths.append(out_img)
        cropped_gt_paths.append(out_gt)

    return sorted(cropped_image_paths), sorted(cropped_gt_paths)


def run_amg_evaluation(
    image_paths,
    gt_paths,
    experiment_folder,
    model_type,
    backbone,
    device,
    checkpoint_path=None,
    raw_key=None,
    label_key=None,
    cleanup_predictions=False,
):
    """Run automatic mask generation (AMG) and evaluate on 2D crops."""
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS[backbone][model_type]

    cropped_image_paths, cropped_gt_paths = _save_2d_crops(
        image_paths, gt_paths, experiment_folder, raw_key=raw_key, label_key=label_key,
    )

    inference_root = inference.run_amg(
        image_paths=cropped_image_paths,
        image_key=None,
        experiment_folder=experiment_folder,
        model_type=model_type,
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    save_path = os.path.join(experiment_folder, "results", "amg.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prediction_paths = sorted(glob(os.path.join(inference_root, "*")))
    results = run_evaluation(gt_paths=cropped_gt_paths, prediction_paths=prediction_paths, save_path=save_path)
    print(results)

    if cleanup_predictions:
        shutil.rmtree(inference_root)


def run_interactive_evaluation_2d(
    image_paths,
    gt_paths,
    experiment_folder,
    model_type,
    backbone,
    device,
    start_with_box=True,
    use_masks=False,
    checkpoint_path=None,
    n_iterations=8,
    raw_key=None,
    label_key=None,
    cleanup_predictions=False,
):
    """Run iterative-prompting interactive segmentation and evaluate on 2D crops."""
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS[backbone][model_type]

    cropped_image_paths, cropped_gt_paths = _save_2d_crops(
        image_paths, gt_paths, experiment_folder, raw_key=raw_key, label_key=label_key,
    )

    prediction_root = inference.run_interactive_segmentation_2d(
        image_paths=cropped_image_paths,
        gt_paths=cropped_gt_paths,
        image_key=None,
        gt_key=None,
        prediction_dir=experiment_folder,
        model_type=model_type,
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        start_with_box_prompt=start_with_box,
        device=device,
        n_iterations=n_iterations,
        use_masks=use_masks,
    )

    mask_tag = "with" if use_masks else "without"
    result_folder = os.path.join(experiment_folder, "results", f"iterative_prompting_{mask_tag}_mask")
    os.makedirs(result_folder, exist_ok=True)

    prompt_tag = "start_box" if start_with_box else "start_point"
    csv_path = os.path.join(result_folder, f"iterative_prompts_{prompt_tag}.csv")

    if os.path.exists(csv_path):
        print(f"Results already stored at '{csv_path}'.")
        return

    list_of_results = []
    prediction_folders = sorted(glob(os.path.join(prediction_root, "iteration*")))
    for pred_folder in prediction_folders:
        print("Evaluating", os.path.basename(pred_folder))
        pred_paths = sorted(glob(os.path.join(pred_folder, "*")))

        msas, sa50s, sa75s = [], [], []
        for gt_path, pred_path in tqdm(
            zip(cropped_gt_paths, pred_paths), desc="Evaluate", total=len(cropped_gt_paths)
        ):
            gt = imageio.imread(gt_path)
            gt = connected_components(gt)
            pred = imageio.imread(pred_path)
            msa, scores = mean_segmentation_accuracy(pred, gt, return_accuracies=True)
            msas.append(msa)
            sa50s.append(scores[0])
            sa75s.append(scores[5])

        list_of_results.append(pd.DataFrame.from_dict({
            "mSA": [np.mean(msas)], "SA50": [np.mean(sa50s)], "SA75": [np.mean(sa75s)],
        }))

    res_df = pd.concat(list_of_results, ignore_index=True)
    res_df.to_csv(csv_path)
    print(res_df)

    if cleanup_predictions:
        shutil.rmtree(prediction_root)


def run_automatic_evaluation_2d(
    image_paths,
    gt_paths,
    experiment_folder,
    device,
    dataset_name,
    checkpoint_path=None,
    crop_shape=CROP_SHAPE_2D,
    raw_key=None,
    label_key=None,
):
    """Run automatic segmentation (directed distances) and evaluate on 2D crops."""
    if checkpoint_path is None:
        checkpoint_path = UNISAM2_CHECKPOINT

    save_path = os.path.join(experiment_folder, "results", "automatic.csv")
    if os.path.exists(save_path):
        print(f"Results already stored at '{save_path}'.")
        return

    model = load_unisam2_model(checkpoint_path, device)

    cropped_image_paths, cropped_gt_paths = _save_2d_crops(
        image_paths, gt_paths, experiment_folder, crop_shape, raw_key=raw_key, label_key=label_key,
    )

    all_gt, all_seg = [], []
    for img_path, gt_path in tqdm(
        zip(cropped_image_paths, cropped_gt_paths),
        total=len(cropped_image_paths),
        desc="Automatic",
    ):
        image = imageio.imread(img_path).astype("float32")
        gt = connected_components(imageio.imread(gt_path)).astype("uint32")
        out = predict_unisam2(model, image, ndim=2, device=device)
        seg = postprocess_unisam2(out, dataset_name)
        all_gt.append(gt)
        all_seg.append(seg)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg, save_path=save_path)
    print(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate micro_sam2 for 2D segmentation.")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, choices=DATASETS_2D)
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT)
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t",
                        help="SAM2 model size (amg/interactive modes only).")
    parser.add_argument("-b", "--backbone", type=str, default="sam2.1",
                        help="SAM2 backbone version (amg/interactive modes only).")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=["box", "point"])
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None)
    parser.add_argument("--automatic_checkpoint", type=str, default=None)
    parser.add_argument("-iter", "--n_iterations", type=int, default=8)
    parser.add_argument("--use_masks", action="store_true", help="Use logits masks across iterations.")
    parser.add_argument("--cleanup_predictions", action="store_true",
                        help="Delete stored predictions after CSV is saved.")
    parser.add_argument(
        "--mode", type=str, default="all", choices=["all", "amg", "interactive", "automatic"],
        help="Which evaluations to run: 'all' runs AMG + interactive + automatic, or pick one.",
    )
    args = parser.parse_args()

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths, gt_paths, raw_key, label_key = get_data_paths(args.dataset_name, args.input_path)
    run_all = (args.mode == "all")

    if run_all or args.mode == "amg":
        run_amg_evaluation(
            image_paths=image_paths,
            gt_paths=gt_paths,
            experiment_folder=args.experiment_folder,
            model_type=args.model_type,
            backbone=args.backbone,
            device=device,
            checkpoint_path=args.checkpoint_path,
            raw_key=raw_key,
            label_key=label_key,
            cleanup_predictions=args.cleanup_predictions,
        )

    if run_all or args.mode == "interactive":
        run_interactive_evaluation_2d(
            image_paths=image_paths,
            gt_paths=gt_paths,
            experiment_folder=args.experiment_folder,
            model_type=args.model_type,
            backbone=args.backbone,
            device=device,
            start_with_box=(args.prompt_choice == "box"),
            use_masks=args.use_masks,
            checkpoint_path=args.checkpoint_path,
            n_iterations=args.n_iterations,
            raw_key=raw_key,
            label_key=label_key,
            cleanup_predictions=args.cleanup_predictions,
        )

    if run_all or args.mode == "automatic":
        run_automatic_evaluation_2d(
            image_paths=image_paths,
            gt_paths=gt_paths,
            experiment_folder=args.experiment_folder,
            device=device,
            dataset_name=args.dataset_name,
            checkpoint_path=args.automatic_checkpoint,
            raw_key=raw_key,
            label_key=label_key,
        )


if __name__ == "__main__":
    main()
