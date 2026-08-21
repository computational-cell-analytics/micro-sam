"""Benchmark evaluation of the interactive segmentation baselines, i.e. everything but micro-sam2.

Supported methods:
  nninteractive: nnInteractive interactive segmentation (3d only)
  sam: Pretrained SAM v1 interactive segmentation (2d only)
  sam3: SAM3 interactive segmentation (2d and 3d)
  micro-sam: micro-sam v1 finetuned interactive, slice-wise (vit_b_lm for LM, vit_b_em_organelles for EM)
  microsam_vol: micro-sam v1 finetuned interactive, volumetric projection (3d LM only)

The SAM2 engine itself, pretrained or jointly finetuned, is evaluated by
evaluate_interactive_segmentation.py, which those two share.

Usage examples:
    python evaluate_interactive_baselines.py -d embedseg -e <exp> --method nninteractive -p box
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam3
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method micro-sam
    python evaluate_interactive_baselines.py -d embedseg -e <exp> --method microsam_vol -m vit_b_lm -p box
"""

import os
import sys
import shutil
import argparse

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from tqdm import tqdm
from skimage.measure import label as connected_components

import torch

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_EM,
    check_data_download, interactive_result_name, interactive_run_tag, load_data, n_samples,
    run_dataset_evaluation,
)

METHODS = ["nninteractive", "sam3", "sam", "micro-sam", "microsam_vol"]

NNINTERACTIVE_CHECKPOINT = "/mnt/vast-nhr/home/archit/u12090/nnInteractive/pretrained_weights/nnInteractive_v1.0"
SAM3_ROOT = "/mnt/vast-nhr/home/archit/u12090/SAM3_Experiments"

SAM_V1_MODEL_TYPE = "vit_b"
MICROSAM_V1_LM_MODEL = "vit_b_lm"
MICROSAM_V1_EM_MODEL = "vit_b_em_organelles"

EM_DATASETS = set(DATASETS_3D_EM)


def _get_largest_region_center(mask):
    labeled = connected_components(mask)
    counts = np.bincount(labeled.ravel())[1:] if labeled.max() > 0 else np.array([])
    if len(counts) == 0:
        return None
    region = labeled == (counts.argmax() + 1)
    return [int(np.round(c.mean())) for c in np.where(region)]


def _get_correction_points(gt_mask, pred_mask):
    """Return one positive FN point and one negative FP point if available."""
    positive = _get_largest_region_center(gt_mask & ~pred_mask)
    negative = _get_largest_region_center(~gt_mask & pred_mask)
    return positive, negative


def _get_middle_slice_prompt(gt_mask):
    """Return a representative z slice and the 2D object mask on this slice."""
    z_indices = np.where(gt_mask)[0]
    z_mid = int(np.round((int(z_indices.min()) + int(z_indices.max())) / 2.0))
    z_values = np.unique(z_indices)
    z = min(z_values, key=lambda zz: abs(int(zz) - z_mid))
    mask_2d = gt_mask[z]
    return int(z), mask_2d


def _load_nninteractive(checkpoint_path, device):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    session = nnInteractiveInferenceSession(device=torch.device(device), verbose=False)
    session.initialize_from_trained_model_folder(checkpoint_path, use_fold=0)
    return session


def _segment_nninteractive_iterative(volume, labels, session, start_with_box, n_iterations):
    # set_image resets the session (nullifies target_buffer), so set_target_buffer must come after.
    session.set_image(volume[np.newaxis].astype("float32"))  # [1, Z, Y, X]
    buffer = np.zeros(volume.shape, dtype="float32")
    session.set_target_buffer(buffer)

    gt_ids = np.unique(labels)[1:]
    seg_per_iter = [np.zeros(volume.shape, dtype="uint32") for _ in range(n_iterations)]

    for gt_id in gt_ids:
        gt_mask = labels == gt_id
        session.reset_interactions()
        z, gt_mask_2d = _get_middle_slice_prompt(gt_mask)
        yx_coords = np.where(gt_mask_2d)

        if start_with_box:
            bbox = [
                [z, z + 1],
                [int(yx_coords[0].min()), int(yx_coords[0].max()) + 1],
                [int(yx_coords[1].min()), int(yx_coords[1].max()) + 1],
            ]
            session.add_bbox_interaction(bbox, include_interaction=True)
        else:
            center = (z, int(np.round(yx_coords[0].mean())), int(np.round(yx_coords[1].mean())))
            session.add_point_interaction(center, include_interaction=True)

        pred_mask = buffer > 0.5
        seg_per_iter[0][pred_mask] = gt_id

        for it in range(1, n_iterations):
            positive_point, negative_point = _get_correction_points(gt_mask, pred_mask)
            if positive_point is not None:
                session.add_point_interaction(tuple(positive_point), include_interaction=True)
                pred_mask = buffer > 0.5
            if negative_point is not None:
                session.add_point_interaction(tuple(negative_point), include_interaction=False)
                pred_mask = buffer > 0.5
            seg_per_iter[it][pred_mask] = gt_id

    return seg_per_iter


def run_nninteractive_evaluation(
    dataset_name, data_root, experiment_folder, device,
    checkpoint_path=None, start_with_box=True, n_iterations=8,
):
    if dataset_name not in DATASETS_3D:
        raise ValueError(f"nnInteractive is 3D-only; got '{dataset_name}'.")
    if checkpoint_path is None:
        checkpoint_path = NNINTERACTIVE_CHECKPOINT

    prompt_str = "box" if start_with_box else "point"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, f"{dataset_name}_nninteractive_{prompt_str}_iter{it:02d}.csv")
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

    session = _load_nninteractive(checkpoint_path, device)
    n = n_samples(dataset_name, data_root)
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels, valid_roi in tqdm(load_data(dataset_name, data_root, ndim=3), total=n, desc="nninteractive"):
        segs = _segment_nninteractive_iterative(raw, labels, session, start_with_box, n_iterations)
        all_gt.append(labels)
        for it, seg in enumerate(segs):
            if valid_roi is not None:
                seg[~valid_roi] = 0
            all_seg_per_iter[it].append(seg)

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        results = run_dataset_evaluation(all_gt, all_seg_per_iter[it], dataset_name, save_path)
        print(f"Iteration {it:02d}: {results}")


def _load_sam_v1(model_type, checkpoint, device):
    from micro_sam.v1.util import get_sam_model
    return get_sam_model(model_type=model_type, checkpoint_path=checkpoint, device=device)


def _write_2d_inputs(dataset_name, data_root, input_dir, gt_dir, min_size=0):
    """Write the cropped images and labels the SAM v1 2d inference reads, and return their paths."""
    image_paths, gt_paths = [], []
    n = n_samples(dataset_name, data_root)
    it = tqdm(load_data(dataset_name, data_root, 2, min_size), total=n, desc="save-crops")
    for sample_id, (raw, labels, _) in enumerate(it):
        if labels.max() == 0:  # Inference skips these, so they must not be scored either.
            continue

        image_path = os.path.join(input_dir, f"{sample_id:05d}.tif")
        gt_path = os.path.join(gt_dir, f"{sample_id:05d}.tif")
        raw = np.clip(np.round(raw), 0, 255).astype("uint8")
        imageio.imwrite(image_path, raw, compression="zlib")
        imageio.imwrite(gt_path, labels.astype("uint32"), compression="zlib")
        image_paths.append(image_path)
        gt_paths.append(gt_path)
    return image_paths, gt_paths


def run_sam_v1_evaluation(
    dataset_name, data_root, experiment_folder, device,
    model_type="vit_b_lm", checkpoint=None, start_with_box=True, n_iterations=8, ndim=None, name_tag="micro-sam",
    use_masks=False, min_size=0,
):
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2

    if ndim == 3:
        raise ValueError(
            "micro-sam v1 3D interactive evaluation must use the volumetric implementation. "
            "Run finetuning/v2/evaluation/evaluate_micro_sam_volumetric.py instead."
        )

    if name_tag == "micro-sam" and dataset_name in EM_DATASETS:
        raise ValueError(f"micro-sam interactive does not support EM datasets (LM model only); got '{dataset_name}'.")

    prompt_str = "box" if start_with_box else "point"
    run_tag = interactive_run_tag(ndim=2, use_masks=use_masks, min_size=min_size)
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, interactive_result_name(
            dataset_name, name_tag, model_type, prompt_str, it,
            ndim=2, use_masks=use_masks, min_size=min_size,
        ))
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

    predictor = _load_sam_v1(model_type, checkpoint, device)
    from micro_sam.v1.evaluation.inference import run_inference_with_iterative_prompting

    # Inputs, embeddings and predictions outlive the process so a preempted or timed-out job resumes
    # per image. '/tmp' is a small RAM-backed tmpfs on the compute nodes, so it is avoided here.
    work_dir = os.path.join(
        experiment_folder, "predictions", f"{name_tag}_{model_type}", dataset_name, f"{prompt_str}{run_tag}"
    )
    input_dir = os.path.join(work_dir, "inputs", "images")
    gt_dir = os.path.join(work_dir, "inputs", "labels")
    embedding_dir = os.path.join(work_dir, "embeddings")
    prediction_dir = os.path.join(work_dir, "predictions")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    image_paths, gt_paths = _write_2d_inputs(dataset_name, data_root, input_dir, gt_dir, min_size)

    run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_dir,
        prediction_dir=prediction_dir,
        start_with_box_prompt=start_with_box,
        n_iterations=n_iterations,
        use_masks=use_masks,
    )

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        pred_dir = os.path.join(prediction_dir, f"iteration{it:02d}")
        pred_paths = [os.path.join(pred_dir, os.path.basename(path)) for path in image_paths]
        results = run_dataset_evaluation(gt_paths, pred_paths, dataset_name, save_path)
        print(f"Iteration {it:02d}: {results}")

    shutil.rmtree(work_dir, ignore_errors=True)


def run_sam3_evaluation(
    dataset_name, data_root, experiment_folder,
    start_with_box=True, n_iterations=8, ndim=None,
):
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2

    sys.path.insert(0, SAM3_ROOT)
    from micro_sam3.evaluation.inference import (
        build_sam3_image_predictor, build_sam3_video_predictor,
        run_interactive_segmentation_2d_sam3, run_interactive_segmentation_3d_sam3,
    )

    prompt_str = "box" if start_with_box else "point"
    dim_suffix = "" if ndim == 2 else "_3d"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, f"{dataset_name}_sam3{dim_suffix}_{prompt_str}_iter{it:02d}.csv")
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

    if ndim == 2:
        model, processor = build_sam3_image_predictor()
        predictor = None
    else:
        predictor = build_sam3_video_predictor()
        model, processor = None, None

    n = n_samples(dataset_name, data_root)
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels, valid_roi in tqdm(load_data(dataset_name, data_root, ndim), total=n, desc=f"sam3-{ndim}d"):
        if ndim == 2:
            segs = run_interactive_segmentation_2d_sam3(
                image=raw, gt=labels, model=model, processor=processor,
                start_with_box_prompt=start_with_box, n_iterations=n_iterations,
            )
        else:
            segs = run_interactive_segmentation_3d_sam3(
                raw=raw, gt=labels, predictor=predictor,
                start_with_box_prompt=start_with_box, n_iterations=n_iterations,
            )
        all_gt.append(labels)
        for it, seg in enumerate(segs):
            if valid_roi is not None:
                seg[~valid_roi] = 0
            all_seg_per_iter[it].append(seg)

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        results = run_dataset_evaluation(all_gt, all_seg_per_iter[it], dataset_name, save_path)
        print(f"Iteration {it:02d}: {results}")


def run_microsam_volumetric_evaluation(
    dataset_name, data_root, experiment_folder, model_type=None, checkpoint=None,
    prompt_choice="box", full_grid_search=False, store_segmentation=True, min_size=0,
):
    """Evaluate micro-sam v1 with its volumetric projection instead of slice-wise prompting.

    The prompt of an object is placed on one slice and projected through the volume, which is the
    mode the micro-sam v1 annotator offers for 3d data.
    """
    from micro_sam.v1.evaluation.multi_dimensional_segmentation import (
        run_multi_dimensional_segmentation_grid_search
    )

    if dataset_name not in DATASETS_3D:
        raise ValueError(f"The volumetric micro-sam v1 evaluation is 3d only; got '{dataset_name}'.")
    if dataset_name in EM_DATASETS:
        raise ValueError(f"Volumetric micro-sam v1 only supports LM datasets; got '{dataset_name}'.")

    if model_type is None:
        model_type = MICROSAM_V1_LM_MODEL

    interactive_seg_mode = "points" if prompt_choice == "point" else "box"
    # The projection settings the annotator defaults to. A full sweep is opt-in, since it re-runs
    # the projection for every combination.
    grid_search_values = None if full_grid_search else {
        "iou_threshold": [0.8], "projection": ["mask"], "box_extension": [0.025],
    }

    n = n_samples(dataset_name, data_root)
    rows = []
    it = tqdm(load_data(dataset_name, data_root, 3, min_size), total=n, desc="microsam_vol")
    for sample_id, (raw, labels, _) in enumerate(it):
        sample_name = f"sample_{sample_id:05d}"
        result_dir = os.path.join(
            experiment_folder, "results", f"{dataset_name}_microsam_vol_{model_type}_{prompt_choice}", sample_name,
        )
        embedding_path = os.path.join(
            experiment_folder, "embeddings", f"{dataset_name}_microsam_vol_{model_type}", sample_name,
        )

        best_params_path = run_multi_dimensional_segmentation_grid_search(
            volume=raw,
            ground_truth=labels,
            model_type=model_type,
            checkpoint_path=checkpoint,
            embedding_path=embedding_path,
            result_dir=result_dir,
            interactive_seg_mode=interactive_seg_mode,
            grid_search_values=grid_search_values,
            min_size=min_size,
            store_segmentation=store_segmentation,
            verbose=False,
        )

        best_params = pd.read_csv(best_params_path)
        best_params.insert(0, "sample_id", sample_id)
        rows.append(best_params)

    if not rows:
        return

    summary_path = os.path.join(
        experiment_folder, "results", f"{dataset_name}_microsam_vol_{model_type}_{prompt_choice}.csv",
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(summary_path, index=False)
    print(f"Stored the summary at '{summary_path}'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", required=True, choices=sorted(set(DATASETS_2D + DATASETS_3D)))
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=("box", "point"))
    parser.add_argument("-iter", "--n_iterations", type=int, default=8, help="Iterative prompting rounds.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Override the default checkpoint path.")
    parser.add_argument(
        "-m", "--model_type", type=str, default=None,
        help="Model type override, e.g. vit_b for sam, hvit_t for sam2."
    )
    parser.add_argument("--ndim", type=int, default=None, choices=(2, 3), help="Defaults to the dataset's own.")
    parser.add_argument(
        "--min_size", type=int, default=0,
        help="Drop ground-truth objects below this many pixels, from both prompting and scoring. "
             "Cropping leaves unrecoverable slivers at the crop faces."
    )
    parser.add_argument(
        "--use_masks", action="store_true",
        help="Feed the previous logits masks back as mask prompts. SAM v1 is not trained with them."
    )
    parser.add_argument("--full_grid_search", action="store_true", help="Sweep the projection of microsam_vol.")
    args = parser.parse_args()

    check_data_download(args.dataset_name, args.input_path)

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_with_box = (args.prompt_choice == "box")

    if args.method == "nninteractive":
        run_nninteractive_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, checkpoint_path=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations,
        )

    elif args.method == "sam3":
        run_sam3_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
        )

    elif args.method == "sam":
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=args.model_type or SAM_V1_MODEL_TYPE, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="sam",
            use_masks=args.use_masks, min_size=args.min_size,
        )

    elif args.method == "micro-sam":
        is_em = args.dataset_name in EM_DATASETS
        model_type = args.model_type or (MICROSAM_V1_EM_MODEL if is_em else MICROSAM_V1_LM_MODEL)
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=model_type, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="micro-sam",
            use_masks=args.use_masks, min_size=args.min_size,
        )

    elif args.method == "microsam_vol":
        run_microsam_volumetric_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            model_type=args.model_type, checkpoint=args.checkpoint, prompt_choice=args.prompt_choice,
            full_grid_search=args.full_grid_search, min_size=args.min_size,
        )

    else:
        raise ValueError(f"Unknown method: '{args.method}'.")


if __name__ == "__main__":
    main()
