"""Benchmark evaluation of interactive segmentation baselines.

Supported methods:
  nninteractive: nnInteractive interactive segmentation (3D only)
  sam: Pretrained SAM v1 interactive segmentation (2D and 3D)
  sam2: Pretrained SAM2 interactive segmentation (2D and 3D)
  micro-sam: micro-sam v1 finetuned interactive (vit_b_lm LM / vit_b_em_organelles EM)
  micro_sam2: Jointly finetuned SAM2 interactive segmentation (2D and 3D)
  sam3: SAM3 interactive segmentation (2D and 3D)

Usage examples:
    python evaluate_interactive_baselines.py -d embedseg -e <exp> --method nninteractive -p box
    python evaluate_interactive_baselines.py -d embedseg -e <exp> --method nninteractive -p point -iter 4
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam2
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method micro-sam
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method micro_sam2 -m hvit_b
    python evaluate_micro_sam_volumetric.py -d embedseg -e <exp> -m vit_b_lm -p box
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam3
    python evaluate_interactive_baselines.py -d embedseg -e <exp> --method sam3 --ndim 3
"""

import os
import sys
import shutil

import imageio.v3 as imageio
import numpy as np
from skimage.measure import label as connected_components
from tqdm import tqdm

import torch

from micro_sam.v1.evaluation.evaluation import run_evaluation
from micro_sam.v2.normalization import normalize_raw

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_EM, CHECKPOINT_PATHS,
    export_joint_checkpoint, get_data_paths,
)
from baselines_common import MAX_EVALUATION_SAMPLES, _load_data
from common import check_data_download

_METHODS = ["nninteractive", "sam3", "sam", "sam2", "micro-sam", "micro_sam2"]

NNINTERACTIVE_CHECKPOINT = "/mnt/vast-nhr/home/archit/u12090/nnInteractive/pretrained_weights/nnInteractive_v1.0"
_SAM3_ROOT = "/mnt/vast-nhr/home/archit/u12090/SAM3_Experiments"

_SAM2_MODEL_TYPE = "hvit_t"
_SAM_V1_MODEL_TYPE = "vit_b"
_MICROSAM_V1_LM_MODEL = "vit_b_lm"
_MICROSAM_V1_EM_MODEL = "vit_b_em_organelles"

_EM_DATASETS = set(DATASETS_3D_EM)


def _normalize_raw_to_unit(raw):
    """Normalize raw input to float32 [0, 1] for SAM2-style preprocessing."""
    if raw.size == 0:
        return raw.astype("float32", copy=False)
    return normalize_raw(raw)


def _to_sam2_uint8(raw):
    """Convert raw input to uint8 while preserving [0, 1] normalization semantics."""
    return normalize_raw(raw, output_dtype="uint8")


def _get_corrective_point(gt_mask, pred_mask):
    """Center of the largest FN (positive) or FP (negative) region.

    Returns (coords, is_positive) or (None, None) if prediction is perfect.
    coords is a list [d0, d1, ...] matching the mask dimensionality.
    """
    fn_labeled = connected_components(gt_mask & ~pred_mask)
    fp_labeled = connected_components(~gt_mask & pred_mask)
    fn_counts = np.bincount(fn_labeled.ravel())[1:] if fn_labeled.max() > 0 else np.array([])
    fp_counts = np.bincount(fp_labeled.ravel())[1:] if fp_labeled.max() > 0 else np.array([])
    fn_max = int(fn_counts.max()) if len(fn_counts) > 0 else 0
    fp_max = int(fp_counts.max()) if len(fp_counts) > 0 else 0
    if fn_max == 0 and fp_max == 0:
        return None, None
    if fn_max >= fp_max:
        region = fn_labeled == (fn_counts.argmax() + 1)
        is_positive = True
    else:
        region = fp_labeled == (fp_counts.argmax() + 1)
        is_positive = False
    coords = [int(np.round(c.mean())) for c in np.where(region)]
    return coords, is_positive


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
    n = min(len(get_data_paths(dataset_name, data_root)[0]), MAX_EVALUATION_SAMPLES)
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels, valid_roi in tqdm(_load_data(dataset_name, data_root, ndim=3), total=n, desc="nninteractive"):
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
        results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg_per_iter[it], save_path=save_path)
        print(f"Iteration {it:02d}: {results}")


def _load_sam_v1(model_type, checkpoint, device):
    from micro_sam.v1.util import get_sam_model
    return get_sam_model(model_type=model_type, checkpoint_path=checkpoint, device=device)


def _write_sam_v1_2d_inputs(dataset_name, data_root, input_dir, gt_dir):
    image_paths, gt_paths = [], []
    n = min(len(get_data_paths(dataset_name, data_root)[0]), MAX_EVALUATION_SAMPLES)
    it = tqdm(_load_data(dataset_name, data_root, 2), total=n, desc="save-crops")
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


def _write_sam2_2d_inputs(dataset_name, data_root, input_dir, gt_dir):
    image_paths, gt_paths = [], []
    n = min(len(get_data_paths(dataset_name, data_root)[0]), MAX_EVALUATION_SAMPLES)
    it = tqdm(_load_data(dataset_name, data_root, 2), total=n, desc="save-crops")
    for sample_id, (raw, labels, _) in enumerate(it):
        if labels.max() == 0:  # Inference skips these, so they must not be scored either.
            continue

        image_path = os.path.join(input_dir, f"{sample_id:05d}.tif")
        gt_path = os.path.join(gt_dir, f"{sample_id:05d}.tif")
        imageio.imwrite(image_path, _to_sam2_uint8(raw), compression="zlib")
        imageio.imwrite(gt_path, labels.astype("uint32"), compression="zlib")
        image_paths.append(image_path)
        gt_paths.append(gt_path)
    return image_paths, gt_paths


def run_sam_v1_evaluation(
    dataset_name, data_root, experiment_folder, device,
    model_type="vit_b_lm", checkpoint=None, start_with_box=True, n_iterations=8, ndim=None, name_tag="micro-sam",
    use_masks=False,
):
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2

    if ndim == 3:
        raise ValueError(
            "micro-sam v1 3D interactive evaluation must use the volumetric implementation. "
            "Run finetuning/v2/evaluation/evaluate_micro_sam_volumetric.py instead."
        )

    if name_tag == "micro-sam" and dataset_name in _EM_DATASETS:
        raise ValueError(f"micro-sam interactive does not support EM datasets (LM model only); got '{dataset_name}'.")

    prompt_str = "box" if start_with_box else "point"
    mask_str = "with_masks" if use_masks else "without_masks"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, f"{dataset_name}_{name_tag}_{model_type}_{prompt_str}_{mask_str}_iter{it:02d}.csv")
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
        experiment_folder, "predictions", f"{name_tag}_{model_type}", dataset_name, f"{prompt_str}_{mask_str}"
    )
    input_dir = os.path.join(work_dir, "inputs", "images")
    gt_dir = os.path.join(work_dir, "inputs", "labels")
    embedding_dir = os.path.join(work_dir, "embeddings")
    prediction_dir = os.path.join(work_dir, "predictions")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    image_paths, gt_paths = _write_sam_v1_2d_inputs(dataset_name, data_root, input_dir, gt_dir)

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
        results = run_evaluation(gt_paths=gt_paths, prediction_paths=pred_paths, save_path=save_path)
        print(f"Iteration {it:02d}: {results}")

    shutil.rmtree(work_dir, ignore_errors=True)


def run_sam3_evaluation(
    dataset_name, data_root, experiment_folder,
    start_with_box=True, n_iterations=8, ndim=None,
):
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2

    sys.path.insert(0, _SAM3_ROOT)
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

    n = min(len(get_data_paths(dataset_name, data_root)[0]), MAX_EVALUATION_SAMPLES)
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels, valid_roi in tqdm(_load_data(dataset_name, data_root, ndim), total=n, desc=f"sam3-{ndim}d"):
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
        results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg_per_iter[it], save_path=save_path)
        print(f"Iteration {it:02d}: {results}")


def run_sam2_evaluation(
    dataset_name, data_root, experiment_folder, device,
    model_type=_SAM2_MODEL_TYPE, checkpoint_path=None,
    start_with_box=True, n_iterations=8, ndim=None, name_tag="sam2", use_masks=True, mask_threshold=0.0,
):
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS[model_type]

    prompt_str = "box" if start_with_box else "point"
    dim_suffix = "" if ndim == 2 else "_3d"
    # The 3d path always feeds the logits masks through the video predictor, so it has no mask tag.
    mask_str = "" if ndim == 3 else ("_with_masks" if use_masks else "_without_masks")
    if ndim == 2 and mask_threshold != 0.0:
        mask_str += f"_t{mask_threshold:g}"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(
            results_dir, f"{dataset_name}_{name_tag}_{model_type}{dim_suffix}_{prompt_str}{mask_str}_iter{it:02d}.csv"
        )
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

    from micro_sam.v2.evaluation.inference import run_interactive_segmentation_2d, run_interactive_segmentation_3d

    if ndim == 2:
        # Inputs and predictions outlive the process so a preempted or timed-out job resumes per
        # image. '/tmp' is a small RAM-backed tmpfs on the compute nodes, so it is avoided here.
        # Keyed by prompt and mask settings too, so concurrent runs of the same model and dataset do
        # not share a tree and delete each other's predictions on cleanup.
        prediction_root = os.path.join(
            experiment_folder, "predictions", f"{name_tag}_{model_type}", dataset_name, f"{prompt_str}{mask_str}"
        )
        input_dir = os.path.join(prediction_root, "inputs", "images")
        gt_dir = os.path.join(prediction_root, "inputs", "labels")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        image_paths, gt_paths = _write_sam2_2d_inputs(dataset_name, data_root, input_dir, gt_dir)

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
        for it, save_path in enumerate(save_paths):
            if os.path.exists(save_path):
                continue
            pred_dir = os.path.join(prediction_dir, f"iteration{it:02d}")
            pred_paths = [os.path.join(pred_dir, os.path.basename(path)) for path in image_paths]
            results = run_evaluation(gt_paths=gt_paths, prediction_paths=pred_paths, save_path=save_path)
            print(f"Iteration {it:02d}: {results}")

        shutil.rmtree(prediction_root, ignore_errors=True)
    else:
        n = min(len(get_data_paths(dataset_name, data_root)[0]), MAX_EVALUATION_SAMPLES)
        # Keyed by model type and dataset, since cached predictions are reused across runs and the
        # per-sample names are otherwise identical for every dataset.
        prediction_root = os.path.join(experiment_folder, "predictions", f"{name_tag}_{model_type}", dataset_name)
        all_gt = []
        all_valid_rois = []
        pred_paths_per_iter = [[] for _ in range(n_iterations)]

        for sample_id, (raw, labels, valid_roi) in enumerate(
            tqdm(_load_data(dataset_name, data_root, ndim=3), total=n, desc=f"{name_tag}-3d")
        ):
            if labels.max() == 0:  # Skip empty crops, as the 2d path does.
                continue

            sample_prediction_dir = run_interactive_segmentation_3d(
                raw=np.stack([_to_sam2_uint8(frame) for frame in raw]),
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
            for it in range(n_iterations):
                pred_paths_per_iter[it].append(
                    os.path.join(sample_prediction_dir, f"iteration{it}", f"{sample_id:05d}.tif")
                )

        os.makedirs(results_dir, exist_ok=True)
        for it, save_path in enumerate(save_paths):
            if os.path.exists(save_path):
                continue
            preds = []
            for pred_path, valid_roi in zip(pred_paths_per_iter[it], all_valid_rois):
                pred = imageio.imread(pred_path)
                if valid_roi is not None:
                    pred[~valid_roi] = 0
                preds.append(pred)
            results = run_evaluation(gt_paths=all_gt, prediction_paths=preds, save_path=save_path)
            print(f"Iteration {it:02d}: {results}")


def main():
    import argparse
    all_datasets = sorted(set(DATASETS_2D + DATASETS_3D))
    parser = argparse.ArgumentParser(description="Evaluate interactive segmentation baselines.")
    parser.add_argument("-d", "--dataset_name", required=True, choices=all_datasets)
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("--method", type=str, default="nninteractive", choices=_METHODS)
    parser.add_argument(
        "-p", "--prompt_choice", type=str, default="box", choices=["box", "point"],
        help="First prompt type (default: box)."
    )
    parser.add_argument(
        "-iter", "--n_iterations", type=int, default=8, help="Number of iterative prompting rounds (default: 8)."
    )
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Override default checkpoint path.")
    parser.add_argument(
        "-m", "--model_type", type=str, default=None,
        help="Model type override (e.g. vit_b for sam, hvit_t for sam2/micro_sam2)."
    )
    parser.add_argument(
        "--joint_checkpoint", type=str, default="best", choices=["best", "latest"],
        help="Which joint trainer checkpoint the micro_sam2 weights are taken from (default: best)."
    )
    parser.add_argument(
        "--ndim", type=int, default=None, choices=[2, 3],
        help="Dimensionality override (default: inferred from dataset)."
    )
    parser.add_argument(
        "--mask_threshold", type=float, default=0.0,
        help="Threshold on the predicted mask logits (SAM2 default 0.0). The best value is dataset "
             "dependent, so tune it rather than assuming a global optimum."
    )
    parser.add_argument(
        "--use_masks", action=argparse.BooleanOptionalAction, default=None,
        help="Feed the previous logits masks as mask prompts. Defaults to on for SAM2, off for SAM v1."
    )
    args = parser.parse_args()

    # SAM2 is trained with mask logits on every correction click, SAM v1 only with a probability.
    use_masks_sam2 = True if args.use_masks is None else args.use_masks
    use_masks_sam_v1 = bool(args.use_masks)

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
        mt = args.model_type or _SAM_V1_MODEL_TYPE
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="sam",
            use_masks=use_masks_sam_v1,
        )

    elif args.method == "micro-sam":
        is_em = args.dataset_name in _EM_DATASETS
        mt = args.model_type or (_MICROSAM_V1_EM_MODEL if is_em else _MICROSAM_V1_LM_MODEL)
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="micro-sam",
            use_masks=use_masks_sam_v1,
        )

    elif args.method == "sam2":
        mt = args.model_type or _SAM2_MODEL_TYPE
        run_sam2_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, checkpoint_path=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
            name_tag="sam2", use_masks=use_masks_sam2, mask_threshold=args.mask_threshold,
        )

    elif args.method == "micro_sam2":
        mt = args.model_type or _SAM2_MODEL_TYPE
        # The joint checkpoint bundles both branches, so it is split on first use.
        checkpoint = args.checkpoint or export_joint_checkpoint(mt, args.joint_checkpoint)[0]
        # 'best' keeps the plain tag so existing results stay addressable.
        tag = "micro_sam2" if args.joint_checkpoint == "best" else f"micro_sam2_{args.joint_checkpoint}"
        run_sam2_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, checkpoint_path=checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
            name_tag=tag, use_masks=use_masks_sam2, mask_threshold=args.mask_threshold,
        )

    else:
        raise ValueError


if __name__ == "__main__":
    main()
