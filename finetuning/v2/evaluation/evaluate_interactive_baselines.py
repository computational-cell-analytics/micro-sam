"""Benchmark evaluation of interactive segmentation baselines.

Supported methods:
  nninteractive: nnInteractive interactive segmentation (3D only)
  sam: Pretrained SAM v1 interactive segmentation (2D and 3D)
  sam2: Pretrained SAM2 interactive segmentation (2D and 3D)
  micro-sam: micro-sam v1 finetuned interactive (vit_b_lm LM / vit_b_em_organelles EM)
  micro_sam2: Finetuned SAM2 interactive segmentation (2D and 3D)
  sam3: SAM3 interactive segmentation (2D and 3D)

Usage examples:
    python evaluate_interactive_baselines.py -d lucchi -e <exp> --method nninteractive -p box
    python evaluate_interactive_baselines.py -d lucchi -e <exp> --method nninteractive -p point -iter 4
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam2
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method micro-sam
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method micro_sam2
    python evaluate_micro_sam_volumetric.py -d embedseg -e <exp> -m vit_b_lm -p box
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam3
    python evaluate_interactive_baselines.py -d lucchi -e <exp> --method sam3 --ndim 3
"""

import os
import sys
import tempfile

import imageio.v3 as imageio
import numpy as np
from skimage.measure import label as connected_components
from torch_em.transform.raw import normalize
from tqdm import tqdm

import torch

from micro_sam.evaluation.evaluation import run_evaluation

from common import DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_EM, CHECKPOINT_PATHS, get_data_paths
from baselines_common import MAX_EVALUATION_SAMPLES, _load_data

_METHODS = ["nninteractive", "sam3", "sam", "sam2", "micro-sam", "micro_sam2"]

NNINTERACTIVE_CHECKPOINT = "/mnt/vast-nhr/home/archit/u12090/nnInteractive/pretrained_weights/nnInteractive_v1.0"
_SAM3_ROOT = "/mnt/vast-nhr/home/archit/u12090/SAM3_Experiments"

MICROSAM2_INTERACTIVE_CHECKPOINT = (
    "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/interactive/v1/checkpoints/checkpoint.pt"
)

_SAM2_BACKBONE = "sam2.1"
_SAM2_MODEL_TYPE = "hvit_t"
_SAM_V1_MODEL_TYPE = "vit_b"
_MICROSAM_V1_LM_MODEL = "vit_b_lm"
_MICROSAM_V1_EM_MODEL = "vit_b_em_organelles"

_EM_DATASETS = set(DATASETS_3D_EM)


def _normalize_raw_to_unit(raw):
    """Normalize raw input to float32 [0, 1] for SAM2-style preprocessing."""
    raw = raw.astype("float32", copy=False)
    if raw.size == 0:
        return raw
    if raw.min() < 0 or raw.max() > 1:
        raw = normalize(raw)
    return np.clip(raw, 0, 1)


def _to_sam2_uint8(raw):
    """Convert raw input to uint8 while preserving [0, 1] normalization semantics."""
    return np.round(_normalize_raw_to_unit(raw) * 255).astype("uint8")


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
    from micro_sam.util import get_sam_model
    return get_sam_model(model_type=model_type, checkpoint_path=checkpoint, device=device)


def _write_sam_v1_2d_inputs(dataset_name, data_root, input_dir, gt_dir):
    image_paths, gt_paths = [], []
    n = min(len(get_data_paths(dataset_name, data_root)[0]), MAX_EVALUATION_SAMPLES)
    for sample_id, (raw, labels, _) in enumerate(tqdm(_load_data(dataset_name, data_root, 2), total=n, desc="save-crops")):
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
    for sample_id, (raw, labels, _) in enumerate(tqdm(_load_data(dataset_name, data_root, 2), total=n, desc="save-crops")):
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
            "micro-sam v1 3D interactive evaluation should use the volumetric implementation. "
            "Run finetuning/v2/evaluation/evaluate_micro_sam_volumetric.py instead."
        )

    if name_tag == "micro-sam" and dataset_name in _EM_DATASETS:
        raise ValueError(f"micro-sam interactive does not support EM datasets (LM model only); got '{dataset_name}'.")

    prompt_str = "box" if start_with_box else "point"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, f"{dataset_name}_{name_tag}_{model_type}_{prompt_str}_iter{it:02d}.csv")
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

    predictor = _load_sam_v1(model_type, checkpoint, device)
    from micro_sam.evaluation.inference import run_inference_with_iterative_prompting

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "images")
        gt_dir = os.path.join(tmpdir, "labels")
        prediction_dir = os.path.join(tmpdir, "predictions")
        embedding_dir = os.path.join(tmpdir, "embeddings")
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
    model_type=_SAM2_MODEL_TYPE, backbone=_SAM2_BACKBONE, checkpoint_path=None,
    start_with_box=True, n_iterations=8, ndim=None, name_tag="sam2", use_masks=False,
):
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS[backbone][model_type]

    prompt_str = "box" if start_with_box else "point"
    dim_suffix = "" if ndim == 2 else "_3d"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, f"{dataset_name}_{name_tag}_{model_type}{dim_suffix}_{prompt_str}_iter{it:02d}.csv")
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

    from micro_sam.v2.evaluation.inference import run_interactive_segmentation_2d, run_interactive_segmentation_3d

    if ndim == 2:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "images")
            gt_dir = os.path.join(tmpdir, "labels")
            prediction_dir = os.path.join(tmpdir, "predictions")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            image_paths, gt_paths = _write_sam2_2d_inputs(dataset_name, data_root, input_dir, gt_dir)

            prediction_dir = run_interactive_segmentation_2d(
                image_paths=image_paths,
                gt_paths=gt_paths,
                image_key=None,
                gt_key=None,
                prediction_dir=prediction_dir,
                model_type=model_type,
                backbone=backbone,
                checkpoint_path=checkpoint_path,
                start_with_box_prompt=start_with_box,
                device=device,
                n_iterations=n_iterations,
                use_masks=use_masks,
                ensure_8bit=False,
            )

            os.makedirs(results_dir, exist_ok=True)
            for it, save_path in enumerate(save_paths):
                if os.path.exists(save_path):
                    continue
                pred_dir = os.path.join(prediction_dir, f"iteration{it:02d}")
                pred_paths = [os.path.join(pred_dir, os.path.basename(path)) for path in image_paths]
                results = run_evaluation(gt_paths=gt_paths, prediction_paths=pred_paths, save_path=save_path)
                print(f"Iteration {it:02d}: {results}")
    else:
        n = min(len(get_data_paths(dataset_name, data_root)[0]), MAX_EVALUATION_SAMPLES)
        prediction_root = os.path.join(experiment_folder, "predictions", name_tag)
        all_gt = []
        all_valid_rois = []
        pred_paths_per_iter = [[] for _ in range(n_iterations)]

        for sample_id, (raw, labels, valid_roi) in enumerate(
            tqdm(_load_data(dataset_name, data_root, ndim=3), total=n, desc=f"{name_tag}-3d")
        ):
            sample_prediction_dir = run_interactive_segmentation_3d(
                raw=np.stack([_to_sam2_uint8(frame) for frame in raw]),
                labels=labels,
                model_type=model_type,
                backbone=backbone,
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
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="Override default checkpoint path."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default=None,
        help="Model type override (e.g. vit_b for sam, hvit_t for sam2/micro_sam2)."
    )
    parser.add_argument(
        "--ndim", type=int, default=None, choices=[2, 3],
        help="Dimensionality override (default: inferred from dataset)."
    )
    parser.add_argument(
        "--use_masks", action="store_true",
        help="Use previous logits/masks as mask prompts during SAM2 iterative prompting."
    )
    args = parser.parse_args()

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
            use_masks=args.use_masks,
        )

    elif args.method == "micro-sam":
        is_em = args.dataset_name in _EM_DATASETS
        mt = args.model_type or (_MICROSAM_V1_EM_MODEL if is_em else _MICROSAM_V1_LM_MODEL)
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="micro-sam",
            use_masks=args.use_masks,
        )

    elif args.method == "sam2":
        mt = args.model_type or _SAM2_MODEL_TYPE
        run_sam2_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, backbone=_SAM2_BACKBONE,
            checkpoint_path=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
            name_tag="sam2", use_masks=args.use_masks,
        )

    elif args.method == "micro_sam2":
        mt = args.model_type or _SAM2_MODEL_TYPE
        run_sam2_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, backbone=_SAM2_BACKBONE,
            checkpoint_path=args.checkpoint or MICROSAM2_INTERACTIVE_CHECKPOINT,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
            name_tag="micro_sam2", use_masks=args.use_masks,
        )

    else:
        raise ValueError


if __name__ == "__main__":
    main()
