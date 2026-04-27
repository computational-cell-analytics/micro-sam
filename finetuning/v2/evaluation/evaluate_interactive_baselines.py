"""Benchmark evaluation of interactive segmentation baselines.

Supported methods:
  nninteractive: nnInteractive interactive segmentation (3D only)
  sam: Pretrained SAM v1 interactive segmentation (2D and 3D)
  sam2: Pretrained SAM2 interactive segmentation (2D and 3D)
  micro-sam: micro-sam v1 finetuned interactive (vit_b_lm LM / vit_b_em_organelles_v1 EM)
  micro-samv2: Finetuned SAM2 interactive segmentation (2D and 3D)
  microsam1: micro-sam v1 interactive segmentation with iterative prompting (2D and 3D)
  sam3: SAM3 interactive segmentation (2D and 3D)

Usage examples:
    python evaluate_interactive_baselines.py -d lucchi   -e <exp> --method nninteractive -p box
    python evaluate_interactive_baselines.py -d lucchi   -e <exp> --method nninteractive -p point -iter 4
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam2
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method micro-sam
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method micro-samv2
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method microsam1
    python evaluate_interactive_baselines.py -d embedseg -e <exp> --method microsam1 --ndim 3
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam3
    python evaluate_interactive_baselines.py -d lucchi   -e <exp> --method sam3 --ndim 3
"""

import os
import sys
import tempfile

import numpy as np
from skimage.measure import label as connected_components
from torch_em.transform.raw import normalize
from tqdm import tqdm

import torch

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import _get_batched_prompts

from common import DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_EM, CHECKPOINT_PATHS, get_data_paths
from baselines_common import _load_data

_METHODS = ["nninteractive", "microsam1", "sam3", "sam", "sam2", "micro-sam", "micro-samv2"]

NNINTERACTIVE_CHECKPOINT = "/mnt/vast-nhr/home/archit/u12090/nnInteractive/pretrained_weights/nnInteractive_v1.0"
_SAM3_ROOT = "/mnt/vast-nhr/home/archit/u12090/SAM3_Experiments"

MICROSAM2_INTERACTIVE_CHECKPOINT = (
    "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/interactive/v1/checkpoints/checkpoint.pt"
)

_SAM2_BACKBONE = "sam2.1"
_SAM2_MODEL_TYPE = "hvit_t"
_SAM_V1_MODEL_TYPE = "vit_b"
_MICROSAM_V1_LM_MODEL = "vit_b_lm"
_MICROSAM_V1_EM_MODEL = "vit_b_em_organelles_v1"

_EM_DATASETS = set(DATASETS_3D_EM)


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

    for obj_id, gt_id in enumerate(gt_ids, start=1):
        gt_mask = labels == gt_id
        coords = np.where(gt_mask)
        session.reset_interactions()

        if start_with_box:
            bbox = [[int(coords[i].min()), int(coords[i].max()) + 1] for i in range(3)]
            session.add_bbox_interaction(bbox, include_interaction=True)
        else:
            center = tuple(int(np.round(coords[i].mean())) for i in range(3))
            session.add_point_interaction(center, include_interaction=True)

        seg_per_iter[0][buffer > 0.5] = obj_id

        for it in range(1, n_iterations):
            point_coords, is_positive = _get_corrective_point(gt_mask, buffer > 0.5)
            if point_coords is not None:
                session.add_point_interaction(tuple(point_coords), include_interaction=is_positive)
            seg_per_iter[it][buffer > 0.5] = obj_id

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
    n = len(get_data_paths(dataset_name, data_root)[0])
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels in tqdm(_load_data(dataset_name, data_root, ndim=3), total=n, desc="nninteractive"):
        segs = _segment_nninteractive_iterative(raw, labels, session, start_with_box, n_iterations)
        all_gt.append(labels)
        for it, seg in enumerate(segs):
            all_seg_per_iter[it].append(seg)

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg_per_iter[it], save_path=save_path)
        print(f"Iteration {it:02d}: {results}")


def _load_microsam1(model_type, checkpoint, device):
    from micro_sam.util import get_sam_model
    return get_sam_model(model_type=model_type, checkpoint_path=checkpoint, device=device)


def _segment_microsam1_2d_iterative(image, gt, predictor, start_with_box, n_iterations, dilation=5):
    """Iterative 2D interactive segmentation with micro-sam v1 SAM predictor.

    Returns a list of n_iterations segmentation arrays (H, W) uint32.
    """
    H, W = image.shape[:2]
    gt_ids = np.unique(gt)[1:]

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    predictor.set_image(image.astype("uint8"))

    seg_per_iter = [np.zeros((H, W), dtype="uint32") for _ in range(n_iterations)]

    for gt_id in gt_ids:
        gt_mask = gt == gt_id
        binary_gt = gt_mask.astype("uint32")

        if start_with_box:
            _, _, raw_boxes = _get_batched_prompts(
                gt=binary_gt, gt_ids=[1], use_points=False, use_boxes=True,
                n_positives=1, n_negatives=0, dilation=dilation,
            )
            x, y, w, h = raw_boxes[0]
            box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            masks, scores, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=box_xyxy, multimask_output=False,
            )
            acc_points, acc_labels = None, None
        else:
            raw_pts, raw_lbls, _ = _get_batched_prompts(
                gt=binary_gt, gt_ids=[1], use_points=True, use_boxes=False,
                n_positives=1, n_negatives=0, dilation=0,
            )
            acc_points = np.array(raw_pts[0], dtype=np.float32)  # (1, 2) xy
            acc_labels = np.array(raw_lbls[0])                   # (1,)
            masks, scores, _ = predictor.predict(
                point_coords=acc_points, point_labels=acc_labels,
                multimask_output=True,
            )
            masks = masks[np.argmax(scores)][None]

        pred_mask = masks[0] > 0
        seg_per_iter[0][pred_mask] = gt_id

        for it in range(1, n_iterations):
            yx, positive = _get_corrective_point(gt_mask, pred_mask)
            if yx is not None:
                new_pt = np.array([[yx[1], yx[0]]], dtype=np.float32)  # (x, y)
                new_lbl = np.array([1 if positive else 0])
                acc_points = new_pt if acc_points is None else np.concatenate([acc_points, new_pt])
                acc_labels = new_lbl if acc_labels is None else np.concatenate([acc_labels, new_lbl])
                masks, _, _ = predictor.predict(
                    point_coords=acc_points, point_labels=acc_labels,
                    multimask_output=False,
                )
                pred_mask = masks[0] > 0
            seg_per_iter[it][pred_mask] = gt_id

    return seg_per_iter


def _segment_microsam1_3d_iterative(raw, gt, predictor, start_with_box, n_iterations, dilation=3):
    """Slice-by-slice 3D iterative interactive segmentation with micro-sam v1.

    Returns a list of n_iterations segmentation arrays (Z, Y, X) uint32.
    """
    n_frames = raw.shape[0]
    seg_per_iter = [np.zeros_like(gt, dtype="uint32") for _ in range(n_iterations)]

    for z in range(n_frames):
        slice_raw = raw[z].astype("float32")
        slice_gt = gt[z]

        if len(np.unique(slice_gt)) == 1:
            continue

        if slice_raw.max() > 255:
            slice_raw = normalize(slice_raw) * 255

        slice_segs = _segment_microsam1_2d_iterative(
            slice_raw, slice_gt, predictor, start_with_box, n_iterations, dilation,
        )
        for it in range(n_iterations):
            seg_per_iter[it][z] = slice_segs[it]

    return seg_per_iter


def run_microsam1_evaluation(
    dataset_name, data_root, experiment_folder, device,
    model_type="vit_b_lm", checkpoint=None, start_with_box=True, n_iterations=8, ndim=None,
):
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2

    prompt_str = "box" if start_with_box else "point"
    dim_suffix = "" if ndim == 2 else "_3d"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(
            results_dir,
            f"{dataset_name}_microsam1_{model_type}{dim_suffix}_{prompt_str}_iter{it:02d}.csv",
        )
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

    predictor = _load_microsam1(model_type, checkpoint, device)
    n = len(get_data_paths(dataset_name, data_root)[0])
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels in tqdm(_load_data(dataset_name, data_root, ndim), total=n, desc=f"microsam1-{ndim}d"):
        if ndim == 3:
            segs = _segment_microsam1_3d_iterative(raw, labels, predictor, start_with_box, n_iterations)
        else:
            segs = _segment_microsam1_2d_iterative(raw, labels, predictor, start_with_box, n_iterations)
        all_gt.append(labels)
        for it, seg in enumerate(segs):
            all_seg_per_iter[it].append(seg)

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg_per_iter[it], save_path=save_path)
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

    n = len(get_data_paths(dataset_name, data_root)[0])
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels in tqdm(_load_data(dataset_name, data_root, ndim), total=n, desc=f"sam3-{ndim}d"):
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
            all_seg_per_iter[it].append(seg)

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg_per_iter[it], save_path=save_path)
        print(f"Iteration {it:02d}: {results}")


def _load_sam2_image_predictor(model_type, backbone, checkpoint_path, device):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from micro_sam.v2.util import get_sam2_model
    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=checkpoint_path, backbone=backbone,
    )
    return SAM2ImagePredictor(model)


def _segment_sam2_2d_iterative(image, gt, predictor, start_with_box, n_iterations, dilation=5):
    """Iterative 2D interactive segmentation with SAM2ImagePredictor.

    Returns a list of n_iterations segmentation arrays (H, W) uint32.
    """
    H, W = image.shape[:2]
    gt_ids = np.unique(gt)[1:]

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    predictor.set_image(image.astype("uint8"))

    seg_per_iter = [np.zeros((H, W), dtype="uint32") for _ in range(n_iterations)]

    for gt_id in gt_ids:
        gt_mask = gt == gt_id
        binary_gt = gt_mask.astype("uint32")

        if start_with_box:
            _, _, raw_boxes = _get_batched_prompts(
                gt=binary_gt, gt_ids=[1], use_points=False, use_boxes=True,
                n_positives=0, n_negatives=0, dilation=dilation,
            )
            # raw_boxes[0] is (4,) in (x0, y0, x1, y1) format — SAM2 expects xyxy
            masks, scores, logits = predictor.predict(
                point_coords=None, point_labels=None,
                box=raw_boxes[0], multimask_output=False,
            )
            acc_points, acc_labels = None, None
        else:
            raw_pts, raw_lbls, _ = _get_batched_prompts(
                gt=binary_gt, gt_ids=[1], use_points=True, use_boxes=False,
                n_positives=1, n_negatives=0, dilation=0,
            )
            acc_points = raw_pts[0].astype(np.float32)  # (1, 2) in (x, y)
            acc_labels = raw_lbls[0]                     # (1,)
            masks, scores, logits = predictor.predict(
                point_coords=acc_points, point_labels=acc_labels, multimask_output=True,
            )
            best = np.argmax(scores)
            masks = masks[best][None]
            logits = logits[best][None]

        pred_mask = masks[0] > 0
        seg_per_iter[0][pred_mask] = gt_id

        for it in range(1, n_iterations):
            yx, positive = _get_corrective_point(gt_mask, pred_mask)
            if yx is not None:
                new_pt = np.array([[yx[1], yx[0]]], dtype=np.float32)  # (x, y)
                new_lbl = np.array([1 if positive else 0])
                acc_points = new_pt if acc_points is None else np.concatenate([acc_points, new_pt])
                acc_labels = new_lbl if acc_labels is None else np.concatenate([acc_labels, new_lbl])
                masks, _, logits = predictor.predict(
                    point_coords=acc_points, point_labels=acc_labels,
                    mask_input=logits, multimask_output=False,
                )
                pred_mask = masks[0] > 0
            seg_per_iter[it][pred_mask] = gt_id

    return seg_per_iter


def _segment_sam2_3d_iterative(
    raw, labels, model_type, backbone, checkpoint_path, device, start_with_box, n_iterations
):
    """Iterative 3D interactive segmentation using SAM2 video predictor.

    Writes volume slices to a temporary directory so SAM2's video predictor can
    init its inference state, then runs object-by-object iterative prompting.

    Returns a list of n_iterations segmentation arrays (Z, Y, X) uint32.
    """
    import imageio.v3 as _iio
    from micro_sam.v2.util import get_sam2_model

    seg_per_iter = [np.zeros_like(labels, dtype="uint32") for _ in range(n_iterations)]
    gt_ids = list(np.unique(labels))[1:]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(raw):
            frame_u8 = frame.astype("uint8")
            if frame_u8.ndim == 2:
                frame_u8 = np.stack([frame_u8] * 3, axis=-1)
            _iio.imwrite(os.path.join(tmpdir, f"{i:04d}.tif"), frame_u8)

        predictor = get_sam2_model(
            model_type=model_type, device=device, checkpoint_path=checkpoint_path,
            input_type="videos", backbone=backbone,
        )
        inference_state = predictor.init_state(video_path=tmpdir)

        for gt_id in gt_ids:
            gt_mask_3d = labels == gt_id
            z_slices = np.where(gt_mask_3d)[0]
            z_center = int(np.ceil(np.mean([z_slices.min(), z_slices.max()])))

            gt_mask_2d = (labels[z_center] == gt_id).astype("uint32")
            if start_with_box:
                _, _, raw_boxes = _get_batched_prompts(
                    gt=gt_mask_2d, gt_ids=[1], use_points=False, use_boxes=True,
                    n_positives=0, n_negatives=0, dilation=5,
                )
                init_pts, init_lbls, init_box = None, None, raw_boxes[0]
            else:
                raw_pts, raw_lbls, _ = _get_batched_prompts(
                    gt=gt_mask_2d, gt_ids=[1], use_points=True, use_boxes=False,
                    n_positives=1, n_negatives=0, dilation=5,
                )
                init_pts = raw_pts[0].astype(np.float32)  # (1, 2) in (x, y)
                init_lbls = raw_lbls[0]                    # (1,)
                init_box = None

            # Add the initial prompt for this object.
            predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=z_center, obj_id=gt_id,
                points=init_pts, labels=init_lbls, box=init_box, clear_old_points=False,
            )

            for it in range(n_iterations):
                video_segs = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    video_segs[out_frame_idx] = {
                        oid: (out_mask_logits[i_] > 0.0).cpu().numpy()
                        for i_, oid in enumerate(out_obj_ids)
                    }

                seg_iter = np.zeros(labels.shape, dtype=bool)
                for z_idx, frame_segs in video_segs.items():
                    if gt_id in frame_segs:
                        seg_iter[z_idx] = frame_segs[gt_id].squeeze()
                seg_per_iter[it][seg_iter] = gt_id

                if it < n_iterations - 1:
                    coords, is_positive = _get_corrective_point(gt_mask_3d, seg_iter)
                    if coords is not None:
                        # coords = [z, y, x] for 3D masks
                        z_new = coords[0]
                        new_pt = np.array([[coords[2], coords[1]]], dtype=np.float32)  # (x, y)
                        new_lbl = np.array([1 if is_positive else 0])
                        predictor.add_new_points_or_box(
                            inference_state=inference_state, frame_idx=z_new, obj_id=gt_id,
                            points=new_pt, labels=new_lbl, box=None, clear_old_points=False,
                        )

            predictor.reset_state(inference_state)

    return seg_per_iter


def run_sam2_evaluation(
    dataset_name, data_root, experiment_folder, device,
    model_type=_SAM2_MODEL_TYPE, backbone=_SAM2_BACKBONE, checkpoint_path=None,
    start_with_box=True, n_iterations=8, ndim=None, name_tag="sam2",
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

    n = len(get_data_paths(dataset_name, data_root)[0])
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    if ndim == 2:
        predictor = _load_sam2_image_predictor(model_type, backbone, checkpoint_path, device)
        for raw, labels in tqdm(_load_data(dataset_name, data_root, ndim=2), total=n, desc=f"{name_tag}-2d"):
            segs = _segment_sam2_2d_iterative(raw, labels, predictor, start_with_box, n_iterations)
            all_gt.append(labels)
            for it, seg in enumerate(segs):
                all_seg_per_iter[it].append(seg)
    else:
        for raw, labels in tqdm(_load_data(dataset_name, data_root, ndim=3), total=n, desc=f"{name_tag}-3d"):
            segs = _segment_sam2_3d_iterative(
                raw, labels, model_type, backbone, checkpoint_path, device, start_with_box, n_iterations,
            )
            all_gt.append(labels)
            for it, seg in enumerate(segs):
                all_seg_per_iter[it].append(seg)

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg_per_iter[it], save_path=save_path)
        print(f"Iteration {it:02d}: {results}")


def main():
    import argparse
    all_datasets = sorted(set(DATASETS_2D + DATASETS_3D))
    parser = argparse.ArgumentParser(description="Evaluate interactive segmentation baselines.")
    parser.add_argument("-d", "--dataset_name", required=True, choices=all_datasets)
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("--method", type=str, default="nninteractive", choices=_METHODS)
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=["box", "point"],
                        help="First prompt type (default: box).")
    parser.add_argument("-iter", "--n_iterations", type=int, default=8,
                        help="Number of iterative prompting rounds (default: 8).")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Override default checkpoint path.")
    parser.add_argument("-m", "--model_type", type=str, default=None,
                        help="Model type override (e.g. vit_b for sam, hvit_t for sam2/micro-samv2).")
    parser.add_argument("--ndim", type=int, default=None, choices=[2, 3],
                        help="Dimensionality override (default: inferred from dataset).")
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
    elif args.method == "microsam1":
        run_microsam1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=args.model_type or "vit_b_lm", checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
        )
    elif args.method == "sam3":
        run_sam3_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
        )
    elif args.method == "sam":
        mt = args.model_type or _SAM_V1_MODEL_TYPE
        run_microsam1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
        )
    elif args.method == "micro-sam":
        is_em = args.dataset_name in _EM_DATASETS
        mt = args.model_type or (_MICROSAM_V1_EM_MODEL if is_em else _MICROSAM_V1_LM_MODEL)
        run_microsam1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
        )
    elif args.method == "sam2":
        mt = args.model_type or _SAM2_MODEL_TYPE
        run_sam2_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, backbone=_SAM2_BACKBONE,
            checkpoint_path=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
            name_tag="sam2",
        )
    elif args.method == "micro-samv2":
        mt = args.model_type or _SAM2_MODEL_TYPE
        run_sam2_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, backbone=_SAM2_BACKBONE,
            checkpoint_path=args.checkpoint or MICROSAM2_INTERACTIVE_CHECKPOINT,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim,
            name_tag="microsam2",
        )


if __name__ == "__main__":
    main()
