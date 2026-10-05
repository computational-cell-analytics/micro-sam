"""Benchmark evaluation of the interactive segmentation baselines, i.e. everything but micro-sam2.

Supported methods:
  nninteractive: nnInteractive interactive segmentation (3d only)
  sam: Pretrained SAM v1 interactive segmentation (2d only)
  sam3: SAM3 interactive segmentation (2d and 3d)
  micro-sam: micro-sam v1 finetuned interactive, slice-wise (vit_b_lm for LM, vit_b_em_organelles for EM)
  microsam_vol: SAM v1 or micro-sam v1 with iterative prompts and volumetric projection, as the 3d annotator (3d only)

The SAM2 engine itself, pretrained or jointly finetuned, is evaluated by
evaluate_interactive_segmentation.py, which those two share.

Usage examples:
    python evaluate_interactive_baselines.py -d embedseg -e <exp> --method nninteractive -p box
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method sam3
    python evaluate_interactive_baselines.py -d livecell -e <exp> --method micro-sam
    python evaluate_interactive_baselines.py -d embedseg -e <exp> --method microsam_vol -m vit_b_lm -p box
    python evaluate_interactive_baselines.py -d cremi -e <exp> --method microsam_vol -p point --correction box
"""

import os
import sys
import shutil
import argparse
from itertools import islice
from multiprocessing import get_context
from functools import lru_cache, partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.measure import label as connected_components

import torch

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_EM,
    check_data_download, interactive_result_name, interactive_run_tag, load_data, n_samples,
    run_dataset_evaluation,
)

METHODS = ["nninteractive", "sam3", "sam", "micro-sam", "microsam_vol"]

NNINTERACTIVE_CHECKPOINT = "/mnt/vast-nhr/home/archit/u12090/nnInteractive/pretrained_weights/nnInteractive_v1.0"
SAM3_ROOT = "/mnt/vast-nhr/home/archit/u12090/SAM3_Experiments"

SAM_V1_MODEL_TYPE = "vit_b"
MICROSAM_V1_LM_MODEL = "vit_b_lm"
MICROSAM_V1_EM_MODEL = "vit_b_em_organelles"

EM_DATASETS = set(DATASETS_EM)


def _load_samples(dataset_name, data_root, ndim, min_size=0, limit=None):
    """Return the test samples and their count, cut to the first 'limit' samples for a check."""
    samples, total = load_data(dataset_name, data_root, ndim, min_size), n_samples(dataset_name, data_root)
    if limit is not None:
        samples, total = islice(samples, limit), min(total, limit)
    return samples, total


def _get_largest_region_center(mask):
    """Return the most interior point of the largest connected region of a mask, or None if it is empty.

    The centroid is not used, since it can lie outside the region. False foreground often forms a
    shell around the object, and the centroid of such a shell is a point on the true object.
    """
    labeled = connected_components(mask)
    if labeled.max() == 0:
        return None
    region_id = np.bincount(labeled.ravel())[1:].argmax() + 1
    bbox = find_objects(labeled)[region_id - 1]
    region = np.pad(labeled[bbox] == region_id, 1)
    distances = distance_transform_edt(region)
    point = np.unravel_index(distances.argmax(), distances.shape)
    return [int(c) - 1 + sl.start for c, sl in zip(point, bbox)]


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
            center = (z, *_get_largest_region_center(gt_mask_2d))
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
    checkpoint_path=None, start_with_box=True, n_iterations=8, limit=None,
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
    samples, n = _load_samples(dataset_name, data_root, 3, limit=limit)
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels, valid_roi in tqdm(samples, total=n, desc="nninteractive"):
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


def _write_2d_inputs(dataset_name, data_root, input_dir, gt_dir, min_size=0, limit=None):
    """Write the cropped images and labels the SAM v1 2d inference reads, and return their paths."""
    image_paths, gt_paths = [], []
    samples, n = _load_samples(dataset_name, data_root, 2, min_size, limit)
    it = tqdm(samples, total=n, desc="save-crops")
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
    use_masks=False, min_size=0, limit=None,
):
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2

    if ndim == 3:
        raise ValueError(
            "The slice-wise micro-sam v1 evaluation is 2d only. Use '--method microsam_vol' for a volume."
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
    image_paths, gt_paths = _write_2d_inputs(dataset_name, data_root, input_dir, gt_dir, min_size, limit)

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
    start_with_box=True, n_iterations=8, ndim=None, limit=None,
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

    samples, n = _load_samples(dataset_name, data_root, ndim, limit=limit)
    all_gt = []
    all_seg_per_iter = [[] for _ in range(n_iterations)]

    for raw, labels, valid_roi in tqdm(samples, total=n, desc=f"sam3-{ndim}d"):
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


@lru_cache(maxsize=1)
def _load_sam_v1_once(model_type, checkpoint, device):
    """Load the SAM v1 predictor once per worker process."""
    return _load_sam_v1(model_type, checkpoint, device)


def _segment_volume(raw, labels, model_type, checkpoint, device, options):
    """Segment one volume with iterative prompts, in a worker process."""
    from micro_sam.v1.evaluation.multi_dimensional_segmentation import evaluate_interactive_volume_segmentation
    predictor = _load_sam_v1_once(model_type, checkpoint, device)
    return evaluate_interactive_volume_segmentation(predictor, raw, labels, **options)


def run_microsam_volumetric_evaluation(
    dataset_name, data_root, experiment_folder, device, model_type=None, checkpoint=None, start_with_box=True,
    n_iterations=8, min_size=0, start_slice="center", correction="box_and_points", use_masks=False, projection="mask",
    iou_threshold=0.8, box_extension=0.025, seed=None, n_workers=4, limit=None,
):
    """Evaluate SAM v1 or micro-sam v1 with iterative prompts and volumetric projection, as in the 3d annotator.

    See `evaluate_interactive_volume_segmentation` for the prompts and the corrections. Each worker process
    segments one volume at a time on the same GPU.
    """
    if dataset_name not in DATASETS_3D:
        raise ValueError(f"The volumetric micro-sam v1 evaluation is 3d only. '{dataset_name}' is 2d.")
    if model_type is None:
        model_type = MICROSAM_V1_EM_MODEL if dataset_name in EM_DATASETS else MICROSAM_V1_LM_MODEL

    # The name holds every setting that changes the numbers.
    prompt = "box" if start_with_box else "point"
    prompt += f"_{start_slice}{seed if start_slice == 'random' else ''}_{correction}"
    prompt += f"{'_with_masks' if use_masks else ''}_{projection}_iou{iou_threshold:g}_ext{box_extension:g}"
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, interactive_result_name(
            dataset_name, "microsam_vol", model_type, prompt, it, ndim=3, min_size=min_size,
        ))
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

    # Skip the volumes without objects: there is nothing to prompt, and nothing to score.
    samples = [s for s in _load_samples(dataset_name, data_root, 3, min_size, limit)[0] if s[1].max() > 0]
    options = dict(
        start_prompt="box" if start_with_box else "points", start_slice=start_slice, n_iterations=n_iterations,
        correction=correction, use_previous_mask=use_masks, projection=projection, iou_threshold=iou_threshold,
        box_extension=box_extension, seed=seed,
    )
    segment = partial(_segment_volume, model_type=model_type, checkpoint=checkpoint, device=device, options=options)
    # CUDA cannot run in a forked process, so the workers start with 'spawn'.
    with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as pool:
        segs_per_volume = list(tqdm(
            pool.map(segment, *zip(*[(raw, labels) for raw, labels, _ in samples])),
            total=len(samples), desc="microsam_vol",
        ))

    all_gt = [labels for _, labels, _ in samples]
    all_seg_per_iter = [
        [seg if roi is None else np.where(roi, seg, 0) for seg, (_, _, roi) in zip(segs, samples)]
        for segs in zip(*segs_per_volume)
    ]
    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        results = run_dataset_evaluation(all_gt, all_seg_per_iter[it], dataset_name, save_path)
        print(f"Iteration {it:02d}: {results}")


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
        help="Feed the previous masks back as mask prompts. SAM v1 is not trained with them."
    )
    parser.add_argument(
        "--start_slice", default="center", choices=("begin", "center", "end", "random"),
        help="microsam_vol only. The slice of an object that gets the first prompt."
    )
    parser.add_argument(
        "--correction", default="box_and_points", choices=("box", "points", "box_and_points"),
        help="microsam_vol only. The prompt on the worst slice of each iteration."
    )
    parser.add_argument("--projection", default="mask", help="microsam_vol only. The projection from slice to slice.")
    parser.add_argument("--iou_threshold", type=float, default=0.8, help="microsam_vol only. Stop the projection.")
    parser.add_argument("--box_extension", type=float, default=0.025, help="microsam_vol only. Extend the box.")
    parser.add_argument("--seed", type=int, default=None, help="microsam_vol only. The seed of a random start slice.")
    parser.add_argument("--n_workers", type=int, default=4, help="microsam_vol only. The volumes segmented at once.")
    parser.add_argument("--n_samples", type=int, default=None, help="Score only the first N samples, for a check.")
    args = parser.parse_args()

    check_data_download(args.dataset_name, args.input_path)

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_with_box = (args.prompt_choice == "box")

    if args.method == "nninteractive":
        run_nninteractive_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, checkpoint_path=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, limit=args.n_samples,
        )

    elif args.method == "sam3":
        run_sam3_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, limit=args.n_samples,
        )

    elif args.method == "sam":
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=args.model_type or SAM_V1_MODEL_TYPE, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="sam",
            use_masks=args.use_masks, min_size=args.min_size, limit=args.n_samples,
        )

    elif args.method == "micro-sam":
        is_em = args.dataset_name in EM_DATASETS
        model_type = args.model_type or (MICROSAM_V1_EM_MODEL if is_em else MICROSAM_V1_LM_MODEL)
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=model_type, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="micro-sam",
            use_masks=args.use_masks, min_size=args.min_size, limit=args.n_samples,
        )

    elif args.method == "microsam_vol":
        run_microsam_volumetric_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder, device=device,
            model_type=args.model_type, checkpoint=args.checkpoint, start_with_box=start_with_box,
            n_iterations=args.n_iterations, min_size=args.min_size, start_slice=args.start_slice,
            correction=args.correction, use_masks=args.use_masks, projection=args.projection,
            iou_threshold=args.iou_threshold, box_extension=args.box_extension, seed=args.seed,
            n_workers=args.n_workers, limit=args.n_samples,
        )

    else:
        raise ValueError(f"Unknown method: '{args.method}'.")


if __name__ == "__main__":
    main()
