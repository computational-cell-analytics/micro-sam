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
import uuid
import shutil
import argparse
from itertools import islice
from multiprocessing import get_context
from functools import lru_cache, partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

import torch
from bioimage_cpp.segmentation import label as relabel_components

from micro_sam.prompt_generators import get_error_centers
from micro_sam.v1.evaluation.inference import _get_batched_prompts

from common import (
    CLICK_PROTOCOLS, DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_EM,
    check_data_download, interactive_result_name, interactive_run_tag, interactive_sample_ids_3d, load_data, n_samples,
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


# The baselines without a micro-sam pipeline (nnInteractive, SAM3) are prompted with the protocol of the micro-sam2
# evaluation (see 'micro_sam.v2.evaluation.inference'), so that every method gets the same prompts: the connected
# components of the labels are prompted with the same first box or point, the 'center' correction clicks are placed
# on the worst slice in 3d, and the smaller object is kept where two predicted objects overlap.
PROMPT_DILATION = 5
PROMPT_SEED = 0


def _prompting_labels(labels):
    """Return the objects that get prompted: the connected components of the labels."""
    return relabel_components(labels).astype("uint32")


def _first_prompt_2d(mask, start_with_box):
    """Return the first prompt of one object in 2d: its box as (x0, y0, x1, y1), or a point as (x, y)."""
    points, _, boxes = _get_batched_prompts(
        gt=mask.astype("uint32"), gt_ids=[1], use_points=not start_with_box, use_boxes=start_with_box,
        n_positives=0 if start_with_box else 1, n_negatives=0, dilation=PROMPT_DILATION,
    )
    return [float(v) for v in (boxes[0] if start_with_box else points[0][0])]


def _center_slice(mask):
    """Return the slice of the first prompt of one object in 3d: the middle of its z-extent."""
    z = np.where(mask.any(axis=(1, 2)))[0]
    return int(np.ceil(np.mean([z.min(), z.max()])))


def _worst_slice_clicks(gt_mask, pred_mask):
    """Return the correction clicks of one object in 3d: the 'center' clicks on its slice with the most errors.

    Returns:
        The slice, the positive click as (x, y) or None, and the negative click as (x, y) or None.
    """
    z_slices = np.where(gt_mask.any(axis=(1, 2)))[0]
    z = int(z_slices[np.argmax([np.sum(gt_mask[z] != pred_mask[z]) for z in z_slices])])
    pos, neg = get_error_centers(gt_mask[z], pred_mask[z])
    return z, pos, neg


def _add_object(segmentation, areas, mask, gt_id):
    """Add one predicted object to a segmentation. Where objects overlap, the smaller one is kept."""
    area = mask.sum()
    existing = segmentation[mask]
    segmentation[mask] = np.where(areas[existing] > area, gt_id, existing)
    areas[gt_id] = area


def _empty_predictions(labels, n_iterations):
    """Return the segmentations and object areas of all iterations, before any object is added."""
    segmentations = [np.zeros(labels.shape, dtype="uint32") for _ in range(n_iterations)]
    areas = [np.full(int(labels.max()) + 1, np.inf) for _ in range(n_iterations)]
    return segmentations, areas


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

    labels = _prompting_labels(labels)
    np.random.seed(PROMPT_SEED)
    seg_per_iter, areas = _empty_predictions(labels, n_iterations)
    for gt_id in np.setdiff1d(np.unique(labels), [0]):
        gt_mask = labels == gt_id
        session.reset_interactions()
        z = _center_slice(gt_mask)
        prompt = [int(round(v)) for v in _first_prompt_2d(gt_mask[z], start_with_box)]
        if start_with_box:
            x0, y0, x1, y1 = prompt
            session.add_bbox_interaction([[z, z + 1], [y0, y1], [x0, x1]], include_interaction=True)
        else:
            session.add_point_interaction((z, prompt[1], prompt[0]), include_interaction=True)
        pred_mask = buffer > 0.5
        _add_object(seg_per_iter[0], areas[0], pred_mask, gt_id)

        for it in range(1, n_iterations):
            z, pos, neg = _worst_slice_clicks(gt_mask, pred_mask)
            for click, include in ((pos, True), (neg, False)):
                if click is not None:
                    session.add_point_interaction((z, int(click[1]), int(click[0])), include_interaction=include)
            pred_mask = buffer > 0.5
            _add_object(seg_per_iter[it], areas[it], pred_mask, gt_id)

    return seg_per_iter


def _cached_prediction_paths(prediction_root, sample_id, n_iterations):
    """The per-iteration prediction files of one sample."""
    sample_dir = os.path.join(prediction_root, f"sample_{sample_id:05d}")
    return [os.path.join(sample_dir, f"iteration{it:02d}.tif") for it in range(n_iterations)]


def _write_cached_predictions(segs, paths):
    """Write the predictions of one sample. Each file is renamed into place, so no partial file is left."""
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    for seg, path in zip(segs, paths):
        partial_path = f"{path[:-4]}.{uuid.uuid4().hex}.partial.tif"
        imageio.imwrite(partial_path, seg, compression="zlib")
        os.replace(partial_path, path)


def run_nninteractive_evaluation(
    dataset_name, data_root, experiment_folder, device,
    checkpoint_path=None, start_with_box=True, n_iterations=8, shard=None, score_only=False, limit=None,
):
    """Run nnInteractive with iterative prompting on the 3d test split and write one result CSV per iteration.

    The predictions are cached per sample, so a preempted job resumes per sample. With 'shard' set to
    (shard_index, n_shards), only the samples with 'sample_id % n_shards == shard_index' are predicted
    and the function returns before scoring. A later unsharded call scores from the cache. With
    'score_only' it fails if a sample is missing rather than predict it.
    """
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

    prediction_root = os.path.join(experiment_folder, "predictions", "nninteractive", dataset_name, prompt_str)
    selected = interactive_sample_ids_3d(dataset_name, data_root, limit)
    sample_ids = selected if shard is None else selected[shard[0]::shard[1]]
    samples = load_data(dataset_name, data_root, 3, sample_ids=set(sample_ids))

    # The model is loaded on the first sample that needs it, so that scoring from the cache loads no model.
    session = None
    all_gt, all_valid_rois, missing = [], [], []
    pred_paths_per_iter = [[] for _ in range(n_iterations)]
    samples = tqdm(samples, total=len(sample_ids), desc="nninteractive")
    for sample_id, (raw, labels, valid_roi) in zip(sample_ids, samples):
        if labels.max() == 0:  # Nothing to prompt, and nothing to score, as in the micro-sam2 evaluation.
            continue
        paths = _cached_prediction_paths(prediction_root, sample_id, n_iterations)
        if not all(os.path.exists(path) for path in paths):
            if score_only:
                missing.append(sample_id)
                continue
            if session is None:
                session = _load_nninteractive(checkpoint_path, device)
            segs = _segment_nninteractive_iterative(raw, labels, session, start_with_box, n_iterations)
            _write_cached_predictions(segs, paths)
        all_gt.append(labels)
        all_valid_rois.append(valid_roi)
        for it, path in enumerate(paths):
            pred_paths_per_iter[it].append(path)

    if shard is not None:
        print(f"Shard {shard[0]}/{shard[1]} finished predicting {len(sample_ids)} sample(s).")
        return
    if missing:
        raise RuntimeError(
            f"{len(missing)} sample(s) of '{dataset_name}' have no predictions yet: {missing}. Run their shards first."
        )

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        predictions = []
        for path, valid_roi in zip(pred_paths_per_iter[it], all_valid_rois):
            prediction = imageio.imread(path)
            if valid_roi is not None:
                prediction[~valid_roi] = 0
            predictions.append(prediction)
        results = run_dataset_evaluation(all_gt, predictions, dataset_name, save_path)
        print(f"Iteration {it:02d}: {results}")


def _load_sam_v1(model_type, checkpoint, device):
    from micro_sam.v1.util import get_sam_model
    return get_sam_model(model_type=model_type, checkpoint_path=checkpoint, device=device)


def _write_atomic(path, data):
    """Write a tif under a unique temporary name, then rename it, so no reader sees a partial file."""
    partial_path = f"{path[:-4]}.{uuid.uuid4().hex}.partial.tif"
    imageio.imwrite(partial_path, data, compression="zlib")
    os.replace(partial_path, path)


def _write_2d_inputs(dataset_name, data_root, input_dir, gt_dir, min_size=0, limit=None):
    """Write the cropped images and labels the SAM v1 2d inference reads, and return their paths.

    The shards of a dataset share these files, so existing ones are kept and new ones written atomically.
    """
    image_paths, gt_paths = [], []
    samples, n = _load_samples(dataset_name, data_root, 2, min_size, limit)
    it = tqdm(samples, total=n, desc="save-crops")
    for sample_id, (raw, labels, _) in enumerate(it):
        if labels.max() == 0:  # Inference skips these, so they must not be scored either.
            continue

        image_path = os.path.join(input_dir, f"{sample_id:05d}.tif")
        gt_path = os.path.join(gt_dir, f"{sample_id:05d}.tif")
        if not os.path.exists(image_path):
            _write_atomic(image_path, np.clip(np.round(raw), 0, 255).astype("uint8"))
        if not os.path.exists(gt_path):
            _write_atomic(gt_path, labels.astype("uint32"))
        image_paths.append(image_path)
        gt_paths.append(gt_path)
    return image_paths, gt_paths


def run_sam_v1_evaluation(
    dataset_name, data_root, experiment_folder, device,
    model_type="vit_b_lm", checkpoint=None, start_with_box=True, n_iterations=8, ndim=None, name_tag="micro-sam",
    use_masks=False, min_size=0, limit=None, click_protocol="random", shard=None, score_only=False,
):
    """Run the slice-wise SAM v1 iterative prompting on the 2d test split and write one result CSV per iteration.

    With 'shard' set to (shard_index, n_shards), only that stride of the images is predicted and the
    function returns before scoring. The predictions are cached per image, so a later unsharded call
    scores without predicting again. With 'score_only' that call fails if an image is missing rather
    than predict it.
    """
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2

    if ndim == 3:
        raise ValueError(
            "The slice-wise micro-sam v1 evaluation is 2d only. Use '--method microsam_vol' for a volume."
        )

    # Of the EM data, micro-sam v1 covers only the mitochondria and nuclei, with the organelle models.
    if name_tag == "micro-sam" and dataset_name in EM_DATASETS and "em_organelles" not in model_type:
        raise ValueError(
            f"micro-sam interactive runs on EM only with the 'vit_*_em_organelles' models; got '{model_type}' on "
            f"'{dataset_name}'."
        )

    prompt_str = "box" if start_with_box else "point"
    run_tag = interactive_run_tag(ndim=2, use_masks=use_masks, min_size=min_size, click_protocol=click_protocol)
    results_dir = os.path.join(experiment_folder, "results")
    save_paths = [
        os.path.join(results_dir, interactive_result_name(
            dataset_name, name_tag, model_type, prompt_str, it,
            ndim=2, use_masks=use_masks, min_size=min_size, click_protocol=click_protocol,
        ))
        for it in range(n_iterations)
    ]
    if all(os.path.exists(p) for p in save_paths):
        print(f"Results already stored at '{results_dir}'.")
        return

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

    if score_only:
        missing = [
            path for path in image_paths
            if not all(
                os.path.exists(os.path.join(prediction_dir, f"iteration{it:02d}", os.path.basename(path)))
                for it in range(n_iterations)
            )
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(image_paths)} image(s) of '{dataset_name}' have no predictions yet. "
                "Run their shards first."
            )
    else:
        from micro_sam.v1.evaluation.inference import run_inference_with_iterative_prompting
        predict_image_paths, predict_gt_paths = image_paths, gt_paths
        if shard is not None:
            predict_image_paths, predict_gt_paths = image_paths[shard[0]::shard[1]], gt_paths[shard[0]::shard[1]]
        run_inference_with_iterative_prompting(
            predictor=_load_sam_v1(model_type, checkpoint, device),
            image_paths=predict_image_paths,
            gt_paths=predict_gt_paths,
            embedding_dir=embedding_dir,
            prediction_dir=prediction_dir,
            start_with_box_prompt=start_with_box,
            n_iterations=n_iterations,
            use_masks=use_masks,
            click_protocol=click_protocol,
        )

    if shard is not None:
        print(f"Shard {shard[0]}/{shard[1]} finished predicting {len(predict_image_paths)} image(s).")
        return

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        pred_dir = os.path.join(prediction_dir, f"iteration{it:02d}")
        pred_paths = [os.path.join(pred_dir, os.path.basename(path)) for path in image_paths]
        results = run_dataset_evaluation(gt_paths, pred_paths, dataset_name, save_path)
        print(f"Iteration {it:02d}: {results}")

    shutil.rmtree(work_dir, ignore_errors=True)


def _segment_sam3_2d(image, labels, model, processor, start_with_box, n_iterations):
    """Prompt SAM3 for every object of an image, with the protocol of the micro-sam2 evaluation.

    The box stays in all iterations, the correction clicks add up, and the previous mask logits are fed back,
    as for micro-sam2 and SAM2.
    """
    from PIL import Image

    # SAM3 reads the size of an array as channels-first, so the image is passed as an RGB PIL image, as in its
    # examples. Grayscale images are repeated into the three channels.
    rgb = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    state = processor.set_image(Image.fromarray(np.clip(np.round(rgb), 0, 255).astype("uint8")))

    labels = _prompting_labels(labels)
    np.random.seed(PROMPT_SEED)
    seg_per_iter, areas = _empty_predictions(labels, n_iterations)
    for gt_id in np.setdiff1d(np.unique(labels), [0]):
        gt_mask = labels == gt_id
        prompt = _first_prompt_2d(gt_mask, start_with_box)
        box = np.array([prompt], dtype="float32") if start_with_box else None
        points = None if start_with_box else np.array([prompt], dtype="float32")
        point_labels = None if start_with_box else np.array([1])
        mask_input, pred_mask = None, None
        for it in range(n_iterations):
            if it > 0:
                pos, neg = get_error_centers(gt_mask, pred_mask)
                clicks = [(click, label) for click, label in ((pos, 1), (neg, 0)) if click is not None]
                if clicks:
                    new_points = np.array([click for click, _ in clicks], dtype="float32")
                    new_labels = np.array([label for _, label in clicks])
                    points = new_points if points is None else np.concatenate([points, new_points])
                    point_labels = new_labels if point_labels is None else np.concatenate([point_labels, new_labels])
            # A single mask for the first point, as in the 'center' protocol of micro-sam2.
            masks, _, low_res_masks = model.predict_inst(
                state, point_coords=points, point_labels=point_labels, box=box, mask_input=mask_input,
                multimask_output=False,
            )
            pred_mask = masks[0] > 0
            mask_input = low_res_masks[:1]
            _add_object(seg_per_iter[it], areas[it], pred_mask, gt_id)
    return seg_per_iter


def _segment_sam3_3d(volume, labels, predictor, start_with_box, n_iterations):
    """Prompt SAM3 for every object of a volume, with the protocol of the micro-sam2 evaluation.

    The first prompt is on the middle slice of the object, the 'center' correction clicks are on its worst slice,
    and the masks are propagated through the volume after every prompt, as for micro-sam2 and SAM2.
    """
    import tempfile
    from micro_sam3.evaluation.inference import _propagate_sam3, _volume_to_frames

    n_frames, height, width = volume.shape
    labels = _prompting_labels(labels)
    np.random.seed(PROMPT_SEED)
    seg_per_iter, areas = _empty_predictions(labels, n_iterations)

    def to_points(clicks):
        coords = torch.tensor([[x / width, y / height] for (x, y), _ in clicks], dtype=torch.float32)
        return coords, torch.tensor([label for _, label in clicks], dtype=torch.int32)

    def propagate(frame):
        pred_mask = np.zeros(labels.shape, dtype=bool)
        for z, mask in _propagate_sam3(predictor, state, frame, n_frames).items():
            pred_mask[z] = mask
        return pred_mask

    tmp_dir = tempfile.mkdtemp()
    try:
        state = predictor.init_state(video_path=_volume_to_frames(volume, tmp_dir))
        for gt_id in np.setdiff1d(np.unique(labels), [0]):
            gt_mask = labels == gt_id
            predictor.clear_all_points_in_video(state)
            z = _center_slice(gt_mask)
            prompt = _first_prompt_2d(gt_mask[z], start_with_box)
            if start_with_box:
                x0, y0, x1, y1 = prompt
                box = np.clip(np.array([[x0 / width, y0 / height, x1 / width, y1 / height]], dtype="float32"), 0, 1)
                predictor.add_new_points_or_box(inference_state=state, frame_idx=z, obj_id=int(gt_id), box=box)
            else:
                coords, point_labels = to_points([(prompt, 1)])
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=z, obj_id=int(gt_id), points=coords, labels=point_labels,
                    clear_old_points=False,
                )
            pred_mask = propagate(z)
            _add_object(seg_per_iter[0], areas[0], pred_mask, gt_id)

            for it in range(1, n_iterations):
                z, pos, neg = _worst_slice_clicks(gt_mask, pred_mask)
                clicks = [(click, label) for click, label in ((pos, 1), (neg, 0)) if click is not None]
                if clicks:
                    coords, point_labels = to_points(clicks)
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=z, obj_id=int(gt_id), points=coords, labels=point_labels,
                        clear_old_points=False,
                    )
                    pred_mask = propagate(z)
                _add_object(seg_per_iter[it], areas[it], pred_mask, gt_id)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return seg_per_iter


def run_sam3_evaluation(
    dataset_name, data_root, experiment_folder,
    start_with_box=True, n_iterations=8, ndim=None, shard=None, score_only=False, limit=None,
):
    """Run SAM3 with iterative prompting on the test split and write one result CSV per iteration.

    The predictions are cached per sample, with the same sharding and scoring as 'run_nninteractive_evaluation'.
    """
    if ndim is None:
        ndim = 3 if dataset_name in DATASETS_3D else 2

    sys.path.insert(0, SAM3_ROOT)
    from micro_sam3.evaluation.inference import build_sam3_image_predictor, build_sam3_video_predictor

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

    prediction_root = os.path.join(experiment_folder, "predictions", f"sam3{dim_suffix}", dataset_name, prompt_str)
    if ndim == 3:
        selected = interactive_sample_ids_3d(dataset_name, data_root, limit)
    else:
        total = n_samples(dataset_name, data_root)
        selected = list(range(total if limit is None else min(total, limit)))
    sample_ids = selected if shard is None else selected[shard[0]::shard[1]]
    samples = load_data(dataset_name, data_root, ndim, sample_ids=set(sample_ids))

    # The model is loaded on the first sample that needs it, so that scoring from the cache loads no model.
    model, processor, predictor = None, None, None
    all_gt, all_valid_rois, missing = [], [], []
    pred_paths_per_iter = [[] for _ in range(n_iterations)]
    samples = tqdm(samples, total=len(sample_ids), desc=f"sam3-{ndim}d")
    for sample_id, (raw, labels, valid_roi) in zip(sample_ids, samples):
        if labels.max() == 0:  # Nothing to prompt, and nothing to score, as in the micro-sam2 evaluation.
            continue
        paths = _cached_prediction_paths(prediction_root, sample_id, n_iterations)
        if not all(os.path.exists(path) for path in paths):
            if score_only:
                missing.append(sample_id)
                continue
            if ndim == 2:
                if model is None:
                    model, processor = build_sam3_image_predictor()
                segs = _segment_sam3_2d(raw, labels, model, processor, start_with_box, n_iterations)
            else:
                if predictor is None:
                    predictor = build_sam3_video_predictor()
                segs = _segment_sam3_3d(raw, labels, predictor, start_with_box, n_iterations)
            _write_cached_predictions(segs, paths)
        all_gt.append(labels)
        all_valid_rois.append(valid_roi)
        for it, path in enumerate(paths):
            pred_paths_per_iter[it].append(path)

    if shard is not None:
        print(f"Shard {shard[0]}/{shard[1]} finished predicting {len(sample_ids)} sample(s).")
        return
    if missing:
        raise RuntimeError(
            f"{len(missing)} sample(s) of '{dataset_name}' have no predictions yet: {missing}. Run their shards first."
        )

    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        predictions = []
        for path, valid_roi in zip(pred_paths_per_iter[it], all_valid_rois):
            prediction = imageio.imread(path)
            if valid_roi is not None:
                prediction[~valid_roi] = 0
            predictions.append(prediction)
        results = run_dataset_evaluation(all_gt, predictions, dataset_name, save_path)
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
    iou_threshold=0.8, box_extension=0.025, seed=None, n_workers=4, shard=None, score_only=False, limit=None,
):
    """Evaluate SAM v1 or micro-sam v1 with iterative prompts and volumetric projection, as in the 3d annotator.

    See `evaluate_interactive_volume_segmentation` for the prompts and the corrections. Each worker process
    segments one volume at a time on the same GPU. The predictions are cached per volume, so a preempted job
    resumes per volume. With 'shard' set to (shard_index, n_shards), only the volumes with
    'sample_id % n_shards == shard_index' are predicted and the function returns before scoring. With
    'score_only' it fails if a volume is missing rather than predict it.
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

    prediction_root = os.path.join(
        experiment_folder, "predictions", f"microsam_vol_{model_type}", dataset_name,
        f"{prompt}{interactive_run_tag(ndim=3, min_size=min_size)}",
    )
    selected = interactive_sample_ids_3d(dataset_name, data_root, limit)
    sample_ids = selected if shard is None else selected[shard[0]::shard[1]]
    loaded = load_data(dataset_name, data_root, 3, min_size=min_size, sample_ids=set(sample_ids))

    # Skip the volumes without objects: there is nothing to prompt, and nothing to score.
    samples, todo, missing = [], [], []
    for sample_id, (raw, labels, valid_roi) in zip(sample_ids, loaded):
        if labels.max() == 0:
            continue
        paths = _cached_prediction_paths(prediction_root, sample_id, n_iterations)
        if not all(os.path.exists(path) for path in paths):
            if score_only:
                missing.append(sample_id)
                continue
            todo.append((raw, labels, paths))
        samples.append((labels, valid_roi, paths))

    options = dict(
        start_prompt="box" if start_with_box else "points", start_slice=start_slice, n_iterations=n_iterations,
        correction=correction, use_previous_mask=use_masks, projection=projection, iou_threshold=iou_threshold,
        box_extension=box_extension, seed=seed,
    )
    segment = partial(_segment_volume, model_type=model_type, checkpoint=checkpoint, device=device, options=options)
    if todo:
        # CUDA cannot run in a forked process, so the workers start with 'spawn'.
        with ProcessPoolExecutor(min(n_workers, len(todo)), mp_context=get_context("spawn")) as pool:
            segs_per_volume = pool.map(segment, *zip(*[(raw, labels) for raw, labels, _ in todo]))
            for segs, (_, _, paths) in tqdm(zip(segs_per_volume, todo), total=len(todo), desc="microsam_vol"):
                _write_cached_predictions(segs, paths)

    if shard is not None:
        print(f"Shard {shard[0]}/{shard[1]} finished predicting {len(sample_ids)} sample(s).")
        return
    if missing:
        raise RuntimeError(
            f"{len(missing)} sample(s) of '{dataset_name}' have no predictions yet: {missing}. Run their shards first."
        )

    all_gt = [labels for labels, _, _ in samples]
    os.makedirs(results_dir, exist_ok=True)
    for it, save_path in enumerate(save_paths):
        if os.path.exists(save_path):
            continue
        predictions = []
        for _, valid_roi, paths in samples:
            prediction = imageio.imread(paths[it])
            if valid_roi is not None:
                prediction[~valid_roi] = 0
            predictions.append(prediction)
        results = run_dataset_evaluation(all_gt, predictions, dataset_name, save_path)
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
    parser.add_argument(
        "--click_protocol", type=str, default="random", choices=CLICK_PROTOCOLS,
        help="sam and micro-sam only. The correction clicks: 'random' positions in the error regions, or the "
             "'center' of the largest error regions with a single mask for the first point, see 'CLICK_PROTOCOLS'.",
    )
    parser.add_argument("--n_samples", type=int, default=None, help="Score only the first N samples, for a check.")
    parser.add_argument(
        "--shard_index", type=int, default=None,
        help="This shard's index (0-based), for splitting a dataset's images or volumes across parallel jobs. "
             "Requires --n_shards. A later unsharded call over the same dataset does the scoring.",
    )
    parser.add_argument("--n_shards", type=int, default=None, help="Total number of shards. Requires --shard_index.")
    parser.add_argument(
        "--score_only", action="store_true",
        help="Score the predictions of finished shards, and fail rather than predict if one is missing.",
    )
    args = parser.parse_args()

    if (args.shard_index is None) != (args.n_shards is None):
        raise ValueError("--shard_index and --n_shards must be given together.")
    if args.score_only and args.shard_index is not None:
        raise ValueError("--score_only scores the whole dataset, so it cannot be combined with a shard.")
    if args.click_protocol != "random" and args.method not in ("sam", "micro-sam"):
        raise ValueError("--click_protocol is supported for sam and micro-sam only.")

    check_data_download(args.dataset_name, args.input_path)

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_with_box = (args.prompt_choice == "box")
    shard = None if args.shard_index is None else (args.shard_index, args.n_shards)

    if args.method == "nninteractive":
        run_nninteractive_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, checkpoint_path=args.checkpoint, start_with_box=start_with_box,
            n_iterations=args.n_iterations, shard=shard, score_only=args.score_only, limit=args.n_samples,
        )

    elif args.method == "sam3":
        run_sam3_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, shard=shard,
            score_only=args.score_only, limit=args.n_samples,
        )

    elif args.method == "sam":
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=args.model_type or SAM_V1_MODEL_TYPE, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="sam",
            use_masks=args.use_masks, min_size=args.min_size, limit=args.n_samples,
            click_protocol=args.click_protocol, shard=shard, score_only=args.score_only,
        )

    elif args.method == "micro-sam":
        is_em = args.dataset_name in EM_DATASETS
        model_type = args.model_type or (MICROSAM_V1_EM_MODEL if is_em else MICROSAM_V1_LM_MODEL)
        run_sam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=model_type, checkpoint=args.checkpoint,
            start_with_box=start_with_box, n_iterations=args.n_iterations, ndim=args.ndim, name_tag="micro-sam",
            use_masks=args.use_masks, min_size=args.min_size, limit=args.n_samples,
            click_protocol=args.click_protocol, shard=shard, score_only=args.score_only,
        )

    elif args.method == "microsam_vol":
        run_microsam_volumetric_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder, device=device,
            model_type=args.model_type, checkpoint=args.checkpoint, start_with_box=start_with_box,
            n_iterations=args.n_iterations, min_size=args.min_size, start_slice=args.start_slice,
            correction=args.correction, use_masks=args.use_masks, projection=args.projection,
            iou_threshold=args.iou_threshold, box_extension=args.box_extension, seed=args.seed,
            n_workers=args.n_workers, shard=shard, score_only=args.score_only, limit=args.n_samples,
        )

    else:
        raise ValueError(f"Unknown method: '{args.method}'.")


if __name__ == "__main__":
    main()
