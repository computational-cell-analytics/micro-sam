import os
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

import torch

from torch_em.transform.raw import normalize
from torch_em.util.segmentation import size_filter

from elf.io import open_file

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from micro_sam.util import segmentation_to_one_hot, mask_data_to_segmentation
from micro_sam.prompt_generators import IterativePromptGenerator
from micro_sam.evaluation.inference import (
    _get_batched_prompts, _get_batched_iterative_prompts, _save_segmentation,
)

from micro_sam.v2.util import _get_device, get_sam2_model, precompute_image_embeddings


def _embedding_tensors_to_numpy(embeddings):
    """Move SAM2 video embeddings to CPU arrays for CustomVideoPredictor.init_state."""
    def _convert(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_convert(v) for v in value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    return _convert(embeddings)


#
# Automatic Mask Generation (AMG) for Images
#


def run_amg(
    image_paths: List[Union[os.PathLike, str]],
    image_key: Optional[str],
    experiment_folder: Union[os.PathLike, str],
    model_type: str,
    backbone: str,
    checkpoint_path: Union[os.PathLike, str],
    device: Optional[Union[torch.device, str]] = None,
    min_object_size: int = 0,
    ensure_8bit: bool = True,
):
    """Functionality for automatic mask generation (AMG) for 2d images.
    """
    # where the predictions are saved
    prediction_dir = os.path.join(experiment_folder, "amg", "inference")
    os.makedirs(prediction_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Run inference for automatic mask generation"):
        image_name = Path(os.path.basename(image_path)).with_suffix(".tif")

        # We skip the images that already have been segmented.
        prediction_path = os.path.join(prediction_dir, image_name)
        if os.path.exists(prediction_path):
            continue

        if image_key is None:
            image = imageio.imread(image_path)
        else:
            image = open_file(image_path)[image_key][:]

        if ensure_8bit and image.max() > 255:
            image = normalize(image) * 255

        if image.ndim == 2:  # Convert single channel images to RGB images.
            image = np.stack([image] * 3, axis=-1)

        model = get_sam2_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path, backbone=backbone)

        mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            pred_iou_thresh=0.6,  # default: 0.8
            stability_score_thresh=0.6,  # default: 0.95
        )

        outputs = mask_generator.generate(image.astype("uint8"))  # NOTE: Done as this is what SAM2 expects as inputs.

        if len(outputs) == 0:  # i.e. no segmentations were found.
            segmentation = np.zeros(image.shape[:2], dtype="uint32")
        else:
            segmentation = mask_data_to_segmentation(
                masks=outputs, with_background=True, min_object_size=min_object_size
            )

        imageio.imwrite(prediction_path, segmentation, compression=5)

    return prediction_dir


#
# Interactive Segmentation for Images
#


@torch.no_grad()
def _run_interactive_segmentation_2d_per_image(
    image: np.ndarray,
    gt: np.ndarray,
    prediction_paths: Union[os.PathLike, str],
    predictor,
    start_with_box_prompt: bool = False,
    dilation: int = 5,
    device: Optional[Union[torch.device, str]] = None,
    n_iterations: int = 8,
    use_masks: bool = False,
    batch_size: int = 32,
) -> None:
    """Functionality for interactive segmentation per 2d image.
    """
    device = _get_device(device)

    # Let's define the iterative prompt generator.
    prompt_generator = IterativePromptGenerator()

    # Preparing prompts for the first iteration: use multimasking only if we have a single positive prompt without box
    if start_with_box_prompt:
        use_boxes, use_points = True, False
        n_positive = 0
        multimasking = False
    else:
        use_boxes, use_points = False, True
        n_positive = 1
        multimasking = True

    # Expects RGB-style images.
    predictor.set_image(image.astype("uint8"))  # NOTE: Done as this is what SAM2 expects as inputs.

    gt_ids = np.unique(gt)[1:]

    points, point_labels, boxes = _get_batched_prompts(
        gt=gt,
        gt_ids=gt_ids,
        use_points=use_points,
        use_boxes=use_boxes,
        n_positives=n_positive,
        n_negatives=0,
        dilation=dilation,
    )

    sampled_binary_y = segmentation_to_one_hot(segmentation=gt.astype("int64"), segmentation_ids=gt_ids)

    for iteration in range(n_iterations):
        if iteration == 0:  # logits masks cannot be used for the first iteration.
            logits_masks = None
        else:
            if not use_masks:  # logits masks should not be used when not desired.
                logits_masks = None

        # scripts for batched inference
        # TODO: we move this in a separate script the same as micro-sam
        n_prompts = boxes.shape[0] if use_boxes else points.shape[0]
        n_batches = int(np.ceil(float(n_prompts) / batch_size))

        masks, logits = [], []
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_stop = min((batch_idx + 1) * batch_size, n_prompts)

            batch_boxes = boxes[batch_start: batch_stop] if use_boxes else None
            batch_points = points[batch_start: batch_stop] if (use_points or points is not None) else None
            batch_point_labels = point_labels[batch_start: batch_stop] if (use_points or point_labels is not None) \
                else None
            batch_logits_masks = logits_masks[batch_start: batch_stop] if use_masks and logits_masks is not None \
                else None

            batch_masks, batch_scores, batch_logits = predictor.predict(
                point_coords=batch_points,
                point_labels=batch_point_labels,
                box=batch_boxes,
                mask_input=batch_logits_masks,
                multimask_output=multimasking,
            )

            if batch_scores.ndim == 2:
                max_index = batch_scores.argmax(axis=1)
            elif batch_scores.ndim == 1:
                max_index = batch_scores.argmax()
                max_index = np.array([max_index])
                batch_masks, batch_logits = batch_masks[None], batch_logits[None]
            else:
                raise ValueError

            if multimasking:
                batch_masks = np.stack([batch_masks[i, max_id][None] for i, max_id in enumerate(max_index)])
                batch_logits = np.stack([batch_logits[i, max_id][None] for i, max_id in enumerate(max_index)])

            masks.append(batch_masks)
            logits.append(batch_logits)

        masks = np.concatenate(masks)
        logits_masks = np.concatenate(logits)

        # switching off multimasking after first iter, as next iters (with multiple prompts) don't expect multimasking
        multimasking = False

        next_coords, next_labels = _get_batched_iterative_prompts(
            sampled_binary_gt=sampled_binary_y,
            masks=torch.from_numpy(masks).to(torch.float32),
            batch_size=batch_size,
            prompt_generator=prompt_generator
        )
        next_coords, next_labels = next_coords.detach().cpu().numpy(), next_labels.detach().cpu().numpy()

        if points is not None:
            points = np.concatenate([points, next_coords], axis=1)
        else:
            points = next_coords

        if point_labels is not None:
            point_labels = np.concatenate([point_labels, next_labels], axis=1)
        else:
            point_labels = next_labels

        _save_segmentation(masks=torch.from_numpy(masks), prediction_path=prediction_paths[iteration])


def run_interactive_segmentation_2d(
    image_paths: List[Union[os.PathLike, str]],
    gt_paths: List[Union[os.PathLike, str]],
    image_key: Optional[str],
    gt_key: Optional[str],
    prediction_dir: Union[os.PathLike, str],
    model_type: str,
    backbone: str,
    checkpoint_path: Union[os.PathLike, str],
    start_with_box_prompt: bool = False,
    device: Optional[Union[torch.device, str]] = None,
    n_iterations: int = 8,
    dilation: int = 5,
    batch_size: int = 32,
    use_masks: bool = False,
    ensure_8bit: bool = True,
):
    """Functionality for interactive segmentation in 2d images using iterative prompting.
    """
    if len(image_paths) != len(gt_paths):
        raise ValueError(f"Expect same number of images and gt images, got {len(image_paths)}, {len(gt_paths)}")

    # create all prediction folders for all intermediate iterations'
    prediction_dir = os.path.join(prediction_dir, "start_with_box" if start_with_box_prompt else "start_with_point")

    if use_masks:
        print("The iterative prompting will make use of logits masks from previous iterations.")
        prediction_dir = os.path.join(prediction_dir, "with_masks")
    else:
        prediction_dir = os.path.join(prediction_dir, "without_masks")

    for i in range(n_iterations):
        os.makedirs(os.path.join(prediction_dir, f"iteration{i:02}"), exist_ok=True)

    model = get_sam2_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path, backbone=backbone)
    predictor = SAM2ImagePredictor(model)

    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Run inference with iterative prompting",
    ):
        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        image_name = Path(os.path.basename(image_path)).with_suffix(".tif")

        # We skip the images that have already been segmented
        prediction_paths = [os.path.join(prediction_dir, f"iteration{i:02}", image_name) for i in range(n_iterations)]
        if all(os.path.exists(prediction_path) for prediction_path in prediction_paths):
            continue

        if image_key is None:
            image = imageio.imread(image_path)
        else:
            image = open_file(image_path)[image_key][:]

        if ensure_8bit and image.max() > 255:
            image = normalize(image) * 255

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        if gt_key is None:
            gt = imageio.imread(gt_path)
        else:
            gt = open_file(gt_path)[gt_key][:]

        # Run connected components on labels
        gt = connected_components(gt).astype("uint32")

        if len(np.unique(gt)) == 1:
            continue  # skipping the image as there are no labels.

        _run_interactive_segmentation_2d_per_image(
            image=image,
            gt=gt,
            prediction_paths=prediction_paths,
            predictor=predictor,
            start_with_box_prompt=start_with_box_prompt,
            device=device,
            dilation=dilation,
            n_iterations=n_iterations,
            use_masks=use_masks,
            batch_size=batch_size,
        )

    return prediction_dir


#
# Interactive Segmentation for Multi-Dimensional Inputs
#


def _convert_volumes_to_frames(raw, frames_dir):
    """Converts whole volumes to individual frames and stores them sequentially in a directory.
    """
    # Let's convert the inputs to individual slices as tifs and pair them with it's respective labels
    if os.path.exists(frames_dir):  # We remove the previously stored frames.
        shutil.rmtree(frames_dir)

    image_dir = os.path.join(frames_dir, "images")  # Store slices in the directory.
    os.makedirs(image_dir, exist_ok=True)

    # Save all slices in order.
    [imageio.imwrite(os.path.join(image_dir, f"{i:04}.tif"), fraw) for i, fraw in enumerate(raw)]

    return image_dir


def run_interactive_segmentation_3d(
    raw: np.ndarray,
    labels: np.ndarray,
    model_type: str,
    backbone: str,
    checkpoint_path: Union[os.PathLike, str],
    start_with_box_prompt: bool,
    prediction_dir: Union[os.PathLike, str],
    prediction_fname: str = "segmentation",
    dilation: int = 5,
    device: Optional[Union[torch.device, str]] = None,
    first_prompt_at_center: bool = True,  # NOTE: With our current schema, we always put the first prompt in center.
    min_size: int = 0,
    batch_size: int = 16,
    n_iterations: int = 8,
    run_connected_components: bool = True,
) -> str:
    """Run interactive segmentation on 3d inputs using Segment Anything 2.

    Args:
        raw: The raw input array.
        labels: The corresponding ground-truth array with instance labels.
        model_type: The choice of Segment Anything 2 model.
        backbone: The backbone of Segment Anything 2 model. Either 'sam2' or 'sam2.1'.
        checkpoint_path: The filepath to the model checkpoints.
        start_with_box_prompt: Whether the iterative prompting starts with 'box' as first prompt
            (else 'points' will be used).
        prediction_dir: The filepath where the predictions will be stored.
        prediction_fname: The name for storing the segmentations.
        dilation: The dilation factor for the radius around the ground-truth object around which
            points will not be sampled.
        device: The torch device.
        first_prompt_at_center: Whether to place the first prompt at center.
        min_size: The minimum pixel criterion to accept instance objects for interactive segmentation.
        batch_size: The batch size to compute prompts.
        n_iterations: The number of iterations for iterative prompting.
        run_connected_components: Whether to ensure individual instances and filter out small objects.

    Returns:
        The folder where segmentations are stored.
    """
    prediction_dir = os.path.join(
        prediction_dir, "interactive_segmentation_3d", "start_with_box" if start_with_box_prompt else "start_with_point",
        "without_masks",
    )

    prediction_paths = [
        os.path.join(
            prediction_dir, f"iteration{i}", Path(prediction_fname).with_suffix(".tif")
        ) for i in range(n_iterations)
    ]
    if all([os.path.exists(_path) for _path in prediction_paths]):
        print(f"The results are stored at '{prediction_dir}'.")
        return prediction_dir

    # Get the device.
    device = _get_device(device)

    if run_connected_components:
        # Ensuring sequential instance labels.
        labels = connected_components(labels).astype(labels.dtype)
        # NOTE: There are some objects covering very few pixels, eg. in lucchi, causing troubles in iterative prompting.
        labels = size_filter(seg=labels, min_size=min_size)

    # Preparing prompts for the first iteration
    if start_with_box_prompt:
        use_boxes, use_points = True, False
        n_positive = 0
    else:
        use_boxes, use_points = False, True
        n_positive = 1

    # Get the SAM2 predictor.
    predictor = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=checkpoint_path, input_type="videos", backbone=backbone,
    )
    # Match training: frames that receive correction clicks become conditioning frames so
    # their corrected predictions are preserved and used as memory for neighboring frames.
    predictor.add_all_frames_to_correct_as_cond = True

    # Initialize the inference state
    volume_embeddings = _embedding_tensors_to_numpy(
        precompute_image_embeddings(predictor=predictor, input_=raw, ndim=3)
    )
    inference_state = predictor.init_state(volume=raw, volume_embeddings=volume_embeddings)

    gt_ids = list(np.unique(labels))[1:]  # Ignoring the background label
    segmentation = []
    for gt_id in tqdm(gt_ids, desc="Segmenting per object in the volume"):
        _per_iter_segmentation = _run_interactive_segmentation_3d_per_object(
            gt_ids=gt_id,
            labels=labels,
            predictor=predictor,
            inference_state=inference_state,
            first_prompt_at_center=first_prompt_at_center,
            use_points=use_points,
            use_boxes=use_boxes,
            n_positive=n_positive,
            dilation=dilation,
            batch_size=batch_size,
            n_iterations=n_iterations,
        )

        for _iter in range(n_iterations):
            if gt_id == gt_ids[0]:  # Add segmentations to the list for first object.
                segmentation.append(_per_iter_segmentation[_iter])
            else:  # Merge incoming segmentations per object to existing seg. array.
                segmentation[_iter] += _per_iter_segmentation[_iter]

    for i, prediction_path in enumerate(prediction_paths):
        os.makedirs(Path(prediction_path).parent, exist_ok=True)
        imageio.imwrite(os.path.join(prediction_path), segmentation[i], compression="zlib")

    return prediction_dir


@torch.no_grad()
def _run_interactive_segmentation_3d_per_object(
    gt_ids: Union[List[int], int],
    labels: np.ndarray,
    predictor,
    inference_state,
    first_prompt_at_center: bool,
    use_points: bool,
    use_boxes: bool,
    n_positive: int,
    dilation: int = 5,
    batch_size: int = 32,
    n_iterations: int = 8,
):
    """Functionality for interactive segmentation using iterative prompting.
    """
    if not isinstance(gt_ids, list):
        gt_ids = [gt_ids]

    # Get prompts for the desired frame and desired object id for the first iteration.
    id_to_prompts = _extract_prompts_per_object(
        labels=labels,
        gt_ids=gt_ids,
        first_prompt_at_center=first_prompt_at_center,
        use_points=use_points,
        use_boxes=use_boxes,
        n_positive=n_positive,
        dilation=dilation,
    )

    # Get segmentation for iterative prompting per object.
    preds_per_object = _get_iteratively_prompted_segmentation_per_image_dir(
        inference_state=inference_state,
        labels=labels,
        id_to_prompts=id_to_prompts,
        predictor=predictor,
        batch_size=batch_size,
        n_iterations=n_iterations,
    )

    assert len(gt_ids) == len(preds_per_object), "The number of label ids should match the number of objects segmented."

    seg_per_iterations = []
    for gt_id, _preds_per_iters in zip(gt_ids, preds_per_object):  # Access interactive segmentation per object.
        assert len(_preds_per_iters) == n_iterations

        for _iter in range(n_iterations):  # Acces per-object per-iteration predictions.
            _pred_per_iter = _preds_per_iters[_iter]
            assert isinstance(_pred_per_iter, np.ndarray)

            # Replace the id of foreground with the expect object id
            _pred_per_iter[_pred_per_iter > 0] = gt_id

            # Let's merge the prediction per iteration at label id.
            if gt_id == gt_ids[0]:  # If the array is not a part of the list.
                seg_per_iterations.append(_pred_per_iter)
            else:  # If this is a existing array, we merge the incoming labels
                seg_per_iterations[_iter] += _pred_per_iter

    for seg in seg_per_iterations:
        if seg.shape != labels.shape:
            raise AssertionError(f"'{seg.shape}' not equal to the expect shape: '{labels.shape}'")

    return seg_per_iterations


def _extract_prompts_per_object(
    labels, gt_ids, first_prompt_at_center, use_points, use_boxes, n_positive, dilation,
):
    """Extracts input prompts per object for interactive segmentation.
    """
    ids_to_prompts = {}
    for gt_id in gt_ids:
        # The first thing to do is find where all the objects start from and end to
        z_slices = np.where(labels == gt_id)[0]
        z_range = (z_slices.min(), z_slices.max())

        if first_prompt_at_center:  # choose the middle slice of the object
            z_choice = int(np.ceil(np.mean(z_range)))
        else:  # choose the first slice of the object
            z_choice = z_range[0]

        points, point_labels, boxes = _get_batched_prompts(
            gt=(labels[z_choice] == gt_id).astype("uint32"),  # binary label of the particular slice per object
            gt_ids=[1],  # see above line: binary labels per object
            use_points=use_points, use_boxes=use_boxes, n_positives=n_positive, n_negatives=0, dilation=dilation,
        )
        ids_to_prompts[gt_id] = [z_choice, points, point_labels, boxes]

    return ids_to_prompts


@torch.no_grad()
def _get_iteratively_prompted_segmentation_per_image_dir(
    inference_state, labels, id_to_prompts, predictor, batch_size=32, n_iterations=8,
):
    """Functionality for inference of 3d interactive segmentation.
    """
    # Let's define the iterative prompt generator.
    prompt_generator = IterativePromptGenerator()

    pred_per_object = []
    for _obj_id, _input_prompts in id_to_prompts.items():
        z_cond, _point, _label, _box = _input_prompts  # z_cond stays fixed for all iterations

        list_of_iterations = list(range(n_iterations))
        pred_per_iteration = []
        _corr_points = None  # (x,y) correction points for the next iteration
        _corr_labels = None
        _corr_frame = None   # z-slice where corrections will be applied

        for iteration in list_of_iterations:
            if iteration == 0:
                # First iteration: initial box or point at the conditioning frame.
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state, frame_idx=z_cond, obj_id=_obj_id,
                    points=_point, labels=_label, box=_box,
                )
            else:
                # Subsequent iterations: add corrective points at the worst-predicted frame.
                # The video predictor automatically loads prev_sam_mask_logits from the
                # previous propagation at that frame, matching the mask_inputs feedback used
                # during training (SAM2Train._iter_correct_pt_sampling, line 492).
                for per_point, per_label in zip(_corr_points, _corr_labels):
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state, frame_idx=_corr_frame, obj_id=_obj_id,
                        points=np.array([per_point]), labels=np.array([per_label]),
                        box=None, clear_old_points=False,
                    )

            # Propagate the masklets throughout the frames using the input prompts in selected frames
            forward_video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                forward_video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Propagate reverse in time if needed.
            reverse_video_segments = {}
            if len(forward_video_segments) < labels.shape[0]:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state, reverse=True,
                ):
                    reverse_video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                    }
                reverse_video_segments = dict(reversed(list(reverse_video_segments.items())))

            video_segments = {**reverse_video_segments, **forward_video_segments}

            segmentation = []
            for slice_idx in video_segments.keys():
                per_slice_seg = np.zeros(labels.shape[-2:])
                for _instance_idx, _instance_mask in video_segments[slice_idx].items():
                    per_slice_seg[_instance_mask.squeeze()] = _instance_idx
                segmentation.append(per_slice_seg)

            segmentation = (np.stack(segmentation) > 0).astype("uint64")
            pred_per_iteration.append(segmentation)

            if len(list_of_iterations) > 1 and iteration < list_of_iterations[-1]:
                # Find the z-slice where the prediction is worst for this object.
                # Restrict to slices where the object exists so IterativePromptGenerator
                # always has FN pixels (or overlap) to sample a positive point from.
                gt_3d = (labels == _obj_id)
                obj_z_slices = np.where(gt_3d.any(axis=(1, 2)))[0]
                errors_per_slice = np.array([
                    np.sum(gt_3d[z] != (segmentation[z] > 0)) for z in obj_z_slices
                ])
                z_worst = int(obj_z_slices[np.argmax(errors_per_slice)])

                gt_slice = gt_3d[z_worst].astype("int64")
                pred_slice = (segmentation[z_worst] > 0).astype("int64")
                next_coords, next_labels = _get_batched_iterative_prompts(
                    sampled_binary_gt=torch.from_numpy(gt_slice)[None, None].to(torch.float32),
                    masks=torch.from_numpy(pred_slice)[None, None].to(torch.float32),
                    batch_size=batch_size,
                    prompt_generator=prompt_generator,
                )
                next_coords = next_coords.detach().cpu().numpy()  # [1, 2, 2]: (obj, pos+neg, xy)
                next_labels = next_labels.detach().cpu().numpy()  # [1, 2]
                _corr_points = next_coords[0]   # [[x_pos, y_pos], [x_neg, y_neg]]
                _corr_labels = next_labels[0]   # [1, 0]
                _corr_frame = z_worst

        pred_per_object.append(pred_per_iteration)

        # Reset the state after tracking
        predictor.reset_state(inference_state)

    return pred_per_object
