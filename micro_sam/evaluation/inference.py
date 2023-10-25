"""Inference with Segment Anything models and different prompt strategies.
"""

import os
import pickle
import warnings

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import imageio.v3 as imageio
import numpy as np
import torch

from skimage.segmentation import relabel_sequential
from tqdm import tqdm

from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from .. import util as util
from ..instance_segmentation import mask_data_to_segmentation
from ..prompt_generators import PointAndBoxPromptGenerator, IterativePromptGenerator
from ..training import get_trainable_sam_model, ConvertToSamInputs


def _load_prompts(
    cached_point_prompts, save_point_prompts,
    cached_box_prompts, save_box_prompts,
    image_name
):

    def load_prompt_type(cached_prompts, save_prompts):
        # Check if we have saved prompts.
        if cached_prompts is None or save_prompts:  # we don't have cached prompts
            return cached_prompts, None

        # we have cached prompts, but they have not been loaded yet
        if isinstance(cached_prompts, str):
            with open(cached_prompts, "rb") as f:
                cached_prompts = pickle.load(f)

        prompts = cached_prompts[image_name]
        return cached_prompts, prompts

    cached_point_prompts, point_prompts = load_prompt_type(cached_point_prompts, save_point_prompts)
    cached_box_prompts, box_prompts = load_prompt_type(cached_box_prompts, save_box_prompts)

    # we don't have anything cached
    if point_prompts is None and box_prompts is None:
        return None, cached_point_prompts, cached_box_prompts

    if point_prompts is None:
        input_point, input_label = [], []
    else:
        input_point, input_label = point_prompts

    if box_prompts is None:
        input_box = []
    else:
        input_box = box_prompts

    prompts = (input_point, input_label, input_box)
    return prompts, cached_point_prompts, cached_box_prompts


def _get_batched_prompts(
    gt,
    gt_ids,
    use_points,
    use_boxes,
    n_positives,
    n_negatives,
    dilation,
    transform_function,
):
    # Initialize the prompt generator.
    prompt_generator = PointAndBoxPromptGenerator(
        n_positive_points=n_positives, n_negative_points=n_negatives,
        dilation_strength=dilation, get_point_prompts=use_points,
        get_box_prompts=use_boxes
    )

    # Generate the prompts.
    center_coordinates, bbox_coordinates = util.get_centers_and_bounding_boxes(gt)
    center_coordinates = [center_coordinates[gt_id] for gt_id in gt_ids]
    bbox_coordinates = [bbox_coordinates[gt_id] for gt_id in gt_ids]
    masks = util.segmentation_to_one_hot(gt.astype("int64"), gt_ids)

    input_points, input_labels, input_boxes, _ = prompt_generator(
        masks, bbox_coordinates, center_coordinates
    )

    # apply the transforms to the points and boxes
    if use_boxes:
        input_boxes = torch.from_numpy(
            transform_function.apply_boxes(input_boxes.numpy(), gt.shape)
        )
    if use_points:
        input_points = torch.from_numpy(
            transform_function.apply_coords(input_points.numpy(), gt.shape)
        )

    return input_points, input_labels, input_boxes


def _run_inference_with_prompts_for_image(
    predictor,
    gt,
    use_points,
    use_boxes,
    n_positives,
    n_negatives,
    dilation,
    batch_size,
    cached_prompts,
):
    # We need the resize transformation for the expected model input size.
    transform_function = ResizeLongestSide(1024)
    gt_ids = np.unique(gt)[1:]

    if cached_prompts is None:
        input_points, input_labels, input_boxes = _get_batched_prompts(
            gt, gt_ids, use_points, use_boxes, n_positives, n_negatives, dilation, transform_function,
        )
    else:
        input_points, input_labels, input_boxes = cached_prompts

    # Make a copy of the point prompts to return them at the end.
    prompts = deepcopy((input_points, input_labels, input_boxes))

    # Transform the prompts into batches
    device = predictor.device
    input_points = None if input_points is None else torch.tensor(np.array(input_points), dtype=torch.float32).to(device)
    input_labels = None if input_labels is None else torch.tensor(np.array(input_labels), dtype=torch.float32).to(device)
    input_boxes = None if input_boxes is None else torch.tensor(np.array(input_boxes), dtype=torch.float32).to(device)

    # Use multi-masking only if we have a single positive point without box
    multimasking = False
    if not use_boxes and (n_positives == 1 and n_negatives == 0):
        multimasking = True

    # Run the batched inference.
    n_samples = input_boxes.shape[0] if input_points is None else input_points.shape[0]
    n_batches = int(np.ceil(float(n_samples) / batch_size))
    masks, ious = [], []
    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_stop = min((batch_idx + 1) * batch_size, n_samples)

            batch_points = None if input_points is None else input_points[batch_start:batch_stop]
            batch_labels = None if input_labels is None else input_labels[batch_start:batch_stop]
            batch_boxes = None if input_boxes is None else input_boxes[batch_start:batch_stop]

            batch_masks, batch_ious, _ = predictor.predict_torch(
                point_coords=batch_points, point_labels=batch_labels,
                boxes=batch_boxes, multimask_output=multimasking
            )
            masks.append(batch_masks)
            ious.append(batch_ious)
    masks = torch.cat(masks)
    ious = torch.cat(ious)
    assert len(masks) == len(ious) == n_samples

    # TODO we should actually use non-max suppression here
    # I will implement it somewhere to have it refactored
    instance_labels = np.zeros_like(gt, dtype=int)
    for m, iou, gt_idx in zip(masks, ious, gt_ids):
        best_idx = torch.argmax(iou)
        best_mask = m[best_idx]
        instance_labels[best_mask.detach().cpu().numpy()] = gt_idx

    return instance_labels, prompts


def get_predictor(
    checkpoint_path: Union[str, os.PathLike],
    model_type: str,
    device: Optional[str] = None,
    return_state: bool = False,
    is_custom_model: Optional[bool] = None,
) -> SamPredictor:
    """Get the segment anything predictor from an exported or custom checkpoint.

    Args:
        checkpoint_path: The checkpoint filepath.
        model_type: The type of the model, either vit_h, vit_b or vit_l.
        return_state: Whether to return the complete state of the checkpoint in addtion to the predictor.
        is_custom_model: Whether this is a custom model or not.
    Returns:
        The segment anything predictor.
    """
    device = util._get_device(device)

    # By default we check if the model follows the torch_em checkpint naming scheme to check whether it is a
    # custom model or not. This can be over-ridden by passing True or False for is_custom_model.
    is_custom_model = checkpoint_path.split("/")[-1] == "best.pt" if is_custom_model is None else is_custom_model

    if is_custom_model:  # Finetuned SAM model
        predictor = util.get_custom_sam_model(
            checkpoint_path=checkpoint_path, model_type=model_type, device=device, return_state=return_state
        )
    else:  # Vanilla SAM model
        assert not return_state
        predictor = util.get_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path)  # type: ignore
    return predictor


def precompute_all_embeddings(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
) -> None:
    """Precompute all image embeddings.

    To enable running different inference tasks in parallel afterwards.

    Args:
        predictor: The SegmentAnything predictor.
        image_paths: The image file paths.
        embedding_dir: The directory where the embeddings will be saved.
    """
    for image_path in tqdm(image_paths, desc="Precompute embeddings"):
        image_name = os.path.basename(image_path)
        im = imageio.imread(image_path)
        embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")
        util.precompute_image_embeddings(predictor, im, embedding_path, ndim=2)


def _precompute_prompts(gt_path, use_points, use_boxes, n_positives, n_negatives, dilation, transform_function):
    name = os.path.basename(gt_path)

    gt = imageio.imread(gt_path).astype("uint32")
    gt = relabel_sequential(gt)[0]
    gt_ids = np.unique(gt)[1:]

    input_point, input_label, input_box = _get_batched_prompts(
        gt, gt_ids, use_points, use_boxes, n_positives, n_negatives, dilation, transform_function
    )

    if use_boxes and not use_points:
        return name, input_box
    return name, (input_point, input_label)


def precompute_all_prompts(
    gt_paths: List[Union[str, os.PathLike]],
    prompt_save_dir: Union[str, os.PathLike],
    prompt_settings: List[Dict[str, Any]],
) -> None:
    """Precompute all point prompts.

    To enable running different inference tasks in parallel afterwards.

    Args:
        gt_paths: The file paths to the ground-truth segmentations.
        prompt_save_dir: The directory where the prompt files will be saved.
        prompt_settings: The settings for which the prompts will be computed.
    """
    os.makedirs(prompt_save_dir, exist_ok=True)
    transform_function = ResizeLongestSide(1024)

    for settings in tqdm(prompt_settings, desc="Precompute prompts"):

        use_points, use_boxes = settings["use_points"], settings["use_boxes"]
        n_positives, n_negatives = settings["n_positives"], settings["n_negatives"]
        dilation = settings.get("dilation", 5)

        # check if the prompts were already computed
        if use_boxes and not use_points:
            prompt_save_path = os.path.join(prompt_save_dir, "boxes.pkl")
        else:
            prompt_save_path = os.path.join(prompt_save_dir, f"points-p{n_positives}-n{n_negatives}.pkl")
        if os.path.exists(prompt_save_path):
            continue

        results = []
        for gt_path in tqdm(gt_paths, desc=f"Precompute prompts for p{n_positives}-n{n_negatives}"):
            prompts = _precompute_prompts(
                gt_path,
                use_points=use_points,
                use_boxes=use_boxes,
                n_positives=n_positives,
                n_negatives=n_negatives,
                dilation=dilation,
                transform_function=transform_function,
            )
            results.append(prompts)

        saved_prompts = {res[0]: res[1] for res in results}
        with open(prompt_save_path, "wb") as f:
            pickle.dump(saved_prompts, f)


def _get_prompt_caching(prompt_save_dir, use_points, use_boxes, n_positives, n_negatives):

    def get_prompt_type_caching(use_type, save_name):
        if not use_type:
            return None, False, None

        prompt_save_path = os.path.join(prompt_save_dir, save_name)
        if os.path.exists(prompt_save_path):
            print("Using precomputed prompts from", prompt_save_path)
            # We delay loading the prompts, so we only have to load them once they're needed the first time.
            # This avoids loading the prompts (which are in a big pickle file) if all predictions are done already.
            cached_prompts = prompt_save_path
            save_prompts = False
        else:
            print("Saving prompts in", prompt_save_path)
            cached_prompts = {}
            save_prompts = True
        return cached_prompts, save_prompts, prompt_save_path

    # Check if prompt serialization is enabled.
    # If it is then load the prompts if they are already cached and otherwise store them.
    if prompt_save_dir is None:
        print("Prompts are not cached.")
        cached_point_prompts, cached_box_prompts = None, None
        save_point_prompts, save_box_prompts = False, False
        point_prompt_save_path, box_prompt_save_path = None, None
    else:
        cached_point_prompts, save_point_prompts, point_prompt_save_path = get_prompt_type_caching(
            use_points, f"points-p{n_positives}-n{n_negatives}.pkl"
        )
        cached_box_prompts, save_box_prompts, box_prompt_save_path = get_prompt_type_caching(
            use_boxes, "boxes.pkl"
        )

    return (cached_point_prompts, save_point_prompts, point_prompt_save_path,
            cached_box_prompts, save_box_prompts, box_prompt_save_path)


def run_inference_with_prompts(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    gt_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    prediction_dir: Union[str, os.PathLike],
    use_points: bool,
    use_boxes: bool,
    n_positives: int,
    n_negatives: int,
    dilation: int = 5,
    prompt_save_dir: Optional[Union[str, os.PathLike]] = None,
    batch_size: int = 512,
) -> None:
    """Run segment anything inference for multiple images using prompts derived form groundtruth.

    Args:
        predictor: The SegmentAnything predictor.
        image_paths: The image file paths.
        gt_paths: The ground-truth segmentation file paths.
        embedding_dir: The directory where the image embddings will be saved or are already saved.
        use_points: Whether to use point prompts.
        use_boxes: Whetehr to use box prompts
        n_positives: The number of positive point prompts that will be sampled.
        n_negativess: The number of negative point prompts that will be sampled.
        dilation: The dilation factor for the radius around the ground-truth object
            around which points will not be sampled.
        prompt_save_dir: The directory where point prompts will be saved or are already saved.
            This enables running multiple experiments in a reproducible manner.
        batch_size: The batch size used for batched prediction.
    """
    if not (use_points or use_boxes):
        raise ValueError("You need to use at least one of point or box prompts.")

    if len(image_paths) != len(gt_paths):
        raise ValueError(f"Expect same number of images and gt images, got {len(image_paths)}, {len(gt_paths)}")

    (cached_point_prompts, save_point_prompts, point_prompt_save_path,
     cached_box_prompts, save_box_prompts, box_prompt_save_path) = _get_prompt_caching(
         prompt_save_dir, use_points, use_boxes, n_positives, n_negatives
     )

    os.makedirs(prediction_dir, exist_ok=True)
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Run inference with prompts"
    ):
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(gt_path)

        # We skip the images that already have been segmented.
        prediction_path = os.path.join(prediction_dir, image_name)
        if os.path.exists(prediction_path):
            continue

        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        im = imageio.imread(image_path)
        gt = imageio.imread(gt_path).astype("uint32")
        gt = relabel_sequential(gt)[0]

        embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")
        image_embeddings = util.precompute_image_embeddings(predictor, im, embedding_path, ndim=2)
        util.set_precomputed(predictor, image_embeddings)

        this_prompts, cached_point_prompts, cached_box_prompts = _load_prompts(
            cached_point_prompts, save_point_prompts,
            cached_box_prompts, save_box_prompts,
            label_name
        )
        instances, this_prompts = _run_inference_with_prompts_for_image(
            predictor, gt, n_positives=n_positives, n_negatives=n_negatives,
            dilation=dilation, use_points=use_points, use_boxes=use_boxes,
            batch_size=batch_size, cached_prompts=this_prompts
        )

        if save_point_prompts:
            cached_point_prompts[label_name] = this_prompts[:2]
        if save_box_prompts:
            cached_box_prompts[label_name] = this_prompts[-1]

        # It's important to compress here, otherwise the predictions would take up a lot of space.
        imageio.imwrite(prediction_path, instances, compression=5)

    # Save the prompts if we run experiments with prompt caching and have computed them
    # for the first time.
    if save_point_prompts:
        with open(point_prompt_save_path, "wb") as f:
            pickle.dump(cached_point_prompts, f)
    if save_box_prompts:
        with open(box_prompt_save_path, "wb") as f:
            pickle.dump(cached_box_prompts, f)


def _save_segmentation(masks, prediction_path):
    # masks to segmentation
    masks = masks.cpu().numpy().squeeze().astype("bool")
    shape = masks.shape[-2:]
    masks = [{"segmentation": mask, "area": mask.sum()} for mask in masks]
    segmentation = mask_data_to_segmentation(masks, shape, with_background=True)
    imageio.imwrite(prediction_path, segmentation)


def _run_inference_with_iterative_prompting_for_image(
    model,
    image,
    gt,
    n_iterations,
    device,
    use_boxes,
    prediction_paths,
    batch_size,
):
    assert len(prediction_paths) == n_iterations, f"{len(prediction_paths)}, {n_iterations}"
    to_sam_inputs = ConvertToSamInputs()

    image = torch.from_numpy(
        image[None, None] if image.ndim == 2 else image[None]
    )
    gt = torch.from_numpy(gt[None].astype("int32"))

    n_pos = 0 if use_boxes else 1
    batched_inputs, sampled_ids = to_sam_inputs(image, gt, n_pos=n_pos, n_neg=0, get_boxes=use_boxes)

    input_images = torch.stack([model.preprocess(x=x["image"].to(device)) for x in batched_inputs], dim=0)
    image_embeddings = model.image_embeddings_oft(input_images)

    multimasking = n_pos == 1
    prompt_generator = IterativePromptGenerator()

    n_samples = len(sampled_ids[0])
    n_batches = int(np.ceil(float(n_samples) / batch_size))

    for iteration in range(n_iterations):
        final_masks = []
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_stop = min((batch_idx + 1) * batch_size, n_samples)

            this_batched_inputs = [{
                k: v[batch_start:batch_stop] if k in ("point_coords", "point_labels") else v
                for k, v in batched_inputs[0].items()
            }]

            sampled_binary_y = torch.stack([
                torch.stack([_gt == idx for idx in sampled[batch_start:batch_stop]])[:, None]
                for _gt, sampled in zip(gt, sampled_ids)
            ]).to(torch.float32)

            batched_outputs = model(
                this_batched_inputs,
                multimask_output=multimasking if iteration == 0 else False,
                image_embeddings=image_embeddings
            )

            masks, logits_masks = [], []
            for m in batched_outputs:
                mask, l_mask = [], []
                for _m, _l, _iou in zip(m["masks"], m["low_res_masks"], m["iou_predictions"]):
                    best_iou_idx = torch.argmax(_iou)
                    mask.append(torch.sigmoid(_m[best_iou_idx][None]))
                    l_mask.append(_l[best_iou_idx][None])
                mask, l_mask = torch.stack(mask), torch.stack(l_mask)
                masks.append(mask)
                logits_masks.append(l_mask)

            masks, logits_masks = torch.stack(masks), torch.stack(logits_masks)
            masks = (masks > 0.5).to(torch.float32)
            final_masks.append(masks)

            for _pred, _gt, _inp, logits in zip(masks, sampled_binary_y, this_batched_inputs, logits_masks):
                next_coords, next_labels, _, _ = prompt_generator(_gt, _pred)
                updated_point_coords = torch.cat([_inp["point_coords"], next_coords], dim=1) \
                    if "point_coords" in _inp.keys() else next_coords
                updated_point_labels = torch.cat([_inp["point_labels"], next_labels], dim=1) \
                    if "point_labels" in _inp.keys() else next_labels

                _inp["point_coords"] = updated_point_coords
                _inp["point_labels"] = updated_point_labels
                _inp["mask_inputs"] = logits

        final_masks = torch.cat(final_masks, dim=1)
        _save_segmentation(final_masks, prediction_paths[iteration])


def run_inference_with_iterative_prompting(
    checkpoint_path: Union[str, os.PathLike],
    model_type: str,
    image_paths: List[Union[str, os.PathLike]],
    gt_paths: List[Union[str, os.PathLike]],
    prediction_root: Union[str, os.PathLike],
    use_boxes: bool,
    device: Optional[str] = None,
    n_iterations: int = 8,
    batch_size: int = 32,
) -> None:
    """@private"""
    warnings.warn("The iterative prompting functionality is not working correctly yet.")

    device = util._get_device(device)
    model = get_trainable_sam_model(model_type, checkpoint_path)

    # create all prediction folders
    for i in range(n_iterations):
        os.makedirs(os.path.join(prediction_root, f"iteration{i:02}"), exist_ok=True)

    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Run inference with prompts"
    ):
        image_name = os.path.basename(image_path)

        prediction_paths = [os.path.join(prediction_root, f"iteration{i:02}", image_name) for i in range(n_iterations)]
        if all(os.path.exists(prediction_path) for prediction_path in prediction_paths):
            continue

        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path).astype("uint32")
        gt = relabel_sequential(gt)[0]

        with torch.no_grad():
            _run_inference_with_iterative_prompting_for_image(
                model, image, gt, n_iterations, device, use_boxes, prediction_paths, batch_size,
            )
