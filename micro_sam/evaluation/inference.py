"""Inference with Segment Anything models and different prompt strategies.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import imageio.v3 as imageio
from skimage.segmentation import relabel_sequential

import torch

from segment_anything import SamPredictor

from .. import util as util
from ..inference import batched_inference
from ..instance_segmentation import (
    mask_data_to_segmentation, get_predictor_and_decoder,
    AutomaticMaskGenerator, InstanceSegmentationWithDecoder,
)
from . import instance_segmentation
from ..prompt_generators import PointAndBoxPromptGenerator, IterativePromptGenerator


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

    points, point_labels, boxes, _ = prompt_generator(
        masks, bbox_coordinates, center_coordinates
    )

    def to_numpy(x):
        if x is None:
            return x
        return x.numpy()

    return to_numpy(points), to_numpy(point_labels), to_numpy(boxes)


def _run_inference_with_prompts_for_image(
    predictor,
    image,
    gt,
    use_points,
    use_boxes,
    n_positives,
    n_negatives,
    dilation,
    batch_size,
    cached_prompts,
    embedding_path,
):
    gt_ids = np.unique(gt)[1:]
    if cached_prompts is None:
        points, point_labels, boxes = _get_batched_prompts(
            gt, gt_ids, use_points, use_boxes, n_positives, n_negatives, dilation
        )
    else:
        points, point_labels, boxes = cached_prompts

    # Make a copy of the point prompts to return them at the end.
    prompts = deepcopy((points, point_labels, boxes))

    # Use multi-masking only if we have a single positive point without box
    multimasking = False
    if not use_boxes and (n_positives == 1 and n_negatives == 0):
        multimasking = True

    instance_labels = batched_inference(
        predictor, image, batch_size,
        boxes=boxes, points=points, point_labels=point_labels,
        multimasking=multimasking, embedding_path=embedding_path,
        return_instance_segmentation=True,
    )

    return instance_labels, prompts


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


def _precompute_prompts(gt_path, use_points, use_boxes, n_positives, n_negatives, dilation):
    name = os.path.basename(gt_path)

    gt = imageio.imread(gt_path).astype("uint32")
    gt = relabel_sequential(gt)[0]
    gt_ids = np.unique(gt)[1:]

    input_point, input_label, input_box = _get_batched_prompts(
        gt, gt_ids, use_points, use_boxes, n_positives, n_negatives, dilation
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
    """Run segment anything inference for multiple images using prompts derived from groundtruth.

    Args:
        predictor: The SegmentAnything predictor.
        image_paths: The image file paths.
        gt_paths: The ground-truth segmentation file paths.
        embedding_dir: The directory where the image embddings will be saved or are already saved.
        use_points: Whether to use point prompts.
        use_boxes: Whether to use box prompts
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
        this_prompts, cached_point_prompts, cached_box_prompts = _load_prompts(
            cached_point_prompts, save_point_prompts,
            cached_box_prompts, save_box_prompts,
            label_name
        )
        instances, this_prompts = _run_inference_with_prompts_for_image(
            predictor, im, gt, n_positives=n_positives, n_negatives=n_negatives,
            dilation=dilation, use_points=use_points, use_boxes=use_boxes,
            batch_size=batch_size, cached_prompts=this_prompts,
            embedding_path=embedding_path,
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
    masks = masks.cpu().numpy().squeeze(1).astype("bool")
    masks = [{"segmentation": mask, "area": mask.sum()} for mask in masks]
    segmentation = mask_data_to_segmentation(masks, with_background=True)
    imageio.imwrite(prediction_path, segmentation, compression=5)


def _get_batched_iterative_prompts(sampled_binary_gt, masks, batch_size, prompt_generator):
    n_samples = sampled_binary_gt.shape[0]
    n_batches = int(np.ceil(float(n_samples) / batch_size))
    next_coords, next_labels = [], []
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_stop = min((batch_idx + 1) * batch_size, n_samples)

        batch_coords, batch_labels, _, _ = prompt_generator(
            sampled_binary_gt[batch_start: batch_stop], masks[batch_start: batch_stop]
        )
        next_coords.append(batch_coords)
        next_labels.append(batch_labels)

    next_coords = torch.concatenate(next_coords)
    next_labels = torch.concatenate(next_labels)

    return next_coords, next_labels


@torch.no_grad()
def _run_inference_with_iterative_prompting_for_image(
    predictor,
    image,
    gt,
    start_with_box_prompt,
    dilation,
    batch_size,
    embedding_path,
    n_iterations,
    prediction_paths,
    use_masks=False,
    verbose=True,
) -> None:
    prompt_generator = IterativePromptGenerator()

    gt_ids = np.unique(gt)[1:]

    # Use multi-masking only if we have a single positive point without box
    if start_with_box_prompt:
        use_boxes, use_points = True, False
        n_positives = 0
        multimasking = False
    else:
        use_boxes, use_points = False, True
        n_positives = 1
        multimasking = True

    points, point_labels, boxes = _get_batched_prompts(
        gt, gt_ids,
        use_points=use_points,
        use_boxes=use_boxes,
        n_positives=n_positives,
        n_negatives=0,
        dilation=dilation
    )

    sampled_binary_gt = util.segmentation_to_one_hot(gt.astype("int64"), gt_ids)

    for iteration in range(n_iterations):
        if iteration == 0:  # logits mask can not be used for the first iteration.
            logits_masks = None
        else:
            if not use_masks:  # logits mask should not be used when not desired.
                logits_masks = None

        batched_outputs = batched_inference(
            predictor=predictor,
            image=image,
            batch_size=batch_size,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            multimasking=multimasking,
            embedding_path=embedding_path,
            return_instance_segmentation=False,
            logits_masks=logits_masks,
            verbose=verbose,
        )

        # switching off multimasking after first iter, as next iters (with multiple prompts) don't expect multimasking
        multimasking = False

        masks = torch.stack([m["segmentation"][None] for m in batched_outputs]).to(torch.float32)

        next_coords, next_labels = _get_batched_iterative_prompts(
            sampled_binary_gt, masks, batch_size, prompt_generator
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

        if use_masks:
            logits_masks = torch.stack([m["logits"] for m in batched_outputs])

        _save_segmentation(masks, prediction_paths[iteration])


def run_inference_with_iterative_prompting(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    gt_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    prediction_dir: Union[str, os.PathLike],
    start_with_box_prompt: bool,
    dilation: int = 5,
    batch_size: int = 32,
    n_iterations: int = 8,
    use_masks: bool = False,
    verbose=True,
) -> None:
    """Run segment anything inference for multiple images using prompts iteratively
        derived from model outputs and groundtruth

    Args:
        predictor: The SegmentAnything predictor.
        image_paths: The image file paths.
        gt_paths: The ground-truth segmentation file paths.
        embedding_dir: The directory where the image embeddings will be saved or are already saved.
        prediction_dir: The directory where the predictions from SegmentAnything will be saved per iteration.
        start_with_box_prompt: Whether to use the first prompt as bounding box or a single point
        dilation: The dilation factor for the radius around the ground-truth object
            around which points will not be sampled.
        batch_size: The batch size used for batched predictions.
        n_iterations: The number of iterations for iterative prompting.
        use_masks: Whether to make use of logits from previous prompt-based segmentation.
        verbose: Whether to show the outputs of the progress bar.
    """
    if len(image_paths) != len(gt_paths):
        raise ValueError(f"Expect same number of images and gt images, got {len(image_paths)}, {len(gt_paths)}")

    # create all prediction folders for all intermediate iterations
    for i in range(n_iterations):
        os.makedirs(os.path.join(prediction_dir, f"iteration{i:02}"), exist_ok=True)

    if use_masks:
        print("The iterative prompting will make use of logits masks from previous iterations.")

    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths),
        total=len(image_paths),
        desc="Run inference with iterative prompting for all images",
        disable=not verbose,
    ):
        image_name = os.path.basename(image_path)

        # We skip the images that already have been segmented
        prediction_paths = [os.path.join(prediction_dir, f"iteration{i:02}", image_name) for i in range(n_iterations)]
        if all(os.path.exists(prediction_path) for prediction_path in prediction_paths):
            continue

        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path).astype("uint32")
        gt = relabel_sequential(gt)[0]

        embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")

        _run_inference_with_iterative_prompting_for_image(
            predictor=predictor,
            image=image,
            gt=gt,
            start_with_box_prompt=start_with_box_prompt,
            dilation=dilation,
            batch_size=batch_size,
            embedding_path=embedding_path,
            n_iterations=n_iterations,
            prediction_paths=prediction_paths,
            use_masks=use_masks,
            verbose=verbose,
        )


#
# AMG FUNCTION
#


def run_amg(
    checkpoint: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    val_image_paths: List[Union[str, os.PathLike]],
    val_gt_paths: List[Union[str, os.PathLike]],
    test_image_paths: List[Union[str, os.PathLike]],
    iou_thresh_values: Optional[List[float]] = None,
    stability_score_values: Optional[List[float]] = None,
    verbose: bool = True,
) -> str:
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    predictor = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint)
    amg = AutomaticMaskGenerator(predictor)
    amg_prefix = "amg"

    # where the predictions are saved
    prediction_folder = os.path.join(experiment_folder, amg_prefix, "inference")
    os.makedirs(prediction_folder, exist_ok=True)

    # where the grid-search results are saved
    gs_result_folder = os.path.join(experiment_folder, amg_prefix, "grid_search")
    os.makedirs(gs_result_folder, exist_ok=True)

    grid_search_values = instance_segmentation.default_grid_search_values_amg(
        iou_thresh_values=iou_thresh_values,
        stability_score_values=stability_score_values,
    )

    instance_segmentation.run_instance_segmentation_grid_search_and_inference(
        segmenter=amg,
        grid_search_values=grid_search_values,
        val_image_paths=val_image_paths,
        val_gt_paths=val_gt_paths,
        test_image_paths=test_image_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_folder,
        result_dir=gs_result_folder,
        verbose_gs=verbose,
    )
    return prediction_folder


#
# INSTANCE SEGMENTATION FUNCTION
#


def run_instance_segmentation_with_decoder(
    checkpoint: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    val_image_paths: List[Union[str, os.PathLike]],
    val_gt_paths: List[Union[str, os.PathLike]],
    test_image_paths: List[Union[str, os.PathLike]],
) -> str:
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    predictor, decoder = get_predictor_and_decoder(model_type=model_type, checkpoint_path=checkpoint)
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)
    seg_prefix = "instance_segmentation_with_decoder"

    # where the predictions are saved
    prediction_folder = os.path.join(experiment_folder, seg_prefix, "inference")
    os.makedirs(prediction_folder, exist_ok=True)

    # where the grid-search results are saved
    gs_result_folder = os.path.join(experiment_folder, seg_prefix, "grid_search")
    os.makedirs(gs_result_folder, exist_ok=True)

    grid_search_values = instance_segmentation.default_grid_search_values_instance_segmentation_with_decoder()

    instance_segmentation.run_instance_segmentation_grid_search_and_inference(
        segmenter, grid_search_values,
        val_image_paths, val_gt_paths, test_image_paths,
        embedding_dir=embedding_folder, prediction_dir=prediction_folder,
        result_dir=gs_result_folder,
    )
    return prediction_folder
