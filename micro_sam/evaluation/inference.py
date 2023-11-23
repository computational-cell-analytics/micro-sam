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
from ..instance_segmentation import mask_data_to_segmentation
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


def get_predictor(
    checkpoint_path: Union[str, os.PathLike],
    model_type: str,
    device: Optional[Union[str, torch.device]] = None,
    return_state: bool = False,
    is_custom_model: Optional[bool] = None,
) -> SamPredictor:
    """Get the segment anything predictor from an exported or custom checkpoint.

    Args:
        checkpoint_path: The checkpoint filepath.
        model_type: The type of the model, either vit_h, vit_b or vit_l.
        device: The device to use.
        return_state: Whether to return the complete state of the checkpoint in addtion to the predictor.
        is_custom_model: Whether this is a custom model or not.
    Returns:
        The segment anything predictor.
    """
    device = util.get_device(device)

    # By default we check if the model follows the torch_em checkpint naming scheme to check whether it is a
    # custom model or not. This can be over-ridden by passing True or False for is_custom_model.
    is_custom_model = checkpoint_path.split("/")[-1] == "best.pt" if is_custom_model is None else is_custom_model

    if is_custom_model:  # Finetuned SAM model
        predictor = util.get_custom_sam_model(
            checkpoint_path=checkpoint_path, model_type=model_type, device=device, return_state=return_state
        )
    else:  # Vanilla SAM model
        assert not return_state
        predictor = util.get_sam_model(
            model_type=model_type, device=device, checkpoint_path=checkpoint_path
        )  # type: ignore
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
    masks = masks.cpu().numpy().squeeze().astype("bool")
    shape = masks.shape[-2:]
    masks = [{"segmentation": mask, "area": mask.sum()} for mask in masks]
    segmentation = mask_data_to_segmentation(masks, shape, with_background=True)
    imageio.imwrite(prediction_path, segmentation, compression=5)


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
    prediction_paths
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
        batched_outputs = batched_inference(
            predictor, image, batch_size,
            boxes=boxes, points=points, point_labels=point_labels,
            multimasking=multimasking, embedding_path=embedding_path,
            return_instance_segmentation=False
        )

        # switching off multimasking after first iter, as next iters (with multiple prompts) don't expect multimasking
        multimasking = False

        masks = torch.stack([m["segmentation"][None] for m in batched_outputs]).to(torch.float32)

        next_coords, next_labels, _, _ = prompt_generator(sampled_binary_gt, masks)
        next_coords, next_labels = next_coords.detach().cpu().numpy(), next_labels.detach().cpu().numpy()

        if points is not None:
            points = np.concatenate([points, next_coords], axis=1)
        else:
            points = next_coords

        if point_labels is not None:
            point_labels = np.concatenate([point_labels, next_labels], axis=1)
        else:
            point_labels = next_labels

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
        dilation: The dilation factor for the radius around the ground-truth obkect
            around which points will not be sampled.
        batch_size: The batch size used for batched predictions.
        n_iterations: The number of iterations for iterative prompting.
    """
    if len(image_paths) != len(gt_paths):
        raise ValueError(f"Expect same number of images and gt images, got {len(image_paths)}, {len(gt_paths)}")

    # create all prediction folders for all intermediate iterations
    for i in range(n_iterations):
        os.makedirs(os.path.join(prediction_dir, f"iteration{i:02}"), exist_ok=True)

    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Run inference with iterative prompting for all images"
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
            predictor, image, gt, start_with_box_prompt=start_with_box_prompt,
            dilation=dilation, batch_size=batch_size, embedding_path=embedding_path,
            n_iterations=n_iterations, prediction_paths=prediction_paths
        )
