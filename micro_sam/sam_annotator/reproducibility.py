import os
import warnings
from typing import Optional, Union

import numpy as np
from elf.io import open_file
from tqdm import tqdm

from .. import util
from ..automatic_segmentation import get_predictor_and_segmenter
from ..instance_segmentation import mask_data_to_segmentation

from ._widgets import _mask_matched_objects
from .util import prompt_segmentation


def _load_model_from_commit_file(f):
    # Check which segmentation mode is used, by going through the commit history
    # and checking if we have committed the 'auto_segmentation' layer.
    # If we did, then we derive which mode was used from the serialized parameters.
    amg = None
    commit_history = f.attrs["commit_history"]
    for commit in commit_history:
        layer, options = next(iter(commit.items()))
        if layer == "auto_segmentation":
            amg = not ("boundary_distance_threshold" in options)

    # Get the predictor and segmenter.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=f.attrs["model_name"],
        # checkpoint="",  TODO we need to also serialize this
        amg=amg,
        is_tiled=f.attrs["tile_shape"] is not None,
    )
    return predictor, segmenter


def _check_data_hash(f, input_data, input_path):
    data_hash = util._compute_data_signature(input_data)
    expected_hash = f.attrs["data_signature"]
    if data_hash != expected_hash:
        raise RuntimeError(
            f"The hash of the input data loaded from {input_path} is {data_hash}, "
            f"which does not match the expected hash {expected_hash}."
        )


def _write_masks_with_preservation(prev_seg, seg, preserve_mode, preservation_threshold, object_ids):
    # Make sure the segmented ids and the object ids match (warn if not),
    # and apply the id offset to match them.
    segmented_ids = np.setdiff1d(np.unique(seg), [0])
    if len(segmented_ids) != len(object_ids):
        warnings.warn(
            f"The number of objects found by running auto segmentation (={len(segmented_ids)})"
            f" does not match the number of expected objects (={len(object_ids)})."
        )
    id_offset = int(np.min(object_ids)) - 1
    seg[seg != 0] += id_offset

    # Write the new segmentation to the previous segmentation, taking the preservation rules into account.
    mask = seg != 0
    if preserve_mode != "none":
        preserve_mask = prev_seg != 0
        if preserve_mask.sum() != 0:
            # In the mode 'objects' we preserve committed objects instead, by comparing the overlaps
            # of already committed and newly committed objects.
            if preserve_mode == "objects":
                preserve_mask = _mask_matched_objects(seg, prev_seg, preservation_threshold)
            mask[preserve_mask] = 0
    prev_seg[mask] = seg[mask]
    return prev_seg


def _rerun_interactive_segmentation(segmentation, f, predictor, image_embeddings, annotator_class, options):
    object_ids = options.pop("object_ids")
    preserve_mode, preservation_threshold = options.pop("preserve_mode"), options.pop("preservation_threshold")
    g = f["prompts"]

    if annotator_class == "Annotator2d":
        # Load the serialized prompts for all objects in this commit.
        boxes, masks = [], []
        points, labels = [], []
        for object_id in object_ids:
            prompt_group = g[str(object_id)]
            if "point_prompts" in prompt_group:
                points.append(prompt_group["point_prompts"][:])
                labels.append(prompt_group["point_labels"][:])
            if "prompts" in prompt_group:
                boxes.append(prompt_group["prompts"][:])
            if "mask" in prompt_group:
                masks.append(prompt_group["mask"][:])

        # TODO
        if not masks:
            masks = None
        if points:
            points, labels = np.array(points), np.array(labels)
        # else:
        #     points, labels = None, None

        batched = len(object_ids) > 1
        seg = prompt_segmentation(
            predictor, points, labels, boxes, masks, segmentation.shape, image_embeddings=image_embeddings,
            multiple_box_prompts=True, batched=batched, previous_segmentation=segmentation,
        )

    # TODO implement batched segmentation for these cases.
    elif annotator_class == "AnnotatorTracking":
        pass
    elif annotator_class == "Annotator3d":
        pass
    else:
        raise RuntimeError(f"Invalid annotator class {annotator_class}.")

    return _write_masks_with_preservation(segmentation, seg, preserve_mode, preservation_threshold, object_ids)


def _rerun_automatic_segmentation(
    image, segmentation, predictor, segmenter, image_embeddings, annotator_class, options
):
    object_ids = options.pop("object_ids")
    preserve_mode, preservation_threshold = options.pop("preserve_mode"), options.pop("preservation_threshold")
    with_background, min_object_size = options.pop("with_background"), options.pop("min_object_size")

    # If there was nothing committed then we don't need to rerun the automatic segmentation.
    if len(object_ids) == 0:
        return segmentation

    if annotator_class == "Annotator2d":
        segmenter.initialize(image=image, image_embeddings=image_embeddings)
        seg = segmenter.generate(**options)
        seg = mask_data_to_segmentation(seg, with_background=with_background, min_object_size=min_object_size)
    # TODO implement auto segmentation for these cases.
    elif annotator_class == "AnnotatorTracking":
        pass
    elif annotator_class == "Annotator3d":
        pass
    else:
        raise RuntimeError(f"Invalid annotator class {annotator_class}.")

    return _write_masks_with_preservation(segmentation, seg, preserve_mode, preservation_threshold, object_ids)


def rerun_segmentation_from_commit_file(
    commit_file: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike, np.ndarray],
    input_key: Optional[str] = None,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
) -> np.ndarray:
    """

    Args:
        commit_file:
        input_path:
        input_key:
        embedding_path:
    """
    # Load the image data and open the zarr commit file.
    input_data = util.load_image_data(input_path, key=input_key)
    with open_file(commit_file, mode="r") as f:

        # Get the annotator class.
        if "annotator_class" not in f.attrs:
            raise RuntimeError(
                f"You have saved the {commit_file} in a version that does not yet support rerunning the segmentation."
            )
        annotator_class = f.attrs["annotator_class"]
        ndim = 2 if annotator_class == "Annotator2d" else 3

        # Check that the stored data hash and the input data hash match.
        _check_data_hash(f, input_data, input_path)

        # Load the model according to the model description stored in the commit file.
        predictor, segmenter = _load_model_from_commit_file(f)

        # Compute oder load the image embeddings.
        image_embeddings = util.precompute_image_embeddings(
            predictor=predictor,
            input_=input_data,
            save_path=embedding_path,
            ndim=ndim,
            tile_shape=f.attrs["tile_shape"],
            halo=f.attrs["halo"],
        )

        # Go through the commit history and redo the action of each commit.
        # Actions can be:
        # - Committing an automatic segmentation result.
        # - Committing an interactive segmentation result.
        commit_history = f.attrs["commit_history"]

        # Rerun the commit history.
        # TODO check if this works correctly for 3d data
        shape = image_embeddings["original_size"]
        segmentation = np.zeros(shape, dtype="uint32")
        for commit in tqdm(commit_history, desc="Rerunning commit history"):
            layer, options = next(iter(commit.items()))
            if layer == "current_object":
                segmentation = _rerun_interactive_segmentation(
                    segmentation, f, predictor, image_embeddings, annotator_class, options
                )
            elif layer == "auto_segmentation":
                segmentation = _rerun_automatic_segmentation(
                    input_data, segmentation, predictor, segmenter, image_embeddings, annotator_class, options
                )
            else:
                raise RuntimeError(f"Invalid layer {layer} in commit_historty.")

    return segmentation


def load_committed_objects_from_commit_file(commit_file: Union[str, os.PathLike]) -> np.ndarray:
    """
    Args:
        commit_file

    Returns:
        AAA
    """
    with open_file(commit_file, mode="r") as f:
        return f["committed_objects"][:]


# TODO
def continue_annotation_from_commit_file(
    commit_file: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike],
    input_key: Optional[str] = None,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """
    Args:
        commit_file:
        input_path:
        input_key:
        embedding_path:
    """
    committed_objects = load_committed_objects_from_commit_file(commit_file)
    with open_file(commit_file, mode="r") as f:
        if "annotator_class" not in f.attrs:
            raise RuntimeError(
                f"You have saved the {commit_file} in a version that does not yet support rerunning the segmentation."
            )
        annotator_class = f.attrs["annotator_class"]


# TODO CLI for 'continue_annotation_from_commit_file'
def main():
    pass
