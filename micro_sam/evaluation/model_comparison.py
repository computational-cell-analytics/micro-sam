"""Functionality for qualitative comparison of Segment Anything models on microscopy data.
"""

import os
from functools import partial
from glob import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import skimage.draw as draw
from scipy.ndimage import binary_dilation
from skimage import exposure
from skimage.segmentation import relabel_sequential, find_boundaries

from tqdm import tqdm
from typing import Optional, Union

from .. import util
from ..prompt_generators import PointAndBoxPromptGenerator
from ..prompt_based_segmentation import segment_from_box, segment_from_points


#
# Compute all required data for the model comparison
#


def _predict_models_with_loader(loader, n_samples, prompt_generator, predictor1, predictor2, predictor3, output_folder):
    i = 0
    os.makedirs(output_folder, exist_ok=True)

    for x, y in tqdm(loader, total=n_samples):
        out_path = os.path.join(output_folder, f"sample_{i}.h5")

        im = x.numpy().squeeze()
        if im.ndim == 3 and im.shape[0] == 3:
            im = im.transpose((1, 2, 0))

        gt = y.numpy().squeeze().astype("uint32")
        gt = relabel_sequential(gt)[0]

        emb1 = util.precompute_image_embeddings(predictor1, im, ndim=2)
        util.set_precomputed(predictor1, emb1)

        emb2 = util.precompute_image_embeddings(predictor2, im, ndim=2)
        util.set_precomputed(predictor2, emb2)

        if predictor3 is not None:
            emb3 = util.precompute_image_embeddings(predictor3, im, ndim=2)
            util.set_precomputed(predictor3, emb3)

        with h5py.File(out_path, "a") as f:
            f.create_dataset("image", data=im, compression="gzip")

        gt_ids = np.unique(gt)[1:]
        centers, boxes = util.get_centers_and_bounding_boxes(gt)
        centers = [centers[gt_id] for gt_id in gt_ids]
        boxes = [boxes[gt_id] for gt_id in gt_ids]

        object_masks = util.segmentation_to_one_hot(gt.astype("int64"), gt_ids)
        coords, labels, boxes, _ = prompt_generator(
            segmentation=object_masks,
            bbox_coordinates=boxes,
            center_coordinates=centers,
        )

        for idx, gt_id in tqdm(enumerate(gt_ids), total=len(gt_ids)):

            # Box prompts:
            # Reorder the coordinates so that they match the normal python convention.
            box = boxes[idx][[1, 0, 3, 2]]
            mask1_box = segment_from_box(predictor1, box)
            mask2_box = segment_from_box(predictor2, box)
            mask1_box, mask2_box = mask1_box.squeeze(), mask2_box.squeeze()

            if predictor3 is not None:
                mask3_box = segment_from_box(predictor3, box)
                mask3_box = mask3_box.squeeze()

            # Point prompts:
            # Reorder the coordinates so that they match the normal python convention.
            point_coords, point_labels = np.array(coords[idx])[:, ::-1], np.array(labels[idx])
            mask1_points = segment_from_points(predictor1, point_coords, point_labels)
            mask2_points = segment_from_points(predictor2, point_coords, point_labels)
            mask1_points, mask2_points = mask1_points.squeeze(), mask2_points.squeeze()

            if predictor3 is not None:
                mask3_points = segment_from_points(predictor3, point_coords, point_labels)
                mask3_points = mask3_points.squeeze()

            gt_mask = gt == gt_id
            with h5py.File(out_path, "a") as f:
                g = f.create_group(str(gt_id))
                g.attrs["point_coords"] = point_coords
                g.attrs["point_labels"] = point_labels
                g.attrs["box"] = box

                g.create_dataset("gt_mask", data=gt_mask, compression="gzip")
                g.create_dataset("box/mask1", data=mask1_box.astype("uint8"), compression="gzip")
                g.create_dataset("box/mask2", data=mask2_box.astype("uint8"), compression="gzip")
                g.create_dataset("points/mask1", data=mask1_points.astype("uint8"), compression="gzip")
                g.create_dataset("points/mask2", data=mask2_points.astype("uint8"), compression="gzip")

                if predictor3 is not None:
                    g.create_dataset("box/mask3", data=mask3_box.astype("uint8"), compression="gzip")
                    g.create_dataset("points/mask3", data=mask3_points.astype("uint8"), compression="gzip")

        i += 1
        if i >= n_samples:
            return


def generate_data_for_model_comparison(
    loader: torch.utils.data.DataLoader,
    output_folder: Union[str, os.PathLike],
    model_type1: str,
    model_type2: str,
    n_samples: int,
    model_type3: Optional[str] = None,
    checkpoint1: Optional[Union[str, os.PathLike]] = None,
    checkpoint2: Optional[Union[str, os.PathLike]] = None,
    checkpoint3: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Generate samples for qualitative model comparison.

    This precomputes the input for `model_comparison` and `model_comparison_with_napari`.

    Args:
        loader: The torch dataloader from which samples are drawn.
        output_folder: The folder where the samples will be saved.
        model_type1: The first model to use for comparison.
            The value needs to be a valid model_type for `micro_sam.util.get_sam_model`.
        model_type2: The second model to use for comparison.
            The value needs to be a valid model_type for `micro_sam.util.get_sam_model`.
        n_samples: The number of samples to draw from the dataloader.
        checkpoint1: Optional checkpoint for the first model.
        checkpoint2: Optional checkpoint for the second model.
    """
    prompt_generator = PointAndBoxPromptGenerator(
        n_positive_points=1,
        n_negative_points=0,
        dilation_strength=3,
        get_point_prompts=True,
        get_box_prompts=True,
    )
    predictor1 = util.get_sam_model(model_type=model_type1, checkpoint_path=checkpoint1)
    predictor2 = util.get_sam_model(model_type=model_type2, checkpoint_path=checkpoint2)

    if model_type3 is not None:
        predictor3 = util.get_sam_model(model_type=model_type3, checkpoint_path=checkpoint3)
    else:
        predictor3 = None

    _predict_models_with_loader(loader, n_samples, prompt_generator, predictor1, predictor2, predictor3, output_folder)


#
# Visual evaluation according to metrics
#


def _evaluate_samples(f, prefix, min_size):
    eval_result = {
        "gt_id": [],
        "score1": [],
        "score2": [],
    }
    for name, group in f.items():
        if name == "image":
            continue

        gt_mask = group["gt_mask"][:]

        size = gt_mask.sum()
        if size < min_size:
            continue

        m1 = group[f"{prefix}/mask1"][:]
        m2 = group[f"{prefix}/mask2"][:]

        score1 = util.compute_iou(gt_mask, m1)
        score2 = util.compute_iou(gt_mask, m2)

        eval_result["gt_id"].append(name)
        eval_result["score1"].append(score1)
        eval_result["score2"].append(score2)

    eval_result = pd.DataFrame.from_dict(eval_result)
    eval_result["advantage1"] = eval_result["score1"] - eval_result["score2"]
    eval_result["advantage2"] = eval_result["score2"] - eval_result["score1"]
    return eval_result


def _overlay_mask(image, mask, alpha=0.6):
    assert image.ndim in (2, 3)
    # overlay the mask
    if image.ndim == 2:
        overlay = np.stack([image, image, image]).transpose((1, 2, 0))
    else:
        overlay = image
    assert overlay.shape[-1] == 3
    mask_overlay = np.zeros_like(overlay)
    mask_overlay[mask == 1] = [255, 0, 0]
    alpha = alpha
    overlay = alpha * overlay + (1.0 - alpha) * mask_overlay
    return overlay.astype("uint8")


def _enhance_image(im, do_norm=True):
    # apply CLAHE to improve the image quality
    if do_norm:
        im -= im.min(axis=(0, 1), keepdims=True)
        im /= (im.max(axis=(0, 1), keepdims=True) + 1e-6)
    im = exposure.equalize_adapthist(im)
    im *= 255
    return im


def _overlay_outline(im, mask, outline_dilation):
    outline = find_boundaries(mask)
    if outline_dilation > 0:
        outline = binary_dilation(outline, iterations=outline_dilation)
    overlay = im.copy()
    overlay[outline] = [255, 255, 0]
    return overlay


def _overlay_box(im, prompt, outline_dilation):
    start, end = prompt
    rr, cc = draw.rectangle_perimeter(start, end=end, shape=im.shape[:2])

    box_outline = np.zeros(im.shape[:2], dtype="bool")
    box_outline[rr, cc] = 1
    if outline_dilation > 0:
        box_outline = binary_dilation(box_outline, iterations=outline_dilation)

    overlay = im.copy()
    overlay[box_outline] = [0, 255, 255]

    return overlay


# NOTE: we currently only support a single point
def _overlay_points(im, prompt, radius):
    coords, labels = prompt
    # make sure we have a single positive prompt, other options are
    # currently not supported
    assert coords.shape[0] == labels.shape[0] == 1
    assert labels[0] == 1

    rr, cc = draw.disk(coords[0], radius, shape=im.shape[:2])
    overlay = im.copy()
    draw.set_color(overlay, (rr, cc), [0, 255, 255], alpha=1.0)

    return overlay


def _compare_eval(
    f, eval_result, advantage_column, n_images_per_sample, prefix,
    sample_name, plot_folder, point_radius, outline_dilation, have_model3,
):
    result = eval_result.sort_values(advantage_column, ascending=False).iloc[:n_images_per_sample]
    n_rows = result.shape[0]

    image = f["image"][:]
    is_box_prompt = prefix == "box"
    overlay_prompts = partial(_overlay_box, outline_dilation=outline_dilation) if is_box_prompt else\
        partial(_overlay_points, radius=point_radius)

    def make_square(bb, shape):
        box_shape = [b.stop - b.start for b in bb]
        longest_side = max(box_shape)
        padding = [(longest_side - sh) // 2 for sh in box_shape]
        bb = tuple(
            slice(max(b.start - pad, 0), min(b.stop + pad, sh)) for b, pad, sh in zip(bb, padding, shape)
        )
        return bb

    def plot_ax(axis, i, row):
        g = f[row.gt_id]

        gt = g["gt_mask"][:]
        mask1 = g[f"{prefix}/mask1"][:]
        mask2 = g[f"{prefix}/mask2"][:]

        # The mask3 is just for comparison purpose, we just plot the crops as it is.
        if have_model3:
            mask3 = g[f"{prefix}/mask3"][:]

        fg_mask = (gt + mask1 + mask2) > 0
        # if this is a box prompt we dilate the mask so that the bounding box
        # can be seen
        if is_box_prompt:
            fg_mask = binary_dilation(fg_mask, iterations=5)
        bb = np.where(fg_mask)
        bb = tuple(
            slice(int(b.min()), int(b.max() + 1)) for b in bb
        )
        bb = make_square(bb, fg_mask.shape)

        offset = np.array([b.start for b in bb])
        if is_box_prompt:
            prompt = g.attrs["box"]
            prompt = np.array(
                [prompt[:2], prompt[2:]]
            ) - offset
        else:
            prompt = (g.attrs["point_coords"] - offset, g.attrs["point_labels"])

        im = _enhance_image(image[bb])
        gt, mask1, mask2 = gt[bb], mask1[bb], mask2[bb]

        if have_model3:
            mask3 = mask3[bb]

        im1 = _overlay_mask(im, mask1)
        im1 = _overlay_outline(im1, gt, outline_dilation)
        im1 = overlay_prompts(im1, prompt)
        ax = axis[0] if i is None else axis[i, 0]
        ax.axis("off")
        ax.imshow(im1)

        # We put the third set of comparsion point in between
        # so that the comparison looks -> default, generalist, specialist
        if have_model3:
            im3 = _overlay_mask(im, mask3)
            im3 = _overlay_outline(im3, gt, outline_dilation)
            im3 = overlay_prompts(im3, prompt)
            ax = axis[1] if i is None else axis[i, 1]
            ax.axis("off")
            ax.imshow(im3)

            nexax = 2
        else:
            nexax = 1

        im2 = _overlay_mask(im, mask2)
        im2 = _overlay_outline(im2, gt, outline_dilation)
        im2 = overlay_prompts(im2, prompt)
        ax = axis[nexax] if i is None else axis[i, nexax]
        ax.axis("off")
        ax.imshow(im2)

    cols = 3 if have_model3 else 2
    if plot_folder is None:
        fig, axis = plt.subplots(n_rows, cols)
        for i, (_, row) in enumerate(result.iterrows()):
            plot_ax(axis, i, row)
        plt.show()
    else:
        for i, (_, row) in enumerate(result.iterrows()):
            fig, axis = plt.subplots(1, cols)
            plot_ax(axis, None, row)
            plt.subplots_adjust(wspace=0.05, hspace=0)
            plt.savefig(os.path.join(plot_folder, f"{sample_name}_{i}.svg"), bbox_inches="tight")
            plt.close()


def _compare_prompts(
    f, prefix, n_images_per_sample, min_size, sample_name, plot_folder,
    point_radius, outline_dilation, have_model3,
):
    box_eval = _evaluate_samples(f, prefix, min_size)
    if plot_folder is None:
        plot_folder1, plot_folder2 = None, None
    else:
        plot_folder1 = os.path.join(plot_folder, "advantage1")
        plot_folder2 = os.path.join(plot_folder, "advantage2")
        os.makedirs(plot_folder1, exist_ok=True)
        os.makedirs(plot_folder2, exist_ok=True)
    _compare_eval(
        f, box_eval, "advantage1", n_images_per_sample, prefix, sample_name, plot_folder1,
        point_radius, outline_dilation, have_model3,
    )
    _compare_eval(
        f, box_eval, "advantage2", n_images_per_sample, prefix, sample_name, plot_folder2,
        point_radius, outline_dilation, have_model3,
    )


def _compare_models(
    path, n_images_per_sample, min_size, plot_folder, point_radius, outline_dilation, have_model3,
):
    sample_name = Path(path).stem
    with h5py.File(path, "r") as f:
        if plot_folder is None:
            plot_folder_points, plot_folder_box = None, None
        else:
            plot_folder_points = os.path.join(plot_folder, "points")
            plot_folder_box = os.path.join(plot_folder, "box")
        _compare_prompts(
            f, "points", n_images_per_sample, min_size, sample_name, plot_folder_points,
            point_radius, outline_dilation, have_model3,
        )
        _compare_prompts(
            f, "box", n_images_per_sample, min_size, sample_name, plot_folder_box,
            point_radius, outline_dilation, have_model3,
        )


def model_comparison(
    output_folder: Union[str, os.PathLike],
    n_images_per_sample: int,
    min_size: int,
    plot_folder: Optional[Union[str, os.PathLike]] = None,
    point_radius: int = 4,
    outline_dilation: int = 0,
    have_model3=False,
) -> None:
    """Create images for a qualitative model comparision.

    Args:
        output_folder: The folder with the data precomputed by `generate_data_for_model_comparison`.
        n_images_per_sample: The number of images to generate per precomputed sample.
        min_size: The min size of ground-truth objects to take into account.
        plot_folder: The folder where to save the plots. If not given the plots will be displayed.
        point_radius: The radius of the point overlay.
        outline_dilation: The dilation factor of the outline overlay.
    """
    files = glob(os.path.join(output_folder, "*.h5"))
    for path in tqdm(files):
        _compare_models(
            path, n_images_per_sample, min_size, plot_folder, point_radius, outline_dilation, have_model3,
        )


#
# Quick visual evaluation with napari
#


def _check_group(g, show_points):
    import napari

    image = g["image"][:]
    gt = g["gt_mask"][:]
    if show_points:
        m1 = g["points/mask1"][:]
        m2 = g["points/mask2"][:]
        points = g.attrs["point_coords"]
    else:
        m1 = g["box/mask1"][:]
        m2 = g["box/mask2"][:]
        box = g.attrs["box"]
        box = np.array([
            [box[0], box[1]], [box[2], box[3]]
        ])

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(gt)
    v.add_labels(m1)
    v.add_labels(m2)
    if show_points:
        # TODO use point labels for coloring
        v.add_points(
            points,
            edge_color="#00FF00",
            symbol="o",
            face_color="transparent",
            edge_width=0.5,
            size=12,
        )
    else:
        v.add_shapes(
            box, face_color="transparent", edge_color="green", edge_width=4,
        )
    napari.run()


def model_comparison_with_napari(output_folder: Union[str, os.PathLike], show_points: bool = True) -> None:
    """Use napari to display the qualtiative comparison results for two models.

    Args:
        output_folder: The folder with the data precomputed by `generate_data_for_model_comparison`.
        show_points: Whether to show the results for point or for box prompts.
    """
    files = glob(os.path.join(output_folder, "*.h5"))
    for path in files:
        print("Comparing models in", path)
        with h5py.File(path, "r") as f:
            for name, g in f.items():
                if name == "image":
                    continue
                _check_group(g, show_points=show_points)
