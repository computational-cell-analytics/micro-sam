import os
from glob import glob

import h5py
import numpy as np
from skimage.segmentation import relabel_sequential

from tqdm import tqdm

from .. import util
from ..prompt_generators import PointAndBoxPromptGenerator
from ..prompt_based_segmentation import segment_from_box, segment_from_points


#
# Compute all required data for the model comparison
#


def predict_models_with_loader(loader, n_samples, prompt_generator, predictor1, predictor2, output_folder):
    i = 0
    os.makedirs(output_folder, exist_ok=True)

    for x, y in tqdm(loader, total=n_samples):
        out_path = os.path.join(output_folder, f"sample_{i}.h5")

        im = x.numpy().squeeze()
        gt = y.numpy().squeeze().astype("uint32")
        gt = relabel_sequential(gt)[0]

        emb1 = util.precompute_image_embeddings(predictor1, im, ndim=2)
        util.set_precomputed(predictor1, emb1)

        emb2 = util.precompute_image_embeddings(predictor2, im, ndim=2)
        util.set_precomputed(predictor2, emb2)

        centers, boxes = util.get_centers_and_bounding_boxes(gt)

        gt_ids = np.unique(gt)[1:]
        for gt_id in tqdm(gt_ids):

            point_coords, point_labels, box, _ = prompt_generator(gt, gt_id, boxes[gt_id], centers[gt_id])

            # TODO multimask output???
            box = np.array(box[0])
            mask1_box = segment_from_box(predictor1, box)
            mask2_box = segment_from_box(predictor2, box)
            mask1_box, mask2_box = mask1_box.squeeze(), mask2_box.squeeze()

            # TODO multimask output if we just have a single point
            point_coords, point_labels = np.array(point_coords), np.array(point_labels)
            mask1_points, ious, _ = segment_from_points(predictor1, point_coords, point_labels, return_all=True)
            mask2_points, ious, _ = segment_from_points(predictor2, point_coords, point_labels, return_all=True)
            mask1_points, mask2_points = mask1_points.squeeze(), mask2_points.squeeze()

            with h5py.File(out_path, "a") as f:
                g = f.create_group(str(gt_id))
                g.attrs["point_coords"] = point_coords
                g.attrs["point_labels"] = point_labels
                g.attrs["box"] = box

                gt_mask = (gt == gt_id).astype("uint8")
                g.create_dataset("image", data=im, compression="gzip")
                g.create_dataset("gt_mask", data=gt_mask, compression="gzip")
                g.create_dataset("box/mask1", data=mask1_box.astype("uint8"), compression="gzip")
                g.create_dataset("box/mask2", data=mask2_box.astype("uint8"), compression="gzip")
                g.create_dataset("points/mask1", data=mask1_points.astype("uint8"), compression="gzip")
                g.create_dataset("points/mask2", data=mask2_points.astype("uint8"), compression="gzip")

        i += 1
        if i >= n_samples:
            return


# TODO expose more params (prompt generation scheme)
def generate_data_for_model_comparison(loader, output_folder, model_type1, model_type2, n_samples):
    prompt_generator = PointAndBoxPromptGenerator(
        n_positive_points=1,
        n_negative_points=0,
        dilation_strength=3,
        get_point_prompts=True,
        get_box_prompts=True,
    )
    predictor1 = util.get_sam_model(model_type=model_type1)
    predictor2 = util.get_sam_model(model_type=model_type2)
    predict_models_with_loader(loader, n_samples, prompt_generator, predictor1, predictor2, output_folder)


#
# Visual evaluation accroding to metrics
#

# TODO


#
# Visual evaluation with napari
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


def model_comparison_with_napari(output_folder, show_points=True):
    files = glob(os.path.join(output_folder, "*.h5"))
    for path in files:
        print("Comparing models in", path)
        with h5py.File(path, "r") as f:
            for g in f.values():
                _check_group(g, show_points=show_points)
