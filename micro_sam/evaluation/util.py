import os
import pickle
from glob import glob

import imageio.v2 as imageio
import numpy as np
import torch

from skimage.segmentation import relabel_sequential
from tqdm import tqdm
from segment_anything.utils.transforms import ResizeLongestSide

from .. import util as sam_util
from ..prompt_generators import PointAndBoxPromptGenerator


def get_predictor_for_amg(ckpt, model_type):
    """ Initializes the SAM predictor based on finetuned / vanilla checkpoints
    """
    if ckpt.split("/")[-1] == "best.pt":  # Finetuned SAM model
        predictor = sam_util.custom_sam_model(checkpoint_path=ckpt, model_type=model_type)
    else:  # Vanilla SAM model
        predictor = sam_util.get_sam_model(model_type=model_type, checkpoint_path=ckpt)  # type: ignore
    return predictor


def get_prompted_segmentations_sam(predictor, img_dir, gt_dir, root_embedding_dir, pred_dir, n_positive, n_negative,
                                   dilation, get_points=False, get_boxes=False, batch_size=512, _name=None):
    """ Function to get prompted segmentations from SAM
    """
    _n = _name.split("_")[0]
    point_save_path = f"{_n}-points-p{n_positive}-n{n_negative}.pkl"
    if n_positive > 1 and n_negative > 0:
        if os.path.exists(point_save_path):
            print("Using point prompts saved previously")

    f_save_points = []

    for ctype in ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]:
        for i, img_path in enumerate(tqdm(glob(os.path.join(img_dir, f"{ctype}*")), desc=f"Run inference for {ctype}")):
            img_id = os.path.split(img_path)[-1]

            # We skip the images which already have been segmented
            if os.path.exists(os.path.join(pred_dir, img_id)):
                continue

            gt_path = os.path.join(gt_dir, ctype, img_id)

            im = imageio.imread(img_path)
            gt = imageio.imread(gt_path)
            gt = relabel_sequential(gt)[0]

            embedding_path = os.path.join(root_embedding_dir, f"{img_id[:-4]}.zarr")
            image_embeddings = sam_util.precompute_image_embeddings(predictor, im, embedding_path)
            predictor = sam_util.set_precomputed(predictor, image_embeddings)

            im = np.stack((im,)*3, axis=-1)
            predictor.set_image(im)

            past_inputs = None
            if n_positive > 1 and n_negative > 0:
                if os.path.exists(point_save_path):
                    with open(point_save_path, 'rb') as file:
                        myvar = pickle.load(file)
                    past_inputs = myvar[i][img_id]

            instances, save_points = sam_predictor(img_id, gt, predictor, n_positive=n_positive, n_negative=n_negative,
                                                   dilation=dilation, get_points=get_points, get_boxes=get_boxes,
                                                   batch_size=batch_size, past_inputs=past_inputs)
            f_save_points.append(save_points)
            imageio.imsave(os.path.join(pred_dir, img_id), instances)

    if n_positive > 1 and n_negative > 0:
        if not os.path.exists(point_save_path):
            print(f"Saving points at {point_save_path}")
            with open(point_save_path, "wb") as file:
                pickle.dump(f_save_points, file)


def sam_predictor(
    img_id, gt, predictor, n_positive=1, n_negative=0, dilation=5, get_points=False, get_boxes=False,
    batch_size=512, past_inputs=None
):
    """ Generates instance segmentation per image from each assigned prompting method
    """
    # returns the set of cell coordinates and respective bboxes for all instances
    center_coordinates, bbox_coordinates = sam_util.get_centers_and_bounding_boxes(gt)

    prompt_generator = PointAndBoxPromptGenerator(n_positive_points=n_positive, n_negative_points=n_negative,
                                                  dilation_strength=dilation, get_point_prompts=get_points,
                                                  get_box_prompts=get_boxes)
    transform_function = ResizeLongestSide(1024)  # from the model
    gt_ids = np.unique(gt)[1:]
    instance_labels, save_points = batched_prompts_per_image(img_id, gt, gt_ids, center_coordinates, bbox_coordinates,
                                                             prompt_generator, get_points, get_boxes, n_positive,
                                                             n_negative, predictor, transform_function,
                                                             batch_size=batch_size, past_inputs=past_inputs)
    return instance_labels, save_points


def batched_prompts_per_image(img_id, gt, gt_ids, center_coordinates, bbox_coordinates, prompt_generator,
                              get_points, get_boxes, n_positive, n_negative, predictor, transform_function,
                              batch_size, past_inputs):
    """Generates the batch-level instance segmentations from the predictor
    """
    input_point, input_label, input_box = [], [], []
    for gt_id in gt_ids:
        centers, bboxes = center_coordinates.get(gt_id), bbox_coordinates.get(gt_id)
        input_point_list, input_label_list, input_box_list, objm = prompt_generator(gt, gt_id, bboxes, centers)

        if get_points:
            if len(input_point_list) != (n_positive + n_negative):
                # to stay consistent, we add random points in the background of an object
                # if there's no neg region around the object - usually happens with small rois
                needed_points = (n_positive + n_negative) - len(input_point_list)
                more_neg_points = np.where(objm == 0)
                chosen_idxx = np.random.choice(len(more_neg_points[0]), size=needed_points)
                for idx in chosen_idxx:
                    input_point_list.append((more_neg_points[0][idx], more_neg_points[1][idx]))
                    input_label_list.append(0)
            assert len(input_point_list) == (n_positive + n_negative)
            _ip = [ip[::-1] for ip in input_point_list]  # to match the coordinate system used by SAM
            # NOTE: ADDL. STEP (transform coords as per expected format - see predictor.predict function for details)
            _ip = transform_function.apply_coords(np.array(_ip), gt.shape)
            input_point.append(_ip)
            input_label.append(input_label_list)

        if get_boxes:
            # indexes hard-coded to adapt with SAM's bbox format
            # default format: [a, b, c, d] -> SAM's format: [b, a, d, c]
            _ib = [input_box_list[0][1], input_box_list[0][0],
                   input_box_list[0][3], input_box_list[0][2]]
            # NOTE: ADDL. STEP (transform boxes as per expected format - see predictor.predict function for details)
            _ib = transform_function.apply_boxes(np.array(_ib), gt.shape)
            input_box.append(_ib)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_point = torch.tensor(np.array(input_point)).to(device) if len(input_point) > 0 else None
    input_label = torch.tensor(np.array(input_label)).to(device) if len(input_label) > 0 else None
    input_box = torch.tensor(np.array(input_box)).to(device) if len(input_box) > 0 else None

    # batched_inputs going into the predictor
    multimasking = False
    if n_positive == 1 and n_negative == 0:
        if not get_boxes:
            multimasking = True

    masks = []
    ious = []

    if past_inputs is not None:
        input_point, input_label = past_inputs

    n_samples = input_box.shape[0] if input_point is None else input_point.shape[0]
    n_batches = int(np.ceil(float(n_samples) / batch_size))

    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_stop = min((batch_idx + 1) * batch_size, n_samples)
            # print(batch_idx, batch_start, batch_stop, min(batch_size, n_samples))

            batch_points = None if input_point is None else input_point[batch_start:batch_stop]
            batch_labels = None if input_label is None else input_label[batch_start:batch_stop]
            batch_boxes = None if input_box is None else input_box[batch_start:batch_stop]

            batch_masks, batch_ious, _ = predictor.predict_torch(
                point_coords=batch_points, point_labels=batch_labels,
                boxes=batch_boxes, multimask_output=multimasking
            )
            masks.append(batch_masks)
            ious.append(batch_ious)
    masks = torch.cat(masks)
    ious = torch.cat(ious)
    assert len(masks) == len(ious) == n_samples

    save_points = {
        img_id: (input_point, input_label)
    }

    # TODO we should actually use non-max suppression here
    # I will implement it somewhere to have it refactored
    instance_labels = np.zeros_like(gt, dtype=int)
    for m, iou, gt_idx in zip(masks, ious, gt_ids):
        best_idx = torch.argmax(iou)
        best_mask = m[best_idx]
        instance_labels[best_mask.detach().cpu().numpy()] = gt_idx

    return instance_labels, save_points
