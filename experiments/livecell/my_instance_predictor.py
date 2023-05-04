import os
import vigra
import napari
import argparse
import numpy as np
import seaborn as sns
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio
from elf.evaluation import mean_average_precision

from segment_anything import SamAutomaticMaskGenerator

from micro_sam.prompt_generators import PointPromptGenerator
from micro_sam.util import get_cell_center_coordinates, get_sam_model


def sam_predictor(
        image, gt, predictor, view,
        prompt_generator=PointPromptGenerator(n_positive_points=1, n_negative_points=0, dilation_strength=3)
):
    """
    Generates instance segmentation from each assigned seed
    """
    # currently works with regionprops
    # returns the set of cell coordinates and respective bboxes for all instances
    center_coordinates, bbox_coordinates = get_cell_center_coordinates(gt)

    instance_labels = np.zeros((gt.shape), dtype=int)

    for i, my_prompts in enumerate(zip(center_coordinates, bbox_coordinates)):
        centers, bboxes = my_prompts   # center and bbox coordinates per instance

        input_point_list, input_label_list, input_box, objm = prompt_generator(gt, i, centers, bboxes)

        _ip = [ip[::-1] for ip in input_point_list]  # to match the coordinate system used by SAM

        input_point = np.array(_ip)
        input_label = np.array(input_label_list)

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False
        )

        if view:
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(gt)
            v.add_labels(objm)
            v.add_labels(masks)
            v.add_points(np.array(input_point_list))  # using the "usual" coordinate system for visualisation
            napari.run()

        instance_labels[masks[0]] = i+1

    return instance_labels


def sam_predictions_for_livecell(args, check_instances=False):
    """
    Gets instance labels for all the livecell images
    """
    _, predictor = get_sam_model(
        my_ckpt_path="C:/Users/anwai/projects/micro-sam/experiments/livecell/checkpoints/sam_vit_h_4b8939.pth")

    root_gt_dir = args.input[0] + "annotations/livecell_test_images/"
    root_img_dir = args.input[0] + "images/livecell_test_images/"

    assert os.path.exists(root_gt_dir), "The path provided doesn't have the LiveCELL images"

    if args.cell_type is None:
        gt_cell_dirs = root_gt_dir + "*"
    else:
        gt_cell_dirs = root_gt_dir + f"{args.cell_type[0]}"

    for gt_cell_dir in glob(gt_cell_dirs):
        for gt_path in tqdm(glob(gt_cell_dir + "/*")):
            gt = imageio.imread(gt_path)
            gt, _, _ = vigra.analysis.relabelConsecutive(gt.astype("uint32"))

            my_fname = os.path.split(gt_path)[-1]

            img_path = root_img_dir + my_fname
            image = imageio.imread(img_path)
            image = np.stack((image,)*3, axis=-1)  # to satisfy the requirement of channel dimensions for SAM

            predictor.set_image(image)

            instances = sam_predictor(image, gt, predictor, view=False)

            if check_instances:
                v = napari.Viewer()
                v.add_image(image)
                v.add_labels(instances)
                napari.run()

            if args.save:
                imageio.imsave(args.pred_path[0] + f"{my_fname}", instances)


def auto_mask_gen(args):
    """
    Automatically generates instance labels from SAM
    """
    model, _ = get_sam_model()
    mask_generator = SamAutomaticMaskGenerator(model)

    root_img_dir = "/scratch/usr/nimanwai/data/livecell/images/livecell_test_images/"

    if args.cell_type is None:
        img_dir = root_img_dir + "*"
    else:
        img_dir = root_img_dir + f"{args.cell_type[0]}_*"

    for i, img_path in enumerate(tqdm(glob(img_dir))):
        img_name = os.path.split(img_path)[-1]

        image = imageio.imread(img_path)

        instance_labels = np.zeros((image.shape[0], image.shape[1]), dtype=int)

        masks = mask_generator.generate(image)

        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        for i, ann in enumerate(sorted_anns):
            if i > 0:
                m = ann['segmentation']
                instance_labels[m] = i

        if args.save:
            imageio.imsave(os.path.join(args.pred_path[0], f"{img_name}"), instance_labels)


def analyse_sam_predictions(args):
    """
    Analyse the prediction generated from SAM (prompted/auto) based on IoU-50
    - calculates the evaluation metrics (IoU, average precision & mean average precision) on the instance segmentations
    - produces histograms (based on IoU-50) on LiveCELL test images
    """
    assert args.hist or args.iou is True, "choose either --hist or --iou"

    root_gt_dir = args.input[0] + "annotations/livecell_test_images/"

    if args.cell_type is None:
        gt_cell_dirs = root_gt_dir + "*"
    else:
        gt_cell_dirs = root_gt_dir + f"{args.cell_type[0]}"

    list_ious = []
    list_m_aps = []

    for gt_cell_dir in glob(gt_cell_dirs):
        for gt_path in tqdm(glob(gt_cell_dir + "/*")):
            gt = imageio.imread(gt_path)

            my_fname = os.path.split(gt_path)[-1]

            pred_path = args.pred_path[0] + my_fname
            pred = imageio.imread(pred_path)

            raise NotImplementedError
            m_aps, aps = mean_average_precision(pred, gt, return_aps=True)  # TODO
            my_iou = intersection_over_union(pred, gt, threshold=0.5)  # TODO

            list_m_aps.append(m_aps)
            list_ious.append(my_iou)

    print(np.mean(list_m_aps))
    print(np.mean(list_ious))

    if args.hist:
        # Histogram for LiveCELL Image's Count of IoU-50 Scores
        fig = sns.displot(list_ious, bins=16, kde=True)
        fig.set(xlabel="IoU-50",
                title=f"{args.cell_type[0]} \nMean Iou: {round(np.mean(list_ious), 3)}"
                if args.cell_type is not None else f"LiveCELL \nMean IoU: {round(np.mean(list_ious), 3)}",
                xlim=(0, 1), ylim=(0, len(list_ious)))
        fig.savefig("figs/hist-livecell.png"
                    if args.cell_type is None else f"figs/hist-{args.cell_type[0]}.png")

    if args.iou:
        # Plot for IoU-50-per-image
        fig = sns.lineplot(list_ious)
        fig1 = fig.get_figure()
        fig1.savefig("figs/iou-per-val-livecell.png"
                     if args.cell_type is None else f"figs/iou-per-val-{args.cell_type[0]}.png")


def assort_qualitative_results(args):
    root_pred_dir = "/scratch/usr/nimanwai/predictions/"
    root_gt_dir = args.input[0] + "annotations/livecell_test_images/"
    root_img_dir = args.input[0] + "images/livecell_test_images/"

    if args.cell_type is not None:
        root_gt_dir = root_gt_dir + f"{args.cell_type[0]}/{args.cell_type[0]}_*"
    else:
        root_gt_dir = root_gt_dir + "*/*"

    for gt_path in tqdm(glob(root_gt_dir)):
        img_name = os.path.split(gt_path)[-1]
        img = imageio.imread(os.path.join(root_img_dir, img_name))

        gt = imageio.imread(gt_path)
        prompt_pred = imageio.imread(os.path.join(root_pred_dir, "sam_manual", img_name))
        auto_pred = imageio.imread(os.path.join(root_pred_dir, "sam_auto", img_name))

        # Obtain overlay of seed points
        coordinates = get_cell_center_coordinates(gt)

        coord_list = []
        label_list = [1] * len(coordinates)
        for centers in coordinates:
            coord_list.append(centers[::-1])

        input_point, input_label = np.array(coord_list), np.array(label_list)

        # TODO - use napari to visualise the cells
        raise NotImplementedError

        imageio.imsave(f"images/{img_name[:-4]}.png", img)


def main(args):
    assert args.prompt or args.auto is True, "Provide a mode for using SAM [--prompt/--auto]"
    assert args.input is not None, "Provide the directory for LiveCELL dataset [-i/--input]"

    if args.pred_path is not None:
        os.makedirs(args.pred_path[0], exist_ok=True)

    if args.prompt:
        print("Generating prompted instance-level predictions on LiveCELL test set..")
        sam_predictions_for_livecell(args)

    elif args.auto:
        print("Generating automatic masks from SAM on LiveCELL test set..")
        auto_mask_gen(args)

    elif args.eval:
        print("Analysis of the SAM predictions..")
        analyse_sam_predictions(args)

    elif args.assess:
        print("Qualitative assessment of SAM..")
        assort_qualitative_results(args)


def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, nargs="+")
    parser.add_argument("-c", "--cell_type", nargs="+", default=None)
    parser.add_argument("--pred_path", type=str, default=None, nargs="+")
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--iou", action='store_true')
    parser.add_argument("--hist", action='store_true')

    parser.add_argument("--prompt", action='store_true')
    parser.add_argument("--auto", action='store_true')

    parser.add_argument("--eval", action='store_true')

    parser.add_argument("--assess", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = my_args()
    main(args)
