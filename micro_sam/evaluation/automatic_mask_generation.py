import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import imageio.v2 as imageio
from elf.evaluation import mean_segmentation_accuracy

from micro_sam.instance_segmentation import mask_data_to_segmentation
from micro_sam.util import precompute_image_embeddings
from micro_sam.instance_segmentation import AutomaticMaskGenerator

from .inference import get_predictor


def get_range_of_search_values(input_vals):
    if isinstance(input_vals, list):
        search_range = np.arange(input_vals[0], input_vals[1] + 0.01, 0.01)
        search_range = [round(e, 2) for e in search_range]
    else:
        search_range = [input_vals]
    return search_range


def _grid_search(img_name, image, gt, img_save_path, amg):
    search_range_for_iou = get_range_of_search_values([0.6, 0.9])
    search_range_for_ss = get_range_of_search_values([0.6, 0.95])

    net_list = []
    for iou_thresh in search_range_for_iou:
        for stability_thresh in search_range_for_ss:
            masks = amg.generate(pred_iou_thresh=iou_thresh, stability_score_thresh=stability_thresh)
            instance_labels = mask_data_to_segmentation(masks, image.shape, with_background=True)
            m_sas, sas = mean_segmentation_accuracy(instance_labels, gt, return_accuracies=True)  # type: ignore

            result_dict = {
                "cell_name": img_name,
                "pred_iou_thresh": iou_thresh,
                "stability_score_thresh": stability_thresh,
                "mSA": m_sas,
                "SA50": sas[0],
                "SA75": sas[5]
            }
            tmp_df = pd.DataFrame([result_dict])
            net_list.append(tmp_df)

    img_gs_df = pd.concat(net_list)
    img_gs_df.to_csv(img_save_path, index=False)


def per_image_amg(args):
    predictor = get_predictor(args.ckpt, args.model)
    amg = AutomaticMaskGenerator(predictor)

    f = open(os.path.join(args.input, "val.json"))
    data = json.load(f)
    livecell_val_ids = [i["file_name"] for i in data["images"]]

    root_img_dir = os.path.join(args.input, "images", "livecell_train_val_images")
    root_gt_dir = os.path.join(args.input, "annotations", "livecell_train_val_images")

    for img_name in tqdm(livecell_val_ids, desc=f"Grid search for {args.name}"):
        # check if the grid search is completed already
        img_save_dir = os.path.join(args.save, args.name)
        os.makedirs(img_save_dir, exist_ok=True)
        img_save_path = os.path.join(img_save_dir, f"{img_name[:-4]}.csv")
        if os.path.exists(img_save_path):
            continue

        image = imageio.imread(os.path.join(root_img_dir, img_name))
        gt = imageio.imread(os.path.join(root_gt_dir, img_name.split("_")[0], img_name))

        embedding_path = os.path.join(args.embedding_path, "embeddings", args.name, f"{img_name[:-4]}.zarr")
        image_embeddings = precompute_image_embeddings(predictor, image, embedding_path)
        amg.initialize(image, image_embeddings)

        _grid_search(img_name, image, gt, img_save_path, amg)


def analyse_amg_grid_search_per_param(args):
    # mean over all image results per parameter for getting the best parameter
    search_range_for_iou = get_range_of_search_values([0.6, 0.9])
    search_range_for_ss = get_range_of_search_values([0.6, 0.95])
    list_of_combs = [(r1, r2) for r1 in search_range_for_iou for r2 in search_range_for_ss]

    img_save_dir = glob(os.path.join(args.save, args.name, "*"))

    f_list = []
    for i, comb in enumerate(tqdm(list_of_combs)):
        tmp_list_of_maps = []
        for p in sorted(img_save_dir):
            df = pd.read_csv(p)
            map_list = df["mSA"].tolist()
            tmp_list_of_maps.append(map_list[i])
        result = {
            "pred_iou_thresh": comb[0],
            "stability_score_thresh": comb[1],
            "Mean mSA": np.mean(tmp_list_of_maps)
        }
        f_list.append(pd.DataFrame([result]))

    df = pd.concat(f_list)
    csv_path = f"{args.name}.csv"
    df.to_csv(csv_path, index=False)


def get_auto_segmentation_from_gs(args):
    # obtain the best parameters from the analysis above
    df = pd.read_csv(f"{args.name}.csv")
    idxs = [i for i, x in enumerate(df["Mean mSA"]) if x == max(df["Mean mSA"])]
    f_iou_thresh, f_ss_thresh = [], []
    for idx in idxs:
        chosen_row = df.loc[[idx]]
        iou_thresh = chosen_row["pred_iou_thresh"].tolist()[0]
        ss_thresh = chosen_row["stability_score_thresh"].tolist()[0]
        print(f"{args.name} has the best performance at IoU thresh: {iou_thresh} and Stability Score: {ss_thresh}")
        f_iou_thresh.append(iou_thresh)
        f_ss_thresh.append(ss_thresh)

    iou_thresh, ss_thresh = f_iou_thresh[0], f_ss_thresh[0]

    save_auto_pred_dir = os.path.join(args.save_auto_pred, args.name, "auto_gs")
    os.makedirs(save_auto_pred_dir, exist_ok=True)

    predictor = get_predictor(args.ckpt, args.model)
    amg = AutomaticMaskGenerator(predictor)

    root_img_dir = glob(os.path.join(args.input, "images", "livecell_test_images", "*"))

    for img_path in tqdm(root_img_dir, desc="Auto Predictions"):
        img_name = os.path.basename(img_path)
        image = imageio.imread(img_path)

        embedding_path = os.path.join(args.embedding_path, "embeddings", args.name, f"{img_name[:-4]}.zarr")
        image_embeddings = precompute_image_embeddings(predictor, image, embedding_path)
        amg.initialize(image, image_embeddings)

        masks = amg.generate(pred_iou_thresh=iou_thresh, stability_score_thresh=ss_thresh)
        instance_labels = mask_data_to_segmentation(masks, image.shape, with_background=True)
        imageio.imsave(os.path.join(save_auto_pred_dir, img_name), instance_labels)


def main(args):
    per_image_amg(args)
    analyse_amg_grid_search_per_param(args)
    get_auto_segmentation_from_gs(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./livecell", help="Provide path where LIVECell data exists")
    parser.add_argument("-e", "--embedding_path", type=str, default="./embeddings/",
                        help="Path where the image embeddings will be saved")
    parser.add_argument("-s", "--save", type=str, default="./grid_search/",
                        help="Path where the grid search results per image will be saved")
    parser.add_argument("--name", type=str, default="livecell",
                        help="Name of the specific model to make distinction in all embeddings")
    parser.add_argument("-m", "--model", type=str, default="vit_b")
    parser.add_argument("-c", "--ckpt", type=str,
                        help="Checkpoint of the custom / vanilla model to get the automatic masks from")
    parser.add_argument("--save_auto_pred", type=str, default="./predictions/",
                        help="Path to save the grid search-based amg predictions")
    args = parser.parse_args()
    main(args)
