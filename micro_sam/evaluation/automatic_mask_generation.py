import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import imageio.v2 as imageio
from elf.evaluation import mean_segmentation_accuracy

from micro_sam.instance_segmentation import mask_data_to_segmentation
from micro_sam.util import precompute_image_embeddings
from micro_sam.instance_segmentation import AutomaticMaskGenerator


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


def per_image_amg(
        predictor,
        image_paths,
        gt_paths,
        embedding_dir
):
    amg = AutomaticMaskGenerator(predictor)

    for _img_path, _gt_path in tqdm(zip(image_paths, gt_paths), desc="Grid search..."):
        # check if the grid search is completed already
        img_name = os.path.basename(_img_path)
        gs_save_dir = "./grid_search"
        os.makedirs(gs_save_dir, exist_ok=True)
        gs_save_path = os.path.join(gs_save_dir, f"{img_name[:-4]}.csv")
        if os.path.exists(gs_save_path):
            continue

        image = imageio.imread(_img_path)
        gt = imageio.imread(_gt_path)

        embedding_path = os.path.join(embedding_dir, "embeddings", f"{img_name[:-4]}.zarr")
        image_embeddings = precompute_image_embeddings(predictor, image, embedding_path)
        amg.initialize(image, image_embeddings)

        _grid_search(img_name, image, gt, gs_save_path, amg)


def get_auto_segmentation_from_gs(
        predictor,
        image_paths,
        save_pred_dir,
        embedding_dir
):
    # mean over all image results per parameter for getting the best parameter
    search_range_for_iou = get_range_of_search_values([0.6, 0.9])
    search_range_for_ss = get_range_of_search_values([0.6, 0.95])
    list_of_combs = [(r1, r2) for r1 in search_range_for_iou for r2 in search_range_for_ss]

    gs_save_dir = glob("./grid_search/*.csv")

    f_list = []
    for i, comb in enumerate(tqdm(list_of_combs)):
        tmp_list_of_maps = []
        for p in sorted(gs_save_dir):
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

    # obtain the best parameters from the analysis above
    idxs = [i for i, x in enumerate(df["Mean mSA"]) if x == max(df["Mean mSA"])]
    f_iou_thresh, f_ss_thresh = [], []
    for idx in idxs:
        chosen_row = df.loc[[idx]]
        iou_thresh = chosen_row["pred_iou_thresh"].tolist()[0]
        ss_thresh = chosen_row["stability_score_thresh"].tolist()[0]
        print(f"The AMG grid search has the best performance at \
              IoU thresh: {iou_thresh} and Stability Score: {ss_thresh}")
        f_iou_thresh.append(iou_thresh)
        f_ss_thresh.append(ss_thresh)

    iou_thresh, ss_thresh = f_iou_thresh[0], f_ss_thresh[0]

    os.makedirs(save_pred_dir, exist_ok=True)

    amg = AutomaticMaskGenerator(predictor)

    for img_path in tqdm(image_paths, desc="Auto Predictions..."):
        img_name = os.path.basename(img_path)
        image = imageio.imread(img_path)

        embedding_path = os.path.join(embedding_dir, "embeddings", f"{img_name[:-4]}.zarr")
        image_embeddings = precompute_image_embeddings(predictor, image, embedding_path)
        amg.initialize(image, image_embeddings)

        masks = amg.generate(pred_iou_thresh=iou_thresh, stability_score_thresh=ss_thresh)
        instance_labels = mask_data_to_segmentation(masks, image.shape, with_background=True)
        imageio.imsave(os.path.join(save_pred_dir, img_name), instance_labels)
