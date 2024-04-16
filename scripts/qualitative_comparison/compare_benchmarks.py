import os
from glob import glob
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.evaluation.livecell import _get_livecell_paths
from micro_sam.evaluation.model_comparison import _enhance_image, _overlay_outline, _overlay_mask


ROOT = "/scratch/projects/nim00007/sam/"


# a function to generate a random color map for a label image
def get_random_colors(labels):
    n_labels = len(np.unique(labels)) - 1
    cmap = [[0, 0, 0]] + np.random.rand(n_labels, 3).tolist()
    cmap = colors.ListedColormap(cmap)
    return cmap


def compare_livecell_cellpose_vs_ais(all_images, all_gt):
    cellpose_root = os.path.join(ROOT, "experiments/benchmarking/cellpose/livecell/livecell/predictions/")
    amg_root = os.path.join(ROOT, "experiments/new_models/v3/specialist/lm/livecell/vit_l/amg/inference/")
    ais_root = os.path.join(
        ROOT, "experiments/new_models/v3/specialist/lm/livecell/vit_l/instance_segmentation_with_decoder/inference/"
    )

    # let's identify the best performing cases first
    all_scores = []
    for image_path, gt_path in zip(all_images, all_gt):
        image_id = Path(image_path).stem

        # cellpose_seg = imageio.imread(os.path.join(cellpose_root, f"{image_id}.tif"))
        ais_seg = imageio.imread(os.path.join(ais_root, f"{image_id}.tif"))
        # amg_seg = imageio.imread(os.path.join(amg_root, f"{image_id}.tif"))
        gt = imageio.imread(gt_path)

        score = {
            "name": image_id,
            "score": mean_segmentation_accuracy(ais_seg, gt)
        }
        all_scores.append(pd.DataFrame.from_dict([score]))

    results = pd.concat(all_scores, ignore_index=True)

    # once we have identified the best performing cases, we get the comparison plots next


def compare_covid_if_cellpose_vs_ais(
    all_images, all_gt, resource_choice="cpu_32G-mem_16-cores", model_choice="vit_b"
):
    base_dir = "/scratch/usr/nimanwai/experiments/resource-efficient-finetuning/"
    def_amg_dir = f"{base_dir}/vanilla/vit_b/amg/inference"
    gen_amg_dir = f"{base_dir}/generalist/vit_b/amg/inference"
    gen_ais_dir = f"{base_dir}/generalist/vit_b/instance_segmentation_with_decoder/inference"

    root_dir = f"/scratch/usr/nimanwai/experiments/resource-efficient-finetuning/{resource_choice}/{model_choice}/"
    assert os.path.exists(root_dir)

    ais_dir = os.path.join(root_dir, "freeze-None", "10-images", "instance_segmentation_with_decoder", "inference")
    all_res = []
    for gt_path in tqdm(all_gt):
        image_id = os.path.split(gt_path)[-1]

        gt = imageio.imread(gt_path)
        ais_seg = imageio.imread(os.path.join(ais_dir, image_id))

        all_res.append(
            pd.DataFrame.from_dict(
                [{
                    "name": image_id.split(".")[0],
                    "score": mean_segmentation_accuracy(ais_seg, gt)
                }]
            )
        )

    all_res = pd.concat(all_res, ignore_index=True)

    sscores = np.array(all_res["score"]).argsort()[::-1][:5]
    best_image_ids = [all_res.iloc[sscore]["name"] for sscore in sscores]

    def _get_image(experiment, n_images, image_id):
        _path = os.path.join(
            root_dir, "freeze-None", n_images,
            "instance_segmentation_with_decoder" if experiment == "ais" else experiment,
            "inference", f"{image_id}.tif"
        )
        image = imageio.imread(_path)
        return image

    for image_path, gt_path in zip(all_images, all_gt):
        image_id = Path(image_path).stem
        if image_id not in best_image_ids:
            continue

        gt = imageio.imread(gt_path)

        image = imageio.imread(image_path)
        image = _enhance_image(image)
        image = _overlay_mask(image, gt)
        image = _overlay_outline(image, gt, 0)

        # now, we (ideally) would like to see the plots from all the models
        amg_vanilla = imageio.imread(os.path.join(def_amg_dir, f"{image_id}.tif"))
        amg_gen = imageio.imread(os.path.join(gen_amg_dir, f"{image_id}.tif"))
        ais_gen = imageio.imread(os.path.join(gen_ais_dir, f"{image_id}.tif"))

        amg_10 = _get_image("amg", "10-images", image_id)
        ais_10 = _get_image("ais", "10-images", image_id)

        fig, ax = plt.subplots(1, 6, figsize=(30, 20), sharex=True, sharey=True)

        ax[0].imshow(image, cmap="gray")
        # ax[0].imshow(boundaries, cmap="magma", interpolation="none", alpha=0.9 * (boundaries > 0))
        # ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(amg_vanilla, cmap=get_random_colors(amg_vanilla), interpolation="nearest")
        # ax[1].set_title("AMG (Default)")
        ax[1].axis("off")

        ax[2].imshow(amg_gen, cmap=get_random_colors(amg_gen), interpolation="nearest")
        # ax[2].set_title("AMG (LM)")
        ax[2].axis("off")

        ax[3].imshow(ais_gen, cmap=get_random_colors(ais_gen), interpolation="nearest")
        # ax[3].set_title("AIS (LM)")
        ax[3].axis("off")

        ax[4].imshow(amg_10, cmap=get_random_colors(amg_10), interpolation="nearest")
        # ax[4].set_title("AMG (Finetuned)")
        ax[4].axis("off")

        ax[5].imshow(ais_10, cmap=get_random_colors(ais_10), interpolation="nearest")
        # ax[5].set_title("AIS (Finetuned)")
        ax[5].axis("off")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"./{image_id}_{model_choice}.svg", bbox_inches="tight")
        plt.close()


def compare_mitolab_mitonet_vs_ais(all_images, all_gt, dataset_name):
    mitonet_root = os.path.join(ROOT, "experiments/benchmarking/mitonet/")
    amg_root = os.path.join(ROOT, f"experiments/new_models/v3/generalist/em/{dataset_name}/vit_l/amg/inference/")
    ais_root = os.path.join(
        ROOT,
        f"experiments/new_models/v3/specialist/em/{dataset_name}/vit_l/instance_segmentation_with_decoder/inference/"
    )

    all_scores = []
    for image_path, gt_path in zip(all_images, all_gt):
        image_id = Path(image_path).stem

        # cellpose_seg = imageio.imread(os.path.join(cellpose_root, f"{image_id}.tif"))
        ais_seg = imageio.imread(os.path.join(ais_root, f"{image_id}.tif"))
        # amg_seg = imageio.imread(os.path.join(amg_root, f"{image_id}.tif"))
        gt = imageio.imread(gt_path)

        score = {
            "image": image_id,
            "msa": mean_segmentation_accuracy(ais_seg, gt)
        }
        all_scores.append(pd.DataFrame.from_dict([score]))

    results = pd.concat(all_scores, ignore_index=True)

    # once this is done, we check the n best cases and compare them with amg and ais


def get_paths(dataset_name):
    if dataset_name == "livecell":
        image_paths, gt_paths = _get_livecell_paths(
            input_folder=os.path.join(ROOT, "data", "livecell"), split="test"
        )

    elif dataset_name == "covid_if":
        root_dir = os.path.join(ROOT, "data", dataset_name, "slices", "test")
        image_paths = [_path for _path in glob(os.path.join(root_dir, "raw", "*.tif"))]
        gt_paths = [_path for _path in glob(os.path.join(root_dir, "labels", "*.tif"))]

    elif dataset_name == "lucchi":
        root_dir = os.path.join(ROOT, "data", dataset_name, "slices")
        image_paths = [_path for _path in glob(os.path.join(root_dir, "raw", "*.tif"))]
        gt_paths = [_path for _path in glob(os.path.join(root_dir, "labels", "*.tif"))]

    elif dataset_name.startswith("mitolab"):
        root_dir = os.path.join(ROOT, "data", "mitolab", "slices", dataset_name, "test")
        image_paths = [_path for _path in glob(os.path.join(root_dir, "raw", "*.tif"))]
        gt_paths = [_path for _path in glob(os.path.join(root_dir, "labels", "*.tif"))]

    else:
        raise ValueError

    return sorted(image_paths), sorted(gt_paths)


def compare_experiments(dataset_name):
    all_images, all_gt = get_paths(dataset_name=dataset_name)

    if dataset_name == "livecell":
        compare_livecell_cellpose_vs_ais(all_images, all_gt)

    elif dataset_name.startswith("mitolab") or dataset_name == "lucchi":
        compare_mitolab_mitonet_vs_ais(all_images, all_gt, dataset_name)

    elif dataset_name == "covid_if":
        compare_covid_if_cellpose_vs_ais(all_images, all_gt)


def main():
    compare_experiments("covid_if")


if __name__ == "__main__":
    main()
