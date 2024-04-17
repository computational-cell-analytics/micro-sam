import os
from glob import glob
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components

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


def compare_cellpose_vs_ais(all_images, all_gt, dataset_name):
    amg_vanilla_root = os.path.join(ROOT, f"experiments/new_models/v3/vanilla/lm/{dataset_name}/vit_l/amg/inference/")

    if dataset_name == "livecell":
        usam_method = "specialist"
        cp_method = "livecell"
    else:
        usam_method = "generalist"
        cp_method = "cyto2"

    amg_spec_root = os.path.join(
        ROOT, f"experiments/new_models/v3/{usam_method}/lm/{dataset_name}/vit_l/amg/inference/"
    )
    ais_spec_root = os.path.join(
        ROOT,
        f"experiments/new_models/v3/{usam_method}/lm/{dataset_name}/vit_l/instance_segmentation_with_decoder/inference/"
    )
    assert os.path.exists(amg_vanilla_root), amg_vanilla_root
    assert os.path.exists(amg_spec_root), amg_spec_root
    assert os.path.exists(ais_spec_root), ais_spec_root

    cellpose_root = os.path.join(ROOT, f"experiments/benchmarking/cellpose/{dataset_name}/{cp_method}/predictions/")

    all_res = []
    for gt_path in tqdm(all_gt):
        image_id = os.path.split(gt_path)[-1]

        gt = imageio.imread(gt_path)
        ais_seg = imageio.imread(os.path.join(ais_spec_root, image_id))

        score = {
            "name": image_id.split(".")[0],
            "score": mean_segmentation_accuracy(ais_seg, gt)
        }
        all_res.append(pd.DataFrame.from_dict([score]))

    all_res = pd.concat(all_res, ignore_index=True)

    sscores = np.array(all_res["score"]).argsort()[::-1]
    best_image_ids = [all_res.iloc[sscore]["name"] for sscore in sscores]

    for image_path, gt_path in zip(all_images, all_gt):
        image_id = os.path.split(image_path)[-1]
        if Path(image_id).stem not in best_image_ids:
            continue

        gt = imageio.imread(gt_path)

        image = imageio.imread(image_path)
        image = _enhance_image(image, do_norm=True if dataset_name == "covid_if" else False)
        image = _overlay_mask(image, gt, alpha=0.95)
        image = _overlay_outline(image, gt, 0)

        amg_vanilla = imageio.imread(os.path.join(amg_vanilla_root, image_id))
        amg_spec = imageio.imread(os.path.join(amg_spec_root, image_id))
        ais_spec = imageio.imread(os.path.join(ais_spec_root, image_id))

        cellpose_seg = imageio.imread(
            os.path.join(
                cellpose_root, image_id if dataset_name == "livecell" else f"covid_if_{Path(image_id).stem}_00001.tif"
            )
        )

        fig, ax = plt.subplots(1, 5, figsize=(30, 20), sharex=True, sharey=True)

        ax[0].imshow(image, cmap="gray")
        ax[0].axis("off")

        ax[1].imshow(cellpose_seg, cmap=get_random_colors(cellpose_seg), interpolation="nearest")
        ax[1].axis("off")

        ax[2].imshow(amg_vanilla, cmap=get_random_colors(amg_vanilla), interpolation="nearest")
        ax[2].axis("off")

        ax[3].imshow(amg_spec, cmap=get_random_colors(amg_spec), interpolation="nearest")
        ax[3].axis("off")

        ax[4].imshow(ais_spec, cmap=get_random_colors(ais_spec), interpolation="nearest")
        ax[4].axis("off")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"./figs/{dataset_name}-cellpose/{Path(image_id).stem}.svg", bbox_inches="tight")
        plt.close()


def compare_covid_if_rf(
    all_images, all_gt, resource_choice="cpu_32G-mem_16-cores"
):
    base_dir = "/scratch/usr/nimanwai/experiments/resource-efficient-finetuning/"

    def_amg_dir = f"{base_dir}/vanilla/vit_b/amg/inference"
    gen_ais_dir = f"{base_dir}/generalist/vit_b/instance_segmentation_with_decoder/inference"

    root_dir = f"/scratch/usr/nimanwai/experiments/resource-efficient-finetuning/{resource_choice}"
    assert os.path.exists(root_dir)

    ais_dir = os.path.join(
        root_dir, "vit_b_lm", "freeze-None", "10-images", "instance_segmentation_with_decoder", "inference"
    )
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

    for image_path, gt_path in zip(all_images, all_gt):
        image_id = os.path.split(image_path)[-1]
        if Path(image_id).stem not in best_image_ids:
            continue

        gt = imageio.imread(gt_path)

        image = imageio.imread(image_path)
        image = _enhance_image(image)
        image = _overlay_mask(image, gt)
        image = _overlay_outline(image, gt, 0)

        amg_vanilla = imageio.imread(os.path.join(def_amg_dir, image_id))
        ais_default_ft = imageio.imread(
            os.path.join(
                root_dir, "vit_b", "freeze-None", "10-images",
                "instance_segmentation_with_decoder", "inference", image_id
            )
        )

        ais_gen = imageio.imread(os.path.join(gen_ais_dir, image_id))
        ais_gen_ft = imageio.imread(
            os.path.join(
                root_dir, "vit_b_lm", "freeze-None", "10-images",
                "instance_segmentation_with_decoder", "inference", image_id
            )
        )

        fig, ax = plt.subplots(1, 5, figsize=(30, 20), sharex=True, sharey=True)

        ax[0].imshow(image, cmap="gray")
        ax[0].axis("off")

        ax[1].imshow(amg_vanilla, cmap=get_random_colors(amg_vanilla), interpolation="nearest")
        ax[1].axis("off")

        ax[2].imshow(ais_default_ft, cmap=get_random_colors(ais_default_ft), interpolation="nearest")
        ax[2].axis("off")

        ax[3].imshow(ais_gen, cmap=get_random_colors(ais_gen), interpolation="nearest")
        ax[3].axis("off")

        ax[4].imshow(ais_gen_ft, cmap=get_random_colors(ais_gen_ft), interpolation="nearest")
        ax[4].axis("off")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"./{Path(image_id).stem}.svg", bbox_inches="tight")
        plt.close()


def compare_mitolab_mitonet_vs_ais(all_images, all_gt, dataset_name):
    amg_vanilla_root = os.path.join(ROOT, f"experiments/new_models/v3/vanilla/em/{dataset_name}/vit_l/amg/inference/")
    amg_gen_root = os.path.join(ROOT, f"experiments/new_models/v3/generalist/em/{dataset_name}/vit_l/amg/inference/")
    ais_gen_root = os.path.join(
        ROOT,
        f"experiments/new_models/v3/generalist/em/{dataset_name}/vit_l/instance_segmentation_with_decoder/inference/"
    )
    assert os.path.exists(amg_vanilla_root), amg_vanilla_root
    assert os.path.exists(amg_gen_root), amg_gen_root
    assert os.path.exists(ais_gen_root), ais_gen_root

    if dataset_name.endswith("tem"):
        dname = "tem"
        seg_dir = os.path.join(ROOT, "experiments", "benchmarking", "mitonet", "segmentations", "tem", "segmentations")
        all_segs = sorted(glob(os.path.join(seg_dir, "*")))
        mitonet_seg_vol = [imageio.imread(_path) for _path in all_segs]
        offset = 0
    else:
        dsplit = dataset_name.split("/")
        if len(dsplit) > 1:
            dname = f"{dsplit[0]}_{dsplit[1]}"
        else:
            dname = dataset_name

        mitonet_seg_path = glob(
            os.path.join(
                ROOT, "experiments", "benchmarking", "mitonet", "segmentations", dname, "*mitonet_seg.tif"
            )
        )
        mitonet_seg_vol = imageio.imread(mitonet_seg_path[0])
        offset = int(Path(all_gt[0]).stem.split("_")[-1])

    assert len(mitonet_seg_vol) == len(all_gt)

    all_res = []
    for gt_path in all_gt:
        image_id = os.path.split(gt_path)[-1]

        gt = imageio.imread(gt_path)
        ais_seg = imageio.imread(os.path.join(ais_gen_root, image_id))

        score = {
            "name": image_id.split(".")[0],
            "score": mean_segmentation_accuracy(ais_seg, gt)
        }
        all_res.append(pd.DataFrame.from_dict([score]))

    all_res = pd.concat(all_res, ignore_index=True)

    sscores = np.array(all_res["score"]).argsort()[::-1][:10]
    best_image_ids = [all_res.iloc[sscore]["name"] for sscore in sscores]

    for image_path, gt_path in zip(all_images, all_gt):
        image_id = os.path.split(image_path)[-1]
        if Path(image_id).stem not in best_image_ids:
            continue

        gt = imageio.imread(gt_path).astype("int")

        image = imageio.imread(image_path).astype("int")

        # image = _enhance_image(image, do_norm=False)
        image = _overlay_mask(image, gt, alpha=0.95)
        image = _overlay_outline(image, gt, 0)

        amg_vanilla = imageio.imread(os.path.join(amg_vanilla_root, image_id))
        amg_gen = imageio.imread(os.path.join(amg_gen_root, image_id))
        ais_gen = imageio.imread(os.path.join(ais_gen_root, image_id))

        if dname == "tem":
            mitonet_seg = connected_components(imageio.imread(os.path.join(seg_dir, image_id)))
        else:
            id_val = int(Path(image_id).stem.split("_")[-1]) - offset
            mitonet_seg = connected_components(mitonet_seg_vol[id_val])

        fig, ax = plt.subplots(1, 5, figsize=(30, 20), sharex=True, sharey=True)

        ax[0].imshow(image, cmap="gray")
        ax[0].axis("off")

        ax[1].imshow(mitonet_seg, cmap=get_random_colors(mitonet_seg), interpolation="nearest")
        ax[1].axis("off")

        ax[2].imshow(amg_vanilla, cmap=get_random_colors(amg_vanilla), interpolation="nearest")
        ax[2].axis("off")

        ax[3].imshow(amg_gen, cmap=get_random_colors(amg_gen), interpolation="nearest")
        ax[3].axis("off")

        ax[4].imshow(ais_gen, cmap=get_random_colors(ais_gen), interpolation="nearest")
        ax[4].axis("off")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"./{Path(image_id).stem}.svg", bbox_inches="tight")
        plt.close()


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
        image_paths = [_path for _path in glob(os.path.join(root_dir, "raw", "lucchi_test*.tif"))]
        gt_paths = [_path for _path in glob(os.path.join(root_dir, "labels", "lucchi_test*.tif"))]

    elif dataset_name.startswith("mitolab"):
        root_dir = os.path.join(ROOT, "data", "mitolab", "slices", dataset_name.split("/")[-1], "test")
        image_paths = [_path for _path in glob(os.path.join(root_dir, "raw", "*"))]
        gt_paths = [_path for _path in glob(os.path.join(root_dir, "labels", "*"))]

    else:
        raise ValueError

    assert len(image_paths) == len(gt_paths)

    return sorted(image_paths), sorted(gt_paths)


def compare_experiments(dataset_name, for_cellpose=True):
    all_images, all_gt = get_paths(dataset_name=dataset_name)

    if dataset_name == "livecell":
        compare_cellpose_vs_ais(all_images, all_gt, dataset_name="livecell")

    elif dataset_name.startswith("mitolab") or dataset_name == "lucchi":
        compare_mitolab_mitonet_vs_ais(all_images, all_gt, dataset_name)

    elif dataset_name == "covid_if":
        if for_cellpose:
            compare_cellpose_vs_ais(all_images, all_gt, dataset_name="covid_if")
        else:
            compare_covid_if_rf(all_images, all_gt)


def main():
    # compare_experiments("covid_if")

    # compare_experiments("mitolab/c_elegans")
    # compare_experiments("mitolab/fly_brain")
    # compare_experiments("mitolab/glycolytic_muscle")
    # compare_experiments("mitolab/hela_cell")
    # compare_experiments("mitolab/tem")
    # compare_experiments("lucchi")

    compare_experiments("livecell")

    # compare_experiments("covid_if", for_cellpose=True)


if __name__ == "__main__":
    main()
