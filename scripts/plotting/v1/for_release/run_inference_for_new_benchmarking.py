import os
from glob import glob
from natsort import natsorted

import pandas as pd
import imageio.v3 as imageio

import matplotlib.pyplot as plt
from matplotlib import rcParams

from elf.evaluation import mean_segmentation_accuracy

from torch_em.util.util import get_random_colors


rcParams['text.usetex'] = False

ROOT = "/home/nimanwai/micro-sam/development"

DATASET_NAME_MAPPING = {
    "arvidsson": "Arvidsson",
    "bitdepth_nucseg": "BitDepth NucSeg",
    "cellbindb": "CellBinDB",
    "cellpose": r"$\mathit{CellPose}$",
    "covid_if": "Covid IF",
    "deepbacs": r"$\mathit{DeepBacs}$",
    "deepseas": "DeepSeas",
    "dynamicnuclearnet": r"$\mathit{DynamicNuclearNet}$",
    "gonuclear": "GoNuclear",
    "hpa": "HPA",
    "ifnuclei": "IFNuclei",
    "livecell": r"$\mathit{LIVECell}$",
    "lizard": "Lizard",
    "neurips_cellseg": r"$\mathit{NeurIPS}$ $\mathit{CellSeg}$",
    "organoidnet": "OrganoIDNet",
    "orgasegment": r"$\mathit{OrgaSegment}$",
    "plantseg_root": r"$\mathit{PlantSeg}$ $\mathit{(Root)}$",
    "tissuenet": r"$\mathit{TissueNet}$",
    "toiam": "TOIAM",
    "vicar": "VICAR",
    "yeaz": r"$\mathit{YeaZ}$",
    "lucchi": "Lucchi",
    "mitolab_3d": "MitoLab (vEM)",
    "mitolab_tem": "MitoLab (TEM)",
    "uro_cell": "UroCell",
    "vnc": "VNC",
}


def get_comparison_plot_for_new_lm_models(metric, model_type):
    old_mfolder = f"{model_type}_v3"
    new_mfolder = f"{model_type}_v4"

    old_results = natsorted(glob(os.path.join(ROOT, old_mfolder, "*", "results", "ais_2d.csv")))
    new_results = natsorted(glob(os.path.join(ROOT, new_mfolder, "*", "results", "ais_2d.csv")))

    results = []
    for old_res_path, new_res_path in zip(old_results, new_results):
        dname = old_res_path.rsplit("/")[-3]
        assert dname in new_res_path, (old_res_path, new_res_path)

        res = {
            "dataset": DATASET_NAME_MAPPING[dname],
            f"{model_type}_lm": pd.read_csv(old_res_path)[metric][0],
            f"{model_type}_lm (NEW)": pd.read_csv(new_res_path)[metric][0],
        }
        results.append(pd.DataFrame.from_dict([res]))

    results = pd.concat(results, ignore_index=True)

    plt.figure(figsize=(12, 8))
    plt.plot(results["dataset"], results[f"{model_type}_lm"], marker="o", label="ViT-Base (v3)", linestyle="-")
    plt.plot(
        results["dataset"], results[f"{model_type}_lm (NEW)"], marker="s",
        label=r"$\mathit{(NEW)}$ ViT-Base (v4)", linestyle="--"
    )

    plt.xticks(rotation=90, fontweight="bold", fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Mean Segmentation Accuracy", fontsize=14, fontweight="bold")
    plt.title(r"$\mu$SAM LM Generalist Model", fontsize=12)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12, frameon=False)

    plt.savefig("./test_lm.png", dpi=300, bbox_inches="tight")


def get_comparison_plot_for_new_em_models(metric, model_type):
    old_mfolder = f"{model_type}_em_organelles_old"
    new_mfolder = f"{model_type}_em_organelles_v4"

    old_results = natsorted(glob(os.path.join(ROOT, old_mfolder, "*", "results", "ais_2d.csv")))
    new_results = natsorted(glob(os.path.join(ROOT, new_mfolder, "*", "results", "ais_2d.csv")))

    results = []
    for old_res_path, new_res_path in zip(old_results, new_results):
        dname = old_res_path.rsplit("/")[-3]
        assert dname in new_res_path, (old_res_path, new_res_path)

        res = {
            "dataset": DATASET_NAME_MAPPING[dname],
            f"{model_type}_em_organelles": pd.read_csv(old_res_path)[metric][0],
            f"{model_type}_em_organelles (NEW)": pd.read_csv(new_res_path)[metric][0],
        }
        results.append(pd.DataFrame.from_dict([res]))

    results = pd.concat(results, ignore_index=True)

    plt.figure(figsize=(12, 8))
    plt.plot(
        results["dataset"], results[f"{model_type}_em_organelles"], marker="o", label="ViT-Base (v2)", linestyle="-"
    )
    plt.plot(
        results["dataset"], results[f"{model_type}_em_organelles (NEW)"], marker="s",
        label=r"$\mathit{(NEW)}$ ViT-Base (v4)", linestyle="--"
    )

    plt.xticks(rotation=90, fontweight="bold", fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Mean Segmentation Accuracy", fontsize=14, fontweight="bold")
    plt.title(r"$\mu$SAM EM Organelles Generalist Model", fontsize=12)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12, frameon=False)

    plt.savefig("./test_em.png", dpi=300, bbox_inches="tight")


def plot_qualitative_ais():
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data"

    new_res_paths = glob(os.path.join("vit_b_new_model", "*", "ais_2d", "inference"))
    old_res_paths = glob(os.path.join("vit_b_old_model", "*", "ais_2d", "inference"))

    for npath, opath in zip(new_res_paths, old_res_paths):

        # Get the dataset name
        dataset = npath.rsplit("/")[-3]

        # Get all image and label paths
        from micro_sam.evaluation.benchmark_datasets import _get_image_label_paths
        image_paths, gt_paths = _get_image_label_paths(path=os.path.join(data_dir, dataset), ndim=2)

        # Get the predictions.
        old_predictions = [
            imageio.imread(os.path.join(opath, os.path.basename(p))) for p in image_paths
        ]
        new_predictions = [
            imageio.imread(os.path.join(npath, os.path.basename(p))) for p in image_paths
        ]

        assert old_predictions and len(old_predictions) == len(new_predictions)

        gts = [imageio.imread(p) for p in gt_paths]

        # Compute the scores and rank the best
        old_scores = [mean_segmentation_accuracy(p, gt) for p, gt in zip(old_predictions, gts)]
        new_scores = [mean_segmentation_accuracy(p, gt) for p, gt in zip(new_predictions, gts)]
        diff = [ns - os for ns, os in zip(new_scores, old_scores)]

        # Sort according to best results
        image_score_pairs = sorted(zip(diff, image_paths), key=lambda x: x[0], reverse=True)
        _, sorted_image_paths = zip(*image_score_pairs)

        # Let's plot the best ones now
        for i, image_path in enumerate(sorted_image_paths):

            if i == 3:
                break

            fig, axes = plt.subplots(1, 3, figsize=(10, 6))

            image = imageio.imread(image_path)
            if dataset != "tissuenet":
                from torch_em.transform.raw import normalize
                image = (normalize(image) * 255).astype("uint8")

            axes[0].imshow(image, cmap="gray")
            axes[0].axis("off")

            opred = imageio.imread(os.path.join(opath, os.path.basename(image_path)))
            axes[1].imshow(opred, cmap=get_random_colors(opred))
            axes[1].axis("off")

            npred = imageio.imread(os.path.join(npath, os.path.basename(image_path)))
            axes[2].imshow(npred, cmap=get_random_colors(npred))
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("./test.svg", bbox_inches="tight")
            plt.savefig("./test.png", bbox_inches="tight")
            plt.close()


def main():
    # get_comparison_plot_for_new_lm_models(metric="mSA", model_type="vit_t")
    get_comparison_plot_for_new_lm_models(metric="mSA", model_type="vit_b")
    # get_comparison_plot_for_new_lm_models(metric="mSA", model_type="vit_l")

    get_comparison_plot_for_new_em_models(metric="mSA", model_type="vit_b")

    # plot_qualitative_ais()


main()
