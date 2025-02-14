import os
from glob import glob
from natsort import natsorted

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams


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
}


def get_comparison_plot_for_new_models(metric, model_type):
    old_mfolder = f"{model_type}_old_model"
    new_mfolder = f"{model_type}_new_model"

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
    print(results)

    plt.figure(figsize=(12, 8))
    plt.plot(results["dataset"], results[f"{model_type}_lm"], marker="o", label="ViT-Base (v2)", linestyle="-")
    plt.plot(
        results["dataset"], results[f"{model_type}_lm (NEW)"], marker="s",
        label=r"$\mathit{(NEW)}$ ViT-Base (v3)", linestyle="--"
    )

    plt.xticks(rotation=90, fontweight="bold", fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Mean Segmentation Accuracy", fontsize=14, fontweight="bold")
    plt.title(r"$\mu$SAM LM Generalist Model", fontsize=12)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12, frameon=False)

    plt.savefig("./test.png", dpi=300, bbox_inches="tight")


def main():
    # get_comparison_plot_for_new_models(metric="mSA", model_type="vit_t")
    get_comparison_plot_for_new_models(metric="mSA", model_type="vit_b")
    # get_comparison_plot_for_new_models(metric="mSA", model_type="vit_l")


main()
