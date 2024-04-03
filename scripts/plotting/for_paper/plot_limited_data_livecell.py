import os
from glob import glob
from pathlib import Path
from natsort import natsorted

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = "/home/anwai/results/dfki/R3/"

PALETTE = {
    "ais": "#045275",
    "amg": "#089099",
    "point": "#7CCBA2",
    r"i$_{p}$": "#FCDE9C",
    "box": "#F0746E",
    r"i$_{b}$": "#90477F"
}

plt.rcParams.update({"font.size": 24})


def get_limited_data_livecell(res_root, model):
    # experiments from dfki
    all_results = []
    all_experiments_dir = natsorted(glob(os.path.join(res_root, model, "*")))
    for experiment_dir in all_experiments_dir:
        experiment_name = os.path.split(experiment_dir)[-1]

        ais = pd.read_csv(os.path.join(experiment_dir, "results_ais", "instance_segmentation_with_decoder.csv"))
        amg = pd.read_csv(os.path.join(experiment_dir, "results_amg", "amg.csv"))
        ip = pd.read_csv(os.path.join(experiment_dir, "results_ip", "iterative_prompts_start_point.csv"))
        ib = pd.read_csv(os.path.join(experiment_dir, "results_ipb", "iterative_prompts_start_box.csv"))

        res = {
            "experiment": int(experiment_name.split("_")[0]),
            "ais": ais.iloc[0]["msa"],
            "amg": amg.iloc[0]["msa"],
            "point": ip.iloc[0]["msa"],
            "box": ib.iloc[0]["msa"],
            r"i$_{p}$": ip.iloc[-1]["msa"],
            r"i$_{b}$": ib.iloc[-1]["msa"]
        }
        all_results.append(pd.DataFrame.from_dict([res]))

    # res_df = pd.concat(all_results, ignore_index=True)
    return all_results


def get_vanilla_and_finetuned_results(res_root, model):
    all_results = get_limited_data_livecell(res_root, model)

    def _get_results(method):
        assert method in ["vanilla", "specialist"]
        root_dir = f"/home/anwai/results/micro-sam/livecell/{method}/{model}"

        amg = pd.read_csv(os.path.join(root_dir, "amg.csv"))
        ip = pd.read_csv(os.path.join(root_dir, "iterative_prompts_start_point.csv"))
        ib = pd.read_csv(os.path.join(root_dir, "iterative_prompts_start_box.csv"))

        have_ais = False
        if method == "specialist":
            ais = pd.read_csv(os.path.join(root_dir, "instance_segmentation_with_decoder.csv"))
            have_ais = True

        res = {
            "experiment": 0 if method == "vanilla" else 100,
            "amg": amg.iloc[0]["msa"],
            "point": ip.iloc[0]["msa"],
            "box": ib.iloc[0]["msa"],
            r"i$_{p}$": ip.iloc[-1]["msa"],
            r"i$_{b}$": ib.iloc[-1]["msa"]
        }
        if have_ais:
            res["ais"] = ais.iloc[0]["msa"]

        return pd.DataFrame.from_dict([res])

    all_results.insert(0, _get_results("vanilla"))
    all_results.insert(len(all_results), _get_results("specialist"))
    res_df = pd.concat(all_results, ignore_index=True)
    return res_df


def get_plots(res_root, model, for_supp=None):
    plt.figure(figsize=(20, 10))
    res = get_vanilla_and_finetuned_results(res_root, model)
    sns.lineplot(
        data=pd.melt(res, "experiment"), x="experiment", y="value", hue="variable", marker="d", palette=PALETTE
    )
    plt.ylabel("Segmentation Accuracy", labelpad=15)
    plt.xlabel("Percent of Data", labelpad=15)
    plt.legend(loc="lower center", ncol=6)
    if for_supp is None:
        plt.title("Finetuning with Reduced Data")
        save_path = f"livecell_{model}_reduce_data.svg"
    else:
        plt.title(for_supp)
        save_path = f"livecell_supplementary_{model}_reduce_data.svg"

    plt.savefig(save_path)
    plt.savefig(Path(save_path).with_suffix(".pdf"))


def main():
    # for figure 2
    get_plots(ROOT, "vit_l")

    # for supplementary figure 1
    get_plots(ROOT, "vit_b", "ViT Base")
    get_plots(ROOT, "vit_l", "ViT Large")
    get_plots(ROOT, "vit_h", "ViT Huge")


if __name__ == "__main__":
    main()
