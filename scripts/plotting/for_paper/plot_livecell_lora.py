import os
from glob import glob
from natsort import natsorted

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = "/media/anwai/ANWAI/micro-sam/for_revision_2/livecell_results"

PALETTE = {
    "AIS": "#045275",
    "AMG": "#FCDE9C",
    "Point": "#7CCBA2",
    r"I$_{P}$": "#089099",
    "Box": "#90477F",
    r"I$_{B}$": "#F0746E"
}

NAME_MAPS = {
    "vanilla": "Default",
    "lora_1": "LoRA (Rank 1)",
    "lora_2": "LoRA (Rank 2)",
    "lora_4": "LoRA (Rank 4)",
    "lora_8": "LoRA (Rank 8)",
    "lora_16": "LoRA (Rank 16)",
    "full_ft": "Full Finetuning",
}

plt.rcParams.update({"font.size": 30})


def _get_livecell_lora_data():
    # experiments from carolin on livecell lora
    all_results = []
    all_experiments_dir = natsorted(glob(os.path.join(ROOT, "*")))
    for experiment_dir in all_experiments_dir:
        experiment_name = os.path.split(experiment_dir)[-1]

        ais = pd.read_csv(os.path.join(experiment_dir, "results", "instance_segmentation_with_decoder.csv"))
        amg = pd.read_csv(os.path.join(experiment_dir, "results", "amg.csv"))
        ip = pd.read_csv(os.path.join(experiment_dir, "results", "iterative_prompts_start_point.csv"))
        ib = pd.read_csv(os.path.join(experiment_dir, "results", "iterative_prompts_start_box.csv"))

        res = {
            "experiment": experiment_name,
            "AIS": ais.iloc[0]["msa"],
            "AMG": amg.iloc[0]["msa"],
            "Point": ip.iloc[0]["msa"],
            "Box": ib.iloc[0]["msa"],
            r"I$_{P}$": ip.iloc[-1]["msa"],
            r"I$_{B}$": ib.iloc[-1]["msa"]
        }
        all_results.append(pd.DataFrame.from_dict([res]))

    # NOTE: this is done to plot "full_finetuning" results at the end of the lineplot.
    all_results = all_results[1:] + [all_results[0]]

    return all_results


def _get_vanilla_and_finetuned_results():
    all_results = _get_livecell_lora_data()

    def _get_results(method):
        assert method in ["vanilla", "specialist"]
        root_dir = f"/home/anwai/results/micro-sam/livecell/{method}/vit_b"

        amg = pd.read_csv(os.path.join(root_dir, "amg.csv"))
        ip = pd.read_csv(os.path.join(root_dir, "iterative_prompts_start_point.csv"))
        ib = pd.read_csv(os.path.join(root_dir, "iterative_prompts_start_box.csv"))

        have_ais = False
        if method == "specialist":
            ais = pd.read_csv(os.path.join(root_dir, "instance_segmentation_with_decoder.csv"))
            have_ais = True

        res = {
            "experiment": method,
            "AMG": amg.iloc[0]["msa"],
            "Point": ip.iloc[0]["msa"],
            "Box": ib.iloc[0]["msa"],
            r"I$_{P}$": ip.iloc[-1]["msa"],
            r"I$_{B}$": ib.iloc[-1]["msa"]
        }
        if have_ais:
            res["AIS"] = ais.iloc[0]["msa"]

        return pd.DataFrame.from_dict([res])

    all_results.insert(0, _get_results("vanilla"))
    res_df = pd.concat(all_results, ignore_index=True)
    return res_df


def _get_plots():
    plt.figure(figsize=(30, 15))
    res = _get_vanilla_and_finetuned_results()
    ax = sns.lineplot(
        data=pd.melt(res, "experiment"),
        x="experiment", y="value", hue="variable", marker="d",
        palette=PALETTE, markersize=20, linewidth=3,
    )

    ax.set_yticks(np.linspace(0.1, 1, 10)[:-1])

    plt.ylabel("Mean Segmentation Accuracy", labelpad=10, fontweight="bold")
    plt.xlabel("Finetuning Strategy", labelpad=10, fontweight="bold")
    plt.legend(loc="lower center", ncol=7)

    plt.xticks(np.arange(7), [exp_name for exp_name in NAME_MAPS.values()])

    plt.title("")
    plt.tight_layout()
    plt.savefig("s14_c.png")
    plt.savefig("s14_c.svg")
    plt.savefig("s14_c.pdf")


def main():
    _get_plots()


if __name__ == "__main__":
    main()
