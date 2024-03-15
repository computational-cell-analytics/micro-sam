import os
from glob import glob
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_all_evaluation import EXPERIMENT_ROOT


TOP_BAR_COLOR, BOTTOM_BAR_COLOR = "#F0746E", "#01B3B7"

MODEL_CHOICE = "vit_l"
COMPARE_WITH = "specialist"


def gather_livecell_results(model_type, experiment_name, benchmark_choice, results_with_logits):
    if results_with_logits:
        sub_dir_experiment = "new_models/test/input_logits"
    else:
        sub_dir_experiment = "new_models/v2"

    result_paths = glob(
        os.path.join(
            EXPERIMENT_ROOT, sub_dir_experiment, experiment_name, "lm", "livecell", model_type, "results", "*"
        )
    )

    ais_score = None
    for result_path in sorted(result_paths):
        if os.path.split(result_path)[-1].startswith("grid_search_"):
            continue

        res = pd.read_csv(result_path)
        setting_name = Path(result_path).stem
        if setting_name == "amg":
            amg_score = res.iloc[0]["msa"]
        elif setting_name.startswith("instance"):
            ais_score = res.iloc[0]["msa"]
        elif setting_name.endswith("box"):
            iterative_prompting_box = res["msa"]
            ib_score = [ibs for ibs in iterative_prompting_box]
        elif setting_name.endswith("point"):
            iterative_prompting_point = res["msa"]
            ip_score = [ips for ips in iterative_prompting_point]

    ip_score = pd.concat([
        pd.DataFrame(
            [{"iteration": idx + 1, "name": "point", "result": ip}]
        ) for idx, ip in enumerate(ip_score)
    ], ignore_index=True)

    ib_score = pd.concat([
        pd.DataFrame(
            [{"iteration": idx + 1, "name": "box", "result": ib}]
        ) for idx, ib in enumerate(ib_score)
    ], ignore_index=True)

    # let's get benchmark results
    cellpose_res = pd.read_csv(
        os.path.join(
            EXPERIMENT_ROOT, "benchmarking", "cellpose", "livecell", "results", f"cellpose-{benchmark_choice}.csv"
        )
    )["msa"][0]

    return amg_score, ais_score, ib_score, ip_score, cellpose_res


def get_barplots(name, ax, ib_data, ip_data, amg, cellpose, ais=None):
    sns.barplot(x="iteration", y="result", hue="name", data=ib_data, ax=ax, palette=[TOP_BAR_COLOR])
    # NOTE: this is the snippet which creates hatches on the iterative prompting starting with box.
    # all_containers = ax.containers[-1]
    # for k in range(len(all_containers)):
    #     ax.patches[k].set_hatch('//')
    #     ax.patches[k].set_edgecolor('k')

    sns.barplot(x="iteration", y="result", hue="name", data=ip_data, ax=ax, palette=[BOTTOM_BAR_COLOR])
    ax.set(xlabel=None, ylabel=None)
    ax.legend(title="Settings", bbox_to_anchor=(1, 1))
    ax.set_title(name, fontsize=13, fontweight="bold")

    ax.axhline(y=amg, label="amg", color="#DC3977")
    if ais is not None:
        ax.axhline(y=ais, label="ais", color="#E19951")
    ax.axhline(y=cellpose, label="cellpose", color="#5454DA")


def plot_for_livecell(benchmark_choice, results_with_logits):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex="col", sharey="row")
    amg_vanilla, _, ib_vanilla, ip_vanilla, cellpose_res = gather_livecell_results(
        MODEL_CHOICE, "vanilla", benchmark_choice, results_with_logits
    )
    get_barplots("Default SAM", ax[0], ib_vanilla, ip_vanilla, amg_vanilla, cellpose_res)

    amg, ais, ib, ip, cellpose_res = gather_livecell_results(
        MODEL_CHOICE, COMPARE_WITH, benchmark_choice, results_with_logits
    )
    get_barplots("Finetuned SAM", ax[1], ib, ip, amg, cellpose_res, ais)

    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    fig.legend(all_lines, all_labels, loc="upper left")

    plt.text(
        x=0.872, y=1.05, s=" X-Axis: Models \n Y-Axis: Segmentation Quality ", ha='center', va='center',
        transform=plt.gca().transAxes, bbox={"facecolor": "None", "edgecolor": "#045275", "boxstyle": "round"}
    )

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.865, right=0.95, left=0.075, bottom=0.05)
    fig.suptitle("LIVECell", fontsize=26, x=0.515, y=0.95)
    plt.savefig("livecell.png")
    plt.close()


def main(args):
    plot_for_livecell(
        benchmark_choice="livecell",
        results_with_logits=args.use_masks
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_masks", action="store_true")
    args = parser.parse_args()
    main(args)
