import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_all_evaluation import EXPERIMENT_ROOT


TOP_BAR_COLOR, BOTTOM_BAR_COLOR = "#F0746E", "#01B3B7"

MODEL_CHOICE = "vit_l"
ALL_MODELS = ["vit_t", "vit_b", "vit_l", "vit_h"]
COMPARE_WITH = "specialist"
MODEL_NAME_MAP = {
    "vit_t": "ViT Tiny",
    "vit_b": "ViT Base",
    "vit_l": "ViT Large",
    "vit_h": "ViT Huge"
}
FIG_ASPECT = (30, 15)

# increase the font size globally
plt.rcParams.update({'font.size': 24})


def gather_livecell_results(model_type, experiment_name, benchmark_choice, result_type="default"):
    if result_type == "default":
        sub_dir_experiment = "new_models/v2"
    elif result_type == "with_logits":
        sub_dir_experiment = "new_models/test/input_logits"
    elif result_type.startswith("run"):
        sub_dir_experiment = f"new_models/test/{result_type}"

    result_paths = glob(
        os.path.join(
            EXPERIMENT_ROOT, sub_dir_experiment, experiment_name, "lm", "livecell", model_type, "results", "*"
        )
    )

    amg_score, ais_score, ib_score, ip_score = None, None, None, None
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
            [{"iteration": idx, "name": "point", "result": ip}]
        ) for idx, ip in enumerate(ip_score)
    ], ignore_index=True)

    ib_score = pd.concat([
        pd.DataFrame(
            [{"iteration": idx, "name": "box", "result": ib}]
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
    if "error" in ib_data:
        ax.errorbar(
            x=ib_data['iteration'], y=ib_data['result'], yerr=ib_data['error'], fmt='none', c='black', capsize=10
        )

    # NOTE: this is the snippet which creates hatches on the iterative prompting starting with box.
    # all_containers = ax.containers[-1]
    # for k in range(len(all_containers)):
    #     ax.patches[k].set_hatch('//')
    #     ax.patches[k].set_edgecolor('k')

    sns.barplot(x="iteration", y="result", hue="name", data=ip_data, ax=ax, palette=[BOTTOM_BAR_COLOR])
    if "error" in ip_data:
        ax.errorbar(
            x=ip_data['iteration'], y=ip_data['result'], yerr=ip_data['error'], fmt='none', c='black', capsize=10
        )
    ax.set(xlabel=None, ylabel=None)
    ax.legend(title="Settings", bbox_to_anchor=(1, 1))
    ax.set_title(name, fontweight="bold")

    if amg is not None:
        ax.axhline(y=amg, label="amg", color="#DC3977")
    if ais is not None:
        ax.axhline(y=ais, label="ais", color="#E19951")
    ax.axhline(y=cellpose, label="cellpose", color="#5454DA")


def plot_for_livecell(benchmark_choice, results_with_logits):
    result_type = "with_logits" if results_with_logits else "default"

    fig, ax = plt.subplots(1, 2, figsize=FIG_ASPECT, sharex="col", sharey="row")
    amg_vanilla, _, ib_vanilla, ip_vanilla, cellpose_res = gather_livecell_results(
        MODEL_CHOICE, "vanilla", benchmark_choice, result_type
    )
    get_barplots("Default SAM", ax[0], ib_vanilla, ip_vanilla, amg_vanilla, cellpose_res)

    amg, ais, ib, ip, cellpose_res = gather_livecell_results(
        MODEL_CHOICE, COMPARE_WITH, benchmark_choice, result_type
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

    ax.set_yticks(np.linspace(0.1, 0.8, 8))
    for ax in fig.axes:
        ax.set_xticks(np.linspace(1, 7, 7))

    plt.text(
        x=0.8, y=1.12, s=" X-Axis: Iterative Prompting \n Y-Axis: Segmentation Accuracy ", ha='center', va='center',
        transform=plt.gca().transAxes, bbox={"facecolor": "None", "edgecolor": "#D6D6D6", "boxstyle": "round"}
    )

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.78, right=0.95, left=0.13, bottom=0.05)
    _path = "livecell_with_logits.svg" if results_with_logits else "livecell.svg"
    plt.savefig(_path)
    plt.savefig(Path(_path).with_suffix(".pdf"))
    plt.close()


def plot_all_livecell(benchmark_choice, model_type):
    amg_vanilla_list, ib_vanilla_list, ip_vanilla_list = [], [], []
    cellpose_res_list = []
    amg_list, ais_list, ib_list, ip_list = [], [], [], []
    for current_run in range(1, 6):
        fig, ax = plt.subplots(1, 2, figsize=FIG_ASPECT, sharex="col", sharey="row")
        amg_vanilla, _, ib_vanilla, ip_vanilla, cellpose_res = gather_livecell_results(
            model_type, "vanilla", benchmark_choice, f"run_{current_run}"
        )
        amg, ais, ib, ip, cellpose_res = gather_livecell_results(
            model_type, COMPARE_WITH, benchmark_choice, f"run_{current_run}"
        )
        amg_vanilla_list.append(amg_vanilla)
        ib_vanilla_list.append(ib_vanilla)
        ip_vanilla_list.append(ip_vanilla)
        cellpose_res_list.append(cellpose_res)
        amg_list.append(amg)
        ais_list.append(ais)
        ib_list.append(ib)
        ip_list.append(ip)

    def _create_res_from_list(res_list):
        tmp_df = pd.concat(res_list)
        this_res = []
        for idx in tmp_df.index.unique():
            this_res.append(
                pd.DataFrame.from_dict(
                    [{
                        "iteration": idx,
                        "name": tmp_df.iloc[idx]["name"],
                        "result": tmp_df.loc[idx]["result"].mean(),
                        "error": tmp_df.loc[idx]["result"].std()
                    }]
                )
            )
        return pd.concat(this_res, ignore_index=True)

    ib_vanilla_res = _create_res_from_list(ib_vanilla_list)
    ip_vanilla_res = _create_res_from_list(ip_vanilla_list)
    ib_res = _create_res_from_list(ib_list)
    ip_res = _create_res_from_list(ip_list)

    ais_res = np.mean([res for res in ais_list if res is not None])
    cellpose_res = np.mean([res for res in cellpose_res_list if res is not None])
    amg_vanilla_res = np.mean([res for res in amg_vanilla_list if res is not None])
    amg_res = np.mean([res for res in amg_list if res is not None])

    get_barplots("Default SAM", ax[0], ib_vanilla_res, ip_vanilla_res, amg_vanilla_res, cellpose_res)
    get_barplots("Finetuned SAM", ax[1], ib_res, ip_res, amg_res, cellpose_res, ais_res)

    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    ax.set_yticks(np.linspace(0.1, 0.8, 8))
    for ax in fig.axes:
        ax.set_xticks(np.linspace(1, 7, 7))

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.865, right=0.95, left=0.075, bottom=0.05)
    fig.suptitle(MODEL_NAME_MAP[model_type], fontsize=26, x=0.515, y=0.97)
    _path = f"livecell_supplementary_{model_type}.svg"
    plt.savefig(_path)
    plt.savefig(Path(_path).with_suffix(".pdf"))
    plt.close()


def main():
    plot_for_livecell(benchmark_choice="livecell", results_with_logits=False)
    plot_for_livecell(benchmark_choice="livecell", results_with_logits=True)

    plot_all_livecell(benchmark_choice="livecell", model_type="vit_t")
    plot_all_livecell(benchmark_choice="livecell", model_type="vit_b")
    plot_all_livecell(benchmark_choice="livecell", model_type="vit_l")
    plot_all_livecell(benchmark_choice="livecell", model_type="vit_h")


if __name__ == "__main__":
    main()
