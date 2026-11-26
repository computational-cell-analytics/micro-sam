import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/test/"

PALETTE = {
    "AIS": "#045275",
    "AMG": "#FCDE9C",
    "Point": "#7CCBA2",
    r"I$_{P}$": "#089099",
    "Box": "#90477F",
    r"I$_{B}$": "#F0746E"
}

plt.rcParams.update({'font.size': 30})


def _open_csv_file(csv_path):
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        return None


def _get_results(experiment_folder):
    amg = _open_csv_file(os.path.join(experiment_folder, "amg.csv"))
    ais = _open_csv_file(os.path.join(experiment_folder, "instance_segmentation_with_decoder.csv"))
    itp_p = _open_csv_file(os.path.join(experiment_folder, "iterative_prompts_start_point.csv"))
    itp_b = _open_csv_file(os.path.join(experiment_folder, "iterative_prompts_start_box.csv"))

    amg = None if amg is None else amg.iloc[0]["msa"]
    ais = None if ais is None else ais.iloc[0]["msa"]
    _1p = None if itp_p is None else itp_p.iloc[0]["msa"]
    _box = None if itp_b is None else itp_b.iloc[0]["msa"]
    _itp_p = None if itp_p is None else itp_p.iloc[-1]["msa"]
    _itp_b = None if itp_b is None else itp_b.iloc[-1]["msa"]

    return amg, ais, _1p, _box, _itp_p, _itp_b


def get_partial_finetuning_combinations():
    # let's get all combinations need for the freezing backbone experiments
    backbone_combinations = ["image_encoder", "prompt_encoder", "mask_decoder"]

    all_combinations = []
    for i in range(len(backbone_combinations)):
        _one_set = itertools.combinations(backbone_combinations, r=i)
        for _per_combination in _one_set:
            if len(_per_combination) == 0:
                all_combinations.append(None)
            else:
                all_combinations.append(list(_per_combination))

    return all_combinations


def get_partial_finetuning_plots():
    exp_root = os.path.join(EXPERIMENT_ROOT, "freezing-livecell")

    all_combinations = get_partial_finetuning_combinations()

    # custom naming for plotting purpose
    partial_finetuning_combinations = [
        r"${all}$",
        "Prompt Encoder,\nMask Decoder",
        "Image Encoder,\nMask Decoder",
        "Image Encoder,\nPrompt Encoder",
        "Mask Decoder", "Prompt Encoder", "Image Encoder"
    ]

    res_list = []

    for _setup, _plot_object in zip(all_combinations, partial_finetuning_combinations):
        if isinstance(_setup, list):
            experiment_folder = os.path.join(exp_root, "freeze-")
            for _name in _setup:
                experiment_folder += f"{_name}-"
            experiment_folder = experiment_folder[:-1]
        else:
            experiment_folder = os.path.join(exp_root, f"freeze-{_setup}")

        experiment_folder = os.path.join(experiment_folder, "results")
        amg, ais, _1p, _box, _itp_p, _itp_b = _get_results(experiment_folder)

        res = [
            {"name": _plot_object, "type": "AMG", "results": amg},
            {"name": _plot_object, "type": "AIS", "results": ais},
            {"name": _plot_object, "type": "Point", "results": _1p},
            {"name": _plot_object, "type": "Box", "results": _box},
            {"name": _plot_object, "type": r"I$_{P}$", "results": _itp_p},
            {"name": _plot_object, "type": r"I$_{B}$", "results": _itp_b}
        ]
        res = [pd.DataFrame(_res, index=[i]) for i, _res in enumerate(res)]
        res = pd.concat(res, ignore_index=True)
        res_list.append(res)

    res_df = pd.concat(res_list)

    plt.figure(figsize=(30, 15))

    ax = sns.barplot(x="name", y="results", hue="type", data=res_df, palette=PALETTE, hue_order=PALETTE.keys())
    ax.set_yticks(np.linspace(0.1, 1, 10))
    lines, labels = ax.get_legend_handles_labels()
    for line, label in zip(lines, labels):
        if label == "AIS":
            for k in range(len(line)):
                line.patches[k].set_hatch('///')
                line.patches[k].set_edgecolor('white')

    plt.xlabel("Finetuned Parts (SAM)", labelpad=15, fontweight="bold")
    plt.ylabel("Mean Segmentation Accuracy", labelpad=10, fontweight="bold")
    plt.legend(loc="upper center", ncol=6)
    plt.tight_layout()

    # plt.subplots_adjust(top=0.9, right=0.95, left=0.15, bottom=0.1)

    save_path = "2_b.svg"
    plt.savefig(save_path)
    plt.savefig(Path(save_path).with_suffix(".pdf"))
    print(f"Plot saved at {save_path}")


def get_n_objects_plots(max_objects=45):
    exp_root = os.path.join(EXPERIMENT_ROOT, "n_objects_per_batch")

    amg_list, ais_list, _1p_list, _box_list, _itp_p_last_list, _itp_b_last_list = [], [], [], [], [], []
    all_n_objects = [*np.arange(0, max_objects+1)][1:]  # the first element is always 0, we don't want it
    for i in all_n_objects:
        experiment_folder = os.path.join(exp_root, f"{i}", "results")
        amg, ais, _1p, _box, _itp_p, _itp_b = _get_results(experiment_folder)

        amg_list.append(amg)
        ais_list.append(ais)
        _1p_list.append(_1p)
        _box_list.append(_box)
        _itp_p_last_list.append(_itp_p)
        _itp_b_last_list.append(_itp_b)

    plt.figure(figsize=(30, 30))

    res = {
        "name": list(range(1, len(amg_list) + 1)),
        "amg": amg_list,
        "ais": ais_list,
        "point": _1p_list,
        "box": _box_list,
        r"i$_{p}$": _itp_p_last_list,
        r"i$_{b}$": _itp_b_last_list
    }

    res_df = pd.DataFrame.from_dict(res)

    palette = sns.color_palette('viridis', 3)
    dark_palette = [(max(0, rgb[0] - 0.2), max(0, rgb[1] - 0.2), max(0, rgb[2] - 0.2)) for rgb in palette]

    ax = sns.stripplot(
        x="variable", y="value", hue="name", data=pd.melt(res_df, ["name"]),
        dodge=True, alpha=.5, palette='viridis', legend=None,
        edgecolor=dark_palette, linewidth=0.75, s=10
    )
    ax.set(xlabel=None, ylabel=None)

    norm = plt.Normalize(1, max_objects)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=plt.gca(), alpha=.5, aspect=20)
    ticks = np.arange(0, max_objects+1, 5)[1:]
    ticks = [1, *ticks]
    colorbar.set_ticks(ticks)

    def _get_all_interval_fills(name, color):
        interval_data = res_df.groupby('name')[name].agg(['mean', 'std'])
        interval_data['lower'] = interval_data['mean'].min()
        interval_data['upper'] = interval_data['mean'].max()
        ax.fill_between(
            x=np.linspace(-1, 7),
            y1=interval_data["lower"].iloc[0] - 0.005,
            y2=interval_data["upper"].iloc[0] + 0.005,
            color=color,
            alpha=0.1,
        )

    _get_all_interval_fills("amg", color="#440154")
    _get_all_interval_fills("ais", color="#440154")
    _get_all_interval_fills("point", color="#440154")
    _get_all_interval_fills("box", color="#440154")
    _get_all_interval_fills(r"i$_{p}$", color="#440154")
    _get_all_interval_fills(r"i$_{b}$", color="#440154")

    ax.set_xticks(np.arange(0, 6))
    ax.set_xticklabels(["AMG", "AIS", "Point", "Box", r"I$_{P}$", r"I$_{B}$"])

    ax.set_xlim(-0.5, len(res_df.columns[1:]) - 0.5)

    plt.suptitle(r"Number of Objects ${(Per}$ ${Image)}$", x=0.45, y=0.9)
    plt.xlabel("Inference Settings", labelpad=15, fontweight="bold")
    plt.ylabel("Mean Segmentation Accuracy", labelpad=30, fontweight="bold")

    save_path = "livecell_vit_b_n_objects.svg"
    plt.savefig(save_path)
    plt.savefig(Path(save_path).with_suffix(".pdf"))
    plt.close()
    print(f"Plot saved at {save_path}")


def main():
    # get_partial_finetuning_plots()
    get_n_objects_plots()


if __name__ == "__main__":
    main()
