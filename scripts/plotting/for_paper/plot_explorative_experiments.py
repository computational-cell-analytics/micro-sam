import os
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/test/"

PALETTE = {
    "ais": "#045275",
    "amg": "#089099",
    "point": "#7CCBA2",
    r"i$_{p}$": "#FCDE9C",
    "box": "#F0746E",
    r"i$_{b}$": "#90477F"
}


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
    partial_finetuning_combinations = ["all", "PE, MD", "IE, MD", "IE, PE", "MD", "PE", "IE"]

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
            {"name": _plot_object, "type": "amg", "results": amg},
            {"name": _plot_object, "type": "ais", "results": ais},
            {"name": _plot_object, "type": "point", "results": _1p},
            {"name": _plot_object, "type": "box", "results": _box},
            {"name": _plot_object, "type": r"i$_{p}$", "results": _itp_p},
            {"name": _plot_object, "type": r"i$_{b}$", "results": _itp_b}
        ]
        res = [pd.DataFrame(_res, index=[i]) for i, _res in enumerate(res)]
        res = pd.concat(res, ignore_index=True)
        res_list.append(res)

    res_df = pd.concat(res_list)

    plt.figure(figsize=(20, 10))

    ax = sns.barplot(x="name", y="results", hue="type", data=res_df, palette=PALETTE, hue_order=PALETTE.keys())

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.3f')

    ax.set(xlabel="Finetuned Parts", ylabel="Segmentation Quality")
    plt.legend(title="Settings", loc="upper left")
    plt.title("Partial Finetuning")

    save_path = "livecell_vit_l_partial_finetuning.png"
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")


def get_n_objects_plots(max_objects=45):
    exp_root = os.path.join(EXPERIMENT_ROOT, "n_objects_per_batch")

    amg_list, ais_list, _1p_list, _box_list, _itp_p_last_list, _itp_b_last_list = [], [], [], [], [], []
    all_n_objects = [*range(1, max_objects+1)]
    for i in all_n_objects:
        experiment_folder = os.path.join(exp_root, f"{i}", "results")
        amg, ais, _1p, _box, _itp_p, _itp_b = _get_results(experiment_folder)

        amg_list.append(amg)
        ais_list.append(ais)
        _1p_list.append(_1p)
        _box_list.append(_box)
        _itp_p_last_list.append(_itp_p)
        _itp_b_last_list.append(_itp_b)

    plt.figure(figsize=(10, 10))

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

    ax = sns.lineplot(x="name", y="value", hue="variable", data=pd.melt(res_df, ["name"]))
    ax.set(xlabel="n_objects", ylabel="Segmentation Quality")

    plt.legend(title="Settings", loc="upper left")
    plt.title("Number of Objects for Finetuning")

    save_path = "livecell_vit_b_n_objects.png"
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")


def main():
    get_partial_finetuning_plots()
    get_n_objects_plots()


if __name__ == "__main__":
    main()
