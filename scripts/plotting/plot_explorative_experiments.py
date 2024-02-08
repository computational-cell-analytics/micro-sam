import os
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/test/"


def get_partial_finetuning_plots():
    exp_root = os.path.join(EXPERIMENT_ROOT, "partial-finetuning")

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

        amg = pd.read_csv(os.path.join(experiment_folder, "amg.csv"))
        ais = pd.read_csv(os.path.join(experiment_folder, "instance_segmentation_with_decoder.csv"))
        itp_p = pd.read_csv(os.path.join(experiment_folder, "iterative_prompts_start_point.csv"))
        itp_b = pd.read_csv(os.path.join(experiment_folder, "iterative_prompts_start_box.csv"))

        res = [
            {"name": _plot_object, "type": "amg", "results": amg.iloc[0]["msa"]},
            {"name": _plot_object, "type": "ais", "results": ais.iloc[0]["msa"]},
            {"name": _plot_object, "type": "1pn0", "results": itp_p.iloc[0]["msa"]},
            {"name": _plot_object, "type": "box", "results": itp_b.iloc[0]["msa"]},
            {"name": _plot_object, "type": r"i$_{p}$", "results": itp_p.iloc[-1]["msa"]},
            {"name": _plot_object, "type": r"i$_{b}$", "results": itp_b.iloc[-1]["msa"]}
        ]
        res = [pd.DataFrame(_res, index=[i]) for i, _res in enumerate(res)]
        res = pd.concat(res, ignore_index=True)
        res_list.append(res)

    res_df = pd.concat(res_list)

    plt.figure(figsize=(10, 10))

    ax = sns.barplot(x="name", y="results", hue="type", data=res_df)
    ax.set(xlabel="Finetuned Parts", ylabel="Segmentation Quality")
    plt.legend(title="Settings", bbox_to_anchor=(1, 1))
    plt.title("Partial Finetuning")

    save_path = "partial_finetuning.png"
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")


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


def get_n_objects_plots(max_objects=45):
    exp_root = os.path.join(EXPERIMENT_ROOT, "n_objects_per_batch")

    amg_list, ais_list, _1p_list, _box_list, _itp_p_last_list, _itp_b_last_list = [], [], [], [], [], []
    all_n_objects = [*range(1, max_objects+1)]
    for i in all_n_objects:
        experiment_folder = os.path.join(exp_root, f"{i}", "results")

        amg = pd.read_csv(os.path.join(experiment_folder, "amg.csv"))
        ais = pd.read_csv(os.path.join(experiment_folder, "instance_segmentation_with_decoder.csv"))
        itp_p = pd.read_csv(os.path.join(experiment_folder, "iterative_prompts_start_point.csv"))
        itp_b = pd.read_csv(os.path.join(experiment_folder, "iterative_prompts_start_box.csv"))

        amg_list.append(amg.iloc[0]["msa"])
        ais_list.append(ais.iloc[0]["msa"])
        _1p_list.append(itp_p.iloc[0]["msa"])
        _box_list.append(itp_b.iloc[0]["msa"])
        _itp_p_last_list.append(itp_p.iloc[-1]["msa"])
        _itp_b_last_list.append(itp_b.iloc[-1]["msa"])

    plt.figure(figsize=(10, 10))

    res = {
        "name": list(range(1, len(amg_list) + 1)),
        "amg": amg_list,
        "ais": ais_list,
        "1pn0": _1p_list,
        "box": _box_list,
        r"i$_{p}$": _itp_p_last_list,
        r"i$_{b}$": _itp_b_last_list
    }

    res_df = pd.DataFrame.from_dict(res)

    ax = sns.lineplot(x="name", y="value", hue="variable", data=pd.melt(res_df, ["name"]))
    ax.set(xlabel="n_objects", ylabel="Segmentation Quality")

    plt.legend(title="Settings", bbox_to_anchor=(1, 1))
    plt.title("Number of Objects for Finetuning")

    save_path = "n_objects.png"
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")


def main():
    get_n_objects_plots()
    get_partial_finetuning_plots()


if __name__ == "__main__":
    main()
