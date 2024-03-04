import os
from pathlib import Path
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/v2"

MODELS = ["vit_t", "vit_b", "vit_l", "vit_h"]


def plot_evaluation_for_datasets(dataset_name, modality, all_models):
    experiment_dirs = glob(os.path.join(EXPERIMENT_ROOT, "*", modality, dataset_name))

    # adding a fixed color palette to each experiments, for consistency in plotting the legends
    palette = {"amg": "C0", "ais": "C1", "box": "C2", "i_b": "C3", "point": "C4", "i_p": "C5"}

    fig, ax = plt.subplots(1, len(experiment_dirs), figsize=(20, 10), sharex="col", sharey="row")

    for idx, experiment_dir in enumerate(experiment_dirs):
        res_df_per_model = []
        for model_type in all_models:
            res_list_per_experiment = []
            for i, result_path in enumerate(sorted(glob(os.path.join(experiment_dir, model_type, "results", "*")))):
                # avoid using the grid-search parameters' files
                if os.path.split(result_path)[-1].startswith("grid_search_"):
                    continue

                res = pd.read_csv(result_path)
                setting_name = Path(result_path).stem
                if setting_name == "amg" or setting_name.startswith("instance"):  # saving results from amg or ais
                    res_df = pd.DataFrame(
                        {
                            "name": model_type,
                            "type": Path(result_path).stem if len(setting_name) == 3 else "ais",
                            "results": res.iloc[0]["msa"]
                        }, index=[i]
                    )
                else:  # saving results from iterative prompting
                    prompt_name = Path(result_path).stem.split("_")[-1]
                    res_df = pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "name": model_type,
                                    "type": prompt_name,
                                    "results": res.iloc[0]["msa"]
                                }, index=[i]
                            ),
                            pd.DataFrame(
                                {
                                    "name": model_type,
                                    "type": f"i_{prompt_name[0]}",
                                    "results": res.iloc[-1]["msa"]
                                }, index=[i]
                            )
                        ]
                    )

                res_list_per_experiment.append(res_df)

            res_df_per_experiment = pd.concat(res_list_per_experiment, ignore_index=True)
            res_df_per_model.append(res_df_per_experiment)

        res_df_per_dataset = pd.concat(res_df_per_model)
        container = sns.barplot(
            x="name", y="results", hue="type", data=res_df_per_dataset, ax=ax[idx], palette=palette
        )
        ax[idx].set(xlabel="Model Size", ylabel="Segmentation Quality")
        ax[idx].legend(title="Settings", bbox_to_anchor=(1, 1))

        # adding the numnbers over the barplots
        for j in container.containers:
            container.bar_label(j, fmt='%.2f')

        # titles for each subplot
        subplot_name = os.path.split(experiment_dir[:experiment_dir.find(f"/{modality}")])[-1]
        ax[idx].title.set_text(subplot_name)

    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    fig.legend(all_lines, all_labels)
    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.95)
    fig.suptitle(dataset_name, fontsize=20)

    save_path = f"figures/v2/{dataset_name}.png"

    try:
        plt.savefig(save_path)
    except FileNotFoundError:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        plt.savefig(save_path)

    plt.close()
    print(f"Plot saved at {save_path}")


def for_all_em():
    all_datasets = [
        "mitoem/rat", "mitoem/human", "platynereis/nuclei", "mitolab/c_elegans", "mitolab/fly_brain",
        "mitolab/glycolytic_muscle", "mitolab/hela_cell", "mitolab/lucchi_pp", "mitolab/salivary_gland",
        "mitolab/tem", "lucchi", "nuc_mm/mouse", "nuc_mm/zebrafish", "uro_cell", "sponge_em",
        "platynereis/cilia", "cremi", "platynereis/cells", "axondeepseg", "snemi", "isbi"
    ]
    for dataset_name in all_datasets:
        plot_evaluation_for_datasets(dataset_name, "em", all_models=MODELS)


def for_all_lm():
    all_datasets = [
        "livecell", "tissuenet", "deepbacs", "plantseg/root", "covid_if", "plantseg/ovules",
        "hpa", "neurips-cell-seg", "lizard", "mouse-embryo"
    ]
    for dataset_name in all_datasets:
        plot_evaluation_for_datasets(dataset_name, "lm", all_models=MODELS)


def main():
    for_all_lm()
    for_all_em()


if __name__ == "__main__":
    main()
