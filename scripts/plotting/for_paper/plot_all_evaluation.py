import os
from pathlib import Path
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments"

# adding a fixed color palette to each experiments, for consistency in plotting the legends
PALETTE = {"ais": "#045275", "amg": "#089099", "point": "#7CCBA2", "i_p": "#FCDE9C", "box": "#F0746E", "i_b": "#7C1D6F"}


def gather_all_results(dataset, modality, model_type):
    res_list_per_dataset = []
    for experiment_dir in glob(os.path.join(EXPERIMENT_ROOT, "new_models", "*", modality, dataset)):
        experiment_name = os.path.split(experiment_dir[:experiment_dir.find(f"/{modality}")])[-1]

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
                        "name": experiment_name,
                        "type": Path(result_path).stem if len(setting_name) == 3 else "ais",
                        "results": res.iloc[0]["msa"]
                    }, index=[i]
                )
            else:  # saving results from iterative prompting
                prompt_name = Path(result_path).stem.split("_")[-1]
                res_df = pd.concat(
                    [
                        pd.DataFrame(
                            {"name": experiment_name, "type": prompt_name, "results": res.iloc[0]["msa"]},
                            index=[i]
                        ),
                        pd.DataFrame(
                            {"name": experiment_name, "type": f"i_{prompt_name[0]}", "results": res.iloc[-1]["msa"]},
                            index=[i]
                        )
                    ], ignore_index=True
                )
            res_list_per_experiment.append(res_df)
        res_df_per_experiment = pd.concat(res_list_per_experiment, ignore_index=True)
        res_list_per_dataset.append(res_df_per_experiment)

    res_df_per_dataset = pd.concat(res_list_per_dataset)
    return res_df_per_dataset


def get_benchmark_results(dataset_name, benchmark_name, benchmark_choice):
    filename = f"{benchmark_name}-{benchmark_choice}.csv"
    if benchmark_name == "cellpose":
        filename = f"{benchmark_name}-{benchmark_choice}.csv"
    else:
        filename = f"{benchmark_name}.csv"

    if dataset_name == "plantseg_root":
        _splits = dataset_name.split("_")
        dataset_name = f"{_splits[0]}/{_splits[1]}"

    res_path = os.path.join(
        EXPERIMENT_ROOT, "benchmarking", benchmark_name, dataset_name, "results", filename
    )
    res = pd.read_csv(res_path)
    return res["msa"][0]


def get_barplots(ax, dataset_name, modality, model_type, benchmark_choice=None):
    plt.rcParams["hatch.linewidth"] = 1.5
    res_df = gather_all_results(dataset_name, modality, model_type)
    sns.barplot(x="name", y="results", hue="type", data=res_df, ax=ax, palette=PALETTE, hue_order=PALETTE.keys())
    lines, labels = ax.get_legend_handles_labels()
    for line, label in zip(lines, labels):
        if label == "ais":
            for k in range(len(line)):
                line.patches[k].set_hatch('///')
                line.patches[k].set_edgecolor('white')

    ax.set(xlabel=None, ylabel=None)
    ax.legend(title="Settings", bbox_to_anchor=(1, 1))
    ax.title.set_text(dataset_name)

    if dataset_name != "ctc" and modality != "em":  # HACK: as we don't have mitonet results now
        benchmark_name = "cellpose" if modality == "lm" else "mitonet"
        benchmark_res = get_benchmark_results(dataset_name, benchmark_name, benchmark_choice)
        ax.axhline(y=benchmark_res, label=benchmark_name, color="#E31A1C")


def plot_evaluation_for_lm_datasets(model_type):
    modality = "lm"
    fig, ax = plt.subplots(3, 3, figsize=(20, 15))

    # choices:
    # "livecell", "tissuenet", "deepbacs", "covid_if", "plantseg_root", "hpa",
    # "ctc", "plantseg_ovules", "neurips-cell-seg", "lizard", "mouse-embryo"

    get_barplots(ax[0, 0], "livecell", modality, model_type, benchmark_choice="livecell")
    get_barplots(ax[0, 1], "deepbacs", modality, model_type, benchmark_choice="cyto")
    get_barplots(ax[0, 2], "tissuenet", modality, model_type, benchmark_choice="cyto")
    get_barplots(ax[1, 0], "plantseg_root", modality, model_type, benchmark_choice="cyto")
    get_barplots(ax[1, 1], "covid_if", modality, model_type, benchmark_choice="cyto")
    get_barplots(ax[1, 2], "neurips-cell-seg", modality, model_type, benchmark_choice="cyto")
    get_barplots(ax[2, 0], "ctc", modality, model_type, benchmark_choice="cyto")
    get_barplots(ax[2, 1], "lizard", modality, model_type, benchmark_choice="cyto")
    get_barplots(ax[2, 2], "hpa", modality, model_type, benchmark_choice="cyto")

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

    fig.text(0.5, 0.01, 'Models', ha='center', fontdict={"size": 22})
    fig.text(0.01, 0.5, 'Segmentation Quality', va='center', rotation='vertical', fontdict={"size": 22})

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.95, left=0.05, bottom=0.05)
    fig.suptitle("Light Microscopy", fontsize=20)
    plt.savefig(f"lm_{model_type}_evaluation.png", transparent=True)
    plt.close()


def plot_evaluation_for_em_datasets(model_type):
    modality = "em"
    fig, ax = plt.subplots(3, 3, figsize=(20, 15))

    # choices:
    # for mito-nuc
    #    "mitoem/rat", "mitoem/human", "platynereis/nuclei", "mitolab/c_elegans", "mitolab/fly_brain",
    #    "mitolab/glycolytic_muscle", "mitolab/hela_cell", "mitolab/lucchi_pp", "mitolab/salivary_gland",
    #    "mitolab/tem", "lucchi", "nuc-mm/mouse", "nuc-mm/zebrafish", "uro_cell", "sponge_em",

    # for boundaries
    #    "platynereis/cilia", "cremi", "platynereis/cells", "axondeepseg", "snemi", "isbi"

    get_barplots(ax[0, 0], "mitoem/rat", modality, model_type)
    get_barplots(ax[0, 1], "mitoem/human", modality, model_type)
    get_barplots(ax[0, 2], "platynereis/nuclei", modality, model_type)
    get_barplots(ax[1, 0], "lucchi", modality, model_type)
    get_barplots(ax[1, 1], "mitolab/fly_brain", modality, model_type)
    get_barplots(ax[1, 2], "uro_cell", modality, model_type)
    get_barplots(ax[2, 0], "nuc-mm/mouse", modality, model_type)
    get_barplots(ax[2, 1], "sponge_em", modality, model_type)
    get_barplots(ax[2, 2], "mitolab/c_elegans", modality, model_type)

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

    fig.text(0.5, 0.01, 'Models', ha='center', fontdict={"size": 22})
    fig.text(0.01, 0.5, 'Segmentation Quality', va='center', rotation='vertical', fontdict={"size": 22})

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.95, left=0.05, bottom=0.05)
    fig.suptitle("Electron Microscopy", fontsize=20)
    plt.savefig(f"em_{model_type}_evaluation.png", transparent=True)
    plt.close()


def main():
    plot_evaluation_for_lm_datasets("vit_h")
    plot_evaluation_for_em_datasets("vit_h")


if __name__ == "__main__":
    main()
