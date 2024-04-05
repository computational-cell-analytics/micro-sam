import os
from pathlib import Path
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/"

# adding a fixed color palette to each experiments, for consistency in plotting the legends
PALETTE = {
    "ais": "#045275",
    "amg": "#089099",
    "point": "#7CCBA2",
    r"i$_{p}$": "#FCDE9C",
    "box": "#F0746E",
    r"i$_{b}$": "#90477F"
}

TITLE = {
    "livecell": "$\it{LIVECell}$",
    "tissuenet/one_chan": "$\it{TissueNet}$",
    "tissuenet/multi_chan": "$\it{TissueNet}$",
    "deepbacs": "$\it{DeepBacs}$",
    "covid_if": "Covid IF",
    "plantseg/root": "$\it{PlantSeg}$ $\it{(Root)}$",
    "hpa": "HPA",
    "ctc": "Cell Tracking Challenge",
    "plantseg/ovules": "PlantSeg (Ovules)",
    "neurips-cell-seg/tuning": "$\it{NeurIPS}$ $\it{CellSeg}$",
    "lizard": "Lizard",
    "mouse-embryo": "Mouse Embryo",
    "mitoem/rat": "$\it{MitoEM}$ $\it{(Rat)}$",
    "mitoem/human": "$\it{MitoEM}$ $\it{(Human)}$",
    "platynereis/nuclei": "$\it{Platynereis}$ $\it{(Nuclei)}$",
    "mitolab/c_elegans": "MitoLab (C. elegans)",
    "mitolab/fly_brain": "MitoLab (Fly Brain)",
    "mitolab/glycolytic_muscle": "MitoLab (Glycolytic Muscle)",
    "mitolab/hela_cell": "MitoLab (HeLa Cell)",
    "mitolab/lucchi_pp": "MitoLab (Lucchi++)",
    "mitolab/salivary_gland": "MitoLab (Salivary Gland: Rat)",
    "mitolab/tem": "MitoLab (TEM)",
    "lucchi": "Lucchi",
    "nuc_mm/mouse": "NucMM (Mouse)",
    "nuc_mm/zebrafish": "NucMM (Zebrafish)",
    "uro_cell": "UroCell",
    "sponge_em": "Sponge EM"
    }

FIG_ASPECT = (25, 22)

plt.rcParams.update({'font.size': 24})


def gather_all_results(dataset, modality, model_type):
    res_list_per_dataset = []
    for experiment_dir in glob(os.path.join(EXPERIMENT_ROOT, "new_models", "v2", "*", modality, dataset)):
        experiment_name = os.path.split(experiment_dir[:experiment_dir.find(f"/{modality}")])[-1]

        if experiment_name == "vanilla":  # easy fix to alter the name
            experiment_name = "default"

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
                            {
                                "name": experiment_name,
                                "type": r"i$_{p}$" if prompt_name[0] == "p" else r"i$_{b}$",
                                "results": res.iloc[-1]["msa"]
                            },
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
        if dataset_name == "plantseg_root":
            _splits = dataset_name.split("_")
            dataset_name = f"{_splits[0]}/{_splits[1]}"
        res_path = os.path.join(EXPERIMENT_ROOT, "benchmarking", benchmark_name, dataset_name, "results", filename)

    elif benchmark_name == "mitonet":
        dname_split = dataset_name.split("/")
        if len(dname_split) > 1:
            _name = f"{dname_split[0]}_{dname_split[1]}"
        else:
            _name = dataset_name
        filename = f"{benchmark_name}_{_name}.csv"
        res_path = os.path.join(EXPERIMENT_ROOT, "benchmarking", benchmark_name, filename)

    try:
        res = pd.read_csv(res_path)
        return res["msa"][0]
    except FileNotFoundError:
        return None


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
    ax.title.set_color("#212427")
    ax.set_title(TITLE[dataset_name], fontweight="bold")

    if dataset_name != "ctc":
        benchmark_name = "cellpose" if modality == "lm" else "mitonet"
        benchmark_res = get_benchmark_results(dataset_name, benchmark_name, benchmark_choice)
        if benchmark_res is not None:
            ax.axhline(y=benchmark_res, label=benchmark_name, color="#DC3977")


def _get_plot_postprocessing(fig, experiment_title, save_path):
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
        x=0.25, y=4.1, s=" X-Axis: Models \n Y-Axis: Segmentation Accuracy ", ha='left',
        transform=plt.gca().transAxes, bbox={"facecolor": "None", "edgecolor": "#D6D6D6", "boxstyle": "round"}
    )

    plt.subplots_adjust(top=0.8, right=0.95, left=0.11, bottom=0.05, hspace=0.4)
    plt.show()
    plt.savefig(Path(save_path).with_suffix(".pdf"))
    plt.savefig(save_path)
    plt.close()
    print(f"Plots saved at {save_path}")


def plot_evaluation_for_lm_datasets(model_type):
    modality = "lm"
    fig, ax = plt.subplots(3, 3, figsize=FIG_ASPECT)

    # choices:
    # "livecell", "tissuenet" (one_chan OR multi_chan), "deepbacs", "covid_if", "plantseg/root", "hpa",
    # "plantseg/ovules", "neurips-cell-seg/tuning" (/all; /self), "lizard", "mouse-embryo"
    # "dynamicnuclearnet", "pannuke"

    get_barplots(ax[0, 0], "livecell", modality, model_type, benchmark_choice="livecell")
    get_barplots(ax[0, 1], "deepbacs", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[0, 2], "tissuenet/multi_chan", modality, model_type, benchmark_choice="tissuenet")
    get_barplots(ax[1, 0], "plantseg/root", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[1, 1], "neurips-cell-seg/tuning", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[1, 2], "covid_if", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[2, 0], "plantseg/ovules", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[2, 1], "lizard", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[2, 2], "hpa", modality, model_type, benchmark_choice="cyto2")

    _get_plot_postprocessing(
        fig=fig, experiment_title="Light Microscopy", save_path=f"lm_{model_type}_evaluation.svg"
    )


def plot_evaluation_for_em_datasets(model_type):
    modality = "em"
    fig, ax = plt.subplots(3, 3, figsize=FIG_ASPECT)

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
    get_barplots(ax[1, 2], "mitolab/c_elegans", modality, model_type)
    get_barplots(ax[2, 0], "uro_cell", modality, model_type)
    get_barplots(ax[2, 1], "nuc_mm/mouse", modality, model_type)
    get_barplots(ax[2, 2], "sponge_em", modality, model_type)

    _get_plot_postprocessing(
        fig=fig, experiment_title="Electron Microscopy", save_path=f"em_{model_type}_evaluation.svg"
    )


def plot_evaluation_for_all_em_datasets(model_type):
    modality = "em"
    fig, ax = plt.subplots(4, 4, figsize=FIG_ASPECT)

    # choices:
    # for mito-nuc
    #    "mitoem/rat", "mitoem/human", "platynereis/nuclei", "mitolab/c_elegans", "mitolab/fly_brain",
    #    "mitolab/glycolytic_muscle", "mitolab/hela_cell", "mitolab/lucchi_pp", "mitolab/salivary_gland",
    #    "mitolab/tem", "lucchi", "nuc-mm/mouse", "nuc-mm/zebrafish", "uro_cell", "sponge_em",
    #    "vnc", "asem/mito"

    get_barplots(ax[0, 0], "mitoem/rat", modality, model_type)
    get_barplots(ax[0, 1], "mitoem/human", modality, model_type)
    get_barplots(ax[0, 2], "platynereis/nuclei", modality, model_type)
    get_barplots(ax[0, 3], "mitolab/c_elegans", modality, model_type)
    get_barplots(ax[1, 0], "mitolab/fly_brain", modality, model_type)
    get_barplots(ax[1, 1], "mitolab/glycolytic_muscle", modality, model_type)
    get_barplots(ax[1, 2], "mitolab/hela_cell", modality, model_type)
    get_barplots(ax[1, 3], "mitolab/salivary_gland", modality, model_type)
    get_barplots(ax[2, 0], "mitolab/tem", modality, model_type)
    get_barplots(ax[2, 1], "lucchi", modality, model_type)
    get_barplots(ax[2, 2], "nuc_mm/mouse", modality, model_type)
    get_barplots(ax[2, 3], "nuc_mm/zebrafish", modality, model_type)
    get_barplots(ax[3, 0], "uro_cell", modality, model_type)
    get_barplots(ax[3, 1], "sponge_em", modality, model_type)
    get_barplots(ax[3, 2], "sponge_em", modality, model_type)
    get_barplots(ax[3, 3], "sponge_em", modality, model_type)

    _get_plot_postprocessing(
        fig=fig, experiment_title="Electron Microscopy", save_path=f"em_{model_type}_all_evaluation.png"
    )


def main():
    plot_evaluation_for_lm_datasets("vit_l")
    plot_evaluation_for_em_datasets("vit_l")

    # plot_evaluation_for_all_em_datasets("vit_h")
    # TODO: plot for all lm datasets


if __name__ == "__main__":
    main()
