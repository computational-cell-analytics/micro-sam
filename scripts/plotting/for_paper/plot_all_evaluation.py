import os
from pathlib import Path
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/"

# adding a fixed color palette to each experiments, for consistency in plotting the legends
PALETTE = {
    "AIS": "#045275",
    "AMG": "#FCDE9C",
    "Point": "#7CCBA2",
    r"I$_{P}$": "#089099",
    "Box": "#90477F",
    r"I$_{B}$": "#F0746E"
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
    "dynamicnuclearnet": "DynamicNuclearNet",
    "pannuke": "PanNuke",
    "mitoem/rat": "$\it{MitoEM}$ $\it{(Rat)}$",
    "mitoem/human": "$\it{MitoEM}$ $\it{(Human)}$",
    "platynereis/nuclei": "$\it{Platynereis}$ $\it{(Nuclei)}$",
    "platynereis/cilia": "Platynereis (Cilia)",
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
    "sponge_em": "Sponge EM",
    "vnc": "VNC",
    "asem/mito": "ASEM (Mito)",
    "cremi": "CREMI",
    "asem/er": "ASEM (ER)",
}

MODELS = {
    "vit_h": 'ViT Huge',
    "vit_l": 'ViT Large',
    "vit_b": 'ViT Base',
    "vit_t": 'ViT Tiny'
}

FIG_ASPECT = (30, 30)

plt.rcParams.update({'font.size': 30})


def gather_all_results(dataset, modality, model_type):
    res_list_per_dataset = []
    for experiment_dir in glob(os.path.join(EXPERIMENT_ROOT, "new_models", "v2", "*", modality, dataset)):
        experiment_name = os.path.split(experiment_dir[:experiment_dir.find(f"/{modality}")])[-1]

        if experiment_name == "vanilla":  # easy fix to alter the name
            experiment_name = "default"

        experiment_name = experiment_name.title()

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
                        "type": Path(result_path).stem.upper() if len(setting_name) == 3 else "AIS",
                        "results": res.iloc[0]["msa"]
                    }, index=[i]
                )
            else:  # saving results from iterative prompting
                prompt_name = Path(result_path).stem.split("_")[-1]
                res_df = pd.concat(
                    [
                        pd.DataFrame(
                            {"name": experiment_name, "type": prompt_name.title(), "results": res.iloc[0]["msa"]},
                            index=[i]
                        ),
                        pd.DataFrame(
                            {
                                "name": experiment_name,
                                "type": r"I$_{P}$" if prompt_name[0] == "p" else r"I$_{B}$",
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


def get_barplots(ax, dataset_name, modality, model_type, benchmark_choice=None, title_as_model_name=False):
    plt.rcParams["hatch.linewidth"] = 1.5
    res_df = gather_all_results(dataset_name, modality, model_type)
    sns.barplot(x="name", y="results", hue="type", data=res_df, ax=ax, palette=PALETTE, hue_order=PALETTE.keys())
    lines, labels = ax.get_legend_handles_labels()
    for line, label in zip(lines, labels):
        if label == "AIS":
            for k in range(len(line)):
                line.patches[k].set_hatch('///')
                line.patches[k].set_edgecolor('white')

    ax.set(xlabel=None, ylabel=None)
    ax.legend(title="Settings", bbox_to_anchor=(1, 1))
    ax.title.set_color("#212427")
    ax.set_title(
        TITLE[dataset_name] if not title_as_model_name else MODELS[model_type], fontweight="bold"
    )

    if dataset_name != "ctc":
        benchmark_name = "cellpose" if modality == "lm" else "mitonet"
        benchmark_res = get_benchmark_results(dataset_name, benchmark_name, benchmark_choice)
        if benchmark_res is not None:
            benchmark_title = "CellPose" if modality == "lm" else "MitoNet"
            ax.axhline(y=benchmark_res, label=benchmark_title, color="#DC3977", lw=3)


def _get_plot_postprocessing(
    fig, experiment_title, save_path, ignore_legends=False, title_loc=0.93,
    ylabel_choice=None, bba=None, adj_params={}, y_loc=None, x_loc=None
):
    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    if ignore_legends:
        hspace = 0.4
        bbox_to_anchor = (0.5, 0.06)
        plt.suptitle(experiment_title, fontsize=36, y=title_loc)
        if ylabel_choice == "em":
            if experiment_title == "ViT Tiny":
                y = 1.7
            else:
                y = 1.55
            x = -5.9
        else:
            if experiment_title == "ViT Tiny":
                y = 1.85
            else:
                y = 2.75
            x = -3.5
    else:
        bbox_to_anchor = (0.5, 0.04)
        hspace = 0.4
        x, y = -5.8, 1

    if y_loc is not None:
        y = y_loc

    if x_loc is not None:
        x = x_loc

    plt.text(x=x, y=y, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=36)

    if bbox_to_anchor is not None:
        bbox_to_anchor = bba

    fig.legend(all_lines, all_labels, loc="lower center", ncols=7, bbox_to_anchor=bbox_to_anchor)

    plt.subplots_adjust(hspace=hspace, **adj_params)
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
    get_barplots(ax[2, 2], "mouse-embryo", modality, model_type, benchmark_choice="cyto2")

    _get_plot_postprocessing(
        fig=fig,
        experiment_title="Light Microscopy",
        save_path=f"lm_{model_type}_evaluation.svg",
        bba=(0.5, 0.05),
        y_loc=0.88
    )


def plot_evaluation_for_all_lm_datasets(model_type):
    modality = "lm"
    fig, ax = plt.subplots(6, 2, figsize=(25, 30))

    # choices:
    # "livecell", "tissuenet" (one_chan OR multi_chan), "deepbacs", "covid_if", "plantseg/root", "hpa",
    # "plantseg/ovules", "neurips-cell-seg/tuning" (/all; /self), "lizard", "mouse-embryo"
    # "dynamicnuclearnet", "pannuke"

    get_barplots(ax[0, 0], "livecell", modality, model_type, benchmark_choice="livecell")
    get_barplots(ax[0, 1], "deepbacs", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[1, 0], "tissuenet/multi_chan", modality, model_type, benchmark_choice="tissuenet")
    get_barplots(ax[1, 1], "plantseg/root", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[2, 0], "neurips-cell-seg/tuning", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[2, 1], "covid_if", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[3, 0], "dynamicnuclearnet", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[3, 1], "plantseg/ovules", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[4, 0], "mouse-embryo", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[4, 1], "hpa", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[5, 0], "pannuke", modality, model_type, benchmark_choice="cyto2")
    get_barplots(ax[5, 1], "lizard", modality, model_type, benchmark_choice="cyto2")

    _get_plot_postprocessing(
        fig=fig,
        experiment_title=MODELS[model_type],
        save_path=f"lm_all_{model_type}_evaluation.svg",
        ignore_legends=True,
        ylabel_choice="lm",
        bba=(0.5, 0.05),
        x_loc=-3.25
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
    get_barplots(ax[1, 2], "mitolab/tem", modality, model_type)
    get_barplots(ax[2, 0], "uro_cell", modality, model_type)
    get_barplots(ax[2, 1], "nuc_mm/mouse", modality, model_type)
    get_barplots(ax[2, 2], "vnc", modality, model_type)

    _get_plot_postprocessing(
        fig=fig,
        experiment_title="Electron Microscopy",
        save_path=f"em_{model_type}_evaluation.svg",
        bba=(0.5, 0.05),
        y_loc=0.7,
    )


def plot_evaluation_for_all_em_datasets(model_type):
    modality = "em"
    fig, ax = plt.subplots(5, 3, figsize=(30, 30))

    # choices:
    # for mito-nuc
    #    "mitoem/rat", "mitoem/human", "platynereis/nuclei", "mitolab/c_elegans", "mitolab/fly_brain",
    #    "mitolab/glycolytic_muscle", "mitolab/hela_cell", "mitolab/lucchi_pp", "mitolab/salivary_gland",
    #    "mitolab/tem", "lucchi", "nuc-mm/mouse", "nuc-mm/zebrafish", "uro_cell", "sponge_em",
    #    "vnc", "asem/mito"

    get_barplots(ax[0, 0], "mitoem/rat", modality, model_type)
    get_barplots(ax[0, 1], "mitoem/human", modality, model_type)
    get_barplots(ax[0, 2], "platynereis/nuclei", modality, model_type)
    get_barplots(ax[1, 0], "mitolab/c_elegans", modality, model_type)
    get_barplots(ax[1, 1], "mitolab/fly_brain", modality, model_type)
    get_barplots(ax[1, 2], "mitolab/glycolytic_muscle", modality, model_type)
    get_barplots(ax[2, 0], "mitolab/hela_cell", modality, model_type)
    get_barplots(ax[2, 1], "mitolab/tem", modality, model_type)
    get_barplots(ax[2, 2], "lucchi", modality, model_type)
    get_barplots(ax[3, 0], "nuc_mm/mouse", modality, model_type)
    get_barplots(ax[3, 1], "uro_cell", modality, model_type)
    get_barplots(ax[3, 2], "sponge_em", modality, model_type)
    get_barplots(ax[4, 0], "vnc", modality, model_type)
    get_barplots(ax[4, 1], "asem/mito", modality, model_type)
    get_barplots(ax[4, 2], "platynereis/cilia", modality, model_type)

    _get_plot_postprocessing(
        fig=fig,
        experiment_title=MODELS[model_type],
        save_path=f"em_all_{model_type}_evaluation.svg",
        ignore_legends=True,
        ylabel_choice="em",
        bba=(0.5, 0.05),
        x_loc=-5.75
    )


def plot_em_specialists(dataset_name):
    modality = "em"
    fig, ax = plt.subplots(1, 4, figsize=(40, 12.5), sharey=True)

    get_barplots(ax[0], dataset_name, modality, "vit_t", title_as_model_name=True)
    get_barplots(ax[1], dataset_name, modality, "vit_b", title_as_model_name=True)
    get_barplots(ax[2], dataset_name, modality, "vit_l", title_as_model_name=True)
    get_barplots(ax[3], dataset_name, modality, "vit_h", title_as_model_name=True)

    if dataset_name != "cremi":
        dsplit = dataset_name.split("/")
        dataset_name = dsplit[0].upper() + f" ({dsplit[1].upper()})"

    _get_plot_postprocessing(
        fig=fig,
        experiment_title=dataset_name.upper(),
        save_path=f"em_{dataset_name}_specialist_evaluation.svg",
        ignore_legends=True,
        title_loc=0.98,
        ylabel_choice="em",
        bba=(0.5, -0.01),
        adj_params={"wspace": 0.05, "top": 0.85, "bottom": 0.15},
        y_loc=0.05, x_loc=-10.7,
    )


def main():
    # plot_evaluation_for_lm_datasets("vit_l")
    # plot_evaluation_for_em_datasets("vit_l")

    # all_models = ["vit_t", "vit_b", "vit_l", "vit_h"]
    # for model in all_models:
    #     plot_evaluation_for_all_em_datasets(model)
    #     plot_evaluation_for_all_lm_datasets(model)

    plot_em_specialists("cremi")
    plot_em_specialists("asem/er")


if __name__ == "__main__":
    main()
