import os
import re
import subprocess
from glob import glob
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CMD = "python submit_all_evaluation.py "
CHECKPOINT_ROOT = "/scratch/usr/nimanwai/experiments/micro-sam/unetr-decoder-updates/"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/test/unetr-decoder-updates"


def run_eval_process(cmd):
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def run_specific_experiment(dataset_name, model_type, setup):
    all_checkpoint_dirs = sorted(glob(os.path.join(CHECKPOINT_ROOT, f"{setup}-*")))
    for checkpoint_dir in all_checkpoint_dirs:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoints", model_type, "lm_generalist_sam", "best.pt")

        experiment_name = checkpoint_dir.split("/")[-1]
        experiment_folder = os.path.join(EXPERIMENT_ROOT, experiment_name, dataset_name, model_type)

        cmd = CMD + f"-d {dataset_name} " + f"-m {model_type} " + "-e generalist "
        cmd += f"--checkpoint_path {checkpoint_path} "
        cmd += f"--experiment_path {experiment_folder}"
        print(f"Running the command: {cmd} \n")
        _cmd = re.split(r"\s", cmd)
        run_eval_process(_cmd)


def _get_plots(dataset_name, model_type):
    experiment_dirs = sorted(glob(os.path.join(EXPERIMENT_ROOT, "*")))

    # adding a fixed color palette to each experiments, for consistency in plotting the legends
    palette = {"amg": "C0", "ais": "C1", "box": "C2", "i_b": "C3", "point": "C4", "i_p": "C5"}

    fig, ax = plt.subplots(1, len(experiment_dirs), figsize=(20, 10), sharex="col", sharey="row")

    for idx, _experiment_dir in enumerate(experiment_dirs):
        all_result_paths = sorted(glob(os.path.join(_experiment_dir, dataset_name, model_type, "results", "*")))
        res_list_per_experiment = []
        for i, result_path in enumerate(all_result_paths):
            # avoid using the grid-search parameters' files
            _tmp_check = os.path.split(result_path)[-1]
            if _tmp_check.startswith("grid_search_"):
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

        container = sns.barplot(
            x="name", y="results", hue="type", data=res_df_per_experiment, ax=ax[idx], palette=palette
        )
        ax[idx].set(xlabel="Experiments", ylabel="Segmentation Quality")
        ax[idx].legend(title="Settings", bbox_to_anchor=(1, 1))

        # adding the numbers over the barplots
        for j in container.containers:
            container.bar_label(j, fmt='%.2f')

        # titles for each subplot
        ax[idx].title.set_text(_experiment_dir.split("/")[-1])

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

    save_path = f"figures/{dataset_name}/{model_type}.png"

    try:
        plt.savefig(save_path)
    except FileNotFoundError:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        plt.savefig(save_path)

    plt.close()
    print(f"Plot saved at {save_path}")


def run_one_setup(all_dataset_list, all_model_list, setup):
    for dataset_name in all_dataset_list:
        for model_type in all_model_list:
            run_specific_experiment(dataset_name=dataset_name, model_type=model_type, setup=setup)
            breakpoint()


def for_all_lm(setup):
    assert setup in ["conv-transpose", "bilinear"]

    # let's run for in-domain
    run_one_setup(
        all_dataset_list=["tissuenet", "deepbacs", "plantseg/root", "livecell", "neurips-cell-seg"],
        all_model_list=["vit_t", "vit_b", "vit_l", "vit_h"],
        setup=setup
    )


def _run_evaluations():
    os.chdir("../")
    # for_all_lm("conv-transpose")
    for_all_lm("bilinear")


def _get_all_plots():
    all_datasets = ["tissuenet", "deepbacs", "plantseg/root", "livecell", "neurips-cell-seg"]
    all_models = ["vit_t", "vit_b", "vit_l", "vit_h"]

    for dataset_name in all_datasets:
        for model_type in all_models:
            _get_plots(dataset_name, model_type)


def main():
    # _run_evaluations()
    _get_all_plots()


if __name__ == "__main__":
    main()
