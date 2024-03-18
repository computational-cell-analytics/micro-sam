import os
from glob import glob
from pathlib import Path
from natsort import natsorted

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = "/scratch/users/archit/experiments"

PALETTE = {"ais": "#440154", "amg": "#404788", "point": "#1F968B", "box": "#73D055"}


def _get_all_results(name, all_res_paths):
    all_res_list, all_box_res_list = [], []
    for i, res_path in enumerate(all_res_paths):
        res = pd.read_csv(res_path)
        res_name = Path(res_path).stem

        if res_name.startswith("grid_search"):
            continue

        if res_name.endswith("point"):
            res_name = "point"
        elif res_name.endswith("box"):
            res_name = "box"
        elif res_name.endswith("decoder"):
            res_name = "ais"

        res_df = pd.DataFrame(
            {"name": name, "type": res_name, "results": res.iloc[0]["msa"]}, index=[i]
        )
        if res_name == "box":
            all_box_res_list.append(res_df)
        else:
            all_res_list.append(res_df)

    all_res_df = pd.concat(all_res_list, ignore_index=True)
    all_box_res_df = pd.concat(all_box_res_list, ignore_index=True)

    return all_res_df, all_box_res_df


def plot_all_experiments():
    all_experiment_paths = glob(os.path.join(ROOT, "*"))
    benchmark_experiment_paths, resource_experiment_paths = [], []
    for experiment_path in all_experiment_paths:
        if experiment_path.endswith("generalist") or experiment_path.endswith("vanilla"):
            benchmark_experiment_paths.append(experiment_path)
        else:
            resource_experiment_paths.append(experiment_path)

    # let's get the benchmark results
    all_benchmark_results, all_benchmark_box_results = {}, {}
    for be_path in sorted(benchmark_experiment_paths):

        experiment_name = os.path.split(be_path)[-1]
        all_res_paths = glob(os.path.join(be_path, "*", "results", "*"))
        all_model_paths = glob(os.path.join(be_path, "*"))

        for model_path in all_model_paths:
            model_name = os.path.split(model_path)[-1]
            all_res_paths = sorted(glob(os.path.join(model_path, "results", "*")))
            benchmark_df, benchmark_box_df = _get_all_results("initial", all_res_paths)
            this_name = model_name if experiment_name == "vanilla" else f"{model_name}_lm"
            all_benchmark_results[this_name] = benchmark_df
            all_benchmark_box_results[this_name] = benchmark_box_df

    # now, let's get the resource efficient fine-tuning
    for exp_path in sorted(resource_experiment_paths):
        fig, ax = plt.subplots(2, 2, figsize=(15, 10), sharey="row")
        resource_name = os.path.split(exp_path)[-1]
        all_model_paths = glob(os.path.join(exp_path, "*"))
        idx = 0
        for model_epath in sorted(all_model_paths):
            model_name = os.path.split(model_epath)[-1]
            if model_name[:5] == "vit_t":  # we don't plot results for vit_t
                continue

            print(f"Results for {resource_name} on {model_name}")
            all_image_setting_paths = natsorted(glob(os.path.join(model_epath, "*", "*")))
            all_res_list, all_box_res_list = [], []
            for image_epath in all_image_setting_paths:
                image_setting = os.path.split(image_epath)[-1]
                all_res_paths = sorted(glob(os.path.join(image_epath, "results", "*")))
                per_image_df, per_image_box_df = _get_all_results(image_setting.split("-")[0], all_res_paths)
                all_res_list.append(per_image_df)
                all_box_res_list.append(per_image_box_df)

            this_res = pd.concat([all_benchmark_results[model_name], *all_res_list])
            this_box_res = pd.concat([all_benchmark_box_results[model_name], *all_box_res_list])

            sns.lineplot(
                x="name", y="results", hue="type", data=this_res,
                ax=ax[0, idx], palette=PALETTE, hue_order=PALETTE.keys()
            )
            _title = "Generalist" if model_name.endswith("lm") else "Vanilla"
            ax[0, idx].set_title(_title, fontsize=13, fontweight="bold")
            ax[0, idx].set(xlabel=None, ylabel=None)

            sns.lineplot(
                x="name", y="results", hue="type", data=this_box_res,
                ax=ax[1, idx], palette=PALETTE, hue_order=PALETTE.keys()
            )
            ax[1, idx].set_title(_title, fontsize=13, fontweight="bold")
            ax[1, idx].set(xlabel=None, ylabel=None)

            idx += 1

        if idx > 0:
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
                x=0.6425, y=2.35, s=" X-Axis: Images \n Y-Axis: Segmentation Quality ", ha='left',
                transform=plt.gca().transAxes, bbox={"facecolor": "None", "edgecolor": "#D6D6D6", "boxstyle": "round"}
            )

            plt.subplots_adjust(top=0.865, right=0.95, left=0.075, bottom=0.05)
            fig.suptitle("Resource Efficient Finetuning (Covid IF)", fontsize=26, x=0.515, y=0.95)

            save_path = f"./figures/{resource_name}/results.png"
            try:
                plt.savefig(save_path)
            except FileNotFoundError:
                os.makedirs(os.path.split(save_path)[0])
                plt.savefig(save_path)

            plt.close()
            print()


def main():
    plot_all_experiments()


if __name__ == "__main__":
    main()
