import os
from glob import glob
from pathlib import Path
from natsort import natsorted

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = "/scratch/users/archit/experiments"


def plot_all_experiments():
    all_experiment_paths = glob(os.path.join(ROOT, "*"))
    benchmark_experiment_paths, resource_experiment_paths = [], []
    for experiment_path in all_experiment_paths:
        if experiment_path.endswith("generalist") or experiment_path.endswith("vanilla"):
            benchmark_experiment_paths.append(experiment_path)
        else:
            resource_experiment_paths.append(experiment_path)

    # let's get the benchmark results
    all_benchmark_results = {}
    for be_path in sorted(benchmark_experiment_paths):
        experiment_name = os.path.split(be_path)[-1]
        all_res_paths = glob(os.path.join(be_path, "*", "results", "*"))
        all_model_paths = glob(os.path.join(be_path, "*"))
        for model_path in all_model_paths:
            model_name = os.path.split(model_path)[-1]
            all_res_paths = sorted(glob(os.path.join(model_path, "results", "*")))
            benchmark_list = []
            for i, res_path in enumerate(all_res_paths):
                res = pd.read_csv(res_path)
                res_name = Path(res_path).stem

                if res_name.startswith("grid_search"):
                    continue

                if res_name.endswith("point"):
                    res_name = "point"
                elif res_name.endswith("box"):
                    # res_name = "box"
                    continue
                elif res_name.endswith("decoder"):
                    res_name = "ais"

                res_df = pd.DataFrame(
                    {"name": experiment_name, "type": res_name, "results": res.iloc[0]["msa"]}, index=[i]
                )
                benchmark_list.append(res_df)

            benchmark_df = pd.concat(benchmark_list, ignore_index=True)
            this_name = model_name if experiment_name == "vanilla" else f"{model_name}_lm"
            all_benchmark_results[this_name] = benchmark_df

    # now, let's get the resource efficient fine-tuning
    for exp_path in resource_experiment_paths:
        resource_name = os.path.split(exp_path)[-1]
        all_model_paths = glob(os.path.join(exp_path, "*"))
        for model_epath in all_model_paths:
            model_name = os.path.split(model_epath)[-1]
            print(f"Results for {resource_name} on {model_name}:")
            all_image_setting_paths = natsorted(glob(os.path.join(model_epath, "*", "*")))
            all_res_list = []
            for image_epath in all_image_setting_paths:
                image_setting = os.path.split(image_epath)[-1]
                all_res_paths = sorted(glob(os.path.join(image_epath, "results", "*")))
                per_image_setting = []
                for i, res_path in enumerate(all_res_paths):
                    res_name = Path(res_path).stem
                    if res_name.startswith("grid_search"):
                        continue

                    res = pd.read_csv(res_path)
                    score = res.iloc[0]["msa"]

                    if res_name.endswith("point"):
                        res_name = "point"
                    elif res_name.endswith("box"):
                        # res_name = "box"
                        continue
                    elif res_name.endswith("decoder"):
                        res_name = "ais"

                    res_df = pd.DataFrame(
                        {"name": image_setting, "type": res_name, "results": score}, index=[i]
                    )
                    per_image_setting.append(res_df)

                per_image_df = pd.concat(per_image_setting, ignore_index=True)
                all_res_list.append(per_image_df)

            this_res = pd.concat([all_benchmark_results[model_name], *all_res_list])
            sns.lineplot(x="name", y="results", hue="type", data=this_res)
            plt.title("Generalist" if model_name.endswith("lm") else "Vanilla")

            save_path = f"./figures/{resource_name}/{model_name}.png"
            try:
                plt.savefig(save_path)
            except FileNotFoundError:
                os.makedirs(os.path.split(save_path)[0])
                plt.savefig(save_path)

            plt.close()
            breakpoint()


def main():
    plot_all_experiments()


if __name__ == "__main__":
    main()
