import os
from glob import glob
from pathlib import Path

from math import pi
import pandas as pd
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/v2/test/n_objects_per_batch/"


def _get_result_dict(res_df, result_path, experiment_name):
    res_name = Path(result_path).stem

    if res_name.startswith("grid"):
        return
    elif res_name.endswith("box"):
        res_name = "box"
    elif res_name.endswith("point"):
        res_name = "point"
    elif res_name.startswith("instance"):
        res_name = "ais"

    return {
        "name": res_name,
        "results": res_df.iloc[0]["msa"],
        "type": experiment_name
    }


def plot_n_objects():
    all_experiment_paths = glob(os.path.join(EXPERIMENT_ROOT, "*"))
    for experiment_path in all_experiment_paths:
        experiment_name = os.path.split(experiment_path)[-1]
        all_result_paths = glob(os.path.join(experiment_path, "results", "*"))
        all_res_list = []
        for result_path in all_result_paths:
            res_df = pd.read_csv(result_path)
            res_dict = _get_result_dict(res_df, result_path, experiment_name)
            this_res = pd.DataFrame.from_dict([res_dict])
            all_res_list.append(this_res)

        all_res = pd.concat(all_res_list, ignore_index=True)

        # parameters to define the essentials for the plot
        N = 5
        angles = [n / float(N) * 2 * pi for n in range(N)]

        # let's get the spider plot
        ax = plt.subplot(111, polar=True)


def main():
    plot_n_objects()


if __name__ == "__main__":
    main()
