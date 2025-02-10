import os
from glob import glob
from natsort import natsorted

import pandas as pd


ROOT = "/home/nimanwai/micro-sam/development"


def get_comparison_plot_for_new_models(metric, model_type):
    old_mfolder = f"{model_type}_old_model"
    new_mfolder = f"{model_type}_new_model"

    old_results = natsorted(glob(os.path.join(ROOT, old_mfolder, "*", "results", "ais_2d.csv")))
    new_results = natsorted(glob(os.path.join(ROOT, new_mfolder, "*", "results", "ais_2d.csv")))

    results = []
    for old_res_path, new_res_path in zip(old_results, new_results):
        dname = old_res_path.rsplit("/")[-3]
        assert dname in new_res_path, (old_res_path, new_res_path)

        res = {
            "dataset": dname,
            f"{model_type}_lm": pd.read_csv(old_res_path)[metric][0],
            f"{model_type}_lm (NEW)": pd.read_csv(new_res_path)[metric][0],
        }
        results.append(pd.DataFrame.from_dict([res]))

    results = pd.concat(results, ignore_index=True)
    print(results)
    print()


def main():
    get_comparison_plot_for_new_models(metric="mSA", model_type="vit_b")
    get_comparison_plot_for_new_models(metric="mSA", model_type="vit_l")


main()
