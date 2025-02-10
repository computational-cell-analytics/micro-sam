import os
from glob import glob
from natsort import natsorted

import pandas as pd


ROOT = "/home/nimanwai/micro-sam/development"


def get_comparison_plot_for_new_models(metric, model_type):
    old_mfolder = "vit_b_old_model" if model_type == "vit_b_lm" else "vit_l_old_model"
    new_mfolder = "vit_b_new_model" if model_type == "vit_b_lm" else "vit_l_new_model"

    old_results = natsorted(glob(os.path.join(ROOT, old_mfolder, "*", "results", "ais_2d.csv")))
    new_results = natsorted(glob(os.path.join(ROOT, new_mfolder, "*", "results", "ais_2d.csv")))

    results = []
    for old_res_path, new_res_path in zip(old_results, new_results):
        old_res = pd.read_csv(old_res_path)
        new_res = pd.read_csv(new_res_path)

        dname = old_res_path.rsplit("/")[-3]
        res = {
            "dataset": dname,
            model_type: old_res[metric][0],
            f"{model_type} (NEW)": new_res[metric][0],
        }
        results.append(pd.DataFrame.from_dict([res]))

    results = pd.concat(results, ignore_index=True)
    print(results)
    breakpoint()


def main():
    get_comparison_plot_for_new_models(metric="mSA", model_type="vit_b_lm")


main()
