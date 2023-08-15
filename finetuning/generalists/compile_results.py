import os
from glob import glob

import pandas as pd

from evaluate_generalist import EXPERIMENT_ROOT
from util import EM_DATASETS, LM_DATASETS


def get_results(model, ds):
    res_folder = os.path.join(EXPERIMENT_ROOT, model, ds, "results")
    res_paths = sorted(glob(os.path.join(res_folder, "box", "*.csv"))) +\
        sorted(glob(os.path.join(res_folder, "points", "*.csv")))

    amg_res = os.path.join(res_folder, "amg.csv")
    if os.path.exists(amg_res):
        res_paths.append(amg_res)

    results = []
    for path in res_paths:
        prompt_res = pd.read_csv(path)
        prompt_name = os.path.splitext(os.path.relpath(path, res_folder))[0]
        prompt_res.insert(0, "prompt", [prompt_name])
        results.append(prompt_res)
    results = pd.concat(results)
    results.insert(0, "dataset", results.shape[0] * [ds])

    return results


def compile_results(models, datasets, out_path):
    results = []

    for model in models:
        model_results = []

        for ds in datasets:
            ds_results = get_results(model, ds)
            model_results.append(ds_results)

        model_results = pd.concat(model_results)
        model_results.insert(0, "model", [model] * model_results.shape[0])
        results.append(model_results)

    results = pd.concat(results)
    results.to_csv(out_path, index=False)


def compile_em():
    compile_results(
        ["vit_h", "vit_h_em", "vit_b", "vit_b_em"],
        EM_DATASETS,
        os.path.join(EXPERIMENT_ROOT, "evaluation-em.csv")
    )


# TODO
def compile_lm():
    compile_results(
        ["vit_h", "vit_h_lm", "vit_b", "vit_b_lm"],
        LM_DATASETS,
        os.path.join(EXPERIMENT_ROOT, "evaluation-lm.csv")
    )


def main():
    compile_em()


if __name__ == "__main__":
    main()
