import os
from glob import glob
import pandas as pd
from natsort import natsorted


ROOT = "/home/anwai/results/dfki/R3/"


def limited_data_livecell(res_root, model):
    # experiments from VKS and NK (DFKI)
    all_results = []
    all_experiments_dir = natsorted(glob(os.path.join(res_root, model, "*")))
    for experiment_dir in all_experiments_dir:
        experiment_name = os.path.split(experiment_dir)[-1]

        ais = pd.read_csv(os.path.join(experiment_dir, "results_ais", "instance_segmentation_with_decoder.csv"))
        amg = pd.read_csv(os.path.join(experiment_dir, "results_amg", "amg.csv"))
        ip = pd.read_csv(os.path.join(experiment_dir, "results_ip", "iterative_prompts_start_point.csv"))
        ib = pd.read_csv(os.path.join(experiment_dir, "results_ipb", "iterative_prompts_start_box.csv"))

        res = {
            "experiment": experiment_name,
            "ais": ais.iloc[0]["msa"],
            "amg": amg.iloc[0]["msa"],
            "point": ip.iloc[0]["msa"],
            "box": ib.iloc[0]["msa"],
            "ip": ip.iloc[-1]["msa"],
            "ib": ib.iloc[-1]["msa"]
        }
        all_results.append(pd.DataFrame.from_dict([res]))

    res_df = pd.concat(all_results, ignore_index=True)
    return res_df


def main():
    res = limited_data_livecell(ROOT, "vit_b")
    print(res)


if __name__ == "__main__":
    main()
