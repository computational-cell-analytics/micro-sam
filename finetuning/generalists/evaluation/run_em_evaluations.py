import re
import subprocess


CMD = "python submit_em_evaluation.py "


def run_eval_process(cmd):
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def run_specific_experiment(dataset_name, model_type, experiment_set, roi):
    cmd = CMD + f"-d {dataset_name} " + f"-m {model_type} " + f"-e {experiment_set} " + f"-r {roi}"
    print(f"Running the command: {cmd} \n")

    _cmd = re.split(r"\s", cmd)

    run_eval_process(_cmd)


def for_all_em():
    def run_one_setup(all_dataset_list, all_model_list, all_experiment_set_list, roi):
        for dataset_name in all_dataset_list:
            for model_type in all_model_list:
                for experiment_set in all_experiment_set_list:
                    run_specific_experiment(dataset_name, model_type, experiment_set, roi)

    # let's run for organelles
    run_one_setup(
        all_dataset_list=[
            "mitoem/rat", "mitoem/human", "platynereis/nuclei", "mitolab/c_elegans", "mitolab/fly_brain",
            "mitolab/glycolytic_muscle", "mitolab/hela_cell", "mitolab/lucchi_pp", "mitolab/salivary_gland",
            "mitolab/tem", "lucchi", "nuc-mm/mouse", "nuc-mm/zebrafish", "uro_cell", "sponge_em", "platynereis/cilia",
        ],
        all_model_list=["vit_b", "vit_h"],
        all_experiment_set_list=["vanilla", "generalist"],
        roi="organelles"
    )

    # next, let's run for boundaries
    run_one_setup(
        all_dataset_list=[
            "cremi", "platynereis/cells", "axondeepseg", "snemi", "isbi"
        ],
        all_model_list=["vit_b", "vit_h"],
        all_experiment_set_list=["vanilla", "generalist"],
        roi="boundaries"
    )


def main():
    for_all_em()


if __name__ == "__main__":
    main()
