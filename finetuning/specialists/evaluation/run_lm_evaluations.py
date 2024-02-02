import re
import subprocess


CMD = "python submit_lm_evaluation.py "


def run_eval_process(cmd):
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def run_specific_experiment(dataset_name, model_type, experiment_set):
    cmd = CMD + f"-d {dataset_name} " + f"-m {model_type} " + f"-e {experiment_set}"
    print(f"Running the command: {cmd} \n")

    _cmd = re.split(r"\s", cmd)

    run_eval_process(_cmd)


def for_all_lm():
    def run_one_setup(all_dataset_list, all_model_list, all_experiment_set_list):
        for dataset_name in all_dataset_list:
            for model_type in all_model_list:
                for experiment_set in all_experiment_set_list:
                    run_specific_experiment(dataset_name, model_type, experiment_set)

    # let's run for in-domain
    run_one_setup(
        all_dataset_list=["tissuenet", "deepbacs", "plantseg_root"],
        all_model_list=["vit_b", "vit_h"],
        all_experiment_set_list=["vanilla", "generalist", "specialist"]
    )

    # next, let's run for out-of-domain
    run_one_setup(
        all_dataset_list=["covid_if", "plantseg_ovules", "hpa", "lizard", "mouse-embryo", "ctc", "neurips-cell-seg"],
        all_model_list=["vit_b", "vit_h"],
        all_experiment_set_list=["vanilla", "generalist"]
    )


def main():
    for_all_lm()


if __name__ == "__main__":
    main()
