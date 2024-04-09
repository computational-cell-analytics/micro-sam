import re
import subprocess


CMD = "python submit_all_evaluation.py "
ALl_MODELS = ["vit_t", "vit_b", "vit_l", "vit_h"]


def run_eval_process(cmd):
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def run_specific_experiment(dataset_name, model_type, experiment_set, roi, specific_script):
    cmd = CMD + f"-d {dataset_name} " + f"-m {model_type} " + f"-e {experiment_set} " + f"-r {roi}"
    if specific_script is not None:
        cmd += f" -s {specific_script}"
    print(f"Running the command: {cmd} \n")
    _cmd = re.split(r"\s", cmd)
    run_eval_process(_cmd)


def run_one_setup(all_dataset_list, all_model_list, all_experiment_set_list, roi, specific_script):
    for dataset_name in all_dataset_list:
        for model_type in all_model_list:
            for experiment_set in all_experiment_set_list:
                if experiment_set == "vanilla" and specific_script == "evaluate_instance_segmentation":
                    # we don't perform ais on vanilla models
                    continue

                run_specific_experiment(dataset_name, model_type, experiment_set, roi, specific_script)


def for_all_lm(specific_script):
    # let's run for in-domain
    run_one_setup(
        all_dataset_list=["tissuenet", "deepbacs", "plantseg/root", "livecell", "neurips-cell-seg"],
        all_model_list=ALl_MODELS,
        all_experiment_set_list=["vanilla", "generalist", "specialist"],
        roi="lm",
        specific_script=specific_script
    )

    # next, let's run for out-of-domain
    run_one_setup(
        all_dataset_list=["covid_if", "plantseg/ovules", "hpa", "lizard", "mouse-embryo"],
        all_model_list=ALl_MODELS,
        all_experiment_set_list=["vanilla", "generalist"],
        roi="lm",
        specific_script=specific_script
    )


def for_all_em(specific_script):
    # let's run for organelles
    run_one_setup(
        all_dataset_list=[
            "mitoem/rat", "mitoem/human", "platynereis/nuclei", "mitolab/c_elegans", "mitolab/fly_brain",
            "mitolab/glycolytic_muscle", "mitolab/hela_cell", "mitolab/lucchi_pp", "mitolab/salivary_gland",
            "mitolab/tem", "lucchi", "nuc_mm/mouse", "nuc_mm/zebrafish", "uro_cell", "sponge_em", "platynereis/cilia",
        ],
        all_model_list=ALl_MODELS,
        all_experiment_set_list=["vanilla", "generalist"],
        roi="organelles",
        specific_script=specific_script
    )

    # next, let's run for boundaries
    run_one_setup(
        all_dataset_list=["cremi", "platynereis/cells", "axondeepseg", "snemi", "isbi"],
        all_model_list=ALl_MODELS,
        all_experiment_set_list=["vanilla", "generalist"],
        roi="boundaries",
        specific_script=specific_script
    )


def for_livecell(specific_script):
    # let's run for livecell
    run_one_setup(
        all_dataset_list=["livecell"],
        all_model_list=ALl_MODELS,
        all_experiment_set_list=["vanilla", "generalist", "specialist"],
        roi="lm",
        specific_script=specific_script
    )


def for_custom_livecell(specific_script):
    EXP_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models"

    def _run_custom_livecell(model_type, experiment_set):
        cmd = CMD
        cmd += "-d livecell "
        cmd += f"-m {model_type} "
        cmd += f"-e {experiment_set} "
        cmd += "-r lm "
        cmd += f"--experiment_path {EXP_ROOT}/test/input_logits/{experiment_set}/lm/livecell/{model_type} "
        cmd += f"-s {specific_script} "
        cmd += "--use_masks"

        print(f"Running the command: {cmd} \n")
        _cmd = re.split(r"\s", cmd)
        run_eval_process(_cmd)

    all_experiments = ["vanilla", "generalist", "specialist"]
    for model_type in ALl_MODELS:
        for experiment_set in all_experiments:
            _run_custom_livecell(model_type, experiment_set)


def for_variance_in_livecell(run_set):
    EXP_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/test"

    def _run_custom_livecell(model_type, experiment_set):
        cmd = CMD
        cmd += "-d livecell "
        cmd += f"-m {model_type} "
        cmd += f"-e {experiment_set} "
        cmd += "-r lm "
        cmd += f"--experiment_path {EXP_ROOT}/{run_set}/{experiment_set}/lm/livecell/{model_type}"

        print(f"Running the command: {cmd} \n")
        _cmd = re.split(r"\s", cmd)
        run_eval_process(_cmd)

    all_experiments = ["vanilla", "generalist", "specialist"]
    for model_type in ALl_MODELS:
        for experiment_set in all_experiments:
            _run_custom_livecell(model_type, experiment_set)


def main(args):
    # for_all_lm(specific_script=args.specific_script)
    # for_all_em(specific_script=args.specific_script)
    # for_livecell(specific_script=args.specific_script)

    # for_custom_livecell(specific_script="iterative_prompting")

    for_variance_in_livecell("run_1")
    for_variance_in_livecell("run_2")
    for_variance_in_livecell("run_3")
    for_variance_in_livecell("run_4")
    for_variance_in_livecell("run_5")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--specific_script", type=str, default=None)
    args = parser.parse_args()
    main(args)
