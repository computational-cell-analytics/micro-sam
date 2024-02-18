import common

# NOTE:
# current resources available:
#   - xps13 (CPU compute) (local)
#   - medium (CPU compute partition) (SCC)
#   - gtx1080: 8GB (SCC)
#   - rtx5000: 16GB (SCC)
#   - v100: 32GB (SCC / Grete)
#   - A100: 40GB (Grete)
#   - A100: 80GB (Grete)
def write_batch_sript(tier_choice, env_name, resource_name):
    "Writing scripts for resource-efficient trainings for micro-sam finetuning on Covid-IF."
    batch_script = common.base_slurm_script(
        tier_choice=tier_choice,
        env_name=env_name,
        resource_name=resource_name
    )

    python_script = "python covid_if_finetuning.py "

    # add parameters to the python script
    python_script += f"-i {input_path} "  # path to the covid-if data
    python_script += f"-s {save_root} "  # path to save the model checkpoints and logs
    python_script += f"-m {model_type} "  # choice of vit
    python_script += f"--n_objects {n_objects} "  # number of objects per batch for finetuning
    python_script += f"--n_samples {n_samples} "  # number of samples we train for

    # let's add the python script to the bash script
    batch_script += python_script

    with open(script_name, "w") as f:
        f.write(batch_script)



def main(args):
    # TODO: run the finetuning scripts for the different tiers of finetuning
    write_batch_sript(...)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tier", type=int, default=1, help="The different tiers of resource-efficient finetuning.")
    args = parser.parse_args()
    main(args)
