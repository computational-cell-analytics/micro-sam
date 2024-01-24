import os
from subprocess import run

import pandas as pd

# TODO we need to make sure that this has the corrected training data for the proper training
DATA_ROOT = "/scratch/projects/nim00007/data/LiveCELL"
SAVE_ROOT = "/scratch/projects/nim00007/sam/livecell_grid_search"

LRS = [1e-4, 5e-5, 1e-5, 5e-6]


def _get_name_and_checkpoint(lr, use_adamw):
    name = f"vit_b-lr{lr}"
    if use_adamw:
        name += "-adamw"
    checkpoint = os.path.join(SAVE_ROOT, "checkpoints", name, "best.pt")
    return name, checkpoint


def precompute_embeddings():
    for lr in LRS:
        for use_adamw in [True, False]:
            name, ckpt = _get_name_and_checkpoint(lr, use_adamw)
            if not os.path.exists(ckpt):
                print("Skipping:", ckpt)
                continue
            cmd = ["sbatch", "precompute_embeddings.sbatch", "--name", f"livecell_grid_search/{name}",
                   "-m", "vit_b", "-c", ckpt]
            run(cmd)


def run_evaluations():
    for lr in LRS:
        for use_adamw in [True, False]:
            name, ckpt = _get_name_and_checkpoint(lr, use_adamw)
            if not os.path.exists(ckpt):
                print("Skipping:", ckpt)
                continue
            # iterative prompting (start with prompt)
            cmd = ["sbatch", "iterative_prompting.sbatch", "--name", f"livecell_grid_search/{name}",
                   "-m", "vit_b", "-c", ckpt]
            run(cmd)
            # iterative prompting (start with box)
            cmd = ["sbatch", "iterative_prompting.sbatch", "--name", f"livecell_grid_search/{name}",
                   "-m", "vit_b", "-c", ckpt, "--box"]
            run(cmd)
            # instance segmentation
            cmd = ["sbatch", "evaluate_instance_segmentation.sbatch", "--name", f"livecell_grid_search/{name}",
                   "-m", "vit_b", "-c", ckpt]
            run(cmd)


def accumulate_results():
    result_root = "/scratch/projects/nim00007/sam/experiments/livecell/livecell_grid_search"

    # TODO add the instance segmentation result
    exp_names = [
        "iterative_prompts_start_box.csv", "iterative_prompts_start_point.csv", "instance_segmentation_with_decoder.csv"
    ]

    for exp_name in exp_names:
        results = []
        for lr in LRS:
            for use_adamw in [True, False]:
                name = f"vit_b-lr{lr}"
                if use_adamw:
                    name += "-adamw"

                result_path = os.path.join(result_root, name, "results", exp_name)
                if not os.path.exists(result_path):
                    continue

                this_result = pd.read_csv(result_path)
                this_result = this_result.rename(columns={"Unnamed: 0": "iteration"})

                this_result["lr"] = [lr] * len(this_result)
                this_result["optimizer"] = ["adamw" if use_adamw else "adam"] * len(this_result)

                results.append(this_result)

        results = pd.concat(results)
        out_path = os.path.join(result_root, exp_name)
        results.to_csv(out_path, index=False)


def main():
    # precompute_embeddings()
    # run_evaluations()
    accumulate_results()


if __name__ == "__main__":
    main()
