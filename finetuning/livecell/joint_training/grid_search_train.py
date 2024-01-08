from subprocess import run

# TODO we need to make sure that this has the corrected training data for the proper training
DATA_ROOT = "/scratch/projects/nim00007/data/LiveCELL"
SAVE_ROOT = "/scratch-grete/projects/nim00007/sam/livecell_grid_search"


def run_grid_search(dry_run):
    lrs = [1e-4, 5e-5, 1e-5, 5e-6]
    for lr in lrs:
        for use_adamw in [True, False]:
            name = f"vit_b-lr{lr}"
            if use_adamw:
                name += "-adamw"
            cmd = ["sbatch", "submit_training.sh", "-i", DATA_ROOT, "-s", SAVE_ROOT,
                   "--iterations", "25000", "--name", name, "--lr", str(lr)]
            if use_adamw:
                cmd.append("--use_adamw")
            if dry_run:
                print(cmd)
            else:
                run(cmd)


run_grid_search(False)
