from subprocess import run

DATA_ROOT = "/scratch/usr/nimanwai/data/livecell"
SAVE_ROOT = "/scratch/usr/nimanwai/experiments/micro-sam/parameters_ablation/"


def run_grid_search(dry_run):
    lrs = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
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


run_grid_search(dry_run=False)
