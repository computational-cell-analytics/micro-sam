# Resource Efficient Finetuning: time taken to achieve the best epoch per setting
# a: rtx5000
#   - (vit_b) 1-image: 1192.79, 2-image: 725.15, 5-image: 3759.01, 10-image: 2427.18
#   - (vit_b_lm) 1-image: 2089.22, 2-image: 1622.69, 5-image: 3477.83, 10-image: 1869.33

# b: v100
#   - (vit_b) 1-image: 752.39 (9/100), 2-image: 2051.77 , 5-image: 1653.99, 10-image: 2998.08
#   - (vit_b_lm) 1-image: 1874.83, 2-image: 3205.59 , 5-image: 3196.15, 10-image: 2612.99

# c: cpu32gb
#   - (vit_b) 1-image: 6302.03, 2-image: 29153.65, 5-image: 53502.85, 10-image: 20885.33
#   - (vit_b_lm) 1-image: 21711.23, 2-image: 34443.09, 5-image: 32750.22, 10-image: 19229.85

# d: cpu64gb
#   - (vit_b) 1-image: 11439.01, 2-image: 26225.69, 5-image: 18675.01, 10-image: 50894.71
#   - (vit_b_lm) 1-image: 23291.25, 2-image: 40262.73, 5-image: 33137.21, 10-image: 47490.61


import os
from glob import glob

from micro_sam.util import _load_checkpoint


ROOT = "/scratch/usr/nimanwai/experiments/resource-efficient-finetuning/"


def _load_per_model(checkpoint):
    state, model_state = _load_checkpoint(checkpoint)
    print("Time taken to train for the best epoch:", state["train_time"])
    print("The best epoch attained at:", state["epoch"])
    print()


def check_models(setting, model):
    all_ckpt_paths = sorted(
        glob(os.path.join(ROOT, setting, model, "freeze-*", "*", "checkpoints", "*", "*", "best.pt"))
    )
    for ckpt in all_ckpt_paths:
        print(ckpt)
        _load_per_model(ckpt)


def main():
    settings = ["v100", "rtx5000", "gtx1080", "cpu_32G-mem_16-cores", "cpu_64G-mem_16-cores"]
    models = ["vit_b", "vit_b_lm"]
    for setting in settings:
        for model in models:
            check_models(setting, model)

        breakpoint()


if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action="ignore")
    main()
