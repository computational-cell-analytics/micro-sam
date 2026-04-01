import os
from glob import glob
from natsort import natsorted

import pandas as pd

from micro_sam.util import _load_checkpoint


ROOT = "/media/anwai/ANWAI/micro-sam/resource-efficient-finetuning/"


def _stats_per_model(checkpoint):
    state, model_state = _load_checkpoint(checkpoint)
    time_in_seconds = state["train_time"]
    minutes, seconds = divmod(time_in_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    total_time = "%d:%02d:%02d" % (hours, minutes, seconds)

    # Let's create a dataframe and store all the results.
    desired_path = checkpoint[len(ROOT):]
    _splits = desired_path.split("/")
    experiment_name = _splits[0]
    finetuned_model = _splits[1] + "-" + _splits[2]
    n_images = _splits[4].split("-")[0]

    outputs = {
        "experiment_name": experiment_name,
        "finetuned_model": finetuned_model,
        "number_of_images": n_images,
        "best_epoch": state["epoch"],
        "time_in_minutes": total_time,
    }
    outputs = pd.DataFrame([outputs])

    return outputs


def check_models(setting, model):
    all_ckpt_paths = natsorted(
        glob(os.path.join(ROOT, setting, model, "*", "freeze-*", "*", "checkpoints", "*", "*", "best.pt"))
    )
    all_outputs = [_stats_per_model(ckpt) for ckpt in all_ckpt_paths]
    outputs = pd.concat(all_outputs, ignore_index=True)
    print(outputs)


def main():
    settings = ["V100", "RTX5000", "GTX1080", "cpu_32G-mem_16-cores", "cpu_64G-mem_16-cores"]
    models = ["vit_b", "vit_b_lm"]
    for setting in settings:
        for model in models:
            check_models(setting, model)

        breakpoint()


if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action="ignore")
    main()
