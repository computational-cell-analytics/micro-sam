import argparse
import os
import warnings
from glob import glob

from micro_sam.util import get_custom_sam_model
from util import evaluate_checkpoint_for_datasets

CHECKPOINT_ROOT = "/scratch-grete/projects/nim00007/sam/LM/generalist"
EXPERIMENT_ROOT = "/scratch-grete/projects/nim00007/sam/experiments/generalists/lm"
# We evaluate these three datasets for the training evolution.
# These are chosen based on observations from preliminary experiments.
# - covid-if: out-of-domain dataset that shows the expected improvement (over vanilla).
# - deepbacs: in domain dataset where we see the biggest gap to the specialist.
# - plantseg-root: out-of-domain dataset that doesn't show an improvement.
DATASETS = ("covid-if", "deepbacs", "plantseg-root")


def evaluate_training_evolution(model_type):
    checkpoints = sorted(glob(
        os.path.join(CHECKPOINT_ROOT, model_type, "*.pt")
    ))
    assert len(checkpoints) > 0

    epochs = []
    for checkpoint in checkpoints:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor, state = get_custom_sam_model(checkpoint, model_type=model_type, return_state=True)
        epoch = state["epoch"] + 1

        if epoch in epochs:
            continue

        print("Run evaluation for", model_type, "epoch", epoch)
        experiment_root = os.path.join(EXPERIMENT_ROOT, f"{model_type}-epoch-{epoch}")
        evaluate_checkpoint_for_datasets(
            None, None, experiment_root, DATASETS,
            run_default_evaluation=True, run_amg=False,
            predictor=predictor,
        )

        epochs.append(epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type")
    args = parser.parse_args()
    evaluate_training_evolution(args.model_type)


if __name__ == "__main__":
    main()
