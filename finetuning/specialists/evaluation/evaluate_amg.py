import argparse
import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.util import run_amg

from util import get_experiment_folder, DATA_ROOT


def run_specialist_amg(name, model_type, checkpoint, domain):
    experiment_folder = get_experiment_folder(domain, name, model_type)
    val_image_paths, val_gt_paths, test_image_paths = DATA_ROOT[domain]
    prediction_folder = run_amg(
        checkpoint, model_type, experiment_folder,
        val_image_paths, val_gt_paths, test_image_paths
    )
    return prediction_folder


def eval_specialist_amg(name, prediction_folder, model_type, domain):
    print("Evaluating", prediction_folder)
    pred_paths, gt_paths = ...
    save_path = os.path.join(get_experiment_folder(domain, name, model_type), "results", "amg.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, required=True, help="Name of the specialist domain.")
    parser.add_argument("-n", "--name", required=True, help="Name for the experimental folder.")
    parser.add_argument("-m", "--model", type=str, help="Provide the model type to initialize the predictor.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="The finetuned specialist checkpoint.")
    args = parser.parse_args()

    name = args.name
    assert args.domain in DATA_ROOT.keys(), f"{args.domain} is not the correct specialist domain, please check."

    checkpoint = f"/scratch/usr/nimanwai/micro-sam/checkpoints/{args.model}/{args.domain}_sam/best.pt"
    assert os.path.exists(checkpoint), checkpoint

    prediction_folder = run_specialist_amg(name, args.model, checkpoint, args.domain)
    eval_specialist_amg(name, prediction_folder)


if __name__ == "__main__":
    main()
