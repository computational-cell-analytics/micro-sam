import argparse
import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.livecell import run_livecell_amg
from util import DATA_ROOT, get_checkpoint, get_experiment_folder, get_pred_and_gt_paths


def run_amg(name, model_type, checkpoint):
    if checkpoint is None:
        checkpoint, model_type = get_checkpoint(name)
    experiment_folder = get_experiment_folder(name)
    input_folder = DATA_ROOT
    prediction_folder = run_livecell_amg(
        checkpoint,
        input_folder,
        model_type,
        experiment_folder,
        n_val_per_cell_type=25,
    )
    return prediction_folder


def eval_amg(name, prediction_folder):
    print("Evaluating", prediction_folder)
    pred_paths, gt_paths = get_pred_and_gt_paths(prediction_folder)
    save_path = os.path.join(get_experiment_folder(name), "results", "amg.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument(
        "-m", "--model", type=str,  # options: "vit_h", "vit_h_generalist", "vit_h_specialist"
        help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = parser.parse_args()

    name = args.name

    prediction_folder = run_amg(name, args.model, args.checkpoint)
    eval_amg(name, prediction_folder)


if __name__ == "__main__":
    main()
