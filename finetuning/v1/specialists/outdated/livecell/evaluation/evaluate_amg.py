import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.livecell import run_livecell_amg
from util import DATA_ROOT, VANILLA_MODELS, get_pred_and_gt_paths, get_default_arguments


def run_amg(model_type, checkpoint, experiment_folder):
    input_folder = DATA_ROOT

    if checkpoint is None:
        checkpoint = VANILLA_MODELS[model_type]

    prediction_folder = run_livecell_amg(
        checkpoint,
        input_folder,
        model_type,
        experiment_folder,
        n_val_per_cell_type=25,
    )
    return prediction_folder


def eval_amg(prediction_folder, experiment_folder):
    print("Evaluating", prediction_folder)
    pred_paths, gt_paths = get_pred_and_gt_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "amg.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()

    prediction_folder = run_amg(args.model, args.checkpoint, args.experiment_folder)
    eval_amg(prediction_folder, args.experiment_folder)


if __name__ == "__main__":
    main()
