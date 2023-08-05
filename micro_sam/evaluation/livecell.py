import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

from tqdm import tqdm

from . import inference, evaluation
from .experiments import default_experiment_settings, full_experiment_settings

CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


#
# Inference
#


def _get_livecell_paths(input_folder):
    assert os.path.exists(input_folder), "Please download the LIVECell Dataset"
    img_dir = os.path.join(input_folder, "images", "livecell_test_images")
    assert os.path.exists(img_dir), "The LiveCELL Dataset is incomplete"
    gt_dir = os.path.join(input_folder, "annotations", "livecell_test_images")
    assert os.path.exists(gt_dir), "The LiveCELL Dataset is incomplete"
    image_paths, gt_paths = [], []
    for ctype in CELL_TYPES:
        for img_path in glob(os.path.join(img_dir, f"{ctype}*")):
            image_paths.append(img_path)
            img_name = os.path.basename(img_path)
            gt_path = os.path.join(gt_dir, ctype, img_name)
            assert os.path.exists(gt_path), gt_path
            gt_paths.append(gt_path)
    return image_paths, gt_paths


def livecell_inference(
    checkpoint,
    input_folder,
    model_type,
    experiment_folder,
    use_points,
    use_boxes,
    n_positives=None,
    n_negatives=None,
    prompt_folder=None,
    predictor=None,
):
    """Run inference for livecell with a fixed prompt setting.
    """
    image_paths, gt_paths = _get_livecell_paths(input_folder)
    if predictor is None:
        predictor = inference.get_predictor(checkpoint, model_type)

    if use_boxes and use_points:
        assert (n_positives is not None) and (n_negatives is not None)
        setting_name = f"box/p{n_positives}-n{n_negatives}"
    elif use_boxes:
        setting_name = "box/p0-n0"
    elif use_points:
        assert (n_positives is not None) and (n_negatives is not None)
        setting_name = f"points/p{n_positives}-n{n_negatives}"
    else:
        raise ValueError("You need to use at least one of point or box prompts.")

    # we organize all folders with data from this experiment beneath 'experiment_folder'
    prediction_folder = os.path.join(experiment_folder, setting_name)  # where the predicted segmentations are saved
    os.makedirs(prediction_folder, exist_ok=True)
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    # NOTE: we can pass an external prompt folder, to make re-use prompts from another experiment
    # for reproducibility / fair comparison of results
    if prompt_folder is None:
        prompt_folder = os.path.join(experiment_folder, "prompts")
        os.makedirs(prompt_folder, exist_ok=True)

    inference.run_inference_with_prompts(
        predictor,
        image_paths,
        gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_folder,
        prompt_save_dir=prompt_folder,
        use_points=use_points,
        use_boxes=use_boxes,
        n_positives=n_positives,
        n_negatives=n_negatives,
    )


def _run_multiple_prompt_settings(args, prompt_settings):

    predictor = inference.get_predictor(args.checkpoint, args.model_type)
    image_paths, _ = _get_livecell_paths()
    embedding_folder = os.path.join(args.experiment_folder, "embeddings")
    inference.precompute_image_embeddings(predictor, image_paths, embedding_folder)

    for settings in prompt_settings:
        livecell_inference(
            args.ckpt,
            args.input,
            args.model,
            args.experiment_folder,
            use_points=settings["use_points"],
            use_boxes=settings["use_boxes"],
            n_positives=settings["n_positives"],
            n_negatives=settings["n_negatives"],
            prompt_folder=args.prompt_folder,
            predictor=predictor
        )


def _run_single_prompt_setting(args):
    livecell_inference(
        args.ckpt,
        args.input,
        args.model,
        args.experiment_folder,
        args.points,
        args.box,
        args.positive,
        args.negative,
        args.prompt_folder,
    )


# TODO add grid-search / automatic instance segmentation
# TODO enable over-riding paths for convenience (to set paths on GRETE)
def run_livecell_inference():
    parser = argparse.ArgumentParser()

    # the checkpoint, input and experiment folder
    parser.add_argument("-c", "--ckpt", type=str, required=True,
                        help="Provide model checkpoints (vanilla / finetuned).")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Provide the data directory for LIVECell Dataset.")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True,
                        help="Provide the path where all data for the inference run will be stored.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Pass the checkpoint-specific model name being used for inference.")

    # the experiment type:
    # - standard: run the following prompt settings:
    #   - p1-n0
    #   - p2-n4
    #   - box
    # - full:
    #   - TODO define it
    # if none of the two are active then the prompt setting arguments will be parsed
    # and used to run inference for a single prompt setting
    parser.add_argument("-f", "--full_experiment", action="store_true")
    parser.add_argument("-s", "--standard_experiment", action="store_true")

    # the prompt settings for an individual inference run
    parser.add_argument("--box", action="store_true", help="Activate box-prompted based inference")
    parser.add_argument("--points", action="store_true", help="Activate point-prompt based inference")
    parser.add_argument("-p", "--positive", type=int, default=1, help="No. of positive prompts")
    parser.add_argument("-n", "--negative", type=int, default=0, help="No. of negative prompts")

    # optional external prompt folder
    parser.add_argument("--prompt_folder", help="")

    args = parser.parse_args()
    if args.full_experiment:
        prompt_settings = full_experiment_settings(args.box)
        _run_multiple_prompt_settings(args, prompt_settings)
    elif args.standard_experiment:
        prompt_settings = default_experiment_settings()
        _run_multiple_prompt_settings(args, prompt_settings)
    else:
        _run_single_prompt_setting(args)


#
# Evaluation
#


def evaluate_livecell_predictions(gt_dir, pred_dir, verbose):
    assert os.path.exists(gt_dir), gt_dir
    assert os.path.exists(pred_dir), pred_dir

    msas, sa50s, sa75s = [], [], []
    msas_ct, sa50s_ct, sa75s_ct = [], [], []

    for ct in tqdm(CELL_TYPES, desc="Evaluate livecell predictions", disable=not verbose):

        gt_pattern = os.path.join(gt_dir, f"{ct}/*.tif")
        gt_paths = glob(gt_pattern)
        assert len(gt_paths) > 0, "gt_pattern"

        pred_paths = [
            os.path.join(pred_dir, os.path.basename(path)) for path in gt_paths
        ]

        this_msas, this_sa50s, this_sa75s = evaluation._run_evaluation(
            gt_paths, pred_paths, False
        )

        msas.extend(this_msas), sa50s.extend(this_sa50s), sa75s.extend(this_sa75s)
        msas_ct.append(np.mean(this_msas))
        sa50s_ct.append(np.mean(this_sa50s))
        sa75s_ct.append(np.mean(this_sa75s))

    result_dict = {
        "cell_type": CELL_TYPES + ["Total"],
        "msa": msas_ct + [np.mean(msas)],
        "sa50": sa50s_ct + [np.mean(sa50s_ct)],
        "sa75": sa75s_ct + [np.mean(sa75s_ct)],
    }
    df = pd.DataFrame.from_dict(result_dict)
    df = df.round(decimals=4)
    return df


def run_livecell_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Provide the data directory for LIVECell Dataset"
    )
    parser.add_argument(
        "-e", "--experiment_folder", required=True,
        help="Provide the path where the inference data is stored."
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Force recomputation of already cached eval results."
    )
    args = parser.parse_args()

    gt_dir = os.path.join(args.input, "annotations", "livecell_test_images")
    assert os.path.exists(gt_dir), "The LiveCELL Dataset is incomplete"

    experiment_folder = args.experiment_folder
    save_root = os.path.join(experiment_folder, "results")

    inference_root_names = ["points", "box"]
    for inf_root in inference_root_names:

        pred_folders = sorted(glob(os.path.join(experiment_folder, inf_root, "*")))
        save_folder = os.path.join(save_root, inf_root)
        os.makedirs(save_folder, exist_ok=True)

        for pred_folder in tqdm(pred_folders, desc=f"Evaluate predictions for {inf_root} prompt settings"):
            exp_name = os.path.basename(pred_folder)
            save_path = os.path.join(save_folder, f"{exp_name}.csv")
            if os.path.exists(save_path) and not args.force:
                continue
            results = evaluate_livecell_predictions(gt_dir, pred_folder, verbose=False)
            results.to_csv(save_path, index=False)
