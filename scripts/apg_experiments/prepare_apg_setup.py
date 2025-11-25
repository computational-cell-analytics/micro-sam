import os
from glob import glob
from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths

from elf.evaluation import mean_segmentation_accuracy, matching

from micro_sam.evaluation.inference import run_apg
from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.livecell import _get_livecell_paths
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def run_ais(model_type="vit_b_lm"):
    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    image_paths, label_paths = get_livecell_paths(path=data_dir, split="test")

    # Prepare the predictor
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type)

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Run AIS",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run AIS
        segmentation = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=image, verbose=False,
        )

        # Evalate results.
        msa, sas = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
        stats = matching(segmentation, labels)

        msas.append(msa)
        sa50s.append(sas[0])
        precisions.append(stats["precision"])
        recalls.append(stats["recall"])
        f1s.append(stats["f1"])

    print(
        "The final scores are - mSA:", np.mean(msas), "SA50:",  np.mean(sa50s),
        "Precision:", np.mean(precisions), "Recall:", np.mean(recalls), "F1 Score:", np.mean(f1s)
    )


def run_apg_grid_search(model_type="vit_b_lm"):
    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    val_image_paths, val_label_paths = get_livecell_paths(path=data_dir, split="val")
    test_image_paths, _ = get_livecell_paths(path=data_dir, split="test")

    # HACK: Let's test the pipeline super quickly on a very few validation images.
    val_image_paths, val_label_paths = val_image_paths[:10], val_label_paths[:10]

    experiment_folder = "./experiments/livecell"  # HACK: Hard-coded to something random atm.
    prediction_folder = run_apg(
        checkpoint=None,
        model_type=model_type,
        experiment_folder=experiment_folder,
        val_image_paths=val_image_paths,
        val_gt_paths=val_label_paths,
        test_image_paths=test_image_paths,
    )

    # Get the prediction paths
    prediction_paths = sorted(glob(os.path.join(prediction_folder, "*.tif")))
    _, label_paths = _get_livecell_paths(input_folder=data_dir)
    res = run_evaluation(label_paths, prediction_paths, os.path.join(experiment_folder, "results", "apg.csv"))
    print(res)


def main():
    # run_apg_grid_search()
    run_ais()


if __name__ == "__main__":
    main()
