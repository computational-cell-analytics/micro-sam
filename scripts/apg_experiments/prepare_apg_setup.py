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
from micro_sam.instance_segmentation import (
    AutomaticPromptGenerator, get_predictor_and_decoder, mask_data_to_segmentation
)

from tukra.inference.get_cellpose import segment_using_cellpose
from tukra.inference.get_instanseg import segment_using_instanseg


def run_default_ais(model_type="vit_b_lm"):
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
            predictor=predictor, segmenter=segmenter, input_path=image, verbose=False, ndim=2,
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


def run_default_apg(model_type="vit_b_lm"):
    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    image_paths, label_paths = get_livecell_paths(path=data_dir, split="test")

    # Prepare the predictor
    predictor, decoder = get_predictor_and_decoder(model_type=model_type)
    segmenter = AutomaticPromptGenerator(predictor, decoder)

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Run APG",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run APG
        segmenter.initialize(image)
        segmentation = segmenter.generate()
        segmentation = mask_data_to_segmentation(segmentation, with_background=False)

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


def run_default_amg(model_type="vit_b_lm"):
    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    image_paths, label_paths = get_livecell_paths(path=data_dir, split="test")

    # Prepare the predictor
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, amg=True)

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Run AMG",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run AIS
        segmentation = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=image, verbose=False, ndim=2,
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


def run_default_cellpose(model_choice):
    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    image_paths, label_paths = get_livecell_paths(path=data_dir, split="test")

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc=f"Run {model_choice}",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run CellPose
        segmentation = segment_using_cellpose(image=image, model_choice=model_choice)

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


def run_default_cellsam():
    from cellSAM import cellsam_pipeline

    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    image_paths, label_paths = get_livecell_paths(path=data_dir, split="test")

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Run CellSAM",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run CellSAM
        segmentation = cellsam_pipeline(image, use_wsi=False)

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


def run_default_instanseg():
    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    image_paths, label_paths = get_livecell_paths(path=data_dir, split="test")

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Run InstanSeg",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run InstanSeg
        segmentation = segment_using_instanseg(
            image, model_type="fluorescence_nuclei_and_cells", target="cells", verbose=False,
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
    # Get the image paths for LIVECell.w
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

    # run_default_ais()
    # run_default_apg()
    # run_default_amg("vit_b")
    # run_default_amg("vit_b_lm")

    # run_default_cellpose("cyto3")
    # run_default_cellpose("cpsam")
    run_default_cellsam()
    # run_default_instanseg()


if __name__ == "__main__":
    main()
