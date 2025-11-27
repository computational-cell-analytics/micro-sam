from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy, matching

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.instance_segmentation import (
    AutomaticPromptGenerator, get_predictor_and_decoder, mask_data_to_segmentation
)

from tukra.inference.get_cellpose import segment_using_cellpose
from tukra.inference.get_instanseg import segment_using_instanseg

from util import get_image_label_paths


NAME = "dsb"


def run_default_ais(model_type="vit_b_lm"):
    image_paths, label_paths = get_image_label_paths(dataset_name=NAME, split="test")

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
    image_paths, label_paths = get_image_label_paths(dataset_name=NAME, split="test")

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
        segmentation = segmenter.generate(
            prompt_selection=["center_distances", "boundary_distances", "connected_components"]
        )
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
    image_paths, label_paths = get_image_label_paths(dataset_name=NAME, split="test")

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
    image_paths, label_paths = get_image_label_paths(dataset_name=NAME, split="test")

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

    image_paths, label_paths = get_image_label_paths(dataset_name=NAME, split="test")

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Run CellSAM",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run CellSAM
        segmentation = cellsam_pipeline(image, use_wsi=False)

        # NOTE: For images where no objects could be found, a weird segmentation is returned.
        if segmentation.ndim == 3 and segmentation.shape[0] == 3:
            segmentation = segmentation[0]

        assert labels.shape == segmentation.shape

        # Evaluate results.
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
    image_paths, label_paths = get_image_label_paths(dataset_name=NAME, split="test")

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


def main():
    run_default_ais()
    # run_default_apg()
    # run_default_amg("vit_b")
    # run_default_amg("vit_b_lm")

    # run_default_cellpose("cyto3")
    # run_default_cellpose("cpsam")
    # run_default_cellsam()

    # run_default_ais("vit_b_histopathology")
    # run_default_instanseg()
    # run_default_cellvit()


if __name__ == "__main__":
    main()
