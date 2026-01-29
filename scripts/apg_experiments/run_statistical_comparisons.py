import os
from tqdm import tqdm

import pandas as pd
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy, matching

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from util import get_image_label_paths


def store_quantitative_comparisons(dataset_name, model_type):

    # Set the output directory.
    output_dir = "quantitative_results"
    output_path = os.path.join(output_dir, f"{dataset_name}_{model_type}_metrics.csv")
    if os.path.exists(output_path):
        print("The results have already been computed")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get the image and label paths.
    image_paths, label_paths = get_image_label_paths(dataset_name=dataset_name, split="test")

    # Prepare the model-level predictors for micro-sam
    predictor_ais, segmenter_ais = get_predictor_and_segmenter(model_type=model_type, segmentation_mode="ais")
    predictor_apg, segmenter_apg = get_predictor_and_segmenter(model_type=model_type, segmentation_mode="apg")

    # Get the cellpose model to run CPSAM on the fly.
    from cellpose import models
    cpsam_model = models.CellposeModel(gpu=True, model_type="cpsam")

    # Initialize list to store all results
    results = []

    for image_path, label_path in tqdm(
        zip(image_paths, label_paths), desc="Store quantitative results per image", total=len(image_paths),
    ):
        # Get the image and labels
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        # Store the filenames
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)

        # Initialize row for this image-label pair
        row = {'image': image_name, 'label': label_name}

        # Run AIS
        segmentation = automatic_instance_segmentation(
            input_path=image, verbose=False, ndim=2, predictor=predictor_ais, segmenter=segmenter_ais,
        )
        # Calculate the scores for AIS.
        msa, sas = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
        stats = matching(segmentation, labels)

        row['ais_msa'] = msa
        row['ais_sa50'] = sas[0]
        row['ais_precision'] = stats['precision']
        row['ais_recall'] = stats['recall']
        row['ais_f1'] = stats['f1']

        # Run APG
        segmentation = automatic_instance_segmentation(
            input_path=image, verbose=False, ndim=2, predictor=predictor_apg, segmenter=segmenter_apg,
        )
        # Calculate the scores for APG.
        try:
            msa, sas = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
        except ValueError:
            breakpoint()
        stats = matching(segmentation, labels)

        row['apg_msa'] = msa
        row['apg_sa50'] = sas[0]
        row['apg_precision'] = stats['precision']
        row['apg_recall'] = stats['recall']
        row['apg_f1'] = stats['f1']

        # Run CellPoseSAM
        segmentation, _, _ = cpsam_model.eval(image)
        # Calculate the scores for CPSAM.
        msa, sas = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
        stats = matching(segmentation, labels)

        row['cpsam_msa'] = msa
        row['cpsam_sa50'] = sas[0]
        row['cpsam_precision'] = stats['precision']
        row['cpsam_recall'] = stats['recall']
        row['cpsam_f1'] = stats['f1']

        # Add row to results
        results.append(row)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Save to CSV
    output_path = os.path.join(output_dir, f"{dataset_name}_{model_type}_metrics.csv")
    df.to_csv(output_path, index=False)


def main(args):
    store_quantitative_comparisons(args.dataset_name, args.model_type)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, required=True)
    args = parser.parse_args()
    main(args)
