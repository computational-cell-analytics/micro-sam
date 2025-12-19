import argparse
import json
import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import imageio.v3 as imageio
import numpy as np

from elf.evaluation import mean_segmentation_accuracy

from util import get_image_label_paths

DEFAULT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam/apg_cc"


def posthoc_analysis(dataset_name, input_root, worst_k, output_name):
    import matplotlib.pyplot as plt

    # Check the dataset.
    dataset_dir = os.path.join(input_root, dataset_name)
    if not os.path.exists(dataset_dir):
        datasets = [os.path.basename(path) for path in glob(os.path.join(input_root, "*"))]
        raise ValueError(f"Invalid dataset: {dataset_name}. Available datasets: {datasets}")

    # Let's check inference results of APG and check where APG does the worst.
    try:
        ds_name = dataset_name[:(dataset_name.find("apg") - 1)] if "apg" in dataset_name else dataset_name
        image_paths, label_paths = get_image_label_paths(dataset_name=ds_name, split="test")
    except Exception:
        print("Cannot get the dataset:", ds_name)
        return

    # Get all the predictions.
    # prediction_paths = natsorted(glob(os.path.join(dataset_dir, "apg", "inference", "*.tif")))
    prediction_paths = natsorted(glob(os.path.join(dataset_dir, "*.tif")))
    try:
        assert len(label_paths) == len(prediction_paths), \
            f"Inconsistent labels and predictions: {len(label_paths)}, {len(prediction_paths)}"
    except AssertionError:
        print("Predictions are missing for", dataset_name)
        return

    # Make a simple check: the images with worst mSA will be stored locally.
    results = []
    for i, (curr_label, curr_seg) in tqdm(
        enumerate(zip(label_paths, prediction_paths)), total=len(label_paths), desc=f"Post-hoc for {dataset_name}",
    ):
        label = imageio.imread(curr_label)
        seg = imageio.imread(curr_seg)

        # Evaluate the images and store them in one place.
        results.append((mean_segmentation_accuracy(seg, label), i))

    msas = [res[0] for res in results]
    print("Overall result for", dataset_name, "based on", len(results), "images:")
    print("mSA:", np.mean(msas), "+-", np.std(msas))

    # Let's assort the results
    results.sort(key=lambda x: x[0])

    # Find the k worst results
    worst_res = results[:worst_k]

    # Store the comparisons and keep track of the respective paths.
    result_info = {
        "image_paths": [],
        "label_paths": [],
        "prediction_paths": [],
        "msas": [],
    }
    out_folder = f"{output_name}/figures/{dataset_name}"
    os.makedirs(out_folder, exist_ok=True)
    for rank, (msa, i) in enumerate(worst_res, start=1):
        image_path, label_path, pred_path = image_paths[i], label_paths[i], prediction_paths[i]
        fname = os.path.basename(image_path)
        im = imageio.imread(image_path).astype("uint8")
        label = imageio.imread(label_path)
        seg = imageio.imread(pred_path)

        result_info["image_paths"].append(image_path)
        result_info["label_paths"].append(label_path)
        result_info["prediction_paths"].append(pred_path)
        result_info["msas"].append(msa)

        fig, ax = plt.subplots(1, 3, figsize=(15, 10))

        if dataset_name == "tissuenet":
            im = im[:, :, 0]
            ax[0].imshow(im, cmap="gray")
        elif dataset_name == "pannuke":
            ax[0].imshow(im)
        else:
            ax[0].imshow(im, cmap="gray")

        ax[0].axis("off")
        ax[0].set_title(fname)

        ax[1].imshow(label)
        ax[1].set_title(f"GT (rank {rank}, mSA={msa:.3f})")
        ax[1].axis("off")

        ax[2].imshow(seg)
        ax[2].set_title("Prediction")
        ax[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_name}/figures/{dataset_name}/bad_res_{rank:03}.png")
        plt.close()

    with open(os.path.join(out_folder, "summary.json"), "w") as f:
        json.dump(result_info, f)


def run_posthoc_analysis(datasets, input_root, worst_k):
    output_name = os.path.basename(input_root)
    if os.path.exists(os.path.join(input_root, "inference")):
        input_root = os.path.join(input_root, "inference")
    if datasets is None:
        datasets = [os.path.basename(path) for path in glob(os.path.join(input_root, "*"))]
    for dataset in datasets:
        posthoc_analysis(dataset, input_root, worst_k, output_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datasets", nargs="+")
    parser.add_argument("-i", "--input_root", default=DEFAULT_ROOT)
    parser.add_argument("-k", "--worst_k", type=int, default=10)
    args = parser.parse_args()
    run_posthoc_analysis(args.datasets, args.input_root, args.worst_k)


if __name__ == "__main__":
    main()
