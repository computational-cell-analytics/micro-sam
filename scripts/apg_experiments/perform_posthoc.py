import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy

from util import get_image_label_paths


def posthoc_analysis(dataset_name):
    # Let's check inference results of APG and check where APG does the worst.
    image_paths, label_paths = get_image_label_paths(
        dataset_name=dataset_name, split="test",
    )

    # Get the predicted paths
    base_dir = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam/apg_experiments"
    prediction_paths = natsorted(
        glob(os.path.join(base_dir, dataset_name, "apg", "inference", "*.tif"))
    )

    # Make a simple check: the images with worst mSA will be stored locally.
    results = []
    for i, (curr_label, curr_seg) in tqdm(
        enumerate(zip(label_paths, prediction_paths)), total=len(label_paths), desc="Post-hoc",
    ):
        label = imageio.imread(curr_label)
        seg = imageio.imread(curr_seg)

        # Evaluate the images and store them in one place.
        results.append((mean_segmentation_accuracy(seg, label), i))

    # Let's assort the results
    results.sort(key=lambda x: x[0])

    # Find the k worst results
    worst_k = 10  # The number of bad segmentations to check for.
    worst_res = results[:worst_k]

    # Finally, store comparisons.
    os.makedirs(f"figures/{dataset_name}", exist_ok=True)
    for rank, (msa, i) in enumerate(worst_res, start=1):
        im = imageio.imread(image_paths[i]).astype("uint8")
        label = imageio.imread(label_paths[i])
        seg = imageio.imread(prediction_paths[i])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 10))

        if dataset_name == "tissuenet":
            im = im[:, :, 0]
            print(im.max(), im.min())
            ax[0].imshow(im, cmap="gray")
        elif dataset_name == "pannuke":
            ax[0].imshow(im)
        else:
            ax[0].imshow(im, cmap="gray")

        ax[0].axis("off")
        ax[0].set_title("Image")

        ax[1].imshow(label)
        ax[1].set_title(f"GT (rank {rank}, mSA={msa:.3f})")
        ax[1].axis("off")

        ax[2].imshow(seg)
        ax[2].set_title("Prediction")
        ax[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"figures/{dataset_name}/bad_res_{rank}.png")


def main():
    posthoc_analysis("tissuenet")


if __name__ == "__main__":
    main()
