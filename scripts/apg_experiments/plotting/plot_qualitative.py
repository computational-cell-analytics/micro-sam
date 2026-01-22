import os
import sys
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import imageio.v3 as imageio
import matplotlib.pyplot as plt

from torch_em.util.util import get_random_colors

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.evaluation.model_comparison import _overlay_outline
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


sys.path.append("..")


ROOT = "/mnt/vast-nhr/projects/cidas/cca"


def plot_quali(dataset_name):
    # Get APG result paths.
    pred_paths = natsorted(
        glob(
            os.path.join(
                ROOT, "experiments", "micro_sam", "apg_baselines", "cc_without_box",
                "inference", f"{dataset_name}_apg_*", "*"
            )
        )
    )

    # Get the image and label paths
    from util import get_image_label_paths
    image_paths, label_paths = get_image_label_paths(dataset_name=dataset_name, split="test")

    # Get the cellpose model to run CPSAM on the fly.
    from cellpose import models
    model = models.CellposeModel(gpu=True, model_type="cpsam")

    # Let's iterate over each image, label and predictions.
    results = []
    for image_path, label_path, pred_path in tqdm(
        zip(image_paths, label_paths, pred_paths), desc="Qualitative analysis", total=len(image_paths)
    ):
        # Load everything first
        image = imageio.imread(image_path)
        label = imageio.imread(label_path)
        pred = imageio.imread(pred_path)

        assert label.shape == pred.shape

        # Let's run CellPoseSAM to get the best relative results for APG.
        seg, _, _ = model.eval(image)

        # Compare both
        apg_msa = mean_segmentation_accuracy(pred, label)
        cpsam_msa = mean_segmentation_accuracy(seg, label)

        # Next, find the relative score and store it for the dataset
        diff = apg_msa - cpsam_msa
        results.append((diff, image_path, label_path, pred_path, apg_msa, cpsam_msa))

    # Let's fetch top-k where APG wins
    k = 10
    apg_wins = [res for res in results if res[0] > 0]
    apg_wins.sort(key=lambda x: x[0], reverse=True)
    top_k = apg_wins[:k]

    # Prepare other model stuff
    # SAM-related models
    predictor_amg, segmenter_amg = get_predictor_and_segmenter(model_type="vit_b", segmentation_mode="amg")
    predictor_ais, segmenter_ais = get_predictor_and_segmenter(model_type="vit_b_lm", segmentation_mode="ais")

    # Plot each of the top-k images as a separate horizontal triplet
    for rank, (diff, image_path, label_path, pred_path, apg_msa, cpsam_msa) in enumerate(top_k, 1):
        from micro_sam.util import _to_image
        image = _to_image(imageio.imread(image_path))
        gt = imageio.imread(label_path)

        # APG prediction is read from disk, CPSAM is recomputed for plotting
        amg_pred = automatic_instance_segmentation(
            input_path=image, verbose=False, ndim=2, predictor=predictor_amg, segmenter=segmenter_amg,
        )
        ais_pred = automatic_instance_segmentation(
            input_path=image, verbose=False, ndim=2, predictor=predictor_ais, segmenter=segmenter_ais,
        )
        apg_pred = imageio.imread(pred_path)
        cpsam_pred, _, _ = model.eval(image)

        from cellSAM import cellsam_pipeline
        cellsam_pred = cellsam_pipeline(image, use_wsi=False)

        # Prepare image as expected (normalize and overlay labels)
        curr_image = _overlay_outline(image, gt, outline_dilation=1)

        fig, axes = plt.subplots(1, 6, figsize=(15, 5), constrained_layout=True)
        axes[0].imshow(curr_image)
        axes[0].axis("off")

        axes[1].imshow(amg_pred, cmap=get_random_colors(amg_pred))
        axes[1].axis("off")

        axes[2].imshow(ais_pred, cmap=get_random_colors(ais_pred))
        axes[2].axis("off")

        axes[3].imshow(cpsam_pred, cmap=get_random_colors(cpsam_pred))
        axes[3].axis("off")

        axes[4].imshow(cellsam_pred, cmap=get_random_colors(cellsam_pred))
        axes[4].axis("off")

        axes[6].imshow(apg_pred, cmap=get_random_colors(apg_pred))
        axes[6].axis("off")

        os.makedirs("./quali_figures", exist_ok=True)
        plt.savefig(f"./quali_figures/{dataset_name}_{rank}.png", dpi=200)
        plt.close()

        breakpoint()


def main():
    plot_quali("dsb")


if __name__ == "__main__":
    main()
