import os

import matplotlib.pyplot as plt

import torch

from torch_em.data import MinInstanceSampler
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data.datasets import get_livecell_loader, get_mitolab_loader

from micro_sam.evaluation.model_comparison import _enhance_image

from compare_benchmarks import get_random_colors


ROOT = "/scratch/projects/nim00007/sam/data/"


def _get_dataloaders(dataset):
    label_dtype = torch.float32

    label_transform = PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=True,
        min_size=25,
    )

    if dataset == "livecell":
        loader = get_livecell_loader(
            path=os.path.join(ROOT, "livecell"),
            split="train",
            patch_shape=(512, 512),
            batch_size=1,
            label_transform=label_transform,
            label_dtype=label_dtype,
            sampler=MinInstanceSampler(10),
        )

    elif dataset == "mitolab":
        loader = get_mitolab_loader(
            path=os.path.join(ROOT, "mitolab"),
            split="train",
            patch_shape=(512, 512),
            batch_size=1,
            label_transform=label_transform,
            label_dtype=label_dtype,
            sampler=MinInstanceSampler(10),
        )
        loader.dataset.max_sampling_attempts = 5000

    else:
        raise ValueError

    return loader


def _get_plots(dataset):
    loader = _get_dataloaders(dataset)

    cmap_for_dist = "nipy_spectral"

    for i, (x, y) in enumerate(loader):
        x, y = x.squeeze().numpy(), y.squeeze().numpy()
        instances, binary, center_dist, boundary_dist = y

        fig, ax = plt.subplots(1, 5, figsize=(30, 20))

        ax[0].imshow(_enhance_image(x), cmap="gray")
        ax[0].axis("off")

        ax[1].imshow(center_dist, cmap=cmap_for_dist, interpolation="nearest")
        ax[1].axis("off")

        ax[2].imshow(boundary_dist, cmap=cmap_for_dist, interpolation="nearest")
        ax[2].axis("off")

        ax[3].imshow(binary, cmap=cmap_for_dist, interpolation="nearest")
        ax[3].axis("off")

        ax[4].imshow(instances, cmap=get_random_colors(instances), interpolation="nearest")
        ax[4].axis("off")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"./figs/{dataset}/{dataset}_{i}.svg", bbox_inches="tight")
        plt.close()


def main():
    # _get_plots("livecell")
    _get_plots("mitolab")


if __name__ == "__main__":
    main()
