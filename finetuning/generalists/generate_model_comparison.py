import os

import imageio.v3 as imageio
import micro_sam.evaluation.model_comparison as comparison
import torch_em

from util import get_data_paths, EM_DATASETS, LM_DATASETS

OUTPUT_ROOT = "/scratch-grete/projects/nim00007/sam/experiments/model_comparison"


def _get_patch_shape(path):
    im_shape = imageio.imread(path).shape[:2]
    patch_shape = tuple(min(sh, 512) for sh in im_shape)
    return patch_shape


def raw_trafo(raw):
    raw = raw.transpose((2, 0, 1))
    print(raw.shape)
    return raw


def get_loader(dataset):
    image_paths, gt_paths = get_data_paths(dataset, split="test")
    image_paths, gt_paths = image_paths[:100], gt_paths[:100]

    with_channels = dataset in ("hpa", "lizard")

    label_transform = torch_em.transform.label.connected_components
    loader = torch_em.default_segmentation_loader(
        image_paths, None, gt_paths, None,
        batch_size=1, patch_shape=_get_patch_shape(image_paths[0]),
        shuffle=True, n_samples=25, label_transform=label_transform,
        with_channels=with_channels, is_seg_dataset=not with_channels
    )
    return loader


def generate_comparison_for_dataset(dataset, model1, model2):
    output_folder = os.path.join(OUTPUT_ROOT, dataset)
    if os.path.exists(output_folder):
        return output_folder
    print("Generate model comparison data for", dataset)
    loader = get_loader(dataset)
    comparison.generate_data_for_model_comparison(loader, output_folder, model1, model2, n_samples=25)
    return output_folder


def create_comparison_images(output_folder, dataset):
    plot_folder = os.path.join(OUTPUT_ROOT, "images", dataset)
    if os.path.exists(plot_folder):
        return
    comparison.model_comparison(
        output_folder, n_images_per_sample=25, min_size=100,
        plot_folder=plot_folder, outline_dilation=1
    )


def generate_comparison_em():
    model1 = "vit_h"
    model2 = "vit_h_em"
    for dataset in EM_DATASETS:
        folder = generate_comparison_for_dataset(dataset, model1, model2)
        create_comparison_images(folder, dataset)


def generate_comparison_lm():
    model1 = "vit_h"
    model2 = "vit_h_lm"
    for dataset in LM_DATASETS:
        folder = generate_comparison_for_dataset(dataset, model1, model2)
        create_comparison_images(folder, dataset)


def main():
    generate_comparison_lm()
    # generate_comparison_em()


if __name__ == "__main__":
    main()
