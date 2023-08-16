import os

import imageio.v3 as imageio
import micro_sam.evaluation.model_comparison as comparison
import torch_em

from util import get_data_paths, EM_DATASETS

OUTPUT_ROOT = "/scratch-grete/projects/nim00007/sam/experiments/model_comparison"


def _get_patch_shape(path):
    im_shape = imageio.imread(path).shape[:2]
    patch_shape = tuple(min(sh, 512) for sh in im_shape)
    return patch_shape


def get_loader(dataset):
    image_paths, gt_paths = get_data_paths(dataset, split="test")
    image_paths, gt_paths = image_paths[:100], gt_paths[:100]

    label_transform = torch_em.transform.label.connected_components
    loader = torch_em.default_segmentation_loader(
        image_paths, None, gt_paths, None,
        batch_size=1, patch_shape=_get_patch_shape(image_paths[0]),
        shuffle=True, n_samples=25, label_transform=label_transform,
    )
    return loader


def generate_comparison_for_dataset(dataset, model1, model2):
    output_folder = os.path.join(OUTPUT_ROOT, dataset)
    if os.path.exists(output_folder):
        return
    print("Generate model comparison data for", dataset)
    loader = get_loader(dataset)
    comparison.generate_data_for_model_comparison(loader, output_folder, model1, model2, n_samples=25)


# TODO
def create_comparison_images():
    pass


def generate_comparison_em():
    model1 = "vit_h"
    model2 = "vit_h_em"
    for dataset in EM_DATASETS:
        generate_comparison_for_dataset(dataset, model1, model2)
        create_comparison_images()


def main():
    # generate_comparison_lm()
    generate_comparison_em()


if __name__ == "__main__":
    main()
