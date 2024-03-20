import os

from torch_em.data import datasets
from torch_em.transform.label import connected_components

from micro_sam.evaluation.model_comparison import (
    generate_data_for_model_comparison, model_comparison, model_comparison_with_napari
)


ROOT = "/media/anwai/ANWAI/data"


def compare_experiments_for_dataset(
    dataset_name, standard_model, finetuned_model, checkpoint1=None, checkpoint2=None, view_napari=False
):
    output_folder = f"./model_comparison/{dataset_name}/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = fetch_data_loaders(dataset_name)
        generate_data_for_model_comparison(
            loader=loader,
            output_folder=output_folder,
            model_type1=standard_model,
            model_type2=finetuned_model,
            n_samples=10,
            checkpoint1=checkpoint1,
            checkpoint2=checkpoint2
        )

    model_comparison(
        output_folder=output_folder,
        n_images_per_sample=8,
        min_size=100,
        plot_folder=f"./candidates/{dataset_name}",
        point_radius=3,
        outline_dilation=0
    )
    if view_napari:
        model_comparison_with_napari(output_folder, show_points=True)


def fetch_data_loaders(dataset_name):
    if dataset_name == "lucchi":
        loader = datasets.get_lucchi_loader(
            os.path.join(ROOT, "lucchi", "t"), "train", (1, 512, 512), 1, ndim=2, download=True,
            label_transform=connected_components
        )

    elif dataset_name == "livecell":
        loader = datasets.get_livecell_loader(os.path.join(ROOT, "livecell"), "train", (512, 512), 1)

    elif dataset_name == "deepbacs":
        loader = datasets.get_deepbacs_loader(
            os.path.join(ROOT, "deepbacs"), "test", bac_type="mixed", download=True,
            patch_shape=(512, 512), batch_size=1, shuffle=False, n_samples=100
        )

    elif dataset_name == "tissuenet":
        loader = datasets.get_tissuenet_loader(
            os.path.join(ROOT, "tissuenet"), "train", raw_channel="rgb", label_channel="cell",
            patch_shape=(256, 256), batch_size=1, shuffle=True,
        )

    elif dataset_name == "plantseg_root":
        loader = datasets.get_plantseg_loader(
            os.path.join(ROOT, "plantseg"), "root", "test", (1, 512, 512), 1, ndim=2, download=True
        )

    return loader
