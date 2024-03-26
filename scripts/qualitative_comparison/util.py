import os

from torch_em.data import datasets
from torch_em.transform.label import connected_components

from micro_sam.evaluation.model_comparison import (
    generate_data_for_model_comparison, model_comparison
)


ROOT = "/scratch/projects/nim00007/sam/data/"


def compare_experiments_for_dataset(
    dataset_name,
    experiment_folder,
    standard_model,
    finetuned_model,
    checkpoint1=None,
    checkpoint2=None,
    view_napari=False
):
    output_folder = os.path.join(
        experiment_folder, "model_comparison", dataset_name, f"{standard_model}-{finetuned_model}"
    )
    plot_folder = os.path.join(experiment_folder, "candidates", dataset_name)
    if not os.path.exists(output_folder):
        loader = fetch_data_loaders(dataset_name)
        generate_data_for_model_comparison(
            loader=loader,
            output_folder=output_folder,
            model_type1=standard_model,
            model_type2=finetuned_model[:5],
            n_samples=10,
            checkpoint1=checkpoint1,
            checkpoint2=checkpoint2
        )

    model_comparison(
        output_folder=output_folder,
        n_images_per_sample=8,
        min_size=100,
        plot_folder=plot_folder,
        point_radius=3,
        outline_dilation=0
    )
    if view_napari:
        from micro_sam.evaluation.model_comparison import model_comparison_with_napari
        model_comparison_with_napari(output_folder, show_points=True)


def fetch_data_loaders(dataset_name):
    if dataset_name == "lucchi":
        loader = datasets.get_lucchi_loader(
            os.path.join(ROOT, "lucchi", "t"), "test", (1, 512, 512), 1, ndim=2, download=True,
            label_transform=connected_components
        )

    elif dataset_name == "livecell":
        loader = datasets.get_livecell_loader(
            os.path.join(ROOT, "livecell"), "test", (512, 512), 1
        )

    elif dataset_name == "deepbacs":
        loader = datasets.get_deepbacs_loader(
            os.path.join(ROOT, "deepbacs"), "test", bac_type="mixed", download=True,
            patch_shape=(512, 512), batch_size=1, shuffle=False, n_samples=100
        )

    elif dataset_name == "tissuenet":
        loader = datasets.get_tissuenet_loader(
            os.path.join(ROOT, "tissuenet"), "test", raw_channel="rgb", label_channel="cell",
            patch_shape=(256, 256), batch_size=1, shuffle=True,
        )

    elif dataset_name == "plantseg_root":
        loader = datasets.get_plantseg_loader(
            os.path.join(ROOT, "plantseg"), "root", "test", (1, 512, 512), 1, ndim=2, download=True
        )

    elif dataset_name == "neurips_cellseg":
        loader = datasets.get_neurips_cellseg_supervised_loader(
            os.path.join(ROOT, "neurips_cellseg"), "test", (512, 512), 1
        )

    elif dataset_name == "covid_if":
        loader = datasets.get_covid_if_loader(
            os.path.join(ROOT, "covid_if"), (512, 512), 1, ndim=2, download=True
        )

    elif dataset_name == "plantseg_ovules":
        loader = datasets.get_plantseg_loader(
            os.path.join(ROOT, "plantseg"), "ovules", "test", (1, 512, 512), 1, ndim=2, download=True
        )

    elif dataset_name == "hpa":
        loader = datasets.get_hpa_segmentation_loader(
            os.path.join(ROOT, "hpa"), "test", (512, 512), 1, download=True
        )

    elif dataset_name == "lizard":
        loader = datasets.get_lizard_loader(
            os.path.join(ROOT, "lizard"), (512, 512), 1, download=True
        )

    elif dataset_name == "mouse_embryo":
        loader = datasets.get_mouse_embryo_loader(
            os.path.join(ROOT, "mouse_embryo"), "membrane", "test", (1, 512, 512), download=True, ndim=2
        )

    elif dataset_name == "dsb":
        loader = datasets.get_dsb_loader(
            os.path.join(ROOT, "dsb"), "test", (256, 256), 1, download=True
        )

    elif dataset_name == "dynamicnuclearnet":
        loader = datasets.get_dynamicnuclearnet_loader(
            os.path.join(ROOT, "dynamicnuclearnet"), "test", (512, 512), 1
        )

    elif dataset_name == "pannuke":
        loader = datasets.get_pannuke_loader(
            os.path.join(ROOT, "pannuke"), (1, 512, 512), 1, ndim=2, download=True,
        )

    elif dataset_name == "mitoem_rat":
        loader = datasets.get_mitoem_loader(
            os.path.join(ROOT, "mitoem"), "val", (1, 512, 512), 1, "rat", ndim=2,
        )

    elif dataset_name == "mitoem_human":
        loader = datasets.get_mitoem_loader(
            os.path.join(ROOT, "mitoem"), "val", (1, 512, 512), 1, "human", ndim=2,
        )

    elif dataset_name == "platy_nuclei":
        loader = datasets.get_platynereis_nuclei_loader(
            os.path.join(ROOT, "platynereis"), (512, 512), 1, ndim=2, download=True
        )

    elif dataset_name == "platy_cilia":
        loader = datasets.get_platynereis_cilia_loader(
            os.path.join(ROOT, "platynereis"), (512, 512), 1, ndim=2, download=True
        )

    elif dataset_name == "uro_cell":
        loader = datasets.get_uro_cell_loader(
            os.path.join(ROOT, "uro_cell"), "mito", (512, 512), 1, download=True
        )

    elif dataset_name == "nuc_mm_mouse":
        loader = datasets.get_nuc_mm_loader(
            os.path.join(ROOT, "nuc_mm"), "mouse", "val", (512, 512), 1, download=True
        )

    elif dataset_name == "nuc_mm_zebrafish":
        loader = datasets.get_nuc_mm_loader(
            os.path.join(ROOT, "nuc_mm"), "zebrafish", "val", (512, 512), 1, download=True
        )

    elif dataset_name == "sponge_em":
        loader = datasets.get_sponge_em_loader(
            os.path.join(ROOT, "sponge_em"), "instances", (512, 512), 1, download=True
        )

    elif dataset_name == "asem":
        loader = datasets.get_asem_loader(
            os.path.join(ROOT, "asem"), (1, 512, 512), 1, ndim=2, organelles="mito"
        )

    return loader
