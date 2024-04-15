import os

import numpy as np

from elf.io import open_file

from torch_em.data import datasets
from torch_em.data import MinInstanceSampler
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
    intermediate_model=None,
    checkpoint1=None,
    checkpoint2=None,
    checkpoint3=None,
    view_napari=False,
    n_samples=20,
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
            model_type3=intermediate_model[:5],
            n_samples=n_samples,
            checkpoint1=checkpoint1,
            checkpoint2=checkpoint2,
            checkpoint3=checkpoint3,
        )

    model_comparison(
        output_folder=output_folder,
        n_images_per_sample=10,
        min_size=100,
        plot_folder=plot_folder,
        point_radius=3,
        outline_dilation=0,
        have_model3=intermediate_model is not None
    )
    if view_napari:
        from micro_sam.evaluation.model_comparison import model_comparison_with_napari
        model_comparison_with_napari(output_folder, show_points=True)


def compute_platy_rois(root, sample_ids, ignore_label, file_template, label_key):
    rois = {}
    for sample_id in sample_ids:
        path = os.path.join(root, (file_template % sample_id))
        with open_file(path, "r") as f:
            labels = f[label_key][:]
        valid_coordinates = np.where(labels != ignore_label)
        roi = tuple(slice(
            int(coord.min()), int(coord.max()) + 1
        ) for coord in valid_coordinates)
        rois[sample_id] = roi
    return rois


def fetch_data_loaders(dataset_name):
    sampler = MinInstanceSampler()

    # LM LOADERS

    if dataset_name == "livecell":
        loader = datasets.get_livecell_loader(
            path=os.path.join(ROOT, "livecell"), split="test", patch_shape=(512, 512),
            batch_size=1, shuffle=True, sampler=sampler,
        )

    elif dataset_name == "deepbacs":
        loader = datasets.get_deepbacs_loader(
            path=os.path.join(ROOT, "deepbacs"), split="test", bac_type="mixed", patch_shape=(512, 512),
            batch_size=1, shuffle=True, n_samples=100, sampler=sampler,
        )

    elif dataset_name == "tissuenet":
        loader = datasets.get_tissuenet_loader(
            path=os.path.join(ROOT, "tissuenet"), split="test", raw_channel="rgb",  # rgb / nucleus
            label_channel="cell", patch_shape=(256, 256), batch_size=1, shuffle=True, sampler=sampler,
        )

    elif dataset_name == "plantseg_root":
        loader = datasets.get_plantseg_loader(
            path=os.path.join(ROOT, "plantseg"), name="root", split="test", patch_shape=(1, 512, 512),
            batch_size=1, ndim=2, shuffle=True, sampler=sampler,
        )

    elif dataset_name == "neurips_cellseg":
        loader = datasets.get_neurips_cellseg_supervised_loader(
            root=os.path.join(ROOT, "neurips-cell-seg", "zenodo"), split="test",
            patch_shape=(512, 512), batch_size=1, shuffle=True, sampler=sampler,
        )

    elif dataset_name == "covid_if":
        loader = datasets.get_covid_if_loader(
            path=os.path.join(ROOT, "covid_if"), patch_shape=(512, 512),
            batch_size=1, ndim=2, shuffle=True, sampler=sampler,
        )

    elif dataset_name == "plantseg_ovules":
        loader = datasets.get_plantseg_loader(
            path=os.path.join(ROOT, "plantseg"), name="ovules", split="train", patch_shape=(1, 512, 512),
            batch_size=1, ndim=2, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "hpa":
        loader = datasets.get_hpa_segmentation_loader(
            path=os.path.join(ROOT, "hpa", "test"), split="train", patch_shape=(512, 512), download=True,
            batch_size=1, sampler=sampler, shuffle=True, channels=["protein", "nuclei", "er"],
        )

    elif dataset_name == "lizard":
        loader = datasets.get_lizard_loader(
            path=os.path.join(ROOT, "lizard"), patch_shape=(512, 512),
            batch_size=1, shuffle=True, sampler=sampler
        )

    elif dataset_name == "mouse_embryo":
        loader = datasets.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse-embryo"), name="membrane", split="train",
            patch_shape=(1, 512, 512), batch_size=1, ndim=2, shuffle=True, sampler=sampler,
        )

    elif dataset_name == "dynamicnuclearnet":
        loader = datasets.get_dynamicnuclearnet_loader(
            path=os.path.join(ROOT, "dynamicnuclearnet"), split="test", patch_shape=(512, 512),
            batch_size=1, download=False, sampler=sampler,
        )

    elif dataset_name == "pannuke":
        loader = datasets.get_pannuke_loader(
            path=os.path.join(ROOT, "pannuke"), patch_shape=(1, 512, 512),
            batch_size=1, ndim=2, shuffle=True, sampler=sampler,
        )

    # EM LOADERS

    elif dataset_name == "lucchi":
        loader = datasets.get_lucchi_loader(
            path=os.path.join(ROOT, "lucchi"), split="test", patch_shape=(1, 512, 512), batch_size=1,
            ndim=2, label_transform=connected_components, shuffle=True, sampler=sampler,
        )

    elif dataset_name == "mitoem_rat":
        loader = datasets.get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"), splits="val", patch_shape=(1, 512, 512),
            batch_size=1, samples=["rat"], ndim=2, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "mitoem_human":
        loader = datasets.get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"), splits="val", patch_shape=(1, 512, 512),
            batch_size=1, samples=["human"], ndim=2, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "platy_nuclei":
        sample_ids = [11, 12]
        rois = compute_platy_rois(
            os.path.join(ROOT, "platynereis"), sample_ids, ignore_label=-1,
            file_template="nuclei/train_data_nuclei_%02i.h5", label_key="volumes/labels/nucleus_instance_labels"
        )
        loader = datasets.get_platynereis_nuclei_loader(
            path=os.path.join(ROOT, "platynereis"), patch_shape=(1, 512, 512), batch_size=1,
            sample_ids=sample_ids, rois=rois, ndim=2, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "platy_cilia":
        sample_ids = [1, 2, 3]
        rois = compute_platy_rois(
            os.path.join(ROOT, "platynereis"), sample_ids, ignore_label=-1,
            file_template="cilia/train_data_cilia_%02i.h5", label_key="volumes/labels/segmentation"
        )
        loader = datasets.get_platynereis_cilia_loader(
            path=os.path.join(ROOT, "platynereis"), patch_shape=(1, 512, 512), batch_size=1,
            ndim=2, download=True, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "uro_cell":
        loader = datasets.get_uro_cell_loader(
            path=os.path.join(ROOT, "uro_cell"), target="mito", patch_shape=(1, 512, 512),
            ndim=2, batch_size=1, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "nuc_mm_mouse":
        loader = datasets.get_nuc_mm_loader(
            path=os.path.join(ROOT, "nuc_mm"), sample="mouse", split="train", patch_shape=(1, 512, 512),
            batch_size=1, ndim=2, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "nuc_mm_zebrafish":
        loader = datasets.get_nuc_mm_loader(
            path=os.path.join(ROOT, "nuc_mm"), sample="zebrafish", split="train", patch_shape=(1, 256, 256),
            ndim=2, batch_size=1, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "sponge_em":
        loader = datasets.get_sponge_em_loader(
            path=os.path.join(ROOT, "sponge_em"), mode="instances", patch_shape=(1, 512, 512),
            ndim=2, batch_size=1, sampler=sampler, shuffle=True,
        )

    elif dataset_name == "mitolab_c_elegans":
        loader = datasets.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"), dataset_id=1, batch_size=1,
            patch_shape=(1, 512, 512), sampler=sampler, shuffle=True,
        )

    elif dataset_name == "mitolab_fly_brain":
        loader = datasets.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"), dataset_id=2, batch_size=1,
            patch_shape=(1, 512, 512), sampler=sampler, shuffle=True,
        )

    elif dataset_name == "mitolab_glycotic_muscle":
        loader = datasets.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"), dataset_id=3, batch_size=1,
            patch_shape=(1, 512, 512), sampler=sampler, shuffle=True,
        )

    elif dataset_name == "mitolab_hela_cell":
        loader = datasets.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"), dataset_id=4, batch_size=1,
            patch_shape=(1, 512, 512), sampler=sampler, shuffle=True,
        )

    elif dataset_name == "mitolab_tem":
        loader = datasets.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"), dataset_id=7, batch_size=1,
            patch_shape=(1, 512, 512), sampler=sampler, shuffle=True,
        )

    elif dataset_name == "asem_mito":
        loader = datasets.get_asem_loader(
            path=os.path.join(ROOT, "asem"), patch_shape=(1, 512, 512), batch_size=1,
            ndim=2, organelles="mito", sampler=sampler, shuffle=True, label_transform=connected_components
        )

    elif dataset_name == "vnc":
        loader = datasets.get_vnc_mito_loader(
            path=os.path.join(ROOT, "vnc"), patch_shape=(1, 512, 512),
            ndim=2, sampler=sampler, batch_size=1,
        )

    elif dataset_name == "asem_er":
        loader = datasets.get_asem_loader(
            path=os.path.join(ROOT, "asem"), patch_shape=(1, 512, 512), batch_size=1,
            ndim=2, organelles="er", sampler=sampler, shuffle=True, label_transform=connected_components
        )

    elif dataset_name == "cremi":
        loader = datasets.get_cremi_loader(
            path=os.path.join(ROOT, "cremi"), patch_shape=(1, 512, 512), ndim=2, batch_size=1,
            defect_augmentation_kwargs=None, sampler=sampler, shuffle=True, label_transform=connected_components
        )

    else:
        raise ValueError

    return loader
