from util import compare_experiments_for_dataset


VIT_B_PARAMS = {
    "experiment_folder": "/scratch/projects/nim00007/sam/experiments/new_models/qualitative",
    "standard_model": "vit_b",
    "finetuned_model": "vit_b_em_organelles_v2",
    "checkpoint1": None,
    "checkpoint2": "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt"
}


VIT_L_PARAMS = {
    "experiment_folder": "/scratch/projects/nim00007/sam/experiments/new_models/qualitative",
    "standard_model": "vit_l",
    "finetuned_model": "vit_l_em_organelles_v2",
    "checkpoint1": None,
    "checkpoint2": "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_l/mito_nuc_em_generalist_sam/best.pt"
}


def compare_em():
    # for figure 1 (we use 'vit_b' here)
    # all_datasets = ["mitoem_rat", "lucchi", "platy_nuclei"]
    # params = VIT_B_PARAMS

    # for figure 4 (we use 'vit_l' here)
    # all_datasets = ["mitolab_tem", "nuc_mm_mouse", "platy_nuclei", "mitoem_human"]
    # params = VIT_L_PARAMS

    # for all datasets (we use `vit_l` here)
    all_datasets = [
        "mitoem_rat", "mitoem_human", "platy_nuclei", "mitolab_c_elegans",
        "mitolab_fly_brain", "mitolab_glycotic_muscle", "mitolab_hela_cell",
        "mitolab_tem", "lucchi", "nuc_mm_mouse", "uro_cell", "sponge_em",
        "vnc", "asem_mito", "platy_cilia"
    ]
    params = VIT_L_PARAMS

    for dataset in all_datasets:
        compare_experiments_for_dataset(dataset, **params)

    # proof of concept experiments:
    # cremi specialist, asem (er)


def main():
    compare_em()


if __name__ == "__main__":
    main()
