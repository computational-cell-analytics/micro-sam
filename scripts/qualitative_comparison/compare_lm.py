from util import compare_experiments_for_dataset


VIT_B_PARAMS = {
    "experiment_folder": "/scratch/projects/nim00007/sam/experiments/new_models/qualitative",
    "standard_model": "vit_b",
    "finetuned_model": "vit_b_lm_v2",
    "checkpoint1": None,
    "checkpoint2": "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/lm_generalist_sam/best.pt"
}


VIT_L_PARAMS = {
    "experiment_folder": "/scratch/projects/nim00007/sam/experiments/new_models/qualitative",
    "standard_model": "vit_l",
    "finetuned_model": "vit_l_lm_v2",
    "checkpoint1": None,
    "checkpoint2": "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_l/lm_generalist_sam/best.pt"
}


def compare_lm():
    # for figure 1 (we use 'vit_b')
    # all_datasets = ["livecell", "deepbacs", "dynamicnuclearnet", "plantseg_root"]
    # params = VIT_B_PARAMS

    # for figure 3 (we use 'vit_l')
    # all_datasets = ["covid_if", "lizard", "mouse_embryo", "plantseg_ovules"]
    # params = VIT_L_PARAMS

    # for all datasets (we use 'vit_l')
    all_datasets = [
        "livecell", "deepbacs", "tissuenet", "plantseg_root", "neurips_cellseg", "covid_if",
        "dynamicnuclearnet", "plantseg_ovules", "mouse_embryo", "hpa", "pannuke", "lizard",
    ]
    params = VIT_L_PARAMS

    for dataset in all_datasets:
        compare_experiments_for_dataset(dataset, **params)


def main():
    compare_lm()


if __name__ == "__main__":
    main()
