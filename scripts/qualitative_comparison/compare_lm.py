from util import compare_experiments_for_dataset


def compare_lm(
    experiment_folder, standard_model, finetuned_model, checkpoint1=None, checkpoint2=None
):
    # for figure 1
    all_datasets = ["livecell", "deepbacs", "dynamicnuclearnet", "plantseg_root"]

    # TODO: for all datasets
    # all_datasets = [
    #     "livecell", "deepbacs", "tissuenet", "neurips_cellseg", "covid_if", "hpa",
    #     "plantseg_ovules", "lizard", "mouse_embryo", "dynamicnuclearnet", "pannuke"
    # ]

    for dataset in all_datasets:
        compare_experiments_for_dataset(
            dataset, experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
        )


def main():
    compare_lm(
        experiment_folder="/scratch/projects/nim00007/sam/experiments/new_models/qualitative",
        standard_model="vit_b",
        finetuned_model="vit_b_lm_v2",
        checkpoint1=None,
        checkpoint2="/scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/lm_generalist_sam/best.pt"
    )


if __name__ == "__main__":
    main()
