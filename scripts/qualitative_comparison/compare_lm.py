from util import compare_experiments_for_dataset


def compare_lm(
    experiment_folder, standard_model, finetuned_model, checkpoint1=None, checkpoint2=None
):
    # compare_experiments_for_dataset(
    #     "livecell", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    # )
    # compare_experiments_for_dataset(
    #     "deepbacs", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    # )
    # compare_experiments_for_dataset(
    #     "plantseg_root", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    # )
    # compare_experiments_for_dataset(
    #     "tissuenet", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    # )
    # compare_experiments_for_dataset(
    #     "neurips_cellseg", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    # )  # TODO
    compare_experiments_for_dataset(
        "covid_if", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    )
    compare_experiments_for_dataset(
        "plantseg_ovules", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    )
    compare_experiments_for_dataset(
        "hpa", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    )
    compare_experiments_for_dataset(
        "lizard", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    )
    compare_experiments_for_dataset(
        "mouse_embryo", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    )
    compare_experiments_for_dataset(
        "dsb", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    )
    compare_experiments_for_dataset(
        "dynamicnuclearnet", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    )
    compare_experiments_for_dataset(
        "pannuke", experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
    )


def main():
    compare_lm(
        experiment_folder="/scratch/projects/nim00007/sam/experiments/new_models/",
        standard_model="vit_b",
        finetuned_model="vit_b_lm_v2",
        checkpoint1=None,
        checkpoint2="/scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/lm_generalist_sam/best.pt"
    )


if __name__ == "__main__":
    main()
