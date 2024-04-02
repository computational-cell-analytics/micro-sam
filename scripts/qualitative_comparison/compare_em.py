from util import compare_experiments_for_dataset


def compare_em(
    experiment_folder, standard_model, finetuned_model, checkpoint1=None, checkpoint2=None
):
    # for figure 1
    all_datasets = ["mitoem_rat", "lucchi", "platy_nuclei"]
    for dataset in all_datasets:
        compare_experiments_for_dataset(
            dataset, experiment_folder, standard_model, finetuned_model, checkpoint1, checkpoint2
        )

    # TODO:
    # mitoem_rat, mitoem_human, platy_nuclei, mitolab (see what's relevant),
    # nucmm_mouse, nucmm_zebrafish (?), platy_cilia, uro_cell, sponge_em, asem (mito)

    # proof of concept experiments:
    # cremi specialist: see if it works for other boundary structures
    #   - platy_cells, axondeepseg, snemi, isbi
    # asem (er) specialist ()


def main():
    compare_em(
        experiment_folder="/scratch/projects/nim00007/sam/experiments/new_models/qualitative",
        standard_model="vit_b",
        finetuned_model="vit_b_em_organelles_v2",
        checkpoint1=None,
        checkpoint2="/scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt"
    )


if __name__ == "__main__":
    main()
