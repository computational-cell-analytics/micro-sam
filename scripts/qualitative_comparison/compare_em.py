from util import compare_experiments_for_dataset


ROOT = "/media/anwai/ANWAI/data"


def compare_em(
    standard_model, finetuned_model, checkpoint1=None, checkpoint2=None
):
    compare_experiments_for_dataset("lucchi", standard_model, finetuned_model)

    # TODO:
    # mitoem_rat, mitoem_human, platy_nuclei, mitolab (see what's relevant),
    # nucmm_mouse, nucmm_zebrafish (?), platy_cilia, uro_cell, sponge_em, asem (mito)

    # proof of concept experiments:
    # cremi specialist: see if it works for other boundary structures
    #   - platy_cells, axondeepseg, snemi, isbi
    # asem (er) specialist ()


def main():
    compare_em("vit_b", "vit_b_em_organelles")


if __name__ == "__main__":
    main()
