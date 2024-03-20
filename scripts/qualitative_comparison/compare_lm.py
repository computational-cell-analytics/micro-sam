from util import compare_experiments_for_dataset


ROOT = "/media/anwai/ANWAI/data"


def compare_lm(
    standard_model, finetuned_model, checkpoint1=None, checkpoint2=None
):
    compare_experiments_for_dataset("livecell", standard_model, finetuned_model)
    compare_experiments_for_dataset("deepbacs", standard_model, finetuned_model)
    compare_experiments_for_dataset("plantseg_root", standard_model, finetuned_model)

    # compare_experiments_for_dataset("tissuenet", standard_model, finetuned_model)
    compare_experiments_for_dataset("neurips_cellseg", standard_model, finetuned_model)
    compare_experiments_for_dataset("covid_if", standard_model, finetuned_model)
    compare_experiments_for_dataset("plantseg_ovules", standard_model, finetuned_model)
    compare_experiments_for_dataset("hpa", standard_model, finetuned_model)
    compare_experiments_for_dataset("lizard", standard_model, finetuned_model)
    compare_experiments_for_dataset("mouse_embryo", standard_model, finetuned_model)
    compare_experiments_for_dataset("dsb", standard_model, finetuned_model)
    compare_experiments_for_dataset("dynamicnuclearnet", standard_model, finetuned_model)


def main():
    compare_lm("vit_b", "vit_b_lm")


if __name__ == "__main__":
    main()
