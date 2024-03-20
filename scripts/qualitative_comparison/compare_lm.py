from util import compare_experiments_for_dataset


ROOT = "/media/anwai/ANWAI/data"


def compare_lm(
    standard_model, finetuned_model, checkpoint1=None, checkpoint2=None
):
    compare_experiments_for_dataset("livecell", standard_model, finetuned_model)
    compare_experiments_for_dataset("deepbacs", standard_model, finetuned_model)
    compare_experiments_for_dataset("plantseg_root", standard_model, finetuned_model)

    # compare_experiments_for_dataset("tissuenet", standard_model, finetuned_model)


def main():
    compare_lm("vit_b", "vit_b_lm")


if __name__ == "__main__":
    main()
