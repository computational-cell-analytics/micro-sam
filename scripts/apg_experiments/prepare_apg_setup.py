from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths

from micro_sam.evaluation.inference import run_apg


def run_apg_grid_search(model_type="vit_b_lm"):
    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    val_image_paths, val_label_paths = get_livecell_paths(path=data_dir, split="val")
    test_image_paths, test_label_paths = get_livecell_paths(path=data_dir, split="test")

    experiment_folder = "./experiments/livecell"  # HACK: Hard-coded to something random atm.
    run_apg(
        checkpoint=None,
        model_type=model_type,
        experiment_folder=experiment_folder,
        val_image_paths=val_image_paths,
        val_gt_paths=val_label_paths,
        test_image_paths=test_image_paths,
    )


def main():
    # test_apg_on_livecell()
    run_apg_grid_search()


if __name__ == "__main__":
    main()
