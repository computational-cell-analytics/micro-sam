import os
from glob import glob

from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths

from micro_sam.evaluation.inference import run_apg
from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.livecell import _get_livecell_paths


def run_apg_grid_search(model_type="vit_b_lm"):
    # Get the image paths for LIVECell.
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
    val_image_paths, val_label_paths = get_livecell_paths(path=data_dir, split="val")
    test_image_paths, _ = get_livecell_paths(path=data_dir, split="test")

    # HACK: Let's test the pipeline super quickly on a very few validation images.
    val_image_paths, val_label_paths = val_image_paths[:10], val_label_paths[:10]

    experiment_folder = "./experiments/livecell"  # HACK: Hard-coded to something random atm.
    prediction_folder = run_apg(
        checkpoint=None,
        model_type=model_type,
        experiment_folder=experiment_folder,
        val_image_paths=val_image_paths,
        val_gt_paths=val_label_paths,
        test_image_paths=test_image_paths,
    )

    # Get the prediction paths
    prediction_paths = sorted(glob(os.path.join(prediction_folder, "*.tif")))
    _, label_paths = _get_livecell_paths(input_folder=data_dir)
    res = run_evaluation(label_paths, prediction_paths, os.path.join(experiment_folder, "results", "apg.csv"))
    print(res)


def main():
    # test_apg_on_livecell()
    run_apg_grid_search()


if __name__ == "__main__":
    main()
