import os
from glob import glob

from torch_em.data.datasets import get_dsb_loader

from micro_sam.evaluation.model_comparison import (
    generate_data_for_model_comparison, model_comparison
)


OUT_FOLDER = "./results"


def create_data():
    n_samples = 4
    results = glob(os.path.join(OUT_FOLDER, "*.h5"))
    if len(results) == n_samples:
        return

    loader = get_dsb_loader(
        "/home/anwai/data", "train", patch_shape=(256, 256), batch_size=1, download=True,
    )

    generate_data_for_model_comparison(
        loader, output_folder=OUT_FOLDER,
        model_type1="vit_b", model_type2="vit_b_lm", n_samples=n_samples,
    )


def main():
    create_data()
    model_comparison(
        OUT_FOLDER, n_images_per_sample=4, min_size=50,
        # Uncomment this to save the plots to file instead of directly displaying them.
        # plot_folder="./plots",
    )


if __name__ == "__main__":
    main()
