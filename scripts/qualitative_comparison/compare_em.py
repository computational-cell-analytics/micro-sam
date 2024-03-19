import os

from micro_sam.evaluation.model_comparison import generate_data_for_model_comparison, model_comparison

from util import fetch_data_loaders


ROOT = "/media/anwai/ANWAI/data"


def compare_lucchi():
    standard_model = "vit_b"
    finetuned_model = "vit_b_em_organelles"

    output_folder = f"./model_comparison/lucchi/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = fetch_data_loaders("lucchi")
        generate_data_for_model_comparison(loader, output_folder, standard_model, finetuned_model, n_samples=10)

    model_comparison(
        output_folder, n_images_per_sample=8, min_size=250, plot_folder="./candidates/lucchi", outline_dilation=0
    )
    # model_comparison_with_napari(output_folder, show_points=True)


def main():
    compare_lucchi()
    # compare_snemi()


if __name__ == "__main__":
    main()
