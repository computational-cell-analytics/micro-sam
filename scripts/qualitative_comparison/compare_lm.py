import os

from micro_sam.evaluation.model_comparison import (
    generate_data_for_model_comparison, model_comparison
)

from util import fetch_data_loaders


ROOT = "/media/anwai/ANWAI/data"


def compare_deepbacs():
    standard_model = "vit_b"
    finetuned_model = "vit_b_lm"

    output_folder = f"./model_comparison/deepbacs/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = fetch_data_loaders("deepbacs")
        generate_data_for_model_comparison(loader, output_folder, standard_model, finetuned_model, n_samples=10)

    model_comparison(
        output_folder, n_images_per_sample=8, min_size=100,
        plot_folder="./candidates/deepbacs", point_radius=2, outline_dilation=0
    )
    # model_comparison_with_napari(output_folder, show_points=True)


def compare_livecell():
    standard_model = "vit_b"
    finetuned_model = "vit_b_lm"

    output_folder = f"./model_comparison/livecell/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = fetch_data_loaders("livecell")
        generate_data_for_model_comparison(loader, output_folder, standard_model, finetuned_model, n_samples=10)

    model_comparison(
        output_folder, n_images_per_sample=8, min_size=100,
        plot_folder="./candidates/livecell", point_radius=3, outline_dilation=0
    )
    # model_comparison_with_napari(output_folder, show_points=True)


def compare_tissuenet():
    standard_model = "vit_b"
    finetuned_model = "vit_b_lm"

    output_folder = f"./model_comparison/tissuenet/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = fetch_data_loaders("tissuenet")
        generate_data_for_model_comparison(loader, output_folder, standard_model, finetuned_model, n_samples=10)

    model_comparison(
        output_folder, n_images_per_sample=8, min_size=100, plot_folder="./candidates/tissuenet",
        point_radius=3, outline_dilation=0
    )
    # model_comparison_with_napari(output_folder, show_points=True)


def compare_plantseg_root():
    standard_model = "vit_b"
    finetuned_model = "vit_b_lm"

    output_folder = f"./model_comparison/plantseg_root/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = fetch_data_loaders("plantseg_root")
        generate_data_for_model_comparison(loader, output_folder, standard_model, finetuned_model, n_samples=10)

    model_comparison(
        output_folder, n_images_per_sample=8, min_size=100, plot_folder="./candidates/plantseg_root",
        point_radius=3, outline_dilation=0
    )
    # model_comparison_with_napari(output_folder, show_points=True)


def main():
    # compare_deepbacs()
    # compare_livecell()
    # compare_tissuenet()
    compare_plantseg_root()


if __name__ == "__main__":
    main()
