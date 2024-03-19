import os
from torch_em.data.datasets import get_lucchi_loader, get_snemi_loader
from torch_em.transform.label import connected_components
from micro_sam.evaluation.model_comparison import generate_data_for_model_comparison, model_comparison


def compare_lucchi():
    standard_model = "vit_h"
    finetuned_model = "vit_h_lm"

    output_folder = f"./model_comparison/lucchi/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = get_lucchi_loader(
            "./data/lucchi", "train", (1, 512, 512), 1, ndim=2, download=True,
            label_transform=connected_components
        )
        generate_data_for_model_comparison(loader, output_folder, standard_model, finetuned_model, n_samples=10)

    model_comparison(output_folder, n_images_per_sample=8, min_size=250, plot_folder="./candidates/lucchi",
                     outline_dilation=0)
    # model_comparison_with_napari(output_folder, show_points=True)


def compare_snemi():
    standard_model = "vit_h"
    finetuned_model = "vit_h_lm"

    output_folder = f"./model_comparison/snemi/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = get_snemi_loader(
            "./data/snemi", (1, 512, 512), 1, ndim=2, download=True,
            label_transform=connected_components
        )

        generate_data_for_model_comparison(loader, output_folder, standard_model, finetuned_model, n_samples=10)

    model_comparison(
        output_folder, n_images_per_sample=8, min_size=500, plot_folder="./candidates/snemi",
        outline_dilation=0
    )
    # model_comparison_with_napari(output_folder, show_points=True)


# For Fig 1:
# Lucchi, SpongeEM, Nuc-MM, ?

# For Fig 4:
# MitoEM, Lucchi, (choose one from) Mitolab, Nuc-MM
def main():
    # compare_lucchi()
    compare_snemi()


if __name__ == "__main__":
    main()