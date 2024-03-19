import os
from torch_em.data.datasets import (
    get_livecell_loader,
    get_deepbacs_loader,
    get_tissuenet_loader,
)
from micro_sam.evaluation.model_comparison import (
    generate_data_for_model_comparison, model_comparison
)

ROOT = "/media/anwai/ANWAI/data"


def compare_deepbacs():
    standard_model = "vit_b"
    finetuned_model = "vit_b_lm"

    output_folder = f"./model_comparison/deepbacs/{standard_model}-{finetuned_model}"
    if not os.path.exists(output_folder):
        loader = get_deepbacs_loader(
            os.path.join(ROOT, "deepbacs"), "test", bac_type="mixed", download=True,
            patch_shape=(512, 512), batch_size=1, shuffle=False, n_samples=100
        )
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
        loader = get_livecell_loader(os.path.join(ROOT, "livecell"), "train", (512, 512), 1)
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
        get_tissuenet_loader(
            os.path.join(ROOT, "tissuenet"), "train", raw_channel="rgb", label_channel="cell",
            patch_shape=(256, 256), batch_size=1, shuffle=True,
        )

    model_comparison(
        output_folder, n_images_per_sample=8, min_size=100, plot_folder="./candidates/tissuenet",
        point_radius=3, outline_dilation=0
    )
    # model_comparison_with_napari(output_folder, show_points=True)


def main():
    # compare_deepbacs()
    compare_livecell()
    # compare_tissuenet()


if __name__ == "__main__":
    main()