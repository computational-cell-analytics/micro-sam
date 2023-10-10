import os
from glob import glob

from bioimageio.core.build_spec import build_model


def _get_livecell_path(input_dir):
    test_img_paths = glob(os.path.join(input_dir, "images", "livecell_test_images", "*"))
    return test_img_paths[0]


def _get_modelzoo_yaml():
    input_path = _get_livecell_path("/scratch/usr/nimanwai/data/livecell")

    build_model(
        weight_uri="~/.sam_models/vit_t_mobile_sam.pth",
        test_inputs=[input_path],
        test_outputs=["./results/"],
        input_axes=["bcyx"],
        output_axes=["bcyx"],
        name="dinosaur",
        description="lorem ipsum",
        authors=[{"name": "Anwai Archit", "affiliation": "Uni Goettingen"},
                 {"name": "Constantin Pape", "affiliation": "Uni Goettingen"}],
        tags=["instance segmentation", "segment anything"],
        license="LOREM IPSUM",  # FIXME
        documentation="README.md",  # TODO - check out what to put here
        cite=[{"text": "Archit, ..., Pape et al. Segment Anything for Microscopy",
               "doi": "10.1101/2023.08.21.554208"}],
        output_path="my_micro_sam.zip"
    )


def main():
    _get_modelzoo_yaml()


if __name__ == "__main__":
    main()
