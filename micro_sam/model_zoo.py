import os
import numpy as np
from glob import glob
import imageio.v2 as imageio

from bioimageio.core.build_spec import build_model


def _get_livecell_npy_path(input_dir):
    test_img_paths = sorted(glob(os.path.join(input_dir, "images", "livecell_test_images", "*")))
    input_image = imageio.imread(test_img_paths[0])
    save_image_path = "./test-livecell-image.npy"
    np.save(save_image_path, input_image)

    # TODO: probably we need the prompt inputs here as well

    # TODO: get output paths
    # outputs: model(inputs) -> outputs: converted to numpy
    save_output_path = "<RANDOM_PATH>.npy"

    return [save_image_path], [save_output_path]


def _get_documentation(doc_path):
    with open(doc_path, "w") as f:
        f.write("# Segment Anything for Microscopy\n")
        f.write("Lorem Ipsum\n")
    return doc_path


def _get_modelzoo_yaml():
    input_list, output_list = _get_livecell_npy_path("/scratch/usr/nimanwai/data/livecell")

    build_model(
        weight_uri="~/.sam_models/vit_t_mobile_sam.pth",
        test_inputs=input_list,  # type: ignore
        test_outputs=output_list,  # type: ignore
        input_axes=["bcyx"],
        output_axes=["bcyx"],
        name="dinosaur",
        description="Finetuned Segment Anything models for Microscopy",
        authors=[{"name": "Anwai Archit", "affiliation": "Uni Goettingen"},
                 {"name": "Constantin Pape", "affiliation": "Uni Goettingen"}],
        tags=["instance-segmentation", "segment-anything"],
        license="CC-BY-4.0",  # TODO: check with Constantin
        documentation=_get_documentation("./doc.md"),
        cite=[{"text": "Archit, ..., Pape et al. Segment Anything for Microscopy", "doi": "10.1101/2023.08.21.554208"}],
        output_path="./modelzoo/my_micro_sam.zip"
    )


def main():
    _get_modelzoo_yaml()


if __name__ == "__main__":
    main()
