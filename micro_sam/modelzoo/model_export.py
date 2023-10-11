import os
import argparse
import numpy as np
from glob import glob

import imageio.v2 as imageio

import torch

from micro_sam import util

from .predictor_adaptor import PredictorAdaptor
from ..prompt_based_segmentation import _compute_box_from_mask

from bioimageio.core.build_spec import build_model


def _get_model(image, model_type):
    "Returns the model and predictor while initializing with the model checkpoints"
    predictor, sam_model = util.get_sam_model(model_type=model_type, return_sam=True)  # type: ignore
    image_embeddings = util.precompute_image_embeddings(predictor, image)
    util.set_precomputed(predictor, image_embeddings)
    return predictor, sam_model


def _get_livecell_npy_paths(
        input_dir: str,
        model_type: str
):
    test_img_paths = sorted(glob(os.path.join(input_dir, "images", "livecell_test_images", "*")))
    chosen_input = test_img_paths[0]

    input_image = imageio.imread(chosen_input)

    fname = os.path.split(chosen_input)[-1]
    cell_type = fname.split("_")[0]
    label_image = imageio.imread(os.path.join(input_dir, "annotations", "livecell_test_images", cell_type, fname))

    save_image_path = "./test-livecell-image.npy"
    np.save(save_image_path, input_image[None, None])

    _, sam_model = _get_model(input_image, model_type)
    get_instance_segmentation = PredictorAdaptor(sam_model=sam_model)

    box_prompts = _compute_box_from_mask(label_image)
    save_box_prompt_path = "./test-box-prompts.npy"
    np.save(save_box_prompt_path, box_prompts)

    input_image = util._to_image(input_image).transpose(2, 0, 1)

    instances = get_instance_segmentation(
        input_image=torch.from_numpy(input_image)[None],
        image_embeddings=None,
        box_prompts=torch.from_numpy(box_prompts)[None]
    )

    save_output_path = "./test-livecell-output.npy"
    np.save(save_output_path, instances.numpy())

    return save_image_path, save_output_path


def _get_documentation(doc_path):
    with open(doc_path, "w") as f:
        f.write("# Segment Anything for Microscopy\n")
        f.write("We extend Segment Anything, a vision foundation model for image segmentation ")
        f.write("by training specialized models for microscopy data.\n")
    return doc_path


def _get_sam_checkpoints(model_type):
    checkpoint = util._get_checkpoint(model_type, None)
    print(f"{model_type} is available at {checkpoint}")
    return checkpoint


def get_modelzoo_yaml(
        image_path: str,
        box_prompts_path: str,
        model_type: str,
        output_path: str,
        doc_path: str
):
    input_path, output_path = _get_livecell_npy_paths(input_dir=image_path, model_type=model_type)
    _checkpoint = _get_sam_checkpoints(model_type)

    input_list = [input_path, box_prompts_path]
    output_list = [output_path]

    architecture_path = os.path.join(os.path.split(__file__)[0], "predictor_adaptor.py")

    build_model(
        weight_uri=_checkpoint,
        test_inputs=input_list,  # type: ignore
        test_outputs=output_list,  # type: ignore
        input_axes=["bcyx"],
        output_axes=["bcyx"],
        name="dinosaur",
        description="Finetuned Segment Anything models for Microscopy",
        authors=[{"name": "Anwai Archit", "affiliation": "Uni Goettingen"},
                 {"name": "Constantin Pape", "affiliation": "Uni Goettingen"}],
        tags=["instance-segmentation", "segment-anything"],
        license="CC-BY-4.0",
        documentation=_get_documentation(doc_path),
        cite=[{"text": "Archit, ..., Pape et al. Segment Anything for Microscopy", "doi": "10.1101/2023.08.21.554208"}],
        output_path=output_path,
        architecture=architecture_path
    )


def _get_modelzoo_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str,
                        help="Path to the raw inputs' directory")
    parser.add_argument("-m", "--model_type", type=str, default="vit_b",
                        help="Name of the model to get the SAM checkpoints")
    parser.add_argument("-o", "--output_path", type=str, default="./models/micro_sam.zip",
                        help="Path to the output bioimage modelzoo-format SAM model")
    parser.add_argument("-d", "--doc_path", type=str, default="./documentation.md",
                        help="Path to the documentation")
    parser.add_argument("--boxes_path", type=str, default=None,
                        help="Path to the saved box prompts")
    return parser
