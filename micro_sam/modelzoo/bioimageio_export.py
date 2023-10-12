import os
import numpy as np
from typing import Optional, Union

import torch

from bioimageio.core.build_spec import build_model

from .. import util
from ..prompt_generators import PointAndBoxPromptGenerator
from .predictor_adaptor import PredictorAdaptor


def _get_model(image, model_type, checkpoint_path):
    "Returns the model and predictor while initializing with the model checkpoints"
    predictor, sam_model = util.get_sam_model(model_type=model_type, return_sam=True)  # type: ignore
    image_embeddings = util.precompute_image_embeddings(predictor, image)
    util.set_precomputed(predictor, image_embeddings)
    return predictor, sam_model


# TODO use tempfile
def _create_test_inputs_and_outputs(image, labels, model_type, checkpoint_path):

    # For now we just generate a single box prompt here, but we could also generate more input prompts.
    generator = PointAndBoxPromptGenerator(0, 0, 4, False, True)
    centers, bounding_boxes = util.get_centers_and_bounding_boxes(labels)
    masks = util.segmentation_to_one_hot(labels.astype("int64"), segmentation_ids=[1])
    _, _, box_prompts, _ = generator(masks, [bounding_boxes[1]])
    box_prompts = box_prompts.numpy()

    save_image_path = "./test-livecell-image.npy"
    np.save(save_image_path, image[None, None])

    _, sam_model = _get_model(image, model_type, checkpoint_path)
    predictor = PredictorAdaptor(sam_model=sam_model)

    save_box_prompt_path = "./test-box-prompts.npy"
    np.save(save_box_prompt_path, box_prompts)

    input_ = util._to_image(image).transpose(2, 0, 1)

    # TODO embeddings are also expected output
    instances = predictor(
        input_image=torch.from_numpy(input_)[None],
        image_embeddings=None,
        box_prompts=torch.from_numpy(box_prompts)[None]
    )

    save_output_path = "./test-livecell-output.npy"
    np.save(save_output_path, instances.numpy())

    return [save_image_path, save_box_prompt_path], [save_output_path]


def _get_documentation(doc_path):
    with open(doc_path, "w") as f:
        f.write("# Segment Anything for Microscopy\n")
        f.write("We extend Segment Anything, a vision foundation model for image segmentation ")
        f.write("by training specialized models for microscopy data.\n")
    return doc_path


def export_bioimageio_model(
    image: np.ndarray,
    label_image: np.ndarray,
    model_type: str,
    model_name: str,
    output_path: Union[str, os.PathLike],
    doc_path: Optional[Union[str, os.PathLike]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
):
    input_paths, result_paths = _create_test_inputs_and_outputs(
        image, label_image, model_type, checkpoint_path
    )
    checkpoint = util._get_checkpoint(model_type, checkpoint_path=checkpoint_path)

    architecture_path = os.path.join(os.path.split(__file__)[0], "predictor_adaptor.py")

    if doc_path is None:
        doc_path = "./doc.md"
        _get_documentation(doc_path)

    build_model(
        weight_uri=checkpoint,
        test_inputs=input_paths,  # type: ignore
        test_outputs=result_paths,  # type: ignore
        input_axes=["bcyx"],
        output_axes=["bcyx"],
        name=model_name,
        description="Finetuned Segment Anything models for Microscopy",
        authors=[{"name": "Anwai Archit", "affiliation": "Uni Goettingen"},
                 {"name": "Constantin Pape", "affiliation": "Uni Goettingen"}],
        tags=["instance-segmentation", "segment-anything"],
        license="CC-BY-4.0",
        documentation=doc_path,
        cite=[{"text": "Archit, ..., Pape et al. Segment Anything for Microscopy", "doi": "10.1101/2023.08.21.554208"}],
        output_path=output_path,
        architecture=architecture_path
    )
