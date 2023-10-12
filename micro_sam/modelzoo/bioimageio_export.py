import os
from tempfile import NamedTemporaryFile as tmp_file
import numpy as np
from typing import Optional, Union

import torch

from bioimageio.core.build_spec import build_model

from .. import util
from ..prompt_generators import PointAndBoxPromptGenerator
from .predictor_adaptor import PredictorAdaptor


def _get_model(image, model_type, checkpoint_path):
    "Returns the model and predictor while initializing with the model checkpoints"
    predictor, sam_model = util.get_sam_model(model_type=model_type, return_sam=True,
                                              checkpoint_path=checkpoint_path)  # type: ignore
    image_embeddings = util.precompute_image_embeddings(predictor, image)
    util.set_precomputed(predictor, image_embeddings)
    return predictor, sam_model


def _create_test_inputs_and_outputs(
    image,
    labels,
    model_type,
    checkpoint_path,
    input_path,
    box_path,
    mask_path,
    score_path,
    embed_path,
):

    # For now we just generate a single box prompt here, but we could also generate more input prompts.
    generator = PointAndBoxPromptGenerator(0, 0, 4, False, True)
    centers, bounding_boxes = util.get_centers_and_bounding_boxes(labels)
    masks = util.segmentation_to_one_hot(labels.astype("int64"), segmentation_ids=[1])  # type: ignore
    _, _, box_prompts, _ = generator(masks, [bounding_boxes[1]])
    box_prompts = box_prompts.numpy()

    save_image_path = input_path.name
    np.save(save_image_path, image[None, None])

    _, sam_model = _get_model(image, model_type, checkpoint_path)
    predictor = PredictorAdaptor(sam_model=sam_model)

    save_box_prompt_path = box_path.name
    np.save(save_box_prompt_path, box_prompts)

    input_ = util._to_image(image).transpose(2, 0, 1)

    masks, scores, embeddings = predictor(
        input_image=torch.from_numpy(input_)[None],
        image_embeddings=None,
        box_prompts=torch.from_numpy(box_prompts)[None]
    )

    np.save(mask_path.name, masks.numpy())
    np.save(score_path.name, scores.numpy())
    np.save(embed_path.name, embeddings.numpy())

    return [save_image_path, save_box_prompt_path], [mask_path.name, score_path.name, embed_path.name]


def _write_documentation(doc_path, doc):
    with open(doc_path, "w") as f:
        if doc is None:
            f.write("# Segment Anything for Microscopy\n")
            f.write("We extend Segment Anything, a vision foundation model for image segmentation ")
            f.write("by training specialized models for microscopy data.\n")
        else:
            f.write(doc)
    return doc_path


# TODO enable over-riding the authors and citation and tags from kwargs
# TODO support RGB sample inputs
def export_bioimageio_model(
    image: np.ndarray,
    label_image: np.ndarray,
    model_type: str,
    model_name: str,
    output_path: Union[str, os.PathLike],
    doc: Optional[str] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    **kwargs
) -> None:
    """Export SAM model to BioImage.IO model format.

    The exported model can be uploaded to [bioimage.io](https://bioimage.io/#/) and
    be used in tools that support the BioImage.IO model format.

    Args:
        image: The image for generating test data.
        label_image: The segmentation correspoding to `image`.
            It is used to derive prompt inputs for the model.
        model_type: The type of the SAM model.
        model_name: The name of the exported model.
        output_path: Where the exported model is saved.
        doc: Documentation for the model.
        checkpoint_path: Optional checkpoint for loading the SAM model.
        kwargs: optional keyword arguments for the 'build_model' function
            that converts to the modelzoo format.
    """
    with (
        tmp_file(suffix=".md") as tmp_doc_path,
        tmp_file(suffix=".npy") as tmp_input_path,
        tmp_file(suffix=".npy") as tmp_boxes_path,
        tmp_file(suffix=".npy") as tmp_mask_path,
        tmp_file(suffix=".npy") as tmp_score_path,
        tmp_file(suffix=".npy") as tmp_embed_path,
    ):
        input_paths, result_paths = _create_test_inputs_and_outputs(
            image, label_image, model_type, checkpoint_path,
            input_path=tmp_input_path,
            box_path=tmp_boxes_path,
            mask_path=tmp_mask_path,
            score_path=tmp_score_path,
            embed_path=tmp_embed_path,
        )
        checkpoint = util._get_checkpoint(model_type, checkpoint_path=checkpoint_path)

        architecture_path = os.path.join(os.path.split(__file__)[0], "predictor_adaptor.py")

        doc_path = tmp_doc_path.name
        _write_documentation(doc_path, doc)

        build_model(
            weight_uri=checkpoint,  # type: ignore
            test_inputs=input_paths,
            test_outputs=result_paths,
            input_axes=["bcyx", "bic"],
            # FIXME this causes some error in build-model
            # input_names=["image", "box-prompts"],
            output_axes=["bcyx", "bic", "bcyx"],
            # FIXME this causes some error in build-model
            # output_names=["masks", "scores", "image_embeddings"],
            name=model_name,
            description="Finetuned Segment Anything models for Microscopy",
            authors=[{"name": "Anwai Archit", "affiliation": "Uni Goettingen"},
                     {"name": "Constantin Pape", "affiliation": "Uni Goettingen"}],
            tags=["instance-segmentation", "segment-anything"],
            license="CC-BY-4.0",
            documentation=doc_path,  # type: ignore
            cite=[{"text": "Archit, ..., Pape et al. Segment Anything for Microscopy",
                   "doi": "10.1101/2023.08.21.554208"}],
            output_path=output_path,  # type: ignore
            architecture=architecture_path,
            **kwargs,
        )

        # TODO actually test the model
