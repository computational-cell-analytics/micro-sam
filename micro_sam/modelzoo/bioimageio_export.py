import os
from pathlib import Path
from tempfile import NamedTemporaryFile as tmp_file
from typing import Optional, Union

import bioimageio.spec.model.v0_5 as spec
import numpy as np
import torch

from bioimageio.spec import save_bioimageio_package


from .. import util
from ..prompt_generators import PointAndBoxPromptGenerator
from .predictor_adaptor import PredictorAdaptor

# TODO extend the defaults
DEFAULTS = {
    "authors": [
        spec.Author(name="Anwai Archit", affiliation="University Goettingen", github_user="anwai98"),
        spec.Author(name="Constantin Pape", affiliation="University Goettingen", github_user="constantinpape"),
    ],
    "description": "Finetuned Segment Anything Model for Microscopy",
    "cite": [
        spec.CiteEntry(text="Archit et al. Segment Anything for Microscopy", doi=spec.Doi("10.1101/2023.08.21.554208")),
    ]
}


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
    masks = util.segmentation_to_one_hot(labels.astype("int64"), segmentation_ids=[1, 2])  # type: ignore
    _, _, box_prompts, _ = generator(masks, [bounding_boxes[1], bounding_boxes[2]])
    box_prompts = box_prompts.numpy()[None]

    predictor = PredictorAdaptor(model_type=model_type)
    predictor.load_state_dict(torch.load(checkpoint_path))

    save_box_prompt_path = box_path.name
    np.save(save_box_prompt_path, box_prompts)

    input_ = util._to_image(image).transpose(2, 0, 1)[None]
    save_image_path = input_path.name
    np.save(save_image_path, input_)

    masks, scores, embeddings = predictor(
        image=torch.from_numpy(input_),
        embeddings=None,
        box_prompts=torch.from_numpy(box_prompts)
    )

    np.save(mask_path.name, masks.numpy())
    np.save(score_path.name, scores.numpy())
    np.save(embed_path.name, embeddings.numpy())

    # TODO autogenerate the cover and return it too.

    inputs = {
        "image": save_image_path,
        "box_prompts": save_box_prompt_path,
    }
    outputs = {
        "mask": mask_path.name,
        "score": score_path.name,
        "embeddings": embed_path.name
    }
    return inputs, outputs


# TODO url with documentation for the modelzoo interface, and just add it to defaults
def _write_documentation(doc_path, doc):
    with open(doc_path, "w") as f:
        if doc is None:
            f.write("# Segment Anything for Microscopy\n")
            f.write("We extend Segment Anything, a vision foundation model for image segmentation ")
            f.write("by training specialized models for microscopy data.\n")
        else:
            f.write(doc)
    return doc_path


def _get_checkpoint(model_type, checkpoint_path):
    if checkpoint_path is None:
        model_registry = util.models()
        checkpoint_path = model_registry.fetch(model_type)
    return checkpoint_path


def export_bioimageio_model(
    image: np.ndarray,
    label_image: np.ndarray,
    model_type: str,
    name: str,
    output_path: Union[str, os.PathLike],
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
        name: The name of the exported model.
        output_path: Where the exported model is saved.
        checkpoint_path: Optional checkpoint for loading the SAM model.
    """
    with (
        tmp_file(suffix=".md") as tmp_doc_path,
        tmp_file(suffix=".npy") as tmp_input_path,
        tmp_file(suffix=".npy") as tmp_boxes_path,
        tmp_file(suffix=".npy") as tmp_mask_path,
        tmp_file(suffix=".npy") as tmp_score_path,
        tmp_file(suffix=".npy") as tmp_embed_path,
    ):
        checkpoint_path = _get_checkpoint(model_type, checkpoint_path=checkpoint_path)
        input_paths, result_paths = _create_test_inputs_and_outputs(
            image, label_image, model_type, checkpoint_path,
            input_path=tmp_input_path,
            box_path=tmp_boxes_path,
            mask_path=tmp_mask_path,
            score_path=tmp_score_path,
            embed_path=tmp_embed_path,
        )
        input_descriptions = [
            # First input: the image data.
            spec.InputTensorDescr(
                id=spec.TensorId("image"),
                axes=[
                    spec.BatchAxis(),
                    # NOTE: to support 1 and 3 channels we can add another preprocessing.
                    # Best solution: Have a pre-processing for this! (1C -> RGB)
                    spec.ChannelAxis(channel_names=[spec.Identifier(cname) for cname in "RGB"]),
                    spec.SpaceInputAxis(id=spec.AxisId("y"), size=spec.ARBITRARY_SIZE),
                    spec.SpaceInputAxis(id=spec.AxisId("x"), size=spec.ARBITRARY_SIZE),
                ],
                test_tensor=spec.FileDescr(source=input_paths["image"]),
                data=spec.IntervalOrRatioDataDescr(type="uint8")
            ),

            # Second input: the box prompts (optional)
            spec.InputTensorDescr(
                id=spec.TensorId("box_prompts"),
                optional=True,
                axes=[
                    spec.BatchAxis(),
                    spec.IndexAxis(
                        id=spec.AxisId("object"),
                        size=spec.ARBITRARY_SIZE
                    ),
                    # TODO double check the axis names
                    spec.ChannelAxis(channel_names=[spec.Identifier(bname) for bname in "hwxy"]),
                ],
                test_tensor=spec.FileDescr(source=input_paths["box_prompts"]),
                data=spec.IntervalOrRatioDataDescr(type="int64")
            ),

            # TODO
            # Third input: the point prompts (optional)

            # TODO
            # Fourth input: the mask prompts (optional)

            # Fifth input: the image embeddings (optional)
            spec.InputTensorDescr(
                id=spec.TensorId("embeddings"),
                optional=True,
                axes=[
                    spec.BatchAxis(),
                    # NOTE: we currently have to specify all the channel names
                    # (It would be nice to also support size)
                    spec.ChannelAxis(channel_names=[spec.Identifier(f"c{i}") for i in range(256)]),
                    spec.SpaceInputAxis(id=spec.AxisId("y"), size=64),
                    spec.SpaceInputAxis(id=spec.AxisId("x"), size=64),
                ],
                test_tensor=spec.FileDescr(source=result_paths["embeddings"]),
                data=spec.IntervalOrRatioDataDescr(type="float32")
            ),

        ]

        output_descriptions = [
            # First output: The mask predictions.
            spec.OutputTensorDescr(
                id=spec.TensorId("masks"),
                axes=[
                    spec.BatchAxis(),
                    spec.IndexAxis(
                        id=spec.AxisId("object"),
                        size=spec.SizeReference(
                            tensor_id=spec.TensorId("box_prompts"), axis_id=spec.AxisId("object")
                        )
                    ),
                    # NOTE: this could be a 3 once we use multi-masking
                    spec.ChannelAxis(channel_names=[spec.Identifier("mask")]),
                    spec.SpaceOutputAxis(
                        id=spec.AxisId("y"),
                        size=spec.SizeReference(
                            tensor_id=spec.TensorId("image"), axis_id=spec.AxisId("y"),
                        )
                    ),
                    spec.SpaceOutputAxis(
                        id=spec.AxisId("x"),
                        size=spec.SizeReference(
                            tensor_id=spec.TensorId("image"), axis_id=spec.AxisId("x"),
                        )
                    )
                ],
                data=spec.IntervalOrRatioDataDescr(type="uint8"),
                test_tensor=spec.FileDescr(source=result_paths["mask"])
            ),

            # The score predictions
            spec.OutputTensorDescr(
                id=spec.TensorId("scores"),
                axes=[
                    spec.BatchAxis(),
                    spec.IndexAxis(
                        id=spec.AxisId("object"),
                        size=spec.SizeReference(
                            tensor_id=spec.TensorId("box_prompts"), axis_id=spec.AxisId("object")
                        )
                    ),
                    # NOTE: this could be a 3 once we use multi-masking
                    spec.ChannelAxis(channel_names=[spec.Identifier("mask")]),
                ],
                data=spec.IntervalOrRatioDataDescr(type="float32"),
                test_tensor=spec.FileDescr(source=result_paths["score"])
            ),

            # The image embeddings
            spec.OutputTensorDescr(
                id=spec.TensorId("embeddings"),
                axes=[
                    spec.BatchAxis(),
                    spec.ChannelAxis(channel_names=[spec.Identifier(f"c{i}") for i in range(256)]),
                    spec.SpaceOutputAxis(id=spec.AxisId("y"), size=64),
                    spec.SpaceOutputAxis(id=spec.AxisId("x"), size=64),
                ],
                data=spec.IntervalOrRatioDataDescr(type="float32"),
                test_tensor=spec.FileDescr(source=result_paths["embeddings"])
            )
        ]

        # TODO sha256
        architecture_path = os.path.join(os.path.split(__file__)[0], "predictor_adaptor.py")
        architecture = spec.ArchitectureFromFileDescr(
            source=Path(architecture_path),
            callable="PredictorAdaptor",
            kwargs={"model_type": model_type}
        )

        weight_descriptions = spec.WeightsDescr(
            pytorch_state_dict=spec.PytorchStateDictWeightsDescr(
                source=Path(checkpoint_path),
                architecture=architecture,
                pytorch_version=spec.Version(torch.__version__),
            )
        )

        doc_path = tmp_doc_path.name
        _write_documentation(doc_path, kwargs.get("documentation", None))

        # TODO tags, dependencies, other stuff ...
        model_description = spec.ModelDescr(
            name=name,
            description=kwargs.get("description", DEFAULTS["description"]),
            authors=kwargs.get("authors", DEFAULTS["authors"]),
            cite=kwargs.get("cite", DEFAULTS["cite"]),
            license=spec.LicenseId("MIT"),
            documentation=Path(doc_path),
            git_repo=spec.HttpUrl("https://github.com/computational-cell-analytics/micro-sam"),
            inputs=input_descriptions,
            outputs=output_descriptions,
            weights=weight_descriptions,
        )

        # TODO test the model.

        save_bioimageio_package(model_description, output_path=output_path)
