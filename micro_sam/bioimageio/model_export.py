import os
import tempfile

from pathlib import Path
from typing import Optional, Union

import bioimageio.spec.model.v0_5 as spec
import matplotlib.pyplot as plt
import numpy as np
import torch

from bioimageio.spec import save_bioimageio_package


from .. import util
from ..prompt_generators import PointAndBoxPromptGenerator
from ..evaluation.model_comparison import _enhance_image, _overlay_outline, _overlay_box
from .predictor_adaptor import PredictorAdaptor

DEFAULTS = {
    "authors": [
        spec.Author(name="Anwai Archit", affiliation="University Goettingen", github_user="anwai98"),
        spec.Author(name="Constantin Pape", affiliation="University Goettingen", github_user="constantinpape"),
    ],
    "description": "Finetuned Segment Anything Model for Microscopy",
    "cite": [
        spec.CiteEntry(text="Archit et al. Segment Anything for Microscopy", doi=spec.Doi("10.1101/2023.08.21.554208")),
    ],
    "tags": ["segment-anything", "instance-segmentation"]
}


def _create_test_inputs_and_outputs(
    image,
    labels,
    model_type,
    checkpoint_path,
    tmp_dir,
):

    # For now we just generate a single box prompt here, but we could also generate more input prompts.
    generator = PointAndBoxPromptGenerator(0, 0, 4, False, True)
    centers, bounding_boxes = util.get_centers_and_bounding_boxes(labels)
    masks = util.segmentation_to_one_hot(labels.astype("int64"), segmentation_ids=[1, 2])  # type: ignore
    _, _, box_prompts, _ = generator(masks, [bounding_boxes[1], bounding_boxes[2]])
    box_prompts = box_prompts.numpy()[None]

    predictor = PredictorAdaptor(model_type=model_type)
    predictor.load_state_dict(torch.load(checkpoint_path))

    save_box_prompt_path = os.path.join(tmp_dir, "box_prompts.npy")
    np.save(save_box_prompt_path, box_prompts)

    input_ = util._to_image(image).transpose(2, 0, 1)[None]
    save_image_path = os.path.join(tmp_dir, "input.npy")
    np.save(save_image_path, input_)

    masks, scores, embeddings = predictor(
        image=torch.from_numpy(input_),
        embeddings=None,
        box_prompts=torch.from_numpy(box_prompts)
    )

    mask_path = os.path.join(tmp_dir, "mask.npy")
    score_path = os.path.join(tmp_dir, "scores.npy")
    embed_path = os.path.join(tmp_dir, "embeddings.npy")
    np.save(mask_path, masks.numpy())
    np.save(score_path, scores.numpy())
    np.save(embed_path, embeddings.numpy())

    # TODO autogenerate the cover and return it too.

    inputs = {
        "image": save_image_path,
        "box_prompts": save_box_prompt_path,
    }
    outputs = {
        "mask": mask_path,
        "score": score_path,
        "embeddings": embed_path
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


def _write_dependencies(dependency_file, require_mobile_sam):
    content = """name: sam
channels:
    - pytorch
    - conda-forge
dependencies:
    - segment-anything"""
    if require_mobile_sam:
        content += """
    - pip:
        - git+https://github.com/ChaoningZhang/MobileSAM.git"""
    with open(dependency_file, "w") as f:
        f.write(content)


def _generate_covers(input_paths, result_paths, tmp_dir):
    image = np.load(input_paths["image"]).squeeze()
    prompts = np.load(input_paths["box_prompts"])
    mask = np.load(result_paths["mask"])

    # create the image overlay
    if image.ndim == 2:
        overlay = np.stack([image, image, image]).transpose((1, 2, 0))
    elif image.shape[0] == 3:
        overlay = image.transpose((1, 2, 0))
    else:
        overlay = image
    overlay = _enhance_image(overlay.astype("float32"))

    # overlay the mask as outline
    overlay = _overlay_outline(overlay, mask[0, 0, 0], outline_dilation=2)

    # overlay the bounding box prompt
    prompt = prompts[0, 0][[1, 0, 3, 2]]
    prompt = np.array([prompt[:2], prompt[2:]])
    overlay = _overlay_box(overlay, prompt, outline_dilation=4)

    # write  the cover image
    fig, ax = plt.subplots(1)
    ax.axis("off")
    ax.imshow(overlay.astype("uint8"))
    cover_path = os.path.join(tmp_dir, "cover.jpeg")
    plt.savefig(cover_path, bbox_inches="tight")
    plt.close()

    covers = [cover_path]
    return covers


def export_sam_model(
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = _get_checkpoint(model_type, checkpoint_path=checkpoint_path)
        input_paths, result_paths = _create_test_inputs_and_outputs(
            image, label_image, model_type, checkpoint_path, tmp_dir,
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
                    spec.IndexInputAxis(
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
                    # NOTE: we use the data dependent size here to avoid dependency on optional inputs
                    spec.IndexOutputAxis(
                        id=spec.AxisId("object"), size=spec.DataDependentSize(),
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
                    # NOTE: we use the data dependent size here to avoid dependency on optional inputs
                    spec.IndexOutputAxis(
                        id=spec.AxisId("object"), size=spec.DataDependentSize(),
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

        architecture_path = os.path.join(os.path.split(__file__)[0], "predictor_adaptor.py")
        architecture = spec.ArchitectureFromFileDescr(
            source=Path(architecture_path),
            callable="PredictorAdaptor",
            kwargs={"model_type": model_type}
        )

        dependency_file = os.path.join(tmp_dir, "environment.yaml")
        _write_dependencies(dependency_file, require_mobile_sam=model_type.startswith("vit_t"))

        weight_descriptions = spec.WeightsDescr(
            pytorch_state_dict=spec.PytorchStateDictWeightsDescr(
                source=Path(checkpoint_path),
                architecture=architecture,
                pytorch_version=spec.Version(torch.__version__),
                dependencies=spec.EnvironmentFileDescr(source=dependency_file),
            )
        )

        doc_path = os.path.join(tmp_dir, "documentation.md")
        _write_documentation(doc_path, kwargs.get("documentation", None))

        covers = _generate_covers(input_paths, result_paths, tmp_dir)

        model_description = spec.ModelDescr(
            name=name,
            inputs=input_descriptions,
            outputs=output_descriptions,
            weights=weight_descriptions,
            description=kwargs.get("description", DEFAULTS["description"]),
            authors=kwargs.get("authors", DEFAULTS["authors"]),
            cite=kwargs.get("cite", DEFAULTS["cite"]),
            license=spec.LicenseId("CC-BY-4.0"),
            documentation=Path(doc_path),
            git_repo=spec.HttpUrl("https://github.com/computational-cell-analytics/micro-sam"),
            tags=kwargs.get("tags", DEFAULTS["tags"]),
            covers=covers,
            # TODO attach the decoder weights if given
            # Can be list of files???
            # attachments=[spec.FileDescr(source=file_path) for file_path in attachment_files]
            # TODO write the config
            # dict with yaml values, key must be a str
            # micro_sam: ...
            # config=
        )

        # TODO test the model.
        # Should work, but not tested with optional.

        save_bioimageio_package(model_description, output_path=output_path)
