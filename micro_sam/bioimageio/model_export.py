import os
import tempfile

from pathlib import Path
from typing import Optional, Union

import bioimageio.core
import bioimageio.spec.model.v0_5 as spec
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray

from bioimageio.spec import save_bioimageio_package
from bioimageio.core.digest_spec import create_sample_for_model


from .. import util
from ..prompt_generators import PointAndBoxPromptGenerator
from ..evaluation.model_comparison import _enhance_image, _overlay_outline, _overlay_box
from ..prompt_based_segmentation import _compute_logits_from_mask
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
    "tags": ["segment-anything", "instance-segmentation"],
}


def _create_test_inputs_and_outputs(
    image,
    labels,
    model_type,
    checkpoint_path,
    tmp_dir,
):
    # For now we just generate a single box prompt here, but we could also generate more input prompts.
    generator = PointAndBoxPromptGenerator(
        n_positive_points=1,
        n_negative_points=2,
        dilation_strength=2,
        get_point_prompts=True,
        get_box_prompts=True,
    )
    centers, bounding_boxes = util.get_centers_and_bounding_boxes(labels)
    masks = util.segmentation_to_one_hot(labels.astype("int64"), segmentation_ids=[1, 2])  # type: ignore
    point_prompts, point_labels, box_prompts, _ = generator(masks, [bounding_boxes[1], bounding_boxes[2]])

    box_prompts = box_prompts.numpy()[None]
    point_prompts = point_prompts.numpy()[None]
    point_labels = point_labels.numpy()[None]

    # Generate logits from the two
    mask_prompts = np.stack(
        [
            _compute_logits_from_mask(labels == 1),
            _compute_logits_from_mask(labels == 2),
        ]
    )[None]

    predictor = PredictorAdaptor(model_type=model_type)
    predictor.load_state_dict(torch.load(checkpoint_path))

    input_ = util._to_image(image).transpose(2, 0, 1)[None]
    image_path = os.path.join(tmp_dir, "input.npy")
    np.save(image_path, input_)

    masks, scores, embeddings = predictor(
        image=torch.from_numpy(input_),
        embeddings=None,
        box_prompts=torch.from_numpy(box_prompts),
        point_prompts=torch.from_numpy(point_prompts),
        point_labels=torch.from_numpy(point_labels),
        mask_prompts=torch.from_numpy(mask_prompts),
    )

    box_prompt_path = os.path.join(tmp_dir, "box_prompts.npy")
    point_prompt_path = os.path.join(tmp_dir, "point_prompts.npy")
    point_label_path = os.path.join(tmp_dir, "point_labels.npy")
    mask_prompt_path = os.path.join(tmp_dir, "mask_prompts.npy")
    np.save(box_prompt_path, box_prompts.astype("int64"))
    np.save(point_prompt_path, point_prompts)
    np.save(point_label_path, point_labels)
    np.save(mask_prompt_path, mask_prompts)

    mask_path = os.path.join(tmp_dir, "mask.npy")
    score_path = os.path.join(tmp_dir, "scores.npy")
    embed_path = os.path.join(tmp_dir, "embeddings.npy")
    np.save(mask_path, masks.numpy())
    np.save(score_path, scores.numpy())
    np.save(embed_path, embeddings.numpy())

    inputs = {
        "image": image_path,
        "box_prompts": box_prompt_path,
        "point_prompts": point_prompt_path,
        "point_labels": point_label_path,
        "mask_prompts": mask_prompt_path,
    }
    outputs = {
        "mask": mask_path,
        "score": score_path,
        "embeddings": embed_path
    }
    return inputs, outputs


def _write_documentation(doc, model_type, tmp_dir):
    tmp_doc_path = os.path.join(tmp_dir, "documentation.md")

    if doc is None:
        with open(tmp_doc_path, "w") as f:
            f.write("# Segment Anything for Microscopy\n")
            f.write("We extend Segment Anything, a vision foundation model for image segmentation ")
            f.write("by training specialized models for microscopy data.\n")
        return tmp_doc_path

    elif os.path.exists(doc):
        return doc

    else:
        with open(tmp_doc_path, "w") as f:
            f.write(doc)
        return tmp_doc_path


def _get_checkpoint(model_type, checkpoint_path, tmp_dir):
    # If we don't have a checkpoint we get the corresponding model from the registry.
    if checkpoint_path is None:
        model_registry = util.models()
        checkpoint_path = model_registry.fetch(model_type)
        return checkpoint_path, None

    # Otherwise we have to load the checkpoint to see if it is the state dict of an encoder,
    # or the checkpoint for a custom SAM model.
    state, model_state = util._load_checkpoint(checkpoint_path)

    if "model_state" in state:  # This is a finetuning checkpoint -> we have to resave the state.
        new_checkpoint_path = os.path.join(tmp_dir, f"{model_type}.pt")
        torch.save(model_state, new_checkpoint_path)

        # We may also have an instance segmentation decoder in that case.
        # If we have it we also resave this one and return it.
        if "decoder_state" in state:
            decoder_path = os.path.join(tmp_dir, f"{model_type}_decoder.pt")
            decoder_state = state["decoder_state"]
            torch.save(decoder_state, decoder_path)
        else:
            decoder_path = None

        return new_checkpoint_path, decoder_path

    else:  # This is a SAM encoder state -> we don't have to resave.
        return checkpoint_path, None


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


def _check_model(model_description, input_paths, result_paths):
    # Load inputs.
    image = xarray.DataArray(np.load(input_paths["image"]), dims=tuple("bcyx"))
    embeddings = xarray.DataArray(np.load(result_paths["embeddings"]), dims=tuple("bcyx"))
    box_prompts = xarray.DataArray(np.load(input_paths["box_prompts"]), dims=tuple("bic"))
    point_prompts = xarray.DataArray(np.load(input_paths["point_prompts"]), dims=tuple("biic"))
    point_labels = xarray.DataArray(np.load(input_paths["point_labels"]), dims=tuple("bic"))
    mask_prompts = xarray.DataArray(np.load(input_paths["mask_prompts"]), dims=tuple("bicyx"))

    # Load outputs.
    mask = np.load(result_paths["mask"])

    with bioimageio.core.create_prediction_pipeline(model_description) as pp:

        # Check with all prompts. We only check the result for this setting,
        # because this was used to generate the test data.
        sample = create_sample_for_model(
            model=model_description,
            image=image,
            box_prompts=box_prompts,
            point_prompts=point_prompts,
            point_labels=point_labels,
            mask_prompts=mask_prompts,
            embeddings=embeddings,
        ).as_single_block()
        prediction = pp.predict_sample_block(sample)

        predicted_mask = prediction.blocks["masks"].data.data
        assert predicted_mask.shape == mask.shape
        assert np.allclose(mask, predicted_mask)

        # Run the checks with partial prompts.
        prompt_kwargs = [
            # With boxes.
            {"box_prompts": box_prompts},
            # With point prompts.
            {"point_prompts": point_prompts, "point_labels": point_labels},
            # With masks.
            {"mask_prompts": mask_prompts},
            # With boxes and points.
            {"box_prompts": box_prompts, "point_prompts": point_prompts, "point_labels": point_labels},
            # With boxes and masks.
            {"box_prompts": box_prompts, "mask_prompts": mask_prompts},
            # With points and masks.
            {"mask_prompts": mask_prompts, "point_prompts": point_prompts, "point_labels": point_labels},
        ]

        for kwargs in prompt_kwargs:
            sample = create_sample_for_model(
                model=model_description, image=image, embeddings=embeddings, **kwargs
            ).as_single_block()
            prediction = pp.predict_sample_block(sample)
            predicted_mask = prediction.blocks["masks"].data.data
            assert predicted_mask.shape == mask.shape


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
        checkpoint_path, decoder_path = _get_checkpoint(model_type, checkpoint_path, tmp_dir)
        input_paths, result_paths = _create_test_inputs_and_outputs(
            image, label_image, model_type, checkpoint_path, tmp_dir,
        )
        input_descriptions = [
            # First input: the image data.
            spec.InputTensorDescr(
                id=spec.TensorId("image"),
                axes=[
                    spec.BatchAxis(size=1),
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
                    spec.BatchAxis(size=1),
                    spec.IndexInputAxis(
                        id=spec.AxisId("object"),
                        size=spec.ARBITRARY_SIZE
                    ),
                    spec.ChannelAxis(channel_names=[spec.Identifier(bname) for bname in "hwxy"]),
                ],
                test_tensor=spec.FileDescr(source=input_paths["box_prompts"]),
                data=spec.IntervalOrRatioDataDescr(type="int64")
            ),

            # Third input: the point prompt coordinates (optional)
            spec.InputTensorDescr(
                id=spec.TensorId("point_prompts"),
                optional=True,
                axes=[
                    spec.BatchAxis(size=1),
                    spec.IndexInputAxis(
                        id=spec.AxisId("object"),
                        size=spec.ARBITRARY_SIZE
                    ),
                    spec.IndexInputAxis(
                        id=spec.AxisId("point"),
                        size=spec.ARBITRARY_SIZE
                    ),
                    spec.ChannelAxis(channel_names=[spec.Identifier(bname) for bname in "xy"]),
                ],
                test_tensor=spec.FileDescr(source=input_paths["point_prompts"]),
                data=spec.IntervalOrRatioDataDescr(type="int64")
            ),

            # Fourth input: the point prompt labels (optional)
            spec.InputTensorDescr(
                id=spec.TensorId("point_labels"),
                optional=True,
                axes=[
                    spec.BatchAxis(size=1),
                    spec.IndexInputAxis(
                        id=spec.AxisId("object"),
                        size=spec.ARBITRARY_SIZE
                    ),
                    spec.IndexInputAxis(
                        id=spec.AxisId("point"),
                        size=spec.ARBITRARY_SIZE
                    ),
                ],
                test_tensor=spec.FileDescr(source=input_paths["point_labels"]),
                data=spec.IntervalOrRatioDataDescr(type="int64")
            ),

            # Fifth input: the mask prompts (optional)
            spec.InputTensorDescr(
                id=spec.TensorId("mask_prompts"),
                optional=True,
                axes=[
                    spec.BatchAxis(size=1),
                    spec.IndexInputAxis(
                        id=spec.AxisId("object"),
                        size=spec.ARBITRARY_SIZE
                    ),
                    spec.ChannelAxis(channel_names=["channel"]),
                    spec.SpaceInputAxis(id=spec.AxisId("y"), size=256),
                    spec.SpaceInputAxis(id=spec.AxisId("x"), size=256),
                ],
                test_tensor=spec.FileDescr(source=input_paths["mask_prompts"]),
                data=spec.IntervalOrRatioDataDescr(type="float32")
            ),

            # Sixth input: the image embeddings (optional)
            spec.InputTensorDescr(
                id=spec.TensorId("embeddings"),
                optional=True,
                axes=[
                    spec.BatchAxis(size=1),
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
                    spec.BatchAxis(size=1),
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
                    spec.BatchAxis(size=1),
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
                    spec.BatchAxis(size=1),
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

        doc_path = _write_documentation(kwargs.get("documentation", None), model_type, tmp_dir)

        covers = kwargs.get("covers", None)
        if covers is None:
            covers = _generate_covers(input_paths, result_paths, tmp_dir)
        else:
            assert all(os.path.exists(cov) for cov in covers)

        # the uploader information is only added if explicitly passed
        extra_kwargs = {}
        if "id" in kwargs:
            extra_kwargs["id"] = kwargs["id"]
        if "id_emoji" in kwargs:
            extra_kwargs["id_emoji"] = kwargs["id_emoji"]
        if "uploader" in kwargs:
            extra_kwargs["uploader"] = kwargs["uploader"]

        if decoder_path is not None:
            extra_kwargs["attachments"] = [spec.FileDescr(source=decoder_path)]

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
            **extra_kwargs,
            # TODO write specific settings in the config
            # dict with yaml values, key must be a str
            # micro_sam: ...
            # config=
        )

        _check_model(model_description, input_paths, result_paths)

        save_bioimageio_package(model_description, output_path=output_path)
