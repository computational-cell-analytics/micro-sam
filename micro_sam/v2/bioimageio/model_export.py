"""Export micro-sam2 models to the BioImage.IO model format.

An exported model does interactive segmentation with box, point or mask prompts. If the weights hold
a UniSAM2 decoder, it also does automatic instance segmentation (AIS) when you give no prompts.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

import bioimageio.core
import bioimageio.spec.model.v0_5 as spec
from bioimageio.spec import save_bioimageio_package
from bioimageio.core.digest_spec import create_sample_for_model

import torch

from ...util import load_checkpoint
from ...v1.prompt_based_segmentation import _compute_logits_from_mask
from ..util import FINETUNED_MODELS, _download_finetuned_sam2_model, _read_sam2_checkpoint, detect_sam2_backbone
from .predictor_adaptor import SAM2PredictorAdaptor


DEFAULTS = {
    "authors": [
        spec.Author(name="Anwai Archit", affiliation="University Goettingen", github_user="anwai98"),
        spec.Author(name="Constantin Pape", affiliation="University Goettingen", github_user="constantinpape"),
    ],
    "description": "Segment Anything 2 for Microscopy (micro-sam2), for interactive and automatic segmentation",
    "cite": [
        spec.CiteEntry(
            text="Archit et al. Segment Anything for Microscopy", doi=spec.Doi("10.1038/s41592-024-02580-4"),
        ),
    ],
    "tags": [
        "segment-anything", "segment-anything-2", "micro-sam", "instance-segmentation", "interactive-segmentation",
    ],
}

ANY_SIZE = spec.ParameterizedSize(min=1, step=1)
# The SAM2 mask decoder takes mask prompts as logits of this size.
MASK_PROMPT_SIZE = 256


def _get_weights(model_type, checkpoint_path, decoder_path, tmp_dir):
    """Save the SAM2 weights and the decoder weights in one file, as the adaptor loads them.

    Returns:
        The path of the weights, the SAM2 backbone name and whether the weights hold a decoder.
    """
    if checkpoint_path is None:
        if model_type not in FINETUNED_MODELS:
            raise ValueError(f"'{model_type}' is not a finetuned micro-sam2 model. Give a 'checkpoint_path'.")
        checkpoint_path, _, decoder_path = _download_finetuned_sam2_model(model_type)

    state, model_state, encoder_only = _read_sam2_checkpoint(checkpoint_path)
    if encoder_only:
        raise ValueError(f"'{checkpoint_path}' holds only an image encoder. Give the checkpoint of the SAM2 model.")
    if isinstance(state, dict) and state.get("peft_kwargs") is not None:
        raise ValueError("The export does not support PEFT checkpoints. Merge the PEFT weights first.")

    # A joint checkpoint holds the decoder together with the SAM2 weights.
    decoder_state = state.get("decoder_state") if isinstance(state, dict) else None
    if decoder_path is not None:
        decoder_state = load_checkpoint(decoder_path)
        decoder_state = decoder_state.get("model_state", decoder_state)
    if decoder_state is not None:
        if "out_conv.weight" not in decoder_state:
            raise ValueError("The decoder weights hold no UniSAM2 decoder.")
        # The decoder reuses the SAM2 image encoder, so the encoder weights are saved only once.
        encoder_state = {k: v for k, v in decoder_state.items() if k.startswith("encoder.inner.")}
        for key, value in encoder_state.items():
            if not torch.equal(value, model_state["image_encoder." + key[len("encoder.inner."):]]):
                raise ValueError("The image encoder of the decoder weights differs from the SAM2 model.")
        decoder_state = {k: v for k, v in decoder_state.items() if not k.startswith("encoder.inner.")}

    weights = {"model_state": model_state}
    if decoder_state is not None:
        weights["decoder_state"] = decoder_state
    weight_path = os.path.join(tmp_dir, f"{model_type}.pt")
    torch.save(weights, weight_path)
    return weight_path, detect_sam2_backbone(model_state), decoder_state is not None


def _to_image_tensor(image):
    """Bring a 2d or channel-last image to the input layout: batch x channel x y x x."""
    image = np.asarray(image, dtype="float32")
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(f"The image must be 2d or channel-last with 1 or 3 channels, got shape {image.shape}.")
    return image.transpose(2, 0, 1)[None]


def _create_test_inputs(image, label_image, tmp_dir):
    """Save the test image and a box, a point and a mask prompt for each of the two largest objects."""
    inputs = {"image": os.path.join(tmp_dir, "image.npy")}
    np.save(inputs["image"], _to_image_tensor(image))

    seg_ids, sizes = np.unique(label_image[label_image != 0], return_counts=True)
    if len(seg_ids) == 0:
        raise ValueError("The label image has no objects to derive the test prompts from.")
    boxes, points, masks = [], [], []
    for seg_id in seg_ids[np.argsort(sizes)[::-1][:2]]:
        masks.append(_compute_logits_from_mask(label_image == seg_id))
        coords = np.argwhere(label_image == seg_id)
        (y0, x0), (y1, x1) = coords.min(axis=0), coords.max(axis=0)
        boxes.append([x0, y0, x1, y1])
        # The object pixel that is nearest to the center, so that the point is inside the object.
        y, x = coords[np.argmin(((coords - coords.mean(axis=0)) ** 2).sum(axis=1))]
        points.append([[x, y]])

    prompts = {
        "box_prompts": np.array(boxes, dtype="int64")[None],
        "point_prompts": np.array(points, dtype="int64")[None],
        "point_labels": np.ones((1, len(points), 1), dtype="int64"),
        "mask_prompts": np.stack(masks).astype("float32")[None],
    }
    for name, prompt in prompts.items():
        inputs[name] = os.path.join(tmp_dir, f"{name}.npy")
        np.save(inputs[name], prompt)
    return inputs


def _to_label_image(masks):
    """Merge the binary masks of the objects (1 x OBJECTS x 1 x H x W) into one label image."""
    labels = np.zeros(masks.shape[-2:], dtype="uint32")
    for seg_id, mask in enumerate(masks[0, :, 0], start=1):
        labels[mask > 0] = seg_id
    return labels


def _write_documentation(documentation, model_type, with_decoder, tmp_dir):
    if documentation is not None and os.path.exists(documentation):
        return documentation

    doc_path = os.path.join(tmp_dir, "documentation.md")
    if documentation is None:
        automatic = (
            "Automatic instance segmentation: give only the image. The model segments all objects with its "
            "UniSAM2 decoder (AIS) and returns one mask per object.\n"
        ) if with_decoder else (
            "This model has no decoder for automatic instance segmentation, so you must give prompts.\n"
        )
        documentation = (
            f"# Segment Anything 2 for Microscopy: {model_type}\n\n"
            "micro-sam2 finetunes Segment Anything 2 (SAM2) for microscopy. This model does interactive "
            "segmentation with prompts and, if it has a decoder, automatic instance segmentation.\n\n"
            "## How to use the model\n\n"
            "- The image has 1 or 3 channels and any intensity range. The model normalizes it.\n"
            "- Interactive segmentation: give box prompts (x_min, y_min, x_max, y_max), point prompts (x, y) "
            "with point labels (1 positive, 0 negative), or mask prompts (logits of size 256 x 256). The model "
            "returns one mask and one quality score per object.\n"
            f"- {automatic}\n"
            "The tool [micro_sam](https://github.com/computational-cell-analytics/micro-sam) also loads the "
            "weights of this model, for the napari annotation tools, the command line and the python library.\n"
        )
    with open(doc_path, "w") as f:
        f.write(documentation)
    return doc_path


def _write_dependencies(tmp_dir):
    dependency_file = os.path.join(tmp_dir, "environment.yaml")
    with open(dependency_file, "w") as f:
        f.write(
            "name: micro-sam2\nchannels:\n    - conda-forge\ndependencies:\n    - micro_sam >=2.0\n    - pip\n"
            "    - pip:\n        - git+https://github.com/facebookresearch/sam2.git\n"
        )
    return dependency_file


def _generate_cover(input_paths, masks, show_boxes, tmp_dir):
    """Show the image and the outlines of the predicted objects side by side."""
    image = np.load(input_paths["image"])[0].transpose(1, 2, 0).astype("float32")
    low, high = np.percentile(image, (1, 99))
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    overlay = image.copy()
    for mask in masks[0, :, 0]:
        overlay[find_boundaries(mask > 0, mode="thick")] = (1.0, 0.8, 0.0)
    if show_boxes:
        for x0, y0, x1, y1 in np.load(input_paths["box_prompts"])[0]:
            overlay[y0:y1 + 1, [x0, x1]] = overlay[[y0, y1], x0:x1 + 1] = (0.0, 0.6, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, data in zip(axes, (image, overlay)):
        ax.imshow(data)
        ax.axis("off")
    cover_path = os.path.join(tmp_dir, "cover.png")
    plt.savefig(cover_path, bbox_inches="tight", dpi=100)
    plt.close()
    return cover_path


def _build_model_description(name, input_paths, output_paths, weights, doc_path, covers, **kwargs):
    image_shape = np.load(input_paths["image"], mmap_mode="r").shape
    channel_names = ["gray"] if image_shape[1] == 1 else ["red", "green", "blue"]

    def _test_tensor(name):
        return spec.FileDescr(source=input_paths[name])

    object_axis = spec.IndexInputAxis(id=spec.AxisId("object"), size=ANY_SIZE)
    point_axis = spec.IndexInputAxis(id=spec.AxisId("point"), size=ANY_SIZE)
    inputs = [
        spec.InputTensorDescr(
            id=spec.TensorId("image"),
            description="The image. The model normalizes the intensities.",
            axes=[
                spec.BatchAxis(size=1),
                spec.ChannelAxis(channel_names=[spec.Identifier(c) for c in channel_names]),
                spec.SpaceInputAxis(id=spec.AxisId("y"), size=ANY_SIZE),
                spec.SpaceInputAxis(id=spec.AxisId("x"), size=ANY_SIZE),
            ],
            test_tensor=_test_tensor("image"),
            data=spec.IntervalOrRatioDataDescr(type="float32"),
        ),
        spec.InputTensorDescr(
            id=spec.TensorId("box_prompts"),
            description="One box per object: x_min, y_min, x_max, y_max.",
            optional=True,
            axes=[
                spec.BatchAxis(size=1), object_axis,
                spec.ChannelAxis(channel_names=[spec.Identifier(c) for c in ("x_min", "y_min", "x_max", "y_max")]),
            ],
            test_tensor=_test_tensor("box_prompts"),
            data=spec.IntervalOrRatioDataDescr(type="int64"),
        ),
        spec.InputTensorDescr(
            id=spec.TensorId("point_prompts"),
            description="The point coordinates per object: x, y.",
            optional=True,
            axes=[
                spec.BatchAxis(size=1), object_axis, point_axis,
                spec.ChannelAxis(channel_names=[spec.Identifier(c) for c in "xy"]),
            ],
            test_tensor=_test_tensor("point_prompts"),
            data=spec.IntervalOrRatioDataDescr(type="int64"),
        ),
        spec.InputTensorDescr(
            id=spec.TensorId("point_labels"),
            description="The label of each point: 1 is positive, 0 is negative.",
            optional=True,
            axes=[spec.BatchAxis(size=1), object_axis, point_axis],
            test_tensor=_test_tensor("point_labels"),
            data=spec.IntervalOrRatioDataDescr(type="int64"),
        ),
        spec.InputTensorDescr(
            id=spec.TensorId("mask_prompts"),
            description="One mask prompt per object, as logits.",
            optional=True,
            axes=[
                spec.BatchAxis(size=1), object_axis,
                spec.ChannelAxis(channel_names=[spec.Identifier("mask")]),
                spec.SpaceInputAxis(id=spec.AxisId("y"), size=MASK_PROMPT_SIZE),
                spec.SpaceInputAxis(id=spec.AxisId("x"), size=MASK_PROMPT_SIZE),
            ],
            test_tensor=_test_tensor("mask_prompts"),
            data=spec.IntervalOrRatioDataDescr(type="float32"),
        ),
    ]

    object_out_axis = spec.IndexOutputAxis(id=spec.AxisId("object"), size=spec.DataDependentSize())
    outputs = [
        spec.OutputTensorDescr(
            id=spec.TensorId("masks"),
            description="One binary mask per object.",
            axes=[
                spec.BatchAxis(size=1), object_out_axis,
                spec.ChannelAxis(channel_names=[spec.Identifier("mask")]),
                spec.SpaceOutputAxis(
                    id=spec.AxisId("y"),
                    size=spec.SizeReference(tensor_id=spec.TensorId("image"), axis_id=spec.AxisId("y")),
                ),
                spec.SpaceOutputAxis(
                    id=spec.AxisId("x"),
                    size=spec.SizeReference(tensor_id=spec.TensorId("image"), axis_id=spec.AxisId("x")),
                ),
            ],
            data=spec.IntervalOrRatioDataDescr(type="uint8"),
            test_tensor=spec.FileDescr(source=output_paths["masks"]),
        ),
        spec.OutputTensorDescr(
            id=spec.TensorId("scores"),
            description="The predicted quality of each mask. The scores are 1 for automatic segmentation.",
            axes=[spec.BatchAxis(size=1), object_out_axis, spec.ChannelAxis(channel_names=[spec.Identifier("score")])],
            data=spec.IntervalOrRatioDataDescr(type="float32"),
            test_tensor=spec.FileDescr(source=output_paths["scores"]),
        ),
    ]

    extra_kwargs = {key: kwargs[key] for key in ("id", "id_emoji", "uploader", "version") if key in kwargs}
    return spec.ModelDescr(
        name=name,
        description=kwargs.get("description", DEFAULTS["description"]),
        authors=kwargs.get("authors", DEFAULTS["authors"]),
        cite=kwargs.get("cite", DEFAULTS["cite"]),
        license=spec.LicenseId(kwargs.get("license", "CC-BY-4.0")),
        documentation=spec.FileDescr(source=Path(doc_path)),
        git_repo=spec.HttpUrl("https://github.com/computational-cell-analytics/micro-sam"),
        tags=kwargs.get("tags", DEFAULTS["tags"]),
        covers=[spec.FileDescr(source=Path(cover)) for cover in covers],
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        **extra_kwargs,
    )


def _predict(model_description, inputs):
    """Run the model through BioImage.IO on the CPU."""
    sample = create_sample_for_model(model=model_description, inputs=inputs)
    with bioimageio.core.create_prediction_pipeline(model_description, devices=["cpu"]) as pp:
        prediction = pp.predict_sample_without_blocking(sample)
    return {name: np.asarray(prediction.members[name].data) for name in ("masks", "scores")}


def _check_automatic(model_description, image_path, expected_masks):
    """Check that BioImage.IO runs AIS when it gets only the image, with the expected objects."""
    result = _predict(model_description, {"image": np.load(image_path)})
    if result["masks"].shape != expected_masks.shape or not np.array_equal(result["masks"], expected_masks):
        raise RuntimeError(
            f"Automatic segmentation through BioImage.IO gives {result['masks'].shape[1]} objects, "
            f"but the model gives {expected_masks.shape[1]} objects."
        )


def export_sam2_model(
    image: np.ndarray,
    label_image: Optional[np.ndarray],
    model_type: str,
    name: str,
    output_path: Union[str, os.PathLike],
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    decoder_path: Optional[Union[str, os.PathLike]] = None,
    **kwargs,
) -> None:
    """Export a micro-sam2 model to the BioImage.IO model format.

    The exported model does interactive segmentation with box, point and mask prompts. If the weights
    hold a UniSAM2 decoder, it also does automatic instance segmentation (AIS) when you give only the image.
    The package test runs box and point prompts. For weights with a decoder, the function also checks
    AIS through BioImage.IO before it saves the package.

    Args:
        image: The test image, 2d or channel-last with 1 or 3 channels.
        label_image: The segmentation of the test image. The function derives the test prompts from it.
            For weights with a decoder, the AIS result of the model is used by default.
        model_type: The name of a finetuned micro-sam2 model, e.g. 'hvit_t_cells'. With `checkpoint_path`,
            the name of your model.
        name: The name of the exported model.
        output_path: The path of the exported package, a zip file.
        checkpoint_path: The path of your SAM2 checkpoint. By default the weights of `model_type` are downloaded.
            A joint checkpoint with 'model_state' and 'decoder_state' also gives the decoder.
        decoder_path: The path of your UniSAM2 decoder checkpoint.
        kwargs: Optional metadata: 'description', 'authors', 'cite', 'tags', 'license', 'documentation'
            (text or a file path), 'covers' (file paths), 'id', 'id_emoji', 'uploader' and 'version'.

    Raises:
        RuntimeError: If the exported model fails a check.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        weight_path, backbone, with_decoder = _get_weights(model_type, checkpoint_path, decoder_path, tmp_dir)
        if not with_decoder and label_image is None:
            raise ValueError("Give a 'label_image' to derive the test prompts for weights without a decoder.")

        # The reference outputs come from the adaptor on the CPU, as BioImage.IO runs it in the test.
        adaptor = SAM2PredictorAdaptor(backbone)
        adaptor.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        adaptor.eval()
        image_tensor = torch.from_numpy(_to_image_tensor(image))
        automatic_masks = None
        if with_decoder:
            automatic_masks, _ = adaptor(image_tensor)
            automatic_masks = automatic_masks.numpy()
            if automatic_masks.shape[1] == 0:
                raise RuntimeError("AIS finds no objects in the test image. Use an image with objects.")
            if label_image is None:
                label_image = _to_label_image(automatic_masks)

        input_paths = _create_test_inputs(image, label_image, tmp_dir)
        masks, scores = adaptor(**{key: torch.from_numpy(np.load(path)) for key, path in input_paths.items()})
        output_paths = {key: os.path.join(tmp_dir, f"{key}.npy") for key in ("masks", "scores")}
        np.save(output_paths["masks"], masks.numpy())
        np.save(output_paths["scores"], scores.numpy())
        del adaptor

        weights = spec.WeightsDescr(
            pytorch_state_dict=spec.PytorchStateDictWeightsDescr(
                source=Path(weight_path),
                architecture=spec.ArchitectureFromFileDescr(
                    source=Path(os.path.join(os.path.dirname(__file__), "predictor_adaptor.py")),
                    callable="SAM2PredictorAdaptor",
                    kwargs={"model_type": backbone},
                ),
                pytorch_version=spec.Version(torch.__version__),
                dependencies=spec.FileDescr(source=Path(_write_dependencies(tmp_dir))),
            )
        )
        doc_path = _write_documentation(kwargs.pop("documentation", None), model_type, with_decoder, tmp_dir)
        covers = kwargs.pop("covers", None)
        if covers is None:
            # The cover shows all objects for AIS, and the prompted objects otherwise.
            cover_masks = masks.numpy() if automatic_masks is None else automatic_masks
            covers = [_generate_cover(input_paths, cover_masks, automatic_masks is None, tmp_dir)]
        model_description = _build_model_description(
            name, input_paths, output_paths, weights, doc_path, covers, **kwargs,
        )

        summary = bioimageio.core.test_model(model_description, devices=["cpu"])
        if summary.status != "passed":
            summary.display()
            raise RuntimeError(f"The exported model '{name}' fails the BioImage.IO test.")
        # The package test runs only the prompts, so the function checks AIS here.
        if with_decoder:
            _check_automatic(model_description, input_paths["image"], automatic_masks)

        save_bioimageio_package(model_description, output_path=output_path)
