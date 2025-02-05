import os
import warnings
from typing import Optional, Union

import torch

from segment_anything.utils.onnx import SamOnnxModel

try:
    import onnxruntime
    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

from ..util import get_sam_model


ENCODER_CONFIG = """name: "%s"
backend: "pytorch"
platform: "pytorch_libtorch"

max_batch_size : 1
input [
  {
    name: "input0__0"
    data_type: TYPE_FP32
    dims: [3, -1, -1]
  }
]
output [
  {
    name: "output0__0"
    data_type: TYPE_FP32
    dims: [256, 64, 64]
  }
]

parameters: {
  key: "INFERENCE_MODE"
  value: {
    string_value: "true"
  }
}"""


DECODER_CONFIG = """name: "%s"
backend: "onnxruntime"
platform: "onnxruntime_onnx"

parameters: {
  key: "INFERENCE_MODE"
  value: {
    string_value: "true"
  }
}

instance_group {
  count: 1
  kind: KIND_CPU
}"""


def _to_numpy(tensor):
    return tensor.cpu().numpy()


def export_image_encoder(
    model_type: str,
    output_root: Union[str, os.PathLike],
    export_name: Optional[str] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Export SAM image encoder to torchscript.

    The torchscript image encoder can be used for predicting image embeddings
    with a backed, e.g. with [the bioengine](https://github.com/bioimage-io/bioengine-model-runner).

    Args:
        model_type: The SAM model type.
        output_root: The output root directory where the exported model is saved.
        export_name: The name of the exported model.
        checkpoint_path: Optional checkpoint for loading the exported model.
    """
    if export_name is None:
        export_name = model_type
    name = f"sam-{export_name}-encoder"

    output_folder = os.path.join(output_root, name)
    weight_output_folder = os.path.join(output_folder, "1")
    os.makedirs(weight_output_folder, exist_ok=True)

    predictor = get_sam_model(model_type=model_type, checkpoint_path=checkpoint_path)
    encoder = predictor.model.image_encoder

    encoder.eval()
    input_ = torch.rand(1, 3, 1024, 1024)
    traced_model = torch.jit.trace(encoder, input_)
    weight_path = os.path.join(weight_output_folder, "model.pt")
    traced_model.save(weight_path)

    config_output_path = os.path.join(output_folder, "config.pbtxt")
    with open(config_output_path, "w") as f:
        f.write(ENCODER_CONFIG % name)


def export_onnx_model(
    model_type: str,
    output_root: Union[str, os.PathLike],
    opset: int = 17,
    export_name: Optional[str] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    return_single_mask: bool = True,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics: bool = False,
    quantize_model: bool = False,
) -> None:
    """Export SAM prompt encoder and mask decoder to onnx.

    The onnx encoder and decoder can be used for interactive segmentation in the browser.
    This code is adapted from
    https://github.com/facebookresearch/segment-anything/blob/main/scripts/export_onnx_model.py

    Args:
        model_type: The SAM model type.
        output_root: The output root directory where the exported model is saved.
        opset: The ONNX opset version. The recommended opset version is 17.
        export_name: The name of the exported model.
        checkpoint_path: Optional checkpoint for loading the SAM model.
        return_single_mask: Whether the mask decoder returns a single or multiple masks.
        gelu_approximate: Whether to use a GeLU approximation, in case the ONNX backend
            does not have an efficient GeLU implementation.
        use_stability_score: Whether to use the stability score instead of the predicted score.
        return_extra_metrics: Whether to return a larger set of metrics.
        quantize_model: Whether to also export a quantized version of the model.
            This only works for onnxruntime < 1.17.
    """
    if export_name is None:
        export_name = model_type
    name = f"sam-{export_name}-decoder"

    output_folder = os.path.join(output_root, name)
    weight_output_folder = os.path.join(output_folder, "1")
    os.makedirs(weight_output_folder, exist_ok=True)

    _, sam = get_sam_model(model_type=model_type, checkpoint_path=checkpoint_path, return_sam=True)
    weight_path = os.path.join(weight_output_folder, "model.onnx")

    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules:
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}}

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(weight_path, "wb") as f:
            print(f"Exporting onnx model to {weight_path}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    if onnxruntime_exists:
        ort_inputs = {k: _to_numpy(v) for k, v in dummy_inputs.items()}
        # set cpu provider default
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(weight_path, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")

    # This requires onnxruntime < 1.17.
    # See https://github.com/facebookresearch/segment-anything/issues/699#issuecomment-1984670808
    if quantize_model:
        assert onnxruntime_exists
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic

        quantized_path = os.path.join(weight_output_folder, "model_quantized.onnx")
        quantize_dynamic(
            model_input=weight_path,
            model_output=quantized_path,
            # optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )

    config_output_path = os.path.join(output_folder, "config.pbtxt")
    with open(config_output_path, "w") as f:
        f.write(DECODER_CONFIG % name)


def export_bioengine_model(
    model_type: str,
    output_root: Union[str, os.PathLike],
    opset: int,
    export_name: Optional[str] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    return_single_mask: bool = True,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics: bool = False,
) -> None:
    """Export SAM model to a format compatible with the BioEngine.

    [The bioengine](https://github.com/bioimage-io/bioengine-model-runner) enables running the
    image encoder on an online backend, so that SAM can be used in an online tool, or to predict
    the image embeddings via the online backend rather than on CPU.

    Args:
        model_type: The SAM model type.
        output_root: The output root directory where the exported model is saved.
        opset: The ONNX opset version.
        export_name: The name of the exported model.
        checkpoint_path: Optional checkpoint for loading the SAM model.
        return_single_mask: Whether the mask decoder returns a single or multiple masks.
        gelu_approximate: Whether to use a GeLU approximation, in case the ONNX backend
            does not have an efficient GeLU implementation.
        use_stability_score: Whether to use the stability score instead of the predicted score.
        return_extra_metrics: Whether to return a larger set of metrics.
    """
    export_image_encoder(model_type, output_root, export_name, checkpoint_path)
    export_onnx_model(
        model_type=model_type,
        output_root=output_root,
        opset=opset,
        export_name=export_name,
        checkpoint_path=checkpoint_path,
        return_single_mask=return_single_mask,
        gelu_approximate=gelu_approximate,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )
