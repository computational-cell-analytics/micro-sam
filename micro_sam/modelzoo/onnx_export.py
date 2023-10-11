import os
import warnings

import torch
from segment_anything.utils.onnx import SamOnnxModel

try:
    import onnxruntime
    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

from ..util import get_sam_model


# TODO check if this is still correct
DECODER_CONFIG = """name: "sam-decoder"
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


def to_numpy(tensor):
    return tensor.cpu().numpy()


# ONNX export script adapted from
# https://github.com/facebookresearch/segment-anything/blob/main/scripts/export_onnx_model.py
def export_onnx(
    model_type,
    output_root,
    opset: int,
    checkpoint_path=None,
    return_single_mask: bool = True,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics: bool = False,
):
    output_folder = os.path.join(output_root, "sam-decoder")
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

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

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
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        # set cpu provider default
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(weight_path, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")

    config_output_path = os.path.join(output_folder, "config.pbtxt")
    with open(config_output_path, "w") as f:
        f.write(DECODER_CONFIG)
