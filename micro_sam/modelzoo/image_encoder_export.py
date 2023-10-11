import os
import torch
from ..util import get_sam_model


ENCODER_CONFIG = """name: "sam-backbone"
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


def export_image_encoder(
    model_type,
    output_root,
    checkpoint_path=None,
):
    output_folder = os.path.join(output_root, "sam-backbone")
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
        f.write(ENCODER_CONFIG)
