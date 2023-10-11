# TODO combined script
from micro_sam.modelzoo.image_encoder_export import export_image_encoder
from micro_sam.modelzoo.onnx_export import export_onnx

model_type = "vit_b"
export_image_encoder(model_type, "./test-export")
export_onnx(model_type, "./test-export", opset=12)
