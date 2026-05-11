# Example for model finetuning

This folder contains example scripts that show how to finetune a SAM model on your own data and how the finetuned model can then be used:
- `finetune_hela.py`: Shows how to finetune the model on new data. Set `train_instance_segmentation` (line 130) to `True` in order to also train a decoder for automatic instance segmentation.
- `finetune_hela_instance_segmentation.py`: Shows how to train **only** the instance segmentation decoder (UNETR) without updating the interactive segmentation components (prompt encoder and mask decoder). Use this when you only need automatic instance segmentation (AIS) and not interactive segmentation. The trained model can be exported with `export_instance_segmentation_model` and used with the micro_sam AIS functionality.
- `annotator_with_finetuned_model.py`: Use the finetuned model in the 2d annotator.
