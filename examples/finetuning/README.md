# Example for model finetuning

This folder contains example scripts that show how to finetune a SAM model on your own data and how the finetuned model can then be used:
- `finetune_hela.py`: Shows how to finetune the model on new data. Set `train_instance_segmentation` (line 165) to `True` in order to also train a decoder for automatic instance segmentation.
- `annotator_with_finetuned_model.py`: Use the finetuned model in the 2d annotator.
- `instance_segmentation_with_finetuned_model`: Use the finetuned model for automatic instance segmentation (only if you have trained with `train_instance_segmentation = True`).
