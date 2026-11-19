# Example for model finetuning

This folder contains example scripts that show how to finetune a micro-sam2 model on your own data and how the finetuned model can then be used:
- `finetune_hela.py`: Shows how to finetune the model for interactive segmentation on new data, with `micro_sam.v2.training.train_sam2`.
- `finetune_hela_instance_segmentation.py`: Shows how to finetune the model for automatic instance segmentation (AIS and APG), with `micro_sam.v2.training.train_automatic`. It trains the image encoder together with the decoder, but not the prompt encoder and mask decoder for interactive segmentation. Use this when you mainly need automatic instance segmentation.
- `annotator_with_finetuned_model.py`: Use the finetuned model in the annotator. The model size is read from the checkpoint.

The checkpoints are saved in `./checkpoints/<name>/best.pt` and can be passed directly as `checkpoint_path` to the annotator, or as `checkpoint` to `micro_sam.v2.automatic_segmentation.get_predictor_and_segmenter`.
