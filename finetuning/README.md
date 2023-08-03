# Segment Anything Finetuning

Preliminary examples for fine-tuning segment anything on custom datasets.

## LiveCELL

**Finetuning**

Run the script `livecell_finetuning.py` for fine-tuning a model on LiveCELL.

**Inference**

The script `livecell_inference.py` can be used to run inference on the test set. It supports different arguments for inference with different configurations.
For example run
```
python livecell_inference.py -c checkpoints/livecell_sam/best.pt --p predictions/finetuned-vitb-v1 -i /scratch/projects/nim00007/data/LiveCELL/ --points --positive 1 --negative 0
```
for inference with 1 positive point prompt and no negative point prompt (the prompts are derived from ground-truth).

The arguments `-c`, `-p` and `-i` specify where the checkpoint for the model is, where the predictions from the model will be saved, and where the input (LiveCELL) dataset is stored.

Our standard evaluation procedure uses the following three settings:
- `--points --positive 1 --negative 0` (using point prompts with a single positive point)
- `--points --positive 2 --negative 4` (using point prompts with two positive points and four negative points)
- `--box` (using box prompts)

**Evaluation**

The script `livecell_evaluation.py` can then be used to evaluate the results from the inference runs.


