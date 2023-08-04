# Segment Anything Finetuning

Preliminary examples for fine-tuning segment anything on custom datasets.

## LiveCELL

**Finetuning**

Run the script `livecell_finetuning.py` for fine-tuning a model on LiveCELL.

**Inference**

The script `livecell_inference.py` can be used to run inference on the test set. It supports different arguments for inference with different configurations.
For example run
```
python livecell_inference.py -c checkpoints/livecell_sam/best.pt -m vit_b -e experiment -i /scratch/projects/nim00007/data/LiveCELL/ --points --positive 1 --negative 0
```
for inference with 1 positive point prompt and no negative point prompt (the prompts are derived from ground-truth).

The arguments `-c`, `-e` and `-i` specify where the checkpoint for the model is, where the predictions from the model and other experiment data will be saved, and where the input dataset (LiveCELL) is stored.

Our standard evaluation procedure uses the following three settings:
- `--points --positive 1 --negative 0` (using point prompts with a single positive point)
- `--points --positive 2 --negative 4` (using point prompts with two positive points and four negative points)
- `--box` (using box prompts)

**Evaluation**

The script `livecell_evaluation.py` can then be used to evaluate the results from the inference runs.
E.g. run the script like below to evaluate the previous predictions.
```
python livecell_evaluation.py -i /scratch/projects/nim00007/data/LiveCELL -p predictions --name finetuned_livecell
```
This will create a folder `results` with csv tables with the results per cell type and averaged over all images.
