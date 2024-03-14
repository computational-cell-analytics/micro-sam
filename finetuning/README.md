# Segment Anything Finetuning

Code for finetuning segment anything data on microscopy data and evaluating the finetuned models.

## Example: LIVECell

**Finetuning**

Run the script `livecell_finetuning.py` for fine-tuning a model on LIVECell. Run the following script for details on how to run the scripts:

```
python livecell_finetuning.py -h
```

**Inference**

TODO: update this as this changes

The script `livecell_inference.py` can be used to run inference on the test set. It supports different arguments for inference with different configurations.
For example run:
```
python livecell_inference.py -c checkpoints/livecell_sam/best.pt -m vit_b -e experiment -i /scratch/projects/nim00007/data/LiveCELL --points --positive 1 --negative 0
```
for inference with 1 positive point prompt and no negative point prompt (the prompts are derived from ground-truth).

The arguments `-c`, `-e` and `-i` specify where the checkpoint for the model is, where the predictions from the model and other experiment data will be saved, and where the input dataset (LiveCELL) is stored.

To run the default set of experiments from our publication use the command
```
python livecell_inference.py -c checkpoints/livecell_sam/best.pt -m vit_b -e experiment -i /scratch/projects/nim00007/data/LiveCELL -d --prompt_folder /scratch/projects/nim00007/sam/experiments/prompts/livecell 
```

Here `-d` automatically runs the evaluation for these settings:
- `--points --positive 1 --negative 0` (using point prompts with a single positive point)
- `--points --positive 2 --negative 4` (using point prompts with two positive points and four negative points)
- `--points --positive 4 --negative 8` (using point prompts with four positive points and eight negative points)
- `--box` (using box prompts)

In addition `--prompt_folder` specifies a folder with precomputed prompts. Using pre-computed prompts significantly speeds up the experiments and enables running them in a reproducible manner. (Without it the prompts will be recalculated each time.)

You can also evaluate the automatic instance segmentation functionality, by running
```
python livecell_inference.py -c checkpoints/livecell_sam/best.pt -m vit_b -e experiment -i /scratch/projects/nim00007/data/LiveCELL -a 
```

This will first perform a grid-search for the best parameters on a subset of the validation set and then run inference on the test set. This can take up to a day.

**Evaluation**

TODO: update this as this changes as well

The script `livecell_evaluation.py` can then be used to evaluate the results from the inference runs.
E.g. run the script like below to evaluate the previous predictions.
```
python livecell_evaluation.py -i /scratch/projects/nim00007/data/LiveCELL -e experiment
```
This will create a folder `experiment/results` with csv tables with the results per cell type and averaged over all images.


## Finetuning and evaluation code

TODO: hint the users to the `generalist/`, `specialist/` and `evaluation/` scripts with a bit of hints

The subfolders contain the code for different finetuning and evaluation experiments for microscopy data:
- `livecell`: TODO
- `generalist`: TODO

Note: we still need to clean up most of this code and will add it later.
