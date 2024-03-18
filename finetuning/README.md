# Segment Anything Finetuning

Code for finetuning segment anything data on microscopy data and evaluating the finetuned models.

## Example: LIVECell

**Finetuning**

The script `livecell_finetuning.py` can be used for finetuning Segment Anything models on LIVECell. Run `python livecell_finetuning.py -h` for details on how to run the scripts.
Here is an example run for finetuning:

```bash
$ python livecell_finetuning.py -i /path/to/livecell
                                -m vit_b
                                -s /path/to/save/checkpoints
                                --export_path /path/to/save/exported/model.pth
```
The arguments `-i`, `-m`, `-s` and `--export_path`specify where the input dataset (LIVECell) is stored, which Segment Anything model to finetune, where the checkpoints and logs for the finetuned models will be stored, and the exported model for making use of the annotator tool, respectively.

**Inference**

The script `livecell_inference.py` can be used to run inference on the test set. It supports different arguments for inference with different configurations. Run `python livecell_inference.py -h` for details on how to run the scripts.
Here is an example run for inference:

```bash
$ python livecell_inference.py -c /path/to/saved/checkpoints
                               -i /path/to/livecell
                               -e /path/to/store/experiment
                               -m vit_b
                               # choice of inference:
                               #    - ('-p') precompute image embeddings
                               #    - ('-ip') iterative prompting-based interactive instance segmentation 
                               #        - default: starting with point
                               #        - ('-b') starting with box
                               #        - ('--use_masks') use logits from previous iteration's segmentation iteratively
                               #    - ('-amg') automatic mask generation
                               #    - ('-ais') automatic instance segmentation
```
The arguments `-c`, `-i`, `-e` and `m` specify where the checkpoint for the model is, where the input dataset (LiveCELL) is stored, where the predictions from the model and other experiment data will be saved, and the model name for the model checkpoint, respectively.

To run the default set of experiments from our publication use the command:
```bash
$ python livecell_inference.py -c /path/to/saved/checkpoints -i /path/to/livecell -e /path/to/store/experiment -m vit_b -p  # precompute the embeddings
$ python livecell_inference.py -c /path/to/saved/checkpoints -i /path/to/livecell -e /path/to/store/experiment -m vit_b -ip  # iterative prompting starting with point
$ python livecell_inference.py -c /path/to/saved/checkpoints -i /path/to/livecell -e /path/to/store/experiment -m vit_b -ip -b  # iterative prompting starting with box
```

Here, `-ip` stands for iterative prompting-based interactive instance segmentation (i.e. starting with placing 1 positive point prompt OR a box to segment the object, and continuing prompt-based segmentation by placing additional points subjected to the model's prediction - where the model makes mistakes (to add a negative point prompt) and where the model missed to segment the object of interest (to add a positive point prompt))

You can also evaluate the automatic instance segmentation functionality, by running
```bash
$ python livecell_inference.py -c /path/to/saved/checkpoints -i /path/to/livecell -e /path/to/store/experiment -m vit_b -amg  # automatic mask generation
$ python livecell_inference.py -c /path/to/saved/checkpoints -i /path/to/livecell -e /path/to/store/experiment -m vit_b -ais  # automatic instance segmentation
```

This will first perform a grid-search for the best parameters on a subset of the validation set and then run inference on the test set. This can take up to a day.

**Evaluation**

The script `livecell_evaluation.py` can be used to evaluate the results from the inference runs on the test set. Run `python livecell_evaluation.py -h` for details on how to run the scripts.
Here is an example run for evaluation:

```bash
$ python livecell_evaluation.py -i /path/to/livecell -e /path/to/stored/experiments
```
This will create a folder `experiment/results` with csv tables with the results per cell type and averaged over all images.


## Finetuning and Evaluation code

TODO: hint the users to the `generalist/`, `specialist/` and `evaluation/` scripts with a bit of hints

The subfolders contain the code for different finetuning and evaluation experiments for microscopy data:
- `livecell`: TODO
- `generalist`: TODO
- `specialist`: TODO
- `evaluation`: TODO

Note: we still need to clean up most of this code and will add it later.
