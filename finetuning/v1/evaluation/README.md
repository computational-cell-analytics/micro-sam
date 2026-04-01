# Segment Anything Evaluation

Scripts for evaluating Segment Anything models and the finetuned `micro_sam` models.

## Benchmark Methods

- `cellpose_baseline.py`: Script to benchmark CellPose in light microscopy datasets.
- `mitonet_baseline.py`: Script to benchmark MitoNet (Empanada) on mitochondria segmentation in electron microscopy datasets.
    - NOTE: The MitoNet scripts are for assorting inputs to perform segmentation using the napari plugin.

## Evaluation Code

To run the evaluations on your custom dataset, you need to adapt* the scripts a bit.

- `precompute_embeddings.py`: Script to precompute the image embeddings and store it for following evaluation scripts.
- `evaluate_amg.py`: Script to run Automatic Mask Generation (AMG), the "Segment Anything" feature.
- `evaluate_instance_segmentation`: Script to run Automatic Instance Segmentation (AIS), the new feature in micro-sam with added decoder to perform instance segmentation.
- `iterative_prompting.py`: Script to run iterative prompting** (interactive instance segmentation) with respect to the true labels.

Know more about the scripts above and the expected arguments using `<SCRIPT>.py -h`.

TLDR: The most important arguments to be passed are the hinted below:
```bash
python <SCRIPT>.py -m <MODEL_NAME>  # the segment anything model type
                   -c <CHECKPOINT_PATH>  # path to the model checkpoint (default or finetuned models)
                   -e <EXPERIMENT_FOLDER>  # path to store all the evaluations
                   -d <DATASET_NAME>  # not relevant*
                   # arguments relevant for iterative prompting**
                   --box  # starting with box prompt
                   --use_masks  # use logits masks from previous iterations
```

### How to run the evaluation scripts on your own data?

Below are some specifications on how to run the evaluation scripts (automatic mask generation (AMG), automatic instance segmentation (AIS) and interactive instance segmentation (using iterative prompting starting with points / box and improving the segmentation quality with positive and / or negative points))
- \*adapt: For making your scripts work on your custom dataset, you need to overwrite the `get_paths` function to create your own heuristic to pass the paths to all the images (and respective labels). Here's an example overwrite (REMEMBER: to comment out the `get_paths` import from `util`):

    ```python
    import os
    from glob import glob

    def get_paths(dataset="sample", split="test"):
        root_dir = "/path/to/my/data"
        image_paths = sorted(glob(os.path.join(root_dir, split, "images", "*")))
        gt_paths = sorted(glob(os.path.join(root, split, "gt", "*")))
        return image_paths, gt_paths

    ```
    - We also need the validation set to perform grid-search for the automatic segmentation methods. It's essential to obtain the post-processing parameters for the specific dataset. Make sure to have a validation set (you can choose the validation set subjected to your finetuning, or just make a very small split out of the available dataset)

- **iterative prompting: The method to start interactive segmentation with initial prompts (could be either a positive point or a box), and continuing to (automatically) add point prompts (8 times) to: a) rectify where the model makes mistakes (using negative point prompt) and b) indicate an expected region missed out by the model (using a positive point prompt).


## Additional Scripts:

- `preprocess_data.py`: The scripts used to preprocess open-source datasets for quantitative benchmarking.
- `util.py`: The scripts for taking care of evaluation-based functionalities.
- `time-benchmarking.md`: The inference time benchmarking for Segment Anything models.
- `run_all_evaluation.py`: Convenience scripts to run inference by submitting batch jobs to HLRN via slurm.
- `submit_all_evaluation.py`: Convenience scripts responsible for creating the slurm scripts to submit the batch jobs to HLRN.
- `explorative_experiments.py` (OUTDATED): The scripts to study the different model performances on LIVECell.
