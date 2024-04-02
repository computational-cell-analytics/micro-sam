# Segment Anything Evaluation

Scripts for evaluating Segment Anything models and the finetuned models.

## Benchmark Methods

- `cellpose_baseline.py`: Script to benchmark CellPose in light microscopy datasets.
- `mitonet_baseline.py`: Script to benchmark MitoNet (Empanada) on mitochondria segmentation in electron microscopy datasets.

## Evaluation Code

NOTE: To run the evaluations on your custom dataset, you need to adapt* the scripts a bit.

- `precompute_embeddings.py`: Script to precompute the image embeddings and store it for following evaluation scripts.
- `evaluate_amg.py`: Script to run Automatic Mask Generation (AMG), the "Segment Anything" feature.
- `evaluate_instance_segmentation`: Script to run Automatic Instance Segmentation (AIS), the new feature in micro-sam with added decoder to perform instance segmentation.
- `iterative_prompting.py`: Script to run iterative prompting** (interactive instance segmentation) with respect to the true labels.

To know more about the scripts above and the expected arguments, you can check it out using `<SCRIPT>.py -h`.

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

### NOTE:
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

- **iterative prompting: The method to start interactive segmentation with initial prompts (could be either a positive point or a box), and continuing to (automatically) add point prompts (8 times) to: a) rectify where the model makes mistakes (using negative point prompt) and b) indicate an expected region missed out by the model (using a positive point prompt).