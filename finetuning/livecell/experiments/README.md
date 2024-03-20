# Supplementary Experiments:

Additional experiments on LIVECell dataset
- Experiments with `n_objects_per_batch`: Range of the number of objects (per-image-per-batch)
for training the LIVECell specialist (using `vit_l`).
- Experiments with freezing different backbones: freezing different parts of the Segment Anything model.

## Description:

- `submit_experiment_finetuning.py`: Script to submit the finething for both the above mentioned experiments.
- `submit_experiment_evaluation.py`: Script to submit the inference and evaluation for both the above mentioned experiments.
- `run_experiment_evaluation.py`: Scripts to automatically submit all the jobs from `submit_experiment_evaluation.py` for all the combinations of the experiments mentioned above.
