# TODO refactor so that this starts the relevant slurm jobs.
# For now we just use this for debugging.

from micro_sam.evaluation.livecell import run_livecell_amg
from util import DATA_ROOT, get_checkpoint, get_experiment_folder

model_name = "vit_b"
checkpoint, model_type = get_checkpoint(model_name)
experiment_folder = get_experiment_folder(model_name)
input_folder = DATA_ROOT

run_livecell_amg(
    checkpoint, model_type, input_folder, experiment_folder,
    iou_thresh_values=[0.7, 0.75, 0.8, 0.85, 0.9],
    stability_score_values=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    verbose_gs=True,
    n_val_per_cell_type=10,
)
