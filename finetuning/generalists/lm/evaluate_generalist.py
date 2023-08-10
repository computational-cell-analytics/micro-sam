from util import evaluate_checkpoint_for_datasets


# TODO extend this to run the full evaluation protocol for a generalist.

checkpoint = "/scratch-grete/projects/nim00007/sam/LM/generalist/vit_b/epoch-30.pt"
root = "/scratch-grete/projects/nim00007/sam/experiments/generalists/lm/test"
datasets = ["covid-if"]

evaluate_checkpoint_for_datasets(
    checkpoint=checkpoint,
    model_type="vit_b",
    experiment_root=root,
    datasets=datasets,
    run_default_evaluation=True,
    run_amg=True,
    max_num_val_images=10,
)
