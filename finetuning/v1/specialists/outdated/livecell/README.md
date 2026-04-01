# Finetuning Segment Anything for LIVECell

These experiments are implemented for a slurm cluster with access to GPUs (and you ideally need access to A100s or H100s with 80GB of memory, if you only use ViT Base then using a GPU with 32 GB or 40 GB should suffice.)

## Training

The relevant scripts are located in the top-level `finetuning` directory at: `finetuning/livecell_finetuning.py`.

## Evaluation

To run the evaluation experiments for the Segment Anything Models on LIVECell follow these steps:

- Preparation:
    - Go to the `evaluation` directory.
    - Adapt the path to the models, the data folder and the experiment folder in `util.py`
    - Make sure you have the LiveCELL data downloaded in the data folder. If not you can run `python util.py` and it will automatically downloaded.
    - Adapt the settings in `util/precompute_embeddings.sh` and `util/precompute_prompts.sh` to your slurm set-up.
- Precompute the embeddings by running `sbatch precompute_embeddings.sbatch <MODEL_NAME>` for all the models you want to evaluate.
    - This will submit a slurm job that needs access to a GPU.
- Precompute the prompts by running `python precompute_prompts.py -f`.
    - This will submit a slurm array job that precomputes the prompts for all images. In total 31 jobs will be started. Note: these jobs do not need access to a GPU, the computation is purely CPU based.
- Run inference via `python inference.py -n <MODEL_NAME>`.
    - This will submit a slurm array job that runs the prediction for all prompt settings for this model. In total 61 jobs will be started, but each should only take 10-20 minutes (on a A100 and depending on the model type).
- Run the evaluation of inference results via `sbatch evaluation.sbatch -n <MODEL_NAME>`.
    - This will submit a single slurm job that does not need a GPU.
