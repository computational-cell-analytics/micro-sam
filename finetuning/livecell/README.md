# Finetuning Segment Anything for LiveCELL

TODO: explain the set-up

These experiments are implemented for a slurm cluster with access to GPUs (and you ideally need access to A100s or H100s with 80GB of memory, if you only use ViT-b then using a GPU with 32 GB or 40 GB should suffice.)

## Training

TODO: add training code and explain how to run it

## Evaluation

To run the evaluation experiments for the Segment Anything Models on LiveCELL follow these steps:

- Preparation:
    - Adapt the path to the models, the data folder and the experiment folder in `evaluation/util.py`
    - Make sure you have the LiveCELL data downloaded in the data folder. If not you can run `python evaluation/util.py` and it will automatically downloaded.
    - Adapt the settings in `util/precompute_embeddings.sh` and `util/precompute_prompts.sh` to your slurm set-up.
- Precompute the embeddings by running `sbatch precompute_embeddings.sbatch <MODEL_TYPE>` for all the model types you want to test.
    - This will submit a slurm job that needs access to a GPU.
- Precompute the prompts by running `python precompute_prompts.py -f`.
    - This will submit a slurm array job that precomputes the prompts for all images. (In total 26 jobs will be started.) Note: these jobs do not need access to a GPU, the computation is purely CPU based.
- TODO: explain how to run the evaluation script.
