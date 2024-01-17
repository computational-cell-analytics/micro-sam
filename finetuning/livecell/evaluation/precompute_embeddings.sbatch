#! /bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -t 120
#SBATCH -p grete:shared
#SBATCH -A nim00007
#SBATCH -G A100:1

source ~/.bashrc
micromamba activate main
python precompute_embeddings.py -c /scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/lm_generalist_sam/best.pt \
                                -m vit_b \
                                -e /scratch/projects/nim00007/sam/experiments/new_models/generalists/lm/livecell/vit_b/
