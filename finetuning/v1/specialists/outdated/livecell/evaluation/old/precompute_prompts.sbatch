#! /bin/bash
#SBATCH -p standard96
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -t 720

source activate sam
python precompute_prompts.py $@
