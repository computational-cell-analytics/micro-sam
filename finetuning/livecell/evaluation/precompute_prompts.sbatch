#! /bin/bash
#SBATCH -p standard96
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -t 1000

source activate sam
python precompute_prompts.py $@
