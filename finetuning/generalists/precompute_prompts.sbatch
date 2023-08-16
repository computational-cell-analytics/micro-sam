#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 2000
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007

source activate sam
python precompute_prompts.py $@
