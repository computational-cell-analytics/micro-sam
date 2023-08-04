#! /bin/bash
#SBATCH -N 1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 128G
#SBATCH -t 2880

source activate sam
python precompute_prompts.py
