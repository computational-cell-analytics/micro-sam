#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 96G
#SBATCH -t 2880
#SBATCH -p grete:shared
#SBATCH -G A100:1

source activate sam
python evaluate_training_evolution.py $@
