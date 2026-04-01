#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 96G
#SBATCH -t 240
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007

source activate sam
python evaluate_training_evolution.py $@
