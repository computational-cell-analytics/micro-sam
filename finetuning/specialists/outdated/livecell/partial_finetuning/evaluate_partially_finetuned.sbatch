#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 96G
#SBATCH -t 2880
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007

source activate sam
python evaluate_partially_finetuned.py $@
