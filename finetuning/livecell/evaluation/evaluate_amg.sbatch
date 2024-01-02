#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 2880
#SBATCH -p grete:shared
#SBATCH -G A100:1

source activate sam
python grid_search_and_inference.py $@
