#! /bin/bash
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 60
#SBATCH -p grete:shared
#SBATCH -G A100:1

source activate sam
python inference.py $@
