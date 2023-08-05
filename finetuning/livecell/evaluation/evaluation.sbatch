#! /bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 720
#SBATCH -p grete:shared
#SBATCH -G A100:1

source activate sam
python evaluation.py $@ 
