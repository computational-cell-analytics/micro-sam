#! /bin/bash
#SBATCH -c 16
#SBATCH --mem 128G
#SBATCH -t 2800
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --constraint=80gb

source activate sam
python train_histopathology_generalist.py $@
