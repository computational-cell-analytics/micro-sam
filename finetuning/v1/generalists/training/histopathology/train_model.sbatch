#! /bin/bash
#SBATCH -c 16
#SBATCH --mem 128G
#SBATCH -t 7-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --constraint=80gb
#SBATCH --qos=7d
#SBATCH --job-name=sam_histopathology

source ~/.bashrc
mamba activate sam
python train_histopathology_generalist.py $@
