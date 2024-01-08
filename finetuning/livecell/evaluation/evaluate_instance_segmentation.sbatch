#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 6:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007

source ~/.bashrc
micromamba activate main
python evaluate_instance_segmentation.py $@
