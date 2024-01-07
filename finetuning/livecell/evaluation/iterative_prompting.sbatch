#! /bin/bash
#SBATCH -c 16
#SBATCH --mem 48G
#SBATCH -t 6:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --job-name=sam-iterative-prompting

source activate sam
python iterative_prompting.py $@