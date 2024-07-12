#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem 128G
#SBATCH --job-name=mito-net


source /home/nimlufre/.bashrc
conda activate sam

python /home/nimlufre/micro-sam/development/train_3d_model_with_lucchi_without_decoder.py
