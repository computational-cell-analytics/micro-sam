#! /bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -t 120
#SBATCH -p grete:shared
#SBATCH -G A100:1

source activate sam
python precompute_embeddings.py -n $1 
