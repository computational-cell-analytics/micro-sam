#! /bin/bash
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 6:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007

source activate sam
python iterative_prompting.py -c /scratch/usr/nimanwai/micro-sam/checkpoints/vit_h/livecell_sam/best.pt \
                              -m vit_h \
                              -e /scratch/projects/nim00007/sam/experiments/new_models/specialists/lm/livecell/vit_h/
