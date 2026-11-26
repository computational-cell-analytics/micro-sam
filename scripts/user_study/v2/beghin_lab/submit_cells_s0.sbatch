#!/bin/bash
# The cells of the whole s0 level of the Beghin EM spheroid in one job: the cell predictions (predict_volume, only
# foreground and boundary) and the block-wise segmentation seeded with the nuclei (segment_cells_volume). Both steps
# continue where a previous run stopped (per z slab and per stage), so the job can be requeued.
# Measured: ~9.3 GB of GPU memory, ~50 GB of host memory for a slab of 64 (+ 8 context) slices of s0.
#SBATCH --job-name=beghin_cells_s0
#SBATCH --account=nim00007
#SBATCH --partition=grete:preemptible
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --constraint=inet
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions/cells_s0_%j.log

set -eo pipefail
source ~/.bashrc
micromamba activate new-stack
set -u

cd /mnt/vast-nhr/home/pape41/u12086/Work/my_projects/micro-sam/scripts/user_study/v2/beghin_lab
python -u explore_beghin_em_data.py predict_volume --level 0 --target cells --channels foreground boundary \
    --slab_size 64
python -u explore_beghin_em_data.py segment_cells_volume --level 0 --num_workers 8
