#!/bin/bash
# Evaluate one trained AIS decoder variant on the AIS benchmarks: stage the checkpoint, cache the predictions
# of the 2d manifests (primary, training_extra, holdout) and the 3d crop manifests (primary, holdout) on the
# cluster, then run the library defaults (and the two contact configurations for five-channel decoders) on the
# caches, one CPU task per (subset, configuration), chained with SLURM dependencies.
#
#     bash evaluate_ais_decoder.sh <variant> [best|latest]
#
# Afterwards: python optimization/report_ais_decoders.py --subsets primary training_extra --ndim 2 [--output ...]
set -eo pipefail
VARIANT=${1:?variant}
WHICH=${2:-best}
ROOT=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization
REPO=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/micro-sam
OPT=$REPO/finetuning/v2/evaluation/optimization
PY=/mnt/vast-nhr/home/pape41/u12086/Work/software/micromamba/envs/envs/new-stack/bin/python
export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=$ROOT/ais_decoder_training/staged
export MICRO_SAM2_JOINT_EXPORT_ROOT=$ROOT/model_exports

$PY $REPO/finetuning/v2/generalist/ais_decoder/stage_ais_decoder_checkpoint.py --variant "$VARIANT" --which "$WHICH"
CHANNELS=$($PY -c "import json; print(json.load(open('$ROOT/ais_decoder_training/staged/joint_sam2_hvit_t_multi_gpu/$VARIANT.json'))['output_channels'])")
echo "staged $VARIANT ($WHICH): $CHANNELS output channels"

CONFIGS=""
if [ "$CHANNELS" -gt 4 ]; then
    CONFIGS="--configs $OPT/configs/ais_contact_ridge.json $OPT/configs/ais_contact_mask.json"
fi

newest_job_id() { cat "$(ls -td $ROOT/jobs/*_"$1" | head -1)/job_id.txt"; }

cd $OPT
$PY ais_campaign_tasks.py predict --name "dec_${VARIANT}_predict2d" --preset 2d --kind v5 \
    --subsets primary training_extra holdout --extra "--joint-checkpoint $VARIANT"
J2D=$(newest_job_id "dec_${VARIANT}_predict2d")
$PY ais_campaign_tasks.py predict --name "dec_${VARIANT}_predict3d" --preset 3d --kind apg3d \
    --subsets primary holdout --extra "--joint-checkpoint $VARIANT"
J3D=$(newest_job_id "dec_${VARIANT}_predict3d")
echo "predict jobs: 2d $J2D, 3d $J3D"

$PY ais_campaign_tasks.py screen --name "dec_${VARIANT}_screen2d" --preset cpu --kind v5 \
    --subsets primary training_extra holdout --extra "--joint-checkpoint $VARIANT --ndim 2" $CONFIGS \
    --dependency "afterok:$J2D"
$PY ais_campaign_tasks.py screen --name "dec_${VARIANT}_screen3d" --preset cpu --kind apg3d \
    --subsets primary holdout --extra "--joint-checkpoint $VARIANT" $CONFIGS --dependency "afterok:$J3D"
echo "screens submitted (afterok the predictions); check with: squeue -u \$USER"
