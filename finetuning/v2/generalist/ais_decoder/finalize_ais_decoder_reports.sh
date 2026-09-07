#!/bin/bash
# Wait until the screens of every variant have finished (all tasks of the newest dec_<variant>_screen{2d,3d} job
# directories carry a .done marker), then write the final comparison tables and the field diagnostics.
#
#     bash finalize_ais_decoder_reports.sh [max_wait_seconds]
#
# Outputs: <root>/ais/reports/decoders_final_{dev,holdout}{,_datasets,_mechanisms}.csv,
#          <root>/ais/reports/decoders_final_3d*.csv, <root>/ais/reports/decoder_fields_<variant>*.csv
set -o pipefail
MAX_WAIT=${1:-32400}
ROOT=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization
REPO=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/micro-sam
OPT=$REPO/finetuning/v2/evaluation/optimization
PY=/mnt/vast-nhr/home/pape41/u12086/Work/software/micromamba/envs/envs/new-stack/bin/python
V4=$ROOT/v4_geodesic_checkpoints/joint_sam2_hvit_t_multi_gpu/best.pt
EPOCH=856a433c4b33348e1d85c4c13278f057
export MICRO_SAM2_JOINT_EXPORT_ROOT=$ROOT/model_exports

screens_done() {  # all tasks of the newest job dir of this name have a .done marker
    local dir
    dir=$(ls -td "$ROOT"/jobs/*_"$1" 2>/dev/null | head -1)
    [ -n "$dir" ] || return 1
    local n_tasks n_done
    n_tasks=$(wc -l < "$dir/tasks.txt")
    n_done=$(ls "$dir"/logs/*.done 2>/dev/null | wc -l)
    [ "$n_done" -ge "$n_tasks" ]
}

waited=0
while true; do
    pending=""
    for v in baseline contact fgcal both; do
        for kind in screen2d screen3d; do
            screens_done "dec_${v}_${kind}" || pending="$pending dec_${v}_${kind}"
        done
    done
    if [ -z "$pending" ]; then echo "$(date +%H:%M) all screens done"; break; fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then echo "$(date +%H:%M) giving up waiting for:$pending"; break; fi
    echo "$(date +%H:%M) waiting for:$pending"
    sleep 300; waited=$((waited + 300))
done

cd "$OPT"
export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=$ROOT/ais_decoder_training/staged
$PY report_ais_decoders.py --variants baseline contact fgcal both --production-checkpoint "$V4" \
    --configs current-defaults contact-ridge contact-mask --subsets primary training_extra --ndim 2 --epoch $EPOCH \
    --output "$ROOT/ais/reports/decoders_final_dev" 2>&1 | grep -v "Warning\|warnings.warn"
$PY report_ais_decoders.py --variants baseline contact fgcal both --production-checkpoint "$V4" \
    --configs current-defaults contact-ridge contact-mask --subsets holdout --ndim 2 --epoch $EPOCH \
    --output "$ROOT/ais/reports/decoders_final_holdout" 2>&1 | grep -v "Warning\|warnings.warn"
$PY report_ais_decoders.py --variants baseline contact fgcal both --production-checkpoint "$V4" \
    --configs current-defaults contact-ridge --kind apg3d --subsets primary holdout --ndim 3 --epoch $EPOCH \
    --output "$ROOT/ais/reports/decoders_final_3d" 2>&1 | grep -v "Warning\|warnings.warn"
for v in baseline contact fgcal both; do
    [ -f "$ROOT/ais_decoder_training/staged/joint_sam2_hvit_t_multi_gpu/$v.pt" ] || continue
    $PY diagnose_decoder_fields.py --joint-checkpoint "$v" --subset primary training_extra --ndim 2 \
        --output "$ROOT/ais/reports/decoder_fields_$v.csv" 2>&1 | grep -v "Warning\|warnings.warn" | tail -14
done
echo "$(date +%H:%M) finalisation done"
