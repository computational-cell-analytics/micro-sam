#!/bin/bash
# Once the 2d prediction caches of baseline and contact exist, submit their grid sweeps (and the contact
# configuration screens for the five-channel 'contact' decoder); then, when every variant's sweeps are done,
# rank each sweep (report_ais_sweep.py) into <root>/ais/reports/dec_<variant>_sweep_dev.csv.
#
#     bash launch_tuning_after_caches.sh [max_wait_seconds]
set -o pipefail
MAX_WAIT=${1:-32400}
ROOT=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization
OPT=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/micro-sam/finetuning/v2/evaluation/optimization
PY=/mnt/vast-nhr/home/pape41/u12086/Work/software/micromamba/envs/envs/new-stack/bin/python
export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=$ROOT/ais_decoder_training/staged
export MICRO_SAM2_JOINT_EXPORT_ROOT=$ROOT/model_exports
PRIMARY="livecell tissuenet dynamicnuclearnet deepbacs dic_hepg2"
EXTRA="yeaz neurips_cellseg deepseas puma tnbc covid_if"
CONTACT_CONFIGS="$OPT/configs/ais_contact_ridge_w0.5.json $OPT/configs/ais_contact_ridge_w2.0.json $OPT/configs/ais_contact_ridge_w4.0.json $OPT/configs/ais_contact_mask_t0.3.json $OPT/configs/ais_contact_mask_t0.7.json $OPT/configs/ais_contact_ridge1_mask0.5.json $OPT/configs/ais_contact_ridge.json $OPT/configs/ais_contact_mask.json"

tasks_done() {  # all tasks of the newest job dir of this name carry a .done marker
    local dir; dir=$(ls -td "$ROOT"/jobs/*_"$1" 2>/dev/null | head -1); [ -n "$dir" ] || return 1
    [ "$(ls "$dir"/logs/*.done 2>/dev/null | wc -l)" -ge "$(wc -l < "$dir/tasks.txt")" ]
}
wait_for() {  # wait_for <max seconds> <names...>
    local limit=$1; shift; local waited=0
    while true; do
        local pending=""
        for name in "$@"; do tasks_done "$name" || pending="$pending $name"; done
        [ -z "$pending" ] && return 0
        [ "$waited" -ge "$limit" ] && { echo "$(date +%H:%M) timeout waiting for:$pending"; return 1; }
        echo "$(date +%H:%M) waiting for:$pending"; sleep 300; waited=$((waited + 300))
    done
}

cd "$OPT"
declare -A launched
while true; do
    for v in baseline contact; do
        [ -n "${launched[$v]}" ] && continue
        if tasks_done "dec_${v}_predict2d"; then
            echo "$(date +%H:%M) caches of $v ready, submitting sweeps"
            $PY ais_campaign_tasks.py sweep --name "dec_${v}_sweep_primary" --preset cpu --kind v5 --subsets primary \
                --grid configs/ais_grid_lm_v4.json --datasets $PRIMARY --num-shards 1 --extra "--joint-checkpoint $v"
            $PY ais_campaign_tasks.py sweep --name "dec_${v}_sweep_extra" --preset cpu --kind v5 --subsets training_extra \
                --grid configs/ais_grid_lm_v4.json --datasets $EXTRA --num-shards 1 --extra "--joint-checkpoint $v"
            if [ "$v" = "contact" ]; then
                $PY ais_campaign_tasks.py screen --name "dec_${v}_contact_screen" --preset cpu --kind v5 \
                    --subsets primary training_extra holdout --no-defaults --configs $CONTACT_CONFIGS \
                    --extra "--joint-checkpoint $v --ndim 2"
            fi
            launched[$v]=1
        fi
    done
    [ -n "${launched[baseline]}" ] && [ -n "${launched[contact]}" ] && break
    [ "$MAX_WAIT" -le 0 ] && { echo "$(date +%H:%M) gave up waiting for the caches"; break; }
    sleep 300; MAX_WAIT=$((MAX_WAIT - 300))
done

names=""
for v in baseline contact fgcal both; do names="$names dec_${v}_sweep_primary dec_${v}_sweep_extra"; done
wait_for 14400 $names || true
for v in baseline contact fgcal both; do
    tasks_done "dec_${v}_sweep_primary" && tasks_done "dec_${v}_sweep_extra" || { echo "sweeps of $v incomplete, skipping the ranking"; continue; }
    $PY report_ais_sweep.py --grid configs/ais_grid_lm_v4.json --subset primary training_extra --joint-checkpoint "$v" \
        --top 25 --output "$ROOT/ais/reports/dec_${v}_sweep_dev.csv" 2>&1 | grep -v "Warning\|warnings.warn" | tail -40
done
echo "$(date +%H:%M) tuning launcher done"
