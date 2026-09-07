#!/bin/bash
# Round 2 of the AIS decoder campaign (boundary / boundary_fgcal), unattended:
#   1. screen the shared tuned configuration `dec-top1` (plain, ridge 1, ridge 2 + mask 0.3) on the 2d caches
#      of the two new decoders, chained `afterok` on their prediction jobs,
#   2. wait until every round-2 screen has finished (screen2d / screen3d from evaluate_ais_decoder.sh,
#      top_screen from step 1, contact_screen from launch_tuning_after_caches.sh),
#   3. write the conclusive overview of all six decoders and the field diagnostics of the two new ones.
#
#     bash finalize_round2_reports.sh [max_wait_seconds_for_the_caches]
#
# Submit it with --dependency afterany on the two ais_eval_<variant> jobs, and run a frozen copy: bash
# re-reads a running script by byte offset, so editing this file while a job sleeps in a wait loop breaks it.
#
# Outputs under <root>/ais/reports/: decoders_all_defaults_{dev,holdout}*.csv, decoders_all_tuned_{dev,holdout}*.csv,
# decoders_all_3d*.csv, decoders_boundary_contact_dev*.csv, decoder_fields_{boundary,boundary_fgcal}*.csv
set -o pipefail
NEW=${NEW_VARIANTS:-"boundary boundary_fgcal"}
ALL=${VARIANTS:-"baseline contact fgcal both boundary boundary_fgcal"}
MAX_WAIT=${1:-7200}
ROOT=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization
REPO=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/micro-sam
OPT=$REPO/finetuning/v2/evaluation/optimization
PY=/mnt/vast-nhr/home/pape41/u12086/Work/software/micromamba/envs/envs/new-stack/bin/python
V4=$ROOT/v4_geodesic_checkpoints/joint_sam2_hvit_t_multi_gpu/best.pt
EPOCH=856a433c4b33348e1d85c4c13278f057
export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=$ROOT/ais_decoder_training/staged
export MICRO_SAM2_JOINT_EXPORT_ROOT=$ROOT/model_exports
TOP_CONFIGS="$OPT/configs/ais_dec_top1.json $OPT/configs/ais_dec_top1_ridge1.json $OPT/configs/ais_dec_top1_ridge2_mask0.3.json"
REPORTS=$ROOT/ais/reports

job_dir() { ls -td "$ROOT"/jobs/*_"$1" 2>/dev/null | head -1; }
tasks_done() {  # all tasks of the newest job dir of this name carry a .done marker
    local dir; dir=$(job_dir "$1"); [ -n "$dir" ] || return 1
    [ "$(ls "$dir"/logs/*.done 2>/dev/null | wc -l)" -ge "$(wc -l < "$dir/tasks.txt")" ]
}
wait_for() {  # wait_for <max seconds> <names...>
    local limit=$1; shift; local waited=0
    while true; do
        local pending=""
        for name in "$@"; do tasks_done "$name" || pending="$pending $name"; done
        [ -z "$pending" ] && { echo "$(date +%H:%M) all done"; return 0; }
        [ "$waited" -ge "$limit" ] && { echo "$(date +%H:%M) timeout waiting for:$pending"; return 1; }
        echo "$(date +%H:%M) waiting for:$pending"; sleep 300; waited=$((waited + 300))
    done
}

cd "$OPT" || exit 1

# 1. The dec-top1 screens of the two new decoders, on their 2d prediction jobs.
for v in $NEW; do
    waited=0
    while [ -z "$(job_dir "dec_${v}_predict2d")" ]; do
        [ "$waited" -ge "$MAX_WAIT" ] && { echo "$(date +%H:%M) no dec_${v}_predict2d job dir, skipping its top screen"; break; }
        echo "$(date +%H:%M) waiting for the dec_${v}_predict2d job dir"; sleep 120; waited=$((waited + 120))
    done
    d=$(job_dir "dec_${v}_predict2d"); [ -n "$d" ] || continue
    if [ -n "$(job_dir "dec_${v}_top_screen")" ]; then echo "$(date +%H:%M) dec_${v}_top_screen exists already"; continue; fi
    j=$(cat "$d/job_id.txt")
    echo "$(date +%H:%M) submitting dec_${v}_top_screen (afterok:$j)"
    $PY ais_campaign_tasks.py screen --name "dec_${v}_top_screen" --preset cpu --kind v5 \
        --subsets primary training_extra holdout --no-defaults --configs $TOP_CONFIGS \
        --extra "--joint-checkpoint $v --ndim 2" --dependency "afterok:$j"
done

# 2. Every round-2 screen (the contact screens come from launch_tuning_after_caches.sh).
names=""
for v in $NEW; do
    names="$names dec_${v}_screen2d dec_${v}_screen3d dec_${v}_top_screen dec_${v}_contact_screen"
done
wait_for 28800 $names || true

# 3. The conclusive overview of all six decoders.
echo "$(date +%H:%M) writing the overview"
$PY report_ais_decoders.py --variants $ALL --production-checkpoint "$V4" --baseline-variant baseline \
    --configs current-defaults contact-ridge contact-mask --subsets primary training_extra --ndim 2 --epoch $EPOCH \
    --output "$REPORTS/decoders_all_defaults_dev" 2>&1 | grep -v "Warning\|warnings.warn"
$PY report_ais_decoders.py --variants $ALL --production-checkpoint "$V4" --baseline-variant baseline \
    --configs current-defaults contact-ridge contact-mask --subsets holdout --ndim 2 --epoch $EPOCH \
    --output "$REPORTS/decoders_all_defaults_holdout" 2>&1 | grep -v "Warning\|warnings.warn"
for pair in dev:"primary training_extra" holdout:holdout; do
    $PY report_ais_decoders.py --variants $ALL --production-checkpoint "$V4" --baseline-variant baseline \
        --baseline-config dec-top1 --configs current-defaults dec-top1 dec-fgcal-top1 dec-top1-ridge1 dec-top1-ridge2-mask0.3 \
        --subsets ${pair##*:} --ndim 2 --epoch $EPOCH \
        --output "$REPORTS/decoders_all_tuned_${pair%%:*}" 2>&1 | grep -v "Warning\|warnings.warn"
done
$PY report_ais_decoders.py --variants $ALL --production-checkpoint "$V4" --baseline-variant baseline \
    --configs current-defaults contact-ridge --kind apg3d --subsets primary holdout --ndim 3 --epoch $EPOCH \
    --output "$REPORTS/decoders_all_3d" 2>&1 | grep -v "Warning\|warnings.warn"
# The boundary channel's ridge and mask settings against the decoder's own defaults.
for v in $NEW; do
    $PY report_ais_decoders.py --variants "$v" --baseline-variant "$v" --configs current-defaults \
        contact-ridge-w0.5 contact-ridge contact-ridge-w2.0 contact-ridge-w4.0 contact-mask-t0.3 contact-mask \
        contact-mask-t0.7 contact-ridge1-mask0.5 --subsets primary training_extra --ndim 2 --epoch $EPOCH \
        --output "$REPORTS/decoders_${v}_contact_dev" 2>&1 | grep -v "Warning\|warnings.warn"
done
# Field diagnostics. --contact-mode must match the training target of the fifth channel, otherwise the head's
# precision is scored against a target that calls its correct pixels negative; `both` is rescored in the
# touching mode so that the round-1 reference carries the new recall_touching / recall_bg_boundary columns.
for pair in boundary:all boundary_fgcal:all both:touching; do
    v=${pair%%:*}; mode=${pair##*:}
    case " $NEW both " in *" $v "*) ;; *) continue ;; esac
    [ -f "$ROOT/ais_decoder_training/staged/joint_sam2_hvit_t_multi_gpu/$v.pt" ] || continue
    $PY diagnose_decoder_fields.py --joint-checkpoint "$v" --subset primary training_extra --ndim 2 \
        --contact-mode "$mode" --output "$REPORTS/decoder_fields_$v.csv" 2>&1 | grep -v "Warning\|warnings.warn" | tail -14
done
echo "$(date +%H:%M) round-2 finalisation done"
