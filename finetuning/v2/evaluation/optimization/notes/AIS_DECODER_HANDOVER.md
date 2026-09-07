# Hand-over: AIS decoder campaign, round 2 (full-boundary channel)

Written 2026-09-07 17:00 for the successor session. Everything below is committed on branch `ais-train-optim`.
Read first: `AIS_DECODER_TRAINING.md` (decision log, sections 4.0-4.4 hold the results and conclusions of round 1)
and the memory note `ais-decoder-campaign-state`. `<root>` = `/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization`,
`<camp>` = `<root>/ais_decoder_training`, `<opt>` = `finetuning/v2/evaluation/optimization`,
`<dec>` = `finetuning/v2/generalist/ais_decoder`, python = `micromamba activate new-stack`.

## 1. Task

Round 1 trained four decoders (baseline, contact, fgcal, both) and found that the contact-only fifth channel
(touching boundaries, <1 % of the pixels, ill-defined) is a dataset-dependent trade and under-confident. The user
wants round 2 with the **proper boundary loss**: the fifth channel holds the inner boundary of every object
(to neighbours and to background alike), which is the classical target and coincides with the zero level set of
the geodesic distance channels. Two new decoders, then the same evaluation and tuning as round 1, then one
conclusive overview of all six decoders.

| variant | fifth channel | foreground loss | status |
|---|---|---|---|
| `boundary` | inner boundary of every object, dilated by 1 (`contact_mode="all"`), Dice + BCE | Dice (unchanged) | to train |
| `boundary_fgcal` | same | Dice + boundary-weighted BCE (`boundary_weight=4`, radius 2) | to train |

The code is in place and tested (commit of 2026-09-07 17:00): `micro_sam/v2/transforms/labels.py::object_boundaries`
and the `contact_mode` argument of the label transforms; `<dec>/ais_decoder_lib.py::VARIANTS` has both entries;
`build_loaders` passes the mode. Everything downstream treats the channel as "contact" (loss `contact=True`,
sigmoid activation, `flow_instance_segmentation(contact=..., contact_weight=..., contact_mask_threshold=...)`,
configs `ais_contact_*.json`, `report_ais_decoders.py`), so no other change is needed. Unit tests:
`python -m pytest -o addopts="" test/test_v2_label_transforms.py` (8 pass).

## 2. Launch the two trainings (identical budget to round 1)

**Done at 16:59 on 2026-09-07 (seven 3g slices were free, 98 A100 jobs pending):** `boundary` = job 15776831,
`boundary_fgcal` = job 15776833 (grete:preemptible, 3g.40gb, 14 h; expected to finish ~06:00 on 2026-09-08 at
1.03 s/it), evaluation drivers 15776838 / 15776839 (`afterany`), round-2 tuning launcher 15776840
(WAIT_VARIANTS="boundary boundary_fgcal", VARIANTS = all six, polls up to ~12.8 h, then ranks every sweep). The
smoke test of `boundary_fgcal` passed on the session slice (5 target channels, loss 2.04 at batch 2, 3.7 GiB).
What remains for the successor: monitor (section 4), submit the `dec-top1` screens of the two new decoders once
their 2d caches exist (section 3, second block), then the overview (section 5). The commands below document what
was launched and serve as the fallback if a job has to be resubmitted.

Identical settings to round 1 so the six decoders are comparable: 48000 iterations, batch 8, `--epoch-scale 4`
(635 iterations per epoch), lr 5e-5, 12 loader workers, 16 CPUs, 64 G. Measured speeds: A100-40GB 0.467 s/it
(6.4 h), 3g.40gb slice 1.03 s/it (12.9 h; the 14 h limit leaves 1 h of margin).

Pick the GPU pool by the queue at launch time (both commands are ready; `--dry` prints the sbatch script):

```bash
# 3g slices free?  (each node has 4; the second number is the allocated count)
for n in ggpu158 ggpu192; do scontrol show node $n | grep -oE "gres/gpu:3g.40gb=[0-9]+" | tr '\n' ' '; echo "<- $n"; done
# A100 queue depth on grete:shared
squeue -p grete:shared -t PENDING -h -o "%b %r" | grep -c "A100:1"
```
At 16:52 seven 3g slices were free and 98 single-A100 jobs were pending (this morning the A100 waits were 4-23 min,
so re-check). Rule: free 3g slices -> use them (guaranteed start, done in ~13 h); otherwise A100
(`--partition grete:shared --gres A100:1 --time 12:00:00`), and if an A100 job has not started within an hour,
cancel it and fall back to a 3g slice. Do not run both pools for the same variant (same checkpoint directory).

```bash
cd /mnt/vast-nhr/home/pape41/u12086/Work/my_projects/micro-sam
PY=/mnt/vast-nhr/home/pape41/u12086/Work/software/micromamba/envs/envs/new-stack/bin/python
$PY finetuning/v2/generalist/ais_decoder/submit_ais_decoder_training.py --variants boundary boundary_fgcal \
    --iterations 48000 --batch-size 8 --epoch-scale 4 --partition grete:preemptible --gres 3g.40gb:1 --time 14:00:00
```
Then chain the evaluation drivers (`afterany`, so a time-out still evaluates the last `best.pt`), one per job id
printed above:
```bash
R=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/ais_decoder_training
DRV=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/micro-sam/finetuning/v2/generalist/ais_decoder/evaluate_ais_decoder.sh
for pair in boundary:<JOBID_B> boundary_fgcal:<JOBID_BF>; do v=${pair%%:*}; j=${pair##*:}
  sbatch --parsable --job-name=ais_eval_$v --dependency=afterany:$j -p grete:preemptible -G 1g.10gb:1 -c 2 --mem=16G \
    -t 01:00:00 --constraint=inet -A nim00007 -o $R/logs/slurm/ais_eval_${v}_%j.out -e $R/logs/slurm/ais_eval_${v}_%j.err \
    --wrap "set -eo pipefail; source ~/.bashrc; set -u; micromamba activate new-stack; export PYTHONUNBUFFERED=1; bash $DRV $v best"
done
```
The driver stages `<camp>/staged/joint_sam2_hvit_t_multi_gpu/<variant>.pt` (v4 SAM2 state + trained decoder),
caches the predictions of v5 primary / training_extra / holdout and apg3d primary / holdout, and runs the library
defaults plus `contact-ridge` (weight 1) and `contact-mask` (0.5) on the caches (`afterok` on the caches).

Sanity checks after the jobs start (the first iterations appear after ~2 min): `tr '\r' '\n' < <camp>/logs/slurm/ais_decoder_<v>_<job>.err | grep it/s | tail -1`;
`<camp>/checkpoints/ais_decoder_<v>/{best,latest}.pt` appear after the first epoch (~11 min on 3g).
A smoke test of the pipeline was run on the session slice before this hand-over (`--variant boundary_fgcal --smoke 5`).

## 3. Then the tuning of the two new decoders (once their 2d caches exist)

Same as round 1, unattended: the launcher waits for the caches, submits the grid sweeps (1728 combinations, CPU)
and the eight contact-ridge / mask screens, then ranks every sweep.
```bash
WAIT_VARIANTS="boundary boundary_fgcal" VARIANTS="baseline contact fgcal both boundary boundary_fgcal" \
  sbatch --parsable --job-name=ais_decoder_tuning2 -p grete:preemptible -G 1g.10gb:1 -c 2 --mem=16G -t 14:00:00 --constraint=inet \
  -A nim00007 -o $R/logs/slurm/ais_decoder_tuning2_%j.out -e $R/logs/slurm/ais_decoder_tuning2_%j.err \
  --wrap "set -o pipefail; source ~/.bashrc; micromamba activate new-stack; export PYTHONUNBUFFERED=1; export WAIT_VARIANTS VARIANTS; bash /mnt/vast-nhr/home/pape41/u12086/Work/my_projects/micro-sam/finetuning/v2/generalist/ais_decoder/launch_tuning_after_caches.sh 46000"
```
(`sbatch --export=ALL` is the default, so the two variables reach the script.) Also screen the shared tuned
configuration of round 1 for both new decoders, so all six can be read at one setting:
```bash
cd <opt>; export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<root>/ais_decoder_training/staged MICRO_SAM2_JOINT_EXPORT_ROOT=<root>/model_exports
J2D=$(cat $(ls -td <root>/jobs/*_dec_boundary_predict2d | head -1)/job_id.txt)   # same for boundary_fgcal
$PY ais_campaign_tasks.py screen --name dec_boundary_top_screen --preset cpu --kind v5 --subsets primary training_extra holdout \
    --no-defaults --configs configs/ais_dec_top1.json configs/ais_dec_top1_ridge1.json configs/ais_dec_top1_ridge2_mask0.3.json \
    --extra "--joint-checkpoint boundary --ndim 2" --dependency afterok:$J2D
```
Important: the cached sweep scorer (`parameter_search.score_image_sparse_cached`) ignores the contact keywords, so
ridge / mask settings are evaluated only through `screen` with config files, never through `sweep`.

## 4. Jobs of round 1 still running at hand-over time (monitor, do not resubmit)

| job | what | expected |
|---|---|---|
| 15772287 `ais_decoder_finalize` | waits for all round-1 screens, then writes `<root>/ais/reports/decoders_final_{dev,holdout,3d}*.csv` and `decoder_fields_<variant>.csv` for the four variants | ~18:00 (gives up ~18:10, wall limit 19:50) |
| 15772853 `ais_decoder_tuning` | waits for the baseline / contact sweeps, then ranks all four sweeps into `<root>/ais/reports/dec_<variant>_sweep_dev.csv` | ~18:30 |
| 15776127 / 15776128 (baseline), 15776228 / 15776229 (contact) | grid sweeps, 11 CPU tasks each | ~18:00 |
| 15776088 (baseline), 15776115 (contact) | 3d screens | ~17:30 |

`squeue -u $USER -h -o "%i %j %T %M %R" | sort -k2` shows them; task markers are `logs/<tag>.done` / `.failed` in the
newest `<root>/jobs/<timestamp>_<name>/`. If the finalize job gave up before the 3d screens finished, rerun
`bash <dec>/finalize_ais_decoder_reports.sh 0` (VARIANTS defaults to the four round-1 names) in a CPU job.

## 5. The conclusive overview (when everything is in)

Reference for every comparison is the fine-tuned `baseline`; the production decoder is included as the second
reference. Run from `<opt>` with `MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<root>/ais_decoder_training/staged`:
```bash
V4=<root>/v4_geodesic_checkpoints/joint_sam2_hvit_t_multi_gpu/best.pt; E=856a433c4b33348e1d85c4c13278f057
ALL="baseline contact fgcal both boundary boundary_fgcal"
# defaults, dev and holdout
$PY report_ais_decoders.py --variants $ALL --production-checkpoint $V4 --baseline-variant baseline --configs current-defaults contact-ridge \
    --subsets primary training_extra --ndim 2 --epoch $E --output <root>/ais/reports/decoders_all_defaults_dev
$PY report_ais_decoders.py --variants $ALL --production-checkpoint $V4 --baseline-variant baseline --configs current-defaults contact-ridge \
    --subsets holdout --ndim 2 --epoch $E --output <root>/ais/reports/decoders_all_defaults_holdout
# shared tuned configuration (reference baseline at dec-top1), dev and holdout
$PY report_ais_decoders.py --variants $ALL --production-checkpoint $V4 --baseline-variant baseline --baseline-config dec-top1 \
    --configs current-defaults dec-top1 dec-fgcal-top1 dec-top1-ridge1 dec-top1-ridge2-mask0.3 --subsets primary training_extra --ndim 2 --epoch $E \
    --output <root>/ais/reports/decoders_all_tuned_dev      # and --subsets holdout
# contact / boundary ridge and mask settings of a five-channel decoder against its own defaults
$PY report_ais_decoders.py --variants boundary --baseline-variant boundary --configs current-defaults contact-ridge-w0.5 contact-ridge \
    contact-ridge-w2.0 contact-ridge-w4.0 contact-mask-t0.3 contact-mask contact-mask-t0.7 contact-ridge1-mask0.5 --subsets primary training_extra --ndim 2 --epoch $E
# each decoder at its own sweep optimum (dev-tuned; read the model comparison on the holdout): <root>/ais/reports/dec_<variant>_sweep_dev.csv
# 3d crops (regression instrument only; a 2d-only fine-tune regresses volumes, and the round-1 five-channel decoder collapsed under the filter there)
$PY report_ais_decoders.py --variants $ALL --production-checkpoint $V4 --baseline-variant baseline --configs current-defaults contact-ridge \
    --kind apg3d --subsets primary holdout --ndim 3 --epoch $E --output <root>/ais/reports/decoders_all_3d
# field diagnostics (contact / boundary head Dice, precision, recall; flow cosines; fg area ratio)
$PY diagnose_decoder_fields.py --joint-checkpoint boundary --subset primary training_extra --ndim 2 --output <root>/ais/reports/decoder_fields_boundary.csv
```
Read-outs the user needs (see `AIS_DECODER_TRAINING.md` section 4.4 for the round-1 verdicts): balanced mSA and the
generalization gate (>= 9 / 11 up, worst > -2 %, balanced >= +2 %) against baseline on dev, confirmed on holdout;
the seeded-merge share and the unseeded / absorbed shares; `fg_area_ratio` and `matched_iou`; whether the boundary
head is confident (recall at 0.5) where the contact head was not; and whether the losses on deepbacs / dic_hepg2 /
covid_if / deepseas that the contact channel caused through the shared features disappear with the boundary target.
Write the tables into section 4.5 of `AIS_DECODER_TRAINING.md`, update section 4.4 if the verdict on point 1.1
changes, update the memory note, commit.

## 6. Pitfalls met in round 1 (all fixed in the code, listed so they are not re-debugged)

- Python 3.14 starts DataLoader workers through a fork server: 30-60 s per worker, every epoch for validation.
  `train_ais_decoder.py` forces `fork`. Do not remove.
- Files with fewer than three objects (yeaz frames) make torch_em's sampler raise after 500 attempts; the subset
  wrappers redraw. torch_em splits `n_samples` over files for zarr / h5 lists, hence `RandomSubsetDataset`.
- Trainer checkpoints pickle the datasets: import `ais_decoder_lib` before `torch.load` of a `best.pt`
  (`stage_ais_decoder_checkpoint.py` does); the staged files are lean and load in seconds.
- Always export `MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<root>/ais_decoder_training/staged` and
  `MICRO_SAM2_JOINT_EXPORT_ROOT=<root>/model_exports` before any benchmark command or submission (pinned into job.sh).
- The session cwd drifts after `cd`; use absolute paths. `.sh` files are git-ignored: `git add -f`.
- The CPU preset takes 16 cores per task; ~2-4 tasks run at once, so 40 queued tasks take ~1.5 h. `scontrol hold`
  the sweep arrays if screens must go first, `scontrol release` afterwards.
- `boundary_magnitude_max=0.4` removes every instance of a decoder whose magnitude does not dip at boundaries
  (the round-1 five-channel decoder in 3d); the fine-tuned decoders emit magnitude ~0 in the background, so the
  filter's premise is gone for them anyway.
- The session runs inside an interactive SLURM job on ggpu137 (1 CPU, 1g.20gb slice, 12 h); chain everything with
  dependencies so nothing depends on the session staying alive.
