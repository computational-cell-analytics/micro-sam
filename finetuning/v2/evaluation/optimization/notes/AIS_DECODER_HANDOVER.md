# Hand-over: AIS decoder campaign, round 2 (full-boundary channel) - reading the results

Written 2026-09-07 18:00 for the successor session, replacing the 17:00 version. Everything is committed on
branch `ais-train-optim`. Read first: `AIS_DECODER_TRAINING.md` - sections 4.0-4.4 hold round 1, **4.5** the
round-1 completions, **5.1-5.3** the round-2 launch and the chain. Memory note `ais-decoder-campaign-state`.
`<root>` = `/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization`,
`<camp>` = `<root>/ais_decoder_training`, `<opt>` = `finetuning/v2/evaluation/optimization`,
`<dec>` = `finetuning/v2/generalist/ais_decoder`, `<rep>` = `<root>/ais/reports`,
python = `micromamba activate new-stack`.

## 1. What is left to do

**Nothing has to be submitted.** Round 2 is chained end to end (section 5.3 of the notes); the successor reads
the tables the chain writes and finishes the write-up:

1. Read the round-2 tables (section 3 below) and write them into **section 5.5** of `AIS_DECODER_TRAINING.md` (5.4 already records what the ridge and
   mask modes mean once the channel is a full boundary, and the signature to look for).
2. Decide point 1.1 (the fifth channel) with the boundary target on the evidence, and update section 4.4 point 3
   if the verdict changes. The user's rule: only cross-dataset wins count - balanced mSA plus the gate
   (>= 9 / 11 up, worst > -2 %, balanced >= +2 %) against the fine-tuned `baseline` on dev, confirmed on the
   holdout. No per-dataset fits.
3. Write the conclusive overview of all six decoders (**section 6**), update the memory note, commit.

## 2. State at hand-over

| job | what | expected |
|---|---|---|
| 15776831 `boundary`, 15776833 `boundary_fgcal` | the two trainings, 48000 iterations at 1.08 it/s on 3g.40gb slices (ggpu158 / ggpu192), started 17:16 | done 05:30-06:00, wall limit 07:16 |
| 15776838 / 15776839 `ais_eval_<variant>` | `afterany` the training: stage, cache v5 primary / training_extra / holdout and apg3d primary / holdout, then the `current-defaults`, `contact-ridge` and `contact-mask` screens | ~06:00, screens ~07:00 |
| 15777359 `ais_decoder_tuning2` | `afterany` both evaluations: waits for the 2d caches, submits the two grid sweeps (1728 combinations) and the eight-configuration contact screen per new variant, then ranks all six sweeps into `<rep>/dec_<variant>_sweep_dev.csv` | ~06:05, rankings ~11:00 |
| 15777505 `ais_decoder_finalize_r2` | `afterany` both evaluations: submits the `dec-top1` screens of the two new decoders `afterok` their prediction jobs, waits for every round-2 screen (up to 8 h), then writes the overview tables and the field diagnostics | ~06:05, tables ~11:00-13:00 |
| 15776127/28, 15776228/29 | the round-1 `baseline` / `contact` grid sweeps, 18 of 22 tasks left at 17:30, roughly serial at ~6 min | ~19:30 |
| 15772853 `ais_decoder_tuning` | the round-1 launcher; ranks the `baseline` / `contact` sweeps if they finish before it gives up at 20:03 | 20:03 |

Round 1 is otherwise complete: `<rep>/decoders_{defaults,tuned,final}_*`, `decoders_final_3d*`,
`decoder_fields_{production,baseline,contact,fgcal,both}*`. If `dec_baseline_sweep_dev.csv` /
`dec_contact_sweep_dev.csv` are missing, `tuning2` writes them (it ranks all six variants); to do it by hand:

```bash
cd <opt>; export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<root>/ais_decoder_training/staged
for v in baseline contact; do $PY report_ais_sweep.py --grid configs/ais_grid_lm_v4.json \
    --subset primary training_extra --joint-checkpoint $v --top 25 --output <rep>/dec_${v}_sweep_dev.csv; done
```

Check the chain with `squeue -u $USER -h -o "%i %j %T %M %R" | sort -k2`; task markers are `logs/<tag>.done` /
`.failed` in the newest `<root>/jobs/<timestamp>_<name>/`; drivers log to `<camp>/logs/slurm/`.

## 3. The tables to read, and the read-outs the user needs

Reference for every comparison is the fine-tuned `baseline`; the production decoder is the second reference.

| file under `<rep>` | what |
|---|---|
| `decoders_all_defaults_{dev,holdout}{,_datasets,_mechanisms}.csv` | all six plus production under the library defaults, `contact-ridge` and `contact-mask` |
| `decoders_all_tuned_{dev,holdout}*.csv` | all six at the shared tuned `dec-top1` (reference: baseline at `dec-top1`), with the ridge / mask variants |
| `decoders_all_3d*.csv` | apg3d primary + holdout; regression instrument only (see 4.5: even the plain fine-tune loses 25-60 % per LM family, and the round-1 five-channel decoders are 0 because `boundary_magnitude_max` removes every instance) |
| `decoders_{boundary,boundary_fgcal}_contact_dev*.csv` | the eight ridge / mask settings against the decoder's own defaults |
| `dec_{boundary,boundary_fgcal}_sweep_dev.csv` | each new decoder at its own sweep optimum (dev-tuned; read the model comparison on the holdout) |
| `decoder_fields_{boundary,boundary_fgcal}{,_summary}.csv` | field diagnostics, scored with `--contact-mode all` |

Read-outs:

1. **The gate.** Balanced mSA and the gate against `baseline` on dev, confirmed on the holdout, at the defaults
   *and* at `dec-top1`. Round-1 numbers to beat: `contact` -4.7 % / -5.2 % (defaults), -4.2 % / -3.8 %
   (`dec-top1`); `both` -1.3 % / -1.9 % and +1.3 % / +1.8 %; `fgcal` +0.6 % / +1.1 % and +2.4 % / +1.7 %.
2. **Is the head confident now?** The round-1 contact head was precise but under-confident exactly on the
   datasets whose merges motivated it. `recall_touching` of the round-1 `contact` decoder (per-dataset medians):
   dynamicnuclearnet 0.72, yeaz 0.76, livecell 0.63, covid_if 0.59, tissuenet **0.19**, neurips **0.13**, puma
   0.12, tnbc 0.03, deepbacs 0.015, dic_hepg2 **0.001**. The boundary head has to lift tissuenet, neurips,
   deepbacs and dic_hepg2; `recall_bg_boundary` (0.00-0.15 for `contact`) shows whether it also learned the
   background-facing rim, i.e. whether it learned the target at all.
3. **Do the shared-feature losses disappear?** Round 1 lost -12 % deepbacs, -21 % covid_if, -26 % deepseas,
   -38 % dic_hepg2 through the shared features, not through the ridge (the head never fired there). Read the
   per-dataset columns and the mechanism shares: dic_hepg2 lost seeds (unseeded 43.9 -> 57.1 %), deepbacs split
   its rods (1.7 -> 4.1 %). If the boundary target removes these, point 1.1 becomes a candidate again.
4. **The merges it was for.** Seeded-merge share at the defaults and at `dec-top1`, with and without the ridge.
   Round 1: baseline 8.1 % / 13.4 %, `contact` + ridge 4.5 % / 4.1 %, `both` + ridge 4.2 % / 4.0 %.
5. **Extent.** `fg_area_ratio` and `matched_iou` per dataset - never the summary CSV's mean (deepseas 12-91 and
   neurips 2.3-12 dominate it; see 4.5).
6. **The ridge / mask setting** of a denser head: with a few percent of the pixels positive the mask mode at 0.5
   may finally do something (it was inert in round 1 because the head rarely exceeded 0.5).

## 4. If something went wrong

- **A training timed out** (wall 07:16): `afterany` still fires, and the driver stages `best.pt` of the last
  finished epoch, so the chain completes on a slightly shorter run. Note the epoch in the write-up.
- **A job was preempted** (everything runs on `grete:preemptible`): resubmit the driver by hand, e.g.
  `bash <dec>/evaluate_ais_decoder.sh boundary best`, or the frozen copies under `<camp>/jobs/frozen/`
  (`finalize_round2_<ts>.sh`, `launch_tuning_<ts>.sh`, `finalize_<ts>.sh`).
- **Reports come out empty**: check `--epoch 856a433c4b33348e1d85c4c13278f057` still matches
  `implementation_checksum()`. It hashes `benchmark_ais_optimization.py`, `common.py`, `parameter_search.py`,
  `micro_sam/v2/instance_segmentation.py` and `micro_sam/v2/postprocessing.py` - **do not touch those five while
  the chain is in flight**, or the new runs get a different epoch and every filtered report goes blank.
  `micro_sam/v2/transforms/labels.py` and the readers are not hashed, so the round-2 code changes did not move it.
- **A screen or sweep task failed**: `<root>/jobs/<ts>_<name>/logs/<tag>.failed` holds the reason; rerun the one
  command from `tasks.txt`.

## 5. Pitfalls (met, fixed, listed so they are not re-debugged)

- **Never submit a repo path for a long-running driver.** Bash re-reads a running script by byte offset, so
  editing the file while a job sleeps in a wait loop breaks the parse - that is how the round-1 finalisation died
  after waiting 7.4 h (notes 5.2). Copy it to `<camp>/jobs/frozen/<name>_<timestamp>.sh` and submit the copy.
- **Our own array can block our own jobs.** An unschedulable job at the head of a partition blocks every
  lower-priority job of the same user; the round-2 trainings pended 17 minutes behind our own sweep array with
  seven 3g slices free (notes 5.1). Diagnosis: `squeue -u $USER -O "jobid,name,state,reason,priority"` and look
  for `TopOfQueue`. Fix: `scontrol update jobid=<array> nice=100`, or `scontrol hold` / `release`.
- **`sbatch --test-only` is worthless on `grete:preemptible`** - it ignores preemption and returned the same
  10-hour-away estimate for every pool while jobs started immediately.
- `diagnose_decoder_fields.py --contact-mode` must match the training target of the fifth channel (`touching`
  for `contact` / `both`, `all` for `boundary` / `boundary_fgcal`), otherwise the head's precision is scored
  against a target that calls its correct pixels negative.
- The cached sweep scorer ignores the contact keywords, so ridge / mask settings are only ever evaluated through
  `screen` with config files, never through `sweep`.
- Python 3.14 starts DataLoader workers through a fork server (30-60 s each, every epoch);
  `train_ais_decoder.py` forces `fork`. Do not remove.
- Files with fewer than three objects (yeaz frames) make torch_em's sampler raise after 500 attempts; the subset
  wrappers redraw. `RandomSubsetDataset` exists because torch_em splits `n_samples` over files.
- Trainer checkpoints pickle the datasets: import `ais_decoder_lib` before `torch.load` of a `best.pt`.
- Always export `MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<camp>/staged` and
  `MICRO_SAM2_JOINT_EXPORT_ROOT=<root>/model_exports` before any benchmark command.
- The session cwd drifts after `cd`; use absolute paths. `.sh` files are git-ignored: `git add -f`.
- The CPU preset takes 16 cores per task and roughly one task runs at a time, so a 22-task sweep needs ~2 h.
