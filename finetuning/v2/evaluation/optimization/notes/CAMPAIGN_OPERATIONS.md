# Campaign operations

How the APG optimization jobs are run on grete, and the rules that keep their numbers comparable.
Written for the campaigns started on 2026-09-02; the facts about the cluster were verified then.

## Cluster and environment

- Environment: `micromamba activate new-stack`. The `super` environment that
  `submit_all_evaluations.py` and `parameter_search.py` default to does not exist on this host.
- Partition `grete:preemptible` (2-day limit): GRES `1g.10gb:1` (plentiful), `1g.20gb:1` (8 slices),
  `2g.20gb:1` (16 slices), `3g.40gb:1` (8). `grete:interactive` allows two jobs per user for 12 h.
  Every job needs `--constraint=inet`. Account `nim00007`; QOS `2h` and `normal` only.
- Script body order matters: `set -eo pipefail`, then `source ~/.bashrc`, then `set -u`. An earlier
  `set -euo pipefail` before the bashrc failed on `/etc/bashrc`'s unset `BASHRCSOURCED`.

## Submitting

`submit_optimization_jobs.py` writes `<output-root>/jobs/<timestamp>_<name>/` with `tasks.txt`
(`tag<TAB>command`), `job.sh`, `logs/` and `submit.json` (argv, resources, git revision, dirty flag),
then submits `job.sh` as one array (`--array=0-N%throttle`, `--requeue`, `--open-mode=append`).
`apg_campaign_tasks.py` builds the task lists for the benchmark, the screens, the trainers and
per-sample scripts. Always run `--dry-run` first and read `job.sh`; `sbatch --test-only job.sh`
checks the header against the cluster. `--local` runs the same tasks sequentially on the session GPU.

Every task leaves `logs/<tag>.done` or `logs/<tag>.failed`. A dependent stage waits on `.done`
markers or on `--dependency afterok:<job id>` (the id is in `job_id.txt`), never on the mere existence
of an output file: the trainers write once at the end and a preempted write leaves a truncated file.

## Preemption and retries

- The benchmark and every screen resume per sample from `samples.csv` (atomic writes), so a requeued
  task continues where it stopped. The trainers do not resume; submit them with `--attempts 1` and let
  a requeue restart them from scratch.
- `status <job dir>` shows Slurm state, exit code, restarts and the marker per task. A task with
  `Restarts > 0` or a second `[date] task ...` banner in its `.out` was preempted.
- Quality results are unaffected by a restart. A *timing* trial that restarted mid-way is discarded
  and re-run under a fresh trial id.
- `DependencyNeverSatisfied`: `scancel` the dependent job and resubmit with `--resume-from <job dir>`.

## Timing trials

The 3D notes measured 3-7% node-level drift in generation time with byte-identical code. Rules:

- Canonical timing runs are serialized on one GRES type (`--throttle 1`), bracketed by a defaults run
  before and after (`apg_campaign_tasks.py benchmark --serialize --bracket`). If the two brackets differ
  by more than about 2% (2D) or 3% (3D), the cost column of that trial is discarded and it is repeated.
- Three trials per configuration; the comparator takes per-dataset medians.
- The comparator pairs runs only within one implementation checksum and one hardware identity, so a
  `1g.10gb` screen and a `1g.20gb` canonical run never collide, and never compare.

## Implementation checksum epochs

`benchmark_apg_optimization._implementation_checksum()` hashes eight files: the benchmark itself,
`common.py`, `parameter_search.py`, and `micro_sam/v2/{automatic_prompt_generation, multimask_selection,
instance_segmentation, postprocessing, prompt_based_segmentation}.py`. Any edit re-keys every new run.

- Batch edits to those files into one commit at an epoch boundary, then re-run the three controls.
- Never edit them while an array is queued: a task reads the working tree when it starts.
- Record the old and new checksum in the notes, as `APG_3D_OPTIMIZATION.md` does for its six epochs.
- Value at the start of the 2026-09 campaign: `aeb1aca09a5fff43d2b8bb8bacff2b06`.

## Decision log

One dated subsection per round in `APG_2D_OPTIMIZATION.md` or `APG_3D_OPTIMIZATION.md`: the job
directories, trial ids, config checksums, the comparator output path (`campaign_*.json` beside the
output root) and the accept/reject line with its gate. `submit.json` in every job directory keeps the
argv and git revision that produced it.

## Configuration files

`optimization/configs/` holds the benchmark and screen inputs. Dict-shaped files
(`{"name", "params_2d", "params_3d"}`) go to `benchmark_apg_optimization.py --config`; list-shaped files
(`[{"name", "params_2d"}, ...]`) go to `screen_apg_refinement.py --configs`. Every campaign config pins
`candidate_threshold`, `dt` and `max_overlap` explicitly, because the accepted 2D runs used
`1.5 / 0.25 / 0.15` while the per-model hvit_t defaults resolve to `3.0 / 0.5 / 0.3`.

## Pinned proposal settings for every 2d screen

The learned 2d artifacts were extracted with `candidate_threshold=1.5, dt=0.25, sigma=0.5,
min_candidate_size=4, foreground_threshold=0.7`. A screen that calls `propose()` with the library
defaults regenerates a different prompt set (the per-model hvit_t defaults resolve to 3.0 / 0.5) and
the OOF identity check in `screen_apg_multimask._oof_predictions_for_sample` refuses it. Every script
that replays OOF predictions therefore proposes with `PINNED_PROPOSAL_2D` from `screen_apg_multimask.py`
(`train_apg_refinement_gate.py`, `screen_apg_compact_selector.py`, `screen_apg_multimask.py`); the
refinement screen takes the values from its configuration list, which must pin them too.

## Never edit a checksum file while arrays run

Epoch 3 happened by accident: a harness-only edit to `benchmark_apg_optimization.py` re-keyed every 3d
run directory in the middle of the C1 arrays. The aggregates recover from it (`sibling_run_dirs`), but
the rule stands: batch edits to the eight checksum files, and never during an array.

- Never index a compressed `NpzFile` inside a loop: every access decompresses the whole array; read each
  array once into a local variable (the C3 aggregate OOM at 128 GB, 2026-09-03).

- `multimask_selection/token_lowres_v1/training_extra_features.npz` holds library-default proposals (trainer
  extraction without `proposal_settings`); replay it with `screen_apg_compact_selector.py --proposal-settings
  library`. Primary feature datasets under `candidate_supply/` are pinned.
- `screen_apg_candidate_supply.feature_path()` always names its output `primary_features_<setting>.npz`; extracting
  another subset through `stage_extract` into the canonical directory silently skips (file exists) or, if renamed
  afterwards, destroys the primary file. Extract other subsets into a scratch directory and move the result.

## Continuation checklist for the 2026-09-03 campaign

Everything below runs from `finetuning/v2/evaluation`; `<root>` is the APG output root.

1. **3d arrays** (2g.20gb, `<root>/jobs/2026-09-03_*`): `c3_extract_primary` (track cache),
   `c2_union_point_recall` (hybrid candidates through propagation), `c2_hybrid_em` (slice-wise EM). When
   an array is complete:
   - `python optimization/screen_apg_3d_hybrid.py aggregate --subset primary --variant union-point
     --encoding standalone --scoring plain --linker multicut --beta 0.5 --selector-artifact <selector.pt>`
   - `python optimization/screen_apg_3d_hybrid.py aggregate --subset primary --variant hybrid-2d
     --encoding standalone --scoring plain --linker multicut --beta <beta> --selector-artifact <selector.pt>`
     for each beta replayed from the slice cache (`run ... --beta 0.3` is CPU-only once the cache exists).
   - `c3_train_replay` (job 15716576) waits on the extraction array and writes `<root>/3d_v2/c3/`
     (`training/candidates.npz`, `models/`, `screen_primary/summary.csv`). Read the `__family_macro__`
     rows and the paired bootstrap columns `macro_delta_*`.
2. **2d**: `e2_train_screen_v2` writes `<root>/candidate_supply_screening/hvit_t/<identity>/summary.csv`.
   The holdout confirmation of any E2 winner is a benchmark run with the winner's settings pinned in a
   config and the pooled artifact passed as `--multimask-scorer-artifact`.
3. **Holdout / test for 3d winners**: `benchmark_apg_3d.py run --subset holdout|test ...` arrays via
   `apg_campaign_tasks.py per-sample`; the `test` manifest has to be built first
   (`apg3d_manifest.py build --subset test`, CPU, reads labels of ~50 volumes).
4. **v4 confirmation** (geodesic): stage a checkpoint root with
   `<stage>/joint_sam2_hvit_t_multi_gpu/best.pt -> .../joint/v4/checkpoints/joint_sam2_hvit_t_geodesic_multi_gpu/best.pt`,
   export `MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<stage>` (the submitter pins it into job scripts), use a
   separate `--output-root`, re-extract and re-fit the selector and gate (`train_apg_multimask_selector.py`,
   `train_apg_refinement_gate.py` with `PINNED_PROPOSAL_2D`), then the holdout benchmark trials.
5. **Decision log**: one dated subsection per round in `APG_2D_OPTIMIZATION.md` /
   `APG_3D_OPTIMIZATION.md`; comparator outputs beside the output root (`campaign2_*.json`).

### State at the close of session 2 (2026-09-03, ~08:20)

- 2D: E2 winner confirmed on the holdout (`apg_e2_winner.json` + pooled selector `…-pooled6.pt`):
  efficiency gate accepted vs the re-fitted selector, quality gate not reached (+1.4% macro, −18.5% runtime).
  Decision files `e2_winner_vs_*.json` at the output root. Nothing 2D is running.
- 3D running/queued: `c3_extract_primary` (15716172) → `c3_train_replay_v3` (15721312; training done, single-threaded CPU replay slow, may need parallelizing) writes
  `3d_v2/c3/screen_primary/summary.csv`; `c5_holdout_apg3d_defaults` (15720688) and
  `c5_holdout_apg3d_refine_points_boxes` (15720689) on the 18 holdout crops; the slice-wise hybrid holdout
  run (`jobs/*c5_holdout_hybrid_local`) on the session GPU (aggregate with
  `screen_apg_3d_hybrid.py aggregate --subset holdout ... --beta 0.3`).
- Test manifest built (`3d_v2/manifest_test_apg3d-v1.json`, 56 crops) but never run.
- Next: read the C3 screen (gate: lower CI of the family-macro delta > 0, no family < −1%, unseen macro ≥ 0,
  passes ≤ 0.92x); if passed, `benchmark_apg_3d.py run --config <filter config> --volume-candidate-scorer-artifact`
  on primary + holdout; C4 union-through-filter needs an extractor `--prompts-from` pass (not implemented);
  then test arrays once and the v4 (geodesic) confirmations for the 2D and 3D shortlist.
- 2D direction change (08:30): only generalizing changes count (see APG_2D_OPTIMIZATION.md "Direction
  change"); the G campaign (generic features, LODO-first design) is running: `g1_generic_grid` (15720917) →
  `g2_generic_replay` (15720930). Read the G2 summaries (LODO vs baseline per dataset over 11 datasets), then
  decide the shortlist for production generalization and whether a library feature-selection hook is needed.
- G campaign jobs at the close of session 2 (09:30): `g2_generic_replay` primary tasks (15720930),
  `g2_extra_library_replay` (15721431), `g1p_pinned_grid` (15721581, partial-pinned extra features) → `g2p_pinned_replay` (15721582, primary task only), and the
  clean chain `extract_extra_pinned_v2` (15722759) → `g1p_v2_pinned_grid` (15722760) → `g2p_v2_pinned_replay`
  (15722761). The clean pinned corpus is `candidate_supply/{primary,training_extra}_features_ct1p5_fg0p7.npz`
  once the extraction lands; `models/` = mixed corpus, `models_pinned/` = partial-pinned, `models_pinned_v2/` = clean. Interim verdict (G1 proxies,
  G2 wave 2 on primary): generic features with linear models are safe but at baseline; H64 MLPs overfit
  dataset interactions even on generic inputs; per-image standardization and adaptive thresholds do not
  transfer. If G2/G2p confirm this, the 2D learned-selector line is closed and the generalizable 2D levers
  left are proposal-side (E2's ct 2.0 / mo 0.3 / ms 25 re-screened with the plain filter) and refinement.
- 09:50: the plain-path E2 setting is the one 2D candidate under the generalization rule (see the 2D notes,
  "E2 on the plain path"); jobs `e2plain_generalization` (15721819, holdout rows pending) and `e2plain_production`
  (15722179, 23 datasets). Next: `evaluate_apg_generalization.py --report`, then three bracketed timing trials of
  `apg_e2_plain_t0p5.json` vs `apg_control_registry_defaults.json` on the holdout (local 1g.20gb, `--serialize
  --bracket`), and the v4 (geodesic) sign check.

### State at the close of session 2, final (2026-09-03, 10:25)

- 2D under the generalization rule: closed negative for every learned or proposal-side change tried (see the 2D
  notes, "Where this leaves 2d under the generalization rule"). Still running at close: `e2plain_production`
  (tissuenet), the partial-corpus `g2p_pinned_replay` primary task, and the clean chain
  `g1p_v2_pinned_grid` → `g2p_v2_pinned_replay` (15722760/15722761, primary) and `g2p_v2_extra_replay` (15723413, extra); read them with
  `evaluate_apg_generalization.py report` and `summarize_generic_replay.py <run dirs>`; they are expected to
  confirm the verdict, not change it.
- 3D: `c3_train_replay_v3` (15721312) is in its slow single-threaded CPU replay; the G4 protocol in the 3D notes
  says how to read it. If it timed out (12 h), parallelize the replay before rerunning. The 3D holdout
  confirmations of the baseline and refinement are recorded; the hybrid is closed.
- Untracked new files this session: `optimization/{summarize_generic_selector_grid,summarize_generic_replay}.py`,
  `optimization/configs/apg_e2_{winner,plain_t0p5,plain_t0p6}.json`, extended trainer/screen/aggregate code,
  and tests in `test/test_train_apg_multimask_selector.py`. Nothing committed.

### How to find the results of the jobs left running at the close of session 2

Output root `R=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization`.

- **C3 learned-filter replay** (job 15721312, job dir `$R/jobs/20260903_090840_c3_train_replay_v3/`): state from
  `ls $R/jobs/20260903_090840_c3_train_replay_v3/logs/` (`.done` or `.failed` marker; `sacct -j 15721312 -X`), stdout
  in `logs/*.out`. Result: `$R/3d_v2/c3/screen_primary/summary.csv` (policy rows incl. `lodo`, `__family_macro__`,
  `__unseen_macro__`, `macro_delta_*` bootstrap) and `samples.csv`; models and training results in
  `$R/3d_v2/c3/models/`, aggregated candidates `$R/3d_v2/c3/training/candidates.npz`, chain script
  `$R/3d_v2/c3/train_and_replay.sh`, track cache `$R/3d_v2/cache/primary/103ee82e4d89/`. If the job timed out
  (12 h) with no summary, rerun only the last line of the chain script after parallelizing
  `screen_apg_3d_filter.py` (unpack each crop's tracks once, multiprocess over crops). Read under the G4 protocol
  in `APG_3D_OPTIMIZATION.md`.
- **Clean pinned selector screens** (jobs 15722761 primary and 15723413 extra; job dirs
  `$R/jobs/20260903_095755_g2p_v2_pinned_replay/`, `$R/jobs/20260903_102842_g2p_v2_extra_replay/`): run dirs are
  `$R/compact_selector_screening/hvit_t/85fb099c4bb038fa0ab9bddd6151689e/<hash>/` whose `metadata.json` has
  `oof_paths` under `generalization_g1/models_pinned_v2/` (subset `primary` / `training_extra`); join with
  `python optimization/summarize_generic_replay.py <primary run dir> <extra run dir> --output
  $R/multimask_selection/generalization_g1/g2p_v2_11datasets.csv`. Expected: no candidate above the
  predicted-IoU baseline by more than noise.
- **Partial-corpus primary screen** (job 15721582, `$R/jobs/20260903_092542_g2p_pinned_replay/`): same reader,
  models under `generalization_g1/models_pinned/`; secondary.

## Session 3 (2026-09-03/04): the structural 2d screen

- `optimization/screen_apg_structural.py`: `cache --subset S` (GPU; one encoding per image, the (4, Y, X) prediction
  as `.prediction.npy` and the proposals of every prompt type as `.pkl` under
  `<root>/structural_2d/cache/<subset>/<identity>/`, identity = proposal params + prompt types + checkpoint +
  implementation checksum), then `oracle` / `replay --workers 16` (CPU; `structural_2d/{oracle,replay}/<subset>/…`)
  and `report --subsets primary training_extra` (`structural_2d/reports/<checkpoint>/<subsets>/`: `gates.csv`,
  `per_dataset_relative.csv`, `identity.json`). The report refuses replays from different checkpoints and checks the
  registry replay against the canonical benchmark run of the same checkpoint image by image.
- The replay grid is fixed in `variant_grid()`; every variant is a `select()` option on cached proposals (fusion,
  arbitration, fixed / adaptive threshold) or a cached prompt type. A run of the 33 variants over 240 images takes
  two minutes on the `cpu` preset (16 CPUs, 1g.10gb GRES, `--qos 2h --time 02:00:00` schedules fastest).
- The session GPU on ggpu137 is a MIG 1g.10gb slice with one CPU: run the caches there, the replays on grete.
- Task files for `submit_optimization_jobs.py submit --tasks-file` go to the scratchpad; the `2h` QOS accepts about
  ten queued array tasks per user, the normal QOS (`2d` preset) has no such limit.
- v4 geodesic: stage `<root>/v4_geodesic_checkpoints/joint_sam2_hvit_t_multi_gpu/best.pt -> joint/v4/…/best.pt`,
  export `MICRO_SAM2_JOINT_CHECKPOINT_ROOT` before calling the submitter (it pins the variable into `job.sh`), keep
  the same output root (run directories, caches and reports are keyed by the checkpoint checksum). The v4 decoders
  are half-width (`initial_features=32`); `UniSAM2` rebuilds its decoder at that width since session 3, because the
  installed `torch_em` 0.10.1 ignores the argument.

## Session 3, refinement campaign (2026-09-03, 21:45-23:05)

- Epoch 5 `4fa97979b2aa4173e3c1d3fd38d00b66` (refinement kwargs `protect_neighbours`, `negative_scope`,
  `gate="isolated"`, `isolated_fallback`, `touch_radius`; benchmark `IMAGE_DIAGNOSTICS`). Registry controls at epoch 5
  exist for both checkpoints and all three manifests and equal epoch 4 per image.
- Refinement screens take a *list-shaped* config (`configs/apg_r_refinement_screen.json`: `[{"name", "params_2d"}]`,
  every entry pinning the proposal keys identically), the canonical runs and the production evaluator take the
  *single-object* `apg_s_*.json` shape; keep both in sync by generating the single-object files from the list entries.
  `refinement_kwargs` passes through every layer unvalidated until `_parse_refinement`.
- `report_refinement_screen.py --latest --subsets primary training_extra` (with `MICRO_SAM2_JOINT_CHECKPOINT_ROOT`
  exported for v4) joins the newest screen per subset of the current checkpoint, applies the campaign rule against the
  `none` entry, checks `none` against the registry benchmark and `pb` against the canonical `s-refine-pb` run per image,
  and writes `costs.csv` (full-prompt / box-only / isolated fractions, negatives per instance, protected pixels).
  Screens have no `status` field: a `summary.csv` beside `metadata.json` is the completion marker.
- Background shell commands inherit the session's drifted cwd: launch scripts with absolute paths (two launches
  failed on `optimization/...` vs `finetuning/v2/evaluation/...` this session).
- Timing trials: a 5% bracket drift on one node (146 / 154 s vs 138 / 139 s) is normal on `grete:preemptible`; read
  per-dataset medians over three trials, as `s8_timing_v4_holdout` did.

## Visual case check (2026-09-04)

`optimization/visualize_refinement_cases.py --dataset <name> --variant <screen entry> --n 5 --checkpoint {v2,v4}`
needs a refinement screen that contains the variant and `none` for the manifest holding the dataset; it recomputes
the shown cases with the real model on the session slice (about a minute per dataset) and writes to
`<root>/structural_2d/visual/<checkpoint>/<dataset>/<variant>/{improvements,decreases}/` plus `ranking.csv`. Run it
before reading a screen's per-dataset table: the 2d campaigns of 2026-09-03 were decided on mSA movements that turned
out to be one-pixel boundary conventions on small objects (see the closing section of `APG_2D_OPTIMIZATION.md`).
