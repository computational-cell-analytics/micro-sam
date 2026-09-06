# Experimental set-up of the APG optimization campaigns

The reproducible set-up behind the 2026-09 APG (automatic prompt generation) optimization campaigns,
recorded so that the same data, splits, metrics, gates and cluster mechanics can be reused for the
next instance-segmentation optimization (AIS, the decoder-based mode). Everything named here exists on
branch `apg-clean-up`; the experiment code the campaigns themselves ran, and every refuted mechanism,
is preserved unchanged on branch `apg-optim-fable` (commit `356b76d`, on origin).

Paths are relative to `finetuning/v2/evaluation/` unless they start with `micro_sam/`. `<root>` is the
output root, `/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization`.

## 1. What is generic and what is APG-specific

Generic (reusable as-is for AIS):

- the dataset registry, validation splits and size floors in `common.py`;
- the 2D subset manifests and their builder in `optimization/benchmark_apg_optimization.py`;
- the 3D crop manifests and their builder in `optimization/apg3d_manifest.py`;
- the per-sample metrics (`parameter_search.compute_metrics`) and the aggregation rules;
- the comparator and its acceptance gates (`optimization/compare_apg_optimization.py`);
- the SLURM array submitter and task registry (`optimization/submit_optimization_jobs.py`,
  `optimization/apg_campaign_tasks.py`);
- the implementation-checksum epochs and the output-root layout.

APG-specific (an AIS campaign needs an analogue, see section 13): the parameter resolution over
`generate()` keywords (`common.GENERATE_PARAM_KEYS`, `common.resolve_params`,
`benchmark_apg_3d.resolve_volume_params`), the segmenter construction (`common.build_apg_segmenter`),
the refinement statistics columns, and the configuration files under `optimization/configs/`.

## 2. Environment and cluster

- Environment: `micromamba activate new-stack`. Both submitters (`submit_all_evaluations.py`,
  `parameter_search.py`) activate it by default since 2026-09-06; the earlier default `super` does not exist
  on grete.
- Partition `grete:preemptible` (2-day limit). GRES pools: `1g.10gb:1` (plentiful), `1g.20gb:1`
  (8 slices), `2g.20gb:1` (16 slices), `3g.40gb:1` (8). `grete:interactive` allows two jobs per user
  for 12 h. Every job needs `--constraint=inet`. Account `nim00007`; QOS `2h` and `normal` only.
- Job script body order: `set -eo pipefail`, then `source ~/.bashrc`, then `set -u` (an early
  `set -u` fails on `/etc/bashrc`).
- Presets (`submit_optimization_jobs.PRESETS`):

  | preset     | GRES        | memory | time     | QOS  | CPUs |
  |------------|-------------|--------|----------|------|------|
  | `2d`       | `1g.10gb:1` | 16G    | 08:00:00 |      | 4    |
  | `2d-short` | `1g.10gb:1` | 16G    | 02:00:00 | `2h` | 4    |
  | `3d`       | `2g.20gb:1` | 32G    | 12:00:00 |      | 4    |
  | `3d-large` | `2g.20gb:1` | 64G    | 12:00:00 |      | 4    |
  | `cpu`      | `1g.10gb:1` | 64G    | 04:00:00 |      | 16   |

- The submitter writes `<root>/jobs/<timestamp>_<name>/` with `tasks.txt` (`tag<TAB>command`),
  `job.sh`, `logs/`, `submit.json` (argv, resources, git revision, dirty flag) and `job_id.txt`, and
  submits `job.sh` as one array (`--array=0-N%throttle`, default throttle 8, `--requeue`,
  `--open-mode=append`). Every task leaves `logs/<tag>.done` or `logs/<tag>.failed`; dependent
  stages wait on those markers or on `--dependency afterok:<job id>`, never on an output file.
  `status <job dir>` reports state, exit code, restarts and marker per task; `--resume-from <job dir>`
  re-submits the unfinished tasks; `--local` runs the same tasks sequentially on the session GPU.
  `MICRO_SAM2_JOINT_CHECKPOINT_ROOT` and `MICRO_SAM2_JOINT_EXPORT_ROOT` are pinned into `job.sh`
  (`PINNED_ENV_VARS`), so a job resolves the same checkpoints as the shell that submitted it.
- Production evaluations go through `submit_all_evaluations.py` (one job per dataset and mode, 8 h,
  `grete:preemptible`, `--constraint=inet`): 2D jobs `1g.10gb:1` / 16G, 3D jobs `1g.20gb:1` / 64G, both
  checkpoint variables pinned into the script; `--gpu`, `--memory`, `--env`, `--dry` override or inspect.
- Always `--dry-run` first and read `job.sh`; `sbatch --test-only job.sh` checks the header.
- Runs resume per sample from `samples.csv` (2D) or `crops/*.json` (3D), both written atomically, so a
  requeued task continues where it stopped.

## 3. Data and splits (`common.py`)

- Data root (read-only): `/mnt/vast-nhr/projects/cidas/cca/data` (`DATA_ROOT`).
- Production 2D datasets (`DATASETS_2D`, 23): livecell, arvidsson, bitdepth_nucseg, cellbindb,
  cellpose_data, covid_if, cvz_fluo, deepbacs, deepseas, dic_hepg2, dsb, dynamicnuclearnet, hpa,
  microbeseg, neurips_cellseg, omnipose, puma, segpc, tissuenet, tnbc, usiigaci, vicar, yeaz.
- Production 3D datasets: LM (`DATASETS_3D_LM`, 10): blastospim, cartocell, celegans_atlas,
  cellseg_3d, embedseg, gonuclear, mouse_embryo, nis3d, plantseg, pnas_arabidopsis; EM
  (`DATASETS_3D_EM`, 4): platynereis_nuclei, cremi, snemi, humanneurons. The dense (multicut) pipeline
  and the CREMI score apply to `DATASETS_DENSE` = the EM datasets without platynereis_nuclei.
- Ground-truth size floor (`GT_MIN_SIZE_2D`, measured, never tuned): livecell 50, cellpose_data 20,
  deepbacs 50, dynamicnuclearnet 50, tissuenet 10, u20s 10, vicar 25, yeaz 10.
- Evaluation crops: `CROP_SHAPE_2D = (512, 512)`, `CROP_SHAPE_3D = (8, 512, 512)`.
- Tuning splits (`VAL_SPLITS`): `val` for livecell, tissuenet, dynamicnuclearnet, deepbacs, dic_hepg2,
  celegans_atlas, covid_if, yeaz, neurips_cellseg, puma, tnbc; `train` for embedseg and deepseas
  (never seen in joint training); none for gonuclear, cremi, snemi, platynereis_nuclei, humanneurons,
  which hold out a z-slab instead (`VAL_Z_RANGE`: cremi 0:32, snemi 0:8, gonuclear 32:96,
  humanneurons 0:16, counted from what `load_volume` keeps, so snemi starts at original slice 70).
  platynereis_nuclei tunes on three of its twelve volumes with 16-slice windows
  (`PLATYNEREIS_NUCLEI_VAL_SAMPLES = {1: (28, 44), 5: (2, 18), 8: (99, 115)}`). A dataset whose `val`
  split is the evaluated split is absent from `VAL_SPLITS`; `has_val_split` refuses to tune there.
- Two incompletely annotated LIVECell test images are excluded (`LIVECELL_EXCLUDED_TEST_IMAGES`).
- SNEMI production scoring reads original slices 81:89 (drop z < 70, then the centre 8 of the
  remaining 30). Every tuning crop must stay clear of that slab (section 6).
- Seen-in-training: CREMI samples A and B and all EmbedSeg training sub-datasets were joint-training
  data; celegans_atlas, gonuclear, CREMI C, snemi and humanneurons were not; platynereis_nuclei is
  uncertain ("maybe"). The eleven-dataset 2D development corpus (section 5) leaves twelve production
  datasets strictly unseen by any tuning: arvidsson, bitdepth_nucseg, cellbindb, cellpose_data,
  cvz_fluo, dsb, hpa, microbeseg, omnipose, segpc, usiigaci, vicar.

## 4. Checkpoints

- Joint checkpoints live under `_joint_checkpoint_root()` =
  `$MICRO_SAM2_JOINT_CHECKPOINT_ROOT` or `/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/joint/v2/checkpoints`,
  as `joint_sam2_<model_type>_multi_gpu/<checkpoint>.pt` (`get_joint_checkpoint`). The root is read
  on every call, never at import time, so set the variable before the process starts.
- `export_joint_checkpoint` splits a joint file into `<name>.pt` (SAM2) and `<name>_decoder.pt`
  (UniSAM2) under `_joint_export_root()` (`$MICRO_SAM2_JOINT_EXPORT_ROOT` or `.../exported/joint/v2`),
  keyed by `checkpoint_checksum` (xxh128 of the file bytes), so a re-trained `best` never reuses a
  stale export. A run's checkpoint identity is `combine_checkpoint_checksums(joint, decoder)`.
- The campaigns used `hvit_t` only, developing on joint/v2 `best` (checksum `85fb099c…`) and
  confirming on joint/v4 geodesic (checkpoint checksum `5a729846…`, an improvement of +9-10 % over v2
  on the 2D primary and holdout subsets and +6-7 % on the 3D subsets; it is now the default model).
- v4 staging recipe: create `<root>/v4_geodesic_checkpoints/joint_sam2_hvit_t_multi_gpu/best.pt` as a
  symlink to `.../joint/v4/checkpoints/joint_sam2_hvit_t_geodesic_multi_gpu/best.pt`, then
  `export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<root>/v4_geodesic_checkpoints` before submitting. The v4
  decoders are 32 features wide (the v2 ones 64); the loader reads the width off `out_conv.weight` and
  passes `initial_features` through `UniSAM2` to torch_em's `UNETR3D`, which honours it from torch_em
  0.10.4 on (0.10.1 silently built a 64-wide decoder, so v4 checkpoints need the newer torch_em).
- 3D campaign roots per checkpoint: v2 under `<root>/3d_v2`, v4 geodesic under `<root>/3d_v4geo`
  (`package_apg3d_cases.CHECKPOINTS`).

## 5. 2D subsets (`optimization/benchmark_apg_optimization.py`)

Manifest schema version 5; files `<root>/subset_manifest_v5{,_holdout,_training_extra,_deep3d}.json`
(`_default_manifest_path`). Each manifest records its `manifest_checksum`, `selection_policy`,
`schema_version` and `data_root`; `_validate_manifest` requires the exact schema version.

| subset           | definition                                         | content                                                                 | checksum                           |
|------------------|----------------------------------------------------|-------------------------------------------------------------------------|------------------------------------|
| primary          | `SAMPLE_COUNTS_2D`                                 | livecell 80 (10 per each of 8 `LIVECELL_TYPES`), tissuenet 40, dynamicnuclearnet 40, deepbacs 30, dic_hepg2 50 = 240 images, plus one 12-slice volume each of celegans_atlas, embedseg, gonuclear, cremi, snemi (245 samples) | `0f8fb67b3650a71f9f44b53037e89546` |
| holdout          | `SAMPLE_COUNTS_2D_HOLDOUT`, image-disjoint          | 80 / 40 / 40 / 30 / 43 = 233 images plus the same 5 volumes (238 samples); deepbacs is reused verbatim (`HOLDOUT_REUSED_DATASETS`) because all 30 validation images are primary | `bf8f3c28befe1fb06d62309dc302d1c4` |
| training_extra   | `TRAINING_EXTRA_DATASETS`, `SAMPLE_COUNTS_2D_TRAINING_EXTRA` (caps) | yeaz 40, neurips_cellseg 40, deepseas 40, puma 26 (cap 40), covid_if 5, tnbc 6 (cap 20) = 157 images, no volumes | `cee6224d6a93cec5a54a5c522a0f7bf5` |
| deep3d variant   | `--crops-3d deep`, `CROP_SHAPE_3D_DEEP = (32, 512, 512)` | the 240 primary images with 32-slice volumes; SNEMI 30 slices overlap the production slab, so this is a regression instrument, not a tuning set | `f611a7125383e850798d0b5bf696f6f7` |

- The eleven-dataset development corpus of the 2026-09 campaigns is primary + training_extra
  (397 images). Primary and holdout carry the quality figure; training_extra never enters them.
- Selection rule (`_scan_2d_dataset`, `_select_2d_samples`): every validation image is centre-cropped
  to 512², relabelled by connected components, and crop-severed slivers below `GT_MIN_SIZE_2D` are
  dropped; the complexity of an image is the mean percentile rank of its object count and foreground
  fraction (`_percentile_ranks`, `_add_complexity`); the images nearest to evenly spaced complexity
  quantiles are taken (`_quantile_targets`, `_select_nearest`); LIVECell is stratified by cell type.
  `sample_id = "<dataset>:<xxh128(dataset, raw path, label path, roi)[:12]>"` (`_sample_identity`).
- The training_extra `"role": "selector-training-only"` string in `_selection_policy` is part of the
  manifest identity and therefore frozen, although the selectors it named are gone.
- Standard 3D crops of this benchmark: `CROP_SHAPE_3D = (12, 512, 512)` (celegans `(32, 140, 512)`),
  `CANDIDATE_GRID_3D = (4, 3, 3)`, one crop per volume dataset at the 0.5 complexity target.

## 6. 3D subsets (`optimization/apg3d_manifest.py`)

Schema `apg3d-v1`, campaign root `<root>/3d_v2`, files `manifest_{primary,holdout,test}_apg3d-v1.json`.
Constants: `DEEP_DEPTH = 32` (`MIN_REALIZED_DEPTH = 24` slices of annotation make a crop deep),
`OBJECTS_PER_PASS = 16`, `SEED = 17`, `SNEMI_TEST_SLAB = (81, 89)`.

| subset  | crops | per source                                                                                                                          | checksum                           |
|---------|-------|-------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| primary | 57    | gonuclear 10, embedseg_platy_nuclei 8, humanneurons 8, embedseg_platy_ish 7, celegans_atlas 6, snemi 6, cremi_seen 4, cremi 3, embedseg_skull 3, platynereis_nuclei 2 | `1cf951b9f2784b4ad9ee6126496cbf61` |
| holdout | 18    | gonuclear 3, celegans_atlas 2, cremi_seen 2, embedseg_platy_ish 2, embedseg_platy_nuclei 2, embedseg_skull 2, snemi 2, cremi 1, humanneurons 1, platynereis_nuclei 1 | `5ddc25cb56afb79195e72c773ac711d5` |
| test    | 56    | 8 crops each of the seven `TEST_ONLY_DATASETS` (blastospim, cartocell, cellseg_3d, mouse_embryo, nis3d, plantseg, pnas_arabidopsis); built, never run | `33043f1ef079b10e6f6562297b3ff83c` |

- The ten tuning sources (`tuning_source_specs`) carry a family, a legal z range, a crop shape, a
  holdout rule, a `seen_in_training` flag and a metric mode: celegans_atlas (unseen, crop
  `(32, 140, 512)`), embedseg_skull / embedseg_platy_nuclei / embedseg_platy_ish (seen), gonuclear
  (unseen, z 32:96), cremi C (unseen, dense metric), cremi_seen A+B (seen, dense), snemi (unseen,
  dense, crop `(11, 512, 512)`), humanneurons (unseen, dense, crop `(16, 512, 512)`), platynereis_nuclei
  ("maybe", 16-slice windows, invalid labels masked). EmbedSeg Mouse-Organoid-Cells is excluded (four
  annotated cells per volume).
- SNEMI legal z is original 70:81 and 89:100 (`_snemi_legal_z`): held out from training and clear of
  the production slab. `validate_manifest` refuses any crop overlapping the slab. The stopped 2026-09-02
  3D campaign under `<root>/3d_campaign` had this leak; its manifests are kept only for sample-id
  continuity (`LEGACY_MANIFESTS`).
- Holdout is volume-disjoint (per-source basename rules), except for single-volume sources which hold
  out one spatial quadrant (`QUADRANT_HOLDOUT`: cremi (738, 738), snemi (512, 512), humanneurons
  (1536, 1536)); every crop overlapping the quadrant leaves primary.
- Folds are grouped by source volume (`_assign_folds`, sha256-stable, five folds), which is the
  leakage-safe fold definition any learned component must use. Crop selection reuses the 2D
  complexity-quantile machinery (`_balanced_select`, distinct sources preferred).

## 7. Running a benchmark and the metrics

- 2D/standard-3D canonical run: `optimization/benchmark_apg_optimization.py --config <json>
  [--subset primary|holdout|training_extra] [--serial --trial-id <id>]`; results under
  `<root>/hvit_t/<checkpoint id>/<manifest ck>-<params ck>-<implementation ck>/` with `samples.csv`
  and a summary keyed by manifest, parameters, implementation and hardware identity
  (`_hardware_identity`).
- Deep-3D run: `optimization/benchmark_apg_3d.py run --subset <s> --config <json> --sample-index <i>`
  (one crop per task, `apg_campaign_tasks.py per-sample`), then `aggregate`; run directories
  `<campaign root>/runs/<subset>/<config name>-<params hash 12>-<implementation ck 12>`; the
  aggregate also reads sibling directories of other implementation checksums
  (`sibling_run_dirs`), current implementation winning. Case inspection: `package_apg3d_cases.py`
  (HDF5 per crop) and `view_apg3d_cases.py` (napari).
- Per sample: `parameter_search.compute_metrics` gives `msa` (`elf.evaluation.mean_segmentation_accuracy`)
  and, for `metric_mode="dense"`, `cremi`, `vi_split`, `vi_merge`, `adapted_rand`. 2D segmentations pass
  through `drop_severed_objects` first, symmetric with the ground-truth filtering.
- 2D aggregation (`_summarize`): per-dataset mean and std, then the row `__dataset_balanced__` = the
  equal-weight mean of the per-dataset means. This is "balanced mSA".
- 3D aggregation (`benchmark_apg_3d.summarize`): per-dataset mean with a 2000-sample bootstrap CI,
  then `__family_macro__` (mean over families), `__dataset_balanced__`, `__unseen_macro__`
  (`seen_in_training == "False"`) and `__legacy_macro__` (`LEGACY_FAMILIES`). Object counts per crop
  (`object_counts`: `gt_objects`, `severed_objects`, `merged`, `unmatched`, `genuine_misses`) are the
  first thing to read on small-object data, where mSA swings on 1-2 px rims.
- Runtime and peak CUDA memory are recorded per sample; the comparator pairs runs only within one
  implementation checksum and one hardware identity.

## 8. Parameters, defaults and controls

- 2D parameters are resolved by `common.resolve_params(overrides, ndim, model_type)` over
  `GENERATE_PARAM_KEYS` from `micro_sam.v2.automatic_prompt_generation.default_prompt_generation`;
  volumes by `benchmark_apg_3d.resolve_volume_params`, which starts from the library's volume defaults
  (sigma, minimum candidate size, overlap and size floor differ between an image and a volume).
- Config files are `{"name", "params_2d", "params_3d"}` (`common.load_apg_overrides`, also accepted
  by `evaluate_automatic_segmentation.py --apg_params` with `--result_tag`).
- Three sets of 2D proposal settings existed, and every config pinned `candidate_threshold`, `dt` and
  `max_overlap` explicitly because two of them are both called "defaults":
  - registry defaults for `hvit_t` since commit `9fd3b57`: `candidate_threshold 3.0, dt 0.5,
    max_overlap 0.3` (`configs/apg_control_registry_defaults.json`, empty overrides);
  - campaign defaults of the historical accepted runs: `candidate_threshold 1.5, dt 0.25, sigma 0.5,
    min_candidate_size 4, foreground_threshold 0.7, max_overlap 0.15, min_size 50`
    (`configs/apg_control_campaign_defaults.json`);
  - the pinned proposal settings every learned 2D artifact was extracted with, `{"candidate_threshold":
    1.5, "dt": 0.25, "sigma": 0.5, "min_candidate_size": 4, "foreground_threshold": 0.7}` (formerly
    `PINNED_PROPOSAL_2D` in the removed `screen_apg_multimask.py`).
- 3D controls: `configs/apg3d_defaults.json` (library volume defaults),
  `configs/apg3d_legacy_defaults.json` (`sigma 0.5, min_candidate_size 4, min_size 50, dt 0.25,
  max_overlap 0.15`, the continuity control for the pre-`3d_v2` runs) and
  `configs/apg3d_refine_points_boxes.json` (`refinement: "points+boxes"`, the kept second round).

## 9. Acceptance gates (`optimization/compare_apg_optimization.py`)

`--target` selects the gate; every check must pass. Dataset set: `EXPECTED_DATASETS` (the five
primary datasets per dimensionality). Runtime and quality changes are relative to the baseline run;
timing trials are grouped per implementation checksum and hardware identity and the per-dataset medians
compared.

- quality: balanced macro mSA ≥ +5 %; at most two datasets below −5 %; every dataset's runtime
  ≤ +10 %, or the exception "macro ≥ +10 % and every dataset positive".
- efficiency: every dataset's quality ≥ −0.5 %; every dataset ≥ +5 % speed-up.
- refinement: macro mSA improves; every dataset ≥ −1 %; aggregate runtime ≤ +10 %; every dataset's
  runtime ≤ +15 %; peak CUDA memory ≤ +10 %.
- replacement: macro ≥ −0.5 %; every dataset ≥ −2 % relative or ≥ −0.005 absolute; aggregate speed-up
  ≥ +5 %; every dataset's runtime ≤ +2 %; peak CUDA memory ≤ +10 %.

The generalization gate used by the 2026-09 structural and refinement screens (implemented in the
removed `screen_apg_structural.gate_table`, recorded here): over the eleven development datasets a
change passes if it is up on at least 9 of 11 datasets, no dataset loses more than 2 % relative and
0.005 absolute balanced mSA, and the balanced gain is at least +2 %. The production variant relaxed the
loss limit to −5 % / 0.005. Under the user's generalization rule only changes that improve consistently
across datasets count; per-dataset fits and dataset-gated modes are rejected outright.

## 10. Timing-trial protocol

Node-level drift of 3-7 % in generation time was measured with byte-identical code, so:

- canonical timing runs are serialized on one GRES type (`--throttle 1`) and bracketed by a defaults
  run before and after (`apg_campaign_tasks.py benchmark --serialize --bracket`); if the brackets differ
  by more than about 2 % (2D) or 3 % (3D) the cost column of that trial is discarded and repeated;
- three trials per configuration, the comparator takes per-dataset medians;
- a timing trial that was preempted and restarted is discarded and re-run under a fresh trial id
  (quality results are unaffected by restarts).

## 11. Implementation checksum epochs

`benchmark_apg_optimization._implementation_checksum()` is the xxh128 over the bytes of
`IMPLEMENTATION_FILES`: the benchmark itself, `common.py`, `parameter_search.py` and
`micro_sam/v2/{automatic_prompt_generation, instance_segmentation, postprocessing,
prompt_based_segmentation}.py` (seven files; `micro_sam/v2/multimask_selection.py` was the eighth
until it was removed). Any edit to these files re-keys every new run directory.

Rules: batch edits to checksum files into one commit at an epoch boundary and re-run the controls;
never edit them while an array is queued (a task reads the working tree when it starts); record the
old and new value in the notes.

Epochs of the 2026-09 campaigns: `aeb1aca09a5fff43d2b8bb8bacff2b06` (campaign start) →
`d11e2404…` (phase 0 hooks) → `14800942…` (NaN stability fix) → `26a1003788ea2825356b486da1496fd7`
(harness-only edit, accidental) → `41abe8ca0cf86fadcf5d46ea183bb296` (structural hooks) →
`4fa97979b2aa4173e3c1d3fd38d00b66` (refinement kwargs; the last epoch of `apg-optim-fable`) →
`f76ee7170ca77da882c0078dfaa5b301` (this branch after the clean-up commit; the baselines of section 14) →
`e1903b1b3c1e4e3610c71e1d0bd81f1d` (2026-09-06, harness-only: the `parameter_search.py` job template activates `new-stack`,
results unaffected). Historical run directories
stay valid records under their own epochs; the 3D aggregate reads them through `sibling_run_dirs`.

## 12. Output root layout

```
<root>/subset_manifest_v5{,_holdout,_training_extra,_deep3d}.json   2D manifests (legacy v3/v4 beside them)
<root>/hvit_t/<checkpoint id>/<manifest>-<params>-<implementation>/  canonical 2D / standard-3D runs
<root>/model_exports/                                              exported joint checkpoints
<root>/jobs/<timestamp>_<name>/                                    submitter job directories
<root>/3d_v2/manifest_{primary,holdout,test}_apg3d-v1.json          3D manifests
<root>/3d_v2/runs/{primary,holdout}/<config>-<hash>-<impl>/          3D runs on joint/v2
<root>/3d_v4geo/                                                   the same layout on joint/v4 geodesic
<root>/3d_cases/<subset>/                                          packaged napari cases (README inside)
<root>/v4_geodesic_checkpoints/                                    staged checkpoint root for the env var
<root>/prompt_state_replay/                                        benchmark_prompt_state_replay outputs
<root>/campaign*_*.json, e2_*.json                                 comparator decision files
```

Historical trees written only by code that lives on `apg-optim-fable` (data, readable there):
`multimask_selection/`, `structural_2d/`, `refinement_screening/`, `candidate_supply_screening/`,
`compact_selector_screening/`, `mask_head_filter_screening/`, `multimask_screening/`,
`production_generalization/`, `3d_campaign/` (the leaky first 3D campaign) and
`3d_v2/{c3, cache, hybrid, screens}`.

## 13. Setting up an AIS campaign on this set-up

1. Reuse the manifests unchanged: the 2D subsets (section 5) through `benchmark_apg_optimization.py`'s
   loaders, the 3D crops through `apg3d_manifest.load_sample` / `load_labels`. Do not rebuild them; a
   rebuilt manifest must reproduce the recorded checksum.
2. Replace the parameter resolution: instead of `resolve_params` over `generate()` keywords, resolve
   over the AIS post-processing grids in `parameter_search.py` (`LM_GRID` for the sparse flow
   pipeline, `EM_GRID` for the dense multicut pipeline; `tuning_config` picks the grid per dataset).
   Build the model with `common.build_ais_model_from_checkpoint` or `common.build_model(mode="ais")`,
   predict with `common.predict_unisam2` and post-process with `common.postprocess_unisam2`.
3. Keep the runner conventions: one run directory per (manifest, parameters, implementation,
   hardware), atomic per-sample CSV/JSON, `__dataset_balanced__` for 2D and the family/unseen macros
   with object counts for 3D.
4. Add the AIS files to `IMPLEMENTATION_FILES` in place of `automatic_prompt_generation.py` and record
   the resulting epoch here before the first canonical run.
5. Keep the gates of section 9 and the timing protocol of section 10; `EXPECTED_DATASETS` in the
   comparator is the one hard-coded dataset set to generalize if the subsets change.
6. Judge every candidate under the generalization rule: development on primary + training_extra,
   confirmation on holdout, one production run on the 23 (2D) or the test manifest (3D) at the very
   end, with the twelve strictly unseen 2D datasets as the out-of-domain check.

Status (2026-09-06): implemented as `optimization/benchmark_ais_optimization.py` (`predict` caches the
decoder predictions per manifest sample under `<root>/ais/predictions/`, `run` / `screen` / `sweep` /
`oracle` / `report` work on the cache), task builder `optimization/ais_campaign_tasks.py`, configurations
`optimization/configs/ais_*.json`, decision log `notes/AIS_V4_OPTIMIZATION.md`. The AIS implementation
checksum covers five files (the benchmark, `common.py`, `parameter_search.py`,
`micro_sam/v2/{instance_segmentation, postprocessing}.py`). AIS epochs: `f57b117edfda5420d9df761b1db4db2d`
(frozen Phase 0 harness) → `5700c6e0f471b360013551a442b1e53d` (harness only: refined loss decomposition
columns) → `a65e2eb08c23538f11544860736961a3` (epoch A1, 2026-09-06: opt-in `boundary_magnitude_max`
filter in `micro_sam/v2/postprocessing.py`, default off; the cached sweep scorer applies it) →
`576a85c8ffd4314627812fd30a3c1223` (epoch A2, 2026-09-06: the optimized `hvit_t` defaults with volume overrides, the fast filter
and dimension-aware `default_postprocessing`). Decision log and results: `notes/AIS_V4_OPTIMIZATION.md`.

## 14. Baseline results of the cleaned harness (2026-09-06)

Reruns of the default settings with the joint/v4 hvit_t geodesic checkpoint (checksum `5a729846…`) on the
harness of this branch (implementation epoch `f76ee7170ca77da882c0078dfaa5b301`), run to verify the
clean-up and to serve as the baselines for the next optimization. Everything below is on this machine.

### 14.1 APG, 2D subset benchmark (registry defaults, trial `verify-1`)

Run directories under `<root>/hvit_t/5a729846c141daf73c27b24f52d8af4f/`, each with `summary.csv`,
`samples.csv` and `metadata.json`; the parameter checksum is `d914b807f7c6719914ae4b3e6fbcac80` (the
recorded v4 controls of session 3 carry `9a58f84a…` for the same configuration, because the resolved
parameter dict lost the removed keys):

| subset | run directory | balanced mSA | per-dataset mSA |
|---|---|---:|---|
| primary | `0f8fb67b3650a71f9f44b53037e89546-d914b807f7c6719914ae4b3e6fbcac80-f76ee7170ca77da882c0078dfaa5b301` | 0.295460 | livecell 0.391248, tissuenet 0.289299, dynamicnuclearnet 0.461289, deepbacs 0.320974, dic_hepg2 0.014491 |
| holdout | `bf8f3c28befe1fb06d62309dc302d1c4-d914b807f7c6719914ae4b3e6fbcac80-f76ee7170ca77da882c0078dfaa5b301` | 0.289588 | livecell 0.390674, tissuenet 0.293459, dynamicnuclearnet 0.434222, deepbacs 0.320974, dic_hepg2 0.008612 |
| training_extra | `cee6224d6a93cec5a54a5c522a0f7bf5-d914b807f7c6719914ae4b3e6fbcac80-f76ee7170ca77da882c0078dfaa5b301` | 0.463378 | yeaz 0.677109, neurips_cellseg 0.240862, puma 0.523091, tnbc 0.419334, covid_if 0.744593, deepseas 0.175276 |

All 630 per-sample scores are identical to the recorded v4 controls (`…-9a58f84a…-4fa97979…`). Run
locally on a `1g.20gb` slice; runtimes are therefore not comparable with the recorded `1g.10gb` runs.

### 14.2 APG, 3D deep crops (`configs/apg3d_defaults.json`, trial `verify-1`)

Run directories `<root>/3d_v4geo/runs/{primary,holdout}/apg3d-defaults-70b90e407d4b-f76ee7170ca7/` with
`crops/*.json`, `samples.csv` and `summary.csv` (written by `benchmark_apg_3d.py aggregate`); job
directories `<root>/jobs/20260906_132801_verify_v4_3d_defaults_primary` and
`<root>/jobs/20260906_132802_verify_v4_3d_defaults_holdout` (`1g.20gb:1`, throttle 8).

| subset | crops | family macro | dataset balanced | unseen macro | matched / misses / gt objects |
|---|---:|---:|---:|---:|---|
| primary | 57 | 0.32703 (recorded 0.32666) | 0.32798 | 0.32711 | 3751 / 1805 / 13793 (recorded 3750 / 1806) |
| holdout | 18 | 0.34239 (identical) | 0.36990 | 0.31947 | 1162 / 564 / 2127 (identical) |

Per-source means (primary): celegans_atlas 0.1521, cremi 0.1341, cremi_seen 0.0300, embedseg_platy_ish
0.4458, embedseg_platy_nuclei 0.2569, embedseg_skull 0.6602, gonuclear 0.3536, humanneurons 0.4367,
platynereis_nuclei 0.2514, snemi 0.5590. Holdout: celegans_atlas 0.1375, cremi 0.1623, cremi_seen 0.3168,
embedseg_platy_ish 0.3899, embedseg_platy_nuclei 0.5222, embedseg_skull 0.6819, gonuclear 0.4399,
humanneurons 0.3410, platynereis_nuclei 0.1907, snemi 0.5167.

56 of the 57 primary crops and all 18 holdout crops are identical to the recorded v4 run
(`…-4fa97979b2aa`). The crop `gonuclear:4ea4ece3dbe1` is nondeterministic run to run (0.1364, 0.1579 and
0.1623 were observed across four runs, two of them with the pre-clean-up code on the same node): its 26
anchor candidates are the same, but one borderline anchor decision flips, which moves one merged object.

### 14.3 AIS and APG defaults on the production test splits

`evaluate_automatic_segmentation.py --skip_tuning` for both modes on nine datasets, results in
`/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/v4_geodesic_cleanup_verification/results/`
as `<dataset>_micro_sam2_hvit_t_{ais,apg}_default_ckpt-5a729846c141daf73c27b24f52d8af4f.csv` (one row:
mSA/SA50/SA75/precision/recall/F1, or cremi/vi_split/vi_merge/adapted_rand for the dense EM datasets). The
job scripts and logs are under `finetuning/v2/evaluation/gpu_jobs/20260906_132906/` (git-ignored).

| dataset | metric | AIS v4 default | APG v4 default | AIS v2 ref | APG v2 ref |
|---|---|---:|---:|---:|---:|
| livecell | mSA | 0.2575 | 0.3863 | 0.2533 | 0.3422 |
| deepbacs | mSA | 0.2056 | 0.4097 | 0.2940 | 0.4133 |
| dsb | mSA | 0.4631 | 0.5587 | 0.4248 | 0.5167 |
| dynamicnuclearnet | mSA | 0.5083 | 0.4648 | 0.5075 | 0.4744 |
| gonuclear | mSA | 0.2689 | 0.3730 | 0.2459 | 0.3570 |
| embedseg | mSA | 0.4105 | 0.6402 | | |
| cremi | CREMI (lower is better) | 0.4858 | 0.4418 | | |
| snemi | CREMI | 0.9085 | 0.5958 | | |
| humanneurons | CREMI | 0.6999 | 0.3429 | 0.6698 | 0.3992 |

References: v2 `best` defaults for both modes in
`experiments/v2_registry_default_evaluation/results/<dataset>_micro_sam2_hvit_t_{ais,apg}_default_ckpt-85fb099c….csv`;
an earlier v4 AIS default run of 2026-08-30 (six datasets) in
`experiments/v4_joint_evaluation_hvit_t_geodesic/results/`, which the rerun matches within 3 % everywhere
(livecell −0.1 %, gonuclear +0.2 %, embedseg +2.9 %, cremi +0.2 %, snemi −0.5 %, humanneurons +0.0 %).
Against v2: APG v4 gains on average (+4.5 % over the five mSA datasets; livecell +12.9 %, dsb +8.1 %,
gonuclear +4.5 %, deepbacs −0.9 %, dynamicnuclearnet −2.0 %) and improves the CREMI score on humanneurons;
AIS v4 is mixed (−2.0 % on average: dsb +9.0 %, gonuclear +9.4 %, livecell +1.7 %, dynamicnuclearnet +0.1 %,
deepbacs −30.1 %) and worsens humanneurons. APG beats AIS on every dataset except dynamicnuclearnet, as
under v2. Note that deepbacs APG gained +28.5 % on its validation subset (section 14.1 vs the v2 control)
but is flat on the test split.
