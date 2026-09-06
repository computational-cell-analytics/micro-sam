# AIS optimization for the joint/v4 geodesic `hvit_t` model

Decision log of the AIS (decoder-based automatic instance segmentation) optimization campaign started
2026-09-06 on branch `ais-v4-optim` (forked from `apg-clean-up` at `4a3ef31`). Set-up, data, gates and
cluster mechanics: `EXPERIMENTAL_SETUP.md`; the plan: `~/.claude/plans/please-plan-a-campagin-cozy-willow.md`.
Paths are relative to `finetuning/v2/evaluation/`; `<root>` is
`/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization`.

## Why

The v4 decoder predicts the geodesic hybrid field (`micro_sam/v2/transforms/labels.py`,
`GeodesicHybridDistanceTransform`): the direction of every pixel's vector is the gradient of the geodesic
distance from the object's centre, so `-d` converges to one sink per object; the magnitude is the
per-object normalised distance to the object's own boundary. The v2 decoder predicted the Euclidean
vector to the nearest boundary, whose negation converges onto the medial axis. The AIS post-processing
(`micro_sam/v2/postprocessing.py`, `flow_instance_segmentation`) and its `hvit_t` defaults (fg 0.5,
density 10, min_size 100, sigma 0.5, n_iter 50, dt 0.5, fg_weight 0.5) were derived for the v2 field:
a fixed 25 px travel, an absolute density threshold, a height map from the inverted magnitude.

Scope decided with the user on 2026-09-06: sparse (flow) pipeline first, dense (multicut) afterwards;
the deliverable is new library logic and new `hvit_t` defaults in `postprocessing.py`; numpy prototypes
for primitives bioimage-cpp lacks, C++ port before the library switch if such a variant wins. `hvit_t`
only, no learned components, no per-dataset modes.

## Harness (Phase 0, 2026-09-06)

`optimization/benchmark_ais_optimization.py` predicts every manifest sample once and caches the
`(4, *spatial)` float32 prediction with its labels under
`<root>/ais/predictions/<checkpoint id>/<manifest checksum>/<sample id>.npz` (`predict`); every
configuration (`run`, `screen`), the parameter grid (`sweep`) and the diagnostics then run on the cache
on CPU. Run directories follow the APG layout, `<root>/ais/hvit_t/<checkpoint id>/<manifest ck>-<params
ck>-<implementation ck>/` with `samples.csv`, `summary.csv`, `metadata.json`, so
`compare_apg_optimization.py` reads them. The implementation checksum covers the benchmark, `common.py`,
`parameter_search.py`, `micro_sam/v2/instance_segmentation.py` and `micro_sam/v2/postprocessing.py`.

Per-sample columns beyond the metrics: `matched` / `unmatched` / `severed_objects` / `genuine_misses`
(the `benchmark_apg_3d.object_counts` definitions, computed from one contingency table), and the seed
diagnostics of a pipeline mirrored step by step (`sparse_pipeline`): `n_seeds`, `gt_with_0_seeds`,
`gt_with_1_seed`, `gt_with_2plus_seeds`, `background_seeds` (majority pixel in the background),
`seeded_unmatched` (seeded, lost in the watershed), `matched_before_min_size`, `fg_iou` and
`pipeline_mismatch` (mirrored segmentation differs from the library's; the bit-identity check of an epoch).
`report` joins the subsets of a screen and applies the generalization gate (up on all but two datasets,
no dataset below both −2 % and −0.005, balanced gain ≥ +2 %). Configuration files:
`configs/ais_*.json` (`{"name", "mode", "params_2d", "params_3d"}`; a flat dict is sparse overrides,
`{"sparse": ..., "dense": ...}` sets both). Task builder: `optimization/ais_campaign_tasks.py`
(`predict`, `screen`, `sweep`). Unit tests: `test/test_ais_optimization.py` (15 tests).

Smoke test (deepbacs, 30 primary images, library defaults, session A100): balanced mSA 0.1604; of 892
ground-truth objects 607 matched, 33 without a seed, **268 with two or more seeds**, 248 background seeds;
mirrored pipeline identical on all 30 images. The v4 field over-seeds the rods: a first sign that the
default travel (25 px) and the absolute density threshold do not fit a centre-directed field.

## Log

- 2026-09-06 19:30: harness written, unit tests green, smoke test passed. Prediction caching of the
  v5 subsets (primary, training_extra, holdout) and the deep 3d crops (apg3d primary, holdout) started
  on the session GPU.
- 2026-09-06 20:15: harness frozen (AIS epoch `f57b117edfda5420d9df761b1db4db2d`, commit of this state). The oracle markers were
  changed from one pixel to the 3-neighbourhood inside the object after the first oracle run: the geodesic
  magnitude is zero at an object's centre pixel (gradient of a field at its source), so the inverted
  magnitude height map has a one-pixel spike there and the monotone flooding of `bioimage_cpp`'s watershed
  floods a seed on a spike last (one pixel left to the object). Predicted seeds are multi-pixel blobs, so
  the pipeline itself is unaffected, but any seed logic that places small seeds at the magnitude peak must
  keep this in mind. Prediction caches: v5 primary 245, training_extra 157, holdout 238 samples (float32,
  labels included); apg3d primary / holdout in progress.
