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

## Phase 1 (2026-09-06 evening): baseline, travel ladder (D1) and oracles (D3)

All on the cached joint/v4 geodesic predictions (checksum `5a729846…`), library defaults, AIS epoch
`f57b117edfda5420d9df761b1db4db2d` (the epoch of the frozen Phase 0 harness; run directories under
`<root>/ais/hvit_t/5a729846…/`). Per-dataset mSA of the defaults:

| subset | balanced | livecell | tissuenet | dynamicnuclearnet | deepbacs | dic_hepg2 | volumes (n = 1 each) |
|---|---:|---:|---:|---:|---:|---:|---|
| primary (245) | 0.1841 | 0.2683 | 0.2102 | 0.5422 | 0.1604 | 0.0019 | celegans 0.131, embedseg 0.165, gonuclear 0.340, cremi CREMI 1.057, snemi CREMI 1.054 |
| holdout (238) | 0.1826 | 0.2726 | 0.2112 | 0.5223 | 0.1604 | 0.0021 | (same volumes) |
| training_extra (157) | 0.4183 | yeaz 0.6128, neurips_cellseg 0.2168, deepseas 0.1016, puma 0.4668, covid_if 0.7411, tnbc 0.3705 | | | | | |

For comparison, APG defaults on the same manifests: primary 0.2955, holdout 0.2896, training_extra 0.4634
(EXPERIMENTAL_SETUP.md §14.1). The dense multicut on the 12-slice cremi / snemi crops over-segments
massively (1831 and 3005 instances for 134 and 96 objects); Phase 5 material.

Object fates of the defaults (primary + training_extra, 2d): livecell 17389 objects, 8323 matched, 2735
without a seed, 2014 with two or more seeds, **6338 seeded but unmatched**; tissuenet 4011 / 2115 / 512 /
490 / 1384; deepbacs 892 / 607 / 33 / 268 / 252 (plus 248 background seeds); neurips_cellseg 5766 / 2141 /
1424 / 421 / 2204; deepseas 250 objects but 504 background seeds; yeaz 450 of 2689 objects split;
dic_hepg2 foreground IoU 0.09 (the foreground channel fails on DIC, nothing to post-process). The
"seeded but unmatched" category dominates everywhere; the refined decomposition (split / merged /
undersized / oversized, absorbed / missing) was added to the harness afterwards.

### D1: travel ladder (`configs/ais_s0_travel_*.json`, cluster job 15766947, report `ais/reports/s0_travel_ladder_dev.csv`)

Relative change of mSA against the defaults (travel 25 px) on the development manifests:

| travel (px) | balanced (16 datasets) | livecell | tissuenet | dynamicnuclearnet | deepbacs | yeaz | neurips_cellseg | deepseas | puma | tnbc | 0-seed Δ | 2+-seed Δ | bg-seed Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.5 | +8.5 % | −6.5 | −5.7 | +0.3 | +1.8 | −3.0 | −0.7 | −3.9 | +0.2 | −3.5 | +2705 | −2437 | −1799 |
| 50 | −2.5 % | +0.4 | −0.7 | 0.0 | −3.9 | −1.5 | +1.4 | −7.8 | −0.3 | −1.1 | −455 | +808 | +987 |
| 100 | −1.7 % | +0.2 | −0.9 | −0.2 | +5.5 | −0.9 | +1.8 | −6.5 | −0.5 | −2.2 | −131 | +438 | +1419 |
| 200 | +5.6 % | +0.3 | −0.9 | −0.2 | +10.6 | −0.6 | +3.1 | −4.6 | −0.2 | −1.7 | +202 | −28 | +1398 |
| 400 | +7.2 % | +0.3 | −0.9 | −0.2 | +14.6 | −0.7 | +3.1 | −1.6 | −0.2 | −1.7 | +311 | −192 | +1324 |

The balanced figures are inflated by the single 12-slice embedseg volume (+81-130 %) and the near-zero
dic_hepg2; on the images the travel moves deepbacs (+14.6 % at 400 px) and neurips_cellseg (+3 %) and
costs everything else a little. Longer travel trades splits for background seeds (+1324 at 400 px) and
does not seed more objects. **Verdict: the travel is a secondary knob; run to convergence only together
with a seed rule that suppresses the background sinks.** No candidate passes the gate.

### D3: oracles (`oracle`, primary / training_extra / holdout, defaults; `<root>/ais/oracles/`)

mSA when one part of the pipeline is replaced by the ground truth (predicted parts otherwise), primary
+ training_extra images:

| dataset | baseline | GT seeds | GT height map | GT seeds + GT height map | GT foreground | GT seeds + GT foreground |
|---|---:|---:|---:|---:|---:|---:|
| livecell | 0.268 | 0.319 (+19 %) | 0.539 (+101 %) | 0.612 | 0.422 (+57 %) | 0.504 |
| tissuenet | 0.210 | 0.226 (+7 %) | 0.351 (+67 %) | 0.357 | 0.345 (+64 %) | 0.394 |
| dynamicnuclearnet | 0.542 | 0.568 (+5 %) | 0.556 (+3 %) | 0.575 | 0.972 (+79 %) | 0.988 |
| deepbacs | 0.160 | 0.266 (+66 %) | 0.407 (+154 %) | 0.488 | 0.621 (+287 %) | 0.921 |
| yeaz | 0.613 | 0.628 (+2 %) | 0.687 (+12 %) | 0.702 | 0.885 (+44 %) | 0.910 |
| neurips_cellseg | 0.217 | 0.332 (+53 %) | 0.317 (+46 %) | 0.431 | 0.602 (+178 %) | 0.738 |
| deepseas | 0.102 | 0.184 (+81 %) | 0.168 (+65 %) | 0.261 | 0.806 | 0.923 |
| puma | 0.467 | 0.507 (+8 %) | 0.500 (+7 %) | 0.520 | 0.883 (+89 %) | 0.928 |
| tnbc | 0.371 | 0.412 (+11 %) | 0.395 (+7 %) | 0.415 | 0.849 | 0.964 |
| covid_if | 0.741 | 0.755 (+2 %) | 0.783 (+6 %) | 0.791 | 0.864 | 0.885 |

Reading: (1) on the touching-cell data (livecell, tissuenet, deepbacs) a perfect ridge map with the
predicted seeds and foreground doubles the score, so the assignment step (height map / watershed) is the
largest lever that post-processing controls; (2) perfect seeds add +5-20 % on most datasets and +50-80 %
where background seeds and misses are frequent (deepbacs, deepseas, neurips_cellseg); (3) the ground-truth
foreground ceiling is the largest everywhere, but it leaks the instance separation wherever objects do not
touch (dynamicnuclearnet, puma, tnbc, yeaz: nuclei), so it mixes foreground extent with separation. The
part of it that is extent (mSA on small nuclei swings on one boundary pixel) is reachable only through
the foreground threshold and the instance extent rule, which the sweep and the height-map work cover.
Holdout reproduces the primary picture (livecell 0.273 → 0.557 with GT ridges, tissuenet 0.211 → 0.351).

Priorities for Phase 2/3 from D1-D3: height-map ridge terms (H1 divergence, H2 direction discontinuity)
and trajectory assignment (A1) first, seed rules that suppress background sinks and merge multi-sink
objects second (S1, S2, S5), travel to convergence as a parameter of both. The seed-variant prototype
(`scratchpad/proto_seeds.py`, cluster job) screens all of these on 8 images per dataset before any
library edit.

### Probe of the predicted field (four primary images per dataset, 2026-09-06 19:25)

| dataset | \|d\| background p50 / p90 | \|d\| foreground p10 / p50 | IoU(fg > 0.5, GT fg) | area(fg > 0.5) / area(GT) | \|d\| at the object centre / 5×5 ring |
|---|---|---|---:|---:|---:|
| livecell | 1.04 / 1.13 | 0.24 / 0.48 | 0.87 | 1.13 | 1.00 |
| tissuenet | 1.02 / 1.13 | 0.18 / 0.48 | 0.58 | 1.17 | 1.01 |
| dynamicnuclearnet | 1.01 / 1.11 | 0.17 / 0.57 | 0.67 | 1.48 | 0.93 |
| deepbacs | 1.02 / 1.11 | 0.22 / 0.53 | 0.48 | 3.49 | 1.00 |

Three consequences. (1) The decoder predicts the label transform's fill value (magnitude ≈ 1) in the
background although the distance loss is masked there, so the magnitude cannot serve as a foreground cue,
and along a ray from an object's centre the magnitude runs 1 → 0 (boundary) → 1 (background): the
boundary is the magnitude *minimum*, which the inverted-magnitude height map already turns into a ridge.
(2) The thresholded foreground is systematically too large, by half on the nuclei and 3.5× on the thin
deepbacs rods; the seeded watershed then floods every instance out to the foreground edge, which is why
the ground-truth-foreground oracle is so far above everything else. The instance extent is therefore a
first-order problem: either a higher foreground threshold (the sweep must go beyond 0.7) or, scale-free,
an extent defined by the flow (pixels whose trajectory reaches the instance's sink, no refill; the halo
pixels carry the background direction and do not converge). (3) The one-pixel magnitude dip at the
centre of the training target is not reproduced by the network (ratio ≈ 1.0), so seeds placed at the
magnitude maximum are safe in practice; the oracle-marker precaution stays.

### D2: fate of every ground-truth object under the defaults (primary + training_extra images, epoch `5700c6e0…`)

Percent of ground-truth objects. "matched" at IoU 0.5; "seed0" = no seed component inside; "absorbed" /
"missing" = unseeded objects mostly covered by a neighbour's instance / by nothing; "seeded lost" =
seeded but unmatched, decomposed into "split" (two or more seeds), "merged" (the object's instance covers
at least half of another object too), "under" / "over" (extent errors); "bg seeds" = seed components
whose majority pixel is background, as percent of the object count; "iou" = mean IoU of the matched
objects; "pred/gt" = predicted over ground-truth instance count.

| dataset | gt | matched | iou | seed0 | absorbed | missing | seeded lost | split | merged | under | over | bg seeds | fg IoU | pred/gt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| livecell | 17389 | 47.9 | 0.75 | 15.7 | 11.2 | 4.4 | 36.4 | 6.7 | **24.6** | 1.1 | 4.0 | 8.3 | 0.82 | 0.71 |
| tissuenet | 4011 | 52.7 | 0.71 | 12.8 | 4.7 | 8.1 | 34.5 | 5.3 | **20.4** | 3.0 | 5.8 | 2.3 | 0.71 | 0.67 |
| neurips_cellseg | 5766 | 37.1 | 0.74 | 24.7 | 10.2 | 14.4 | 38.2 | 3.4 | **16.8** | 9.7 | 8.4 | 24.3 | 0.59 | 0.74 |
| deepbacs | 892 | 68.0 | 0.67 | 3.7 | 3.5 | 0.2 | 28.3 | **14.5** | 9.8 | 0.3 | 3.7 | **27.8** | 0.64 | 1.06 |
| deepseas | 250 | 63.6 | 0.67 | 11.6 | 7.6 | 3.6 | 25.2 | 6.0 | 10.4 | 4.0 | 4.8 | **201.6** | 0.45 | 1.72 |
| dynamicnuclearnet | 2592 | 94.0 | 0.80 | 1.1 | 0.3 | 0.8 | 4.9 | 0.3 | 0.9 | 1.5 | 2.2 | 10.4 | 0.75 | 1.03 |
| puma | 3293 | 88.1 | 0.79 | 4.1 | 1.0 | 3.1 | 7.8 | 1.0 | 1.7 | 0.6 | 4.5 | 10.8 | 0.77 | 0.98 |
| tnbc | 399 | 83.7 | 0.77 | 3.5 | 0.8 | 2.8 | 12.8 | 1.8 | 2.3 | 1.3 | 7.5 | 20.8 | 0.66 | 1.00 |
| yeaz | 2689 | 86.9 | 0.84 | 6.6 | 2.3 | 4.3 | 6.5 | 2.9 | 1.9 | 0.8 | 0.9 | 3.4 | 0.87 | 0.92 |
| covid_if | 481 | 91.3 | 0.90 | 5.0 | 2.1 | 2.9 | 3.7 | 0.2 | 1.2 | 0.0 | 2.3 | 0.8 | 0.92 | 0.93 |
| dic_hepg2 | 490 | 1.8 | 0.63 | 32.9 | 0.6 | 32.2 | 65.3 | 47.1 | 0.0 | 13.7 | 4.5 | 105.3 | 0.09 | 1.57 |

The size filter alone (`min_size` 100, refilled by the neighbours) costs tissuenet 302 of 2417 matches
(7.5 %), neurips_cellseg 110, livecell 94, puma 86 (`matched_before_min_size` column); the ground-truth
size floors are 10-50 px, so a shared default has to be lower.

Mechanisms to address, in order of the objects they cost:

- **M1 merges** (livecell 36 % merged + absorbed, tissuenet 25 %, neurips_cellseg 27 %): mostly two seeded
  objects whose basins are not separated, i.e. the ridge of the height map is too weak or misplaced
  (the D3 height-map oracle with the same predicted seeds doubles livecell). With `foreground_weight` 0.5
  the ridge between touching cells is only the magnitude term, 0.5·(1 − |d|), a quarter above the
  interior, and any gap in the magnitude dip along the contact line lets one basin flood the other. The
  direction of the field flips across the contact line whatever the magnitude does, so a direction
  discontinuity ridge (H2) or the trajectory assignment (A1) attacks this directly.
- **M2 unseeded objects** (neurips_cellseg 25 %, livecell 16 %, tissuenet 13 %): the absolute density
  threshold (10) is unreachable for small objects, whose converged particles number about their area
  divided by the sink footprint. Relative or particle-count seeds (S1, S2) with a lower `min_size`.
- **M3 background seeds** (deepseas 2× the object count, deepbacs 28 %, neurips_cellseg 24 %, tnbc 21 %):
  false-positive foreground regions converge into sinks. The predicted field there is the background
  fill (|d| ≈ 1 throughout, no dip at the region's edge, no converging structure), so an instance-level
  field-consistency filter (magnitude along the instance boundary, or the divergence at the sink) is a
  cheap, label-free way to drop them.
- **M4 extent** (neurips_cellseg 18 % under/over, tnbc 9 %, matched IoU 0.67-0.75 on the cell datasets):
  the foreground over-predicts (see the probe), so the instance boundary sits outside the object. A
  higher foreground threshold or a flow-defined extent (A1 without refill).

### Direction structure at the contact lines (8 images each, livecell / tissuenet)

Merged pairs mostly hold **distinct** seeds (livecell 449 pairs with distinct seeds vs 154 sharing a
seed component; tissuenet 52 vs 17), so the merges are an assignment failure, as the D3 oracle said.
But the predicted direction field does not flip sharply at a contact line: the cosine between the flow
1 px on either side of a contact pixel is +0.90 (median) against +0.97 inside; only at ±3-4 px does it
reach 0 / −0.35 (interior +0.70 / +0.49), and it also reverses around every object centre. The
divergence separates contacts from interiors only weakly (+0.03 vs −0.02). The magnitude dip is the
sharper cue: |d| 0.17 at contacts against 0.34 inside (p50), i.e. a ridge of a quarter of the height
range with gaps. The network smooths the target's discontinuities over a 6-8 px band.

### Prototype 1: seed and assignment variants (`proto_seeds.py`, cluster job 15767065, 8 / 6 images per dataset)

Balanced mSA over the prototype images (primary: deepbacs, dic_hepg2, dynamicnuclearnet, livecell,
tissuenet; extra: covid_if, deepseas, neurips_cellseg, puma, tnbc, yeaz). Baseline (travel 25, defaults)
0.232 / 0.383.

| variant | primary | extra | note |
|---|---:|---:|---|
| foreground threshold 0.7, travel 400 | 0.245 | 0.382 | deepbacs +51 %, tissuenet −6 %, dynamicnuclearnet −2.5 %, tnbc −15 % |
| foreground threshold 0.6, travel 400 | 0.243 | 0.388 | the best on both, small per-dataset losses (tissuenet −3 %) |
| density threshold 50 at travel 400 | 0.240 | 0.387 | dynamicnuclearnet +3 %, livecell −12 % |
| travel 400, defaults otherwise | 0.235 | 0.381 | |
| divergence / direction ridges (H1, H2) | 0.235 / 0.234 | 0.380 / 0.381 | no effect, as the direction analysis predicts |
| relative density seeds (S1, 0.25-0.5 of the local maximum) | 0.205 | 0.353 | livecell −52 % (large cells split) |
| particle-count sinks (S2) | 0.207 | 0.352 | livecell −50 %, deepbacs +17 % |
| trajectory assignment, refilled (A1) | 0.219 | 0.364 | worse everywhere; loose mask (0.3) without refill 0.186 / 0.323: the halo converges too |
| magnitude cores (S3), divergence sinks (S4) | ≤ 0.20 | ≤ 0.36 | worse |

Reading: the structural seed rules over-segment the large livecell cells (their predicted field has
several weak sinks and a jittering centre) while they help the small-object data, so a scale-free seed
rule alone does not generalize; the assignment by trajectories inherits the blurred field and is worse
than the watershed; per-pixel direction ridges are empty. The gains that do generalize on this small
sample are the foreground threshold (0.6-0.7: the over-predicted foreground, mechanism M4) and running the
flow to convergence, both parameters. Next: the height-map prototype (`proto_h.py`, job 15767091:
sharpened magnitude dips, relative magnitude, multi-offset reversal ridges, background-instance filters)
and a case study of the merged pairs.

### Prototype 2: height maps and instance filters (`proto_h.py`, job 15767091; seeds = converged density, threshold 10)

Reference `lib_fw0.5` (the library height map, travel 400): balanced 0.235 primary / 0.381 extra.

| variant | primary | extra | per dataset |
|---|---:|---:|---|
| **boundary-magnitude filter 0.4** (drop instances whose boundary median \|d\| > 0.4) | **0.247** | **0.398** | deepbacs +9 %, dynamicnuclearnet +8 %, deepseas ×4, neurips_cellseg +72 %, tnbc +1 %, nothing down |
| boundary filter 0.6 | 0.244 | 0.387 | same direction, smaller |
| mean-magnitude filter 0.7 | 0.237 | 0.383 | weaker |
| sharpened dips exp(−\|d\|/τ), powers, relative magnitude | 0.233-0.237 | 0.373-0.381 | ±1 %, relative magnitude −5 % on yeaz |
| foreground weight 0 / 0.25 / 0.75 | 0.233 / 0.236 / 0.226 | 0.377 / 0.380 / 0.379 | the current 0.5 is fine |
| multi-offset reversal ridges (k = 2-4) | 0.18-0.19 | 0.26-0.32 | catastrophic: the field also reverses around every centre |

Reading: the shape of the height map is not the lever; the background-instance filter is the first
label-free rule that improves every dataset it touches (mechanism M3). It is scale-free (a real object
has a magnitude dip along its whole boundary, a false foreground region carries the background fill).

### Why the merges happen (seed-quality probe, 6 images per dataset)

At the density peak (the sink) the predicted magnitude is small: |d| 0.04-0.22 for the seeds of matched
objects. The network smears the target's zero at the centre over the whole centre region, so every
proper seed sits on a **peak** of the inverted-magnitude height map. With the monotone flooding of the
watershed a seed's front never drops below the seed's own height, so a seed whose centre dip is deeper
than the contact-line dip to its neighbour (contact |d| 0.17 median) loses the object to the neighbour:
in the merged pairs of livecell the losing seed's own instance is 8.5 px (median) before the size
filter, of tissuenet 1 px. The same property silently suppresses spurious seeds, which is why zeroing
the height under all seeds (halving livecell's merges, +7 % matched) still lowered mSA on every dataset:
the spurious seeds then flood too (deepbacs 0.178 → 0.125). Seed-quality cues that separate proper
seeds from background seeds: the foreground probability at the peak (proper p10 0.8-0.99, background
p50 0.5-0.7), the converged particle count (proper p10 ≥ 35-460, background p50 15-43; but the extra
seeds of split objects sit in between), and the inward flux ratio of the flow on a ring of radius 6
(proper p50 0.8-0.97, background 0.1-0.3 on deepbacs, dynamicnuclearnet, deepseas, livecell; not on the
small tissuenet objects). neurips_cellseg's "background" seeds are confident cells the labels do not
contain (fg 0.94, flux 0.98), out of reach for post-processing.

Prototype 3 (`proto_merge.py`): seed floors (none / zero / ring minimum) × size floor (100 / 25) ×
boundary filter (off / 0.4) × decoder-consistency merge of adjacent instances without a dip on their
shared boundary (off / 0.7 / 0.85), at foreground 0.5 and 0.6.

### Prototype 3: seed floors and the decoder-consistency merge (`proto_merge.py`, job 15767152)

72 variants (foreground 0.5 / 0.6 × floor none / zero / ring × size floor 100 / 25 × boundary filter
off / 0.4 × merge off / 0.7 / 0.85), the same images as before. Top of the tables: primary
`fg0.6 · mono · min_size 25 · filter 0.4` 0.2515 and `fg0.6 · mono · 100 · filter 0.4` 0.2498 (baseline
0.2350); extra `fg0.5 · mono · 100 · filter 0.4` 0.398 (baseline 0.381). Every floor variant lands below
the monotone flooding with the same filter, on both subsets. Livecell (fg 0.5, size floor 100, filter
0.4): mono 0.2639 (599 matched, 880 predicted, 140 merged); ring floor 0.2486 (680 matched, **1259
predicted**, 45 merged); zero floor 0.2593 (657 / 1097 / 73). The floors recover the merged objects and
release just as many extra instances: the seeds the monotone flooding silently suppressed were the extra
sinks inside the large cells. Only tissuenet gains from a floor (+5 % with size floor 25), because its
losses are small objects deleted by the size filter. The merge rule (edge / interior magnitude ratio
0.7 / 0.85) removes some of the extra instances but merges real neighbours as well (livecell mono
0.2639 → 0.2560 at 0.7).

Pair probe (adjacent instances under the zero floor, 8 images): same-object pairs vs different-object
pairs on livecell (230 / 1735): edge-over-interior magnitude ratio p50 0.58 vs 0.35 (p25 0.37 vs p75
0.48 overlap), mid-segment magnitude minimum 0.10 vs 0.07, foreground along the boundary 0.83 vs 0.87,
peak distance 21 vs 35 px, size ratio 0.30 vs 0.56. No cue separates the two populations; a rule that
merges most same-object pairs also merges about a fifth of the real pairs. **M1 is not fixable with
label-free rules on this field**: the decoder blurs the field over 6-8 px, so whether two sinks belong to
one object is not decidable from the prediction; the monotone flooding's implicit arbitration (the seed
with the lower height floods) is as good as any explicit rule tried. The remedy is training-side (a
sharper field, or a target with an explicit contact channel), out of scope here.

### Decision (2026-09-06 20:15): what goes into epoch A1

- **Library (opt-in keyword)**: `boundary_magnitude_max` in `flow_instance_segmentation`, implemented by
  `drop_instances_without_boundary_dip` (median |d| along an instance's boundary above the threshold →
  the instance is a false foreground region). Default None (off) for every backbone; the default path is
  bit-identical. The dense pipeline is untouched (Phase 5).
- **Parameters for the shared-default sweep**: travel to convergence (`n_iter` 800 at `dt` 0.5, the tracer
  stops early), `foreground_threshold` (0.4-0.7; the datasets disagree in sign, so it is a compromise:
  tissuenet under-covers, deepbacs over-covers), `min_size` (25-100), `density_threshold` (5-50),
  `foreground_weight`, `boundary_magnitude_max` (off / 0.4 / 0.6).
- Not adopted: seed floors, decoder-consistency merge, relative or particle-count seeds, trajectory
  assignment, height-map transforms, direction / divergence ridges (all recorded above with numbers).

- 2026-09-06 20:30: **epoch A1 `a65e2eb08c23538f11544860736961a3`** (from `5700c6e0…`): `micro_sam/v2/postprocessing.py` gains
  `drop_instances_without_boundary_dip` and the opt-in keyword `boundary_magnitude_max` (default None in
  every backbone's table), mirrored in the harness (`sparse_pipeline`) and in the cached sweep scorer
  (`parameter_search.score_image_sparse_cached`); tests in `test/test_v2_automatic_segmentation.py`. The
  default path is unchanged; the baselines are rerun under this epoch and checked per sample.

### Look ahead to Phase 5: the dense multicut on the deep EM crops (2026-09-06 21:00)

AIS defaults on the apg3d manifests (epoch `5700c6e0…`): family macro **0.091** primary / 0.109 holdout
(APG: 0.327 / 0.342). The sparse LM families: celegans_atlas 0.10, gonuclear 0.20 (352 of 726 objects
split, **1127 background seeds**), embedseg_platy_ish 0.35, embedseg_platy_nuclei 0.23, embedseg_skull 0.10,
platynereis_nuclei 0.08 (720 background seeds for 127 objects). The dense EM families: cremi CREMI 0.94
with 22004 instances for 840 objects, cremi_seen 0.40 (27698 / 7469), snemi 0.82 (14066 / 526),
humanneurons 1.30 (49939 / 1601).

One cremi and one snemi crop by hand: the slice-wise oversegmentation already produces 15104 / 5566
fragments for 295 / 93 objects (the EM foreground is predicted at 0.71 on average with 29 % of the
neuron voxels below 0.5, so the seeds shatter every cross-section and most fragment boundaries look like
membranes: median edge boundary value 0.74 / 0.81), and the multicut at `beta` 0.5 → 0.95 goes from 7400
to 11456 instances (CREMI 1.02 → 2.08) — in elf's `compute_edge_costs` a **higher beta cuts more**, the
opposite of the `run_multicut` docstring ("higher values favour more merging"), and `EM_GRID` (0.5-0.8)
never enters the merging regime (< 0.5). Both the seeding granularity (fewer, larger fragments; the
boundary filter does not apply, the fragments are not instances) and the beta range are Phase 5 items.

- 2026-09-06 21:05: epoch A1 baselines (`current-defaults`) on v5 primary / training_extra / holdout and
  apg3d primary / holdout are identical per sample to the epoch `5700c6e0…` runs (755 samples: mSA and
  instance counts equal, 0 pipeline mismatches); balanced 0.1841 / 0.4183 / 0.1826, apg3d dataset-balanced
  0.1091 / 0.1221. Filter screens `a1_filter_2d` (job 15767179, 27 tasks) and `a1_filter_3d` (15767180,
  18 tasks) and the sweeps `a1_sweep_primary` / `a1_sweep_extra` (15767181 / 15767182, grid
  `configs/ais_grid_lm_v4.json`, 1728 combinations) submitted at 19:55.

## Epoch A1 screen (2026-09-06 21:10, jobs 15767179 / 15767180, reports `ais/reports/a1_filter_*.csv`)

Relative change of mSA against the defaults; "balanced" over the eleven development datasets (2D) or the
six sparse LM sources of the deep 3D crops (the dense EM sources are untouched by these parameters).

| configuration | 2D dev balanced | up / 11 | worst | 2D holdout (5) | 3D LM primary (6) | 3D LM holdout (6) |
|---|---:|---:|---:|---:|---:|---:|
| filter 0.4, travel 25 (defaults otherwise) | **+1.4 %** | **9** | −0.1 % | +1.0 % | **+4.7 %, 5/6 up, passes** | **+9.7 %, 5/6 up, passes** |
| filter 0.6, travel 25 | +0.9 % | 7 | 0.0 % | +0.5 % | +0.8 % | +4.5 % |
| travel 400 alone | +0.3 % | 3 | −1.7 % (tnbc) | +2.5 % | +22 % (skull +281 %, platy_ish −6 %, platy_nuclei −6 %) | +19 % |
| filter 0.4, travel 400 | +1.7 % | 6 | −0.9 % (tissuenet) | **+3.5 %, 5/5 up, passes** | +27 % (2 sources down) | +28 % |
| filter 0.3, travel 400 | +1.8 % | 6 | −1.1 % | +3.5 % | +32 % | +35 % |
| filter 0.4, travel 400, min_size 25 | −0.3 % | 3 | −6.7 % (tnbc) | +4.5 % (tissuenet +12 %) | +2.6 % | +2.2 % |
| filter 0.4, travel 400, foreground 0.6 | −0.9 % | 5 | −6.5 % (dnn) | +1.0 % | +29 % | +27 % |

Per dataset, filter 0.4 at travel 25 (2D): deepseas +23.6 %, dic_hepg2 +10.6 %, deepbacs +4.4 %,
neurips_cellseg +4.2 %, dynamicnuclearnet +1.4 %, tnbc +0.9 %, puma +0.2 %, covid_if / livecell / yeaz
0.0 %, tissuenet −0.1 %; 3D primary: embedseg_skull +25 %, platynereis_nuclei +16 %, gonuclear +5 %,
celegans_atlas +1.4 %, platy_nuclei +0.4 %, platy_ish 0.0 %; 3D holdout: platy_nuclei +35 %, platynereis
+29 %, skull +8 %, celegans +1 %. Travel 400 with the filter reaches +19 % on deepbacs and +18 % on
deepseas but costs covid_if, tissuenet, tnbc, yeaz 0.5-0.9 % each and, in 3D, the two large EmbedSeg
nuclei sources 5-7 % (the converged sinks split large nuclei).

Reading: the boundary filter is a generalizing improvement (never below −0.1 % on any of the 22 dataset
× subset cells, up wherever background seeds exist) but alone it stays under the +2 % balanced bar in
2D; the travel is the second lever in 3D and on the small-object 2D data and needs a compensating change
where it splits large objects. The shared-default sweep (`configs/ais_grid_lm_v4.json`, 1728
combinations over the eleven 2D datasets; a reduced grid over the six 3D LM sources) decides the
combination.

## Phase 4: shared-default sweep, 2D (2026-09-06 21:40, jobs 15767181 / 15767182, ranking `ais/reports/a1_sweep_dev_ranking.csv`)

Grid `configs/ais_grid_lm_v4.json` (1728 combinations: foreground 0.4-0.7, density 5-50, size floor
25 / 50 / 100, sigma 0.5 / 1.0, travel 25 / 400 px, foreground weight 0.25 / 0.5 / 0.75, filter off / 0.4 /
0.6) scored on every image of the eleven development datasets from the cache (`sweep`, 8-20 s per
image). `report_ais_sweep.py` ranks the combinations as shared defaults against the library defaults.
**7 of 1728 pass the gate**; all seven keep foreground 0.5, density 10, foreground weight 0.5 and use
sigma 1.0.

| combination (changes to the defaults) | balanced | gain | up / 11 | worst | per dataset |
|---|---:|---:|---:|---:|---|
| defaults | 0.3357 | | | | |
| filter 0.4 | 0.3404 | +1.4 % | 9 | −0.1 % | deepseas +24, dic +11, deepbacs +4, neurips +4 |
| sigma 1.0 | 0.3420 | +1.9 % | 8 | −1.3 % (tnbc) | deepbacs +10, deepseas +30, livecell +2.9, neurips +6, tissuenet −1.2 |
| min_size 50 | 0.3333 | −0.7 % | 2 | −13.5 % | tissuenet +9.7, everything else down: the lower floor alone admits the small spurious seeds |
| sigma 1.0 + min_size 50 | 0.3430 | +2.2 % | 9 | −1.1 % | tissuenet +6.4 (the smoothing removes the spurious seeds the lower floor would keep) |
| **C1: sigma 1.0 + min_size 50 + filter 0.4** | **0.3437** | **+2.4 %** | **9** | **−0.8 % (tnbc)** | covid −0.2, deepbacs +12.7, deepseas +31.9, dic +59.6, dnn +0.4, livecell +3.1, neurips +4.4, puma +0.4, tissuenet +6.3, tnbc −0.8, yeaz +0.5 |
| C1 + travel 400 | 0.3437 | +2.4 % | 6 | −0.8 % | deepbacs +17.4, tissuenet +8.8, livecell +4.0, dnn +1.1; deepseas +20, five datasets −0.0 to −0.8 |
| sigma 1.0 + travel 400 + filter 0.4 (size floor 100) | 0.3431 | +2.2 % | 9 | −0.5 % | |

The best by mean ratio to each dataset's optimum (0.878) is foreground 0.4 / filter 0.6, which fails the
gate (6 up); C1 is second (0.877). Reading: the sweep changes the *interpretation* of the seeding rather
than its logic: a wider smoothing of the convergence density (sigma 1.0) merges the jittering sinks of a
large cell into one seed and drops the isolated one-pixel seeds (tissuenet's were 1 px), which is what
the seed floors and the merge rule tried and failed to do structurally; with those gone the size floor can
follow the ground-truth floors (50), and the boundary filter removes the remaining false regions. Travel
to convergence is neutral in 2D (same balanced, more datasets marginally down); the 3D sweep decides it.

Confirmation (job `a1_confirmation`, one task per manifest, trial `timing-1`, control and candidates on
the same node): `configs/ais_c1_sigma1_ms50_filter0p4.json`, `ais_c1_t400.json`, `ais_c3_sigma1_ms50.json`
on v5 primary / training_extra / holdout and apg3d primary / holdout.

## Phase 4: shared-default sweep, 3D LM crops (2026-09-06 22:15, job 15767263, ranking `ais/reports/a1_sweep_3d_primary_ranking.csv`)

Grid `configs/ais_grid_lm3d_v4.json` (576 combinations; size floor 50 / 100 / 200 voxels, foreground
weight fixed at 0.5) on the six sparse LM sources of the 57 primary deep crops. **22 of 576 pass the gate**
(all six sources up in the best of them). Balanced over the six sources, defaults 0.1714:

| combination (changes to the defaults) | balanced | gain | up / 6 | worst | celegans | platy_ish | platy_nuclei | skull | gonuclear | platynereis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 (sigma 1.0, min_size 50, filter 0.4) | 0.1804 | +5.3 % | 3 | −6.1 % | −6.1 | −0.3 | −0.7 | +27 | +8.7 | +43 |
| sigma 1.0, filter 0.4 (min_size 100) | 0.1910 | +11.4 % | 5 | −5.8 % | −5.8 | +2.5 | +0.7 | +56 | +14 | +66 |
| sigma 1.0, filter 0.4, min_size 200 | 0.2030 | +18.4 % | 5 | −4.6 % | −4.6 | +2.9 | +1.0 | +109 | +16 | +95 |
| **sigma 1.0, filter 0.4, min_size 200, foreground 0.6** | **0.2081** | **+21.4 %** | **6** | **+1.4 %** | +8.2 | +1.4 | +2.0 | +108 | +25 | +106 |
| same with foreground 0.7 | 0.2073 | +20.9 % | 5 | −4.3 % | +18 | −4.3 | +0.1 | +100 | +32 | +108 |
| sigma 1.0, filter 0.4, min_size 100, travel 400 | | +29.4 % | 3 | −3.5 % | | | | | | |
| top by balanced: density 20, foreground 0.6, min_size 200, sigma 1.0, filter 0.4 | 0.2644 | +54 % | 3 | −7.3 % | the two EmbedSeg platy sources lose |

Joint view over the 384 combinations both grids share: 4 pass the 2D gate, 11 the 3D gate, **none both**.
The disagreement is the size floor (50 px is right for the 2D nuclei data, 200 voxels for the volumes: a
voxel floor of 50 keeps fragments that no 3D object is) and the foreground threshold (celegans_atlas turns
from −6 % to +8 % between 0.5 and 0.6, while 0.6 costs dynamicnuclearnet, tissuenet and tnbc in 2D). Sigma
1.0, the boundary filter at 0.4, density 10, foreground weight 0.5 and the default travel are shared by the
winners of both dimensions. Travel to convergence does not enter the 3D winners either (it splits the large
EmbedSeg nuclei), so the runtime stays as it is.

Proposal: dimension-aware defaults, as the APG module has for volumes (`default_prompt_generation(...,
is_volume=True)`): images `{sigma 1.0, min_size 50, boundary_magnitude_max 0.4}` on top of the current
table; volumes additionally `{min_size 200, foreground_threshold 0.6}`. The volume part is confirmed on the
3D holdout before anything is promoted (`configs/ais_c1v_volume.json`).

## Confirmation of C1 (2026-09-06 22:30, job 15767364, trial `timing-1`, control and candidates on one node per manifest)

| configuration | 2D dev (11) | up | worst | 2D holdout (5) | up | worst | 3D LM primary (6) | 3D LM holdout (6) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **C1** sigma 1.0, min_size 50, filter 0.4 | **+2.4 %, passes** | 9 | −0.8 % | **+4.3 %, passes** | 5 | +0.9 % | +5.6 % (3 up, celegans −6.1 %) | +1.4 % (2 up, celegans −5.0 %) |
| C1 + travel 400 | +2.4 % | 6 | −0.8 % | +5.7 %, passes | 5 | +0.6 % | +18 % (2 up, platy_ish −7 %) | +21 % (2 up, celegans −7 %) |
| C3 sigma 1.0, min_size 50 (no filter) | +2.2 %, passes | 9 | −1.1 % | +4.1 % | 4 | −3.4 % (dic) | +2.1 % | −2.6 % |

2D holdout per dataset, C1: deepbacs +12.7 %, dic_hepg2 +2.3 %, dynamicnuclearnet +0.9 %, livecell +2.8 %,
tissuenet +8.1 %. Object counts on the development images: +526 matched, −2589 objects with two or more
seeds, −2453 background seeds, −1364 splits (of 38 000 objects). C1 confirms in 2D; as a volume setting it
fails on celegans_atlas, which is what the volume overrides (foreground 0.6, size floor 200) address
(confirmation job `a1_confirmation_volume`, trial `timing-2`).

Post-processing time on the same node: v5 primary 6.0 s (defaults) → 15.3 s (C1) for 240 images, i.e. +0.04 s
per image, all of it the boundary filter (sigma / size floor alone: 12.1 s → the density smoothing is not
the cost; C3 on holdout 5.2 s); apg3d primary 32.8 s → 159 s for 57 crops (+2.2 s per 32-slice crop).

Runtime of the confirmation (2D images, prediction time from the A100 cache records, post-processing on
one cluster node): C1 with the first filter implementation +4 % to +31 % total per dataset (the filter's
`scipy.ndimage.median` per instance cost 0.06 s per image and 2.9 s per 32-slice crop); the same
configuration without the filter (C3) +0 % to +9 % (livecell +9.3 %, the wider density smoothing). The filter
was then rewritten (2026-09-06 22:45): the inner boundary from axis shifts and every instance's median from one
`lexsort` over the boundary pixels (mean of the two middle values for even counts, as `ndimage.median`) —
identical output on 60 images and 3 crops, 18× faster (3 ms per 512² image, 0.12 s per crop), so the
runtime overhead of C1 is that of C3.

`default_postprocessing` gained an `ndim` argument and the table a `sparse_volume` sub-table of volume
overrides (empty until the volume confirmation), which `flow_instance_segmentation` resolves from the
foreground's dimensionality and the harness from `params_2d` / `params_3d`.


## Volume confirmation and promotion (2026-09-06 23:05, job 15767420, trial `timing-2`)

`c1v-volume` (images: sigma 1.0, min_size 50, filter 0.4; volumes: the same plus min_size 200 and
foreground 0.6): 3D LM primary **+22.7 %, 6 / 6 up** (celegans +8.2, platy_ish +1.4, platy_nuclei +2.0,
skull +108, gonuclear +25, platynereis +90); 3D LM holdout **+22.0 %, 5 / 6 up, worst −0.2 %** (celegans
+20, platy_ish −0.2, platy_nuclei +38, skull +98, gonuclear +1.6, platynereis +131); the twelve-slice
volumes of the 2D manifests: celegans +14.5 %, embedseg +39.6 %, gonuclear +18 %; the images are C1
(+2.4 %). Without the foreground change (`c1v-ms200`) holdout is +19.8 % with celegans −1.8 %.

**Epoch A2 `576a85c8ffd4314627812fd30a3c1223`: promoted.** `DEFAULT_POSTPROCESSING["hvit_t"]["sparse"]` = foreground 0.5,
density 10, min_size 50, sigma 1.0, n_iter 50, dt 0.5, foreground weight 0.5, boundary_magnitude_max 0.4;
`["sparse_volume"]` = min_size 200, foreground 0.6. The other backbones keep their registry values and
an empty volume table; the dense pipeline is unchanged. The old values remain reachable as an explicit
configuration (`configs/ais_control_v4_old_defaults.json`, filter off via `Infinity`).

## Phase 6: canonical runs, production and the 3D test manifest (submitted 2026-09-06 23:15)

- Canonical A2 screens (job 15767503 `a2_canonical`, trial `a2-1`): `v4-old-defaults` (explicit old values)
  against `current-defaults` (the promoted library defaults) on v5 primary / training_extra / holdout and
  apg3d primary / holdout, plus `--ndim 2` runs of v5 primary and holdout for
  `compare_apg_optimization.py --target quality`.
- Production (`submit_all_evaluations.py --segmentation_type automatic --segmentation_mode ais
  --all_datasets --modality lm -m hvit_t --skip_tuning`, experiment folder
  `experiments/v4_geodesic_ais_optimization`): jobs 15767555-57 with the new defaults (result tag
  `a2-defaults`) and 15767589-91 with `--ais_params configs/ais_control_v4_old_defaults.json` (tag
  `old-defaults`), 33 LM datasets each (23 2d + 10 3d LM; the dense EM pipeline is unchanged, its §14.3
  numbers stand). Reader: `report_ais_production.py -e <folder> --baseline default_old-defaults --candidate
  default_a2-defaults`, which reports the twelve strictly unseen 2d datasets separately.
- 3D test manifest (`manifest_test_apg3d-v1.json`, 56 crops of the seven test-only LM datasets, opened
  once): predictions cached on the session GPU, then `screen` old vs new defaults (trial `test-1`).
