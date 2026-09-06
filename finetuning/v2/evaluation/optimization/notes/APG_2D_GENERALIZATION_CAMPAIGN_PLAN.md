# APG 2d: campaign plan for generalizing improvements over the defaults

> **Status on branch `apg-clean-up` (2026-09).** This note is the historical record of experiments whose
> mechanisms were tested, refuted and removed from the library and the evaluation harness on this branch.
> The complete state that produced these numbers (library hooks, scripts, configs, artifact loaders, tests)
> is preserved unchanged on branch `apg-optim-fable` (commit `356b76d`, on origin). What remains here is the
> generic harness (`benchmark_apg_optimization.py`, `benchmark_apg_3d.py`, `apg3d_manifest.py`,
> `compare_apg_optimization.py`, `submit_optimization_jobs.py`, `apg_campaign_tasks.py`) and the plain
> refinement round (`generate(refinement=..., refinement_kwargs=...)`); the reproducible set-up is in
> `EXPERIMENTAL_SETUP.md`. Removed items named below are listed under "Status on this branch" at the end
> of this note. This is the plan of the structural campaign (fusion, arbitration, recall, calibration); every
> experiment in it was run and closed negative, see the "Generalization campaign of 2026-09-03/04" section
> of `APG_2D_OPTIMIZATION.md`.

Written 2026-09-03 at the close of the generalization campaign; to be executed in a fresh session. Background
and evidence: `APG_2D_OPTIMIZATION.md` (dated sections of 2026-09-03), `FURTHER_APG_OPTIM.md` ("Session of
2026-09-03"), operations in `CAMPAIGN_OPERATIONS.md`.

## 1. Goal and rule

Find changes to the 2d Automatic Prompt Generator (hvit_t, joint/v2 `best`, confirmed once on v4 geodesic)
that improve on the per-model registry defaults **consistently across datasets**: every dataset up, or most
datasets up with regressions that are minor (working definition: no dataset below −2% relative, or below
−0.005 absolute where the baseline is near zero, on both the development corpus and the production splits).
Learned rankers, per-dataset fitting, rank-based per-image thresholds and global-scalar re-tuning are
excluded: the previous campaign showed every one of them is either non-transferable or a wash.

## 2. What the previous campaign established (constraints on this one)

- SAM2's own predicted IoU is the best dataset-independent ranking of its three masks; head choice carries
  the in-domain gain of any selector and does not transfer; learned filters fail catastrophically on
  appearance shifts (DIC HepG2 −84% out of domain).
- The optimum of every global scalar (candidate threshold, filter threshold, merge overlap, size floor) is
  dataset-dependent; the registry defaults are near the best single compromise (proposal-side winner: +3.5%
  on validation splits, +0.6% on test splits, ±10% per dataset in both directions).
- Recall is the largest loss: 9-19% of objects are never seeded; the candidate ladder is exhausted; the
  merge decides 5-10% of objects; refinement is worth +0.75% (15% signed gate) and is tuned out.
- The decoder's own prediction (foreground, center and boundary distances) and the AIS instance
  segmentation are computed by APG already and are unused after seeding. Agreement between independent
  segmentations of the same object is the one label-free signal not yet exploited.

## 3. Protocol (fixed before any experiment)

**Development corpus.** The eleven datasets with legal validation splits: primary five (240 images:
livecell, tissuenet, dynamicnuclearnet, deepbacs, dic_hepg2) and `training_extra` six (157 images: yeaz,
neurips_cellseg, puma, tnbc, covid_if, deepseas). Manifests `subset_manifest_v5.json`,
`subset_manifest_v5_training_extra.json`; holdout `subset_manifest_v5_holdout.json` (233 images) for the
canonical timing trials.

**Baselines.** Registry defaults (`configs/apg_control_registry_defaults.json`) on all three manifests at the
current implementation checksum (exists for primary/training_extra/holdout, trial `plain-1`, and three
bracketed holdout timing trials). Re-run after every checksum epoch.

**Metric and reading.** Dataset-balanced mSA plus per-dataset deltas against the registry defaults; a
change is a candidate only if it is up on ≥ 9 of 11 datasets with no dataset below the minor-regression
line, and the balanced gain is ≥ +2%. Report precision/recall-style object counts (seeded / proposed /
scored / merged, `predicted_objects` vs `gt_objects`) next to mSA so over-segmentation and misses are
visible. Nothing is learned in this campaign, so OOF/LODO are not needed; the protection against
overfitting the corpus is that every knob a change introduces is fixed a priori or by a label-free rule,
and that the production splits are opened exactly once for the final shortlist.

**Production check.** `evaluate_apg_generalization.py` (register each candidate in `CONFIGS`; add
`--apg_params`/artifacts as needed) on all 23 datasets; the twelve outside the development corpus
(arvidsson, bitdepth_nucseg, cellbindb, cellpose_data, cvz_fluo, dsb, hpa, microbeseg, omnipose, segpc,
usiigaci, vicar) decide; ≤ 3 candidates opened.

**Runtime.** Three serialized bracketed timing trials on the holdout on one 1g.20gb (`apg_campaign_tasks.py
benchmark --serialize --bracket`), comparator `--target quality|efficiency` for the record; runtime cap
+15% per dataset unless the quality gain is ≥ +5% on every dataset (the existing exception).

**Hardware.** Screens and production arrays on `grete:preemptible` 1g.10gb (`2d` preset); timing trials
on the session 1g.20gb; the second `grete:interactive` slot for a second serialized lane.

**Hygiene.** Library hooks default-off and bit-identical when off, batched into one checksum epoch before
the round's runs, unit-tested in `test/test_v2_automatic_prompt_generation.py`; never edit a checksum file
while arrays run; every config JSON pins all proposal parameters; results and decisions logged per round
in `APG_2D_OPTIMIZATION.md`.

## 4. Experiments, in priority order

### P0. Headroom oracles (CPU, first hour)

From existing production and validation results, compute per dataset: max(AIS, APG) − APG (fusion
ceiling), and the recall ceiling from the E2 recall diagnostic (objects never seeded). Both decide how much
of P1 and P3 is worth building; if the fusion ceiling is < +3% balanced, P1 is demoted below P2.

### P1. AIS/APG fusion per object (the main bet)

*Hypothesis.* SAM2 masks win on separated, well-contrasted objects; the decoder's watershed instances win on
dense, touching, low-contrast data (tnbc, deepseas, puma, cellbindb), where APG loses. Choosing per object by
agreement is label-free and adds information APG discards today.

*Design.* After the APG merge, for every AIS instance: (a) if it overlaps an accepted SAM2 mask with IoU ≥ a
(a = 0.5, fixed), keep the SAM2 mask; (b) if it overlaps only rejected or no SAM2 masks and its size ≥
`min_size`, add the AIS instance ("recall fallback"); (c) where a SAM2 mask covers ≥ 2 AIS instances each of
substantial size (split-merge conflict), resolve by SAM2's stability: keep the mask if stability ≥ s (s = 0.9
fixed), else the AIS instances. Variants: fallback only (b); conflict only (c); both. No dataset-tuned knob:
a and s are fixed and reported; a sensitivity check at a ∈ {0.4, 0.6}, s ∈ {0.85, 0.95} is reported, not
optimized.

*Implementation.* Opt-in in `AutomaticPromptGenerator.select()` (`fusion=None|"fallback"|"conflict"|"both"`),
AIS instances from the existing decoder prediction via the AIS path already in the library; harness config
keys; unit tests with synthetic masks (fallback adds, conflict resolves, off is bit-identical). One
checksum epoch shared with P2/P3 hooks.

*Cost.* One `select()`-level variant; a `propose()` cache makes the screen CPU-cheap after one GPU pass per
manifest (~15 min on 1g.10gb per manifest). Production array 23 tasks (~2 h wall).

*Gate.* Protocol gate on the eleven datasets; then production once.

### P2. Decoder-arbitrated merge (replace the overlap threshold)

*Hypothesis.* The score-ordered greedy merge drops a lower-scored mask when its overlap exceeds
`max_overlap`; the threshold's optimum is dataset-dependent (0.15 vs 0.3 moved deepseas and dic_hepg2 in
opposite directions). Assigning contested pixels by the decoder's center distance (nearest seed in
distance-transform terms) and keeping both objects removes the threshold.

*Design.* `merge_by_score(..., arbitration="drop"|"split")`: in split mode, overlapping masks both survive;
contested pixels go to the mask whose seed is closer along the decoder's distance prediction (fall back to
Euclidean seed distance); a mask that loses > 50% of its area to arbitration is dropped as before.
Variants: split with `max_overlap` 0.3 / 0.5 / 1.0 (1.0 = pure arbitration).

*Implementation.* `micro_sam/v2/postprocessing.py` (checksum file) opt-in; tests on synthetic overlaps.
Screens are CPU replays of cached proposals.

*Gate.* Protocol gate; report over-segmentation counts explicitly (this is where deepseas lost under E2).

### P3. Seeding recall (largest ceiling, highest risk)

*3a. Box prompts from decoder components.* Replace or complement the center point with the component's
bounding box (from the decoder foreground / distance components) as the first-pass prompt. Untested so far
(only blanket box *refinement* was, and rejected on cost). Same number of SAM2 decoder calls. Variants:
box only; point+box; box only for components with occupancy < 0.5 (elongated). Fixed rule, no tuning.

*3b. Foreground-residual prompting.* After the merge, connected components of predicted foreground not
covered by accepted masks (area ≥ `min_size`) get one point prompt each (second `propose` pass, same
filter); the earlier "recovery" variant that was rejected in-domain was not foreground-driven and is
re-measured under this protocol only if 3a does not close the recall gap.

*Implementation.* `propose(prompt_type="point"|"box"|"point_box")` and a `recover_residual=True` opt-in in
`select()`; both in `automatic_prompt_generation.py`; tests. Cost ≈ one extra GPU screen per manifest per
variant (~15 min); 3b adds ≈ 30% runtime, so it must clear the runtime cap or the all-datasets exception.

*Gate.* Protocol gate; seeded/merged object counts must show the recall came from previously unseeded objects.

### P4. Label-free per-image filter calibration against the decoder (weak prior, cheap)

Choose the predicted-IoU filter threshold per image from a fixed grid {0.4, 0.5, 0.6, 0.7} by maximizing the
agreement between accepted masks and predicted foreground (Dice of their union with the foreground map
above 0.5), not by rank. CPU replay of cached proposals; gate as above. If it fails (likely), it closes the
threshold-adaptation line for good.

### P5. Checkpoint check

Winners of P1-P4 (or, if none, the registry defaults vs the campaign defaults) re-run on joint/v4 geodesic
(`MICRO_SAM2_JOINT_CHECKPOINT_ROOT` symlink recipe in the plan file / `CAMPAIGN_OPERATIONS.md`): decision
stands only if the sign agrees.

## 5. Sequencing and budget

| step | work | wall |
|---|---|---|
| day 1 morning | P0 oracles; library epoch with P1/P2/P3a hooks + tests; registry controls re-run | 3 h |
| day 1 afternoon | proposal cache per manifest (GPU); P1 and P2 CPU screens; P3a GPU screen | 4 h |
| day 2 | read; P3b / P4 only if warranted; production array for ≤ 3 candidates; holdout timing trials | 1 day |
| day 3 | v4 sign check; notes; decision | half day |

GPU budget ≈ 10-15 GPU-h on 1g.10gb plus the session slice. Decision log: one dated subsection per step in
`APG_2D_OPTIMIZATION.md`.

## 6. Deliverables

Per candidate: eleven-dataset table (mSA, object counts, per-dataset delta), production table over 23
datasets with the twelve strictly unseen marked, three bracketed timing trials and comparator files, v4
sign, and a one-paragraph verdict under the rule. Library changes stay opt-in; no default changes, no
bundled artifacts; nothing committed unless asked.

## 7. What not to do

No learned scorer of any kind; no per-dataset or per-image rank thresholds; no re-tuning of global scalars
as a result (controls only); no dataset-keyed modes; no opening of production splits before the shortlist.


## 8. Execution status (2026-09-03, session 3)

Executed in one session; decision log in `APG_2D_OPTIMIZATION.md`, "Generalization campaign of 2026-09-03/04", operations
in `CAMPAIGN_OPERATIONS.md`, "Session 3". Outcome: P0 done (production and development oracles); P1, P2, P3a, P3b, P4
built (epoch 4, `41abe8ca…`), screened on the eleven datasets and the holdout, and re-screened on joint/v4 geodesic: none
passes the gate (best: arbitration, a wash; fusion −1 to −4%; box prompts −5 to −10%; the adaptive threshold degenerates
to a fixed 0.4). No production run, no timing trials. Side result: joint/v4 geodesic with the registry defaults is
+9-10% over v2 on the primary and holdout manifests; the v4 decoders needed a `UniSAM2` width fix to load.

## Status on this branch

- Removed (all on `apg-optim-fable`): `evaluate_apg_generalization.py`, `screen_apg_structural.py`, the
  library hooks `fusion`, `arbitration`, `prompt_type`, `recover_residual` and their helpers
  (`fuse_with_instances`, `decoder_basins`, `residual_point_prompts`), and the `configs/apg_s_*.json`
  variants.
- The 9-of-11 generalization gate of section 3 is documented in `EXPERIMENTAL_SETUP.md`, section 9; its
  implementation (`gate_table`) went with `screen_apg_structural.py`.
- `configs/apg_control_registry_defaults.json` (the control of section 3) is kept.
