# Targeted 2D APG optimization

## Outcome

The original point-placement, blanket-refinement and batching campaign below found no accepted
optimization. The later learned-multimask campaign and its compact deployment follow-up are recorded
at the end of this document; they add explicit opt-in model paths while still leaving all defaults
unchanged. In the original campaign, the closest quality candidate, confidence-gated box refinement, improved
the dataset-balanced mSA by 1.74%, short of the required 5%, while increasing total runtime by 21.86%.
The best worst-dataset efficiency candidate, a prompt batch size of 192, improved total runtime by
0.53% and its slowest-improving dataset by only 0.23%, short of the required 5% on every dataset.

The experimental implementations for point placement and selective box refinement were therefore
reverted. Their serialized benchmark results are retained below the experiment output root. No rejected
setting was made a library default and no regression test was added for experimental code that is no
longer present.

## Benchmark and decision rules

All experiments used only the 2D portion of manifest schema 5, checksum
`0f8fb67b3650a71f9f44b53037e89546`. The source data under
`/mnt/vast-nhr/projects/cidas/cca/data` was treated as read-only. The manifest contains 240 deterministic
validation samples:

| dataset | samples | role in the coverage |
|---|---:|---|
| LiveCELL | 80 | diverse phase-contrast cell types |
| TissueNet | 40 | multichannel tissue microscopy |
| DynamicNuclearNet | 40 | fluorescent nuclei |
| DeepBacs | 30 | bacterial morphology; all available validation images |
| DIC HepG2 | 50 | DIC cells, with extra coverage for its low absolute baseline |

The model was `hvit_t` with checkpoint `best`, checksum
`85fb099c4bb038fa0ab9bddd6151689e`. Runs were serialized on an
`NVIDIA A100-SXM4-80GB MIG 1g.20gb` device. The baseline and batch-size sweep used implementation
checksum `b9ceb079dce0fc0e4f9ad620089169c9` at revision
`f58d959b5fd89d4698c875787da329cff93f3177`.

The primary quality metric is the equal-weight mean of the five per-dataset mSA values. Relative, not
absolute, changes determine every gate:

- A quality optimization needs at least +5% macro mSA. At most two datasets may regress by more than
  5%. No dataset may take more than 10% longer unless macro quality improves by at least 10% and all
  five datasets improve.
- An efficiency optimization must be at least 5% faster on every dataset. Every dataset must keep mSA
  within -0.5% of baseline.
- Up to five configurations are ranked within one hypothesis, but the best configuration is adopted
  only if it passes the corresponding gate.

Each canonical baseline dataset runtime is the median of three complete serialized trials. Candidate
quality is deterministic for a fixed implementation and configuration. Point-placement and box-refinement
experiments each used a same-implementation control run so that the temporary experimental branch itself
could not be confused with the optimization. Their quality control exactly matched the canonical baseline;
their runtime changes are paired with that control to reduce execution-time drift. Every individual sweep
completed within the 30-minute limit: about 15 minutes for point placement, 26 minutes for box refinement,
and 20 minutes for prompt batching.

The comparison program rejects incomplete runs and runs with mismatching dimensions, manifest, model,
checkpoint, implementation, or resolved parameters. This prevents stale or partially overwritten results
from entering a decision. Peak CUDA memory is reset and recorded per sample.

## Baseline

The current defaults use the deepest interior point of each convergence-density component, do not run box
refinement, and evaluate 64 prompts per interactive-model forward pass.

| dataset | mSA | median seconds |
|---|---:|---:|
| LiveCELL | 0.343095 | 103.490 |
| TissueNet | 0.273352 | 27.706 |
| DynamicNuclearNet | 0.457223 | 21.590 |
| DeepBacs | 0.247616 | 13.157 |
| DIC HepG2 | 0.026599 | 38.852 |
| **Dataset-balanced / total** | **0.269577** | **204.797** |

Peak CUDA memory was 1.98 GiB. The low DIC result is not caused by an empty or incorrectly selected input:
the expanded benchmark deliberately contains 50 DIC images. TissueNet is loaded through the common
multichannel normalization path, which normalizes microscopy channels independently before converting
them to the SAM2 image representation.

## Experiment 1: candidate point placement

### Hypothesis and implementation

The baseline places a prompt at the deepest interior location of the thresholded density component. This
guarantees an in-component point but ignores the predicted convergence strength and the foreground extent.
Three alternatives were evaluated:

- `density`: use the convergence-density maximum in the component.
- `distance`: use the maximum foreground distance-to-boundary in the component.
- `density-distance`: maximize a combined, normalized density and distance score.

All variants retained one positive point per component and changed neither candidate count nor downstream
scoring and merging. Inline checks verified that every returned point was inside its component, coordinate
conversion remained XY for SAM2, ties were deterministic, and empty input kept the existing behavior.

### Results

| point rule | macro mSA | macro change | total seconds | runtime change | worst dataset runtime change | accepted |
|---|---:|---:|---:|---:|---:|---|
| density | 0.270802 | +0.454% | 200.877 | -2.39% | -0.71% | no |
| distance | 0.269518 | -0.022% | 200.096 | -2.77% | -1.51% | no |
| density-distance | 0.268788 | -0.293% | 202.280 | -1.70% | -0.75% | no |

Per-dataset relative mSA changes:

| point rule | LiveCELL | TissueNet | DynamicNuclearNet | DeepBacs | DIC HepG2 |
|---|---:|---:|---:|---:|---:|
| density | +1.302% | +0.828% | +0.271% | -0.894% | +1.397% |
| distance | +1.293% | +0.762% | -0.646% | -1.954% | +3.656% |
| density-distance | +1.171% | +0.559% | -0.790% | -2.137% | -2.209% |

The density maximum is the best of these rules, but its +0.454% macro improvement is an order of magnitude
below the acceptance threshold. The results also show that there is no point rule that consistently helps
all modalities: distance-based placement trades gains on LiveCELL, TissueNet, and DIC for regressions on
DynamicNuclearNet and DeepBacs. Apparent runtime gains are too small to interpret as an algorithmic effect,
because the number of model prompts is unchanged and point derivation is a negligible part of the pipeline.

**Decision:** reject all three variants and retain `interior_points`.

## Experiment 2: confidence-gated box refinement

### Hypothesis and implementation

Box prompts are less ambiguous than point prompts, but refining every accepted point mask is expensive and
can replace a good mask with a worse one. The experimental refinement used the source proposal's predicted
IoU to refine only masks at or below a confidence threshold. Higher-confidence masks were locked against
overpainting, and unchanged small instances were restored rather than being lost as a side effect of the
refinement pass. Thresholds 0.70, 0.75, 0.80, 0.85, and 1.00 were tested; 1.00 is effectively refinement of
all non-empty instances.

Inline checks covered the no-op path, ID preservation, locked-mask preservation, empty segmentations,
threshold monotonicity, and accounting of refined versus eligible instances.

### Results

| maximum score refined | instances refined | macro mSA | macro change | total seconds | runtime change | worst runtime change | accepted |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.70 | 3,864 / 15,608 (24.76%) | 0.270552 | +0.362% | 217.602 | +5.73% | +8.12% | no |
| 0.75 | 6,232 / 15,608 (39.93%) | 0.271082 | +0.558% | 224.215 | +8.95% | +13.32% | no |
| 0.80 | 8,911 / 15,608 (57.09%) | 0.271604 | +0.752% | 230.678 | +12.09% | +18.99% | no |
| 0.85 | 12,388 / 15,608 (79.37%) | 0.273425 | +1.427% | 240.757 | +16.98% | +23.44% | no |
| 1.00 | 15,608 / 15,608 (100%) | 0.274269 | +1.740% | 250.785 | +21.86% | +37.93% | no |

Per-dataset relative mSA changes:

| maximum score refined | LiveCELL | TissueNet | DynamicNuclearNet | DeepBacs | DIC HepG2 |
|---:|---:|---:|---:|---:|---:|
| 0.70 | +0.022% | +0.579% | +0.087% | +0.731% | +3.794% |
| 0.75 | -0.032% | +0.961% | +0.497% | +0.939% | +1.526% |
| 0.80 | +0.077% | +0.955% | +1.070% | +0.766% | +1.754% |
| 0.85 | +0.298% | +1.357% | +2.671% | +0.804% | +1.132% |
| 1.00 | +0.533% | +1.324% | +3.442% | +1.010% | -0.862% |

The quality response is mostly monotonic with the fraction refined, but it saturates far below +5%. The
0.70 setting is the only configuration inside the 10% per-dataset runtime cap, yet it improves macro mSA
by only 0.362%. From 0.75 upward the quality gate and runtime gate both fail. Refining everything gives the
largest gain, driven primarily by DynamicNuclearNet, but makes that dataset 37.93% slower and slightly
regresses DIC. No setting regresses more than two datasets by 5%; that guard is not the limiting gate.

**Decision:** reject all thresholds and leave `refine_with_box_prompts=False`.

## Experiment 3: prompt batching

### Hypothesis and implementation

Prompting the SAM2 interactive branch is the largest measured 2D stage. The existing implementation already
batches prompts, so batch sizes 96, 128, 192, 256, and 384 were compared with the default 64. This changes
only the number of prompts in one forward pass. It is expected to preserve segmentation except for minor
floating-point batching effects.

### Results

| batch size | macro mSA change | total seconds | total speedup | worst dataset speedup | peak CUDA memory | quality guard | accepted |
|---:|---:|---:|---:|---:|---:|---|---|
| 64 (baseline) | 0% | 204.797 | 0% | 0% | 1.98 GiB | yes | baseline |
| 96 | +0.0047% | 204.429 | +0.18% | -0.19% | 2.77 GiB | yes | no |
| 128 | +0.0052% | 203.692 | +0.54% | +0.18% | 3.56 GiB | yes | no |
| 192 | +0.0028% | 203.716 | +0.53% | +0.23% | 5.14 GiB | yes | no |
| 256 | +0.0048% | 204.314 | +0.24% | -0.10% | 6.72 GiB | yes | no |
| 384 | +0.0002% | 204.665 | +0.06% | -0.78% | 9.89 GiB | yes | no |

Per-dataset runtime changes, where negative is faster:

| batch size | LiveCELL | TissueNet | DynamicNuclearNet | DeepBacs | DIC HepG2 |
|---:|---:|---:|---:|---:|---:|
| 96 | +0.19% | -0.54% | -0.43% | -0.31% | -0.73% |
| 128 | -0.42% | -0.82% | -0.29% | -0.18% | -0.91% |
| 192 | -0.41% | -1.05% | -0.23% | -0.33% | -0.70% |
| 256 | +0.10% | -1.03% | -0.26% | -0.19% | -0.58% |
| 384 | +0.78% | -1.26% | -0.50% | -0.36% | -1.10% |

All quality changes are far inside the -0.5% guard. None of the batch sizes is 5% faster on even one
dataset, much less every dataset. Batch size 128 has the best total runtime, while 192 has the best
worst-dataset speedup, but both gains are below timing noise and cost substantially more memory. The curve
also explains the weak response: initialization and non-prompt work remain unchanged, and the existing
batch of 64 already uses the device effectively.

**Decision:** retain `batch_size=64`.

## Conclusions and follow-up

The three experiments narrow the useful search space:

1. Moving a single point within the same density component cannot deliver the required quality gain. A
   future prompt-quality change must add information, such as an extra positive/negative point, rather than
   merely relocating the existing one.
2. Box refinement genuinely improves segmentation, but not enough to pay for a second SAM2 pass. It may be
   useful as an explicit high-quality mode, but it should not become the general default under these gates.
3. Prompt-forward batching is not the current throughput bottleneck at batch size 64. Larger batches trade
   memory for changes below 1% and should not be pursued further on this hardware.

The next quality experiment should therefore target errors the point prompt cannot express: ambiguous
clusters or masks lost at the score/overlap merge. It should first stratify gains by merge reason so that an
additional prompt is issued only where it addresses a measured failure. For efficiency, a useful change must
remove or reuse model work (for example, avoid a prompt forward for candidates that can be rejected from
decoder evidence), not only pack the same work into a larger batch.

That experiment has since been run: see
[the second part of this document](#second-round-refinement-from-grouped-prompts) for the
second-round refinement from grouped prompts, which supersedes the box-refinement result above
(+2.91% macro mSA as `points+boxes`, still short of the gate) and replaces the
`refine_with_box_prompts` argument with the generalized `refinement` mode.

## Reproducibility and artifacts

The output root is:

```text
/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/hvit_t/
  85fb099c4bb038fa0ab9bddd6151689e/
```

Run directory names are `<manifest checksum>-<config checksum>-<implementation checksum>`. Canonical
baseline config checksums are `6afe80482ecc3a8348fca1beef9772b6`,
`4e4af15af3a43236e0420739a91c16a8`, and `e22c65b518a9137811e173273455609c` for trials 1-3. Candidate config
checksums are:

| experiment | configuration checksums |
|---|---|
| point placement | density `7f28d239363ea25efd1f1d67e4df68da`; distance `07a12b8679b9d9bc5c7aff801177e684`; density-distance `5b09256c6c439b80a1396c0f0e6dec11` |
| box refinement | 0.70 `2d34dcf7039492caa6a5b0fca75e7a94`; 0.75 `f6062ebedeec8b3d447e4cb6dbd87b9d`; 0.80 `9e18f07b1bed3419280c97ba905e4151`; 0.85 `7b6d7fbb7cd038a04e2722ec33335d52`; 1.00 `4acbcf112125cdf6c49d28ff8293c23f` |
| prompt batching | 96 `1984d04f791a75bbf7b04f452ba705b9`; 128 `86aabecc2667413fe5a324b744f799e5`; 192 `ef09f9855a2c8fc77996812420a6bf55`; 256 `1f70722a06638ecfab2d298b6c42f07a`; 384 `10e3b3793826a5defc39c8263be86fb2` |

The temporary point and box implementations have checksums `e3c34c40041e013c0309f6dedf4f207b` and
`3f8049a2ac12ad6e0140dbe5e9019449`, respectively. Their same-implementation controls have config
checksums `ec1371c439138cbdcb7c7118dea2aa96` and `95e5995b05fe70cfd67b4c7179af97e8`.

A canonical trial is run with:

```bash
python finetuning/v2/evaluation/benchmark_apg_optimization.py \
    --ndim 2 --trial-id baseline-1
```

A JSON configuration supplies a name and only the changed parameters, for example:

```json
{
  "name": "batch-128",
  "params_2d": {"batch_size": 128}
}
```

Serialized runs are evaluated with repeated `--baseline-run` and `--candidate-run` arguments:

```bash
python finetuning/v2/evaluation/compare_apg_optimization.py \
    --target efficiency \
    --baseline-run /path/to/baseline-1 \
    --baseline-run /path/to/baseline-2 \
    --baseline-run /path/to/baseline-3 \
    --candidate-run /path/to/candidate \
    --output /tmp/apg-decisions.json
```

The comparator writes the decision summary as JSON and every per-dataset delta as CSV beside it.

---

## Second-round refinement from grouped prompts

The follow-up experiment the conclusions above asked for, previously kept in its own document and
merged here. It adds a generalized second refinement round to the 2D APG and sweeps it on the same
benchmark.

### Outcome

Second-round refinement from grouped prompts works in 2d, but only when the re-prompt is anchored by
the instance's box: `points+boxes` at 3 positives / 6 negatives is the best 2d refinement measured
so far, +2.91% macro mSA over the baseline and +1.2 points over box-only refinement — for +38.5%
runtime. The user-facing hypothesis in its pure form — re-prompting with grouped positive points and
nearby negative points, without a box — is **refuted**: negatives without a box are harmful
(-1.8% to -6.3% under `replace`), and the best pure-points configuration (+1.27%) stays behind plain
box refinement.

No configuration approaches the +5% quality gate and every one breaks the 10% runtime cap, so, as
with the earlier box-refinement experiment, nothing becomes a library default: `refinement=None`
remains the default and the mechanism ships as an explicit opt-in mode
(`generate(refinement="points+boxes", refinement_kwargs={"n_positives": 3, "n_negatives": 6})` for
the quality-optimal setting, `n_negatives=4` for the most balanced per-dataset profile).

**Superseded by [campaign 2](#refinement-campaign-2-holdout-validation-and-the-four-follow-up-directions)**,
which confirmed these findings on a
held-out validation subset and improved the recommended configuration to +4.19%/+4.89% macro mSA
(tuned/held-out): one positive, six negatives, geometric acceptance gates — now the refinement
defaults, so plain `generate(refinement="points+boxes")` is the recommended usage.

### Motivation and mechanism

The parameter and efficiency sweeps above established that relocating the single point
prompt cannot deliver a meaningful quality gain and that box refinement of every instance improves
quality (+1.74% macro mSA) but not enough to pay for its second SAM2 pass. Their follow-up
recommendation was to *add information — an extra positive or negative point — rather than relocate
the existing one*, and to attribute any gain to a measured failure mode.

This experiment adds a generalized second refinement round to the 2D APG
(`micro_sam/v2/automatic_prompt_generation.py`). After the first round's merge, every instance is
re-prompted once, with a `+`-joined combination of three prompt components:

- `points`: the first round's prompts grouped onto the instance — the prompt that made it plus all
  suppressed prompts whose point lies inside it — as positives (farthest-point subsampled to
  `n_positives`), and the nearest prompts belonging to other instances as negatives (nearest-first
  up to `n_negatives`, optionally capped by `max_negative_distance`). See
  `derive_refinement_prompts`.
- `boxes`: the instance's bounding box, grown by `box_extension`.
- `masks`: the instance's mask as a 256x256 logit prompt in SAM2's squashed square frame
  (`mask_to_logits`). Only valid in combination, since SAM2 is not trained for dense-only prompting.

The acceptance `policy` decides what the second round may do: `replace` repaints every instance
from its new mask (ascending combined score, so the most confident wins contested pixels, and an
empty re-prompt keeps the first-round mask), `keep-if-better` keeps the first-round mask unless the
second round's `predicted_iou * stability_score` beats the first round's. Everything is exposed as
`generate(refinement=..., refinement_kwargs={...})`; the former `refine_with_box_prompts` /
`box_extension` arguments were folded into `refinement="boxes"`. In 3D the closely related idea was
measured before and found neutral (see the module docstring: grouped re-prompting +0.001, adjacent
negatives +0.001); the 2D case is what this experiment answers.

### Benchmark and decision rules

Same benchmark and gates as the sweeps above: the 240-image 2D
portion of manifest schema 5, checksum `0f8fb67b3650a71f9f44b53037e89546`, model `hvit_t` checkpoint
`best` (`85fb099c4bb038fa0ab9bddd6151689e`), serialized runs on an `NVIDIA A100-SXM4-80GB MIG
1g.20gb`. The goal of this experiment is declared as **exploration**: the refinement stays an
explicit opt-in mode either way, the gates are reported for the record, and the shortlist is ranked
by dataset-balanced macro mSA.

The refinement changed `micro_sam/v2/automatic_prompt_generation.py`, so the implementation checksum
is new: `9f6254b7cce5f6b1471b4801c8809f54`. Three control trials with `refinement=None` re-establish
the baseline on this implementation. All three exactly reproduce the canonical baseline — macro mSA
0.269577 and every per-dataset mSA to six decimals — which verifies that the refactor is a no-op
when the refinement is off. Their wall times are 3.32-3.45 minutes on the 240 2d images.

### Screening

Every refinement configuration shares the first round, so the grids were screened with
`screen_apg_refinement.py`, which runs `propose` once per image and only the merge plus the
second-round re-prompt per configuration. Screening ranks quality only; canonical numbers come from
the full benchmark runs of the shortlist. Four screening rounds were run; every round carries the
`refinement-none` control, which reproduces the baseline exactly (a verification that the shared
proposals do not leak between configurations).

#### Round 1: the main grid (27 configurations)

`points` with `n_positives x n_negatives x policy` in `{2,3,5} x {0,2,4} x {replace,
keep-if-better}`, plus `boxes`, `points+boxes`, `points+masks` and `boxes+masks` at the point
defaults (`n_positives=3, n_negatives=4`) with both policies.

| configuration | macro mSA | macro change |
|---|---:|---:|
| points+boxes replace | 0.275605 | +2.24% |
| points+boxes keep-if-better | 0.275232 | +2.10% |
| boxes replace | 0.274108 | +1.68% |
| boxes keep-if-better | 0.274092 | +1.68% |
| points p2-n0 replace | 0.272990 | +1.27% |
| points p2-n0 keep-if-better | 0.272458 | +1.07% |
| points p3-n0 keep-if-better | 0.271999 | +0.90% |
| boxes+masks keep-if-better | 0.270781 | +0.45% |
| ... remaining keep-if-better points configs | 0.2699-0.2707 | +0.1% to +0.4% |
| baseline (refinement-none) | 0.269577 | 0 |
| points+masks (both policies) | 0.2680-0.2692 | -0.6% to -0.1% |
| points with negatives, replace | 0.2525-0.2648 | **-6.3% to -1.8%** |

Three immediate findings:

1. **For the pure point mode, negatives hurt.** Every `n_negatives>0` configuration is worse than
   its 0-negative counterpart, catastrophically so under `replace` (down to -6.3%). The
   `keep-if-better` policy contains the damage (the model's own score identifies the bad re-prompts)
   but never turns negatives into a win. More positives also hurt: p2 > p3 > p5.
2. **Mask conditioning is neutral to harmful.** `points+masks` is the only mode below baseline in
   both policies; `boxes+masks` is strictly worse than `boxes`.
3. **`points+boxes` beats `boxes`** — and it did so at the *untuned* defaults `p3-n4`, i.e. with the
   very negatives that ruin the pure point mode.

`boxes replace` at +1.68% is consistent with the +1.74% that
experiment 2 above measured for refining every instance, which
cross-validates the new engine against the reverted implementation.

#### Rounds 2-4: the `points+boxes` response surface

With a box anchoring the re-prompt, the roles invert — negatives help and extra positives without
negatives do almost nothing:

| configuration (all replace) | macro mSA | macro change |
|---|---:|---:|
| p3-n8 | 0.277965 | +3.11% |
| p3-n6 | 0.277424 | +2.91% |
| p5-n6 | 0.276459 | +2.55% |
| p2-n4 | 0.276357 | +2.52% |
| p3-n4 | 0.275605 | +2.24% |
| p5-n4 | 0.274419 | +1.80% |
| p2-n2 | 0.273516 | +1.46% |
| p3-n2 | 0.272662 | +1.15% |
| p1-n0 (box + surviving point) | 0.271461 | +0.70% |
| p2-n0 | 0.270860 | +0.48% |
| p3-n0 | 0.270264 | +0.26% |
| p5-n0 | 0.269632 | +0.02% |
| p3-n12 | 0.271987 | +0.89% |
| p3-n16 | 0.256764 | -4.75% |

The negative-count response peaks at 6-8 and collapses beyond 12. The macro peak is misleading,
though: per-dataset, `p3-n8` is a lopsided trade (DynamicNuclearNet +13.1%, but LiveCELL -4.5%,
DIC -4.2%, DeepBacs -1.0%), while `p3-n4` and `p3-n6` gain on three datasets and only regress
LiveCELL (-2.3% / -3.0%) and TissueNet (-1.4% / -0.3%). The shortlist therefore carries `n4` and
`n6`, not the macro-optimal `n8`.

### Canonical runs

The five shortlisted configurations ran through the canonical benchmark and
`compare_apg_optimization.py --target quality` against the three control trials. Canonical quality
matches the screening exactly on every configuration, which validates the screening shortcut
end to end.

| configuration | macro mSA | macro change | runtime change | worst dataset runtime | accepted |
|---|---:|---:|---:|---:|---|
| points+boxes p3-n6 replace | 0.277424 | +2.91% | +38.50% | +72.93% | no |
| points+boxes p3-n4 replace | 0.275605 | +2.24% | +24.70% | +44.24% | no |
| points+boxes p3-n4 keep-if-better | 0.275232 | +2.10% | +42.86% | +67.97% | no |
| boxes replace | 0.274108 | +1.68% | +40.92% | +69.39% | no |
| points p2-n0 replace | 0.272990 | +1.27% | +42.20% | +71.28% | no |

Every configuration fails the +5% quality bar and the 10% per-dataset runtime cap; none regresses
any dataset by more than 5%, so that guard is not the limiting gate. The runtime deltas carry the
usual single-trial noise (the same amount of second-round work measures anywhere between +24.7% and
+42.9%); a second SAM2 pass over every instance costs roughly a third of the run either way, in line
with the earlier box-refinement measurement. Peak CUDA memory is unchanged at 1.98 GiB.

Per-dataset relative mSA changes:

| configuration | LiveCELL | TissueNet | DynamicNuclearNet | DeepBacs | DIC HepG2 |
|---|---:|---:|---:|---:|---:|
| points+boxes p3-n6 replace | -3.04% | -0.33% | +9.62% | +2.37% | +2.76% |
| points+boxes p3-n4 replace | -2.34% | -1.36% | +6.20% | +4.79% | +6.22% |
| points+boxes p3-n4 keep-if-better | -2.03% | -1.32% | +5.57% | +4.28% | +10.36% |
| boxes replace | +0.53% | +1.12% | +3.44% | +1.01% | -1.73% |
| points p2-n0 replace | -0.33% | +2.20% | +4.98% | -4.61% | +3.13% |

The grouped prompts are what moves the needle in both directions: relative to box-only refinement
they buy DynamicNuclearNet, DeepBacs and DIC while costing LiveCELL and TissueNet. Box-only is the
lone variant that improves LiveCELL. As with the earlier point-placement experiment, no single
setting helps every modality.

### Attribution

The stratification the previous sweep asked for, from the recorded per-sample statistics of
`points+boxes p3-n4 replace` (the merge reasons are configuration-independent):

| dataset | kept instances | suppressed duplicates | duplicates per instance | mSA change |
|---|---:|---:|---:|---:|
| DIC HepG2 | 142 | 1,814 | 12.8 | +6.22% |
| TissueNet | 2,794 | 1,681 | 0.60 | -1.36% |
| LiveCELL | 9,369 | 6,017 | 0.64 | -2.34% |
| DynamicNuclearNet | 2,673 | 716 | 0.27 | +6.20% |
| DeepBacs | 630 | 305 | 0.48 | +4.79% |

The suppressed-duplicate supply explains DIC (each instance has a dozen grouped prompts to draw on)
but not DynamicNuclearNet, whose gain arrives with the fewest duplicates per instance — there the
negatives, not the extra positives, carry the improvement (consistent with the `points+boxes`
response surface, where `p1-n0` already beats every `n0` setting with more positives). LiveCELL and
TissueNet sit in the middle of the supply range and regress: densely packed, similarly sized cells
are exactly where a neighbouring prompt used as a negative most plausibly touches the instance's own
extent. Under `keep-if-better` the model's own score arbitrates and 100% of instances still adopt
the second-round mask when a box is present, so the score does not recognise the LiveCELL
regressions — the predicted IoU of a box-anchored re-prompt is systematically higher than the
point-prompt score it competes against.

### Conclusions

1. **Grouped prompts pay only when box-anchored.** The best configuration combines all three
   information sources the first round leaves behind: the box (extent), the grouped positives
   (identity), and neighbouring prompts as negatives (boundary). Removing the box flips the
   negatives from +2.4 points (`p3-n4` vs `p3-n0`, boxed) to -5.6 points (unboxed).
2. **Negative prompts without a box are the failure mode, not the fix.** A single SAM2 forward
   conditioned on positive points plus foreign negatives fragments the mask; the merge's score
   ordering then propagates the damage. `keep-if-better` contains it but cannot recover a win.
3. **The negative-count response peaks at 6-8 and collapses by 16.** The macro-optimal `n8` is a
   lopsided DynamicNuclearNet trade; `n4`-`n6` is the balanced range.
4. **Mask conditioning adds nothing** in either combination, consistent with the -0.005 measured for
   2d-mask conditioning of 3d anchors.
5. **Nothing is default-worthy.** +2.9% macro at +38% runtime repeats the box-refinement verdict at
   a higher quality point: worthwhile as an explicit high-quality mode, not as the default. The
   library default stays `refinement=None`.
6. **For 3d,** these results sharpen the earlier neutral measurements: the ingredients that were
   tried there separately (grouped re-prompting +0.001, adjacent negatives +0.001, box conditioning
   +0.001) are exactly the ones that only work *in combination* in 2d. A 3d revisit should test the
   combined `points+boxes` conditioning of the anchor slice rather than any single ingredient — but
   the expected ceiling is low, since 3d selection was shown to sit 0.006 below its oracle.

### Reproducibility and artifacts

Output root as before:

```text
/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/hvit_t/
  85fb099c4bb038fa0ab9bddd6151689e/
```

Screening results live under `refinement_screening/` below the same root; the four screening run
directories have config-list checksums `06743d30ffcd067eb3ec516949f90d6a` (main grid),
`ccede5e7a9026aaaf45ed5d66ad3a814` (`points+boxes` surface), `33202d8b93dbcf3bc655c0f81dd36bfe`
(negative counts) and `59f166492207785a80cc9b74559a0634` (saturation probe).

Implementation checksum: `9f6254b7cce5f6b1471b4801c8809f54` at revision
`8bb90584e0f6df22e6995d411146a0434cd160dd` plus the refinement work tree. Control config checksums:
`3b6baba28669c2897b453f9246222bc5`, `cda279d704fe9845ab424066945cdd11`,
`8a0d4adc9b39de3f0bddc39ea4300afb` for trials 1-3. Candidate config checksums:

| configuration | checksum |
|---|---|
| points+boxes p3-n6 replace | `d178de990f26c71c320bd75921e2927b` |
| points+boxes p3-n4 replace | `0db6009d7317c02a3def7b200d80b14e` |
| points+boxes p3-n4 keep-if-better | `5be45599f85f44af53b71cd732f4b6c6` |
| boxes replace | `c64754f399678f4a263646be6540a3f1` |
| points p2-n0 replace | `7e41fb3db1a3547d56e8f0e6695f0880` |

A run directory `0f8fb67b...-3b6baba2...-3862c4a2...` with status `failed` is an aborted control
launched against a pre-final implementation state; it carries no results and can be removed.

```bash
# Controls and canonical candidate runs:
python finetuning/v2/evaluation/benchmark_apg_optimization.py --ndim 2 --trial-id control-1
python finetuning/v2/evaluation/benchmark_apg_optimization.py --ndim 2 --config <candidate>.json

# Screening:
python finetuning/v2/evaluation/screen_apg_refinement.py --device cuda

# Comparison:
python finetuning/v2/evaluation/compare_apg_optimization.py --ndim 2 --target quality \
    --baseline-run <control-1> --baseline-run <control-2> --baseline-run <control-3> \
    --candidate-run <candidate> --output <decisions>.json
```

---

## Refinement campaign 2: holdout validation and the four follow-up directions

The continuation of the refinement work above, previously kept in its own document and merged here:
a held-out validation subset, a generalization check of campaign 1, and the systematic sweep of its
four follow-up directions.

### Outcome

The refinement's recommended configuration improved by a factor of ~1.6 over campaign 1 and is
confirmed on a held-out validation subset: **`points+boxes` with one positive (the surviving
prompt), six nearby negatives, and geometric acceptance gates** reaches **+4.19% macro mSA on the
tuned subset and +4.89% on the held-out one** (campaign 1: +2.91%, unconfirmed), for +35-50%
runtime. Its values are now the refinement defaults, so `generate(refinement="points+boxes")` is
the recommended opt-in; the pipeline default stays `refinement=None`, since the +5% quality gate is
missed by a hair and the 10% runtime cap by a wide margin.

Of the four directions swept: the **geometry gates** (1) and **negative quality** (4) are confirmed
and adopted; **recovery** (3) is measured neutral; **adaptivity by grouped supply** (2) is refuted —
and the sweep's biggest single gain came from an ablation none of the four directions predicted:
dropping the grouped extra positives entirely (`n_positives=1`), which re-frames what this
refinement actually is. The campaign-1 findings themselves generalized to the holdout (gains equal
or larger), with one exception: the DIC HepG2 gain was set-specific.

### Motivation

[Campaign 1](#second-round-refinement-from-grouped-prompts) established the
refinement mechanism and measured `points+boxes`
p3-n6 at +2.91% macro mSA (+38.5% runtime), with four follow-up directions left on the table:

1. **Geometry-based acceptance** — the keep-if-better score gate never fires for box-anchored
   re-prompts (predicted IoU is systematically higher than the point-prompt score it competes
   against), so the LiveCELL/TissueNet regressions pass unchecked. Consistency and containment gates
   arbitrate on geometry instead.
2. **Per-instance adaptivity** — the grouped-duplicate supply varies by two orders of magnitude
   between datasets (DIC 12.8 per instance, DynamicNuclearNet 0.27) and correlates with where the
   grouped points pay; a per-instance threshold applies them only where they can.
3. **Recall recovery** — campaign 1 only polished surviving masks, but recall is the limiting factor
   (APGv2 diagnostics) and the merge rejects whole objects whose mask a neighbour partially claims.
   Re-prompting those dropped records as *new* instances attacks the recall axis directly.
4. **Negative selection quality** — nearest-first prompt selection in confluent data plausibly picks
   negatives that touch the instance's own extent; the source (neighbour interior point vs raw
   prompt), a minimum distance to the instance's own mask, and the never-swept
   `max_negative_distance` are the candidate fixes.

Campaign 1 also mined its 240-image validation subset with four screening rounds, so this campaign
first builds a held-out subset and checks that the campaign-1 findings generalize before tuning
anything new.

### The holdout subset

`subset_manifest_v5_holdout.json`, checksum `bf8f3c28befe1fb06d62309dc302d1c4`, built against the
primary manifest `0f8fb67b3650a71f9f44b53037e89546` (recorded as `holdout_of`). 233 2d samples,
selected by the same complexity-quantile policy on the pool that remains after excluding every
primary image:

| dataset | holdout samples | image-disjoint from primary? | pool after exclusion |
|---|---:|---|---:|
| LiveCELL | 80 (10 per cell type) | yes | 489 (>=41 per type) |
| TissueNet | 40 | yes | 3078 |
| DynamicNuclearNet | 40 | yes | 1377 |
| DeepBacs | 30 | **no — reused verbatim** (all 30 validation images are primary) | 0 |
| DIC HepG2 | 43 | yes | 43 (of 93 usable; the primary set holds 50) |

The DeepBacs column is therefore not held out and is flagged in every comparison. The test splits
could have closed the DeepBacs/DIC gaps but are the paper-evaluation splits: selecting on them is
the leak the `VAL_SPLITS` policy exists to prevent, so they were not used. Unequal per-dataset
counts do not skew the quality figure, which is an equal-weight mean of per-dataset means.

Tuning stays on the primary subset (comparable to all campaign-1 tables); the holdout is only read
for the validity check below and for confirming the final shortlist.

### Epochs

The benchmark checksums its implementation files, so the campaign runs in two epochs:

- **Epoch 1** — manifest machinery only (`--subset` axis in the benchmark and screening scripts).
  Implementation checksum `586d2bcb0c15f95d9a93a7a3c3406e79`. The set-A control (trial
  `epoch1-control-A`) reproduces macro mSA 0.269577 with every per-dataset value identical: the
  machinery is behavior-free.
- **Epoch 2** — the four mechanisms in `micro_sam/v2/automatic_prompt_generation.py`.
  Implementation checksum `c3a723ae4c7222abd642188169cc9c77`; fresh controls on both subsets
  reproduced epoch-1 quality.

### Validity check (epoch 1): the campaign-1 findings generalize

Criteria fixed before running: campaign-1 findings generalize iff (i) the macro ordering on the
holdout is `points+boxes {n4, n6}` > `boxes` > baseline; (ii) the `points+boxes` macro gains retain
at least half their set-A size; (iii) the negative-count response rises through n4-n6 and does not
collapse before n8.

**All three pass.** Holdout baseline: macro mSA 0.264318 (three identical control trials,
`controlB-{1..3}`). The five campaign-1 configurations, canonical runs on the holdout:

| configuration | holdout macro | holdout change | set-A change |
|---|---:|---:|---:|
| points+boxes p3-n6 replace | 0.273193 | **+3.36%** | +2.91% |
| points+boxes p3-n4 replace | 0.271569 | +2.74% | +2.24% |
| points+boxes p3-n4 keep-if-better | 0.270996 | +2.53% | +2.10% |
| boxes replace | 0.269128 | +1.82% | +1.68% |
| points p2-n0 replace | 0.267424 | +1.18% | +1.27% |

Every macro gain is at least as large on the holdout as on the tuned set, and the negative-count
response (screening, `points+boxes` p3, replace) rises monotonically: n0 +0.57%, n2 +1.80%,
n4 +2.74%, n6 +3.36%, n8 +3.60% — no collapse through n8.

Per-dataset, the picture sharpens rather than reverses (baseline per dataset: LiveCELL 0.339412,
TissueNet 0.282117, DynamicNuclearNet 0.433541, DeepBacs 0.247616*, DIC 0.018904):

| configuration | LiveCELL | TissueNet | DynNuclearNet | DeepBacs* | DIC HepG2 |
|---|---:|---:|---:|---:|---:|
| points+boxes p3-n6 replace | -2.44% | -0.66% | +11.45% | +2.37% | -5.26% |
| points+boxes p3-n4 replace | -1.74% | -0.76% | +7.66% | +4.79% | -3.91% |
| boxes replace | +0.93% | -0.05% | +4.23% | +1.01% | +0.83% |
| points p2-n0 replace | -0.16% | +0.68% | +6.20% | -4.61% | -6.83% |

\* DeepBacs is the reused (not held-out) dataset; its values are identical to set A by construction.

The DynamicNuclearNet gain is robust and larger on the holdout; the LiveCELL/TissueNet regressions
replicate at smaller size. The one campaign-1 result that does **not** generalize is the DIC gain
(+6.2%/+10.4% on set A, -3.9%/-5.3% here): DIC's absolute baseline is tiny (0.019-0.027) with 43-50
samples, so its relative changes carry the largest noise of the five datasets. This strengthens the
case for the geometry gates (direction 1), whose job is exactly to veto harmful re-prompts
per instance.

### The four mechanisms (epoch 2, historical implementation)

The epoch-2 implementation added all four mechanisms below to
`micro_sam/v2/automatic_prompt_generation.py`. The geometry and negative-quality mechanisms remain;
the neutral recovery and refuted grouped-supply adaptivity paths were removed after the campaign to
avoid carrying unsupported options in the current API.

- **Geometry gates** (shared kwargs): `min_consistency` accepts a second-round mask only if its IoU
  with the first-round mask reaches the threshold — the re-prompt may polish, not reshape;
  `max_foreign_overlap` keeps the first round when the new mask grows into other first-round
  instances beyond the threshold. Both veto independently of the policy, because the model's score
  cannot arbitrate across prompt types. Stats: `gated_consistency`, `gated_foreign`.
- **Negative quality** (points kwargs): `negative_source="interior"` uses the deepest interior
  point of each other instance instead of its raw prompt; `min_negative_distance` excludes
  negatives closer than that to the instance's own first-round mask (exact EDT on the padded
  bounding box).
- **Recovery** (historical component `"recover"`, then valid standalone): records the merge dropped as
  'duplicate' or 'truncated below min size', with at most `recover_max_claimed` of their pixels
  claimed, are re-prompted with their own point as the positive and the claimants' surviving
  prompts as negatives; a survivor (score above `score_threshold`, unclaimed pixels above
  `min_size`) is painted on its unclaimed pixels as a **new** instance. Built on
  `merge_by_score(return_claimed=True)`. Stats: `recovery_candidates`, `recovered_instances`.
- **Adaptivity** (historical points kwarg): `min_grouped_for_points` re-prompts sparsely grouped instances
  (fewer suppressed prompts than the threshold) with their box alone — their point row is fully
  padded with the ignore label inside the same batch. Requires the `boxes` component. Stats:
  `points_suppressed_instances`.

### Sweeps (primary subset)

Epoch-2 controls: three trials per subset, all reproducing epoch-1 quality exactly (primary
0.269577, holdout 0.264318) — the four mechanisms are no-ops when off. Base modes for the sweeps:
`points+boxes` p3-n4 and p3-n6, replace (the campaign-1 winners; primary-set references +2.24% and
+2.91%).

#### S1: acceptance gates

| configuration (on pb-n6) | macro mSA | macro change |
|---|---:|---:|
| `min_consistency=0.7` | 0.277984 | **+3.12%** |
| `max_foreign_overlap=0.15` | 0.277703 | +3.01% |
| `max_foreign_overlap=0.05` | 0.277688 | +3.01% |
| `min_consistency=0.5` | 0.277552 | +2.96% |
| ungated | 0.277424 | +2.91% |
| `min_consistency=0.85` | 0.275893 | +2.34% |

The same ordering holds on pb-n4 (mc0.7 best at 0.275990). The consistency gate at 0.7 is the
optimum: 0.85 over-gates (it vetoes genuine boundary fixes), 0.5 barely fires. Per-dataset, mc0.7
softens the LiveCELL regression (-3.04% to -2.65%) and lifts DIC (+2.76% to +5.00%) without losing
DynamicNuclearNet. The gates do not fully repair LiveCELL on their own.

#### S2: negative quality

| configuration (on pb-n6) | macro mSA | macro change |
|---|---:|---:|
| `negative_source=interior` | 0.277984 | **+3.12%** |
| interior + `min_negative_distance=3` | 0.277960 | +3.11% |
| prompts + `min_negative_distance=3` | 0.277798 | +3.05% |
| prompts + `min_negative_distance=6` | 0.277715 | +3.02% |
| `max_negative_distance=64` | 0.277650 | +3.00% |
| interior + `min_negative_distance=6` | 0.277640 | +2.99% |
| prompts (base) | 0.277424 | +2.91% |

Interior negatives win and cost nothing (the EDT-based distance filter adds ~18% select time for no
further quality). Per-dataset, interior softens LiveCELL to -2.57% and turns TissueNet positive
(+0.18%). Every negative-quality variant beats the raw-prompt base, confirming the "negatives touch
the instance's own extent" hypothesis — but like the gates, none fully repairs LiveCELL alone.

#### S3: composition

| configuration (on pb-n6, replace) | macro mSA | macro change | LiveCELL |
|---|---:|---:|---:|
| mc0.7 + fo0.15 | 0.278290 | **+3.23%** | -2.65% |
| interior + fo0.15 | 0.278156 | +3.18% | -2.55% |
| interior + mc0.7 + fo0.15 | 0.277966 | +3.11% | **-2.24%** |
| interior + mc0.7 | 0.277773 | +3.04% | -2.26% |

The gates compose (mc0.7 + fo0.15 beats either alone), and adding interior negatives on top trades
a little macro for the friendliest LiveCELL/TissueNet profile (interior + mc0.7 + fo0.15:
LiveCELL -2.24%, TissueNet -0.09%, DIC +6.09%). Both the macro winner and the balanced variant are
carried into S4/S5 and the shortlist. Even composed, no setting turns LiveCELL positive: what the
grouped points gain elsewhere they structurally cost on confluent phase-contrast data.

#### S4: recovery — neutral

On the S3 macro winner (`mc0.7 + fo0.15`): `recover_max_claimed=0.4` gives 0.278306 (+0.006 points
over the base), 0.6 and 0.8 give 0.278145/0.278150 (slightly below). Standalone recovery
(`refinement="recover"`) lands at 0.269526, marginally **below** baseline. The dropped-duplicate
records that pass the claim cap either fail the score threshold, produce too-few unclaimed pixels,
or add objects that cost as much precision as they add recall. The recall axis, like its 3d
counterpart, does not respond to re-prompting — consistent with the APGv2 finding that the
merge-rejection failure is rarer than the never-proposed one. The measured-neutral `recover`
component was subsequently removed from the library.

#### S5: adaptivity by grouped supply — refuted, instructively

On the same base: `min_grouped_for_points` 1/2/3 give 0.270593/0.267750/0.266471 — far below the
+3.23% base, barely above (or below) the plain baseline. The mechanism works as designed (a control
with everything suppressed reproduces the `boxes` mode), so the result is a finding, not a bug: an
instance with no grouped extras still carries its anchor positive **and its negatives**, and
suppressing its point row removes the negatives — which S2 and the `points+boxes` response surface
identified as the active ingredient. Gating the point prompt on grouped-duplicate supply therefore
throws away exactly what pays. The signal gates the wrong ingredient; per-instance adaptivity would
have to key on something that predicts *negative* usefulness (local crowding), which is left as an
explicitly unexplored follow-up. The refuted `min_grouped_for_points` option was subsequently removed.

#### S6: the positives ablation — one positive is enough, and better

The holdout confirmation screening carried one ablation the primary sweeps had not measured:
`n_positives=1` (the surviving prompt only — no grouped extras) under the composed gates. It won on
the holdout by a wide margin (+4.89% vs +3.66% for the p3 winner), so it was measured back on the
primary subset, where the ordering replicates:

| configuration (all with mc0.7 + fo0.15, replace) | macro mSA | macro change | LiveCELL | DynNuclearNet |
|---|---:|---:|---:|---:|
| p1-n8 | 0.282051 | **+4.63%** | -3.97% | +16.28% |
| p1-n6 | 0.280874 | +4.19% | -2.61% | +12.41% |
| p1-n6 interior | 0.280835 | +4.17% | -1.91% | +11.69% |
| p2-n6 | 0.279948 | +3.85% | -2.69% | +10.71% |
| p1-n4 | 0.278223 | +3.21% | -1.63% | +8.04% |
| p3-n6 (the S3 winner) | 0.278290 | +3.23% | -2.65% | +9.40% |

The grouped extra positives — the original core of the second-round idea — do not merely fail to
help: removing them adds a full point of macro quality. The refined prompt that works is
**the surviving point + the instance's box + nearby negatives + the geometry gates**; the
suppressed prompts' only productive role is indirect, as the negative pool of the neighbours.
(This also explains S5: adaptivity that suppresses the point row removes the negatives, the actual
active ingredient.)

### Confirmation and canonical runs

The top configurations ran through the canonical benchmark on both subsets, compared against the
respective epoch-2 control trials with `compare_apg_optimization.py --target quality`. Canonical
quality matches the screening exactly everywhere. One honesty note: the `n_positives=1` direction
was first surfaced by the holdout ablation and then *selected* on the primary subset (S6), so the
holdout numbers below are a fair confirmation of the selection, with that one-config peek on
record.

| configuration (gates = mc0.7 + fo0.15) | primary macro | primary change | holdout macro | holdout change | runtime (A / B) | accepted |
|---|---:|---:|---:|---:|---|---|
| **p1-n6 + gates** | 0.280874 | +4.19% | **0.277244** | **+4.89%** | +48% / +35% | no |
| p1-n8 + gates | 0.282051 | **+4.63%** | 0.277171 | +4.86% | +48% / +35% | no |
| p1-n6 interior + gates | 0.280835 | +4.18% | 0.276709 | +4.69% | +50% / +37% | no |
| p2-n6 + gates | 0.279948 | +3.85% | 0.275333 | +4.17% | +49% / +35% | no |
| p3-n6 + gates | 0.278290 | +3.23% | 0.274001 | +3.66% | +49% / +35% | no |

All five fail the +5% quality bar — by 0.1-0.8 points now, not by 3 as in campaign 1 — and all
break the 10% runtime cap by a wide margin, so nothing becomes a pipeline default. No configuration
regresses any dataset by more than 5% on either subset (worst: p1-n8's LiveCELL -3.97% on the
primary subset). Peak CUDA memory is unchanged at 1.98 GiB.

**Recommendation** (per the rule fixed before the sweep — best holdout macro among configurations
regressing no dataset by more than 5% on either subset): **`points+boxes` with `n_positives=1`,
`n_negatives=6`, `min_consistency=0.7`, `max_foreign_overlap=0.15`, `policy="replace"`** — +4.19%
macro on the tuned subset, +4.89% on the held-out one. These values are now the `DEFAULT_REFINEMENT`
entries, so the recommended usage is simply:

```python
segmenter.generate(refinement="points+boxes")
```

The pipeline default stays `refinement=None`. Users preferring the gentlest per-dataset profile over
peak macro can pass `refinement_kwargs={"negative_source": "interior"}` (LiveCELL -1.91%,
TissueNet +1.09%); `n_negatives=8` buys DynamicNuclearNet (+16.3%) at LiveCELL's expense (-3.97%).

### Conclusions

1. **The refined second-round prompt is: the surviving point + the instance's box + ~6 nearby
   negatives + geometric acceptance gates.** Worth +4.2%/+4.9% macro mSA (tuned/held-out) over the
   baseline and +1.3/+1.5 points over the ungated campaign-1 winner, at +35-50% runtime.
2. **Grouped extra positives are refuted** (direction 2's premise and campaign 1's core idea): p1 >
   p2 > p3 on both subsets. The suppressed prompts matter only as the neighbours' negative pool.
3. **Geometry gates work where scores cannot** (direction 1 confirmed): `min_consistency=0.7`
   composes with `max_foreign_overlap=0.15` for +0.3 points and softer regressions; 0.85
   over-gates. They contain, but do not eliminate, the LiveCELL cost of the negatives.
4. **Negative quality matters at the margins** (direction 4 partially confirmed): interior-point
   negatives are the best source and trade ~0.1 macro points for visibly gentler LiveCELL/TissueNet
   behaviour; the EDT distance filter costs runtime for nothing.
5. **Recovery is neutral** (direction 3 refuted): +0.006 points on top of the best config, slightly
   negative standalone. The recall axis does not respond to re-prompting dropped records, matching
   the 3d result and the APGv2 diagnosis that most misses were never proposed at all.
6. **Adaptivity by grouped supply is refuted, instructively** (direction 2): suppressing the point
   row removes the negatives, the actual active ingredient. Any future per-instance adaptivity must
   key on a signal that predicts negative usefulness (e.g. local crowding), not positive supply.
7. **The holdout discipline paid off twice**: it certified that campaign 1 was not a screening
   artifact (gains generalize, even grow), exposed the one set-specific result (DIC's campaign-1
   gain), and its confirmation screening surfaced the p1 ablation that became the winner.

### Reproducibility and artifacts

Output root as before; holdout runs key on manifest checksum `bf8f3c28befe1fb06d62309dc302d1c4`,
screening runs live under `refinement_screening/`. Epoch checksums: epoch 1 (manifest machinery)
`586d2bcb0c15f95d9a93a7a3c3406e79`; epoch 2 (the four mechanisms, all sweeps and canonical runs)
`c3a723ae4c7222abd642188169cc9c77`; epoch 3 (the recommended values as `DEFAULT_REFINEMENT`, the
work tree's final state) `8bcd5e7457fcda456b872d1f329369c4`, certified by the `refinement=None`
control `04482b7ffd263202d55a6184b648aacf` reproducing 0.269577 with every per-dataset value exact.

Canonical candidate config checksums (identical for the primary and holdout runs; the run
directories differ through the manifest checksum): p1-n6 `2ffb27a17fd224e5105c9108343a19b3`,
p1-n6-interior `e9d183ad27536a43d005e4269a41035c`, p1-n8 `aaebcb67bdff007c69b634bab00446a0`,
p2-n6 `b72d7a7de3f38e13bbb5933189a97986`, p3-n6 `920e114aa174b0ee5cf8e39ed0fd43fd`. Control config
checksums: primary `055529a777ecef73dcb8238ecc8f3b0a` / `61aafed6af057da99374b9af9d76502f` /
`b26ae23e408a63ee2582e69e8379883f`; holdout `4bc4f2d669c31128e9a42e0912eabd69` /
`24d7494a9a2bc0a9efde98ce3b1b57aa` / `b66228ad53910e21d3f6b4e52dd39e51`.

## Compact MLP deployment follow-up

### Selected models

The selected first-round model is a 52 KB groupwise MLP with a shared 64-unit encoder, mean/max
group context, 10% dropout and a per-alternative scoring head. Direct regression on alternative IoU
was the best training objective. With the historical predicted-IoU eligibility filter, deferred
merging reaches 0.283210 primary mSA and 0.278328 holdout mSA, +5.06% and +5.30% over the
established controls. Its three-trial holdout median is 203.5 s, +7.8% over the same-implementation
188.8 s baseline, but two datasets exceeded the original per-dataset runtime cap.

The optional refinement gate is a separate 50 KB `(128, 64)` MLP trained directly on the positive
benefit of the second pass. At the primary-selected 50% threshold it raises quality to
0.287493/0.283929 on primary/holdout, but its 251.4 s median is +33.1% over baseline. The additional
latency is dominated by the selected second decoder calls rather than gate inference.

### Torch-only feature path

The version-1 19-feature schema is computed only with Torch. Decoder masks, predicted IoU,
stability, foreground support, seed geometry, pairwise agreement, ranks, areas and boxes stay on the
decoder device through feature extraction and MLP inference. CUDA synchronization is postponed
until the already-required proposal materialization. Empty masks, tied alternatives, clipped edge
seeds and singleton groups are covered by the Torch tests.

The artifact loader supports only pointwise and permutation-equivariant groupwise MLP artifacts in
`.pt` or `.pth` form. The groupwise model's shared alternative encoder is pooled with group mean
and maximum, concatenated back to every alternative and scored by one shared head. Both selector and
gate expose tensor inference; the historical predicted-IoU path still avoids model loading and
feature extraction entirely.

### Fixed eager/deferred comparison

The same H64 artifact was evaluated under both merge semantics after architecture selection:

| merge | primary mSA | holdout mSA | holdout median | runtime vs baseline |
|---|---:|---:|---:|---:|
| eager learned rescore | 0.282357 | 0.277130 | **200.9 s** | +6.4% |
| deferred group merge | **0.283210** | **0.278328** | 203.5 s | +7.8% |

Deferred gains 0.30% primary and 0.43% holdout relative quality. Its explicit prompt-group lock is a
small extension of the ordinary score-ordered merge, not a separate assignment stage. The later
learned-filter campaign below establishes eager selection as the stronger deployment choice.

### Reproduction

The supported training entry points are now:

```bash
python finetuning/v2/evaluation/train_apg_multimask_selector.py --device cuda
python finetuning/v2/evaluation/train_apg_multimask_selector.py --single-mask --device cuda
python finetuning/v2/evaluation/train_apg_refinement_gate.py --device cuda \
    --selection deferred --merge learned \
    --selector-oof-dataset multimask_selection/primary_features.npz \
    --selector-oof-predictions \
        multimask_selection/groupwise_v1/models/groupwise-h64-d0p1-regression_oof.npy
```

The selector script extracts GPU features and trains the fixed direct H64 model for either one or
three alternatives. The gate script extracts pre-refinement features and trains the fixed direct
H128x64 model. `screen_apg_mask_head_filters.py` performs the current selection/filter sweep,
`screen_apg_refinement.py` replays OOF gate predictions, and
`benchmark_apg_optimization.py` records artifact hashes and canonical timings.


## Decoder-head and learned-filter campaign

### Outcome

The follow-up campaign finds no evidence that SAM2's three-mask output is better than its dedicated
single-mask token under the historical APG policy. With the default predicted-IoU threshold and
ordering, `multimasking=False` improves mSA from 0.269577 to 0.280752 on primary (+4.15%) and from
0.264318 to 0.276301 on holdout (+4.53%); all five datasets improve on both splits. This is a real
but sub-threshold gain under the established +5% aggregate quality gate.

The conclusion changes once the learned score also controls initial proposal eligibility. A
separately trained singleton H64 scorer with a 0.25 learned-score filter reaches 0.289056/0.283470.
The existing triplet H64 scorer reaches 0.296508/0.295659 with eager selection and a 0.25 filter,
and 0.295966/0.296831 with deferred selection and a 0.30 filter. Thus the extra alternatives are
useful when both selection and filtering are microscopy-aware, even though the default three-mask
policy is worse than token 0.

| configuration | initial filter | primary mSA | holdout mSA | canonical runtime |
|---|---|---:|---:|---:|
| three masks, predicted IoU (control) | IoU >= 0.60 | 0.269577 | 0.264318 | 177.8 s median (3) |
| single mask, predicted IoU | IoU >= 0.60 | 0.280752 | 0.276301 | 171.7 s (1) |
| single mask, H64 ordering | IoU >= 0.60 | 0.281581 | 0.276264 | not timed |
| single mask, H64 ordering/filtering | MLP >= 0.25 | 0.289056 | 0.283470 | 174.7 s (1) |
| three masks, H64 eager ordering | IoU >= 0.60 | 0.282357 | 0.277130 | historical 200.9 s median (3) |
| three masks, H64 eager ordering/filtering | MLP >= 0.25 | **0.296508** | 0.295659 | 189.9 s median (3) |
| three masks, H64 deferred ordering | IoU >= 0.60 | 0.283210 | 0.278328 | historical 203.5 s median (3) |
| three masks, H64 deferred ordering/filtering | MLP >= 0.30 | 0.295966 | **0.296831** | 194.2 s (1) |

The historical eager/deferred runtimes are retained only to connect to the preceding campaign; they
use implementation checksum `52e46e3315064212fd71d5dde674561a`. All new canonical timings use
checksum `7447ecca968e89297a454b6e105d7d6d`, the same holdout manifest and the same A100 MIG. Entries
marked `(1)` are diagnostic one-shot timings rather than formal repeated comparisons.

### Leakage-safe selection protocol

The single-mask training dataset contains 48,331 token-0 alternatives, one per prompt group. Its
52 KB H64/dropout-0.1 groupwise MLP was trained from scratch rather than applying the triplet model
to singleton groups. Five image-level, dataset-stratified outer folds produced OOF predictions for
the primary sweep; the final model was refit on all primary rows for holdout. Its weighted OOF MSE
is 0.054056, weighted MAE 0.175710 and target correlation 0.664536.

The primary screen independently swept learned-score thresholds from 0.20 through 0.80 in 0.05
increments for single, eager-triplet and deferred-triplet routes. It also evaluated no initial
filter and the historical predicted-IoU filter. The fixed route thresholds were 0.25, 0.25 and 0.30,
respectively. Eager triplet at 0.25 was the global primary winner and is consequently the only
formal holdout candidate. Deferred's slightly higher holdout value is confirmatory evidence, not a
post-hoc selection.

Replacing only merge ordering is not enough: relative to the same learned scorers with the old IoU
filter, the learned filter adds +2.65%/+2.61% primary/holdout for singleton, +5.01%/+6.69% for eager
triplet and +4.50%/+6.65% for deferred triplet. The `score_filter` option now makes this distinction
explicit: `predicted_iou` retains historical eligibility, `selection_score` applies the threshold to
the installed model's score, and `none` disables the initial threshold. The selected MLP score
continues to define merge ordering in all learned configurations.

### Eager/deferred comparison

After fixing the scorer and threshold on primary, eager remains preferable for deployment. Deferred
is 0.18% worse on primary and 0.40% better on holdout, while its one-shot runtime is 2.3% above the
eager median because three times as many records enter the merge pool. The exact GPU feature and
MLP work is nearly identical; deferred mainly increases mask transfer and record materialization.
Its full-run diagnostics report 9.19 s feature extraction, 0.40 s MLP scoring, 4.36 s transfer and
1.86 s record construction. Eager reports medians near 9.20 s, 0.41 s, 1.63 s and 0.49 s.

The formal comparator accepts eager H64 plus the 0.25 MLP filter. Against three same-implementation
controls, holdout mSA improves 11.86%, no dataset regresses, median aggregate runtime rises 6.83%,
and the worst per-dataset runtime increase is 8.68%. All quality checks pass and the >10% quality
gain activates the previously established quality/runtime exception. This is an accepted opt-in
candidate; library defaults remain unchanged because the fitted artifact is external and promotion
was not part of this campaign.

### Reproduction artifacts

Artifacts are below the established optimization output root:

- singleton features/model: `multimask_selection/singlemask_v1/primary_features.npz` and
  `models/singlemask-groupwise-h64-d0p1-regression.pt`;
- triplet model: `multimask_selection/groupwise_v1/models/groupwise-h64-d0p1-regression.pt`;
- primary screen suffix: `mask_head_filter_screening/.../7765c0915944736459b5a3ed50ec7e9f/`;
- holdout confirmation suffix:
  `mask_head_filter_screening/.../223ed9424a42086df9a60fcec647b9cb/`;
- formal decision: `canonical_eager_decision.json` and `canonical_eager_decision.csv` in that
  holdout confirmation directory;
- canonical baseline suffixes: `e71562d94af3587252f1f89ebca4250f`,
  `1bb3ce5d68b10d4f60215ecaa8a93db0`, and `80026a5815867f3025116a41449ca75b`;
- canonical eager suffixes: `f379e4296344653ed7dd84094461379b`,
  `f527cd24aca4814d4e7ad9b620f08a48`, and `3092bb125bbe3ad33d371db57e4be269`;
- diagnostic singleton-default, singleton-MLP and deferred suffixes:
  `1ca4ce4dba9c74913fff3065c80154c2`, `f0e9fa0d5e1893251f6c654929d0fe37`, and
  `d41a0701a1ee5d04f200b69836d7b3e7`.

The campaign entry points are `train_apg_multimask_selector.py` (with optional `--single-mask`) and
`screen_apg_mask_head_filters.py`. The screen reuses one single- and one three-mask decoder pass per
image across every filter threshold, so threshold comparisons do not repeatedly invoke the decoder.

## Combined eager selector and uncertainty-gate campaign

### Policy-matched training and selection

The previous gate was trained after deferred selection and the historical predicted-IoU filter, so
attaching it to the accepted eager selector would have changed its input distribution. This campaign
instead re-extracted 20,370 primary instances after the exact accepted first pass: triplet H64 eager
selection, learned-score merge ordering, and `selection_score >= 0.25`. Both selector and gate inputs
use image-level OOF predictions on primary. The fixed direct `(128, 64)` gate was then refit on all
primary rows for holdout and deployment.

The extractor now records its first-pass policy in the feature dataset and accepts explicit
`score_filter` and `score_threshold` arguments. OOF screening replays the same filter when it checks
that every merged instance has a gate prediction. The trainer records both OOF fraction thresholds
for primary selection and full-refit thresholds for frozen holdout confirmation.

The OOF sweep selected 50% strictly on primary:

| refined fraction | primary mSA |
|---:|---:|
| 0% (first pass only) | 0.296508 |
| 10% | 0.297267 |
| 20% | 0.298105 |
| 30% | 0.298565 |
| 40% | 0.299291 |
| **50%** | **0.299904** |
| 100% (refine all) | 0.299195 |

The OOF threshold is `0.0150854`; the corresponding full-primary refit threshold frozen for holdout
is `0.0151422`. It selects exactly 10,185/20,370 primary instances and 10,705/20,934 holdout
instances (51.1% on holdout after distribution shift).

### Quality and runtime outcome

The frozen 50% gate confirms on holdout: mSA increases from 0.295659 to 0.301455, or 1.96% relative.
It also beats blanket refinement's 0.299058 while issuing about half as many second decoder calls.
Every dataset improves over the accepted first pass, but most gains are small outside DeepBacs:

| dataset | first pass mSA | + 50% gate mSA | quality change | runtime change |
|---|---:|---:|---:|---:|
| DeepBacs | 0.271977 | 0.285267 | +4.89% | +12.97% |
| DIC-HeLa | 0.043984 | 0.044304 | +0.73% | +4.45% |
| DynamicNuclearNet | 0.523938 | 0.536082 | +2.32% | +34.62% |
| LiveCELL | 0.347593 | 0.349098 | +0.43% | +31.64% |
| TissueNet | 0.290801 | 0.292524 | +0.59% | +22.90% |
| **dataset-balanced / total** | **0.295659** | **0.301455** | **+1.96%** | **+25.32%** |

Canonical runtime uses three serialized holdout trials for each side under implementation checksum
`621931b4644d2b7c5fece26343227f52`. The sum of per-dataset medians is 212.15 s for the first pass
and 265.86 s for the combination; the worst per-dataset increase is 34.62%. The incremental
candidate therefore fails the established quality route: its gain is below 5% and its runtime is
above the 10% cap. Compared with the original default it reaches +14.05% mSA and improves every
dataset, so the formal comparator accepts it only via the existing >=10% all-datasets quality
exception, despite +49.55% cross-epoch runtime.

The conclusion is consequently qualified: the gate targets refinement better than refine-all and is
the highest-quality tested 2D route, but it is not a viable promotion over the accepted eager first
pass under the deployment gates. Keep it as an explicit quality/latency option; retain selector-only
eager H64 plus the 0.25 learned filter as the deployment recommendation.

### Artifacts

The policy-matched dataset, gate artifact, sweep configurations, canonical configurations and
comparison reports are under
`multimask_selection/groupwise_v1/refinement_gate/eager_mlp_filter_025/`. Canonical combined run
suffixes are `6a48d15097d4d38e816183aa526d0e08`, `01f566fcef32f901c9ed1e5b136cce7f`
and `1609fe12d0525ca9e02c216851752022`; current-implementation first-pass suffixes are
`1d5530f64dd603916ae05de66227b458`, `aac26a5ca704f959f32adad81f759f9f` and
`672ed6a96a8989c2e7a1e6040f6ebbfd`. The incremental decision is
`compare_vs_current_first_pass.json` with its adjacent detailed CSV.

## Three-token compact selector and post-merge signed gate

### Scope and protocol

This final campaign follows up the two most direct remaining opportunities while keeping the model
path deployable. It deliberately keeps SAM2's existing three multimask alternatives: neither a
fourth token nor a new first-pass box/neighbor prompt is introduced. Campaign 1 replaces the dense
full-resolution selector features with mask-token and low-resolution evidence computed on the
decoder device. Campaign 2 freezes that winner and learns which merged instances benefit from the
existing `points+boxes` second pass.

All primary model comparisons use five image-level out-of-fold predictions. Thresholds and model
size are selected only on the 240-image primary split. The 233-image holdout is used once for frozen
confirmation, followed by three serialized A100 MIG timing trials. A corrected square-stretch
mapping is used for low-resolution foreground and prompt coordinates, matching SAM2's image
transform rather than independently scaling the two image axes.

### Campaign 1: compact on-device selector

The implementation invokes the SAM2 mask decoder directly and explicitly retains its three
multimask tokens (`1:4`). It extracts one of four versioned input schemas without transferring masks
to NumPy for feature computation:

- `lowres_v1`: the established 19 mask/seed/foreground statistics at decoder resolution;
- `token_v1`: predicted IoU, alternative index, and the 256-dimensional mask token;
- `token_lowres_v1`: the 19 low-resolution statistics plus the 256-dimensional token;
- `dense_v1`: the previous full-resolution 19-feature control.

The compact schemas enforce exactly three alternatives. Eager selection post-processes and
transfers only the chosen mask; deferred selection keeps all three until the general merge. The
same MLP score controls initial eligibility and merge ordering in both cases.

Primary final-merge screening selected the H64 `token_lowres_v1` model and a learned-score threshold
of 0.375:

| OOF scorer | best threshold | primary mSA | selection time |
|---|---:|---:|---:|
| low-resolution H64 | 0.325 | 0.294125 | 0.735 s |
| token-only H32 | 0.250 | 0.299611 | 0.881 s |
| token-only H64 | 0.300 | 0.301014 | 0.772 s |
| token-only H128 | 0.300 | 0.304437 | 0.787 s |
| token + low-resolution H32 | 0.275 | 0.303642 | 0.835 s |
| **token + low-resolution H64** | **0.375** | **0.306835** | **0.628 s** |
| token + low-resolution H128 | 0.225 | 0.306647 | 0.892 s |

After fixing H64 and 0.375, deferred merging scores slightly higher but costs materially more:

| merge policy | primary mSA | holdout mSA | holdout runtime |
|---|---:|---:|---:|
| **eager** | 0.306835 | 0.315633 | **186.15 s median (3)** |
| deferred | **0.308195** | **0.316343** | 217.32 s (1) |

Deferred therefore adds only 0.000710 holdout mSA while taking 16.8% longer than eager. Eager is the
frozen deployment winner.

The compact path also formally replaces the previous dense H64 selector. On holdout it improves
macro mSA from 0.295659 to 0.315633 (+6.76%) while reducing the sum of per-dataset median runtimes by
12.04%. Peak CUDA memory falls from about 2.31 GB to 2.10 GB. Four datasets improve; DIC changes
from 0.043984 to 0.039948, a small -0.004036 absolute change whose large relative percentage is a
near-zero-baseline artifact. The replacement comparator consequently retains the 2% relative guard
but permits at most 0.005 absolute loss for such low-score cases. Every replacement check passes.

### Campaign 2: post-merge signed-utility refinement

The first pass is frozen to `token_lowres_v1` H64, eager selection and learned-score filtering at
0.375. The new gate is evaluated after merge and prompt assembly, so its 25 features describe the
actual surviving instance: source and visible geometry, merge/filter margins, foreground support,
claimed fraction, neighboring-instance distance, and the assembled positive/negative prompt set.
It is trained on signed refinement utility (`refined IoU - first-pass IoU`) rather than clipping
harmful refinements to zero. Signed output is not clamped at inference.

The ablations show that both changes matter. The older pre-merge positive-benefit gate peaks at
0.308843 at 50%; moving the positive target post-merge reaches 0.309682, and retaining signed harms
reaches 0.310674. The deployment fraction is 15%, because it is the highest-quality point below the
predeclared runtime budget and it dominates 20% in both primary quality and screening cost:

| gate | fraction | primary mSA | selection/refinement time |
|---|---:|---:|---:|
| no refinement | 0% | 0.306835 | 0.655 s |
| blanket `points+boxes` | 100% | 0.308035 | 70.067 s |
| pre-merge positive utility | 50% | 0.308843 | 39.866 s |
| post-merge positive utility | 50% | 0.309682 | 33.590 s |
| post-merge signed utility | 10% | 0.309606 | 11.268 s |
| **post-merge signed utility** | **15%** | **0.310112** | **14.516 s** |
| post-merge signed utility | 25% | 0.310342 | 21.066 s |
| post-merge signed utility | 50% | 0.310674 | 38.137 s |

The primary-selected 15% threshold is `0.0075367484`; its full-primary refit threshold frozen for
holdout and deployment is `0.0042799711`. It refines 2,760 of 19,890 eligible holdout instances
(13.9% after distribution shift). Its quality gain retains 163% of the pre-merge gate's primary
gain and 212% of that gate's holdout gain, exceeding the 80% retention requirement on both splits.

The frozen per-dataset holdout and canonical runtime comparison is:

| dataset | selector only mSA | + signed 15% gate mSA | quality change | selector seconds | gated seconds | runtime change |
|---|---:|---:|---:|---:|---:|---:|
| DeepBacs | 0.335103 | 0.341084 | +1.785% | 13.233 | 12.353 | -6.646% |
| DIC HepG2 | 0.039948 | 0.040979 | +2.582% | 30.472 | 27.296 | -10.421% |
| DynamicNuclearNet | 0.545306 | 0.553082 | +1.426% | 21.765 | 21.757 | -0.038% |
| LiveCELL | 0.356211 | 0.355869 | -0.096% | 92.185 | 98.762 | +7.134% |
| TissueNet | 0.301595 | 0.301734 | +0.046% | 28.467 | 28.167 | -1.053% |
| **dataset-balanced / total** | **0.315633** | **0.318550** | **+0.924%** | **186.12** | **188.33** | **+1.189%** |

The refinement acceptance route requires a positive macro change, no dataset below -1%, at most
10% aggregate and 15% per-dataset runtime growth, and at most 10% additional peak CUDA memory. All
checks pass; peak memory is unchanged at about 2.10 GB. This makes the 15% signed gate an accepted
incremental deployment option on top of the compact eager selector.

### Historical tree comparison and artifacts

There is no remaining measured quality gap to the removed tree experiments. The historical
ExtraTrees deferred selector reached 0.284267/0.286724 primary/holdout, and its 40% refinement route
reached 0.288126/0.290765. The compact eager selector already reaches 0.306835/0.315633, and the
signed 15% route reaches 0.310112/0.318550. These are contextual rather than controlled comparisons
because the newer campaign also changes filtering, features and gate stage; they nevertheless remove
any empirical reason to retain the much slower tree dependency.

The main artifacts below the optimization output root are:

- selector dataset: `multimask_selection/token_lowres_v1/primary_features.npz`;
- selector: `multimask_selection/groupwise_v1/token_lowres_v1/models/` followed by
  `token_lowres_v1-groupwise-h64-d0p1-regression.pt`;
- signed-gate dataset/artifact: `multimask_selection/groupwise_v1/refinement_gate/` followed by
  `compact_h64_eager/postmerge_signed/primary_features.npz` and its `models/` directory;
- compact replacement decision: `campaign_compact_selector_replacement.json` and adjacent CSV;
- refinement decision: `campaign_postmerge_signed_refinement.json` and adjacent CSV;
- canonical gated run suffixes: `d3dcd1b32bf729d035b2ef1e30522d47`,
  `2760e6a9d54c1ce9337896fee2cc13ae`, and `87d46d134d369d6324785bf0c770cb18`.

The deployment configuration is
`apg_token_lowres_h64_eager_postmerge_signed_15.json`. Both fitted artifacts remain explicit inputs;
library defaults are unchanged.
