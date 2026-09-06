# Targeted 2D APG optimization

## Outcome

The original point-placement, blanket-refinement and batching campaign below found no accepted
optimization. The end of this document records the later learned-multimask campaign and its compact deployment follow-up.
They add explicit opt-in model paths and leave all defaults unchanged.
In the original campaign, confidence-gated box refinement was the closest quality candidate.
It improved the dataset-balanced mSA by 1.74%, which was short of the required 5%.
It increased the total runtime by 21.86%.
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
python finetuning/v2/evaluation/optimization/benchmark_apg_optimization.py \
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
python finetuning/v2/evaluation/optimization/compare_apg_optimization.py \
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
python finetuning/v2/evaluation/optimization/benchmark_apg_optimization.py --ndim 2 --trial-id control-1
python finetuning/v2/evaluation/optimization/benchmark_apg_optimization.py --ndim 2 --config <candidate>.json

# Screening:
python finetuning/v2/evaluation/optimization/screen_apg_refinement.py --device cuda

# Comparison:
python finetuning/v2/evaluation/optimization/compare_apg_optimization.py --ndim 2 --target quality \
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
python finetuning/v2/evaluation/optimization/train_apg_multimask_selector.py --device cuda
python finetuning/v2/evaluation/optimization/train_apg_multimask_selector.py --single-mask --device cuda
python finetuning/v2/evaluation/optimization/train_apg_refinement_gate.py --device cuda \
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
`optimization/configs/apg_token_lowres_h64_eager_postmerge_signed_15.json`. Both fitted artifacts remain explicit inputs;
library defaults are unchanged.

---

## Campaign of 2026-09-03: generalization, candidate supply, refinement retune

Started 2026-09-03 on branch `apg-optim-fable` (from `dev` at `f9c2abb`). Everything below runs
on hvit_t joint/v2 `best` (`85fb099c…`). Implementation checksum of the epoch after the Phase 0
edits (evaluation plumbing, `training_extra` manifest subset, the volume hooks of the 3d campaign):
`d11e240452d84052916d0f16e1a1cfb1`; the campaign's job directories live under `<output root>/jobs/`, see
`CAMPAIGN_OPERATIONS.md`.

### Phase 0: two defaults, and what the accepted runs reproduce to

The accepted 2d runs (`campaign_postmerge_signed_refinement.json`) resolved to
`candidate_threshold=1.5, dt=0.25, max_overlap=0.15, sigma=0.5, min_candidate_size=4, min_size=50`,
while the per-model hvit_t defaults of commit 9fd3b57 resolve to `3.0 / 0.5 / 0.3`. Every campaign
configuration in `optimization/configs/` now pins the former; `apg_control_registry_defaults.json`
is the library default and `apg_control_campaign_defaults.json` the pinned one.

Reproduction on the current implementation, one trial each (quality only):

| configuration | primary | recorded | holdout | recorded |
|---|---:|---:|---:|---:|
| registry defaults (3.0 / 0.5 / 0.3) | 0.268121 | - | 0.265033 | - |
| campaign defaults (1.5 / 0.25 / 0.15) | 0.269264 | 0.269577 | 0.264336 | 0.264318 |
| accepted selector only (refit artifact) | 0.327071* | 0.306835 (OOF) | 0.314209 | 0.315633 |
| accepted selector + 15% gate | 0.330903* | 0.310112 (OOF) | 0.317531 | 0.318550 |

\* The recorded primary values of the learned configurations are out-of-fold screening numbers;
a refit artifact evaluated on the images it was fitted on is optimistic by construction, which is
what the primary rows show. The holdout is the honest comparison: the selector reproduces to within
-0.45% (0.314209 vs 0.315633) and the pinned campaign defaults to within 1e-5. The residual is
behaviour drift between the accepted runs' revision (`4224b5c9` on `apg-optim`, not an ancestor of
`f9c2abb`) and the current tree; the new controls are the reference from here on. The drift also
shows in the selector features: the stored `token_lowres_v1` OOF dataset no longer matches the
features the current tree regenerates (the screens' identity check refuses it, see E3a below). The
candidate sets themselves moved - 450 prompts regenerated against 471 stored on one LIVECell image,
441 against 438 on a TissueNet one - so the flow-density candidates, not only the features, differ
from the accepted runs' tree. The learned artifacts are therefore re-extracted and re-fitted on the
current implementation before any OOF-based screen. On the holdout the learned gain survives the drift: +18.9% (selector) and +20.1%
(selector + gate) over the campaign defaults, against +19.4% / +20.5% recorded.

### Epoch 2 (2026-09-03 05:00): empty alternatives no longer poison the selector features

An alternative whose mask is empty at both stability offsets has a 0/0 stability score. The learned
selector's feature path carried that NaN into the group's features, and `_apply_prompts` refused it:
the E2 extraction crashed on a DynamicNuclearNet image at `foreground_threshold=0.5`, and the E1
production run of the selector on `cvz_fluo` crashed the same way. `_apply_prompts` now maps that
stability to 0 (`torch.nan_to_num`) before the features are built; a finite run is unchanged.
Implementation checksum after the edit: `14800942c30b0c62ee919988feffd64a`. The 3d aggregates read
sibling run directories of one configuration across checksums and record whether they mixed, because
the C1 arrays were running through this boundary (the volume path does not touch the edited code).

### Epoch 3 (2026-09-03 05:03): harness-only edit

`benchmark_apg_optimization.py` learned to cap the `training_extra` counts at a dataset's pool size
(PUMA has 26 validation images). The file is part of the implementation checksum, so the checksum
moved to `26a1003788ea2825356b486da1496fd7` without any change to what a run computes; the C1 arrays
now span three checksums and their aggregates say so (`mixed_implementations`). From here on the eight
checksum files are frozen until the round's canonical runs are in.

### Phase 0 timing: the two control configurations on the holdout

Three serialized, bracketed trials on the session's 1g.20gb slice (`phase0_holdout_timing_controls`),
233 images each; the brackets are registry-default runs before and after.

| configuration | holdout mSA | seconds per trial | median |
|---|---:|---|---:|
| registry defaults (ct 3.0, dt 0.5, mo 0.3) | 0.265033 | 144.6 / 144.3 / 144.5 | 144.5 |
| campaign defaults (ct 1.5, dt 0.25, mo 0.15) | 0.264336 | 180.5 / 179.9 / 180.1 | 180.1 |
| brackets (registry defaults) | 0.265033 | 147.7, 144.5 / 144.3, 144.6 / 144.4, 151.9 | - |

The pinned campaign defaults cost 25% more than the library's per-model defaults for the same
holdout quality: the lower candidate threshold proposes more prompts. The brackets drifted by at most
5%, so the cost figures stand. The learned configurations are timed against these once the re-fitted
artifacts exist.

### E2, first result: the candidate supply saturates at the current threshold

The extraction over ten proposal settings (`candidate_threshold` 3.0 / 2.0 / 1.5 / 1.0 / 0.5 x
`foreground_threshold` 0.7 / 0.5) records, per image, the objects a prompt lands in (*seeded*) and
the objects some alternative matches at IoU 0.5 (*proposed*). Fraction of the primary objects:

| dataset | objects | seeded at ct 3.0 | ct 2.0 | ct 1.5 | ct 1.0 | ct 0.5 | proposed at 1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LIVECell | 17389 | 0.766 | 0.804 | 0.810 | 0.730 | 0.331 | 0.748 |
| TissueNet | 4011 | 0.847 | 0.906 | 0.911 | 0.886 | 0.489 | 0.866 |
| DynamicNuclearNet | 2592 | 0.976 | 0.983 | 0.984 | 0.984 | 0.963 | 0.963 |
| DeepBacs | 892 | 0.946 | 0.963 | 0.946 | 0.864 | 0.682 | 0.862 |
| DIC HepG2 | 490 | 0.649 | 0.798 | 0.820 | 0.816 | 0.792 | 0.429 |

(all at `foreground_threshold=0.7`; 0.5 moves each cell by at most 0.02, mostly up, for 10-30% more
prompts). Lowering the threshold does not buy candidates: below 1.5 the density components fuse, the
prompt count falls and with it the seeded fraction (LIVECell 0.33 at 0.5). The pinned 1.5 is already
the recall optimum of the ladder, and the objects it never seeds - 19% of LIVECell, 18% of DIC, 9% of
TissueNet - are a ceiling no threshold reaches. Any further recall has to come from a different
proposal mechanism, not from this threshold. The learned-threshold / overlap / size screen still runs
on the settings that keep the seeding (ct 1.0 / 1.5 / 2.0, both foreground thresholds), but its
headroom is bounded by the small seeded gains above.

### E1 result: the accepted selector does not generalize beyond the datasets it was fitted on

Production evaluation (`evaluate_automatic_segmentation.py --mode apg --skip_tuning`, test splits,
hvit_t joint/v2 `best`) on all 23 2d datasets; `*` marks the five datasets the selector and gate
were fitted on. mSA per dataset (relative change against the registry defaults in brackets):

| dataset | registry defaults | campaign defaults | selector only | selector + 15% gate |
|---|---:|---:|---:|---:|
| arvidsson | 0.5961 | 0.6018 | 0.4421 (-25.8%) | 0.4454 (-25.3%) |
| bitdepth_nucseg | 0.2751 | 0.2661 | 0.2442 (-11.2%) | 0.2413 (-12.3%) |
| cellbindb | 0.2282 | 0.2328 | 0.1999 (-12.4%) | 0.2036 (-10.8%) |
| cellpose_data | 0.3128 | 0.3112 | 0.2929 (-6.3%) | 0.2928 (-6.4%) |
| covid_if | 0.7661 | 0.7665 | 0.7427 (-3.1%) | 0.7401 (-3.4%) |
| cvz_fluo | 0.1875 | 0.1868 | 0.1941 (+3.5%) | 0.1947 (+3.9%) |
| deepbacs * | 0.4139 | 0.4067 | 0.3978 (-3.9%) | 0.3959 (-4.4%) |
| deepseas | 0.1342 | 0.1249 | 0.1588 (+18.4%) | 0.1594 (+18.8%) |
| dic_hepg2 * | 0.0187 | 0.0198 | 0.0423 (+126.3%) | 0.0425 (+127.4%) |
| dsb | 0.5178 | 0.5195 | 0.4779 (-7.7%) | 0.4805 (-7.2%) |
| dynamicnuclearnet * | 0.4491 | 0.4485 | 0.5303 (+18.1%) | 0.5402 (+20.3%) |
| hpa | 0.0000 | 0.0000 | 0.0076 (n/a) | 0.0083 (n/a) |
| livecell * | 0.3408 | 0.3366 | 0.3533 (+3.7%) | 0.3528 (+3.5%) |
| microbeseg | 0.1690 | 0.1724 | 0.1667 (-1.4%) | 0.1657 (-2.0%) |
| neurips_cellseg | 0.3603 | 0.3538 | 0.3356 (-6.8%) | 0.3363 (-6.7%) |
| omnipose | 0.3803 | 0.3620 | 0.3663 (-3.7%) | 0.3657 (-3.8%) |
| puma | 0.1206 | 0.1175 | 0.0924 (-23.3%) | 0.0918 (-23.9%) |
| segpc | 0.0282 | 0.0311 | 0.0215 (-23.5%) | 0.0225 (-20.2%) |
| tissuenet * | 0.2857 | 0.2856 | 0.3098 (+8.4%) | 0.3112 (+8.9%) |
| tnbc | 0.1100 | 0.1064 | 0.0589 (-46.4%) | 0.0601 (-45.4%) |
| usiigaci | 0.0895 | 0.0901 | 0.0999 (+11.7%) | 0.0994 (+11.1%) |
| vicar | 0.3357 | 0.3438 | 0.2821 (-16.0%) | 0.2870 (-14.5%) |
| yeaz | 0.7095 | 0.7067 | 0.7113 (+0.3%) | 0.7107 (+0.2%) |

Macros (equal weight per dataset):

| configuration | seen (5) | unseen (18) | all (23) |
|---|---:|---:|---:|
| registry defaults | 0.3016 | 0.2956 | 0.2969 |
| campaign defaults | 0.2994 (-0.7%) | 0.2941 (-0.5%) | 0.2952 (-0.6%) |
| selector only | 0.3267 (+8.3%) | 0.2719 (-8.0%) | 0.2838 (-4.4%) |
| selector + 15% gate | 0.3285 (+8.9%) | 0.2725 (-7.8%) | 0.2847 (-4.1%) |

The learned selector gains on every dataset it was fitted on (DynamicNuclearNet +18%, DIC HepG2
more than doubles from a near-zero base, LIVECell +4%, TissueNet +9%) and loses on ten of the
eighteen it was not: arvidsson -26%, tnbc -46%, puma -23%, vicar -16%, dsb -8%, cellbindb -12%,
neurips_cellseg -7%, cellpose -6%, bitdepth_nucseg -11%, segpc -24%. The three unseen datasets it
helps (deepseas +18%, usiigaci +12%, cvz_fluo +4%) are fluorescence nuclei or cells, close to the
training modalities. This is the same failure the 3d probe showed on C. elegans slices: the score
learned on five light-microscopy datasets encodes their appearance, not mask quality in general.
Consequences: the accepted configuration must not become a default; the +19% holdout figure is a
within-distribution number; E4 (pooling six more datasets, with a leave-one-dataset-out diagnostic)
is now the decisive 2d experiment. The gate adds +0.6% on the seen datasets and +0.2% on the unseen
ones on top of the selector, in line with its earlier +0.9%. The campaign defaults (1.5 / 0.25 /
0.15) are 0.6% below the per-model registry defaults on the test splits, so the earlier per-model
tuning did its job; the HPA split scores zero with every configuration (a modality the model does
not segment at all, nine protein-channel images) and is kept in the macros as a constant.

### E4 result: pooling six more datasets does not make the selector transfer

Selectors re-fitted on the current implementation (`token_lowres_v1`, H64), screened on the primary
images with image-level OOF predictions; the leave-one-dataset-out (LODO) rows score every dataset
with a model that never saw it. Balanced primary mSA at the best threshold; the campaign-default
baseline on this implementation is 0.269264:

| selector | training data | OOF (in distribution) | LODO (held-out dataset) |
|---|---|---:|---:|
| primary only | 5 datasets, 240 images | 0.3057 (t 0.375) | 0.2535 (t 0.25) |
| pooled | + yeaz, neurips_cellseg, deepseas, puma, tnbc, covid_if (157 images) | 0.2972 (t 0.40) | 0.2637 (t 0.25) |

In distribution the re-fitted primary selector reproduces the recorded 0.3068. Held out, both
selectors fall *below* the predicted-IoU baseline: the learned score does not beat SAM2's own IoU on
a dataset it has not seen, and pooling six more datasets softens that loss (0.2535 to 0.2637) without
turning it. The per-dataset correlations say the same: OOF 0.62-0.79 against LODO 0.08 (DIC HepG2),
0.32 (DynamicNuclearNet), 0.59-0.64 (LIVECell, TissueNet, DeepBacs) for the primary selector; the
pooled one holds 0.60-0.77 on the datasets closest to its training mix (yeaz, covid_if, LIVECell,
TissueNet) and 0.08-0.51 elsewhere. Together with E1 this settles the deployment question: the
selector is a within-distribution optimization. E4b asks which inputs carry the dataset identity -
the 256 mask-token dimensions or the 19 geometry statistics - by fitting both halves separately with
the same LODO protocol.

### Canonical holdout runs of the re-fitted artifacts (current implementation)

Three serialized, bracketed trials each on the session's 1g.20gb slice; comparator decisions in
`campaign2_selector_vs_registry.json`, `campaign2_selector_vs_campaign_defaults.json` and
`campaign2_gate15_vs_selector.json` beside the output root.

| configuration | holdout mSA | median seconds | vs registry defaults | vs campaign defaults |
|---|---:|---:|---:|---:|
| registry defaults | 0.265033 | 144.4 | - | - |
| campaign defaults | 0.264336 | 180.1 | -0.3% / +24.7% | - |
| re-fitted selector, t 0.375 | 0.321314 | 177.5 | **+21.2%** / +22.9% | **+21.6%** / -1.4% |
| re-fitted selector + 15% signed gate | 0.323727 | 191.1 | +22.1% / +32.3% | +22.5% / +6.1% |

Both learned configurations pass their gates on the holdout: the selector clears the +5% quality
bar by a wide margin (the runtime cap is waived by the >=10% all-datasets exception against the
registry defaults, and not even needed against the campaign defaults, where the learned filter is
marginally faster than predicted-IoU filtering), and the gate passes the incremental refinement route
(+0.75% macro, worst dataset runtime +14.4%, no dataset loss). Re-fitting on the current
implementation recovered what the drift had cost (0.3213 against 0.3142 with the old artifact). These
are within-distribution numbers; E1 above is what they are worth elsewhere.

### E3a result: the refinement round has nothing left to tune

Sixty-one configurations on the re-fitted selector and gate, primary OOF replay (job
`e3a_retune_refit_v2`), control 0.304733 (selector only):

| configuration | primary mSA | change | select seconds (240 images) |
|---|---:|---:|---:|
| ungated points+boxes, n8, single-mask head | 0.307909 | +1.04% | 76.8 |
| gated 15%, n6, mc 0.6, fo 0.15, single-mask | 0.307256 | +0.83% | 16.3 |
| gated 15%, n6, mc 0.7, fo 0.15 (current defaults) | 0.307029 | +0.75% | 16.3 |
| gated 15%, n4, any gate | 0.3066-0.3067 | +0.6% | 16.3 |
| any configuration with `multimasking=True` in the refinement pass | 0.3010-0.3035 | -1.2% to -0.4% | 16.3-79.0 |

Loosening the consistency gate from 0.7 to 0.6 is worth +0.0002; the negative count and the foreign
gate move the third decimal; refining every instance buys +0.3 points over the 15% gate for five
times the second-round time. Asking the refinement decoder for three masks and picking the best by
predicted IoU is the one change that hurts, so a learned head choice in the refinement pass (E3b) is
not pursued. E3 is closed: the accepted refinement defaults stand.

### E4b result: the mask tokens carry the dataset identity

Same protocol as E4, with the selector fitted on the two halves of `token_lowres_v1` separately
(`lowres_v1`: 19 mask, seed and foreground statistics; `token_v1`: predicted IoU, alternative index
and the 256 mask-token dimensions). Balanced primary mSA at the best threshold; baseline 0.269264:

| inputs | training | OOF | LODO |
|---|---|---:|---:|
| tokens + statistics (E4) | primary | 0.3057 | 0.2535 |
| tokens only | primary | 0.3043 | 0.2498 |
| statistics only | primary | 0.2954 | 0.2547 |
| tokens + statistics (E4) | pooled | 0.2972 | 0.2637 |
| tokens only | pooled | 0.2954 | 0.2576 |
| statistics only | pooled | 0.2925 | 0.2685 |

The tokens are the in-distribution gain (0.3043 against 0.2954 for statistics alone) and the
out-of-distribution loss (LODO 0.2498 against 0.2547). The statistics transfer - their held-out
correlations stay at 0.59-0.72 on every dataset but DIC, where the tokens' fall to 0.28-0.55 - but on
a dataset they have not seen they only reach the predicted-IoU baseline (0.2685 pooled LODO against
0.2693), never above it. No learned variant beats SAM2's own IoU score on an unseen dataset.

### Where this leaves 2d

1. The learned selector and gate are real within a dataset: +21% holdout mSA on the current tree,
   +8% on the seen datasets' test splits, both gates passed. They are not a general default: -8% on
   the eighteen unseen production datasets, and leave-one-dataset-out never beats predicted IoU.
2. The candidate ladder is exhausted (E2): the pinned threshold is the recall optimum and 9-19% of the
   objects are never seeded; the refinement round is tuned out (E3a).
3. The deployment that follows from 1 is per-dataset fitting inside the existing tuning protocol: a
   dataset with a validation split (`common.VAL_SPLITS`, thirteen of them) fits its own selector
   there, exactly as `parameter_search.py` already tunes its thresholds there, and a dataset without
   one keeps predicted IoU. That is the follow-up this campaign hands over; it needs no new method.

### E2, second result: a slightly higher candidate threshold with a looser merge

The screen (`screen_apg_candidate_supply.py`, job `e2_train_screen_v2`): one selector pooled over
six proposal settings (ct 1.0 / 1.5 / 2.0 x fg 0.5 / 0.7, image-level OOF), then every setting
replayed against learned thresholds 0.25-0.60, `max_overlap` 0.15 / 0.3 / 0.5 and `min_size` 25 / 50.
Balanced primary mSA:

| configuration | mSA | prompts (240 images) | objects seeded / proposed / scored / merged of 25374 |
|---|---:|---:|---|
| **ct 2.0, fg 0.7, t 0.35, mo 0.3, ms 25** | **0.3115** | 34632 | 21406 / 18235 / 17215 / 17391 |
| ct 2.0, fg 0.7, t 0.30, mo 0.3, ms 25 | 0.3113 | 34632 | 21406 / 18235 / 17358 / 17533 |
| ct 2.0, fg 0.5, t 0.40, mo 0.3, ms 25 | 0.3106 | 36388 | 21526 / 18358 / 17081 / 17255 |
| ct 2.0, fg 0.7, t 0.35, mo 0.15, ms 25 | 0.3102 | 34632 | 21406 / 18235 / 16925 / 17104 |
| ct 1.5, fg 0.7, t 0.375, mo 0.15, ms 50 (the accepted setting) | 0.3046 | 48240 | 21536 / 18208 / 16466 / 16699 |

Three small effects add up to +2.3% over the accepted setting with 28% fewer prompts: candidate
threshold 2.0 seeds as many objects as 1.5 with far fewer prompts (the extra prompts at 1.5 are
duplicates the merge then has to reject), `max_overlap` 0.3 keeps 700 more objects through the merge,
and the size floor of 25 keeps the smallest ones. The learned filter's threshold moves from 0.375 to
0.35 for the pooled selector. The setting is confirmed on the holdout with three bracketed timing
trials against the re-fitted selector (`holdout_timing_e2_winner`).

### E2 holdout confirmation: accepted on the efficiency route, not the quality route

Three serialized bracketed trials on the local 1g.20gb (`holdout_timing_e2_winner`, checksum
`26a1003788ea…`, pooled selector `…-pooled6.pt`, config `apg_e2_winner.json`). Holdout balanced mSA
0.325675 in all three trials (deterministic) against 0.314209 for the re-fitted selector-only trials and
0.265033 for the campaign-defaults brackets, which did not drift (144-162 s; the first bracket carried the
warm-up). The comparator's equal-weight macro over the five datasets is 0.3213 → 0.3257 for the selector.

| comparison (`compare_apg_optimization.py`) | macro mSA | macro runtime | verdict |
|---|---:|---:|---|
| vs re-fitted selector-only, `--target quality` | +1.4% | -18.5% | rejected (needs +5%) |
| vs re-fitted selector-only, `--target efficiency` | same | same | accepted |
| vs campaign defaults, `--target quality` | +23.2% | -19.7% | accepted |

Per dataset against the re-fitted selector: deepbacs +4.2%, dic_hepg2 +8.3%, livecell +1.8%, tissuenet
−0.1%, dynamicnuclearnet −0.5%; runtime −7% to −28% on every dataset (fewer prompts through the SAM2
decoder and the merge). The E2 setting is therefore a strictly better operating point for the learned
selector: it does not clear the +5% quality bar on its own, it passes the efficiency gate (every dataset
≥ 5% faster, no dataset loses more than 0.5%), and against the defaults it is accepted with the same
margin as the selector plus the runtime saving. Decision files at the output root:
`e2_winner_vs_refit_selector_quality.json`, `e2_winner_vs_refit_selector_efficiency.json`,
`e2_winner_vs_campaign_defaults.json`. It inherits the E1 caveat (the selector inside it is
within-distribution); the proposal-side changes (ct 2.0, mo 0.3, ms 25) are selector-agnostic and could be
re-screened with the plain predicted-IoU filter for the generalist setting.

### Direction change (2026-09-03, 08:30): only what generalizes counts

The user's decision after the E1/E4/E4b results: per-dataset selector fitting is not an option, and neither
is any dataset-specific mode. An APG change is acceptable only if it improves consistently on all datasets,
or improves enough on most that minor regressions are tolerable. The hand-over recommendation in "Where this
leaves 2d" is withdrawn. The suspected cause of the generalization failure is the embedding-derived input
(the 256 mask-token dimensions); the next round therefore returns to generic inputs (SAM2's predicted IoU
and stability, the low-resolution mask statistics) and treats out-of-domain generalization as a
first-class development metric.

### G campaign: a selector that transfers (design)

**What E4b already tells us.** With the 19 generic statistics alone, pooled over 11 datasets, the
leave-one-dataset-out (LODO) mSA on the primary images is 0.2685 against 0.2693 for predicted IoU: the
statistics stop the damage but do not yet gain. The in-domain OOF gain of the statistics is +9.7%
(primary) / +8.6% (pooled). So the question is not whether tokens hurt (they do) but which generic
formulation carries a gain across a dataset boundary.

**Development corpus and metrics (the design change).** The eleven 2d datasets with a legal validation
split (`common.VAL_SPLITS`): the five primary datasets (240 images) and the six `training_extra` datasets
(157 images: yeaz, neurips_cellseg, deepseas, puma, tnbc, covid_if). Features for all of them exist
(`multimask_selection/token_lowres_v1/candidate_supply/primary_features_ct1p5_fg0p7.npz` and
`training_extra_features.npz`; the first 19 columns are the generic statistics, so no re-extraction).
Every candidate is scored twice on the same images and reported dataset-balanced over the eleven:

- in-domain: image-level out-of-fold (OOF) mSA;
- out-of-domain: LODO mSA, every dataset scored by a model that never saw it;

both relative to the predicted-IoU baseline on the same proposals (the baseline is replayed as a
candidate whose values are column 0 of the feature dataset, thresholds 0.5 / 0.6), with per-dataset
deltas. Acceptance for the development stage: LODO macro > baseline, no dataset below −2% in LODO, and a
retained in-domain gain. The twelve production datasets never used for training (arvidsson,
bitdepth_nucseg, cellbindb, cellpose_data, cvz_fluo, dsb, hpa, microbeseg, omnipose, segpc, usiigaci,
vicar) are opened once, at the end, for at most three shortlisted candidates through
`evaluate_apg_generalization.py`; the holdout timing trials follow for the final one.

**Factors (trainer options added today, `train_apg_multimask_selector.py`).**

1. Feature set (`--feature-set`): `lowres_all` (19), `iou_stab` (SAM2's own two scores and their
   product), `sam_scores` (the score-derived seven), `scale_free` (no absolute size or distance),
   `no_decoder` (no foreground-map features), `scale_free_no_decoder`.
2. Per-image standardization (`--per-image replace|append`): z-scoring every feature within its image
   removes dataset-level offsets and scales (an image-relative "is this mask better than its peers").
3. Model (`--model linear|mlp`, widths 16/64): a linear scorer cannot memorize dataset interactions.
4. Target (`--target iou|matched`): IoU regression against classifying IoU ≥ 0.5, the definition of a
   match that the metric uses and that does not shift with a dataset's difficulty.
5. Decomposition (`screen_apg_compact_selector.py --score-filter predicted_iou`): learned head selection
   with SAM2's own filter, to see which of the two decisions transfers.

**Stages.** G1 (CPU array, `cpu` preset): fit the grid with `--lodo` on the pooled eleven; the trainer now
writes LODO proxies per dataset (matched-AUC and selected IoU of the model and of predicted IoU) so the
grid can be pruned without GPU. G2 (GPU replay, `1g.10gb`): the top configurations by LODO proxy replayed
on both manifests (`--subset primary`, `--subset training_extra`) with OOF and LODO files and the
baseline candidate, thresholds 0.3-0.6. G3: shortlist to production generalization and holdout timing;
a winner with a feature subset or per-image standardization needs a small library hook (feature
selection / image statistics in `multimask_selection.load_feature_scorer`) before it can run end to end,
which opens a new checksum epoch. G4: carry the finding into 3d (the anchor-slice scorer and the
candidate filter must use the same generic inputs).

### G1 result: on the leave-one-dataset-out proxies, generic features only reach predicted IoU

48 fits (`g1_generic_grid`, ~2-15 min each on CPU), pooled over the eleven datasets with source-grouped
folds; proxy = matched-AUC (does the score rank IoU ≥ 0.5 masks above the rest) per held-out dataset minus
the same AUC for SAM2's predicted IoU. Calibration of the proxy on the token selector whose mSA we know:
its in-domain OOF proxy is +0.040 / +0.054 (primary / extra datasets; +9% / +8.6% mSA) and its LODO proxy
−0.044 / −0.011 (−2% mSA), so ±0.005 is noise level.

| model | inputs | LODO proxy, mean over 11 | worst dataset | datasets improved | in-domain OOF proxy |
|---|---|---:|---:|---:|---:|
| linear, matched target | lowres_all, per-image z appended | +0.005 | −0.040 (dynamicnuclearnet) | 8 / 11 | +0.009 |
| linear, IoU regression | scale_free | +0.003 | −0.026 | 6 / 11 | — |
| linear, matched | iou_stab, per-image z appended | +0.003 | −0.011 | 5 / 11 | — |
| linear, any other | any | −0.007 … +0.002 | −0.011 … −0.066 | 2-8 / 11 | ≤ +0.011 |
| MLP H64, any target | any generic set | −0.002 … −0.031 | −0.025 … −0.206 (tnbc, dic_hepg2) | 2-6 / 11 | — |

Two clear statements. (1) The 64-unit MLP overfits dataset interactions even on generic inputs: every
one of its 24 variants is negative out of domain, and its worst-dataset loss is 5-20 AUC points. Model
capacity, not only the token inputs, was part of the E1 failure. (2) A linear scorer on generic statistics
is safe (it never loses more than a few points anywhere) but, on a ranking proxy, adds nothing over
predicted IoU: SAM2's own IoU head is already a near-optimal dataset-independent ranking of its
alternatives. The proxy is threshold-free within a dataset, so it cannot see the one effect the per-image
variants are designed for, a threshold that calibrates itself per image; that is measured by the G2 replays
(mSA with one global threshold over eleven datasets), which therefore include the per-image-standardized
linear models and two unsupervised adaptive candidates: the per-image percentile of predicted IoU
(`iou_rank`) and its average with the raw value (`iou_blend`). Jobs: `g2_generic_replay` (15720930, wave 1:
plain linear / H64 per feature set), `g2b_perimage_replay_extra` and the local `g2b_perimage_replay_primary_local`
(wave 2). Summary: `multimask_selection/generalization_g1/models/g1_proxy_summary.csv`.

### G2 wave 2, primary manifest: per-image standardization and adaptive thresholds do not transfer either

Local replay (`g2b_perimage_replay_primary_local`, 240 images, 15 candidates × 7 thresholds; summary
`generalization_g1/g2b_primary_summary.csv`). Dataset-balanced mSA over the five primary datasets, each
candidate at its best threshold, against SAM2's predicted IoU at its best threshold (0.45, 0.2790):

| candidate | OOF (in-domain) | LODO (out-of-domain) | worst dataset (LODO) |
|---|---:|---:|---:|
| `iou_blend` (½ predicted IoU + ½ its per-image percentile; no learning) | +0.5% | same | −1.4% deepbacs |
| linear, iou_stab + per-image z, matched target | +0.1% | −0.3% | −2.5% deepbacs |
| `iou_rank` (per-image percentile of predicted IoU; no learning) | −0.8% | same | −3.4% dic_hepg2 |
| linear, iou_stab + per-image z, IoU regression | −1.9% | −2.1% | −4.4% deepbacs |
| linear, lowres_all + per-image z, matched | −1.9% | −6.7% | −72% dic_hepg2 |
| linear, sam_scores + per-image z, either target | −3.1% … −3.8% | −3.3% … −3.5% | −6% … −11% |
| linear, lowres_all + per-image z, regression | −2.4% | −6.1% | −44% dic_hepg2 |

Nothing in this wave beats predicted IoU by more than noise, in-domain or out. Image-relative
standardization does not supply the per-dataset calibration the token selector had learned: DIC HepG2, the
dataset where the token selector more than doubled mSA, is where the standardized statistics collapse
(−44% to −72% under LODO), because its images have few, low-scoring candidates and any within-image
normalization inflates them. The unsupervised adaptive filters bracket the effect: a pure per-image
percentile threshold is worse than an absolute one, the blend is +0.5%. Wave 1 (plain linear and H64 on
each feature set, and the predicted-IoU-filter decomposition) and the six extra datasets complete the
picture (`g2_generic_replay`, `g2_extra_library_replay`).

### G2 wave 1, primary manifest: the in-domain gain sits in the head choice, the out-of-domain loss in the filter

`g2_generic_replay` primary tasks (240 images; summaries `generalization_g1/g2_primary_{selection_score,predicted_iou}_summary.csv`).
Dataset-balanced mSA over the five primary datasets at each candidate's best threshold, against predicted IoU at
its best threshold (0.2790 with the learned-filter path at 0.45, 0.2783 with SAM2's filter at 0.5):

| candidate | learned filter, OOF | learned filter, LODO (worst dataset) | SAM2 filter, OOF | SAM2 filter, LODO (worst) |
|---|---:|---:|---:|---:|
| H64, lowres_all (19 statistics) | +4.8% | −4.3% (dic_hepg2 −84%) | +4.5% | −2.5% (tissuenet −4.7%) |
| H64, scale_free_no_decoder | +2.4% | −6.9% (dic_hepg2 −79%) | +2.9% | −4.9% (dynamicnuclearnet −8.7%) |
| H64, sam_scores | −0.6% | −3.3% (dic_hepg2 −14%) | +0.2% | −2.4% |
| H64, iou_stab | −1.1% | −1.8% (dic_hepg2 −34%) | −0.5% | −0.5% (deepbacs −1.8%) |
| linear, iou_stab | −1.0% | −1.8% | −0.9% | −0.9% |
| linear, lowres_all / sam_scores / scale_free_no_decoder | −2.6% … −4.6% | −4.5% … −5.4% | −2.4% … −4.5% | −4.5% … −4.8% |

Three things follow. (1) With SAM2's own filter (learned head choice only), the generic H64 model still
gains +4.5% in-domain, so most of the in-domain gain of any selector is the choice among the three masks, not
the filter. (2) The catastrophic out-of-domain losses (−80% on DIC HepG2) come from the learned *filter*: a
score fitted on ten datasets puts DIC's few low-scoring candidates below any global threshold. With SAM2's
filter the same model loses 2.5% out of domain, spread evenly. (3) Neither route yields an out-of-domain
gain: the head choice learned from generic statistics is −0.5% to −2.5% on a dataset it has not seen, and the
linear scorers are below baseline even in-domain. Caveat on this wave: every learned candidate peaked at the
lowest threshold of the 0.3-0.6 grid, so the learned-filter rows are upper-bounded by the grid edge; the
0.1-0.25 range is added (`g2_primary_lowthr` locally, and the pinned chain's tasks were extended) before the
learned-filter numbers are final.

### G2 over eleven datasets: no generic learned selector transfers by more than noise

Primary (240 images, pinned proposals) and training_extra (157 images, library-default proposals, each
dataset against its own baseline) joined; dataset-balanced mSA over the eleven datasets, each candidate at
its best threshold, predicted IoU at its best (0.3158). Tables:
`generalization_g1/g2_{w1_selection_score,w1_predicted_iou,w2_selection_score}_11datasets.csv` (with
per-dataset companions).

| candidate | filter | OOF (in-domain) | LODO (out-of-domain) | LODO: datasets up / below −2% / worst |
|---|---|---:|---:|---|
| H64, lowres_all | learned | +3.3% | −4.3% | 3 / 5 / dic_hepg2 −84% |
| H64, lowres_all | SAM2's | +2.8% | −1.2% | 5 / 4 / neurips_cellseg −8.8% |
| H64, iou_stab | SAM2's | −0.0% | −0.0% | 6 / 0 / deepbacs −1.8% |
| H64, scale_free_no_decoder | learned | +0.4% | −5.9% | 2 / 9 / dic_hepg2 −79% |
| linear, iou_stab | learned | −0.1% | −1.0% | 3 / 4 / deepseas −4.8% |
| linear, iou_stab + per-image z, matched target | learned (0.35) | **+0.7%** | **+0.4%** | 8 / 0 / deepbacs −1.6% |
| `iou_blend` (no learning) | learned (0.45) | +0.5% | same | 8 / 2 / tnbc −4.9% |
| `iou_rank` (no learning) | learned (0.30) | −0.9% | same | 2 / 3 / tnbc −8.6% |
| every other linear or H64 variant | either | −0.3% … −4.6% | −1.0% … −5.9% | ≤ 5 / 3-9 / −8% … −84% |

Verdict of the G campaign so far. On a dataset the scorer has not seen, nothing beats SAM2's predicted IoU
by more than noise: the best out-of-domain candidate, a three-feature linear scorer on predicted IoU,
stability and their product with their per-image z-scores, is +0.4% with no dataset below −1.6%, i.e. safe
and consistent but too small to matter, and its in-domain gain is +0.7%. The in-domain gains that remain
(+3% for the 19-statistics MLP) come with out-of-domain losses on a third of the datasets, and the MLP's
loss is largest exactly where the token selector had gained most (DIC HepG2), because its learned
threshold does not calibrate across appearance. The user's hypothesis is confirmed in one half: removing the
mask tokens removes the catastrophic transfer failure (−8% unseen macro in E1 becomes −1% with generic
inputs and SAM2's filter, or +0.4% with the tiny linear model). It does not hold in the other half: generic
inputs do not carry a transferable gain, because SAM2's IoU head is already the best dataset-independent
ranking of its own three masks that these statistics can express. The learned head-choice/filter line for
2d is therefore closed under the generalization rule, unless the pinned re-run (`g2p_pinned_replay`) or
the low-threshold sweep (`g2_primary_lowthr_local`) contradicts it; both are read next session. The 2d
levers that remain are non-learned and selector-agnostic: the E2 proposal-side setting (candidate
threshold 2.0, overlap 0.3, size floor 25) re-screened on the plain predicted-IoU path across all eleven
datasets, and the refinement round. For 3d the consequence is that the anchor filter (C3) must be judged
on its unseen-source LODO with generic inputs only, and that a plain-predicted-IoU anchor policy is the
reference to beat.

### G2 low-threshold sweep (primary): the grid edge hid nothing

`g2_primary_lowthr_local` added thresholds 0.10-0.25 for the wave-1 candidates on the learned-filter path
(joined table `generalization_g1/g2_w1_selection_score_primary_allthr.csv`). In-domain, the H64
19-statistics model moves from +4.8% (0.30) to +5.2% (0.25); out of domain every learned filter stays
negative at its own best threshold: H64 iou_stab −1.2% (0.15), linear iou_stab −1.2% (0.20), H64 sam_scores
−3.1%, H64 lowres_all −3.9% (dic_hepg2 −81%), the rest −4.5% to −6.7%. The wave-1 verdict stands.

### G1 on the pinned corpus: same ranking, same size

`g1p_pinned_grid` (24 linear fits done at 09:45; `generalization_g1/models_pinned/g1_proxy_summary.csv`).
With all eleven datasets on the pinned proposals, the linear matched-target scorers gain a little on the
LODO proxy: lowres_all + per-image z +0.009 (9 / 11 datasets, worst −0.036 dynamicnuclearnet), scale_free
+0.006 (9 / 11), lowres_all +0.006; iou_stab variants +0.001 (worst −0.013); regression targets −0.009 to
+0.004. In the token-selector calibration (+0.04 proxy ≈ +9% mSA) this is at most +2% out-of-domain mSA
before the filter threshold is applied globally; `g2p_pinned_replay` measures it. The clean pinned corpus
(`extract_extra_pinned_v2`: 135,447 extra alternatives with the full setting; `g1p_v2_pinned_grid`,
`models_pinned_v2/g1_proxy_summary.csv`) gives the same ranking and size: lowres_all + per-image z, linear,
matched +0.008 (9 / 11, worst −0.038), scale_free +0.007, iou_stab +0.002; the mixed-corpus conclusion did not
depend on the proposal mismatch.

### E2 on the plain path: the proposal-side setting is the first 2d change that travels

`e2plain_generalization` (single runs, 1g.10gb, trial `plain-1`; holdout rows added when its runs finish).
No learned component: candidate threshold 2.0, dt 0.25, sigma 0.5, min_candidate_size 4, foreground 0.7,
`max_overlap` 0.3, `min_size` 25, SAM2's predicted-IoU head and filter. Balanced mSA:

| manifest | registry defaults (3.0 / 0.6 / 0.3) | campaign defaults (1.5 / 0.5 / 0.15) | E2 plain, filter 0.6 | **E2 plain, filter 0.5** |
|---|---:|---:|---:|---:|
| primary (5 datasets, 240 images) | 0.2681 | 0.2693 | 0.2695 | **0.2796 (+4.3%)** |
| training_extra (6 datasets, 157 images) | 0.3433 | 0.3425 | 0.3437 | **0.3486 (+1.5%)** |
| primary seconds | 168 | 210 | 170 | 191 |
| training_extra seconds | 111 | 176 | 117 | 117 |

Per dataset against the registry defaults, E2 plain at 0.5: livecell +5.5%, tissuenet +5.8%,
dynamicnuclearnet −0.0%, deepbacs +6.7%, dic_hepg2 +24%; yeaz −0.4%, neurips_cellseg +1.4%, puma +9.2%,
tnbc +14.2%, covid_if +0.8%, deepseas −6.3%. Nine of eleven datasets improve, one is flat, one (deepseas,
0.115 → 0.108) loses. At filter 0.6 the setting is a wash, so the gain is the combination of the lower
candidate threshold with the 0.5 filter and the looser merge, and it costs 5-13% runtime against the
registry defaults (it is still faster than the campaign defaults it replaces). This is the first 2d change in
the campaign that meets the generalization rule on the eleven legal datasets, with one regression to weigh.
Holdout (233 images, single runs on 1g.10gb): registry 0.2650, campaign 0.2643, E2 plain 0.6 0.2659, **E2 plain
0.5 0.2742 (+3.5%)**; per dataset livecell +4.7%, tissuenet +4.2%, dynamicnuclearnet −1.3%, deepbacs +6.7%,
dic_hepg2 +40% (0.018 → 0.025); 165 s against 164 s for the registry defaults on that node. The holdout
reproduces the primary picture without re-tuning. Decision path: the twelve never-used production datasets
once (`e2plain_production`), three bracketed timing trials on the session GPU (`holdout_timing_e2_plain`), and
the v4 sign check.

### E2 plain, canonical holdout timing trials

`holdout_timing_e2_plain`: three serialized bracketed trials on the session 1g.20gb (checksum `26a1003788ea…`),
0.274202 in all three (deterministic); brackets 144-163 s (the second trial's opening bracket drifted +13%,
so its cost column is discounted). Comparator decisions at the output root (`e2_plain_vs_{registry,campaign}_{quality,efficiency}.json`):

| against | macro mSA | total runtime | worst-dataset runtime | verdict of the formal gates |
|---|---:|---:|---:|---|
| registry defaults (3 trials) | +3.5% | +1.2% | +5.9% | quality: below the +5% bar; efficiency: not a speedup |
| campaign defaults (3 trials) | +3.7% | −18.8% | −7.5% | quality: below +5%; efficiency: dynamicnuclearnet −1.3% exceeds the 0.5% allowance |

The formal gates were written for the learned-selector campaign and ask for +5%; the plain E2 setting is a
+3.5% change with no dataset below −5% on the holdout and, with the eleven-dataset development table above,
nine datasets up, one flat, one down 6% (deepseas). Under the generalization rule it is the one 2d candidate
worth carrying to the twelve never-used production datasets (`e2plain_production`, running) and the v4 sign
check; whether a 6% loss on one dataset out of eleven counts as a tolerable minor regression is the user's
call, and the deepseas loss should be looked at (its images are dense and low-contrast; the lower candidate
threshold and looser merge may admit fragments) before the setting is adopted. Per-image check: on the 40
deepseas images E2 plain predicts 8.3 objects per image against 6.3 annotated and 6.0 for the registry defaults,
so the loss is over-segmentation (false positives or split cells), not missed objects; a per-image object-count
or fragment-size guard is the natural fix if the setting goes forward.

### E2 plain on the production test splits: the gain does not carry over

`e2plain_production` (all 23 datasets, 10:27; `evaluate_apg_generalization.py report`,
`production_generalization/v2_best/generalization_summary.csv`). mSA against the registry defaults on the test
splits; `*` marks the eleven datasets whose validation splits chose the setting:

| dataset | registry | E2 plain | change | | dataset | registry | E2 plain | change |
|---|---:|---:|---:|---|---|---:|---:|---:|
| arvidsson | 0.5961 | 0.5240 | −12.1% | | livecell * | 0.3408 | 0.3593 | +5.4% |
| bitdepth_nucseg | 0.2751 | 0.2699 | −1.9% | | tissuenet * | 0.2857 | 0.2975 | +4.1% |
| cellbindb | 0.2282 | 0.2387 | +4.6% | | dynamicnuclearnet * | 0.4491 | 0.4500 | +0.2% |
| cellpose_data | 0.3128 | 0.3137 | +0.3% | | deepbacs * | 0.4139 | 0.4153 | +0.3% |
| cvz_fluo | 0.1875 | 0.2058 | +9.8% | | dic_hepg2 * | 0.0187 | 0.0243 | +30% |
| dsb | 0.5178 | 0.5252 | +1.4% | | yeaz * | 0.7095 | 0.7020 | −1.1% |
| hpa | 0.0000 | 0.0000 | — | | neurips_cellseg * | 0.3603 | 0.3591 | −0.3% |
| microbeseg | 0.1690 | 0.1903 | +12.6% | | puma * | 0.1206 | 0.1355 | +12.4% |
| omnipose | 0.3803 | 0.3927 | +3.3% | | tnbc * | 0.1100 | 0.1058 | −3.8% |
| segpc | 0.0282 | 0.0246 | −12.8% (−0.004 abs.) | | covid_if * | 0.7661 | 0.7677 | +0.2% |
| usiigaci | 0.0895 | 0.1031 | +15.2% | | deepseas * | 0.1342 | 0.1287 | −4.1% |
| vicar | 0.3357 | 0.3381 | +0.7% | | | | | |

Macros over all 23 datasets: +0.6% overall, +0.1% on the report's eighteen "unseen" datasets, +2.5% on the
five seen ones. On the twelve datasets that played no part in choosing the setting: eight up
(microbeseg +13%, usiigaci +15%, cvz_fluo +10%, cellbindb +5%, omnipose +3%), one flat (HPA at zero), three
down (arvidsson −12%, bitdepth_nucseg −2%, segpc −13% on a near-zero base). The +3.5% to +4.3% measured on the
validation splits of the eleven tuning datasets shrinks to about zero on the test splits, and arvidsson's 12%
loss is not a minor regression. Verdict under the generalization rule: the plain E2 setting is a **wash**, not
a win: it moves individual datasets by ±10% in both directions and the average not at all. It is not adopted.
The one useful reading is the same as for the selectors: the per-model registry defaults, tuned per dataset
family by `parameter_search.py`, are already close to the best single setting for hvit_t, and what remains is
dataset-specific, which is exactly what is not wanted.

### Where this leaves 2d under the generalization rule (10:25)

Nothing tried in 2d transfers: not the token selector (−8% unseen), not a generic-feature selector (linear at
baseline, MLP negative out of domain), not per-image calibration, not the proposal-side setting (a wash on the
test splits). SAM2's own predicted IoU with the per-model defaults is the best general-purpose 2d
configuration measured. The open, unexplored generalizable directions are structural rather than a score:
(a) making the merge itself smarter without learning (the overlap arbitration decides 5-10% of objects and
the E2 gain on the validation splits came mostly from `max_overlap` 0.3 with a smaller size floor); (b) a
test-time, label-free per-image adaptation of the filter threshold that is not a rank (the percentile
variants failed), for example an image-level agreement between the decoder foreground and the accepted masks;
(c) checking whether the joint v4 (geodesic) checkpoint changes the picture. None of these is a learned
selector, and each needs the eleven-dataset LODO-style protocol used here plus the production splits once.

### Experiments in flight (job ids)

- E1 domain generalization: `evaluate_apg_generalization.py`, 4 configurations x 23 production
  datasets, array `15715958`. Report: `production_generalization/v2_best/generalization_decision.json`.
- E2 candidate supply: `screen_apg_candidate_supply.py --stage extract` (job `15715936`), then a
  pooled OOF selector over 10 proposal settings and the threshold / overlap / size screen.
- E3a refinement retune on the frozen selector: `screen_apg_refinement.py --configs
  configs/apg_refinement_retune_screen.json` (61 configurations). The first submission (job
  `15715957`) failed the OOF identity check (`Regenerated proposal features differ from the OOF
  dataset`); it is re-run once the selector and gate are re-fitted on the current implementation.
- `g1_generic_grid` (15720917, 48 CPU tasks, 08:33): generic-feature selector grid with LODO proxies,
  artifacts under `multimask_selection/generalization_g1/models/`; smoke-tested locally first.
- `g2_generic_replay` (15720930, afterany on G1): OOF/LODO replay screens on the primary and training_extra
  manifests, learned filter (thresholds 0.3-0.6) and predicted-IoU filter (0.5/0.6), with the
  predicted-IoU baseline candidate; results under `compact_selector_screening/hvit_t/<ckpt>/<hash>/summary.csv`
  (metadata.json carries `subset` and `score_filter`).
- `g2b_perimage_replay_extra` (15721326, 1g.10gb) and `g2b_perimage_replay_primary_local` (session GPU, 09:12):
  wave 2 of the G2 replay, per-image-standardized linear models (lowres_all / iou_stab / sam_scores ×
  regression / matched, OOF and LODO) plus the unsupervised `iou_rank` and `iou_blend` candidates and the
  predicted-IoU baseline, thresholds 0.3-0.6, learned filter.
- The three training_extra replay tasks (wave 1 and 2) failed with "Proposal ... missing from the selector
  dataset": `training_extra_features.npz` was extracted by the trainer's plain path, i.e. with the library's
  per-model proposal defaults, not `PINNED_PROPOSAL_2D` (so E4's pooled selector was trained on two proposal
  distributions). The screen gained `--proposal-settings {pinned,library}`; resubmitted as
  `g2_extra_library_replay` (15721431, 3 tasks, 09:17). Each dataset is compared with its own baseline on the
  same proposals, so the deltas stay valid; a pinned re-extraction of training_extra is the clean follow-up.
- Pinned re-extraction of the six extra datasets on the session GPU (09:25, 167 s):
  `token_lowres_v1/candidate_supply/training_extra_features_ct1p5_fg0p7.npz` (110,571 alternatives against
  68,688 with the library defaults). With it, all eleven datasets share one proposal distribution.
- `g1p_pinned_grid` (15721581; 27 CPU fits: the six feature sets × per-image none/append × linear × iou/matched, plus
  three H64 references) → `g2p_pinned_replay` (15721582, afterany; primary and training_extra replays with
  `--proposal-settings pinned`, 7 linear/H64 candidates as OOF and LODO, the baseline, `iou_rank`, `iou_blend`).
  Artifacts under `generalization_g1/models_pinned/`. This is the clean version of G1/G2; read it with
  `summarize_generic_selector_grid.py` and `summarize_generic_replay.py <run dirs>`.
- `e2plain_generalization` (15721819, 12 runs on 1g.10gb, 09:31): the selector-agnostic E2 proposal setting on the
  plain predicted-IoU path (`configs/apg_e2_plain_t0p5.json`, `apg_e2_plain_t0p6.json`: ct 2.0, dt 0.25, sigma 0.5,
  min_candidate_size 4, fg 0.7, max_overlap 0.3, min_size 25, predicted-IoU filter 0.5 / 0.6) against the registry
  and campaign defaults on primary, training_extra and holdout (trial `plain-1`). Read with
  `benchmark_apg_optimization.py`'s summaries in `hvit_t/<ckpt>/<manifest>-<config>-<impl>/summary.csv`; the
  question is whether the +2.3% / −28% prompts of E2 survive without any learned component and across eleven
  datasets. If it does, it is the one 2d change that meets the generalization rule.
- `g2_primary_lowthr_local` (session GPU, 09:29): wave-1 candidates at thresholds 0.1-0.25 on primary.
- `e2plain_production` (15722179, 23 runs on 1g.10gb, 09:46): the plain E2 setting (`apg_e2_plain_t0p5.json`) on
  every 2d production test split, `production_generalization/v2_best`, result tag `e2-plain-t0p5`; registered in
  `evaluate_apg_generalization.py` CONFIGS. Read with `evaluate_apg_generalization.py --report`; for this
  configuration the strictly unseen set is the twelve datasets outside the eleven with validation splits.
- Correction (09:58): the 09:25 re-extraction of the extra datasets passed only the candidate and foreground
  thresholds, so dt / sigma / min_candidate_size stayed at the library values; the file is kept as
  `candidate_supply/training_extra_features_ct1p5_fg0p7_partialpinned.npz`, and `g1p_pinned_grid` /
  `models_pinned/` and the running `g2p_pinned_replay` primary task are a *partial-pinned* corpus (their
  training_extra screen failed on the prompt mismatch). The clean chain is `extract_extra_pinned_v2` (15722759,
  through `screen_apg_candidate_supply.stage_extract` with the full setting) → `g1p_v2_pinned_grid` (15722760,
  artifacts `models_pinned_v2/`) → `g2p_v2_pinned_replay` (15722761). A first version of that chain was cancelled
  before it ran because its script would have renamed the primary feature file; the fixed script extracts into
  `candidate_supply/training_extra_tmp/` and moves the result to `training_extra_features_ct1p5_fg0p7.npz`.
- 10:28: the clean chain's training_extra screen failed once because the baseline / `iou_rank` / `iou_blend`
  candidate files for that stem had been built from the partial extraction (110,571 rows vs 135,447); rebuilt
  from the clean file and resubmitted as `g2p_v2_extra_replay` (15723413). The primary task of
  `g2p_v2_pinned_replay` (15722761) is running. Join both with `summarize_generic_replay.py`.

## Generalization campaign of 2026-09-03/04: structural, label-free changes (plan `APG_2D_GENERALIZATION_CAMPAIGN_PLAN.md`)

Session 3, started 2026-09-03 ~19:30 on `apg-optim-fable`. Everything below runs on hvit_t joint/v2 `best`
(`85fb099c…`) against the per-model registry defaults (ct 3.0, dt 0.5, sigma 0.5, mcs 4, fg 0.7, filter 0.6,
`max_overlap` 0.3, `min_size` 50). Rule: a change counts only if it is up on ≥ 9 of the 11 development datasets
with no dataset below −2% relative (−0.005 absolute allowance near zero) and a balanced gain ≥ +2%, then once on the
production splits. Nothing is learned or tuned; every constant a change introduces is fixed a priori and reported.

### P0, production side: the dataset-level fusion ceiling is one dataset

From the existing production results (`production_generalization/v2_best`, APG registry defaults, test splits) and
the AIS results of the joint evaluation (`experiments/v2_joint_evaluation/results/*_auto_tuned.csv`, per-dataset
tuned AIS post-processing, so an optimistic AIS): `max(AIS, APG) − APG` per dataset, table in
`production_generalization/v2_best/p0_fusion_ceiling_production.csv`.

| | balanced mSA | relative to APG |
|---|---:|---:|
| APG registry defaults (23 datasets) | 0.2969 | - |
| max(AIS tuned, APG) per dataset | 0.3091 | +4.1% |
| the same without dynamicnuclearnet | | +0.4% |

AIS beats APG on only four datasets: dynamicnuclearnet (0.707 vs 0.449, the whole ceiling), deepseas (+8.6%),
cellbindb (+4.4%), usiigaci (+2.2%); everywhere else APG is ahead by 4-55%. Caveat: the AIS evaluation's checkpoint
is not recorded in its result files (assumed the same joint v2 `best`), and per-dataset tuning inflates it. The
dataset-level ceiling is formally above the plan's +3% demotion line but rests on one dataset; the per-object
ceiling on the development corpus (same checkpoint, AIS defaults) is the decisive P0 reading and comes from the
cache stage of `screen_apg_structural.py` (`oracle`). P1 stays in the plan; every hook was built in one epoch
anyway, so the order only affects the reading.

### Epoch 4 (2026-09-03 ~20:00): the structural hooks, default-off and bit-identical when off

Implementation checksum `26a1003788ea2825356b486da1496fd7` → `41abe8ca0cf86fadcf5d46ea183bb296`. One edit to the
checksum files (`automatic_prompt_generation.py`, `common.py` GENERATE_PARAM_KEYS, the benchmark's
IMAGE_DIAGNOSTICS), all opt-in through `generate()` / `propose()` / `select()`:

- `prompt_type` (P3a): `point` (default), `box`, `point_box`, `box_thin`. The box is the bounding box of the
  candidate's basin in the decoder's seeded watershed (`decoder_basins`: the same heightmap and markers the sparse
  post-processing finishes its instances with, the density components as markers); `box_thin` boxes only the
  candidates whose basin fills < 0.5 of the box, the rest keep the point. Every record keeps the point as its seed.
- `arbitration` (P2): `drop` (default), `decoder`, `euclidean`. `merge_by_score(arbitration="split", basins=...)`:
  a non-duplicate candidate keeps the contested pixels whose basin marker is its own prompt (`decoder`, basins from
  the decoder watershed seeded at the filtered records' prompts; Euclidean seed distance for pixels in no basin) or
  that lie closer to its seed (`euclidean`); a candidate winning < 50% of its area is "arbitrated away", an accepted
  mask that falls below 50% of its painted area (or `min_size`) is "split away". `max_overlap` still rejects
  duplicates; 1.0 is pure arbitration.
- `fusion` (P1): None (default), `fallback`, `conflict`, `both` (`fuse_with_instances`, AIS instances from
  `flow_instance_segmentation` with the model's registry defaults). Fallback adds a decoder instance when no
  accepted mask reaches IoU 0.5 with it and ≤ 50% of it is claimed, on its free pixels (≥ `min_size`). Conflict:
  a mask covering ≥ 2 instances (each ≥ 50% inside and ≥ `min_size` inside) is kept at stability ≥ 0.9, else
  replaced by the instances. Constants `FUSION_AGREEMENT_IOU` 0.5, `FUSION_STABILITY_THRESHOLD` 0.9,
  `FUSION_COVERAGE` 0.5, fixed.
- `recover_residual` (P3b): one interior point per uncovered connected foreground component ≥ `min_size` after the
  merge (and after refinement / fusion), SAM2's predicted-IoU choice, the same filter threshold, merged onto the free
  pixels (`merge_by_score(initial=...)`).
- Diagnostics: `fusion_fallback_added`, `fusion_conflicts`, `fusion_conflicts_split`, `arbitration_dropped`,
  `residual_prompts`, `residual_added` in `_last_generation_stats` and the benchmark's samples.csv; merge reasons
  gain "arbitrated away" / "split away".

Tests: 13 new unit tests in `test/test_v2_automatic_prompt_generation.py` (129 pass), `test/test_screen_apg_structural.py`.
Smoke test on four primary images: the registry defaults reproduce the `plain-1` control's mSA exactly; the options
run at 0.3-0.5 s per image on the session 1g.10gb.

Configs: `configs/apg_s_*.json` (all proposal parameters pinned to the registry values). Round 1 GPU arrays on
`grete:preemptible` 1g.10gb, trial `plain-1`: registry control, `s-box`, `s-point-box`, `s-box-thin`, `s-residual`
on primary (15732201), training_extra (15732202), holdout (15732206). Cache pass (`screen_apg_structural.py cache`)
on the session slice for the three manifests → `structural_2d/cache/<subset>/<identity>/`; the CPU screens
(`replay`, 29 variants: fusion modes + sensitivity a ∈ {0.4, 0.6}, s ∈ {0.85, 0.95}; arbitration decoder /
euclidean × `max_overlap` 0.3 / 0.5 / 1.0; box prompt types alone and with fusion / arbitration; adaptive
foreground-agreement threshold {0.4, 0.5, 0.6, 0.7} (P4)) and the `oracle` (P0 on the development corpus) follow.

### Round 1 (GPU, 2026-09-03 20:20): the registry control reproduces bit for bit; box prompts and residual recovery fail

Identity: the epoch-4 registry control (`plain-1`, 1g.10gb) equals the epoch-3 control on every image of the three
manifests (max |Δ mSA| = 0, no object-count difference), so the hooks are bit-identical when off on real data too.
Runtime of the control moved 168 → 174 s (primary), 111 → 118 s, 164 → 175 s across nodes: node drift, not code.

**P3a, box prompts** (`s-box`, `s-point-box`, `s-box-thin`; box = the candidate's decoder basin, see epoch 4):

| dataset | registry | box | point+box | box_thin | | residual (P3b) |
|---|---:|---:|---:|---:|---|---:|
| livecell | 0.3465 | −13.3% | −12.9% | −6.3% | | +1.1% |
| tissuenet | 0.2709 | −2.5% | −3.1% | −3.9% | | +1.5% |
| dynamicnuclearnet | 0.4485 | +6.9% | +4.6% | −0.0% | | +0.2% |
| deepbacs | 0.2498 | −11.0% | −7.0% | −5.3% | | −0.3% |
| dic_hepg2 | 0.0249 | −75.8% | −78.5% | −52.2% | | +1.1% |
| yeaz | 0.6986 | +1.2% | −0.2% | −0.2% | | −0.2% |
| neurips_cellseg | 0.2370 | −24.3% | −26.2% | −13.7% | | −0.1% |
| puma | 0.1306 | −13.4% | −4.7% | −9.9% | | +2.7% |
| tnbc | 0.1453 | −38.8% | −34.7% | −20.8% | | −1.3% |
| covid_if | 0.7334 | +2.4% | +2.1% | −0.1% | | +0.2% |
| deepseas | 0.1151 | −35.1% | −35.1% | −20.1% | | −4.9% |
| **balanced (11)** | 0.3091 | **−6.3%** (3 up / 8 down) | **−6.3%** (2 / 9) | **−4.7%** (0 / 11) | | **+0.1%** (6 / 5) |
| holdout balanced (5) | 0.2785 | −6.8% | −6.6% | −5.2% | | +0.5% |

The decoder basin is the wrong extent exactly where APG is weakest: on dense, touching data (tnbc, deepseas, puma,
neurips) the basins are fragments or fused neighbours and a box prompt commits SAM2 to that error, where a point
leaves it free. The gains on the separated, well-contrasted nuclei (dynamicnuclearnet +6.9%, covid_if +2.4%) do not
compensate. `box_thin` (box only for basins filling < 0.5 of their box) is down on all eleven datasets. **P3a is
rejected**; the same number of decoder calls, so no cost story either.

**P3b, foreground-residual prompting** (`s-residual`): 3425 residual prompts on primary (240 images) add 326 objects
(10%), 4970 add 189 on training_extra; +0.1% balanced, six datasets up by ≤ 2.7%, deepseas −4.9%, +7-15% runtime.
It does not reach the +2% bar and adds a regression: **rejected**. The recall the E2 diagnostic left on the table
(9-19% of objects never seeded) is not recoverable by prompting the uncovered foreground: what SAM2 returns there
mostly fails the predicted-IoU filter or duplicates an accepted mask. P3 is closed; P1/P2/P4 follow as CPU replays.

P5 preparation (20:28): staged checkpoint root
`<root>/v4_geodesic_checkpoints/joint_sam2_hvit_t_multi_gpu/best.pt -> joint/v4/checkpoints/joint_sam2_hvit_t_geodesic_multi_gpu/best.pt`
(checksum `5a729846c141daf73c27b24f52d8af4f`), same output root (run directories are keyed by the checkpoint checksum).
v4 controls (registry and campaign defaults, trial `v4-1`, 1g.10gb) on the three manifests: jobs 15732353/54/55.

### P0, development side (primary, 20:30): AIS matches objects APG misses, but the dataset-level ceiling is dynamicnuclearnet alone

`screen_apg_structural.py oracle --subset primary` (cache `structural_2d/cache/primary/7c7fe1d6…`, same checkpoint,
AIS = `flow_instance_segmentation` with the hvit_t registry defaults, no tuning; `structural_2d/oracle/primary/`):

| dataset | APG mSA | AIS mSA | max per image | APG recall | AIS recall | union recall | seeded | proposed (IoU ≥ 0.5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| livecell | 0.3465 | 0.2528 | 0.3491 | 0.512 | 0.424 | 0.568 | 0.807 | 0.683 |
| tissuenet | 0.2709 | 0.2157 | 0.2811 | 0.627 | 0.569 | 0.707 | 0.878 | 0.754 |
| dynamicnuclearnet | 0.4485 | 0.5544 | 0.5573 | 0.903 | 0.919 | 0.940 | 0.980 | 0.908 |
| deepbacs | 0.2498 | 0.1986 | 0.2680 | 0.585 | 0.620 | 0.704 | 0.974 | 0.700 |
| dic_hepg2 | 0.0249 | 0.0110 | 0.0287 | 0.074 | 0.071 | 0.112 | 0.782 | 0.218 |
| balanced | 0.2681 | 0.2465 | 0.2968 (+10.7%) | 0.564 | 0.497 | 0.624 | 0.842 | 0.709 |

Dataset-level fusion ceiling: 0 on four datasets, +23.6% on dynamicnuclearnet (the same picture as the production
side). The per-image oracle (+10.7%) and the object-level union recall (+6 points over APG) say the two segmentations
do complement each other object by object, so the per-object fusion (P1) has real headroom in principle. The recall
ceiling is confirmed on the current proposals: 84% of the objects are seeded and 71% have a proposal at IoU ≥ 0.5
(dic_hepg2 22%), against 56% matched after the filter and the merge; the filter and the merge, not the seeding, lose
the 15 points in between.

### P1, P2, P4 over the eleven development datasets (CPU replays, 2026-09-03 20:40): nothing passes the gate

`screen_apg_structural.py replay` on the cached registry proposals of primary (240 images) and training_extra
(157), joined by `report --subsets primary training_extra` (`structural_2d/reports/primary+training_extra/`; the
registry replay reproduces the canonical runs on all 397 images). Balanced mSA of the registry defaults over the
eleven datasets: 0.3091. Relative change per dataset (%):

| dataset | registry | fusion fallback | fusion conflict | fusion both | arb. decoder mo 0.3 | arb. decoder mo 0.5 | arb. euclid. mo 0.5 | arb. decoder mo 1.0 | adaptive fg | fixed 0.5 | fixed 0.4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| livecell | 0.3465 | −0.7 | −0.0 | −0.7 | 0.0 | −0.1 | +0.3 | −0.7 | +5.2 | +5.4 | +5.2 |
| tissuenet | 0.2709 | +1.3 | −0.0 | +1.3 | +0.2 | +0.7 | +0.9 | −0.4 | +4.9 | +4.5 | +4.9 |
| dynamicnuclearnet | 0.4485 | −0.8 | 0.0 | −0.8 | 0.0 | 0.0 | −0.1 | −0.3 | +0.8 | −0.1 | +0.8 |
| deepbacs | 0.2498 | +4.9 | −0.9 | +4.1 | −0.1 | −0.2 | +0.4 | −0.2 | +6.5 | +5.6 | +6.5 |
| dic_hepg2 | 0.0249 | −10.5 | −27.7 | −30.9 | −1.0 | −1.2 | −2.5 | −26.9 | +25.9 | +24.8 | +22.6 |
| yeaz | 0.6986 | −0.1 | 0.0 | −0.1 | 0.0 | 0.0 | −0.0 | +0.2 | −0.5 | −0.4 | −0.6 |
| neurips_cellseg | 0.2370 | −1.6 | +0.2 | −1.4 | 0.0 | +0.7 | +0.6 | −5.0 | +1.6 | +1.7 | +1.2 |
| puma | 0.1306 | −7.2 | −0.1 | −7.3 | −0.1 | −0.3 | −0.2 | +1.2 | +9.3 | +11.1 | +9.4 |
| tnbc | 0.1453 | −13.8 | +1.0 | −13.3 | +0.1 | +0.1 | +0.1 | −0.5 | +1.3 | +11.8 | +7.6 |
| covid_if | 0.7334 | −0.2 | 0.0 | −0.2 | 0.0 | −0.2 | −0.2 | −0.2 | −0.4 | −0.3 | −0.4 |
| deepseas | 0.1151 | −8.9 | −0.2 | −9.0 | −0.1 | −0.3 | −0.3 | −5.5 | −4.9 | −2.8 | −5.1 |
| **balanced** | 0.3091 | **−1.1** (2 up / 9 down) | −0.2 (2 / 6) | −1.3 (2 / 9) | **+0.0** (6 / 4) | +0.0 (5 / 6) | +0.1 (5 / 6) | −0.9 (2 / 9) | **+1.9** (8 / 3) | +2.3 (7 / 4) | +2.1 (8 / 3) |
| predicted / gt objects | 0.62 | 0.82 | 0.62 | 0.82 | 0.62 | 0.62 | 0.62 | 0.62 | 0.85 | 0.76 | 0.86 |

**P1, fusion.** The fallback adds 7,658 decoder instances to the 23,566 accepted masks (397 images) and matches 2,123
more objects, so three in four additions are false positives or fragments: dense, low-contrast data (tnbc −13.8%,
deepseas −8.9%, puma −7.2%, dic −10.5%) pays, deepbacs (+4.9%) and tissuenet (+1.3%) gain. The conflict rule fires
on 485 masks and splits 55 (stability < 0.9), for a loss on dic_hepg2 and nothing elsewhere. The sensitivity
variants (a 0.4 / 0.6, s 0.85 / 0.95) are all between −1.1% and −1.9%; s 0.95 splits more and loses more. The
object-level headroom of P0 (union recall +6-8 points) is real, but no label-free rule tells the good decoder
instances from the bad ones: the masks the SAM2 filter rejected were rejected for a reason. **Rejected.**

**P2, arbitration.** With `max_overlap` 0.3 the decoder arbitration changes 3 accepted masks in 397 images and the
result is the registry's to four digits (+0.01%); at 0.5 it is still ±1% per dataset; at 1.0 (pure arbitration)
7,789 masks are arbitrated or split away and the balanced score falls 0.9% (dic −27%, deepseas −5.5%, neurips −5%):
two prompts in one object then split it along the basin boundary, which is exactly what duplicate suppression
prevents. The Euclidean variant behaves the same. The overlap threshold is not where the 5-10% of contested objects
are decided in a way a label-free rule could improve on: the score order already agrees with the decoder's basins
where it matters. **A wash; rejected.**

**P4, adaptive threshold by foreground agreement.** The rule picks 0.4 on 228 of 240 primary images (0.5-0.7 only on
dic_hepg2): the Dice of the mask union with the predicted foreground is monotone in the threshold, so the "adaptation"
is a global retune to 0.4 in disguise, and its table is the fixed-0.4 table (+2.1% vs +1.9%). Both, and the fixed
0.5, gain 5-25% on livecell, tissuenet, deepbacs, dic, puma, tnbc and lose 3-5% on deepseas (over-segmentation, as
under E2): 7-8 datasets up, deepseas below the line, below the 9-of-11 rule, and a global-scalar retune is excluded
by the campaign rule anyway. E2 on the plain path already showed what happens to such a threshold on the production
splits (+3.5% development → +0.6% test, arvidsson −12%). **The threshold-adaptation line is closed.**

No variant of the 33 passes the protocol gate (≥ 9 of 11 up, no dataset below −2% / −0.005, balanced ≥ +2%), so no
candidate is carried to the production splits and no timing trials are run (the CPU cost of the options is
recorded: fusion +0.08 s per image for the AIS pass, decoder arbitration +0.01 s, both negligible on the GPU path).

### Holdout replay (233 images, 20:45): the same picture

`structural_2d/reports/85fb099c…/holdout/`; the registry replay reproduces the holdout control on all 233 images.
Balanced over the five holdout datasets (registry 0.2650): fixed 0.4 / adaptive +3.5% and fixed 0.5 +3.3% (4 up,
dynamicnuclearnet −1.0%, dic_hepg2 +35-42% on a 0.018 base); arbitration decoder / euclidean at mo 0.3-0.5 within
±0.2%; fusion fallback −0.1% (deepbacs +4.9%, dic −19%), conflict −0.2%, both −0.3%; box prompts −5 to −7%; mo 1.0
−1.0%. Nothing changes the eleven-dataset verdict.

### P5 blocker and fix (20:50): the v4 decoders could not be loaded by this environment

Every v4 run failed at model load: the joint v4 checkpoints (`initial_features=32` in `train_automatic`) have a
half-width UNETR3D decoder (`out_conv` 4×32, first block 256), while the installed `torch_em` 0.10.1 hardcodes the
width at 64 and silently swallows the `initial_features` argument that `UniSAM2` and `get_unisam2_model` pass
through. Fix in `micro_sam/v2/models/util.py` (not a checksum file, so epoch 4 stands): `UniSAM2` takes
`initial_features` explicitly and, when the built `out_conv` width differs, rebuilds the decoder tail at the
requested width from torch_em's own blocks (`_rebuild_decoder`, mirroring `UNETR3D.__init__`). The v4 geodesic
export then loads strictly with every tensor equal; v2 loads as before. This also means a `train_automatic` run in
this environment now really trains at the width it asks for. v4 controls resubmitted (`s5b_v4_controls_*`,
15732478/79/80) with the v4 cache (`s5b_v4_cache`, 15732481, primary + training_extra) for the replay-based sign
check of the negatives.

### P5, controls on joint/v4 geodesic (21:05): the checkpoint moves more than any setting did

Registry defaults, trial `v4-1`, 1g.10gb, same manifests (`hvit_t/5a729846…/`):

| manifest | v2 registry | v4 registry | change | v4 campaign defaults vs v4 registry | seconds v2 → v4 |
|---|---:|---:|---:|---:|---:|
| primary (5) | 0.2681 | 0.2955 | +10.2% | −0.1% | 174 → 144 |
| training_extra (6) | 0.3433 | 0.4634 | +35.0% | −1.1% | 118 → 85 |
| holdout (5) | 0.2650 | 0.2896 | +9.3% | +0.3% | 175 → 138 |

Per dataset (v4 vs v2, registry defaults, primary / training_extra): livecell +12.9%, tissuenet +6.8%,
dynamicnuclearnet +2.9%, deepbacs +28.5%, dic_hepg2 −42% (0.025 → 0.015), covid_if +1.5%, deepseas +52%,
neurips_cellseg +1.6%, puma 0.131 → 0.523, tnbc 0.145 → 0.419, yeaz −3.1%. Whether the v4 training saw puma / tnbc /
deepseas training splits is not recorded in the checkpoint's `init` (dataset objects only) and should be checked
before reading the training_extra jump as generalization; the primary and holdout gains (+9-10%) are on the five
datasets both checkpoints were tuned against. The campaign defaults against the registry defaults have the same sign
on v4 as on v2 (a wash within ±1%), so that fallback comparison of the plan stands. The structural variants are
replayed on v4 next (`s5b_v4_cache` → CPU replay) to check the sign of the negatives.

### P5, the structural variants on joint/v4 geodesic (21:15): every sign agrees

`structural_2d/reports/5a729846…/primary+training_extra/` (cache `s5b_v4_cache`, replays `s5b_v4_cpu`; the registry
replay reproduces the v4 controls on all 397 images). Balanced over the eleven datasets, v4 registry defaults 0.3871:
fusion fallback / both **−4.0%** (0 of 11 up; the decoder's instances are now clearly worse than v4's SAM2 masks),
conflict −0.0%; arbitration decoder / euclidean at mo 0.3-0.5 **+0.0 to +0.1%** (7 up / 1-2 down by ≤ 0.5% each:
noise), mo 1.0 −2.3%; box prompts −10%, box_thin −2.8%; fixed 0.5 / adaptive-without-0.4 **+0.9%** (6 / 5), fixed 0.4
+0.3% (4 / 7), fixed 0.7 −5.2%. The threshold gain of v2 (+2%) shrinks to +1% on v4 with more datasets down, which
is the checkpoint dependence the rule is meant to exclude. Every negative of the v2 screen keeps its sign on v4;
nothing turns positive.

### Where this leaves 2d after the structural campaign (2026-09-03, 21:20)

The four structural, label-free levers the plan named have all been built, screened on the eleven development
datasets, confirmed on the holdout and re-screened on the second checkpoint, and none passes the rule:

| lever | v2, 11 datasets | v4, 11 datasets | verdict |
|---|---:|---:|---|
| P1 AIS/APG fusion (fallback / conflict / both, fixed a = 0.5, s = 0.9) | −1.1 / −0.2 / −1.3% | −4.0 / −0.0 / −4.0% | rejected: the uncovered decoder instances are mostly wrong |
| P2 decoder-arbitrated merge (mo 0.3 / 0.5 / 1.0) | +0.0 / +0.0 / −0.9% | +0.0 / +0.1 / −2.3% | wash: score order already agrees with the basins |
| P3a box prompts from decoder basins (box / point+box / thin) | −6.3 / −6.3 / −4.7% | −10 / −10 / −2.8% | rejected: wrong extents where APG is weakest |
| P3b foreground-residual prompting | +0.1% (deepseas −4.9%, +8-15% runtime) | not run | rejected |
| P4 adaptive threshold by foreground agreement | +1.9% (= fixed 0.4; 8 / 11, deepseas −5%) | +0.4% | degenerate; threshold line closed |

The headroom is real but not reachable label-free: 84% of objects are seeded and 71% have a proposal at IoU ≥ 0.5
against 56% matched (v2 primary), and the AIS/APG union recall is 6-8 points above APG alone; every rule that tries
to collect it (adding decoder instances, arbitrating overlaps, re-prompting the residual, loosening the filter) adds
more false positives than objects on the dense datasets (tnbc, deepseas, puma, dic_hepg2) and only the separated
nuclei gain. Together with the previous campaign this closes the 2d APG optimization line for hvit_t under the
generalization rule: learned selectors, proposal-side scalars, per-image thresholds and structural changes all move
individual datasets by ±10% and the cross-dataset balance by nothing that survives the production splits. The one
large, general effect measured in this session is the checkpoint: joint/v4 geodesic with the unchanged registry
defaults is +9-10% on the five primary datasets (holdout confirmed) and 17% faster than v2; that, not the APG, is
where the next gain is.

No candidate reached the production splits or the timing trials; the 23-dataset production run was not opened.
Library: the hooks stay opt-in and default-off (checksum epoch 4, bit-identical when off on 630 images and two
checkpoints); `UniSAM2` now loads the half-width v4 decoders. Nothing committed.

### Refinement re-examination (2026-09-03, 21:40): the recommended `points+boxes` on eleven datasets and on v4

Asked whether the refinement strategies leave room, the accepted opt-in (`points+boxes`, p1-n6, mc 0.7, fo 0.15,
replace; `configs/apg_s_refine_pb*.json`, proposals pinned to the registry defaults) and its `boxes`-only ablation
were run for the first time on the six training_extra datasets and on joint/v4 geodesic (jobs `s6_refine_*`).
Relative mSA vs the registry control:

| dataset | v2 pb | v2 pb interior | v2 boxes | v4 pb | v4 pb interior | v4 boxes |
|---|---:|---:|---:|---:|---:|---:|
| livecell | −1.5 | −1.6 | +1.0 | −1.4 | −1.7 | +0.6 |
| tissuenet | +1.3 | +1.4 | +1.6 | +1.4 | +1.4 | +2.0 |
| dynamicnuclearnet | **+12.1** | +12.4 | +5.0 | **+8.7** | +8.6 | +1.4 |
| deepbacs | +3.8 | +4.1 | +1.7 | +2.9 | +1.7 | +0.6 |
| dic_hepg2 | +4.1 | +6.6 | −0.2 | +1.2 | +3.7 | +1.8 |
| yeaz | +1.1 | +1.4 | +2.0 | +1.4 | +1.6 | +1.8 |
| neurips_cellseg | −0.6 | −0.8 | +1.9 | +4.4 | +4.5 | +2.9 |
| puma | **−11.9** | −11.9 | −4.0 | **−4.5** | −4.5 | −0.7 |
| tnbc | −2.1 | −2.6 | −1.1 | +3.4 | +3.8 | +5.6 |
| covid_if | +0.9 | +1.0 | +0.6 | +0.5 | +0.6 | −0.3 |
| deepseas | −2.2 | −1.7 | −0.8 | −1.1 | +1.2 | −3.1 |
| **balanced (11)** | **+1.6** (6 up) | +1.8 (6) | +1.5 (7) | **+1.4** (8 up) | +1.5 (9) | +1.1 (8) |
| holdout (5) | +4.7 | +4.9 | +2.3 | +4.2 | +3.9 | +1.8 |
| runtime (primary) | +40% | +40% | +37% | +50% | +51% | +48% |

Readings. (1) The historical +4.2% / +4.9% was a five-dataset figure: the holdout reproduces it (+4.7% v2, +4.2%
v4), but the six extra datasets pull the eleven-dataset balance to +1.4-1.8%, with puma below the regression line
on both checkpoints (−11.9% v2, −4.5% v4), so the refinement fails the campaign rule as it stands, at +40-50%
runtime. (2) The gain is a size-bias correction: on the accepted first-round masks of dynamicnuclearnet 75% are
undersized (median predicted/GT area 0.70, pixel precision 0.99, recall 0.70, on v2 and v4 alike), and the box
prompt fills the object; deepbacs and v2 puma are oversized (median 1.3, precision 0.73), where the negatives cut;
livecell / tissuenet / yeaz are balanced (ratio 0.9-1.0), where the negatives only cost. (3) The geometric gates do
not select: 15,629 of 15,800 primary instances take the second-round mask (consistency vetoes 113, foreign 58), so
the mode is blanket replacement. (4) A label-free per-instance gate on decoder-foreground disagreement (uncovered
foreground in a 3-px ring, foreground inside the mask) does not target the bias: on dynamicnuclearnet the decoder
foreground agrees with the undersized masks (gate would select 2-4% of instances, Spearman with the area ratio
−0.09 / −0.36), and the sign of the correlation flips between datasets (`reports/instance_fg_disagreement_v2_v4.csv`,
`reports/matched_object_iou_v2_v4.csv`). (5) The first-round predicted IoU was tested as a gate in the first
campaign (least gain on the least confident masks) and the learned signed gate is the only selector that worked,
which the generalization rule excludes. What is left untested and label-free: protecting neighbours' first-round
pixels from the second-round repaint (the theft mechanism on dense nuclei) and drawing negatives only from touching
instances; both aim at the puma / livecell loss, neither at a new gain. Expected ceiling of such a probe: +2-3%
balanced at +40-50% runtime, on either checkpoint.

## Refinement campaign of 2026-09-03 (session 3, 21:45): label-free rules against the second round's losses

Plan `~/.claude/plans/glistening-dazzling-fountain.md`, approved 21:45. Develop on joint/v4 geodesic, confirm on v2;
the refinement stays an opt-in mode (runtime reported, no cap); production splits once, v4 only, for ≤ 2 candidates.
Rule as before: ≥ 9 of 11 development datasets up, none below −2% / −0.005, balanced ≥ +2%, on both checkpoints.

### Epoch 5 (22:00): three label-free refinement kwargs, 2d only

Implementation checksum `41abe8ca0cf86fadcf5d46ea183bb296` → `4fa97979b2aa4173e3c1d3fd38d00b66`
(`automatic_prompt_generation.py` and the benchmark's `IMAGE_DIAGNOSTICS`). All default-off, `refinement=None` and
the historical `points+boxes` unchanged (135 tests, 11 new):

- `protect_neighbours` (shared): the second-round mask is clipped to background + the instance's own first-round
  pixels before the gates, so a re-prompt grows into free space or shrinks but never onto a neighbour; a mask
  clipped to nothing keeps the first round. The consistency IoU sees the clipped mask (weakly relaxed), the
  foreign-overlap gate never fires for protected masks; `keep-if-better` still scores the unclipped mask. Stat
  `refinement_protected_pixels`.
- `negative_scope="touching"` (points): negatives only from instances within `touch_radius` (shared, 2 px,
  Euclidean; 1 = 4-connected contact only, 2 includes diagonal contact and one-pixel gaps) of the instance's mask,
  none without a touching neighbour; works for both negative sources by owner id. `_touching_instances` compares the
  label image with its shifted copies (six offsets at r 2), independent of the instance count. Stat
  `refinement_negatives`.
- `gate="isolated"` (+ `isolated_fallback=None|"boxes"`): only instances without a touching neighbour take the
  full second pass; the touching ones keep the first round or, with the fallback, are re-prompted with the box
  alone in their own batches. Stats `refinement_isolated_instances`, `refinement_fallback_instances`.
- All six image-only kwargs (`IMAGE_ONLY_REFINEMENT_KWARGS`) are rejected by the volume surface instead of being
  silently accepted (the previous derivation-by-subtraction would have accepted them); the one existing test
  assertion about the volume surface was updated accordingly.

Screen grid `configs/apg_r_refinement_screen.json` (11 entries, proposals pinned to the registry defaults, current
refinement defaults otherwise): none, pb, boxes, pb-protect, pb-touch, pb-touch-protect, pb-isolated,
pb-isolated-boxes, pb-isolated-boxes-protect, and the sensitivity rows pb-touch-protect-r1 / -r4 (reported only).
`screen_apg_refinement.py` now flattens the gate and refinement counters; `report_refinement_screen.py` joins the
screens of one checkpoint, applies the rule against `none`, checks `none` against the epoch-5 registry benchmark and
`pb` against the epoch-4 `apg_s_refine_pb` run image by image, and writes the per-dataset cost columns.

Jobs (22:03): screens `s7_refine_screen_v2` (15733313, primary + training_extra), `s7_refine_screen_v4` (15733314,
primary; the v4 training_extra screen runs on the session slice, log
`structural_2d/logs/screen_refine_v4_training_extra.log`); epoch-5 registry controls `s7_e5_controls_{v4,v2}_*`
(15733315-15733320).

### Screen (22:25): no label-free rule passes; the crowding gate is the one cost saver

Screens `refinement_screening/hvit_t/{5a729846…,85fb099c…}/<manifest>-9d4ab712…-4fa97979…/`, reports
`structural_2d/refinement_reports/<checkpoint>/primary+training_extra/`. Identity: the `none` entry equals the
epoch-5 registry benchmark and the `pb` entry the epoch-4 canonical `apg_s_refine_pb` run on every one of the 397
images, on both checkpoints; the epoch-5 registry controls equal epoch 4 on all six manifest / checkpoint pairs.
Relative mSA vs `none` (%), eleven datasets; v4 `none` = 0.3871, v2 `none` = 0.3091:

| dataset | v4 pb | v4 protect | v4 touch | v4 isolated | v4 isolated+boxes | v4 boxes | v2 pb | v2 protect | v2 touch | v2 isolated | v2 isolated+boxes | v2 boxes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| livecell | −1.4 | −1.5 | −0.4 | −0.5 | −0.2 | +0.6 | −1.5 | −1.7 | −0.4 | −0.8 | +0.0 | +1.0 |
| tissuenet | +1.4 | +1.4 | −1.2 | +1.3 | +2.4 | +2.0 | +1.3 | +1.0 | −0.3 | +0.8 | +2.1 | +1.6 |
| dynamicnuclearnet | +8.7 | +8.7 | −4.7 | +8.7 | +8.6 | +1.4 | +12.1 | +12.1 | −1.6 | +11.7 | +11.9 | +5.0 |
| deepbacs | +2.9 | +2.9 | +4.7 | +1.9 | +2.7 | +0.6 | +3.8 | +4.2 | +6.5 | +1.6 | +2.1 | +1.7 |
| dic_hepg2 | +1.2 | +1.2 | −2.0 | +1.2 | +2.1 | +1.8 | +4.1 | +5.1 | +5.0 | +7.1 | +1.0 | −0.2 |
| yeaz | +1.4 | +1.5 | +1.0 | +0.9 | +1.9 | +1.8 | +1.1 | +1.2 | +0.8 | +0.7 | +2.0 | +2.0 |
| neurips_cellseg | +4.4 | +4.4 | +1.7 | +3.3 | +4.2 | +2.9 | −0.6 | −0.4 | +0.2 | −0.7 | −0.2 | +1.9 |
| puma | −4.5 | −4.5 | +0.0 | −4.4 | −4.4 | −0.7 | −11.9 | −11.7 | +6.8 | −11.2 | −11.7 | −4.0 |
| tnbc | +3.4 | +3.4 | +0.8 | +3.5 | +3.4 | +5.6 | −2.1 | −2.3 | +1.9 | −1.8 | −2.1 | −1.1 |
| covid_if | +0.5 | +0.5 | −2.7 | +0.1 | +0.2 | −0.3 | +0.9 | +0.9 | −0.8 | +0.7 | +0.7 | +0.6 |
| deepseas | −1.1 | −1.0 | −1.7 | −1.1 | −1.0 | −3.1 | −2.2 | −3.0 | +1.8 | −1.9 | −2.1 | −0.8 |
| **balanced** | +1.4 (8 up) | +1.4 (8) | −0.5 (5) | +1.2 (8) | **+1.6** (8) | +1.1 (8) | +1.6 (6) | +1.6 (6) | +0.6 (7) | +1.4 (6) | **+1.8** (7) | +1.5 (7) |
| second-pass forwards / eligible | 1.00 | 1.00 | 1.00 | **0.39** | 1.00 (0.61 box-only) | 1.00 | 1.00 | 1.00 | 1.00 | **0.37** | 1.00 (0.63 box-only) | 1.00 |

Sensitivity rows (touch_radius 1 / 4 on pb-touch-protect): v4 −0.4 / −0.5%, v2 +0.8 / +0.5%; the radius does not
change the reading. Readings:

- **Neighbour protection does nothing.** It clips 82k pixels over 397 images (v4; 55k of them on LiveCELL) and moves
  no dataset by more than 0.1 points; puma keeps its −4.5% / −11.7%. The second round does not lose on dense data
  by stealing neighbours' pixels: the loss happens inside the instance's own extent.
- **Puma is not a crowding problem.** At radius 2, 91% (v4) / 73% (v2) of puma's instances are isolated, as are
  94% of dynamicnuclearnet's: the same geometry, opposite responses (+8.7% / −4.5%). `boxes` alone is −0.7% / −4.0%
  on puma, so the negatives cost 4 (v4) to 8 (v2) points there; on v2, touching-only negatives turn puma to +6.8%
  and tnbc / deepseas positive, but cost dynamicnuclearnet's gain on both checkpoints (−1.6% / −4.7%) and covid_if
  on v4 (−2.7%), for +0.6% (v2) / −0.5% (v4) balanced. Where the negatives help (undersized nuclei) and where they
  hurt (small, correctly sized nuclei) is a property of the object, not of its neighbourhood; no label-free rule of
  this campaign separates the two, and the sensitivity rows say the contact radius is not the missing knob.
- **The isolated gate is a cost story, not a quality one.** `pb-isolated` re-prompts 39% / 37% of the instances
  for 87% / 84% of `pb`'s gain (second-pass select time 55 s vs 125 s on v4, 41 s vs 94 s on v2, over 397 images);
  the box fallback for the touching instances brings the quality back to `pb`'s and slightly above (+1.6% / +1.8%,
  tissuenet, LiveCELL and yeaz up) at `pb`'s full cost. Neither passes the rule: 8 (v4) / 7 (v2) datasets up, puma
  below the line on both checkpoints, balanced below +2% on v4.
- The geometric gates still fire on ≤ 1% of the instances (replaced fraction 0.98-1.00 everywhere).

Verdict under the rule: **no candidate.** Two variants are carried to canonical runs for the record only:
`pb-isolated` (the Pareto point on cost) and `pb-isolated-boxes` (the best quality, equal to `pb` in cost), on
primary / training_extra / holdout for both checkpoints (`s8_isolated*`, 15733519-24 and the isolated-only runs), and
three serialized bracketed holdout timing trials on v4 of `pb`, `pb-isolated`, `pb-isolated-boxes`
(`s8_timing_v4_holdout`). The production splits are not opened.

### Canonical runs and timing (23:00): the screen reproduces; the crowding gate buys the second pass at a third of the cost

Canonical benchmark runs (`s8_isolated*`, single runs, 1g.10gb, trial `plain-1` / `v4-1`) reproduce the screen's
per-dataset numbers exactly on both checkpoints. Balanced relative change vs the epoch-5 registry control:

| configuration | v4 dev (11) | v4 holdout (5) | v2 dev (11) | v2 holdout (5) | second-pass forwards | v4 holdout runtime, 3 bracketed trials (median per dataset) |
|---|---:|---:|---:|---:|---:|---:|
| registry defaults | 0.3871 | 0.2896 | 0.3091 | 0.2650 | – | 138 s (brackets 138-154 s) |
| `points+boxes` (pb, current opt-in) | +1.42% (8 up, puma −4.5%) | +4.19% | +1.61% (6 up, puma −11.9%) | +4.71% | 100% | 206 s (+49%) |
| pb + `gate="isolated"` | +1.23% (8 up, puma −4.4%) | +3.87% | +1.36% (6 up, puma −11.2%) | +4.23% | 39% / 37% | **159 s (+15%; −23% vs pb)** |
| pb + isolated + `isolated_fallback="boxes"` | **+1.63%** (8 up, puma −4.4%) | **+4.52%** | **+1.84%** (7 up, puma −11.7%) | +4.74% | 100% (61% box-only) | 209 s (+51%) |
| … + `protect_neighbours` | +1.63% | +4.50% | +1.85% | +4.68% | 100% | not timed (screen: +3% select time) |

Per dataset the gate loses at most 0.9 points against `pb` (deepbacs, neurips), keeps dynamicnuclearnet (+8.7% /
+11.7%) and softens LiveCELL (−0.5% instead of −1.4%); the box fallback lifts tissuenet (+2.4% / +2.1%), yeaz
(+1.9% / +2.0%) and LiveCELL (−0.2% / +0.0%) above `pb` while keeping its gains elsewhere. The first timing trial ran
on a slower node (brackets 146 / 154 s against 138 / 139 s in trials 2 and 3, a 5% bracket drift) and is discounted
in the totals above, which are per-dataset medians over the three trials; LiveCELL carries the saving (89 s against
126 s per 80 images), dic_hepg2 and dynamicnuclearnet (94% isolated) cost the same as `pb`.

### Verdict (23:05)

Under the campaign rule (≥ 9 of 11 development datasets up, none below −2% / −0.005, balanced ≥ +2%, on both
checkpoints) **no refinement variant is a candidate**, so the production splits stay closed and `DEFAULT_REFINEMENT`
is unchanged. What the campaign established, for the record:

1. The second round's loss on dense nuclei and confluent cells is not pixel theft from neighbours: clipping every
   second-round mask to its own first-round extent plus background (82k pixels over 397 images on v4) moves no
   dataset by more than 0.1 points. The loss happens inside the instance, from the negatives (`boxes` alone loses
   0.7% on v4 puma against 4.5% with negatives) on small, correctly sized objects; the same negatives are the whole
   gain on undersized nuclei (dynamicnuclearnet: box alone +1.4%, box + point + negatives +8.7%). Which of the two
   an object is does not show in its neighbourhood (puma and dynamicnuclearnet are both > 90% isolated at radius
   2), in the decoder foreground (the earlier disagreement oracle), or in the first-round score (the first
   campaign): the label-free signals available to the second pass do not tell the objects that benefit from the
   objects that lose.
2. Touching-only negatives are the one rule that fixes puma (v2 +6.8%), and it costs dynamicnuclearnet on both
   checkpoints; the contact radius (1 / 2 / 4) does not change that.
3. The crowding gate is a genuine cost result: re-prompting only instances without a touching neighbour keeps
   87% (v4 dev) / 92% (v4 holdout) of the opt-in refinement's gain at 31% of its added runtime (+15% over the
   defaults instead of +49%). With the box fallback for the touching instances the quality is slightly above `pb`
   (+0.2 points on both checkpoints, LiveCELL no longer negative) at `pb`'s cost.

Both gate variants ship as opt-in kwargs (`refinement="points+boxes", refinement_kwargs={"gate": "isolated"}`,
optionally `"isolated_fallback": "boxes"`), documented in `DEFAULT_REFINEMENT`. Whether the recommended opt-in values
should move to the gate (cheaper, same puma regression) is a product decision, not one the rule makes; the library
defaults were not changed. Epoch 5 (`4fa97979…`) stands; the six registry controls are bit-identical to epoch 4.
Runs: screens `s7_refine_screen_*`, canonical `s8_isolated*`, timing `s8_timing_v4_holdout` (15733546), reports
`structural_2d/refinement_reports/<checkpoint>/primary+training_extra/`. Nothing committed.

## Visual check of the refinement and the end of the 2d APG optimization (2026-09-04)

### The visual check

`optimization/visualize_refinement_cases.py --dataset <name> --variant pb --n 5 --checkpoint v4` ranks a dataset's
images by the per-image mSA change of a screened refinement variant against the `none` control (read from the
refinement screen's `samples.csv`), recomputes the first round, the refinement prompts and the refined result for
the N largest improvements and the N largest decreases with the real model (the recomputed scores are checked
against the screen), and writes one six-panel figure per image: the image, the first-round masks, the refined
masks (ground truth as white outlines on both), the refinement prompts (positives, negatives, boxes) on the
first-round outlines, the pixel-level change (gained / lost / re-assigned, with the median mask-to-truth area ratio
of the matched objects before and after), and the per-object IoU change on the ground-truth footprints. Figures for
all eleven development datasets, v4 geodesic, `points+boxes` with the current defaults:
`<root>/structural_2d/visual/v4/<dataset>/pb/{improvements,decreases}/` (91 figures; `ranking.csv` beside them
holds every image's before / after score); puma also on v2 under `visual/v2/puma/pb/`.

Per-image direction of the refinement (v4, `pb` against the registry defaults):

| dataset | images | up | down | unchanged | median relative change | worst image | best image |
|---|---:|---:|---:|---:|---:|---:|---:|
| dynamicnuclearnet | 40 | 32 | 7 | 1 | +5.8% | −6.9% | +108% |
| neurips_cellseg | 40 | 27 | 11 | 2 | +3.6% | −23% | +134% |
| deepbacs | 30 | 16 | 13 | 1 | +2.2% | −26% | +80% |
| yeaz | 40 | 30 | 10 | 0 | +1.4% | −3.4% | +8.3% |
| tnbc | 6 | 4 | 2 | 0 | +1.0% | −1.5% | +16% |
| tissuenet | 40 | 23 | 16 | 1 | +0.6% | −9.9% | +38% |
| covid_if | 5 | 4 | 1 | 0 | +0.1% | −1.8% | +3.2% |
| deepseas | 40 | 11 | 14 | 15 | 0.0% | −52% | +100% |
| dic_hepg2 | 50 | 1 | 0 | 49 | 0.0% | 0 | 0 |
| livecell | 80 | 17 | 63 | 0 | −1.5% | −7.4% | +6.4% |
| puma | 26 | 4 | 22 | 0 | −5.1% | −20% | +7.0% |

### What the figures show

- **The segmentations are virtually unchanged.** On the puma images that lose 15-20% mSA the refinement adds
  3,000-4,000 pixels and removes 300-400 over 100-150 nuclei: a rim of one to two pixels per object, invisible at
  image scale. The first-round masks already cover 1.3-1.4 times the annotated area on those images (annotations
  drawn tight to the chromatin), the refinement takes the ratio from 1.35 to 1.42, and 124 of 154 objects drop in
  IoU. On the puma image that gains 7% the annotations are larger and the same rim helps. The same holds, with
  smaller amplitude, on LiveCELL (63 of 80 images down by a median 1.5%).
- **mSA is therefore not measuring what we care about here.** mSA averages the matching accuracy over IoU
  thresholds 0.5 to 0.95, so on small objects a one-pixel boundary shift moves many of them across several
  thresholds at once; a change no annotator would notice moves the score by ±20% on a dataset. The per-dataset
  swings that drove both campaigns of this session (puma −4.5% / −12%, dynamicnuclearnet +9% / +12%) are of this
  kind: boundary conventions of the annotation relative to SAM2's mask, not segmentation errors. The large relative
  values on deepseas, neurips and deepbacs (±50-130%) are single images with a near-zero baseline where one object
  crossing IoU 0.5 doubles or halves the score.
- **Where the refinement does something visible, it is small.** Even the best cases (dynamicnuclearnet, the boxes
  filling undersized nuclei) are boundary adjustments of a few pixels; the refinement never adds or removes
  objects (instance counts are identical before and after on every figure), so it cannot touch the recall that
  the diagnostics identified as the dominant loss.

### Decision

The user decided on 2026-09-04 to **stop the 2d APG optimization**: the visible effect of the refinement is
insignificant, and mSA, the score every gate of these campaigns was built on, responds to one-pixel boundary
conventions more than to segmentation quality on the small-object datasets that decided the campaigns. Joint/v4
geodesic becomes the default model independently of this work (its +9-10% over v2 with the unchanged registry
defaults is the one large, general effect measured in this session), and the code is to be simplified.

For the record, what this leaves in the tree, all opt-in and default-off, none of it having passed a gate:
the structural hooks of epoch 4 (`prompt_type`, `arbitration`, `fusion`, `recover_residual`, with
`decoder_basins`, `fuse_with_instances`, `residual_point_prompts`, the arbitrating `merge_by_score`), the
refinement kwargs of epoch 5 (`protect_neighbours`, `negative_scope`, `gate="isolated"`, `isolated_fallback`,
`touch_radius`, `IMAGE_ONLY_REFINEMENT_KWARGS`), the learned selector / gate path of the earlier campaigns, and the
refinement mode itself. The `UniSAM2(initial_features=...)` width fix in `models/util.py` is the one library change
of this session that is needed regardless (v4 checkpoints cannot be loaded without it). The evaluation harness
additions (`screen_apg_structural.py`, `report_refinement_screen.py`, `visualize_refinement_cases.py`, the
`apg_s_*` / `apg_r_*` configs) and the result caches under `structural_2d/` (5.2 GB) document the campaigns and can
be removed once the notes are considered sufficient.

Two lessons for whatever follows: any future evaluation of boundary-level changes should report a
boundary-tolerant or object-count-based measure next to mSA (per-object IoU distributions and the mask-to-truth
area ratio, as the visual tool does, or SA at a single tolerant threshold), and the visual check should come
before the screen, not after it: two campaigns' worth of gates were read from a score whose per-dataset movements
a handful of figures explained in an hour.
