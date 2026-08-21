# Targeted 2D APG optimization

## Outcome

No tested optimization met its acceptance gate, so the 2D APG algorithm and its defaults remain
unchanged. The closest quality candidate, confidence-gated box refinement of every instance, improved
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

**Superseded by [campaign 2](APG_2D_REFINEMENT_2.md)**, which confirmed these findings on a
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
