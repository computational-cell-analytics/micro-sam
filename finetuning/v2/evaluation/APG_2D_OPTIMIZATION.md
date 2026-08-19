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
