# Targeted 3D APG optimization

## Outcome

No tested optimization met its acceptance gate, so the 3D APG algorithm and its defaults remain
unchanged. The most promising quality setting, a lower three-level candidate-threshold ladder,
improved dataset-balanced mSA by 1.31%, short of the required 5%, and increased total runtime by
20.72%. The best efficiency trade-off, early stopping with patience 2, preserved every dataset within
the -0.5% quality guard and reduced total runtime by 6.56%, but four of five datasets were only
0.20-1.81% faster. It therefore failed the requirement of at least 5% speedup on every dataset.

The temporal-filter and anchor-coalescing implementations were experimental. Both were reverted after
their benchmark sweeps failed. Their serialized results are retained below the experiment output root.
No rejected setting was made a library default, and no persistent regression test was added for code
that is no longer present.

## Benchmark and decision rules

All experiments used only the 3D portion of manifest schema 5, checksum
`0f8fb67b3650a71f9f44b53037e89546`. The source data below
`/mnt/vast-nhr/projects/cidas/cca/data` was treated as read-only. One deterministic representative crop
was evaluated for each dataset:

| dataset | crop shape (Z, Y, X) |
|---|---:|
| C. elegans atlas | 32 x 140 x 512 |
| EmbedSeg | 12 x 512 x 512 |
| GoNuclear | 12 x 512 x 512 |
| CREMI | 12 x 512 x 512 |
| SNEMI | 12 x 512 x 512 |

The model was `hvit_t` with checkpoint `best`, checksum
`85fb099c4bb038fa0ab9bddd6151689e`. Runs were serialized on an
`NVIDIA A100-SXM4-80GB MIG 1g.20gb` device. The canonical baseline, candidate ladder, and early-stop
sweep used implementation checksum `ada109a965c5c71aa8ec0ac44ecfd411` at revision
`833d97ae91f8a5f4cc56a10ac79ff527ade8a3ca`.

The primary quality metric is the equal-weight mean of the five per-dataset mSA values. Relative, not
absolute, changes determine every gate:

- A quality optimization needs at least +5% macro mSA. At most two datasets may regress by more than
  5%. No dataset may take more than 10% longer unless macro quality improves by at least 10% and all
  five datasets improve.
- An efficiency optimization must be at least 5% faster on every dataset. Every dataset must keep mSA
  within -0.5% of baseline.
- Up to five configurations may be ranked within one hypothesis, but a setting is adopted only if it
  passes the corresponding gate.

Canonical baseline runtimes are medians of three complete trials. Candidate quality is deterministic
for a fixed implementation and configuration. The temporary temporal-filter and anchor-coalescing
branches each used a same-implementation control, so their timing was not compared across code
checksums. An apparent accepted candidate would have received two additional timing trials; no initial
candidate passed, so those confirmation runs were unnecessary. Every complete hypothesis sweep stayed
within 30 minutes.

The comparison program rejects incomplete runs and runs with mismatching dimensions, manifest, model,
checkpoint, implementation, or resolved parameters. Peak CUDA memory is reset and recorded per crop.

## Baseline

The current 3D defaults use candidate thresholds `(1.5, 10.0)`, score each candidate on its density-peak
slice, propagate up to 16 objects sharing one anchor slice in a pass, propagate through the complete
volume, and merge the resulting cropped masks by score.

| dataset | mSA | median seconds | proposed | scored | anchor slices | passes | frame steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| C. elegans atlas | 0.147862 | 107.912 | 127 | 106 | 18 | 18 | 432 |
| EmbedSeg | 0.645630 | 83.914 | 618 | 199 | 12 | 16 | 192 |
| GoNuclear | 0.492197 | 39.955 | 137 | 84 | 12 | 13 | 156 |
| CREMI | 0.095666 | 77.118 | 976 | 176 | 11 | 17 | 204 |
| SNEMI | 0.530935 | 146.633 | 1,065 | 360 | 12 | 29 | 348 |
| **Dataset-balanced / total** | **0.382458** | **454.532** | **2,923** | **925** | **65** | **93** | **1,332** |

Peak CUDA allocation was 11.15 GB (10.39 GiB). Candidate scoring removes most raw density components,
but propagation still dominates runtime: each pass visits every slice unless early stopping is enabled.

## Experiment 1: candidate-threshold ladders

### Hypothesis

A single low density threshold can merge nearby convergence peaks, while a high threshold can miss weak
objects. The default two-level ladder already combines thresholds 1.5 and 10. Three alternatives tested
whether an intermediate level could recover touching objects or a lower first level could improve recall:

- `(1.5, 5, 10)`
- `(1, 3, 10)`
- `(1.5, 5, 10, 20)`

### Results

| ladder | macro mSA | macro change | total runtime change | worst dataset runtime change | accepted |
|---|---:|---:|---:|---:|---|
| 1.5, 5, 10 | 0.383139 | +0.178% | +4.80% | +8.84% | no |
| 1, 3, 10 | 0.387483 | +1.314% | +20.72% | +40.67% | no |
| 1.5, 5, 10, 20 | 0.383139 | +0.178% | +6.24% | +8.86% | no |

Per-dataset relative mSA changes:

| ladder | C. elegans | EmbedSeg | GoNuclear | CREMI | SNEMI |
|---|---:|---:|---:|---:|---:|
| 1.5, 5, 10 | +2.288% | 0% | 0% | +0.024% | 0% |
| 1, 3, 10 | +2.073% | 0% | +2.457% | +4.572% | +1.054% |
| 1.5, 5, 10, 20 | +2.288% | 0% | 0% | +0.024% | 0% |

The lower ladder improves all affected datasets without a large quality regression, but the gain is far
below 5% and the extra candidates make SNEMI 40.67% slower. Adding threshold 20 changes neither mSA nor
the selected masks relative to `(1.5, 5, 10)` and only adds work.

**Decision:** reject all three ladders and retain `(1.5, 10.0)`.

## Experiment 2: temporal component filtering

### Hypothesis and implementation

Propagation can produce disconnected mask components away from an object's anchor. A temporary filter
kept the anchor component containing the prompt, then walked independently in both z-directions and kept
components touching a one-pixel dilation of the preceding mask. The `connected` variant fell back to the
unfiltered mask after a discontinuity; the `terminate` variant stopped that direction. Branches touching
the previous mask were retained rather than collapsed to one component.

Inline checks covered anchor-component selection, branch retention, connected fallback, directional
termination, and invalid modes. A same-implementation `none` control reproduced the canonical baseline
mSA exactly.

### Results

| filter | macro mSA | macro change | total runtime change | worst runtime change | accepted |
|---|---:|---:|---:|---:|---|
| connected | 0.383240 | +0.205% | +6.57% | +8.69% | no |
| terminate | 0.383375 | +0.240% | +6.65% | +8.88% | no |

`connected` changed only GoNuclear (+0.795%). `terminate` changed GoNuclear by +0.795% and CREMI by
+0.705%; the other datasets were unchanged. The cleanup cost is paid for every propagated object and
slice, while it alters very few final masks. Neither variant approaches the +5% quality target.

**Decision:** reject both filters and remove the experimental implementation.

## Experiment 3: propagation early stopping

### Hypothesis

The existing optional early-stop mechanism ends one propagation direction after every object in the pass
has been empty for a configured number of consecutive frames. Patience values 2, 3, and 4 tested whether
this could become an efficiency default without truncating real objects.

### Results

| patience | macro mSA change | total speedup | worst dataset speedup | worst mSA change | quality guard | accepted |
|---:|---:|---:|---:|---:|---|---|
| 2 | -0.038% | +6.56% | +0.20% | -0.490% | pass | no |
| 3 | -0.078% | +5.70% | +0.44% | -1.003% | fail | no |
| 4 | -0.235% | +4.50% | -0.12% | -3.033% | fail | no |

Per-dataset results for the best setting, patience 2:

| dataset | mSA change | speedup | skipped frame steps |
|---|---:|---:|---:|
| C. elegans atlas | -0.490% | +26.34% | 111 / 432 |
| EmbedSeg | 0% | +0.20% | 0 / 192 |
| GoNuclear | 0% | +1.81% | 5 / 156 |
| CREMI | 0% | +0.35% | 0 / 204 |
| SNEMI | 0% | +0.22% | 0 / 348 |

Patience 3 skipped 91 C. elegans and four GoNuclear frame steps; patience 4 skipped 72 and three. Neither
skipped any work on EmbedSeg, CREMI, or SNEMI. This explains why patience 2 can improve aggregate runtime
while failing the consistency gate. The non-monotonic C. elegans quality response also shows that a later
stop is not necessarily safer: different partial tracks interact during the final score-ordered merge.

**Decision:** retain `early_stop_patience=None` as the general default. Early stopping remains available
as an explicit workload-specific option.

## Experiment 4: anchor-slice coalescing

### Hypothesis and implementation

Objects on different anchor slices require separate propagation passes. A temporary `anchor_stride`
relocated a component's prompt to the nearest globally stride-aligned z-slice intersecting that density
component, with deterministic lower-z tie breaking. Components without a supported aligned slice kept
their original density-peak anchor. Strides 2 and 4 were tested against a same-implementation stride-1
control.

Inline checks covered relocation, global alignment, fallback for short components, point containment,
tie behavior, and invalid strides. The control reproduced canonical baseline mSA exactly.

### Results

| stride | macro mSA change | total speedup | worst dataset speedup | worst mSA change | accepted |
|---:|---:|---:|---:|---:|---|
| 2 | -0.504% | +3.02% | -1.18% | -4.739% | no |
| 4 | +0.054% | +4.69% | +2.13% | -1.883% | no |

Per-dataset relative changes:

| stride | metric | C. elegans | EmbedSeg | GoNuclear | CREMI | SNEMI |
|---:|---|---:|---:|---:|---:|---:|
| 2 | mSA | +1.142% | -4.739% | +2.453% | -3.268% | +1.943% |
| 2 | speed | +4.512% | +11.142% | -1.179% | +0.061% | -0.023% |
| 4 | mSA | -0.406% | +0.450% | -1.883% | +2.195% | +1.111% |
| 4 | speed | +2.133% | +10.929% | +7.249% | +3.135% | +3.079% |

The relocation changes which candidates pass 2D mask scoring, so speed and quality do not respond
monotonically to stride. The propagation telemetry illustrates the limited consolidation:

| dataset | control anchors / passes | stride 2 anchors / passes | stride 4 anchors / passes |
|---|---:|---:|---:|
| C. elegans atlas | 18 / 18 | 13 / 14 | 15 / 16 |
| EmbedSeg | 12 / 16 | 12 / 17 | 12 / 16 |
| GoNuclear | 12 / 13 | 11 / 11 | 11 / 11 |
| CREMI | 11 / 17 | 11 / 16 | 11 / 17 |
| SNEMI | 12 / 29 | 12 / 27 | 12 / 27 |

The 12-slice crops already use nearly every slice as an anchor, and relocation often leaves the number of
unique anchor slices unchanged. Pass count is also controlled by the 16-object batch limit; EmbedSeg at
stride 2 actually needs one extra pass after candidates are redistributed. Stride 4 is the best aggregate
trade-off but fails both the -0.5% per-dataset quality guard and the consistent +5% speed gate.

**Decision:** reject both strides and remove the experimental implementation.

## Conclusions and follow-up

The experiments identify three structural constraints on further 3D optimization:

1. Adding density thresholds increases propagation work much faster than quality. Further recall work
   should reject weak candidates before SAM2 propagation or target measured genuine misses rather than
   broadening the ladder globally.
2. Post-propagation connected-component cleanup is too late and too expensive for the small number of masks
   it changes. Temporal consistency is more promising as a propagation signal or merge score than as a
   per-object, per-slice cleanup pass.
3. Early stopping and anchor coalescing save work only on selected datasets. A general efficiency win needs
   a stopping or batching rule driven by each track's evidence, while preserving enough common behavior to
   improve every workload.

A useful next experiment would measure per-track mask confidence and extent during propagation, then stop
only individual tracks that have remained empty or unstable. The current pass-level early stop cannot save
work when one long-lived object keeps the entire batch active. For quality, candidate diagnostics should be
matched to ground-truth misses before adding prompts, so extra propagation is spent only on objects the
default ladder did not already recover.

## Reproducibility and artifacts

The output root is:

```text
/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/hvit_t/
  85fb099c4bb038fa0ab9bddd6151689e/
```

Run directories are named `<manifest checksum>-<config checksum>-<implementation checksum>`. Canonical
baseline config checksums are `27bebe0dc27b7778e40a8965bce7b60a`,
`59f98b4845ce5f055952a873ca40659f`, and `9ce531cd7258d8e9a14232a96c142de5` for trials 1-3.

| experiment | configuration checksums |
|---|---|
| candidate ladder | 1.5/5/10 `2716c6950768c4e682073fd83ef4de88`; 1/3/10 `b75c6f6e71cd8dd957fa9696a1518e62`; 1.5/5/10/20 `869d7f11143d9b04edaab4fa7c7732ee` |
| temporal filter | control `d31c0c8a412add51b608ade68151395e`; connected `000b9418ea189b6560cae7faa40241d9`; terminate `a421e7a988fc4c3dc023f77946467089` |
| early stopping | patience 2 `ab942765bd00484ce8dbb81390e20b64`; patience 3 `5ba67d34318ac4972a0ce22e25134aa8`; patience 4 `dbfc559606edfe058c76dc7a0948339f` |
| anchor coalescing | control `2f727c5c8cd69dd8c0ff4848d56c4279`; stride 2 `5d0594b5bb0d58bac40f1c5d68eb34a0`; stride 4 `1695e4036ce81b0439395ff92e242c63` |

The temporary temporal-filter implementation checksum is `dd29fc112bded2b9404ac7978e923504`;
the temporary anchor-coalescing checksum is `352853c41b0f219ad217c3be4da3f177`.

A canonical trial is run with:

```bash
python finetuning/v2/evaluation/benchmark_apg_optimization.py \
    --ndim 3 --trial-id baseline-1 --time-budget-minutes 30
```

A JSON configuration supplies a name and only the changed 3D parameters, for example:

```json
{
  "name": "early-stop-patience-2",
  "params_3d": {"early_stop_patience": 2}
}
```

Serialized runs are evaluated with repeated `--baseline-run` and `--candidate-run` arguments and an
explicit dimension:

```bash
python finetuning/v2/evaluation/compare_apg_optimization.py \
    --ndim 3 --target efficiency \
    --baseline-run /path/to/baseline-1 \
    --baseline-run /path/to/baseline-2 \
    --baseline-run /path/to/baseline-3 \
    --candidate-run /path/to/candidate \
    --output /tmp/apg-3d-decisions.json
```

The comparator writes its decision summary as JSON and every per-dataset delta as CSV beside it.
