# Targeted 3D APG optimization

## Outcome

**Propagation early stopping is now on by default with patience 2** (`early_stop_patience` in
`micro_sam/v2/automatic_prompt_generation.py`). It was adopted in experiment 5 as a deliberate,
documented exception to the efficiency gate, which it fails: only two of five datasets are more than
5% faster. The justification is that on 32-slice crops the setting is *output-preserving* — the
segmentation is bit-identical on all five datasets at patience 2, 3 and 4, down to the predicted
object count and seven decimals of mSA. The gate's every-dataset requirement exists to stop a
workload-specific win from being imposed on workloads it would cost something; here a non-benefiting
workload pays nothing measurable, while GoNuclear skips 33.6% of its frame steps and runs 30.1%
faster. The change also aligns the library with the annotator, whose volume widget has defaulted to
patience 2 since it was written.

No other tested optimization met its gate, and no other default changed. The most promising quality
setting, a lower three-level candidate-threshold ladder, improved dataset-balanced mSA by 1.31%,
short of the required 5%, and increased total runtime by 20.72%.

Experiment 3 first measured early stopping on the 12-slice crops and rejected it, concluding that it
"saves work only on selected datasets". Experiment 5 re-measured it on 32-slice crops and shows that
conclusion was half right: the *verdict* stands, but the *diagnosis* was wrong. C. elegans did not
dominate because it was the only dataset whose objects end; it dominated because it was the only crop
deep enough to have slices left to skip. It also had a quality cost there (up to -3.03% mSA) that
turns out to be an artifact of its crop being trimmed to 24 of a declared 32 slices, so the stop was
firing at an unannotated boundary rather than at the end of an object.

The temporal-filter and anchor-coalescing implementations were experimental. Both were reverted after
their benchmark sweeps failed. Their serialized results are retained below the experiment output root.
No rejected setting was made a library default, and no persistent regression test was added for code
that is no longer present.

## Benchmark and decision rules

Experiments 1-4 used only the 3D portion of manifest schema 5, checksum
`0f8fb67b3650a71f9f44b53037e89546`. Experiment 5 used the opt-in deep crop set, checksum
`f611a7125383e850798d0b5bf696f6f7`, selected with `--crops-3d deep`; both manifests are schema 5, and
both they and their runs are retained. The source data below `/mnt/vast-nhr/projects/cidas/cca/data`
was treated as read-only. One deterministic representative crop was evaluated for each dataset:

| dataset | crop shape, experiments 1-4 | crop shape, experiment 5 | propagated slices, 1-4 / 5 |
|---|---:|---:|---:|
| C. elegans atlas | 32 x 140 x 512 | 32 x 140 x 512 | 24 / 32 |
| EmbedSeg | 12 x 512 x 512 | 32 x 512 x 512 | 12 / 32 |
| GoNuclear | 12 x 512 x 512 | 32 x 512 x 512 | 12 / 32 |
| CREMI | 12 x 512 x 512 | 32 x 512 x 512 | 12 / 32 |
| SNEMI | 12 x 512 x 512 | 30 x 512 x 512 | 12 / 30 |

The deep set targets 32 slices with two unavoidable deviations. CREMI's tuning slab is exactly 32
slices, so its crop is the whole slab and there is only one z position. SNEMI stops at 30 because its
entire held-out range is 30 slices: training used slices 0-70 of a 100-slice volume, and reaching 32
would require training data.

The two crop sets differ only in depth and in one selection rule: a deep candidate is accepted only if
its first and last slice are annotated. The loader trims unannotated end slices, so without that rule
a declared depth is not the depth that reaches propagation — which is exactly what happened to the
C. elegans crop of experiments 1-4, whose declared 32 slices became the 24 in the table above. The
candidate grid is shared, so a deep result is attributable to depth.

The model was `hvit_t` with checkpoint `best`, checksum
`85fb099c4bb038fa0ab9bddd6151689e`. Runs were serialized on an
`NVIDIA A100-SXM4-80GB MIG 1g.20gb` device throughout, experiment 5 included. The canonical baseline,
candidate ladder, and early-stop sweep of experiments 1-4 used implementation checksum
`ada109a965c5c71aa8ec0ac44ecfd411` at revision `833d97ae91f8a5f4cc56a10ac79ff527ade8a3ca`.
Experiment 5 used `aef08d8026e3ddba8350370bc994019a` at revision
`c31494020103ec6dccb17bef3aca90f9699da735`; its baseline trial 1 records the parent revision
`2f67909ce9d3f410e75af9f471406b9f14d37c3b` because it ran minutes before that commit, which changed no
file that the implementation checksum covers.

The primary quality metric is the equal-weight mean of the five per-dataset mSA values. Relative, not
absolute, changes determine every gate:

- A quality optimization needs at least +5% macro mSA. At most two datasets may regress by more than
  5%. No dataset may take more than 10% longer unless macro quality improves by at least 10% and all
  five datasets improve.
- An efficiency optimization must be at least 5% faster on every dataset. Every dataset must keep mSA
  within -0.5% of baseline.
- Up to five configurations may be ranked within one hypothesis, but a setting is adopted only if it
  passes the corresponding gate, or if failing it is argued explicitly. Experiment 5 is the one such
  argument made so far, and it rests on the candidate being output-preserving rather than on a
  judgement about how much quality a speedup is worth.

Canonical baseline runtimes are medians of three complete trials. Candidate quality is deterministic
for a fixed implementation and configuration. The temporary temporal-filter and anchor-coalescing
branches each used a same-implementation control, so their timing was not compared across code
checksums. An apparent accepted candidate would have received two additional timing trials; no initial
candidate passed a gate outright, so those confirmation runs were unnecessary. Every complete
hypothesis sweep of experiments 1-4 stayed within 30 minutes. Experiment 5 could not: the deep set costs
5.7x the frame steps of the 12-slice set in total, and 6-10x on the four datasets whose depth actually
changed, so a single deep run takes 52-55 minutes and its six runs took about 5.3 hours of GPU time at a
150-minute per-run budget.

The comparison program rejects incomplete runs and runs with mismatching dimensions, manifest, model,
checkpoint, implementation, or resolved parameters. Peak CUDA memory is reset and recorded per crop.

## Baseline

The 3D defaults *as of experiments 1-4* use candidate thresholds `(1.5, 10.0)`, score each candidate on
its density-peak slice, propagate up to 16 objects sharing one anchor slice in a pass, propagate through
the complete volume, and merge the resulting cropped masks by score. Only the last of these still holds
unconditionally: experiment 5 changed `early_stop_patience` from `None` to 2, so a pass now ends once
every one of its objects has been empty for two consecutive slices. The baseline below therefore
describes the defaults these four experiments were measured against, not the current ones.

| dataset | mSA | median seconds | proposed | scored | anchor slices | passes | frame steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| C. elegans atlas | 0.147862 | 107.912 | 127 | 106 | 18 | 18 | 432 |
| EmbedSeg | 0.645630 | 83.914 | 618 | 199 | 12 | 16 | 192 |
| GoNuclear | 0.492197 | 39.955 | 137 | 84 | 12 | 13 | 156 |
| CREMI | 0.095666 | 77.118 | 976 | 176 | 11 | 17 | 204 |
| SNEMI | 0.530935 | 146.633 | 1,065 | 360 | 12 | 29 | 348 |
| **Dataset-balanced / total** | **0.382458** | **454.532** | **2,923** | **925** | **65** | **93** | **1,332** |

Peak CUDA allocation was 11.15 GB (10.39 GiB). Candidate scoring removes most raw density components,
but propagation still dominates runtime: with `early_stop_patience=None`, as here, each pass visits every
slice of the volume whether or not its objects have ended.

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

**Superseded by experiment 5.** Four of the five crops here are 12 slices deep and a pass costs one
frame step per slice, so this sweep could not distinguish "the objects of a pass do not end" from "the
crop has no slices left to skip". Re-measuring on 32-slice crops separates the two and reverses the
decision. Read the per-dataset table above as a measurement of crop depth, not of the mechanism: the
C. elegans quality regressions in particular do not reproduce at full annotated depth.

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

## Experiment 5: early stopping on 32-slice crops

### Hypothesis

Experiment 3 rejected early stopping because only C. elegans was meaningfully faster, and concluded that
the mechanism helps only selected datasets. But C. elegans was also the only crop deeper than 12 slices,
and a pass costs one frame step per slice, so the sweep confounded two explanations: that the objects of
a pass do not end, and that the crop has no slices left to skip. At patience 2, 111 of the 116 skipped
frame steps came from C. elegans and three of five datasets skipped none at all.

This experiment re-measures the same patience values on the opt-in deep crop set, where every dataset has
30-32 annotated slices, to find out which explanation was doing the work.

### Baseline

Medians of three complete trials on the deep crops. Every crop reaches propagation at its full declared
depth, so `propagated frame steps` equals `passes x depth` exactly on all five datasets.

| dataset | mSA | median seconds | proposed | scored | anchor slices | passes | frame steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| C. elegans atlas | 0.034672 | 198.472 | 209 | 149 | 31 | 31 | 992 |
| EmbedSeg | 0.450988 | 611.438 | 1,372 | 488 | 32 | 45 | 1,440 |
| GoNuclear | 0.510760 | 315.041 | 351 | 245 | 32 | 33 | 1,056 |
| CREMI | 0.148779 | 969.914 | 3,157 | 756 | 31 | 63 | 2,016 |
| SNEMI | 0.425515 | 1,076.169 | 2,243 | 898 | 30 | 70 | 2,100 |
| **Dataset-balanced / total** | **0.314143** | **3,170.859** | **7,332** | **2,536** | **156** | **242** | **7,604** |

Peak CUDA allocation was 11.15 GB, unchanged from the 12-slice runs: the peak is set by the encoder and
scoring phase, not by depth. The deep crops carry 5.7x the frame steps of the 12-slice set (7,604 against
1,332), because nearly every slice is an anchor and so the pass count grows with depth as well.

These absolute numbers are not comparable to the experiment 1-4 baseline. The crops are different, and
mSA generally falls because a deeper crop contains more objects; C. elegans falls furthest, from 0.147862
to 0.034672, because its deep crop is also drawn from a different source volume. Only candidate-against-
deep-baseline comparisons are meaningful here, which is what the gates use.

### Results

| patience | macro mSA change | total speedup | worst dataset speedup | worst mSA change | quality guard | gate |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.000% | +2.18% | -2.41% | 0.000% | pass | fail |
| 3 | 0.000% | +2.00% | -3.10% | 0.000% | pass | fail |
| 4 | 0.000% | +0.78% | -3.95% | 0.000% | pass | fail |

Per-dataset results for patience 2, with the baseline's trial-to-trial spread for scale:

| dataset | mSA change | speedup | baseline spread | skipped frame steps |
|---|---:|---:|---:|---:|
| C. elegans atlas | 0% | +7.04% | 3.5% | 121 / 992 (12.2%) |
| EmbedSeg | 0% | +0.30% | 3.8% | 55 / 1,440 (3.8%) |
| GoNuclear | 0% | **+30.15%** | 4.1% | 355 / 1,056 (33.6%) |
| CREMI | 0% | -2.41% | 4.0% | 64 / 2,016 (3.2%) |
| SNEMI | 0% | -1.69% | 3.8% | 11 / 2,100 (0.5%) |

Three results matter more than the gate verdict.

**The segmentation is bit-identical.** Not "within the -0.5% guard": identical. Every dataset returns the
same predicted object count and the same mSA to seven decimals, at patience 2, 3 and 4 alike. This is
what the mechanism should do — it stops only after every object of a pass has been empty for N
consecutive slices, and empty masks contribute nothing to the score-ordered merge — but experiment 3 did
not show it, because its C. elegans crop was trimmed to 24 of a declared 32 slices and the stop was
firing at an unannotated boundary rather than past the end of an object. That also retires experiment 3's
finding that a later stop is not necessarily safer: patience 2 now dominates 3 and 4 outright, skipping
strictly more work for the same output, so there is no safety argument for a longer patience on this data.

**Depth was not what held the other datasets back.** With 30-32 slices, EmbedSeg still skips 3.8% of its
frame steps, CREMI 3.2% and SNEMI 0.5%. They were never depth-starved; they are structurally unable to
stop. A pass carries up to 16 objects and these crops need 45-70 passes, so at least one object per batch
survives to the end of the volume, and for dense EM neurites that is true by construction rather than by
chance. Experiment 3's diagnosis was therefore wrong even though its verdict was right.

**The aggregate speedup fell, from +6.56% to +2.18%.** The deep total is dominated by CREMI and SNEMI,
2,046 of 3,171 seconds, and they save nothing; GoNuclear's 30% win applies to a 315-second base. The
12-slice headline was flattered by C. elegans being a large share of a small total.

One measurement caveat: the baseline's per-dataset spread across three trials is 3.5-4.1%, so the 5% gate
threshold sits close to the noise floor of a single candidate trial. The CREMI and SNEMI "slowdowns" are
noise around zero, and C. elegans's +7.04% is only twice the spread. GoNuclear's +30.15% is the only
per-dataset speedup that is unambiguous, and it is corroborated by its frame-step count rather than by
timing alone.

**Decision:** adopt `early_stop_patience=2` as the library default, as a documented exception to the
efficiency gate. The gate is failed on three of five datasets and is not waived lightly; the argument is
that its every-dataset requirement protects workloads that would pay for someone else's speedup, and here
the payment is provably zero rather than merely small. What a non-benefiting workload loses is bounded by
the timing noise, and what it computes is unchanged bit for bit.

Unlike the reverted experiments, this one changes code that stays, so it carries regression tests in
`test/test_v2_prompt_based_segmentation.py`: that propagation stops on the second consecutive frame in
which every object is empty, that the non-empty masks are identical to a full propagation's, that an
isolated dropped mask does not trigger a stop, and that the default is 2.

The residual risk is not visible in this benchmark. A stop needs *every* object of the pass to be empty
for two consecutive slices, which is already conservative, but SAM2 can drop a mask and recover it, so a
pass whose objects all vanish for two slices and then resume would be truncated. No pass on these five
crops does that, at any patience value: if one had, the segmentation would not have been bit-identical.
That is evidence about these crops, not a guarantee — the failure mode needs every object of a batch to
lapse together, which is likeliest in a sparse volume where a pass carries few objects. Workloads where
objects are expected to vanish and return should raise the patience, which is why it stays exposed as a
parameter rather than becoming a fixed constant.

## Conclusions and follow-up

The experiments identify three structural constraints on further 3D optimization:

1. Adding density thresholds increases propagation work much faster than quality. Further recall work
   should reject weak candidates before SAM2 propagation or target measured genuine misses rather than
   broadening the ladder globally.
2. Post-propagation connected-component cleanup is too late and too expensive for the small number of masks
   it changes. Temporal consistency is more promising as a propagation signal or merge score than as a
   per-object, per-slice cleanup pass.
3. Anchor coalescing saves work only on selected datasets. Early stopping does too, but experiment 5 shows
   the reason is not crop depth: a pass keeps running while any one of its up-to-16 objects is alive, so
   dense EM volumes cannot stop at any depth. It was adopted anyway because it is output-preserving, which
   makes an uneven benefit acceptable in a way an uneven quality trade-off would not be. The remaining win
   is per-track: stop each track when its own evidence runs out instead of waiting for the whole batch, and
   GoNuclear's 33.6% becomes the floor rather than the outlier.

A useful next experiment would measure per-track mask confidence and extent during propagation, then stop
only individual tracks that have remained empty or unstable. The current pass-level early stop cannot save
work when one long-lived object keeps the entire batch active, and experiment 5 quantifies the headroom:
EmbedSeg, CREMI and SNEMI need 45-70 passes each yet skip 0.5-3.8% of their frame steps, so almost all of
that work is spent propagating objects that have already ended. For quality, candidate diagnostics should be
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

Experiment 5 reuses the trial ids of experiments 1-4, so its configuration checksums are the same six
values listed above: the config identity covers the resolved parameters, dimensions and trial id, but not
the manifest. Its run directories therefore differ from their 12-slice counterparts only in the manifest
and implementation components, which is a check in itself that the parameters are identical. Baseline
trials use `baseline-3d-1`, `baseline-3d-2` and `baseline-3d-3`; the three candidates share
`early-stop-screen-1`.

| experiment 5 run | run directory |
|---|---|
| deep baseline, trials 1-3 | `f611a712…-{27bebe0d…, 59f98b48…, 9ce531cd…}-aef08d80…` |
| deep patience 2 / 3 / 4 | `f611a712…-{ab942765…, 5ba67d34…, dbfc5596…}-aef08d80…` |

A canonical trial is run with:

```bash
python finetuning/v2/evaluation/benchmark_apg_optimization.py \
    --ndim 3 --trial-id baseline-1 --time-budget-minutes 30
```

The deep crop set is opt-in, keeps its own manifest, and needs a larger budget. Building it first is
worthwhile because `--prepare-only` prints the depth each crop will actually propagate through, which is
the check that the declared depth survived the loader's trim:

```bash
python finetuning/v2/evaluation/benchmark_apg_optimization.py --crops-3d deep --prepare-only

python finetuning/v2/evaluation/benchmark_apg_optimization.py \
    --ndim 3 --crops-3d deep --trial-id baseline-3d-1 --time-budget-minutes 150
```

A stored manifest is checked against the active crop variant, so a deep run cannot silently read the
12-slice subset, or the reverse.

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
