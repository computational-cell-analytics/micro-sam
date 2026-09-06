# Targeted 3D APG optimization

## Outcome

**A second refinement round is now available for volumes, as an opt-in.**
`generate(refinement="points+boxes")` re-prompts each candidate on its anchor slice before the
propagation and is worth **+2.28% macro mSA at +1.4% runtime on the 32-slice crops** (+2.26% at about
+9% on the 12-slice ones), with no dataset regressing at depth. It fails the +5% quality gate, so
`refinement=None` remains the pipeline default, but it passes the per-dataset runtime cap at depth,
which no 2d refinement variant does. Experiment 6 has the measurements; its two tuned defaults differ
from the 2d ones (`n_negatives` 4 rather than 6, `min_consistency` 0.85 rather than 0.7), and it also
found that *how many SAM2 calls carry the same prompt* changes quality by 1.75 macro points, which no
2d experiment could have surfaced.

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

## Experiment 6: refining the anchor slice

### Hypothesis

Campaign 1 of the 2d refinement work closed with a prediction about volumes
([`APG_2D_OPTIMIZATION.md`](APG_2D_OPTIMIZATION.md), conclusion 6): the three ingredients that were
each measured neutral in 3d - re-prompting an instance with the prompts grouped onto it (+0.001),
negative prompts from the adjacent instances (+0.001), and conditioning the anchor slice with a box
(+0.001) - are exactly the ones that only pay *in combination* in 2d, where the combined
`points+boxes` re-prompt is worth +4.2% macro mSA. A 3d revisit should therefore test the combination
rather than any single ingredient.

Where it can be tested is forced. A volumetric mask comes from the propagation, and the module
docstring records that a prompt on an already propagated slice turns it into a conditioning frame and
replaces the mask there with a single-point one (-0.034 mSA). A finished track cannot be re-prompted,
so the 2d shape of the mechanism - merge, then re-prompt every instance - has no volumetric
counterpart. What does have one is the **anchor slice**: every candidate is already prompted and
scored there, in 2d, before anything is propagated, and that one prompt is what the whole track grows
from.

### Mechanism

`generate(refinement=..., refinement_kwargs=...)` now accepts a volume. The mode strings and their
components are the 2d ones; the second round runs inside `_score_candidates`, slice by slice, while
the predictor is still pointed at that slice, and what it produces is the conditioning the
propagation starts from rather than a finished mask. `derive_refinement_prompts`,
`_predict_refinement_batch`, `_predict_prompt_batch` and both acceptance gates are the 2d code,
applied in-plane. The current volume kwarg surface omits the 2d-only learned uncertainty-gate options
`gate` and `gate_threshold`, and adds one option with no 2d analogue: **`conditioning`**, which decides
how an accepted round reaches the propagation.

Historically, the campaign also implemented a `recover` component. It kept the 2d shape but had a
stronger volumetric argument: the merge that drops a candidate runs on one slice, and two objects
overlapping in-plane there can be separate everywhere else in z, so a revived candidate was
propagated and the 3d merge arbitrated. The neutral result below led to its later removal.

### Benchmark

Standard 12-slice crops, manifest `0f8fb67b3650a71f9f44b53037e89546`, `hvit_t` checkpoint `best`
(`85fb099c4bb038fa0ab9bddd6151689e`), on an `NVIDIA A100-SXM4-80GB MIG 1g.20gb` - the same device
string every earlier experiment recorded, so runtimes here are comparable to theirs. The screening ran
across six
implementation epochs; the `refinement=None` control returns macro mSA **0.382313017** in all
seventeen of its runs, every per-dataset value identical, so quality is comparable across epochs even
where the comparator's checksum matching is not. Baseline: 0.382313017, and a per-window median total
between 425 and 458 s.

**Timing on this machine needs stating before any cost figure is read.** The baseline moved 427 ->
455 -> 437 -> 458 s across four windows in six hours with byte-identical unrefined code, and the
*within-window* spread ranged from 0.18% to 4.72%. A 1g.20gb slice is supposed to be partitioned, so
that drift is worth knowing about on its own: partitioning the GPU does not partition whatever this
is - host, storage or memory bandwidth. Splitting it shows the noise in generation, not
in the embedding phase, so it is contention on the node. Every round is therefore bracketed by a
control before and after it, and a round whose bracket shows drift has its cost column discarded:
that happened to R2 (+7.67% drift), R3 (+6.84%) and R4. The conditioning round and the composition
round held to -0.67% and -0.01% and their cost figures stand. `boxes` with `conditioning="mask"`
illustrates the hazard: the same code and the same segmentation measured +10.20%, +4.87% and +0.06%
against three different windows' controls.

### The mode grid

Everything at the 2d optimum (p1-n6, mc0.7, fo0.15, replace), `conditioning="prompts"` unless noted.

| mode | macro % | worst dataset |
|---|---:|---:|
| **points+boxes** | **+1.867** | -1.20 |
| points+boxes+masks | +1.833 | -0.15 |
| boxes, `conditioning="mask"` | +0.216 | -4.46 |
| points | +0.111 | -2.41 |
| boxes+masks | +0.080 | -1.58 |
| points+boxes, `conditioning="mask"` | -0.446 | -9.75 |
| boxes | -0.614 | -1.50 |

Every mode kept the baseline's 925 scored candidates and 93 propagation passes, so no second round
fed back into the first.

**The combination transfers; the ingredient ordering does not.** `points+boxes` beats both of its
ingredients alone - `points` +0.11%, `boxes` -0.61% - which is 2d's claim that grouped prompts pay
only when box-anchored, and its converse. But 2d had `boxes` alone at +1.68%, and here a box-only
conditioning frame is *worse* than the original single point. In a volume the box contributes nothing
by itself and earns its place only as the anchor the points hang off. Mask cues add nothing to
`points+boxes`, as in 2d.

### The conditioning axis, which has no 2d counterpart

Four strategies push **the same prompt** - identical box and points, from an identical accepted second
round: 912 replaced, 12 consistency-gated, 1 foreign-gated, 5.82 negatives per candidate, 1216 frame
steps, in every row. They differ only in how many SAM2 calls carry it.

| strategy | pushes per object | macro % | total time % | worst dataset time |
|---|---:|---:|---:|---:|
| `prompts` | 1 box + 6 points | **+1.867** | +8.64 | **+10.08** |
| `prompts-grouped` | 1 box + 1 points | +1.334 | **+1.97** | +2.29 |
| `prompts-joint` | 1 | +0.113 | +0.68 | +0.88 |
| `mask` | 1 (`add_new_mask`) | -0.446 | ~-2 | - |

A push is not bookkeeping. Each `add_new_points_or_box` re-runs the mask decoder on the conditioning
frame and feeds the previous prediction back in as a mask input, so pushing a prompt in more steps
refines the anchor iteratively. One call gives the decoder a single shot at box and points together;
eight give it a box and then six corrections. **Almost all of the benefit is in the first extra
step**: one push to two is worth +1.22 macro points, two to eight only +0.53 more for four times the
added runtime. What pays is letting the decoder see the box before it sees the negatives, not
correcting point by point.

This was found against intent. `prompts` was the original implementation; its runtime was read as
waste and "optimized" into `prompts-joint`, which cost 1.75 macro points. The telemetry identified
the cause: with the second round's counters and the frame steps identical, nothing but the push
mechanism could account for the drop. The same correction applies to the cost - `prompts` was first
reported at +25.25% against a baseline from another window, and is +8.64% against its own.

### The prompt counts

| config | macro % | negatives used per candidate |
|---|---:|---:|
| **n_negatives=4** | **+2.158** | 3.95 |
| n_negatives=6 (2d optimum) | +1.867 | 5.82 |
| n_positives=2 | +0.945 | 3.95 |
| n_negatives=2 | +0.749 | 2.00 |
| n_negatives=8 | +0.655 | 7.62 |
| n_negatives=12 | +0.337 | 11.11 |
| n_negatives=0 | -0.302 | 0 |
| n_positives=3 | -3.056 | 3.95 |

**Negatives are the whole mechanism and they peak at four.** With none the mode falls below baseline
and lands where `boxes` alone did, which it should: one positive and no negatives is a box plus the
original point. 2d peaked at six to eight and collapsed past twelve; 3d peaks earlier. That fits how a
volume must choose them - every negative comes from a candidate anchored on the same slice, so each is
a real in-plane neighbour rather than a prompt from across the image, and four already bound the
object. Supply is not the cap: 11.1 negatives per candidate were available and used at n12.

**Extra positives hurt, more sharply than in 2d.** p1 > p2 > p3, and p3 takes EmbedSeg down 12.7%.
This is 2d's S6 ablation reproducing: the grouped extra positives were the original core of the
second-round idea and are the wrong ingredient in both dimensions.

### The acceptance gates

On base `n_negatives=4`.

| config | macro % | consistency-gated | foreign-gated |
|---|---:|---:|---:|
| **`min_consistency=0.85`** | **+2.264** | 39 | 0 |
| `min_consistency=0.85`, foreign off | +2.264 | 39 | 0 |
| `max_foreign_overlap=None` | +2.195 | 9 | 0 |
| base, mc0.7 + fo0.15 | +2.158 | 9 | 1 |
| `max_foreign_overlap=0.05` | +2.158 | 9 | 7 |
| ungated | +1.962 | 0 | 0 |
| `min_consistency=0.5` | +1.925 | 2 | 2 |
| `min_consistency=0.95` | +1.541 | 374 | 0 |

**The consistency gate carries the gates' whole contribution and 3d wants it tighter than 2d**, with a
clean interior optimum at 0.85. 2d measured the opposite at the top, where 0.85 over-gated (+2.34%
against +3.12% for 0.7). The reversal has a mechanism: in 2d a wrongly accepted re-prompt costs one
instance's mask, in 3d it becomes the conditioning frame of a whole track. At 0.95 the gate vetoes 374
of 925 and starts refusing genuine improvements.

**`max_foreign_overlap` is redundant rather than inert.** Composed with mc0.85 it fires zero times and
removing it changes nothing to six decimals: the consistency gate has already vetoed those same
candidates. Keeping the 0.15 default costs nothing.

**Caveat.** Four of the five datasets return identical mSA across every row of this table - only CREMI
moves, from +1.81% to +7.88%, on a single crop with the lowest absolute baseline of the five (0.0957).
The gate ranking rests on one crop, which is what `../../APGv2.md:77-78` warns against.

### Negative quality and prompt geometry: four directions refuted

| config | macro % | worst dataset |
|---|---:|---:|
| **base** (`box_extension=0`, prompt negatives, replace, no distance cap) | **+2.158** | -1.09 |
| `policy="keep-if-better"` | +1.883 | -1.09 |
| `max_negative_distance=64` | +0.338 | -1.09 |
| `negative_source="interior"` | +0.185 | -1.91 |
| `box_extension=2` | -1.962 | -23.32 |
| `box_extension=4` | -5.122 | -17.08 |

Nothing improves on the defaults, which is worth as much as a win: it closes four axes. A grown box is
far more harmful than in 2d, driven by the C. elegans atlas (-23.3% at extension 2), the most crowded
crop - a box grown into a neighbour becomes a track's conditioning frame, so the error propagates
instead of staying in one mask. Interior negatives *lose*, reversing 2d's best source, plausibly
because a 3d prompt is the density peak of a whole component while the interior point of an obliquely
cut cross-section can sit near the border of a thin sliver. Capping negatives by distance costs 1.8
points, so the negatives that pay are not only the immediate neighbours.

### Recovery: neutral, and quantified

| `recover_max_claimed` | candidates | recovered | macro % |
|---:|---:|---:|---:|
| 0.4 | 16 | 7 | +2.158 |
| 0.6 | 24 | 9 | +2.158 |
| 0.8 | 33 | 9 | +2.158 |
| without `recover` | - | - | +2.158 |
| standalone `recover` | 24 | 9 | +0.000 |

The volumetric argument is stronger than 2d's and still does not pay, for a reason the argument did
not anticipate: there is almost nothing to act on. Of 925 scored candidates the merge offers 16 to 33
whose pixels are lightly enough claimed to qualify, at most 9 survive their re-prompt, and adding 9
objects across five crops moves the dataset-balanced mSA by less than 1e-6 - with or without the rest
of the refinement. The records the per-slice merge drops are nearly all heavily claimed, i.e. real
duplicates rather than lost objects, which matches 2d's S4 and this document's own diagnosis that the
dominant 3d recall failure is the object that was never proposed. The measured-neutral `recover`
option was subsequently removed from both the 2d and 3d library paths.

### What the standard set recommends

Two configurations, both `points+boxes` with `n_negatives=4` and `min_consistency=0.85`:

- **quality**: `conditioning="prompts"` - +2.264% macro for about +9% total runtime and +10.08% on its
  worst dataset, which misses the gates' 10% per-dataset cap by 0.08 points.
- **balance**: `conditioning="prompts-grouped"` - +1.876% macro for +2.60% runtime, comfortably inside
  every runtime constraint, and the only refinement variant in either dimension that is.

Neither approaches the +5% quality gate, so the pipeline default stays `refinement=None`.

### Confirmation on the deep crops

Opt-in 32-slice set, manifest `f611a7125383e850798d0b5bf696f6f7`, selected with `--crops-3d deep`.
`--prepare-only` confirms all five crops reach their declared depth with no trim (32, 32, 32, 32, 30)
at the ROIs and object counts experiment 5 recorded.

Three controls give macro **0.314143** with every per-dataset value within 3.6e-7 of experiment 5's
recorded deep baseline, so the refinement work is behaviour-free at depth as well. Median total
3024 s; the bracketing control closed at 3000 s, -0.78%, so this round's cost figures stand.

| configuration | macro % | total time % | worst dataset mSA | worst dataset time |
|---|---:|---:|---:|---:|
| **n4 + mc0.85, `prompts-grouped`** | **+2.310** | +1.97 | +0.09 | +3.74 |
| **n4 + mc0.85, `prompts`** | +2.277 | **+1.42** | +0.16 | **+2.08** |
| library defaults, n6 + mc0.7, `prompts` | +1.779 | +9.41 | +0.02 | +10.90 |

Per-dataset mSA change:

| configuration | celegans | embedseg | gonuclear | cremi | snemi |
|---|---:|---:|---:|---:|---:|
| n4 + mc0.85, `prompts-grouped` | +7.23 | +1.15 | +3.55 | +0.09 | +2.43 |
| n4 + mc0.85, `prompts` | +8.36 | +1.15 | +3.48 | +0.16 | +2.28 |
| library defaults | +8.46 | +0.80 | +3.52 | +2.24 | +0.02 |

Four results, all of them favourable, and one of them the answer to the question this set exists for.

**The tuning is not an artifact of shallow crops.** Both tuned variants beat the defaults at 32 slices,
+2.28% and +2.31% against +1.78%, reproducing the 12-slice ordering (+2.26% against +1.87%). That was
the real risk: `n_negatives=4` and `min_consistency=0.85` were selected on crops where four of the five
datasets returned identical mSA across entire rounds, R3's whole signal was one CREMI crop and R4's
one C. elegans crop. They hold at depth.

**Nothing regresses at depth.** The worst per-dataset change is +0.02% to +0.16% across all three
candidates, against -1.09% to -1.34% on the 12-slice crops: every deep crop of every candidate
improves. C. elegans gains the most, 7.2-8.5%, and it is also the crop whose absolute baseline is
smallest (0.0347), so its relative change carries the least weight of the five.

The tuning trades between datasets rather than lifting all of them: against the defaults it buys CREMI
down (+2.24% to +0.09%) and SNEMI up (+0.02% to +2.43%), and the macro gain is that trade coming out
ahead. The two tuned variants differ from each other mainly on C. elegans, +7.23% against +8.36%.

**The cost largely disappears at depth.** `prompts` cost about +9% total and +10.08% on its worst
dataset at 12 slices; at 32 it is +1.42% and +2.08%. The refinement's overhead is fixed per candidate
- one extra 2d forward, a few extra anchor-frame pushes - while propagation scales with depth, and the
deep crops carry 5.7x the frame steps. Both tuned variants therefore clear the 10% per-dataset runtime
cap comfortably, which neither did on the shallow set. Deep volumes being the case that matters, this
is the more relevant regime.

**The two conditioning strategies converge.** At 12 slices `prompts` led `prompts-grouped` by 0.39
macro points; at 32 they are within 0.03 points and their runtimes differ by less than the controls'
own 3.0% spread. The iterative refinement of the anchor is worth less when the track is longer, which
is consistent with its mechanism: the anchor is one frame of 32 rather than one of 12.

### Decision

The quality gate needs +5% macro mSA and the best configuration reaches +2.31%, so **no default
changes and `refinement=None` remains the pipeline default**, as in 2d. The gate is failed on quality
alone: the runtime cap, which the 12-slice measurements broke, is passed at depth by both tuned
variants.

The measured optimum becomes the `DEFAULT_REFINEMENT_3D` values, so the recommended usage is

```python
segmenter.generate(refinement="points+boxes")
```

which resolves to `n_positives=1`, `n_negatives=4`, `min_consistency=0.85`,
`max_foreign_overlap=0.15`, `policy="replace"`, `conditioning="prompts"`: +2.28% macro at +1.4%
runtime on the deep crops, +2.26% at about +9% on the shallow ones. Workloads on shallow stacks, where
`prompts` is the expensive variant, should pass `refinement_kwargs={"conditioning": "prompts-grouped"}`
for +1.88% at +2.6%; at depth the two are equivalent and the cheaper one is marginally ahead.

Two of the 2d recommendation's values do **not** carry over and the defaults differ accordingly:
`n_negatives` is 4 rather than 6, and `min_consistency` 0.85 rather than 0.7.

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

Experiment 6 adds three more, all specific to refining a volume rather than an image:

4. **The structure of the 2d refinement transfers; none of its tuned values do.** The combination pays
   where no single ingredient does, one positive beats two beats three, mask cues add nothing, and
   recovery is neutral - all as in 2d. But `boxes` alone is negative here where it was +1.68% in 2d,
   negatives peak at four rather than six to eight, the consistency gate wants 0.85 rather than 0.7,
   `max_foreign_overlap` is redundant rather than additive, interior negatives lose rather than win,
   and a grown box costs four macro points rather than a fraction of one. Each reversal has a
   volumetric mechanism behind it, and the common thread is that a 3d anchor prompt is the conditioning
   frame of a whole track, so both the value of getting it right and the cost of getting it wrong are
   larger than for one 2d mask.
5. **The number of decoder steps that carry a prompt is a parameter, not an implementation detail.**
   Pushing the identical box and points in one SAM2 call, two, or eight gives +0.11%, +1.33% and
   +1.87%, because each push re-runs the mask decoder on the conditioning frame and feeds the previous
   prediction back in. Nearly all of it is in the first extra step: letting the decoder see the box
   before the negatives. Anything that touches how prompts reach the video predictor should measure
   this axis rather than assume batching is free.
6. **A refinement's cost amortizes against depth while its benefit does not decay.** The same
   configuration costs about +9% on 12-slice crops and +1.4% on 32-slice ones, because the overhead is
   fixed per candidate while propagation scales with depth. Optimizations of the per-candidate stages
   should therefore be gated on deep crops; the shallow set overstates their cost by roughly 6x.

The follow-ups experiment 6 leaves open: the recall axis is still untouched, since recovery reaches only
the 16-33 candidates of 925 that the anchor-slice merge drops lightly and the dominant failure remains
the object never proposed; and the shallow benchmark cannot resolve the gates, where four of five
datasets returned identical mSA across whole rounds and one CREMI crop carried the entire signal.

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
python finetuning/v2/evaluation/optimization/benchmark_apg_optimization.py \
    --ndim 3 --trial-id baseline-1 --time-budget-minutes 30
```

The deep crop set is opt-in, keeps its own manifest, and needs a larger budget. Building it first is
worthwhile because `--prepare-only` prints the depth each crop will actually propagate through, which is
the check that the declared depth survived the loader's trim:

```bash
python finetuning/v2/evaluation/optimization/benchmark_apg_optimization.py --crops-3d deep --prepare-only

python finetuning/v2/evaluation/optimization/benchmark_apg_optimization.py \
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
python finetuning/v2/evaluation/optimization/compare_apg_optimization.py \
    --ndim 3 --target efficiency \
    --baseline-run /path/to/baseline-1 \
    --baseline-run /path/to/baseline-2 \
    --baseline-run /path/to/baseline-3 \
    --candidate-run /path/to/candidate \
    --output /tmp/apg-3d-decisions.json
```

The comparator writes its decision summary as JSON and every per-dataset delta as CSV beside it.

### Experiment 6: reproducibility and artifacts

Output root as for the earlier experiments. Model `hvit_t` / `best`
(`85fb099c4bb038fa0ab9bddd6151689e`) throughout, on an `NVIDIA A100-SXM4-80GB MIG 1g.20gb`. Manifests
`0f8fb67b3650a71f9f44b53037e89546` (standard) and `f611a7125383e850798d0b5bf696f6f7` (deep).

The implementation changed six times, so the campaign spans six checksums. Quality is nonetheless
comparable across all of them: the `refinement=None` control returns macro mSA 0.382313017 on the
standard crops in every one of its seventeen runs, with every per-dataset value identical, and
0.314143 on the deep crops. Runtime is not comparable across epochs, which is what the bracketing
controls below are for.

| epoch | implementation | what changed | what it carries |
|---|---|---|---|
| 2 | `d8fafe4f32175b067e54805b7601cd37` | the implementation | R1's mode grid |
| 3 | `8f7ba4cda458c37479c72c0fdd05c21f` | prompts batched into one push | superseded; three modes crashed |
| 4 | `30546874680ed0ebdf479bcc41f099ca` | negative-stride fix | R1 re-measured |
| 5 | `ef3aac3c94a76583e37035f33f9dfd75` | `conditioning` as an explicit axis | the conditioning round, R2, R3, R4 |
| 6 | `efe1ae4b6f23407f256bc1b418dd5be1` | `recover` conditioning fix | R5, composition, the deep confirmation |

Epochs 3 and 4 exist only to undo mistakes: epoch 3 batched the anchor-frame pushes into one call,
which crashed on any frame carrying a single candidate and, once fixed, cost 1.75 macro points by
removing the iterative refinement it was meant to make cheaper. Both are recorded because their runs
are on disk and because the epoch-5 conditioning axis is the measurement that explains them.

Configuration checksums, by round (trial id in brackets):

| round | configurations |
|---|---|
| controls, standard | `d2d0cb8b`, `095496cc`, `c667d849` [`control-3d-1..3`], in every epoch |
| brackets, standard | `ee632c07` [r2], `77cbca7c` [r3], `9c2f2fa8` [r4], `fe11fd69` [r5], `fb305b4e` [e6], `5af3a058` [cond] |
| R1 mode grid [`r1-screen-1`] | boxes `678a375f`; boxes+masks `ca6ffd5e`; boxes/mask `2db8dbd7`; points `fe9e9f9b`; points+boxes `458952ad`; points+boxes/mask `6d4160f8`; points+boxes+masks `6e22acd7` |
| conditioning [`cond-screen-1`] | prompts `131ad19b`; prompts-grouped `80d48a8b`; prompts-joint `55fc4298` |
| R2 [`r2-screen-1`] | n0 `845c1895`; n2 `4e65b8eb`; n4 `76b46b4b`; n8 `5b70c6f8`; n12 `0e094bd1`; p2 `3009f492`; p3 `2b313dc3` |
| R3 [`r3-screen-1`] | mc0.5 `fdbb589a`; mc0.85 `73a64deb`; mc-off `b9d5eab9`; fo0.05 `189a1258`; fo-off `d201ebfe`; ungated `85d4f770` |
| R4 [`r4-screen-1`] | interior `b1596834`; box-ext2 `4b068d93`; box-ext4 `0a569e09`; max-neg-dist64 `e34960f9`; keep-if-better `05f79163` |
| R5 [`r5-screen-1`] | 0.4 `d0aad4c4`; 0.6 `917d9bc2`; 0.8 `561e043c`; standalone `9138fc68` |
| composition [`comp-screen-1`] | mc0.85+fo-off `6b039643`; mc0.95 `2839a7a0`; grouped+mc0.85 `c667bceb` |
| deep controls | `d2d0cb8b`, `095496cc`, `c667d849` [`control-3d-1..3`]; bracket `a4738f11` [`deep-bracket`] |
| deep candidates [`deep-screen-1`] | tuned `0d1909b7`; grouped `2fadfc69`; defaults `f79dcef2` |

A round is run as one configuration per invocation, for example:

```bash
python finetuning/v2/evaluation/optimization/benchmark_apg_optimization.py \
    --ndim 3 --config points-boxes-n4-mc085.json --trial-id r2-screen-1 --time-budget-minutes 60

python finetuning/v2/evaluation/optimization/benchmark_apg_optimization.py \
    --ndim 3 --crops-3d deep --config deep-tuned.json --trial-id deep-screen-1 \
    --time-budget-minutes 150
```

with a configuration naming only the changed 3d parameters:

```json
{
  "name": "deep-tuned",
  "params_3d": {
    "refinement": "points+boxes",
    "refinement_kwargs": {"n_negatives": 4, "min_consistency": 0.85}
  }
}
```

Because those two values are now the `DEFAULT_REFINEMENT_3D` entries, that configuration is what a
bare `{"refinement": "points+boxes"}` resolves to; `deep-defaults` above predates the change and
records the previous values explicitly.

Every round was bracketed by a control before and after it. Where the bracket showed drift the round's
cost column is discarded: R2 (+7.67%), R3 (+6.84%) and R4 are quality-only for that reason. The
conditioning round (-0.01%), the composition round (-0.67%) and the deep confirmation (-0.78%) held,
and their cost figures are the ones quoted.

Regression tests are in `test/test_v2_automatic_prompt_generation.py` and
`test/test_v2_prompt_based_segmentation.py`. Two of them exist because of specific failures rather
than by design: a parametrized stride test over one, two and seven points, after a single-candidate
frame produced a reversed array that torch refuses; and a cross product over six modes and four
conditioning strategies that pushes every candidate through the real propagator, after a producer of
conditioning dicts was found to omit a key the consumer required. Both bugs were shaped the same way -
each half unit-tested, the pair never exercised together.

### Prompt-state replay regression benchmark

`benchmark_prompt_state_replay.py` isolates a separate multi-anchor correctness and efficiency
problem. It reads the standard-manifest C. elegans crop `celegans_atlas:8db1fb8b4013`, selects the
four largest non-z-border objects with distinct center slices, and compares the production shared
state against one independent predictor state per anchor. The oracle sends the intended predictor
operations directly, bypassing `PromptableSegmentation3D` bookkeeping and replay.

The `joint` protocol sends one tight box, one interior positive and up to four nearest foreign-object
negatives in a single call. The `replacement` protocol first sends a grown box with an alternate
positive, replaces it with the joint set, then appends the cleared alternate point. The decoder-only
measurement runs one warm-up plus five repetitions; each protocol also gets one complete propagation.

```bash
python finetuning/v2/evaluation/optimization/benchmark_prompt_state_replay.py --label baseline
python finetuning/v2/evaluation/optimization/benchmark_prompt_state_replay.py --label fixed --expect-exact
```

Both runs used `hvit_t` / `best` (`85fb099c4bb038fa0ab9bddd6151689e`), manifest
`0f8fb67b3650a71f9f44b53037e89546`, and an NVIDIA A100-SXM4-80GB. The baseline and fixed
implementation checksums are `15d81f30dc775292a2200c70f80aa4f8` and
`49dbb7540d5010a3736912e47e6ed2d3`. Results are stored in the default APG output root's
`prompt_state_replay/` directory as `prompt_state_replay_{baseline,fixed}_a8f60972.json`.

| protocol | replay calls, before -> after | median decoder/replay time | full production time | oracle exact | selected-object mSA, before -> after/oracle |
|---|---:|---:|---:|---:|---:|
| joint | 92 -> 12 | 3.518 s -> 1.685 s (2.09x) | 15.508 s -> 10.180 s (1.52x) | no -> yes | 0.568 -> 0.568/0.568 |
| replacement | 168 -> 36 | 3.690 s -> 1.864 s (1.98x) | 11.125 s -> 10.256 s (1.08x) | no -> yes | 0.354 -> 0.522/0.522 |

Before the fix, replay reconstructed unordered active prompt sets: it split batched points into
individual decoder calls, replayed boxes afterward, and retained prompts that a replacement had
cleared. The corrected implementation records successful predictor operations and replays their
order, batching and `clear_old_points` flags verbatim. Active deduplication maps separately mirror
SAM2's mutually exclusive point/box and mask inputs. The fixed masks are bit-identical to the oracle
for every selected object in both protocols, and the predictor-call counts equal the logical minimum
for the initial push, grouped replay and final state restoration.

---

## Campaign of 2026-09-03: a new 3d setup before a new 3d method

Started 2026-09-03 on branch `apg-optim-fable`, hvit_t joint/v2 `best`. Implementation checksum of
the epoch with the opt-in volume hooks: `d11e240452d84052916d0f16e1a1cfb1`.

### What the survey of the stopped funnel campaign changed

1. **The tuning corpus.** The old primary manifest held 32 crops, 22 of them GoNuclear. The new
   manifests (`apg3d_manifest.py`, schema `apg3d-v1`, under `<output root>/3d_v2/`) draw deep crops
   from every source `common.VAL_SPLITS` allows: primary `1cf951b9…` with 57 crops from 10 sources
   (celegans 6, cremi C 3, cremi A/B 4 as seen-in-training, embedseg platy-ISH 7, platy-nuclei 8,
   skull 3, gonuclear 10, humanneurons 8, platynereis_nuclei 2, snemi 6), holdout `5ddc25cb…` with
   18 source-disjoint crops. EmbedSeg's organoid volumes are excluded: they annotate four cells each.
2. **A leak.** The production SNEMI test crop is original z 81:89; the old deep manifest's SNEMI
   range (0:30 after the z>=70 offset) and the funnel campaign's SNEMI crops contained it. The new
   SNEMI crops use z 70:81 and 89:100 and the validator refuses anything overlapping the test slab.
   The old deep manifest stays as a regression instrument for the runs recorded against it.
3. **Seen sources.** CREMI A/B and the EmbedSeg train sub-datasets were joint-training data; every
   crop carries `seen_in_training` and every summary reports an unseen-only macro next to the family
   macro.
4. **Hooks, default-off and bit-identical when off** (`micro_sam/v2/automatic_prompt_generation.py`):
   `derive_volume_prompts(return_metadata=True)` (ladder birth/merge/persistence and component
   statistics, `VOLUME_CANDIDATE_FEATURE_NAMES`), `_score_candidates(candidate_feature_schema=)`
   (the three anchor alternatives' selector features as side information), and in `generate`
   `prompts=`, `candidate_scorer_threshold`, `candidate_order`, `candidate_budget`, `keep_trace`
   with `set_multimask_models(volume_candidate_scorer=)`.

5. **Two 3d "defaults" too.** `common.resolve_params(ndim=3)` fills every parameter from the 2d
   per-model table and only swaps the candidate threshold, so every 3d run of the earlier benchmark
   used `sigma=0.5, min_candidate_size=4, min_size=50, dt=0.25` - the image values - where a plain
   `generate()` on a volume uses `sigma=1.0, min_candidate_size=1, min_size=100, dt=0.5`
   (`default_prompt_generation(is_volume=True)`). The new runner resolves against the volume defaults
   (`benchmark_apg_3d.resolve_volume_params`), so its control is what a user gets;
   `configs/apg3d_legacy_defaults.json` pins the earlier benchmark's values for continuity.

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

### C1: the baseline on the new primary manifest (volume defaults, 57 crops)

One crop per array task on 2g.20gb slices (array `c1_apg3d_defaults_v2`, aggregated across the epoch
checksums it ran through). mSA with a per-crop bootstrap 95% interval; the family macro averages the
ten sources into their eight families first; `merged` counts ground-truth objects matched at IoU 0.5,
`genuine_misses` the unmatched ones the crop did not sever.

| dataset | crops | mSA [95% CI] | objects | merged | genuine misses | passes | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| celegans_atlas | 6 | 0.1183 [0.045, 0.204] | 1119 | 234 | 805 | 153 | 369 |
| cremi | 3 | 0.1078 [0.099, 0.122] | 840 | 177 | 251 | 162 | 1289 |
| cremi_seen | 4 | 0.0302 [0.026, 0.035] | 7469 | 243 | 58 | 243 | 1942 |
| embedseg_platy_ish | 7 | 0.4161 [0.372, 0.466] | 422 | 316 | 71 | 227 | 1040 |
| embedseg_platy_nuclei | 8 | 0.2153 [0.188, 0.249] | 901 | 449 | 348 | 244 | 524 |
| embedseg_skull | 3 | 0.6240 [0.564, 0.722] | 62 | 53 | 5 | 96 | 541 |
| gonuclear | 10 | 0.3379 [0.261, 0.409] | 726 | 511 | 144 | 229 | 603 |
| humanneurons | 8 | 0.3915 [0.362, 0.422] | 1601 | 978 | 304 | 276 | 1138 |
| platynereis_nuclei | 2 | 0.2669 [0.254, 0.280] | 127 | 78 | 44 | 43 | 149 |
| snemi | 6 | 0.5318 [0.502, 0.560] | 526 | 446 | 41 | 115 | 317 |
| __family_macro__ | 57 | 0.3048 | 13793 | 3485 | 2071 | 1788 | 7909 |
| __dataset_balanced__ | 57 | 0.3040 | 13793 | 3485 | 2071 | 1788 | 7909 |
| __unseen_macro__ | 33 | 0.2974 | 4812 | 2346 | 1545 | 935 | 3714 |
| __legacy_macro__ | 47 | 0.2951 | 12065 | 2429 | 1723 | 1469 | 6623 |

Recall remains the dominant loss: 3485 of 13793 objects are matched, and 2071 genuine misses remain
after severed objects are excluded. C. elegans (234 of 1119 matched) and CREMI C (177 of 840) are the
recall-limited sources the earlier notes named; the seen CREMI A/B slabs are worse still (243 of 7469,
mSA 0.03) - they hold thousands of thin processes a 32-slice crop cuts into slivers.

### C1: anchor refinement on the new manifest

`refinement="points+boxes"` (the opt-in of experiment 6) against the volume-default baseline on the
same 57 crops: family macro 0.3074 against 0.3048 (+0.8%), unseen macro 0.3018 against 0.2974
(+1.5%), for +8.9% total time (8614 s against 7909 s). Per source: CREMI C +7.9%, EmbedSeg skull
+4.9%, GoNuclear +3.1%, humanneurons +1.3%, SNEMI +0.1%, EmbedSeg Platynereis-ISH -0.8%, Platynereis
nuclei -2.3%, C. elegans -2.1%, platynereis_nuclei -2.6%. The +2.3% the earlier five-crop deep set
reported shrinks to +0.8% on a balanced set with 57 crops, and three sources lose. The refinement
stays an opt-in; it is not a lever for the +5% gate.

### C2 result: the slice-wise hybrid is a CREMI mode, not a 3d mode

Hybrid-only (2d APG with predicted-IoU scoring on a fresh encode of every slice, then linking) on the
21 EM crops of the primary manifest, linking replayed from the slice cache; the baseline is the
propagation pipeline with the volume defaults on the same crops:

| source | crops | baseline | multicut beta 0.5 | beta 0.4 | beta 0.3 | beta 0.2 | greedy IoU 0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cremi (C) | 3 | 0.108 | 0.120 | 0.149 | **0.157** | 0.147 | 0.142 |
| cremi_seen (A, B) | 4 | 0.030 | 0.027 | 0.027 | 0.027 | 0.027 | 0.028 |
| humanneurons | 8 | 0.391 | 0.157 | 0.193 | 0.219 | 0.237 | 0.181 |
| snemi | 6 | 0.532 | 0.265 | 0.295 | 0.310 | 0.317 | 0.284 |

On CREMI C the hybrid matches 289 of 840 objects against the baseline's 177 and lifts mSA by 46% at
beta 0.3, in a tenth of the time. On humanneurons and SNEMI, where the propagation already matches
most objects (978 of 1601, 446 of 526), the slice masks do not stack into 3d objects at IoU 0.5 and
the hybrid loses 40% however the linking prior is set. The mode is therefore modality-specific -
worth keeping as an explicit option for dense anisotropic EM of the CREMI kind, not as a 3d default -
and its general use is the `union-point` recall variant, whose CREMI crops so far gain 12-29% with the
propagation intact (see the arrays below).

### C2 result, recall variant: hybrid chains as extra prompts recover objects, but not precision

`union-point`: the density ladder's candidates plus one point prompt per hybrid chain whose anchor no
density candidate already covers, all propagated by the unchanged pipeline (array
`c2_union_point_recall`, 27 crops of the recall-limited sources; hybrid slices with predicted-IoU
scoring, multicut beta 0.5):

| source | crops | baseline mSA | union mSA | matched objects | passes | seconds |
|---|---:|---:|---:|---|---|---|
| cremi (C) | 3 | 0.108 | **0.126 (+17%)** | 177 -> 209 of 840 | 162 -> 188 | +36% |
| cremi_seen (A, B) | 4 | 0.030 | 0.031 (+4%) | 243 -> 253 of 7469 | 243 -> 252 | +18% |
| celegans | 6 | 0.118 | 0.114 (-4%) | 234 -> 255 of 1119 | 153 -> 172 | +113% |
| humanneurons | 8 | 0.391 | 0.368 (-6%) | 978 -> 1033 of 1601 | 276 -> 414 | +80% |
| snemi | 6 | 0.532 | 0.523 (-2%) | 446 -> 468 of 526 | 115 -> 152 | +127% |

The extra prompts do what they were meant to do on every source - 21 to 55 more objects matched -
and mSA still falls on three of five, because the same prompts also add tracks that match nothing
and the score-ordered merge keeps them. This is the recall / precision trade the earlier notes
predicted for any candidate expansion, and it is exactly what a pre-propagation filter has to arbitrate:
the C3 replay measures whether the learned candidate score can keep the recovered objects and drop
the rest (the `union` supply is the recall-expansion policy of C4). The cost, +18-127% of the baseline,
comes from the extra passes and from the slice-wise 2d pass itself; a budget on the added prompts is
part of the same replay.

### C0, test manifest (built 2026-09-03 08:20, not yet opened)

`apg3d_manifest.py build --subset test` → `3d_v2/manifest_test_apg3d-v1.json`, checksum
`33043f1ef079b10e6f6562297b3ff83c`, 56 deep crops, 8 per pure-test dataset (blastospim 143 objects,
cartocell 511, cellseg_3d 699, mouse_embryo 576, nis3d 2991, plantseg 1199, pnas_arabidopsis 2213), all
unseen in training. It is to be run once, at the end, for control, `points+boxes` and the C3/C4 winner.

### C2 holdout check of the slice-wise hybrid (2026-09-03, 08:50)

`hybrid-2d`, standalone encoding, plain predicted IoU, multicut beta 0.3 on the 18 holdout crops
(`3d_v2/hybrid/holdout/hybrid-2d-standalone-plain-multicut-473a3b0d5e73-26a1003788ea`), 662 s in total:

| dataset (crops) | mSA | merged / GT objects | genuine misses |
|---|---:|---|---:|
| celegans_atlas (2) | 0.086 | 113 / 384 | 239 |
| cremi (1) | 0.159 | 113 / 297 | 57 |
| cremi_seen (2) | 0.154 | 100 / 274 | 69 |
| embedseg_platy_ish (2) | 0.204 | 86 / 129 | 30 |
| embedseg_platy_nuclei (2) | 0.180 | 16 / 34 | 14 |
| embedseg_skull (2) | 0.618 | 73 / 77 | 2 |
| gonuclear (3) | 0.244 | 278 / 447 | 123 |
| humanneurons (1) | 0.166 | 133 / 265 | 71 |
| platynereis_nuclei (1) | 0.058 | 18 / 46 | 28 |
| snemi (2) | 0.239 | 132 / 174 | 31 |
| family macro (18) | 0.183 | 1062 / 2127 | 664 |

The propagation baseline on the same crops (`c5_holdout_apg3d_defaults`) is the reference for the
per-dataset comparison and is recorded below when its array completes. Under the generalization rule
stated by the user today (a change must help on all datasets or on most with minor regressions), a mode
that wins on one EM family and loses 40% elsewhere is not a candidate, whatever this comparison shows;
the hybrid stays a diagnostic of where slice-wise recall exceeds propagation recall.

### C5 holdout: propagation baseline, and the hybrid against it (2026-09-03, 08:50)

`c5_holdout_apg3d_defaults` (18 crops, `3d_v2/runs/holdout/apg3d-defaults-70b90e407d4b-26a1003788ea`),
3202 s in total, 604 propagation passes; the hybrid columns are the run recorded in the previous section:

| dataset (crops) | baseline mSA | hybrid mSA | hybrid change | baseline merged / GT |
|---|---:|---:|---:|---|
| celegans_atlas (2) | 0.104 | 0.086 | −17% | 89 / 384 |
| cremi (1) | 0.138 | 0.159 | +16% | 80 / 297 |
| cremi_seen (2) | 0.262 | 0.154 | −41% | 99 / 274 |
| embedseg_platy_ish (2) | 0.330 | 0.204 | −38% | 88 / 129 |
| embedseg_platy_nuclei (2) | 0.456 | 0.180 | −60% | 27 / 34 |
| embedseg_skull (2) | 0.675 | 0.618 | −8% | 73 / 77 |
| gonuclear (3) | 0.390 | 0.244 | −37% | 311 / 447 |
| humanneurons (1) | 0.320 | 0.166 | −48% | 141 / 265 |
| platynereis_nuclei (1) | 0.241 | 0.058 | −76% | 24 / 46 |
| snemi (2) | 0.510 | 0.239 | −53% | 149 / 174 |
| family macro (18) | 0.322 | 0.183 | −43% | 1081 / 2127 |
| unseen macro (9) | 0.292 | 0.179 | −39% | 770 / 1567 |

The holdout baseline (family macro 0.3218) sits above the primary one (0.3048), as the holdout draws
more of its crops from the easier nuclei sources. The hybrid's CREMI advantage shrinks to +16% on the single
unseen CREMI holdout crop and turns into −41% on the two CREMI A/B crops, at a fifth of the runtime. This
closes the hybrid as a mode: it does not even hold its one family out of sample. The seeded-object deficit
that motivated it (C. elegans 89 of 384 objects merged, CREMI 80 of 297) remains the recall problem the
propagation pipeline has to solve within itself.

### C5 holdout: anchor refinement with points and boxes (2026-09-03, 08:57)

`c5_holdout_apg3d_refine_points_boxes` (`3d_v2/runs/holdout/apg3d-refine-points-boxes-3826936b5a95-26a1003788ea`),
3375 s against 3202 s for the baseline (+5.4%), same 604 propagation passes:

| dataset (crops) | baseline | refined | change |
|---|---:|---:|---:|
| celegans_atlas (2) | 0.1036 | 0.1043 | +0.6% |
| cremi (1) | 0.1377 | 0.1514 | +9.9% |
| cremi_seen (2) | 0.2621 | 0.2777 | +5.9% |
| embedseg_platy_ish (2) | 0.3305 | 0.3605 | +9.1% |
| embedseg_platy_nuclei (2) | 0.4564 | 0.4636 | +1.6% |
| embedseg_skull (2) | 0.6751 | 0.6739 | −0.2% |
| gonuclear (3) | 0.3896 | 0.3898 | +0.1% |
| humanneurons (1) | 0.3205 | 0.3343 | +4.3% |
| platynereis_nuclei (1) | 0.2414 | 0.2117 | −12.3% |
| snemi (2) | 0.5101 | 0.5067 | −0.7% |
| family macro (18) | 0.3218 | 0.3230 | +0.4% |
| unseen macro (9) | 0.2923 | 0.2973 | +1.7% |
| dataset-balanced (18) | 0.3427 | 0.3474 | +1.4% |

Refinement helps where the anchors are least reliable (EM neurites, the ISH channel, human neurons: +4% to
+10%), is neutral on the nuclei sources, and loses 12% on the single platynereis crop (46 objects, so one
crop's noise is large). Together with the primary result (+0.8% family macro) this is a small but consistent
improvement on the EM/ISH families and a wash elsewhere at +5% runtime; it stays what it was, an opt-in.
Whether a filter-aware version (refine only the candidates the learned filter keeps) does better is a C3/C4
question.

### G4 design: what the 2d generalization result changes for 3d (2026-09-03, 09:35)

The 2d G campaign found that a learned mask scorer on SAM2's mask tokens carries dataset identity, that
generic statistics remove the transfer failure but carry no transferable gain over predicted IoU, and that
a 64-unit MLP overfits dataset interactions even on generic inputs. For the 3d retry this fixes the reading
protocol of C3/C4 before their numbers are looked at:

1. The only acceptable evidence is source-level leave-one-dataset-out on the unseen sources (celegans,
   gonuclear, humanneurons, cremi C, snemi; `screen_apg_3d_filter.py` `lodo` rows and `__unseen_macro__`),
   not the in-domain OOF rows; the in-domain rows are reported but do not decide.
2. Among the 18 C3 filters, the `lowres_v1` schema (19 generic anchor statistics, with or without the 20
   ladder/component features) is the only one eligible; `token_v1` / `token_lowres_v1` rows are diagnostics
   of how much identity the tokens carry in 3d, as in 2d.
3. Width: prefer the smallest hidden size whose LODO gain matches the larger ones; if only H128 gains, treat
   it as in-domain fitting. A linear filter should be added to the C3 grid before any confirmation run
   (`train_apg_3d_filter.py` has no linear option yet; port `GroupwiseLinear` from the 2d trainer).
4. The reference policy is the plain anchor filter (predicted IoU ≥ 0.5, the C1 control), and the plain
   E2-style proposal-side changes are the 3d analogue to test first if no learned filter passes: the density
   ladder and `max_overlap` (3d merge 0.15) and `min_size` (100) were never re-tuned on the 57-crop corpus.
5. Acceptance for a 3d change follows the same rule as 2d: unseen-source LODO family macro up, no source
   below −2%, and confirmed on the 18-crop holdout without re-fitting.

### Experiments in flight (job ids)

- C1 baseline and refinement on the new primary manifest: `benchmark_apg_3d.py`, arrays `15715858`
  and `15715859` were cancelled after 13 crops because they resolved the 3d parameters from the 2d
  table (finding 5); resubmitted as `c1_apg3d_defaults_v2` and `c1_apg3d_refine_points_boxes_v2`
  with the volume defaults, one crop per task on 2g.20gb slices.
- C2 slice-wise hybrid (`screen_apg_3d_hybrid.py`). Probes on three crops, hybrid-only (2d APG on
  every slice, multicut linking) against the propagation baseline with the volume defaults:

  | crop | baseline mSA | hybrid, plain scoring | hybrid, learned selector | hybrid seconds / baseline |
  |---|---:|---:|---:|---:|
  | gonuclear `4ea4ece3dbe1` (19 objects) | 0.133 | 0.019 | 0.013 | 29 / 68 |
  | celegans `18f3659eee71` (72 objects) | 0.289 | 0.181 | 0.048 | 21 / 39 |
  | cremi `28e7f518d8f1` (296 objects) | 0.099 (CREMI 0.990) | 0.107 (CREMI 1.267) | 0.106 (CREMI 1.246) | 68 / 427 |

  Two findings. The learned 2d selector, fitted on five light-microscopy datasets, does not transfer
  to these slices: on C. elegans it cuts the slice-wise result from 0.18 to 0.05, and on the volume's
  own per-slice embeddings (the video model's preprocessing) it collapses even further (0.09 against
  0.27 per-slice mSA on one GoNuclear slice). Plain predicted-IoU scoring is the hybrid's working
  configuration. Second, hybrid-only loses to propagation on the nuclei volumes (small objects whose
  slice masks do not stack into a 3d IoU of 0.5) but recovers far more objects on dense CREMI (95
  matched against 54, mSA +7.5%) at a sixth of the cost, while fragmenting more (VI split 1.43 vs
  0.64). The linking prior matters on dense EM: replaying the cached slices of the CREMI crop with
  the multicut at beta 0.3 gives 0.128 mSA with 216 chains (296 objects), greedy IoU linking at 0.3
  gives 0.128 with 354, while beta 0.7 or a 0.5 IoU threshold fragment the volume (0.056 / 0.069,
  566 / 591 chains). The mode is therefore not a general replacement; on dense EM it is a candidate
  in its own right (array `c2_hybrid_em` caches the slices of every EM crop, beta is swept on the
  cache) and elsewhere its use is as a recall source: the `union-point` variant (density candidates
  plus one prompt per hybrid chain, propagated) runs on the recall-limited datasets as array
  `c2_union_point_recall`.
- C3 track cache (`extract_apg_3d_tracks.py`), trainer (`train_apg_3d_filter.py`) and CPU replay
  (`screen_apg_3d_filter.py`). Identity check on the GoNuclear crop `4ea4ece3dbe1`: the replay's
  control policy (base ladder, predicted IoU >= 0.6, in-plane merge, anchor-score order) gives mSA
  0.132850, 9 objects, 20 candidates and 12 passes, and the pipeline run with the volume defaults gives
  0.1329, 9, 20 and 12 - the cache reproduces the pipeline. The extraction over the 57 primary crops
  is array `c3_extract_primary` (three ladders, all candidates propagated once, about twice the
  baseline's cost per crop). One caveat seen on the way: the flow density is computed with eight
  threads and two runs of the same crop proposed 33 and 34 candidates, so a candidate count can move
  by one between runs; scored candidates, passes and mSA were identical here.
- `c5_holdout_apg3d_defaults` (15720688), `c5_holdout_apg3d_refine_points_boxes` (15720689) and the local
  hybrid holdout run: complete and recorded above (08:50-08:57).
- `c3_train_replay` (15716576) and `_v2` (15720967, 128 GB) were both OOM-killed in the aggregate step: the
  per-candidate loop indexed the compressed `NpzFile` on every iteration, which decompresses the whole
  feature array each time and kept one copy alive per candidate. `train_apg_3d_filter.aggregate` now reads
  each crop's arrays once and stacks them (same output); resubmitted as `c3_train_replay_v3` (15721312) at 09:08. The track cache
  is `3d_v2/cache/primary/103ee82e4d89/` (57 complete crops; `12c3c629b1ae` is the local smoke-test crop).
- `c3_train_replay_v3` (15721312): aggregate (84,485 candidates, 21 s) and the 20 filter fits (done 09:22; LODO
  track-IoU correlations 0.22-0.77 by source) are complete under `3d_v2/c3/{training,models}`; the CPU replay
  (`screen_apg_3d_filter.py`, started 09:22) had printed no finished policy after 50 min at one busy core and
  6 GB. It is single-threaded over ~hundreds of policies × 57 crops; if it has not written
  `3d_v2/c3/screen_primary/summary.csv` by the next session, profile the replay (unpack each crop's tracks once,
  cache the merge inputs, multiprocess over crops) before rerunning. Read it under the G4 protocol above.

## Campaign of 2026-09-05: v4 geodesic first, object counts first, visual check before any screen

Started 2026-09-05 11:25 after the 2d work was closed (see the closing section of `APG_2D_OPTIMIZATION.md`). What
the 2d result changed for 3d, agreed with the user: (1) measure joint/v4 geodesic on the 3d manifests before
anything else (it moved 2d by +10% with unchanged settings and no 3d number exists for it); (2) read object counts
(matched at IoU 0.5, genuine misses, extra predictions) as the primary evidence, mSA second, and look at the
best / worst crops in napari before reading any screen (mSA moved ±20% on 2d datasets for one-pixel boundary
conventions, and a track ending one slice early is the 3d analogue); (3) close the learned lines (the C3 filter,
whose replay produced one policy in 12 h before timing out, and the planned learned refinement gate); (4) keep the
label-free recall work, re-planned on v4 and gated on object counts with false positives counted; (5) the
efficiency items are the safe part. The 3d code path is unchanged by epochs 4 and 5: the first re-run crop
(celegans `18f3659eee71`, v2 defaults) reproduces the recorded 0.2894.

Runs: `benchmark_apg_3d.py run --save-outputs` (new option: each crop's segmentation and its anchors, scored and
merged prompt indices under `<run dir>/outputs/`) for v2 and v4 × volume defaults and `points+boxes` × primary (57)
and holdout (18), one crop per 2g.20gb task, trial `v2-e5` / `v4-e5`, arrays `s9_3d_*` (15751379-15751388). The
3d run identity does not include the checkpoint, so the v4 runs live under their own campaign root
`3d_v4geo/` (manifests linked from `3d_v2/`). Readers: `compare_apg3d_runs.py --subset <s>` (object counts and
mSA per source, macros, v4 vs v2 and refinement deltas), `package_apg3d_cases.py --subset <s> --n 1` (per dataset
the best / worst crop by the refinement's effect on v4 and by the checkpoint's effect, packaged as one HDF5 per crop
under `3d_cases/<subset>/` with raw, ground truth, the four segmentations and the anchors), and
`view_apg3d_cases.py <file.h5>` (napari: image, ground truth, one labels layer per run, anchor points as proposed /
scored / merged; needs a napari environment, not `new-stack`).

### Primary manifest (57 crops, 15:25): the checkpoint is the effect, the refinement changes no object count

`compare_apg3d_runs.py --subset primary` (tables in `3d_cases/primary_comparison*.csv`); all four runs at epoch 5:

| run | family macro mSA | unseen macro | matched / 13,793 | genuine misses | extra predictions |
|---|---:|---:|---:|---:|---:|
| v2 defaults | 0.3048 | 0.2974 | 3,485 | 2,071 | 645 |
| v2 + points+boxes | 0.3074 | 0.3018 | 3,467 | 2,089 | 647 |
| v4 defaults | 0.3267 (+7.2%) | 0.3266 (+9.8%) | 3,750 | 1,806 | 713 |
| v4 + points+boxes | 0.3245 | 0.3261 | 3,747 | 1,809 | 736 |

Per source, v4 with the defaults matches more objects than v2 on nine of ten (celegans 234 → 316, embedseg
nuclei 449 → 546, humanneurons 978 → 1,009, embedseg ISH 316 → 341, gonuclear 511 → 526, snemi 446 → 455) and
fewer on platynereis_nuclei (78 → 73, two crops); extra predictions rise by 68 (gonuclear +30, celegans +30). The
refinement moves the matched count by −18 (v2) and −3 (v4) and every per-source mSA change is within the range
the 2d visual check attributed to boundary conventions (v4: −1.9% family macro, six sources down, embedseg nuclei
−5.5%). Runtime is not comparable between the checkpoints in this table (v4 ran on 1g.20gb, v2 on 2g.20gb slices).

Cases for napari: `3d_cases/primary/` (31 crops, 198 MB; `cases.csv`, `README.md`, `view_apg3d_cases.py` beside
them), per dataset the best and worst crop by the refinement's effect on v4 and by the checkpoint's effect, each
file holding raw, ground truth, the four segmentations, and the proposed / scored / merged anchors of every run.

### Holdout manifest (18 crops, 16:05): the same picture, without re-fitting anything

`compare_apg3d_runs.py --subset holdout` (`3d_cases/holdout_comparison*.csv`):

| run | family macro mSA | unseen macro | matched / 2,127 | genuine misses | extra predictions |
|---|---:|---:|---:|---:|---:|
| v2 defaults | 0.3219 | 0.2913 | 1,085 | 641 | 238 |
| v2 + points+boxes | 0.3225 | 0.2962 | 1,091 | 635 | 235 |
| v4 defaults | 0.3424 (+6.4%) | 0.3195 (+9.7%) | 1,162 | 564 | 244 |
| v4 + points+boxes | 0.3387 | 0.3223 | 1,154 | 572 | 258 |

v4 with the defaults matches 77 more objects (celegans +25, gonuclear +24, cremi A/B +17, embedseg ISH +10) and
misses 77 fewer for six more extra predictions; humanneurons (−4) and the single platynereis crop (−5 of 46, mSA
−20%) go the other way, as platynereis did on the primary manifest. The refinement changes the matched count by
+6 (v2) and −8 (v4). Cases: `3d_cases/holdout/` (17 crops). Combined with the primary manifest the v4 checkpoint
matches 342 more objects out of 15,920 with 74 more extra predictions, and no setting change of either campaign
comes near that; the object-level reading and the napari cases are what decide from here, not mSA.
