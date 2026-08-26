# Further APG Optimization Opportunities

## Scope and status

This note began as a read-only review of the APG2d and APG3d optimization work on the `apg-optim`
branch, followed by small sandbox experiments intended to identify useful next directions. The 2d
directions have now been implemented and evaluated on the established 240-image primary and
233-image holdout subsets. The implementation, leakage-safe sweeps, canonical acceptance result and
reproduction commands are recorded in the final section of
`finetuning/v2/evaluation/APG_2D_OPTIMIZATION.md`; the exploratory measurements below are retained as
the motivation and audit trail.

The first compact-model campaign found a 52 KB groupwise MLP selector. Retaining the historical
predicted-IoU filter left it at 0.283210/0.278328 primary/holdout and over the per-dataset runtime
cap, motivating the later learned-filter campaign. A final compact-feature campaign has now
superseded that path: the three SAM2 mask tokens plus low-resolution mask evidence, H64 eager
selection and a 0.375 learned-score filter reach 0.306835/0.315633. The three-trial holdout median is
186.15 s, 12.04% faster than the dense H64 control while improving its mSA by 6.76%.

The decoder-head/filter follow-up resolves this: the same triplet H64 scorer with eager selection
and a primary-selected 0.25 learned-score filter reaches 0.296508/0.295659. Its three-trial holdout
median is 189.9 s versus 177.8 s for same-implementation controls. The formal comparator accepts it:
+11.86% mSA, +6.83% aggregate runtime, no dataset regression and at most +8.68% runtime on any
dataset. A separately trained singleton MLP reaches 0.289056/0.283470 in 174.7 s in a diagnostic
run; the single-token default is itself better and faster than the historical three-mask/IoU path,
but remains below the +5% quality gate. The accepted eager MLP policy remains opt-in because its
artifact is external; changing library defaults was not part of the campaign. The first gated MLP
follow-up retrained the gate on this exact eager/learned-filter policy. Its primary-selected 50%
route reached 0.299904/0.301455 primary/holdout but added 25.32% runtime. The final campaign instead
scores signed refinement utility after merge and prompt assembly. Its primary-selected 15% route
reaches 0.310112/0.318550 on top of the compact selector and adds only 1.19% aggregate runtime in
canonical trials. It passes the incremental refinement acceptance gates and is the deployment
recommendation when both fitted artifacts are available.

The review covered:

- `finetuning/v2/evaluation/APG_2D_OPTIMIZATION.md`
- `finetuning/v2/evaluation/APG_3D_OPTIMIZATION.md`
- `finetuning/v2/evaluation/APGv2.md`
- `micro_sam/v2/automatic_prompt_generation.py`
- `micro_sam/v2/prompt_based_segmentation.py`
- `micro_sam/v2/models/_video_predictor.py`

The conclusions below distinguish the original reduced-set measurements from hypotheses. Those
sandbox datasets were too small to justify changing a default, but they exposed two signals worth
full evaluation:

1. APG2d's choice among SAM2's already-computed multimask alternatives can be improved.
2. APG3d proposal recall is modality-dependent, so a single broader density threshold is unlikely to
   be the right way to recover missing objects.

## Executive summary

The most promising quality improvement is a microscopy-specific selector for SAM2's multimask
outputs. The current implementation already computes several masks for each prompt, but chooses one
using the decoder's predicted-IoU score alone. On a reduced 50-image evaluation set, an exploratory
ranker using only already-available mask and prompt features improved macro mSA from 0.275459 to
0.293296, or 6.47% relative, without another SAM2 forward pass. An oracle selection among the same
three masks reached 0.309512, showing that there is additional headroom in selection rather than
proposal generation alone.

The complementary APG2d efficiency opportunity is to use this selector's uncertainty to decide
which instances receive the existing `points+boxes` refinement. Multimask selection and selective
refinement are related but not equivalent:

- **Multimask selection** picks a better answer from masks that the first decoder call has already
  produced. Its goal is a near-free quality improvement.
- **Selective refinement** detects cases where none of those first-round alternatives is trustworthy
  and spends a second decoder call only on those cases. Its goal is to retain refinement gains while
  avoiding its 35-50% blanket runtime overhead.

For APG3d, propagation remains the dominant cost. Further fixed threshold ladders, post-hoc temporal
connected-component filtering, anchor coalescing, and simple pass reordering are not attractive.
Better opportunities are:

- rank volumetric density components using persistence and component features before propagation;
- use confidence accumulated over the full propagated trajectory for termination and merging;
- investigate a second anchor only for uncertain or long components;
- reduce mask retention and launch overhead without changing segmentation semantics.

## Existing optimization baseline

### APG2d

The existing measurements identify prompt processing and mask decoding as the main costs. The current
240-image baseline has macro mSA 0.269577 and total runtime 204.797 seconds. On the recorded LiveCELL
profile, the approximate per-image stage times are:

| Stage | Time |
|---|---:|
| Prompt processing | 0.213 s |
| Mask decoder | 0.165 s |
| Image encoding | 0.086 s |
| Interior computation | 0.012 s |
| Flow computation | 0.012 s |
| Merge | 0.002 s |

Several intuitive optimizations have already been explored:

- alternative point placement gave at most a 0.454% quality improvement;
- refining every instance with a box improved quality by 1.74% but increased runtime by 21.86%;
- prompt batch sizes from 96 through 384 did not provide a meaningful gain and increased memory;
- recovering records rejected by intermediate filtering was neutral;
- adding more positive refinement prompts generally hurt compared with one positive prompt.

The later `points+boxes` refinement campaign established a stronger opt-in configuration. One
positive, six negative prompts and geometric acceptance gates improved the primary benchmark by
4.19% and the holdout by 4.89%, but added roughly 35-50% runtime. Negative prompts were the active
ingredient. The quality gain is real, but the cost keeps `refinement=None` as the pipeline default.

### APG3d

The recorded GoNuclear profile attributes about 91% of total runtime to propagation:

| Stage | Time |
|---|---:|
| Propagation | 112.8 s |
| Initialization | 6.1 s |
| Candidate scoring | 3.6 s |
| Prompt derivation | 0.6 s |
| Merge | 0.06 s |

The current implementation already includes several substantial efficiency improvements:

- objects with compatible memory signatures are batched during unconditioned video frames;
- redundant prompt-output consolidation is skipped;
- feature caching adapts to device memory;
- early stopping terminates a pass after all masks have been empty for a patience interval;
- candidates on the same anchor slice are grouped into propagation passes.

The default `early_stop_patience=2` was bit-identical on five deep benchmark crops. It skipped 33.6%
of possible GoNuclear frame steps and 12.2% for C. elegans, but almost none for dense EM data.

Candidate and post-processing experiments found:

- a broader/lower density ladder improved quality by 1.31% but cost 20.72% runtime;
- temporal connected-component filtering improved quality by about 0.2% and cost 6.6%;
- anchor coalescing could save up to 4.69% runtime, but failed quality and consistency checks;
- recovering candidates rejected by intermediate gates was neutral;
- deep anchor refinement improved macro mSA by 2.28% for about 1.4% runtime, but remained below the
  5% quality gate;
- an additional post-propagation point did not provide a useful recovery mechanism.

These results make proposal selectivity and propagation behavior more promising than another broad
global sweep.

## APG2d experiment: multimask selection

### Motivation

`AutomaticPromptGenerator._apply_prompts` asks SAM2 for multimask output, reshapes all returned logits
and scores, and then executes the equivalent of:

```python
best = scores.argmax(dim=1)
logits, scores = logits[index, best], scores[index, best]
```

The non-selected logits are discarded before stability and geometry are evaluated. Thus, the decoder
has already paid most of the cost of producing the alternatives, but APG uses only the generic
predicted-IoU ranking learned by SAM2.

Microscopy provides additional evidence that is not represented explicitly in that scalar:

- whether the mask contains its positive seed;
- whether it captures other proposal seeds;
- foreground-map support and precision;
- mask compactness and bounding-box occupancy;
- consistency or disagreement among the three decoder alternatives;
- the local density of competing prompts;
- stability at the APG mask threshold.

### Initial selector probe

A first probe evaluated ten images, two from each of the five 2D benchmark datasets. The current
selection obtained macro mSA 0.289727. An oracle that chose, per prompt, the alternative producing
the best ground-truth match obtained 0.321208, a 10.9% relative improvement.

This established that useful masks are being discarded, but simple selectors did not capture the
signal. The following all performed worse than predicted IoU alone:

- predicted IoU multiplied by stability;
- admitting all masks to the normal merge;
- penalizing masks that contain foreign prompts;
- simple foreground-overlap penalties;
- fixed score-margin rules.

The negative results matter: preserving all alternatives would increase merge pressure and allows
large or duplicate masks to suppress good records. A selector needs to learn interactions between
features rather than just adding one universal penalty.

### Learned selector probe

The second probe retained all three outputs and constructed per-mask features from data already
available during normal generation: predicted IoU and stability, mask and box geometry, within-group
ranks, seed containment, foreground support, local crowding and pairwise mask agreement.

The reduced-set probe established that these features contain useful selection signal and that the
oracle among the same three masks remains substantially above predicted-IoU argmax. It was used only
to justify a full leakage-safe compact-MLP campaign; the supported implementation now contains only
the selected on-device MLP path.


### Interpretation and limitations

The experiment supports a narrow conclusion: the three first-round alternatives contain substantial
quality headroom, and their correct ordering can be predicted better than by SAM2's generic IoU score
alone.

Important limitations are:

- the split used the first ten primary and evaluation images per dataset rather than the canonical
  full 240/233-image evaluation;
- DeepBacs is reused by the existing benchmark setup and is therefore not a strict domain holdout;
- the ranker was fitted to a small sample and could exploit dataset-specific correlations;
- feature calculation and retaining all masks have some memory/bandwidth cost even without another
  decoder call;
- selecting the locally best mask per prompt is not necessarily the globally best decision after
  APG's score-ordered overlap merge.

The preferred implementation direction was therefore to use the experiment as supervision evidence,
then test either:

1. calibration or fine-tuning of SAM2's existing IoU head on microscopy masks;
2. a small on-device MLP trained from the richer feature set;
3. the generic predicted-IoU selection as a zero-configuration fallback.

Evaluation must retain image-level splits and add domain-level holdouts where possible. It should
measure decoder time, feature time, peak memory, total time and final merged segmentation rather than
only per-prompt mask accuracy.

## APG2d proposal: uncertainty-gated refinement

### How this differs from multimask selection

The first proposal changes which output is retained from the existing first decoder call. It cannot
create a mask outside the three alternatives produced for the original positive prompt.

The refinement proposal is an escalation mechanism. After first-round selection, it identifies an
instance for which the alternatives remain unreliable and calls the decoder again with the existing
`points+boxes` prompt set. The extra box and negative prompts can produce a mask that did not exist in
the first-round alternatives.

The intended control flow is:

```text
first decoder call
        |
        v
rank its existing multimask outputs
        |
        +-- confident --> retain best first-round mask
        |
        +-- uncertain --> run points+boxes refinement --> accept only if gates pass
```

They should therefore be evaluated together but ablated separately:

- selector only: expected to improve quality with small overhead;
- uncertainty gate plus current predicted-IoU selector: tests whether uncertainty alone targets
  refinement effectively;
- selector plus uncertainty-gated refinement: intended final funnel;
- blanket refinement: quality/runtime reference.

### Candidate uncertainty features

Raw predicted IoU was already tested as a refinement gate and was not sufficiently selective. More
informative signals are:

- margin between the best and second-best ranked alternatives;
- pairwise mask disagreement or area spread;
- instability of the selected mask;
- local prompt crowding and distance to neighboring seeds;
- foreign-seed containment;
- poor foreground precision or incomplete foreground coverage;
- disagreement between the generic IoU head and the microscopy-specific ranker.

Local crowding is especially relevant. The optimization notes refute grouped duplicate supply as an
adaptivity signal, but do not test spatial neighborhood ambiguity directly. This is also where
negative prompts, the effective ingredient in the existing refinement, should help most.

The target is not necessarily to exceed blanket refinement's quality. A useful result would preserve
most of its 4-5% gain while reducing the refined fraction enough to fit the runtime gate.

## APG3d experiment: candidate proposal structure

### Current proposal mechanism

`derive_volume_prompts` computes a three-dimensional density field and labels connected components at
each threshold in a global descending ladder. Each component contributes the voxel on the slice with
the largest component-density sum. Duplicates are removed only when two thresholds produce the exact
same anchor voxel.

The component's other information is discarded, including:

- the range of thresholds over which it persists;
- the density prominence of its peak;
- component volume and shape;
- z extent;
- foreground confidence;
- whether it splits or merges with nearby components as the threshold changes.

### Candidate coverage probe

The candidate probe used the existing deep 30-32-slice crop manifest. A ground-truth object was
counted as seeded when at least one candidate anchor voxel lay inside it. This is only a proposal
coverage diagnostic: an anchor inside an object can still fail SAM2 scoring or propagation, and a
candidate outside the annotated object can still be useful near uncertain boundaries.

| Dataset | GT objects | Base candidates | GT objects seeded | Candidates surviving slice scoring | Propagation passes |
|---|---:|---:|---:|---:|---:|
| C. elegans | 205 | 209 | 123 | 149 | 31 |
| EmbedSeg | 39 | 1,372 | 38 | 488 | 45 |
| GoNuclear | 95 | 351 | 85 | 245 | 33 |
| CREMI | 278 | 3,157 | 162 | 756 | 63 |
| SNEMI | 119 | 2,243 | 99 | 898 | 70 |

The contrast between C. elegans/CREMI and EmbedSeg is the important result. Some modalities remain
proposal-recall-limited, while others already generate far more candidates than annotated objects.
A uniformly broader proposal rule cannot address both efficiently.

### Alternative thresholds and peak suppression

A lower threshold ladder recovered additional seeded objects in C. elegans and CREMI:

- C. elegans increased from 123 seeded objects with 209 candidates to 146 with 257 candidates;
- CREMI increased from 162 with 3,157 candidates to 178 with 5,017 candidates;
- EmbedSeg remained at 38 seeded objects with roughly 1,100-1,400 candidates;
- SNEMI gained only one seeded object while increasing from 2,243 to 3,658 candidates.

Fixed h-maxima suppression also failed to transfer consistently. For example, a stronger suppression
could reduce EmbedSeg to 770 candidates without losing seeded objects, but similar settings reduced
coverage elsewhere. A permissive setting increased CREMI coverage to 185 objects but generated more
than 8,000 candidates.

This supports using component persistence and learned/adaptive ranking rather than selecting another
global threshold or h-maxima value.

## APG3d opportunities

### 1. Persistence-aware proposal ranking

Represent the descending threshold sweep as a component hierarchy and retain metadata for each peak.
Potential features are:

- birth threshold, merge threshold and persistence;
- peak and integrated density;
- component voxel count and z extent;
- foreground mean, minimum and quantiles;
- distance and prominence relative to competing peaks;
- anchor-slice foreground and flow consistency;
- results of the existing SAM2 slice score and stability gates.

The objective should account for downstream cost. Removing candidates from a partially filled batch
may not save a propagation pass; removing enough candidates at one anchor to eliminate a pass is more
valuable. A useful ranker should therefore report both object recall and the resulting number of
passes, propagated frame steps and wall time.

An alternative or supplementary recall source is slice-wise 2D density maxima. These should be
introduced only through the same ranking/gating stage, because blindly adding per-slice candidates
would multiply propagation cost.

### 2. Propagation-aware track confidence and retirement

Current early stopping counts an empty frame only when all objects in a pass are empty. One persistent
or leaking object keeps the pass alive for every other object. The final volume record also derives
its quality priority primarily from the anchor-slice candidate rather than evidence across the track.

Useful per-track telemetry could include:

- SAM2 object-score logits per frame;
- mask area and changes in area;
- confidence-weighted foreground agreement;
- centroid motion and overlap with adjacent slices;
- consecutive empty or low-confidence frames;
- distance from the density component's expected z extent.

It could support two related changes:

- deactivate an individual track after its evidence expires, without requiring the whole pass to be
  empty;
- compute a trajectory-level score for final merging and penalize tracks that leak, oscillate or
  decay.

The first change is not guaranteed to provide an object-frame-proportional speedup. The optimized
video predictor batches compatible objects specifically because this path is kernel-launch-bound.
Shrinking a batch can reduce arithmetic and memory without reducing the number of launches. A proper
prototype must therefore compare wall time for multiple active-batch sizes and verify masks exactly
for tracks that remain active.

### 3. Uncertainty-gated multi-anchor scoring

The current anchor is the density peak on the component's most converged slice. A poor cross-section
can cause a real candidate to fail before propagation or give propagation a weak initial mask.

For long or uncertain components, score two well-separated, high-density slices and either:

- propagate from whichever slice produces the stronger accepted mask; or
- condition one object on both slices before propagation.

This differs from adding a corrective prompt after propagation, which the existing experiments found
unhelpful. It changes initialization rather than trying to repair a finished track. It should remain
uncertainty-gated because extra conditioning costs decoder time and can alter batching/state replay.

### 4. Streaming volume-record construction

The current path materializes full-frame binary masks for all objects and frames in
`video_segments`, then scans and crops them into volume records. Constructing cropped or bit-packed
records as propagation yields frames could reduce host memory, transfers and later scans. This is an
output-equivalent engineering optimization and is lower priority because propagation compute, rather
than record construction, dominates runtime.

### 5. Decoder and propagation launch optimization

The prompt batch-size sweep did not improve APG2d and the optimized video predictor describes the 3D
path as kernel-launch-bound. `torch.compile` or CUDA graphs around stable decoder and unconditioned
tracking shapes may reduce launch overhead without changing the algorithm.

This requires careful measurement of:

- compilation and warm-up cost;
- shape variability between batches;
- end-to-end rather than steady-state timing;
- numerical and segmentation equivalence;
- memory growth from compiled graph variants.

It is a plausible efficiency experiment, not a quality direction.

## Simple scheduling experiment

Candidates at an anchor are currently kept in their existing order and chunked into groups of 16. A
sandbox proxy sorted them by predicted component z span, attempting to put similarly sized tracks in
the same pass so pass-level early stopping would trigger sooner.

The estimated useful frame totals were effectively unchanged overall. Examples were:

- C. elegans: 621 frames for both current and span-sorted scheduling;
- GoNuclear: 731 current versus 730 span-sorted;
- CREMI: 1,221 current versus 1,213 span-sorted;
- EmbedSeg: 1,383 current versus 1,426 span-sorted, which was worse.

Even an oracle ordering offered only modest headroom on most of these sets. Pass reordering alone is
therefore not recommended. Direct track evidence is more likely to help than a component-span proxy.

## Lower-priority implementation efficiencies

### Refinement negative-prompt lookup

`derive_refinement_prompts` searches and sorts negative candidates separately for each instance even
though only a small number are retained. Pre-grouping prompts by owner and using a spatial index,
neighborhood grid, or partial top-k selection could reduce CPU work. A diversity-aware selection—one
negative per nearby instance or angular sector—could also improve quality in crowded scenes.

This affects only opt-in refinement and should be benchmarked before complicating the implementation,
because prompt decoding may still dominate the added round.

### Automatic feature-cache policy

The implementation already supports device caching, CPU offload, eager host embeddings and caching
all slices. An automatic policy based on estimated volume feature size and available host/device
memory could avoid cache thrashing on volumes that are just larger than the adaptive device cache.
This is primarily a deployment improvement, not a new algorithmic optimization.

### Bulk conditioning

Conditioning several objects on the same anchor could potentially be grouped further. Initialization
and candidate scoring are small compared with propagation, however, so this should follow propagation
work rather than precede it.

## Directions not recommended without new evidence

The following ideas were either rejected by the branch's full experiments or failed the sandbox
probes:

- increasing the normal 2D prompt batch above 64;
- generic point relocation;
- applying box or grouped refinement to every 2D instance by default;
- selecting first-round masks by predicted IoU times stability;
- merging all first-round multimask outputs;
- using a fixed foreign-prompt penalty as the multimask selector;
- increasing the number of positive refinement prompts;
- recovering records dropped by intermediate merge or score gates;
- another fixed lower 3D density ladder;
- a single fixed h-maxima suppression value across modalities;
- post-hoc temporal connected-component cleanup;
- anchor coalescing or a larger anchor stride;
- sorting propagation passes by a simple component z-span estimate;
- unconditional post-propagation corrective prompts.

These negative results do not prove that every related mechanism is impossible. They show that a new
experiment should exploit additional information—multimask ambiguity, component persistence or
track-level evidence—rather than repeating a broader fixed heuristic.

## Recommended experiment sequence

Phases A and B are now complete. Their accepted outcomes are the three-token `token_lowres_v1` H64
eager selector and its optional post-merge signed-utility 15% refinement gate, documented in
`finetuning/v2/evaluation/APG_2D_OPTIMIZATION.md`. Phases C and D remain future 3D work.

### Phase A: validate first-round multimask ranking

1. Export all three masks and the proposed features without changing final APG output.
2. Use the full primary split for fitting/calibration and the full existing holdout for evaluation.
3. Add at least one domain-held-out evaluation if the available datasets permit it.
4. Evaluate compact MLP sizes and a calibrated IoU head.
5. Measure final merged mSA, per-dataset regressions, total runtime, decoder time and peak memory.
6. Inspect oracle headroom after merging, not only per-prompt oracle matches.

Success should require the branch's full quality gate, no material dataset regression and negligible
runtime overhead.

### Phase B: combine ranking with selective refinement

1. Freeze the best validated first-round selector.
2. Calibrate an uncertainty score from rank margin, disagreement and local crowding.
3. Sweep the fraction of instances refined rather than only a raw confidence threshold.
4. Compare selector-only, gate-only, selector-plus-gate and blanket refinement.
5. Report quality against the actual added decoder calls and end-to-end runtime.

The intended outcome is a Pareto curve: selector-only at near-baseline cost, then progressively more
selective refinement for users who want additional quality.

### Phase C: instrument APG3d before modifying it

1. Record component persistence and geometry at proposal time.
2. Record per-frame object score, area and foreground agreement during propagation.
3. Attribute false negatives to never proposed, slice-score rejection, propagation failure or merge.
4. Measure active object counts and wall time per frame and batch size.
5. Estimate how many candidate removals actually eliminate a propagation pass.

This instrumentation separates three different interventions: improving proposal recall, pruning
expensive false candidates, and terminating bad tracks. Without it, a quality improvement can easily
increase the already dominant propagation cost.

### Phase D: test 3D changes in increasing-risk order

1. Persistence-aware candidate reranking with unchanged propagation.
2. Trajectory-level scoring used only in final merge.
3. Conservative per-track retirement while preserving masks for surviving tracks.
4. Multi-anchor initialization for only uncertain candidates.
5. Compiled/graph-captured propagation after algorithmic behavior is stable.

Every phase should retain the existing deep and shallow benchmark protocols, per-dataset guards,
runtime limits, and exact-output checks for purportedly behavior-preserving changes.

## Overall conclusion

The existing optimization work has already exhausted most broad, global heuristics. The remaining
headroom is more conditional:

- In 2D, exploit information already present among the decoder alternatives, then spend refinement
  only when that information says the first round is genuinely ambiguous.
- In 3D, distinguish recall-limited modalities from candidate-saturated ones, and use evidence over a
  track's lifetime rather than relying exclusively on its density anchor and one slice score.

The 2D sequence is complete: compact three-token ranking produced a significant quality and runtime
improvement, and post-merge signed-utility gating retained more refinement gain with only 1.19%
aggregate overhead. The strongest remaining experiment is now 3D instrumentation followed by
persistence-aware candidate ranking; propagation changes should come only after measuring how active
batch size and per-track retirement translate into actual wall time.

## Ranked 3D agenda after the completed 2D campaigns

### What the 2D results change

The completed 2D work sharpens the 3D agenda in four ways.

1. **A learned score should control eligibility as well as ordering.** In 2D, replacing predicted-IoU
   ordering alone was useful but modest; the large gain appeared when the learned score also controlled
   the initial filter. This is even more important in 3D because a false positive that survives the
   anchor-slice filter pays for a full propagation.
2. **Compact decoder evidence is enough.** Three mask tokens plus low-resolution mask statistics beat
   the dense feature path while being faster and smaller. A 3D model should reuse this architecture and
   on-device extraction, not reintroduce CPU/NumPy mask features, trees, a linear head, or a fourth mask
   token.
3. **Score at the stage where the relevant evidence exists.** The 2D post-merge gate beat a pre-merge
   one because it saw the accepted mask and assembled prompts. In 3D this implies two different scores:
   one before propagation, when the decision is whether a candidate is worth the cost, and one after
   propagation, when the complete trajectory is available for final merge ordering.
4. **Train on signed downstream utility.** Clipping harmful refinements to zero hides exactly the cases
   a gate needs to reject. A 3D refinement or alternate-anchor gate should predict the signed change in
   the completed track or final segmentation, not only anchor-slice IoU.

One detail prevents a mechanical copy of the 2D selector. The ordinary unrefined 3D path uses the
selected anchor mask for score filtering and anchor-slice duplicate suppression, but then conditions
the video predictor from the original point again. Choosing a different one of the three anchor masks
does not directly choose a different propagated mask. The useful target is consequently **track
utility**—whether this prompt should be propagated and how its resulting volume should rank—rather
than only which anchor cross-section has the best 2D IoU. Passing the chosen unrefined mask into the
propagator is not recommended: that path was already measured slightly negative.

The quantitative backdrop is unusually clear. On the deep benchmark, 2,536 candidates become 242
propagation passes and 7,604 pass-frame steps; the baseline takes about 3,171 seconds and peaks at
11.15 GB. Propagation is about 91% of the measured GoNuclear runtime. The lower threshold ladder
recovered only +1.31% macro mSA for +20.72% runtime, whereas the existing anchor refinement reaches
+2.28% for only +1.4% at depth. Any new quality route should therefore improve decisions around the
existing candidates before it broadens their supply.

### Ranking

| priority | option | primary purpose | expected cost/risk | recommendation |
|---:|---|---|---|---|
| 1 | learned candidate-to-track funnel | quality and propagation selectivity | medium; needs a larger leakage-safe 3D training set | pursue first |
| 2 | post-anchor signed gate for the existing refinement | additional quality with bounded second-round work | medium; paired propagation labels are expensive once | run on the frozen priority-1 winner |
| 3 | full-volume feature residency and eager host embeddings | output-exact runtime reduction | low algorithmic risk; bounded GPU/host memory cost | run as the first short performance screen |
| 4 | trajectory-aware per-track retirement | reduce dominant propagation work | high implementation risk; speedup is not proportional to retired objects | instrument first, implement only if the wall-time oracle is large |
| 5 | uncertainty-gated alternate-anchor initialization | recover propagation-decay and bad-anchor failures | high cost and batching risk | run only after error attribution on priorities 1-2 |
| 6 | sparse streaming records and compiled tracking kernels | transfer, host-memory and launch efficiency | engineering-heavy, quality-neutral in intent | follow only after the higher-level path is fixed |

### Prerequisite: a trainable 3D benchmark and reusable track cache

The existing five standard and five deep crops are adequate for deterministic regression and canonical
timing, but not for selecting an MLP. The earlier diagnostics explicitly found that the standard error
on EmbedSeg and GoNuclear exceeds the spread of the parameter grid, and several 3D refinement rounds
were ranked by one changing crop. Before fitting any score, construct a larger optimization manifest
from non-overlapping validation crops and a frozen, source-disjoint holdout wherever the source data
permits it.

Splits must be grouped by source volume, not by candidate or crop. Overlapping crops and crops from the
same small source must never cross an OOF fold. Report both shallow and 30-32-slice results, but make the
deep set the deployment decision because propagation and fixed per-candidate overhead have a different
ratio there. A leave-one-dataset-out diagnostic should accompany the ordinary five-fold OOF screen; it
is not required to win every held-out modality, but it will show whether the model learned general
track evidence or dataset identity.

The extraction format should preserve one stable identity for every density component, prompt group,
anchor alternative and propagated track. It should include:

- the threshold-component lineage and all proposal features described below;
- the three anchor masks' compact features, mask tokens and OOF scores;
- the anchor-slice fate (`low score`, duplicate, truncated, kept, or empty);
- one cached unrefined volume record per unique point prompt;
- per-frame trajectory accumulators and the final candidate-to-ground-truth match;
- for the refinement campaign, paired unrefined and refined tracks for the same candidate.

This cache is what makes a real sweep feasible. The ordinary point-conditioned track is independent of
which anchor alternative supplied the filtering score, so each unique prompt can be propagated once.
OOF scorers, eager/deferred anchor merges, learned thresholds and final merge scores can then be replayed
from cached records without rerunning SAM2 for every configuration. A broadened proposal campaign can
likewise cache each new prompt once, then measure exactly which added tracks survive each policy.

### Priority 1: learned candidate-to-track funnel

This is the best remaining overall option because it can improve quality at both merge stages and can
remove expensive false candidates before propagation. It has two small Torch models with deliberately
different information and responsibilities.

#### 1A. Pre-propagation candidate scorer

Retain the existing three SAM2 multimask outputs on the anchor slice and extract the same
`token_lowres_v1` evidence that won in 2D. Concatenate group context with 3D proposal features that are
currently discarded by `derive_volume_prompts`:

- birth and merge thresholds, persistence, and which ladder levels contain the component;
- peak, integrated and percentile density;
- component volume, bounding-box occupancy, z extent and anchor position within that extent;
- foreground mean/precision and flow agreement in the component and anchor cross-section;
- distance and prominence relative to competing peaks;
- predicted-IoU/stability ranks, mask-token context, alternative agreement and score margin;
- number and distance of same-slice candidates and the expected pass occupancy at that anchor.

Start with direct regression to the matched unrefined track's 3D IoU, because direct IoU regression
won the 2D architecture sweep. Also evaluate a signed marginal-utility target derived by adding or
removing the cached track from the final merge. Keep the model family compact: groupwise H32, H64 and
H128 MLPs, image/source-grouped OOF predictions, and exactly three alternatives.

The learned score should be ablated in three roles:

1. select the anchor alternative used for same-slice duplicate suppression;
2. apply the initial candidate threshold before propagation;
3. provide the provisional final merge order before trajectory evidence is added.

Compare predicted IoU, token-only, low-resolution-only, and token-plus-low-resolution inputs. Compare
eager and deferred anchor merging only after architecture and threshold are fixed. Report candidate
count, unique anchors, `ceil(candidates_at_anchor / 16)` pass count, predicted object count and cached
final mSA. Removing a candidate from a partially filled pass may save quality but not time; a claimed
speedup is credible only when it removes passes or is confirmed end to end.

#### 1B. Post-propagation trajectory scorer

The final merge currently orders a track by its anchor predicted IoU times anchor stability. That score
cannot see a mask that leaks, oscillates, decays, disappears and resumes, or conflicts with the decoder
foreground away from the anchor. Accumulate compact per-track evidence during the already-required
propagation:

- object-score-logit mean, minimum, quantiles and slopes away from the anchor;
- mask area, area ratio to the anchor, abrupt area changes and empty-gap statistics;
- foreground support/precision and convergence-density support along the track;
- adjacent-slice overlap, centroid motion, border contact and directional asymmetry;
- observed z extent versus the proposal component's z extent;
- anchor score, pre-propagation MLP score, refinement state and conditioning strategy.

Compute reductions on the device and transfer only a few scalars per track. Fit a small pointwise MLP
to direct final-track IoU and signed final-merge utility, then use its score for final filtering and
merge ordering. Screen four frozen variants: current anchor score, pre-propagation scorer only,
trajectory scorer only, and the two-stage combination. This is the 3D counterpart of delaying a
decision until the useful evidence exists, and it adds no decoder or propagation call.

#### 1C. Persistence-based recall expansion, only after 1A-1B

Once the learned funnel is validated on the unchanged `(1.5, 10.0)` ladder, add proposal supply from a
lower threshold or explicit component-persistence hierarchy. Do not adopt another fixed ladder. The
new candidates must pass the frozen pre-propagation scorer, and the screen must enforce explicit budgets
on propagated candidates and passes.

The decisive comparison is not broadened ladder versus baseline; that was already rejected. It is:

1. current ladder plus learned funnel;
2. persistence-expanded candidates plus the same frozen funnel;
3. the rejected lower ladder without learned filtering as the cost reference.

Report genuine-miss recovery separately from false candidate propagation. The expansion is useful only
if it recovers seeded objects in recall-limited C. elegans/CREMI without reproducing the candidate
explosion already observed in EmbedSeg/SNEMI.

### Priority 2: signed utility gate for anchor refinement

The existing 3D `points+boxes` refinement is the strongest validated quality mechanism: one positive,
four negatives, `min_consistency=0.85`, and `conditioning="prompts"` give +2.28% macro mSA for +1.4%
runtime at depth. It currently refines every anchor survivor and relies on a geometric consistency gate
afterward. The 2D result suggests replacing that blanket decision with a learned signed gate.

The 3D gate must run **after anchor-slice merge and prompt assembly but before the second decoder call**.
A post-volume-merge gate is too late because re-prompting a completed track changes its conditioning
state. Inputs should combine the frozen priority-1 score with the actual visible anchor mask, claimed
fraction, same-slice neighbors, selected negative prompts, density persistence/z extent, alternative
margin and anchor location.

Build paired supervision by propagating each training candidate twice: once from the ordinary point and
once from the accepted tuned refinement. The target is
`refined track IoU - unrefined track IoU`, retaining negative values. Where cached tracks interact in
the final merge, also record signed change in the final segmentation. Compare:

- the current blanket refinement plus geometric consistency gate;
- a positive-benefit MLP, to isolate the effect of signed training;
- a signed-utility MLP at fixed selected fractions (5, 10, 15, 25, 40, 60 and 100%);
- signed MLP plus the current consistency veto, and signed MLP without it.

Freeze the fraction and threshold on primary OOF predictions, then refit once for holdout. Because the
blanket refinement is already cheap at depth, the primary objective is higher net quality, not merely
fewer decoder calls. A useful gate should retain at least 80% of the blanket gain on both primary and
holdout, introduce no dataset loss beyond 1%, and stay within +10% aggregate/+15% per-dataset runtime
relative to the frozen selector-only route. Prefer a gate that exceeds blanket quality by rejecting
harmful refinements.

### Priority 3: full-volume feature residency

This is the clearest behavior-preserving performance screen and can run independently of learned
scoring. The video predictor caches slice features on the device, but the automatic capacity reserves
only a quarter of free memory. Its own implementation notes that a cache shorter than the volume gives
no full-pass reuse: after walking the volume, slices are fetched again on the next of up to 242 passes.
With lazy zarr-backed embeddings this can include storage reads and host-to-device transfers, a plausible
contributor to the measured node-level generation-time drift.

Screen four initialization policies with identical generation parameters:

1. current adaptive device cache plus lazy embeddings;
2. current device cache plus eager host-resident embeddings (`lazy_embeddings=False`);
3. all slice features resident on the device (`cache_all_slices=True`);
4. eager host embeddings plus full device residency.

At the documented roughly 90 MB per slice, 30-32 deep slices require about 2.7-2.9 GB in addition to the
11.15 GB measured peak, so full residency appears plausible on the 20 GB MIG used for the campaign but
must be measured rather than assumed. Include at least one longer-volume stress case that cannot fit, so
an eventual automatic policy has a tested fallback.

If full residency wins, replace the partial adaptive choice with a binary deployment policy: cache the
whole volume only when measured entry size plus the maximum propagation-state reserve fits with a safety
margin; otherwise keep the small interactive cache. A near-full cache that cannot hold the complete
walk should not consume memory without reuse. This route must be bit-identical, including object ids,
and should be accepted only with at least 5% aggregate speedup, no dataset more than 2% slower, and no
out-of-memory failure under the declared fallback.

### Priority 4: trajectory-aware per-track retirement

Pass-level patience 2 is output-preserving on the deep crops but cannot stop when one long-lived object
keeps a batch active. Per-track retirement is still the largest conceptual propagation reduction, but the
current predictor batches compatible objects and is launch-bound. Retiring fifteen of sixteen objects
may leave one forward launch per frame, so object-frame counts are not a speedup estimate.

Instrument before implementing:

- active and empty objects per frame and direction;
- `object_score_logits`, mask area and foreground support per object;
- memory-signature group sizes passed to `_track_frame_batch`;
- CUDA time for group sizes 1, 2, 4, 8 and 16 with warm caches;
- an oracle last-useful-frame bound from ground truth and a prediction-only conservative bound;
- projected launches, batch elements and actual wall time saved by each bound.

Proceed only if the prediction-only oracle projects at least a 10% aggregate end-to-end win after the
measured batch-size curve. The first prototype should deactivate a track direction only after a run of
empty masks, negative object-score logits and absent foreground/density support beyond the component's
expected z extent. Retired objects emit empty masks while survivors stay batched; their old memories must
not alter surviving tracks. Sweep conservative patience/margin values and compare on top of the existing
pass-level patience 2.

An efficiency promotion needs at least 5% end-to-end speedup on every dataset with no dataset losing
more than 0.5% mSA. If a stricter policy is bit-identical, the output-preserving exception used for
pass-level early stopping may be considered, but only after three timing trials and a sparse-volume test
where masks can disappear and return.

### Priority 5: uncertainty-gated alternate-anchor initialization

Use this only if the trajectory diagnostics still attribute substantial genuine misses to decay from a
poor anchor. Retain each density component's top separated high-density slices and evaluate the safer
variant first: score two candidate slices, choose one anchor, and propagate once. This spends an extra
2D decoder call but keeps one conditioning frame per track.

Only if choose-one-of-two shows clear oracle headroom should a small selected fraction receive two
conditioning frames. Multi-anchor objects have different memory-frame signatures and split out of the
large propagation batch, so even a few can be much more expensive than their decoder calls suggest. The
prompt-state replay fix makes this path correct, not cheap.

Train a signed gate on the change in completed-track utility from the alternate anchor versus the default.
Use component z extent, the default/alternate anchor scores and margins, foreground/density support,
distance between slices, and the priority-1 uncertainty features. Freeze a small fraction on primary and
report decoder calls, unique anchors, propagation passes, memory-signature group sizes and total runtime.
Do not test unconditional two-anchor conditioning, four mask tokens, or the previously rejected strategy
of adding approximate flow boxes and neighboring negatives to the first decoder call.

### Priority 6: secondary engineering work

Two implementation optimizations remain credible after the algorithmic path is fixed:

- **Sparse streaming records.** Consume propagation frames into per-object bounding-box accumulators
  instead of retaining `video_segments`, use GPU row/column reductions, and transfer only occupied crops
  where that beats one batched full-frame transfer. This directly applies the 2D lesson that mask transfer
  and record materialization matter once dense alternatives are removed. It must reproduce volume records
  and final segmentations bit for bit.
- **Compiled tracking kernels.** Compile `_track_frame_batch` or bucket stable group sizes before trying
  CUDA graphs. Measure warm-up, graph count, dynamic memory-signature recompilation, peak memory and full
  end-to-end time. The 7,604 deep pass-frame steps can amortize compilation, but shape variability can
  erase the gain. Do not combine compilation with retirement until each is independently measured.

Multi-GPU propagation of independent passes is also possible, but it is deployment scaling rather than a
single-device APG optimization: it duplicates the video model and feature residency and should be reported
as throughput per volume and per GPU, not as an algorithmic speedup.

### Campaign order and decision gates

The recommended order is:

1. add the larger grouped manifest, component lineage, track cache and trajectory telemetry;
2. run the short feature-residency performance screen while cached track extraction is running;
3. fit the pre-propagation scorer on the unchanged candidate ladder;
4. add trajectory rescoring and freeze the complete candidate-to-track funnel;
5. test persistence-based recall expansion through that frozen funnel;
6. train the signed anchor-refinement gate on top of the winner;
7. implement retirement only if its telemetry oracle clears the wall-time threshold;
8. test alternate anchors only if the remaining errors are demonstrably anchor/decay failures.

Use three distinct acceptance routes rather than forcing every hypothesis through one number:

- **quality:** the established +5% macro mSA gate, at most two datasets below -5%, and no dataset above
  +10% runtime unless the >=10% all-datasets quality exception applies;
- **incremental learned gate:** positive macro gain over the frozen parent, no dataset below -1%, at
  least 80% retention of the referenced blanket gain on primary and holdout, <=10% aggregate and <=15%
  per-dataset runtime growth;
- **output-exact performance:** identical segmentation and object ids, >=5% aggregate speedup, no dataset
  more than 2% slower, and a declared memory fallback that does not OOM. A non-exact performance change
  keeps the stricter existing >=5% per-dataset speedup and <=0.5% quality-loss gate.

Every canonical result needs three timing trials bracketed closely by controls because the earlier 3D
campaign measured 3-7% node-level drift. Report initialization and generation separately, plus cache
hits/bytes, candidate and pass counts, object-frame batch elements, actual tracking launches, refined
fraction, trajectory-feature time, transfer time and peak CUDA memory. These measurements make it
possible to reject a mechanism for the right reason rather than mistaking fewer candidates or objects
for fewer expensive launches.

## Stopped 3D learned-funnel campaign: complete report

### Decision and scope

The learned 3D funnel effort was stopped after the primary OOF campaign and a targeted correction
campaign completed on 24-25 August 2026. No learned artifact met its promotion gate. The holdout was
therefore never opened, no library default was changed, and the conditional refinement, recall-expansion,
and retirement stages were not advanced. The temporary campaign implementation was removed from the
source tree after this report was written. The generated caches, logs, model artifacts, and screening
tables remain under:

```text
/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/3d_campaign
```

They occupy approximately 1.3 GB across 3,971 files and are the reproducibility record for the numbers
below. The campaign used the frozen SAM2 Hiera-tiny checkpoint identified by
`85fb099c4bb038fa0ab9bddd6151689e` and the unchanged `(1.5, 10.0)` 3D proposal ladder.

The campaign was designed to answer four questions raised by the strong 2D learned-scorer result:

1. Was the first 3D loss caused by assigning the wrong target to the three anchor alternatives?
2. Can a pre-propagation model identify useful point-conditioned tracks and avoid propagation work?
3. Does post-propagation trajectory evidence improve final filtering or merge ordering?
4. Would a selected anchor mask become useful if it actually conditioned propagation, and does the 2D
   scorer transfer to that setting?

The quality route required at least +5% dataset-balanced primary mSA, at most two datasets below -5%,
and the documented runtime guard. The output-exact residency route required at least 5% aggregate
speedup with no dataset more than 2% slower. The counterfactual mask-conditioned route required both a
positive lower 95% confidence bound on random-stratum track IoU and recovery of at least 50% of the
mask-track oracle headroom.

### Data, split, and cache design

The original five standard and five deep crops were too small for model selection, so deterministic
manifests were built for 32 primary and 10 sealed holdout deep crops. The primary set contained one
C. elegans crop, three EmbedSeg crops, 22 GoNuclear crops, three CREMI crops, and three SNEMI crops. The
holdout contained one, one, six, one, and one respectively. Folds were grouped by source volume; where
only one source existed, a spatially disjoint crop was reserved for holdout. All reported learned
results are source-grouped five-fold OOF results on primary. Leave-one-dataset-out weighted errors were
also recorded during fitting, but were diagnostic rather than a selection criterion.

The reusable extractor recorded stable candidate identities and component lineage, the three compact
SAM alternatives, mask tokens, low-resolution mask statistics, proposal context, anchor-slice fates,
point-conditioned tracks, cropped bit-packed masks, direct 3D IoU targets, and prediction-only
trajectory reductions. It also instrumented feature-cache hits and misses, tracking launches, realized
batch sizes, active/empty object-frame elements, object-score logits, and deferred CUDA timings. Each
sample had a checksummed identity covering the manifest, checkpoint, implementation, feature schemas,
and prompt parameters. Pass fragments were written atomically so a preempted crop could resume without
mixing incompatible cache identities.

The completed primary cache produced 24,946 proposal candidates. The historical anchor filter and
same-slice merge retained 9,897 point-conditioned parent tracks over 934 anchor slices and 1,204
propagation passes. This distinction became important: a proposal candidate, an anchor alternative,
and a completed track are three different decision units.

### Implementation that was tried

The temporary implementation added the following opt-in facilities without changing defaults:

- a versioned Torch artifact loader with explicit `anchor`, `candidate`, `trajectory`, and `refinement`
  stages and checksummed feature schemas;
- 20 component/proposal features, grouped compact mask alternatives, and 27 full-trajectory features;
- separate installed models for anchor alternative selection, pre-propagation candidate filtering,
  post-propagation track scoring, and the later refinement gate;
- volume score-filter and track-order controls in APG plus benchmark CLI arguments for explicit artifacts;
- cache-residency controls for adaptive/full device features and lazy/eager host embeddings;
- predictor telemetry for feature reuse, launches, batch elements, object-score logits, active tracks,
  and CUDA time by realized group size;
- manifest preparation, restartable extraction, grouped OOF training, cache replay, counterfactual
  mask-conditioned propagation, Slurm submission, and screening tools;
- tests for manifests and identity checks, source-grouped OOF behavior, resume semantics, feature
  construction, scorer-stage validation, replay, and the counterfactual bootstrap gate.

The model family was deliberately compact. Anchor models used three alternatives and direct anchor-plane
IoU. Candidate models used permutation-invariant grouped alternatives and were trained to either direct
point-conditioned track IoU or signed pre-propagation merge utility. Trajectory models used one row per
completed historical track and were trained to direct track IoU or signed final-merge utility. The input
ablations were token-only, low-resolution-only, and token plus low-resolution; widths were H32, H64, and
H128. Thresholds were derived from non-test folds at fixed retained fractions rather than selected on
the evaluated fold.

### Execution history and operational corrections

The first remote extraction array, job `15465049`, and the first two residency arrays, jobs `15465229`
and `15465230`, failed before doing scientific work. Their scripts enabled `set -u` before sourcing the
cluster bash setup, whose `/etc/bashrc` reads an unset `BASHRCSOURCED` variable. The launchers were fixed
to source the environment first and enable nounset afterward. The jobs failed in 23-48 seconds with exit
code 1; none produced results that entered an aggregate.

Local sample-zero validation intentionally invalidated several early cache identities while the feature
ablations and telemetry were completed. These attempts were moved to `stale/` rather than silently
reused. The preserved names identify the transitions: identity-only, pre-ablation schema,
pre-CUDA-telemetry, and a CUDA-counter correction. The final extraction array `15466938` completed all
31 remote crops with exit code 0, while sample zero used the interactive 20 GB MIG. Runtime varied from
about two minutes to 74 minutes because candidate/pass counts differ sharply by crop; accounting and
fragment completeness confirmed that the short jobs were valid completions, not kills. The array was
initially throttled to three and was increased to eight after recognizing that these jobs contend with
all other queued GPU work in the same way as the residency jobs.

The first fitter, job `15466943`, completed in 1:55:06. The corrected target array `15500050` and
counterfactual array `15500154` likewise completed every shard with exit code 0; the long corrected
target shards took 43:25, 1:02:56, and 28:01, while most smaller shards took one to nine minutes. The
corrected primary fitter `15500055` was promoted from `grete:preemptible` to `grete:interactive`, raising
its scheduling priority by approximately 100,000; because the two partitions share the same physical
nodes and do not preempt running jobs, its predicted start initially remained unchanged. Slurm refused
the same promotion for dependent job `15500461` because of the interactive QOS job limit. This caused no
loss because it was still dependency-blocked. The jobs ultimately completed in 1:09:57 and 0:06:11,
respectively, both with exit code 0.

### Output-exact feature-residency screen

Four serial, identically configured deep screens compared adaptive/full device caching with lazy/eager
host embeddings. Job array `15466942` completed all four configurations. The reported quality, proposal
counts, pass counts, and peak CUDA allocation were identical. Aggregate timings were:

| device cache | host embeddings | initialization (s) | generation (s) | total (s) | delta vs adaptive/lazy |
|---|---|---:|---:|---:|---:|
| adaptive | lazy | 89.275 | 2,877.936 | 2,967.212 | control |
| adaptive | eager | 93.637 | 2,885.259 | 2,978.896 | +0.39% |
| full | lazy | 89.461 | 2,898.498 | 2,987.959 | +0.70% |
| full | eager | 94.220 | 2,878.591 | 2,972.812 | +0.19% |

All four reported dataset-balanced mSA `0.314143`, peak CUDA allocation `11,154,618,368` bytes,
11,670 feature-cache hits, 158 misses, 6,756 tracking launches, and 73,742 tracked batch elements. The
158 misses are effectively one miss per volume slice: on a 20 GB MIG the existing adaptive policy
already held the complete 30-32-slice feature set. Full residency therefore created no additional reuse,
and eager host loading only moved small initialization costs. The largest timing difference was 0.7%,
in the wrong direction and well inside the known node variation. No configuration cleared the 5%
performance gate, so no three-trial confirmation or longer-volume fallback test was warranted.

### First learned candidate campaign

The first aggregate treated all 24,946 proposal groups as candidate/track examples and fitted nine
direct-IoU candidate models: three input schemas by three widths. OOF Pearson correlations ranged from
near zero to `0.3862`; the strongest pointwise correlation came from token plus low-resolution H32. The
cache replay tested learned alternative selection, initial filtering, and final merge ordering at a
threshold grid.

The historical control won:

| policy | dataset-balanced mSA | candidates | anchor slices | passes | predicted objects |
|---|---:|---:|---:|---:|---:|
| historical anchor control | **0.337022** | 9,897 | 934 | 1,204 | 1,912 |
| best learned policy, token+lowres H32 selection/filter at 0.425 | 0.334496 | 9,237 | 954 | 1,178 | 1,905 |

The best learned route lost `0.002526` absolute mSA, approximately 0.75% relative. More aggressive
learned selection reduced candidates and passes but lost more quality. No configuration passed the
primary gate, so trajectory fitting, the holdout, and end-to-end learned-artifact confirmation were
stopped at this point.

Reviewing this screen exposed a real objective error. One point-conditioned 3D track IoU had been
broadcast to all three anchor alternatives, although replay used the three predicted scores in an
`argmax`. Such identical labels cannot train an alternative selector. It also obscured that ordinary
unrefined propagation restarts from the original point: choosing another 2D anchor mask changes local
filtering and duplicate suppression, but does not choose the propagated track. The first negative result
was therefore not treated as conclusive; the decisions and supervision were separated in a corrected
campaign.

### Corrected campaign

The corrected aggregate contained 24,946 proposal candidates and 9,897 historical parent tracks. It
trained three independent stages:

- `anchor`: three direct anchor-plane IoUs for the three alternatives;
- `candidate`: one point-conditioned track IoU or signed removal utility for each proposal group;
- `trajectory`: one completed-track IoU or signed final-merge utility for each historical survivor.

Each screen changed only the tested role. Anchor selection retained the historical candidate threshold
and final ordering; candidate filtering retained historical anchor selection and final ordering; the
trajectory screen started from the full historical parent. This prevented one scorer from receiving
credit for a different decision.

#### Corrected anchor selection

The best anchor-plane model was low-resolution-only H64. Its OOF Pearson correlation was `0.68795`, so
the corrected labels produced a learnable 2D target. The downstream effect was nevertheless negligible:

| policy | dataset-balanced mSA | candidates | passes | predicted objects |
|---|---:|---:|---:|---:|
| historical control | 0.337022 | 9,897 | 1,204 | 1,912 |
| low-resolution H64 anchor selection | **0.337415** | 9,889 | 1,202 | 1,910 |

This is only `+0.000393` absolute or `+0.12%` relative. C. elegans, CREMI, EmbedSeg, and GoNuclear were
unchanged; the complete gain came from SNEMI (`0.481382 -> 0.483347`). A model can predict the anchor
cross-section well without materially changing point-conditioned 3D propagation.

#### Corrected pre-propagation candidate filtering

The one modest positive result was a token-only H128 model trained on direct track IoU and used only as
an 85%-retention filter. Learned ordering reduced its gain, and every signed-utility winner was below the
historical control.

| dataset | historical mSA | learned filter mSA | absolute delta |
|---|---:|---:|---:|
| C. elegans | 0.031763 | 0.033744 | +0.001980 |
| CREMI | 0.146845 | 0.143359 | -0.003486 |
| EmbedSeg | 0.629028 | 0.641389 | +0.012360 |
| GoNuclear | 0.396090 | 0.385431 | -0.010659 |
| SNEMI | 0.481382 | 0.489397 | +0.008015 |
| **dataset-balanced** | **0.337022** | **0.338664** | **+0.001642** |

Aggregate candidate count fell from 9,897 to 8,940 (-9.7%), anchor slices from 934 to 895 (-4.2%),
passes from 1,204 to 1,103 (-8.4%), and predicted objects from 1,912 to 1,820 (-4.8%). These are credible
reductions in projected propagation work, but the cache replay time is not an end-to-end GPU timing. The
quality gain is only +0.49% relative, an order of magnitude below the +5% gate, and it trades meaningful
CREMI/GoNuclear losses for EmbedSeg/SNEMI gains. It was therefore not promoted or evaluated on holdout.

#### Corrected post-propagation trajectory scoring

No trajectory policy beat the historical order. The best learned alternative was direct-IoU H64 at
100% retention, meaning it only reordered tracks:

| policy | dataset-balanced mSA | absolute delta |
|---|---:|---:|
| historical control | **0.337022** | control |
| direct H64 trajectory order, 100% retained | 0.332420 | -0.004602 |

Filtering tracks made the result worse. The direct H64 model had OOF Pearson `0.39347`, but pointwise
track-IoU correlation did not preserve the non-local overlap decisions made by final score-ordered merge.
Utility models also lost. The tested 27-feature trajectory representation and pointwise MLP objective
therefore did not justify any deployment or a retirement prototype.

#### Why signed utility did not help

The exact downstream utilities were extremely sparse:

| target | negative | exactly zero | positive | maximum OOF Pearson among tested utility models |
|---|---:|---:|---:|---:|
| candidate removal utility, 24,946 rows | 1.86% | **95.99%** | 2.16% | 0.2603 |
| trajectory removal utility, 9,897 rows | 3.19% | **90.87%** | 5.94% | 0.2539 |

Most tracks either do not survive the final merge or can be removed without changing dataset mSA, so a
regression loss is dominated by exact zeros while the rare nonzero values span roughly `-1.87` to
`+1.99` object-equivalents. Direct-IoU models reached correlations up to `0.4061` for candidates and
`0.3935` for trajectories, and the only positive downstream policy used the direct target. Retaining
the sign was conceptually correct, but it did not solve the rarity and interaction problem.

### Mask-conditioned counterfactual campaign

The ordinary path does not propagate the selected anchor mask, so a staged counterfactual explicitly
called mask-prompt conditioning for each of the three alternatives. The canonical manifest contained
750 groups: for each dataset, 75 random candidates, 40 ground-truth anchor-regret candidates, and 35
low-margin candidates. All three masks were propagated. The screen compared historical predicted-IoU
selection, corrected primary anchor models, one dedicated token+low-resolution H32 model trained on
these tracks, the frozen 2D selector, an anchor-plane-IoU oracle, and a mask-track-IoU oracle.

On the unbiased random stratum, real alternate-mask headroom existed:

| policy | dataset-balanced selected-track IoU | delta vs historical | paired 95% interval | oracle headroom recovered |
|---|---:|---:|---:|---:|
| historical predicted IoU | 0.684892 | control | [0, 0] | 0% |
| anchor-plane-IoU oracle | 0.704042 | +0.019150 | [0.008012, 0.029819] | 46.6% |
| mask-track-IoU oracle | 0.725962 | +0.041070 | [0.032241, 0.050744] | 100% |
| best primary learned policy | 0.684083 | -0.000810 | [-0.007734, 0.006074] | -2.0% |
| dedicated counterfactual H32 model | 0.683653 | -0.001239 | [-0.013889, 0.011358] | -3.0% |
| frozen 2D token+lowres H64 | 0.652010 | -0.032883 | [-0.044242, -0.022727] | -80.1% |

The anchor-plane oracle shows that a better cross-section can sometimes choose a better conditioned
track, but even perfect anchor IoU recovers less than half the full track oracle. Neither learned 3D
model recovered positive random-stratum headroom. The 2D scorer transferred especially poorly and was
significantly worse than predicted IoU, demonstrating a genuine 2D-to-3D decision/domain mismatch.

Cached segmentation replay gave the dedicated counterfactual model `0.340147` dataset-balanced mSA
versus `0.337970` for historical selection, a nominal `+0.002176` with interval
`[0.001951, 0.002442]`. This apparent gain was not a valid promotion signal: approximately 96% of the
macro delta came from the single C. elegans primary crop (`0.032670 -> 0.043160`), and the manifest had
deliberately enriched candidates selected with ground-truth anchor regret. A crop bootstrap cannot
measure C. elegans crop variation when only one primary crop exists. The predeclared random-stratum
track gate protected against that selection effect and returned
`advance_mask_conditioned_anchor: false`. For reference, the segmentation replay oracles reached
`0.340365` for perfect anchor IoU and `0.341952` for perfect conditioned-track IoU, while the frozen 2D
scorer fell to `0.335730`.

### What the campaign established

The original supervision bug was real, but it was not the main explanation for the difference from 2D.
After correction, anchor-plane IoU was learnable while downstream 3D quality barely moved. In 2D the
selected mask is the output that enters filtering and final merge; in ordinary 3D the selected mask is
used locally and the video predictor starts again from the point. The learned action and the evaluated
3D track are therefore only weakly coupled.

There is some learnable pre-propagation eligibility signal: a direct-IoU token model removed about 8.4%
of passes with a small OOF quality gain. It was not sufficiently consistent across modalities and did
not approach the quality gate. Post-propagation features did not yield a safe merge order, because a
pointwise score does not capture interactions among overlapping tracks. Exact signed merge utilities
were too sparse for ordinary regression. Finally, forcing the chosen mask to condition propagation
revealed oracle headroom but not a leakage-safe predictor of that headroom; direct transfer of the 2D
model was decisively negative.

Accordingly, the historical predicted-IoU anchor policy, point-conditioned propagation, and historical
track order remain the supported behavior. No 3D learned artifact should be exported or installed from
this campaign. If the topic is revisited, the only evidence-backed starting points are a more strongly
regularized, cross-dataset candidate eligibility classifier or a prediction-only gate that first
identifies the small subset with alternate-mask regret. Either would require a new primary design rather
than opening the sealed holdout for the models reported here.

### Stages deliberately not run

The signed refinement-gate extractor and trainer were implemented but not executed because there was no
frozen learned-funnel winner on which to condition the experiment. Persistence-expanded proposal supply
was not run because the unchanged-ladder funnel failed. Per-track retirement was instrumented but not
implemented; fewer active objects do not necessarily remove launches in the batched predictor, and no
accepted funnel existed for a final end-to-end screen. Alternate-anchor choose-one/two-conditioning was
not advanced after the counterfactual learned gate failed. Sparse streaming records and compiled
tracking kernels were also left untouched. This respected the campaign's decision gates and avoided
spending the sealed holdout on an exploratory configuration.

### Canonical result files

The most important preserved files are:

```text
3d_campaign/jobs/20260824_082649_residency/logs/residency_15466942_*.out
3d_campaign/screening/candidate/9f476137689a23fe2debed3f23a9eae8/summary.csv
3d_campaign/correction_v2/screening/anchor/6b7c31022c03c4b12c57b4c22d736ae7/summary.csv
3d_campaign/correction_v2/screening/candidate/66f42c2710920d0afbeb874aab978513/summary.csv
3d_campaign/correction_v2/screening/trajectory/7780de64b44ca986c17a7eb6aea7680d/summary.csv
3d_campaign/correction_v2/screening/counterfactual/dae562dc67233831110bcc2925779aa2/gate.json
3d_campaign/correction_v2/screening/counterfactual/dae562dc67233831110bcc2925779aa2/track_diagnostics.csv
```
