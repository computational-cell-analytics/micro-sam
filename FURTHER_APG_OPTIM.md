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
cap, motivating the later learned-filter campaign.

The decoder-head/filter follow-up resolves this: the same triplet H64 scorer with eager selection
and a primary-selected 0.25 learned-score filter reaches 0.296508/0.295659. Its three-trial holdout
median is 189.9 s versus 177.8 s for same-implementation controls. The formal comparator accepts it:
+11.86% mSA, +6.83% aggregate runtime, no dataset regression and at most +8.68% runtime on any
dataset. A separately trained singleton MLP reaches 0.289056/0.283470 in 174.7 s in a diagnostic
run; the single-token default is itself better and faster than the historical three-mask/IoU path,
but remains below the +5% quality gate. The accepted eager MLP policy remains opt-in because its
artifact is external; changing library defaults was not part of the campaign. The gated MLP and
selective-refinement follow-up retrained the gate on this exact eager/learned-filter policy. Its
primary-selected 50% route reaches 0.299904/0.301455 primary/holdout, but adds 25.32% runtime over
the same-implementation first pass. It is therefore an explicit quality/latency option, not the
deployment recommendation; the selector-only route remains the accepted default candidate.

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

The strongest immediate experiment is full validation of multimask ranking. It offers a plausible
quality gain without another model forward and also supplies the uncertainty signal needed to make
the existing refinement cost selective. The strongest 3D experiment is instrumentation followed by
persistence-aware candidate ranking; propagation changes should come only after measuring how active
batch size and per-track retirement translate into actual wall time.
