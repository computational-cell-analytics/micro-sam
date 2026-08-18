# APG v2: what the diagnostics found

The findings of the automatic prompt generation (APG) diagnostics, kept here because the scripts that
produced them are gone. `visualize_apg_2d.py`, `visualize_apg_3d.py` and `sweep_apg_3d_overlap.py`
were deleted on 2026-08-18. They walked the pipeline stage by stage in napari, which needs a desktop,
so they could not run on the cluster where the data and the checkpoints are.

Everything below is measured, not assumed. Where a number has no dataset next to it, it held on every
dataset that was tried.

## How the pipeline runs

An image and a volume take different paths.

- 2d: the flow is integrated in the plane, so a density component is one cross-section of an object.
  One threshold gives the candidates. Every candidate is prompted, and the masks are merged in one pass.
- 3d: the flow is integrated in 3d, so a density component is a whole object. The candidates come from
  a ladder of thresholds rather than one. Every object is prompted once, on the slice its density
  converges on, and the surviving prompts are propagated through the volume by the SAM2 video predictor.

The ladder matters. A single threshold merges the peaks of touching objects into one component, and the
coarser level of the ladder separates them again. It is worth 0.009 mSA across three backbones.

## Where recall is lost

Recall sits well below precision, so objects are lost. The stages an object can drop out of are:

1. seeded: a candidate point lands inside the object at all.
2. proposed: some prompted mask matches it at IoU >= 0.5, before any filtering.
3. scored: that match survives `score_threshold`.
4. merged: it survives the overlap merge, so it is in the output.

The gap between `seeded` and 1.0 is what proposing more candidates could recover. The gap between
`proposed` and `merged` is what the selection throws away, and no extra candidate helps there.

## The two failure mechanisms

The misses split into two kinds that need opposite fixes. Do not treat them as one number.

- **Merge rejection.** The object's own mask propagates well, a neighbour clips about a fifth of it,
  and `merge_by_score` then rejects the whole mask because more than `max_overlap` of it is already
  claimed. The object is left unclaimed. `max_overlap` is the axis that controls this.
- **Never proposed.** No candidate is ever anchored inside the object, so nothing reaches the merge.
  Raising `max_overlap` does nothing here.

Sweeping `max_overlap` over 0.15, 0.3, 0.5 and 0.7 on dense EM data recovers objects on cremi only.
snemi and humanneurons do not respond, because their misses are of the second kind.

A third kind exists on dense volumes: an object that an instance *does* overlap, but below the matching
threshold. That mask is wrong rather than missing. It leaked into a neighbour or stopped short.

## Leaks and propagation decay

On a dense volume the unmatched objects are usually covered by an instance that is not theirs, which
means a propagated mask crossed a membrane and swallowed its neighbour. Measure the leak per instance,
against the ground-truth object it overlaps most. The distribution matters more than the mean: a few
catastrophic merges and a broad boundary bias need different fixes.

A propagated mask is conditioned on one slice and tracked away from it in both directions. If the video
predictor loses the object as it goes, coverage falls with distance from the anchor. Measure it per
(candidate, slice) pair against the object's cross-section on that slice.

## Measurement caveats

These bit us more than once.

- **Severed objects.** A crop cuts objects at its faces, and relabelling promotes each sliver to its own
  object. In a volume an object reduced to one or two slices never reaches `candidate_threshold`, so it
  is never proposed. Counting those as misses blames the method for the crop. `severed_objects` in
  `baselines_common.py` separates them.
- **Do not fix it with `min_size`.** On cremi a neurite cross-section in a thin crop has a median of
  about six voxels, so a size floor deletes genuine annotations instead. Separate the severed objects,
  do not drop them.
- **Report objects, not only the aggregate.** A change that recovers objects while costing precision
  elsewhere is not the same as one that does neither. `genuine_misses` reports both.
- **The 3d benchmark is below its noise floor.** gonuclear and embedseg have too few volumes: the
  standard error exceeds the spread of the whole parameter grid. Do not rank 3d parameters on them.
- **Tuning and evaluation must share one loader.** When they drifted apart, validation tuning became
  anti-predictive.

## Parameters that transfer

- `max_overlap = 0.15` is optimal on eleven of twelve datasets. It is the one axis that transfers.
- `candidate_threshold = 1.5` is the modal per-dataset optimum over twelve datasets, best on six of
  them. A candidate's density scales with the object's size, so a volume needs its own value, which is
  the `(1.5, 10.0)` ladder.
- `n_threads = 8` cuts the flow integration by a factor of five. Sixteen buys nothing more.

## Cost

Measured with `evaluate_apg.py` on one A100 40GB MIG slice, hvit_t, joint v3 checkpoint.

2d, livecell, seconds per image:

| stage | seconds |
|---|---|
| prompt | 0.213 |
| decoder | 0.165 |
| encode | 0.086 |
| interior_points | 0.012 |
| flow_density | 0.012 |
| merge | 0.002 |

Prompting the interactive branch is the largest single stage, and the decoder is next. The candidate
derivation is noise by comparison, so optimising the flow integration further is not worth it.

3d, gonuclear, one volume, seconds:

| stage | seconds |
|---|---|
| propagate | 112.8 |
| initialize | 6.1 |
| score_candidates | 3.6 |
| derive_candidates | 0.6 |
| merge | 0.06 |

Propagation is 91% of the run. Nothing else in the volumetric path is worth optimising until that is.
Holding the tracking state on the device rather than on the host (`offload_to_cpu=False`) gives 1.03x
here for about 17 MB of device memory per slice, with bit-identical output. It is a batch job's to
spend, which is why it is not the library default.

## What moved into the code

The reusable parts of the deleted scripts now live in tracked modules:

- `baselines_common.severed_objects`, `.unmatched_objects` and `.genuine_misses`.
- `micro_sam.v2.automatic_prompt_generation.merge_by_score(..., return_reasons=True)` reports why every
  candidate was kept or dropped: `kept`, `too small`, `duplicate`, or `truncated below min size`. This
  replaces the `merge_with_reasons` fork that the visualizers carried, which had already drifted from
  the function it mirrored.
- `evaluate_3d.py` records `unmatched` and `genuine_misses` next to mSA for every APG variant.

The deleted scripts checked their mirrors against the library on every run (`anchors_match`,
`scoring_matches`, `merge_matches`) and the mirrors agreed. That is why folding the merge fork back into
`merge_by_score` is safe.

## What the walk-through showed

`visualize_apg_2d.py` and `visualize_apg_3d.py` built a napari layer per stage, in this order. Rebuild
from this list if the walk-through is ever needed again. Every intermediate came from the library
itself, so what was shown was what runs.

1. The input crop, the one the evaluation scores.
2. Decoder channel 0, the foreground probability, predicted for the whole image or volume in one pass.
3. The foreground mask at `foreground_threshold`: the pixels a prompt may come from.
4. Decoder channels 1 to 3, the normalized vector to the nearest boundary. 2d uses y and x. 3d adds z,
   which is the channel 2d drops. The y component flips sign across the middle of every object.
5. The length of that vector inside the foreground, which is the distance to the nearest boundary.
6. The flow field, which is minus that vector, so every pixel is pushed away from the boundary.
7. The convergence density: advect every foreground pixel `n_iter` x `dt` along the flow, then smooth.
   One bright peak per object in 2d, one blob per object in 3d, spanning the slices the object spans.
8. What AIS does with the density: seeds at density > 10, then a watershed. Shown next to APG, because
   the two differ only in what turns the same prediction into instances.
9. What APG does instead: candidates at `candidate_threshold`. In 3d this is the ladder, one layer per
   level, showing that the coarse level separates peaks the fine level fuses.
10. The prompts, from `interior_points`, one per candidate component.
11. The masks SAM2 returns per prompt, before the best-scoring one is kept. This is where the ambiguity
    a single point leaves is visible: one object against a cluster.
12. The predicted IoU filter at `score_threshold`, kept in green and dropped in red.
13. 3d only: the propagation passes. Objects sharing an anchor slice propagate together, because the
    video predictor propagates every object of a state from the earliest slice any of them is
    conditioned on.
14. The fate of every candidate in the merge, colour-coded.
15. The unmatched ground-truth objects, split into the ones the crop severed and the ones the method lost.

The candidate fates were: `kept`, `duplicate in 3d`, `duplicate on the anchor slice`, `too small`,
`truncated below min size`, `low score`, `empty on the anchor slice`, `empty after propagation`. The
first four and the last two of the merge are now available from `merge_by_score(return_reasons=True)`.
The anchor-slice fates belong to `_score_candidates` and are not reported by the library.

Two display details that were needed to read a volume at all: a thin slab is nearly flat next to its
512 in-plane pixels, so z was stretched by the physical spacing and then again by a display factor that
each step named. Points are sized in pixels of the image, so a zoomed step had to shrink them.

## Still open

The v3 joint checkpoint improved APG on four of eight datasets but regressed AIS on seven of eight, with
dynamicnuclearnet down 0.166 mSA and embedseg down 0.128 mSA. The decoder got worse for the
post-processing path while the interactive branch improved. Nobody has localised that to a stage yet.
`diagnose_apg_recall.py` is the instrument for it.
