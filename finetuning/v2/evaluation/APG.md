# Automatic Prompt Generation (APG) for micro-sam v2

Status as of 2026-08-09. Sections up to 'Cross-dataset results' are LiveCELL only; that section covers
twelve 2d datasets and supersedes the LiveCELL-only conclusions where they disagree.

**Protocol.** Every number is measured on the 512x512 centre crop of each image, not on the full frame.
Published LiveCELL numbers are on full frames and are not comparable to this table.

**The table below is on a superseded ground truth.** It came from `tune_apg.py`, which scores against the
raw cropped labels. `baselines_common._load_data` instead re-labels with connected components, which
promotes every sliver the crop severed to its own object: on livecell that invents about 9 objects per
image at a median of 20 px, none of which any method can predict, because its own `min_size` forbids them.
Scoring the same predictions both ways differs by 0.010. `GT_MIN_SIZE_2D` now drops those slivers, which is
the canonical protocol; see 'Cross-dataset results' for numbers on it. Apply a size floor only where the
data is cropped: dsb arrives at native resolution, relabelling invents nothing there, and a floor would
delete 5.2% of genuine small nuclei instead.

The full LiveCELL split is 1510 images: the test set has 1512 and two are excluded, see
`LIVECELL_EXCLUDED_TEST_IMAGES`. The full split also needs two environment variables,
`MICRO_SAM_EVAL_MAX_SAMPLES` and
`MICRO_SAM_LIVECELL_PER_CELL_TYPE=0`. Without them the evaluation silently scores the stratified 200-image
subset, which reads about 0.08 mSA higher.

## What it is

The flow post-processing (`micro_sam/v2/postprocessing.py`) turns the UniSAM2 decoder's convergence density
into instances by thresholding it. APG instead proposes candidates *below* that threshold, prompts the joint
model's interactive branch with each one, and keeps the masks the model scores highly. The discrimination
that thresholding cannot do is done by the model.

**Why**: on LiveCELL the density threshold has to serve two conflicting needs. A quarter of objects get no
seed at all, but lowering the threshold enough to find them shatters the objects that were already correct,
because real peaks and noise sit at the same density values.

## Results (LiveCELL, 1510 test images, superseded ground truth)

| method | model | mSA | SA50 | SA75 | precision | recall | cost |
|---|---|---|---|---|---|---|---|
| microsam2_ais | hvit_b | 0.3064 | 0.5570 | 0.3114 | 0.7444 | 0.6652 | 1x |
| microsam2_apg | hvit_b | 0.3951 | 0.6628 | 0.4228 | 0.8555 | 0.7325 | ~4x |
| **microsam2_apg+box** | **hvit_b** | **0.4006** | **0.6670** | **0.4290** | 0.8495 | 0.7420 | ~6x |
| microsam_ais | vit_b_lm | 0.3674 | 0.6220 | 0.3868 | 0.8228 | 0.7030 | 1x |
| microsam_apg | vit_b_lm | 0.3931 | 0.6461 | 0.4209 | 0.8315 | 0.7291 | ~2x |
| cellpose4 cpsam | - | 0.4370 | 0.7160 | 0.4713 | 0.8950 | 0.7743 | 1x |

APG is **+0.089 over our AIS**, and the box-refinement stage adds a further **+0.006**. Cellpose-SAM leads
by 0.036. The costs are measured against AIS on the same 48 images: 9 s, 36 s and 57 s. Most of APG's cost
is the few hundred prompts, not the image encoder.

The remaining gap is **detection, not boundaries**, and more false positives than misses:

| | ours | cellpose | gap |
|---|---|---|---|
| SA50 | 0.6670 | 0.7160 | -0.0490 |
| precision | 0.8495 | 0.8950 | -0.0455 |
| recall | 0.7420 | 0.7743 | -0.0323 |
| SA75 / SA50 | 0.6432 | 0.6582 | -0.0150 |

Boundary quality *given* detection is nearly matched, so almost all of the gap sits at SA50. Before the box
stage the two gaps were about equal (-0.040 and -0.042); the box stage traded precision for recall.

**The margin over v1 on livecell is gone.** It was +0.0075 here, already inside one standard error.
Re-measured on the canonical ground truth it is **-0.002** (0.4033 against v1's 0.4052), and v1 still runs
at its library defaults. Do not claim v2 APG beats v1 APG on livecell, and see 'Cross-dataset results':
across twelve datasets v1 APG at defaults beats the tuned v2 APG on six of them.

APG gains more for v2 than for v1, but the informative reading is that both land near the same place from
decoders that differ by 0.06 in AIS. The decoder's density is not the bottleneck; the prompt, mask and
merge stage is.

## Cross-dataset results (2026-08-09)

Twelve 2d datasets, **both v2 engines grid-searched per dataset** on that dataset's own val split, then
evaluated on test. The v1 rows run at library defaults, which is how the v1 manuscript reports them, so
every v1 comparison favours v2. All numbers on the canonical ground truth (relabelled, `GT_MIN_SIZE_2D`).
These are every 2d dataset in `results_automatic.csv` whose val split is large enough to tune on, plus
deepbacs, yeaz and u20s, whose val splits are not and whose rows should not be cited alone.

| dataset | test | APG v2 | AIS v2 | APG v1 | AIS v1 | APG-AIS | p | box ext |
|---|---|---|---|---|---|---|---|---|
| dynamicnuclearnet | 686 | **0.6695** | 0.5523 | 0.4916 | 0.5092 | +0.117 | 3.5e-49 | 4 |
| livecell | 1510 | 0.4033 | 0.3113 | **0.4052** | 0.3803 | +0.092 | 2.5e-244 | off |
| omnipose | 205 | 0.5511 | 0.5076 | **0.6033** | 0.5657 | +0.044 | 2.6e-09 | 0 |
| deepbacs | 35 | **0.4245** | 0.3823 | 0.4001 | 0.3281 | +0.042 | 4.9e-02 | 0 |
| cellpose | 68 | 0.3449 | 0.3027 | **0.4026** | 0.3695 | +0.042 | 4.3e-07 | off |
| dsb | 50 | **0.5880** | 0.5492 | 0.5846 | 0.5819 | +0.039 | 8.7e-07 | 0 |
| vicar | 528 | 0.4550 | 0.4223 | **0.5385** | 0.5280 | +0.033 | 1.7e-21 | 4 |
| tissuenet | 1288 | 0.3131 | 0.2891 | 0.3273 | **0.3357** | +0.024 | 5.1e-42 | 2 |
| cellbindb | 303 | **0.3327** | 0.3177 | 0.3259 | 0.3165 | +0.015 | 1.3e-03 | off |
| yeaz | 62 | 0.7379 | 0.7318 | **0.8313** | 0.8244 | +0.006 | 5.8e-04 | 2 |
| u20s | 172 | **0.7386** | 0.7370 | 0.7309 | 0.7177 | +0.002 | 0.77 | 0 |
| deepseas | 483 | 0.1721 | **0.1843** | 0.1454 | 0.1387 | **-0.012** | 1.9e-10 | off |

**APG beats AIS on 11 of 12, median margin +0.036**, by a paired Wilcoxon over the per-image scores. It
loses on deepseas, significantly. Read the median rather than livecell: livecell (+0.092) and
dynamicnuclearnet (+0.117) are outliers, and the other ten lie between -0.012 and +0.044. APG is a modest
and mostly reliable improvement over AIS inside v2, not the transformation the livecell number suggests.

### v2 against v1 is a coin flip

APG v2 beats APG v1 on **6 of 12, median +0.0007** - with v1 at library defaults and v2 grid-searched per
dataset. v1 wins livecell, omnipose, cellpose, vicar, tissuenet and yeaz, by 0.06 to 0.09 on omnipose,
cellpose and vicar. AIS v2 also loses to AIS v1 on those, so this tracks the **decoder**, not APG. APG is
not a reason to prefer v2 over v1, and it does not rescue the v2 decoder where v1 is simply better. That
decoder gap is the open question this round surfaced and it was deliberately not pursued.

### Four hypotheses this round refuted

- **'APG wins where AIS's seeding fails.'** Spearman between AIS mSA and APG's margin is **+0.046** over
  twelve datasets, no relationship. It fails at both ends: deepseas has the weakest AIS (0.184) and is the
  one loss, while dynamicnuclearnet has middling AIS (0.552) and the largest margin.
- **'AIS degrades from val to test more than APG.'** AIS drops more on 4 of 9 measured, mean absolute drop
  0.051 for APG against 0.056 for AIS. One dataset's observation, not a mechanism.
- **'`candidate_threshold` scales with object area.'** Grids reaching to 0.1 chose 1.5 on datasets spanning
  5.7x in object size.
- **'`box_extension` is set by crowding.'** dsb is separated nuclei and prefers 0 monotonically, while
  dynamicnuclearnet at half its object radius prefers 4.

Every mechanism inferred from four datasets died at nine or twelve; the plain measurements held. Treat any
explanation in this document that rests on fewer than about eight datasets as a hypothesis.

### What transferred

**`max_overlap` = 0.15 on 11 of 12** is the one axis that genuinely transfers, and it is the library
default. **`candidate_threshold` = 1.5** is the modal and median value but only on 6 of 12: two datasets
prefer 1.0 and cellpose and omnipose peak at 5.0, the top of the grid, so their optima are unmeasured.
Their curves are nearly flat there (0.005 and 0.013 across the whole axis), so the default stands, but it
rests on a plurality rather than a consensus.

**`min_candidate_size` = 1 on 9 of 12**, and sweeping it was not worth it: the two datasets that move gain
+0.0042 (dynamicnuclearnet, 4) and +0.0048 (vicar, 16, at the top of its range). Pinning it at 1 through
the whole round cost nothing.

`sigma`, `foreground_threshold`, `n_iter`, `score_threshold` and `min_size` all differ per dataset with no
pattern. Carrying the livecell configuration elsewhere costs -0.018 (deepbacs) and -0.022
(dynamicnuclearnet) on val, against -0.003 on livecell itself.

### box_extension is required per dataset and nothing predicts it

Selected values over the twelve: off on 4, 0 on 4, 2 on 2, 4 on 2. No correlation with object radius,
modality or AIS strength. On dynamicnuclearnet it is the entire result: extension 4 gives 0.6695 and
extension 0 gives 0.5032, below AIS's 0.5523, the gain being boundary quality (SA75 0.7711 against 0.5067)
at a small cost in detection (f1 0.9526 against 0.9619). Stop looking for a rule.

### Do not select on a small val split

The APG-AIS **sign flipped between val and test on two of the first four datasets**, and the val-to-test
drop averages 0.051 with a maximum of 0.135 (vicar). The grids are not flat: best minus worst mSA inside
one dataset's grid runs from 0.093 to 0.223, so which cell wins matters. The riskiest rows are u20s (val
25, grid spread 0.203, margin +0.002 and not significant) and yeaz (val 37, spread 0.122, margin +0.006).
Per-dataset re-tuning also *lost* on livecell, -0.003 against the eight-round configuration, while
winning on val.

Val sizes vary and are worth knowing when reading a row: livecell 570, cellpose 100, omnipose 309 (it ran
before the 100-image cap was fixed), tissuenet and vicar 101, dsb and dynamicnuclearnet 100, deepseas 73,
cellbindb 60, yeaz 37, deepbacs 30, u20s 25.

## The tuned configuration

```python
foreground_threshold = 0.5  candidate_threshold = 2.25  sigma = 0.5  min_candidate_size = 1
score_threshold = 0.5  max_overlap = 0.20  min_size = 50
refine_with_box_prompts = True  box_extension = 0
n_iter = 25, dt = 0.25
```
val 0.3877 -> test 0.4006. Val underestimates test by about 0.013; the val split is harder.

The previous configuration (`n_iter = 50`, `min_candidate_size = 4`, no box stage) gives val 0.3832 -> test
0.3949. Of the +0.0057 on test, +0.0042 is the box stage alone. The rest is `n_iter = 25` together with
`min_candidate_size = 1`, which help only in combination, because each scores worse on its own on both val
splits. Treat that part as marginal.

`refine_with_points = True` is measured at +0.0007 on val-570 and is not in the table yet, because it has
not been evaluated on test. `min_refined_size = 50`, which applies the merge's size filter to the box
stage's output as well, is +0.0004 on val-570 and is in the same position.

**The library defaults in `DEFAULT_PROMPT_GENERATION` are not this configuration.** `foreground_threshold`
defaults to 0.7 because it inherits `DEFAULT_POSTPROCESSING["sparse"]`, `candidate_threshold` is 2.0, and
every option added after round 2 is off. Pass the configuration above explicitly. The defaults stay
conservative until there is cross-dataset evidence, see open item 2.

## Code

The sweep scripts live outside the repository and are not version controlled, under
`/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/livecell_variant_scripts/`. They also
hardcode an absolute path back into this checkout.

| file | role |
|---|---|
| `micro_sam/v2/automatic_prompt_generation.py` | the implementation, non-tiled and tiled |
| `micro_sam/v2/instance_segmentation.py` | `get_instance_segmentation_generator(segmentation_mode='apg')` |
| `micro_sam/_cli.py` | `micro_sam segment --engine apg` |
| `finetuning/v2/evaluation/evaluate_automatic_baselines.py` | `--method micro_sam2_apg`, `--apg_params` |
| `<scripts>/tune_apg.py` | the grid search, with `ROUNDS` recording what was swept |
| `<scripts>/submit_apg.py` | sharded submission with a dependency-gated merge |
| `<scripts>/build_livecell_table.py` | regenerates `automatic_segmentation_livecell.csv` |
| `<scripts>/tune_apg_watershed.py` | the AIS+APG watershed sweep, with `submit_apg_watershed.py` |
| `<scripts>/probe_truncation.py` | how much of the result the merge's truncation touches |
| `<scripts>/probe_ais_union.py`, `probe_ais_proposals.py` | the two detection-side pairings |
| `finetuning/v2/evaluation/apg_crossdataset.py` | the nine-dataset rounds, with `submit_apg_crossdataset.py` |

`apg_crossdataset.py` is in the repository, unlike the livecell sweep scripts. Its stages are `tune`
(plain APG on val), `ais_tune` (the flow post-processing on the same val, so both engines are tuned),
`box` (`box_extension` with the two merge filters) and `test`. Data comes from
`scripts/apg_experiments/util.py`, the only place with val/test splits per dataset; livecell is the
exception and is read through the v2 centre crop so that its whole 570-image val split is available.

`AutomaticPromptGenerator` subclasses `UniSAM2InstanceSegmentation`, mirroring how v1's APG subclasses
`InstanceSegmentationWithDecoder`. `initialize` encodes the image once and runs the decoder on that
encoding, because the two halves of a joint checkpoint share their image encoder weights. `set_state`
therefore needs either 'image_embeddings' or 'image': without one of them `generate` prompts whatever image
the predictor still holds. The class reuses `_get_centers` (v1) for interior points, and resolves overlaps
with either `merge_by_score` (default) or `micro_sam.util.apply_nms`.

`TiledAutomaticPromptGenerator` inherits from both `AutomaticPromptGenerator` and
`TiledUniSAM2InstanceSegmentation`, so the decoder half is the tiled one and `generate` keeps its
parameter surface. `generate` calls three overridable stages, `_apply_and_merge`, `_residual_rounds` and
`_refine_boxes`, and the tiled class replaces the first and third with per-tile versions. **The prompts
are still derived once, from the stitched prediction**, so a candidate that straddles a tile border is
proposed once; each one is then assigned to the tile whose *inner* block holds its point and prompted
within that tile's halo, which is what keeps an object from being segmented twice while still letting its
mask reach past the border it sits near. Residual rounds are rejected there, because they compose by
claiming pixels across the whole image. With one tile covering the image the tiled path reproduces the
non-tiled one (62 instances either way, mSA 0.5097 against 0.5118, the difference being
`precompute_image_embeddings` against `set_image`).

Reproduce:
```bash
# grid search, see ROUNDS in tune_apg.py for what each round swept
python submit_apg.py -m hvit_b --tag valfull --split val --per_celltype 0 --n_shards 40 --round 1
# evaluate on test
python submit_apg.py -m hvit_b --tag testfull --split test --per_celltype 0 --n_shards 40 --round final_box
```
Both need cached decoder predictions first (`cache_variants.py -s val|test --per_celltype 0`).

## What the sweeps found

Eight rounds. Rounds 1-2 swept the original axes, 2700 then 24 combinations over 569 val images, and round 2
gained +0.0004, so round 1 had already found their optimum. Rounds 3-8 swept the prompt structure, the box
refinement, the false-positive filters and the mask-changing levers. Only the box stage and
`refine_with_points` survived.

- **`foreground_threshold` is inert for candidate proposal.** 0.3/0.5/0.7 differ by 0.0003 in round 1 and
  tie to four decimals in round 2. It does not define object boundaries here, only which pixels can be
  proposed from. It is still load-bearing for `derive_residual_prompts` and `foreground_agreement`.
- **`candidate_threshold` is the axis that matters.** Peak at 2.25, plateau 2.0-2.5, clearly worse at 1.75
  and 5.0. This is APG's analogue of AIS's `density_threshold` (tuned to 5.0), and it must be *lower*.
- **`sigma` stays 0.5**, the same as AIS. Less smoothing leaves more peaks, but the AIS tuning transfers.
- **`score_threshold` is a guard, not a lever.** 0.0 and 0.3 are identical, 0.5 helps slightly, 0.7 costs
  -0.037 and 0.85 collapses to 0.146. v1's APG has no score cutoff at all.
- **`max_overlap` is nearly flat** (0.15/0.20/0.30 within 0.0002).
- **`min_size = 50`** confirmed again, as in all three AIS sweeps.

## Do not retry these

- **Seven seeding mechanisms**, all worse than the plain global cutoff: v1-style magnitude threshold,
  magnitude h-maxima, density h-maxima, magnitude local maxima, local-relative-to-max, per-image percentile,
  cascaded re-split. Range -0.005 to -0.062. At threshold 1.5 two thirds of missed objects do have their own
  component, but that costs 1022 components for 239 objects. Real peaks and noise occupy the same density
  values, so **no thresholding rule separates them**.
- **Box prompts derived from candidate regions** (-0.05 versus points). A candidate region is a fragment, so
  its bounding box tells the model the object is fragment-sized. Boxes from *already-predicted masks* are
  the correct design and are implemented as `refine_with_boxes`. Use **zero** extension: 1 px costs -0.004
  and 2 px costs -0.011, because LiveCELL is confluent and a grown box captures the neighbouring cell.
- **`apply_nms` for the sweep.** It is quadratic over full masks: 0.8-1.1 s per call at about 350
  candidates, against 0.13 s for `merge_by_score`, which also scored marginally *better*.
- **Negative points from neighbouring candidates**: -0.051 at 2 points per candidate and -0.061 at 4. It
  fails through **recall** (0.730 -> 0.657), not precision. A negative point makes SAM2 shrink the mask
  rather than sharpen the boundary against the neighbour. Many candidates also share an object, and a
  `negative_min_distance` of 15 px recovers only 0.008 of the loss.
- **Residual re-seeding** from the foreground the previous round left unclaimed (`derive_residual_prompts`):
  -0.0016. Recall rises by 0.0079 and precision falls by the same. Above a residual score cutoff of 0.85 no
  candidate survives and the output is bit-identical to no re-seeding. **The objects the density misses are
  objects the interactive branch also segments poorly.**
- **Keeping every multimask proposal**: +0.0006 without the box stage and nothing with it. The idea that the
  predicted IoU prefers the cluster is directionally right (recall +0.0026, precision -0.0029) but too small
  to be worth the tripled record count.
- **`min_candidate_size` above 4.** A cliff, not a slope: 4 -> 0.3962, 10 -> 0.3594, 25 -> 0.2161.
- **Retuning the flow travel.** `n_iter = 25` is -0.0028 on its own on the full val split, and helps only
  together with `min_candidate_size = 1`. The AIS-inherited `n_iter = 50` was essentially right.
- **Three false-positive filters.** `min_box_agreement`, the IoU between the point-prompted and the
  box-refined mask, is **inert at every value**: the two masks always agree. `max_size` only ever loses, and
  it loses **recall** (0.7363 -> 0.6854 at 2000) while precision stays flat, so **the large masks are real
  large cells, not clusters**. `min_foreground_agreement = 0.5` gives +0.0003: real precision
  (0.8403 -> 0.8456) paid for in recall (0.7363 -> 0.7328).
- **Eight-fold test-time augmentation of the density.** Worth **+0.034 to the flow pipeline** but only
  **+0.001 to APG** (0.3981 against 0.3970). The two are substitutes: APG exists to recover what a noisy
  density misses, so cleaning the density removes the deficiency APG compensates for. Not worth 8x decoder
  inference on top of APG's 6x. It also refutes the reason for trying it, because `candidate_threshold`
  stays at 2.25 on the cleaner density and 3.0 is already worse.
- **Four of the five mask-changing levers.** `mask_threshold` is monotonically worse on the full split:
  0.3877 at the default 0.0, then 0.3871, 0.3852 and 0.3815 at 1.0, 2.0 and 3.0. Multimasking in the box
  stage costs -0.022, because a box already pins the extent. Clipping the masks to the decoder foreground
  costs -0.002, so **the masks already sit inside it**. A mask-prompt refinement round is -0.000. Hole
  filling does nothing, because SAM2 masks have no enclosed holes: the code only changes the result once the
  area bound is above 100000 pixels. Only `refine_with_points` survives, at +0.0007.
- **Trusting a 48-image probe.** `mask_threshold = 1.0` measured +0.0014 there and -0.0006 on 570 images,
  the largest lever in that probe reversing sign. Screen on 48 images if you must, settle on the full split.
- **Re-cutting the boundaries with the flow watershed** (`refine_with_watershed`), and the two other ways
  of pairing the engines. See the section above: -0.009 to -0.044, and the mechanism it fixes is 0.9% of
  the segmented pixels.

**The pattern is the useful result.** Negative points, residual re-seeding, keeping every proposal,
foreground agreement and `max_size` all trade precision against recall at about 1:1 and net zero. The
candidate set and the merge are already on the Pareto front of what this model's masks support, so moving
along that curve is finished. The two things that worked, box refinement and `refine_with_points`, both
**change what a prompt is** rather than which masks are kept. Spend the next effort on what the masks are.

## Pairing AIS with APG: measured, and it does not work

Three ways to combine the two engines were implemented and measured. All three fail, each for its own
reason, and the reasons are worth more than the attempts.

### 1. AIS boundaries under APG detection: the mechanism is 0.9% of the pixels

The idea was that the greedy merge truncates a mask to whatever pixels a better-scoring mask left, so the
contact line where two cells touch is arbitrary, and the flow watershed would draw it properly.
`refine_with_watershed` does that: the APG instances are the markers, `watershed_heightmap` is the height,
the decoder foreground is the mask.

**The premise is quantitatively wrong.** The merge truncates 37% of the instances, but only **0.9% of the
segmented pixels**, and the oracle that removes truncation entirely (every surviving mask painted whole,
most confident last) scores **0.3581 against the merge's 0.3581** on 24 val images. There is nothing there
to win.

The sweep agrees, on the full 569-image val split against the tuned APG's 0.3877:

| markers | erosion | growth | mSA |
|---|---|---|---|
| box stage | 0 | none | 0.3881 |
| box stage | 1 | none | 0.3787 |
| box stage | 2 | none | 0.3733 |
| box stage | 4 | none | 0.3687 |
| box stage | 0 | fg > 0.5 | 0.3629 |
| box stage | 0 | fg > 0.3 | 0.3442 |

Monotone in both axes. `erosion = 0` with no growth is a **no-op** by construction, so its +0.0004 is not
the watershed at all: it is the `min_size = 50` filter running after the box stage, which is
`min_refined_size` and is worth +0.0004 on its own. Markers from the merge instead of the box stage are
0.002-0.004 worse throughout. Growing into unclaimed foreground repeats what `clip_to_foreground` already
showed, in the other direction and much larger.

### 2. AIS as a source of detections: adding correct objects still loses mSA

AIS matches **4.75% of the ground-truth objects that APG misses**, about 7.5 per image, so the recall
headroom is real. Adding them is not:

| variant | mSA | SA50 |
|---|---|---|
| APG | 0.3642 | 0.6490 |
| oracle union, unclaimed pixels only | 0.3603 | 0.6453 |
| **oracle union, added masks painted whole** | **0.3763** | **0.6926** |
| coverage < 0.1 | 0.3581 | 0.6396 |
| coverage < 0.25 | 0.3566 | 0.6376 |
| coverage < 0.5 | 0.3546 | 0.6342 |

Two things fall out. Painting an added mask into the unclaimed pixels only, the way the merge treats every
later mask, **destroys the gain**: -0.004 instead of +0.012, because the missed object's pixels have
already been claimed by an over-extended neighbour and taking them back is most of the benefit. And the
ceiling with the whole mask is worth having, **+0.012 mSA and +0.044 SA50**, so the direction is not
hopeless in principle. What fails is the selection: picking the flow instances by how little the APG result
covers them loses 0.006, and painting those whole changes nothing (0.3581 either way), so the added
instances are the wrong ones rather than the wrong shape.

### 3. Letting the model judge the AIS proposals: the score does not discriminate

APG's own answer to a selection problem is to prompt the model and keep what it scores highly, and a flow
instance offers a **box**, which is the stronger prompt and is not the fragment-shaped box that already
failed. Re-prompting every poorly-covered flow instance with its bounding box, on 48 val images:

- **20% of the proposals are real** objects APG missed, 6.5 proposals per image.
- The predicted IoU is **0.790 on the real ones and 0.767 on the junk**. A gap of 0.023 is not a filter.
- Every cutoff either keeps the junk or keeps nothing: 0.7 gives 0.3578, 0.85 keeps 0.06 proposals per
  image and lands back on the baseline, 0.9 keeps none.
- The oracle **on the box-refined masks** is 0.3660, against 0.3763 on the raw flow masks. So the
  refinement itself throws away five sixths of the headroom.

This is the same wall as residual re-seeding, from the other side: **the objects APG misses are objects the
interactive branch also segments poorly, however it is prompted about them.**

### What this leaves

- **Boundary work on livecell is finished.** Not because boundaries are perfect, but because the merge's
  arbitrary part is 0.9% of the pixels and the rest of the boundary is SAM2's mask, which the flow field
  does not improve on.
- **The recall headroom needs a detector that is right about being right.** +0.012 mSA sits behind a
  selection rule, and neither coverage nor the model's own score is one.
- **Do not split large masks** still holds: `max_size` showed the big masks are real large cells, so a
  disagreement between the two engines is not evidence of a cluster.
- The scripts are `tune_apg_watershed.py` (with `submit_apg_watershed.py`), `probe_truncation.py`,
  `probe_ais_union.py` and `probe_ais_proposals.py`, alongside the others.

## Open items

1. **3D is unsupported.** `run_microsam2_apg_evaluation` warns and skips 3d datasets, and `initialize`
   raises for `ndim != 2`.
2. **Only two defaults have cross-dataset evidence.** `candidate_threshold` = 1.5 and `max_overlap` = 0.15
   are set from the four-dataset sweep. The rest of `DEFAULT_PROMPT_GENERATION` is still conservative and
   differs from any tuned configuration, which is correct while the remaining axes vary per dataset.
   `tissuenet` and `neurips_cellseg` are the obvious next datasets; neurips is not in
   `scripts/apg_experiments/util.py::get_image_label_paths` and needs its train split downloaded for a val
   split.
3. **`results_automatic.csv` and the published v2 baselines are stale.** They were produced before
   `GT_MIN_SIZE_2D` existed, so they carry the crop-sliver artefact, worth about 0.01 on livecell. The whole
   matrix needs re-running for those numbers to be comparable with anything measured now.
4. **The nms merge path was removed**, along with `resolution`, `nms_threshold` and
   `intersection_over_min`. Greedy beat nms and is 6-8x faster, but that comparison was 24 images and
   predates the box stage, so it is thin. Restoring the option is a few lines if a dataset ever wants it.
5. **The annotator does not offer APG.** The CLI and the Python API do, since `--engine apg` and
   `get_instance_segmentation_generator(segmentation_mode='apg')` were added, but
   `sam_annotator/_widgets.py` still builds its segmenter without an engine choice.
6. **The sweep recomputes everything per `generate` call.** Only 5 parameters affect the prompts; the rest
   are post-filters over the same records. `tune_apg.py` works around this by calling `_apply_prompts`
   directly. Caching the records inside the class would let the sweep use the public API.

## Gotchas that cost time

- **Slurm cannot see `/local`.** Scripts, logs and outputs must live on `/mnt/vast-nhr`. 24 jobs failed in
  24 s with empty logs because of this.
- **The prediction cache is keyed by name, not index.** Filenames embed the index the image had when
  written, so excluding an image shifts every later index; an index-based lookup silently missed 1114 of
  1510 images. `variant_common.cache_index` handles this, and `tune_apg.py` fails loudly if the cache does
  not cover the split.
- **A `None` in a sweep grid becomes a NaN in the merged CSV**, and pandas `groupby` drops those rows
  silently. Sweep an unbounded value as a large integer instead, see `NO_MAX_SIZE`.
- **Memory.** Holding every prompt round's masks at once OOM'd 3 of 60 shards at 16G. The sweep now holds
  one round at a time. One OOM blocks a `--dependency=afterok` merge permanently, which surfaces as
  `DependencyNeverSatisfied`.
- **Slurm jobs import the library from the working tree.** Editing `automatic_prompt_generation.py` while
  shards are queued mixes code versions across one merge.
- **Timing estimates from a single image do not survive 60-way sharding.** Contention roughly tripled the
  per-image cost. Measure under load.
