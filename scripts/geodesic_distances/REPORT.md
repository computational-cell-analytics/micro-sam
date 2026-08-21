# Geodesic distance targets for the AIS v2 automatic branch

Status: oracle evidence collected, implementation landed, **training A/B not yet run**.

## 1. The question

The automatic branch regresses a directed *euclidean* distance field
(`micro_sam.v2.transforms.labels.DirectedPerObjectBoundaryDistanceTransform`, built on
`bioimage_cpp.distance.vector_difference_transform`). Would a *geodesic* field, which constrains
paths to stay inside the object, do better?

## 2. What the field is actually used for

This is the key to everything below. `micro_sam/v2/postprocessing.py` uses the field for **two**
separate jobs:

1. **As a flow.** `flow_instance_segmentation` advects foreground pixels along `-field`, and the
   convergence density seeds a watershed. What matters here is the field's **direction**.
2. **As a ridge.** `watershed_heightmap` takes `norm(field)`, inverts it, and uses it as the wall
   between touching objects. What matters here is the field's **magnitude**.

A representation has to serve both. That is why the obvious geodesic swap fails.

## 3. The four variants

Every variant is `direction x magnitude`, built from at most two geodesic solves per object
(sources = the object's boundary voxels; source = the object's center, argmax of the EDT).

| variant | direction | magnitude |
|---|---|---|
| `euclidean` (current) | toward nearest boundary | distance to boundary |
| `geodesic_boundary` | toward nearest boundary (geodesic) | geodesic distance to boundary |
| `geodesic_center` | away from object center (geodesic) | **1 everywhere** |
| **`geodesic_hybrid`** | away from object center (geodesic) | geodesic distance to boundary |

`hybrid` and `boundary` differ by exactly one factor. Why direction matters: for an elongated
object the nearest boundary is *sideways*, so a boundary-referenced flow collapses pixels onto a
**medial axis** (a line, hence many seeds, hence shattering), while a center-referenced flow
collapses them onto a **point** (one seed). Why magnitude matters: a unit-norm field gives
`watershed_heightmap` no ridge at all, so touching objects merge.

## 4. Oracle results

Method: build the fields from **ground truth** labels, run the AIS v2 post-processing on them,
score against the same ground truth. This bounds what a perfectly trained model could reach with
each representation. `n_iter`, `density_threshold` and `sigma` were swept per variant and the best
setting chosen per (dataset, variant) — never per image. Sweeping all three is essential: the
pipeline defaults are tuned to the euclidean field's magnitude profile, and density smoothing is
the baseline's main fix for its over-segmentation.

Sparse / `flow_instance_segmentation`, **mSA** (higher better):

| dataset | euclidean | geo boundary | geo center | **geo hybrid** |
|---|---|---|---|---|
| DSB (10 img, 2D) | 0.9696 | 0.9671 | 0.8764 | **0.9848** |
| LIVECell (10 img, 2D) | 0.6169 | 0.5796 | 0.2068 | **0.9130** |
| GoNuclear (3 vol, 3D, 48x256x256) | 0.9401 | 0.9449 | 0.7810 | **0.9970** |

LIVECell by cell type (best mSA): A172 0.791 -> **0.984**, SHSY5Y 0.501 -> **0.876**.

Dense / `run_multicut`, 3D fields:

| dataset | metric | euclidean | geo boundary | geo center | geo hybrid |
|---|---|---|---|---|---|
| SNEMI (1 crop 25x512x512) | mSA | 0.0955 | 0.2524 | 0.1699 | **0.2578** |
| SNEMI | CREMI score | 0.2090 | **0.1889** | 0.2027 | 0.2280 |
| CREMI A (1 crop 25x512x512) | mSA | 0.1103 | 0.1059 | **0.1182** | 0.1045 |
| CREMI A | CREMI score | 0.3092 | **0.3034** | 0.4197 | 0.3695 |

CREMI used `--foreground_sigma 1.0`: a binary ground truth boundary map leaves the multicut
watershed on flat plateaus, which has nothing to do with the distance representation.

### Reading it

- **Hybrid wins every sparse dataset.** The margin tracks how far an object's medial axis is from
  being a point: round nuclei have little headroom (DSB +0.015, GoNuclear +0.057), elongated
  LIVECell cells have a lot (+0.30).
- **`geodesic_boundary` is a dead end** — within noise of euclidean everywhere, at ~2x the cost.
  Its magnitude is already inside the hybrid (`|hybrid| == boundary distance` before the
  per-channel normalization).
- **`geodesic_center` alone fails** — no ridge, so touching objects merge (LIVECell: 63 predicted
  objects for 107 ground truth).
- **EM is unresolved.** SNEMI favours geodesic, CREMI is a wash. But every variant scores mSA
  0.10-0.26 there, i.e. the dense path is dominated by the boundary map and the multicut, not by
  the distance representation. One crop each. Not enough to act on.

### A bug that invalidated an earlier version of this table

The first implementation padded the mask when choosing the geodesic boundary sources, which made
the crop's z faces count as object boundary. For volume-spanning objects that zeroed the boundary
*distance* across the entire first and last slice, so `geodesic_boundary` and `geodesic_hybrid`
(both of which use that distance as their magnitude) produced an empty field there: no flow, no
seeds, blank slices, and a severed multicut z-linkage. It cost those two variants roughly 1.1
CREMI score on SNEMI (1.32 -> 0.19 after the fix). `euclidean` and `geodesic_center` were never
affected and came back bit-identical, which is what confirmed the diagnosis. All numbers above are
post-fix.

A slice-wise solving mode was added while chasing that bug and is retained in the exploration
script (`--slicewise`), but its apparent benefit was largely the bug. It is **not** used by the
shipped code. The one real effect it showed is that `euclidean` itself gains on SNEMI from
slice-wise solving (mSA 0.096 -> 0.139), which is worth checking independently of geodesic.

## 5. Tile consistency

The hybrid's center reference is global, so a tile that cuts an object relocates its center. Test:
recompute the field from many differently offset crops that all contain a fixed region, compare
against the full-image field. Median angular deviation, LIVECell tile 256:

| variant | <50% inside | 50-75% | 75-90% | 90-99% | whole |
|---|---|---|---|---|---|
| euclidean | 2.46 deg | 0.69 | 0.74 | 0.06 | 0.06 |
| geodesic hybrid | 63.5 deg | 19.7 | **1.18** | 0.06 | 0.06 |

**The target is stable once >=75% of the object is inside the crop** (1.18 deg, 0% of pixels beyond
30 deg). So this is not a learnability problem — the target is also a deterministic function of the
patch. It is a **tiled inference** constraint: the tile plus halo must contain ~75% of any object
whose pixels are kept, i.e. a halo of roughly half the largest object diameter. Worth checking
against what `batched_tiled_inference` currently uses before deploying the hybrid on tiled data.

## 6. What was implemented

- `micro_sam/v2/transforms/labels.py`: `GeodesicHybridDistanceTransform`, a subclass of
  `DirectedPerObjectBoundaryDistanceTransform` that overrides only
  `compute_normalized_directed_distances`. Subclassing (rather than reimplementing) guarantees the
  relabeling, `distance_fill_value`, channel order and 2D promotion are identical, so the two
  targets differ in exactly one thing.
- `micro_sam/v2/postprocessing.py`: a `"sparse_hybrid"` entry in `DEFAULT_POSTPROCESSING`
  (`n_iter=200, density_threshold=2.0, sigma=2.0`, rest as `"sparse"`). Chosen as the setting with
  the best *worst-case* mSA across DSB, LIVECell and GoNuclear. The `"sparse"` defaults
  (`n_iter=50, density_threshold=10.0, sigma=0.5`) are tuned for the euclidean magnitude profile
  and cost the hybrid ~0.18 mSA on LIVECell.
- No other production code changed. The euclidean transform and the `"sparse"` preset are untouched.

`common.py` builds its hybrid variant by calling the shipped transform, so the oracle sweep
exercises production code. Re-running the sweeps through it reproduces the oracle numbers: **LIVECell mSA 0.9119** at exactly
the `sparse_hybrid` preset, and **DSB 0.9848**. The standalone prototype gave 0.9130 and 0.9848; the
small LIVECell difference is the promoted-3D solve.

That check earned its keep: the first version of the subclass scored 0.8075. Inheriting the parent's
2d-to-3d promotion meant `np.pad(mask, 1)` also padded the singleton z axis, putting background one
voxel from every voxel, flattening the distance field used to pick the object center and making the
argmax arbitrary. `_geodesic_object_center` now leaves singleton axes unpadded. Anyone porting this
to another framing should re-run that sweep rather than trust the field looks reasonable.

Cost: the hybrid runs two geodesic solves per object. Measured in the LIVECell dataloader,
**0.052 -> 0.102 s per batch of 2** with 8 workers (~2x). Should stay hidden behind an `hvit_t` GPU
step; raise `--n_workers` if the GPU idles.

## 7. The A/B test to run

`scripts/geodesic_distances/train_geodesic_livecell.py`. The two arms differ only in
`label_transform2`.

```bash
python train_geodesic_livecell.py dry_run  --target hybrid --input_path /data --download

python train_geodesic_livecell.py train --target euclidean --input_path /data --save_root /runs
python train_geodesic_livecell.py train --target hybrid    --input_path /data --save_root /runs

python train_geodesic_livecell.py evaluate --target euclidean --input_path /data --save_root /runs \
    --result_path /runs/eval_euclidean.json
python train_geodesic_livecell.py evaluate --target hybrid    --input_path /data --save_root /runs \
    --result_path /runs/eval_hybrid.json
```

Defaults: `hvit_t`, SHSY5Y only (largest oracle gap), 512^2 patches, batch 2, 25k iterations,
lr 1e-4. Evaluation sweeps the same grid the oracle used, so the trained numbers are directly
comparable to the 0.617 / 0.913 ceilings, and also reports the score at each target's own preset.

**Success criterion**: the hybrid arm beats the euclidean arm by a margin that survives the
per-image variance. The oracle gap is 0.30 mSA; anything above ~0.05 is meaningful, anything at or
below zero kills the idea.

`lr=1e-4` and `n_iterations=25000` are reasonable defaults, not tuned values (`train_automatic`'s
own default lr is 1e-5). Use the same for both arms — the comparison is the point.

**Not yet done: a single optimizer step.** Shapes, dataloader throughput and plumbing are validated
on CPU; training was never run (no GPU here, and CPU training is blocked by the mixed-precision
issue). Treat the first few hundred iterations as a smoke test.

## 8. Open questions

1. Does the oracle gap survive training? (the A/B above)
2. Does the tiled path need a larger halo for the hybrid, and what does that cost?
3. EM: does the hybrid help on more than one SNEMI crop, and does CREMI's anisotropy really explain
   the difference? Both EM numbers are single crops.
4. Does `euclidean` really gain from slice-wise solving on dense EM (0.096 -> 0.139 on SNEMI)? That
   would be a free improvement to the current pipeline, independent of geodesic.

## 9. Files

| file | role |
|---|---|
| `common.py` | field variants, dataset loading, sweep grids. The hybrid comes from the shipped transform, so the oracle numbers validate production code. |
| `evaluate_geodesic_distances.py` | oracle AIS v2 + mSA / CREMI score, with the parameter sweep |
| `measure_tile_consistency.py` | the section 5 measurement, parallel over crops |
| `visualize_geodesic_distances.py` | napari view of all fields side by side |
| `train_geodesic_livecell.py` | the A/B training and evaluation |

Data used: LIVECell (val), DSB (reduced, test), GoNuclear, CREMI samples A/B/C, SNEMI train.
