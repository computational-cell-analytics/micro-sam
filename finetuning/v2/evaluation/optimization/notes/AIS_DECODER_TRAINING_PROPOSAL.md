# Improving the UniSAM2 decoder for automatic instance segmentation (AIS)

What the 2026-09 AIS post-processing campaign on the joint/v4 geodesic `hvit_t` model found about the
decoder's output, and how the decoder's training should change to remove the two losses that
post-processing cannot reach. Evidence and numbers: `finetuning/v2/evaluation/optimization/notes/AIS_V4_OPTIMIZATION.md`.

## What the decoder predicts today

The automatic branch regresses four channels: a foreground probability (Dice loss against the binary mask)
and three directed-distance channels (MSE, masked to the foreground) whose target is the *geodesic hybrid*
field (`micro_sam/v2/transforms/labels.py`, `GeodesicHybridDistanceTransform`): direction = gradient of the
geodesic distance from the object's centre, magnitude = per-object normalised distance to the object's own
boundary. Post-processing follows the negated field to sinks (seeds), then floods a height map
`0.5 (1 - fg) + 0.5 (1 - |d|)` inside `fg > 0.5`.

Measured properties of the prediction (cached predictions of the 2026-09 development corpus):

- The field is smooth where the target is discontinuous. At a contact line between two touching cells the
  target's direction flips; the prediction turns over a 6-8 px band (cosine between the flow 1 px on either
  side of a contact pixel: +0.90, inside an object +0.97; at ±3 px: ≈0 vs +0.70). The magnitude dips to 0.17
  (median) at contacts against 0.34 inside, instead of to zero, and with gaps along the line.
- At the sink (object centre) the magnitude is 0.04-0.2, not the target's 1: the network smears the
  single-pixel zero of the source over the whole centre region.
- In the background the network emits the label transform's fill value (|d| ≈ 1.0-1.1) even though the
  distance loss is masked there; the halo right around an object, however, carries a smooth continuation
  of the object's field (only 0-13 % of halo pixels exceed 0.8).
- The thresholded foreground has a dataset-dependent bias: area ratio to the ground truth 1.13 (livecell),
  1.17 (tissuenet, yet 39 % of its object pixels fall below 0.5), 1.48 (dynamicnuclearnet), 3.5 (deepbacs,
  thin rods). The matched-object IoU is 0.67-0.84, which caps mSA at the higher IoU thresholds.

## Point 1: contact information (merges)

Merges are the dominant loss on touching-cell data: livecell 25 % of the objects are seeded but end in
an instance that also covers a neighbour, and another 11 % are unseeded and absorbed; tissuenet 20 % + 5 %;
neurips_cellseg 17 % + 10 %. Oracles with the predicted seeds and foreground: a ground-truth ridge doubles
livecell (0.27 → 0.54) and deepbacs, +67 % on tissuenet. Every label-free rule tried on the predicted field
failed to separate "two seeds in one cell" from "two touching cells" (edge/interior magnitude ratio 0.58 vs
0.35 with heavy overlap). The information is not in the prediction.

Proposed changes, in the order I would try them:

1. **An explicit contact channel.** Add a fifth output channel trained on the *touching boundary* mask:
   pixels of an object adjacent to another object (`find_boundaries(labels, mode="inner")` restricted to
   pixels whose neighbourhood contains a second non-zero label, dilated by one pixel so the target is 2-3 px
   wide and learnable). Loss: Dice or focal BCE (the class is rare). Post-processing then adds the channel
   as a ridge term to the height map, or excludes it from the watershed mask and reassigns it afterwards
   (the EM training already does the analogous thing implicitly: `expected_fg = fg & ~boundary`). This is
   the cheapest change with the largest expected return, since the ridge oracle shows the ceiling.
2. **Boundary-excluded foreground for LM, as in the EM recipe.** Train the LM foreground target as
   `fg & ~find_boundaries(labels, mode="inner")` (a one-pixel gap between touching objects, and a
   one-pixel shrink at every boundary). The gap gives the watershed mask a separation it currently lacks
   and the shrink counters the over-prediction of point 4 (see below). Risk: the shrink changes the
   foreground calibration for every dataset by one pixel, which is a lot for 50-pixel objects; it has to be
   paired with a dilation-by-one of every instance after the watershed, which the EM pipeline does not do
   either. Cheaper than 1 (no new channel), less targeted.
3. **Sharpen the field at contacts through the loss.** The masked MSE weights every foreground pixel
   equally; contact pixels are <2 % of them and the network averages the two objects' fields there. Weight
   the distance loss by proximity to a contact (e.g. 5× within 3 px of a touching boundary), or add a
   direction term (`1 - cos` between predicted and target unit vectors, weighted the same way) so that the
   flip is penalised as a direction error and not only through the small magnitude residual. Expect a
   sharper flip, not a sharp one: an L2 regressor will still average within its receptive field.
4. **Instance-affinity output for the merge decision.** A short-offset affinity channel (is the pixel 2 px
   away the same instance?) decided per pixel is what the merge rule needed and could not compute from the
   field. This is the most invasive option (a new head, a new loss, and the post-processing becomes a
   mutex/affinity watershed, which `bioimage_cpp.segmentation.mutex_watershed` provides) and would replace
   the seeded watershed rather than fix it.

What would show that it worked: the merged + absorbed fraction on livecell / tissuenet in the D2
decomposition of the benchmark (`benchmark_ais_optimization.py run`, columns `seeded_merged`,
`unseeded_absorbed`) falls from 36 % / 25 % towards the ridge oracle's level, and the contact-line
cosine at ±1 px turns negative.

## Point 4: instance extent (foreground calibration)

The foreground threshold is a compromise whose sign differs by dataset (tissuenet under-covers, deepbacs
over-covers by 3.5×), so no global rule generalizes, and the halo carries a smooth field, so the magnitude
cannot trim it. The extent is also where APG's advantage over AIS comes from: the same seeds with SAM2
masks score 20 % higher in 2D and 80 % in 3D.

Proposed changes:

1. **Calibrate the foreground target to the boundary, not to the mask.** The Dice loss on the binary mask
   rewards a soft, wide foreground (a boundary pixel predicted at 0.6 costs almost nothing). Options:
   a per-pixel loss with boundary weighting (BCE with weights rising towards the boundary, or a boundary
   Dice term on the ring of ±2 px), or a signed-distance regression for the object extent (predict the
   signed distance to the object boundary, positive inside; the extent is the zero level set, which is
   sub-pixel and calibrated by construction). The signed distance is the natural companion of the geodesic
   channels and can replace the foreground channel entirely.
2. **Resolve the label-convention conflict explicitly.** The over-prediction on deepbacs (thin rods
   annotated tighter than the visible cell) and the under-prediction on tissuenet are label conventions the
   network averages over. Two remedies: (a) a per-dataset boundary offset during training (dilate or erode
   the masks of the datasets whose annotation is systematically tight or loose, measured once against the
   raw intensity edge), so that the network learns one convention; (b) a small conditioning input (the
   dataset's convention as an offset in pixels) which is not available at inference for new data and so is
   the weaker option. (a) is a data-preparation change and costs nothing at inference.
3. **Train the extent at the object's own scale.** The 1024-px resize of the encoder puts a 20-px nucleus
   and a 200-px cell through the same boundary blur. Multi-scale sampling of the training patches (the
   generalist loader already has patch shapes; add a scale augmentation targeting 30-80 px objects) or
   an auxiliary loss on the boundary IoU at the native resolution would sharpen the small objects, where a
   one-pixel error is 10-20 % of the IoU.

What would show that it worked: the area ratio of `fg > 0.5` to the ground truth moves towards 1 on every
dataset at the same threshold, the matched-object IoU (`matched_iou` column) rises from 0.67-0.84, and
the per-dataset optimum of `foreground_threshold` in the sweep collapses to one value.

## Order and cost

Point 1.1 (contact channel) and 4.1 (boundary-calibrated foreground or signed distance) are one training
run each on the existing joint recipe (`finetuning/v2/generalist/train_joint.py`, `distance_type`), with a
new label transform in `micro_sam/v2/transforms/labels.py` and a loss term in
`micro_sam/v2/loss/directed_distance_based.py`. The benchmark and its caches evaluate a new checkpoint end
to end in under an hour (predict once, then the diagnostics), and the oracles give the ceiling for each
change before any post-processing is retuned. Everything the post-processing side can still do without a
better field is listed at the end of `AIS_V4_OPTIMIZATION.md`.
