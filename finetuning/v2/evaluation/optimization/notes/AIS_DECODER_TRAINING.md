# AIS decoder training campaign: contact channel and boundary-calibrated foreground

Decision log of the campaign that trains the changes proposed in `AIS_DECODER_TRAINING_PROPOSAL.md`
(points 1.1 and 4.1) and compares them on the AIS benchmarks. Branch `ais-train-optim`; paths relative to
`finetuning/v2/evaluation/` unless they start with `micro_sam/` or `finetuning/`; `<root>` =
`/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization`.

## 1. Question and design (2026-09-07)

Two losses of the v4 geodesic `hvit_t` decoder cannot be recovered by post-processing (`AIS_V4_OPTIMIZATION.md`,
sections D2/D3): merges of touching cells (livecell 36 %, tissuenet 25 %, neurips 27 % of the objects) and an
over-wide foreground (`fg > 0.5` covers 1.1-3.5x the ground-truth area). Four decoders are trained under
identical conditions and compared with the library's current post-processing defaults (no re-tuning):

| variant | target / loss change | output channels |
|---|---|---|
| `baseline` | none (foreground Dice, three masked-MSE distance channels) | 4 |
| `contact` | fifth channel = touching boundaries (`touching_boundaries`, radius 1, dilation 1), Dice + BCE | 5 |
| `fgcal` | foreground loss = Dice + BCE weighted 5x within +-2 px of every object boundary (`boundary_weight=4`) | 4 |
| `both` | both changes | 5 |

Decisions (with the user): decoder-only training with the image encoder frozen at the v4 joint weights (the
interactive half is untouched; a single GPU suffices); warm start from the v4 decoder (the baseline is then
"v4 decoder + 12 h fine-tune on the tuning data"); the BCE variant of point 4.1, not the signed distance;
training data = train splits of the AIS tuning datasets; the `both` job is submitted after the other three.

Library changes are epoch A5 (commit `d85bccb`, implementation checksum `856a433c4b33348e1d85c4c13278f057`,
previous A4 `184eba917bd0cff28b5719b5584f6967`): `micro_sam/v2/transforms/labels.py` (`touching_boundaries`,
`contact=True`), `micro_sam/v2/loss/directed_distance_based.py` (`contact`, `boundary_weight`),
`micro_sam/v2/models/util.py` (sigmoid on channels >= 4), five-channel plumbing in `instance_segmentation.py`
and `batched_inference.py`, `flow_instance_segmentation(contact=, contact_weight=, contact_mask_threshold=)`
(opt-in; the default path is bit-identical, the three-channel `out[1:]` convenience stays, any other channel
count raises), APG and evaluation mirrors read `[1:4]`, the harness gains `fg_area_ratio` and the configs
`configs/ais_contact_ridge.json` (`contact_weight` 1.0) and `configs/ais_contact_mask.json`
(`contact_mask_threshold` 0.5).

Epoch A5 bit-identity (2026-09-07 02:55): the production decoder's `current-defaults` runs under A5 (jobs
15769317 / 15769318, `--ndim 2` for the images) reproduce the A4 runs sample by sample on v5 primary (240),
training_extra (157), holdout (233) and the apg3d primary (57) and holdout (18) crops (mSA, matched, predicted
objects, merged / absorbed counts, fg and matched IoU all identical; balanced 0.2457 / 0.4253 / 0.2437). These
A5 run directories are the production reference for the decoder comparison (`fg_area_ratio` included).

## 2. Data

Train splits of nine of the eleven tuning datasets, built from torch_em path lists
(`finetuning/v2/generalist/ais_decoder/ais_decoder_lib.py::build_datasets`); the trainer's validation set is
the last 5 % (at least 2 files) of every sorted train list, so the evaluation manifests (val splits) and the
test splits stay untouched. The file lists of every run are written to
`<save-root>/checkpoints/ais_decoder_<variant>/data_manifest.json`.

| dataset | train files (val tail) | samples per epoch | raw handling |
|---|---|---|---|
| livecell (8 cell types) | 3253 minus 30 images that also appear in the val split | 8 x 25 | grayscale, `MinInstanceSampler(6)` |
| tissuenet | 2580 | 200 | `raw/rgb` per-channel percentile normalisation, `labels/cell` |
| dynamicnuclearnet | 4950 | 200 | grayscale |
| deepbacs (mixed) | 125 | 120 | `_to_8bit` |
| dic_hepg2 | 302 | 120 | rgb png, channels kept distinct |
| neurips_cellseg (Training-labeled) | 1000 | 150 | mixed formats, to rgb |
| yeaz bf / phc 2d / phc stacks | 207 / 14 / 14 | 80 / 20 / 20 | stacks read as (1, 512, 512) patches |
| puma nuclei | 138 | 100 | rgb h5 |
| tnbc | 34 | 60 | rgb h5 (channel-first) |

Excluded: **deepseas** (binary masks; connected components merge touching cells, which would teach "no
contact" exactly at contacts and give merged blobs one geodesic centre; its 40 manifest crops are also
train-split files) and **covid_if** (49 files without a split, 5 tuning crops, 44 production-scored). Both stay
evaluation datasets and are unseen for all four models. zarr/h5/stack datasets are wrapped in
`RandomSubsetDataset` because torch_em splits `n_samples` uniformly over the files (200 samples over 2451
tissuenet files would only ever read the first 200 files).

## 3. Training set-up

`finetuning/v2/generalist/ais_decoder/train_ais_decoder.py`: `FrozenEncoderUniSAM2` (32 features wide, the
encoder in eval mode with `requires_grad=False`, so autograd stores no encoder activations), warm start from
the v4 `unetr_state` (a five-channel decoder keeps the four pretrained output rows and a fresh fifth row),
AdamW over the decoder parameters (lr 5e-5, weight decay 0.1), `ReduceLROnPlateau(0.9, patience 10)`,
bf16 autocast, `UniSAM2Trainer` (loss = metric = `DirectedDistanceLoss` of the variant), patch (512, 512),
percentile augmentation as in the generalist recipe. Checkpoints `<root>/ais_decoder_training/checkpoints/
ais_decoder_<variant>/{best,latest}.pt`; staging (`stage_ais_decoder_checkpoint.py`) writes the lean joint-format
file `<root>/ais_decoder_training/staged/joint_sam2_hvit_t_multi_gpu/<variant>.pt` (v4 `model_state` + trained
`unetr_state`) for `MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<root>/ais_decoder_training/staged`,
`--joint-checkpoint <variant>`.

Compute: `grete:shared`, one A100 (`-G A100:1`, 16 CPUs, 64 G, 12 h); the eight `3g.40gb` slices were held by
two-day jobs of one user (preemption off) and recent single-A100 jobs on `grete:shared` started within 1-1.5 h.

### Smoke tests (session slice 1g.20gb, one CPU, 2026-09-07 01:20)

`contact`, batch 4, one loader worker, 20 iterations: loaders built in ~1 min (1270 samples per epoch,
150 validation crops), `check_loader` 3 raw / 5 target channels, loader 3.9 samples/s with one worker
(1.02 s per batch of 4, so 12 workers deliver ~45 samples/s), GPU step 1.24 s per iteration at batch 4
(7.2 GiB allocated, 11.0 GiB reserved), 20 iterations plus a 38-batch validation in 60 s, loss 1.25 -> 1.50
validation metric (untrained fifth channel). Extrapolation to a full A100 (six to seven times the SMs of the
slice): ~0.4 s per iteration at batch 8, ~15 GiB allocated, so batch 8 fits a 40 GB node with margin and the
loader is not the bottleneck.

### Budget and submission (2026-09-07 01:30)

Queue at submission time: two of the eight `3g.40gb` slices free (the other six held by two-day jobs until
2026-09-08 evening), 14 `2g.20gb` slices free, grete:shared with 200 of 244 A100s allocated and about 16 free
on non-reserved nodes but other users' single-A100 arrays pending with reason "WaitingInQueue";
`sbatch --test-only` estimated a 25 h start for a 16-CPU A100 job, which the recent starts (1-1.5 h) contradict.
Decision: identical training for all four (batch 8, `--epoch-scale 4` = 5080 samples / 635 iterations per
epoch, **48000 iterations**, lr 5e-5, 12 loader workers), spread over the two pools so that every model is
done within about 12 h: `baseline` and `contact` on the free `3g.40gb` slices (`grete:preemptible`,
`-t 14:00:00`; ~0.8 s per iteration expected, ~11 h), `fgcal` and `both` on `grete:shared` A100 (`-t 12:00:00`,
~0.4 s per iteration, ~6 h), `both` submitted last with `--dependency=after` on the other three. 48000
iterations x 8 = 384k samples = 76 epochs of the 5080-sample epoch; the hardware only changes the wall time.

First submission (jobs 15769310-13, 01:26): all four died with exit 139 within 3-11 min of training. Two
causes, both fixed before the resubmission: (1) Python 3.14 starts DataLoader workers through a fork server,
so every worker re-imported the environment (30-60 s each, serialised; the two 3g jobs spent nine minutes
before their first iteration, and the non-persistent validation workers would have paid it every epoch) -
`train_ais_decoder.py` now forces the fork start method; (2) a 512^2 zarr file with fewer than three objects
makes torch_em's `MinInstanceSampler` reject the same crop 500 times and raise, which ended the run
(`RandomSubsetDataset` now redraws another file, `FixedSubsetDataset` does the same for validation, and both
wrap every dataset; torch_em's image-collection datasets already rotate images after 50 failed crops). Measured
rejection rates of the container datasets: yeaz phase-contrast stack frames 9/290, yeaz bright field 1/40,
dynamicnuclearnet 0/120, tissuenet 0/60, puma 0/40, tnbc 0/34 - the sparse yeaz frames were the trigger.

Second submission (02:47): `baseline` 15769606 and `contact` 15769607 on `3g.40gb` (started at once, both on
ggpu158), `fgcal` 15769609 on `grete:shared` A100 (started 02:52 on ggpu114 after five minutes in the queue),
`both` 15769611 after the three. Measured speed on a 3g.40gb slice: 1.08 iterations/s at batch 8, so 48000
iterations take about 12.8 h (finish ~15:40). The evaluation is chained by SLURM: `ais_eval_<variant>` jobs
15769618-15769621 run `finetuning/v2/generalist/ais_decoder/evaluate_ais_decoder.sh <variant> best` after their
training succeeds (stage, predict v5 primary / training_extra / holdout and apg3d primary / holdout, then the
`current-defaults` runs plus `contact-ridge` / `contact-mask` for the five-channel decoders on the caches).

## 4. Results

Readout (`report_ais_decoders.py`, reference = the `baseline` decoder under the library defaults; the production
decoder `5a729846...` is the second reference, epoch A5 runs):

- 4.1 Development set (primary + training_extra, eleven datasets, `--ndim 2`): balanced mSA, gate verdict,
  merged + absorbed share, `fg_area_ratio`, `matched_iou` per variant and configuration (`current-defaults`;
  `contact-ridge` and `contact-mask` for the five-channel decoders).
- 4.2 Holdout (five datasets).
- 4.3 3D crops (apg3d primary / holdout): family macros with `current-defaults` (and `contact-ridge`).
- 4.4 Field diagnostics (`diagnose_decoder_fields.py`): contact cosine at +-1 / +-3 px, contact Dice, magnitude
  at contacts vs interior, foreground area ratio at threshold 0.5.
- 4.5 Training curves: validation loss per variant (TensorBoard under `<root>/ais_decoder_training/logs/`).

Training curves at 03:55 (validation loss per epoch of 635 iterations, loss = metric of each variant, so the
values are not comparable across variants): baseline 0.180, 0.174, 0.166, 0.166, 0.162, 0.157 (six epochs);
contact 0.933, 0.799, 0.762, 0.764, 0.733, 0.724 (the fresh contact head dominates the early loss); fgcal 0.493,
0.450, 0.434, 0.438, 0.428, 0.414, 0.410, 0.410, 0.405, 0.414, 0.402 (eleven epochs); both 1.149, 1.071, 1.017,
1.041, 1.008, 0.983, 0.979, 0.965, 0.950. All four decrease; none has plateaued yet.

`fgcal` finished at 09:19 after 6.43 h (48000 iterations, peak 14.1 GiB allocated / 21.2 GiB reserved on an
A100-40GB; best epoch 64 of 76, validation loss 0.367); its evaluation chain (predict arrays 15772069 / 15772070,
screens 15772071 / 15772072) started at 09:21.

### 4.0 Preliminary: `fgcal` against the production decoder (09:35, before the fine-tuned baseline exists)

`current-defaults`, epoch A5, checkpoint `dd52aee4...` vs production `5a729846...`:

| set | production | fgcal | up | worst | merged + absorbed (object-weighted) | unseeded | fg area ratio (mean over datasets) |
|---|---:|---:|---|---|---|---|---|
| dev (11) | 0.3437 | 0.4170 (+21.3 %) | 9 / 11 | covid_if -35.9 %, deepseas -25.8 % | 25.2 % -> 17.5 % | 20.7 % -> 14.1 % | see below |
| holdout (5) | 0.2437 | 0.3938 (+61.6 %) | 5 / 5 | tissuenet +19.0 % | 29.4 % -> 17.9 % | 20.4 % -> 14.5 % | |

Per dataset (dev): livecell 0.277 -> 0.365, tissuenet 0.224 -> 0.263, dynamicnuclearnet 0.545 -> 0.831, deepbacs
0.181 -> 0.326, dic_hepg2 0.003 -> 0.174, neurips 0.226 -> 0.295, yeaz 0.616 -> 0.819, puma 0.469 -> 0.536, tnbc
0.368 -> 0.405, covid_if 0.740 -> 0.474, deepseas 0.134 -> 0.099. Merged + absorbed: livecell 36.9 -> 21.9 %,
tissuenet 25.6 -> 13.7 %, deepbacs 23.2 -> 9.8 %, neurips 28.0 -> 30.8 %. Foreground area ratio at 0.5: deepbacs
1.76 -> 1.25, livecell 1.11 -> 1.06, dynamicnuclearnet 0.92 -> 1.02, tissuenet 0.87 -> 0.75 (more under-coverage),
covid_if 1.05 -> 1.30, puma / tnbc / yeaz ~1.0 in both.

Reading: the two datasets that lose are exactly the two the fine-tuned decoders never saw (covid_if, deepseas),
and dic_hepg2 / dynamicnuclearnet / yeaz (never in the joint training) gain the most, so this comparison mostly
measures "12 h of decoder fine-tuning on the tuning datasets' train splits", not the boundary-weighted loss.
The isolating comparison is against the fine-tuned `baseline` (same data, same budget), pending.

Field diagnostics (`diagnose_decoder_fields.py`, dev caches, per-dataset medians; `ais/reports/decoder_fields_{fgcal,production}.csv`):

| quantity | production | fgcal |
|---|---|---|
| distance magnitude in the background | 0.83-0.86 on every dataset (the label fill value) | 0.03-0.05 |
| magnitude at ground-truth contact pixels | 0.10-0.29 | 0.04-0.11 |
| flow cosine across a contact, +-1 px | 0.63 (tissuenet), 0.71 (dnn), 0.77 (yeaz), 0.88 (livecell) | 0.44, 0.47, 0.37, 0.88 |
| flow cosine across a contact, +-3 px | -0.58, -0.55, -0.52, -0.15 | -0.70, -0.72, -0.85, -0.34 |
| fg IoU at 0.5 (median) | livecell 0.84, dnn 0.78, deepbacs 0.66, neurips 0.65, tissuenet 0.76, covid_if 0.92 | 0.88, 0.93, 0.78, 0.74, 0.74, 0.77 |
| fg area ratio at 0.5 (median) | deepbacs 1.49, dic_hepg2 0.10, dnn 0.90, tissuenet 0.91, covid_if 1.05 | 1.13, 1.04, 1.01, 0.79, 1.28 |

The background magnitude change matters for `boundary_magnitude_max`: the filter assumes a false region's
boundary runs through magnitude ~1; with ~0 in the background it no longer discriminates. Contact flips are
sharper but not negative at +-1 px. Attribution (fine-tuning vs the boundary loss) waits for the baseline.

3D crops (apg3d primary + holdout, 75 crops, `current-defaults`), fgcal vs production: every LM family loses
(celegans_atlas 0.104 -> 0.011, embedseg_platy_ish 0.339 -> 0.135, embedseg_platy_nuclei 0.259 -> 0.086,
embedseg_skull 0.118 -> 0.079, gonuclear 0.256 -> 0.136, platynereis_nuclei 0.068 -> 0.006) and the EM CREMI
scores roughly double (cremi 0.99 -> 2.24, cremi_seen 0.59 -> 2.15, snemi 0.97 -> 1.95, humanneurons 1.35 ->
1.97); the volume foreground balloons (area ratio celegans 1.28 -> 2.38, gonuclear 2.02 -> 2.68) and merges rise
(celegans 33 -> 75 %). Expected for a 2D-only, LM-only decoder fine-tune (the 3D path of the decoder saw no data),
and the reason these decoders cannot replace the production one for volumes or EM; the 3D crops serve as the
regression instrument of the campaign only.

`both` (fgcal + contact, checkpoint `25e2a32a...`, best epoch 73) against production, 2D (09:50):

| set / configuration | production | fgcal defaults | both defaults | both contact-ridge | both contact-mask |
|---|---:|---:|---:|---:|---:|
| dev balanced (11) | 0.3437 | 0.4170 | 0.4090 | 0.4096 | 0.4093 |
| holdout balanced (5) | 0.2437 | 0.3938 | 0.3819 | 0.3819 | 0.3823 |
| dev merged + absorbed, object-weighted | 25.2 % | 17.5 % | 14.8 % | 12.6 % | 13.9 % |
| livecell mSA / merged + absorbed | 0.277 / 36.9 % | 0.365 / 21.9 % | 0.382 / 19.2 % | 0.385 / 15.7 % | 0.384 / 17.8 % |
| tissuenet mSA / merged + absorbed | 0.224 / 25.6 % | 0.263 / 13.7 % | 0.272 / 12.7 % | 0.268 / 11.9 % | 0.271 / 12.5 % |
| neurips mSA / merged + absorbed | 0.226 / 28.0 % | 0.295 / 30.8 % | 0.317 / 22.4 % | 0.314 / 19.7 % | 0.318 / 21.2 % |
| deepbacs mSA | 0.181 | 0.326 | 0.287 | 0.285 | 0.287 |
| deepseas / covid_if mSA (unseen) | 0.134 / 0.740 | 0.099 / 0.474 | 0.046 / 0.462 | 0.046 / 0.460 | 0.046 / 0.462 |

The five-channel model wins on the three touching-cell datasets (livecell, tissuenet, neurips) and loses on
deepbacs and on the two unseen datasets, so its balanced score is 2 % below fgcal. The contact ridge at weight
1.0 removes another 2-4 points of merges on livecell / tissuenet / neurips for +0.1-0.7 % mSA; the mask mode at
0.5 changes little (the contact head is rarely above 0.5). Both post-processing settings are untuned. The
isolating pairs (contact vs baseline, both vs fgcal with the same data) complete when the 3g jobs finish.
Tables: `ais/reports/decoders_prelim_{primary_training_extra,holdout}*.csv`.

`both` on the 3D crops scores exactly 0 on every LM family: the volumes get 180-350 seeds and a foreground
(fg IoU 0.34, area ratio 2.7) but every instance is removed by `boundary_magnitude_max=0.4`, because the
five-channel decoder's magnitude inside the true objects of a volume is 0.86 (median; fgcal 0.35, production
0.27), i.e. its 3D distance field has drifted towards the fill value. Without the filter one celegans crop gives
57 instances (ground truth 72; production 57). A 3D-only effect of the 2D fine-tune, recorded, not pursued.
Tables: `ais/reports/decoders_prelim_3d*.csv`.

Contact head of `both` (`ais/reports/decoder_fields_both.csv`, medians, threshold 0.5): Dice against the true contact
lines livecell 0.57 (precision within 2 px 0.81, recall 0.55), yeaz 0.67 (0.88 / 0.65), dynamicnuclearnet 0.65
(0.94 / 0.51), covid_if 0.45, tissuenet 0.26 (precision 0.93 but recall 0.16), neurips 0.19 (recall 0.02),
dic_hepg2 / deepbacs / tnbc / puma ~0 (dic_hepg2 has 2169 true contact pixels per crop and predicts none). The
head is precise but under-confident on the datasets with the largest merge losses, which is why the mask mode at
0.5 did nothing; a class-weighted or focal contact loss is the recipe change to consider for the big run. The
contact training also sharpened the flow: the +-1 px contact cosine drops from 0.47 / 0.44 / 0.37 (fgcal, dnn /
tissuenet / yeaz) to 0.11 / 0.32 / -0.03.

### 4.6 Tuning launched in the meantime (10:53)

Grid sweeps (`configs/ais_grid_lm_v4.json`, 1728 combinations) on the development caches of fgcal (jobs 15772848 /
15772849) and both (15772850 / 15772851), contact-configuration screens for both (15772852: ridge 0.5 / 2 / 4, mask
0.3 / 0.7, ridge 1 + mask 0.5, on dev and holdout), and a launcher (15772853,
`finetuning/v2/generalist/ais_decoder/launch_tuning_after_caches.sh`) that submits the same for baseline and contact
once their caches exist and then ranks every sweep into `ais/reports/dec_<variant>_sweep_dev.csv`
(`report_ais_sweep.py`, reference = library defaults). Read the model comparison at tuned settings on the holdout,
not on the development set the sweep tuned on.

fgcal sweep ranked (11:55, `ais/reports/dec_fgcal_sweep_dev.csv`, 1728 combinations, reference = library defaults on
the fgcal caches, dev balanced 0.4170): the best shared configuration reaches 0.4298 (+3.1 %, 6 / 11 up, worst
-6.4 %, mean ratio to the per-dataset optimum 0.935); nothing passes the gate. The top rows all use travel 800
(n_iter 800, dt 0.5), density 50 (or 20), sigma 0.5, foreground weight 0.75, min_size 50, filter 0.4 or off - a
different regime from the production defaults (travel 25, density 10, sigma 1.0), consistent with a field that now
converges to sinks (magnitude ~0 in the background, sharper flips). Confirmation of the top-1 and the density-20
variant on dev + holdout: `configs/ais_dec_fgcal_top{1,10}.json`, job dec_fgcal_top_screen.

both sweep ranked (12:35, `ais/reports/dec_both_sweep_dev.csv`, contact terms not part of the cached sweep): best
shared configuration 0.4254 (+4.0 % over its defaults 0.4090, 8 / 11 up, worst -8.2 %), the same regime as fgcal
(travel 800, density 50, sigma 0.5, foreground weight 0.75, min_size 50). At the tuned shared setting fgcal stays
1 % ahead of both on the development set (0.4298 vs 0.4254). Screens of this shared top configuration alone and
with the contact terms (ridge 1; ridge 2 + mask 0.3) on dev + holdout for both: `configs/ais_dec_top1*.json`,
job dec_both_top_screen.

Contact configurations on both (13:25, `ais/reports/dec_both_contact_{primary_training_extra,holdout}*.csv`,
reference = both under the library defaults, other parameters at the defaults): ridge weights 0.5 / 1 / 2 / 4 give
+0.2 / +0.2 / +0.3 / +0.3 % balanced on dev (4-5 of 11 up, worst -1.5 %) and +0.2 / 0.0 / +0.1 / -0.3 % on holdout;
mask thresholds 0.3 / 0.5 / 0.7 give 0.0 / +0.1 / 0.0 % (dev) and -0.0 / +0.1 / +0.1 % (holdout); ridge 1 + mask 0.5
+0.2 / 0.0 %. The ridge does what it is meant to - seeded merges fall from 6.3 % to 4.1 % of the objects on dev (7.0
to 4.7 % on holdout) with no change in unseeded objects - but the recovered objects hardly move mSA at IoU 0.5, so
with these decoders the contact channel is not where the remaining mSA is (merges are down from 13 % to 6 % of the
objects already by the fine-tuning).

Tuned comparison of the two finished decoders (13:40, `ais/reports/dec_tuned_prelim_{primary_training_extra,holdout}*.csv`;
`dec-top1` = travel 800, density 50, sigma 0.5, fw 0.75, min_size 50, filter 0.4, the shared optimum of both sweeps):

| configuration | dev balanced (11) | holdout balanced (5) | holdout seeded merges |
|---|---:|---:|---:|
| production, defaults | 0.3437 | 0.2437 | 16.7 % |
| fgcal, defaults | 0.4170 | 0.3938 | 8.7 % |
| fgcal, dec-top1 | 0.4298 | 0.4094 | 10.9 % |
| both, defaults | 0.4090 | 0.3819 | 7.0 % |
| both, dec-top1 | 0.4254 | 0.4098 | 8.9 % |
| both, dec-top1 + contact ridge 1 | 0.4271 | 0.4113 | 4.5 % |

Holdout per dataset (fgcal top1 / both top1 + ridge): deepbacs 0.389 / 0.340, dic_hepg2 0.226 / 0.250, dynamicnuclearnet
0.822 / 0.825, livecell 0.355 / 0.382, tissuenet 0.255 / 0.261. Reading: at tuned settings the two decoders are within
0.5 % of each other on the holdout; the tuned regime (few, converged seeds) brings merges back for fgcal, which the
contact ridge removes for both without changing mSA; the shared configuration trades livecell / tissuenet for deepbacs
/ dic_hepg2 (the 6 / 11 "up" of the sweep). The isolating pairs against baseline and contact are pending.

`baseline` finished at 15:42 after 12.87 h on a 3g.40gb slice (48000 iterations, best epoch 75 of 76, peak 14.1 GiB);
`contact` at 15:50 after 12.97 h. Their chains (staging, caches, default and contact screens), the finalisation job
and the launcher's sweeps follow automatically; the shared tuned configuration `dec-top1` is screened for baseline
(job dec_baseline_top_screen) and contact as well, so all four decoders can be read at the same tuned setting.

### 4.1 The isolating comparison under the library defaults (16:05; contact pending)

Reference = the fine-tuned `baseline` (same data, budget, initialisation; checkpoint under `staged/baseline.pt`):

| decoder | dev balanced (11) | vs baseline | up / worst | holdout balanced (5) | vs baseline | up / worst | seeded merges dev / holdout |
|---|---:|---:|---|---:|---:|---|---|
| production | 0.3437 | -17.1 % | 2 / 11, dic_hepg2 -98 % | 0.2437 | -37.4 % | 0 / 5 | 13.3 % / 16.7 % |
| baseline | 0.4145 | - | - | 0.3894 | - | - | 8.1 % / 8.8 % |
| fgcal | 0.4170 | +0.6 % | 6 / 11, dic_hepg2 -8.8 % | 0.3938 | +1.1 % | 4 / 5, deepbacs -5.0 % | 8.3 % / 8.7 % |
| both | 0.4090 | -1.3 % | 7 / 11, deepseas -49 % | 0.3819 | -1.9 % | 3 / 5, dic_hepg2 -24 % | 6.3 % / 7.0 % |
| both + contact ridge 1 | 0.4096 | -1.2 % | 7 / 11 | 0.3819 | -1.9 % | 3 / 5 | 4.2 % / 4.8 % |

Per dataset against baseline (dev): fgcal livecell 0.0 %, tissuenet +3.0 %, neurips +5.8 %, puma +4.4 %, yeaz +4.0 %,
dnn +0.7 %, deepbacs -5.0 %, dic_hepg2 -8.8 %, tnbc -2.3 %, covid_if -4.5 %, deepseas +11 %; both livecell +4.6 %,
tissuenet +6.6 %, neurips +13.7 %, yeaz +3.6 %, puma +2.8 %, tnbc +2.2 %, dnn +0.5 %, deepbacs -16.4 %, dic_hepg2
-28.6 %, covid_if -6.9 %, deepseas -48.6 %. Holdout: fgcal deepbacs -5.0 %, dic_hepg2 +7.5 %, dnn +1.4 %, livecell
+0.4 %, tissuenet +5.0 %; both deepbacs -16.4 %, dic_hepg2 -23.7 %, dnn +1.7 %, livecell +5.3 %, tissuenet +10.6 %.

Reading:
1. Almost the entire gain over the production decoder (+21 % dev, +60 % holdout) is the decoder fine-tune on the
   tuning datasets' train splits, with the unchanged loss. The fine-tuned baseline already cuts merges from 13 % to
   8 % of the objects and moves the foreground area ratio to ~1 on most datasets (deepbacs 1.76 -> 1.19).
2. The boundary-weighted foreground loss (point 4.1) adds +0.6 % / +1.1 % balanced, on 6 / 11 and 4 / 5 datasets,
   with a -5 to -9 % loss on deepbacs or dic_hepg2; the foreground area ratio and the merge share are unchanged
   against baseline (tissuenet under-coverage 0.71 -> 0.75). It fails the generalization gate.
3. The contact channel (point 1.1, here on top of fgcal) is a strong, dataset-dependent lever: +5 to +14 % on the
   touching-cell datasets (livecell, tissuenet, neurips) with the merge share down to 6 % (4 % with the ridge), but
   -16 % on deepbacs and -24 to -29 % on dic_hepg2, so the balanced score is 1-2 % below baseline. The contact-vs-
   baseline pair (pending) separates the channel from the fgcal loss it was stacked on.

Mechanisms behind the both-vs-baseline differences (`ais/reports/decoders_isolating_dev_mechanisms.csv`, % of
objects): dic_hepg2 loses seeds (unseeded 43.9 -> 57.1 %, absorbed 34.1 -> 44.3 %; merges unchanged), deepbacs
splits its thin rods (1.7 -> 4.1 %) with a lower matched IoU (0.771 -> 0.744); livecell (merges 11.8 -> 9.2 %,
unseeded 15.2 -> 13.4 %), tissuenet (3.2 -> 2.5 %, 23.9 -> 22.6 %) and neurips (13.2 -> 10.0 %, 17.2 -> 14.9 %) gain on
both counts with higher matched IoU (0.777 -> 0.784, 0.737 -> 0.740, 0.772 -> 0.789). The contact head itself
never fires on dic_hepg2 or deepbacs, so their losses come from the shared features the extra task changed, not
from the ridge.

### 4.2 All four decoders under the library defaults (16:25; `ais/reports/decoders_defaults_{primary_training_extra,holdout}*.csv`)

| decoder (configuration) | dev balanced | vs baseline | up / 11 | worst | holdout balanced | vs baseline | up / 5 | worst | seeded merges dev |
|---|---:|---:|---|---|---:|---:|---|---|---:|
| production | 0.3437 | -17.1 % | 2 | dic_hepg2 -98 % | 0.2437 | -37.4 % | 0 | dic_hepg2 -99 % | 13.3 % |
| baseline | 0.4145 | - | - | - | 0.3894 | - | - | - | 8.1 % |
| fgcal | 0.4170 | +0.6 % | 6 | dic_hepg2 -8.8 % | 0.3938 | +1.1 % | 4 | deepbacs -5.0 % | 8.3 % |
| contact | 0.3952 | -4.7 % | 4 | dic_hepg2 -38 % | 0.3691 | -5.2 % | 2 | dic_hepg2 -40 % | 7.8 % |
| contact + ridge 1 | 0.4004 | -3.4 % | 4 | deepseas -26 % | 0.3770 | -3.2 % | 2 | dic_hepg2 -23 % | 4.5 % |
| both | 0.4090 | -1.3 % | 7 | deepseas -49 % | 0.3819 | -1.9 % | 3 | dic_hepg2 -24 % | 6.3 % |
| both + ridge 1 | 0.4096 | -1.2 % | 7 | deepseas -49 % | 0.3819 | -1.9 % | 3 | dic_hepg2 -21 % | 4.2 % |

contact vs baseline per dataset (dev, defaults / ridge): tissuenet +8.1 / +6.5 %, neurips +5.6 / +7.9 %, livecell
+1.3 / +4.2 %, yeaz +2.3 / +2.1 %, puma -0.7 / -1.8 %, tnbc -0.2 / -1.2 %, dnn -3.1 / -3.5 %, deepbacs -12.0 / -11.5 %,
dic_hepg2 -38.1 / -8.5 %, covid_if -21.1 / -21.0 %, deepseas -25.6 / -25.9 %. Holdout: tissuenet +10.3 / +8.2 %,
livecell +1.5 / +4.8 %, dnn -2.6 / -2.6 %, deepbacs -12.0 / -11.5 %, dic_hepg2 -40.4 / -22.6 %.

Reading (all four, same data, budget and initialisation):
- Point 4.1 (boundary-weighted foreground BCE): +0.6 % / +1.1 % balanced, 6 / 11 and 4 / 5 datasets up, a 5-9 % loss
  on one dataset each time; foreground calibration and merge share unchanged against baseline. A marginal, non-
  uniform effect; it does not pass the gate.
- Point 1.1 (contact channel): a strong dataset-dependent trade, not a general gain: +6 to +8 % on tissuenet and
  neurips, +1 to +5 % on livecell and yeaz, against -12 % on deepbacs, -21 % on covid_if, -26 % on deepseas and
  -38 % on dic_hepg2 (-8.5 % once the ridge recovers the absorbed objects). The losses come through the shared
  features (fewer seeds on dic_hepg2 and covid_if, split rods on deepbacs), not through the ridge; the head itself
  never fires on those datasets. Stacked on fgcal (`both`) the trade is milder (-1.3 % / -1.9 %) with the same sign
  pattern.
- The dominant effect of the campaign is neither: the plain fine-tune on the tuning datasets' train splits lifts the
  decoder from 0.344 to 0.415 (dev) and from 0.244 to 0.389 (holdout), removes 40 % of the merges and calibrates
  the foreground area to ~1 on most datasets. covid_if and deepseas, the two datasets left out of training, lose
  (-33 % and -33 % for baseline vs production), so part of this is in-domain specialisation.

### 4.3 All four decoders at the shared tuned configuration (16:30; `ais/reports/decoders_tuned_{primary_training_extra,holdout}*.csv`)

`dec-top1` (travel 800, density 50, sigma 0.5, foreground weight 0.75, min_size 50, filter 0.4) is the optimum of both
the fgcal and the both sweep; reference = baseline at dec-top1 (0.4200 dev, 0.4025 holdout; its own sweep pending).

| decoder (configuration) | dev balanced | vs baseline | up / 11 | worst | holdout balanced | vs baseline | up / 5 | worst | seeded merges dev |
|---|---:|---:|---|---|---:|---:|---|---|---:|
| baseline (dec-top1) | 0.4200 | - | - | - | 0.4025 | - | - | - | 13.4 % |
| fgcal (dec-top1) | 0.4298 | +2.4 % | 9 | dic_hepg2 -5.1 % | 0.4094 | +1.7 % | 4 | deepbacs -4.5 % | 10.6 % |
| contact (dec-top1) | 0.4024 | -4.2 % | 5 | -26 % | 0.3872 | -3.8 % | 2 | deepbacs -18 % | 12.2 % |
| contact (dec-top1 + ridge 1) | 0.4155 | -1.1 % | 6 | -27 % | 0.4044 | +0.5 % | 3 | deepbacs -14 % | 4.1 % |
| both (dec-top1) | 0.4254 | +1.3 % | 8 | deepseas -47 % | 0.4098 | +1.8 % | 4 | deepbacs -16 % | 7.7 % |
| both (dec-top1 + ridge 1) | 0.4271 | +1.7 % | 8 | deepseas -48 % | 0.4113 | +2.2 % | 4 | deepbacs -16 % | 4.0 % |

Holdout per dataset at dec-top1 (baseline / fgcal / contact + ridge / both + ridge): deepbacs 0.407 / 0.389 / 0.351 /
0.340, dic_hepg2 0.220 / 0.226 / 0.248 / 0.250, dynamicnuclearnet 0.804 / 0.822 / 0.784 / 0.825, livecell 0.341 /
0.355 / 0.381 / 0.382, tissuenet 0.240 / 0.255 / 0.258 / 0.261. The tuned regime (few converged seeds) raises the
merge share of the four-channel decoders from 8 % to 13 %; the contact ridge is the only thing that brings it to 4 %.

### 4.4 Conclusions for the training recipe (2026-09-07, 16:35)

1. In-domain data dominates. A decoder-only fine-tune with the unchanged loss on the tuning datasets' train
   splits gains +21 % (dev) / +60 % (holdout) over the production decoder, halves the merge share and calibrates the
   foreground area; every proposed loss change is a small correction on top of that. For the next big run the
   composition of the training data (which of the evaluation datasets' train splits are included) matters far more
   than the two loss changes.
2. Point 4.1 (boundary-weighted foreground BCE): consistently small and positive. +0.6 / +1.1 % at the defaults,
   +2.4 / +1.7 % at the tuned setting, 9 of 11 dev datasets up at the tuned setting, but a 5-9 % loss on one dataset
   (dic_hepg2 or deepbacs) each time, so it misses the gate's worst-loss bound. It does not change the foreground
   area ratio or the merge share against the fine-tuned baseline. Cheap and safe to include, not decisive.
3. Point 1.1 (contact channel, plain Dice + BCE, ridge in the watershed): a dataset-dependent trade. +6 to +10 % on
   tissuenet, +6 to +8 % on neurips, +1 to +5 % on livecell (the datasets whose merges motivated it), but -12 % on
   deepbacs, -21 % on covid_if, -26 % on deepseas and -38 % on dic_hepg2 through the shared features (seeds lost,
   rods split), with a head that never fires on those datasets. The ridge itself is effective and cheap (merges
   6-13 % -> 4 % at any setting) and recovers half of the dic_hepg2 loss; stacked on fgcal (`both`) the trade
   narrows to -1.3 % / -1.9 % at the defaults and +1.7 % / +2.2 % at the tuned setting. Under the generalization rule
   the channel as trained here is not a win; the levers to try before including it in a big run are a class-
   weighted or focal contact loss (the head is precise but under-confident: recall 0.16 on tissuenet, 0.02 on
   neurips at 0.5) and a lower contact loss weight so that the shared features do not lose seeds on large or thin
   cells. The 3D path of the five-channel decoder also drifted (section 4.0), which a joint 2D + 3D run avoids.
4. Post-processing for fine-tuned decoders: their fields converge (magnitude ~0 in the background, sharper flips),
   and the tuned optimum moves to long travel (800) with a high density threshold (50), sigma 0.5 and foreground
   weight 0.75 (+1.3 to +3.1 % over the current defaults, 6-9 of 11 up, worst -5 to -8 %); the production defaults
   are no longer the right regime for such decoders, and `boundary_magnitude_max` loses its premise.

**Revised by 4.7 (18:15), once baseline had its own sweep:** point 2's "+2.4 % at the tuned setting" is measured
at `dec-top1`, which is fgcal's optimum and 1 % below baseline's own (baseline wants `foreground_threshold` 0.4,
fgcal and both 0.5). At each decoder's own optimum the boundary-weighted foreground loss is worth **+1.3 %**, and
its mechanism is the threshold calibration, not the merge share. Point 4's "the tuned optimum moves to density 50
and sigma 0.5" holds for the loss-changed decoders only; baseline keeps the production density 10 / sigma 1.0 and
only lengthens the travel.

(the 3D tables of all four and the unattended finalisation outputs are in 4.5, the sweep rankings in 4.7)

### 4.5 Round-1 completions: the 3D table of all four, the field diagnostics, the mask mode (17:25)

Written by `ais_decoder_finalize2` (15777315, four minutes once the screens were in; see 5.2 for why the first
attempt died): `ais/reports/decoders_final_{dev,holdout,3d}*.csv` and `decoder_fields_{baseline,contact}*.csv`.

**All four on the 3D crops** (apg3d primary + holdout, 75 crops, `current-defaults`; the balanced score mixes LM
mSA with the negated CREMI error, so read the families, not the aggregate):

| family | production | baseline | fgcal | contact | both |
|---|---:|---:|---:|---:|---:|
| celegans_atlas | 0.104 | 0.040 | 0.011 | 0.000 | 0.000 |
| embedseg_platy_ish | 0.339 | 0.156 | 0.135 | 0.000 | 0.000 |
| embedseg_platy_nuclei | 0.259 | 0.115 | 0.086 | 0.000 | 0.000 |
| embedseg_skull | 0.118 | 0.238 | 0.078 | 0.000 | 0.000 |
| gonuclear | 0.256 | 0.132 | 0.135 | 0.000 | 0.000 |
| platynereis_nuclei | 0.068 | 0.052 | 0.006 | 0.000 | 0.000 |
| cremi / cremi_seen (lower is better) | 0.99 / 0.59 | 1.87 / 1.23 | 2.24 / 2.15 | 2.19 / 2.05 | 2.09 / 1.89 |
| snemi / humanneurons (lower is better) | 0.97 / 1.35 | 1.69 / 1.97 | 1.95 / 1.98 | 2.16 / 2.37 | 1.92 / 2.04 |

The regression is the 2D-only fine-tune itself, not the loss changes: the unchanged-loss `baseline` already loses
25-60 % of every LM family (embedseg_skull is the exception, 0.118 -> 0.238) and adds 0.6-1.0 to every CREMI
error, before any loss change. Both five-channel decoders are exactly 0 on all six LM families because
`boundary_magnitude_max=0.4` removes every instance of a field whose magnitude no longer dips at boundaries
(section 4.0). Read as: a 2D-only decoder fine-tune cannot replace the production decoder for volumes, and the
magnitude filter has to be re-decided for any fine-tuned decoder - not as a verdict on the two loss changes.

**Field diagnostics of all four** (dev, per-dataset medians, `decoder_fields_<variant>_summary.csv`):

- Background distance magnitude 0.83-0.86 (production, the label fill value) -> 0.03-0.08 for *all four*
  fine-tuned decoders. `boundary_magnitude_max` loses its premise for every one of them, not only for fgcal.
- dic_hepg2 is a production-decoder failure, not a loss effect: fg IoU 0.07 at an area ratio of 0.10 (it barely
  predicts foreground there, hence mSA 0.003); every fine-tuned decoder reaches fg IoU 0.89-0.90 at ratio
  1.03-1.08. This single dataset carries most of the +21 % dev gain over production.
- The two datasets held out of training move the wrong way, which is where their losses come from: covid_if
  fg IoU 0.92 -> 0.72-0.77 with the area ratio 1.05 -> 1.26-1.35 (over-coverage), deepseas fg IoU 0.46 ->
  0.16-0.38 with the ratio 1.90 -> 0.53-1.06 (`both` the worst at 0.16 / 0.53, and it is the variant with the
  -49 % deepseas loss).
- The flow flip across a contact (cosine at +-1 px, lower is sharper) is sharpened by the fine-tune and again by
  the contact channel: dynamicnuclearnet 0.71 -> 0.25 (baseline) -> 0.11 (contact), tissuenet 0.63 -> 0.39 ->
  0.28, yeaz 0.77 -> 0.40 -> -0.18. The channel does to the field exactly what it was meant to do; the mSA it
  buys is the question, not the mechanism.
- fgcal against baseline moves the foreground in both directions rather than calibrating it: tissuenet
  under-coverage 0.75 -> 0.79 (better), deepbacs over-coverage 1.03 -> 1.13 (worse), the rest within 0.02.

**The mask mode**, added to the four-way defaults table: `contact` 0.3952 -> 0.3962 (dev) and 0.3691 -> 0.3706
(holdout), `both` 0.4090 -> 0.4093 and 0.3819 -> 0.3823. Confirms 4.2 - the mask is inert because the head
rarely exceeds 0.5.

**The fifth channel of the two round-1 decoders**, rescored with the mode-independent recalls (per-dataset
medians, threshold 0.5, `--contact-mode touching` = the target they were trained on):

| dataset | target px | `contact` pred px / Dice / precision 2px / recall_touching / recall_bg | `both` pred px / Dice / precision / recall_touching / recall_bg |
|---|---:|---|---|
| yeaz | 7570 | 6967 / 0.69 / 0.84 / **0.76** / 0.15 | 5772 / 0.67 / 0.88 / 0.65 / 0.09 |
| dynamicnuclearnet | 170 | 264 / 0.65 / 0.79 / **0.72** / 0.01 | 126 / 0.65 / 0.94 / 0.51 / 0.00 |
| livecell | 11400 | 11954 / 0.60 / 0.77 / **0.63** / 0.07 | 8499 / 0.57 / 0.81 / 0.55 / 0.05 |
| covid_if | 935 | 2000 / 0.36 / 0.35 / 0.59 / 0.03 | 1497 / 0.45 / 0.46 / 0.58 / 0.02 |
| tissuenet | 7117 | 1281 / 0.30 / 0.93 / **0.19** / 0.00 | 950 / 0.26 / 0.93 / 0.16 / 0.00 |
| neurips_cellseg | 1007 | 296 / 0.21 / 0.46 / **0.13** / 0.00 | 14 / 0.20 / 0.10 / 0.02 / 0.00 |
| puma | 233 | 128 / 0.17 / 0.49 / 0.12 / 0.00 | 28 / 0.05 / 0.37 / 0.03 / 0.00 |
| tnbc | 180 | 19 / 0.08 / 0.31 / 0.03 / 0.00 | 0 / 0.01 / 0.00 / 0.00 / 0.00 |
| deepbacs | 132 | 30 / 0.02 / 0.07 / 0.02 / 0.00 | 4 / 0.00 / 0.00 / 0.00 / 0.00 |
| dic_hepg2 | 3790 | 40 / 0.01 / 0.16 / 0.001 / 0.00 | 0 / 0.00 / 0.00 / 0.00 / 0.00 |

Three things this settles for round 2:

1. `recall_bg_boundary` is 0.00-0.15 everywhere, so both heads did learn the *touching* target specifically and
   ignore the background-facing rim - the target definition took, the confidence did not.
2. The head fires where merges are cheap (yeaz, dynamicnuclearnet, livecell: recall 0.63-0.76) and is nearly
   silent exactly where the campaign lost mSA: tissuenet 0.19, neurips 0.13, deepbacs 0.02, dic_hepg2 0.001
   (3790 target pixels per crop, 40 predicted). Those losses therefore cannot come from the ridge - they come
   from the shared features the extra task changed, as section 4.1 concluded.
3. The boundary-weighted foreground loss makes the head *less* confident, not more: every `both` recall is below
   its `contact` counterpart (neurips 0.02 vs 0.13, puma 0.03 vs 0.12, tnbc and deepbacs to zero). The two loss
   changes compete for the same decoder capacity.

The round-2 target has a few percent of the pixels positive instead of under one, which is the structural
version of the "class-weighted or focal contact loss" lever of section 4.4 point 3: if under-confidence was
class imbalance, `boundary` fixes it, and its `recall_touching` on tissuenet / neurips / deepbacs / dic_hepg2
is the number to look at.

**Do not read the `fg_area_ratio` column of the summary CSVs**: it is a mean over datasets, and deepseas (12-91)
and neurips (2.3-12) dominate it because their crops carry few or tiny ground-truth objects. The per-dataset
column of `*_mechanisms.csv` is the readable one (baseline / fgcal / contact / both on deepbacs 1.19 / 1.25 /
1.40 / 1.25, tissuenet 0.71 / 0.75 / 0.77 / 0.74, dic_hepg2 1.04 / 1.11 / 1.16 / 1.12).

### 4.7 Each decoder at its own sweep optimum, and the foreground threshold (19:00)

The sweep rankings (`ais/reports/dec_<variant>_sweep_dev.csv`, 1728 combinations, cached scorer, reference =
that decoder's library defaults) reproduce the screened full-pipeline runs to better than 0.05 %: baseline at
threshold 0.5 / density 50 / sigma 0.5 scores 0.4202 in the sweep against 0.4200 screened, fgcal 0.4298 against
0.4298, both 0.4254 against 0.4254. The sweep numbers below are therefore comparable to sections 4.2 / 4.3.

| decoder | own optimum (dev balanced) | vs baseline's optimum | gain over its defaults | n_up | worst | fg threshold | density / sigma |
|---|---:|---:|---:|---|---|---:|---|
| baseline | 0.4244 | - | +2.4 % | 9 / 11 | -9.1 % | **0.4** | 10 / 1.0 |
| fgcal | 0.4298 | **+1.3 %** | +3.1 % | 6 / 11 | -6.4 % | 0.5 | 50 / 0.5 |
| both | 0.4254 | +0.2 % | +4.0 % | 8 / 11 | -8.2 % | 0.5 | 50 / 0.5 |
| contact | 0.4083 | **-3.8 %** | +3.3 % | 7 / 11 | -9.6 % | **0.6** | 10 / 1.0 |

All four want the long travel (`n_iter` 800, `dt` 0.5), `foreground_weight` 0.75 and `min_size` 50; none passes
the gate; `boundary_magnitude_max` is irrelevant everywhere (0.4, 0.6 and off are within 0.001).

Two things this changes:

1. **The boundary-weighted foreground loss does calibrate the foreground, and the shared configuration hid it in
   the opposite direction.** Every one of baseline's top 20 rows uses `foreground_threshold` 0.4; at 0.5 it only
   reaches 0.4202 (+1.4 %). fgcal and both peak at 0.5. So the plain decoder needs its threshold lowered by a
   tenth to reach its best, the boundary-calibrated ones are optimal at the natural 0.5 - which is exactly what
   point 4.1 claims and what the area-ratio column was too coarse to show. Section 4.3 compared all four at
   `dec-top1` (threshold 0.5), i.e. at fgcal's optimum and 1 % below baseline's, so the +2.4 % it reports for
   fgcal is really **+1.3 %** (0.4298 against baseline's own 0.4244). Point 4.1 is a real but smaller effect,
   and its mechanism is the threshold, not the merge share.
2. **The contact channel inflates the foreground, and the boundary-weighted BCE undoes it.** The optimal
   threshold runs baseline 0.4 -> contact 0.6 -> fgcal / both 0.5. The field diagnostics say the same thing at a
   fixed threshold: contact's `fg_area_ratio` at 0.5 is above baseline's on ten of eleven datasets (deepbacs
   1.21 vs 1.03, neurips 1.15 vs 1.03, tnbc 1.03 vs 0.88, puma 1.03 vs 0.91). The extra task pushes foreground
   probability mass outward, and the calibrated loss pulls it back - which is why `both` sits between the two.
3. **The contact channel is a loss even at its own optimum.** Against baseline's own optimum, fgcal is +1.3 %,
   both +0.2 % and contact **-3.8 %**. Section 4.2 measured -4.7 % at the shared defaults and 4.3 -4.2 % at
   `dec-top1`; giving each decoder its best post-processing moves that by less than one point. The "it was only
   mis-tuned" objection to section 4.4 point 3 is therefore closed: point 1.1 as implemented in round 1 loses.
4. **The "tuned regime moved" conclusion (4.4 point 4) is a property of the loss-changed decoders.** baseline
   and contact keep the production density (10) and sigma (1.0) and only lengthen the travel; fgcal and both move
   to density 50 / sigma 0.5. So the shift to "few, converged seeds" comes with the *foreground* loss change,
   not with decoder fine-tuning as such.

## 5. Round 2: the proper boundary channel (2026-09-07)

Round 1 leaves point 1.1 undecided in the user's reading: the contact-only fifth channel (touching boundaries,
under 1 % of the pixels, ill-defined where three cells meet) is a dataset-dependent trade with an under-confident
head. Round 2 replaces it with the **classical boundary target**: the fifth channel holds the inner boundary of
every object, to neighbours and to background alike (`contact_mode="all"`, dilated by 1), which coincides with the
zero level set of the three geodesic distance channels the decoder already predicts, so the extra task no longer
asks for a quantity the other channels do not encode.

| variant | fifth channel | foreground loss |
|---|---|---|
| `boundary` | inner boundary of every object, Dice + BCE | Dice (unchanged) |
| `boundary_fgcal` | same | Dice + boundary-weighted BCE (`boundary_weight=4`, radius 2) |

`boundary` vs `baseline` isolates the channel, `boundary_fgcal` vs `fgcal` isolates it on top of the calibrated
foreground, and `boundary` vs `contact` isolates the target definition at a fixed loss. Everything downstream still
treats the channel as "contact" (sigmoid activation, `flow_instance_segmentation(contact=, contact_weight=,
contact_mask_threshold=)`, the `ais_contact_*.json` configs), so the round-1 readouts apply unchanged.

### 5.1 Launch (17:16), and why the jobs first refused to start

Submitted at 16:59 for `3g.40gb` slices with seven of the eight free, both jobs stayed `PENDING/WaitingInQueue`
for 17 minutes although slices, CPUs and memory were free on ggpu158 and ggpu192, and `sbatch --test-only`
claimed a start no earlier than 2026-09-08T03:13 for *every* pool (A100 on grete:shared, 3g, 2g, 1g on
preemptible) and independently of `--time`, `-c` and `--mem`. Cause: our own `dec_baseline_sweep_primary` array
sat at `TopOfQueue` on `grete:preemptible` with a marginally higher priority (103063 vs 103021), and an
unschedulable job at the head of the queue blocks the partition in the main scheduling loop for every
lower-priority job of the same user. `scontrol hold` on the four sweep arrays started both trainings within
seconds. The durable fix (the arrays only feed the sweep ranking, so they are the cheapest thing to delay):

```bash
for j in 15776127 15776128 15776228 15776229; do scontrol update jobid=$j nice=100; done
```

which puts the sweeps below the rest of our chain (evaluation, finalisation, tuning launchers) while keeping them
ahead of the other user's queued preemptible job. Note for the next campaign: `--test-only` is worthless on
`grete:preemptible` because it ignores preemption - a sweep task started at 17:11 against a 03:14 estimate for
the same request. The only thing worth checking when a job does not start is whether one of our own arrays is at
the head of the queue.

`boundary` = 15776831 (ggpu158), `boundary_fgcal` = 15776833 (ggpu192), both at 1.08 it/s for batch 8 (the
round-1 3g speed), so 48000 iterations plus the per-epoch validation land at 06:10-06:20 on 2026-09-08 inside
the 14 h limit (07:16). The first log lines confirm the target: `variant boundary: 5 output channels, loss
settings {'contact': True, 'contact_mode': 'all', 'boundary_weight': None}`.

### 5.2 The round-1 finalisation died on an edited script

`ais_decoder_finalize` (15772287) waited 7.4 h for the last screens, printed "all screens done" at 17:11 and then
aborted with `finalize_ais_decoder_reports.sh: line 44: syntax error near unexpected token 'done'`. The file is
syntactically fine; it had been edited at 16:53 while the job slept in the wait loop, and bash re-reads a running
script by byte offset, so the resumed parse landed mid-statement. None of the `decoders_final_*` tables or the
`baseline` / `contact` field diagnostics were written. Rerun as 15777315. **Rule from now on: submit a frozen
copy of every long-running driver**, `<camp>/jobs/frozen/<name>_<timestamp>.sh`, never the repo path.

The same 16:53 edit claimed a second job six hours later: `ais_decoder_tuning` (15772853, running since 12:28)
died at 19:03 with `break: only meaningful in a for, while or until loop` followed by
`syntax error near unexpected token 'done'` in `launch_tuning_after_caches.sh`, and its last log line is the
message of a branch it could not have reached - the signature of a shifted offset. Nothing was lost: it had
already submitted the `baseline` / `contact` sweeps at 16:03, and all four rankings exist (fgcal and both at
11:55 / 12:35, baseline and contact by hand at 18:12 / 18:57, section 4.7). Both files on disk pass `bash -n`
and the frozen copies under `<camp>/jobs/frozen/` are byte-identical to them, so `tuning2` and `finalize_r2`
are unaffected. **One edit to a driver can kill every job currently sleeping in it, hours apart.**

### 5.3 The chain (nothing depends on the session)

The session runs in a 12 h interactive job that ends at 05:06 on 2026-09-08, before the trainings do, so every
step is chained with SLURM dependencies.

| job | what | starts |
|---|---|---|
| 15776831 / 15776833 | the two trainings | running since 17:16, done ~06:15 |
| 15776838 / 15776839 `ais_eval_<variant>` | `afterany` the training: stage, cache v5 primary / training_extra / holdout and apg3d primary / holdout, then the `current-defaults`, `contact-ridge` and `contact-mask` screens | ~06:15 |
| 15777359 `ais_decoder_tuning2` | `afterany` both evaluations (frozen `launch_tuning_after_caches.sh`): waits for the 2d caches, submits the two grid sweeps (1728 combinations) and the eight-configuration contact screen per variant, then ranks all six sweeps into `<root>/ais/reports/dec_<variant>_sweep_dev.csv` | ~06:20 |
| 15777357 `ais_decoder_finalize_r2` | `afterany` both evaluations (frozen `finalize_round2_reports.sh`): submits the `dec-top1` screens of the two new decoders `afterok` their prediction jobs, waits for every round-2 screen, then writes `decoders_all_defaults_{dev,holdout}`, `decoders_all_tuned_{dev,holdout}`, `decoders_all_3d`, `decoders_<variant>_contact_dev` and the field diagnostics of both new decoders | ~06:20 |
| 15777315 `ais_decoder_finalize2` | the round-1 finalisation, rerun from a frozen copy | queued |

`finalize_round2_reports.sh` is new (`finetuning/v2/generalist/ais_decoder/`); it replaces the manual "submit the
`dec-top1` screens once the caches exist, then run the section 5 commands" step of the hand-over, so the
successor only has to read the tables.

### 5.4 What the two post-processing modes mean once the channel is a full boundary

Both modes read the fifth channel unchanged (`micro_sam/v2/postprocessing.py`), but the target swap changes what
they do, which is worth stating before the numbers arrive:

- `contact_weight` adds `w * contact` to the watershed height map. With the touching target the ridge sits only
  between two objects; with the full boundary it also runs along every object's rim to the background. The
  watershed is masked to the foreground, so a rim ridge mostly sits at the mask border and should be close to
  inert - except that the target is dilated by one pixel, so the ridge reaches one pixel *inside* the object and
  can shave structures only a few pixels wide (deepbacs rods, dic_hepg2 filaments).
- `contact_mask_threshold` excludes `contact > t` from the first seeded watershed and lets the instances claim
  those pixels afterwards. With a full boundary this is no longer "keep the contact line free" but the classical
  *erode, flood, dilate back* scheme: the first watershed runs on objects eroded by ~3 pixels. That should help
  wherever objects touch, and it is the mode that was inert in round 1 only because the head rarely exceeded
  0.5 - a confident boundary head makes it active for the first time. The risk is the same one: an object thinner
  than twice the band loses its interior entirely and can end up unseeded.

So the expected signature of the boundary channel, if it works, is: mask mode finally moving the score, the
merge share falling on livecell / tissuenet / neurips, and a *new* kind of loss on the thin-object datasets -
which the mechanism columns separate (`seeded_split` and `gt_with_0_seeds` rather than `seeded_merged`). Both
modes are screened at 0.5 / 1 / 2 / 4 and 0.3 / 0.5 / 0.7 for each new decoder, so this is testable rather than
argued.

### 5.5 `SBATCH_EXPORT=none` silently reverted the round-2 tuning to round 1 (2026-09-08, 06:15)

Both trainings finished cleanly on the first attempt - `boundary` COMPLETED in 12:48:29 (48000 iterations, best
epoch 65 of 76, validation 0.786 at epoch 1 -> 0.576) and `boundary_fgcal` in 12:49 (best 0.804 from 1.076) -
and the evaluation chain staged both checkpoints with five output channels and submitted the caches, screens and
`dec-top1` screens as designed.

`ais_decoder_tuning2` then did the wrong thing: at 06:13 it logged `caches of baseline ready, submitting sweeps`
and re-submitted the four **round-1** sweep arrays plus the 24-task `dec_contact_contact_screen`, and never
submitted anything for the two new decoders. Cause: **`SBATCH_EXPORT=none` is set in this environment**
(`echo $SBATCH_EXPORT`), so `sbatch` does not propagate the submitting environment and the `WAIT_VARIANTS` /
`VARIANTS` variables never reached the script, which fell back to its round-1 defaults. The note in the previous
hand-over - "`sbatch --export=ALL` is the default, so the two variables reach the script" - is wrong on this
system, so the original submission (15776840) carried the same latent bug; only the deliberate stop and restart
of the session caught it, because the failure is silent and produces plausible-looking work.

Recovery (06:15-06:17), all of it visible in `<root>/jobs/`:

- cancelled the five redundant arrays (46 tasks) and `tuning2`, and moved their job directories to
  `<root>/jobs/_superseded/` so that `tasks_done` sees the completed round-1 directories as the newest again
  (it reads `ls -td | head -1`, so an empty newer directory shadows a finished one);
- submitted by hand what the launcher should have: `dec_boundary_sweep_{primary,extra}` (15783735 / 15783736)
  and `dec_boundary_contact_screen` (15783737) on the finished cache, and the same three for `boundary_fgcal`
  (15783738 / 15783739 / 15783740) `afterok` its still-running `predict2d` array;
- `ais_rank_round2` (15783741) ranks both new sweeps `afterany` the four arrays.

Fix in the repository: `launch_tuning_after_caches.sh` now takes the variants as arguments
(`--wait boundary boundary_fgcal --rank baseline contact ... boundary_fgcal`) and only falls back to the
environment when run directly in a shell. **Rule: never pass campaign parameters to a SLURM job through the
environment on this cluster** - put them in the command line or in the frozen script.

### 5.6 Results of the boundary channel (2026-09-08, 06:30; `ais/reports/decoders_r2_early_*`)

Both trainings ran the full budget on the first attempt: `boundary` COMPLETED in 12:48:29 (best epoch 65 of 76,
validation 0.786 at epoch 1 -> 0.576), `boundary_fgcal` in 12:49 (best 0.804 from 1.076). Checkpoints
`66368b4c` and `0753918a`, staged with five output channels.

**1. The head is confident now - the class-imbalance diagnosis of 4.4 point 3 was right.**
`recall_touching` at threshold 0.5 (per-dataset medians, `decoder_fields_boundary_summary.csv`), round-1
`contact` -> round-2 `boundary`: deepbacs 0.015 -> **0.505**, tnbc 0.032 -> **0.587**, puma 0.123 -> **0.566**,
neurips 0.128 -> **0.460**, tissuenet 0.192 -> 0.347, livecell 0.634 -> 0.724, dynamicnuclearnet 0.720 -> 0.899,
yeaz 0.757 -> 0.914, covid_if 0.591 -> 0.740. It learned the actual target rather than collapsing onto the
touching lines (`recall_bg_boundary` 0.43-0.90 against 0.00-0.15 for `contact`) and stayed precise (precision
within 2 px 0.68-0.98, Dice up to 0.87 on dynamicnuclearnet, 0.81 yeaz, 0.69 livecell). Exactly the four
datasets whose merges motivated the channel and whose head was silent in round 1 now fire.

Two exceptions, and the first one taught us something about the instrument. On **`dic_hepg2`** the head has no
pixel above 0.5 on 33 of 50 crops (mean 13 predicted pixels against 8698 target pixels, per-crop maximum 0.41),
so every threshold-0.5 column calls it dead - but its *soft* probability is 0.133 on the true boundary against
0.0020 elsewhere, a 65-fold contrast, i.e. **well localised and merely under-confident** (deepbacs and livecell
run 0.65-0.69 against 0.0003). The consequence is visible in the scores: `contact_weight` adds
`w * contact` to the height map and therefore reads the soft map, so the ridge alone moves dic_hepg2 from
-13.8 % to +16.3 % against baseline at `dec-top1` - a 30-point swing out of a head that "predicts nothing".
`contact_mask_threshold` thresholds instead, and cannot use it. **Read the fifth channel's soft contrast, not
only its Dice and recall at 0.5**; the threshold columns understate a well-localised head. `deepseas` is
genuinely near-silent (Dice 0.056), as expected of binary masks with no true object boundaries.

**2. Under the library defaults** (dev = 11 datasets, holdout = 5, reference = the fine-tuned `baseline`):

| decoder (configuration) | dev balanced | vs baseline | up / 11 | holdout balanced | vs baseline | up / 5 | seeded merges dev |
|---|---:|---:|---|---:|---:|---|---:|
| `boundary` + ridge 1 | **0.4209** | **+1.5 %** | 7 | 0.3865 | -0.7 % | 3 | 4.5 % |
| `boundary` + mask 0.5 | 0.4201 | +1.3 % | 7 | 0.3855 | -1.0 % | 3 | 6.1 % |
| `boundary` defaults | 0.4192 | +1.1 % | 7 | 0.3838 | -1.4 % | 3 | 7.8 % |
| `fgcal` defaults | 0.4170 | +0.6 % | 6 | **0.3938** | **+1.1 %** | 4 | 8.3 % |
| `baseline` defaults | 0.4145 | - | - | 0.3894 | - | - | 8.1 % |
| `boundary_fgcal` defaults | 0.4124 | -0.5 % | 6 | 0.3746 | -3.8 % | 3 | 8.0 % |
| `both` defaults | 0.4090 | -1.3 % | 7 | 0.3819 | -1.9 % | 3 | 6.3 % |
| `contact` defaults | 0.3952 | -4.7 % | 4 | 0.3691 | -5.2 % | 2 | 7.8 % |

**3. At the shared tuned configuration** (`dec-top1`, reference = `baseline` at `dec-top1` = 0.4200 dev / 0.4025
holdout) the boundary channel gives **the best result of the whole campaign**:

| decoder (configuration) | dev balanced | vs baseline | up / 11 | worst | holdout balanced | vs baseline | up / 5 |
|---|---:|---:|---|---|---:|---:|---|
| `boundary` + ridge 1 | **0.4340** | **+3.3 %** | **10** | deepbacs -14.1 % | 0.4085 | +1.5 % | 4 |
| `boundary` + ridge 2 + mask 0.3 | 0.4325 | +3.0 % | 10 | -15.0 % | 0.4069 | +1.1 % | 4 |
| `both` + ridge 1 | 0.4271 | +1.7 % | 8 | deepseas -48 % | **0.4113** | **+2.2 %** | 4 |
| `boundary` (no ridge) | 0.4254 | +1.3 % | 7 | -13.8 % | 0.4022 | -0.1 % | 3 |
| `contact` + ridge 1 | 0.4155 | -1.1 % | 6 | -26.5 % | 0.4044 | +0.5 % | 3 |

**4. The round-1 collateral damage is repaired.** Per dataset at `dec-top1` + ridge 1 (dev), `contact` ->
`boundary`: covid_if -19.2 % -> **+0.2 %**, deepseas -26.5 % -> **+20.0 %**, dic_hepg2 +14.5 % -> +16.3 %,
dynamicnuclearnet -3.6 % -> +1.7 %, puma -1.3 % -> +2.1 %, tnbc +2.6 % -> +8.0 %, and the datasets the channel
was for stay up (livecell +10.7 %, tissuenet +9.1 %, yeaz +3.7 %, neurips +3.4 %). **Exactly one dataset is
down: deepbacs, -14.1 %** - and it is down by 13.7-16.4 % for `contact` and `both` too, so it is a property of
carrying a fifth channel at all, not of the target definition.

**5. deepbacs is the predicted thin-object failure** (5.4), and its mechanism is visible: `boundary` + ridge 1
against `baseline` at `dec-top1` splits more (`seeded_split` 2.8 % against 1.6 % of the objects), matches worse
(`matched_iou` 0.743 against 0.767) and above all over-covers (`fg_area_ratio` **1.36** against 1.19) - while
actually *improving* the two counts the channel targets (merges 2.8 % against 3.6 %, objects without a seed
5.2 % against 7.2 %). The rods are shaved and split, not merged.

**6. Stacking the two loss changes still hurts**, as in round 1: `boundary_fgcal` is below `boundary` everywhere
(-0.5 % against +1.1 % dev, -3.8 % against -1.4 % holdout at the defaults) and its head is a few points less
confident (deepbacs 0.466 against 0.505, puma 0.486 against 0.566, tnbc 0.466 against 0.587). It does calibrate
the foreground it was meant to (`fg_area_ratio` deepbacs 1.06 against 1.17, neurips 1.03 against 1.07, deepseas
0.86 against 1.54) - but that did not buy mSA, and on deepbacs it made the score worse (-13.7 % against -7.7 %
at the defaults), so the over-coverage is not what costs deepbacs its score.

**7. Gate verdict.** `boundary` + ridge 1 on dev: 10 of 11 datasets up, balanced +3.3 % (both bounds met), worst
-14.1 % against the -2 % bound - **it fails the gate on deepbacs alone**. On the holdout it is +1.5 % (4 of 5),
where `both` + ridge 1 reaches +2.2 %. So the proper boundary target turns point 1.1 from a broad
dataset-dependent trade (round 1: 4-6 datasets down, up to -38 %) into a broad gain with one identified,
channel-generic failure. That is a qualitatively different object from round 1 and the first version of the
fifth channel worth carrying further, but it is not yet a pass.

Still running at the time of writing: the 24-configuration contact screens of both new decoders, the
`dec-top1` screens of `boundary_fgcal`, the four grid sweeps and their ranking (`ais_rank_round2`, 15783741), and
the 3d screens. The sweep will say which `foreground_threshold` the boundary decoders want, which is the test of
4.7 point 1 (`boundary`'s median `fg_area_ratio` is above `baseline`'s on nine of eleven datasets, so 0.5-0.6 is
the expectation).

**8. The ridge and the mask separate cleanly** (`decoders_boundary_contact_{dev,holdout}*.csv`, `boundary`
against its own defaults 0.4192 dev / 0.3838 holdout, all other parameters at the library defaults):

| configuration | dev | vs defaults | up / 11 | worst | holdout | vs defaults | seeded merges dev | seeded splits dev |
|---|---:|---:|---|---|---:|---:|---:|---:|
| defaults | 0.4192 | - | - | - | 0.3838 | - | 7.8 % | 1.80 % |
| ridge 0.5 | 0.4207 | +0.35 % | 4 | -1.0 % | **0.3874** | **+0.95 %** | 4.8 % | 2.03 % |
| ridge 1 | 0.4209 | +0.41 % | 4 | -2.1 % | 0.3865 | +0.71 % | 4.5 % | 2.08 % |
| ridge 2 | **0.4214** | **+0.51 %** | 4 | -3.1 % | 0.3857 | +0.51 % | 4.3 % | 2.17 % |
| ridge 4 | 0.4212 | +0.48 % | 4 | -3.3 % | 0.3853 | +0.39 % | 4.3 % | 2.21 % |
| mask 0.3 | 0.4201 | +0.21 % | **7** | **-0.5 %** | 0.3852 | +0.36 % | 5.5 % | 1.86 % |
| mask 0.5 | 0.4201 | +0.20 % | 5 | -0.3 % | 0.3855 | +0.46 % | 6.1 % | 1.79 % |
| mask 0.7 | 0.4198 | +0.13 % | 7 | -0.0 % | 0.3847 | +0.24 % | 7.0 % | 1.67 % |
| ridge 1 + mask 0.5 | 0.4209 | +0.41 % | 4 | -2.1 % | 0.3865 | +0.71 % | 4.5 % | 2.10 % |

This revises the prediction of 5.4 in one respect and confirms it in another. The mask mode *does* move the
score now that the head is confident (+0.2 % dev, +0.36-0.46 % holdout, against 0.0-0.1 % in round 1), and it is
by far the more **uniform** lever: 7 of 11 datasets up with a worst case of -0.5 %, against the ridge's 4 of 11
and -1.0 to -3.3 %. But the shaving of thin objects is a **ridge** effect, not a mask effect: `seeded_split`
rises monotonically with the ridge weight (1.80 % -> 2.21 %) and stays flat or falls under the mask
(1.67-1.86 %) - exactly as the erode-*and-dilate-back* structure of the mask mode implies, which is the half of
5.4's reasoning that was right. The ridge still wins on balanced mSA because it removes almost twice as many
merges (7.8 % -> 4.3 % against 5.5-7.0 %) and because it can exploit an under-confident head (point 1).

**9. No post-processing setting can rescue deepbacs.** `boundary` is already -11.7 % there at `dec-top1` with
neither ridge nor mask, and its `fg_area_ratio` of 1.359 (baseline 1.185) is a property of the decoder, not of
the watershed. The ridge adds 2 points of loss on top (-14.1 %); the loss itself is in the field. So the gate
failure of point 7 is a training-recipe question (deepbacs' thin rods need the foreground calibrated, and
`boundary_fgcal` - which does calibrate it, 1.06 against 1.17 - scores *worse* there, -13.7 % against -7.7 % at
the defaults), not a tuning question.

**10. The threshold test of 4.7 point 1: the full boundary does not inflate the foreground.**
`boundary`'s own sweep optimum (`dec_boundary_sweep_dev.csv`, 1728 combinations) is 0.4273 at
`foreground_threshold` **0.5** (0.4 gives 0.4252, 0.6 gives 0.4246, 0.7 gives 0.4158), so the optimal threshold
runs `baseline` 0.4 -> **`boundary` 0.5** -> `contact` 0.6. The touching target pushed foreground mass outward;
the full boundary does so far more mildly, and the section 5.6 expectation of "0.5-0.6" lands at the benign end.
The optimum also confirms 4.7 point 4: `boundary` keeps the *production* density (10) and sigma (1.0) and only
lengthens the travel to 800, exactly like `baseline` and `contact`, whereas `fgcal` and `both` - the two that
changed the *foreground* loss - move to density 50 / sigma 0.5. The regime shift belongs to the foreground loss,
not to the fifth channel.

**11. Restating the headline honestly.** The +3.3 % of point 3 is measured against `baseline` at `dec-top1`
(0.4200), which is 1 % below baseline's own optimum (0.4244, 4.7) - the same overstatement that 4.7 caught for
`fgcal`. Against baseline's own optimum, `boundary` + ridge 1 at `dec-top1` is **+2.3 %**. The sweep cannot
settle this by itself because the cached scorer ignores the contact keywords, so `boundary`'s own optimum
(0.4273, +0.7 % over baseline's own optimum) is a *ridge-free* number and understates the decoder as much as
`dec-top1` overstates it. Screens of the missing cells were submitted at 06:42: `dec-base-top1` for `baseline`
(job dec_baseline_own_screen) and `dec-bnd-top1` / `dec-bnd-top1-ridge1` for `boundary`
(dec_boundary_own_screen), i.e. each decoder at its own sweep optimum, with and without the ridge.

**12. The corrected comparison: every decoder against `baseline` at ITS OWN optimum** (screens
`dec_baseline_own_screen` / `dec_boundary_own_screen`, `ais/reports/decoders_own_optimum_*`). `baseline` at
`dec-base-top1` scores **0.4244 dev / 0.4052 holdout**, against 0.4200 / 0.4025 at `dec-top1`. Recomputing every
candidate's best configuration against that reference:

| decoder (best configuration) | dev | vs baseline's own optimum | holdout | vs baseline's own optimum |
|---|---:|---:|---:|---:|
| `boundary` + ridge 1 @ `dec-top1` | 0.4340 | **+2.3 %** | 0.4085 | +0.8 % |
| `boundary_fgcal` + ridge 1 @ `dec-top1` | 0.4311 | +1.6 % | 0.4082 | +0.7 % |
| `fgcal` @ `dec-fgcal-top1` | 0.4298 | +1.3 % | 0.4094 | +1.0 % |
| `both` + ridge 1 @ `dec-top1` | 0.4271 | +0.6 % | **0.4113** | **+1.5 %** |
| `baseline` @ `dec-base-top1` | 0.4244 | - | 0.4052 | - |
| `contact` + ridge 1 @ `dec-top1` | 0.4155 | -2.1 % | 0.4044 | -0.2 % |

**This retracts the "10 of 11 datasets up" of point 3.** Against `baseline` at its own optimum, `boundary` +
ridge 1 has **7 of 11 up on dev** and 3 of 5 on the holdout, with four datasets down: deepbacs -10.2 %,
neurips -6.1 %, tissuenet -5.0 %, puma -0.3 % (up: deepseas +28.1 %, dic_hepg2 +26.5 %, tnbc +6.5 %, covid_if
+4.2 %, livecell +4.2 %, yeaz +3.3 %, dynamicnuclearnet +1.0 %). tissuenet alone swings 14 points
(+9.1 % -> -5.0 %) purely from the reference, because `baseline` at threshold 0.4 / density 10 / sigma 1.0 is far
better there than at `dec-top1`. The lesson of 4.7 therefore applies to round 2 in full: **a shared tuned
configuration flatters whichever decoder it was tuned on**, and the only defensible reference is each decoder at
its own optimum.

What survives the correction: the target change is worth **+4.4 points** over round 1 (`contact` -2.1 % ->
`boundary` +2.3 % on dev, both at their best configuration against the same reference), the head is confident
(point 1), and the collateral damage on the unseen datasets is repaired (covid_if +4.2 %, deepseas +28.1 %).
What does not: the dev gain is +2.3 % rather than +3.3 %, it does not confirm on the holdout (+0.8 %, where
round-1 `both` reaches +1.5 %), and four datasets are down rather than one. Under the user's rule - only
cross-dataset wins that hold on the holdout count - **the boundary channel is a real improvement over the
contact channel but still not a win over the plain fine-tune**, and no configuration of any of the six decoders
passes the gate.

**13. `boundary`'s own sweep optimum is not its best configuration once the ridge exists.**
`dec-bnd-top1` (threshold 0.5, density 10, sigma 1.0) scores 0.4273 / 0.4000 and with ridge 1 0.4269 / 0.4007,
against 0.4340 / 0.4085 for `dec-top1` + ridge 1 (density 50, sigma 0.5). The ridge and the seed regime
interact: the ridge pays off in the few-converged-seeds regime, and the sweep - which cannot evaluate the contact
keywords at all - therefore optimises into the wrong basin. dic_hepg2 shows it starkly: -8.6 % at
`dec-bnd-top1-ridge1` against +26.5 % at `dec-top1-ridge1`. **A ridge-blind sweep cannot tune a five-channel
decoder**; the grid needs `contact_weight` as a dimension, which requires teaching the cached scorer the contact
keywords.

**14. `boundary_fgcal`'s ridge and mask** (`decoders_boundary_fgcal_contact_*`): the ridge is worth at most
+0.13 % (dev, w0.5) and the mask +0.24 % (dev, t0.5) / +0.28 % (holdout) against its own defaults - an order of
magnitude less than for `boundary`, and higher ridge weights *hurt* (-0.35 % at w4). Its foreground is already
calibrated, so the seeds it would gain from a ridge are largely there; consistent with point 6.

**15. All six sweep optima, and what each parameter tracks** (`dec_<variant>_sweep_dev.csv`, 1728 combinations
each, cached scorer, reference = that decoder's own library defaults; `n_iter` 800, `dt` 0.5,
`foreground_weight` 0.75 and `min_size` 50 everywhere):

| decoder | own optimum (dev) | `foreground_threshold` | density / sigma | fifth channel | foreground loss |
|---|---:|---:|---|---|---|
| `baseline` | 0.4244 | **0.4** | 10 / 1.0 | - | Dice |
| `contact` | 0.4083 | **0.6** | 10 / 1.0 | touching | Dice |
| `boundary` | 0.4273 | **0.5** | 10 / 1.0 | full boundary | Dice |
| `fgcal` | **0.4298** | 0.5 | **50 / 0.5** | - | Dice + boundary BCE |
| `both` | 0.4254 | 0.5 | **50 / 0.5** | touching | Dice + boundary BCE |
| `boundary_fgcal` | 0.4261 | 0.5 | **50 / 0.5** | full boundary | Dice + boundary BCE |

The two parameters separate the two loss changes with no exceptions across six decoders:

- **`density_threshold` / `sigma` track the foreground loss alone.** All three decoders trained with the
  boundary-weighted foreground BCE want density 50 / sigma 0.5; all three without it want the production
  density 10 / sigma 1.0. The fifth channel has no influence. This settles 4.7 point 4: the move to the
  "few, converged seeds" regime is caused by the foreground loss, not by decoder fine-tuning and not by the
  extra channel.
- **`foreground_threshold` tracks the fifth channel's target.** No channel 0.4, touching boundaries 0.6, full
  boundaries 0.5 - i.e. the auxiliary task pushes foreground probability mass outward in proportion to how
  ill-posed it is, and the calibrated foreground loss pins the threshold at 0.5 whatever the channel does
  (`fgcal`, `both` and `boundary_fgcal` all 0.5).

Ranking at each decoder's own **ridge-free** optimum: `fgcal` 0.4298 > `boundary` 0.4273 > `boundary_fgcal`
0.4261 > `both` 0.4254 > `baseline` 0.4244 > `contact` 0.4083. So without the contact ridge the
boundary-weighted foreground loss is the best single change, and the fifth channel only overtakes it once the
ridge is available (point 12) - which the sweep cannot see (point 13).

**16. The 3d crops: the boundary channel does not repair the volume path, and its LM failure is the foreground**
(`ais/reports/decoders_all_3d*`, apg3d primary + holdout, 75 crops; regression instrument only - the decoders
were fine-tuned on 2d LM data and the 3d path saw none). Per family under `current-defaults`:

| family | production | baseline | fgcal | boundary | boundary_fgcal | contact | both |
|---|---:|---:|---:|---:|---:|---:|---:|
| celegans_atlas | 0.104 | 0.040 | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 |
| embedseg_platy_ish | 0.339 | 0.156 | 0.135 | 0.000 | 0.002 | 0.000 | 0.000 |
| embedseg_platy_nuclei | 0.259 | 0.115 | 0.086 | 0.000 | 0.000 | 0.000 | 0.000 |
| embedseg_skull | 0.118 | 0.238 | 0.078 | 0.000 | 0.009 | 0.000 | 0.000 |
| gonuclear | 0.256 | 0.132 | 0.135 | 0.000 | 0.004 | 0.000 | 0.000 |
| platynereis_nuclei | 0.068 | 0.052 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 |
| cremi / cremi_seen (lower better) | 0.99 / 0.59 | 1.87 / 1.23 | 2.24 / 2.15 | **1.86 / 1.42** | 1.97 / 2.04 | 2.19 / 2.05 | 2.09 / 1.89 |
| snemi / humanneurons (lower better) | 0.97 / 1.35 | 1.69 / 1.97 | 1.95 / 1.98 | **1.87** / 2.03 | 2.02 / 2.16 | 2.16 / 2.37 | 1.92 / 2.04 |

Every five-channel decoder scores exactly 0 on all six LM families, the boundary target included, so the
better-posed channel does **not** repair the volume path. But the mechanism is not the one recorded for round 1
in 4.0 (`boundary_magnitude_max` removing every instance): the mechanism columns show `boundary`'s 3d
**foreground ballooning** - `fg_area_ratio` 6.63 on celegans_atlas and **8.45** on gonuclear, against 2.04 / 3.59
for `baseline` and 1.28 / 2.02 for production - with 2.7 to 9.1 background seeds per ground-truth object and
`matched_iou` undefined because nothing matches at IoU 0.5 at all. Objects are not missing for want of seeds
(`gt_with_0_seeds` 0.29-0.39, no worse than baseline); the volume is simply flooded. `boundary` is the *worst*
of the six on this measure, i.e. the extra 2d task makes the untrained 3d foreground worse the better it is
learned in 2d.

Two things worth carrying to a joint 2d + 3d run:

- **`fgcal` is the only variant that improves the 3d foreground** (gonuclear `fg_area_ratio` 2.68 against
  baseline's 3.59, celegans 2.38 against 2.04 - and it is the only loss change that keeps an LM score at
  baseline level, gonuclear 0.135 against 0.132). The boundary-weighted foreground BCE generalises to the
  dimension it never saw; the fifth channel does the opposite.
- **On EM the boundary channel is harmless**: `boundary` matches `baseline` on cremi (1.86 against 1.87) and is
  the best of the six on cremi_seen (1.42) and snemi (1.87), while `fgcal` is the worst on cremi (2.24). The
  volume regression is specific to LM instance matching, not to volumes as such.
