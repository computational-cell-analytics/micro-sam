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

(to be filled when the trainings have finished)
