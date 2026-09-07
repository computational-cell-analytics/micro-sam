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

(sweep rankings of baseline and contact, the 3D tables of all four and the unattended finalisation outputs are
appended below when they land)

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
