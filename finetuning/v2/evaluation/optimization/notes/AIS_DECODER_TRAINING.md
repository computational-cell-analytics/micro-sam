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

(to be filled when the trainings have finished)
