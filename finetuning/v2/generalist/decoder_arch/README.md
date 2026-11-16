# Decoder architecture evaluation

Plan: `~/.claude/plans/decoder-architecture-evaluation.md`. Experiment root:
`/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_arch`.
Overview of the implementation and the current results: `OVERVIEW.md`.

## Arms

| Arm | `decoder_type` | Decoder | Decoder params |
|---|---|---|---|
| A | `unetr` | current UniSAM2 decoder (`initial_features=32`) | 6.40 M |
| B | `unetr_skip` | A with `deconv1`-`deconv4` replaced by the real stride-8 / stride-4 features and an image stem | 5.58 M |
| C | `hiera_u` | Hiera-U: skip U-decoder, 2D convs on folded slices, per-slice GroupNorm, zero-init z branches | 1.56 M |
| D | `cagd_3d` | CAGD-3D: MedCAGD's decoder (gated skips, context aggregators, refinement, deep supervision), hybrid recipe | 8.51 M |
| E | `c3f2` | C3F2: ConvNeXt blocks with z attention at stride 16, z convs at 8 / 4, 2D upsampling head | 2.15 M |

A-C are step 1, D and E step 2 (same N = 31,800 iterations, manifest, schedule and evaluation).

The decoders live in `micro_sam/v2/models/decoders.py`; `UniSAM2(decoder_type=...)` builds them, checkpoints record
`decoder_config`, and `get_unisam2_model` rebuilds the decoder from it.

## Workflow

```bash
python build_manifest.py --workers 14            # CPU job: manifests/manifest_v1.json + leakage report
python submit_decoder_arch.py calibrate --arms A B C
python summarize_calibration.py                  # calibration/summary.json: the iteration count N
python submit_decoder_arch.py train --arm A --seed 0 --iterations N
```

Every job runs from a frozen copy of this folder under `<root>/jobs/code_<timestamp>`, together with the harness
files the evaluation imports (`finetuning/v2/evaluation/common.py`, `eval_crops.json`, `tuning_crops.json`).

## Evaluation (plan section 6)

```bash
python evaluate_decoder_arch.py place-val-crops        # once: evaluation/val_crops_v1.json (3D tuning crops)
python submit_decoder_arch.py evaluate --run armA_current_seed0 [--after <training job>]
python report_decoder_arch.py                          # evaluation/report/report.md + CSV tables
```

`evaluate` chains three jobs (`afterok`), all resumable:

| Stage | Where | What |
|---|---|---|
| `predict-val` | A100 | decoder outputs of the validation data -> `evaluation/<run>/val_cache/` (float16) |
| `tune` | CPU slice, 16 workers, 2 h (QoS `2h`) | flow post-processing sweep, one set for 2D and one for 3D -> `tuning/best_params.json` |
| `test` | A100 + 14 CPU workers, 2 h (QoS `2h`) | 15 test sets (12 reported) with defaults and tuned parameters -> `test/<dataset>.csv` |

- **Validation data.** 2D: the manifest's validation images (LIVECell 120, CellPose 80, TissueNet 120), loaded
  like the harness loads test images (8-bit range, 512 center crop, severed-object filter). 3D: 18 crops of
  32 x 512 x 512 (the harness test crop shape), at most 3 per validation volume or region, the tiles with the most
  objects, overlapping by at most 10 %.
- **Selection.** The combination with the best mean relative score (a group's mSA over its best mSA). 2D groups:
  the three datasets; 3D groups: NIS3D, PlantSeg root, EmbedSeg (mean of its four collections). The sweep is a grid
  (`eval_lib.GRIDS`, 4800 combinations in 2D, 3840 in 3D, containing the library defaults) plus up to three
  rounds that step past every grid edge the winner sits on.
- **Scorer.** `eval_lib.score_grid` shares the flow density and the watershed between combinations; it reproduces
  `flow_instance_segmentation` exactly (tested) and, unlike the harness sweep, applies the ignore mask (PlantSeg).
- **Test scoring** reproduces the harness (`common.evaluate_samples` + `postprocess_unisam2`) exactly; checked on
  LIVECell, DSB and PlantSeg root. Per sample: mSA / SA50 / SA75 at both parameter sets, foreground IoU and boundary
  recall of the raw outputs, merged / split / missed object shares (`eval_lib.object_errors`), and the decoder's
  time and peak memory for Z = 1 and Z = 8 at 1024 px.
- **Report.** Headline = mean over the 12 reported datasets at each run's own tuned parameters; paired bootstrap
  (2000 resamples within each dataset) against `armA_current_seed0`; seed spread from `armA_current_seed1`; the
  decision rule of the plan (a dataset 'loses' if its relative change is below -5 % with a 95 % interval below 0).

Checks (no SLURM): `python -m pytest -o addopts="" test_eval_lib.py`; any stage with `--eval-root <scratch>
--limit N` (and `tune --grid small`); `report_decoder_arch.py --allow-limited`.

## Deviations from the plan and findings during implementation

- **PlantSeg root leak.** `root_train/Movie2_T00010_crop_gt.h5` holds the same raw volume as
  `root_test/Movie2_T00010_crop_gt.h5` (the files differ only in metadata bytes). The manifest builder drops train
  volumes that share a z-slice with a test volume, so root trains on 20 volumes.
- **TissueNet stratification.** `tissuenet_v1.1.zip` has no `tissue_list` / `platform_list`; its `meta.npy`
  lists the experiment (tissue x platform x lab, 19 groups), which is used as the stratum.
- **LR schedule.** The warm-up (500 iterations) and the cosine decay to zero are applied to both parameter groups
  (encoder peak 1e-5, decoder peak 1e-4).
- **SAM2 stride-4 / stride-8 projections** (`conv_s0` / `conv_s1`, initialised from SAM2.1) train with the encoder
  learning rate.
- **Default decoder quirk.** `torch_em`'s `Decoder._crop` also crops the channel axis: of the synthesised pyramid
  (`deconv1`-`deconv3`, 256 / 128 / 64 channels) only the middle 128 / 64 / 32 channels reach the decoder. Arm B
  feeds these widths directly.
- **Calibration covers step 1 only** (A, B, C). If a step-2 arm is slower, it keeps the same N and gets a longer
  wall time instead of changing N.
- **Evaluation deviations.** GoNuclear is scored as the harness scores it: test volume 1170, 18 crops (the plan said
  all five volumes; the harness keeps 1139 for tuning and has crops only for 1170). The Usiigaci test set is all 87
  labelled images (45 + 5 Mask R-CNN sets and the 37 T98G time points of one field of view, so its bootstrap
  interval is optimistic); like CellPose it has no harness resolver and is resolved in `eval_lib`. CellPose test
  images get the training channel layout (`_prepare_cellpose_raw`). The tiled-vs-untiled check is for step 2.
- **Validation predictions are cached as float16**; the tuned parameters are then applied to float32 test outputs.
- **Tuning wall time.** The `tune` stage requests 4 h without the `2h` QoS (step-1 tuning took 1.6-2.5 h and timed
  out twice at 2 h).

## Step 2: CAGD-3D and C3F2

Both decoders are in `micro_sam/v2/models/decoders.py` (`CAGD3DDecoder`, `C3F2Decoder`), built from the step-1
prototypes and, for CAGD-3D, from the published MedCAGD code (github.com/saadwazir/MedCAGD, `model.py`, `train.py`,
`cfgs.py`). Like Hiera-U they run 2D convolutions on the slices folded into the batch, normalise per slice, pool
globally per slice, and get their z context from zero-initialised residual branches (`ZResidual`: depth-wise z
convolution; `ZAttention`: attention along z with a relative position bias). At initialisation a volume decodes
exactly to its stacked slices (tested in float64).

CAGD-3D, kept from MedCAGD: pyramid widths 64 / 128 / 320 / 512, bottleneck (conv + global context), decoder blocks
with transposed-conv upsampling, SCA-Gate on skips wider than 64 channels (strides 16 and 8), context aggregators
at strides 16 / 8 / 4, refinement blocks (spatial focus at strides 8 / 4 / 2), ECA with learned temperature, MedCAGD
group counts, deep supervision on the stride 16 / 8 / 4 stages with MedCAGD's weights 0.4 / 0.3 / 0.2 (deepest
first) on predictions upsampled to full resolution. Changed:
- SAM2 has no stride-32 level: it is a strided 3x3 convolution of the stride-16 features.
- Bottleneck: attention along z; every conv block: a zero-initialised z branch. The gate has no z branch.
- The gate's spatial competition (softmax over H x W, numerically zero at 1024 px) is rescaled by H x W and clamped
  at 4.
- The last stage runs at stride 2 with the stride-2 image stem as skip (MedCAGD: full resolution, 1.7x the FLOPs),
  then a transposed convolution to full resolution fused with the stride-1 stem.
- The deep supervision heads predict all five channels with the production loss; the boundary channel takes the
  role of MedCAGD's separate edge heads (no extra edge heads). The plan said "downsampled targets"; MedCAGD's code
  upsamples the predictions instead, which is what is implemented.
- ECA: MedCAGD's three pooling scales all reduce to the global mean, so the three averaged convolutions are one.
- Global-context width at least 4 channels (MedCAGD: 1; identical at the default widths).

Training: the loss of arm D is main loss + 0.4 / 0.3 / 0.2 x the auxiliary losses (`DecoderArchTrainer.aux_weights`,
`UniSAM2.auxiliary_outputs`, only produced in training mode). The validation loss (checkpoint selection) is the
main output alone, as for all other arms; the TensorBoard train loss of arm D includes the auxiliary terms.

Calibration: `submit_decoder_arch.py calibrate --arms D E`; `summarize_calibration.py` keeps N from the step-1 arms
and lists the expected hours of every arm, from which the wall time of D and E is set.
