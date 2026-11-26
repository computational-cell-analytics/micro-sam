# Decoder training on frozen uSAM2 encoders

Training of a hybrid 2D/3D segmentation decoder for automatic instance segmentation on top of the image encoder
of a trained uSAM2 model. The encoder is frozen, only the decoder is updated.

- Decoder: Hiera U-Net decoder (`hiera_u`, design C of the decoder comparison on the branch `decoder-tests`, see
  `finetuning/v2/generalist/decoder_arch/OVERVIEW.md` there). It is implemented in `torch_em.model.hybrid_unetr`
  (torch-em branch `advanced-decoders`). The C3F2 decoder (`c3f2`) can be selected with `--decoder`.
- Library code: `micro_sam/v2/training/decoder_training.py` (`FrozenEncoderHybridUNETR`, `DecoderTrainer`,
  `train_decoder`), tests in `test/test_v2_decoder_training.py`.
- Environment: `micromamba activate new-stack`.
- Outputs: `/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training`
  (`data_check/`, `calibration/`, `checkpoints/<run>/`, `logs/`, `jobs/`). Override with `DECODER_SAVE_ROOT`.

## Set-up

The data and the settings are the ones of the automatic segmentation branch in the joint training of the cells
model (`finetuning/v2/generalist/train_joint.py`, `micro_sam.v2.training.train_joint_sam2`):

| Setting | Value |
|---|---|
| Data | `_build_joint_datasets(dataset_choice="all", distance_type="geodesic", with_boundaries=True)`, the datasets of the cells model; 50 validation samples per dataset |
| Targets | Geodesic directed distances with boundaries: foreground, three directed distances, boundary (the instance channel of the joint targets is not used) |
| Loss and metric | `DirectedDistanceLoss(mask_distances_in_bg=True, with_boundaries=True, boundary_dice_weight=0.5)` |
| Optimizer | AdamW, default weight decay (0.01) |
| Scheduler | `ReduceLROnPlateau(mode="min", factor=0.9, patience=10)` on the validation loss, per epoch |
| Precision | bfloat16 autocast |
| Epochs | 75 for hvit_t (73 hvit_s, 82 hvit_b, 79 hvit_l) |
| Gradient clipping, early stopping | none |

Differences to the joint training:

- Learning rate 1e-4 instead of 1e-5. The decoder is trained from scratch; 1e-4 is the decoder rate of the
  decoder comparison.
- The encoder comes from the trained model (`hvit_t_cells`) and is frozen: no gradients, evaluation mode, applied
  under `torch.no_grad()`.
- The projections of the stride-4 and stride-8 features (`decoder.proj_s4`, `decoder.proj_s8`) are initialized
  from `sam_mask_decoder.conv_s0/conv_s1` of the trained model and are trained with the decoder.
- One GPU. The batch size for 2D and the number of slices for 3D are set by the calibration.

The checkpoints (`checkpoints/<run>/{best,latest,epoch-*}.pt`) hold only the decoder weights (`model_state` and
`decoder_state`, both with the prefix `decoder.`), the `decoder_config` and the `encoder_model_type`.
Load them with `FrozenEncoderHybridUNETR.load_decoder_state(checkpoint["decoder_state"])`.

## Workflow

```bash
# 1. Data check and calibration on a H100 (one job).
python submit_decoder_training.py calibrate

# 2. Training with the calibrated settings, as a chain of 48 h jobs that resume from the latest checkpoint.
python submit_decoder_training.py train --batch_size_2d <B> --z_slice <Z> --n_jobs <N>
```

- `check_data.py` builds the loaders, counts datasets and samples per group, and draws samples from every
  dataset (shapes, value ranges, finite targets, foreground). For several numbers of slices it shows which ones all
  3D datasets support. Results: `data_check/`.
- `calibrate_decoder.py sweep` measures memory and time per training step for the candidate settings and
  selects the largest ones below 80 % of the GPU memory. `calibrate_decoder.py loop` runs training iterations and a
  validation pass with the loaders and extrapolates the duration of the training. Results: `calibration/`.
- `train_decoder.py` runs the training. `--resume` continues from the latest checkpoint up to the total number of
  epochs.
- All jobs go to `grete-h100:shared` with one H100 and run from a frozen copy of this folder.

## Calibration results and runs

Calibration on a H100 (93 GiB, 2026-11-16, job 17347836; files in `calibration/`). The data check for 14 slices
passed for all datasets (train: 87 2D datasets with 27,225 samples and 41 3D datasets with 13,594 samples;
validation: 5,159 samples).

| Setting | s / step | Peak reserved memory [GiB] |
|---|---|---|
| 2D, batch 14 | 0.23 | 24.0 |
| 2D, batch 24 | 0.38 | 41.1 |
| 2D, batch 32 | 0.52 | 54.6 |
| 2D, batch 40 | 0.65 | 68.3 |
| 2D, batch 48 | 0.80 | 82.0 |
| 2D, batch 64 | out of memory | |
| 3D, z = 8 | 0.15 | 14.7 |
| 3D, z = 14 | 0.26 | 25.5 |
| 3D, z = 16 | 0.29 | 29.1 |
| 3D, z = 24 | 0.43 | 43.5 |
| 3D, z = 32 | 0.59 | 57.8 |

- Memory is about 1.6 GiB per slice and the throughput is 55 - 60 slices per second for all settings, so larger
  batches do not shorten an epoch. The number of 3D samples per epoch does not depend on z, so the duration of the
  3D part of an epoch is proportional to z.
- Selected: 2D batch size 40 (largest below 80 % of the memory) and z = 14 (the value of the joint training and
  the only one covered by the data check), 3D batch size 1.
- Training loop with these settings (200 iterations and a validation pass): 14,275 batches per epoch (681 2D,
  13,594 3D), 1.93 h per epoch including 8.4 min validation, 145 h for 75 epochs. The 2D batches wait for the
  loader (3.5 s per batch, from 11 measured batches), the 3D batches do not.
- Note: the windowed attention of Hiera fails in bfloat16 for 64 or more images per encoder call
  ('CUDA error: invalid configuration argument'); the encoder wrapper encodes in chunks of 32 images.

Runs:

| Run | Settings | Jobs |
|---|---|---|
| `hiera_u_hvit_t_cells` | 2D batch 40, z = 14, lr 1e-4, 75 epochs | 17348080 - 17348083 (chain of four 48 h jobs, submitted 2026-11-16) |

Job 17348080 ran out of host memory (160 GB) on 2026-11-18 in the validation after its fourth epoch; the
checkpoints of epoch 3 are kept (validation loss 0.422, 0.374, 0.364). The datasets take about 14 GB in the main
process, which the forked loader workers copy step by step, and the memory of the persistent training workers
accumulated over the epochs. Changes: the training workers are restarted in every epoch (`common.build_loaders`),
the jobs request 250 GB, `MALLOC_TRIM_THRESHOLD_` is set as in the joint run, and the training prints the host
memory every 15 minutes (`[memory]` lines in the log). The pending jobs 17348081 - 17348083 were updated in place
(memory request, `common.py` and `train_decoder.py` of their code copy) and job 17390104 was appended to the chain.
