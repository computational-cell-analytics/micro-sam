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
| Scheduler | `ReduceLROnPlateau(mode="min", factor=0.9, patience=10)` on the validation loss, per epoch (`--scheduler_patience`) |
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
- All jobs go to `grete-h100:shared` by default (`--partition`, `--account` and `--env` select another queue,
  account and environment), with one H100, and run from a frozen copy of this folder. `--gpu` and `--constraint`
  select another GPU, e.g. `--partition kisski-h100,kisski --gpu 1 --constraint "inet&80gb_vram"` for the first
  free H100 or A100 with 80 GB.

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
| `c3f2_hvit_t_cells` | as `hiera_u_hvit_t_cells`, but scheduler patience 6 | 17497912, 17497913, 17497915, 17497923 (chain of four 48 h jobs on `grete-h100:shared`, submitted 2026-11-23) |
| `c3f2_hvit_t_cells` (KISSKI) | the same, in the output folder of the KISSKI account | 17486495 - 17486498 (chain of four 48 h jobs on `kisski-h100`, submitted 2026-11-23) |
| `c3f2_hvit_t_cells` (A100) | the same, on an A100 with 80 GB, in `decoder_training_a100` | 17511094, 17511096, 17511098, 17511100 (chain of four 48 h jobs on `grete:shared`, submitted 2026-11-24) |
| `c3f2_hvit_t_cells` (KISSKI, A100) | the same, on an A100 with 80 GB, in `decoder_training_kisski_a100` | 17501954 - 17501957 (chain of four 48 h jobs on `kisski`, submitted 2026-11-23) |

Job 17348080 ran out of host memory (160 GB) on 2026-11-18 in the validation after its fourth epoch; the
checkpoints of epoch 3 are kept (validation loss 0.422, 0.374, 0.364). The datasets take about 14 GB in the main
process, which the forked loader workers copy step by step, and the memory of the persistent training workers
accumulated over the epochs. Changes: the training workers are restarted in every epoch (`common.build_loaders`),
the jobs request 250 GB, `MALLOC_TRIM_THRESHOLD_` is set as in the joint run, and the training prints the host
memory every 15 minutes (`[memory]` lines in the log). The pending jobs 17348081 - 17348083 were updated in place
(memory request, `common.py` and `train_decoder.py` of their code copy) and job 17390104 was appended to the chain.

Job 17348082 failed on 2026-11-22 at its first optimizer step after resuming from epoch 26. The decoder in
torch-em was refactored on 2026-11-21 (the jobs import torch-em from the working tree), which changed the order
of the decoder parameters; the optimizer state is stored by position and no longer fit. The weights are not
affected: the checkpoint of epoch 25 gives the same predictions with the torch-em commit of the training start
(`c5e7087f`) and with the refactored one (maximal difference 6e-7). `DecoderTrainer` now assigns the optimizer
state by parameter name and stores the names in the checkpoints. Job 17479058 was appended to the chain.
Validation loss up to epoch 26: best 0.329 (epoch 25), 0.332 - 0.335 for epochs 17 - 26 otherwise.

The C3F2 run (the ConvNeXt-style decoder, 2.17 M parameters) uses the settings of the Hiera-U run, except for the
patience of `ReduceLROnPlateau` (`--scheduler_patience 6` instead of 10). It was submitted four times (2026-11-23 and
2026-11-24) with the same settings and to different output folders, because the wait for a H100 was not
foreseeable. The chains are independent trainings of the same run:

```bash
# grete-h100:shared, account nim00007, environment new-stack and the output folder above (the defaults).
python submit_decoder_training.py train --decoder c3f2 --batch_size_2d 40 --z_slice 14 --scheduler_patience 6 \
    --n_jobs 4

# The KISSKI H100 nodes from the other account, which cannot write to the output folder above. Outputs:
# /mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training_kisski
python submit_decoder_training.py train --decoder c3f2 --batch_size_2d 40 --z_slice 14 --scheduler_patience 6 \
    --n_jobs 4 --partition kisski-h100 --account kisski_usam2_pape --env main \
    --root /mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training_kisski

# A100 with 80 GB on grete:shared. Outputs: <root>_a100 (set by the script for '--gpu A100:1').
python submit_decoder_training.py train --decoder c3f2 --batch_size_2d 40 --z_slice 14 --scheduler_patience 6 \
    --n_jobs 4 --partition grete:shared --gpu A100:1 --constraint "inet&80gb_vram" --mem 160G
```

The KISSKI account has a chain on A100s as well (`--partition kisski --gpu A100:1 --constraint "inet&80gb_vram"`
and the arguments of its H100 chain). Notes on the A100 chains:

- Expected duration (not measured): 2.7 - 3.3 h per epoch instead of about 2.1 h, i.e. 200 - 250 h for 75 epochs,
  more than the four jobs of the chain cover (about 60 - 70 epochs). The chain was reduced from six jobs on purpose
  (jobs 17511101 and 17511102 were cancelled): the validation loss of Hiera-U was on a plateau from epoch 17 on, so
  the last epochs are not expected to matter; jobs can be appended with `--resume --after <job>`. The duration
  assumes training steps that are 1.5 - 2 times slower than on a H100; the waiting for the 2D loader and the
  validation (about 40 % of an epoch on the H100) do not scale with the GPU.
- Expected GPU memory for 2D batch 40: about 55 GiB of 80 GiB (80 % of the 68 GiB of Hiera-U), not measured.
- The `grete:shared` chain requests 160 GB instead of 250 GB of host memory: the Hiera-U jobs with restarted
  workers peaked at 99 GiB (`[memory]` lines of jobs 17348081 and 17348083), and the A100 nodes with a busy GPU had
  70 - 90 GB free on 2026-11-24, so that 250 GB only fits after two jobs on the same node ended.
- Once one chain trains, cancel the others or keep them as repetitions.

No separate calibration was run for it. Training steps on random data (local 10 GiB MIG slice, 2026-11-23) need
about 80 % of the memory of Hiera-U at the same time per step (2D batch 4: 6.7 vs. 8.4 GiB, 3D z = 4: 7.0 vs.
8.7 GiB), so batch 40 and z = 14 fit on a H100. The same check in the `new-stack` environment (training,
checkpoint, reloading and resuming with `c3f2`) passed with 6.5 and 6.9 GiB.

## Evaluation

`evaluate_decoder.py` scores a decoder checkpoint with automatic instance segmentation (AIS) on the datasets of the
automatic segmentation panels of Figures 2 and 3 of the paper (read from `PANEL_DATASETS` in
`scripts/plotting/v2/for_paper/plots_for_figure{2,3}.py`; Extended Data Figure 1 uses the same datasets) and on the
four supplement datasets that the parameter search tunes on (`SUPPLEMENT_DATASETS`), 85 datasets in total.

```bash
# Check: the first N samples of every dataset, on the local GPU.
python evaluate_decoder.py run --checkpoint <root>/checkpoints/hiera_u_hvit_t_cells/best.pt --n_samples 2 \
    --skip_failed -e <root>/evaluation/smoke
python evaluate_decoder.py report -e <root>/evaluation/smoke --n_samples 2

# Full evaluation: Slurm array over datasets and shards (one A100 per task by default), then the report.
python evaluate_decoder.py submit --checkpoint <root>/checkpoints/hiera_u_hvit_t_cells/best.pt [--dry-run]
python evaluate_decoder.py report -e <root>/evaluation/hiera_u_hvit_t_cells

# With the parameters of the search (next section): one array per setting, after its search job.
python evaluate_decoder.py submit --checkpoint <staged checkpoint> -e <root>/evaluation/<checkpoint> --data 3d \
    --params <root>/tuning/<checkpoint>/3d/best_parameters.json --dependency afterok:<search job> \
    --partition grete:preemptible --gres 1g.10gb:1 --cpus 4 --mem 32G --requeue
python evaluate_decoder.py report -e <root>/evaluation/<checkpoint>   # the tuned results if there are any
```

- The pipeline is the one of `finetuning/v2/evaluation/evaluate_automatic_segmentation.py` (`run_evaluation`): the
  same test samples and crops (`common.load_data`, `eval_crops.json`), padding, upscaling, tiling of large images
  (384 + 2 x 64 pixels), postprocessing (`postprocess_unisam2`) and metrics. The decoder model has the input and
  output convention of the UniSAM2 model, so it runs through the same code.
- The light microscopy volumes and densecell are scored without slivers since 2026-11-23, see `TUNING_DATA.md`.
  Results from before are not compared with results from after: `report` lists these datasets apart, and
  `reference summary` lists the reference results that still need `reference submit`.
- The only difference to the reference: volumes are predicted in blocks of 10 + 2 x 2 = 14 slices, the number of
  slices of the training (`--z_tile`, `--z_halo`; the reference uses 4 + 2 x 2). These blocks need about 9.1 GiB
  of GPU memory and `PYTORCH_ALLOC_CONF=expandable_segments:True` (set by the script).
- Reference: micro-sam2 hvit_t, i.e. `hvit_t_cells` with `hvit_t_cells_decoder` and the library defaults (no
  tuning), in `<root>/evaluation/reference_hvit_t/` (`results/`, `summary.csv`, `sources.json`). It is built with
  `evaluate_decoder.py reference`:
  - `copy` takes the results of the registry evaluation of the paper
    (`experiments/v6_cells_registry_evaluation{,_crops32,_isotropic}/results`, a later folder replaces the result of
    an earlier one, as for the figure tables), without the datasets whose test set-up changed since
    (`REFERENCE_RECOMPUTED`): `platynereis_cells` and `synapsenet_compartments` (now multicut and CREMI score) and
    `cartocell` (six test volumes moved into tuning). The volumes that are scored without slivers
    (`common.SLIVER_DATASETS`, 23 of the evaluated datasets) are not copied either.
  - `submit` recomputes the datasets that are not copied and have no result with the current scoring, with the
    command of the registry evaluation
    (`evaluate_automatic_segmentation.py -m hvit_t --mode ais --skip_tuning -c <hvit_t_cells_decoder>`) and the
    current code; `summary` rewrites the table and lists the results that were scored with slivers.
  - With `--n_samples`, the report compares on the same samples, using the per-sample rows of the reference.
- Postprocessing parameters: without `--params` the library defaults of `hvit_t_cells` (`model_type="hvit_t"`, see
  `DEFAULT_MODEL_TYPE`), which were tuned for the previous decoder (results `..._ais_default_...`). With `--params`
  the result of the parameter search for the checkpoint (results `..._ais_tuned_...`): the best parameters over the
  tuning datasets of the setting of a dataset (2d, 3d or dense), with the `min_size` of the dataset itself where
  it has tuning data. The out-of-domain datasets have none, so they take the `min_size` of the best parameters.
  The parameters must belong to the checkpoint (checksum), and a dataset without a session is an error.
- The checkpoint is staged first (`<experiment>/checkpoints/<run>_epoch<k>_<checksum>.pt`, decoder entries only),
  since the training overwrites `best.pt`. The checksum of the training checkpoint names the result files
  (`results/<dataset>_micro_sam2_hvit_t_ais_default[_n<N>]_ckpt-<checksum>.csv`). torch_em stores the training
  datasets in the checkpoints; they are not rebuilt when loading (`load_training_checkpoint`).
- Errors of a dataset (e.g. out of GPU memory) are listed in `<experiment>/failed.json` with `--skip_failed`.

Smoke test (2026-11-19, `best.pt` after 7 epochs, checksum `88cb7825...`, local MIG 1g.10gb): the first two samples
of all 81 datasets ran without errors (2D about 0.2 s per image, 3D 4 - 38 s per crop including loading and
postprocessing, 9.1 GiB
peak). Means over the panels, new decoder vs. reference on the same samples (`evaluation/smoke/report_n2.csv`;
two samples per dataset, so these numbers only show that the pipeline works):

| Panel | ID new / ref. | OOD new / ref. |
|---|---|---|
| 2D label-free cells (mSA) | 0.449 / 0.439 | 0.182 / 0.205 |
| 2D fluorescence cells (mSA) | 0.397 / 0.375 | 0.351 / 0.363 |
| 2D fluorescence nuclei (mSA) | 0.511 / 0.504 | 0.412 / 0.408 |
| 2D histopathology nuclei (mSA) | 0.436 / 0.434 | 0.266 / 0.250 |
| 3D fluorescence cells (mSA) | 0.447 / 0.471 | 0.319 / 0.285 |
| 3D fluorescence nuclei (mSA) | 0.356 / 0.359 | 0.291 / 0.253 |
| 3D EM neurites (CREMI, lower is better) | 1.087 / 1.102 | 1.001 / 0.963 |

## Post-processing parameter search for a decoder checkpoint

`tune_decoder.py` runs the search of `finetuning/v2/evaluation/parameter_search.py` (`search`) for a decoder
checkpoint, once per setting (`TUNING_DATA.md` lists the tuning data): `2d` (flow, 31 datasets), `3d` (flow, 14
datasets) and `dense` (multicut, 6 datasets). The run of the epoch 25 checkpoint below used the pools before
2026-11-23: 2d without `orgline` (30 datasets), dense with `liconn` and `xpress` (8 datasets).

```bash
python tune_decoder.py predict --checkpoint <staged checkpoint> --data 3d     # GPU, minutes
python tune_decoder.py search --checkpoint <staged checkpoint> --data 3d      # CPU
python tune_decoder.py submit --checkpoint <checkpoint> [--skip_predict] [--dry-run]   # all settings as Slurm jobs
```

- Stages of a search, each on all tuning datasets of the setting; the best combination has the best mean relative
  score over the datasets:
  1. the broad grid (`BROAD_GRIDS_2D`, `BROAD_GRIDS_3D`), without boundary terms;
  2. up to two rounds one step past the grid edges that the best combination sits on;
  3. the boundary stage (`BOUNDARY_GRID`). Sparse: the 20 best combinations with `boundary_weight` (the predicted
     boundary as a ridge on the watershed height map) in {0, 0.25, 0.5, 1} and `boundary_mask_threshold` (the first
     watershed stays off the boundary pixels) in {off, 0.3, 0.5, 0.7}. Dense: the 10 best oversegmentation settings
     with `boundary_weight` in {0.25, 0.5, 0.75, 1} and every `beta` swept so far. For the multicut the weight
     mixes the predicted boundary into the boundary map, which was the inverted foreground only
     (`micro_sam.v2.postprocessing.multicut_boundary_map`; weight 0, the default, is the earlier behaviour);
  4. sparse only: the winner with every `min_size` of `MIN_SIZE_VALUES`. Each dataset takes the value that scores
     best on its tuning data (`min_size_per_dataset`); it keeps the value of the winner unless another one is better.
- `predict` stores the decoder outputs of the tuning samples (`<session>/predictions/`), with the z blocks of 14
  slices and the padding of the small TissueNet images of the test evaluation. `search` scores from them and needs
  no GPU. It reuses the queue of `parameter_search.SlurmRunner` (sharded sweeps, claims, resume) with worker slots
  inside the job. Finished sweeps are kept: a search that is started again continues with the open tasks
  (`--retry_failed` also runs failed ones again).
- Output: `<root>/tuning/<staged checkpoint>/<setting>/` with `best_parameters.json` (checksum, best parameters,
  `min_size_per_dataset`, summary with the scores at the global and the per-dataset `min_size`),
  `selection/<setting>_ais/hvit_t.csv` (the 50 best combinations) and `tuning/<setting>/<stage>/` (all sweeps, with
  the per-sample scores).
- `search --check` uses small grids to check the pipeline; the evaluation refuses such a result for a full run.

Run for `best.pt` after 25 epochs (2026-11-21; staged as `hiera_u_hvit_t_cells_epoch25_1b75589f`, checksum
`1b75589f51a2aa4cfd7762e794a18dfb`, validation loss 0.329):

| Step | Setting | Job | State (2026-11-22 20:15) |
|---|---|---|---|
| Predictions | 2d, 3d, dense | interactive node, no Slurm job | done: 3074 images with labels (of 3094), 66 tiles, 21 crops |
| Search | 2d | 17449319 (`grete:preemptible`, 40 CPU threads) + 4 workers 17450248 (`standard96s:shared`) | done (22 h 51 min) |
| Search | dense | 17449337 (`grete:preemptible`, 24 CPU threads) | done (1 h 54 min) |
| Search | 3d | 17450284 (`grete:preemptible`, 8 CPU threads) | time limit after 24 h, 86 of 238 tasks of the first stage |
| Search | 3d | 17474388 (`grete:preemptible`, 40 CPU threads; preempted 14:52) + worker 17474394 (24 CPU threads), then the driver on the interactive node | done 2026-11-22 18:01 |
| Workers | 3d | 17450326 (10 x 16 CPU threads), 17459478 (20 x 8), `standard96s:shared` | never started (fair share), cancelled |
| Search | dense, neurite datasets only | 17474640 (session `dense_neurites`) | done (22 min) |
| Test | 2d: 48 datasets | 17449378 (array of 82) | done |
| Test | dense: 14 datasets | 17449390 (array of 36) | done |
| Test | 3d: 23 datasets | 17450263 (array of 137, dependency removed) | released 2026-11-22 20:00 |
| Test | dense, neurite-only parameters: 12 datasets | 17474645 (array of 33), folder `..._dense_neurites` | done |

- Folders: `<root>/tuning/hiera_u_hvit_t_cells_epoch25_1b75589f/{2d,3d,dense}/` and
  `<root>/evaluation/hiera_u_hvit_t_cells_epoch25_1b75589f/`. The test arrays run on `grete:preemptible`
  (10 GB slice, 4 CPUs, 32 GB, `--requeue`, 8 tasks at once).
- Queue (2026-11-21): `grete:shared` (A100) had a start estimate of 4-5 days, also with the 2 h QoS; a 10 GB slice
  on `grete:preemptible` started at once (`sbatch --test-only` is wrong for this partition, and it replaces the
  QoS). A GPU request is mandatory there, so the 2d and dense searches hold one slice they do not use. They were
  first queued on `standard96s:shared` (17440356, 17440359; cancelled); the 3d search needs a full CPU node.
- Small jobs instead of one node: `tune_decoder.py worker -e <session> --threads 16 --slots 8` takes tasks from the
  queue of a session that a `search` job drives (scripts `jobs/code_workers_1009/worker_<setting>.sbatch`). It waits
  between the stages and ends with the result, or after 30 minutes without an open task. Measured memory: about
  1 GB (2d) and 2 GB (dense) per task that runs.
- A search job that fails or runs into its time limit leaves its test array pending with an unmet dependency.
  Then: submit the search again (`tune_decoder.py submit --checkpoint <staged> --skip_predict --data <setting>`),
  cancel the array and submit it again with the new job id.
- Dense search without `platynereis_cells` and `synapsenet_compartments` (2026-11-22): the winner only changes in
  `density_threshold` (3.0 instead of 2.0) and `n_iter` (50 instead of 25). Test means (CREMI score, lower is
  better): neurites ID 0.859 (all 8 tuning datasets: 0.858, reference 1.080), OOD 1.053 (1.055, reference 0.964).
  The two cell datasets do not explain the OOD difference to the reference.
- On `grete:preemptible` interactive jobs can hold all 10 GB slices for hours (seen on a Sunday afternoon); jobs
  then wait and running ones are preempted. Do not make a test array depend on a job that may not start.
- The reference results of `platynereis_cells`, `synapsenet_compartments` and `cartocell` were recomputed on the
  interactive node (`reference_hvit_t/logs/recompute_*_local.log`); `evaluate_decoder.py reference summary` rewrites
  the table.
- Checks before the submission (small grids, 1-3 samples of two datasets per setting, scratch folder): prediction,
  all stages of the search, the test run with the resulting parameters and the report ran for all three settings.

### Search and test of a checkpoint as a few jobs

`submit_checkpoint_evaluation.py` replaces the job arrays above (255 test tasks of 2-5 minutes for the epoch 25
checkpoint) by one job per setting that runs the whole pipeline on one 10 GB slice of `grete:preemptible`:
predictions of the tuning data, search on the CPUs of the job, test with the parameters it found. The 2d and 3d
settings have a second job that takes tasks of the same search (`tune_decoder.py worker`) and scores the other half
of the test samples. A fifth job computes the reference results that have no result with the current scoring.

```bash
python submit_checkpoint_evaluation.py --checkpoint <root>/checkpoints/hiera_u_hvit_t_cells/best.pt [--dry-run]
python evaluate_decoder.py reference summary
python evaluate_decoder.py report -e <root>/evaluation/<staged checkpoint>
```

- No job waits for the result of another one; the second 3d job only waits for the start of the first
  (`--dependency=after`) and ends after 3 h without an open task. All jobs have `--requeue` and continue after a
  preemption: prediction markers, finished sweeps and the per-sample result rows (`--n_shards`) are kept, and a
  search with a result is not started again.
- Sizes (`JOBS`): 2d 2 x 32 CPUs / 48 GB, 3d 2 x 32 CPUs / 64 GB, dense 16 CPUs / 40 GB, reference 4 CPUs /
  32 GB. Measured for the epoch 29 checkpoint: 3d search 970 core hours (19.7 h with the two jobs, 42 GB peak),
  3d test 2 h per job, dense 5.7 h in total, reference 7.2 h. The 2d search needs 1200 - 1400 core hours (the
  epoch 25 search: 591 in the driver and 568 in its four worker jobs), too much for one job of 32 CPUs in 48 h.

Run for `best.pt` after 29 epochs (2026-11-23; staged as `hiera_u_hvit_t_cells_epoch29_9a812c61`, checksum
`9a812c61a8bc6bd2859c46dccb157249`, validation loss 0.325), the first run with the tuning pools, the 105 3D crops
and the scoring without slivers of 2026-11-23:

| Job | Steps | Slurm job |
|---|---|---|
| `deceval_2d_epoch29_9a812c61` | predict, search, test of 48 datasets | 17502288 |
| `deceval_3d_epoch29_9a812c61_0` | predict, search, test of 23 datasets | 17502291 |
| `deceval_3d_epoch29_9a812c61_1` | tasks of the 3d search, test of 23 datasets | 17502292 |
| `deceval_dense_epoch29_9a812c61` | predict, search, test of 15 datasets | 17502293 |
| `decref_hvit_t` | reference (hvit_t, defaults) for the 23 volumes scored without slivers | 17502296 |
| `deceval_2d_epoch29_9a812c61_worker` | tasks of the 2d search, test of the 48 datasets in reverse order | 17533378 |

State 2026-11-24 21:30: 3d, dense and reference done (no failed dataset); the 2d search had 283 of the 527 tasks
of its first stage after 23 h, so 17533378 was added (script in the code folder of the run). The recomputed
reference reproduces the earlier scores with slivers within 0.001 mSA (densecell, one volume: 0.327 against 0.335).

Logs: `<root>/evaluation/hiera_u_hvit_t_cells_epoch29_9a812c61/logs/slurm/` and
`<root>/evaluation/reference_hvit_t/logs/slurm/`. Before the recomputed reference is used, compare its column
`mSA_with_slivers` with the mSA of the earlier results (`TUNING_DATA.md`, "Next steps").

State 2026-11-25 23:55: the 2d search stopped on 2026-11-24 21:43. The jobs imported `micro_sam` from the working
tree, where another branch was checked out at 21:34, so every task started after that failed on an import (19
failed tasks, the tasks that were running finished). The two jobs then held their CPUs without work until their
time limits (17502288 failed 2026-11-25 22:04, 17533378 cancelled). 316 of the 527 tasks of the first stage are
kept. Changes:
- `evaluate_decoder.freeze_code` copies the packages `micro_sam` and `torch_em` (`FROZEN_PACKAGES`) into the code
  folder of a run, and the job scripts put it first in `PYTHONPATH`.
- A search worker ends when every open task waits for the cache of a failed task
  (`parameter_search.worker_loop`), and `tune_decoder.py worker` ends after `--max_idle` without a started or
  closed task. Before, both waited for the time limit of their job.
- 2d submitted again with `--data 2d --skip_reference`: 17569447 (predictions are stored, search continues, test
  half 0) and 17569448 (tasks of the search, test half 1), code folder `jobs/code_20261125_234801`. The queue of
  the first run was moved aside before (`2d/queue_before_20261125_2355`): the tasks are keyed by their command,
  which holds the path of the code folder, so a search that continues from another code folder would add every
  open task a second time. With a new queue the 297 stored shards are read (at most 92 s per task) and the
  others run once.

State 2026-11-26 15:45: the run is complete. 17569447 and 17569448 finished at 12:22 (12.5 h; search result at
11:52, the last `orgline` shards took about 8 h each), no failed dataset. `report_tuned.csv` holds all 86 datasets,
each with a result of the checkpoint and a reference result with the same scoring.

## Post-processing parameter search: offline analysis

`analyze_parameter_search.py` (`cache`, `run`, `report`) analyses the existing sweeps without predicting anything: the
v6 global search of hvit_t (30 2D / 15 3D validation datasets, production decoder) and round 0 of the decoder-tests
tuning (runs A0 - E0; 3 2D datasets, 3 3D groups). It compares selection rules by their transfer to held-out
datasets, held-out samples and the other training seed, and reports sensitivity, flatness and per-dataset optima.
Outputs: `<root>/param_analysis/` (`report.md`, `transfer.csv`, `sensitivity.csv`, `flatness.csv`,
`per_dataset_optima.csv`). Note: in the v6 sources the library defaults are the winner of the full search, so their
"defaults" rows are not a held-out comparison.

## Post-processing parameter search: 3D tuning set

`TUNING_DATA.md` lists the current tuning data per setting and the test data with its ID / OOD panels.

The v6 search tuned the 3D parameters on one 32x512x512 center crop per validation volume (15 datasets, 68 crops).
`tune_set.py` builds a larger tuning set from the same held-out data (user decisions, 2026-11-19):

- Only in-domain datasets (`parameter_search.IN_DOMAIN_CELLS["3d"]`) plus `embedseg_platy_ish`; OOD data stays clean.
- Up to 10 tiles of 32x512x512 per dataset, round-robin over the volumes, most objects first (`write`). A last
  edge-aligned tile is added when it covers at least half a tile of new data (`tile_starts(edge_aligned=True)`).
- cartocell's validation data is two tiny volumes, so 6 of its 60 test volumes tune instead
  (`common.CARTOCELL_TUNING_TEST_VOLUMES`). Reference results written before the move are matched per sample
  (`common.legacy_test_sample_ids`, used by `evaluate_decoder.py report`).
- The search scores inside `valid_roi` like the test evaluation (before, predictions in the unannotated tissue of
  plantseg_root and platynereis_cells counted as false positives).
- platynereis_cells moved to the dense pool (user, 2026-11-20): it is segmented with the multicut and ranked by the
  CREMI score in tuning (`parameter_search.IN_DOMAIN_CELLS["dense"]`) and in the test evaluation
  (`common.DATASETS_DENSE`). Its labels hold many small objects that are no cells (ids of a few hundred voxels on
  one or two sections, specks split off the cells), which mSA counts like cells. Label zero and the neuropil label
  are not scored (`common.CREMI_IGNORE_ZERO_LABEL`). The tuning volumes 7-8 are read inside the bounding box of
  their labels (`common.PLATYNEREIS_CELLS_VAL_ROIS`; half of each volume is an unlabelled halo that shows cells)
  and tune on the 6 tiles of these boxes (`tuning_crops.json`). The test crops already tile the labelled box of
  volume 9, which holds no zero label. The sparse scores of the earlier tiles: `tune_set/superseded_20261120*/`.
- synapsenet_compartments moved to the dense pool the same way (user, 2026-11-20): multicut and CREMI score in
  tuning and test, label zero (unannotated parts of the tomograms, 2-46 % of a tile) not scored. More than half of
  its objects are fragments below 1000 voxels next to compartments of 1e5-1e6 voxels. It tunes on one tile per
  tomogram, the whole tuning region (`tuning_crops.json`).
- cartocell is ranked and compared by SA50 instead of mSA (user, 2026-11-21; `common.DATASETS_SA50`, criterion
  `sa50` in `parameter_search.py`, column `SA50` in `evaluate_decoder.py report`, which lists it apart from the mSA
  means). The cells are thin (about 12 x 17 x 17 voxels), so shrinking the ground truth by one voxel already scores
  mSA 0.01. It tunes on the 9 tiles (>= 16 sections, >= 10 cells) of the 6 moved test volumes
  (`tuning_crops.json`); the 2 volumes of the validation folder are of the `train_M1` acquisition, which the test
  split does not hold, and get no tile.
- The tiles of the low-scoring datasets (v6 best mSA < 0.3) are scored with Hiera-U and hvit_t_cells on a compact
  grid (`score` / `submit`, `report`) before the tiles are chosen.
- plantseg_root, nucverse3d, gonuclear, celegans_atlas and embedseg_platy_nuclei tune on tiles that are placed
  like their test crops (user, 2026-11-23;
  `tune_set.TEST_LIKE`): `candidates` adds the tiles centered in the tuning region (`tile_starts(centered=True)`,
  the placement of `eval_crops.json`), and `write` takes up to 10 of them with `parameter_search.spread_tiles`
  (without the tiles below 20 % of the median object count; every volume in proportion to its tiles; within a
  volume at evenly spaced ranks of the object count). Their center crops did not represent the test crops, see
  `TUNING_DATA.md`. nis3d and densecell take all edge-aligned tiles of their slab (`write`, the default choice).
  The 3D tuning set has 105 crops; `TUNING_DATA.md` lists them and the steps for running the search and the
  evaluations again.
- `tune_decoder.py` stores the tuning crops with the predictions of a dataset and stops if they are not the
  current ones (the predictions are stored by sample index).

Subcommands: `candidates` (tiles + label statistics, CPU), `score` / `submit` (GPU), `report`, `write [--picks]`.
Outputs: `<root>/param_analysis/tune_set/` (`candidates.csv`, `scores/<model>/`, `predictions/<model>/` as float16,
`report.md`).

`select_tune_tiles.py` shows the scored tiles of the low-scoring datasets in napari for a manual choice: tiles with
at least 16 slices and 10 objects (no object filter for synapsenet_compartments), without the center crops, per
dataset sorted by the best Hiera-U score. "Include" / "Exclude" (keys I / E, B takes the last decision back) are
stored in `<root>/param_analysis/tune_set/tile_selection.csv`; a restart continues with the first undecided tile.
`--list` prints the tiles and decisions without napari.
