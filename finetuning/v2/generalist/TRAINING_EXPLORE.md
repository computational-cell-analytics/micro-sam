# Joint SAM2 training: geodesic hybrid vs directed distances

Handoff written 2026-08-25. Work was done on `ggpu236` (one full H100, grete-h100) and stopped
part-way so the trainings can be prepared on different hardware. Read "State of the work" for what
is done, and "Instructions for the next agent" for what to pick up.

## The experiment

Two full-scale joint generalist trainings, identical in every respect except the regression target
of the automatic branch:

| | target | flow behaviour |
|---|---|---|
| **geodesic** | `GeodesicHybridDistanceTransform` | direction = gradient of the geodesic field from the object's interior center, magnitude = geodesic distance to the boundary. Converges to **one sink per object**, whatever the shape. |
| **directed** | `DirectedPerObjectBoundaryDistanceTransform` | euclidean vector to the nearest boundary. Converges onto a **medial axis**, so elongated objects over-segment. |

Both start from `hvit_t` SAM 2.1, train interactive (SAM2Train) and automatic (UniSAM2) jointly on
a shared image encoder, on all data including the new histopathology datasets.

Why it matters: the geodesic target has so far only been validated on ground truth fields. This
asks whether the advantage survives when a network has to *learn* the field, at generalist scale,
with the interactive branch competing for the same encoder. It is the direct follow-up to the
LIVECell over-segmentation ceiling, where grid tuning and thresholding were already exhausted.

## Fixed configuration

- model: `hvit_t` (SAM 2.1 - `checkpoint_path=None` resolves to `configs/sam2.1/sam2.1_hiera_t.yaml`
  and `sam2.1_hiera_tiny.pt` from the 092824 release, so nothing extra is needed for "SAM2.1")
- `dataset_choice=all` (light microscopy + electron microscopy + histopathology)
- `initial_features=32` (decoder bottleneck matches the hvit_t embed_dim of 256)
- `n_epochs=100`
- 8 GPUs total
- pinned per-model knobs from `CHOSEN_PARAMETERS["hvit_t"]`: `batch_size_2d=10`, `z_slices=[10]`,
  `max_num_objects=5`

Launch:

```bash
python train_joint.py --model_type hvit_t --dataset_choice all --n_epochs 100 --distance_type geodesic
python train_joint.py --model_type hvit_t --dataset_choice all --n_epochs 100 --distance_type directed
```

The run name carries the distance type (`joint_sam2_hvit_t_{geodesic,directed}_multi_gpu`), so both
runs can share one `SAVE_ROOT` without colliding. `SAVE_ROOT` is read from the environment and
defaults to `/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/joint/v4`.

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It recovers roughly 14 GiB of allocator
fragmentation on the joint model and has never hurt.

## Code changes made (all committed to the working tree, nothing staged or pushed)

The geodesic arm was **not runnable before this work**. Commit `fe252142` defaulted the
*automatic-only* path (`get_dataloaders`, `_build_automatic_datasets`) to the geodesic transform,
but `_build_joint_datasets` still hardcoded `_JointLabelTransform`, which is euclidean. Joint plus
geodesic simply did not exist.

1. `micro_sam/v2/transforms/labels.py` - added `_JointGeodesicLabelTransform`, the
   `GeodesicHybridDistanceTransform` counterpart of `_JointLabelTransform`. Defaults to
   `instances=True`, so it produces the same 5-channel `[instance_ids, fg, d_x, d_y, d_z]` layout
   the joint trainer expects.
2. `micro_sam/v2/datasets/generalist_loader.py` - `_build_joint_datasets` takes
   `distance_type="geodesic" | "directed"` and picks the transform, with a `ValueError` on anything
   else.
3. `micro_sam/v2/training/training.py` - `distance_type` threaded through `train_joint_sam2`,
   `_train_joint_rank` and `train_joint_sam2_multi_gpu`, including docstrings.
4. `finetuning/v2/generalist/train_joint.py` - `--distance_type` argument, explicit
   `initial_features=32`, run name carries the distance type, `SAVE_ROOT` default bumped to `v4`,
   and the multi-GPU branch now keys off `"RANK" in os.environ` instead of
   `torch.cuda.device_count()`.
5. `finetuning/v2/generalist/train_joint_{multi,single}_node.sh` - updated, but see the warning
   below; the real submission scripts live elsewhere.

Total library diff is 48 lines. `flake8 --max-line-length=120` is clean on every file touched.

### One real bug found in the sbatch scripts

`train_joint_multi_node.sh` had `#SBATCH -G H100:4` together with `--nodes=2`. In Slurm, `-G` is
the **total** GPU count for the job, not per node, so that allocation gives 2 GPUs per node while
`torchrun --nproc_per_node=4` asks for 4. The fix is `--gpus-per-node=H100:4`. **If the real
submission scripts carry the same line, this is the single most important thing to port over.**

## Measured numbers

All measured on one full H100 (93.5 GiB) with the real `torchrun` and DDP code path, not a smoke
test.

### Dataset sizing after the histopathology addition

| | value |
|---|---|
| train samples | 19040 (14640 2D / 4400 3D) |
| train samples before histopathology | 13784 (so **+38%**) |
| val samples | 1750 (1250 2D / 500 3D) |
| sub-datasets | 46 train, 35 val |

Batches per rank per epoch, from `DistributedUniBatchSampler` at `batch_size=1`,
`batch_size_2d=10`:

| world size | train | val | val samples/rank |
|---|---|---|---|
| 1 | 5864 | 625 | 1750 |
| 4 | 1466 | 156 | 435 |
| **8** | **733** | **77** | **212** |

The 5864 figure was confirmed against the trainer's own "with 5864 iterations per epoch" line, so
the sampler arithmetic and the loader agree.

### Throughput and the epoch budget

Steady-state **~1.95 s/it**, geodesic target, full mixed dataset, `n_workers=8` on 16 CPUs. This is
better than the 2.47 s/it recorded for this model previously. The dataloader was **not** the
bottleneck, so there is no need to inflate `--cpus-per-gpu` for the geodesic job.

| | 8 GPUs | 4 GPUs |
|---|---|---|
| train time/epoch | ~24 min | ~49 min |
| **100 epochs** | **~60-70 h, fits a 96 h wall** | ~118 h, **does not fit** |

The 4-GPU column explains the historical `TIMEOUT` on the v3 joint run at 4 days. 8 GPUs is
load-bearing, not a nice-to-have.

### Cost of the geodesic target

`__getitem__` over a stride across the whole concatenated train set, 40 samples:

| target | mean | median | p90 | max |
|---|---|---|---|---|
| geodesic | 0.553 s | 0.147 s | 2.74 s | 3.24 s |
| directed | 0.314 s | 0.085 s | 1.43 s | 1.70 s |

Geodesic costs **1.76x** the CPU per sample, which is expected: two geodesic solves plus a distance
transform per object, against one vector distance transform. Both arms run the same number of
epochs so the comparison stays fair; the geodesic job is simply slower in wallclock.

### Memory

~71.4 GiB observed on the GPU during the geodesic run at `batch_size_2d=10, z=10, objs=5`. This is
consistent with the ~74 GiB previously recorded for this configuration and leaves roughly 20 GiB of
headroom on a 93.5 GiB H100. The script's own `[peak-memory]` line was never reached because the
run was stopped early, so treat 71.4 GiB as a good observation rather than a precise peak.

## State of the work: what is and is not verified

### Verified

- Dataset build succeeds for **both** distance types at `dataset_choice=all` including
  histopathology. Both probe scripts exited 0.
- All sizing and timing numbers above.
- **120 training iterations** of joint geodesic on the full dataset through
  `torchrun -> train_joint_sam2_multi_gpu -> _train_joint_rank -> DDP`, with no errors. Both
  branches stepped, mixed batches of 2D and 3D groups, bf16 autocast confirmed active ("Training
  with mixed precision").
- `flake8 --max-line-length=120` clean on every file touched.

### NOT verified - pick these up first

- **The directed arm was never trained.** Its run was still building datasets when everything was
  stopped. Only the dataset build and the `__getitem__` timing were measured for it.
- **No validation pass ever completed**, so the joint validate plus checkpoint-save cycle is
  unproven with the new 5-channel geodesic labels. See the trap below.
- **Only `world_size=1` was exercised.** Real multi-rank DDP - gradient all-reduce across ranks,
  `DistributedUniBatchSampler` sharding, the manual `_sync_automatic_grads` for the shared encoder -
  has not been run with this change.
- The script's own `[peak-memory]` print was never reached.

### The dry-run trap that cost the most time

A short `--n_iterations` run on `dataset_choice=all` is a **bad dry-run shape**. When training hits
the iteration cap, the trainer immediately validates over the *entire* val set - 625 batches at
world_size=1 - and that takes well over an hour at **0% GPU utilisation**, because the interactive
branch's correction-click sampling is single-threaded CPU work. It looks exactly like a hang.

Use `--dataset_choice hp --n_iterations 20` instead. Histopathology is 2D only and its val set is
350 samples, so the whole train, validate, checkpoint cycle finishes in minutes and still exercises
the new data and the new label transform.

More generally: **validation, not training, is the dominant walltime term.** At 8 GPUs it is 212
samples per rank per epoch and stays manageable, but it grows faster than training does if
`N_SAMPLES_VAL` (currently 50, in `generalist_loader.py`) rises or the GPU count drops.

## Open issues that need a decision

### 1. `sparse_hybrid` is orphaned - this one can silently void the whole experiment

`DEFAULT_POSTPROCESSING["sparse_hybrid"]` exists and was tuned for the geodesic target, but
**nothing selects it**. Both `micro_sam/sam_annotator/_widgets.py:4449` and
`micro_sam/v2/automatic_prompt_generation.py:68` hardcode `DEFAULT_POSTPROCESSING["sparse"]`.

Evaluate the geodesic model through any default path and it gets euclidean-tuned parameters -
precisely the configuration the ground truth tuning showed under-performs. The A/B would then
measure the postprocessing mismatch rather than the target. Wire this before the checkpoints land,
not after.

### 2. PanNuke is 21.8% of the entire training set

Inside `_get_hp_datasets`, cpm15, cpm17, monuseg and tnbc are each capped at `n_samples=50`, but
lizard (361), puma (552) and **pannuke (4143)** are uncapped. Histopathology overall is now 27.6%
of training, and the largest non-histopathology dataset in the whole generalist mix is 1000.

That is a 4x over-weighting of one dataset relative to anything else in the mix. It may be
deliberate - the code comment says the composition mirrors patho-sam's generalist training set -
but it materially changes what both models learn and it changes the epoch length. **Not changed;
needs the user's call.**

### 3. PanNuke padding offset is not randomized

`_pannuke_random_resize_and_pad_trafo` randomizes the resize *size* in steps of 64, but the pad is
always `(0, pad_total)` - bottom and right. Tiles therefore always sit in the top-left corner with
a zero-filled L-shaped border, despite commit `770e047f` being titled "Add randomized PanNuke
padding size and offset". Possible position bias. Low priority, but the commit title promises
something the code does not do.

## Instructions for the next agent

Environment rules that apply here: never `pip install` or `micromamba install`; run Python through
`micromamba run -n super`; use `flake8 --max-line-length=120` as the only linter and never run a
formatter; validate by running the real scripts with real data, not inline smoke tests.

1. **Read the diff first.** `git diff micro_sam/ finetuning/` in
   `/mnt/vast-nhr/home/archit/u12090/micro-sam`. It is 48 lines in the library plus the launcher.
   Nothing is staged or committed, and nothing may be committed without the user asking.

2. **Finish the dry runs that were cut short**, on the target hardware, in this order:

   ```bash
   cd /mnt/vast-nhr/home/archit/u12090/micro-sam/finetuning/v2/generalist
   export SAVE_ROOT=<a scratch dir, not the real one>
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

   # a) full train -> validate -> checkpoint cycle, both targets, minutes each
   torchrun --nnodes=1 --nproc_per_node=<n> train_joint.py \
       --model_type hvit_t --dataset_choice hp --n_iterations 20 --distance_type geodesic
   torchrun --nnodes=1 --nproc_per_node=<n> train_joint.py \
       --model_type hvit_t --dataset_choice hp --n_iterations 20 --distance_type directed

   # b) throughput and memory on the real mixed dataset, directed arm (never run)
   #    kill it once the progress bar reaches the iteration cap, or it will validate for an hour
   torchrun --nnodes=1 --nproc_per_node=<n> train_joint.py \
       --model_type hvit_t --dataset_choice all --n_iterations 120 --distance_type directed
   ```

   Step (a) is the one that matters most: it is the only thing that proves the joint validate plus
   checkpoint path works with the new 5-channel geodesic labels.

3. **Run multi-rank at least once** with `--nproc_per_node` greater than 1, even briefly. The
   automatic branch bypasses the DDP wrapper and its encoder gradients are all-reduced by hand in
   `JointSam2Trainer._sync_automatic_grads`; that path has not been exercised since the change.
   `test/test_v2_training.py::TestJointDdpGradientSync` covers the mechanism with fakes and is
   marked slow - worth running too.

4. **Confirm the epoch budget on the actual hardware.** Recompute
   `train_time_per_epoch = 733 * (measured s/it)` at 8 GPUs and add the measured validation time.
   The target is 100 epochs inside the wall. If s/it comes out above roughly 3.5 s on the new
   hardware, 100 epochs stops fitting a 96 h wall and the user needs to know before submitting.

5. **Raise the three open issues above with the user** before either job is submitted. Issue 1 is
   the dangerous one, because it does not fail loudly - it just makes the comparison measure the
   wrong thing.

6. **Do not** commit, push, or submit anything without the user asking.
