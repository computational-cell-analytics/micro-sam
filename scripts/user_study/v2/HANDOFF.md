# Handoff: micro-sam2 user-study data exploration

State as of 2026-10-10, branch `dev`. All segmentation uses micro-sam2 with `hvit_l_cells` unless noted.

Conventions:
- Call the tool "micro-sam2", never "v2".
- Commits are one line with no co-author trailer. Add `[skip ci]` only when the user asks for it.
- Write temporary embeddings to `$TMPDIR` (node-local NVMe), never to `~/.cache/micro_sam`, and remove them in a `finally`.
- Kill runs by PID, never with `pkill -f`.
- Python: `/mnt/lustre-grete/usr/u12090/micromamba/envs/super/bin/python`. It also has cellpose 4.2.1.1 (CellposeSAM),
  and the `cpsam` weights are cached in `~/.cellpose/models`.
- Data root: `/mnt/vast-nhr/projects/cidas/cca` (written below as `cca/`).

Compute:
- The interactive session GPU is a 10 GB MIG slice of an A100, too small for these volumes. Submit jobs instead.
- `grete:shared`, QOS `2h` (at most 10 jobs per user): A100s, use `--constraint="inet&80gb_vram"` for large volumes.
- `jupyter` V100s (32 GB): `--partition=jupyter --qos=2h -G V100:1`. The partition's default QOS (`interactive`) allows
  only 2 jobs per user, which the interactive sessions already use, so `--qos=2h` is required.
- `grete:preemptible` for jobs over 2h: MIG slices, e.g. `--gres=gpu:3g.40gb:1` (40 GB is the largest). Preempted
  jobs are requeued automatically, so check `squeue` for duplicates before resubmitting, or pass `--no-requeue`.
- Logs of the Huisken jobs: `cca/experiments/micro_sam2/huisken_data/logs/`.
- Logs of the Beghin jobs: `cca/experiments/micro_sam2/beghin_data/logs/`. The V100 logs show CUDA OOM warnings from
  the automatic batch size, but the runs complete.

## Scripts in this folder

| File | Status | Purpose |
|------|--------|---------|
| `explore_huisken_data.py` | committed, **with uncommitted changes** (section 1b) | Huisken data: the high-res ExM volume (block AIS/APG, slab-parallel AIS and stitching) and the small ExM AVI and membrane volumes (`-d`/`--sweep`) |
| `explore_beghin_data.py` | committed (`03fc6ca6`), **with uncommitted changes** | Beghin `nus_organoids` (with GT), the CellposeSAM comparison and the MCF7 spheroid DAPI stacks |
| `explore_gruen_data.py` | committed (renamed from `explore_lisa_data.py`) | Gruen data; no new work done yet |
| `submit_huisken_slabs.sh`, `submit_huisken_full.sh` | gitignored (`*.sh`) | SLURM submit scripts for the Huisken slabs |
| `submit_huisken_small.sh` | gitignored | SLURM submit script for the ExM AVI and membrane sweeps (stages `ais`, `best`, `membrane2`); later runs were submitted by hand |

Uncommitted in `explore_beghin_data.py`:
- `-e` overrides the engine of `--best`, e.g. `--best -e ais -n TM00099_cell`.
- `--clean` applies `clean_spheroid_segmentation` instead of the min size and stores `<engine>_clean`.
- The `clean` setting in `BEST_SETTINGS`, used for ZeroG.
- `--cellpose` runs CellposeSAM in both modes (`run_cellpose`).

Library change, committed: `micro_sam/v2/automatic_segmentation.py::automatic_instance_segmentation` takes an optional
`norm_bounds`, which it forwards to `precompute_image_embeddings` for 3D inputs. Tests for it are in
`test/test_v2_automatic_segmentation.py`.

## 1. Huisken high-res ExM volume: done

- Raw data: `cca/data/huisken_data/Pdu_2dpf_ExM/` (`hires_volume.h5` and the `.raw`). The data moved into this
  subfolder; the paths in the script are updated (uncommitted).
- Best setting: AIS on the volume downsampled 2× in zyx, with `min_size 500`. The min size removes the few-pixel
  fragments attached to every nucleus.
- The whole volume was segmented over z 0–2435 (in ds2 coordinates):
  - It runs as slabs of 256 slices, each with a global `norm_bounds`, so that empty slabs don't stretch noise.
  - The slabs are stitched by IoU matching in the overlaps, with union-find.
  - `remove_low_contrast_objects` (contrast ≥ 1.1, background ring of 6 px) removes background noise and keeps the
    dim nuclei at the ends of the volume.
- Outputs, in `cca/experiments/micro_sam2/huisken_data/`:
  - `hires_volume_ds2_z0-2435_hvit_l_cells_ais_min500_slabs256.h5`: the full result, 2160 objects.
  - `hires_volume_ds4_..._slabs256.h5`: a uint16 preview that fits on a 32 GB laptop.
  - `slabs256_*/`: the per-slab results.
- Local crop for viewing: `huisken_crop_z2900-3028_y1024-1536_x1280-1792_hvit_l_cells.h5`.
- SLURM: `grete:shared`, QOS `2h`, `inet&80gb_vram` (40 GB A100s run out of memory). Slab 256 needs 64G of RAM, slab
  512 about 61 GB, which is too tight for 64G.

## 1b. Huisken small volumes (ExM AVI and LSM membrane): done, results with the user

Two smaller volumes, both run with `explore_huisken_data.py` (`SMALL_DATA_PATHS`, `SMALL_SETTINGS`):
- `exm`: `Pdu_2dpf_ExM/Pdu_6dpf_ExM_downsampled_nuclei-sytox-green.avi`, nuclei, very low resolution. It's read
  with cv2, channel 0 of every frame.
- `membrane`: `Pdu_2dpf_membrane/fused_t000050_R0001_C02_Dfinal_cropped.tif`, shape (83, 509, 542), thin membranes.
  The TIF has no voxel size; the z anisotropy is estimated at about 4–5×.

Usage (from this folder):
- `python explore_huisken_data.py --sweep exm --crop dense dim -e ais [-s <settings>] [--tag <tag>]` runs settings on
  the exm crops (`EXM_CROPS`: `dense`, `dim`, `dense_small`). `--sweep membrane` runs on the whole membrane volume.
- `python explore_huisken_data.py -d exm -e ais -s x2_min100_merge_bright` runs one setting on the whole volume.
  Volumes over 128 slices run in resumable z-slabs (128 slices, halo 16, IoU matching and union-find), each saved
  under `<dataset>_slabs128_<engine>_<setting>/`.
- `--view -d <dataset> [--crop <crop>] [--tag <tag>]` opens the results in napari.
- Run parallel jobs with different `--tag`s, since they would otherwise write to the same h5 file.

Settings (`SMALL_DEFAULT_SETTINGS` lists every key): `scale` (zoom before segmentation, labels resampled back with
nearest), `anisotropy`, `norm_percentiles`, `ridge` (Sato filter blended with the data), `clahe`, `gamma`, `invert`,
`tile_shape`/`halo`, `ndim` (2 = per slice), `min_size`, `merge_fragments`, `min_border_intensity`,
`min_mean_intensity`.

### Results

- **exm best: AIS `x2_min100_merge_bright`** (the user picked min100).
  - Upsample 2× in zyx, AIS with min size 100.
  - `merge_fragments 400`: objects under 400 voxels merge into the neighbor covering ≥ 60% of their border. This
    removes the small artifacts inside nuclei the user didn't want.
  - `min_mean_intensity 5` removes the noise objects in the black background (dim nuclei are at ≥ 8).
  - Whole volume: 9364 nuclei, 57 min, 69 GB RAM peak (the unslabbed run hit 256 GB and stalled).
  - 1× merges nuclei, 3× fragments them; CLAHE and gamma add haze objects; percentiles barely matter.
- **membrane best: AIS `ridge_clahe_z4_t128_clean`** (the user's pick: "the best of all worst cases"), 961 cells.
  - Ridge 0.5 + CLAHE 0.02, 4× in z, 128-px tiles with halo 32 (each tile is resized to 1024 px, so this magnifies
    the thin membranes about 5×), min size 200, `min_border_intensity 1500` (removes objects whose border is not on a
    bright membrane, i.e. background).
  - Tiling is required; without it the membranes are too thin. The library only tiles above 768 px on its own.
  - Didn't help: dense mode (user said skip it), inverting, a 64-px halo, 64-px tiles, 2× in yx, 2D per slice,
    gamma, deblurring (Richardson-Lucy, unsharp, DoG in a 2D test). The ridge filter was what recovered the blurred
    tissue.
  - Known limits: straight cuts at tile borders, and some cells missing in the most blurred areas.
- **APG is not worth it on either volume.** On the exm crops it matches AIS at 80–200× the runtime. On the membrane
  (`apg_ridge_clahe_z4_t128_clean`, 94 min) it gives 2423 objects: 635 under 200 voxels (AIS: 5), plus large
  straight-edged objects along tile and block borders covering the background and interior. APG ignores `min_size`.

Tile seams: `micro_sam/v2/batched_inference.py` (around line 1457, `output[(slice(None),) + inner] = prediction[...]`)
pastes the inner tile region of the decoder predictions without blending. A fix would blend the predictions in the
overlap. Proposed, not implemented.

### Outputs

In `cca/experiments/micro_sam2/huisken_data/`: `exm_hvit_l_cells.h5` (whole volume), `exm_z..._hvit_l_cells.h5`
(crops), `membrane_<tag>_hvit_l_cells.h5` (one per tag). Keys are `<engine>_<setting>`, with the settings, object
count and runtime as attributes.

Copies for viewing, in `review/`:
- `exm_full.h5`: **`ais_x2_min100_merge_bright`** and `ais_x2_min100_merge` (without the intensity filter).
- `exm_crop_dense.h5`, `exm_crop_dim.h5`, `exm_crop_dense_small.h5`: the crop sweeps; dim and dense_small also have
  APG.
- `membrane_ridge_clahe.h5`: **`ais_ridge_clahe_z4_t128_clean`** and `apg_ridge_clahe_z4_t128_clean`.
- `membrane_*.h5`: one per membrane experiment (tiling, t128, invert, 2d, clahe, gamma, p95, ridge, deblur2d, ...).

## 2. Beghin `nus_organoids` (with GT): micro-sam2 vs CellposeSAM, done

- Data: `cca/data/beghin_data/nus_organoids`, four uint16 ZYX volumes with an instance GT each.
- Runs (from this folder, on a V100 or A100, see Compute):
  - micro-sam2: `python explore_beghin_data.py --best -o beghin_best_segmentations`
  - CellposeSAM: `python explore_beghin_data.py --cellpose -o beghin_best_segmentations`
- Metrics come from `evaluate()`: mSA, SA50 and SA75; precision, recall and F1 at IoU 0.5; SBD.

### Results

micro-sam2 with the best settings of each volume vs the better of the two CellposeSAM modes (by mSA):

| Volume | Method | mSA | SA50 | SA75 | Prec | Rec | F1 | SBD |
|---|---|---|---|---|---|---|---|---|
| HCT116 monolayer | micro-sam2 | **0.704** | **1.000** | **0.854** | **1.000** | **1.000** | **1.000** | **0.920** |
| | cpsam | 0.536 | 0.778 | 0.739 | 0.833 | 0.921 | 0.875 | 0.794 |
| Colon organoid | micro-sam2 | **0.493** | **0.976** | **0.456** | **1.000** | **0.976** | **0.988** | **0.837** |
| | cpsam | 0.298 | 0.581 | 0.256 | 0.643 | 0.857 | 0.735 | 0.625 |
| ZeroG spheroid | micro-sam2 | **0.417** | **0.795** | **0.436** | **0.873** | **0.899** | **0.886** | **0.767** |
| | cpsam | 0.084 | 0.164 | 0.078 | 0.179 | 0.667 | 0.282 | 0.181 |
| TM00099 | micro-sam2 | **0.198** | **0.588** | **0.047** | **0.777** | **0.708** | **0.741** | **0.675** |
| | cpsam | 0.107 | 0.316 | 0.038 | 0.487 | 0.474 | 0.480 | 0.549 |

Which result each row uses:

| Volume | micro-sam2 (`BEST_SETTINGS`) | cpsam mode |
|---|---|---|
| HCT116 | AIS, scale (1,2,2), min_size 500 | 3d |
| Colon organoid | AIS, scale (2,2,2), anisotropy 4, min_size 500 | 2d_stitch (3d has the higher SBD, 0.655) |
| ZeroG | AIS, spheroid cleanup (`clean`) | 2d_stitch |
| TM00099 | APG, background_sigma 8, norm percentiles (99, 99.99), tiles (4,256,256), halo (2,52,52) | 2d_stitch |

All CellposeSAM scores (`cpsam`, default settings):

| Volume | Mode | Objects (GT) | mSA | SA50 | F1 | SBD | Time |
|---|---|---|---|---|---|---|---|
| HCT116 | 3d | 42 (38) | 0.536 | 0.778 | 0.875 | 0.794 | 284 s |
| | 2d_stitch | 48 | 0.204 | 0.536 | 0.698 | 0.579 | 14 s |
| Colon organoid | 3d | 48 (42) | 0.264 | 0.552 | 0.711 | 0.655 | 290 s |
| | 2d_stitch | 56 | 0.298 | 0.581 | 0.735 | 0.625 | 15 s |
| ZeroG | 3d | 572 (99) | 0.079 | 0.151 | 0.262 | 0.137 | 341 s |
| | 2d_stitch | 369 | 0.084 | 0.164 | 0.282 | 0.181 | 69 s |
| TM00099 | 3d | 267 (359) | 0.076 | 0.240 | 0.387 | 0.470 | 1490 s |
| | 2d_stitch | 349 | 0.107 | 0.316 | 0.480 | 0.549 | 334 s |

- The CellposeSAM modes:
  - `3d`: `do_3D=True`, which predicts the flows on the yx, zx and zy planes and follows their average in 3D
    (the aggregation).
  - `2d_stitch`: 2D on every z plane only, with the masks of neighboring planes stitched at IoU ≥ 0.5
    (`CELLPOSE_STITCH_THRESHOLD`).
  - No `anisotropy` is passed. That likely hurts `3d` on the anisotropic colon organoid and TM00099.
- Observations:
  - micro-sam2 beats CellposeSAM on every volume and metric. Note that micro-sam2 uses settings tuned per volume,
    while CellposeSAM uses its defaults.
  - On ZeroG, CellposeSAM segments the bright speckles around the spheroid (precision < 0.18). micro-sam2 needed the
    spheroid cleanup for the same reason.
  - `3d` is 5–20× slower than `2d_stitch`, and better only on HCT116.

### micro-sam2 notes

- ZeroG: the spheroid cleanup replaced `min_size 2000` (mSA 0.389, SA50 0.729, F1 0.843, SBD 0.723 before), because the
  min size also removed valid small nuclei.
- TM00099 is the hard case:
  - The nuclei cover 0.2% of the volume and sit inside dim cluster bodies. With the default settings, mSA is 0.001.
  - Background subtraction and high-percentile normalization make the nuclei visible.
  - The tile size matters. Larger tiles change the effective scale, and full-volume APG then found only 28 objects.
  - AIS with the same preprocessing and tiling is weaker than APG: mSA 0.135, SA50 0.422, F1 0.594, SBD 0.604.

### Outputs, in `beghin_best_segmentations/`

- Each `{name}_hvit_l_cells.h5` holds `raw`, `gt`, `ais`, `cpsam_3d` and `cpsam_2d_stitch`, and TM00099 also `apg`.
  Every segmentation has its scores (and settings or runtime) as attributes.
- For TM00099, the micro-sam2 result is `apg`.
- `TM00099_cell_crop_experiments.h5` holds the parameter sweeps on a crop containing 77 GT nuclei.
- `python explore_beghin_data.py --view -n <name> -o beghin_best_segmentations` opens a file in napari.

## 3. Beghin MCF7 spheroids (no GT): done, awaiting user feedback

- Data:
  - `cca/data/beghin_data/39_Spheroids_MCF7_GCp24_DAPI_Ki67_B4`
  - `cca/data/beghin_data/3_Spheroids_MCF7_GCp24_DAPI_eCadh_btub`
- Run: `python explore_beghin_data.py --spheroids`. This runs AIS on DAPI (channel 0).
- Output: `cca/experiments/micro_sam2/beghin_data/spheroids/<set>/<stack>_hvit_l_cells.h5`.
  - There are 42 files, 807 MB in total, with 7801 objects.
  - Each file holds only `raw` and `ais_clean`. **Look at `ais_clean`.**
- `clean_spheroid_segmentation` replaces the old `min_size 2000`, which removed valid small nuclei:
  1. Merge small objects (< `LARGE_SIZE` 2000 voxels) into a nucleus when ≥ 60% of their border is on it. These are
     the fragments that the flow/density seeds create inside nuclei.
  2. Remove small objects that are:
     - far from any large object (`NEAR_XY` 15, `NEAR_Z` 5), i.e. frame speckles;
     - under 300 voxels;
     - grainy (intensity CV ≥ 0.45).
  3. Run `remove_noise_objects`, which removes objects with contrast below 1.1, and flat grainy blobs (CV ≥ 0.5 with a
     z-extent of at most 9 slices). This handles the end-slice noise.
- Effect: 7801 objects, against 7235 with min2000. Middle slices typically gain 1–6 nuclei, and the dim top nuclei
  are kept. On ZeroG, which has GT, the same cleanup improves every metric (see section 2).
- Known limits:
  - Objects under 300 voxels are still dropped, so a few tiny nuclei or nuclear caps can be lost.
  - A few speckle-layer objects remain, e.g. in `s20`.
  - The cleanup was checked visually on about 8 stacks only.

## Open items

1. Commit and push the uncommitted `explore_beghin_data.py` and `explore_huisken_data.py` changes (see Scripts) once
   the user approves. Both pass flake8.
2. Huisken small volumes, only if the user wants them: blend the tile predictions in the library to remove the
   membrane seams; filter the membrane APG result (min size, the large blocky objects). User feedback on the exm and
   membrane AIS results is pending.
3. **The user's detailed feedback on the spheroid `ais_clean` is pending.** Adjust the constants in
   `explore_beghin_data.py` (`LARGE_SIZE`, `SMALL_MIN_SIZE`, `SMALL_MAX_CV`, `NEAR_*`, `FRAGMENT_MIN_ENCLOSED`) based on
   it.
4. Gruen data: nothing done yet. A data path rename was offered but not answered.
5. The "new Beke data" was never found. The old Beke evaluation is in `scripts/plotting/v2/interactive_visualization.py`
   and `visualize.py`, with its data in `cca/data/beke_data`.
6. Possible CellposeSAM follow-ups, if a fairer comparison is wanted: pass `anisotropy` to `3d` for the colon organoid
   and TM00099, or apply the spheroid cleanup to its ZeroG result.
