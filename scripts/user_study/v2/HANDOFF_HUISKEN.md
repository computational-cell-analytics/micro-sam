# Handoff: Huisken lab predictions

State as of 2026-10-14, branch `unify-model-cache`. Every Huisken volume now has its final micro-sam2 prediction. The
general user-study handoff is `HANDOFF.md` in this folder.

Conventions:
- Tifs for the collaborators use the encoding of their own Fiji files: tifffile `compression="zlib"` (deflate), strips
  (no `tile=`), classic tiff below 4 GB, no ImageJ metadata. Never zstd, never tiles: Fiji's reader handles neither.
- Predictions are uint16 instance labels. Check `max() <= 65535` before the cast and read every file back to verify it.
- A prediction for a collaborator has the shape of their raw file, so that it overlays it directly. A prediction of a
  crop is placed at the crop in the full volume, zero outside.
- New files on vast: `chgrp cidas` and `chmod g+rw`. Folders owned by u12090 (the user's other account) can only be
  opened up from that account.
- Keep scripts and logs on vast, not in the job scratchpad: it disappears with the SLURM job.
- h5 files for viewing in napari (keys `raw`, `groundtruth`, `segmentation...`, gzip) are built on request in the repo
  root and deleted afterwards. None are kept, rebuild them from the tifs.

Compute from the u28048 account (`kisski_usam2_pape`), which differs from `HANDOFF.md` (written for u12090):
- Python: `/mnt/vast-kisski/projects/kisski_usam2_pape/micromamba_anwai/envs/super/bin/python -u`. The u12090 env
  path in `HANDOFF.md` is not readable from u28048. `micromamba run -n super` buffers stdout until the end.
- `grete:shared` is not available to this account. Use `jupyter` (`--qos=2h -G V100:1`) or `grete:preemptible`
  (`--gres=gpu:2g.20gb:1` or `3g.40gb`).
- The V100s run SAM2 in fp32, the A100 slices in bf16. The embedding cache records the precision, so a V100 job
  recomputes (and overwrites) bf16 embeddings: give parallel jobs their own embedding paths.
- The interactive session GPU is a 2g.20gb slice, enough for the embryo embeddings and the CellposeSAM 2D runs.

## Data layout and final predictions

`/mnt/vast-nhr/projects/cidas/cca/data/huisken_data/`, documented in its `README.md` (the reference for names and
settings). Naming: `<name>.tif` raw, `<name>_groundtruth.tif` annotation, `<name>_prediction[_<method>].tif`. Without a
method suffix, a prediction is the final micro-sam2 result.

| Cohort | Raw | Final prediction | Model, settings |
|---|---|---|---|
| `exm_lsm/high_res/tile` | `S000_..._P04871.raw` / `.h5` (key `raw`), (4871, 2048, 2048) uint16 | `..._prediction.tif` (4871, 2048, 2048), 2160 objects; `..._ds2_prediction.tif` (2435, 1024, 1024) | `hvit_l_cells` AIS on the 2x downsampled volume, `min_size` 500, z-slabs of 256 + halo 16, stitched by IoU, upsampled 2x (nearest), the last slice repeated |
| `exm_lsm/low_res/whole` | `Pdu_6dpf_..._sytox-green.avi` and its `.tif` (765, 565, 1405) uint8 | `..._prediction.tif`, 9364 objects | `hvit_l_cells` AIS `x2_min100_merge_bright` (`SMALL_SETTINGS["exm"]`) |
| `classic_lsm/membrane` | `fused_t000050_R0001_C02_Dfinal_cropped.tif` (83, 509, 542) uint16 | `..._prediction.tif`, 961 objects | `hvit_l_cells` AIS `ridge_clahe_z4_t128_clean` (`SMALL_SETTINGS["membrane"]`) |
| `classic_lsm/early_embryo`, fused t0 | `fused_t000000_R0000_C02_Dfinal.tif` (208, 1024, 1024) | `_prediction.tif` (full size) and `<crop>_prediction.tif`, 23 objects | `hvit_l_cells` APG on the full volume, cropped, see below |
| `classic_lsm/early_embryo`, 3.5hpf | `C2-Pdu_Composite_3.5hpf_HRas-mScarlet_H2B-GFP.tif` (79, 1024, 1024) | `_prediction.tif` (full size) and `<crop>_prediction.tif`, 12 objects | the same as fused t0 |

- The membrane raw is byte-identical to Beke's ownCloud share `F4VXUkpOgwXMiMR` (`Pdu_2dpf.zip`). The `_cropped` is in
  Beke's original name, the share also has Beke's own `auto_segmentation_..._cropped.tif` (not downloaded).
- Only the two embryos have ground truth. Fused t0 also has `_autosegmentation.tif`, Beke's automatic result, not
  ground truth.

## Early embryos: micro-sam2 against CellposeSAM

Crops: fused t0 `_z0-208_y215-727_x253-765`, 3.5hpf `_z0-79_y211-723_x255-767` (`EMBRYO_CROPS`). Scored with
`evaluate_embryo` on `<crop>_groundtruth_filtered.tif`, relabeled and size-filtered at 5000 voxels (24 and 14 cells).

| Embryo | Method | Objects | mSA | SA50 | SA75 | SBD |
|---|---|---|---|---|---|---|
| fused t0 | micro-sam2 (`hvit_l_cells` APG) | 23 | 0.346 | 0.615 | 0.400 | 0.748 |
| fused t0 | CellposeSAM (`2d_stitch`) | 93 | 0.014 | 0.037 | 0.018 | 0.077 |
| 3.5hpf | micro-sam2 (`hvit_l_cells` APG) | 12 | 0.498 | 0.857 | 0.529 | 0.776 |
| 3.5hpf | CellposeSAM (`2d_stitch`) | 22 | 0.303 | 0.565 | 0.200 | 0.548 |

micro-sam2:
- On the 512 px crop, SAM2 resizes every slice to 1024 px, which magnifies the cells 2x beyond the training scale and
  fragments them (fused t0: AIS 0.003, APG 0.139). The fix is the full 1024 px volume, cropped afterwards.
- APG flow: `n_iter` 200, `dt` 2, spacing (2, 1, 1), `min_size` 20000, tuned on fused t0 only. The user chose APG over
  AIS on both embryos after looking at them.
- 3.5hpf, all scores: APG 0.498 (t0 setting), 0.480 (+ `foreground_threshold` 0.4, best SA50 0.929 and SBD 0.826).
  AIS: 0.447 (`foreground_threshold` 0.3), 0.401 (the t0 AIS setting `n200_dt2_fg0.4_s2_d50_z2_min20k`), 0.364 (the
  t0 flow alone). One APG run on 3.5hpf takes ~10 min (V100 or 2g.20gb), on fused t0 ~90 min (2g.20gb).
- Fused t0: AIS 0.347 (20 objects), `hvit_t_cells` APG 0.340.

CellposeSAM (`cpsam`, default settings, no `anisotropy` or `diameter`, as in the Beghin comparison):

| Embryo | Mode | Input | Objects | mSA | Time |
|---|---|---|---|---|---|
| fused t0 | `3d` | crop / full | 706 / 2842 | 0.000 / 0.000 | 227 / 772 s |
| fused t0 | `2d_stitch` | crop / full | 93 / 103 | 0.014 / 0.013 | 99 / 302 s |
| 3.5hpf | `3d` | crop / full | 923 / 714 | 0.006 / 0.005 | 169 / 582 s |
| 3.5hpf | `2d_stitch` | crop / full | 22 / 31 | 0.303 / 0.158 | 42 / 131 s |

- `3d` predicts the flows on the yx, zx and zy planes and follows their average in 3D, it fragments the large cells.
  `2d_stitch` is 2D on every z plane, stitched at IoU 0.5. Unlike micro-sam2, CellposeSAM does not gain from the full
  volume.
- The best mode (`2d_stitch` on the crop) is saved as `<name>_prediction_cellposesam.tif`, full size like the
  micro-sam2 prediction. The older `<crop>_prediction_cellposesam.tif` files are from before this session (settings
  not recorded, their scores match the `3d` mode) and were kept.
- Fairness: micro-sam2 is tuned on fused t0, CellposeSAM uses its defaults. 3.5hpf is the fairer comparison, since
  micro-sam2 was not tuned on it.

## Code state

`explore_huisken_data.py` (committed in `9b61ac03`, plus uncommitted changes, flake8 clean):
- `--ais` next to `--apg` (`run_embryo_automatic(..., mode)`) and `--embedding_root` (default `EMBRYO_OUTPUT_ROOT`,
  owned by u12090 and not group-writable).
- `--cellpose 3d 2d_stitch` with `--embryo` (`run_embryo_cellpose`, `EMBRYO_CROPS`, `CELLPOSE_MODES`): a full embryo
  is segmented whole and cropped, a `_crop` embryo directly; scored on the crop ground truth and saved as
  `<crop>_prediction_cpsam_<mode>[_full].tif`. Validated with `--embryo 3.5hpf_crop --cellpose 2d_stitch` (22 objects,
  mSA 0.303, the same as the array run).
- Still missing: the final micro-sam2 embryo setup. `--embryo t0_crop --ais/--apg` still runs on the crop with the
  default settings (the 0.003 / 0.139 results) and only allows ground truth for `t0_crop` (`EMBRYO_GT_VOLUME`).

Scripts and results in `cca/experiments/micro_sam2/early_embryo/` (logs in `logs/`):
- `embryo_sweep.py`: the fused t0 sweep (AIS from cached decoder predictions, APG on the full volume), results in
  `sweep/`.
- `run_3.5hpf.py`: 3.5hpf with the fused t0 settings (`--precompute`, `--ais`, `--apg <setting>`,
  `--embedding_path`), results in `sweep_3.5hpf/`.
- `run_cpsam.py`: the CellposeSAM job array (`--task 0-7`), the cropped results in `sweep_cpsam/`. Superseded by
  `--cellpose` in `explore_huisken_data.py`.
- `export_hvit_l_cells.py`: wrote the fused t0 tifs.
- The `hvit_l_cells` embeddings of every variant (`*.zarr`, up to ~1.4 GB each).

## Open items

1. Port the final micro-sam2 embryo setup into `explore_huisken_data.py` (full volume, crop afterwards, the flow
   settings, ground truth for both embryos via `EMBRYO_CROPS`), then commit with the user's OK. The uncommitted
   `environment.yaml` / `setup.cfg` pin of `pydantic<2.14` is unrelated.
2. Optional: CellposeSAM with `anisotropy` or `diameter`, for a fairer comparison in `3d` mode.
3. APG drift on the fused t0 crop with `hvit_t_cells` (mSA 0.306 on 2026-09-16, 0.340 now): deferred by the user.
4. Optional: more APG settings on fused t0 (e.g. `foreground_threshold` 0.4, which gave the best SA50 on 3.5hpf).
