# Tuning and test data of the post-processing search (cells)

State of 2026-11-23, branch `ais-hiera-training`. The tables are read from the code: the tuning pools are
`parameter_search.IN_DOMAIN_CELLS`, the pinned tuning crops `evaluation/tuning_crops.json`, the test panels
`PANEL_DATASETS` of `scripts/plotting/v2/for_paper/plots_for_figure{2,3}.py`, and the counts `common.n_samples`.

## What changed (2026-11-20 / 21)

The tiles of the datasets that scored low in the 3D search were checked by hand (`select_tune_tiles.py`). The data
is fine; three datasets were scored in a way that does not fit them.

| dataset | finding | now |
|---|---|---|
| `platynereis_cells` | Half of each tuning volume is an unlabelled halo that shows cells; label zero inside marks cells that were left out; many label ids are no cells (a few hundred voxels on 1-2 sections). | Multicut + CREMI score, in tuning and test (`common.DATASETS_DENSE`). Label zero and the neuropil label are not scored (`common.CREMI_IGNORE_ZERO_LABEL`). Tuning reads the volumes inside the bounding box of their labels (`common.PLATYNEREIS_CELLS_VAL_ROIS`), 6 tiles. The test crops already tile the labelled box of volume 9. |
| `synapsenet_compartments` | Label zero (2-46 % of a tile) marks unannotated parts; more than half of the objects are fragments below 1000 voxels next to compartments of 1e5-1e6 voxels. | Multicut + CREMI score, in tuning and test; label zero not scored. One tile per tuning tomogram, 8 tiles. |
| `cartocell` | The cells are thin (about 12 x 17 x 17 voxels): the ground truth shrunk by one voxel scores mSA 0.01. The 2 volumes of the validation folder are of the `train_M1` acquisition ("cyst ..."), which the test split does not hold. | Flow-based segmentation, ranked and compared by SA50 (`common.DATASETS_SA50`). Tuning on 9 tiles of the 6 test volumes that moved into tuning (`common.CARTOCELL_TUNING_TEST_VOLUMES`); the test split holds the other 54 volumes (68 crops). |

Code: `evaluation/common.py` (pools, regions, ignore label), `evaluation/parameter_search.py` (criterion `sa50`,
CREMI score without label zero, a stored sweep without its ranking metric is recomputed),
`evaluation/tuning_crops.json`, `decoders/evaluate_decoder.py` (`report` compares SA50 for cartocell and lists it
apart from the mSA means), `decoders/tune_set.py`, `decoders/select_tune_tiles.py` (new), tests in
`test/test_parameter_search.py`.

Update 2026-11-21 (search and test of the Hiera-U checkpoint of epoch 25, see README.md):
- The reference results (micro-sam2 hvit_t, library defaults) are in
  `<decoder_training>/evaluation/reference_hvit_t/`: a copy of the registry evaluation, with `platynereis_cells`,
  `synapsenet_compartments` (multicut, CREMI score) and `cartocell` (54 test volumes) recomputed.
- The search has a boundary stage (2D, 3D: `boundary_weight`, `boundary_mask_threshold`; dense: `boundary_weight`,
  which mixes the predicted boundary into the multicut boundary map) and, for 2D and 3D, a `min_size` stage that
  gives the best `min_size` per tuning dataset. The test uses the best parameters over the tuning datasets of a
  setting with the `min_size` of the dataset where it has tuning data (the "tuned in" column below).
- Of the 3094 2D tuning samples 20 have no labelled object and are not scored (pannuke 11, histo_miner 3,
  neurips_cellseg_label_free 2, tsakiroglou 2, enseg 1, livecell 1).
- The test of the new decoder also scores the four supplement datasets.

Update 2026-11-23 (after the comparison of the epoch 25 checkpoint with the reference,
`TUNING_EVALUATION_RESULTS.md`; the search of that checkpoint ran with the pools before this change):
- `orgline` is in the 2D pool with its validation split (515 images, 11 of them without a labelled object). A third
  of its test images hold one organoid of about 100000 pixels, which got no seed with the parameters found without
  it; no other 2D tuning dataset has such objects. The broad grid holds the longer flow travel these objects need:
  on the validation split the epoch 25 checkpoint scores mSA 0.287 at the 2D winner (`n_iter` 50, `dt` 1) and 0.379
  at twice the travel, the same for `n_iter` 50 with `dt` 2 (the form in the grid) and `n_iter` 100 with `dt` 1.
- `liconn` and `xpress` are out of the dense pool. They are no EM volumes (expansion microscopy, X-ray), have one
  tuning crop each and decided the dense parameters: without them the search gives the parameters of the reference.
  Both are still evaluated: `liconn` in the EM neurites panel, `xpress` with the supplement datasets
  (`evaluate_decoder.SUPPLEMENT_DATASETS`; it is not in the panel because its test data is a single slab of 20
  slices).
- `plantseg_root` and `nucverse3d` tune on 10 tiles each that are placed like their test crops
  (`tune_set.TEST_LIKE`): tiles centered in the tuning region (`tile_starts(centered=True)`, which reproduces the
  test crops of `eval_crops.json`), without the tiles below 20 % of the median object count, and of these 10 that
  are spread over the volumes and over the object counts (`parameter_search.spread_tiles`). No model score enters
  the choice. Both scored lower on their tuning crops than on the test crops, which came from the crops and not
  from the volumes:
  - `plantseg_root`: the center crops of the two volumes sit in the dense core of the root (36 and 46 cells); 72
    of the 98 test crops hold at most 20 cells, and the test crops with more than 20 score as low as the center
    crops did (0.21-0.23 against 0.39 for the crops with at most 10 cells, epoch 25 checkpoint). The 10 tiles hold
    3-39 cells (8 of the first volume, 2 of the second).
  - `nucverse3d`: one center crop per collection made the glia slab (11 slices, mSA 0.09) a third of the tuning
    data; it is 2 of the 54 test crops. Now: 1 glia, 5 liver and 4 liver_hcc tiles.
- The light microscopy volumes and `densecell` (`common.SLIVER_DATASETS`, 28 datasets) are scored without slivers,
  in the parameter search and in the test (user, 2026-11-23). The volumes are scored on crops of 32 slices, which
  cut almost every object (99 % of the `plantseg_root` cells), and a part of the pieces is a sliver of a cell
  that lies outside the crop: 48 of the 164 objects in the `plantseg_root` tuning tiles have fewer than 1000 voxels
  and all of them are cut, while the smallest whole cell of the tuning volumes has 31000.
  - Rule (`common.exclude_slivers`): an object that touches a face of its crop and is smaller than a tenth of the
    median object of its dataset (`common.SLIVER_FRACTION`) is a sliver. It is cleared in the ground truth, the
    segmentation is cleared in its place, and the predicted objects on a crop face below the same size are
    dropped. A crop whose objects are all slivers is skipped.
  - The size per dataset and split is measured on the ground truth of all crops of the split
    (`parameter_search.py prepare --sliver_sizes`, about 5 minutes with 4 processes) and stored in
    `evaluation/sliver_sizes.json` with the median and the number of slivers. It has to be measured again when the
    crops of a dataset change. Examples (test split): `plantseg_root` 4217 voxels (619 of 1931 objects are
    slivers), `pnas_arabidopsis` 676 (6209 of 43130), `nucverse3d` 330 (463 of 3612), `celegans_atlas` 85 (194 of
    3102), `embedseg_organoid` 4243 (0 of 192).
  - The results hold `gt_slivers` (the number of ground-truth slivers left out) and, in the test,
    `mSA_with_slivers` (the score from before, to check a rescoring against the old results). A result, sample row
    or sweep of these datasets without `gt_slivers` counts as not computed (`common.result_is_current`,
    `parameter_search.sweep_cache_is_current`, `sweep_is_done`), and `evaluate_decoder.py report` does not compare
    a result with slivers with one without.
  - Effect on the tuning tiles of the epoch 25 checkpoint with the 3D winner (mSA with / without slivers):
    `plantseg_root` 0.256 / 0.353 (58 of 164 objects are slivers), `nucverse3d` 0.307 / 0.342 (73 of 632). The
    order of the `min_size` values does not change (table in `<root>/param_analysis/tune_set/min_size_20261123/`).
  - The EM organelle volumes keep their scoring, so the results of the EM organelle models are not touched. The
    dense datasets are scored by the CREMI score, which has no objects to leave out.

- The other 3D datasets with more tuning data are extended to at most 10 crops (user, 2026-11-23); the 3D tuning
  set has 105 crops (66 before 2026-11-23):
  - `gonuclear` (1 -> 10), `celegans_atlas` (3 -> 10) and `embedseg_platy_nuclei` (1 -> 3) take tiles placed like
    their test crops, like `plantseg_root` and `nucverse3d` (`tune_set.TEST_LIKE`).
  - `nis3d` (1 -> 4) and `densecell` (1 -> 4) take all edge-aligned tiles of their slab: it holds one tile of
    512 pixels without overlap, the same as the center crop. The `nis3d` tiles overlap by 46 pixels, the
    `densecell` tiles (15 slices, 800 x 800) by 224.
  - Unchanged: `pnas_arabidopsis` (20), `embedseg_organoid` (11) and `phmamm` (10) keep one center crop per volume,
    `cartocell` its 9 tiles placed by hand. `wing_disc` (2), `embedseg_mouse_skull` (1) and `embedseg_platy_ish`
    (1) have no further tile.
- `tune_decoder.py` refuses stored predictions that belong to other tuning crops (they are stored by sample
  index), and a search that is started again runs the sweeps of a later stage again if their combinations changed
  (`parameter_search.sweep_has_params`).

Open:
- The dense search of the new decoder includes `platynereis_cells` and `synapsenet_compartments`; the dense defaults
  of the previous decoder (the reference) were tuned on neurites only.
- The figure tables (collector script of the baselines) read mSA for cartocell.

## Next steps: rerunning the search and the evaluations

Nothing ran with the tuning data and the scoring of 2026-11-23 yet. `<root>` is
`/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training`, `<ckpt>` the staged checkpoint, e.g.
`hiera_u_hvit_t_cells_epoch25_1b75589f`. Commands run in `finetuning/v2/decoders`; the Slurm options used so far
are in `README.md` (search drivers and test arrays on `grete:preemptible`, 10 GB slice, `--requeue`).

1. Check the measured sizes: `sliver_sizes.json` has to be measured again after any change of test or tuning
   crops (`python ../evaluation/parameter_search.py prepare --sliver_sizes --threads 4 [-d <datasets>]`, 5
   minutes). It is current for the crops of this commit; a test in `test/test_parameter_search.py` checks the number
   of tuning crops it was measured on.
2. Search, per setting (`tune_decoder.py predict`, then `search`; or `submit` for both as Slurm jobs):
   - 3D: use a new session for a checkpoint that was searched before. The session of the epoch 25 checkpoint
     (`<root>/tuning/<ckpt>/3d`) holds predictions of the old crops for 7 datasets, which `tune_decoder.py`
     refuses, and sweeps scored with slivers. Either move that folder aside or pass `-e <new session>` to
     `predict` and `search` (`submit` has no such option). 105 tiles instead of 66, and the sliver scoring adds
     about 7 %: expect about 1.7 times the earlier search, which took 24 h on 8 threads plus 7 h on 64 threads.
   - 2D: the stored session can continue. `predict` adds `orgline` (515 images), `search` reuses the sweeps of the
     other 30 datasets, sweeps `orgline` and runs the later stages again where the best combinations change.
   - Dense: the stored session can continue. The sweeps of `liconn` and `xpress` are no longer read; the later
     stages run again where the best combinations change. The earlier search took 2 h.
   - A checkpoint that was not searched before needs nothing special.
3. Test of the checkpoint with the new parameters: `evaluate_decoder.py submit --checkpoint <staged checkpoint>
   -e <new folder> --data <setting> --params <session>/best_parameters.json`, one array per setting. Use a new
   evaluation folder: a stored `..._ais_tuned_...` result is taken as done whatever parameters it was computed
   with. Only the results of the volumes scored without slivers are recognised as outdated.
4. Reference (micro-sam2 hvit_t, library defaults): `evaluate_decoder.py reference submit` recomputes the 23
   datasets that `reference summary` lists (137 array tasks), then `reference summary`. Before using it, compare
   the column `mSA_with_slivers` of the recomputed results with the mSA of the copied results
   (`reference_hvit_t/sources.json` names them): the copies come from three registry runs
   (`v6_cells_registry_evaluation{,_crops32,_isotropic}`) with two checkpoint ids, and the recomputation has to
   reproduce them before the scores without slivers are compared.
5. Report: `evaluate_decoder.py report -e <new folder>`. It lists the datasets where only one of the two results
   is scored without slivers instead of comparing them.
6. Not part of this folder: the 3D light microscopy results of the baselines and of the other micro-sam2 models in
   the figure tables were scored with slivers. `evaluate_automatic_baselines.py` and
   `evaluate_automatic_segmentation.py` compute them again when they are run on the same folders
   (`common.result_is_current`); until then these tables and the new results differ in the scoring.
7. Optional: the diagnostics of the loss investigation (`<root>/evaluation/diag_epoch25_*`) were scored with
   slivers as well.

## Settings

| setting | segmentation | ranked by | tuning datasets | tuning samples |
|---|---|---|---|---|
| 2D | flow | mSA | 31 | 3609 |
| 3D | flow | mSA (13 datasets), SA50 (1 dataset) | 14 | 105 |
| dense (3D EM) | multicut | CREMI | 6 | 19 |

Tuning samples are images (2D) or crops of up to 32 x 512 x 512 (3D, dense). OOD datasets are never tuned on.

## Tuning data

### 2D (flow, mSA)

| dataset | tuning samples | ranked by | crops | test panel (ID) |
|---|---|---|---|---|
| `bitdepth_nucseg` | 8 | mSA | images | not in a panel |
| `bmgd` | 82 | mSA | images | fluorescence nuclei |
| `cell_acdc` | 13 | mSA | images | label-free cells |
| `cellbindb` | 55 | mSA | images | fluorescence nuclei |
| `cellular` | 8 | mSA | images | not in a panel |
| `cpm17` | 32 | mSA | images | histopathology nuclei |
| `cvz_fluo_cell` | 15 | mSA | images | fluorescence cells |
| `cvz_fluo_dapi` | 9 | mSA | images | not in a panel |
| `deepbacs_label_free` | 3 | mSA | images | not in a panel |
| `dynamicnuclearnet` | 500 of 1417 | mSA | images | fluorescence nuclei |
| `enseg` | 31 | mSA | images | not in a panel |
| `flywing` | 252 | mSA | images | fluorescence cells |
| `glysac` | 34 | mSA | images | histopathology nuclei |
| `histo_miner` | 212 | mSA | images | histopathology nuclei |
| `ifnuclei` | 5 | mSA | images | fluorescence nuclei |
| `livecell` | 540 | mSA | images | label-free cells |
| `microbeseg` | 5 | mSA | images | not in a panel |
| `monuseg` | 37 | mSA | images | histopathology nuclei |
| `neurips_cellseg_fluorescence` | 32 | mSA | images | fluorescence cells |
| `neurips_cellseg_label_free` | 69 | mSA | images | label-free cells |
| `omnipose_bact_fluor` | 15 | mSA | images | fluorescence cells |
| `omnipose_bact_phase` | 25 | mSA | images | label-free cells |
| `pannuke` | 500 of 505 | mSA | images | histopathology nuclei |
| `puma` | 26 | mSA | images | histopathology nuclei |
| `tissuenet` | 500 of 3118 | mSA | images | fluorescence cells |
| `tsakiroglou` | 4 | mSA | images | not in a panel |
| `u20s` | 20 | mSA | images | fluorescence nuclei |
| `vicar` | 55 | mSA | images | label-free cells |
| `astih` | 5 | mSA | images | not in a panel |
| `tumor_spheroid` | 2 | mSA | images | not in a panel |
| `orgline` | 515 | mSA | images | label-free cells |

### 3D (flow)

| dataset | tuning samples | ranked by | crops | test panel (ID) |
|---|---|---|---|---|
| `plantseg_root` | 10 | mSA | pinned tiles, placed like the test crops | fluorescence cells |
| `pnas_arabidopsis` | 20 | mSA | center crop per volume | fluorescence cells |
| `cartocell` | 9 | SA50 | pinned tiles | fluorescence cells |
| `phmamm` | 10 | mSA | center crop per volume | fluorescence cells |
| `wing_disc` | 2 | mSA | center crop per volume | fluorescence cells |
| `embedseg_organoid` | 11 | mSA | center crop per volume | fluorescence cells |
| `embedseg_mouse_skull` | 1 | mSA | center crop per volume | fluorescence nuclei |
| `embedseg_platy_nuclei` | 3 | mSA | pinned tiles, placed like the test crops | fluorescence nuclei |
| `nis3d` | 4 | mSA | pinned tiles, all tiles of the slab | fluorescence nuclei |
| `celegans_atlas` | 10 | mSA | pinned tiles, placed like the test crops | fluorescence nuclei |
| `gonuclear` | 10 | mSA | pinned tiles, placed like the test crops | fluorescence nuclei |
| `nucverse3d` | 10 | mSA | pinned tiles, placed like the test crops | fluorescence nuclei |
| `densecell` | 4 | mSA | pinned tiles, all tiles of the slab | not in a panel |
| `embedseg_platy_ish` | 1 | mSA | center crop per volume | not in a panel |

### Dense (multicut, CREMI score)

| dataset | tuning samples | ranked by | crops | test panel (ID) |
|---|---|---|---|---|
| `cremi` | 2 | CREMI | center crop per volume | EM neurites |
| `fafb` | 1 | CREMI | center crop per volume | EM neurites |
| `hemibrain` | 1 | CREMI | center crop per volume | EM neurites |
| `zebrafinch_j0126` | 1 | CREMI | center crop per volume | EM neurites |
| `platynereis_cells` | 6 | CREMI | pinned tiles | not in a panel |
| `synapsenet_compartments` | 8 | CREMI | pinned tiles | not in a panel |

`liconn` and `xpress` have validation data (one center crop each) but are not tuned on, see above.

## Test data

ID: the dataset (other images or volumes of it) is in the training data. OOD: it is not. "tuned in" names the pool
an ID dataset has tuning data in; "-" means it has none.

### Overview

| figure | panel | metric | ID datasets (samples) | of these tuned | OOD datasets (samples) |
|---|---|---|---|---|---|
| Fig. 2 (2D) | label-free cells | mSA | 6 (2095) | 6 | 6 (1514) |
| Fig. 2 (2D) | fluorescence nuclei | mSA | 6 (942) | 5 | 6 (1467) |
| Fig. 2 (2D) | histopathology nuclei | mSA | 6 (3047) | 6 | 6 (462) |
| Fig. 2 (2D) | fluorescence cells | mSA | 6 (1440) | 5 | 6 (173) |
| Fig. 3 (3D) | fluorescence cells | mSA / SA50 | 6 (367) | 6 | 3 (327) |
| Fig. 3 (3D) | fluorescence nuclei | mSA | 6 (92) | 6 | 6 (199) |
| Fig. 3 (3D) | EM neurites | CREMI | 6 (134) | 4 | 6 (72) |

Extended Data Fig. 1 uses the same datasets. The EM cell datasets are scored for the supplement only.

### Fig. 2: 2D

| panel | domain | dataset | test samples | segmentation | metric | tuned in |
|---|---|---|---|---|---|---|
| label-free cells | ID | `livecell` | 1510 | flow | mSA | 2D |
| label-free cells | ID | `neurips_cellseg_label_free` | 31 | flow | mSA | 2D |
| label-free cells | ID | `cell_acdc` | 12 | flow | mSA | 2D |
| label-free cells | ID | `vicar` | 55 | flow | mSA | 2D |
| label-free cells | ID | `omnipose_bact_phase` | 143 | flow | mSA | 2D |
| label-free cells | ID | `orgline` | 344 | flow | mSA | 2D |
| label-free cells | OOD | `deepseas` | 576 | flow | mSA | - |
| label-free cells | OOD | `bbbc010` | 20 | flow | mSA | - |
| label-free cells | OOD | `yeastsam` | 44 | flow | mSA | - |
| label-free cells | OOD | `cellapp` | 184 | flow | mSA | - |
| label-free cells | OOD | `bac_mother` | 534 | flow | mSA | - |
| label-free cells | OOD | `ecoli_microcolony_lineage` | 156 | flow | mSA | - |
| fluorescence nuclei | ID | `dsb` | 49 | flow | mSA | - |
| fluorescence nuclei | ID | `dynamicnuclearnet` | 717 | flow | mSA | 2D |
| fluorescence nuclei | ID | `bmgd` | 69 | flow | mSA | 2D |
| fluorescence nuclei | ID | `ifnuclei` | 37 | flow | mSA | 2D |
| fluorescence nuclei | ID | `u20s` | 15 | flow | mSA | 2D |
| fluorescence nuclei | ID | `cellbindb` | 55 | flow | mSA | 2D |
| fluorescence nuclei | OOD | `arvidsson` | 10 | flow | mSA | - |
| fluorescence nuclei | OOD | `hela_cytonuc` | 268 | flow | mSA | - |
| fluorescence nuclei | OOD | `covid_if_nuclei` | 44 | flow | mSA | - |
| fluorescence nuclei | OOD | `cardioblast_nuclei` | 30 | flow | mSA | - |
| fluorescence nuclei | OOD | `mif_tonsil` | 10 | flow | mSA | - |
| fluorescence nuclei | OOD | `micro_bench` | 1105 | flow | mSA | - |
| histopathology nuclei | ID | `glysac` | 25 | flow | mSA | 2D |
| histopathology nuclei | ID | `histo_miner` | 212 | flow | mSA | 2D |
| histopathology nuclei | ID | `monuseg` | 14 | flow | mSA | 2D |
| histopathology nuclei | ID | `cpm17` | 32 | flow | mSA | 2D |
| histopathology nuclei | ID | `puma` | 42 | flow | mSA | 2D |
| histopathology nuclei | ID | `pannuke` | 2722 | flow | mSA | 2D |
| histopathology nuclei | OOD | `cytodark0` | 27 | flow | mSA | - |
| histopathology nuclei | OOD | `pcns` | 272 | flow | mSA | - |
| histopathology nuclei | OOD | `khoshdeli` | 32 | flow | mSA | - |
| histopathology nuclei | OOD | `beacon` | 67 | flow | mSA | - |
| histopathology nuclei | OOD | `cd47_ihcnusc` | 16 | flow | mSA | - |
| histopathology nuclei | OOD | `amonuseg` | 48 | flow | mSA | - |
| fluorescence cells | ID | `tissuenet` | 1324 | flow | mSA | 2D |
| fluorescence cells | ID | `neurips_cellseg_fluorescence` | 19 | flow | mSA | 2D |
| fluorescence cells | ID | `flywing` | 42 | flow | mSA | 2D |
| fluorescence cells | ID | `omnipose_bact_fluor` | 17 | flow | mSA | 2D |
| fluorescence cells | ID | `cvz_fluo_cell` | 23 | flow | mSA | 2D |
| fluorescence cells | ID | `deepbacs_fluorescence` | 15 | flow | mSA | - |
| fluorescence cells | OOD | `covid_if_cells` | 44 | flow | mSA | - |
| fluorescence cells | OOD | `hpa` | 9 | flow | mSA | - |
| fluorescence cells | OOD | `medussa` | 51 | flow | mSA | - |
| fluorescence cells | OOD | `mif_tonsil_cells` | 10 | flow | mSA | - |
| fluorescence cells | OOD | `fusionx` | 7 | flow | mSA | - |
| fluorescence cells | OOD | `wing_disc_timelapse` | 52 | flow | mSA | - |

### Fig. 3: 3D

| panel | domain | dataset | test samples | segmentation | metric | tuned in |
|---|---|---|---|---|---|---|
| fluorescence cells | ID | `cartocell` | 68 | flow | SA50 | 3D |
| fluorescence cells | ID | `embedseg_organoid` | 11 | flow | mSA | 3D |
| fluorescence cells | ID | `phmamm` | 70 | flow | mSA | 3D |
| fluorescence cells | ID | `plantseg_root` | 98 | flow | mSA | 3D |
| fluorescence cells | ID | `pnas_arabidopsis` | 118 | flow | mSA | 3D |
| fluorescence cells | ID | `wing_disc` | 2 | flow | mSA | 3D |
| fluorescence cells | OOD | `cshaper` | 105 | flow | mSA | - |
| fluorescence cells | OOD | `morphonet_patiria` | 4 | flow | mSA | - |
| fluorescence cells | OOD | `plantseg_ovules` | 218 | flow | mSA | - |
| fluorescence nuclei | ID | `celegans_atlas` | 14 | flow | mSA | 3D |
| fluorescence nuclei | ID | `embedseg_mouse_skull` | 2 | flow | mSA | 3D |
| fluorescence nuclei | ID | `embedseg_platy_nuclei` | 2 | flow | mSA | 3D |
| fluorescence nuclei | ID | `gonuclear` | 18 | flow | mSA | 3D |
| fluorescence nuclei | ID | `nis3d` | 2 | flow | mSA | 3D |
| fluorescence nuclei | ID | `nucverse3d` | 54 | flow | mSA | 3D |
| fluorescence nuclei | OOD | `bbbc032` | 18 | flow | mSA | - |
| fluorescence nuclei | OOD | `bbbc033` | 4 | flow | mSA | - |
| fluorescence nuclei | OOD | `bbbc050` | 76 | flow | mSA | - |
| fluorescence nuclei | OOD | `blastospim` | 72 | flow | mSA | - |
| fluorescence nuclei | OOD | `morphonet_celegans` | 25 | flow | mSA | - |
| fluorescence nuclei | OOD | `nus_organoids` | 4 | flow | mSA | - |
| EM neurites | ID | `cremi` | 12 | multicut | CREMI | dense |
| EM neurites | ID | `fafb` | 44 | multicut | CREMI | dense |
| EM neurites | ID | `hemibrain` | 24 | multicut | CREMI | dense |
| EM neurites | ID | `liconn` | 25 | multicut | CREMI | - |
| EM neurites | ID | `snemi` | 4 | multicut | CREMI | - |
| EM neurites | ID | `zebrafinch_j0126` | 25 | multicut | CREMI | dense |
| EM neurites | OOD | `humanneurons` | 16 | multicut | CREMI | - |
| EM neurites | OOD | `ionsem` | 4 | multicut | CREMI | - |
| EM neurites | OOD | `isbi2012` | 1 | multicut | CREMI | - |
| EM neurites | OOD | `microns_basil` | 1 | multicut | CREMI | - |
| EM neurites | OOD | `nisb` | 25 | multicut | CREMI | - |
| EM neurites | OOD | `synapseweb` | 25 | multicut | CREMI | - |

### Supplement

| panel | domain | dataset | test samples | segmentation | metric | tuned in |
|---|---|---|---|---|---|---|
| EM cells | ID | `platynereis_cells` | 12 | multicut | CREMI | dense |
| EM cells | ID | `densecell` | 1 | flow | mSA | 3D |
| EM cells | ID | `synapsenet_compartments` | 8 | multicut | CREMI | dense |
| - | ID | `embedseg_platy_ish` | 2 | flow | mSA | 3D |
| X-ray neurites | ID | `xpress` | 1 | multicut | CREMI | - |
