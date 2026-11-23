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

Not done yet:
- The larger 3D tuning set of `tune_set.py` (up to 10 tiles per dataset) is not written to `tuning_crops.json`:
  the other 3D datasets still tune on one center crop per tuning volume (the counts below).
- The dense search of the new decoder includes `platynereis_cells` and `synapsenet_compartments`; the dense defaults
  of the previous decoder (the reference) were tuned on neurites only.
- The figure tables (collector script of the baselines) read mSA for cartocell.

## Settings

| setting | segmentation | ranked by | tuning datasets | tuning samples |
|---|---|---|---|---|
| 2D | flow | mSA | 31 | 3609 |
| 3D | flow | mSA (13 datasets), SA50 (1 dataset) | 14 | 66 |
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
| `plantseg_root` | 2 | mSA | center crop per volume | fluorescence cells |
| `pnas_arabidopsis` | 20 | mSA | center crop per volume | fluorescence cells |
| `cartocell` | 9 | SA50 | pinned tiles | fluorescence cells |
| `phmamm` | 10 | mSA | center crop per volume | fluorescence cells |
| `wing_disc` | 2 | mSA | center crop per volume | fluorescence cells |
| `embedseg_organoid` | 11 | mSA | center crop per volume | fluorescence cells |
| `embedseg_mouse_skull` | 1 | mSA | center crop per volume | fluorescence nuclei |
| `embedseg_platy_nuclei` | 1 | mSA | center crop per volume | fluorescence nuclei |
| `nis3d` | 1 | mSA | center crop per volume | fluorescence nuclei |
| `celegans_atlas` | 3 | mSA | center crop per volume | fluorescence nuclei |
| `gonuclear` | 1 | mSA | center crop per volume | fluorescence nuclei |
| `nucverse3d` | 3 | mSA | center crop per volume | fluorescence nuclei |
| `densecell` | 1 | mSA | center crop per volume | not in a panel |
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
