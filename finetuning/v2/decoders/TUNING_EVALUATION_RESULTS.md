# Post-processing search and test of the Hiera-U decoder (epoch 25): results

State: 2026-11-22. Commands, folders and job ids are in [README.md](README.md), the tuning data in
[TUNING_DATA.md](TUNING_DATA.md).

## What was compared

- **New decoder:** Hiera-U decoder on the frozen `hvit_t_cells` encoder, `best.pt` after 25 epochs (staged as
  `hiera_u_hvit_t_cells_epoch25_1b75589f`, checksum `1b75589f51a2aa4cfd7762e794a18dfb`). Its AIS post-processing parameters
  were searched on the tuning data, separately for 2D sparse, 3D sparse and 3D dense segmentation. The test uses
  the global winner of each setting; for the sparse settings `min_size` is taken per dataset from the tuning sweep
  (datasets without tuning data use the global value).
- **Reference:** the previous hvit_t decoder with the library defaults, no search (`evaluation/reference_hvit_t`).
- **Test data:** 85 datasets: 48 in 2D, 23 in 3D sparse, 14 in 3D dense. All have a result for both decoders.
- **Metrics:** mSA for sparse segmentation (higher is better; cartocell uses SA50), the CREMI score for dense
  segmentation (lower is better). Delta is new minus reference.

The comparison is not symmetric: the new decoder has searched parameters, the reference has defaults.

## Summary

| Setting | Group | Domain | New, tuned | Reference | Datasets | New better on |
|---|---|---|---|---|---|---|
| 2D, mSA | Label-free cells | ID | 0.425 | 0.446 | 6 | 3 |
| 2D, mSA | Label-free cells | OOD | 0.222 | 0.195 | 6 | 5 |
| 2D, mSA | Fluorescence nuclei | ID | 0.598 | 0.574 | 6 | 5 |
| 2D, mSA | Fluorescence nuclei | OOD | 0.470 | 0.471 | 6 | 4 |
| 2D, mSA | Histopathology nuclei | ID | 0.425 | 0.388 | 6 | 6 |
| 2D, mSA | Histopathology nuclei | OOD | 0.323 | 0.297 | 6 | 6 |
| 2D, mSA | Fluorescence cells | ID | 0.437 | 0.391 | 6 | 5 |
| 2D, mSA | Fluorescence cells | OOD | 0.339 | 0.308 | 6 | 4 |
| 3D, mSA | Fluorescence cells | ID | 0.494 | 0.471 | 5 | 4 (1 equal) |
| 3D, SA50 | Fluorescence cells (cartocell) | ID | 0.306 | 0.146 | 1 | 1 |
| 3D, mSA | Fluorescence cells | OOD | 0.279 | 0.276 | 3 | 2 |
| 3D, mSA | Fluorescence nuclei | ID | 0.339 | 0.360 | 6 | 3 |
| 3D, mSA | Fluorescence nuclei | OOD | 0.256 | 0.292 | 6 | 2 |
| Dense, CREMI score | EM neurites | ID | 0.858 | 1.080 | 6 | 3 |
| Dense, CREMI score | EM neurites | OOD | 1.055 | 0.964 | 6 | 0 |
| Dense, CREMI score | EM cells (supplement) | ID | 0.176 | 0.257 | 2 | 2 |
| 3D, mSA | densecell (supplement) | ID | 0.263 | 0.335 | 1 | 0 |
| 3D, mSA | embedseg_platy_ish (supplement) | ID | 0.339 | 0.288 | 1 | 1 |

- **2D:** the new decoder is better on 38 of 48 datasets and in 6 of the 8 group means. The two exceptions are
  label-free cells in-domain (one large loss, orgline) and fluorescence nuclei out-of-domain (practically equal;
  one larger loss, covid_if_nuclei).
- **3D cells:** better or equal on 8 of 9 datasets; the gains are small except for cartocell and plantseg_root.
- **3D nuclei:** behind the reference in both means. The deficit comes from three large losses
  (embedseg_mouse_skull, bbbc032, bbbc033); without embedseg_mouse_skull the in-domain mean is slightly ahead.
- **Dense, neurites:** better in-domain in the mean, but only through liconn and snemi; worse on all six
  out-of-domain datasets.
- **Dense, EM cells:** better on both datasets, which were part of the tuning data for this setting.

## Search results

| Setting | Tuning datasets | Combinations (`n_combos`) | Mean relative score of the winner | Lowest (dataset) | Same winner without one dataset |
|---|---|---|---|---|---|
| 2D | 30 | 42643 | 0.929 | 0.767 (cellular) | 28 of 30 |
| 3D | 14 | 30548 | 0.921 | 0.823 (densecell) | 9 of 14 |
| Dense | 8 | 1023 | 0.908 | 0.748 (fafb) | 6 of 8 |
| Dense, neurites only | 6 | 1027 | 0.903 | 0.749 (fafb) | 4 of 6 |

The relative score of a dataset compares its score with the global winner to its score with its own best
combination (1 means that the global winner is also the best combination of this dataset).

Winners:

| Parameter | 2D | 3D | Dense | Dense, neurites only |
|---|---|---|---|---|
| `foreground_threshold` | 0.5 | 0.4 | | |
| `foreground_weight` | 0.65 | 0.75 | | |
| `boundary_weight` | 0.25 | 0.5 | 0.75 | 0.75 |
| `boundary_mask_threshold` | 0.5 | none | | |
| `density_threshold` | 30 | 30 | 2.0 | 3.0 |
| `dt` | 1.0 | 1.0 | 1.0 | 1.0 |
| `n_iter` | 50 | 50 | 25 | 50 |
| `sigma` | 0.5 | 0.5 | 1.0 | 1.0 |
| `min_size` (global) | 50 | 500 | | |
| `beta` | | | 0.3 | 0.3 |

- The boundary channel is used in all three settings: as a weight in the watershed height map (2D and 3D), with a
  boundary mask in 2D only, and as a mix weight of 0.75 in the multicut boundary map (dense).
- The 2D winner is stable (same winner for 28 of 30 leave-one-out selections). The 3D and dense winners are less
  stable (9 of 14, 6 of 8).
- In the dense search the winner sits on the edge of the grid for `beta` (0.3) and `dt` (1.0) after the edge
  rounds, and 180 combinations were dropped because they did not have a score for every dataset.
- `min_size` per dataset, 2D: bitdepth_nucseg 100, bmgd 200, cell_acdc 50, cellbindb 10, cellular 1000, cpm17 50, cvz_fluo_cell 10, cvz_fluo_dapi 10, deepbacs_label_free 50, dynamicnuclearnet 25, enseg 50, flywing 10, glysac 50, histo_miner 25, ifnuclei 50, livecell 50, microbeseg 50, monuseg 50, neurips_cellseg_fluorescence 50, neurips_cellseg_label_free 200, omnipose_bact_fluor 25, omnipose_bact_phase 50, pannuke 25, puma 50, tissuenet 25, tsakiroglou 5, u20s 50, vicar 500, astih 50, tumor_spheroid 1000.
- `min_size` per dataset, 3D: plantseg_root 1000, pnas_arabidopsis 500, cartocell 200, phmamm 500, wing_disc 4000, embedseg_organoid 4000, embedseg_mouse_skull 2000, embedseg_platy_nuclei 200, nis3d 500, celegans_atlas 100, gonuclear 1000, nucverse3d 500, densecell 1000, embedseg_platy_ish 200.

## 2D results (mSA, higher is better)

### Label-free cells

| Dataset | Domain | New, tuned | Reference | Delta |
|---|---|---|---|---|
| livecell | ID | 0.360 | 0.302 | +0.058 |
| neurips_cellseg_label_free | ID | 0.303 | 0.290 | +0.012 |
| cell_acdc | ID | 0.732 | 0.762 | -0.030 |
| vicar | ID | 0.490 | 0.537 | -0.046 |
| omnipose_bact_phase | ID | 0.443 | 0.347 | +0.096 |
| orgline | ID | 0.224 | 0.438 | -0.214 |
| deepseas | OOD | 0.128 | 0.073 | +0.055 |
| bbbc010 | OOD | 0.125 | 0.127 | -0.003 |
| yeastsam | OOD | 0.391 | 0.368 | +0.023 |
| cellapp | OOD | 0.222 | 0.211 | +0.011 |
| bac_mother | OOD | 0.188 | 0.112 | +0.076 |
| ecoli_microcolony_lineage | OOD | 0.278 | 0.277 | +0.000 |

### Fluorescence nuclei

| Dataset | Domain | New, tuned | Reference | Delta |
|---|---|---|---|---|
| dsb | ID | 0.590 | 0.559 | +0.031 |
| dynamicnuclearnet | ID | 0.760 | 0.716 | +0.045 |
| bmgd | ID | 0.605 | 0.597 | +0.008 |
| ifnuclei | ID | 0.618 | 0.585 | +0.033 |
| u20s | ID | 0.731 | 0.702 | +0.028 |
| cellbindb | ID | 0.284 | 0.286 | -0.002 |
| arvidsson | OOD | 0.558 | 0.572 | -0.015 |
| hela_cytonuc | OOD | 0.179 | 0.157 | +0.021 |
| covid_if_nuclei | OOD | 0.556 | 0.650 | -0.095 |
| cardioblast_nuclei | OOD | 0.531 | 0.476 | +0.055 |
| mif_tonsil | OOD | 0.189 | 0.168 | +0.020 |
| micro_bench | OOD | 0.809 | 0.803 | +0.006 |

### Histopathology nuclei

| Dataset | Domain | New, tuned | Reference | Delta |
|---|---|---|---|---|
| glysac | ID | 0.349 | 0.324 | +0.025 |
| histo_miner | ID | 0.497 | 0.448 | +0.048 |
| monuseg | ID | 0.378 | 0.341 | +0.037 |
| cpm17 | ID | 0.430 | 0.387 | +0.043 |
| puma | ID | 0.496 | 0.457 | +0.039 |
| pannuke | ID | 0.399 | 0.370 | +0.028 |
| cytodark0 | OOD | 0.451 | 0.387 | +0.064 |
| pcns | OOD | 0.317 | 0.302 | +0.015 |
| khoshdeli | OOD | 0.313 | 0.280 | +0.032 |
| beacon | OOD | 0.307 | 0.294 | +0.013 |
| cd47_ihcnusc | OOD | 0.372 | 0.352 | +0.019 |
| amonuseg | OOD | 0.176 | 0.167 | +0.009 |

### Fluorescence cells

| Dataset | Domain | New, tuned | Reference | Delta |
|---|---|---|---|---|
| tissuenet | ID | 0.378 | 0.315 | +0.062 |
| neurips_cellseg_fluorescence | ID | 0.570 | 0.578 | -0.008 |
| flywing | ID | 0.698 | 0.575 | +0.123 |
| omnipose_bact_fluor | ID | 0.416 | 0.397 | +0.019 |
| cvz_fluo_cell | ID | 0.241 | 0.203 | +0.038 |
| deepbacs_fluorescence | ID | 0.319 | 0.280 | +0.040 |
| covid_if_cells | OOD | 0.256 | 0.263 | -0.007 |
| hpa | OOD | 0.361 | 0.382 | -0.021 |
| medussa | OOD | 0.304 | 0.276 | +0.027 |
| mif_tonsil_cells | OOD | 0.203 | 0.138 | +0.065 |
| fusionx | OOD | 0.335 | 0.280 | +0.055 |
| wing_disc_timelapse | OOD | 0.573 | 0.509 | +0.064 |

Largest losses in 2D: orgline (-0.214), covid_if_nuclei (-0.095), vicar (-0.046), cell_acdc (-0.030). Largest
gains: flywing (+0.123), omnipose_bact_phase (+0.096), bac_mother (+0.076).

## 3D results (mSA, higher is better; cartocell SA50)

### Fluorescence cells

| Dataset | Domain | New, tuned | Reference | Delta |
|---|---|---|---|---|
| cartocell | ID | 0.306 | 0.146 | +0.160 |
| embedseg_organoid | ID | 0.680 | 0.680 | +0.000 |
| phmamm | ID | 0.546 | 0.538 | +0.008 |
| plantseg_root | ID | 0.309 | 0.252 | +0.058 |
| pnas_arabidopsis | ID | 0.639 | 0.604 | +0.035 |
| wing_disc | ID | 0.294 | 0.280 | +0.014 |
| cshaper | OOD | 0.266 | 0.266 | +0.000 |
| morphonet_patiria | OOD | 0.280 | 0.266 | +0.014 |
| plantseg_ovules | OOD | 0.291 | 0.296 | -0.006 |

### Fluorescence nuclei

| Dataset | Domain | New, tuned | Reference | Delta |
|---|---|---|---|---|
| celegans_atlas | ID | 0.148 | 0.122 | +0.026 |
| embedseg_mouse_skull | ID | 0.364 | 0.542 | -0.177 |
| embedseg_platy_nuclei | ID | 0.376 | 0.357 | +0.019 |
| gonuclear | ID | 0.508 | 0.509 | -0.001 |
| nis3d | ID | 0.327 | 0.305 | +0.022 |
| nucverse3d | ID | 0.309 | 0.322 | -0.013 |
| bbbc032 | OOD | 0.131 | 0.302 | -0.171 |
| bbbc033 | OOD | 0.409 | 0.524 | -0.114 |
| bbbc050 | OOD | 0.374 | 0.262 | +0.112 |
| blastospim | OOD | 0.161 | 0.193 | -0.032 |
| morphonet_celegans | OOD | 0.021 | 0.042 | -0.021 |
| nus_organoids | OOD | 0.437 | 0.429 | +0.007 |

### Supplement datasets

| Dataset | Domain | New, tuned | Reference | Delta |
|---|---|---|---|---|
| densecell | ID | 0.263 | 0.335 | -0.072 |
| embedseg_platy_ish | ID | 0.339 | 0.288 | +0.051 |

- embedseg_organoid: the equal mSA is a coincidence (0.67974 and 0.67972); the other metrics differ.
- embedseg_mouse_skull was tested with its per-dataset `min_size` of 2000. Whether this explains the loss of
  0.177 is not checked.
- densecell is the dataset with the lowest relative score in the 3D search (0.823), and it is worse than the
  reference in the test.

## Dense results (CREMI score, lower is better)

**Old scoring.** The dense numbers of this document (this section, the dense rows of the summary tables and the
dense search) counted label zero as an object in the CREMI score, in the search and in the test, for every dataset
except platynereis_cells and synapsenet_compartments. Since 2026-11-27 label zero is ignored for all dense
datasets (`TUNING_DATA.md`, "Update 2026-11-27"). The epoch 25 results were not computed again.

### EM neurites

The last column is the test with the parameters of the search without the two EM cell datasets (next section).

| Dataset | Domain | New, tuned | Reference | Delta | Neurite-only tuning |
|---|---|---|---|---|---|
| cremi | ID | 0.697 | 0.603 | +0.094 | 0.667 |
| fafb | ID | 0.799 | 0.835 | -0.036 | 0.802 |
| hemibrain | ID | 0.366 | 0.256 | +0.109 | 0.377 |
| liconn | ID | 1.141 | 2.436 | -1.295 | 1.161 |
| snemi | ID | 0.799 | 1.007 | -0.208 | 0.804 |
| zebrafinch_j0126 | ID | 1.345 | 1.344 | +0.000 | 1.345 |
| humanneurons | OOD | 1.246 | 1.211 | +0.034 | 1.252 |
| ionsem | OOD | 0.629 | 0.430 | +0.199 | 0.623 |
| isbi2012 | OOD | 1.103 | 0.990 | +0.113 | 1.119 |
| microns_basil | OOD | 1.692 | 1.632 | +0.060 | 1.683 |
| nisb | OOD | 1.165 | 1.137 | +0.028 | 1.165 |
| synapseweb | OOD | 0.493 | 0.384 | +0.109 | 0.475 |

### EM cells (supplement)

| Dataset | Domain | New, tuned | Reference | Delta |
|---|---|---|---|---|
| platynereis_cells | ID | 0.295 | 0.396 | -0.101 |
| synapsenet_compartments | ID | 0.056 | 0.119 | -0.063 |

### Does tuning on the EM cell datasets hurt the neurite results?

Question: `platynereis_cells` and `synapsenet_compartments` contain very different objects than the neurite
datasets; does their presence in the dense tuning data lead to worse neurite results?

- The search was repeated with the six neurite tuning datasets only (cremi, fafb, hemibrain, zebrafinch_j0126,
  liconn, xpress). The winner only changes in `density_threshold` (3.0 instead of 2.0) and `n_iter` (50 instead
  of 25). The two cell datasets alone would prefer `beta` 0.4 and a boundary weight of 0.25, but they do not
  move the joint winner in these parameters.
- Test means with the neurite-only parameters: in-domain 0.859 (all eight tuning datasets: 0.858, reference
  1.080), out-of-domain 1.053 (1.055, reference 0.964).
- Per dataset the changes are small and go in both directions: cremi -0.030 and synapseweb -0.018 improve,
  liconn +0.020 and isbi2012 +0.017 get worse.
- So the two cell datasets do not explain the out-of-domain difference to the reference. The neurite datasets
  disagree among themselves (fafb has the lowest relative score in both searches).

## How the jobs were run, and what to change

Timeline: submitted 2026-11-21 02:26, last result 2026-11-22 22:57, so about 44 hours. Most of this time was
waiting for the scheduler and not computation.

| Step | Computation | Time until the result | Main cause of the difference |
|---|---|---|---|
| Dense search | 1 h 54 min (24 CPU threads) | 9 h | first queued for 7 h as a whole CPU node |
| 2D search | 22 h 51 min (40 threads, plus up to 4 x 16) | 30 h | same; and too few CPUs |
| 3D search | about 410 CPU hours | 40 h | 8 CPU threads for the first 24 h; CPU workers never started |
| 2D test | 2.9 h of task time, 82 tasks | 1 h 23 min after the search | |
| Dense test | 3.0 h of task time, 36 tasks | 1 h 34 min after the search | |
| 3D test | 7.2 h of task time, 137 tasks | 4 h 56 min after the search | 2 h behind a dependency that could not be met |

Mistakes in how the jobs were set up:

1. **Whole CPU nodes.** The searches were first submitted as 48 h jobs with a full node (192 threads) on
   `standard96s:shared`. With the fair share of the account they did not start in 7 hours. The later CPU worker
   arrays for the 3D search (10 x 16 and 20 x 8 threads) waited for 34 hours and never started; only the four
   2D workers with 16 GB started, after 45 minutes to 5 hours. One memory request was not measured
   (490 GB for tasks that need 1 to 2.5 GB each).
2. **3D search with too few CPUs.** After the move to `grete:preemptible` the 3D driver ran with 8 CPU threads
   while its CPU workers were pending. It reached its 24 h limit with 86 of 238 tasks. With the 64 threads that
   were used afterwards the rest of the first stage and all later stages took about 7 hours.
3. **Shards not sized by cost.** Every 3D tuning dataset was split into 16 shards, 308 tasks in total. A shard
   takes 6 to 7 hours for pnas_arabidopsis and 10 to 110 minutes for the other datasets in the first stage. The search therefore
   waited on the 16 pnas_arabidopsis tasks at the end.
4. **Too many small test tasks.** The test arrays have one task per dataset shard: 137 tasks for 3D (median
   2.6 minutes, longest 5.6), 82 for 2D (median 2.0 minutes), 36 for dense (median 5.0 minutes). Each task pays
   for the start of the environment and for loading the model, and each one has to be scheduled on its own; with
   8 tasks at once the 3D array took 2 h 48 min. A task of a few minutes is not worth a scheduler slot.
5. **Dependency on a job that did not start.** The 3D test array was chained with `afterok` to the driver job.
   The driver was preempted, its replacement got no GPU slice, and the search was finished by a driver on the
   interactive node. The tests then waited two hours on a job that had nothing left to do.
6. **Contention on `grete:preemptible`.** On Sunday afternoon interactive jobs held all 10 GB slices. Jobs waited
   for hours (the 33 dense test tasks took from 12:27 to 20:11 for 2.8 hours of task time), and running jobs
   were preempted.

For the next run:

- **Pool the test tasks.** One job should run for 30 to 60 minutes: one job per group of datasets that loads the
  model once and loops over the datasets and shards. For this run that would be about 8 jobs for 3D, 4 for 2D
  and 4 for dense, in place of 255 tasks.
- **Size the search shards by measured time.** Take the per-combination time of each dataset from this run
  (`tuning/<run>/<setting>/queue/claims`), give short datasets one or two shards and split only the long ones
  (pnas_arabidopsis above all) further, so that all tasks take about the same time.
- **Request small jobs with measured memory** (about 1 GB per running 2D task, 2 GB dense, 2.5 GB 3D), and enough
  of them from the start; do not count on `standard96s:shared` for this account.
- **Do not chain the tests to a preemptible driver.** Let the test job wait for `best_parameters.json` itself, or
  submit it when the search is done.
- **Run what fits on the interactive node there.** The dense search (22 minutes on 16 threads) and the dense and
  2D tests are small enough.

## Files

- Search sessions: `<root>/tuning/hiera_u_hvit_t_cells_epoch25_1b75589f/{2d,3d,dense,dense_neurites}/best_parameters.json`
- Test results and table: `<root>/evaluation/hiera_u_hvit_t_cells_epoch25_1b75589f/report_tuned.csv`
- Dense test with neurite-only parameters: `<root>/evaluation/hiera_u_hvit_t_cells_epoch25_1b75589f_dense_neurites/report_tuned.csv`
- Reference: `<root>/evaluation/reference_hvit_t/`
- `<root>` is `/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training`.

## Dense rerun of 2026-11-27 (epoch 29, CREMI score without label zero)

Not the epoch 25 checkpoint of the sections above: the Hiera-U decoder of epoch 29
(`hiera_u_hvit_t_cells_epoch29_9a812c61`) and the reference `hvit_t` (registry decoder `hvit_t_cells`). Label zero
is ignored for every dense dataset, in the search and in the test, and `liconn` and `xpress` are in the dense
tuning pool again (8 datasets, 21 crops; `TUNING_DATA.md`, "Update 2026-11-27"). Both models were searched with
the same tuning data, grid and stages. How the jobs ran: `README.md`, "Dense rerun of 2026-11-27".

### Search results

| Model | beta | boundary_weight | density_threshold | dt | n_iter | sigma | Combinations | Mean CREMI, 8 tuning datasets |
|---|---|---|---|---|---|---|---|---|
| Hiera-U epoch 29, before (old scoring, 6 datasets) | 0.5 | none | 2.0 | 0.5 | 25 | 0.5 | | |
| Hiera-U epoch 29, new | 0.4 | 0.25 | 2.0 | 0.5 | 25 | 0.5 | 1027 | 0.526 |
| hvit_t, library defaults until 2026-11-27 (v6 search, old scoring) | 0.5 | none | 3.0 | 0.5 | 50 | 0.5 | | |
| hvit_t, new (library defaults since 2026-11-27) | 0.4 | none | 3.0 | 0.5 | 100 | 0.25 | 1083 | 0.444 |

- Hiera-U: the result is at the edge of the grid for `boundary_weight` (0.25 is the lowest value of the dense
  boundary grid 0.25 - 1.0; values between 0 and 0.25 were not tested). The best combination without a boundary
  weight is the old winner, at rank 19 with 0.597. Worst tuning dataset: `xpress`. The search without one dataset
  gives the same result for 4 of the 8 datasets.
- hvit_t: not at an edge of the grid, no boundary weight in the result. Worst tuning dataset: `liconn`. The search
  without one dataset gives the same result for 2 of the 8 datasets.

### Test (CREMI score, lower is better)

"Label zero" is the share of the voxels of the test crops that is ignored (`ignored_fraction`). "hvit_t before"
is the superseded reference result with the old scoring (label zero scored as an object, except for
`platynereis_cells` and `synapsenet_compartments`), "Hiera-U before" the superseded epoch 29 result (old scoring,
old search result). "hvit_t defaults" is the test with the library defaults at the time of the run (beta 0.5,
sigma 0.5, n_iter 50). The result of the hvit_t search is the dense library default of `hvit_t` since 2026-11-27
(`micro_sam/_model_settings.py`), so "hvit_t tuned" is the score of the library defaults of today. The stored
reference results named `*_ais_default_*` of the dense datasets are the ones with the defaults from before; they
were not computed again, and `evaluate_decoder.py report` without `--reference_parameters tuned` compares with them.
The dense defaults of `hvit_s`, `hvit_b` and `hvit_l` are still from the search with the old scoring.

| Dataset | Panel | Domain | Label zero | hvit_t before | Hiera-U before | hvit_t defaults | hvit_t tuned | Hiera-U tuned |
|---|---|---|---|---|---|---|---|---|
| cremi | EM neurites | ID | 0.000 | 0.603 | 0.614 | 0.611 | 0.591 | 0.623 |
| fafb | EM neurites | ID | 0.000 | 0.835 | 0.712 | 0.837 | 0.752 | 0.718 |
| hemibrain | EM neurites | ID | 0.011 | 0.256 | 0.318 | 0.222 | 0.265 | 0.318 |
| liconn | EM neurites | ID | 0.427 | 2.436 | 2.425 | 0.487 | 0.387 | 0.370 |
| snemi | EM neurites | ID | 0.191 | 1.007 | 0.968 | 0.253 | 0.202 | 0.217 |
| zebrafinch_j0126 | EM neurites | ID | 0.194 | 1.344 | 1.355 | 0.519 | 0.506 | 0.497 |
| humanneurons | EM neurites | OOD | 0.217 | 1.211 | 1.215 | 0.154 | 0.120 | 0.138 |
| ionsem | EM neurites | OOD | 0.015 | 0.430 | 0.396 | 0.388 | 0.350 | 0.409 |
| isbi2012 | EM neurites | OOD | 0.001 | 0.990 | 1.163 | 0.997 | 1.012 | 1.103 |
| microns_basil | EM neurites | OOD | 0.221 | 1.632 | 1.638 | 0.520 | 0.487 | 0.546 |
| nisb | EM neurites | OOD | 0.179 | 1.137 | 1.104 | 0.369 | 0.260 | 0.313 |
| synapseweb | EM neurites | OOD | 0.403 | 0.384 | 0.447 | 0.947 | 0.963 | 1.046 |
| platynereis_cells | EM cells (supplement) | ID | 0.410 | 0.396 | 0.401 | 0.396 | 0.353 | 0.398 |
| synapsenet_compartments | EM cells (supplement) | ID | 0.072 | 0.119 | 0.058 | 0.119 | 0.141 | 0.058 |
| xpress | X-ray neurites (supplement) | ID | 0.527 | 2.144 | 2.403 | 1.524 | 1.170 | 1.746 |

Means per panel:

| Panel | Domain | Datasets | hvit_t defaults | hvit_t tuned | Hiera-U tuned |
|---|---|---|---|---|---|
| EM neurites | ID | 6 | 0.488 | 0.450 | 0.457 |
| EM neurites | OOD | 6 | 0.562 | 0.532 | 0.592 |
| EM cells (supplement) | ID | 2 | 0.257 | 0.247 | 0.228 |
| X-ray neurites (supplement) | ID | 1 | 1.524 | 1.170 | 1.746 |

- The tuning pool holds `cremi`, `fafb`, `hemibrain`, `zebrafinch_j0126`, `liconn`, `xpress`, `platynereis_cells`
  and `synapsenet_compartments` (other crops than the test crops); `snemi` and the OOD datasets are not in it.
- `synapseweb` is the one dataset with a higher score than before (0.384 to 0.947 for hvit_t with the library
  defaults): 40 % of its test crops is label zero, and with the old scoring this region counted as one object.
  Why the score without it is this high is not looked into.
- `isbi2012`, `microns_basil` and `xpress` have one test crop each, `ionsem` four.
- Check of the recomputation: the reference with the library defaults, scored with label zero (`cremi_with_zero`),
  is within 0.011 of the superseded result for 12 of 13 datasets (`xpress`, one crop: 0.032). It is not equal:
  the superseded results are from 2026-10-25 and 2026-11-02, computed with the code of that time, and the scores
  differ for every crop, also for `cremi` and `fafb`, which have no label zero. Which change of the code causes
  this is not looked into.

Files: `<root>/param_analysis/zero_label_20261127/compare_rerun.csv` (the table above),
`<root>/evaluation/hiera_u_hvit_t_cells_epoch29_9a812c61/report_tuned.csv` (all 86 datasets, reference with the
library defaults) and `report_tuned_vs_tuned_reference.csv` (the 15 dense datasets, reference with its own search
result), `<root>/tuning/{hiera_u_hvit_t_cells_epoch29_9a812c61,hvit_t_cells_registry_a469c3d0}/dense/best_parameters.json`.
