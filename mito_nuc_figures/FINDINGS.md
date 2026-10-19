# Findings for choosing mito / nucleus training data

Date: 2026-10-15. torch-em at main `6c25def2`. 39 torch-em EM loaders with mito and/or nucleus labels plus 194 CellMap crops.
Dataset folders are grouped into `mito_and_nucleus/` (8), `mito_only/` (24) and `nucleus_only/` (8). Every dataset folder has an `info.json` with per-volume shapes, voxel sizes, boxes and visual QA notes; this file collects what matters for the decision.
The tiers below are a draft to discuss, not a decision.

## Draft tiers

### Mitochondria

| Tier | Datasets |
|---|---|
| A: dense, manual, aligned | mitoem (human + rat train/val), mitoemv2 (all except jurkat_train02 and pyra test01), cellmap mito crops, cem (MitoLab 2D + 6 vEM benchmarks), lucchi, kasthuri (mask -1), uro_cell (5 of 8 volumes), cmito, asem 46/58/64/2E, mito_segem testis + brainstem, parlakgul_liver (semantic), densecell (small, semantic), vnc (20 slices) |
| B: usable with care | mitonet_predicted_kidney (pseudo-labels, but look excellent), hemibrain (automated, blocky at 16 nm), microns pinky (only the stitched vol19-34 is dense, the rest need `volumes/mask`), mito_segem intestine (80-90%), tnbc_mito (2D, low-mag images sparse), deepcontact (drop images without mito polygons), hela_mito (only the 5 correct slices, central cell only), betaseg / human_liver_em / human_organoids / waenet ds3 (see sparse annotation), probtem (only 3 labeled images local) |
| Cryo-ET / X-ray | popsicle (= pombe_nucleus_mito, partial per tomogram), npc1_mito (pseudo-label), sxt_ins1e_mito (soft X-ray, 55 cells, semi-automatic) |
| C: exclude | drg_axon_mito and surface_morphometrics (membranes, not filled mitos), mito_anf (about half of the mitos unlabeled), mitoemv2 jurkat_train02 (labels do not match raw), asem 61 / 3A (mostly unlabeled) |

### Nucleus

| Tier | Datasets |
|---|---|
| A: manual, exhaustive | cellmap nuc crops (minus the 15 crops fully inside a nucleus), platynereis nuclei (crop to the annotated box), tumor_spheroid_em 2d_manual (20 independent slices), waenet ds1 (FIB-SEM stack 323x512x896), human_organoids, janelia_nucleus nacc-2/3/4 (1-3 nuclei, border-cut) |
| B: automatic but near-exhaustive | fafb_nuclei, microns_nuclei (worse raw: stripes, folds), janelia_nucleus (13 Cellpose volumes), openorganelle_nucleus (filled-mask volumes, `labels/inference`) |
| Single target cell only | betaseg, human_liver_em (2-8 nuclei per slice), waenet ds4-7, sxt_ins1e_mito (X-ray) |
| C: exclude / fix first | cefa_hela (loader crop misaligned, needs registration), openorganelle ctl-id8-2 (envelope band labeled as nucleus) and liver-4/5/6 (nuclear membrane), nuc_mm zebrafish (labels incomplete), nuc_mm mouse (X-ray, not EM), waenet ds2 (chromosomes, no nucleus) |

## Label problems that silently break training

- **mitoemv2 me2-jurkat_train02**: the labels lie in empty resin and do not match the raw. Exclude it.
- **hela_mito**: 4 of 9 files (z0002/z0004/z0026/z0028) are second annotations of raw z0030/z0060/z0116/z0150. The loader pairs them with the wrong raw slices.
- **surface_morphometrics**: the labels of all float32 tomograms are flipped along y relative to the raw, and the loader never flips them. TF3 and UE8 have broken labels.
- **cefa_hela**: the loader takes the center 2000x2000 crop, but the labels sit at about y=5050, x=5580. In the center crop they do not match the raw at all.
- **human_organoids**: raw and labels are int8, with -128 as background and 127 as foreground. Use `labels > 0`.
- **platynereis nuclei**: -1 marks a large unlabeled halo (29-45% of voxels), which the loader passes through. Use `rois` or a mask.
- **kasthuri**: -1 marks the padding (43% of train voxels), which the loader does not mask.
- **microns**: the uint8 ids reach 255 in the stitched volumes, so ids may collide.
- **parlakgul_liver, betaseg**: labels are 0/255, and touching mitos merge (semantic).
- **openorganelle_nucleus ctl-id8-2** is a nuclear-envelope band, and **liver-4/5/6** are `nuc_mem` traces that vanish at 128 nm or coarser. Both are declared as nucleus.
- **waenet**: dataset 2 holds chromosomes only (its docstring is swapped with dataset 3's), and the train/test split sorts names alphabetically.

## Sparse / partial annotation (unlabeled does not mean background)

- betaseg, hela_mito and waenet ds4-7 label only the central target cell.
- human_liver_em labels mito only in the upper ~70% of each slice, and nuclei sparsely.
- microns pinky: everything except the stitched vol19-34 labels one or a few cells only.
- mito_anf labels only mitos of traced fibers (about half). mito_segem intestine is 80-90% complete.
- popsicle / pombe: classes and z-ranges are annotated per tomogram only (table in `mito_and_nucleus/popsicle/info.json`). The loss must be restricted per tomogram.
- **cellmap**:
  - "Annotated empty" can be wrong: the sum159-4 crop211 mito array is all zero although mitos are visible.
  - 15 nucleus crops lie fully inside a nucleus: hela-2 94/95/96, hela-3 85/86/87, jurkat 91/92/93, macrophage-2 88/89/90, sum159-1 97/98/99.
  - Near-duplicate and nested crops exist (listed in `mito_and_nucleus/cellmap/info.json`). Split by region, not by crop id.
- tnbc_mito: low-magnification images often label one cell or region only.
- deepcontact: 15 of 109 cell and 17 of 73 sem images have no mito polygons although mitos are visible.

## Duplicates and leakage

- **pombe_nucleus_mito masks are byte-identical to popsicle** TS_0006 (train) and TS_045 (test). Use only one of the two.
- **cem MitoLab contains tiles of the public benchmarks**: MitoEM-H/R (6,583 images, 30%), UroCell, kasthuri2011, CREMI, the OpenOrganelle jrc_* cells, perez2014, and Guay2020 (likely DenseCell). This leaks into any test set drawn from those.
- `get_mitolab_paths` splits train/val randomly by image, so neighboring tiles of one volume land in both splits.
- MitoEM: the loader puts slice 499 in both val and test (test has 501 slices).
- microns: the stitched vol19-34 exists twice, as original and `_realigned`.
- tumor_spheroid_em: the 100 nm slices are the 50 nm slices downsampled, so there are only 20 independent images.
- platynereis_nuclei/vframes is a frame export of platynereis train_data_nuclei_01 raw. It is not a separate dataset.

## torch-em loader bugs found

- `mitoem.py:89/107`: the test volume starts at the last val slice.
- `mitoemv2.py`: reads `split.json`, but pyra ships `splits.json`. The n5 conversion loads whole volumes into memory (pyra test01 needs ~67 GB).
- `hela_mito.py:26-36`: wrong raw slice for the GT2 files.
- `surface_morphometrics.py`: no y-flip for float32 tomograms. TF3 and UE8 are broken.
- `drg_axon_mito.py` RUNS lines 42/47/49/72/81/82/85/87: the inner-membrane scale is `-2.0`, but the portal only has `-1.0`. The download crashes at the 5th run.
- `cefa_hela.py`: the center crop is misaligned, and the Zenodo `files-archive` returns 404. It also needs `ascp` and `ncempy`, which are not installed.
- `parlakgul_liver.py:72/88/104/186`: only 6461 works. The raw names, folders and zip names are wrong for the other three, and zip members are sorted alphabetically, which breaks z order.
- `asem.py:39-47/72`: the bucket layout changed (`<id>/<id>.zarr`), and quilt3 is called even with `download=False` (quilt3 is not installed).
- `cmito.py:102`: l3 boxes are converted with the wrong mip (6.144 vs 8 nm), so they are misaligned and the default box is cut by 23%.
- `janelia_nucleus.py:92`: raises for jrc_mus-pancreas-1 (no `em` group). A workaround cache in the loader's own format now exists.
- `cellmap.py:237-238/267-269`: labels are not clipped when the crop exceeds the EM extent (cos7-1a crop254). The int16 cast can overflow uint16 ids, and 255 ("unknown") stays a normal id.
- `betaseg.py:128`: iterates `label_choices`, so passing a list of choices raises UnboundLocalError.
- `nuc_mm.py:27`: the zebrafish part cannot be built, because only 7/27 train and 0/27 val labels exist and the gdrive download skips existing folders.
- `probtem.py:~80`: returns as soon as any test h5 exists. The local copy has no train split and 3 of 6 test targets.
- `waenet.py:6-7/127/169/205`: swapped docstring, per-image class remap by gray-value order, deleted zip, and an alphabetical split.
- `sxt_ins1e_mito.py:66/94`, `pombe_nucleus_mito.py:86`: these create an empty zarr store when `download=False`.
- `mito_segem.py:72`: downloads full slices serially, so one brainstem box takes about 2.5 h. The figure script uses range reads (~11 min).
- `openorganelle_nucleus.py`: has no retry on S3 SSL drops. hippocampus-1 label s1 has corrupt chunks (s0 and s2 are fine). Cropped volumes are trained from a single crop, although hippocampus-1, pancreas-4 and skin-1 hold thousands of nuclei in only 3-6 GB.

## Environment notes for training on these nodes

- The aiohttp-based fsspec HTTP ignores the GWDG proxy. Set `fsspec.config.conf["https"] = {"client_kwargs": {"trust_env": True}}` (same for "http") before using mito_anf, mitonet_predicted_kidney, sxt_ins1e_mito and other HTTP-zarr loaders. s3fs, requests and cloudvolume work without it.
- Not writable for u28048 (owned by u12090), so new data went to sibling folders:
  - `deepcontact_redownload/`: the original zips are broken (0 bytes / a 504 HTML page).
  - `parlakgul_liver_boxes/`
  - `human_liver_em`: small figure cache in `mito_nuc_figures/mito_and_nucleus/human_liver_em/cache`.
- `probtem_redownload/` holds only `__MACOSX` stubs (Drive access denied) and can be deleted.

## New data downloaded into cca/data (2026-10-15)

| Dataset | Size | What was downloaded |
|---|---|---|
| mitoemv2 | 15.9 GB | 7 missing zips, unzipped, zips kept |
| sxt_ins1e_mito | 15 GB | |
| drg_axon_mito | 9.6 GB | 6 full runs + 57 coarse overviews |
| surface_morphometrics | 8.4 GB | 8 of 34 tomograms |
| tnbc_mito | 6.1 GB | full |
| openorganelle_nucleus | 3.2 GB | |
| deepcontact_redownload | 2.9 GB | |
| janelia_nucleus | 1.9 GB | |
| npc1_mito | 1.5 GB | |
| pombe_nucleus_mito | 0.7 GB | |
| asem/mito_crops | 0.7 GB | |
| cmito | 0.6 GB | |
| mitonet_predicted_kidney | 0.35 GB | |
| cefa_hela | 0.2 GB | |
| mito_segem | 0.17 GB | |
| mito_anf | 0.18 GB | |
| parlakgul_liver_boxes | 77 MB | |
| hela_mito | 26 MB | |

The exact boxes and scales used for the streamed datasets are in each `info.json`, so they can be reused for training.
