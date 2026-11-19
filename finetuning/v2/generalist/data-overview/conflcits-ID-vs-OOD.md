# μSAM2 manuscript: conflicts between training data and OOD evaluation data

Checked against: the manuscript (Google Doc, after the Zotero merge, 149 references) and the `paper-figures` branch of micro-sam (commit `09094cd`, 2026-11-01). Reference numbers are the current ones.

## The problem

In the Results section, the OOD datasets are described as datasets "which were not seen during training by any of the methods". **7 of the 24 OOD datasets in Fig. 2 are part of the μSAM2 training data.** In the manuscript, they are cited both in the training-data sentence and in the OOD citation groups.

| Fig. 2 panel | Dataset (code name) | Ref | Also cited in training group | Evidence in code |
|---|---|---|---|---|
| Label-free cells | TOIAM (`toiam`) | 76 (Seiffarth et al.) | cells, 48–76 | trained on train/val movies |
| Label-free cells | mCellSeg (`mcellseg`) | 68 (Alam et al.) | cells, 48–76 | trained on train/val split |
| Label-free cells | CISD (`cisd`) | 59 (Bouyssoux et al.) | cells, 48–76 | trained on train/val slides |
| Histopathology nuclei | Lizard-Mitosis (`lizard_mitosis`) | 108 (Baumann et al.) | histopathology, 95–108 | trained on train/val split |
| Histopathology nuclei | LyNSeC H&E (`lynsec_he`) | 102 (Naji et al.) | histopathology, 95–108 | trained on train/val split |
| Histopathology nuclei | SRSA-Net (`srsanet`) | 103 (Wang et al.) | histopathology, 95–108 | trained on train/val split |
| Fluorescence nuclei | SPATCH DAPI (`spatch_dapi`) | 107 (Ren et al.) | histopathology, 95–108 | SPATCH H&E subsets are trained; the DAPI sections share tissue with them |

Where this comes from in the code:

- `micro_sam/v2/datasets/generalist_loader.py` trains on all of the datasets above. For the first six it uses their train/val splits. For SPATCH it uses the H&E subsets (`SPATCH_HE_SUBSETS`).
- `scripts/plotting/v2/for_paper/plots_for_figure2.py` keeps them in the OOD panels (`PANEL_DATASETS`). Its own comments say so: *"toiam, mcellseg and cisd are in the training data of micro-sam2"* and *"lizard_mitosis, lynsec_he and srsanet are in the training data of micro-sam2"*.
- `finetuning/v2/evaluation/common.py` classifies the first six as **supplementary** (training) datasets. It classifies `spatch_dapi` as a **held-out platform**, not OOD, because it *"shares tissue sections with the sPATCH H&E training data"*.

Because these 7 datasets are test splits of training data (or share tissue with it), the six affected Fig. 2a OOD results are optimistic for μSAM2: label-free cells, fluorescence nuclei and histopathology nuclei, for both AIS and APG. The statement "μSAM2 was the best method in six out of the eight evaluations" may change.

## Possible fix

Use the OOD lists from `common.py`, which only contain datasets kept out of training.

| Panel | Keep | Replace with (from `common.py`) |
|---|---|---|
| Label-free cells | DeepSea, BBBC010, YeastSAM | CellApp (doi:10.1091/mbc.E25-02-0076), BacMother (doi:10.1371/journal.pcbi.1013071), E. coli microcolony lineage (doi:10.1016/j.cels.2018.03.009) |
| Histopathology nuclei | CytoDArk0, PCNS (Hou et al.), Khoshdeli | DeepLIIF (doi:10.1038/s42256-022-00471-x), PanopTILs (doi:10.1038/s41523-024-00663-1). Only 5 in total, so a sixth dataset is needed or the "six OOD datasets" wording must change |
| Fluorescence nuclei | Arvidsson, HeLaCytoNuc, COVID-IF, cardioblast, mIF tonsil | µBench (doi:10.52202/079017-0965) instead of SPATCH DAPI |

Alternatively, keep the 7 datasets but report them as ID or "held-out split" results. They can't stay under the current OOD definition.

After the change, still to do:

1. Re-run or re-plot the affected Fig. 2a panels (and Fig. 2c/d, which use the same data).
2. Update the OOD citation groups (59, 68, 76 / 102, 103, 108 / 107) and the dataset and image counts.
3. Re-check the "best in six out of eight evaluations" statement.

## Minor notes

- The ID and OOD lists in `plots_for_figure2.py` and `common.py` also differ elsewhere. For example, ID fluorescence cells is DeepBacs in the figure but EnSeg in `common.py`. Sync them before generating the final figures.
- Wing disc time-lapse (OOD, ref 140, Dye et al.) is a different dataset from the training datasets FlyWing (ref 61) and Wing disc 3D (ref 72, Paci et al.). It is not a conflict, but all three are Drosophila wing epithelium, so it is close to the training domain.
