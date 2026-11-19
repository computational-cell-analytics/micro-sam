# Evaluation Data References for the μSAM2 `_cells` Models

2D evaluation data of the automatic segmentation benchmark. In-domain (ID) datasets are the held-out test splits of training datasets. Out-of-domain (OOD) datasets are not in the main training panel.

† The dataset is a supplementary training dataset in `finetuning/v2/evaluation/common.py`. Its held-out split is scored in the OOD column.

| Modality | ID | OOD |
|---|---|---|
| Label-free cells | 6 | 6 |
| Fluorescence cells | 6 | 6 |
| Fluorescence nuclei | 6 | 6 |
| Histopathology nuclei | 6 | 6 |
| **Total** | **24** | **24** |


## Label-free Cells

### In-domain (6)

- **LIVECell** (`livecell`): [10.1038/s41592-021-01249-6](https://doi.org/10.1038/s41592-021-01249-6)
- **NeurIPS CellSeg 2022** (`neurips_cellseg_label_free`): [10.1038/s41592-024-02233-6](https://doi.org/10.1038/s41592-024-02233-6)
- **Cell-ACDC** (`cell_acdc`): [10.1186/s12915-022-01372-6](https://doi.org/10.1186/s12915-022-01372-6), data: [10.5281/zenodo.6795124](https://doi.org/10.5281/zenodo.6795124)
- **VICAR** (`vicar`): [10.1364/BOE.433212](https://doi.org/10.1364/BOE.433212)
- **Omnipose** (`omnipose_bact_phase`): [10.1038/s41592-022-01639-4](https://doi.org/10.1038/s41592-022-01639-4)
- **OrgLine** (`orgline`): [10.5281/zenodo.16355179](https://doi.org/10.5281/zenodo.16355179) — A compilation; also cite the source publications: [10.1038/s41598-019-48874-y](https://doi.org/10.1038/s41598-019-48874-y), [10.1038/s42003-024-05966-4](https://doi.org/10.1038/s42003-024-05966-4), [10.1038/s41597-024-03330-z](https://doi.org/10.1038/s41597-024-03330-z), [10.1038/s41598-023-46485-2](https://doi.org/10.1038/s41598-023-46485-2), [10.1371/journal.pcbi.1010584](https://doi.org/10.1371/journal.pcbi.1010584), [10.1007/s13402-024-00958-2](https://doi.org/10.1007/s13402-024-00958-2), [zenodo.org/records/18447547](https://zenodo.org/records/18447547)

### Out-of-domain (6)

- **DeepSea** (`deepseas`): [10.1016/j.crmeth.2023.100500](https://doi.org/10.1016/j.crmeth.2023.100500)
- **BBBC010** (`bbbc010`): [10.1038/nmeth.2083](https://doi.org/10.1038/nmeth.2083) — Broad Bioimage Benchmark Collection. C. elegans.
- **YeastSAM** (`yeastsam`): [10.1101/2025.09.17.676679](https://doi.org/10.1101/2025.09.17.676679), data: [10.5281/zenodo.17204942](https://doi.org/10.5281/zenodo.17204942)
    - Now published: https://doi.org/10.1091/mbc.E25-09-0454
- **TOIAM**† (`toiam`): [10.48550/arXiv.2411.00552](https://doi.org/10.48550/arXiv.2411.00552), data: [10.5281/zenodo.7260137](https://doi.org/10.5281/zenodo.7260137)
    - Now published: https://link.springer.com/chapter/10.1007/978-3-031-91721-9_20
- **mCellSeg**† (`mcellseg`): [10.1016/j.cmpb.2026.109470](https://doi.org/10.1016/j.cmpb.2026.109470), data: [10.5281/zenodo.20174259](https://doi.org/10.5281/zenodo.20174259)
    - CHECK: torch-em cites [10.1016/j.cmpb.2026.108919](https://doi.org/10.1016/j.cmpb.2026.108919) instead.
- **CISD**† (`cisd`): [10.1109/ISBI52829.2022.9761495](https://doi.org/10.1109/ISBI52829.2022.9761495)


## Fluorescence Cells

### In-domain (6)

- **TissueNet** (`tissuenet`): [10.1038/s41587-021-01094-0](https://doi.org/10.1038/s41587-021-01094-0)
- **NeurIPS CellSeg 2022** (`neurips_cellseg_fluorescence`): [10.1038/s41592-024-02233-6](https://doi.org/10.1038/s41592-024-02233-6)
- **FlyWing** (`flywing`): [10.1007/978-3-030-66415-2_21](https://doi.org/10.1007/978-3-030-66415-2_21), [10.1007/978-3-030-11024-6_33](https://doi.org/10.1007/978-3-030-11024-6_33), data: [10.5281/zenodo.5156991](https://doi.org/10.5281/zenodo.5156991)
- **Omnipose** (`omnipose_bact_fluor`): [10.1038/s41592-022-01639-4](https://doi.org/10.1038/s41592-022-01639-4)
- **CVZ-Fluo** (`cvz_fluo_cell`, cell stain): [10.1038/s41597-023-02108-z](https://doi.org/10.1038/s41597-023-02108-z)
- **DeepBacs** (`deepbacs_fluorescence`): [10.1038/s42003-022-03634-z](https://doi.org/10.1038/s42003-022-03634-z)

### Out-of-domain (6)

- **COVID-IF** (`covid_if_cells`): [10.1002/bies.202000257](https://doi.org/10.1002/bies.202000257), data: [10.5281/zenodo.5092850](https://doi.org/10.5281/zenodo.5092850)
- **HPA** (`hpa`): [10.1038/s41592-019-0658-6](https://doi.org/10.1038/s41592-019-0658-6), data: [10.5281/zenodo.4665863](https://doi.org/10.5281/zenodo.4665863)
- **MeDuSSA** (`medussa`): [10.1101/2025.10.26.684635](https://doi.org/10.1101/2025.10.26.684635), data: [S-BIAD2350](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2350) — Bacterial membranes.
    - Now published in https://www.nature.com/articles/s42003-026-10303-y
- **Multiplex IF Tonsil** (`mif_tonsil_cells`): data: [10.5281/zenodo.22107836](https://doi.org/10.5281/zenodo.22107836) — No publication; cite the Zenodo record.
- **FusionX** (`fusionx`): [10.1091/mbc.E25-11-0561](https://doi.org/10.1091/mbc.E25-11-0561), data: [10.5281/zenodo.18596465](https://doi.org/10.5281/zenodo.18596465)
- **Wing disc timelapse** (`wing_disc_timelapse`): [10.7554/eLife.57964](https://doi.org/10.7554/eLife.57964), [10.1242/dev.155069](https://doi.org/10.1242/dev.155069), data: [10.5281/zenodo.22760592](https://doi.org/10.5281/zenodo.22760592)


## Fluorescence Nuclei

### In-domain (6)

- **DSB 2018** (`dsb`, fluorescence images): [10.1038/s41592-019-0612-7](https://doi.org/10.1038/s41592-019-0612-7)
- **DynamicNuclearNet** (`dynamicnuclearnet`): [10.1101/803205](https://doi.org/10.1101/803205)
- **BMGD** (`bmgd`): [10.21203/rs.3.rs-8263420/v1](https://doi.org/10.21203/rs.3.rs-8263420/v1)
- **IFNuclei** (`ifnuclei`): [10.1038/s41597-020-00608-w](https://doi.org/10.1038/s41597-020-00608-w)
- **U2OS (BBBC039)** (`u20s`): [10.1002/cyto.a.23863](https://doi.org/10.1002/cyto.a.23863), [10.1038/nmeth.2083](https://doi.org/10.1038/nmeth.2083)
- **CellBinDB** (`cellbindb`, DAPI, ssDNA, mIF): [10.1093/gigascience/giaf069](https://doi.org/10.1093/gigascience/giaf069), preprint: [10.1101/2024.11.20.619750](https://doi.org/10.1101/2024.11.20.619750)

### Out-of-domain (6)

- **Arvidsson** (`arvidsson`): [10.1016/j.dib.2022.108769](https://doi.org/10.1016/j.dib.2022.108769), data: [10.5281/zenodo.6657260](https://doi.org/10.5281/zenodo.6657260)
- **HeLaCytoNuc** (`hela_cytonuc`): data: [10.14278/rodare.3001](https://doi.org/10.14278/rodare.3001) — No publication; cite the RODARE record.
- **COVID-IF** (`covid_if_nuclei`): [10.1002/bies.202000257](https://doi.org/10.1002/bies.202000257), data: [10.5281/zenodo.5092850](https://doi.org/10.5281/zenodo.5092850)
- **Cardioblast nuclei** (`cardioblast_nuclei`): [10.1083/jcb.202506102](https://doi.org/10.1083/jcb.202506102), data: [10.6019/S-BIAD1410](https://doi.org/10.6019/S-BIAD1410)
- **Multiplex IF Tonsil** (`mif_tonsil`): data: [10.5281/zenodo.22107836](https://doi.org/10.5281/zenodo.22107836) — No publication; cite the Zenodo record.

- **sPATCH** (`spatch_dapi`, Xenium and CosMx DAPI subsets): [10.1038/s41467-025-64292-3](https://doi.org/10.1038/s41467-025-64292-3) — The H&E subsets of the same tissue sections are training data.


## Histopathology Nuclei

### In-domain (6)

- **GLySAC** (`glysac`): [10.1109/JBHI.2022.3149936](https://doi.org/10.1109/JBHI.2022.3149936)
- **Histo-Miner** (`histo_miner`): [10.1371/journal.pcbi.1013907](https://doi.org/10.1371/journal.pcbi.1013907), data: [10.5281/zenodo.15973142](https://doi.org/10.5281/zenodo.15973142)
- **MoNuSeg** (`monuseg`): [10.1109/TMI.2017.2677499](https://doi.org/10.1109/TMI.2017.2677499)
- **CPM-17** (`cpm17`): [10.3389/fbioe.2019.00053](https://doi.org/10.3389/fbioe.2019.00053)
- **PUMA** (`puma`): [10.1093/gigascience/giaf011](https://doi.org/10.1093/gigascience/giaf011), data: [10.5281/zenodo.13859989](https://doi.org/10.5281/zenodo.13859989)
- **PanNuke** (`pannuke`): [10.1007/978-3-030-23937-4_2](https://doi.org/10.1007/978-3-030-23937-4_2), [10.48550/arXiv.2003.10778](https://doi.org/10.48550/arXiv.2003.10778)

### Out-of-domain (6)

- **cytoDArk0** (`cytodark0`): [10.1016/j.compbiomed.2025.111018](https://doi.org/10.1016/j.compbiomed.2025.111018), data: [10.5281/zenodo.13694738](https://doi.org/10.5281/zenodo.13694738) — Nissl stain.
- **PCNS** (`pcns`): [10.1038/s41597-020-0528-1](https://doi.org/10.1038/s41597-020-0528-1), data: [10.7937/TCIA.2019.4A4DKP9U](https://doi.org/10.7937/TCIA.2019.4A4DKP9U)
- **Khoshdeli** (`khoshdeli`): [10.1186/s12859-018-2285-0](https://doi.org/10.1186/s12859-018-2285-0), data: [10.6084/m9.figshare.6944522](https://doi.org/10.6084/m9.figshare.6944522)
- **Lizard-Mitosis**† (`lizard_mitosis`): No DOI ([PMLR v250](https://proceedings.mlr.press/v250/baumann24a.html))
- **LyNSeC**† (`lynsec_he`, H&E subset): [10.1016/j.compbiomed.2024.107978](https://doi.org/10.1016/j.compbiomed.2024.107978), data: [10.5281/zenodo.8065174](https://doi.org/10.5281/zenodo.8065174)
- **SRSA-Net**† (`srsanet`): [10.1016/j.bspc.2024.106143](https://doi.org/10.1016/j.bspc.2024.106143), data: [10.5281/zenodo.7647846](https://doi.org/10.5281/zenodo.7647846) — IHC.
