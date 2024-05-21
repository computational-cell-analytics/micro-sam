# Light Microscopy Datasets

The `LM Generalist v2` model was trained on seven different light microscopy datasets with segmentation annotations for cells and nuclei:

1. [LIVECell](https://sartorius-research.github.io/LIVECell/): containing cell segmentation annotations for phase-contrast microscopy.
2. [DeepBacs](https://github.com/HenriquesLab/DeepBacs): containing segmentation annotations for bacteria in different label-free microscopy modalities.
3. [TissueNet](https://datasets.deepcell.org/): containing cell segmentation annotations in tissues imaged with fluorescence light microscopy.
4. [PlantSeg (Root)](https://osf.io/2rszy/): containing cell segmentation annotations in plant roots imaged with fluorescence lightsheet microscopy.
5. [NeurIPS CellSeg](https://neurips22-cellseg.grand-challenge.org/): containg cell segmentation annotations in phase-contrast, brightfield, DIC and fluorescence microscopy.
6. [CTC (Cell Tracking Challenge)](https://celltrackingchallenge.net/2d-datasets/): containing cell segmentation annotations in different label-free and fluorescence microscopy settings. We make use of the following CTC datasets: `BF-C2DL-HSC`, `BF-C2DL-MuSC`, `DIC-C2DH-HeLa`, `Fluo-C2DL-Huh7`, `Fluo-C2DL-MSC`, `Fluo-N2DH-SIM+`, `PhC-C2DH-U373`, `PhC-C2DL-PSC"`]
7. [DSB Nucleus Segmentation](https://www.kaggle.com/c/data-science-bowl-2018): containing nucleus segmentation annotations in fluorescence microscopy. We make use of [this subset](https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip) of the data.
