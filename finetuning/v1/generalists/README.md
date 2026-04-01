# Segment Anything Finetuning for Generalists Models

Code for finetuning Segment Anything on multiple microscopy datasets.

## Finetuning Scripts

- `training/`: The finetuning scripts for the two microscopy domains covered by `micro_sam`.
    - Light Microscopy (LM)
        - `light_microscopy/obtain_lm_datasets.py`: Scripts for getting the dataloader from multipe LM datasets.
        - `light_microscopy/train_lm_generalist.py`: Finetuning on multiple LM datasets.
    - Electron Microscopy (EM)
        - `mito_nuc/obtain_mito_nuc_em_datasets.py`: Scripts for getting the dataloader from multiple EM datasets.
        - `mito_nuc/train_mito_nuc_em_generalist.py`: Finetuning on multiple EM datasets for segmenting mitochondria and nuclei.


## Experimental Scripts
These scripts are a work-in-progress and often under active development.

- `training/histopathology`: Finetuning Segment Anything on Histopathology datasets.
- `training/electron_microscopy/boundaries`: Finetuning Segment Anything on Electron Microscopy datasets for segmenting boundary-based dense structures.

> For details on how to run the scripts from above: `python <DATASET>_finetuning.py -h`


## Outdated Scripts
The scripts located at `outdated/` are not in working purpose with the latest version of `micro-sam`.
