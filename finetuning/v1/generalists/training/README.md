# Segment Anything Finetuning for Multiple Datasets (Generalists)

Code for finetuning segment anything data on multiple microscopy datasets.

## Finetuning Scripts

- `training/`: The finetuning scripts for different microscopy datasets.
    - `light_microscopy/`
        - `obtain_lm_datasets.py`: Script to get the dataloaders for multiple light microscopy datasets. See [here](https://github.com/computational-cell-analytics/micro-sam/blob/master/doc/bioimageio/lm_v2.md) for details on the datasets used.
        - `train_lm_generalist.py`: Finetuning on Light Microscopy datasets.
    - `electron_microscopy/`
        - `mito_nuc/`
            - `obtain_mito_nuc_em_datasets.py`: Scripts to get the dataloaders for multiple electron microscopy datasets for mitochondria and nuclei segmentation. See [here](https://github.com/computational-cell-analytics/micro-sam/blob/master/doc/bioimageio/em_organelles_v2.md) for details on the datasets used.
            - `train_mito_nuc_em_generalist.py`: Finetuning on Electron Microscopy datasets for Mitochondria and Nuclei.

> For details on how to run the scripts from above: `python <GENERALIST_FINETUNING_SCRIPT>.py -h`

## Experimental Scripts (Work-in-Progress)

> NOTE: The scripts below are very experimental and might not be up-to-date with the latest micro-sam features.

- `training/`:
    - `electron_microscopy/`
        - `boundaries/` (WIP)
            - `obtain_boundaries_em_datasets.py`: Scripts to get the dataloaders for multiple electron microscopy datasets for boundary segmentation.
            - `train_mito_boundaries_generalist.py`: Finetuning on Electron Microscopy datasets for Boundary Structures.
    - `histopathology` (WIP)
        - `obtain_hp_datasets.py`: Scripts to get the dataloaders for multiple histopathology datasets.
        - `train_histopathology_generalist.py`: Finetuning on Histopathology datasets.
