# Segment Anything Finetuning for Specific Datasets

Code for finetuning segment anything data on specific microscopy datasets.

## Finetuning Scripts

- `training/`: The finetuning scripts for different microscopy datasets.
    - `light_microscopy/`
        - `deepbacs_finetuning.py`: Finetuning on DeepBacs data.
        - `tissuenet_finetuning.py`: Finetuning on TissueNet data.
        - `plantseg_root_finetuning.py`: Finetuning on PlantSeg (Root) data.
        - `neurips_cellseg_finetuning.py`: Finetuning on NeurIPS Cell Segmentation data.
    - `electron_microscopy/`
        - `organelles/asem_finetuning.py`: Finetuning on ASEM data.
        - `boundaries/cremi_finetuning.py`: Finetuning on CREMI data.

> For details on how to run the scripts from above: `python <DATASET>_finetuning.py -h`

- `resource_efficient_finetuning`: The experiments for finetuning a custom dataset on limited resources.
