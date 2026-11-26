# Segment Anything Finetuning for Specialist Models

Code for finetuning Segment Anything on specific microscopy datasets.

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


## Experimental Scripts

- `training/histopathology/`: The finetuning scripts for histopathology datasets.
    - `pannuke_finetuning.py`: Finetuning Segment Anything on PanNuke datasets.


## Outdated Scripts
The scripts located at `outdated/` are not in working purpose with the latest version of `micro-sam`.
- It comprises of extensive experiments on "LIVECell" specialist, located at `outdated/livecell/`.