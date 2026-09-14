# μSAM2 Finetuning for Specialist Models

Code for finetuning SAM2 on one microscopy dataset per task. Every task folder holds a 2d and a 3d script, with the same data design: LIVECell for 2d, Lucchi for 3d.

## Finetuning Scripts

- `interactive_segmentation/`: Finetuning SAM2 with its native prompting, for point and box prompts with correction clicks.
    - `train_livecell_2d.py`: Finetuning on LIVECell data.
    - `train_lucchi_3d.py`: Finetuning on Lucchi data.
- `instance_segmentation/`: Finetuning UniSAM2, a UNETR decoder on the SAM2 image encoder, for automatic instance segmentation.
    - `train_livecell_2d.py`: Finetuning on LIVECell data.
    - `train_lucchi_3d.py`: Finetuning on Lucchi data.
- `semantic_segmentation/`: Finetuning SemanticSAM2, the same decoder with a three class head: background, object boundary and object interior.
    - `train_livecell_2d.py`: Finetuning on LIVECell data.
    - `train_lucchi_3d.py`: Finetuning on Lucchi data.
    - `run_inference_livecell_2d.py`: Prediction and scoring on the LIVECell test split.
    - `run_inference_lucchi_3d.py`: Prediction and scoring on the Lucchi test volume.

> For details on how to run the scripts from above: `python <SCRIPT>.py -h`

## Notes

- The semantic segmentation model, loss, training and inference live in `micro_sam.v2`. The scripts here only build the dataloaders.
- Lucchi ships a train and a test volume only, so the test volume validates.
- The semantic loss follows `medico-sam`: `dice_weight * dice + (1 - dice_weight) * cross entropy`.
