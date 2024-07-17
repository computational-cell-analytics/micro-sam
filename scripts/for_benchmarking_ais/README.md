# Benchmarking AIS (Automatic Instance Segmentation)

Targets: Distance-based approach

Combination of Models:
- UNet (trained from scratch)
- UNETR (ViT Large) (trained from scratch)
- UNETR (ViT Large) (pretrained SAM)
- SemanticSam (ViT Large) (trained from scratch)
- SemanticSam (ViT Large) (pretrained SAM)


Datasets:
- LIVECell (Entire dataset)
- Covid IF (Limited data setting)


## Results:

### LIVECell

| Model                     | mSA      |     SA50 |     SA75 |
|---------------------------|----------|----------|----------|
| UNet                      | 0.432179 | 0.713014 | 0.462311 |
| UNETR (vit_l) (scratch)   | 0.419043 | 0.701129 | 0.445928 |
| **UNETR (vit_l) (SAM)**   | 0.431117 | 0.71396  | 0.461748 |
| SemanticSam (vit_l) (SAM) | 0.422658 | 0.703621 | 0.45052  |

### Covid IF

| Model                     | Number of Images | mSA      |     SA50 |     SA75 |
|---------------------------|------------------|----------|----------|----------|
| UNet                      | 1                | | | |
| UNETR (vit_l) (scratch)   | 1                | | | |
| UNETR (vit_l) (SAM)       | 1                | | | |
| SemanticSam (vit_l) (SAM) | 1                | | | |
| UNet                      | 2                | | | |
| UNETR (vit_l) (scratch)   | 2                | | | |
| UNETR (vit_l) (SAM)       | 2                | | | |
| SemanticSam (vit_l) (SAM) | 2                | | | |
| UNet                      | 5                | | | |
| UNETR (vit_l) (scratch)   | 5                | | | |
| UNETR (vit_l) (SAM)       | 5                | | | |
| SemanticSam (vit_l) (SAM) | 5                | | | |
| UNet                      | 10               | | | |
| UNETR (vit_l) (scratch)   | 10               | | | |
| UNETR (vit_l) (SAM)       | 10               | | | |
| SemanticSam (vit_l) (SAM) | 10               | | | |