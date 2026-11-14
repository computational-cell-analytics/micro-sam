# Finetuned Models

In addition to the original SAM2 models, we provide `micro-sam2` models that are finetuned on microscopy data.
They are downloaded automatically the first time you use them.

In the [annotation tools](#annotation-tools), you can select the model in the embedding menu: the `Model` dropdown selects the kind of model, and the model size dropdown (in the `Embedding Settings`) selects its size.

If you are using the [CLI](#using-the-command-line-interface-cli) or the [python library](#using-the-python-library), then you can specify the model via the `model_type` parameter (`-m` in the CLI), which is a single name composed of the model size and the kind of model, for example `hvit_t_cells`.

We provide the following models (`Model` dropdown name in brackets):
- The original SAM2 models (`Natural Images`): `hvit_t`, `hvit_s`, `hvit_b` and `hvit_l`. The letter after `hvit` is the size of the image encoder: tiny, small, base or large. These models support interactive segmentation and automatic mask generation (AMG), but not AIS or APG, because they have no decoder.
- The `micro-sam2` models for cells and nuclei in light microscopy and neurites in electron microscopy, in 2D and 3D (`Microscopy`): `hvit_t_cells`, `hvit_s_cells`, `hvit_b_cells` and `hvit_l_cells`.
- The `micro-sam2` model for mitochondria in electron microscopy (`Mitochondria (EM)`): `hvit_s_em_mito`.
- The `micro-sam2` model for nuclei in electron microscopy (`Nuclei (EM)`): `hvit_s_em_nuclei`.

The `micro-sam2` models support interactive segmentation and all automatic segmentation methods (AMG, AIS and APG, see [APG](#apg)). The default model is `hvit_t_cells`.

## Choosing a Model

As a rule of thumb:
- Use the `Microscopy` models (`hvit_*_cells`) for segmenting cells or nuclei in light microscopy and neurites in electron microscopy, in 2D and 3D.
    - `hvit_t_cells` is the fastest and the default. The larger models (`hvit_b_cells`, `hvit_l_cells`) are more accurate, especially for automatic segmentation, but need more computational resources.
- Use `hvit_s_em_mito` for segmenting mitochondria and `hvit_s_em_nuclei` for segmenting nuclei in electron microscopy.
- For other use-cases, try one of the `micro-sam2` models first, or one of the original SAM2 models (`hvit_b`, `hvit_l`) for interactive segmentation.
- If none of the models works well enough for your data, finetune a model on your own data, see [training your own model](#training-your-own-model).

## Models of the Previous Version

The SAM1 models of the previous version of `micro_sam` are still available in the `micro_sam.v1` subpackage and the `micro_sam v1` command line tools:
- The original Segment Anything models: `vit_h`, `vit_l`, `vit_b` and `vit_t` (from [Mobile SAM](https://arxiv.org/abs/2306.14289)).
- The light microscopy models from [Segment Anything for Microscopy](https://www.nature.com/articles/s41592-024-02580-4): `vit_l_lm`, `vit_b_lm` and `vit_t_lm`.
- The electron microscopy models: `vit_l_em_organelles`, `vit_b_em_organelles` and `vit_t_em_organelles`.
- The medical imaging model from [MedicoSAM](https://arxiv.org/abs/2501.11734): `vit_b_medical_imaging`.
- The histopathology models from [PathoSAM](https://arxiv.org/abs/2502.00408): `vit_h_histopathology`, `vit_l_histopathology` and `vit_b_histopathology`.

These models are also available in the [BioImage.IO Model Zoo](https://bioimage.io/#/).

### Older SAM1 Models

Older versions of the SAM1 models are available on Zenodo or on the BioImage.IO modelzoo:

#### v3 Models

<!---
It seems like there's no easy way to link to previous versions on BioImage.IO right now.
-->
An improved version of the light microscopy that were trained on a larger dataset compared to the v2 light microscopy models:
- [vit_t_lm](TODO): the ViT-Tiny model for segmenting cells and nuclei in LM.
- [vit_b_lm](TODO): the ViT-Base model for segmenting cells and nuclei in LM.
- [vit_l_lm](TODO): the ViT-Large model for segmenting cells and nuclei in LM.

#### v2 Models

The models at the point of the [publication](https://www.nature.com/articles/s41592-024-02580-4). The results reported in the paper refer to these models:
- [vit_t_lm](https://zenodo.org/records/11111329): the ViT-Tiny model for segmenting cells and nuclei in LM.
- [vit_b_lm](https://zenodo.org/records/11103798): the ViT-Base model for segmenting cells and nuclei in LM.
- [vit_l_lm](https://zenodo.org/records/11111177): the ViT-Large model for segmenting cells and nuclei in LM.
- [vit_t_em_organelles](https://doi.org/10.5281/zenodo.11110950): the ViT-Base model for segmenting mitochondria and nuclei in EM.
- [vit_b_em_organelles](https://doi.org/10.5281/zenodo.11111293): the ViT-Base model for segmenting mitochondria and nuclei in EM.
- [vit_l_em_organelles](https://doi.org/10.5281/zenodo.11111054): the ViT-Large model for segmenting mitochondria and nuclei in EM.

#### v1 Models

The initial models published with our preprint:
- [vit_b_em_boundaries](https://zenodo.org/records/10524894): for segmenting compartments delineated by boundaries such as cells or neurites in EM.
- [vit_b_em_organelles](https://zenodo.org/records/10524828): for segmenting mitochondria, nuclei or other organelles in EM.
- [vit_b_lm](https://zenodo.org/records/10524791): for segmenting cells and nuclei in LM.
- [vit_h_em](https://zenodo.org/records/8250291): for general EM segmentation.
- [vit_h_lm](https://zenodo.org/records/8250299): for general LM segmentation.

We do not recommend to use these models since our new models improve upon them significantly. But we provide the links here in case they are needed to reproduce older segmentation workflows.

We provide additional models that were used for experiments in our publication on Zenodo:
- [LIVECell Specialist Models](https://doi.org/10.5281/zenodo.11115426)
- [TissueNet Specialist Models](https://doi.org/10.5281/zenodo.11115998)
- [NeurIPS CellSeg Specialist Models](https://doi.org/10.5281/zenodo.11116407)
- [DeepBacs Specialist Models](https://doi.org/10.5281/zenodo.11115827)
- [PlantSeg (Root) Specialist Models](https://doi.org/10.5281/zenodo.11116603)
- [CREMI Specialist Models](https://doi.org/10.5281/zenodo.11117314)
- [ASEM (ER) Specialist Models](https://doi.org/10.5281/zenodo.11117144)
- [The LM Generalist Model with ViT-H backend (vit_h_lm)](https://doi.org/10.5281/zenodo.11117559)
- [The EM Generalist Model with ViT-H backend (vit_h_em_organelles)](https://doi.org/10.5281/zenodo.11117495)
- [Finetuned Models for the user studies](https://doi.org/10.5281/zenodo.11117615)
