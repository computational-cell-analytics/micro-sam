# Finetuned Models

In addition to the original Segment Anything models, we provide models that are finetuned on microscopy data, histopathology data and medical imaging.
They are available in the [BioImage.IO Model Zoo](https://bioimage.io/#/) and are also hosted on Zenodo.

In the [annotation tools](#annotation-tools), you can select the model from the top widget:
<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/model-type-selector.png" width="512">

You can use the `Model` dropdown (left side) to select which kind of model to use and the `model size` dropdown (right side, available after opening the `Embedding Settings` menu) to select the size of the model.
By default, the `base` size is used.

If you are using the [CLI](#using-the-command-line-interface-cli) or the [python library](#using-the-python-library), then you can specify the model to use via the `model_type` parameter, which consists of a single name composed of model type and size. See for example the [2d annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py#L62).

Specifically, we provide the following model types and sizes:
- The original Segment Anything Models (`Natural Images (SAM)`):
    - `vit_h`: Segment Anything model with ViT Huge image encoder.
    - `vit_l`: Segment Anything model with ViT Large image encoder.
    - `vit_b`: Segment Anything model with ViT Base image encoder.
    - `vit_t`: Segment Anything model with ViT Tiny image encoder. From [Mobile SAM](https://arxiv.org/abs/2306.14289).
- The light microscopy generalist models from [Segment Anything for Microscopy](https://www.nature.com/articles/s41592-024-02580-4) (`Light Microscopy`).
    - `vit_l_lm`: Model for cells and nuclei in light microscopy data with ViT Large image encoder. ([idealistic-rat on BioImage.IO](https://bioimage.io/#/?id=idealistic-rat))
    - `vit_b_lm`: Model for cells and nuclei in light microscopy data with ViT Base image encoder. ([diplomatic-bug on BioImage.IO](https://bioimage.io/#/?id=diplomatic-bug))
    - `vit_t_lm`: Model for cells and nuclei in light microscopy data with ViT Tiny image encoder. ([faithful-chicken BioImage.IO](https://bioimage.io/#/?id=faithful-chicken))
- The electron microscopy generalist models from [Segment Anything for Microscopy](https://www.nature.com/articles/s41592-024-02580-4) (`Electron Microscopy`).
    - `vit_l_em_organelles`: Model for mitochodria and nuclei in electron microscopy data with ViT Large image encoder. ([humorous-crab on BioImage.IO](https://bioimage.io/#/?id=humorous-crab))
    - `vit_b_em_organelles`: Model for mitochodria and nuclei in electron microscopy data with ViT Base image encoder. ([noisy-ox on BioImage.IO](https://bioimage.io/#/?id=noisy-ox))
    - `vit_t_em_organelles`: Model for mitochodria and nuclei in electron microscopy data with ViT Tiny image encoder. ([greedy-whale on BioImage.IO](https://bioimage.io/#/?id=greedy-whale))
- The medical imaging generalist models from [MedicoSAM](https://arxiv.org/abs/2501.11734) (`Medical Imaging`).
    - `vit_b_medical_imaging`: Model for medical imaging data with ViT Base image encoder.
- The histopathology generalist models from [PathoSAM](https://arxiv.org/abs/2502.00408) (`Histopathology`).
    - `vit_h_histopathology`: Model for nuclei in histopathology with ViT Huge image encoder.
    - `vit_l_histopathology`: Model for nuclei in histopathology with ViT Large image encoder.
    - `vit_b_histopathology`: Model for nuclei in histopathology with ViT Base image encoder.

See the two figures below of the improvements through the finetuned models for LM and EM data. 

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/lm_comparison.png" width="768">

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/em_comparison.png" width="768">

Note: if you have a `micro_sam` version older than v1.4.0, then the model selection dialogue in the [annotation tools](#annotation-tools) looks differently. In these versions, you have to select the model by its full name:
<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/model-type-selector-old.png" width="384">

## Choosing a Model 

As a rule of thumb:
- Use the `Light Microscopy` model (`vit_b_lm`) for segmenting cells or nuclei in light microscopy. The larger model (`vit_l_lm`) yields a bit better segmentation quality, especially for automatic segmentation, but needs more computational resources.
- Use the `Electron Microscopy` models (`vit_b_em_organelles` or `vit_l_em_organelles`) for segmenting mitochondria, nuclei or other roundish organelles in electron microscopy.
- Use the `Medical Imaging` model (`vit_b_medical_imaging`) for interactive segmentation of medical imaging data (X-Ray, CT, MRI, Ultrasound, etc.).
- Use the `Histopathology` models (`vit_b_histopathology` or `vit_l_histopathology`) for segmenting nuclei in histopathology images.
- For other use-cases, use one of the original SAM models (`SAM (Natural Images)`, `vit_b` or `vit_l`).
- We have not seen any advantages of using the largest model (`vit_h`), so in general don't recommend to use it as it needs significantly more resources to run.
- The `vit_t_...` models run much faster than other models, but yield inferior quality for many applications. It can still make sense to try them for your use-case if your working on a laptop and want to annotate many images or volumetric data. 

See also the figures above for examples where the finetuned models work better than the default models.
We are working on further improving these models and adding new models for other biomedical imaging domains.


## Other Models

Previous versions of our models are available on Zenodo or on the BioImage.IO modelzoo:

### v3 Models

<!---
It seems like there's no easy way to link to previous versions on BioImage.IO right now.
-->
An improved version of the light microscopy that were trained on a larger dataset compared to the v2 light microscopy models:
- [vit_t_lm](TODO): the ViT-Tiny model for segmenting cells and nuclei in LM.
- [vit_b_lm](TODO): the ViT-Base model for segmenting cells and nuclei in LM.
- [vit_l_lm](TODO): the ViT-Large model for segmenting cells and nuclei in LM.

### v2 Models

The models at the point of the [publication](https://www.nature.com/articles/s41592-024-02580-4). The results reported in the paper refer to these models:
- [vit_t_lm](https://zenodo.org/records/11111329): the ViT-Tiny model for segmenting cells and nuclei in LM.
- [vit_b_lm](https://zenodo.org/records/11103798): the ViT-Base model for segmenting cells and nuclei in LM.
- [vit_l_lm](https://zenodo.org/records/11111177): the ViT-Large model for segmenting cells and nuclei in LM.
- [vit_t_em_organelles](https://doi.org/10.5281/zenodo.11110950): the ViT-Base model for segmenting mitochondria and nuclei in EM.
- [vit_b_em_organelles](https://doi.org/10.5281/zenodo.11111293): the ViT-Base model for segmenting mitochondria and nuclei in EM.
- [vit_l_em_organelles](https://doi.org/10.5281/zenodo.11111054): the ViT-Large model for segmenting mitochondria and nuclei in EM.

### v1 Models

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
