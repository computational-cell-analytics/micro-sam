# Finetuned Models

In addition to the original Segment Anything models, we provide models that are finetuned on microscopy data.
They are available in the [BioImage.IO Model Zoo](https://bioimage.io/#/) and are also hosted on Zenodo.

We currently offer the following models:

- `vit_h`: Default Segment Anything model with ViT Huge backbone.
- `vit_l`: Default Segment Anything model with ViT Large backbone.
- `vit_b`: Default Segment Anything model with ViT Base backbone.
- `vit_t`: Segment Anything model with ViT Tiny backbone. From the [Mobile SAM publication](https://arxiv.org/abs/2306.14289).
- `vit_l_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with ViT Large backbone. ([Zenodo](https://doi.org/10.5281/zenodo.11111176)) ([idealistic-rat on BioImage.IO](TODO))
- `vit_b_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with ViT Base backbone. ([Zenodo](https://zenodo.org/doi/10.5281/zenodo.11103797)) ([diplomatic-bug on BioImage.IO](TODO))
- `vit_t_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with ViT Tiny backbone. ([Zenodo](https://doi.org/10.5281/zenodo.11111328)) ([faithful-chicken BioImage.IO](TODO))
- `vit_l_em_organelles`: Finetuned Segment Anything model for mitochodria and nuclei in electron microscopy data with ViT Large backbone. ([Zenodo](https://doi.org/10.5281/zenodo.11111054)) ([humorous-crab on BioImage.IO](TODO))
- `vit_b_em_organelles`: Finetuned Segment Anything model for mitochodria and nuclei in electron microscopy data with ViT Base backbone. ([Zenodo](https://doi.org/10.5281/zenodo.11111293)) ([noisy-ox on BioImage.IO](TODO))
- `vit_t_em_organelles`: Finetuned Segment Anything model for mitochodria and nuclei in electron microscopy data with ViT Tiny backbone. ([Zenodo](https://doi.org/10.5281/zenodo.11110950)) ([greedy-whale on BioImage.IO](TODO))

See the two figures below of the improvements through the finetuned model for LM and EM data. 

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/lm_comparison.png" width="768">

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/em_comparison.png" width="768">

You can select which model to use in the [annotation tools](#annotation-tools) by selecting the corresponding name in the `Model:` drop-down menu in the embedding menu:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/model-type-selector.png" width="256">

To use a specific model in the python library you need to pass the corresponding name as value to the `model_type` parameter exposed by all relevant functions.
See for example the [2d annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py#L62).


## Choosing a Model 

As a rule of thumb:
- Use the `vit_l_lm` or `vit_b_lm` model for segmenting cells or nuclei in light microscopy. The larger model (`vit_l_lm`) yields a bit better segmentation quality, especially for automatic segmentation, but needs more computational resources.
- Use the `vit_l_em_organelles` or `vit_b_em_organelles` models for segmenting mitochondria, nuclei or other  roundish organelles in electron microscopy.
- For other use-cases use one of the default models.
- The `vit_t_...` models run much faster than other models, but yield inferior quality for many applications. It can still make sense to try them for your use-case if your working on a laptop and want to annotate many images or volumetric data. 

See also the figures above for examples where the finetuned models work better than the default models.
We are working on further improving these models and adding new models for other biomedical imaging domains.


## Other Models

Previous versions of our models are available on Zenodo:
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
