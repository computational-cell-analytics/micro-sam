# Finetuned models

In addition to the original Segment Anything models, we provide models that are finetuned on microscopy data.
The additional models are available in the [bioimage.io modelzoo](https://bioimage.io/#/) and are also hosted on zenodo.

We currently offer the following models:
- `vit_h`: Default Segment Anything model with vit-h backbone.
- `vit_l`: Default Segment Anything model with vit-l backbone.
- `vit_b`: Default Segment Anything model with vit-b backbone.
- `vit_t`: Segment Anything model with vit-tiny backbone. From the [Mobile SAM publication](https://arxiv.org/abs/2306.14289). 
- `vit_l_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with vit-l backbone. ([zenodo](TODO), [bioimage.io](TODO))
- `vit_b_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with vit-b backbone. ([zenodo](TODO), [bioimage.io](TODO))
- `vit_t_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with vit-t backbone. ([zenodo](TODO), [bioimage.io](TODO))
- `vit_l_em_organelles`: Finetuned Segment Anything model for mitochodria and nuclei in electron microscopy data with vit-l backbone. ([zenodo](TODO), [bioimage.io](TODO))
- `vit_b_em_organelles`: Finetuned Segment Anything model for mitochodria and nuclei in electron microscopy data with vit-b backbone. ([zenodo](TODO), [bioimage.io](TODO))
- `vit_t_em_organelles`: Finetuned Segment Anything model for mitochodria and nuclei in electron microscopy data with vit-t backbone. ([zenodo](TODO), [bioimage.io](TODO))

See the two figures below of the improvements through the finetuned model for LM and EM data. 

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/lm_comparison.png" width="768">

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/em_comparison.png" width="768">

You can select which model to use for annotation by selecting the corresponding name in the embedding menu:

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


## Older Models

Previous versions of our models are available on zenodo:
- [vit_b_em_boundaries](https://zenodo.org/records/10524894): for segmenting compartments delineated by boundaries such as cells or neurites in EM.
- [vit_b_em_organelles](https://zenodo.org/records/10524828): for segmenting mitochondria, nuclei or other organelles in EM.
- [vit_b_lm](https://zenodo.org/records/10524791): for segmenting cells and nuclei in LM.
- [vit_h_em](https://zenodo.org/records/8250291): for general EM segmentation.
- [vit_h_lm](https://zenodo.org/records/8250299): for general LM segmentation.

We do not recommend to use these models since our new models improve upon them significantly. But we provide the links here in case they are needed to reproduce older segmentation workflows.
