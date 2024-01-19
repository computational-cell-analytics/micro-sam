# Finetuned models

In addition to the original Segment anything models, we provide models that finetuned on microscopy data using the functionality from `micro_sam.training`.
The models are hosted on zenodo. We currently offer the following models:
- `vit_h`: Default Segment Anything model with vit-h backbone.
- `vit_l`: Default Segment Anything model with vit-l backbone.
- `vit_b`: Default Segment Anything model with vit-b backbone.
- `vit_t`: Segment Anything model with vit-tiny backbone. From the [Mobile SAM publication](https://arxiv.org/abs/2306.14289). 
- `vit_b_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with vit-b backbone.
- `vit_b_em_organelles`: Finetuned Segment Anything model for mitochodria and nuclei in electron microscopy data with vit-b backbone.
- `vit_b_em_boundaries`: Finetuned Segment Anything model for neurites and cells in electron microscopy data with vit-b backbone.

See the two figures below of the improvements through the finetuned model for LM and EM data. 

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/lm_comparison.png" width="768">

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/em_comparison.png" width="768">

You can select which of the models is used in the annotation tools by selecting the corresponding name from the `Model Type` menu:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/model-type-selector.png" width="256">

To use a specific model in the python library you need to pass the corresponding name as value to the `model_type` parameter exposed by all relevant functions.
See for example the [2d annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py#L62) where `use_finetuned_model` can be set to `True` to use the `vit_b_lm` model.

Note that we are still working on improving these models and may update them from time to time. All older models will stay available for download on zenodo, see [model sources](#model-sources) below


## Which model should I choose?

As a rule of thumb:
- Use the `vit_b_lm` model for segmenting cells or nuclei in light microscopy.
- Use the `vit_b_em_organelles` models for segmenting mitochondria, nuclei or other organelles in electron microscopy.
- Use the `vit_b_em_boundaries` models for segmenting cells or neurites in electron microscopy.
- For other use-cases use one of the default models.

See also the figures above for examples where the finetuned models work better than the vanilla models.
Currently the model `vit_h` is used by default.

We are working on further improving these models and adding new models for other biomedical imaging domains.


## Model Sources

Here is an overview of all finetuned models we have released to zenodo so far:
- [vit_b_em_boundaries](https://zenodo.org/records/10524894): for segmenting compartments delineated by boundaries such as cells or neurites in EM.
- [vit_b_em_organelles](https://zenodo.org/records/10524828): for segmenting mitochondria, nuclei or other organelles in EM.
- [vit_b_lm](https://zenodo.org/records/10524791): for segmenting cells and nuclei in LM.
- [vit_h_em](https://zenodo.org/records/8250291): this model is outdated.
- [vit_h_lm](https://zenodo.org/records/8250299): this model is outdated.

Some of these models contain multiple versions.
