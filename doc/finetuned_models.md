# Finetuned models

We provide models that were finetuned on microscopy data using `micro_sam.training`. They are hosted on zenodo. We currently offer the following models:
- `vit_h`: Default Segment Anything model with vit-h backbone.
- `vit_l`: Default Segment Anything model with vit-l backbone.
- `vit_b`: Default Segment Anything model with vit-b backbone.
- `vit_h_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with vit-h backbone.
- `vit_b_lm`: Finetuned Segment Anything model for cells and nuclei in light microscopy data with vit-b backbone.
- `vit_h_em`: Finetuned Segment Anything model for neurites and cells in electron microscopy data with vit-h backbone.
- `vit_b_em`: Finetuned Segment Anything model for neurites and cells in electron microscopy data with vit-b backbone.

See the two figures below of the improvements through the finetuned model for LM and EM data. 

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/lm_comparison.png" width="768">

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/em_comparison.png" width="768">

You can select which of the models is used in the annotation tools by selecting the corresponding name from the `Model Type` menu:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/model-type-selector.png" width="256">

To use a specific model in the python library you need to pass the corresponding name as value to the `model_type` parameter exposed by all relevant functions.
See for example the [2d annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py#L62) where `use_finetuned_model` can be set to `True` to use the `vit_h_lm` model.

## Which model should I choose?

As a rule of thumb:
- Use the `_lm` models for segmenting cells or nuclei in light microscopy.
- Use the `_em` models for segmenting ceells or neurites in electron microscopy.
    - Note that this model does not work well for segmenting mitochondria or other organelles becuase it is biased towards segmenting the full cell / cellular compartment.
- For other cases use the default models.

See also the figures above for examples where the finetuned models work better than the vanilla models.
Currently the model `vit_h` is used by default.

We are working on releasing more fine-tuned models, in particular for mitochondria and other organelles in EM.
