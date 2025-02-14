# Segment Anything for Light Microscopy

This is a [Segment Anything](https://segment-anything.com/) model that was specialized for light microscopy with [micro_sam](https://github.com/computational-cell-analytics/micro-sam).
This model uses a %s vision transformer as image encoder.

Segment Anything is a model for interactive and automatic instance segmentation.
We improve it for light microscopy by finetuning on a large and diverse microscopy dataset.
It should perform well for cell and nucleus segmentation in fluorescent, label-free and other light microscopy datasets.

See [the dataset overview](https://github.com/computational-cell-analytics/micro-sam/blob/master/doc/datasets/lm_v%i.md) for further informations on the training data and the [micro_sam documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html) for details on how to use the model for interactive and automatic segmentation.


## Validation

The easiest way to validate the model is to visually check the segmentation quality for your data.
If you have annotations you can use for validation you can also quantitative validation, see [here for details](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#9-how-can-i-evaluate-a-model-i-have-finetuned).
Please note that the required quality for segmentation always depends on the analysis task you want to solve.
