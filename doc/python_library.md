# How to use the Python Library

The python library can be imported via
```python
import micro_sam
```

It implements functionality for running Segment Anything for 2d and 3d data, provides more instance segmentation functionality and several other helpful functions for using Segment Anything.
This functionality is used to implement the `micro_sam` annotation tools, but you can also use it as a standalone python library. Check out the documentation under `Submodules` for more details on the python library.

## Finetuned models

We provide finetuned Segment Anything models for microscopy data. They are still in an experimental stage and we will upload more and better models soon, as well as the code for fine-tuning.
For using the preliminary models, check out the [2d annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py#L62) and set `use_finetuned_model` to `True`.

We currently provide support for the following models:
- `vit_h`: The default Segment Anything model with vit-h backbone.
- `vit_l`: The default Segment Anything model with vit-l backbone.
- `vit_b`: The default Segment Anything model with vit-b backbone.
- `vit_h_lm`: The preliminary finetuned Segment Anything model for light microscopy data with vit-h backbone.
- `vit_b_lm`: The preliminary finetuned Segment Anything model for light microscopy data with vit-b backbone.

These are also the valid names for the `model_type` parameter in `micro_sam`. The library will automatically choose and if necessary download the corresponding model.

See the difference between the normal and finetuned Segment Anything ViT-h model on an image from [LiveCELL](https://sartorius-research.github.io/LIVECell/):

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/vanilla-v-finetuned.png" width="768">

