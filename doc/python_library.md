# How to use the Python Library

The python library can be imported via
```python
import micro_sam
```

The library
- implements function to apply Segment Anything to 2d and 3d data more conviently in `micro_sam.prompt_based_segmentation`.
- provides more and imporoved automatic instance segmentation functionality in `micro_sam.instance_segmentation`.
- implements training functionality that can be used for finetuning on your own data in `micro_sam.training`.
- provides functionality for quantitative and qualitative evaluation of Segment Anything models in `micro_sam.evaluation`.

This functionality is used to implement the interactive annotation tools and can also be used as a standalone python library.
Some preliminary examples for how to use the python library can be found [here](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/use_as_library). Check out the `Submodules` documentation for more details.

## Training your own model

We reimplement the training logic described in the [Segment Anything publication](https://arxiv.org/abs/2304.02643) to enable finetuning on custom data.
We use this functionality to provide the [finetuned microscopy models](#finetuned-models) and it can also be used to finetune models on your own data.
In fact the best results can be expected when finetuning on your own data, and we found that it does not require much annotated training data to get siginficant improvements in model performance.
So a good strategy is to annotate a few images with one of the provided models using one of the interactive annotation tools and, if the annotation is not working as good as expected yet, finetune on the annotated data.
<!--
TODO: provide link to the paper with results on how much data is needed
-->

The training logic is implemented in `micro_sam.training` and is based on [torch-em](https://github.com/constantinpape/torch-em). Please check out [examples/finetuning](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/finetuning) to see how you can finetune on your own data with it. The script `finetune_hela.py` contains an example for finetuning on a small custom microscopy dataset and `use_finetuned_model.py` shows how this model can then be used in the interactive annotatin tools.

More advanced examples, including quantitative and qualitative evaluation, of finetuned models can be found in [finetuning](https://github.com/computational-cell-analytics/micro-sam/tree/master/finetuning), which contains the code for training and evaluating our microscopy models.
