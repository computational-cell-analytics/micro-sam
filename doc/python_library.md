# Using the Python Library

The python library can be imported via
```python
import micro_sam
```

This library extends the [Segment Anything library](https://github.com/facebookresearch/segment-anything) and
- implements functions to apply Segment Anything to 2d and 3d data in `micro_sam.prompt_based_segmentation`.
- provides improved automatic instance segmentation functionality in `micro_sam.instance_segmentation`.
- implements training functionality that can be used for finetuning on your own data in `micro_sam.training`.
- provides functionality for quantitative and qualitative evaluation of Segment Anything models in `micro_sam.evaluation`.

You can import these sub-modules via
```python
import micro_sam.prompt_based_segmentation
import micro_sam.instance_segmentation
# etc.
```

This functionality is used to implement the interactive annotation tools in `micro_sam.sam_annotator` and can be used as a standalone python library.
We provide jupyter notebooks that demonstrate how to use it [here](https://github.com/computational-cell-analytics/micro-sam/tree/master/notebooks). You can find the full library documentation by scrolling to the end of this page. 

## Training your own model

We reimplement the training logic described in the [Segment Anything publication](https://arxiv.org/abs/2304.02643) to enable finetuning on custom data.
We use this functionality to provide the [finetuned microscopy models](#finetuned-models) and it can also be used to train models on your own data.
In fact the best results can be expected when finetuning on your own data, and we found that it does not require much annotated training data to get siginficant improvements in model performance.
So a good strategy is to annotate a few images with one of the provided models using our interactive annotation tools and, if the model is not working as good as required for your use-case, finetune on the annotated data.
<!--
TODO: provide link to the paper with results on how much data is needed
-->

The training logic is implemented in `micro_sam.training` and is based on [torch-em](https://github.com/constantinpape/torch-em). Check out [the finetuning notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/micro-sam-finetuning.ipynb) to see how to use it.

We also support training an additional decoder for automatic instance segmentation. This yields better results than the automatic mask generation of segment anything and is significantly faster.
The notebook explains how to activate training it together with the rest of SAM and how to then use it.

More advanced examples, including quantitative and qualitative evaluation, of finetuned models can be found in [finetuning](https://github.com/computational-cell-analytics/micro-sam/tree/master/finetuning), which contains the code for training and evaluating [our models](finetuned-models).
