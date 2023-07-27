# How to use the Python Library

The python library can be imported via
```python
import micro_sam
```

It implements functionality for running Segment Anything for 2d and 3d data, provides more instance segmentation functionality and several other helpful functions for using Segment Anything.
This functionality is used to implement the `micro_sam` annotation tools, but you can also use it as a standalone python library.

## Finetuned models

We provide fine-tuned Segment Anything models for microscopy data. They are still in an experimental stage and we will upload more and better models soon, as well as the code for fine-tuning.
For using the current models, check out the [2d annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/sam_annotator_2d.py#L62) and set `use_finetuned_model` to `True`.
