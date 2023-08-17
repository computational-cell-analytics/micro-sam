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
Check out the documentation under `Submodules` for more details.

## Training your own model

TODO
