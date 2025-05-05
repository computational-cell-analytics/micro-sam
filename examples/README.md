# Examples

Examples for using the `micro_sam` annotation tools:
- `annotator_2d.py`: run the interactive 2d annotation tool.
- `annotator_3d.py`: run the interactive 3d annotation tool.
- `annotator_tracking.py`: run the interactive tracking annotation tool.
- `image_series_annotator.py`: run the annotation tool for a series of images.

And examples for using the `micro_sam` automatic segmentation feature:
- `quick_start.py`: run the automatic segmentation feature of `micro_sam` on an example 2d image.
- `automatic_segmentation.py`: run the automatic segmentation feature (with an extensive description) of `micro_sam` on some example data.

We provide Jupyter Notebooks for using automatic segmentation and finetuning on some example data in the [notebooks](../notebooks/) folder.

The folder `finetuning` contains example scripts that show how a Segment Anything model can be fine-tuned
on custom data with the `micro_sam.train` library, and how the finetuned models can then be used within the annotation tools.

The folder `use_as_library` contains example scripts that show how `micro_sam` can be used as a python
library to apply Segment Anything to multi-dimensional data.
