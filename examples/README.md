# Examples

Examples for using the `micro_sam` annotation tools:
- `annotator_2d.py`: Run the interactive 2d annotation tool.
- `annotator_3d.py`: Run the interactive 3d annotation tool.
- `annotator_tracking.py`: Run the interactive tracking annotation tool.
- `image_series_annotator.py`: Run the annotation tool for a series of images.

And python scripts for automatic segmentation and tracking:
- `automatic_segmentation.py`: Run automatic segmentation on 2d images.
- `automatic_tracking.py`: Run automatic tracking on 2d timeseries images.

We provide Jupyter Notebooks for using automatic segmentation and finetuning on some example data in the [notebooks](../notebooks/) folder.

The folder `finetuning` contains example scripts that show how a Segment Anything model can be fine-tuned
on custom data with the `micro_sam.train` library, and how the finetuned models can then be used within the annotation tools.

The folder `use_as_library` contains example scripts that show how `micro_sam` can be used as a python
library to apply Segment Anything to multi-dimensional data.
