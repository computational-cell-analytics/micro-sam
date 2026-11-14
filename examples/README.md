# Examples

Examples for using the `micro_sam` annotation tools with the micro-sam2 models:
- `annotator_2d.py`: Run the interactive annotation tool for 2d images.
- `annotator_3d.py`: Run the interactive annotation tool for a 3d volume.
- `annotator_tracking.py`: Run the interactive tracking annotation tool.
- `image_series_annotator.py`: Run the annotation tool for a series of images.
- `pixel_classifier.py`: Run the pixel classification tool.
- `object_classifier.py`: Run the object classification tool, for single images and for a batch of images.

And python scripts for automatic segmentation and tracking:
- `quick_start.py`: Run the automatic segmentation on an example 2d image.
- `automatic_segmentation.py`: Run the automatic segmentation (AIS, APG or AMG), with a description of the options.
- `automatic_tracking.py`: Run automatic tracking on a 2d timeseries.

We provide Jupyter Notebooks for using automatic segmentation and finetuning on some example data in the [notebooks](../notebooks/) folder.

The folder `finetuning` contains example scripts that show how a micro-sam2 model can be finetuned
on custom data, and how the finetuned model can then be used within the annotation tools.

The folder `use_as_library` contains example scripts that show how `micro_sam` can be used as a python
library to segment 2d images, large images with tiling and 3d volumes.

The folder `bioimageio` contains example scripts for exporting models to the bioimage.io format.
`export_model_for_bioimageio.py` exports micro-sam2 models, for interactive and automatic segmentation, and SAM1 models.
The export for the BioEngine (`export_model_for_bioengine.py`) supports the SAM1 models of micro-sam v1 only.

Each script also shows the corresponding call of the `micro_sam` command line interface, where there is one.
Run `micro_sam --help` to see all commands.
