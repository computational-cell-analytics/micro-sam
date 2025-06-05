# Release Overview

**New in version 1.5.0**
- Preliminary version of object classification tool.
- Enabling support for napari v6, zarr v3 and numpy v2.
- Add support for training models for automatic instance segmentation-only.

**New in version 1.4.0**

This release includes three main changes:
- Preliminary support for automatic tracking via [Trackastra](https://github.com/weigertlab/trackastra) integration.
- Changes in the GUI to make the model names more informative.
- Much easier installation on Windows.

**New in version 1.3.1**

Fixing minor issues with 1.3.0 and adding new section in documentation for our data submission initiative.

**New in version 1.3.0**

This release introduces a new light microscopy model that was trained on a larger dataset and clearly improves automatic segmentation.

**New in version 1.2.2**

Fixing minor issues with 1.2.1 for making automatic segmentation CLI more flexible.

**New in version 1.2.1**
This version introduces several changes that are part of three of our recent publications that are built on top of micro_sam:

- [medico-sam](https://github.com/computational-cell-analytics/medico-sam), which improves SAM for medical images
- [peft-sam](https://github.com/computational-cell-analytics/peft-sam), which investigates parameter efficient finetuning for SAM
- [patho-sam](https://github.com/computational-cell-analytics/patho-sam), which improves SAM for histopathology

**New in version 1.2.0**

The main changes in this version are:

- Installation using only conda-forge dependencies and simplified installation instructions (on Linux and Mac OS).
- Fix annotation in napari widgets with scale factors.
- Support for several parameter-efficient training methods.

**New in version 1.1.1**

Fixing minor issues with 1.1.0 and enabling pytorch 2.5 support.

**New in version 1.1.0**

This version introduces several improvements:

- Bugfixes and several minor improvements
- Compatibility with napari >=0.5
- Automatic instance segmentation CLI
- Initial support for parameter efficient fine-tuning and automatic semantic segmentation in 2d and 3d (not available in napari plugin, part of the python library)

**New in version 1.0.1**

Use stable URL for model downloads and fix issues in state precomputation for automatic segmentation.

**New in version 1.0.0**

This release mainly fixes issues with the previous release and marks the napari user interface as stable.

**New in version 0.5.0**

This version includes a lot of new functionality and improvements. The most important changes are:
- Re-implementation of the annotation tools. The tools are now implemented as napari plugin.
- Using our improved functionality for automatic instance segmentation in the annotation tools, including automatic segmentation for 3D data.
- New widgets to use the finetuning and image series annotation functionality from napari.
- Improved finetuned models for light microscopy and electron microscopy data that are available via bioimage.io.

**New in version 0.4.1**

- Bugfix for the image series annotator. Before the automatic segmentation did not work correctly.

**New in version 0.4.0**

- Significantly improved model finetuning
- Update the finetuned models for microscopy, see [details in the doc](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models)
- Training decoder for direct instance segmentation (not available via the GUI yet)
- Refactored model download functionality using [pooch](https://pypi.org/project/pooch/)

**New in version 0.3.0**

- Support for ellipse and polygon prompts
- Support for automatic segmentation in 3d
- Training refactoring and speed-up of fine-tuning

**New in version 0.2.1 and 0.2.2**

- Several bugfixes for the newly introduced functionality in 0.2.0.

**New in version 0.2.0**

- Functionality for training / finetuning and evaluation of Segment Anything Models
- Full support for our finetuned segment anything models
- Improvements of the automated instance segmentation functionality in the 2d annotator
- And several other small improvements

**New in version 0.1.1**

- Fine-tuned segment anything models for microscopy (experimental)
- Simplified instance segmentation menu
- Menu for clearing annotations

**New in version 0.1.0**

- We support tiling in all annotators to enable processing large images.
- Implement new automatic instance segmentation functionality:
    - That is faster.
    - Enables interactive update of parameters.
    - And also works for large images by making use of tiled embeddings.
- Implement the `image_series_annotator` for processing many images in a row.
- Use the data hash in pre-computed embeddings to warn if the input data changes.
- Create a simple GUI to select which annotator to start.
- And made many other small improvements and fixed bugs.

**New in version 0.0.2**

- We have added support for bounding box prompts, which provide better segmentation results than points in many cases.
- Interactive tracking now uses a better heuristic to propagate masks across time, leading to better automatic tracking results.
- And have fixed several small bugs.
