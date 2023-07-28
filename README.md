[![DOC](https://shields.mitmproxy.org/badge/docs-pdoc.dev-brightgreen.svg)](https://computational-cell-analytics.github.io/micro-sam/)
[![Conda](https://anaconda.org/conda-forge/micro_sam/badges/version.svg)](https://anaconda.org/conda-forge/micro_sam)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7919746.svg)](https://doi.org/10.5281/zenodo.7919746)

# SegmentAnything for Microscopy

Tools for segmentation and tracking in microscopy build on top of [SegmentAnything](https://segment-anything.com/).
Segment and track objects in microscopy images interactively with a few clicks!

We implement napari applications for:
- interactive 2d segmentation (Left: interactive cell segmentation)
- interactive 3d segmentation (Middle: interactive mitochondria segmentation in EM)
- interactive tracking of 2d image data (Right: interactive cell tracking)

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d04cb158-9f5b-4460-98cd-023c4f19cccd" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfca3d9b-dba5-440b-b0f9-72a0683ac410" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/aefbf99f-e73a-4125-bb49-2e6592367a64" width="256">

**Beta version**

This is an advanced beta version. While many features are still under development, we aim to keep the user interface and python library stable.
Any feedback is welcome, but please be aware that the functionality is under active development and that some features may not be thoroughly tested yet.
We will soon provide a stand-alone application for running the `micro_sam` annotation tools, and plan to also release it as [napari plugin](https://napari.org/stable/plugins/index.html) in the future.

If you run into any problems or have questions please open an issue on Github or reach out via [image.sc](https://forum.image.sc/) using the tag `micro-sam` and tagging @constantinpape.


## Installation and Usage

You can install `micro_sam` via conda:
```
conda install -c conda-forge micro_sam
```
You can then start the `micro_sam` tools by running `$ micro_sam.annotator` in the command line.

Please check out [the documentation](https://computational-cell-analytics.github.io/micro-sam/) for more details on the installation and usage of `micro_sam`.


## Citation

If you are using this repository in your research please cite
- [SegmentAnything](https://arxiv.org/abs/2304.02643)
- and our repository on [zenodo](https://doi.org/10.5281/zenodo.7919746) (we are working on a publication)


## Related Projects

There are a few other napari plugins build around Segment Anything:
- https://github.com/MIC-DKFZ/napari-sam (2d and 3d support)
- https://github.com/JoOkuma/napari-segment-anything (only 2d support)
- https://github.com/hiroalchem/napari-SAM4IS

Compared to these we support more applications (2d, 3d and tracking), and provide finetuning methods and finetuned models for microscopy data.
[WebKnossos](https://webknossos.org/) also offers integration of SegmentAnything for interactive segmentation.


## Release Overview

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


<!---
## Contributing

```
micro_sam <- library with utility functionality for using SAM for microscopy data
    /sam_annotator <- the napari plugins for annotation
```
-->
