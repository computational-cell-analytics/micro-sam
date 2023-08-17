# Segment Anything for Microscopy

Segment Anything for Microscopy implements automatic and interactive annotation for microscopy data. It is built on top of [Segment Anything](https://segment-anything.com/) by Meta AI and specializes it for microscopy and other bio-imaging data.
Its core components are:
- The `micro_sam` tools for interactive data annotation with [napari](https://napari.org/stable/).
- The `micro_sam` library to apply Segment Anything to 2d and 3d data or fine-tune it on your data.
- The `micro_sam` models that are fine-tuned on publicly available microscopy data.

Our goal is to build fast and interactive annotation tools for microscopy data, like interactive cell segmentation from bounding boxes:

![box-prompts](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d04cb158-9f5b-4460-98cd-023c4f19cccd)

`micro_sam` is under active development, but our goal is to keep the changes to the user interface and the interface of the python library as small as possible.
On our roadmap for more functionality are:
- Providing an installer for running `micro_sam` as a standalone application.
- Releasing more and better finetuned models as well as the code for fine-tuning.
- Integration of the finetuned models with [bioimage.io](https://bioimage.io/#/)
- Implementing a napari plugin for `micro_sam`.

If you run into any problems or have questions please open an issue on Github or reach out via [image.sc](https://forum.image.sc/) using the tag `micro-sam` and tagging @constantinpape.


## Quickstart

You can install `micro_sam` via conda:
```
$ conda install -c conda-forge micro_sam napari pyqt
```
We also provide experimental installers for all operating systems.
For more details on the available installation options check out [the installation section](#installation).

After installing `micro_sam` you can run the annotation tool via `$ micro_sam.annotator`, which opens a menu for selecting the annotation tool and its inputs.
See [the annotation tool section](#annotation-tools) for an overview and explanation of the annotation functionality.

The `micro_sam` python library can be used via
```python
import micro_sam
```
It is explained in more detail [here](#how-to-use-the-python-library).

Our support for finetuned models is still experimental. We will soon release better finetuned models and host them on zenodo.
For now, check out [the example script for the 2d annotator](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/sam_annotator_2d.py#L62) to see how the finetuned models can be used within `micro_sam`.


## Citation

If you are using `micro_sam` in your research please cite
- [SegmentAnything](https://arxiv.org/abs/2304.02643)
- and our repository on [zenodo](https://doi.org/10.5281/zenodo.7919746)

We will release a preprint soon that describes this work and can be cited instead of zenodo.
