# Segment Anything for Microscopy

Segment Anything for Microscopy implements automatic and interactive annotation for microscopy data. It is built on top of [Segment Anything](https://segment-anything.com/) by Meta AI and specializes it for microscopy and other biomedical imaging data.
Its core components are:
- The `micro_sam` tools for interactive data annotation, built as [napari](https://napari.org/stable/) plugin.
- The `micro_sam` library to apply Segment Anything to 2d and 3d data or fine-tune it on your data.
- The `micro_sam` models that are fine-tuned on publicly available microscopy data and that are available on [BioImage.IO](https://bioimage.io/#/).

Based on these components `micro_sam` enables fast interactive and automatic annotation for microscopy data, like interactive cell segmentation from bounding boxes:

![box-prompts](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d04cb158-9f5b-4460-98cd-023c4f19cccd)

`micro_sam` is now available as stable version 1.0 and we will not change its user interface significantly in the foreseeable future.
We are still working on improving and extending its functionality. The current roadmap includes:
- Releasing more and better finetuned models for the biomedical imaging domain.
- Integrating parameter efficient training and compressed models for efficient fine-tuning and faster inference.
- Improving the 3D segmentation and tracking functionality.

If you run into any problems or have questions please [open an issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) or reach out via [image.sc](https://forum.image.sc/) using the tag `micro-sam`.


## Quickstart

You can install `micro_sam` via mamba:
```
$ mamba install -c conda-forge micro_sam
```
We also provide installers for Windows and Linux. For more details on the available installation options, check out [the installation section](#installation).

After installing `micro_sam` you can start napari and select the annotation tool you want to use from `Plugins -> SegmentAnything for Microscopy`. Check out the [quickstart tutorial video](https://youtu.be/gcv0fa84mCc) for a short introduction and [the annotation tool section](#annotation-tools) for details.

The `micro_sam` python library can be imported via

```python
import micro_sam
```

It is explained in more detail [here](#using-the-python-library).

We provide different finetuned models for microscopy that can be used within our tools or any other tool that supports Segment Anything. See [finetuned models](#finetuned-models) for details on the available models.
You can also train models on your own data, see [here for details](#training-your-own-model).

## Citation

If you are using `micro_sam` in your research please cite
- our [preprint](https://doi.org/10.1101/2023.08.21.554208)
- and the original [Segment Anything publication](https://arxiv.org/abs/2304.02643).
- If you use a `vit-tiny` models, please also cite [Mobile SAM](https://arxiv.org/abs/2306.14289).
