# Segment Anything for Microscopy

Segment Anything for Microscopy implements automatic and interactive annotation for microscopy data. It is built on top of [Segment Anything 2 (SAM2)](https://ai.meta.com/sam2/) by Meta AI and specializes it for microscopy and other biomedical imaging data.
Its core components are:
- The `micro_sam` tools for interactive data annotation, built as [napari](https://napari.org/stable/) plugin.
- The `micro_sam` library to apply SAM2 to 2d and 3d data, to track objects over time, and to finetune it on your data.
- The `micro-sam2` models, which are finetuned on publicly available microscopy data.

Based on these components `micro_sam` enables fast interactive and automatic annotation for microscopy data, like interactive cell segmentation from bounding boxes:

![box-prompts](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d04cb158-9f5b-4460-98cd-023c4f19cccd)

The current version of `micro_sam` (micro-sam2) is based on SAM2. The functionality of the previous version, based on the original Segment Anything model (SAM1), is still available in the `micro_sam.v1` subpackage and in the `micro_sam v1` command line tools.

If you run into any problems or have questions please [open an issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) or reach out via [image.sc](https://forum.image.sc/) using the tag `micro-sam`.
You can follow recent updates on `micro_sam` in our [news feed](https://forum.image.sc/t/microsam-news-feed).


## Quickstart

You can install `micro_sam` via conda:
```bash
conda install -c conda-forge micro_sam
```
For more details on the available installation options, check out [the installation section](#installation).

After installing `micro_sam`, you can start napari from within your environment using

```bash
napari
```
After starting napari, you can select the annotation tool you want to use from `Plugins -> Segment Anything for Microscopy`. See [the annotation tool section](#annotation-tools) for details.

You can also start the annotation tools and the automatic segmentation from the command line:
```bash
# Interactive segmentation of a 2d or 3d image.
micro_sam annotator segmentation -i image.tif
# Automatic segmentation of an image.
micro_sam inference segmentation -i image.tif -o segmentation.tif
```
See [the command line section](#using-the-command-line-interface-cli) for details.

The `micro_sam` python library can be imported via

```python
import micro_sam
```

It is explained in more detail [here](#using-the-python-library).

We also have a quick start notebook [here](https://github.com/computational-cell-analytics/micro-sam/blob/main/notebooks/quick_start.ipynb).

We provide different finetuned models for microscopy. See [finetuned models](#finetuned-models) for details on the available models.
You can also train models on your own data, see [here for details](#training-your-own-model).

## Citation

If you are using `micro_sam` in your research please cite
- our [paper](https://www.nature.com/articles/s41592-024-02580-4) (published in Nature Methods)
- and the [Segment Anything 2 publication](https://arxiv.org/abs/2408.00714).
- If you use automatic tracking, please also cite [Trackastra](https://arxiv.org/abs/2405.15700).
- If you use automatic prompt generation (APG), please also cite [our APG paper](https://openreview.net/forum?id=xFO3DFZN45).
- If you use the SAM1 functionality in `micro_sam.v1`, please also cite the [original Segment Anything publication](https://arxiv.org/abs/2304.02643), and [Mobile SAM](https://arxiv.org/abs/2306.14289) if you use a `vit_t` model.
