# APG

`micro_sam` supports three different methods for automatic instance segmentation:
- Automatic Mask Generation (AMG) covers the image with a grid of points. These points are used as prompts and the resulting masks are merged via non-maximum suppression (NMS) to obtain the instance segmentation. This method has been introduced by the original SAM publication. It works with every model, also with the SAM2 models without a decoder.
- Automatic Instance Segmentation (AIS) uses an additional segmentation decoder, which we introduced in the `micro_sam` publication. This decoder predicts the foreground probability, the distances to the object centers and boundaries, and the object boundaries. The post-processing derives the instances from these predictions.
- Automatic Prompt Generation (APG) is an instance segmentation approach that we introduced in [a new paper](https://openreview.net/forum?id=xFO3DFZN45). It derives point prompts from the predictions of the segmentation decoder (see AIS) and segments each object with the interactive part of the model. In 3d, it propagates the prompts through the volume.

AIS and APG need a model with a decoder, for example the `micro-sam2` models (see [finetuned models](#finetuned-models)) or a model that you trained for automatic segmentation (see [training your own model](#training-your-own-model)).
A model that was trained for interactive segmentation only has no decoder, so AIS and APG are not available with it.

In our experiments, APG yields the best overall instance segmentation results, and AIS is the fastest. APG on volumes and timeseries needs a GPU.

The segmentation method can be selected in the annotation tools (the `mode` of the automatic segmentation menu), with `--engine` in the [CLI](#using-the-command-line-interface-cli) and with `segmentation_mode` in `micro_sam.v2.automatic_segmentation.get_predictor_and_segmenter`. For details on how to use the different methods check out the [automatic segmentation notebook](https://github.com/computational-cell-analytics/micro-sam/blob/main/notebooks/automatic_segmentation.ipynb) and [examples/automatic_segmentation.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/automatic_segmentation.py).
