# APG

`micro_sam` supports three different modes for instance segmentation:
- Automatic Mask Generation (AMG) covers the image with a grid of points. These points are used as prompts and the resulting masks are merged via non-maximum suppression (NMS) to obtain the instance segmentation. This method has been introduced by the original SAM publication.
- Automatic Instance Segmentation (AIS) uses an additional segmentation decoder, which we introduced in the `micro_sam` publication. This decoder predicts foreground probabilities as well as the normalized distances to cell centroids and boundaries. These predictions are used as input to a waterhsed to obtain the instances.
- Automatic Prompt Generation (APG) is an instance segmentation approach that we introduced in [a new paper](https://openreview.net/forum?id=xFO3DFZN45). It derives point prompts from the segmentation decoder (see AIS) and merges the resulting masks via NMS.

In our experiments, APG yields the best overall instance segmentation results (compared to AMG and AIS) and is competitive with [CellPose-SAM](https://doi.org/10.1101/2025.04.28.651001
), the state-of-the-art model for cell instance segmentation.

The segmentation mode can be selected with the argument `mode` or `segmentation_mode` in the [CLI](#using-the-command-line-interface-cli) and [python functionality](https://computational-cell-analytics.github.io/micro-sam/micro_sam/automatic_segmentation.html). For details on how to use the different automatic segmentation modes check out the [automatic segmentation
notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/automatic_segmentation.ipynb). The code for the experiments comparing the different segmentation modes (from [the new paper](https://openreview.net/forum?id=xFO3DFZN45)) can be found [here](https://github.com/computational-cell-analytics/micro-sam/tree/master/scripts/apg_experiments).
