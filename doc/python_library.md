# Using the Python Library

The python library can be imported via

```python
import micro_sam
```

This library extends the [Segment Anything library](https://github.com/facebookresearch/segment-anything) and

- implements functions to apply Segment Anything to 2d and 3d data in `micro_sam.prompt_based_segmentation`.
- provides improved automatic instance segmentation functionality in `micro_sam.instance_segmentation`.
- implements training functionality that can be used for finetuning Segment Anything on your own data in `micro_sam.training`.
- provides functionality for quantitative and qualitative evaluation of Segment Anything models in `micro_sam.evaluation`.

You can import these sub-modules via

```python
import micro_sam.prompt_based_segmentation
import micro_sam.instance_segmentation
# etc.
```

This functionality is used to implement the interactive annotation tools in `micro_sam.sam_annotator` and can be used as a standalone python library.
We provide jupyter notebooks that demonstrate how to use it [here](https://github.com/computational-cell-analytics/micro-sam/tree/master/notebooks). You can find the full library documentation by scrolling to the end of this page. 

## Training your Own Model

We reimplement the training logic described in the [Segment Anything publication](https://arxiv.org/abs/2304.02643) to enable finetuning on custom data.
We use this functionality to provide the [finetuned microscopy models](#finetuned-models) and it can also be used to train models on your own data.
In fact the best results can be expected when finetuning on your own data, and we found that it does not require much annotated training data to get significant improvements in model performance.
So a good strategy is to annotate a few images with one of the provided models using our interactive annotation tools and, if the model is not working as good as required for your use-case, finetune on the annotated data.
We recommend checking out our latest [preprint](https://doi.org/10.1101/2023.08.21.554208) for details on the results on how much data is required for finetuning Segment Anything.

The training logic is implemented in `micro_sam.training` and is based on [torch-em](https://github.com/constantinpape/torch-em). Check out [the finetuning notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/sam_finetuning.ipynb) to see how to use it.
We also support training an additional decoder for automatic instance segmentation. This yields better results than the automatic mask generation of segment anything and is significantly faster.
The notebook explains how to train it together with the rest of SAM and how to then use it.

More advanced examples, including quantitative and qualitative evaluation, can be found in [the finetuning directory](https://github.com/computational-cell-analytics/micro-sam/tree/master/finetuning), which contains the code for training and evaluating [our models](finetuned-models). You can find further information on model training in the [FAQ section](fine-tuning-questions).

Here is a list of resources, together with their recommended training settings, for which we have tested model finetuning:

| Resource Name               | Capacity | Model Type | Batch Size | Finetuned Parts              | Number of Objects|
|-----------------------------|----------|------------|------------|------------------------------|------------------|
| CPU                         | 32GB     | ViT Base   | 1          | *all*                        | 10               |
| CPU                         | 64GB     | ViT Base   | 1          | *all*                        | 15               |
| GPU (NVIDIA GTX 1080Ti)     | 8GB      | ViT Base   | 1          | Mask Decoder, Prompt Encoder | 10               |
| GPU (NVIDIA Quadro RTX5000) | 16GB     | ViT Base   | 1          | *all*                        | 10               |
| GPU (Tesla V100)            | 32GB     | ViT Base   | 1          | *all*                        | 10               |
| GPU (NVIDIA A100)           | 80GB     | ViT Tiny   | 2          | *all*                        | 50               |
| GPU (NVIDIA A100)           | 80GB     | ViT Base   | 2          | *all*                        | 40               |
| GPU (NVIDIA A100)           | 80GB     | ViT Large  | 2          | *all*                        | 30               |
| GPU (NVIDIA A100)           | 80GB     | ViT Huge   | 2          | *all*                        | 25               |

> NOTE: If you use the [finetuning UI](#finetuning-ui) or `micro_sam.training.training.train_sam_for_configuration` you can specify the hardware configuration and the best settings for it will be set automatically. If your hardware is not in the settings we have tested choose the closest match. You can set the training parameters yourself when using `micro_sam.training.training.train_sam`. Be aware that the choice for the number of objects per image, the batch size, and the type of model have a strong impact on the VRAM needed for training and the duration of training. See the [finetuning notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/sam_finetuning.ipynb) for an overview of these parameters.
