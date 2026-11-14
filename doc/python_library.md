# Using the Python Library

The python library can be imported via

```python
import micro_sam
```

The functionality for SAM2 and the `micro-sam2` models is in the `micro_sam.v2` subpackage:
- `micro_sam.v2.util` loads the models (`get_sam2_model`, `get_sam2_image_predictor`) and computes the image embeddings (`precompute_image_embeddings`).
- `micro_sam.v2.prompt_based_segmentation` segments objects from point, box and mask prompts in 2d and 3d.
- `micro_sam.v2.automatic_segmentation` runs automatic instance segmentation (`get_predictor_and_segmenter`, `automatic_instance_segmentation`) and automatic tracking (`automatic_tracking`). See [APG](#apg) for the available methods.
- `micro_sam.v2.semantic_segmentation` applies a model trained for semantic segmentation.
- `micro_sam.v2.training` trains models on your own data, see below.
- `micro_sam.v2.evaluation` evaluates the interactive and automatic segmentation.

`micro_sam.object_classification` and `micro_sam.pixel_classification` implement the object and pixel classifiers, and `micro_sam.sam_annotator` implements the [annotation tools](#annotation-tools), which can also be started from python.

For example, automatic segmentation of an image with the `micro-sam2` model for cells:
```python
from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

predictor, segmenter = get_predictor_and_segmenter(model_type="hvit_t_cells", segmentation_mode="ais")
segmentation = automatic_instance_segmentation(predictor, segmenter, input_path="image.tif")
```

The functionality of the previous version for SAM1 models is in the `micro_sam.v1` subpackage, for example `micro_sam.v1.instance_segmentation` and `micro_sam.v1.training`.

We provide jupyter notebooks that demonstrate how to use the library [here](https://github.com/computational-cell-analytics/micro-sam/tree/main/notebooks), and example scripts [here](https://github.com/computational-cell-analytics/micro-sam/tree/main/examples). You can find the full library documentation by scrolling to the end of this page.

## Training your Own Model

You can finetune the `micro-sam2` models, or the SAM2 models, on your own data.
We use this functionality to train the [finetuned microscopy models](#finetuned-models).
In fact the best results can be expected when finetuning on your own data, and it does not require much annotated training data to get significant improvements in model performance.
So a good strategy is to annotate a few images with one of the provided models using our interactive annotation tools and, if the model does not work as well as required for your use-case, finetune it on the annotated data.

You need images and the corresponding label images, for example annotations created with the `micro_sam` annotation tools. The data can be:
- image files, such as tif or png, given as single files, lists of files, or a folder and a pattern like `*.tif`.
- container files, such as hdf5, zarr, n5 or OME-Zarr, given as the file path and the key of the data in the file.

The data can be 2d or 3d. The labels are instance segmentations, where each object has its own id, or class labels for semantic segmentation.

There are four training modes:
- `interactive`: train for interactive segmentation (`micro_sam.v2.training.train_sam2`).
- `automatic`: train the decoder for automatic segmentation (AIS, APG) together with the image encoder (`micro_sam.v2.training.train_automatic`).
- `joint`: train for both interactive and automatic segmentation (`micro_sam.v2.training.train_joint_sam2`).
- `semantic`: train for semantic segmentation into a given number of classes (`micro_sam.v2.training.train_semantic`).

When you start from a `micro-sam2` model, for example `hvit_t_cells`, the training starts from its image encoder and its decoder. For semantic segmentation, only the output layer for the classes is new.
A model trained in the `interactive` mode has no decoder, so it cannot be used for automatic segmentation.

`micro_sam.v2.training.default_sam2_loader` creates the data loaders for all modes. For example, to train for interactive and automatic segmentation on tif images:
```python
from micro_sam.v2.training import default_sam2_loader, train_joint_sam2

loader_kwargs = dict(patch_shape=(512, 512), mode="joint", batch_size=1)
train_loader = default_sam2_loader("train/images", "*.tif", "train/labels", "*.tif", **loader_kwargs)
val_loader = default_sam2_loader("val/images", "*.tif", "val/labels", "*.tif", is_train=False, **loader_kwargs)
train_joint_sam2("my_model", "hvit_t_cells", train_loader=train_loader, val_loader=val_loader)
```

For a container file, pass the file path and the key, for example `default_sam2_loader("data.h5", "raw", "data.h5", "labels", ...)`.
For 3d training, pass a 3d `patch_shape`, for example `(8, 512, 512)`.

The same is available in the command line with `micro_sam train`. For example:
```bash
# Train for interactive and automatic segmentation, with 10% of the files split off for validation.
micro_sam train -i images/ --image_key '*.tif' -l labels/ --label_key '*.tif' --mode joint

# Train for automatic segmentation on 3d data in hdf5 files.
micro_sam train -i train.h5 --image_key raw -l train.h5 --label_key labels \
    --val_images val.h5 --val_labels val.h5 --mode automatic --patch_shape 8,512,512

# Train for semantic segmentation into three classes.
micro_sam train -i images/ --image_key '*.png' -l labels/ --label_key '*.png' --mode semantic --num_classes 3
```
Run `micro_sam train --help` for all options.

The trained model is saved as `checkpoints/<NAME>/best.pt`. You can use it in the annotation tools and the automatic segmentation with the `checkpoint_path` argument in python, or with `-c` in the command line. The model size is read from the checkpoint.

To share your model, export it to the [BioImage.IO](https://bioimage.io) model format with `micro_sam.v2.bioimageio.export_sam2_model`, see [the export example](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/bioimageio/export_model_for_bioimageio.py). The exported model does interactive segmentation and, if it has a decoder, automatic segmentation.

Check out [the finetuning notebook](https://github.com/computational-cell-analytics/micro-sam/blob/main/notebooks/sam_finetuning.ipynb) and [the finetuning examples](https://github.com/computational-cell-analytics/micro-sam/tree/main/examples/finetuning) for more details.
The training needs a GPU for reasonable training times. The number of objects per image (`max_num_objects`), the batch size, the patch shape and the model size have a strong impact on the GPU memory that is needed. If the training runs out of memory, reduce them. You can find further information on model training in the [FAQ section](#fine-tuning-questions).
