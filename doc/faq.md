# FAQ

Here we provide frequently asked questions and common issues.
If you encounter a problem or question not addressed here feel free to [open an issue](https://github.com/computational-cell-analytics/micro-sam/issues) or to ask your question on [image.sc](https://forum.image.sc/) with the tag `micro-sam`.

## Installation questions

### 1. How to install `micro_sam`?
The [installation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation) for `micro_sam` is supported in three ways: [from mamba](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-mamba) (recommended), [from source](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-source) and [from installers](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-installer). Check out our [tutorial video](TODO) to get started with `micro_sam`, briefly walking you through the installation process and how to start the tool.

### 2. I cannot install `micro_sam` using the installer, I am getting some errors.
The installer should work out-of-the-box on Windows and Linux platforms. Please open an issue to report the error you encounter.
>NOTE: The installers enable using `micro_sam` without mamba or conda. However, we recommend the installation from mamba / from source to use all its features seamlessly. Specifically, the installers currently only support the CPU and won't enable you to use the GPU (if you have one). 

### 3. What is the minimum system requirement for `micro_sam`?
From our experience, the `micro_sam` annotation tools work seamlessly on most laptop or workstation CPUs and with > 8GB RAM.
You might encounter some slowness for $\leq$ 8GB RAM. The resources `micro_sam`'s annotation tools have been tested on are:
- Windows:
    - Windows 10 Pro, Intel i5 7th Gen, 8GB RAM
- Linux:
    - Ubuntu 22.04, Intel i7 12th Gen, 32GB RAM
- Mac:
    - macOS Sonoma 14.4.1
        - M1 Chip, 8GB RAM
        - M3 Max Chip, 36GB RAM

Having a GPU will significantly speed up the annotation tools and especially the model finetuning.

### 4. What is the recommended PyTorch version?
`micro_sam` has been tested mostly with CUDA 12.1 and PyTorch [2.1.1, 2.2.0]. However, the tool and the library is not constrained to a specific PyTorch or CUDA version. So it should work fine with the standard PyTorch installation for your system.

### 5. I am missing a few packages (eg. `ModuleNotFoundError: No module named 'elf.io`). What should I do?
With the latest release 1.0.0, the installation from mamba and source should take care of this and install all the relevant packages for you.
So please reinstall `micro_sam`.

### 6. Can I install `micro_sam` using pip?
The installation is not supported via pip.

### 7. I get the following error: `importError: cannot import name 'UNETR' from 'torch_em.model'`.
It's possible that you have an older version of `torch-em` installed. Similar errors could often be raised from other libraries, the reasons being: a) Outdated packages installed, or b) Some non-existent module being called. If the source of such error is from `micro_sam`, then `a)` is most likely the reason . We recommend installing the latest version following the [installation instructions](https://github.com/constantinpape/torch-em?tab=readme-ov-file#installation).


## Usage questions

<!---
TODO provide relevant links here.
-->
### 1. I have some micropscopy images. Can I use the annotator tool for segmenting them?
Yes, you can use the annotator tool for:
- Segmenting objects in 2d images (using automatic and/or interactive segmentation).
- Segmenting objects in 3d volumes (using automatic and/or interactive segmentation for the entire object(s)).
- Tracking objects over time in time-series data.
- Segmenting objects in a series of 2d / 3d images.
- (OPTIONAL) You can finetune the Segment Anything / `micro_sam` models on your own microscopy data, in case the provided models do not suffice your needs. One caveat: You need to annotate a few objects before-hand (`micro_sam` has the potential of improving interactive segmentation with only a few annotated objects) to proceed with the supervised finetuning procedure.

<!---
TODO fill me in
-->
### 2. Which model should I use for my data?
TODO: quick explanation and link to the corresponding section in the doc.

### 3. I have high-resolution microscopy images, 'micro_sam' does not seem to work.
The Segment Anything model expects inputs of shape 1024 x 1024 pixels. Inputs that do not match this size will be internally resized to match it. Hence, applying Segment Anything to a much larger image will often lead to inferior results, or somethimes not work at all. To address this, `micro_sam` implements tiling: cutting up the input image into tiles of a fixed size (with a fixed overlap) and running Segment Anything for the individual tiles. You can activate tiling with the `tile_shape` parameter, which determines the size of the inner tile and `halo`, which determines the size of the additional overlap.
- If you are using the `micro_sam` annotation tools, you can specify the values for the `tile_shape` and `halo` via the `tile_x`, `tile_y`, `halo_x` and `halo_y` parameters in the `Embedding Settings` drop-down menu.
- If you are using the `micro_sam` library in a python script, you can pass them as tuples, e.g. `tile_shape=(1024, 1024), halo=(256, 256)`. See also the [wholeslide annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py#L47-L63).
- If you are using the command line functionality, you can pass them via the options `--tile_shape 1024 1024 --halo 256 256`.
> NOTE: It's recommended to choose the `halo` so that it is larger than half of the maximal radius of the objects you want to segment.

### 4. The computation of image embeddings takes very long in napari.
`micro_sam` pre-computes the image embeddings produced by the vision transformer backbone in Segment Anything, and (optionally) store them on disc. I fyou are using a CPU, this step can take a while for 3d data or time-series (you will see a progress bar in the command-line interface / on the bootom right of napari). If you have access to a GPU without graphical interface (e.g. via a local computer cluster or a cloud provider), you can also pre-compute the embeddings there and then copy them over to your laptop / local machine to speed this up.
- You can use the command `micro_sam.precompute_embeddings` for this (it is installed with the rest of the software). You can specify the location of the precomputed embeddings via the `embedding_path` argument.
- You can cache the computed embedding in the napari tool (to avoid recomputing the embeddings again) by passing the path to store the embeddings in the `embeddings_save_path` option in the `Embedding Settings` drop-down. You can later load the precomputed image embeddings by entering the path to the stored embeddings there as well.

### 5. Can I use `micro_sam` on a CPU?
Most other processing steps that are very fast even on a CPU, the automatic segmentation step for the default Segment Anything models (typically called as the "Segment Anything" feature or AMG - Automatic Mask Generation) takes several minutes without a GPU (depending on the image size). For large volumes and time-series, segmenting an object interactively in 3d / tracking across time can take a couple of seconds with a CPU (it is very fast with a GPU).
> HINT: All the tutorial videos have been created on CPU resources.

### 6. I generated some segmentations from another tool, can I use it as a starting point in `micro_sam`?
You can save and load the results from the `committed_objects` layer to correct segmentations you obtained from another tool (e.g. CellPose) or  save intermediate annotation results. The results can be saved via `File` -> `Save Selected Layers (s) ...` in the napari menu-bar on top (see the tutorial videos for details). They can be loaded again by specifying the corresponding location via the `segmentation_result` parameter in the CLI or python script (2d and 3d segmentation).
If you are using an annotation tool you can load the segmentation you want to edit as segmentation layer and renae it to `committed_objects`.

### 7. I am using `micro_sam` for segmenting objects. I would like to report the steps for reproducability. How can this be done?
The annotation steps and segmentation results can be saved to a zarr file by providing the `commit_path` in the `commit` widget. This file will contain all relevant information to reproduce the segmentation.
> NOTE: This feature is still under development and we have not implemented rerunning the segmentation from this file yet. See [this issue](https://github.com/computational-cell-analytics/micro-sam/issues/408) for details.

### 8. I want to segment complex objects. Both the default Segment Anything models and the `micro_sam` generalist models do not work for my data. What should I do?
`micro_sam` supports interactive annotation using positive and negative point prompts, box prompts and polygon drawing. You can combine multiple types of prompts to improve the segmentation quality. In case the aforementioned suggestions do not work as desired, `micro_sam` also supports finetuning a model on your data (see the next section). We recommend the following: a) Check which of the provided models performs relatively good on your data, b) Choose the best model as the starting point to train your own specialist model for the desired segmentation task.

### 9. I am using the annotation tool and napari outputs the following error: `While emmitting signal ... an error ocurred in callback ... This is not a bug in psygnal. See ... above for details.`
These messages occur when an internal error happens in `micro_sam`. In most cases this hapepns due to inconsistent annotations and you can fix them by clearing the annotations and continuing.
We are currently aware of the following erros that cause these messages:
- While using `Automatic Segmentation` from the finetuned Segment Anything models, the model does not segment any objects. This often happens with the finetuned models, which need lower values for the `pred_iou_thresh` and the `stability_score_thresh`. You can change these in the GUI (under `Settings` in the `Automatic Segmentation` widget console for AMG-based segmentation). The best approach is the following: lower these thresholds to smaller values (e.g. 0.5 for both). This segments the objects you are interested in, however it also segments unwanted objects. Then choose higher values for the thresholds until the unwanted objects disappear (rerunning this step with different values will be fast after you run the segmentation the first time with lower values).
- While using interactive annotations, this could happen if the prompts are placed outside the image, the object is segmented using a negative point prompt, or the projection along multidimensional inputs places a projected prompt along the consecutive slices outside the image (etc.). The best approach here is: to clear the annotations and place fresh annotations to segment the object with an updated prompt-placement heuristic.

### 10. The objects are not segmented in my 3d data using the interactive annotation tool.
The first thing to check is: a) make sure you are using the latest version of `micro_sam` (pull the latest commit from master if your installation is from source, or update the installation from conda / mamba using `mamba update micro_sam`), and b) try out the steps from the [3d annotator tutorial video](TODO) to verify if this shows the same behaviour (or the same errors) as you faced. For 3d images, it's important to pass the inputs in the python axis convention, ZYX.
c) try using a different model and change the projection mode for 3d segmentation. This is also explained in the video.

### 11. I have very small or fine-grained structures in my high-resolution microscopic images. Can I use `micro_sam` to annotate them?
Segment Anything does not work well for very small or fine-grained objects (e.g. filaments). In these cases, you could try to use tiling to improve results (see [Point 3](#3-i-have-high-resolution-large-tomograms-micro-sam-does-not-seem-to-work) above for details).

### 12. napari seems to be very slow for large images.
Editing (drawing / erasing) very large 2d images or 3d volumes is known to be slow at the moment, as the objects in the layers are stored in-memory. See the related [issue](https://github.com/computational-cell-analytics/micro-sam/issues/39).

<!---
TODO continue from here
-->
### 13. While computing the embeddings (and / or automatic segmentation), a window stating: `"napari" is not responding.` pops up.
We think that `napari` expects to maintain an idle state while performing expensive computations in the background (like, computing the image embeddings, automatic mask generation, etc.). Our recommendation would be to avoid accessing any features in the napari GUI until the progress bar (on bottom-right) completes the background computations.


## Fine-tuning questions

### 1. I have a microscopy dataset I would like to fine-tune Segment Anything for. Is it possible using 'micro_sam'?
Yes, you can fine-tune Segment Anything on your own dataset. Here's how you can do it:
- Check out the [tutorial notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/micro-sam-finetuning.ipynb) on how to fine-tune Segment Anything in a few lines of code.
- Check out the [example](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/finetuning) script for fine-tune using the python library.
- TODO also mention the finetuining UI.

### 2. I would like to fine-tune Segment Anything on open-source cloud services (e.g. Kaggle Notebooks), is it possible?
Yes, you can fine-tune Segment Anything on your custom datasets on Kaggle (and [BAND](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-micro_sam-on-band)). Check out our [tutorial notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/micro-sam-finetuning.ipynb) for this.

### 3. I have finetuned Segment Anything on my microscopy data. How can I use it for annotating new images?
`micro_sam` is flexible in supporting the loading of custom weights out-of-the-box in the napari tool (add the path of your model checkpoints to `custom_weight_paths` in the `Settings` drop-down menu in the `Compute Embeddings` console), for initializing Segment Anything with your own finetuned models and using it for automatic and interactive segementation.

### 4. What is the background of the new AIS (Automatic Instance Segmentation) feature in `micro_sam`?
`micro_sam` introduces a new segmentation decoder to the Segment Anything backbone, for enabling faster and accurate automatic instance segmentation, by learning the [distances](https://github.com/constantinpape/torch-em/blob/main/torch_em/transform/label.py#L284) per object (to the center and to the boundary) and the foreground region, and performs seeded watershed-based [postprocessing](https://github.com/constantinpape/torch-em/blob/main/torch_em/util/segmentation.py#L122) to obtain the instances. However, it's a flexible wrap around the Segment Anything model, which provides the users to either fine-tune the Segment Anything model as it is, or the choice to fine-tune the Segment Anything model with an additional instance segmentation decoder (see the [example](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/finetuning#example-for-model-finetuning) for finetuning with both the objectives). The finetuned models provided by `micro_sam` use the AIS feature for improving the segmentation experience for the light microscopy and electron microscopy domains.

### 5. I have a NVIDIA RTX 4090Ti GPU with 24GB VRAM. Can I finetune Segment Anything?
Finetuning Segment Anything is possible in most consumer-grade GPU and CPU resources (the latter is infamous for taking significantly more time than the former). For the mentioned resource, it should be possible to finetune a ViT Base (also abbreviated as `vit_b`) by reducing the number of objects per image to ~15 (from our experience, this parameters complements to a fair share of impact on the GPU memory consumption, however the segmentation quality for finetuned models with a (much) lower number of objects than mentioned above is not detrimental - our latest [preprint](#how-to-cite-our-work) details out the aforementioned discussion).

### 6. I want to create dataloaders for my own data, for finetuning Segment Anything.
Thanks to `torch-em`, a) Creating PyTorch-supported datasets and dataloaders using the python library is convenient and supported for various data formats and multiple data structures. See the [tutorial](https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb) notebook on how to create dataloaders using `torch-em` and the [documentation](https://github.com/constantinpape/torch-em/blob/main/doc/datasets_and_dataloaders.md) on supporting the details for creating your own datasets and dataloaders; and b) Finetuning using the `napari` tool eases the aforementioned process, by allowing you to add the input parameters (path to the directory for inputs, training parameters, etc.) directly in the tool.
> NOTE: If you have images with large input shapes with a sparse density of instance segmentations, we recommend using [`sampler`](https://github.com/constantinpape/torch-em/blob/main/torch_em/data/sampler.py) for choosing the patches with valid segmentation for the finetuning purpose (see the [example](https://github.com/computational-cell-analytics/micro-sam/blob/master/finetuning/specialists/training/light_microscopy/plantseg_root_finetuning.py#L29) for PlantSeg (Root) specialist model in `micro_sam`).
