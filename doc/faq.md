# FAQ

Here we provide frequently asked questions and common issues.
If you encounter a problem or question not addressed here feel free to [open an issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) or to ask your question on [image.sc](https://forum.image.sc/) with the tag `micro-sam`.

## Installation questions


### 1. How to install `micro_sam`?
The [installation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation) for `micro_sam` is supported in three ways: [from mamba](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-mamba) (recommended), [from source](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-source) and [from installers](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-installer). Check out our [tutorial video](https://youtu.be/gcv0fa84mCc) to get started with `micro_sam`, briefly walking you through the installation process and how to start the tool.


### 2. I cannot install `micro_sam` using the installer, I am getting some errors.
The installer should work out-of-the-box on Windows and Linux platforms. Please open an issue to report the error you encounter.
>NOTE: The installers enable using `micro_sam` without mamba or conda. However, we recommend the installation from mamba / from source to use all its features seamlessly. Specifically, the installers currently only support the CPU and won't enable you to use the GPU (if you have one). 


### 3. What is the minimum system requirement for `micro_sam`?
From our experience, the `micro_sam` annotation tools work seamlessly on most laptop or workstation CPUs and with > 8GB RAM.
You might encounter some slowness for $\leq$ 8GB RAM. The resources `micro_sam`'s annotation tools have been tested on are:
- Windows:
    - Windows 10 Pro, Intel i5 7th Gen, 8GB RAM
    - Windows 10 Enterprise LTSC, Intel i7 13th Gen, 32GB RAM
    - Windows 10 Pro for Workstations, Intel Xeon W-2295, 128GB RAM
    
- Linux:
    - Ubuntu 20.04, Intel i7 11th Gen, 32GB RAM
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
So please reinstall `micro_sam`, following [the installation guide](#installation).

### 6. Can I install `micro_sam` using pip?
We do *not* recommend installing `micro-sam` with pip. It has several dependencies that are only avaoiable from conda-forge, which will not install correctly via pip.

Please see [the installation guide](#installation) for the recommended way to install `micro-sam`.

The PyPI page for `micro-sam` exists only so that the [napari-hub](https://www.napari-hub.org/) can find it.

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
- In addition, you can finetune the Segment Anything / `micro_sam` models on your own microscopy data, in case the provided models do not suffice your needs. One caveat: You need to annotate a few objects before-hand (`micro_sam` has the potential of improving interactive segmentation with only a few annotated objects) to proceed with the supervised finetuning procedure.


### 2. Which model should I use for my data?
We currently provide three different kind of models: the default models `vit_h`, `vit_l`, `vit_b` and `vit_t`; the models for light microscopy `vit_l_lm`, `vit_b_lm` and `vit_t_lm`; the models for electron microscopy `vit_l_em_organelles`, `vit_b_em_organelles` and `vit_t_em_organelles`.
You should first try the model that best fits the segmentation task your interested in, the `lm` model for cell or nucleus segmentation in light microscopy or the `em_organelles` model for segmenting nuclei, mitochondria or other roundish organelles in electron microscopy.
If your segmentation problem does not meet these descriptions, or if these models don't work well, you should try one of the default models instead.
The letter after `vit` denotes the size of the image encoder in SAM, `h` (huge) being the largest and `t` (tiny) the smallest. The smaller models are faster but may yield worse results. We recommend to either use a `vit_l` or `vit_b` model, they offer the best trade-off between speed and segmentation quality.
You can find more information on model choice [here](#choosing-a-model).


### 3. I have high-resolution microscopy images, `micro_sam` does not seem to work.
The Segment Anything model expects inputs of shape 1024 x 1024 pixels. Inputs that do not match this size will be internally resized to match it. Hence, applying Segment Anything to a much larger image will often lead to inferior results, or sometimes not work at all. To address this, `micro_sam` implements tiling: cutting up the input image into tiles of a fixed size (with a fixed overlap) and running Segment Anything for the individual tiles. You can activate tiling with the `tile_shape` parameter, which determines the size of the inner tile and `halo`, which determines the size of the additional overlap.
- If you are using the `micro_sam` annotation tools, you can specify the values for the `tile_shape` and `halo` via the `tile_x`, `tile_y`, `halo_x` and `halo_y` parameters in the `Embedding Settings` drop-down menu.
- If you are using the `micro_sam` library in a python script, you can pass them as tuples, e.g. `tile_shape=(1024, 1024), halo=(256, 256)`. See also the [wholeslide annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py#L47-L63).
- If you are using the command line functionality, you can pass them via the options `--tile_shape 1024 1024 --halo 256 256`.
> NOTE: It's recommended to choose the `halo` so that it is larger than half of the maximal radius of the objects you want to segment.


### 4. The computation of image embeddings takes very long in napari.
`micro_sam` pre-computes the image embeddings produced by the vision transformer backbone in Segment Anything, and (optionally) stores them on disc. I fyou are using a CPU, this step can take a while for 3d data or time-series (you will see a progress bar in the command-line interface / on the bottom right of napari). If you have access to a GPU without graphical interface (e.g. via a local computer cluster or a cloud provider), you can also pre-compute the embeddings there and then copy them over to your laptop / local machine to speed this up.
- You can use the command `micro_sam.precompute_embeddings` for this (it is installed with the rest of the software). You can specify the location of the pre-computed embeddings via the `embedding_path` argument.
- You can cache the computed embedding in the napari tool (to avoid recomputing the embeddings again) by passing the path to store the embeddings in the `embeddings_save_path` option in the `Embedding Settings` drop-down. You can later load the pre-computed image embeddings by entering the path to the stored embeddings there as well.


### 5. Can I use `micro_sam` on a CPU?
Most other processing steps are very fast even on a CPU, the automatic segmentation step for the default Segment Anything models (typically called as the "Segment Anything" feature or AMG - Automatic Mask Generation) however takes several minutes without a GPU (depending on the image size). For large volumes and time-series, segmenting an object interactively in 3d / tracking across time can take a couple of seconds with a CPU (it is very fast with a GPU).
> HINT: All the tutorial videos have been created on CPU resources.


### 6. I generated some segmentations from another tool, can I use it as a starting point in `micro_sam`?
You can save and load the results from the `committed_objects` layer to correct segmentations you obtained from another tool (e.g. CellPose) or save intermediate annotation results. The results can be saved via `File` -> `Save Selected Layers (s) ...` in the napari menu-bar on top (see the tutorial videos for details). They can be loaded again by specifying the corresponding location via the `segmentation_result` parameter in the CLI or python script (2d and 3d segmentation).
If you are using an annotation tool you can load the segmentation you want to edit as segmentation layer and rename it to `committed_objects`.


### 7. I am using `micro_sam` for segmenting objects. I would like to report the steps for reproducability. How can this be done?
The annotation steps and segmentation results can be saved to a Zarr file by providing the `commit_path` in the `commit` widget. This file will contain all relevant information to reproduce the segmentation.
> NOTE: This feature is still under development and we have not implemented rerunning the segmentation from this file yet. See [this issue](https://github.com/computational-cell-analytics/micro-sam/issues/408) for details.


### 8. I want to segment objects with complex structures. Both the default Segment Anything models and the `micro_sam` generalist models do not work for my data. What should I do?
`micro_sam` supports interactive annotation using positive and negative point prompts, box prompts and polygon drawing. You can combine multiple types of prompts to improve the segmentation quality. In case the aforementioned suggestions do not work as desired, `micro_sam` also supports finetuning a model on your data (see the next section on [finetuning](#fine-tuning-questions)). We recommend the following: a) Check which of the provided models performs relatively good on your data, b) Choose the best model as the starting point to train your own specialist model for the desired segmentation task.


### 9. I am using the annotation tool and napari outputs the following error: `While emmitting signal ... an error ocurred in callback ... This is not a bug in psygnal. See ... above for details.`
These messages occur when an internal error happens in `micro_sam`. In most cases this is due to inconsistent annotations and you can fix them by clearing the annotations.
We want to remove these errors, so we would be very grateful if you can [open an issue](https://github.com/computational-cell-analytics/micro-sam/issues) and describe the steps you did when encountering it.


### 10. The objects are not segmented in my 3d data using the interactive annotation tool.
The first thing to check is: a) make sure you are using the latest version of `micro_sam` (pull the latest commit from master if your installation is from source, or update the installation from conda / mamba using `mamba update micro_sam`), and b) try out the steps from the [3d annotation tutorial video](https://youtu.be/nqpyNQSyu74) to verify if this shows the same behaviour (or the same errors) as you faced. For 3d images, it's important to pass the inputs in the python axis convention, ZYX.
c) try using a different model and change the projection mode for 3d segmentation. This is also explained in the video.


### 11. I have very small or fine-grained structures in my high-resolution microscopic images. Can I use `micro_sam` to annotate them?
Segment Anything does not work well for very small or fine-grained objects (e.g. filaments). In these cases, you could try to use tiling to improve results (see [Point 3](#3-i-have-high-resolution-large-tomograms-micro-sam-does-not-seem-to-work) above for details).


### 12. napari seems to be very slow for large images.
Editing (drawing / erasing) very large 2d images or 3d volumes is known to be slow at the moment, as the objects in the layers are stored in-memory. See the related [issue](https://github.com/computational-cell-analytics/micro-sam/issues/39).


### 13. While computing the embeddings (and / or automatic segmentation), a window stating: `"napari" is not responding` pops up.
This can happen for long running computations. You just need to wait a bit longer and the computation will finish.


## Fine-tuning questions


### 1. I have a microscopy dataset I would like to fine-tune Segment Anything for. Is it possible using `micro_sam`?
Yes, you can fine-tune Segment Anything on your own dataset. Here's how you can do it:
- Check out the [tutorial notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/micro-sam-finetuning.ipynb) on how to fine-tune Segment Anything with our `micro_sam.training` library.
- Or check the [examples](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/finetuning) for additional scripts that demonstrate finetuning.
- If you are not familiar with coding in python at all then you can also use the [graphical interface for finetuning](finetuning-ui). But we recommend using a script for more flexibility and reproducibility.


### 2. I would like to fine-tune Segment Anything on open-source cloud services (e.g. Kaggle Notebooks), is it possible?
Yes, you can fine-tune Segment Anything on your custom datasets on Kaggle (and [BAND](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-micro_sam-on-band)). Check out our [tutorial notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/micro-sam-finetuning.ipynb) for this.


<!---
TODO: we should improve this explanation and add a small image that visualizes the labels.
-->
### 3. What kind of annotations do I need to finetune Segment Anything?
Annotations are referred to the instance segmentation labels, i.e. each object of interests in your microscopy images have an individual id to uniquely identify all the segmented objects. You can obtain them by `micro_sam`'s annotation tools. In `micro_sam`, it's expected to provide dense segmentations (i.e. all objects per image are annotated) for finetuning Segment Anything with the additional decoder, however it's okay to use sparse segmentations (i.e. few objects per image are annotated) for just finetuning Segment Anything (without the additional decoder).


### 4. I have finetuned Segment Anything on my microscopy data. How can I use it for annotating new images?
You can load your finetuned model by entering the path to its checkpoint in the `custom_weights_path` field in the `Embedding Settings` drop-down menu.
If you are using the python library or CLI you can specify this path with the `checkpoint_path` parameter.


### 5. What is the background of the new AIS (Automatic Instance Segmentation) feature in `micro_sam`?
`micro_sam` introduces a new segmentation decoder to the Segment Anything backbone, for enabling faster and accurate automatic instance segmentation, by predicting the [distances to the object center and boundary](https://github.com/constantinpape/torch-em/blob/main/torch_em/transform/label.py#L284) as well as predicting foregrund, and performing [seeded watershed-based postprocessing](https://github.com/constantinpape/torch-em/blob/main/torch_em/util/segmentation.py#L122) to obtain the instances.


### 6. I want to finetune only the Segment Anything model without the additional instance decoder.
The instance segmentation decoder is optional. So you can only finetune SAM or SAM and the additional decoder. Finetuning with the decoder will increase training times, but will enable you to use AIS. See [this example](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/finetuning#example-for-model-finetuning) for finetuning with both the objectives.

> NOTE: To try out the other way round (i.e. the automatic instance segmentation framework without the interactive capability, i.e. a UNETR: a vision transformer encoder and a convolutional decoder), you can take inspiration from this [example on LIVECell](https://github.com/constantinpape/torch-em/blob/main/experiments/vision-transformer/unetr/for_vimunet_benchmarking/run_livecell.py).


### 7. I have a NVIDIA RTX 4090Ti GPU with 24GB VRAM. Can I finetune Segment Anything?
Finetuning Segment Anything is possible in most consumer-grade GPU and CPU resources (but training being a lot slower on the CPU). For the mentioned resource, it should be possible to finetune a ViT Base (also abbreviated as `vit_b`) by reducing the number of objects per image to 15.
This parameter has the biggest impact on the VRAM consumption and quality of the finetuned model.
You can find an overview of the resources we have tested for finetuning [here](#training-your-own-model).
We also provide a the convenience function `micro_sam.training.train_sam_for_configuration` that selects the best training settings for these configuration. This function is also used by the finetuning UI.

             
### 8. I want to create a dataloader for my data, to finetune Segment Anything.
Thanks to `torch-em`, a) Creating PyTorch datasets and dataloaders using the python library is convenient and supported for various data formats and data structures.
See the [tutorial notebook](https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb) on how to create dataloaders using `torch-em` and the [documentation](https://github.com/constantinpape/torch-em/blob/main/doc/datasets_and_dataloaders.md) for details on creating your own datasets and dataloaders; and b) finetuning using the `napari` tool eases the aforementioned process, by allowing you to add the input parameters (path to the directory for inputs and labels etc.) directly in the tool.
> NOTE: If you have images with large input shapes with a sparse density of instance segmentations, we recommend using [`sampler`](https://github.com/constantinpape/torch-em/blob/main/torch_em/data/sampler.py) for choosing the patches with valid segmentation for the finetuning purpose (see the [example](https://github.com/computational-cell-analytics/micro-sam/blob/master/finetuning/specialists/training/light_microscopy/plantseg_root_finetuning.py#L29) for PlantSeg (Root) specialist model in `micro_sam`).


### 9. How can I evaluate a model I have finetuned?
To validate a Segment Anything model for your data, you have different options, depending on the task you want to solve and whether you have segmentation annotations for your data.

- If you don't have any annotations you will have to validate the model visually. We suggest doing this with the `micro_sam` GUI tools. You can learn how to use them in the `micro_sam` documentation.
- If you have segmentation annotations you can use the `micro_sam` python library to evaluate the segmentation quality of different models. We provide functionality to evaluate the models for interactive and for automatic segmentation:
    - You can use `micro_sam.evaluation.evaluation.run_evaluation_for_iterative_prompting` to evaluate models for interactive segmentation.
    - You can use `micro_sam.evaluation.instance_segmentation.run_instance_segmentation_grid_search_and_inference` to evaluate models for automatic segmentation.

We provide an [example notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/inference_and_evaluation.ipynb) that shows how to use this evaluation functionality.
