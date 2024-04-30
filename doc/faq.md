# FAQ

We have assembled the frequently asked questions and encountered issues in one place. We welcome you to have a look and see if one of the mentions answers your question(s). Feel free to open an issue in `micro-sam` if your problem needs attention.

## Installation

### 1. How to install `micro-sam`?
Ans. The [installation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation) for `micro-sam` is supported in three ways: [from mamba](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-mamba) (recommended), [from source](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-source) and [from installers](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-installer). Check out our [tutorial video](TODO) to get started with `micro-sam`, briefly walking you through the installation and the tool.

### 2. I cannot install `micro-sam` using the installer, I am getting some errors.
Ans. The installer should work out-of-the-box on Windows and Linux platforms. It's currently supported only for CPU users. [Here](https://github.com/computational-cell-analytics/micro-sam/issues/541) is a known issue for the Windows installer, which is already fixed in version 1.0.0. Feel free to report any issues encountered.
>NOTE: The installers are a quick-starter for `micro-sam`. We highly recommend using the installation from mamba / from source for making use of all the supported functionalities seamlessly.


### 3. What is the minimum system requirement for `micro-sam`?
Ans. From our experience, `micro-sam` works seamlessly on most CPU resources over 8GB RAM for the annotator tool. You might encounter some `napari`-related slowness for $\leq$ 8GB RAM. The resources `micro-sam`'s annotation tool has been tested on are:
- Windows:
    - Windows 10 Pro, Intel i5 7th Gen, 8GB RAM
- Linux:
    - Ubuntu 22.04, Intel i7 12th Gen, 32GB RAM
- Mac:
    - macOS Sonoma 14.4.1
        - M1 Chip, 8GB RAM
        - M3 Max Chip, 36GB RAM

### 4. What is the recommended PyTorch version to install?
Ans. `micro-sam` has been experimented on CUDA 12.1 and PyTorch [2.1.1, 2.2.0]. However, the tool and the library is not constrained to any PyTorch or CUDA versions (should work fine with standard pytorch installation, depending on your OS platform).

### 5. I am missing a few packages (eg. `ModuleNotFoundError: No module named 'elf.io`). What should I do?
Ans. With the latest release 1.0.0, the installation from mamba and source should take care of this and install all the relevant packages for you.

### 6. Can I install `micro-sam` using pip?
Ans. The installation is not supported for installation using pip (neither from pypi builds nor from the source repository).

### 7. I get the following error: `importError: cannot import name 'UNETR' from 'torch_em.model'`.
Ans. It's possible that you have an older version of `torch-em` installed. Similar errors could often be raised from other libraries, reason being: a) Outdated packages installed, or b) Some non-existent module being called. If the source of such error is from `micro-sam`, then `a)` is most likely the reason . We recommend installing the latest version from the installation instructions [here](https://github.com/constantinpape/torch-em?tab=readme-ov-file#installation) (or from the latest release of the related dependency causing this error).


## Usage (Annotator Tool / Library)

### 1. I have some micropscopy images. Can I use the annotator tool for segmentation?
Ans. Yes, you can use the annotator tool for:
- Segmenting objects in 2d images (using automatic and/or interactive segmentation).
- Segmenting objects in 3d volumes (using automatic and/or interactive segmentation for the entire object(s)).
- Tracking objects over time in time-series data.
- Segmenting objects in a series of 2d / 3d images.
- (OPTIONAL) You can finetune the Segment Anything / `micro-sam` models on your own microscopy data, in case the provided models do not suffice your needs. One caveat: You need to annotate a few objects before-hand (`micro-sam` has the potential of improving interactive segmentation with only a few annotated objects) to proceed with the supervised finetuning procedure.


### 2. I have high-resolution large tomograms, 'micro-sam' does not seem to work.
Ans. Segment Anything model backbone expects inputs of shape 1024 x 1024 pixels. Inputs that do not match this size will be internally resized to match it. Hence, applying Segment Anything to a much larger image will often lead to inferior results, or somethimes not working at all. To address this, `micro-sam` implements tiling: cutting up the input image into tiles of a fixed size (with a fixed overlap) and running Segment Anything for the individual tiles. You can activate tiling by passing the parameters `tile_shape`, which determines the size of the inner tile and `halo`, which determines the size of the additional overlap.
- If you are using the `micro-sam` napari-based GUI, you can specify the values for the `tile_shape` and `halo` via the `tile_x`, `tile_y`, `halo_x` and `halo_y` parameters in the `Settings` drop-down menu in the `Compute Embeddings` widget console.
- If you are using the `micro_sam` library in a python script, you can pass them as tuples, e.g. `tile_shape=(1024, 1024), halo=(256, 256)`. See also the [wholeslide annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py#L47-L63).
- If you are using the command line functionality, you can pass them via the options `--tile_shape 1024 1024 --halo 256 256`.
> NOTE 1: It's recommended to choose the `halo` such that it is larger than half of the maximal radius of the objects you want to segment.


### 3. The computation of image embeddings takes very long on my images in napari.
Ans. `micro-sam` pre-computes the image embeddings produced by the vision transformer backbone in Segment Anything, and (optionally) store them on disc. I fyou are using a CPU, this step can take a while for 3d data or time-series (you will see a progress bar in the command-line interface, and a rotating one inside the napari tool on bottom right). If you have access to a GPU without graphical interface (e.g. via a local computer cluster or a cloud provider), you can also pre-compute the embeddings there and then copy them over to your laptop / local machine to speed this up.
- You can use the command `micro_sam.precompute_embeddings` for this (it is installed with the rest of the applications). You can specify the location of the precomputed embeddings via the `embedding_path` argument.
- You can cache the computed embedding in the napari tool (to avoid recomputing the embeddings again-and-again) by passing the path to store the embeddings in the `embeddings_save_path` option in the `Settings` drop-down menu in the `Compute Embeddings` widget console. You can later load the precomputed image embeddings by entering the path to the stored embeddings.


### 4. Can I use `micro-sam` on a CPU?
Ans. Most other processing steps that are very fast even on a CPU, the automatic segmentation step for the default Segment Anything models (typically called as the "Segment Anything" feature or AMG - Automatic Mask Generation) takes several minutes without a GPU (depending on the image size). For large volumes and time-series, segmenting an object interactively in 3d / tracking across time can take a couple of seconds with a CPU (it is very fast with a GPU).

> HINT: All the tutorial videos have been created on CPU resources.


### 5. I generated some segmentations from another tool, can I use it as a starting point in `micro-sam`?
Ans. You can save and load the results from the `committed_objects` layer to correct segmentations you obtained from another tool (e.g. CellPose) or o save intermediate annotation results. The results can be saved via `File` -> `Save Selected Layers (s) ...` in the napari menu-bar on top (see the tutorial videos for details). They can be loaded again by specifying the corresponding location via the `segmentation_result` (2d and 3d segmentation).


### 6. I am using `micro-sam` for segmenting objects. I would like to report the steps for reproducability. How can this be done?
Ans. TODO: @CP: We discussed this about storing the input prompts smh, I totally forgot the status of it. Could you check this one out? Thanks!


### 7. I have complex objects to segment. Both, the default Segment Anything models and `micro-sam` generalist models do not work for my data. What should I do?
Ans. `micro-sam` supports interactive annotation using positive and negative point prompts, box prompts and freehand polygon tool. You can combine multiple types of prompts from above to improve the segmentation quality. In case the aforementioned suggestions do not work as desired, `micro-sam` now supports finetuning using the napari-tool and the python library. We recommend the following: a) Check which of the provided model(s) perform relatively good on your data, b) Choose the better performing model as the starting point to train your own specialist model for the desired segmentation task. 


## Fine-tuning

### 1. I have a microscopy dataset I would like to fine-tune Segment Anything for. Is it possible using 'micro-sam'?
Ans. Yes, you can fine-tune Segment Anything on your own dataset. Here's how you can do it:
- Check out the [tutorial notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/micro-sam-finetuning.ipynb) on how to fine-tune Segment Anything in a few lines of code.
- Check out the [example](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/finetuning) script for fine-tune using the python library.


### 2. I would like to fine-tune Segment Anything on open-source cloud services (e.g. Kaggle Notebooks), is it possible?
Ans. Yes, you can fine-tune Segment Anything on your custom datasets on Kaggle (and [BAND](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-micro_sam-on-band)). Check out our [tutorial notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/micro-sam-finetuning.ipynb) for this.


### 3. I have finetuned Segment Anything on my microscopy data. How can I use it for annotating new images?
Ans. `micro-sam` is flexible in supporting the loading of custom weights out-of-the-box in the napari tool (add the path of your model checkpoints to `custom_weight_paths` in the `Settings` drop-down menu in the `Compute Embeddings` console), for initializing Segment Anything with your own finetuned models and using it for automatic and interactive segementation.


## Known Limitations

- Segment Anything does not work well for very small or fine-grained objects (e.g. filaments)


## How to cite our work?

- Segment Anything for Microscopy (pre-print): https://doi.org/10.1101/2023.08.21.554208

```
@article{archit2023segment,
    title={Segment Anything for Microscopy},
    author={Archit, Anwai and Nair, Sushmita and Khalid, Nabeel and Hilt, Paul and Rajashekar, Vikas and Freitag, Marei and Gupta, Sagnik and Dengel, Andreas and Ahmed, Sheraz and Pape, Constantin},
    journal={bioRxiv},
    pages={2023--08},
    year={2023},
    publisher={Cold Spring Harbor Laboratory},
    url={https://doi.org/10.1101/2023.08.21.554208}
}
