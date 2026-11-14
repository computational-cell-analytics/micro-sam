# FAQ

Here we provide frequently asked questions and common issues.
If you encounter a problem or question not addressed here feel free to [open an issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) or to ask your question on [image.sc](https://forum.image.sc/) with the tag `micro-sam`.

## Installation questions


### 0. I am not familiar with `conda` / `mamba` / `micromamba` and would like to get started with a python package manager.
If you are getting started from scratch, we recommend using `micromamba` to setup your own package manager and creating a new environment with the required dependencies. You can install `micromamba` by following the instructions [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) under the `Automatic install` section. If you are using Linux, macOS, or Git Bash on Windows, the simplest way to of installing `micromamba` is to simply execute the installation script below in your preferred shell:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

If you are using Windows PowerShell, you should be able to install `micromamba` by running the following command:

```bash
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)
```

Once you have `micromamba` installed, you can create a new environment with the required dependencies by following the [installation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation) instructions for `micro_sam` (remember to replace `conda` by `micromamba` in the installation commands. For example: to check the list of available environments, `conda env list` becomes `micromamba env list`).

If you have any troubles in the aforementioned steps, remember to first check the [official documentation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for troubleshooting details.


### 1. How to install `micro_sam`?
The [installation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation) for `micro_sam` is supported in four ways: [from pip](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-pip), [from conda](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-conda), [from source](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-source), and [from installers](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-installer).
We recommend installation from pip or conda for most users.
Check out our [tutorial video](https://youtu.be/gcv0fa84mCc) to get started with `micro_sam`, briefly walking you through the installation process and how to start the tool.


### 2. I cannot install `micro_sam` using the installer, I am getting some errors.
The installer should work out-of-the-box on Windows and Linux platforms. Please open an issue to report the error you encounter.
>NOTE: The installers enable using `micro_sam` without conda. However, we recommend the installation from pip, from conda, or from source to use all its features seamlessly. Specifically, the installers are out-of-date and currently only support the CPU and won't enable you to use the GPU (if you have one). 


### 3. What is the minimum system requirement for `micro_sam`?
From our experience, the `micro_sam` annotation tools work seamlessly on most laptop or workstation CPUs and with > 8GB RAM.
You might encounter some slowness for >= 8GB RAM. The resources `micro_sam`'s annotation tools have been tested on are:
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
`micro_sam` needs PyTorch 2.5 or newer. Apart from that, the tool and the library are not constrained to a specific PyTorch or CUDA version, so they should work fine with the standard PyTorch installation for your system.


### 5. I am missing a few packages (eg. `ModuleNotFoundError: No module named 'elf.io`). What should I do?
With the latest release 1.0.0, the installation from conda and source should take care of this and install all the relevant packages for you.
So please reinstall `micro_sam`, following [the installation guide](#installation).


### 6. Can I install `micro_sam` using pip?

Yes! We recently made available `micro_sam` with pip. Check out [the installation guide](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-pip) for details.


### 7. I get the following error: `importError: cannot import name 'UNETR' from 'torch_em.model'`.
It's possible that you have an older version of `torch-em` installed. Similar errors could often be raised from other libraries, the reasons being: a) Outdated packages installed, or b) Some non-existent module being called. If the source of such error is from `micro_sam`, then `a)` is most likely the reason . We recommend installing the latest version following the [installation instructions](https://github.com/constantinpape/torch-em?tab=readme-ov-file#installation).


### 8. My system does not have a internet connection. Where should I put the model checkpoints for the `micro-sam` models?
The simplest way is to download the models on a computer with internet connection with `micro_sam info --download <MODEL>`, for example `micro_sam info --download hvit_t_cells`, and to copy the model cache directory to the computer without internet connection.
You can find the model cache directory by running `python -c "from micro_sam import util; print(util.get_model_cache_directory())"`.

You can also download the checkpoints by hand and move them to the model cache directory. The files must be named like the keys of `FINETUNED_URLS` in `micro_sam/v2/util.py`. A `micro-sam2` model has two files, for example `hvit_t_cells` and `hvit_t_cells_decoder`.
```bash
# Find the model cache directory.
python -c "from micro_sam import util; print(util.get_model_cache_directory())"
# Create it, if it does not exist, and move the checkpoints there with the expected names.
mkdir -p <MODEL_CACHE_DIRECTORY>
mv <DOWNLOADED_MODEL> <MODEL_CACHE_DIRECTORY>/hvit_t_cells
mv <DOWNLOADED_DECODER> <MODEL_CACHE_DIRECTORY>/hvit_t_cells_decoder
```


### 9. napari crashes with a segmentation fault (`Fatal Python error: Segmentation fault`) when I open one of the `micro_sam` annotators (Linux).
This is caused by a corrupted or stale system font cache, not by `micro_sam` itself. Qt segfaults inside `fontconfig` while laying out text. This shows up in two common variants:
- On launch, before any widget appears, while Qt renders the napari welcome screen. The crash traceback ends in `libfontconfig` (e.g. `FcCharSetHasChar` / `FcCharSetFindLeafForward`), reached via `QFontEngineMultiFontConfig::shouldLoadFontEngineForCharacter` during font fallback.
- After the annotator opens, when the collapsible "Settings" section (from `superqt`) draws small arrow glyphs (`▲` / `▼`). Here Qt may resolve the default font to an unusable font file (e.g. a `.woff` web font) and Qt's bundled FreeType/HarfBuzz segfaults rendering those glyphs. The traceback ends in `superqt/collapsible/_collapsible.py` in `_convert_string_to_icon`.

Both have the same cause and the same fix. The most reliable fix is to clear the user font cache directory and rebuild it, since a stale cache often holds entries from several `fontconfig` versions (e.g. `*.cache-7`, `*.cache-10`, `*.cache-11`) that a newer `fontconfig` then misreads:
```bash
rm -rf ~/.cache/fontconfig && fc-cache -f
```
Rebuilding without clearing (`fc-cache -f` alone) is sometimes enough for the `.woff` variant, but is not reliable for the cross-version cache problem. You can verify the fix by checking that the default font no longer resolves to a `.woff` (or otherwise broken) file:
```bash
fc-match "Sans Serif"   # should return a real font such as NotoSans-Regular.ttf, not a .woff
```

### 10. napari fails to render with `RuntimeError: Cannot FRAMEBUFFER object ... because it does not exist` or `OpenGL.error.Error: Attempt to retrieve context when no valid context` (Linux / Wayland).
This is an OpenGL context mismatch, not a `micro_sam` or napari bug. napari renders points with the vispy `gl+` backend, which routes OpenGL calls through `PyOpenGL`. On a Wayland session `PyOpenGL` defaults to its EGL backend and looks for the current context with `eglGetCurrentContext()`. However, when the napari window runs under XWayland (Qt falls back to the `xcb` platform because the Wayland Qt plugin is missing), the actual OpenGL context is a GLX context, which EGL cannot see. `PyOpenGL` then reports "no valid context" and the draw fails. The crash only appears once the canvas actually draws (e.g. after opening an image), not on the empty welcome screen.

To fix it, tell `PyOpenGL` to use the GLX backend so it matches the context Qt created. For a single run:
```bash
PYOPENGL_PLATFORM=glx napari
```
Or set it for the whole shell session and then launch napari:
```bash
export PYOPENGL_PLATFORM=glx
napari
```
To make it permanent for a conda / micromamba environment, add it to an activation script so it is set automatically on every `activate`:
```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export PYOPENGL_PLATFORM=glx' > "$CONDA_PREFIX/etc/conda/activate.d/pyopengl_platform.sh"
```


## Usage questions

<!---
TODO provide relevant links here.
-->
### 1. I have some microscopy images. Can I use the annotator tool for segmenting them?
Yes, you can use the annotator tool for:
- Segmenting objects in 2d images (using automatic and/or interactive segmentation).
- Segmenting objects in 3d volumes (using automatic and/or interactive segmentation for the entire object(s)).
- Tracking objects over time in time-series data.
- Segmenting objects in a series of 2d / 3d images.
- In addition, you can finetune the Segment Anything / `micro_sam` models on your own microscopy data, in case the provided models do not suffice your needs. One caveat: You need to annotate a few objects before-hand (`micro_sam` has the potential of improving interactive segmentation with only a few annotated objects) to proceed with the supervised finetuning procedure.


### 2. Which model should I use for my data?
We provide the original SAM2 models (`hvit_t`, `hvit_s`, `hvit_b` and `hvit_l`) and the `micro-sam2` models: `hvit_t_cells`, `hvit_s_cells`, `hvit_b_cells` and `hvit_l_cells` for cells and nuclei in light microscopy and neurites in electron microscopy (2D and 3D), `hvit_s_em_mito` for mitochondria and `hvit_s_em_nuclei` for nuclei in electron microscopy.
You should first try the `micro-sam2` model that best fits your segmentation task. If your segmentation problem does not meet these descriptions, or if these models don't work well, you can try one of the original SAM2 models for interactive segmentation, or finetune a model on your data.
The letter after `hvit` denotes the size of the image encoder: `t` (tiny), `s` (small), `b` (base) and `l` (large). The smaller models are faster, the larger models are more accurate, especially for automatic segmentation.
You can find more information on model choice [here](#choosing-a-model).


### 3. I have high-resolution microscopy images, `micro_sam` does not seem to work.
The model resizes every input to 1024 x 1024 pixels. Applying it to a much larger image shrinks the objects and leads to inferior results. To address this, `micro_sam` implements tiling: it cuts the input image into tiles of a fixed size (with a fixed overlap) and runs the model on the individual tiles.
Tiling is turned on by default for images that are larger than 768 pixels, with a tile shape of 512 x 512 and an overlap (halo) of 128 pixels. You can change the tiling with the `tile_shape` and `halo` parameters:
- In the annotation tools, set `tile_x`, `tile_y`, `halo_x` and `halo_y` in the `Embedding Settings`.
- In a python script, pass them as tuples, e.g. `tile_shape=(1024, 1024), halo=(256, 256)`. See also the [wholeslide annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/annotator_2d.py).
- In the command line, pass them with `--tile_shape` and `--overlap`, e.g. `--tile_shape 1024 1024 --overlap 256 256` for `micro_sam annotator` and `--tile_shape 1024,1024 --overlap 256,256` for `micro_sam inference`.
> NOTE: It's recommended to choose the `halo` so that it is larger than half of the maximal radius of the objects you want to segment.


### 4. The computation of image embeddings takes very long in napari.
`micro_sam` pre-computes the image embeddings produced by the image encoder, and (optionally) stores them on disc. If you are using a CPU, this step can take a while for 3d data or time-series (you will see a progress bar on the bottom right of napari). If you have access to a GPU without graphical interface (e.g. via a local computer cluster or a cloud provider), you can also pre-compute the embeddings there and then copy them over to your laptop / local machine to speed this up.
- You can use the command `micro_sam precompute-embeddings` for this (it is installed with the rest of the software). You can specify the location of the pre-computed embeddings via the `-e` / `--embedding_path` argument.
- You can cache the computed embedding in the napari tool (to avoid recomputing the embeddings again) by passing the path to store the embeddings in the `embeddings_save_path` option in the `Embedding Settings` drop-down. You can later load the pre-computed image embeddings by entering the path to the stored embeddings there as well.


### 5. Can I use `micro_sam` on a CPU?
Yes. Interactive segmentation of a 2d image is fast on a CPU. Computing the embeddings, automatic segmentation, and segmenting or tracking objects in volumes and time-series take longer without a GPU. Automatic prompt generation (APG) for volumes and time-series needs a GPU.


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
The first thing to check is: a) make sure you are using the latest version of `micro_sam` (pull the latest commit from main if your installation is from source, or update the installation from conda using `conda update micro_sam`), and b) make sure the tool treats your data as a 3d volume: you can check and change this in the embedding menu. For 3d images, it's important to pass the inputs in the python axis convention, ZYX.
c) try using a different model, and add prompts in more slices of the object before you segment it in the volume with `Apply to Volume`.


### 11. I have very small or fine-grained structures in my high-resolution microscopic images. Can I use `micro_sam` to annotate them?
The models do not work well for very small or fine-grained objects (e.g. filaments). In these cases, you could try to use tiling to improve results (see question 3 above for details).


### 12. napari seems to be very slow for large images.
Editing (drawing / erasing) very large 2d images or 3d volumes is known to be slow at the moment, as the objects in the layers are stored in-memory. See the related [issue](https://github.com/computational-cell-analytics/micro-sam/issues/39).


### 13. While computing the embeddings (and / or automatic segmentation), a window stating: `"napari" is not responding` pops up.
This can happen for long running computations. You just need to wait a bit longer and the computation will finish.


### 14. I have 3D RGB microscopy volumes. How does `micro_sam` handle these images?
The expected shape of a 3D volume with channels is `(Z, Y, X, C)`, with the channels in the last axis. A volume with 2 or 4 channels is mapped to 3 channels: 2 channels are padded with an empty channel, and of 4 channels the first 3 are used. A volume with a single channel is replicated to 3 channels, because the model expects 3-channel inputs.


### 15. I want to use a model stored in a different directory than the `micro_sam` cache. How can I do this?
The `micro_sam` command line tools (`micro_sam annotator`, `micro_sam inference` and `micro_sam precompute-embeddings`) accept the argument `-c` / `--checkpoint` to pass a model checkpoint. If you start a `micro_sam` annotator from the `napari` plugin menu, you can provide the path to the model checkpoint under the `Embedding Settings` drop-down in the `custom weights path` option. In python, pass it as `checkpoint_path`.

The model size is read from the checkpoint, so you do not have to match it with the model type.

For automatic segmentation, the model also needs its decoder. Give a checkpoint that holds the decoder, such as the `best.pt` of joint or automatic training, or a model file with its decoder file next to it: for example `hvit_t_cells.pt` with `hvit_t_cells_decoder.pt`. The decoder file is found by its name. In python and in the command line of the annotators, you can also give the decoder file with `decoder_path` (`--decoder_path`).


### 16. Some parameters in the annotation tools are unclear to me.
`micro-sam` has tooltips for menu options across all widgets (i.e. an information window will appear if you hover over name of the menu), which briefly describe the utility of the specific menu option.


### 17. I want to use an older version of the pretrained models.
The SAM1 models of the previous version are available in the `micro_sam.v1` subpackage and the `micro_sam v1` command line tools. Older versions of these models are still available on zenodo. You can find the download links for all of them [here](#older-sam1-models).


### 18. I would like to evaluate the instance segmentation quantitatively. Can you suggest how to do that?
`micro_sam v1 evaluate` computes the mean segmentation accuracy (introduced in the Pascal VOC challenge) of predicted instance segmentations with the corresponding ground-truth annotations. It works for segmentations from any method. Please see our paper (`Methods` -> `Inference and Evaluation` for more details about it) and `micro_sam v1 evaluate --help` for more details.


### 19. I get `RuntimeError: GET was unable to find an engine to execute this computation` on a V100 GPU (*"or any older GPU"*).
This is a known issue for a combination of older generation GPUs (eg. V100s) and pytorch compiled with the latest CUDA Toolkit (eg. CUDA 12.9 and PyTorch 2.8 has been tested to throw this error on V100s).
Here's what you can do to solve this issue:
- Use a PyTorch/CUDA build that is known to work with V100, for example CUDA 12.1 or 11.8 with a compatible PyTorch version (please check your installed CUDA drivers).
- Run on CPU (slower, but works).

### 20. The automatic instance segmentation does a very poor job for my microscopy images.
This can have multiple reasons that you can address by choosing the best settings for your image:
- First, choose the most suitable model for your image type and segmentation task:
    - `hvit_t_cells` (or a larger `hvit_*_cells` model) for any cell or nucleus segmentation task in light microscopy or related tasks such as segmentation of organoids or similar structures.
    - `hvit_s_em_mito` for segmenting mitochondria and `hvit_s_em_nuclei` for segmenting nuclei in electron microscopy.
- Try the other automatic segmentation methods: APG is often more accurate than AIS, see [APG](#apg).
- If you have multi-channel images then you have to pass those in a format and channel order compatible with `micro_sam`:
    - If your image has 2 or 3 channels you can use it as is. The channels must be given in the last axis.
        - If you have images with a fluorescent marker staining the cytosol or membrane and another marker staining the nucleus, then pass the cytosol/membrane maker as first channel and the nucleus marker as second channel.
    - If you have more than 3 channels then you have to reduce the number of channels to 3, either by selecting the most relevant channels (see previous item) or by averaging or otherwise combining channels in a suitable fashion.
- Large images are tiled by default (see question 3 above). If the objects are very large or very small compared to the tiles, change the tile shape and the halo.
- If your image contains very small objects that you want to segment then decrease the minimum object size by passing a low value (e.g. 10) for `min_size` to the CLI / python function you are using.

If none of these steps help then the available `micro_sam` models are likely not well suited for your data. Note that you can still use `micro_sam` for your segmentation problem, but you will have to fine-tune your own model (see the next section).


## Fine-tuning questions

### 1. I have a microscopy dataset I would like to fine-tune Segment Anything for. Is it possible using `micro_sam`?
Yes, you can finetune the `micro-sam2` and SAM2 models on your own dataset. Here's how you can do it:
- Use `micro_sam train` in the command line, or the `micro_sam.v2.training` library, see [training your own model](#training-your-own-model).
- Check out the [tutorial notebook](https://github.com/computational-cell-analytics/micro-sam/blob/main/notebooks/sam_finetuning.ipynb) and the [examples](https://github.com/computational-cell-analytics/micro-sam/tree/main/examples/finetuning) for scripts that demonstrate finetuning.


### 2. I would like to fine-tune Segment Anything on open-source cloud services (e.g. Kaggle Notebooks), is it possible?
Yes, you can fine-tune Segment Anything on your custom datasets on Kaggle (and [BAND](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-micro_sam-on-band)). Check out our [tutorial notebook](https://github.com/computational-cell-analytics/micro-sam/blob/main/notebooks/sam_finetuning.ipynb) for this.


<!---
TODO: we should improve this explanation and add a small image that visualizes the labels.
-->
### 3. What kind of annotations do I need to finetune Segment Anything?
You need instance segmentation labels, i.e. each object of interest in your microscopy images has an individual id to uniquely identify it. You can obtain them with `micro_sam`'s annotation tools. For training the decoder for automatic segmentation (the `automatic` and `joint` training modes), the segmentations must be dense (i.e. all objects per image are annotated). For training interactive segmentation only (the `interactive` mode), sparse segmentations (i.e. few objects per image are annotated) are fine. For semantic segmentation (the `semantic` mode), you need class labels.


### 4. I have finetuned Segment Anything on my microscopy data. How can I use it for annotating new images?
You can load your finetuned model by entering the path to its checkpoint in the `custom weights path` field in the `Embedding Settings` drop-down menu.
If you are using the python library you can specify this path with the `checkpoint_path` parameter, and in the command line with `-c` / `--checkpoint`.
A model that you trained in the `interactive` mode has no decoder, so you can use it for interactive segmentation, but not for automatic segmentation.


### 5. What is the background of the AIS (Automatic Instance Segmentation) feature in `micro_sam`?
`micro_sam` adds a segmentation decoder to the image encoder, for enabling faster and accurate automatic instance segmentation. The decoder predicts the foreground, the directed distances to the object center and boundary, and the object boundaries. The post-processing derives the instances from these predictions, see [APG](#apg).


### 6. What is the background of the new APG (Automatic Prompt Generation) feature in `micro_sam`?
APG (automatic prompt generation) is an automatic instance segmentation method. It derives point prompts from the predictions of the segmentation decoder, and segments each object with these prompts with the interactive part of the model. You can use it with `micro_sam inference segmentation --engine apg`, with `segmentation_mode="apg"` in python, and in the annotation tools. See [APG](#apg) for details.


### 7. I want to finetune only the Segment Anything model without the additional instance decoder.
The decoder for automatic segmentation is optional. Train with the `interactive` mode to finetune only for interactive segmentation, with the `automatic` mode to finetune only the decoder and the image encoder, or with the `joint` mode for both. Training with the decoder will increase training times, but will enable you to use AIS and APG. See [training your own model](#training-your-own-model) for details.


### 8. I have a NVIDIA RTX 4090Ti GPU with 24GB VRAM. Can I finetune Segment Anything?
Yes, finetuning is possible on most consumer-grade GPUs. The number of objects per image (`max_num_objects`, `--max_num_objects` in the command line) has the biggest impact on the GPU memory consumption and the quality of the finetuned model. The batch size, the patch shape and the model size also matter. If the training runs out of memory, reduce them, for example the number of objects per image.


### 9. I want to create a dataloader for my data, to finetune Segment Anything.
`micro_sam.v2.training.default_sam2_loader` creates the data loaders for image files (e.g. tif or png) and for container files (e.g. hdf5, zarr, n5, OME-Zarr), for all training modes, see [training your own model](#training-your-own-model).
It is based on `torch-em`. See the [tutorial notebook](https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb) on how to create dataloaders using `torch-em` and the [documentation](https://github.com/constantinpape/torch-em/blob/main/doc/datasets_and_dataloaders.md) for details on creating your own datasets and dataloaders.
> NOTE: If you have images with large input shapes with a sparse density of instance segmentations, we recommend using a [`sampler`](https://github.com/constantinpape/torch-em/blob/main/torch_em/data/sampler.py) for choosing the patches with valid segmentation. You can pass it to `default_sam2_loader` with the `sampler` argument.


### 10. How can I evaluate a model I have finetuned?
To validate a model for your data, you have different options, depending on the task you want to solve and whether you have segmentation annotations for your data.

- If you don't have any annotations you will have to validate the model visually. We suggest doing this with the `micro_sam` GUI tools. You can learn how to use them in the `micro_sam` documentation.
- If you have segmentation annotations you can use the `micro_sam` python library to evaluate the segmentation quality of different models. We provide functionality to evaluate the models for interactive and for automatic segmentation:
    - You can use `micro_sam.v2.evaluation.inference.run_interactive_segmentation_2d` and `micro_sam.v2.evaluation.evaluation.run_evaluation_for_iterative_prompting` to evaluate models for interactive segmentation.
    - You can segment your images with `micro_sam inference segmentation` and evaluate the results with `micro_sam v1 evaluate` to evaluate models for automatic segmentation.

We provide an [example notebook](https://github.com/computational-cell-analytics/micro-sam/blob/main/notebooks/inference_and_evaluation.ipynb) that shows how to use this evaluation functionality.


