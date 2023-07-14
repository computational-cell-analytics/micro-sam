[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7919746.svg)](https://doi.org/10.5281/zenodo.7919746)

# SegmentAnything for Microscopy

Tools for segmentation and tracking in microscopy build on top of [SegmentAnything](https://segment-anything.com/).
Segment and track objects in microscopy images interactively with a few clicks!

We implement napari applications for:
- interactive 2d segmentation
- interactive 3d segmentation
- interactive tracking of 2d image data

**Early beta version**

This is an early beta version. Any feedback is welcome, but please be aware that the functionality is under active development and that several features are not finalized or thoroughly tested yet.
Once the functionality has matured we plan to release the interactive annotation applications as [napari plugins](https://napari.org/stable/plugins/index.html).

If you run into any problems or have questions please open an issue or reach out via [image.sc](https://forum.image.sc/) using the tag `micro-sam` and tagging @constantinpape.

**New in version 0.02**

- We have added support for bounding box prompts (see the gif below), which provide better segmentation results than points in many cases.
- Interactive tracking now uses a better heuristic to propagate masks across time, leading to better automatic tracking results.
- And have fixed several small bugs.

![box-prompts](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d04cb158-9f5b-4460-98cd-023c4f19cccd)


## Functionality overview

We implement applications for fast interactive 2d and 3d segmentation as well as tracking.
- Left: interactive 2d segmentation
- Middle: interactive 3d segmentation
- Right: interactive tracking

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d5ee2080-ab08-4716-b4c4-c169b4ed29f5" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfca3d9b-dba5-440b-b0f9-72a0683ac410" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/aefbf99f-e73a-4125-bb49-2e6592367a64" width="256">

## Installation

We require these dependencies:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [SegmentAnything](https://github.com/facebookresearch/segment-anything#installation)
- [napari](https://napari.org/stable/)
- [elf](https://github.com/constantinpape/elf)

We recommend to use conda and provide two environment files with all necessary requirements:
- `environment_gpu.yaml`: sets up an environment with GPU support.
- `environment_cpu.yaml`: sets up an environment with CPU support.

To install via conda first clone this repository:
```
git clone https://github.com/computational-cell-analytics/micro-sam
```
and
```
cd micro_sam
```

Then create either the GPU or CPU environment via

```
conda env create -f <ENV_FILE>.yaml
```
Then activate the environment via
```
conda activate sam
```
And install our napari applications and the `micro_sam` library via
```
pip install -e .
```

**Troubleshooting:**

- On some systems `conda` is extremely slow and cannot resolve the environment in the step `conda env create ...`. You can use `mamba` instead, which is a faster re-implementation of `conda`. It can resolve the environment in less than a minute on any system we tried. Check out [this link](https://mamba.readthedocs.io/en/latest/installation.html) for how to install `mamba`. Once you have installed it, run `mamba env create -f <ENV_FILE>.yaml` to create the env.
- Installation on MAC with a M1 or M2 processor:
    - The pytorch installation from `environment_cpu.yaml` does not work with a MAC that has an M1 or M2 processor. Instead you need to:
        - Create a new environment: `mamba create -c conda-forge python pip -n sam`
        - Activate it va `mamba activate sam`
        - Follow the instructions for how to install pytorch for MAC via conda from [pytorch.org](https://pytorch.org/).
        - Install additional dependencies: `mamba install -c conda-forge napari python-elf tqdm`
        - Install SegmentAnything: `pip install git+https://github.com/facebookresearch/segment-anything.git`
        - Install `micro_sam` by running `pip install -e .` in this folder.
    - **Note:** we have seen many issues with the pytorch installation on MAC. If a wrong pytorch version is installed for you (which will cause pytorch errors once you run the application) please try again with a clean `mambaforge` installation. Please install the `OS X, arm64` version from [here](https://github.com/conda-forge/miniforge#mambaforge).

## Usage

After installing the `micro_sam` python application the three interactive annotation tools can be started from the command line or from a python script (see details below).
They are built with napari to implement the viewer and user interaction. If you are not familiar with napari yet, [start here](https://napari.org/stable/tutorials/fundamentals/quick_start.html).
To use the apps the functionality of [napari point layers](https://napari.org/stable/howtos/layers/points.html), [napari shape layers](https://napari.org/stable/howtos/layers/shapes.html) and [napari labels layers](https://napari.org/stable/howtos/layers/labels.html) is of particular importance.

**Note:** the screenshots and tutorials do not show how to use bounding boxes for prompts yet. You can use the `box_prompts` layer for them in all three tools, and they can be used as a replacement or in combination with the point prompts.

### 2D Segmentation

The application for 2d segmentation can be started in two ways:
- Via the command line with the command `micro_sam.annotator_2d`. Run `micro_sam.annotator_2d -h` for details.
- From a python script with the function `micro_sam.sam_annotator.annotator_2d`. Check out [examples/sam_annotator_2d](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/sam_annotator_2d.py) for details. 

Below you can see the interface of the application for a cell segmentation example:

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/041585a6-0b72-4e4b-8df3-42135f4334c5" width="768">

The most important parts of the user interface are:
1. The napari layers that contain the image, segmentations and prompts:
    - `prompts`: point layer that is used to provide prompts to SegmentAnything. Positive prompts (green points) for marking the object you want to segment, negative prompts (red points) for marking the outside of the object.
    - `current_object`: label layer that contains the object you're currently segmenting.
    - `committed_objects`: label layer with the objects that have already been segmented.
    - `auto_segmentation`: label layer results from using SegmentAnything for automatic instance segmentation.
    - `raw`: image layer that shows the image data.
2. The prompt menu for changing the currently selected point from positive to negative and vice versa. This can also be done by pressing `t`.
3. The menu for automatic segmentation. Pressing `Segment All Objects` will run automatic segmentation (this can take few minutes if you are using a CPU). The results will be displayed in the `auto_segmentation` layer. 
4. The menu for interactive segmentation. Pressing `Segment Object` (or `s`) will run segmentation for the current prompts. The result is displayed in `current_object`
5. The menu for commiting the segmentation. When pressing `Commit` (or `c`) the result from the selected layer (either `current_object` or `auto_segmentation`) will be transferred from the respective layer to `committed_objects`.

Check out [this video](https://youtu.be/DfWE_XRcqN8) for an overview of the interactive 2d segmentation functionality.

### 3D Segmentation

The application for 3d segmentation can be started as follows:
- Via the command line with the command `micro_sam.annotator_3d`. Run `micro_sam.annotator_3d -h` for details.
- From a python script with the function `micro_sam.sam_annotator.annotator_3d`. Check out [examples/sam_annotator_3d](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/sam_annotator_3d.py) for details.

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/0a6fb19e-7db5-4188-9371-3c238671f881" width="768">

The most important parts of the user interface are listed below. Most of these elements are the same as for [the 2d segmentation app](https://github.com/computational-cell-analytics/micro-sam#2d-segmentation).
1. The napari layers that contain the image, segmentation and prompts. Same as for [the 2d segmentation app](https://github.com/computational-cell-analytics/micro-sam#2d-segmentation) but without the `auto_segmentation` layer.
2. The prompt menu.
3. The menu for interactive segmentation.
4. The 3d segmentation menu. Pressing `Segment Volume` (or `v`) will extend the segmentation for the current object across the volume.
5. The menu for committing the segmentation.

Check out [this video](https://youtu.be/5Jo_CtIefTM) for an overview of the interactive 3d segmentation functionality.

### Tracking

The application for interactive tracking (of 2d data) can be started as follows:
- Via the command line with the command `micro_sam.annotator_tracking`. Run `micro_sam.annotator_tracking -h` for details.
- From a python script with the function `micro_sam.sam_annotator.annotator_tracking`. Check out [examples/sam_annotator_tracking](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/sam_annotator_tracking.py) for details. 

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfb80f17-a370-4cbc-aaeb-29de93444090" width="768">

The most important parts of the user interface are listed below. Most of these elements are the same as for [the 2d segmentation app](https://github.com/computational-cell-analytics/micro-sam#2d-segmentation).
1. The napari layers thaat contain the image, segmentation and prompts. Same as for [the 2d segmentation app](https://github.com/computational-cell-analytics/micro-sam#2d-segmentation) but without the `auto_segmentation` layer, `current_tracks` and `committed_tracks` are the equivalent of `current_object` and `committed_objects`.
2. The prompt menu.
3. The menu with tracking settings: `track_state` is used to indicate that the object you are tracking is dividing in the current frame. `track_id` is used to select which of the tracks after divsion you are following.
4. The menu for interactive segmentation.
5. The tracking menu. Press `Track Object` (or `v`) to track the current object across time.
6. The menu for committing the current tracking result.

Check out [this video](https://youtu.be/PBPW0rDOn9w) for an overview of the interactive tracking functionality.

### Tips & Tricks

- By default, the applications pre-compute the image embeddings produced by SegmentAnything and store them on disc. If you are using a CPU this step can take a while for 3d data or timeseries (you will see a progress bar with a time estimate). If you have access to a GPU without graphical interface (e.g. via a local computer cluster or a cloud provider), you can also pre-compute the embeddings there and then copy them to your laptop / local machine to speed this up. You can use the command `micro_sam.precompute_embeddings` for this (it is installed with the rest of the applications). You can specify the location of the precomputed embeddings via the `embedding_path` argument.
- Most other processing steps are very fast even on a CPU, so interactive annotation is possible. An exception is the automatic segmentation step (2d segmentation), which takes several minutes without a GPU (depending on the image size). For large volumes and timeseries segmenting an object in 3d / tracking across time can take a couple settings with a CPU (it is very fast with a GPU).
- You can also try using a smaller version of the SegmentAnything model to speed up the computations. For this you can pass the `model_type` argument and either set it to `vit_l` or `vit_b` (default is `vit_h`). However, this may lead to worse results.
- You can save and load the results from the `committed_objects` / `committed_tracks` layer to correct segmentations you obtained from another tool (e.g. CellPose) or to save intermediate annotation results. The results can be saved via `File->Save Selected Layer(s) ...` in the napari menu (see the tutorial videos for details). They can be loaded again by specifying the corresponding location via the `segmentation_result` (2d and 3d segmentation) or `tracking_result` (tracking) argument.

### Known limitations

- SegmentAnything does not work well for very small or fine-graind objects (e.g. filaments).
- For the automatic segmentation functionality we currently rely on the automatic mask generation provided by SegmentAnything. It is slow and often misses objects in microscopy images. For now, we only offer this functionality in the 2d segmentation app; we are working on improving it and extending it to 3d segmentation and tracking.
- Prompt bounding boxes do not provide the full functionality for tracking yet (they cannot be used for divisions or for starting new tracks). See also https://github.com/computational-cell-analytics/micro-sam/issues/23.

### Using the micro_sam library

After installation the `micro_sam` python library is available, which provides several utility functions for using SegmentAnything with napari. Check out [examples/image_series_annotator.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/image_series_annotator_app.py) for an example application for segmenting objects in an image series built with it.

<!---
## Contributing

```
micro_sam <- library with utility functionality for using SAM for microscopy data
    /sam_annotator <- the napari plugins for annotation
```
TODO: related projects
-->


## Citation

If you are using this repository in your research please cite
- [SegmentAnything](https://arxiv.org/abs/2304.02643)
- and our repository on [zenodo](https://doi.org/10.5281/zenodo.7919746) (we are working on a publication)


## Related Projects

There are two other napari plugins build around segment anything:
- https://github.com/MIC-DKFZ/napari-sam (2d and 3d support)
- https://github.com/JoOkuma/napari-segment-anything (only 2d support)

Compared to these we support more applications (2d, 3d and tracking), and aim to further extend and specialize SegmentAnything for microscopy data.
[WebKnossos](https://webknossos.org/) also offers integration of SegmentAnything for interactive segmentation.
