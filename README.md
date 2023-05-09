# SegmentAnything for Microscopy

Tools for segmentation and tracking in microscopy build on top of [SegmentAnything](https://segment-anything.com/).
We implement napari applications for:
- interactive 2d segmentation
- interactive 3d segmentation
- interactive tracking of 2d image data

**Early beta version**

This is an early beta version. Any feedback is welcome, but please be aware that the functionality is under active development and that several features are not finalized or thoroughly tested yet.
Once the functionality has matured we plan to release the interactive annotation applications as [napari plugins](https://napari.org/stable/plugins/index.html).


## Functionality overview

We implement applications for fast interactive 2d and 3d segmentation as well as tracking.
- Left: interactive 2d segmentation
- Middle: interactive 3d segmentation
- Right: interactive tracking

(TODO add gif for 2d)

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/ca4d6bcc-8674-455b-95c6-0eb516d2bc76" width="384">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/a4a048de-fd3d-4718-b386-2160ac94bbf0" width="384">

## Installation

We require these dependencies:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [SegmentAnything](https://github.com/facebookresearch/segment-anything#installation)
- [napari](https://napari.org/stable/)
- [elf](https://github.com/constantinpape/elf)

We recommend to use conda and provide two conda environment files with all necessary requirements:
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

## Usage

After the installation the three applications for interactive annotations can be started from the command line or within a python script:
- **2d segmentation**: via the command `micro_sam.annotator_2d` or with the function `micro_sam.sam_annotator.annotator_2d` from python. Run `micro_sam.annotator_2d -h` or check out [examples/sam_annotator_2d](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/sam_annotator_2d.py) for details. 
- **3d segmentation**: via the command `micro_sam.annotator_3d` or with the function `micro_sam.sam_annotator.annotator_3d` from python. Run `micro_sam.annotator_3d -h` or check out [examples/sam_annotator_3d](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/sam_annotator_3d.py) for details. 
- **tracking**: via the command `micro_sam.annotator_tracking` or with the function `micro_sam.sam_annotator.annotator_tracking` from python. Run `micro_sam.annotator_tracking -h` or check out [examples/sam_annotator_tracking](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/sam_annotator_tracking.py) for details. 

All three applications are built with napari. If you are not familiar with napari yet, start [here](https://napari.org/stable/tutorials/fundamentals/quick_start.html).

### 2D Segmentation

TODO annotated screenshot + link to tutorial video.

### 3D Segmentation

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/3c35ba63-1b67-48df-9b11-94919bdc7c79" width="1024">

GUI-Elements 3d annotation:
1. TODO
2. TODO
3. TODO
4. TODO
5. TODO

Check out [this video](https://youtu.be/5Jo_CtIefTM) for an overview of the interactive 3d segmentation functionality.

### Tracking

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/1fdffe3c-ff10-4d06-a1ba-9974a673b846" width="1024">

GUI-Elements tracking:
1. TODO
2. TODO
3. TODO
4. TODO
5. TODO
6. TODO

TODO link to video tutorial

### Tips & Tricks

TODO
- speeding things up: precomputing the embeddings with a gpu, making input images smaller
- correcting existing segmentaitons via `segmentation_results`
- saving and loading intermediate results via segmentation results

### Limitations

TODO
- automatic instance segmentation limitations


## Using the micro_sam library

TODO
- link to the example image series application


## Contributing

```
micro_sam <- library with utility functionality for using SAM for microscopy data
    /sam_annotator <- the napari plugins for annotation
```


## Citation

If you are using this repository in your research please cite
- [SegmentAnything](https://arxiv.org/abs/2304.02643)
- and our repository on [zenodo](TODO) (we are working on a publication)
