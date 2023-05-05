# SegmentAnything for Microscopy

Tools for segmentation and tracking in microscopy build on top of [SegmentAnything](https://segment-anything.com/).
We implement napari applications for:
- interactive 2d segmentation
- interactive 3d segmentation
- interactive tracking of 2d image data

**Early beta version**

This is an early beta version. Any feedback is welcome, but please be aware that the functionality is evolving fast and not fully tested.

## Functionality overview

TODO

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

TODO

### Tips & Tricks

TODO

## Using the micro_sam library

TODO

## Contributing

```
micro_sam <- library with utility functionality for using SAM for microscopy data
    /sam_annotator <- the napari plugins for annotation
```

## Citation

If you are using this repository in your research please cite
- [SegmentAnything](https://arxiv.org/abs/2304.02643)
- and our repository on [zenodo](TODO) (we are working on a full publication)
