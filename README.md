# SegmentAnything for Microscopy

## Installation and Requirements

Install [PyTorch](https://pytorch.org/get-started/locally/), [SegmentAnything](https://github.com/facebookresearch/segment-anything#installation) and elf in your conda env.
Then install the `micro_sam` library via
```
pip install -e .
```

## Structure

```
micro_sam <- library with utility functionality for using SAM for microscopy data
    /sam_annotator <- the napari plugins for annotation
```
