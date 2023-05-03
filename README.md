# SegmentAnything for Microscopy

## Installation and Requirements

Install pytorch, SegmentAnything and elf in your conda env.
Then install the `micro_sam` library via
```
pip instal -e .
```

## Structure

```
micro_sam <- library with utility functionality for using SAM for microscopy data
    /sam_annotator <- the napari plugins for annotation
```
