# Segment Anything for Microscopy

Segment Anything for Microscopy implements automatic and interactive annotation for microscopy data. It is built on top of [Segment Anything](https://segment-anything.com/) by Meta AI and specializes it for microscopy and other bio-imaging data.
Its core components are:
- The `micro_sam` tool: implements interactive data annotation using [napari](https://napari.org/stable/).
- The `micro_sam` python library: implements functionality for applying Segment Anything to multi-dimensional data, and to fine-tune it on custom datasets.
- The `micro_sam` models: new Segment Anything models that were fine-tuned on publicly available microscopy data.
