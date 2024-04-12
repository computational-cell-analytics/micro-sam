# FAQ

We have assembled the frequently asked questions and encountered issues in one place. We welcome you to have a look and see if one of the mentions answers your question(s). Feel free to open an issue in `micro-sam` if your problem needs attention.

## Installation

### 1. How to install 'micro-sam'?
Ans. The [installation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation) for `micro-sam` is supported in three ways: [from mamba](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-mamba) (recommended), [from source](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-source) and [from installers](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-installer). Check out our [tutorial video]() to get started with `micro-sam`, briefly walking you through the installation and the tool.

### 2. What is the minimum system requirement for 'micro-sam'?
Ans. From our experience, `micro-sam` works seamlessly on most CPU resources over 8GB RAM for the annotator tool. You might encounter some `napari`-related slowness for $\leq$ 8GB RAM.

### 3. What is the recommended PyTorch version to install?
Ans. `micro-sam` has been experimented on CUDA 12.1. However, the framework is not constrained to any CUDA versions.

### 4. 
Ans.

## Usage

### 1. I have high-resolution large tomograms, 'micro-sam' does not seem to work.
Ans. TODO: Tiling hint

### 2. I have micropscopy images. Can I use the annotator tool for segmentation?
Ans. TODO (smh wrap the question to hint for 2d, 3d, tracking and image series all at once)

### 3. I would like to fine-tune Segment Anything on Google Colab / Kaggle, is it possible?
Ans. Yes, you can fine-tune Segment Anything on your custom datasets on Google Colab, Kaggle and [BAND](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-micro_sam-on-band). Check out our [tutorial video]() and the [tutorial notebook]() for this.

### 4. I am using 'micro-sam' for segmenting objects. I would like to report the steps for reproducability. How can this be done?
Ans. TODO

### 5. I have complex objects to segment. Both, the default and generalist models do not work for me. What should I do?
Ans. TODO: Hint towards finet-uning a specialist.

## Fine-tuning

### 1. I have a microscopy dataset I would like to fine-tune Segment Anything for. Is it possible using 'micro-sam'?
Ans. Yes, you can fine-tune Segment Anything on your own dataset. Here's how you can do it:
- Check out the [tutorial notebook]() on how to fine-tune in a few lines of code.
- Check out the [example](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/finetuning) script for fine-tune using the python library.
- Check out the [video]() on how to fine-tune using the napari tool.


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
