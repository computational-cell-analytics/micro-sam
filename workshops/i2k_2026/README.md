# I2K Workshop: Segment Anything for Microscopy

This document walks you through the preparation for the I2K 2026 workshop on "Segment Anything for Microscopy".
In this workshop, we use the new `micro_sam` models that are based on Segment Anything 2 (SAM2).

## Workshop Overview

The workshop is divided into three parts:
1. Short introduction to `micro_sam` and the new SAM2-based models.
2. Using the `micro_sam` napari plugin for interactive and automatic segmentation in 2D and 3D.
3. Using the plugin for object and pixel classification in 2D and 3D, on your own data or on the example data.

We will walk through the `micro_sam` plugin in part 2. In part 3, you can then apply it to your own data, or to the example data that is most similar to your application.

**Please read the [Workshop Preparation](#workshop-preparation) section carefully and follow the relevant steps before the workshop, so that we can get started right away.**

## Workshop Preparation

To prepare for the workshop, please do the following:
- Install the latest version of `micro_sam`, see [Installation](#installation) for details.
- Download the models and the example data, and precompute the image embeddings, see [here](#download-models-and-data).
- Decide what you want to do in the 3rd part of the workshop. You have the following options:
    - Interactive and automatic segmentation in 2D, see [2D segmentation](#2d-segmentation).
    - Interactive and automatic segmentation in 3D, see [3D LM segmentation](#3d-lm-segmentation).
    - Interactive and automatic segmentation in 3D EM, see [3D EM segmentation](#3d-em-segmentation).
    - Object classification in 2D and 3D, see [object classification](#object-classification).
    - Pixel classification in 2D and 3D, see [pixel classification](#pixel-classification).

We use the model `hvit_t_cells` for all applications. It is a SAM2 model that we finetuned for microscopy.
The scripts run best on a computer with a GPU. You can also run them on a laptop with a CPU, but the image embeddings then take longer to compute, especially for the 3D data.

If you want to learn more about the `micro_sam` napari plugin or python library you can check out the [documentation](https://computational-cell-analytics.github.io/micro-sam/) and our [tutorial videos](https://youtube.com/playlist?list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&si=3q-cIRD6KuoZFmAM).

### Installation

Please make sure to install the latest version of `micro_sam` before the workshop. You can find the instructions for the installation [here](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation).

### Download Models and Data

We provide a script to download the model used in the workshop. To run the script you first need to use `git` to download this repository:
```bash
git clone https://github.com/computational-cell-analytics/micro-sam
```
then go to this directory:
```bash
cd micro-sam/workshops/i2k_2026
```
and run the script:
```bash
python download_models.py
```

We also provide a script to download the example data. You can download all datasets by running:
```bash
python download_datasets.py -i data
```

You can also download a single dataset with the argument `-d`, for example `python download_datasets.py -i data -d nuclei_2d`. The available datasets are `cells_2d`, `nuclei_2d`, `histopatho`, `nuclei_3d`, `neurons_em` and `cells_em`.
Add the argument `-v` to view the data in napari after the download.

Then precompute the image embeddings and the automatic segmentation state for the example data, so that the napari tools start fast:
```bash
python precompute.py
```

This takes about a minute on a GPU. It covers the first ten images of each 2D dataset and all 3D volumes.

All scripts below read the data from the folder `data`. They cache the image embeddings in the folder `embeddings`, so that the second start of a tool for the same data is fast.
Use the argument `-i` to read the data from another folder and the argument `-e` to cache the embeddings in another folder.

### 2D Segmentation

You can use the [annotation tool](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#annotator-2d) to run interactive and automatic segmentation for cells or nuclei in 2D images. We have prepared three example datasets for the workshop:
- `cells_2d`: Cells imaged in phase-contrast microscopy, from the [LIVECell dataset](https://doi.org/10.1038/s41592-021-01249-6).
- `nuclei_2d`: Nuclei imaged in fluorescence microscopy, from the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018).
- `histopatho`: Nuclei in H&E stained tissue, from the LyNSeC dataset.

You can run automatic segmentation for an image of one of these datasets with a script, which then shows the result in napari:
```bash
python automatic_segmentation_2d.py -d cells_2d
```

The script stores the segmentation in the folder `segmentations`.
You can choose the automatic segmentation mode with the argument `--mode`:
- `apg`: Automatic prompt generation, which uses the decoder predictions as prompts for SAM.
- `ais`: Automatic instance segmentation with the decoder.
- `amg`: Automatic mask generation, which uses a grid of point prompts.

By default, the script uses `ais` for `cells_2d` and `apg` for all other datasets, because these modes work best for the data.

After this you can start the annotation tool for the same image:
```bash
python annotator_2d.py -d cells_2d
```

The annotation tool loads the automatic segmentation of the default mode from the folder `segmentations`, so that you can correct it with interactive segmentation. If the segmentation does not exist, the tool starts without it.

**If you want to bring your own data for annotation, please store it as tif images. You DO NOT have to provide segmentation masks; we include them here only for reference and they are not needed for annotation with `micro_sam`.**

### 3D LM Segmentation

You can use the [3D annotation tool](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#annotator-3d) to run interactive and automatic segmentation for cells or nuclei in volume light microscopy. We have prepared an example dataset for the workshop:
- `nuclei_3d`: A crop of a volume with nuclei in a plant tissue, from the GoNuclear dataset.

You can run automatic segmentation for this volume with a script, which then shows the result in napari:
```bash
python automatic_segmentation_3d.py -d nuclei_3d
```

By default, the script uses automatic instance segmentation (`ais`), which takes about a minute on a GPU.
Automatic prompt generation (`apg`) is more accurate, but it takes about two to three minutes on a GPU. It propagates the prompt for every object through the volume. On a small GPU it can take ten minutes or longer, so please run it before the workshop:
```bash
python automatic_segmentation_3d.py -d nuclei_3d --mode apg
```

After this you can start the 3D annotation tool for the same volume:
```bash
python annotator_3d.py -d nuclei_3d
```

The annotation tool loads the APG segmentation from the folder `segmentations`, so that you can correct it with interactive segmentation. If the segmentation does not exist, the tool starts without it.

### 3D EM Segmentation

You can use the [3D annotation tool](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#annotator-3d) to run interactive and automatic segmentation for cells or organelles in volume electron microscopy. We have prepared two example datasets for the workshop:
- `neurons_em`: A small crop of a volume with neurons in the fly brain, from sample C of the [CREMI challenge](https://cremi.org/).
- `cells_em`: A small crop with cells from an EM volume of **Platynereis dumerilii**, from [Hernandez et al.](https://www.cell.com/cell/fulltext/S0092-8674(21)00876-X). You can also segment cellular ultrastructures such as nuclei.

You can run automatic segmentation for one of these volumes with a script, which then shows the result in napari:
```bash
python automatic_segmentation_3d.py -d neurons_em
python automatic_segmentation_3d.py -d cells_em
```

For these volumes, APG works much better than AIS, so the script uses `apg` by default. APG takes about two minutes for each of these crops on a GPU.
The script post-processes the AIS predictions with multicut, because the objects are densely packed.

After this you can start the 3D annotation tool for the same volume, which loads the APG segmentation so that you can correct it:
```bash
python annotator_3d.py -d neurons_em
```

Note: The automatic segmentation of `hvit_t_cells` is less accurate for the neurons in `neurons_em` than for the other data. You can [finetune a model](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#training-your-own-model) on your own annotations to improve it.

### Object Classification

You can use the [object classifier](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#object-classification) to classify segmented objects, for example to sort nuclei into different cell types.
You label a few objects for each class in napari. Then the tool trains a random forest on the SAM features of the objects and predicts the class of all other objects.

You can start the object classifier for the first ten images of the `histopatho` dataset:
```bash
python object_classifier.py -d histopatho
```

The script first segments the nuclei with the default mode of the dataset, or it loads the segmentations from the folder `segmentations`. Then it opens the object classifier, which lets you go through the images one after another.
You can also use the datasets `cells_2d` and `nuclei_2d`, and you can select the number of images with the argument `-n`.
The script stores the new segmentations in the folder `segmentations` and the classification results in the folder `classification`.

You can also classify the nuclei in the `nuclei_3d` volume:
```bash
python object_classifier.py -d nuclei_3d
```

The script loads the APG segmentation from the folder `segmentations`, or it computes the segmentation if it does not exist.
The object classifier computes the features of all nuclei before the first training, which takes about a minute.

### Pixel Classification

You can use the pixel classifier for semantic segmentation, for example to separate the foreground from the background.
You paint a few strokes for each class in napari. Then the tool trains a random forest on the SAM features of the pixels and predicts the class of all other pixels.

You can start the pixel classifier for the first ten images of the `cells_2d` dataset:
```bash
python pixel_classifier.py -d cells_2d
```

You can also use the datasets `nuclei_2d` and `histopatho`, and you can select the number of images with the argument `-n`.
The script stores the classification results in the folder `classification`.

You can also use the pixel classifier for the `nuclei_3d` volume:
```bash
python pixel_classifier.py -d nuclei_3d
```

### Advanced applications: scripting with `micro_sam`

If you want to develop applications based on `micro_sam` you can use
the [micro_sam python library](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-the-python-library) to implement your own functionality.
The scripts in this folder are a good starting point. They use the python library to load the data, to run automatic segmentation and to start the napari tools.
