# I2K Workshop: Segment Anything for Microscopy 

This document walks you through the preparation for the upcoming I2K workshops on "Segment Anything for Microscopy". We will give two workshops:
1. In person at I2K 2024 (Milan, Italy)
2. Online at Virtual I2K 2024

## Workshop Overview

The workshop will be one hour and will be divided into three parts:
1. Short introduction ([slides](https://docs.google.com/presentation/d/1Bw0gQ9Xio0HozKVaJl9-mxJBmCsQPh-1/edit?usp=sharing&ouid=113044948772353505255&rtpof=true&sd=true), 5-10 minutes)
2. Using the `micro_sam` napari plugin for interactive 2D and 3D segmentation (10-15 minutes)
3. Using the plugin on your own or example data, finetuning a custom model or an advanced application (35-40 minutes)

We will walk through how to use the `micro_sam` plugin for interactive segmentation in part 2, so that you can then try it out on your own data (or the example data that is most similar to your targeted application) in part 3.
Alternatively you can also work on model finetuning or an advanced application, such as using our python library to build your own annotation scripts, in part 3.

**Please read the [Workshop Preparation](#workshop-preparation) section carefully and follow the relevant steps before the workshop, so that we can get started during the workshop right away.**

## Workshop Preparation

To prepare for the workshop please do the following:
- Install the latest version of `micro_sam`, see [Installation](#installation) for details.
- Download the pre-computed embeddings for the first 3D segmentation data, see [here](#download-embeddings-for-3d-segmentation).
- Decide what you want to do in the 3rd part of the workshop and follow the respective preparation steps. You have the following options:
    - High-throughput annotation of cells (or other structures) in 2D images, see [high-throughput annotation](#high-throughput-image-annotation).
    - 3D segmentation in light or electron mciroscopy, see [3D LM segmentation](#3d-lm-segmentation) and [3D EM segmentation](#3d-em-segmentation).
    - Finetuning a SAM model, see [model finetuning](#model-finetuning).
    - Writing your own scripts using the `micro_sam` python library, see [scripting](#scripting-with-micro_sam).

If you want to learn more about the `micro_sam` napari plugin or python library you can check out the [documentation](https://computational-cell-analytics.github.io/micro-sam/) or our [tutorial videos](https://youtube.com/playlist?list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&si=3q-cIRD6KuoZFmAM).

### Installation

Please make sure to install the latest version of `micro_sam` (version 1.1) before the workshop using `conda` (or `mamba`).
You can create a new environment and install `micro_sam` like this:
```bash
$ conda create -c conda-forge -n micro_sam python=3.11
$ conda activate micro_sam
$ conda install -c pytorch -c conda-forge "micro_sam>=1.1" "torch_em>=0.7.4"
```
If you already have an installation of `micro_sam` please update it by running the last command in your respective environment. You can find more information on the installation [here](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation).

### Download Embeddings for 3D EM Segmentation

We provide a script to download the image embeddings for the 3D segmentation problem in part 2.
The image embeddings are necessary to run interactive segmentation. Computing them on the CPU can take some time for volumetric data, but we support precomputing them and have done this for this dataset so that we can start with the interactive segmentation during the workshop right away.

To run the script you first need to use `git` to download this repository:
```bash
$ git clone https://github.com/computational-cell-analytics/micro-sam
```
then go to this directory:
```bash
$ cd micro_sam/workshops/i2k_2024
```
and download the embeddings:
```bash
$ python download_embeddings.py -d lucchi -e embeddings
```

### High-throughput Image Annotation

TODO

### 3D LM Segmentation

TODO

### 3D EM Segmentation

TODO

### Model Finetuning

TODO

### Scripting with micro_sam

TODO

### Precompute Embeddings

TODO


<!---

### Step 1: Download the Datasets

- We provide the script `download_datasets.py` for automatic download of datasets to be used for interactive annotation using `micro-sam`.
- You can run the script as follows:
```bash
$ python download_datasets.py -i <DATA_DIRECTORY> -d <DATASET_NAME>
```
where, `DATA_DIRECTORY` is the filepath to the directory where the datasets will be downloaded, and `DATASET_NAME` is the name of the dataset (run `python download_datasets.py -h` in the terminal for more details).

> NOTE: We have chosen a) subset of the CellPose `cyto` dataset, b) one volume from the EmbedSeg `Mouse-Skull-Nuclei-CBG` dataset from the train split (namely, `X1.tif`), c) one volume from the Platynereis `Membrane` dataset from the train split (namely, `train_data_membrane_02.n5`) and d) the entire `HPA` dataset for the following tasks in `micro-sam`.

### Step 2: Download the Precomputed Embeddings

- We provide the script `download_embeddings.py` for automatic download of precompute image embeddings for volumetric data to be used for interactive annotation using `micro-sam`.
- You can run the script as follows:

```bash
$ python download_embeddings -e <EMBEDDING_DIRECTORY> -d <DATASET_NAME>
```
where, `EMBEDDING_DIRECTORY` is the filepath to the directory where the precomputed image embeddings will be downloaded, and `DATASET_NAME` is the name of the dataset (run `python download_embeddings.py -h` in the terminal for more details).

### Additional Section: Precompute the Embeddings Yourself!

Here is an example guide to precompute the image embeddings (eg. for volumetric data).

#### EmbedSeg

```bash
$ micro_sam.precompute_embeddings -i data/embedseg/Mouse-Skull-Nuclei-CBG/train/images/X1.tif  # Filepath where inputs are stored.
                                  -m vit_b  # You can provide name for any model of your choice (supported by 'micro-sam') (eg. 'vit_b_lm').
                                  -e embeddings/embedseg/vit_b/embedseg_Mouse-Skull-Nuclei-CBG_train_X1  # Filepath where computed embeddings will be cached.
```

#### Platynereis

```bash
$ micro_sam.precompute_embeddings -i data/platynereis/membrane/train_data_membrane_02.n5  # Filepath where inputs are stored.
                                  -k volumes/raw/s1  # Key to access the data group in container-style data structures.
                                  -m vit_b  # You can provide name for any model of your choice (supported by 'micro-sam') (eg. 'vit_b_em_organelles').
                                  -e embeddings/platynereis/vit_b/platynereis_train_data_membrane_02  # Filepath where computed embeddings will be cached.
```

### Step 3: Run the `micro-sam` Annotators (WIP)

Run the `micro-sam` annotators with the following scripts:

We recommend using the napari GUI for the interactive annotation. You can use the widget to specify all the essential parameters (eg. the choice of model, the filepath to the precomputed embeddings, etc).

TODO: add more details here.

There is another option to use `micro-sam`'s CLI to start our annotator tools.

#### 2D Annotator (Cell Segmentation in Light Microscopy):

```bash
$ micro_sam.annotator_2d -i data/cellpose/cyto/test/...  # Filepath where the 2d image is stored.
                         -m vit_b  # You can provide name for any model of your choice (supported by 'micro-sam') (eg. 'vit_b_lm')
                         [OPTIONAL] -e embeddings/cellpose/vit_b/...  # Filepath where the computed embeddings will be cached (you can choose to not pass it to compute the embeddings on-the-fly).
```

#### 3D Annotator (EmbedSeg - Nuclei Segmentation in Light Microscopy):

```bash
$ micro_sam.annotator_3d -i data/embedseg/Mouse-Skull-Nuclei-CBG/train/images/X1.tif  # Filepath where the 3d volume is stored.
                         -m vit_b  # You can provide name for any model of your choice (supported by 'micro-sam') (eg. 'vit_b_lm')
                         -e embeddings/embedseg/vit_b/embedseg_Mouse-Skull-Nuclei-CBG_train_X1.zarr  # Filepath where the computed embeddings will be cached (we RECOMMEND to provide paths to the downloaded embeddings OR you can choose to not pass it to compute the embeddings on-the-fly).
```

#### 3D Annotator (Platynereis - Membrane Segmentation in Electron Microscopy):

```bash
$ micro_sam.annotator_3d -i data/platynereis/membrane/train_data_membrane_02.n5  # Filepath where the 2d image is stored.
                         -k volumes/raw/s1  # Key to access the data group in container-style data structures.
                         -m vit_b  # You can provide name for any model of your choice (supported by 'micro-sam') (eg. 'vit_b_em_organelles')
                         -e embeddings/platynereis/vit_b/...  # Filepath where the computed embeddings will be cached (we RECOMMEND to provide paths to the downloaded embeddings OR you can choose to not pass it to compute the embeddings on-the-fly).
```

#### Image Series Annotator (Multiple Light Microscopy 2D Images for Cell Segmentation):

```bash
$ micro_sam.image_series_annotator -i ...
                                   -m vit_b  # You can provide name for any model of your choice (supported by 'micro-sam') (eg. 'vit_b_lm')
```

### Step 4: Finetune Segment Anything on Microscopy Images (WIP)

- We provide a notebook `finetune_sam.ipynb` / `finetune_sam.py` for finetuning Segment Anything Model for cell segmentation in confocal microscopy images.

--->
