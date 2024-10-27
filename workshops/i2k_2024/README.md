# I2K Workshop: Segment Anything for Microscopy 

This document walks you through the preparation for the upcoming I2K workshops on "Segment Anything for Microscopy". We will give two workshops:
1. In person at I2K 2024 (Milan, Italy)
2. Online at Virtual I2K 2024

## Workshop Overview

The workshop will be one hour and will be divided into three parts:
1. Short introduction (5-10 minutes, you can find the slides [here](https://docs.google.com/presentation/d/1Bw0gQ9Xio0HozKVaJl9-mxJBmCsQPh-1/edit?usp=sharing&ouid=113044948772353505255&rtpof=true&sd=true))
2. Using the `micro_sam` napari plugin for interactive 2D and 3D segmentation (10-15 minutes)
3. Using the plugin on your own or on example data, finetuning a custom model or an advanced application (35-40 minutes)

We will walk through how to use the `micro_sam` plugin for interactive segmentation in part 2, so that you can then apply it to your own data (or the example data that is most similar to your targeted application) in part 3.
Alternatively you can also work on model finetuning or an advanced application, such as using the `micro_sam` python library for scripting, in part 3.

**Please read the [Workshop Preparation](#workshop-preparation) section carefully and follow the relevant steps before the workshop, so that we can get started right away.**

## Workshop Preparation

To prepare for the workshop, please do the following:
- Install the latest version of `micro_sam`, see [Installation](#installation) for details.
- Download the models and pre-computed embeddings for the common 3D segmentation, see [here](#download-embeddings-for-3d-segmentation).
- Decide what you want to do in the 3rd part of the workshop and follow the respective preparation steps. You have the following options:
    - High-throughput annotation of cells (or other structures) in 2D images, see [high-throughput annotation](#high-throughput-image-annotation).
    - 3D segmentation in light or electron microscopy, see [3D LM segmentation](#3d-lm-segmentation) and [3D EM segmentation](#3d-em-segmentation).
    - Finetuning SAM on custom data, see [model finetuning](#model-finetuning).
    - Writing your own scripts using the `micro_sam` python library, see [scripting](#scripting-with-micro_sam).

You can do all of this on your laptop with a CPU, except for model finetuning, which requires a GPU.
We have prepared a notebook that runs on cloud resources with a GPU for this.

If you want to learn more about the `micro_sam` napari plugin or python library you can check out the [documentation](https://computational-cell-analytics.github.io/micro-sam/) and our [tutorial videos](https://youtube.com/playlist?list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&si=3q-cIRD6KuoZFmAM).

### Installation

Please make sure to install the latest version of `micro_sam` before the workshop using `conda` (or `mamba`).
You can create a new environment and install it like this:
```bash
conda create -c conda-forge -n micro_sam python=3.11 natsort
conda activate micro_sam
conda install -c pytorch -c conda-forge "micro_sam>=1.1" "pytorch>=2.4" "protobuf<5" cpuonly
```
If you already have an installation of `micro_sam` please update it by running the last command in your respective environment. You can find more information about the installation [here](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation).


### Download Embeddings for 3D EM Segmentation

We provide a script to download the models used in the workshop. To run the script you first need to use `git` to download this repository:
```bash
git clone https://github.com/computational-cell-analytics/micro-sam
```
then go to this directory:
```bash
cd micro-sam/workshops/i2k_2024
```
and run the script:
```bash
python download_models.py
```

We also provide a script to download the image embeddings for the 3D segmentation problem in part 2.
The image embeddings are necessary to run interactive segmentation. Computing them on the CPU can take some time for volumetric data, but we support precomputing them and have done this for this data already.
You can download them by running the script:
```bash
python download_embeddings.py -e embeddings -d lucchi
```

### High-throughput Image Annotation

You can use the [Image Series Annotator](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#image-series-annotator) to run interactive segmentation for many images stored in a folder.
This annotation mode is well suited for generating annotations for 2D cell segmentation or similar analysis tasks.

We have prepared an example dataset for the workshop that you can use. It consists of 15 images from the [CellPose](https://www.cellpose.org/) dataset. You can download the data with the script `download_dataset.py`:

```bash
python download_datasets.py -i data -d cells
```

This will download the data to the folder `data/cells` with images stored in the subfolder `images` and segmentation masks in `masks`. After this you can start the image series annotation tool, either via the napari plugin (we will show this in the workshop) or via the command line:

```bash
micro_sam.image_series_annotator -i data/cells/images -o annotations/cells -e embeddings/cells/vit_b_lm -m vit_b_lm
```

Note: You can use `micro_sam` with different models: the original models from Segment Anything and models finetuned for different microscopy segmentation tasks by us.
For cell segmentation you can either use `vit_b` (the original model) or `vit_b_lm` (our model). Our `vit_b_lm` model will be better for most cell segmentation tasks but there may be cases where `vit_b` is better, so it makes sense to test both before annotating your data. Please refer to [our documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models) for details on the models.

**If you want to bring your own data for annotation please store it in a similar format to the example data. Note that we also support tif images and that you DO NOT have to provide segmentation masks; we include them here only for reference and they are not needed for annotation with `micro_sam`.**

### 3D LM Segmentation

You can use the [3D annotation tool](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#annotator-3d) to run interactive segmentation for cells or nuclei in volume light microscopy. We have prepared an example dataset for the workshop that you can use. It consists of a volume with nuclei imaged in light microscopy from the [EmbedSeg publication](https://github.com/juglab/EmbedSeg).

You can download the data with the script `download_dataset.py`:
```bash
python download_datasets.py -i data -d nuclei_3d
```

After this please download the precomputed embeddings:
```bash
python download_embeddings.py -e embeddings -d nuclei_3d
```

You can then start the 3d annotation tool, either via the napari plugin (we will show this in the workshop) or the command line: 
```bash
micro_sam.annotator_3d -i data/nuclei_3d/images/X1.tif -e embeddings/nuclei_3d/vit_b_lm/embedseg_Mouse-Skull-Nuclei-CBG_train_X1.zarr -m vit_b_lm
```

Note: You can use `micro_sam` with different models: the original models from Segment Anything and models finetuned for different microscopy segmentation tasks by us.
For cell or nucleus segmentation you can either use `vit_b` (the original model) or `vit_b_lm` (our model). Our `vit_b_lm` model will be better for most segmentation problems in light microscopy but there may be cases where `vit_b` is better, so it makes sense to test both before annotating your data. Please refer to [our documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models) for details on the models.

**If you want to bring your own data for annotation please store it in a similar format to the example data. You DO NOT have to provide segmentation masks; we include them here only for reference and they are not needed for annotation with micro_sam. Please also precompute the embeddings for your data, see [Precompute Embeddings](#precompute-embeddings) for details.**

### 3D EM Segmentation

You can use the [3D annotation tool](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#annotator-3d) to run interactive segmentation for cells or organelles in volume electron microscopy.
We have prepared an example dataset for the workshop that you can use. It consists of a small crop from an EM volume of **Platynereis dumerilii**, from [Hernandez et al.](https://www.cell.com/cell/fulltext/S0092-8674(21)00876-X). The volume contains several cells, so you can segment the cells or cellular ultrastructures such as nuclei or mitochondria.

You can download the data with the script `download_dataset.py`:

```bash
python download_datasets.py -i data -d volume_em
```

After this please download the precomputed embeddings:

```bash
python download_embeddings.py -e embeddings -d volume_em
```

You can then start the 3d annotation tool, either via the napari plugin (we will show this in the workshop) or the command line:

```bash
micro_sam.annotator_3d -i data/volume_em/images/train_data_membrane_02.tif -e embeddings/volume_em/vit_b/platynereis_membrane_train_data_membrane_02.zarr -m vit_b
```

Note: You can use `micro_sam` with different models: the original models from Segment Anything and models finetuned for different microscopy segmentation tasks by us.
For segmentation in EM you can either use `vit_b` (the original model) or `vit_b_em_organelles` (our model). Our `vit_b_lm` model will likely be better for nucleus or mitochondrium segmentation, for other structures `vit_b` will likely be better, so it makes sense to test both before annotating your data. Please refer to [our documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models) for details on our models.

**If you want to bring your own data for annotation please store it in a similar format to the example data. You DO NOT have to provide segmentation masks; we include them here only for reference and they are not needed for annotation with micro_sam. Please also precompute the embeddings for your data, see [Precompute Embeddings](#precompute-embeddings) for details.**

### Model Finetuning

You can [finetune a model](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#training-your-own-model) for interactive and automatic segmentation on custom data.
We provide an example notebook `finetune_sam.ipynb` and script `finetune_sam.py` for this, as well as example data from the [HPA Segmentation and Classification Challenge](https://www.kaggle.com/c/hpa-single-cell-image-classification).

You can download the sample data by running:
```bash
python download_datasets.py -i data -d hpa
```

Note: You need a GPU in order to finetune the model (finetuning on the CPU is possible but takes too long for the workshop).
We have prepared the notebook so that it can be run on [kaggle](ttps://www.kaggle.com/code/) with a GPU, which you can use for the course. If you want to use this option please make sure that you can log in there before the workshop.

**If you want to bring your own data for training please store it in a similar format to the example data. You have to bring both images and annotations (= instance segmentation masks) for training. If you want to use kaggle please also upload your data so that you can retrieve it within the notebook.**

### Advanced applications: scripting with `micro_sam`

If you want to develop applications based on `micro_sam` you can use
the [micro_sam python library](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-the-python-library) to implement your own functionality.
For example, you could implement a script to segment cells based on prompts derived from a nucleus segmentation via [batched inference](https://computational-cell-analytics.github.io/micro-sam/micro_sam/inference.html#batched_inference).
Or a script to automatically segment data with a finetuned model using [automatic segmentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam/automatic_segmentation.html).


### Precompute Embeddings

You can use the command line to precompute embeddings for volumetric segmentation.
Here is the example script for pre-computing the embeddings on the [3D nucleus segmentation data](#3d-lm-segmentation).

```bash
micro_sam.precompute_embeddings -i data/nuclei_3d/images/X1.tif  # Filepath where inputs are stored.
                                -m vit_b  # You can provide name for a model of your choice (supported by 'micro-sam') (eg. 'vit_b_lm').
                                -e embeddings/vit_b/nuclei_3d_X1  # Filepath where computed embeddings will be stored.
```

You need to adapt the path to the data, choose the model you want to use (`vit_b`, `vit_b_lm`, `vit_b_em_organelles`) and adapt the path where the embeddings should be saved.

This step will take ca. 30 minutes for a volume with 200 image planes on a CPU.
