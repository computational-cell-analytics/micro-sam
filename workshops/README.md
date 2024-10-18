# Hands-On Analysis using `micro-sam`

## Upcoming Workshops:
1. I2K 2024 (Milan, Italy)
2. Virtual I2K 2024 (Online)

## Introduction

In this document, we walk you through different steps involved to participate in hands-on image annotation experiments our tool.

- Here is our [official documentation](https://computational-cell-analytics.github.io/micro-sam/) for detailed explanation of our tools, library and the finetuned models.
- Here is the playlist for our [tutorial videos](https://youtube.com/playlist?list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&si=3q-cIRD6KuoZFmAM) hosted on YouTube, elaborating in detail on the features of our tools.

## Steps:

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
