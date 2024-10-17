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
where, `DATA_DIRECTORY` is the filepath where the datasets will be downloaded, and `DATASET_NAME` is the name of the dataset (see `python download_datasets.py -h` for the details)

### Step 2: Download the Precomputed Embeddings

- ...

> platynereis/membrane/train_data_membrane_02.n5 (Platynereis Cells)

> embedseg/Mouse-Skull-Nuclei-CBG/train/images/X1.tif (EmbedSeg)

> CellPose: chosen 15 images.

### Step 3: Run the `micro-sam` Annotators

- ...

### Step 4: Finetune Segment Anything on Microscopy Images

- ...
