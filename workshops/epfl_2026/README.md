# EPFL 2026 Workshop: Segment Anything for Microscopy

This document walks you through the preparation and applications for a workshop on "Segment Anything for Microscopy" (`micro_sam`) given at EPFL on the 31st of March 2026.


## Workshop overview

The workshop will be divided into the following parts:
1. Installation of `micro_sam` on your laptop.
2. Short introduction. You can find the slides [here](https://docs.google.com/presentation/d/1c0kSomnr2Dd_CgjBQnBFUKByj62yHilz/edit?usp=sharing&ouid=113044948772353505255&rtpof=true&sd=true).
3. Tutorial on using the `micro_sam` napari plugin for 2d and 3d segmentation.
4. Using `micro_sam` for different [applications](example-applications), on your own or on example data.
5. Questions & further steps.

**Please read the [Workshop Preparation](#workshop-preparation) section carefully and follow it to get started.**


## Workshop preparation
 
To prepare for the workshop, please:
- Install the latest version of `micro_sam`, see the [installation section of the documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-conda) for details.
- Download the models and the data for 3D segmentation, see [here](#model-and-data-download).
- If you have any data you want to try `micro_sam` on: Bring it to the workshop in tiff format. 

You can run the workshop on your laptop. It works best on a Macbook with a M-series chip.
Advanced applications, like training a custom `micro_sam` model or processing large amounts of data, require a GPU.

If you want to learn more about `micro_sam` you can check out the [documentation](https://computational-cell-analytics.github.io/micro-sam/) and [video tutorials](https://youtube.com/playlist?list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&si=3q-cIRD6KuoZFmAM).

### Model and data download

We provide a script to download the models used in the workshop. To run it you first need to activate the conda environment where you have installed `micro_sam`.
Then, if you have not done so already, download the `micro_sam` repository via git:
```bash
git clone https://github.com/computational-cell-analytics/micro-sam
cd micro-sam
```

Go to the workshop directory:
```bash
cd workshops/epfl_2026
```

and run the script:
```bash
python download_models.py
```

We also provide a script to download the embeddings for 3D segmentation. They are necessary to use the plugin. Computing them on the CPU will take some time for volumetric data, so we have precomputed them for you.
You can download them by running:
```bash
python download_embeddings.py -d lucchi
```

## Example applications

We have prepared four example applications for the workshop:
- 3D segmentation in light microscopy ([3D LM segmentation](#3d-lm-segmentation)) 
- 3D segmentation in electron microscopy ([3D EM segmentation](#3d-em-segmentation)).
- High-throughput image annotation ([high-throughput-annotation](#high-throughput-image-annotation)).
- **Advanced**: Finetuning on custom data ([model finetuning](#model-finetuning)).

### 3D LM Segmentation

You can use the [3D annotation tool](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#annotator-3d) to segment cells or nuclei in volumetric light microscopy. We have prepared an example dataset (from the [EmbedSeg publication](https://github.com/juglab/EmbedSeg)) that you can use.

Download the data with the script `download_dataset.py`:
```bash
python download_datasets.py -d nuclei_3d
```

Download the precomputed embeddings:
```bash
python download_embeddings.py -d nuclei_3d
```

You can then start the 3d annotation tool, either via the napari plugin (we will show this in the workshop) or the command line:
```bash
micro_sam.annotator_3d -i data/nuclei_3d/images/X1.tif -e embeddings/nuclei_3d/embeds-nuclei3d.zarr
```

### 3D EM Segmentation

You can use the [3D annotation tool](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#annotator-3d) to segment structures in volume electron microscopy.
We have prepared an example dataset that you can use. It consists of a small crop from an EM volume from [Hernandez et al.](https://doi.org/10.1016/j.cell.2021.07.017). The volume contains several cells, wherein you can segment cellular ultrastructure such as nuclei or mitochondria.

You can download the data with the script `download_dataset.py`:
```bash
python download_datasets.py -d volume_em
```

After this, download the precomputed embeddings:
```bash
python download_embeddings.py -d volume_em
```

You can then start the 3d annotation tool, either via the napari plugin (we will show this in the workshop) or the command line:
```bash
micro_sam.annotator_3d -i data/volume_em/images/train_data_membrane_02.tif -e embeddings/volume_em/embeds_platy.zarr -m vit_b_em_organelles
```

### High-throughput Image Annotation

You can use the [Image Series Annotator](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#image-series-annotator) to run interactive segmentation for multiple images stored in a folder.
This annotation mode is well suited for generating annotations for 2D cell segmentation or similar analysis tasks.

We have prepared an example dataset for the workshop that you can use. It consists of 15 images from the [CellPose](https://www.cellpose.org/) dataset. You can download the data with the script `download_dataset.py`:

```bash
python download_datasets.py -d cells
```

After this you can start the image series annotation tool via the command line:

```bash
micro_sam.image_series_annotator -i data/cells/images -o annotations/cells -e embeddings/embeds_cells.zarr
```

### Model Finetuning

You can [finetune a model](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#training-your-own-model) for interactive and automatic segmentation on your own data. This will improve it significantly for similar data.
We provide an example script `finetune_sam.py` and example data from the [HPA Challenge](https://www.kaggle.com/c/hpa-single-cell-image-classification) for model finetuning.

You can download the sample data by running:
```bash
python download_datasets.py -d hpa
```

Note: You need a GPU in order to finetune the model. So if your laptop does not have a GPU, you have to set up `micro_sam` on a workstation with a GPU, a computer cluster, or similar.
If you want to finetune on your own data, please store it in a similar format to the example data. You have to bring both images and annotations (= instance segmentation masks) for training.


<!---
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
-->
