# Using the Command Line Interface (CLI)

`micro_sam` provides one command, `micro_sam`, with subcommands for annotation, automatic segmentation, embedding computation and training.
Run `micro_sam --help` for an overview, and add `--help` to any subcommand for its options, for example `micro_sam annotator segmentation --help`.

The subcommands are:
- `micro_sam info`: Display `micro_sam` and system information. `micro_sam info --download <MODEL>` downloads a model in advance.
- `micro_sam annotator`: Start the [annotation tools](#annotation-tools).
    - `micro_sam annotator segmentation`: Interactive segmentation of a 2d or 3d image.
    - `micro_sam annotator tracking`: Interactive tracking of cells in a timeseries.
    - `micro_sam annotator batch`: Annotation of all images in a folder, for segmentation, tracking, or object or pixel classification (`--task`).
    - `micro_sam annotator object-classification`: Train and apply an object classifier interactively.
    - `micro_sam annotator pixel-classification`: Train and apply a pixel classifier interactively.
- `micro_sam inference`: Automatic prediction for one or many images.
    - `micro_sam inference segmentation`: Automatic instance segmentation. Select the method with `--engine`: `ais` (the default), `apg` or `amg`, see [APG](#apg).
    - `micro_sam inference tracking`: Automatic tracking for a timeseries.
    - `micro_sam inference object-classification` and `micro_sam inference pixel-classification`: Apply a trained classifier.
- `micro_sam precompute-embeddings`: Compute the image embeddings in advance and save them, so that the annotation tools and the automatic segmentation can reuse them.
- `micro_sam train`: Train a model on your own data, see [training your own model](#training-your-own-model).
- `micro_sam train-generalist`: Train a generalist model on the collection of datasets we use for the `micro-sam2` models.
- `micro_sam v1`: The commands of the previous version for SAM1 models, for example `micro_sam v1 automatic_segmentation` and `micro_sam v1 evaluate`.

The automatic segmentation uses the best post-processing parameters for the chosen model by default.
You can change a parameter by passing it to `micro_sam inference segmentation` as `--<PARAMETER_NAME> <VALUE>`, for example `--min_size 100` to remove smaller objects.
The defaults are listed by `micro_sam.v2.postprocessing.default_postprocessing` (AIS) and `micro_sam.v2.automatic_prompt_generation.default_prompt_generation` (APG).

Some examples:
```bash
# Automatic segmentation of all tif images in a folder (quote the pattern so that the shell does not expand it).
micro_sam inference segmentation -i images/ -o segmentations/ --pattern '*.tif'

# Automatic segmentation with APG and a larger model.
micro_sam inference segmentation -i image.tif -o segmentation.tif -m hvit_b_cells --engine apg

# Automatic tracking of a timeseries, saved in the Cell Tracking Challenge format.
micro_sam inference tracking -i timeseries.tif -o tracking_result/

# Precompute the embeddings and start the annotator with them.
micro_sam precompute-embeddings -i image.tif -e embeddings.zarr -m hvit_t_cells
micro_sam annotator segmentation -i image.tif -e embeddings.zarr -m hvit_t_cells

# Train a model for interactive and automatic segmentation on your own data.
micro_sam train -i images/ --image_key '*.tif' -l labels/ --label_key '*.tif' --mode joint
```
