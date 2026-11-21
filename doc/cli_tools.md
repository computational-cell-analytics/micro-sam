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

The commands read these input data:
- Image files, for example tif or png.
- Container files: hdf5, zarr, OME-Zarr or n5. Use `-k` / `--key` to select the data in the file.
- Folders with image or container files. Use `--pattern` to select the files, for example `--pattern '*.ome.zarr'`.

The key is optional for zarr, OME-Zarr and n5 files:
- For an OME-Zarr file, the commands use the highest resolution of the image. For a segmentation input, for example `--segmentation_key`, they use the single label image in `labels/`.
- For a zarr or n5 file, the commands use the single array in the file.
- If a file has more than one image or array, the command shows an error that lists them. Then give the key, for example `-k labels/nuclei`.

The commands `micro_sam inference segmentation`, `pixel-classification` and `object-classification` write these outputs with `-o` / `--output`:
- A tif file for a single input. A path without an extension also writes a tif file.
- An OME-Zarr file (`*.ome.zarr`) for a single input. The command writes the result as an OME-Zarr label image:
    - If the file is an OME-Zarr image, for example the input itself, the command writes the label image `labels/<name>`. The label image has the same scale levels as the image.
    - If the file does not exist, the command creates it and writes the label image at its root.
    - Use `--label_name` to give the name. By default, the name is the name of the image for an OME-Zarr input, and the file name for other inputs.
- A folder with a tif file for each input, if there are multiple inputs.

If a result exists already (the tif file, or the label image in the OME-Zarr file), the command skips this input and shows an info. Use `--overwrite` to replace the result.

An OME-Zarr label image also stores how the command made it. The commands write this metadata to the attributes of the label image, according to the [provenance zarr convention](https://github.com/BioImageTools/zarr-convention-provenance/blob/v0.2.0/README.md):
- `provenance:steps` holds one step for the command. The python function `automatic_instance_segmentation` writes the same step.
- `software` gives the `micro_sam` version and the function or command.
- `parameters` gives the model, the segmentation mode, the tiling and the parameters of the segmentation, with the default values.
- `inputs` gives the image, the model, and the decoder or classifier. A model from the model registry has its download URL and checksum. A local file has only its file name, and a local model or classifier also has its checksum. The metadata does not contain folder names.

A tif file does not store this metadata.

The automatic segmentation uses the best post-processing parameters for the chosen model by default.
You can change a parameter by passing it to `micro_sam inference segmentation` as `--<PARAMETER_NAME> <VALUE>`, for example `--min_size 100` to remove smaller objects.
The defaults are listed by `micro_sam.v2.postprocessing.default_postprocessing` (AIS) and `micro_sam.v2.automatic_prompt_generation.default_prompt_generation` (APG).

Some examples:
```bash
# Automatic segmentation of all tif images in a folder (quote the pattern so that the shell does not expand it).
micro_sam inference segmentation -i images/ -o segmentations/ --pattern '*.tif'

# Automatic segmentation with APG and a larger model.
micro_sam inference segmentation -i image.tif -o segmentation.tif -m hvit_b_cells --engine apg

# Automatic segmentation of all OME-Zarr images in a folder, without a key.
micro_sam inference segmentation -i images/ -o segmentations/ --pattern '*.ome.zarr'

# Automatic segmentation of an OME-Zarr image. The result is written to 'labels/<name>' of the same file.
micro_sam inference segmentation -i image.ome.zarr -o image.ome.zarr --label_name cells

# Automatic tracking of a timeseries, saved in the Cell Tracking Challenge format.
micro_sam inference tracking -i timeseries.tif -o tracking_result/

# Precompute the embeddings and start the annotator with them.
micro_sam precompute-embeddings -i image.tif -e embeddings.zarr -m hvit_t_cells
micro_sam annotator segmentation -i image.tif -e embeddings.zarr -m hvit_t_cells

# Train a model for interactive and automatic segmentation on your own data.
micro_sam train -i images/ --image_key '*.tif' -l labels/ --label_key '*.tif' --mode joint
```
