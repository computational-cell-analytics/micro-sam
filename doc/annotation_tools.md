# Annotation Tools

`micro_sam` provides applications for fast interactive 2d and 3d segmentation, tracking, and object and pixel classification.
See an example for interactive cell segmentation in phase-contrast microscopy (left), interactive segmentation
of mitochondria in volume EM (middle) and interactive tracking of cells (right).

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d5ee2080-ab08-4716-b4c4-c169b4ed29f5" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfca3d9b-dba5-440b-b0f9-72a0683ac410" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/aefbf99f-e73a-4125-bb49-2e6592367a64" width="256">

The annotation tools can be started from the napari plugin menu (`Plugins -> Segment Anything for Microscopy`), the command line or from python scripts.
They are built as napari plugin and make use of existing napari functionality wherever possible. If you are not familiar with napari, we recommend to [start here](https://napari.org/stable/tutorials/fundamentals/quick_start.html).
The `micro_sam` tools mainly use the [point layer](https://napari.org/stable/howtos/layers/points.html), [shape layer](https://napari.org/stable/howtos/layers/shapes.html) and [label layer](https://napari.org/stable/howtos/layers/labels.html).

The plugin menu contains the following tools:
- [Segmentation Annotator](#segmentation-annotator): interactive and automatic segmentation of 2d images and 3d volumes.
- [Tracking Annotator](#tracking-annotator): interactive and automatic tracking of cells in a timeseries.
- [Batch Annotator](#batch-annotator): annotation of all images in a folder, one after the other.
- [Object Classifier](#object-classifier): classification of segmented objects.
- [Pixel Classifier](#pixel-classifier): semantic segmentation by classifying every pixel.

The plugin menu also contains sample data, and `Set cache directory` for choosing where the models and embeddings are stored.

You can find additional information on the annotation tools [in the FAQ section](#usage-questions).

HINT: If you would like to start napari to use `micro-sam` from the plugin menu, you must start it by activating the environment where `micro-sam` has been installed using:

```bash
conda activate <ENVIRONMENT_NAME>
napari
```


## Segmentation Annotator

The segmentation annotator is used for 2d images and 3d volumes. It can be started by
- clicking `Segmentation Annotator` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator segmentation -i <IMAGE>` in the command line.
- calling `micro_sam.sam_annotator.annotator` in a python script. Check out [examples/annotator_2d.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/annotator_2d.py) and [examples/annotator_3d.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/annotator_3d.py) for details.

The tool detects if the image is 2d, 2d with color channels, or 3d. You can change this in the embedding menu if the detection is wrong, for example for a 3d array that is a 2d image with channels.

The tool uses the following napari layers:
- `prompts`: shape layer for box prompts. Prompts can be given as rectangle, ellipse or polygon.
- `point_prompts`: point layer for point prompts. Positive prompts (green points) mark the object you want to segment, negative prompts (red points) mark the outside of the object.
- `current_object`: label layer for the object(s) you are currently segmenting.
- `auto_segmentation`: label layer with the results of the automatic segmentation.
- `committed_objects`: label layer with the objects that you have already segmented.

The tool has the following menus:
- The embedding menu, for selecting the image and the model, and computing the image embeddings with `Compute Embeddings`. The `Embedding Settings` contain the settings for saving the embeddings to a file and loading them from it, for tiled embeddings of large images, for custom model weights and for the device.
- The menu for interactive segmentation. `Segment Object` (or `S`) segments the object from the current prompts and shows it in `current_object`. Press `T` to switch a prompt between positive and negative.
    - For 3d data, `Apply to Volume` extends the segmentation of the current object through the whole volume. Otherwise only the current slice is segmented.
- The menu for automatic segmentation. `Automatic Segmentation` segments all objects and shows them in `auto_segmentation`. The `mode` selects the method: `apg`, or AIS with the `sparse` or `dense` post-processing (see [APG](#apg)). Automatic segmentation needs a model with a decoder, for example one of the `micro-sam2` models.
    - For 3d data, `Apply to Volume` segments the whole volume. Otherwise only the current slice is segmented.
- The menu for committing the segmentation. `Commit` (or `C`) transfers the result of the selected layer (`current_object` or `auto_segmentation`) to `committed_objects`. The layer of the last segmentation is selected automatically, but you can change it.
- The menu for clearing the annotations. `Clear Annotations` (or `Shift + C`) clears the current prompts and the current segmentation.

Point prompts and box prompts can be combined.

## Tracking Annotator

The tracking annotator is used for 2d timeseries. It can be started by
- clicking `Tracking Annotator` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator tracking -i <TIMESERIES>` in the command line.
- calling `micro_sam.sam_annotator.annotator_tracking` in a python script. Check out [examples/annotator_tracking.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/annotator_tracking.py) for details.

Most elements are the same as in the [segmentation annotator](#segmentation-annotator):
- The tracking settings: `track_state` marks that the object you are tracking divides in the current frame, and `track_id` selects which of the tracks after a division you are following.
- The menu for interactive tracking: `Segment Object` (or `S`) segments the object in the current frame. With `Apply to Volume`, the object is tracked through the whole timeseries.
- The menu for automatic tracking: `Automatic Tracking` segments each frame and links the objects across frames with [trackastra](https://github.com/weigertlab/trackastra). Automatic tracking can take quite long without a GPU.
- `Commit` stores the tracks in `committed_objects` and shows them in the `tracks` layer. The tracking results can be exported in the Cell Tracking Challenge, GEFF and TrackMate formats.

The tracking annotator only supports 2d image data with a time dimension. Volumetric data with a time dimension is not supported.

## Batch Annotator

The batch annotator runs one of the other tools for all images in a folder, one image after the other. This makes it convenient to annotate many images without having to restart the tool for every image. It can be started by
- clicking `Batch Annotator` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator batch -i <FOLDER> -o <OUTPUT_FOLDER> --task <TASK>` in the command line.
- calling `micro_sam.sam_annotator.batch_annotator` or `micro_sam.sam_annotator.image_folder_annotator` in a python script. Check out [examples/image_series_annotator.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/image_series_annotator.py) for details.

You select the task (segmentation, tracking, object classification or pixel classification), the folder with your images, a pattern for selecting a subset of the images (for example `*.tif`), and the folder where the results are saved. The other options are the settings of the embedding menu.
`Next Image` (or `N`) saves the result of the current image and opens the next one. The classification tasks can also go back to the previous image (`Shift + N`), and with `Keep Classifier` the classifier is trained on the annotations of all images so far.
For segmentation and tracking, images that already have a result in the output folder are skipped, so you can continue an annotation session later.

## Object Classifier

The object classifier classifies segmented objects into the classes you choose. It can be started by:
- clicking `Object Classifier` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator object-classification -i <IMAGE> -s <SEGMENTATION>` in the command line.
- calling `micro_sam.sam_annotator.object_classifier.object_classifier` in a python script. Check out [examples/object_classifier.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/object_classifier.py) for details.

Load an image and a segmentation (for example from the segmentation annotator), and compute the image embeddings as in the other tools.
Then mark some objects of each class in the `annotations` layer: one brush stroke or dot per object is enough. You can add classes with `Add Class`.
`Train and Predict` (or `Shift + T`) trains a random forest classifier on the objects you have marked, using the image embeddings of each object as features, and applies it to all objects. The result is shown in the `prediction` layer.
You can export the trained classifier and apply it to other images with `micro_sam inference object-classification`.

## Pixel Classifier

The pixel classifier assigns a class to every pixel, for example to segment tissue types or cell compartments. It can be started by:
- clicking `Pixel Classifier` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator pixel-classification -i <IMAGE>` in the command line.
- calling `micro_sam.sam_annotator.pixel_classifier.pixel_classifier` in a python script. Check out [examples/pixel_classifier.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/pixel_classifier.py) for details.

Draw brush strokes for each class in the `annotations` layer and click `Train and Predict` (or `Shift + T`). A random forest classifier is trained on the image embeddings of the annotated pixels and applied to the whole image. The result is shown in the `prediction` layer. You can add more strokes where the prediction is wrong and train again.
Optional settings add classical image features, or upsample the embeddings with [AnyUp](https://github.com/wimmerth/anyup) for finer details.
You can export the trained classifier and apply it to other images with `micro_sam inference pixel-classification`.

## Training a Model

The previous version of `micro_sam` had a graphical interface for finetuning. To train a model on your own data, use the command line (`micro_sam train`) or the python library instead, see [training your own model](#training-your-own-model).
