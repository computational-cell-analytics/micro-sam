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

All tools need the image embeddings, which you compute with `Compute Embeddings` after selecting the image and the model. Tiling (in `Embedding Settings`) is the most important setting for large images, see the question on high-resolution images in [the FAQ](#usage-questions).
Keep the image layer at the bottom of the napari layer list, so that the prompts, annotations and segmentations are shown on top of it.

The segmentation and tracking tools work best with a GPU or on a Mac with an M-series chip. On a CPU they are slow, and 3d segmentation and automatic tracking take very long. The classifiers also work well on a CPU.

The videos for the previous version show the general workflow, but the user interface has changed since then: [2d annotation](https://www.youtube.com/watch?v=9xjJBg_Bfuc&list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&index=2), [3d annotation](https://www.youtube.com/watch?v=nqpyNQSyu74&list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&index=3) and [tracking](https://www.youtube.com/watch?v=1gg8OPHqOyc&list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&index=4).


## Segmentation Annotator

The segmentation annotator is used for 2d images and 3d volumes. It can be started by
- clicking `Segmentation Annotator` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator segmentation -i <IMAGE>` in the command line.
- calling `micro_sam.sam_annotator.annotator` in a python script. Check out [examples/annotator_2d.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/annotator_2d.py) and [examples/annotator_3d.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/annotator_3d.py) for details.

The tool detects if the image is 2d, 2d with color channels, or 3d. You can change this in the embedding menu if the detection is wrong, for example for a 3d array that is a 2d image with channels.

The tool uses the following napari layers:
- `prompts`: shape layer for box and scribble prompts. A rectangle, ellipse or polygon marks the object, like a box. A path or line (`Add path`, `Add polyline` or `Add line` in the layer controls) is a scribble, which can be positive (on the object) or negative (outside of it).
- `point_prompts`: point layer for point prompts. Positive prompts (green points) mark the object you want to segment, negative prompts (red points) mark the outside of the object.
- `current_object`: label layer for the object(s) you are currently segmenting.
- `auto_segmentation`: label layer with the results of the automatic segmentation.
- `committed_objects`: label layer with the objects that you have already segmented.

The tool has the following menus:
- The embedding menu, for selecting the image and the model, and computing the image embeddings with `Compute Embeddings`. The `Embedding Settings` contain the settings for saving the embeddings to a file and loading them from it, for tiled embeddings of large images, for custom model weights and for the device.
- The menu for interactive segmentation. `Segment Object` (or `S`) segments the object from the current prompts and shows it in `current_object`. Press `T` (or use the `prompt` dropdown) to switch the next point or scribble between positive and negative.
    - With `Batched`, several objects are segmented at once: one object per box or per positive point. `Batched` is not available for tiled embeddings or with scribble prompts.
    - For 3d data, `Apply to Volume` extends the segmentation of the current object through the whole volume. Otherwise only the current slice is segmented.
- The menu for automatic segmentation. `Automatic Segmentation` segments all objects and shows them in `auto_segmentation`. The `mode` selects the method: `apg`, or AIS with the `sparse` or `dense` post-processing (see [APG](#apg)). Use `sparse` for cells and nuclei in light microscopy and `dense` for neurites in electron microscopy. `apg` refines the decoder predictions with automatically generated prompts and can be used for both. Automatic segmentation needs a model with a decoder, for example one of the `micro-sam2` models.
    - The `Advanced Settings` are post-processing parameters. The model predictions are computed only once, so changing them and running again is fast.
    - For 3d data, `Apply to Volume` segments the whole volume. Otherwise only the current slice is segmented.
- The menu for committing the segmentation. `Commit` (or `C`) transfers the result of the selected layer (`current_object` or `auto_segmentation`) to `committed_objects`. The layer of the last segmentation is selected automatically, but you can change it.
- The menu for clearing the annotations. `Clear Annotations` (or `Shift + C`) clears the current prompts and the current segmentation.

Point, box and scribble prompts can be combined.
You can also combine automatic and interactive segmentation: commit the result of `Automatic Segmentation`, and then segment the missing objects interactively and commit them.

## Tracking Annotator

The tracking annotator is used for 2d timeseries. It can be started by
- clicking `Tracking Annotator` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator tracking -i <TIMESERIES>` in the command line.
- calling `micro_sam.sam_annotator.annotator_tracking` in a python script. Check out [examples/annotator_tracking.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/annotator_tracking.py) for details.

Most elements are the same as in the [segmentation annotator](#segmentation-annotator):
- The tracking settings: `track_state` marks that the object you are tracking divides in the current frame, and `track_id` selects which of the tracks after a division you are following.
- The menu for interactive tracking: `Segment Object` (or `S`) segments the object in the current frame. With `Apply to Volume`, the object is tracked through the whole timeseries.
- The menu for automatic tracking: `Automatic Tracking` segments each frame and links the objects across frames with [trackastra](https://github.com/weigertlab/trackastra). Keep the segmentation `mode` at `sparse` for cells. Automatic tracking can take quite long without a GPU.
- `Commit` stores the tracks in `committed_objects` and shows them in the `tracks` layer. `Export` saves the tracking results in the Cell Tracking Challenge (CTC), GEFF or TrackMate XML format.

To track a cell that divides, use point prompts:
1. Track the mother cell as usual. In the last frame before the division, place a point on it, set `track_state` to `division` and run `Segment Object` again with `Apply to Volume`.
2. Go to the first frame after the division and place a point on one of the daughter cells. Set `track_state` back to `track`, set `track_id` to a new id (for example from 1 to 2), and track the daughter cell until its next division or the end of the timeseries.
3. Go back to the first frame after the division, place a point on the other daughter cell, set `track_id` to the next new id (for example 3), and track it in the same way.

The tracking annotator only supports 2d image data with a time dimension. Volumetric data with a time dimension is not supported.

## Batch Annotator

The batch annotator runs one of the other tools for all images in a folder, one image after the other. This makes it convenient to annotate many images without having to restart the tool for every image. It can be started by
- clicking `Batch Annotator` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator batch -i <FOLDER> -o <OUTPUT_FOLDER> --task <TASK>` in the command line.
- calling `micro_sam.sam_annotator.batch_annotator` or `micro_sam.sam_annotator.image_folder_annotator` in a python script. Check out [examples/image_series_annotator.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/image_series_annotator.py) for details.

You select the task (segmentation, tracking, object classification or pixel classification), the folder with your images, a pattern for selecting a subset of the images (for example `*.tif`), and the folder where the results are saved. For object classification, you can also select a folder with one segmentation per image. The other options are the settings of the embedding menu. Click `Annotate Images` to start.
`Next Image` (or `N`) saves the result of the current image and opens the next one. The classification tasks can also go back to the previous image (`Shift + N`), and with `Keep Classifier` the classifier is trained on the annotations of all images so far.
Images that already have a result in the output folder are skipped, so you can continue an annotation session later. For segmentation, uncheck `Continue Annotation` to start again at the first image, for example to review the results.

## Object Classifier

The object classifier classifies segmented objects into the classes you choose. It can be started by:
- clicking `Object Classifier` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator object-classification -i <IMAGE> -s <SEGMENTATION>` in the command line.
- calling `micro_sam.sam_annotator.object_classifier.object_classifier` in a python script. Check out [examples/object_classifier.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/object_classifier.py) for details.

Load an image and a segmentation (for example from the segmentation annotator), select the segmentation layer in the `Segmentation` dropdown, and compute the image embeddings as in the other tools.
Then mark some objects of each class in the `annotations` layer with the paint brush: one brush stroke or dot per object is enough. You can add classes with `Add Class`, and switch to the previous or next class with `Alt + Left` and `Alt + Right`. `Shift + S` shows or hides the segmentation.
`Train and Predict` (or `Shift + T`) trains a random forest classifier on the objects you have marked, using the image embeddings of each object as features, and applies it to all objects. The result is shown in the `prediction` layer.
You can export the trained classifier and apply it to other images with `micro_sam inference object-classification`.

The classifiers can use other models than the segmentation and tracking tools. The `Model` dropdown offers the SAM2 models for `Natural Images` and the `micro-sam2` models for `Microscopy`. With `Advanced Models` in the `Embedding Settings`, you can also select SAM1, SAM3, the `micro_sam` (SAM1) models, PathoSAM, MedicoSAM, DINOv2, DINOv3, UNI and UNI2-h. This also applies to the pixel classifier.

## Pixel Classifier

The pixel classifier assigns a class to every pixel, for example to segment tissue types or cell compartments. It can be started by:
- clicking `Pixel Classifier` in the plugin menu after starting `napari`.
- running `$ micro_sam annotator pixel-classification -i <IMAGE>` in the command line.
- calling `micro_sam.sam_annotator.pixel_classifier.pixel_classifier` in a python script. Check out [examples/pixel_classifier.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/examples/pixel_classifier.py) for details.

Draw brush strokes for each class in the `annotations` layer (switch classes as in the object classifier) and click `Train and Predict` (or `Shift + T`). A random forest classifier is trained on the image embeddings of the annotated pixels and applied to the whole image. The result is shown in the `prediction` layer. You can add more strokes where the prediction is wrong and train again.
Optional settings add classical image features, or upsample the embeddings with [AnyUp](https://github.com/wimmerth/anyup) for finer details.
You can export the trained classifier and apply it to other images with `micro_sam inference pixel-classification`.

## Training a Model

The previous version of `micro_sam` had a graphical interface for finetuning. To train a model on your own data, use the command line (`micro_sam train`) or the python library instead, see [training your own model](#training-your-own-model).
