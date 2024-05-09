# Annotation Tools

`micro_sam` provides applications for fast interactive 2d segmentation, 3d segmentation and tracking.
See an example for interactive cell segmentation in phase-contrast microscopy (left), interactive segmentation
of mitochondria in volume EM (middle) and interactive tracking of cells (right).

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d5ee2080-ab08-4716-b4c4-c169b4ed29f5" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfca3d9b-dba5-440b-b0f9-72a0683ac410" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/aefbf99f-e73a-4125-bb49-2e6592367a64" width="256">

The annotation tools can be started from the napari plugin menu, the command line or from python scripts.
They are built as napari plugin and make use of existing napari functionality wherever possible. If you are not familiar with napari, we recommend to [start here](https://napari.org/stable/tutorials/fundamentals/quick_start.html).
The `micro_sam` tools mainly use the [point layer](https://napari.org/stable/howtos/layers/points.html), [shape layer](https://napari.org/stable/howtos/layers/shapes.html) and [label layer](https://napari.org/stable/howtos/layers/labels.html).

The annotation tools are explained in detail below. We also provide [video tutorials](https://youtube.com/playlist?list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&si=qNbB8IFXqAX33r_Z).

The annotation tools can be started from the napari plugin menu:
<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/napari-plugin.png" width="768">

You can find additional information on the annotation tools [in the FAQ section](#usage-question).


## Annotator 2D

The 2d annotator can be started by
- clicking `Annotator 2d` in the plugin menu.
- running `$ micro_sam.annotator_2d` in the command line.
- calling `micro_sam.sam_annotator.annotator_2d` in a python script. Check out [examples/annotator_2d.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py) for details.

The user interface of the 2d annotator looks like this:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/2d-annotator-menu.png" width="1024">

It contains the following elements:
1. The napari layers for the segmentations and prompts:
    - `prompts`: shape layer that is used to provide box prompts to Segment Anything. Prompts can be given as rectangle (marked as box prompt in the image), ellipse or polygon.
    - `point_prompts`: point layer that is used to provide point prompts to Segment Anything. Positive prompts (green points) for marking the object you want to segment, negative prompts (red points) for marking the outside of the object.
    - `committed_objects`: label layer with the objects that have already been segmented.
    - `auto_segmentation`: label layer with the results from automatic instance segmentation.
    - `current_object`: label layer for the object(s) you're currently segmenting.
2. The embedding menu. For selecting the image to process, the Segment Anything model that is used and computing its image embeddings. The `Embedding Settings` contain advanced settings for loading cached embeddings from file or for using tiled embeddings.
3. The prompt menu for changing whether the currently selected point is a positive or a negative prompt. This can also be done by pressing `T`.
4. The menu for interactive segmentation. Clicking `Segment Object` (or pressing `S`) will run segmentation for the current prompts. The result is displayed in `current_object`. Activating `batched` enables segmentation of multiple objects with point prompts. In this case one object will be segmented per positive prompt.
5. The menu for automatic segmentation. Clicking `Automatic Segmentation` will segment all objects n the image. The results will be displayed in the `auto_segmentation` layer. We support two different methods for automatic segmentation: automatic mask generation (supported for all models) and instance segmentation with an additional decoder (only supported for our models).
Changing the parameters under `Automatic Segmentation Settings` controls the segmentation results, check the tooltips for details.
6. The menu for commiting the segmentation. When clicking `Commit` (or pressing `C`) the result from the selected layer (either `current_object` or `auto_segmentation`) will be transferred from the respective layer to `committed_objects`.
When `commit_path` is given the results will automatically be saved there.
7. The menu for clearing the current annotations. Clicking `Clear Annotations` (or pressing `Shift + C`) will clear the current annotations and the current segmentation.

Point prompts and box prompts can be combined. When you're using point prompts you can only segment one object at a time, unless the `batched` mode is activated. With box prompts you can segment several objects at once, both in the normal and `batched` mode.

Check out [the video tutorial](https://youtu.be/9xjJBg_Bfuc) for an in-depth explanation on how to use this tool.


## Annotator 3D

The 3d annotator can be started by
- clicking `Annotator 3d` in the plugin menu.
- running `$ micro_sam.annotator_3d` in the command line.
- calling `micro_sam.sam_annotator.annotator_3d` in a python script. Check out [examples/annotator_3d.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_3d.py) for details.

The user interface of the 3d annotator looks like this:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/3d-annotator-menu.png" width="1024">

Most elements are the same as in [the 2d annotator](#annotator-2d):
1. The napari layers that contain the segmentations and prompts.
2. The embedding menu.
3. The prompt menu.
4. The menu for interactive segmentation in the current slice.
5. The menu for interactive 3d segmentation. Clicking `Segment All Slices` (or pressing `Shift + S`) will extend the segmentation of the current object across the volume by projecting prompts across slices. The parameters for prompt projection can be set in `Segmentation Settings`, please refer to the tooltips for details.
6. The menu for automatic segmentation. The overall functionality is the same as [for the 2d annotator](#annotator-2d). To segment the full volume `Apply to Volume` needs to be checked, otherwise only the current slice will be segmented. Note that 3D segmentation can take quite long without a GPU.
7. The menu for committing the current object.
8. The menu for clearing the current annotations. If `all slices` is set all annotations will be cleared, otherwise they are only cleared for the current slice.

You can only segment one object at a time using the interactive segmentation functionality with this tool.

Check out [the video tutorial](https://youtu.be/nqpyNQSyu74) for an in-depth explanation on how to use this tool.


## Annotator Tracking

The tracking annotator can be started by
- clicking `Annotator Tracking` in the plugin menu.
- running `$ micro_sam.annotator_tracking` in the command line.
- calling `micro_sam.sam_annotator.annotator_tracking` in a python script. Check out [examples/annotator_tracking.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_tracking.py) for details. 

The user interface of the tracking annotator looks like this:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/tracking-annotator-menu.png" width="1024">

Most elements are the same as in [the 2d annotator](#annotator-2d):
1. The napari layers that contain the segmentations and prompts. Same as for [the 2d segmentation application](#annotator-2d) but without the `auto_segmentation` layer.
2. The embedding menu.
3. The prompt menu.
4. The menu with tracking settings: `track_state` is used to indicate that the object you are tracking is dividing in the current frame. `track_id` is used to select which of the tracks after division you are following.
5. The menu for interactive segmentation in the current frame.
6. The menu for interactive tracking. Click `Track Object` (or press `Shift + S`) to segment the current object across time.
7. The menu for committing the current tracking result.
8. The menu for clearing the current annotations.

The tracking annotator only supports 2d image data with a time dimension, volumetric data + time is not supported. We also do not support automatic tracking yet.

Check out [the video tutorial](https://youtu.be/1gg8OPHqOyc) for an in-depth explanation on how to use this tool.


## Image Series Annotator

The image series annotation tool enables running the [2d annotator](#annotator-2d) or [3d annotator](#annotator-3d) for multiple images that are saved in a folder. This makes it convenient to annotate many images without having to restart the tool for every image. It can be started by
- clicking `Image Series Annotator` in the plugin menu.
- running `$ micro_sam.image_series_annotator` in the command line.
- calling `micro_sam.sam_annotator.image_series_annotator` in a python script. Check out [examples/image_series_annotator.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/image_series_annotator.py) for details. 

When starting this tool via the plugin menu the following interface opens:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/series-menu.png" width="512">

You can select the folder where your images are saved with `Input Folder`. The annotation results will be saved in `Output Folder`.
You can specify a rule for loading only a subset of images via `pattern`, for example `*.tif` to only load tif images. Set `is_volumetric` if the data you want to annotate is 3d. The rest of the options are settings for the image embedding computation and are the same as for the embedding menu (see above).
Once you click `Annotate Images` the images from the folder you have specified will be loaded and the annotation tool is started for them.

This menu will not open if you start the image series annotator from the command line or via python. In this case the input folder and other settings are passed as parameters instead.

Check out [the video tutorial](https://youtu.be/HqRoImdTX3c) for an in-depth explanation on how to use the image series annotator.


## Finetuning UI

We also provide a graphical inferface for fine-tuning models on your own data. It can be started by clicking `Finetuning` in the plugin menu.

**Note:** if you know a bit of python programming we recommend to use a script for model finetuning instead. This will give you more options to configure the training. See [these instructions](#training-your-own-model) for details.

When starting this tool via the plugin menu the following interface opens:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/finetuning-menu.png" width="512">

You can select the image data via `Path to images`. You can either load images from a folder or select a single image file. By providing `Image data key` you can either provide a pattern for selecting files from the folder or provide an internal filepath for HDF5, Zarr or similar fileformats.

You can select the label data via `Path to labels` and `Label data key`, following the same logic as for the image data. The label masks are expected to have the same size as the image data. You can for example use annotations created with one of the `micro_sam` annotation tools for this, they are stored in the correct format. See [the FAQ](#fine-tuning-questions) for more details on the expected label data.

The `Configuration` option allows you to choose the hardware configuration for training. We try to automatically select the correct setting for your system, but it can also be changed. Details on the configurations can be found [here](#training-your-own-model).
