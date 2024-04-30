# Annotation Tools

`micro_sam` provides applications for fast interactive 2d segmentation, 3d segmentation and tracking.
See an example for interactive cell segmentation in phase-contrast microscopy (left), interactive segmentation
of mitochondria in volume EM (middle) and interactive tracking of cells (right).

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d5ee2080-ab08-4716-b4c4-c169b4ed29f5" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfca3d9b-dba5-440b-b0f9-72a0683ac410" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/aefbf99f-e73a-4125-bb49-2e6592367a64" width="256">

The annotation tools can be started from the napari plugin menu, the command line or from python scripts.
They are built as napari plugin and make use of existing napari functionality wherever possible. If you are not familiar with napari yet, [start here](https://napari.org/stable/tutorials/fundamentals/quick_start.html).
The `micro_sam` tools mainly use [the point layer](https://napari.org/stable/howtos/layers/points.html), [shape layer](https://napari.org/stable/howtos/layers/shapes.html) and [label layer](https://napari.org/stable/howtos/layers/labels.html).

The annotation tools are explained in detail below. We also provide [video tutorials](TODO).

The annotation tools can be started from the napari plugin menu:
<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/napari-plugin.png" width="768">


## Annotator 2D

The 2d annotator can be started by
- clicking `Annotator 2d` in the plugin menu.
- running `$ micro_sam.annotator_2d` in the command line.
- calling `micro_sam.sam_annotator.annotator_2d` in a python script. Check out [examples/annotator_2d.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py) for details. 

The user interface of the 2d annotator looks like this:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/2d-annotator-menu.png" width="1024">

It contains the following elements:
1. The napari layers for the segmentations and prompts:
    - `prompts`: shape layer that is used to provide box prompts to SegmentAnything. Annotations can be given as rectangle (box prompt in the image), ellipse or polygon.
    - `point_prompts`: point layer that is used to provide point prompts to SegmentAnything. Positive prompts (green points) for marking the object you want to segment, negative prompts (red points) for marking the outside of the object.
    - `committed_objects`: label layer with the objects that have already been segmented.
    - `auto_segmentation`: label layer with the results from automatic instance segmentation.
    - `current_object`: label layer for the object(s) you're currently segmenting.
2. The embedding menu. For selecting the image to process, the Segment Anything model that is used and computing the image embeddings with the model. The `Embedding Settings` contain advanced settings for loading cached embeddings from file or using tiled embeddings.
3. The prompt menu for changing whether the currently selected point is a positive or a negative prompt. This can also be done by pressing `T`.
4. The menu for interactive segmentation. Clicking `Segment Object` (or pressing `S`) will run segmentation for the current prompts. The result is displayed in `current_object`. Activating `batched` enables segmentation of multiple objects with point prompts. In this case an object will be segmented per positive prompt.
5. The menu for automatic segmentation. Clicking `Automatic Segmentation` will segment all objects n the image. The results will be displayed in the `auto_segmentation` layer. We support two different methods for automatic segmentation: automatic mask generation (supported for all models) and instance segmentation with an additional decoder (only supported for our models).
Changing the parameters under `Automatic Segmentation Settings` controls the segmentation results, check the tooltips for details.
6. The menu for commiting the segmentation. When clicking `Commit` (or pressing `C`) the result from the selected layer (either `current_object` or `auto_segmentation`) will be transferred from the respective layer to `committed_objects`.
When `commit_path` is given the results will automatically be saved there.
7. The menu for clearing the current annotations. Clicking `Clear Annotations` (or pressing `Shift + C`) will clear the current annotations and the current segmentation.

Note that point prompts and box prompts can be combined. When you're using point prompts you can only segment one object at a time, unless the `batched` mode is activated. With box prompts you can segment several objects at once, both in the normal and `batched` mode.

Check out [this video](TODO) for a tutorial for this tool.


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
4. The menu for interactive segmentation.
5. The menu for interactive 3d segmentation. Clicking `Segment All Slices` (or `Shift + S`) will extend the segmentation for the current object across the volume by projecting prompts across slices. The parameters for prompt projection can be set in `Segmentation Settings`, please refer to the tooltips for details.
6. The menu for automatic segmentation. The overall functionality is the same as [for the 2d annotator](#annotator-2d). To segment the full volume `Apply to Volume` needs to be checked, otherwise only the current slice will be segmented. Note that 3D segmentation can take quite long without a GPU.
7. The menu for committing the current object.
8. The menu for clearing the current annotations. If `all slices` is set all annotations will be cleared, otherwise they are only cleared for the current slice.

Note that you can only segment one object at a time using the interactive segmentation functionality with this tool.

Check out [this video](TODO) for a tutorial for the 3d annotation tool.


## Annotator Tracking

The tracking annotator can be started by
- clicking `Annotator Tracking` in the plugin menu.
- running `$ micro_sam.annotator_tracking` in the command line.
- calling `micro_sam.sam_annotator.annotator_tracking` in a python script. Check out [examples/annotator_tracking.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_tracking.py) for details. 

The user interface of the tracking annotator looks like this:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/tracking-annotator-menu.png" width="1024">

Most elements are the same as in [the 2d annotator](#annotator-2d):
1. The napari layers that contain the segmentations and prompts. Same as for [the 2d segmentation app](#annotator-2d) but without the `auto_segmentation` layer.
2. The embedding menu.
3. The prompt menu.
4. The menu with tracking settings: `track_state` is used to indicate that the object you are tracking is dividing in the current frame. `track_id` is used to select which of the tracks after division you are following.
5. The menu for interactive segmentation.
6. The menu for interactive tracking menu. Click `Track Object` (or press `Shift + S`) to segment the current object across time.
7. The menu for committing the current tracking result.
8. The menu for clearing the current annotations.

Note that the tracking annotator only supports 2d image data, volumetric data is not supported. We also do not support automatic tracking yet.

Check out [this video](TODO) for a tutorial for how to use the tracking annotation tool.


## Image Series Annotator

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/series-menu.png" width="1024">

We also provide the `image series annotator`, which can be used for running the 2d annotator for several images in a folder. You can start by clicking `Image series annotator` in the GUI, running `micro_sam.image_series_annotator` in the command line or from a [python script](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/image_series_annotator.py).


## Finetuning Tool

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/finetuning-menu.png" width="1024">


## Tips & Tricks

- Segment Anything was trained with a fixed image size of 1024 x 1024 pixels. Inputs that do not match this size will be internally resized to match it. Hence, applying Segment Anything to a much larger image will often lead to inferior results, because it will be downsampled by a large factor and the objects in the image become too small.
To address this image we implement tiling: cutting up the input image into tiles of a fixed size (with a fixed overlap) and running Segment Anything for the individual tiles.
You can activate tiling by passing the parameters `tile_shape`, which determines the size of the inner tile and `halo`, which determines the size of the additional overlap.
    - If you're using the `micro_sam` GUI you can specify the values for the `halo` and `tile_shape` via the `Tile X`, `Tile Y`, `Halo X` and `Halo Y` by clicking on `Embeddings Settings`.
    - If you're using a python script you can pass them as tuples, e.g. `tile_shape=(1024, 1024), halo=(128, 128)`. See also [the wholeslide_annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/0921581e2964139194d235a87cb002d3f3667f45/examples/annotator_2d.py#L40).
    - If you're using the command line functions you can pass them via the options `--tile_shape 1024 1024 --halo 128 128`
    - Note that prediction with tiling only works when the embeddings are cached to file, so you must specify an `embedding_path` (`-e` in the CLI).
    - You should choose the `halo` such that it is larger than half of the maximal radius of the objects your segmenting.
- The applications pre-compute the image embeddings produced by SegmentAnything and (optionally) store them on disc. If you are using a CPU this step can take a while for 3d data or timeseries (you will see a progress bar with a time estimate). If you have access to a GPU without graphical interface (e.g. via a local computer cluster or a cloud provider), you can also pre-compute the embeddings there and then copy them to your laptop / local machine to speed this up. You can use the command `micro_sam.precompute_embeddings` for this (it is installed with the rest of the applications). You can specify the location of the precomputed embeddings via the `embedding_path` argument.
  - If you use the GUI to save or load embeddings, simply specify an `embeddings save path`. Existing embeddings are loaded from the specified path or embeddings are computed and the path is used to save them. 
- Most other processing steps are very fast even on a CPU, so interactive annotation is possible. An exception is the automatic segmentation step (2d segmentation), which takes several minutes without a GPU (depending on the image size). For large volumes and timeseries segmenting an object in 3d / tracking across time can take a couple settings with a CPU (it is very fast with a GPU).
- You can also try using a smaller version of the SegmentAnything model to speed up the computations. For this you can pass the `model_type` argument and either set it to `vit_b` or to `vit_l` (default is `vit_h`). However, this may lead to worse results.
- You can save and load the results from the `committed_objects` / `committed_tracks` layer to correct segmentations you obtained from another tool (e.g. CellPose) or to save intermediate annotation results. The results can be saved via `File -> Save Selected Layer(s) ...` in the napari menu (see the tutorial videos for details). They can be loaded again by specifying the corresponding location via the `segmentation_result` (2d and 3d segmentation) or `tracking_result` (tracking) argument.

## Known limitations

- Segment Anything does not work well for very small or fine-grained objects (e.g. filaments).
- For the automatic segmentation functionality we currently rely on the automatic mask generation provided by SegmentAnything. It is slow and often misses objects in microscopy images.
- Prompt bounding boxes do not provide the full functionality for tracking yet (they cannot be used for divisions or for starting new tracks). See also [this github issue](https://github.com/computational-cell-analytics/micro-sam/issues/23).
