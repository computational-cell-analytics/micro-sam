# Annotation Tools

`micro_sam` provides applications for fast interactive 2d segmentation, 3d segmentation and tracking.
See an example for interactive cell segmentation in phase-contrast microscopy (left), interactive segmentation
of mitochondria in volume EM (middle) and interactive tracking of cells (right).

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d5ee2080-ab08-4716-b4c4-c169b4ed29f5" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfca3d9b-dba5-440b-b0f9-72a0683ac410" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/aefbf99f-e73a-4125-bb49-2e6592367a64" width="256">

The annotation tools can be started from the `micro_sam` GUI, the command line or from python scripts. The `micro_sam` GUI can be started by
```
$ micro_sam.annotator
```

They are built using [napari](https://napari.org/stable/) and [magicgui](https://pyapp-kit.github.io/magicgui/) to provide the viewer and user interface.
If you are not familiar with napari yet, [start here](https://napari.org/stable/tutorials/fundamentals/quick_start.html).
The `micro_sam` tools use [the point layer](https://napari.org/stable/howtos/layers/points.html), [shape layer](https://napari.org/stable/howtos/layers/shapes.html) and [label layer](https://napari.org/stable/howtos/layers/labels.html).

The annotation tools are explained in detail below. In addition to the documentation here we also provide [video tutorials](https://www.youtube.com/watch?v=ket7bDUP9tI&list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO).


## Starting via GUI

The annotation toools can be started from a central GUI, which can be started with the command `$ micro_sam.annotator` or using the executable [from an installer](#from-installer).

In the GUI you can select with of the four annotation tools you want to use:
<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/micro-sam-gui.png">

And after selecting them a new window will open where you can select the input file path and other optional parameter. Then click the top button to start the tool. **Note: If you are not starting the annotation tool with a path to pre-computed embeddings then it can take several minutes to open napari after pressing the button because the embeddings are being computed.**


## Annotator 2D

The 2d annotator can be started by
- clicking `2d annotator` in the `micro_sam` GUI.
- running `$ micro_sam.annotator_2d` in the command line. Run `micro_sam.annotator_2d -h` for details.
- calling `micro_sam.sam_annotator.annotator_2d` in a python script. Check out [examples/annotator_2d.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_2d.py) for details. 

The user interface of the 2d annotator looks like this:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/2d-annotator-menu.png" width="768">

It contains the following elements:
1. The napari layers for the image, segmentations and prompts:
    - `box_prompts`: shape layer that is used to provide box prompts to SegmentAnything.
    - `prompts`: point layer that is used to provide prompts to SegmentAnything. Positive prompts (green points) for marking the object you want to segment, negative prompts (red points) for marking the outside of the object.
    - `current_object`: label layer that contains the object you're currently segmenting.
    - `committed_objects`: label layer with the objects that have already been segmented.
    - `auto_segmentation`: label layer results from using SegmentAnything for automatic instance segmentation.
    - `raw`: image layer that shows the image data.
2. The prompt menu for changing the currently selected point from positive to negative and vice versa. This can also be done by pressing `t`.
3. The menu for automatic segmentation. Pressing `Segment All Objects` will run automatic segmentation. The results will be displayed in the `auto_segmentation` layer. Change the parameters `pred iou thresh` and `stability score thresh` to control how many objects are segmented.
4. The menu for interactive segmentation. Pressing `Segment Object` (or `s`) will run segmentation for the current prompts. The result is displayed in `current_object`
5. The menu for commiting the segmentation. When pressing `Commit` (or `c`) the result from the selected layer (either `current_object` or `auto_segmentation`) will be transferred from the respective layer to `committed_objects`.
6. The menu for clearing the current annotations. Pressing `Clear Annotations` (or `shift c`) will clear the current annotations and the current segmentation.

Note that point prompts and box prompts can be combined. When you're using point prompts you can only segment one object at a time. With box prompts you can segment several objects at once.

Check out [this video](https://youtu.be/ket7bDUP9tI) for a tutorial for the 2d annotation tool.

We also provide the `image series annotator`, which can be used for running the 2d annotator for several images in a folder. You can start by clicking `Image series annotator` in the GUI, running `micro_sam.image_series_annotator` in the command line or from a [python script](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/image_series_annotator.py).

## Annotator 3D

The 3d annotator can be started by
- clicking `3d annotator` in the `micro_sam` GUI.
- running `$ micro_sam.annotator_3d` in the command line. Run `micro_sam.annotator_3d -h` for details.
- calling `micro_sam.sam_annotator.annotator_3d` in a python script. Check out [examples/annotator_3d.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_3d.py) for details.

The user interface of the 3d annotator looks like this:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/3d-annotator-menu.png" width="768">

Most elements are the same as in [the 2d annotator](#annotator-2d):
1. The napari layers that contain the image, segmentation and prompts. Same as for [the 2d annotator](#annotator-2d) but without the `auto_segmentation` layer.
2. The prompt menu.
3. The menu for interactive segmentation.
4. The 3d segmentation menu. Pressing `Segment All Slices` (or `Shift-S`) will extend the segmentation for the current object across the volume.
5. The menu for committing the segmentation.
6. The menu for clearing the current annotations.

Note that you can only segment one object at a time with the 3d annotator.

Check out [this video](https://youtu.be/PEy9-rTCdS4) for a tutorial for the 3d annotation tool.

## Annotator Tracking

The tracking annotator can be started by
- clicking `Tracking annotator` in the `micro_sam` GUI.
- running `$ micro_sam.annotator_tracking` in the command line. Run `micro_sam.annotator_tracking -h` for details.
- calling `micro_sam.sam_annotator.annotator_tracking` in a python script. Check out [examples/annotator_tracking.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/annotator_tracking.py) for details. 

The user interface of the tracking annotator looks like this:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/tracking-annotator-menu.png" width="768">

Most elements are the same as in [the 2d annotator](#annotator-2d):
1. The napari layers that contain the image, segmentation and prompts. Same as for [the 2d segmentation app](#annotator-2d) but without the `auto_segmentation` layer, `current_tracks` and `committed_tracks` are the equivalent of `current_object` and `committed_objects`.
2. The prompt menu.
3. The menu with tracking settings: `track_state` is used to indicate that the object you are tracking is dividing in the current frame. `track_id` is used to select which of the tracks after division you are following.
4. The menu for interactive segmentation.
5. The tracking menu. Press `Track Object` (or `v`) to track the current object across time.
6. The menu for committing the current tracking result.
7. The menu for clearing the current annotations.

Note that the tracking annotator only supports 2d image data, volumetric data is not supported.

Check out [this video](https://youtu.be/Xi5pRWMO6_w) for a tutorial for how to use the tracking annotation tool.

## Tips & Tricks

- Segment Anything was trained with a fixed image size of 1024 x 1024 pixels. Inputs that do not match this size will be internally resized to match it. Hence, applying Segment Anything to a much larger image will often lead to inferior results, because it will be downsampled by a large factor and the objects in the image become too small.
To address this image we implement tiling: cutting up the input image into tiles of a fixed size (with a fixed overlap) and running Segment Anything for the individual tiles.
You can activate tiling by passing the parameters `tile_shape`, which determines the size of the inner tile and `halo`, which determines the size of the additional overlap.
    - If you're using the `micro_sam` GUI you can specify the values for the `halo` and `tile_shape` via the `Tile X`, `Tile Y`, `Halo X` and `Halo Y`.
    - If you're using a python script you can pass them as tuples, e.g. `tile_shape=(1024, 1024), halo=(128, 128)`. See also [the wholeslide_annotator example](https://github.com/computational-cell-analytics/micro-sam/blob/0921581e2964139194d235a87cb002d3f3667f45/examples/annotator_2d.py#L40).
    - If you're using the command line functions you can pass them via the options `--tile_shape 1024 1024 --halo 128 128`
    - Note that prediction with tiling only works when the embeddings are cached to file, so you must specify an `embedding_path` (`-e` in the CLI).
    - You should choose the `halo` such that it is larger than half of the maximal radius of the objects your segmenting.
- The applications pre-compute the image embeddings produced by SegmentAnything and (optionally) store them on disc. If you are using a CPU this step can take a while for 3d data or timeseries (you will see a progress bar with a time estimate). If you have access to a GPU without graphical interface (e.g. via a local computer cluster or a cloud provider), you can also pre-compute the embeddings there and then copy them to your laptop / local machine to speed this up. You can use the command `micro_sam.precompute_state` for this (it is installed with the rest of the applications). You can specify the location of the precomputed embeddings via the `embedding_path` argument.
- Most other processing steps are very fast even on a CPU, so interactive annotation is possible. An exception is the automatic segmentation step (2d segmentation), which takes several minutes without a GPU (depending on the image size). For large volumes and timeseries segmenting an object in 3d / tracking across time can take a couple settings with a CPU (it is very fast with a GPU).
- You can also try using a smaller version of the SegmentAnything model to speed up the computations. For this you can pass the `model_type` argument and either set it to `vit_b` or to `vit_l` (default is `vit_h`). However, this may lead to worse results.
- You can save and load the results from the `committed_objects` / `committed_tracks` layer to correct segmentations you obtained from another tool (e.g. CellPose) or to save intermediate annotation results. The results can be saved via `File -> Save Selected Layer(s) ...` in the napari menu (see the tutorial videos for details). They can be loaded again by specifying the corresponding location via the `segmentation_result` (2d and 3d segmentation) or `tracking_result` (tracking) argument.

## Known limitations

- Segment Anything does not work well for very small or fine-grained objects (e.g. filaments).
- For the automatic segmentation functionality we currently rely on the automatic mask generation provided by SegmentAnything. It is slow and often misses objects in microscopy images. For now, we only offer this functionality in the 2d segmentation app; we are working on improving it and extending it to 3d segmentation and tracking.
- Prompt bounding boxes do not provide the full functionality for tracking yet (they cannot be used for divisions or for starting new tracks). See also [this github issue](https://github.com/computational-cell-analytics/micro-sam/issues/23).
