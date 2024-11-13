# Using the Command Line Interface (CLI)

`micro-sam` extends access to a bunch of functionalities using the command line interface (CLI) scripts via terminal.

The supported CLIs can be used by
- Running `$ micro_sam.precompute_embeddings` for precomputing and caching the image embeddings.
- Running `$ micro_sam.annotator_2d` for starting the 2d annotator.
- Running `$ micro_sam.annotator_3d` for starting the 3d annotator.
- Running `$ micro_sam.annotator_tracking` for starting the tracking annotator.
- Running `$ micro_sam.image_series_annotator` for starting the image series annotator.
- Running `$ micro_sam.automatic_segmentation` for automatic instance segmentation.
    - We support all post-processing parameters for automatic instance segmentation (for both AMG and AIS).
    - If these parameters are not provided by the user, `micro-sam` makes use of the best post-processing parameters (depending on the choice of model). 
    - The post-processing parameters can be changed by parsing the parameters via the CLI using `--<PARAMETER_NAME> <VALUE>.` For example, one can update the parameter values (eg. `pred_iou_thresh`, `stability_iou_thresh`, etc. - supported by AMG) using
    ```bash
    $ micro_sam.automatic_segmentation ... --pred_iou_thresh 0.6 --stability_iou_thresh 0.6 ...
    ```
    - You can check details for supported parameters and their respective default values at `micro_sam/instance_segmentation.py` under the `generate` method for `AutomaticMaskGenerator` and `InstanceSegmentationWithDecoder` class.

NOTE: For all CLIs above, you can find more details by adding the argument `-h` to the CLI script (eg. `$ micro_sam.annotator_2d -h`).
