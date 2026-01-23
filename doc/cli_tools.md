# Using the Command Line Interface (CLI)

`micro-sam` extends access to a bunch of functionalities using the command line interface (CLI) scripts via terminal.

The supported CLIs can be used by
- Running `$ micro_sam.info` for getting `micro-sam` package information and other necessary system-level information to use `micro-sam`.
- Running `$ micro_sam.precompute_embeddings` for precomputing and caching the image embeddings.
- Running `$ micro_sam.annotator_2d` for starting the 2d annotator.
- Running `$ micro_sam.annotator_3d` for starting the 3d annotator.
- Running `$ micro_sam.annotator_tracking` for starting the tracking annotator.
- Running `$ micro_sam.image_series_annotator` for starting the image series annotator.
- Running `$ micro_sam.train` for finetuning Segment Anything models on your data.
- Running `$ micro_sam.automatic_segmentation` for automatic instance segmentation.
    - We support all post-processing parameters for automatic instance segmentation (for AMG, AIS and APG).
        - The automatic segmentation mode can be controlled by: `--mode <MODE_NAME>`, where the available choice for `MODE_NAME` is `amg` / `ais` / `apg`.
        - AMG is supported by both default Segment Anything models and `micro-sam` models / finetuned models.
        - AIS is supported by `micro-sam` models (or finetuned models; subjected to they are trained with the additional instance segmentation decoder)
        - APG is supported by `micro-sam` models (or finetuned models; subjected to they are trained with the additional instance segmentation decoder)
    - If these parameters are not provided by the user, `micro-sam` makes use of the best post-processing parameters (depending on the choice of model). 
    - The post-processing parameters can be changed by parsing the parameters via the CLI using `--<PARAMETER_NAME> <VALUE>.` For example, one can update the parameter values (eg. `pred_iou_thresh`, `stability_iou_thresh`, etc. - supported by AMG) using `$ micro_sam.automatic_segmentation ... --pred_iou_thresh 0.6 --stability_iou_thresh 0.6 ...`
        - Remember to specify the automatic segmentation mode using `--mode <MODE_NAME>` when using additional post-processing parameters.
    - You can check details for supported parameters and their respective default values at `micro_sam/instance_segmentation.py` under the `generate` method for `AutomaticMaskGenerator`, `InstanceSegmentationWithDecoder` and `AutomaticPromptGenerator` class.
    - A good practice is to set `--ndim <NDIM>`, where `<NDIM>` corresponds to the number of dimensions of input images.
- Running `$ micro_sam.evaluate` for evaluating instance segmentation.

NOTE: For all CLIs above, you can find more details by adding the argument `-h` to the CLI script (eg. `$ micro_sam.annotator_2d -h`).
