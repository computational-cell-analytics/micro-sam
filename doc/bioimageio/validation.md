# Validating SAM Models for Microscopy

To validate a segment anything model for your data you have different options, depending on the task you want to solve and whether you have segmentation annotations for your data.

If you don't have any annotations you will have to validate the model visually. We suggest doing this with the `micro_sam` GUI tools.
You can learn how to use them in [the micro_sam documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html).

If you have segmentation annotations you can use the `micro_sam` library to evaluate the segmentation quality of different SAM models.
We provide functionality to evaluate the models for interactive and for automatic segmentation:
- You can use [run_inference_with_iterative_prompting](https://computational-cell-analytics.github.io/micro-sam/micro_sam/evaluation/inference.html#run_inference_with_iterative_prompting) to evaluate models for interactive segmentation.
- You can use [run_instance_segmentation_grid_search_and_inference](https://computational-cell-analytics.github.io/micro-sam/micro_sam/evaluation/instance_segmentation.html#run_instance_segmentation_grid_search_and_inference) to evaluate models for automatic segmentation.

We provide [an example notebook](https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/automatic_segmentation.ipynb) that shows how to use this evaluation functionality.
