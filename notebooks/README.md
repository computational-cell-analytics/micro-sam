# Example Notebooks

We provide four example notebooks that demonstrate how the `micro_sam` python library can be used for:
- Automatic instance segmentation with minimal effort, using a `micro-sam2` model, in the [`quick_start.ipynb`](./quick_start.ipynb) notebook.
- Automatic instance segmentation with more configuration options, comparing the automatic segmentation methods of `micro-sam2` (AIS, APG) and SAM2 (AMG) in 2d and 3d, in the [`automatic_segmentation.ipynb`](./automatic_segmentation.ipynb) notebook.
- Applying Segment Anything 2 (SAM2) and `micro-sam2` models to many images, for automatic and interactive segmentation, and evaluating the segmentation quality against ground-truth segmentations in [`inference_and_evaluation.ipynb`](./inference_and_evaluation.ipynb).
- Fine-tuning SAM on your custom data to obtain a better model for your segmentation tasks in [`sam_finetuning.ipynb`](./sam_finetuning.ipynb).

If you have an environment with `micro_sam` you can directly run these notebooks. You can also run them on [Kaggle Notebooks](https://www.kaggle.com/code); we install the correct dependencies for it in each notebook.
