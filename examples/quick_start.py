"""A quick start to automatic instance segmentation with the micro-sam2 models.

NOTE: You can install `micro_sam` in a conda environment like: `conda install -c conda-forge micro_sam`
"""

import napari
from skimage.data import cells3d
from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

# Load an example image from the 'scikit-image' library.
image = cells3d()[30, 0]

# Load the micro-sam2 model for light microscopy, with its decoder for automatic segmentation.
predictor, segmenter = get_predictor_and_segmenter(model_type="hvit_t_cells")

# Run automatic instance segmentation (AIS) on the image.
instances = automatic_instance_segmentation(predictor=predictor, segmenter=segmenter, input_path=image)

# Show the image and the instance segmentation.
v = napari.Viewer()
v.add_image(image, name="Image")
v.add_labels(instances, name="Prediction")
napari.run()
