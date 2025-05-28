"""This script is a quick start to the automatic instance segmentation feature
supported by the Segment Anything for Microscopy models.

NOTE: You can install `micro_sam` in a conda environment like: `conda install -c conda-forge micro_sam`
"""

import napari

from skimage.data import cells3d

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


# Load an example image from the 'scikit-image' library.
image = cells3d()[30, 0]

# Load the Segment Anything for Microscopy model.
predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_lm")

# Run automatic instance segmentation (AIS) on our image.
instances = automatic_instance_segmentation(predictor=predictor, segmenter=segmenter, input_path=image)

# Visualize the image and corresponding instance segmentation result.
v = napari.Viewer()
v.add_image(image, name="Image")
v.add_labels(instances, name="Prediction")
napari.run()
