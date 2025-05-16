import os
import imageio.v3 as imageio

from micro_sam.sam_annotator.object_classifier import object_classifier
from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_livecell_example_data

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def livecell_annotator():
    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-livecell-vit_b_lm.zarr")
    model_type = "vit_b_lm"

    # This is the vit-b-lm segmentation
    segmentation = imageio.imread("./livecell-test-seg.tif")

    object_classifier(image, segmentation, embedding_path=embedding_path, model_type=model_type)


def annotator_devel():
    import micro_sam.sam_annotator.object_classifier as clf
    from micro_sam.util import precompute_image_embeddings, get_sam_model

    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-livecell-vit_b_lm.zarr")
    model_type = "vit_b_lm"

    # This is the vit-b-lm segmentation and a test annotaiton.
    segmentation = imageio.imread("./livecell-test-seg.tif")
    annotations = imageio.imread("./livecell-test-annotations.tif")

    predictor = get_sam_model(model_type)
    image_embeddings = precompute_image_embeddings(predictor, image, save_path=embedding_path)
    seg_ids, features = clf._compute_object_features(image_embeddings, segmentation)
    labels = clf._accumulate_labels(segmentation, annotations)
    rf = clf._train_rf(features, labels)
    object_prediction = clf._predict_rf(rf, features, seg_ids)
    prediction = clf._project_prediction(segmentation, object_prediction)

    import napari
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(annotations)
    v.add_labels(prediction)
    napari.run()


livecell_annotator()
# annotator_devel()
