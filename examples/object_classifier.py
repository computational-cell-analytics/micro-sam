import os

import imageio.v3 as imageio
import numpy as np

from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_livecell_example_data, fetch_wholeslide_example_data, fetch_3d_example_data

from elf.io import open_file


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def livecell_annotator():
    from micro_sam.sam_annotator.object_classifier import object_classifier

    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-livecell-vit_b_lm.zarr")
    model_type = "vit_b_lm"

    # This is the vit-b-lm segmentation
    segmentation = imageio.imread("./clf-test-data/livecell-test-seg.tif")

    object_classifier(image, segmentation, embedding_path=embedding_path, model_type=model_type)


def wholeslide_annotator():
    from micro_sam.sam_annotator.object_classifier import object_classifier

    example_data = fetch_wholeslide_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "whole-slide-embeddings-vit_b_lm.zarr")
    model_type = "vit_b_lm"

    segmentation = imageio.imread("./clf-test-data/whole-slide-seg.tif")
    object_classifier(
        image, segmentation, embedding_path=embedding_path, model_type=model_type,
        tile_shape=(1024, 1024), halo=(256, 256),
    )


def lucchi_annotator():
    from micro_sam.sam_annotator.object_classifier import object_classifier

    example_data = fetch_3d_example_data(DATA_CACHE)
    with open_file(example_data) as f:
        raw = f["*.png"][:]

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-lucchi-vit_b_em_organelles.zarr")

    model_type = "vit_b_lm"
    segmentation = imageio.imread("./clf-test-data/lucchi-test-segmentation.tif")

    object_classifier(raw, segmentation, embedding_path=embedding_path, model_type=model_type)


def tiled_3d_annotator():
    from micro_sam.sam_annotator.object_classifier import object_classifier
    from skimage.data import cells3d

    data = cells3d()[30:34, 1]
    embed_path = "./clf-test-data/emebds-3d-tiled.zarr"

    model_type = "vit_b_lm"
    segmentation = imageio.imread("./clf-test-data/tiled-3d-segmentation.tif")

    object_classifier(
        data, segmentation, embedding_path=embed_path, model_type=model_type,
        tile_shape=(128, 128), halo=(32, 32)
    )


def _get_livecell_data():
    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-livecell-vit_b_lm.zarr")

    # This is the vit-b-lm segmentation and a test annotaiton.
    segmentation = imageio.imread("./clf-test-data/livecell-test-seg.tif")
    annotations = imageio.imread("./clf-test-data/livecell-test-annotations.tif")

    model_type = "vit_b_lm"

    return image, segmentation, annotations, model_type, embedding_path, None, None


def _get_wholeslide_data():
    example_data = fetch_wholeslide_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "whole-slide-embeddings-vit_b_lm.zarr")

    # This is the vit-b-lm segmentation and a test annotaiton.
    segmentation = imageio.imread("./clf-test-data/whole-slide-seg.tif")
    annotations = imageio.imread("./clf-test-data/wholeslide-annotations.tif")

    model_type = "vit_b_lm"

    return image, segmentation, annotations, model_type, embedding_path, (1024, 1024), (256, 256)


def _get_lucchi_data():
    example_data = fetch_3d_example_data(DATA_CACHE)
    with open_file(example_data) as f:
        raw = f["*.png"][:]

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-lucchi-vit_b_em_organelles.zarr")

    segmentation = imageio.imread("./clf-test-data/lucchi-test-segmentation.tif")
    annotations = imageio.imread("./clf-test-data/lucchi-annotations.tif")

    model_type = "vit_b_em_organelles"

    return raw, segmentation, annotations, model_type, embedding_path, None, None


def _get_3d_tiled_data():
    from skimage.data import cells3d

    data = cells3d()[30:34, 1]
    embed_path = "./clf-test-data/emebds-3d-tiled.zarr"
    model_type = "vit_b_lm"

    segmentation = imageio.imread("./clf-test-data/tiled-3d-segmentation.tif")
    annotations = imageio.imread("./clf-test-data/tiled-3d-annotations.tif")

    return data, segmentation, annotations, model_type, embed_path, (128, 128), (32, 32)


def annotator_devel():
    from micro_sam.sam_annotator import object_classifier as clf
    from micro_sam.util import precompute_image_embeddings, get_sam_model

    # image, segmentation, annotations, model_type, embedding_path, tile_shape, halo = _get_livecell_data()
    # image, segmentation, annotations, model_type, embedding_path, tile_shape, halo = _get_wholeslide_data()
    # image, segmentation, annotations, model_type, embedding_path, tile_shape, halo = _get_lucchi_data()
    image, segmentation, annotations, model_type, embedding_path, tile_shape, halo = _get_3d_tiled_data()

    predictor = get_sam_model(model_type)
    image_embeddings = precompute_image_embeddings(
        predictor, image, save_path=embedding_path, tile_shape=tile_shape, halo=halo
    )
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


def create_3d_data_with_tiling():
    from skimage.data import cells3d
    from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

    predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_lm", is_tiled=True)
    data = cells3d()[30:34, 1]

    embed_path = "./clf-test-data/emebds-3d-tiled.zarr"
    seg = automatic_instance_segmentation(
        predictor, segmenter, data, embedding_path=embed_path, ndim=3, tile_shape=(128, 128), halo=(32, 32)
    )

    import napari
    v = napari.Viewer()
    v.add_image(data)
    v.add_labels(seg)
    # For annotations.
    v.add_labels(np.zeros_like(seg))
    napari.run()


def histopathology_annotator():
    from torch_em.data.datasets.histopathology.lynsec import get_lynsec_paths
    from micro_sam.sam_annotator import object_classifier as clf
    from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

    predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_histopathology")

    image_paths, _ = get_lynsec_paths(path="./clf-test-data/nuclick", choice="ihc", download=True)
    image_paths = image_paths[:10]

    images, segmentations = [], []
    embedding_paths = []

    for i, image_path in enumerate(image_paths):
        image = imageio.imread(image_path)
        embedding_path = f"./clf-test-data/embeddings_nuclick_{i}.zarr"
        seg_path = f"./clf-test-data/seg-nuclick_{i}.tif"

        if os.path.exists(seg_path):
            segmentation = imageio.imread(seg_path)
        else:
            segmentation = automatic_instance_segmentation(
                predictor, segmenter, embedding_path=embedding_path, input_path=image, ndim=2,
            )
            imageio.imwrite(seg_path, segmentation, compression="zlib")

        images.append(image)
        segmentations.append(segmentation)
        embedding_paths.append(embedding_path)

    clf.image_series_object_classifier(
        images, segmentations, output_folder="./clf-test-data/histo-results",
        embedding_paths=embedding_paths, model_type="vit_b_histopathology", ndim=2,
    )


def batch_prediction():
    import napari
    from torch_em.data.datasets.histopathology.lynsec import get_lynsec_paths
    from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
    from micro_sam.object_classification import run_prediction_with_object_classifier
    from tqdm import tqdm

    predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_histopathology")

    image_paths, _ = get_lynsec_paths(path="./clf-test-data/nuclick", choice="ihc", download=True)
    # Test the batch prediction on the next 5 images.
    image_paths = image_paths[10:12]

    images, segmentations = [], []
    # Prepare images and segmentations
    for image_path in tqdm(image_paths, desc="Segment images"):
        image = imageio.imread(image_path)
        segmentation = automatic_instance_segmentation(predictor, segmenter, input_path=image, ndim=2, verbose=False)
        images.append(image)
        segmentations.append(segmentation)

    rf_path = "clf-test-data/histo-results/rf.joblib"
    print("Start object clf")
    predictions = run_prediction_with_object_classifier(images, segmentations, predictor, rf_path, ndim=2)

    for im, seg, pred in zip(images, segmentations, predictions):
        v = napari.Viewer()
        v.add_image(im)
        v.add_labels(seg)
        v.add_labels(pred)
        napari.run()


def main():
    # create_3d_data_with_tiling()

    # livecell_annotator()
    # wholeslide_annotator()
    # lucchi_annotator()
    # tiled_3d_annotator()
    histopathology_annotator()
    # batch_prediction()

    # annotator_devel()


if __name__ == "__main__":
    main()
