import os

import imageio.v3 as imageio

from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_livecell_example_data, fetch_wholeslide_example_data, fetch_3d_example_data

from elf.io import open_file


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def livecell_pixel_classifier():
    from micro_sam.sam_annotator.pixel_classifier import pixel_classifier

    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-livecell-vit_b_lm.zarr")
    model_type = "vit_b_lm"

    pixel_classifier(image, embedding_path=embedding_path, model_type=model_type)


def wholeslide_pixel_classifier():
    from micro_sam.sam_annotator.pixel_classifier import pixel_classifier

    example_data = fetch_wholeslide_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "whole-slide-embeddings-vit_b_lm.zarr")
    model_type = "vit_b_lm"

    pixel_classifier(
        image, embedding_path=embedding_path, model_type=model_type,
        tile_shape=(1024, 1024), halo=(256, 256),
    )


def lucchi_pixel_classifier():
    from micro_sam.sam_annotator.pixel_classifier import pixel_classifier

    example_data = fetch_3d_example_data(DATA_CACHE)
    with open_file(example_data) as f:
        raw = f["*.png"][:]

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-lucchi-vit_b_em_organelles.zarr")
    model_type = "vit_b_em_organelles"

    pixel_classifier(raw, embedding_path=embedding_path, model_type=model_type)


def tiled_3d_pixel_classifier():
    from micro_sam.sam_annotator.pixel_classifier import pixel_classifier
    from skimage.data import cells3d

    data = cells3d()[30:34, 1]
    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-cells3d-tiled.zarr")
    model_type = "vit_b_lm"

    pixel_classifier(
        data, embedding_path=embedding_path, model_type=model_type,
        tile_shape=(128, 128), halo=(32, 32),
    )


def dino_pixel_classifier():
    from micro_sam.sam_annotator.pixel_classifier import pixel_classifier

    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    # DINOv2 encoder weights download automatically via torch.hub. DINOv3 ('dino_v3_*') weights are
    # license-gated; supply the emailed URL or local path via the MICROSAM_DINOV3_WEIGHTS env var.
    pixel_classifier(image, model_type="dino_v2_vitb")


def main():
    livecell_pixel_classifier()
    # wholeslide_pixel_classifier()
    # lucchi_pixel_classifier()
    # tiled_3d_pixel_classifier()
    # dino_pixel_classifier()


if __name__ == "__main__":
    main()
