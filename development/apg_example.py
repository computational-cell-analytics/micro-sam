import os
import time

import h5py
import napari

from micro_sam.sample_data import sample_data_hela_2d
from micro_sam.instance_segmentation import (
    TiledAutomaticPromptGenerator, AutomaticPromptGenerator, get_predictor_and_decoder
)
from micro_sam.util import precompute_image_embeddings


# TODO example with a custom prompt function
def example_script():
    im = sample_data_hela_2d()[0][0]

    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
    image_embeddings = precompute_image_embeddings(predictor, im, save_path="x.zarr")
    generator = AutomaticPromptGenerator(predictor, decoder)
    generator.initialize(im, image_embeddings=image_embeddings)
    segmentation = generator.generate(intersection_over_min=True)

    v = napari.Viewer()
    v.add_image(im)
    v.add_labels(segmentation)
    napari.run()


def example_script_tiled():
    im = sample_data_hela_2d()[0][0]

    tile_shape, halo = (256, 256), (64, 64)
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
    image_embeddings = precompute_image_embeddings(predictor, im, tile_shape=tile_shape, halo=halo, save_path="y.zarr")
    generator = TiledAutomaticPromptGenerator(predictor, decoder)
    generator.initialize(im, image_embeddings=image_embeddings, tile_shape=tile_shape, halo=halo, verbose=True)
    segmentation = generator.generate(intersection_over_min=False)

    v = napari.Viewer()
    v.add_image(im)
    v.add_labels(segmentation)
    napari.run()


def _require_wsi_data():
    out_path = "./data/wsi.h5"
    if os.path.exists(out_path):
        return out_path

    from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data
    from patho_sam.io.util import read_wsi

    example_data = fetch_wholeslide_histopathology_example_data("./data")
    data = read_wsi(example_data)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")

    os.remove(example_data)
    return out_path


def example_script_wsi():
    tile_shape, halo = (768, 768), (64, 64)
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_histopathology")

    data_path = _require_wsi_data()
    with h5py.File(data_path, "r") as f:
        data = f["data"]
        print("Run prediction for WSI of shape:", data.shape)

        # Processing time: 10:34 min (batch size 24 on an A100 with 80 GB)
        embed_path = "./data/embeds.zarr"
        image_embeddings = precompute_image_embeddings(
            predictor, data, tile_shape=tile_shape, halo=halo, save_path=embed_path, batch_size=24, ndim=2
        )

        # Processing time: 02:14 min (batch size 24 on an A100 with 80 GB)
        generator = TiledAutomaticPromptGenerator(predictor, decoder)
        generator.initialize(
            data, image_embeddings=image_embeddings, tile_shape=tile_shape, halo=halo, verbose=True, batch_size=24
        )

        print("Start generate ...")
        t0 = time.time()
        seg = generator.generate(batch_size=32)
        print("Generate took:", time.time() - t0, "s")
        print(seg.shape)


def main():
    # example_script()
    # example_script_tiled()
    example_script_wsi()


if __name__ == "__main__":
    main()
