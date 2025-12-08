import napari

from micro_sam.sample_data import sample_data_hela_2d
from micro_sam.instance_segmentation import (
    TiledAutomaticPromptGenerator, AutomaticPromptGenerator,
    get_predictor_and_decoder, mask_data_to_segmentation
)
from micro_sam.util import precompute_image_embeddings


def example_script():
    im = sample_data_hela_2d()[0][0]

    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
    image_embeddings = precompute_image_embeddings(predictor, im, save_path="x.zarr")
    generator = AutomaticPromptGenerator(predictor, decoder)
    generator.initialize(im, image_embeddings=image_embeddings)
    segmentation = generator.generate()
    segmentation = mask_data_to_segmentation(segmentation, with_background=False)

    v = napari.Viewer()
    v.add_image(im)
    v.add_labels(segmentation)
    napari.run()


def example_script_tiled():
    im = sample_data_hela_2d()[0][0]

    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
    generator = TiledAutomaticPromptGenerator(predictor, decoder)
    generator.initialize(im, tile_shape=(256, 266), halo=(64, 64), verbose=True)
    segmentation = generator.generate()
    segmentation = mask_data_to_segmentation(segmentation, with_background=False)

    v = napari.Viewer()
    v.add_image(im)
    v.add_labels(segmentation)
    napari.run()


def main():
    example_script()
    # example_script_tiled()


if __name__ == "__main__":
    main()
