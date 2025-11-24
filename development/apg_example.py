import napari

from micro_sam.sample_data import sample_data_hela_2d
from micro_sam.instance_segmentation import AutomaticPromptGenerator, get_predictor_and_decoder


def example_script():
    im = sample_data_hela_2d()[0][0]

    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
    generator = AutomaticPromptGenerator(predictor, decoder)
    generator.initialize(im)
    segmentation = generator.generate()

    v = napari.Viewer()
    v.add_image(im)
    v.add_labels(segmentation)
    napari.run()


def main():
    example_script()


if __name__ == "__main__":
    main()
