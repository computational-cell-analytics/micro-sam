from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

import napari

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.sample_data import sample_data_hela_2d
from micro_sam.instance_segmentation import AutomaticPromptGenerator, get_predictor_and_decoder
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


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


def apg_test_livecell():
    from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths
    image_paths, label_paths = get_livecell_paths(
        path="/mnt/vast-nhr/projects/cidas/cca/data/livecell", split="test",
    )

    # Prepare AIS segmenter.
    predictor_ais, segmenter_ais = get_predictor_and_segmenter(model_type="vit_b_lm")

    # Prepare APG segmenter.
    predictor_apg, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
    segmenter_apg = AutomaticPromptGenerator(predictor_apg, decoder)

    # Run automatic segmentation methods per image.
    ais_scores, apg_scores = [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Run segmentation",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run AIS
        instances_ais = automatic_instance_segmentation(
            predictor=predictor_ais, segmenter=segmenter_ais, input_path=image, ndim=2, verbose=False,
        )

        # Run APG
        segmenter_apg.initialize(image)
        instances_apg = segmenter_apg.generate()

        # Evaluate both segmentations
        ais_scores.append(mean_segmentation_accuracy(instances_ais, labels))
        apg_scores.append(mean_segmentation_accuracy(instances_apg, labels))

    # Calculate mean over all images.
    ais_msa = np.mean(ais_scores)
    apg_msa = np.mean(apg_scores)

    print(ais_msa, apg_msa)


def main():
    # example_script()

    apg_test_livecell()


if __name__ == "__main__":
    main()
