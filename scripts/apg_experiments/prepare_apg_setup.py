from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.instance_segmentation import AutomaticPromptGenerator, get_predictor_and_decoder
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def test_apg_on_livecell():
    from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths
    image_paths, label_paths = get_livecell_paths(
        # path="/mnt/vast-nhr/projects/cidas/cca/data/livecell",
        path="/home/anwai/data/livecell",
        split="test",
        download=True,
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
        # instances_ais = automatic_instance_segmentation(
        #     predictor=predictor_ais, segmenter=segmenter_ais, input_path=image, ndim=2, verbose=False,
        # )
        # ais_scores.append(mean_segmentation_accuracy(instances_ais, labels))

        # Run APG
        segmenter_apg.initialize(image)
        instances_apg = segmenter_apg.generate(
            min_size=25,
            foreground_threshold=0.5,
            min_distance=5,
            threshold_abs=0.25,
            prompt_selection=["center_distances", "boundary_distances", "connected_components"],
            multimasking=False,
            batch_size=32,

        )
        apg_scores.append(mean_segmentation_accuracy(instances_apg, labels))

    # Calculate mean over all images.
    # ais_msa = np.mean(ais_scores)
    # print(ais_msa)

    apg_msa = np.mean(apg_scores)
    print(apg_msa)


def main():
    test_apg_on_livecell


if __name__ == "__main__":
    main()
