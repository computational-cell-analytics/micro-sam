from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy, matching

from micro_sam.instance_segmentation import AutomaticPromptGenerator, get_predictor_and_decoder
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter, automatic_instance_segmentation, mask_data_to_segmentation
)

from tukra.inference.get_cellpose import segment_using_cellpose
from tukra.inference.get_instanseg import segment_using_instanseg

from util import get_image_label_paths


def run_baseline_engine(image, method, **kwargs):
    # OG SAM methods.
    if method in ["ais", "amg"]:
        segmentation = automatic_instance_segmentation(input_path=image, verbose=False, ndim=2, **kwargs)
    elif method == "apg":
        segmenter = kwargs["segmenter"]
        segmenter.initialize(image, ndim=2)
        segmentation = segmenter.generate(
            prompt_selection="boundary_distances",
        )

        if len(segmentation) == 0:
            segmentation = np.zeros(image.shape[:2], dtype="uint32")
        else:
            segmentation = mask_data_to_segmentation(segmentation, with_background=False)

    # Newer SAM methods.
    elif method == "sam2":
        # TODO: Wrap this out in a modular function (like our segmenters)
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from micro_sam2.util import get_sam2_model
        predictor = get_sam2_model(model_type=kwargs["model_type"])
        generator = SAM2AutomaticMaskGenerator(predictor)
        segmentation = generator.generate(image.astype("uint8"))  # HACK: Casting forcefully.

        if len(segmentation) == 0:
            segmentation = np.zeros(image.shape[:2], dtype="uint32")
        else:
            segmentation = mask_data_to_segmentation(segmentation, with_background=True)

    elif method == "sam3":
        # TODO: Wrap this out in a modular function too?
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        inference_state = processor.set_image(image)
        # Prompt the model with text
        segmentation = processor.set_text_prompt(state=inference_state, prompt=kwargs["prompt"])
        segmentation = segmentation["masks"]  # Get the masks only.

        if len(segmentation) == 0:
            segmentation = np.zeros(image.shape[:2], dtype="uint32")
        else:
            # HACK: Let's get a cheap merging strategy
            segmentation = segmentation.squeeze(1).detach().cpu().numpy()
            final_mask = np.zeros(image.shape, dtype="uint32")
            for i, curr_mask in enumerate(segmentation, start=1):
                final_mask[curr_mask] = i
            segmentation = final_mask

    # And external baselines.
    elif method == "cellpose":
        segmentation = segment_using_cellpose(image, kwargs["model_type"])
    elif method == "cellsam":
        from cellSAM import cellsam_pipeline
        segmentation = cellsam_pipeline(image, use_wsi=False)
        # NOTE: For images where no objects could be found, a weird segmentation is returned.
        if segmentation.ndim == 3:
            segmentation = segmentation[0]
    elif method == "instanseg":
        segmentation = segment_using_instanseg(image, verbose=False, **kwargs)
    elif method == "cellvit":
        raise NotImplementedError
    else:
        raise ValueError

    return segmentation


def run_default_baselines(dataset_name, method, model_type, target=None):
    # Get the image and label paths.
    image_paths, label_paths = get_image_label_paths(dataset_name=dataset_name, split="test")

    assert isinstance(method, str)
    kwargs = {}
    if method in ["ais", "amg"]:
        predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, amg=(method == "amg"))
        kwargs["predictor"] = predictor
        kwargs["segmenter"] = segmenter
    elif method == "apg":
        predictor, decoder = get_predictor_and_decoder(model_type=model_type)
        segmenter = AutomaticPromptGenerator(predictor, decoder)
        kwargs["predictor"] = predictor
        kwargs["segmenter"] = segmenter
    elif method == "cellpose":
        kwargs["model_type"] = model_type
    elif method == "instanseg":
        kwargs["model_type"] = model_type
        kwargs["target"] = target
    elif method == "sam2":
        kwargs["model_type"] = model_type
    elif method == "sam3":
        kwargs["prompt"] = target

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc=f"Run '{method}' baseline for '{model_type}'",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run the baseline method.
        segmentation = run_baseline_engine(image, method, **kwargs)

        # Evalate results.
        msa, sas = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
        stats = matching(segmentation, labels)

        msas.append(msa)
        sa50s.append(sas[0])
        precisions.append(stats["precision"])
        recalls.append(stats["recall"])
        f1s.append(stats["f1"])

    print(
        f"The final scores for '{method}' with '{model_type}' on '{dataset_name}' are - mSA:",
        np.mean(msas), "SA50:",  np.mean(sa50s), "Precision:", np.mean(precisions), "Recall:",
        np.mean(recalls), "F1 Score:", np.mean(f1s)
    )


def main(args):
    run_default_baselines(args.dataset_name, args.method, args.model_type, args.target)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("--target", type=str, default=None)  # We need this for InstanSeg and SAM3.
    args = parser.parse_args()
    main(args)
