import os
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy, matching

from micro_sam.util import _to_image
from micro_sam.instance_segmentation import AutomaticPromptGenerator, get_predictor_and_decoder
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from tukra.inference.get_cellpose import segment_using_cellpose

from util import get_image_label_paths


def run_baseline_engine(image, method, **kwargs):
    # OG SAM methods.
    if method in ["ais", "amg"]:
        segmentation = automatic_instance_segmentation(input_path=image, verbose=False, ndim=2, **kwargs)
    elif method == "apg":
        segmenter = kwargs["segmenter"]
        segmenter.initialize(image, ndim=2)
        segmentation = segmenter.generate(
            # refine_with_box_prompts=True,
        )

    # Newer SAM methods.
    elif method == "sam3":  # TODO: Wrap this out in a modular function too?
        processor = kwargs["processor"]
        # Set the image to the processor
        inference_state = processor.set_image(Image.fromarray(_to_image(image)))
        # Prompt the model with text
        processor.reset_all_prompts(inference_state)
        segmentation = processor.set_text_prompt(state=inference_state, prompt=kwargs["prompt"])
        segmentation = segmentation["masks"]  # Get the masks only.

        if len(segmentation) == 0:  # Handles when no objects are segmented.
            segmentation = np.zeros(image.shape[:2], dtype="uint32")
        else:  # HACK: Let's get a cheap merging strategy
            segmentation = segmentation.squeeze(1).detach().cpu().numpy()
            final_mask = np.zeros(image.shape[:2], dtype="uint32")
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
    else:
        raise ValueError

    return segmentation


def run_default_baselines(dataset_name, method, model_type, experiment_folder, target=None):
    # Prepare the results folder.
    res_folder = "./results"  # HACK: I'll store stuff in the cwd for now.
    inference_folder = os.path.join(experiment_folder, "inference", f"{dataset_name}_{method}_{model_type}")
    os.makedirs(res_folder, exist_ok=True)
    os.makedirs(inference_folder, exist_ok=True)

    fnext = (target if model_type == "sam3" else model_type)
    csv_path = os.path.join(res_folder, f"{dataset_name}_{method}_{fnext}.csv")
    if os.path.exists(csv_path):
        print(pd.read_csv(csv_path))
        print(f"The results are computed and stored at '{csv_path}'.")
        return

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
        from micro_sam3.util import get_sam3_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = get_sam3_model(input_type="image")
        kwargs["processor"] = Sam3Processor(model)
        kwargs["prompt"] = target

    msas, sa50s, precisions, recalls, f1s = [], [], [], [], []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths),
        desc=f"Run '{method}' baseline for '{model_type}' on '{dataset_name}'",
    ):
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        # Run the baseline method.
        segmentation = run_baseline_engine(image, method, **kwargs)

        # Evalate results.
        msa, sas = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
        stats = matching(segmentation, labels)

        # Store segmentations
        fname = os.path.join(inference_folder, f"{Path(curr_image_path).stem}.tif")
        imageio.imwrite(fname, segmentation, compression="zlib")

        msas.append(msa)
        sa50s.append(sas[0])
        precisions.append(stats["precision"])
        recalls.append(stats["recall"])
        f1s.append(stats["f1"])

    results = {
        "mSA": np.mean(msas),
        "SA50": np.mean(sa50s),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1": np.mean(f1s),
    }
    results = pd.DataFrame.from_dict([results])
    results.to_csv(csv_path)
    print(results)
    print(f"The results above are stored at '{csv_path}'.")

    # HACK: Force remove the inference folder
    import shutil
    shutil.rmtree(experiment_folder)


def main(args):
    run_default_baselines(args.dataset_name, args.method, args.model_type, args.experiment_folder, args.target)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("--target", type=str, default=None)  # We need this for InstanSeg and SAM3.
    parser.add_argument(
        "--experiment_folder", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam/apg_baselines/cc_without_box"  # noqa
    )
    args = parser.parse_args()
    main(args)
