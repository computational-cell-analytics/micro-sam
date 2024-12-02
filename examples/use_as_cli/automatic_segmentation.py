import os
import itertools
import subprocess
from tqdm import tqdm
from pathlib import Path

import numpy as np


def run_automatic_segmentation(args):
    # Run grid-search for AMG (similar can be implemented over range of AIS parameters).
    pred_iou_thresh_values = np.arange(0.5, 0.9, 0.05)  # default: 0.88
    stability_score_thresh_values = np.arange(0.5, 0.95, 0.05)  # default: 0.95
    # NOTE: You can add additional parameters. Remember to add them to the grid-search below.

    target_dir = "./amg_segmentations"  # Replace this with the target folder where you would like to store seg.
    os.makedirs(target_dir, exist_ok=True)

    counter = 0
    for pred_iou_thresh, stability_score_thresh in tqdm(
        itertools.product(pred_iou_thresh_values, stability_score_thresh_values),
        desc="Grid-search over parameters for AMG",
    ):
        script = ["micro_sam.automatic_segmentation"]
        script.extend(args)  # Add additionally parsed arguments

        if len(args) == 0:
            raise ValueError(
                "You did not provide required arguments to run automatic segmentation CLI. "
                "See 'micro_sam.automatic_segmentation -h' for details."
            )

        idx = args.index("-i") + 1  # We need to find the index for the image filepath
        image_path = args[idx]
        if not os.path.exists(image_path):
            raise AssertionError(f"The image filepath does not exist at {image_path}")

        script.extend(["-o", os.path.join(target_dir, f"{Path(image_path).stem}_{counter}.tif")])  # Specify a seg path.
        script.extend(
            ["--pred_iou_thresh", f"{pred_iou_thresh}", "--stability_score_thresh", f"{stability_score_thresh}"]
        )

        subprocess.run(script)
        counter += 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    _, args = parser.parse_known_args()
    run_automatic_segmentation(args)


if __name__ == "__main__":
    main()
