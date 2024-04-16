import os
from glob import glob
from pathlib import Path

import pandas as pd
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy


CELLPOSE_ROOT = "/scratch/projects/nim00007/sam/experiments/benchmarking/cellpose/livecell/livecell/predictions/"
AMG_ROOT = ""
AIS_ROOT = ""

LIVECELL_ROOT = ""


def compare_cellpose_vs_ais():
    all_livecell_gt = sorted(glob(os.path.join(LIVECELL_ROOT, "*.tif")))

    all_scores = []
    for gt_path in all_livecell_gt:
        image_id = Path(gt_path).stem

        # cellpose_seg = imageio.imread(os.path.join(CELLPOSE_ROOT, f"{image_id}.tif"))
        ais_seg = imageio.imread(os.path.join(AIS_ROOT, f"{image_id}.tif"))
        # amg_seg = imageio.imread(os.path.join(AMG_ROOT, f"{image_id}.tif"))
        gt = imageio.imread(gt_path)

        score = {
            "image": image_id,
            "msa": mean_segmentation_accuracy(ais_seg, gt)
        }
        all_scores.append(pd.DataFrame.from_dict([score]))

    results = pd.concat(all_scores, ignore_index=True)

    # once this is done, we check the n best cases and compare them with amg and ais


def compare_covid_if_cellpose_vs_ais():
    


def main():
    compare_cellpose_vs_ais()


if __name__ == "__main__":
    main()
