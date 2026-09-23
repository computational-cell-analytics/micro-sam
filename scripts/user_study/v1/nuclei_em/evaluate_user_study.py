import os
import pandas as pd

import h5py
import imageio.v3 as imageio
import numpy as np

from elf.evaluation import mean_segmentation_accuracy


DATA_ROOT = "/scratch-grete/projects/nim00007/sam/user_study/nuclei_em/v2"

FILE_NAMES = {
    "default": ["annotations_default.tif"],
    "finetuned": ["annotations_finetuned.tif"],
    "ilastik": ["annotations_ilastik.h5"],
}

# Annotation times.
ANNOTATION_TIMES = {
    "default": [63],
    "finetuned": [24],
    "ilastik": [73],
}


def load_seg(path):
    if path.endswith(".tif"):
        seg = imageio.imread(path)
    else:
        with h5py.File(path, "r") as f:
            seg = f["exported_data"][:].squeeze()
    return seg


def compute_metrics(gt, seg):
    msa, scores = mean_segmentation_accuracy(seg, gt, return_accuracies=True)
    sa50, sa75 = scores[0], scores[5]
    return msa, sa50, sa75


def evaluate_annotation_times():
    method = []
    time_per_object = []
    total_objects_ = []
    total_time_ = []

    sa50s = []

    gt_file = os.path.join(DATA_ROOT, "annotation_volume.h5")
    with h5py.File(gt_file, "r") as f:
        gt = f["labels/nuclei"][:]

    for name in FILE_NAMES:

        total_objects = 0
        total_time = 0.0
        for fname, time in zip(FILE_NAMES[name], ANNOTATION_TIMES[name]):
            seg = load_seg(os.path.join(DATA_ROOT, fname))
            n_objects = len(np.unique(seg)) - 1
            total_time += time
            total_objects += n_objects
            msa, sa50, sa75 = compute_metrics(gt, seg)

        method.append(name)
        total_time_.append(total_time)
        total_objects_.append(total_objects)
        time_per_object.append(float(total_time) / total_objects)
        sa50s.append(sa50)

    table = pd.DataFrame(
        {
            "annotation_method": method,
            "time_per_object [min]": np.round(time_per_object, 1),
            "total_time": total_time_,
            "total_objects": total_objects_,
            # "sa50": sa50s,
        }
    )
    print(table)
    table.to_csv("../results_em_nuclei.csv", index=False)


def main():
    evaluate_annotation_times()


if __name__ == "__main__":
    main()
