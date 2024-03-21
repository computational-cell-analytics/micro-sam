import os
import imageio.v3 as imageio

import numpy as np
import pandas as pd


def _evaluate(result_folders, time_results):
    total_times = []
    n_objects = []
    time_per_object = []

    for model, result_folder in result_folders.items():
        images = time_results["Image Name"].values
        timings = time_results[model].values
        n_object = 0
        time = 0.0

        for im, tim in zip(images, timings):
            if isinstance(tim, float) and np.isnan(tim):
                continue
            # Datetime thinks seconds are minutes ...
            assert tim.second == 0
            n_sec = tim.hour * 60 + tim.minute

            seg_path = os.path.join(result_folder, im.rstrip("'"))
            assert os.path.exists(seg_path), seg_path
            seg = imageio.imread(seg_path)
            this_objects = len(np.unique(seg)) - 1

            time += n_sec
            n_object += this_objects

        total_times.append(time)
        n_objects.append(n_object)
        time_per_object.append(time / n_object)

    result = pd.DataFrame({
        "method": list(result_folders.keys()),
        "time_per_object [s/obj]": time_per_object,
        "total_objects": n_objects,
        "total_time": total_times,
    })
    return result


def evaluate_sam():
    time_results = pd.read_excel("annotation-times-organoids-sam.xlsx")
    result_folders = {
        "Default": "/scratch-grete/projects/nim00007/data/pdo/user_study_v2/result-sam",
        "LM Generalist": "/scratch-grete/projects/nim00007/data/pdo/user_study_v2/result_finetuned_lm",
        # These are the annotations from micro-sam, despite the name.
        "Finetuned": "/scratch-grete/projects/nim00007/data/pdo/user_study_v2/user_study_test_labels",
    }
    return _evaluate(result_folders, time_results)


def evaluate_baselines():
    time_results = pd.read_excel("annotation-times-organoids-baselines.xlsx")
    result_folders = {
        "CellPose Default": "/scratch-grete/projects/nim00007/sam/user_study/organoids/v1/train_data/labels",
        "CellPose HIL": "/scratch-grete/projects/nim00007/sam/user_study/organoids/v1/train_data/labels",
        "CellPose Finetuned": "/scratch-grete/projects/nim00007/sam/user_study/organoids/v1/train_data/labels",
        "Manual": "/scratch-grete/projects/nim00007/sam/user_study/organoids/v1/train_data/labels",
    }
    return _evaluate(result_folders, time_results)


def main():
    result_sam = evaluate_sam()
    result_baseline = evaluate_baselines()
    result = pd.concat([result_sam, result_baseline])
    result.to_csv("../results_organoids.csv")


if __name__ == "__main__":
    main()
