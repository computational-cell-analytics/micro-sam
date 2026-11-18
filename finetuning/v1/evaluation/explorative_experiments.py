import os
import numpy as np
from tqdm import tqdm

import imageio.v3 as imageio
import matplotlib.pyplot as plt

from elf.evaluation import mean_segmentation_accuracy

from util import get_paths, EXPERIMENT_ROOT


def explore_differences(
    dataset_name,
    modality,
    model_type,
    compare_experiments,
    all_settings,
    n_images,
    metric_choice="msa",
    sort_decreasing=True
):
    image_paths, gt_paths = get_paths(dataset_name, split="test")

    assert len(compare_experiments) == 2, "You should provide only two experiment names to compare."
    assert metric_choice in ["msa", "sa50"], "The metric choice is limited to `msa` / `sa50`."

    per_experiment_res = []
    compare_experiments = sorted(compare_experiments)
    all_settings = sorted(all_settings)
    for experiment in compare_experiments:
        experiment_dir = os.path.join(EXPERIMENT_ROOT, experiment, modality, dataset_name, model_type)
        assert os.path.exists(experiment_dir), f"The experiment `{experiment}` does not exist for {dataset_name}"

        print(f"Running experiments for {experiment}...")
        per_setting_res = {}
        for setting in all_settings:
            setting_split = setting.split("/")
            setting_name = setting_split[0] + "_" + setting_split[1]

            res = []
            for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths), desc=setting_name):
                image_id = os.path.split(image_path)[-1]

                prediction_path = os.path.join(experiment_dir, setting, image_id)

                prediction = imageio.imread(prediction_path)
                gt = imageio.imread(gt_path)

                msa, sa_acc = mean_segmentation_accuracy(prediction, gt, return_accuracies=True)
                res.append(msa if metric_choice == "msa" else sa_acc[0])

            per_setting_res[setting_name] = res

        per_experiment_res.append({"name": experiment, "results": per_setting_res})

    experiment1, experiment2 = per_experiment_res

    # now let's calculate the difference between experiments and identify the ones with biggest diff.
    name = f"{compare_experiments[0]}-{compare_experiments[1]}"
    experiment_res = {}
    for (k1, v1), (k2, v2) in zip(experiment1.items(), experiment2.items()):
        if k1 == k2 == "results":
            setting_res = {}
            for (res_k1, res_v1), (res_k2, res_v2) in zip(v1.items(), v2.items()):
                assert res_k1 == res_k2
                res = np.subtract(res_v1, res_v2)
                setting_res[f"{res_k1}"] = list(zip(abs(res).tolist(), res_v1, res_v2))
            experiment_res[name] = setting_res

    image_ids = [os.path.split(image_path)[-1] for image_path in image_paths]
    plot_samples(
        name, modality, dataset_name, model_type, all_settings,
        experiment_res, image_ids, n_images, metric_choice, sort_decreasing
    )


def plot_samples(
    name, modality, dataset_name, model_type, all_settings,
    experiment_res, image_ids, n_images, metric_choice, sort_decreasing
):
    check_samples = {}
    for experiment_name, experiment_value in experiment_res[name].items():
        desired_results = [
            (x, y[1], y[2]) for y, x in sorted(zip(experiment_value, image_ids), reverse=sort_decreasing)
        ][:n_images]
        check_samples[experiment_name] = desired_results

    compare1, compare2 = name.split("-")

    image_paths, gt_paths = get_paths(dataset_name, split="test")

    # generate random colors for instances
    from matplotlib import colors

    def get_random_colors(labels):
        n_labels = len(np.unique(labels)) - 1
        cmap = [[0, 0, 0]] + np.random.rand(n_labels, 3).tolist()
        cmap = colors.ListedColormap(cmap)
        return cmap

    # let's create a directory structure to save results
    for setting in all_settings:
        setting_split = setting.split("/")
        setting_name = setting_split[0] + "_" + setting_split[1]

        save_dir = os.path.join("figures", modality, dataset_name, model_type, name, setting_name)
        os.makedirs(save_dir, exist_ok=True)

        for image_id, metric1, metric2 in check_samples[setting_name]:
            sample1 = imageio.imread(
                os.path.join(EXPERIMENT_ROOT, compare1, modality, dataset_name, model_type, setting, image_id)
            )
            sample2 = imageio.imread(
                os.path.join(EXPERIMENT_ROOT, compare2, modality, dataset_name, model_type, setting, image_id)
            )
            image = imageio.imread(*[image_path for image_path in image_paths if image_id in image_path])
            gt = imageio.imread(*[gt_path for gt_path in gt_paths if image_id in gt_path])

            plt.title(image_id)
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(image, cmap="gray")  # image
            axes[0, 0].title.set_text("Image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(gt, cmap=get_random_colors(gt), interpolation="nearest")  # gt
            axes[0, 1].title.set_text("Labels")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(sample1, cmap=get_random_colors(sample1), interpolation="nearest")  # comparison point 1
            axes[1, 0].title.set_text(f"{compare1}; {metric_choice} - {round(metric1, 3)}")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(sample2, cmap=get_random_colors(sample2), interpolation="nearest")  # comparision point 2
            axes[1, 1].title.set_text(f"{compare2}; {metric_choice} - {round(metric2, 3)}")
            axes[1, 1].axis("off")

            save_path = os.path.join(save_dir, image_id.split(".")[0] + ".png")
            plt.savefig(save_path)

            plt.close()


def main():
    # the inference settings we want to check for
    all_settings = [
        "amg/inference", "instance_segmentation_with_decoder/inference",
        "start_with_box/iteration00", "start_with_box/iteration07",
        "start_with_point/iteration00", "start_with_point/iteration07"
    ]
    # the two experiments we compare between
    compare_experiments = ["generalist", "specialist"]
    n_images = 10
    dataset_name = "livecell"
    modality = "lm"
    model_type = "vit_h"
    metric_choice = "msa"

    explore_differences(
        dataset_name=dataset_name,
        modality=modality,
        model_type=model_type,
        all_settings=all_settings,
        compare_experiments=compare_experiments,
        n_images=n_images,
        metric_choice=metric_choice,
        sort_decreasing=True  # i.e. when `True`, largest gap to smallest gap (and vice-versa)
    )


if __name__ == "__main__":
    main()
