import os
from tqdm import tqdm

import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy

from util import get_paths, EXPERIMENT_ROOT


def explore_differences(
    dataset_name,
    modality,
    model_type,
    compare_experiments,
    all_settings,
    metric_choice="msa"
):
    image_paths, gt_paths = get_paths(dataset_name, split="test")
    image_paths, gt_paths = image_paths[:5], gt_paths[:5]

    assert len(compare_experiments) == 2, "You should provide only two experiment names to compare."

    per_experiment_res = []
    for experiment in compare_experiments:
        experiment_dir = os.path.join(EXPERIMENT_ROOT, experiment, modality, dataset_name, model_type)
        assert os.path.exists(experiment_dir), f"The experiment does not exist for {dataset_name}"

        per_setting_res = {}
        for setting in all_settings:
            setting_split = setting.split("/")
            setting_name = setting_split[0] + "_" + setting_split[1]

            res = []
            for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths), desc=setting_name):
                image_id = os.path.split(image_path)[-1]

                prediction_path = os.path.join(experiment_dir, setting, image_id)

                image = imageio.imread(image_path)
                prediction = imageio.imread(prediction_path)
                gt = imageio.imread(gt_path)

                msa, sa_acc = mean_segmentation_accuracy(prediction, gt, return_accuracies=True)
                res.append(msa if metric_choice == "msa" else sa_acc[0])

            per_setting_res[setting_name] = res

        per_experiment_res.append({"name": experiment, "results": per_setting_res})

    experiment1, experiment2 = per_experiment_res

    # now let's calculate the difference between experiments and identify the ones with biggest diff.
    for (k1, v1), (k2, v2) in zip(experiment1.items(), experiment2.items()):
        if k1 == k2 == "results":
            per_setting_res = {}
            for (res_k1, res_v1), (res_k2, res_v2) in zip(v1.values(), v2.values()):
                assert res_k1 == res_k2
                res = list(set(res_v1) - set(res_v2))
                per_setting_res[f"{res_k1}_difference"] = res

            name = f"{compare_experiments[0]}-{compare_experiments[1]}"
            per_experiment_res[name] = per_setting_res


def main():
    all_settings = [
        "amg/inference", "instance_segmentation_with_decoder/inference",
        "start_with_box/iteration00", "start_with_point/iteration07"
    ]
    compare_experiments = ["generalist", "specialist"]

    explore_differences(
        dataset_name="livecell",
        modality="lm",
        model_type="vit_h",
        all_settings=all_settings,
        compare_experiments=compare_experiments,
        metric_choice="msa"
    )


if __name__ == "__main__":
    main()
