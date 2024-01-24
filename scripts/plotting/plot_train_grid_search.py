import os
import pandas as pd

from util import plot_grid_2d


def main():

    data_root = "results/grid_search_train"

    results_box = pd.read_csv(os.path.join(data_root, "iterative_prompts_start_box.csv"))
    results_point = pd.read_csv(os.path.join(data_root, "iterative_prompts_start_point.csv"))
    result_seg = pd.read_csv(os.path.join(data_root, "instance_segmentation_with_decoder.csv"))

    plot_grid_2d(
        results_point, results_box, data_instance_segmentation=result_seg,
        grid_column1="optimizer", grid_column2="lr",
    )


main()
