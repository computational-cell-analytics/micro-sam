import os
import matplotlib.pyplot as plt
import pandas as pd
from util import plot_iterative_prompting


def _load_sam_results(root):
    point = pd.read_csv(os.path.join(root, "iterative_prompts_start_point.csv"))
    point = point.rename(columns={"Unnamed: 0": "iteration"})

    box = pd.read_csv(os.path.join(root, "iterative_prompts_start_box.csv"))
    box = box.rename(columns={"Unnamed: 0": "iteration"})

    cellpose_path = "results/benchmarking/cellpose/livecell/livecell_cellpose.csv"
    extra = {
        "cellpose": pd.read_csv(cellpose_path),
        "amg": pd.read_csv(os.path.join(root, "amg.csv")),
    }
    ais_path = os.path.join(root, "instance_segmentation_with_decoder.csv")
    if os.path.exists(ais_path):
        extra["ais"] = pd.read_csv(ais_path)
    return point, box, extra


def plot_livecell():
    fig, axes = plt.subplots(2, 2, sharey=True)

    van_point_b, van_box_b, van_extra_b = _load_sam_results("./results/vanilla/lm/livecell/vit_b/results")
    spec_point_b, spec_box_b, spec_extra_b = _load_sam_results("./results/specialists/lm/livecell/vit_b")

    van_point_h, van_box_h, van_extra_h = _load_sam_results("./results/vanilla/lm/livecell/vit_h/results")
    spec_point_h, spec_box_h, spec_extra_h = _load_sam_results("./results/specialists/lm/livecell/vit_h")

    plot_iterative_prompting(van_point_b, van_box_b, van_extra_b, ax=axes[0, 0], show=False)
    axes[0, 0].set_title("VIT-B")
    plot_iterative_prompting(spec_point_b, spec_box_b, spec_extra_b, ax=axes[0, 1], show=False)
    axes[0, 1].set_title("VIT-B-LM")

    plot_iterative_prompting(van_point_h, van_box_h, van_extra_h, ax=axes[1, 0], show=False)
    axes[1, 0].set_title("VIT-H")
    plot_iterative_prompting(spec_point_h, spec_box_h, spec_extra_h, ax=axes[1, 1], show=False)
    axes[1, 1].set_title("VIT-H-LM")

    # TODO add generalist plots

    plt.show()


def main():
    plot_livecell()


if __name__ == "__main__":
    main()
