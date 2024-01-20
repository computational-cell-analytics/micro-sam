import os
import matplotlib.pyplot as plt
import pandas as pd
from util import plot_iterative_prompting


def _load_sam_results(root):
    point = pd.read_csv(os.path.join(root, "iterative_prompts_start_point.csv"))
    point = point.rename(columns={"Unnamed: 0": "iteration"})

    box = pd.read_csv(os.path.join(root, "iterative_prompts_start_box.csv"))
    box = box.rename(columns={"Unnamed: 0": "iteration"})

    # TODO add cellpose baseline
    extra = {
        "amg": pd.read_csv(os.path.join(root, "amg.csv")),
        "ais": pd.read_csv(os.path.join(root, "instance_segmentation_with_decoder.csv")),
    }
    return point, box, extra


def plot_livecell():
    fig, axes = plt.subplots(2)

    spec_point, spec_box, extra = _load_sam_results("./results/specialists/lm/livecell/vit_b")

    plot_iterative_prompting(spec_point, spec_box, extra, ax=axes[0], show=False, sharey=True)
    plt.show()


def main():
    plot_livecell()


if __name__ == "__main__":
    main()
