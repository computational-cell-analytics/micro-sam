import os
import matplotlib.pyplot as plt
import pandas as pd
from util import plot_iterative_prompting


def _load_sam_results(root, mitonet_path):
    point = pd.read_csv(os.path.join(root, "iterative_prompts_start_point.csv"))
    point = point.rename(columns={"Unnamed: 0": "iteration"})

    box = pd.read_csv(os.path.join(root, "iterative_prompts_start_box.csv"))
    box = box.rename(columns={"Unnamed: 0": "iteration"})

    # with mitonet baseline
    extra = {
        "mitonet": pd.read_csv(mitonet_path),
        "amg": pd.read_csv(os.path.join(root, "amg.csv")),
    }
    ais_path = os.path.join(root, "instance_segmentation_with_decoder.csv")
    if os.path.exists(ais_path):
        extra["ais"] = pd.read_csv(ais_path)
    return point, box, extra


def plot_em_mitos(generalist_root, vanilla_root, mitonet_path):
    fig, axes = plt.subplots(2, 2, sharey=True)

    spec_point_b, spec_box_b, spec_extra_b = _load_sam_results(
        os.path.join(generalist_root, "vit_b", "results"), mitonet_path,
    )
    spec_point_h, spec_box_h, spec_extra_h = _load_sam_results(
        os.path.join(generalist_root, "vit_h", "results"), mitonet_path,
    )

    van_point_b, van_box_b, van_extra_b = _load_sam_results(
        os.path.join(vanilla_root, "vit_b", "results"), mitonet_path,
    )
    van_point_h, van_box_h, van_extra_h = _load_sam_results(
        os.path.join(vanilla_root, "vit_h", "results"), mitonet_path,
    )

    plot_iterative_prompting(van_point_b, van_box_b, van_extra_b, ax=axes[0, 0], show=False)
    axes[0, 0].set_title("VIT-B")
    plot_iterative_prompting(spec_point_b, spec_box_b, spec_extra_b, ax=axes[0, 1], show=False)
    axes[0, 1].set_title("VIT-B-EM")

    plot_iterative_prompting(van_point_h, van_box_h, van_extra_h, ax=axes[1, 0], show=False)
    axes[1, 0].set_title("VIT-H")
    plot_iterative_prompting(spec_point_h, spec_box_h, spec_extra_h, ax=axes[1, 1], show=False)
    axes[1, 1].set_title("VIT-H-EM")

    plt.show()


def plot_lucchi():
    plot_em_mitos(
        "results/generalists/em/lucchi/mito_nuc_em_generalist_sam/with_cem",
        "results/vanilla/em/lucchi",
        "results/benchmarking/mitonet/lucchi/results/mitonet.csv"
    )


def plot_mitoem():
    plot_em_mitos(
        "results/generalists/em/mitoem/mito_nuc_em_generalist_sam",
        "results/vanilla/em/mitoem",
        "results/benchmarking/mitonet/mitoem/results/mitonet.csv"
    )


def main():
    # plot_lucchi()
    plot_mitoem()


if __name__ == "__main__":
    main()
