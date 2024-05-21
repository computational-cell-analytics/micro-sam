import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# batch-mode segmentations

VIT_T_GPU_BATCH = [
    {"benchmark": "Embeddings", "runtimes": 0.039},
    {"benchmark": "AIS", "runtimes": 0.259},
    {"benchmark": "AMG", "runtimes": 3.375},
]

VIT_B_GPU_BATCH = [
    {"benchmark": "Embeddings", "runtimes": 0.202},
    {"benchmark": "AIS", "runtimes": 0.255},
    {"benchmark": "AMG", "runtimes": 3.378},
]

VIT_L_GPU_BATCH = [
    {"benchmark": "Embeddings", "runtimes": 0.479},
    {"benchmark": "AIS", "runtimes": 0.247},
    {"benchmark": "AMG", "runtimes": 3.374},
]

VIT_H_GPU_BATCH = [
    {"benchmark": "Embeddings", "runtimes": 0.874},
    {"benchmark": "AIS", "runtimes": 0.249},
    {"benchmark": "AMG", "runtimes": 3.360}
]

VIT_T_CPU_BATCH = [
    {"benchmark": "Embeddings", "runtimes": 0.27},
    {"benchmark": "AIS", "runtimes": 1.23},
    {"benchmark": "AMG", "runtimes": 18.81}
]

VIT_B_CPU_BATCH = [
    {"benchmark": "Embeddings", "runtimes": 1.46},
    {"benchmark": "AIS", "runtimes": 1.24},
    {"benchmark": "AMG", "runtimes": 18.97}
]

VIT_L_CPU_BATCH = [
    {"benchmark": "Embeddings", "runtimes": 3.67},
    {"benchmark": "AIS", "runtimes": 1.23},
    {"benchmark": "AMG", "runtimes": 18.78}
]

VIT_H_CPU_BATCH = [
    {"benchmark": "Embeddings", "runtimes": 5.07},
    {"benchmark": "AIS", "runtimes": 1.22},
    {"benchmark": "AMG", "runtimes": 18.89}
]

# interactive-mode segmentations

VIT_T_GPU_INT = [
    {"benchmark": "Point", "runtimes": 9.62},
    {"benchmark": "Box", "runtimes": 8.56},
]

VIT_B_GPU_INT = [
    {"benchmark": "Point", "runtimes": 9.65},
    {"benchmark": "Box", "runtimes": 8.44},
]

VIT_L_GPU_INT = [
    {"benchmark": "Point", "runtimes": 9.39},
    {"benchmark": "Box", "runtimes": 8.59},
]

VIT_H_GPU_INT = [
    {"benchmark": "Point", "runtimes": 9.60},
    {"benchmark": "Box", "runtimes": 8.64},
]

VIT_T_CPU_INT = [
    {"benchmark": "Point", "runtimes": 27.61},
    {"benchmark": "Box", "runtimes": 25.97},
]

VIT_B_CPU_INT = [
    {"benchmark": "Point", "runtimes": 28.25},
    {"benchmark": "Box", "runtimes": 26.40},
]

VIT_L_CPU_INT = [
    {"benchmark": "Point", "runtimes": 28.05},
    {"benchmark": "Box", "runtimes": 26.37},
]

VIT_H_CPU_INT = [
    {"benchmark": "Point", "runtimes": 28.09},
    {"benchmark": "Box", "runtimes": 27.26},
]

PALETTE = {
    "ViT Tiny": "#089099",
    "ViT Base": "#7CCBA2",
    "ViT Large": "#7C1D6F",
    "ViT Huge": "#F0746E"
}


plt.rcParams.update({'font.size': 30})


def _get_inference_timings_plots():
    fig, ax = plt.subplots(2, 2, figsize=(30, 20), sharey="row")

    df = pd.DataFrame(VIT_T_GPU_BATCH + VIT_B_GPU_BATCH + VIT_L_GPU_BATCH + VIT_H_GPU_BATCH)
    df['model'] = ['ViT Tiny'] * 3 + ['ViT Base'] * 3 + ['ViT Large'] * 3 + ['ViT Huge'] * 3
    sns.barplot(
        x="benchmark", y="runtimes", hue="model", data=df, ax=ax[0, 0], palette=PALETTE
    )
    ax[0, 0].set_xlabel(None)
    ax[0, 0].set_ylabel("Time Per Image $\it{(in}$ $\it{seconds)}$", fontweight="bold", labelpad=10)
    ax[0, 0].set_title("GPU", fontweight="bold")

    df = pd.DataFrame(VIT_T_CPU_BATCH + VIT_B_CPU_BATCH + VIT_L_CPU_BATCH + VIT_H_CPU_BATCH)
    df['model'] = ['ViT Tiny'] * 3 + ['ViT Base'] * 3 + ['ViT Large'] * 3 + ['ViT Huge'] * 3
    sns.barplot(
        x="benchmark", y="runtimes", hue="model", data=df, ax=ax[0, 1], palette=PALETTE
    )
    ax[0, 1].set_xlabel(None)
    ax[0, 1].set_ylabel(None)
    ax[0, 1].set_title("CPU", fontweight="bold")

    df = pd.DataFrame(VIT_T_GPU_INT + VIT_B_GPU_INT + VIT_L_GPU_INT + VIT_H_GPU_INT)
    df['model'] = ['ViT Tiny'] * 2 + ['ViT Base'] * 2 + ['ViT Large'] * 2 + ['ViT Huge'] * 2
    sns.barplot(
        x="benchmark", y="runtimes", hue="model", data=df, ax=ax[1, 0], palette=PALETTE
    )
    ax[1, 0].set_xlabel(None)
    ax[1, 0].set_ylabel("Time Per Object $\it{(in}$ $\it{milliseconds)}$", fontweight="bold", labelpad=10)

    df = pd.DataFrame(VIT_T_CPU_INT + VIT_B_CPU_INT + VIT_L_CPU_INT + VIT_H_CPU_INT)
    df['model'] = ['ViT Tiny'] * 2 + ['ViT Base'] * 2 + ['ViT Large'] * 2 + ['ViT Huge'] * 2
    sns.barplot(
        x="benchmark", y="runtimes", hue="model", data=df, ax=ax[1, 1], palette=PALETTE
    )
    ax[1, 1].set_xlabel(None)
    ax[1, 1].set_ylabel(None)

    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    plt.legend(loc="lower center", ncol=4, bbox_to_anchor=(-0.1, -0.25))
    fig.suptitle("Inference Timings", y=0.95, x=0.51)

    plt.savefig("5_a.svg")
    plt.savefig("5_a.pdf")


def main():
    _get_inference_timings_plots()


main()
