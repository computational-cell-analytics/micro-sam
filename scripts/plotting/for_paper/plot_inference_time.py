from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# replace with path to inference time csv files
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/from_carolin/inference_time/"

PALETTE = {"ViT Tiny": "#089099", "ViT Base": "#7CCBA2", "ViT Large": "#7C1D6F", "ViT Huge": "#F0746E"}
MODELS = ['ViT Huge', 'ViT Large', 'ViT Base', 'ViT Tiny']


# TODO:
# changed to grouped bar plots


VIT_T_GPU = [
    {"benchmark": "embeddings", "runtimes": 0.039},
    {"benchmark": "prompt-p1n0", "runtimes": 0.009},
    {"benchmark": "prompt-box", "runtimes": 0.008},
    {"benchmark": "ais", "runtimes": 0.259},
    {"benchmark": "amg", "runtimes": 3.375}
]

VIT_B_GPU = [
    {"benchmark": "embeddings", "runtimes": 0.202},
    {"benchmark": "prompt-p1n0", "runtimes": 0.009},
    {"benchmark": "prompt-box", "runtimes": 0.008},
    {"benchmark": "ais", "runtimes": 0.255},
    {"benchmark": "amg", "runtimes": 3.378}
]

VIT_L_GPU = [
    {"benchmark": "embeddings", "runtimes": 0.479},
    {"benchmark": "prompt-p1n0", "runtimes": 0.009},
    {"benchmark": "prompt-box", "runtimes": 0.008},
    {"benchmark": "ais", "runtimes": 0.247},
    {"benchmark": "amg", "runtimes": 3.374}
]

VIT_H_GPU = [
    {"benchmark": "embeddings", "runtimes": 0.874},
    {"benchmark": "prompt-p1n0", "runtimes": 0.009},
    {"benchmark": "prompt-box", "runtimes": 0.008},
    {"benchmark": "ais", "runtimes": 0.249},
    {"benchmark": "amg", "runtimes": 3.360}
]

VIT_T_CPU = [
    {"benchmark": "embeddings", "runtimes": 0.27},
    {"benchmark": "prompt-p1n0", "runtimes": 0.027},
    {"benchmark": "prompt-box", "runtimes": 0.025},
    {"benchmark": "ais", "runtimes": 1.23},
    {"benchmark": "amg", "runtimes": 18.81}
]

VIT_B_CPU = [
    {"benchmark": "embeddings", "runtimes": 1.46},
    {"benchmark": "prompt-p1n0", "runtimes": 0.028},
    {"benchmark": "prompt-box", "runtimes": 0.026},
    {"benchmark": "ais", "runtimes": 1.24},
    {"benchmark": "amg", "runtimes": 18.97}
]

VIT_L_CPU = [
    {"benchmark": "embeddings", "runtimes": 3.67},
    {"benchmark": "prompt-p1n0", "runtimes": 0.028},
    {"benchmark": "prompt-box", "runtimes": 0.026},
    {"benchmark": "ais", "runtimes": 1.23},
    {"benchmark": "amg", "runtimes": 18.78}
]

VIT_H_CPU = [
    {"benchmark": "embeddings", "runtimes": 5.07},
    {"benchmark": "prompt-p1n0", "runtimes": 0.028},
    {"benchmark": "prompt-box", "runtimes": 0.027},
    {"benchmark": "ais", "runtimes": 1.22},
    {"benchmark": "amg", "runtimes": 18.89}
]


plt.rcParams.update({"font.size": 30})


def get_radar_plot(ax, dfs, device, show_std=False):
    plt.rcParams["hatch.linewidth"] = 1.5
    cat = dfs[0]['benchmark'].unique().tolist()

    cat = [*cat, cat[0]]  # to close the radar, duplicate the first column
    cat_labels = list(map(lambda x: x.replace('prompt-p1n0', 'point'), cat))
    cat_labels = list(map(lambda x: x.replace('prompt-box', 'box'), cat_labels))

    # normalise the data to a proper scale
    # max_values = (3000, 700) if device == "CPU" else (28, 7)
    max_values = (20, 6) if device == "CPU" else (5, 1)

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(cat))

    for i, df in enumerate(dfs):
        norm = {}
        errors = {}
        for label in cat[:-1]:
            max_value = max_values[0] if label == 'amg' else max_values[1]
            norm[label] = df[df['benchmark'] == label]['runtimes'].item() / max_value
            if show_std:
                errors[label] = df[df['benchmark'] == label]['error'].item() / max_value

        group_norm = list(norm.values())
        group_norm += group_norm[:1]
        group = list(df['runtimes'])
        group += group[:1]

        if show_std:
            err_norm = list(errors.values())
            err_norm = np.array([*err_norm, err_norm[0]])

        ax.plot(label_loc, group_norm, 'o-', color=PALETTE[MODELS[i]], label=MODELS[i])

        # Show errors if wanted
        if show_std:
            ax.fill(label_loc, group_norm + err_norm, facecolor=PALETTE[MODELS[i]], alpha=0.25)
            ax.fill(label_loc,  group_norm - err_norm, facecolor="white", alpha=1)

    # labeling of the axis
    ax.text(0, 0, '0.0', ha='center', va='bottom', fontsize=10)
    for label, _x in zip(cat[:-1], label_loc[:-1]):
        max_value = max_values[0] if label == 'amg' else max_values[1]

        ha = 'center'
        va = 'bottom'
        if label == 'embeddings':
            va = 'top'
        if label == 'prompt-p1n0':
            ha = 'right'
        elif label == 'amg':
            ha = 'left'
        for j in range(1, 6):
            value_str = str(int(round(j/5*max_value))) if device == 'CPU' else str(round(j/5*max_value, 2))
            ax.text(_x, j/5, value_str, va=va, ha=ha, fontsize=10)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(label_loc), cat_labels, fontsize=13)
    ax.set_yticklabels([])
    ax.legend()
    ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.4)

    for label, angle in zip(ax.get_xticklabels(), label_loc):
        if 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        elif angle == 0 or angle == 2*np.pi:
            label.set_horizontalalignment('center')
        else:
            label.set_horizontalalignment('right')

    ax.set_title(f'{device}', y=1.1, fontweight='bold', fontsize=16)


def _get_plots(new_results=True, show_std=False):
    if new_results:
        vit_t_gpu = pd.concat([pd.DataFrame.from_dict([_dict]) for _dict in VIT_T_GPU])
        vit_b_gpu = pd.concat([pd.DataFrame.from_dict([_dict]) for _dict in VIT_B_GPU])
        vit_l_gpu = pd.concat([pd.DataFrame.from_dict([_dict]) for _dict in VIT_L_GPU])
        vit_h_gpu = pd.concat([pd.DataFrame.from_dict([_dict]) for _dict in VIT_H_GPU])

        vit_t_cpu = pd.concat([pd.DataFrame.from_dict([_dict]) for _dict in VIT_T_CPU])
        vit_b_cpu = pd.concat([pd.DataFrame.from_dict([_dict]) for _dict in VIT_B_CPU])
        vit_l_cpu = pd.concat([pd.DataFrame.from_dict([_dict]) for _dict in VIT_L_CPU])
        vit_h_cpu = pd.concat([pd.DataFrame.from_dict([_dict]) for _dict in VIT_H_CPU])
    else:
        vit_t_gpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_t_gpu.csv")
        vit_b_gpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_b_gpu.csv")
        vit_l_gpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_l_gpu.csv")
        vit_h_gpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_h_gpu.csv")

        vit_t_cpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_t_cpu.csv")
        vit_b_cpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_b_cpu.csv")
        vit_l_cpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_l_cpu.csv")
        vit_h_cpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_h_cpu.csv")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

    # The order of the models matters
    get_radar_plot(axs[0], [vit_h_gpu, vit_l_gpu, vit_b_gpu, vit_t_gpu], "GPU", show_std=show_std)
    get_radar_plot(axs[1], [vit_h_cpu, vit_l_cpu, vit_b_cpu, vit_t_cpu], "CPU", show_std=show_std)

    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    fig.legend(all_lines, all_labels, loc="upper left", fontsize=12)
    fig.text(
        x=0.5, y=0.1, s='All measurements in seconds', fontsize=12, verticalalignment='top',
        horizontalalignment='center', bbox=dict(boxstyle='round', edgecolor='lightgrey', facecolor='None')
    )

    plt.subplots_adjust(top=0.85, right=0.9, left=0.1, bottom=0.05, wspace=0.4)
    fig.suptitle("Inference Timings", y=0.97, fontsize=26)

    plt.show()
    print("Saving plot ...  ")
    save_path = "inference_times.svg"
    plt.savefig(save_path)
    plt.savefig(Path(save_path).with_suffix(".pdf"))
    plt.close()


def main():
    show_std = False
    _get_plots(show_std=show_std)


if __name__ == "__main__":
    main()
