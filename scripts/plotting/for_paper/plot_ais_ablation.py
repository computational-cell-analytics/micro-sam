import pandas as pd
import seaborn as sns
from natsort import natsorted

import matplotlib.pyplot as plt
import matplotlib.patches as patches


base_color = '#0562A0'
highlight_color = '#045275'
plt.rcParams.update({'font.size': 30})


# NOTE: the score formats below are a list of numbers: [X, Y, Z],
# where: X is the mSA, Y is SA50 and Z is SA75

LIVECELL_AIS = {
    "unet": [0.4188, 0.699752, 0.443877],
    "unetr_scratch": [0.415419, 0.699897, 0.439006],
    "unetr_sam": [0.445632, 0.726114, 0.479634],
    "semanticsam_scratch": [0.386169, 0.671345, 0.401836],
    "semanticsam_sam": [0.428852, 0.706803, 0.45969]
}

COVID_IF_AIS = {
    "1": {
        "unet": [0.124261, 0.306542, 0.085534],
        "unetr_scratch": [0.150799, 0.372263, 0.101136],
        "unetr_sam": [0.282399, 0.555058, 0.25503],
        "semanticsam_scratch": [0.09322, 0.238215, 0.0615],
        "semanticsam_sam": [0.299337, 0.612757, 0.264384]
    },
    "2": {
        "unet": [0.194456, 0.426158, 0.160465],
        "unetr_scratch": [0.203448, 0.439231, 0.172646],
        "unetr_sam": [0.308674, 0.584671, 0.290992],
        "semanticsam_scratch": [0.117305, 0.285744, 0.083979],
        "semanticsam_sam": [0.311751, 0.632971, 0.281148]
    },
    "5": {
        "unet": [0.243485, 0.495585, 0.219],
        "unetr_scratch": [0.250491, 0.52194, 0.221091],
        "unetr_sam": [0.362728, 0.683941, 0.343065],
        "semanticsam_scratch": [0.136756, 0.32772, 0.100696],
        "semanticsam_sam": [0.320606, 0.649073, 0.290766]
    },
    "10": {
        "unet": [0.29883, 0.588136, 0.280681],
        "unetr_scratch": [0.286946, 0.571417, 0.264325],
        "unetr_sam": [0.401787, 0.729247, 0.39796],
        "semanticsam_scratch": [0.145352, 0.353673, 0.104027],
        "semanticsam_sam": [0.375741, 0.729203, 0.354669]
    }
}

MODEL_NAME_MAPS = {
    "unet": "UNet",
    "unetr_scratch": "UNETR\n$\it{(scratch)}$",
    "unetr_sam": "UNETR\n$\it{(SAM)}$",
    "semanticsam_scratch": "SemSam\n$\it{(scratch)}$",
    "semanticsam_sam": "SemSam\n$\it{(SAM)}$"
}


def make_livecell_barplot():
    labels = list(LIVECELL_AIS.keys())
    model_labels = [MODEL_NAME_MAPS[model] for model in labels]
    scores = [LIVECELL_AIS[model][0] for model in labels]

    max_index = scores.index(max(scores))

    data = {"Model": model_labels, "Score": scores}
    df = pd.DataFrame(data)

    plt.figure(figsize=(20, 15))
    bars = sns.barplot(x="Model", y="Score", data=df, color=base_color)

    for i, bar in enumerate(bars.patches):
        if i == max_index:
            shadow = patches.FancyBboxPatch(
                (bar.get_x() - 0.01, bar.get_y() - 0.01),
                bar.get_width() + 0.02,
                bar.get_height() + 0.0025,
                boxstyle="round,pad=0.011",
                linewidth=2.5,
                edgecolor=None,
                facecolor=highlight_color,
                alpha=0.3,
                zorder=-1
            )
            plt.gca().add_patch(shadow)

    plt.xlabel(None)
    plt.ylabel("Mean Segmentation Accuracy", fontweight="bold")
    plt.title("Automatic Instance Segmentation (LIVECell)")
    plt.ylim(0, max(scores) + 0.05)

    plt.gca().yaxis.labelpad = 30
    plt.gca().xaxis.labelpad = 20

    yticks = [i * 0.05 for i in range(1, int(max(scores) / 0.05) + 2)]
    plt.yticks(yticks)

    plt.tight_layout()
    plt.savefig("s14_1.png")
    plt.savefig("s14_1.svg")
    plt.savefig("s14_1.pdf")


def make_covid_if_lineplot():
    markers = {
        'unet': 'P', 'unetr_scratch': 'X', 'unetr_sam': 'o', 'semanticsam_scratch': '^', 'semanticsam_sam': 'd'
    }
    line_styles = {
        'unet': '-', 'unetr_scratch': '--', 'unetr_sam': '-.', 'semanticsam_scratch': ':', 'semanticsam_sam': '-'
    }

    x = natsorted(COVID_IF_AIS.keys())
    models = list(COVID_IF_AIS[x[0]].keys())

    data = []
    for key in x:
        for model in models:
            data.append({'Key': key, 'Model': model, 'Score': COVID_IF_AIS[key][model][0]})

    df = pd.DataFrame(data)

    plt.figure(figsize=(20, 15))
    for model in models:
        sns.lineplot(
            data=df[df["Model"] == model], x='Key', y='Score', marker=markers[model],
            linestyle=line_styles[model], markersize=15, linewidth=2.5, label=MODEL_NAME_MAPS[model], color=base_color,
        )

    plt.xlabel("Number of Images", fontweight="bold")
    plt.ylabel("Mean Segmentation Accuracy", fontweight="bold")
    plt.title("Automatic Instance Segmentation (Covid IF)")

    plt.gca().yaxis.labelpad = 30
    plt.gca().xaxis.labelpad = 20

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)

    plt.tight_layout()
    plt.savefig("s14_2.png")
    plt.savefig("s14_2.svg")
    plt.savefig("s14_2.pdf")


def main():
    make_livecell_barplot()
    make_covid_if_lineplot()


main()
