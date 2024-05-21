import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


"""
def rename_prompts(values):
    new_vals = []
    for val in values:
        if val == "box/p0-n0":
            new_vals.append("box")
        else:
            new_vals.append(val.split("/")[-1])
    return new_vals


def standard_plot(
    data, model_column="model", prompt_column="prompt", metric_column="msa",
    ax=None, show=False, fontsize=16, with_amg=True,
):
    plt.rcParams.update({"font.size": fontsize})
    hue = "Prompt"
    data = data.rename(columns={metric_column: "Segmentation Quality", prompt_column: hue})

    data[hue] = rename_prompts(data[hue].values)

    prompt_names = ["p1-n0", "p2-n4", "p4-n8", "box"]
    if with_amg:
        prompt_names.append("amg")
    data[hue] = pd.Categorical(data[hue], prompt_names)
    data.sort_values(hue)

    barp = sns.barplot(data, x=model_column, y="Segmentation Quality", hue=hue, ax=ax)
    if show:
        plt.show()
    return barp
"""


# TODO create ax if not given
def plot_iterative_prompting(
    data_points, data_boxes, extra_data=None, fontsize=16, score_name="msa", ax=None, show=True
):
    """Plot evaluation for iterative prompting results.
    """
    plt.rcParams.update({"font.size": fontsize})

    palette_name = None
    sns.set_palette(palette_name)
    palette = sns.color_palette(palette_name)

    hue_column = "Start Prompt"
    data_points[hue_column] = "point"
    data_boxes[hue_column] = "box"
    prompt_names = ["point", "box"]

    data = pd.concat([data_points, data_boxes])

    data[hue_column] = pd.Categorical(data[hue_column], prompt_names)
    data.sort_values(hue_column)

    lp = sns.barplot(data, x="iteration", y=score_name, hue=hue_column, ax=ax)
    if extra_data is not None:
        bounds = ax.get_xbound()
        for i, (name, data) in enumerate(extra_data.items(), 2):
            color = palette[i]
            ax.hlines(data[score_name], xmin=bounds[0], xmax=bounds[1], label=name, colors=color)

    # Make sure everything is in the legend.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    if show:
        plt.show()
    return lp


# TODO support 1d
def plot_grid_2d(
    data_points, data_boxes, grid_column1, grid_column2, data_instance_segmentation=None, fontsize=16, score_name="msa",
):

    grid_columns = [grid_column1, grid_column2]
    grid_values = data_points[grid_columns].groupby(grid_columns).size().reset_index()[grid_columns]

    vals1, vals2 = pd.unique(data_points[grid_column1]).tolist(), pd.unique(data_points[grid_column2]).tolist()
    n1, n2 = len(vals1), len(vals2)
    fig, axes = plt.subplots(n1, n2, sharey=True)

    for i, row in grid_values.iterrows():
        val1, val2 = getattr(row, grid_column1), getattr(row, grid_column2)
        datap = data_points[(data_points[grid_column1] == val1) & (data_points[grid_column2] == val2)]
        datab = data_boxes[(data_boxes[grid_column1] == val1) & (data_boxes[grid_column2] == val2)]

        if data_instance_segmentation is None:
            datai = None
        else:
            datai = data_instance_segmentation[
                (data_instance_segmentation[grid_column1] == val1) & (data_instance_segmentation[grid_column2] == val2)
            ]

        i1, i2 = vals1.index(val1), vals2.index(val2)
        ax = axes[i1, i2]
        plot_iterative_prompting(
            datap, datab, data_instance_segmentation=datai, fontsize=fontsize, score_name=score_name, ax=ax, show=False
        )
        ax.set_title(f"{grid_column1}: {val1}, {grid_column2}: {val2}")
        if i > 0:
            ax.get_legend().remove()

    # plt.tight_layout()
    plt.show()
