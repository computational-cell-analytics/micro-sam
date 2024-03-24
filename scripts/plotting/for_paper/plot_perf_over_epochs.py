import os
from pathlib import Path
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/from_carolin/perf_over_epochs/"

# adding a fixed color palette to each experiments, for consistency in plotting the legends
PALETTE = {"vit_t": "#089099", "vit_b": "#7CCBA2", "vit_l": "#7C1D6F", "vit_h": "#F0746E"}
MODELS = {"vit_t": 'ViT-T', 'vit_b': 'ViT-B', 'vit_l': 'ViT-L', 'vit_h': 'ViT-H'}


def gather_all_results():
    res_list = []
    for experiment_dir in glob(os.path.join(EXPERIMENT_ROOT, "*")):
        if os.path.split(experiment_dir)[-1].startswith("epoch"):
            epoch = os.path.split(experiment_dir)[-1].split('h')[-1]
            for i,result_dir in enumerate(glob(os.path.join(experiment_dir, '*', "results", '*'))):

                if os.path.split(result_dir)[-1].startswith("grid_search_"):
                    continue
                model = result_dir.split("/")[-3]
    
            
                setting_name = Path(result_dir).stem
                result = pd.read_csv(result_dir)

                if setting_name == "amg" or setting_name.startswith('instance'):
                    res_df = pd.DataFrame(
                        {
                            "name": setting_name,
                            "type": "none",
                            "model": model,
                            "epoch": epoch,
                            "result": result.iloc[0]["msa"]
                        }, index=[i]
                    )
                else:
                    prompt_name = Path(result_dir).stem.split("_")[-1]
                    res_df = pd.concat(
                        [
                            pd.DataFrame(
                                {"name": setting_name,
                                    "type": prompt_name,
                                    "model": model,
                                    "epoch": epoch,
                                    "result": result.iloc[0]["msa"]},
                                    index = [i]
                            ),
                            pd.DataFrame(
                                {"name": setting_name,
                                    "type": f"i_{prompt_name[0]}",
                                    "model": model,
                                    "epoch": epoch,
                                    "result": result.iloc[-1]["msa"]},
                                    index=[i]
                            )
                        ], ignore_index=True
                    )
                res_list.append(res_df)
    result = pd.concat(res_list)
    result['epoch'] = result['epoch'].astype(int)

    return result


def get_plots(ax, data, experiment_name):
    
    plt.rcParams["hatch.linewidth"] = 1.5
    
    sns.lineplot(data=data, x='epoch', y='result', ax=ax, hue='model', palette=PALETTE, errorbar='pi', err_style='band')
 
    ax.set(xlabel=None, ylabel=None)
    ax.legend(title="Models", bbox_to_anchor=(1, 1))
    ax.set_title(experiment_name, fontsize=13, fontweight='bold')



def plot_perf_over_epochs():

    all_data = gather_all_results()
    fig, ax = plt.subplots(3,2, figsize=(12,12))

    amg = all_data[all_data["name"] == "amg"]
    ais = all_data[all_data["name"] == "instance_segmentation_with_decoder"]
    point = all_data[all_data["type"] == "point"]
    box = all_data[all_data["type"] == "box"]
    i_point = all_data[all_data["type"] == "i_p"]
    i_box = all_data[all_data["type"] == "i_b"]

    get_plots(ax[0,0], point, "point")
    get_plots(ax[0,1], box, "box")
    get_plots(ax[1,0], ais, "ais")
    get_plots(ax[1,1], amg, "amg")
    get_plots(ax[2,0], i_point, "iterative prompting (start with point)")
    get_plots(ax[2,1], i_box, "iterative prompting (start with box)")


    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()


    fig.legend(all_lines, [MODELS[l] for l in labels], loc="upper left")
    plt.tight_layout()

    fig.text(0.78,0.97, 'X-Axis: Epoch \nY-Axis: Segmentation Quality', verticalalignment='top', 
             horizontalalignment='left', bbox=dict(boxstyle='round', edgecolor='lightgrey', facecolor='None'))

    plt.show()
    plt.subplots_adjust(top=0.9, right=0.9, left=0.1, bottom=0.05)
    
    fig.suptitle("Performance over Epochs", y=0.97, fontsize=26)
    plt.savefig("plot_perf_over_epochs.svg")
    plt.close()



def main():
    plot_perf_over_epochs()

if __name__ == "__main__":
    main()