import os

from torch_em.data import datasets
from torch_em.util.image import load_data


def _get_cellpose_sample_data(path, download):
    # TODO: update to self-chosen samples.
    raw_paths, label_paths = datasets.cellpose.get_cellpose_paths(
        path=path, split="test", choice="cyto", download=download
    )
    return raw_paths, label_paths


def _get_dataset_paths(path, dataset_name, view=False):
    dataset_paths = {
        # 2d LM dataset for cell segmentation
        "cellpose": lambda: _get_cellpose_sample_data(path=os.path.join(path, "cellpose"), download=True),

        # 3d LM dataset for nuclei segmentation
        "embedseg": lambda: datasets.embedseg_data.get_embedseg_paths(
            path=os.path.join(path, "embedseg"), name="Mouse-Skull-Nuclei-CBG", split="train", download=True,
        ),

        # 3d EM dataset for membrane segmentation
        "platynereis_cells": lambda: datasets.platynereis.get_platynereis_paths(
            path=os.path.join(path, "platynereis"), sample_ids=None, name="cells", download=True,
        ),
    }

    dataset_keys = {
        "cellpose": [None, None],
        "embedseg": [None, None],
        "platynereis_cells": ["volumes/raw/s1", "volumes/labels/segmentation/s1"]
    }

    if dataset_name is None:
        dataset_names = [dataset_name]
    else:
        dataset_names = [dataset_name]

    for dname in dataset_names:
        if dname not in dataset_paths:
            raise ValueError(
                f"'{dname}' is not a supported dataset enabled for download. "
                f"Please choose from {list(dataset_paths.keys())}."
            )

        paths = dataset_paths[dname]()
        print(f"'{dataset_name}' is download at {path}.")

        # platynereis/membrane/train_data_membrane_02.n5 (Platynereis Cells)
        # embedseg/Mouse-Skull-Nuclei-CBG/train/images/X1.tif (EmbedSeg)

        if view:
            import napari

            if isinstance(paths, tuple):  # datasets with explicit raw and label paths
                raw_paths, label_paths = paths
            else:
                raw_paths = label_paths = paths

            raw_key, label_key = dataset_keys[dname]
            for raw_path, label_path in zip(raw_paths, label_paths):
                print(raw_path)
                raw = load_data(raw_path, raw_key)
                labels = load_data(label_path, label_key)

                v = napari.Viewer()
                v.add_image(raw)
                v.add_labels(labels)
                napari.run()

                # break  # comment this line out in case you would like to visualize all samples.


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="./data")
    parser.add_argument("-d", "--dataset_name", type=str, default=None)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    _get_dataset_paths(path=args.input_path, dataset_name=args.dataset_name, view=args.view)


if __name__ == "__main__":
    main()
