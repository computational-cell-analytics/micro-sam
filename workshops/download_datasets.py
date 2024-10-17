import os

from torch_em.data import datasets


def _get_cellpose_sample_data(path, download):
    raw_paths = ...
    label_paths = ...

    return raw_paths, label_paths


def _get_dataset_paths(path, dataset_name):
    all_datasets = {
        # 2d LM dataset for cell segmentation
        "cellpose": lambda: _get_cellpose_sample_data(path=os.path.join(path, "cellpose"), download=True),

        # 3d LM dataset fro nuclei segmentation
        "embedseg": lambda: datasets.embedseg_data.get_embedseg_paths(
            path=os.path.join(path, "embedseg"), name="Mouse-Skull-Nuclei-CBG", split="test", download=True,
        ),

        # 3d EM dataset for membrane segmentation
        "platynereis_cells": lambda: datasets.platynereis.get_platynereis_paths(
            path=os.path.join(path, "platynereis"), sample_ids=None, name="cells", download=True,
        ),
    }

    if dataset_name is None:
        dataset_names = [dataset_name]
    else:
        dataset_names = [dataset_name]

    for dname in dataset_names:
        all_datasets[dname]()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="./data")
    parser.add_argument("-d", "--dataset_name", type=str, default=None)
    args = parser.parse_args()

    _get_dataset_paths(path=args.input_path, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
