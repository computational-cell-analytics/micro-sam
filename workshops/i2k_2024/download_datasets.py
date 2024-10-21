import os
from glob import glob
from natsort import natsorted

from torch_em.data import datasets
from torch_em.util.image import load_data


def _download_sample_data(data_dir, url, checksum, download):
    if os.path.exists(data_dir):
        return

    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "data.zip")
    datasets.util.download_source(path=zip_path, url=url, download=download, checksum=checksum)
    datasets.util.unzip(zip_path=zip_path, dst=data_dir)


def _get_cells_sample_data_paths(path, download):
    data_dir = os.path.join(path, "cells")

    url = "https://owncloud.gwdg.de/index.php/s/c96cyWc1PpLAPOn/download"
    checksum = "5d6cb5bc67a2b48c862c200d2df3afdfe6703f9c21bc33a3dd13d2422a396897"

    _download_sample_data(data_dir, url, checksum, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, "masks", "*.png")))

    return raw_paths, label_paths


def _get_hpa_data_paths(path, split, download):
    splits = ["train", "val", "test"]
    assert split in splits, f"'{split}' is not a valid split."

    data_dir = os.path.join(path, "hpa")
    url = "https://owncloud.gwdg.de/index.php/s/IrzUcaMxQKVRLTs/download"
    checksum = "f2c41be1761cdd96635ee30bee9dcbdeda4ebe3ab3467ad410c28417d46cdaad"

    _download_sample_data(data_dir, url, checksum, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, split, "images", "*.tif")))

    if split == "test":  # The 'test' split for HPA does not have labels.
        return raw_paths, None
    else:
        label_paths = natsorted(glob(os.path.join(data_dir, split, "labels", "*.tif")))
        return raw_paths, label_paths


def _get_nuclei_3d_data_paths(path, download):
    data_dir = os.path.join(path, "nuclei_3d")
    url = "https://owncloud.gwdg.de/index.php/s/QdibduvClGmruIV/download"
    checksum = "551d2c55e0e5614ae21c03e75e7a0afb765b312cb569dd4c32d1d634d8798c91"
    _download_sample_data(data_dir, url, checksum, download=download)
    raw_paths = [os.path.join(data_dir, "images", "X1.tif")]
    label_paths = [os.path.join(data_dir, "masks", "Y1.tif")]
    return raw_paths, label_paths


def _get_volume_em_data_paths(path, download):
    data_dir = os.path.join(path, "volume_em")
    url = "https://owncloud.gwdg.de/index.php/s/5CzsV6bsqX0kvSv/download"
    checksum = "e820e2a89ffb5d466fb4646945b8697269501cce18376f47b946c7773ede4653"
    _download_sample_data(data_dir, url, checksum, download=download)
    raw_paths = [os.path.join(data_dir, "images", "train_data_membrane_02.tif")]
    label_paths = [os.path.join(data_dir, "masks", "train_data_membrane_02_labels.tif")]
    return raw_paths, label_paths


def _get_dataset_paths(path, dataset_name, view=False):
    dataset_paths = {
        # 2d LM dataset for cell segmentation
        "cells": lambda: _get_cells_sample_data_paths(path=path, download=True),
        "hpa": lambda: _get_hpa_data_paths(path=path, download=True, split="train"),
        # 3d LM dataset for nuclei segmentation
        "nuclei_3d": lambda: _get_nuclei_3d_data_paths(path=path, download=True),
        # 3d EM dataset for membrane segmentation
        "volume_em": lambda: _get_volume_em_data_paths(path=path, download=True),
    }

    dataset_keys = {
        "cells": [None, None],
        "nuclei_3d": [None, None],
        "volume_em": [None, None]
    }

    if dataset_name is None:  # Download all datasets.
        dataset_names = list(dataset_paths.keys())
    else:  # Download specific datasets.
        dataset_names = [dataset_name]

    for dname in dataset_names:
        if dname not in dataset_paths:
            raise ValueError(
                f"'{dname}' is not a supported dataset enabled for download. "
                f"Please choose from {list(dataset_paths.keys())}."
            )

        paths = dataset_paths[dname]()
        print(f"'{dataset_name}' is download at {path}.")

        if view:
            import napari

            if isinstance(paths, tuple):  # datasets with explicit raw and label paths
                raw_paths, label_paths = paths
            else:
                raw_paths = label_paths = paths

            raw_key, label_key = dataset_keys[dname]
            for raw_path, label_path in zip(raw_paths, label_paths):
                raw = load_data(raw_path, raw_key)
                labels = load_data(label_path, label_key)

                v = napari.Viewer()
                v.add_image(raw)
                v.add_labels(labels)
                napari.run()

                break  # comment this line out in case you would like to visualize all samples.


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download the dataset necessary for the workshop.")
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data will be downloaded. "
        "By default, the data will be stored in your current working directory at './data'."
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default=None,
        help="The choice of dataset you would like to download. By default, it downloads all the datasets. "
        "Optionally, you can choose to download either of 'cells', 'hpa', 'nuclei_3d' or 'volume_em'."
    )
    parser.add_argument(
        "-v", "--view", action="store_true", help="Whether to view the downloaded data."
    )
    args = parser.parse_args()

    _get_dataset_paths(path=args.input_path, dataset_name=args.dataset_name, view=args.view)


if __name__ == "__main__":
    main()
