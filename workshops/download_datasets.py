import os
from glob import glob
from natsort import natsorted

from torch_em.data import datasets
from torch_em.util.image import load_data


def _download_sample_data(path, data_dir, download, url, checksum):
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "data.zip")
    datasets.util.download_source(path=zip_path, url=url, download=download, checksum=checksum)
    datasets.util.unzip(zip_path=zip_path, dst=path)


def _get_cellpose_sample_data_paths(path, download):
    data_dir = os.path.join(path, "cellpose", "cyto", "test")

    url = "https://owncloud.gwdg.de/index.php/s/slIxlmsglaz0HBE/download"
    checksum = "4d1ce7afa6417d051b93d6db37675abc60afe68daf2a4a5db0c787d04583ce8a"

    _download_sample_data(path, data_dir, download, url, checksum)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*_img.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*_masks.png")))

    return raw_paths, label_paths


def _get_hpa_data_paths(path, download):
    urls = [
        "https://owncloud.gwdg.de/index.php/s/zp1Fmm4zEtLuhy4/download",  # train
        "https://owncloud.gwdg.de/index.php/s/yV7LhGbGfvFGRBE/download",  # val
        "https://owncloud.gwdg.de/index.php/s/8tLY5jPmpw37beM/download",  # test
    ]
    checksums = [
        "6e5f3ec6b0d505511bea752adaf35529f6b9bb9e7729ad3bdd90ffe5b2d302ab",  # train
        "4d7a4188cc3d3877b3cf1fbad5f714ced9af4e389801e2136623eac2fde78e9c",  # val
        "8963ff47cdef95cefabb8941f33a3916258d19d10f532a209bab849d07f9abfe",  # test
    ]
    splits = ["train", "val", "test"]

    for url, checksum, split in zip(urls, checksums, splits):
        data_dir = os.path.join(path, split)
        _download_sample_data(path, data_dir, download, url, checksum)

    breakpoint()

    raw_paths = natsorted(glob(os.path.join(data_dir, "*")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*")))

    return raw_paths, label_paths


def _get_dataset_paths(path, dataset_name, view=False):
    dataset_paths = {
        # 2d LM dataset for cell segmentation
        "cellpose": lambda: _get_cellpose_sample_data_paths(path=os.path.join(path, "cellpose"), download=True),
        "hpa": lambda: _get_hpa_data_paths(path=os.path.join(path, "hpa"), download=True),
        # 3d LM dataset for nuclei segmentation
        "embedseg": lambda: datasets.embedseg_data.get_embedseg_paths(
            path=os.path.join(path, "embedseg"), name="Mouse-Skull-Nuclei-CBG", split="train", download=True,
        ),
        # 3d EM dataset for membrane segmentation
        "platynereis": lambda: datasets.platynereis.get_platynereis_paths(
            path=os.path.join(path, "platynereis"), sample_ids=None, name="cells", download=True,
        ),
    }

    dataset_keys = {
        "cellpose": [None, None],
        "embedseg": [None, None],
        "platynereis": ["volumes/raw/s1", "volumes/labels/segmentation/s1"]
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
        "Optionally, you can choose to download either of 'cellpose', 'hpa', 'embedseg' or 'platynereis'."
    )
    parser.add_argument(
        "-v", "--view", action="store_true", help="Whether to view the downloaded data."
    )
    args = parser.parse_args()

    _get_dataset_paths(path=args.input_path, dataset_name=args.dataset_name, view=args.view)


if __name__ == "__main__":
    main()
