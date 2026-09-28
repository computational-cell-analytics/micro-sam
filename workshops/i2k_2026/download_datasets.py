import os
import argparse

import napari

from torch_em.util.image import load_data
from torch_em.data.datasets.light_microscopy.dsb import get_dsb_paths
from torch_em.data.datasets.histopathology.lynsec import get_lynsec_paths
from torch_em.data.datasets.electron_microscopy.cremi import get_cremi_paths
from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths
from torch_em.data.datasets.light_microscopy.gonuclear import get_gonuclear_paths
from torch_em.data.datasets.electron_microscopy.platynereis import get_platynereis_paths


DATASETS_2D = ["cells_2d", "nuclei_2d", "histopatho"]
DATASETS_3D = ["nuclei_3d", "neurons_em", "cells_em"]

# The keys to the raw data and the labels in the files. None means that the file has no keys.
DATASET_KEYS = {
    "cells_2d": [None, None],
    "nuclei_2d": [None, None],
    "histopatho": [None, None],
    "nuclei_3d": ["raw/nuclei", "labels/nuclei"],
    "neurons_em": ["volumes/raw", "volumes/labels/neuron_ids"],
    # The cell labels map the neuropil to an ignore value.
    "cells_em": ["volumes/raw/s1", "volumes/labels/segmentation_corrected/v1/ignore_16777215/s1"],
}

# The automatic segmentation mode per dataset. AIS works better than APG for the LIVECell cells.
# AIS is fast for the nuclei in 3d, and APG works much better than AIS for the EM volumes.
SEGMENTATION_MODES = {
    "cells_2d": "ais",
    "nuclei_2d": "apg",
    "histopatho": "apg",
    "nuclei_3d": "ais",
    "neurons_em": "apg",
    "cells_em": "apg",
}

# The crops of the large volumes, so that the embeddings and APG are fast to compute.
DATASET_ROIS = {
    "nuclei_3d": (slice(88, 152), slice(120, 632), slice(580, 1092)),
    # Slices 14 and 74 of CREMI C are black, so the crop lies between them.
    "neurons_em": (slice(28, 60), slice(369, 881), slice(369, 881)),
    # The crop lies inside the region with cell annotations.
    "cells_em": (slice(45, 85), slice(64, 564), slice(64, 564)),
}


def _get_livecell_data_paths(path, download):
    return get_livecell_paths(path=os.path.join(path, "livecell"), split="test", download=download)


def _get_dsb_data_paths(path, download):
    return get_dsb_paths(path=os.path.join(path, "dsb"), source="reduced", split="test", download=download)


def _get_histopathology_data_paths(path, download):
    return get_lynsec_paths(path=os.path.join(path, "lynsec"), split="test", choice="h&e", download=download)


def _get_gonuclear_data_paths(path, download):
    # We use the volume which is held out from training for the 'hvit_*_cells' models.
    paths = get_gonuclear_paths(path=os.path.join(path, "gonuclear"), sample_ids=(1170,), download=download)
    return paths, paths


def _get_cremi_data_paths(path, download):
    paths = get_cremi_paths(path=os.path.join(path, "cremi"), samples=("C",), download=download)
    return paths, paths


def _get_platynereis_cells_data_paths(path, download):
    paths = get_platynereis_paths(
        path=os.path.join(path, "platynereis"), sample_ids=[2], name="cells", download=download
    )
    return paths, paths


def _get_paths_getters(path):
    return {
        "cells_2d": lambda: _get_livecell_data_paths(path=path, download=True),
        "nuclei_2d": lambda: _get_dsb_data_paths(path=path, download=True),
        "histopatho": lambda: _get_histopathology_data_paths(path=path, download=True),
        "nuclei_3d": lambda: _get_gonuclear_data_paths(path=path, download=True),
        "neurons_em": lambda: _get_cremi_data_paths(path=path, download=True),
        "cells_em": lambda: _get_platynereis_cells_data_paths(path=path, download=True),
    }


def get_dataset_paths(path, dataset_name):
    """Download a dataset, if necessary, and return its raw and label paths.

    Args:
        path: The folder where the datasets are stored.
        dataset_name: The name of the dataset.

    Returns:
        The filepaths to the raw data and to the labels.
    """
    paths_getters = _get_paths_getters(path)
    if dataset_name not in paths_getters:
        raise ValueError(
            f"'{dataset_name}' is not a supported dataset. Please choose from {list(paths_getters.keys())}."
        )
    return paths_getters[dataset_name]()


def get_image_and_labels(path, dataset_name, index=0):
    """Load one image or volume of a dataset together with its labels.

    Args:
        path: The folder where the datasets are stored.
        dataset_name: The name of the dataset.
        index: The index of the image in the dataset.

    Returns:
        The image and the labels.
    """
    raw_paths, label_paths = get_dataset_paths(path, dataset_name)
    raw_key, label_key = DATASET_KEYS[dataset_name]
    roi = DATASET_ROIS.get(dataset_name, slice(None))

    image = load_data(raw_paths[index], raw_key)[roi]
    labels = load_data(label_paths[index], label_key)[roi]

    # The histopathology images are stored as int32 RGB images with values in the range [0, 255].
    if dataset_name == "histopatho":
        image = image.astype("uint8")

    return image, labels


def _download_datasets(path, dataset_name, view=False):
    dataset_names = DATASETS_2D + DATASETS_3D if dataset_name is None else [dataset_name]

    for dname in dataset_names:
        get_dataset_paths(path, dname)
        print(f"'{dname}' is downloaded at {path}.")

        if view:
            image, labels = get_image_and_labels(path, dname)
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(labels)
            napari.run()


def main():
    parser = argparse.ArgumentParser(description="Download the dataset necessary for the workshop.")
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data will be downloaded. "
        "By default, the data will be stored in your current working directory at './data'."
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default=None, choices=DATASETS_2D + DATASETS_3D,
        help="The choice of dataset you would like to download. By default, it downloads all the datasets."
    )
    parser.add_argument("-v", "--view", action="store_true", help="Whether to view the downloaded data.")
    args = parser.parse_args()

    _download_datasets(path=args.input_path, dataset_name=args.dataset_name, view=args.view)


if __name__ == "__main__":
    main()
