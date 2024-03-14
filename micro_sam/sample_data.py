r"""
Sample microscopy data.

You can change the download location for sample data and model weights
by setting the environment variable: MICROSAM_CACHEDIR

By default sample data is downloaded to a folder named 'micro_sam/sample_data'
inside your default cache directory, eg:
    * Mac: ~/Library/Caches/<AppName>
    * Unix: ~/.cache/<AppName> or the value of the XDG_CACHE_HOME environment variable, if defined.
    * Windows: C:\Users\<user>\AppData\Local\<AppAuthor>\<AppName>\Cache

"""

import os
from pathlib import Path
from typing import Union

import imageio.v3 as imageio
import numpy as np
import pooch

from skimage.data import binary_blobs
from skimage.measure import label
from skimage.transform import resize

from .util import get_cache_directory


def fetch_image_series_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample images for the image series annotator.

    Args:
        save_directory: Root folder to save the downloaded data.
    Returns:
        The folder that contains the downloaded data.
    """
    # This sample dataset is currently not provided to napari by the micro-sam
    # plugin, because images are not all the same shape and cannot be combined
    # into a single layer
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "image-series.zip"
    unpack_filenames = [os.path.join("series", f"im{i}.tif") for i in range(3)]
    unpack = pooch.Unzip(members=unpack_filenames)
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/M1zGnfkulWoAhUG/download",
        known_hash="92346ca9770bcaf55248efee590718d54c7135b6ebca15d669f3b77b6afc8706",
        fname=fname,
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    data_folder = os.path.join(save_directory, f"{fname}.unzip", "series")
    assert os.path.exists(data_folder)
    return data_folder


def sample_data_image_series():
    """Provides image series example image to napari.

    Opens as three separate image layers in napari (one per image in series).
    The third image in the series has a different size and modality.
    """
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    base_data_directory = os.path.join(get_cache_directory(), "sample_data")
    data_directory = fetch_image_series_example_data(base_data_directory)
    fnames = os.listdir(data_directory)
    full_filenames = [os.path.join(data_directory, f) for f in fnames]
    full_filenames.sort()
    data_and_image_kwargs = [(imageio.imread(f), {"name": f"img-{i}"}) for i, f in enumerate(full_filenames)]
    return data_and_image_kwargs


def fetch_wholeslide_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample data for the 2d annotator.

    This downloads part of a whole-slide image from the NeurIPS Cell Segmentation Challenge.
    See https://neurips22-cellseg.grand-challenge.org/ for details on the data.

    Args:
        save_directory: Root folder to save the downloaded data.
    Returns:
        The path of the downloaded image.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "whole-slide-example-image.tif"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/6ozPtgBmAAJC1di/download",
        known_hash="3ddb9c9dcc844429932ab951eb0743d5a1af83ee9b0ab54f06ceb2090a606d36",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)


def sample_data_wholeslide():
    """Provides wholeslide 2d example image to napari."""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    base_data_directory = os.path.join(get_cache_directory(), "sample_data")
    filename = fetch_wholeslide_example_data(base_data_directory)
    data = imageio.imread(filename)
    add_image_kwargs = {"name": "wholeslide"}
    return [(data, add_image_kwargs)]


def fetch_livecell_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample data for the 2d annotator.

    This downloads a single image from the LiveCELL dataset.
    See https://doi.org/10.1038/s41592-021-01249-6 for details on the data.

    Args:
        save_directory: Root folder to save the downloaded data.
    Returns:
        The path of the downloaded image.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    fname = "livecell-2d-image.png"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/fSaOJIOYjmFBjPM/download",
        known_hash="4f190983ea672fc333ac26d735d9625d5abb6e4a02bd4d32523127977a31e8fe",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)


def sample_data_livecell():
    """Provides livecell 2d example image to napari."""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    base_data_directory = os.path.join(get_cache_directory(), 'sample_data')
    filename = fetch_livecell_example_data(base_data_directory)
    data = imageio.imread(filename)
    add_image_kwargs = {"name": "livecell"}
    return [(data, add_image_kwargs)]


def fetch_hela_2d_example_data(save_directory: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    """Download the sample data for the 2d annotator.

    This downloads a single image from the HeLa CTC dataset.

    Args:
        save_directory: Root folder to save the downloaded data.
    Returns:
        The path of the downloaded image.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "hela-2d-image.png"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/2sr1DHQ34tV7WEb/download",
        known_hash="908fa00e4b273610aa4e0a9c0f22dfa64a524970852f387908f3fa65238259c7",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)


def sample_data_hela_2d():
    """Provides HeLa 2d example image to napari."""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    base_data_directory = os.path.join(get_cache_directory(), "sample_data")
    filename = fetch_hela_2d_example_data(base_data_directory)
    data = imageio.imread(filename)
    add_image_kwargs = {"name": "hela_2d"}
    return [(data, add_image_kwargs)]


def fetch_3d_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample data for the 3d annotator.

    This downloads the Lucchi++ datasets from https://casser.io/connectomics/.
    It is a dataset for mitochondria segmentation in EM.

    Args:
        save_directory: Root folder to save the downloaded data.
    Returns:
        The folder that contains the downloaded data.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    unpack_filenames = [os.path.join("Lucchi++", "Test_In", f"mask{str(i).zfill(4)}.png") for i in range(165)]
    unpack = pooch.Unzip(members=unpack_filenames)
    fname = "lucchi_pp.zip"
    pooch.retrieve(
        url="http://www.casser.io/files/lucchi_pp.zip",
        known_hash="770ce9e98fc6f29c1b1a250c637e6c5125f2b5f1260e5a7687b55a79e2e8844d",
        fname=fname,
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    lucchi_dir = save_directory.joinpath(f"{fname}.unzip", "Lucchi++", "Test_In")
    return str(lucchi_dir)


def sample_data_3d():
    """Provides Lucchi++ 3d example image to napari."""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    base_data_directory = os.path.join(get_cache_directory(), "sample_data")
    data_directory = fetch_3d_example_data(base_data_directory)
    fnames = os.listdir(data_directory)
    full_filenames = [os.path.join(data_directory, f) for f in fnames]
    full_filenames.sort()
    data = np.stack([imageio.imread(f) for f in full_filenames], axis=0)
    add_image_kwargs = {"name": "lucchi++"}
    return [(data, add_image_kwargs)]


def fetch_tracking_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample data for the tracking annotator.

    This data is the cell tracking challenge dataset DIC-C2DH-HeLa.
    Cell tracking challenge webpage: http://data.celltrackingchallenge.net
    HeLa cells on a flat glass
    Dr. G. van Cappellen. Erasmus Medical Center, Rotterdam, The Netherlands
    Training dataset: http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip (37 MB)
    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/DIC-C2DH-HeLa.zip (41 MB)

    Args:
        save_directory: Root folder to save the downloaded data.
    Returns:
        The folder that contains the downloaded data.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    unpack_filenames = [os.path.join("DIC-C2DH-HeLa", "01", f"t{str(i).zfill(3)}.tif") for i in range(84)]
    unpack = pooch.Unzip(members=unpack_filenames)
    fname = "DIC-C2DH-HeLa.zip"
    pooch.retrieve(
        url="http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip",  # 37 MB
        known_hash="832fed2d05bb7488cf9c51a2994b75f8f3f53b3c3098856211f2d39023c34e1a",
        fname=fname,
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    cell_tracking_dir = save_directory.joinpath(f"{fname}.unzip", "DIC-C2DH-HeLa", "01")
    assert os.path.exists(cell_tracking_dir)
    return str(cell_tracking_dir)


def sample_data_tracking():
    """Provides tracking example dataset to napari."""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    base_data_directory = os.path.join(get_cache_directory(), 'sample_data')
    data_directory = fetch_tracking_example_data(base_data_directory)
    fnames = os.listdir(data_directory)
    full_filenames = [os.path.join(data_directory, f) for f in fnames]
    full_filenames.sort()
    data = np.stack([imageio.imread(f) for f in full_filenames], axis=0)
    add_image_kwargs = {"name": "tracking"}
    return [(data, add_image_kwargs)]


def fetch_tracking_segmentation_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download groundtruth segmentation for the tracking example data.

    This downloads the groundtruth segmentation for the image data from `fetch_tracking_example_data`.

    Args:
        save_directory: Root folder to save the downloaded data.
    Returns:
        The folder that contains the downloaded data.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    unpack_filenames = [os.path.join("masks", f"mask_{str(i).zfill(4)}.tif") for i in range(84)]
    unpack = pooch.Unzip(members=unpack_filenames)
    fname = "hela-ctc-01-gt.zip"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/AWxQMblxwR99OjC/download",
        known_hash="c0644d8ebe1390fb60125560ba15aa2342caf44f50ff0667a0318ea0ac6c958b",
        fname=fname,
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    cell_tracking_dir = save_directory.joinpath(f"{fname}.unzip", "masks")
    assert os.path.exists(cell_tracking_dir)
    return str(cell_tracking_dir)


def sample_data_segmentation():
    """Provides segmentation example dataset to napari."""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    base_data_directory = os.path.join(get_cache_directory(), 'sample_data')
    data_directory = fetch_tracking_segmentation_data(base_data_directory)
    fnames = os.listdir(data_directory)
    full_filenames = [os.path.join(data_directory, f) for f in fnames]
    full_filenames.sort()
    data = np.stack([imageio.imread(f) for f in full_filenames], axis=0)
    add_image_kwargs = {"name": "segmentation"}
    return [(data, add_image_kwargs)]


def synthetic_data(shape, seed=None):
    """Create synthetic image data and segmentation for training."""
    ndim = len(shape)
    assert ndim in (2, 3)
    image_shape = shape if ndim == 2 else shape[1:]
    image = binary_blobs(length=image_shape[0], blob_size_fraction=0.05, volume_fraction=0.15, rng=seed)

    if image_shape[1] != image_shape[0]:
        image = resize(image, image_shape, order=0, anti_aliasing=False, preserve_range=True).astype(image.dtype)
    if ndim == 3:
        nz = shape[0]
        image = np.stack([image] * nz)

    segmentation = label(image)
    image = image.astype("uint8") * 255
    return image, segmentation


def fetch_nucleus_3d_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample data for 3d segmentation of nuclei.

    This data contains a small crop from a volume from the publication
    "Efficient automatic 3D segmentation of cell nuclei for high-content screening"
    https://doi.org/10.1186/s12859-022-04737-4

    Args:
        save_directory: Root folder to save the downloaded data.
    Returns:
        The path of the downloaded image.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "3d-nucleus-data.tif"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/eW0uNCo8gedzWU4/download",
        known_hash="4946896f747dc1c3fc82fb2e1320226d92f99d22be88ea5f9c37e3ba4e281205",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)
