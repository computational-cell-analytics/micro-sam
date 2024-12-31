import os

import numpy as np
from sklearn.model_selection import train_test_split

import torch

import torch_em
from torch_em.data import datasets
from torch_em.transform.raw import normalize
from torch_em.data import MinInstanceSampler, ConcatDataset
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb

from micro_sam.training.util import ResizeRawTrafo, ResizeLabelTrafo


def _to_8bit(raw):
    "Ensures three channels for inputs and rescale them to 8 bit."
    if raw.ndim == 2:
        raw = to_rgb(raw)  # Ensure all images are in 3-channels: triplicate one channel to three channels.
    else:
        if raw.shape[0] != 3:
            assert raw.shape[0] == 1, raw.shape
            raw = np.concatenate([raw] * 3, axis=0)

    raw = normalize(raw) * 255
    return raw


def _identity(x):
    "Ensures three channels for inputs and avoids rescaling inputs."
    x = to_rgb(x)
    return x


def _cellpose_raw_trafo(x):
    """Transforms input images to desired format.
    NOTE: The input channel logic is arranged a bit strangely in `cyto` dataset.
    We take care of it here.
    """
    r, g, b = x

    assert g.max() != 0
    if r.max() == 0:
        # The image is 1 channel and exists in green channel only.
        assert b.max() == 0
        x = np.concatenate([g[None]] * 3, axis=0)

    elif r.max() != 0 and g.max() != 0:
        # The image is 2 channels and we sort the channels such that - 0: cell, 1: nucleus
        x = np.stack([g, r, np.zeros_like(b)], axis=0)

    x = to_rgb(x)  # Ensures three channels for inputs and avoids rescaling inputs.

    return x


def get_concat_lm_datasets(input_path, patch_shape, split_choice):
    assert split_choice in ["train", "val"]

    label_dtype = torch.float32
    sampler = MinInstanceSampler(min_size=10)

    def _get_label_transform(min_size=10):
        label_transform = PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True,
            min_size=min_size
        )
        return label_transform

    def get_livecell_datasets():
        "Datasets for cell segmentation in phase contrast microscopy images."
        all_livecell_datasets = [
            datasets.get_livecell_dataset(
                path=os.path.join(input_path, "livecell"), split=split_choice, patch_shape=patch_shape,
                sampler=sampler, label_dtype=label_dtype, raw_transform=_identity, download=True, cell_types=[ctype],
                label_transform=_get_label_transform(), n_samples=200 if split_choice == "train" else 50,
            ) for ctype in datasets.livecell.CELL_TYPES
        ]
        return all_livecell_datasets

    def get_embedseg_datasets():
        "Datasets for cell and nuclei segmentation in fluorescence microscopy images."
        names = [
            "Mouse-Organoid-Cells-CBG",
            "Mouse-Skull-Nuclei-CBG",
            "Platynereis-ISH-Nuclei-CBG",
            "Platynereis-Nuclei-CBG",
        ]
        all_embedseg_datasets = [
            datasets.get_embedseg_dataset(
                path=os.path.join(input_path, "embedseg"), name=name, patch_shape=(1, *patch_shape),
                download=True, n_samples=500 if split_choice == "train" else 100, raw_transform=_to_8bit,
                label_dtype=label_dtype, label_transform=_get_label_transform(), ndim=2,
                sampler=MinInstanceSampler(min_num_instances=3, min_size=10),
            ) for name in names
        ]
        return all_embedseg_datasets

    def get_yeaz_dataset():
        "Datasets for yeast segmentation in phase contrast and brightfield microscopy images."
        names = ["bf", "phc"]
        all_yeaz_datasets = [
            datasets.get_yeaz_dataset(
                path=os.path.join(input_path, "yeaz"), patch_shape=patch_shape, raw_transform=_to_8bit,
                ndim=2, download=True, split=split_choice, choice=name, label_transform=_get_label_transform(),
                sampler=sampler, label_dtype=label_dtype,
            ) for name in names
        ]
        return all_yeaz_datasets

    def get_cvz_dataset(stain_choice):
        "Datasets for cell and nuclei segmentation in fluorescence microscopy images."
        # NOTE: We create random splits for this dataset for training the generalist.
        raw_paths, label_paths = datasets.cvz_fluo.get_cvz_fluo_paths(
            path=os.path.join(input_path, "cvz"), stain_choice=stain_choice,
        )
        train_raw_paths, test_raw_paths, train_label_paths, test_label_paths = train_test_split(
            raw_paths, label_paths, test_size=0.2, random_state=42,
        )
        ds = torch_em.default_segmentation_dataset(
            raw_paths=train_raw_paths if split_choice == "train" else test_raw_paths, raw_key=None,
            label_paths=train_label_paths if split_choice == "train" else test_label_paths,
            label_key=None, is_seg_dataset=False, patch_shape=patch_shape, sampler=sampler,
            raw_transform=_identity, label_transform=_get_label_transform(), label_dtype=label_dtype,
        )
        return [ds]

    def get_ctc_datasets():
        "Datasets for cell segmentation in different modalities."
        all_ctc_datasets = []
        for dataset_name in datasets.ctc.CTC_CHECKSUMS["train"].keys():
            if dataset_name in ["Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa"]:
                continue

            all_ctc_datasets.append(
                datasets.get_ctc_segmentation_dataset(
                    path=os.path.join(input_path, "ctc"), dataset_name=dataset_name, patch_shape=(1, *patch_shape),
                    sampler=sampler, raw_transform=_to_8bit, label_transform=_get_label_transform(),
                    download=True, label_dtype=label_dtype,
                )
            )
        return all_ctc_datasets

    _datasets = [
        # cell segmentation in tissue microscopy images.
        datasets.get_tissuenet_dataset(
            path=os.path.join(input_path, "tissuenet"), split=split_choice, download=True, patch_shape=patch_shape,
            raw_channel="rgb", label_channel="cell", raw_transform=ResizeRawTrafo((3, *patch_shape), do_rescaling=True),
            label_transform=ResizeLabelTrafo(patch_shape, min_size=10), sampler=sampler, label_dtype=label_dtype,
            n_samples=500 if split_choice == "train" else 100,
        ),
        # bacteria segmentation in label-free microscopy images.
        datasets.get_deepbacs_dataset(
            path=os.path.join(input_path, "deepbacs"), split=split_choice, patch_shape=patch_shape,
            raw_transform=_to_8bit, label_transform=_get_label_transform(), label_dtype=label_dtype,
            download=True, sampler=MinInstanceSampler(min_num_instances=4, min_size=10)
        ),
        # cell segmentation in confocal microscopy images.
        datasets.get_plantseg_dataset(
            path=os.path.join(input_path, "plantseg"), name="root", n_samples=500 if split_choice == "train" else 100,
            patch_shape=(1, *patch_shape), download=True, ndim=2, raw_transform=ResizeRawTrafo((3, *patch_shape)),
            sampler=MinInstanceSampler(min_num_instances=4, min_size=10), split=split_choice, label_dtype=label_dtype,
            label_transform=ResizeLabelTrafo(patch_shape, min_size=10),
        ),
        # cell segmentation in multi-modal microscopy images.
        datasets.get_neurips_cellseg_supervised_dataset(
            root=os.path.join(input_path, "neurips_cellseg"), split=split_choice, label_dtype=label_dtype,
            patch_shape=patch_shape, raw_transform=_to_8bit, label_transform=_get_label_transform(),
            sampler=MinInstanceSampler(min_num_instances=3, min_size=10), download=True,
        ),
        # nuclei segmentation in fluorescence microscopy images.
        datasets.get_dsb_dataset(
            path=os.path.join(input_path, "dsb"), split=split_choice if split_choice == "train" else "test",
            patch_shape=patch_shape, label_transform=_get_label_transform(), sampler=sampler,
            label_dtype=label_dtype, download=True, raw_transform=_identity,
        ),
        # nuclei segmentation in fluorescence microscopy images.
        datasets.get_dynamicnuclearnet_dataset(
            path=os.path.join(input_path, "dynamicnuclearnet"), patch_shape=patch_shape, download=True, sampler=sampler,
            split=split_choice, n_samples=500 if split_choice == "train" else 100, label_dtype=label_dtype,
            raw_transform=_to_8bit, label_transform=_get_label_transform(),
        ),
        # cell segmentation in multiple microscopy imaging modalities.
        datasets.get_cellpose_dataset(
            path=os.path.join(input_path, "cellpose"), patch_shape=patch_shape, choice="cyto", sampler=sampler,
            download=True, split=split_choice if split_choice == "train" else "test", label_dtype=label_dtype,
            label_transform=_get_label_transform(), raw_transform=_cellpose_raw_trafo,
        ),
        # bacteria segmentation in phase contrast and fluorescence microscopy images.
        # worm segmentation in brightfield microscopy images.
        datasets.get_omnipose_dataset(
            path=os.path.join(input_path, "omnipose"), patch_shape=patch_shape, download=True,
            split=split_choice if split_choice == "train" else "test", sampler=sampler,
            label_dtype=label_dtype, raw_transform=_to_8bit, label_transform=_get_label_transform(),
        ),
        # organoid segmentation in brightfield microscopy images.
        datasets.get_orgasegment_dataset(
            path=os.path.join(input_path, "orgasegment"), patch_shape=patch_shape, download=True, split=split_choice,
            raw_transform=_identity, label_transform=_get_label_transform(), label_dtype=label_dtype, sampler=sampler,
        ),
    ]

    # Add LIVECell dataset: cell segmentation for phase contrast microscopy images.
    _datasets.extend(get_livecell_datasets())

    # Add EmbedSeg datasets: cell and nuclei segmentation for fluorescence microscopy images.
    _datasets.extend(get_embedseg_datasets())

    # Add YeaZ datasets: yeast segmentation for brightfield and phase contrast microscopy images.
    _datasets.extend(get_yeaz_dataset())

    # Add CVZ Fluo datasets: cell and nuclei segmentation for fluorescence microscopy images.
    _datasets.extend(get_cvz_dataset("cell"))
    _datasets.extend(get_cvz_dataset("dapi"))

    # Add CTC datasets: cell segmentation for various
    if split_choice == "train":  # NOTE: We use CTC only for training.
        _datasets.extend(get_ctc_datasets())

    generalist_dataset = ConcatDataset(*_datasets)

    # Increasing the sampling attempts for the NeurIPS CellSeg dataset.
    generalist_dataset.datasets[3].max_sampling_attempts = 5000

    return generalist_dataset


def get_generalist_lm_loaders(input_path, patch_shape):
    """This returns the concatenated light microscopy datasets implemented in `torch_em`:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets/light_microscopy.
    It will automatically download all the datasets, except:
    - TissueNet (see `torch_em/data/datasets/light_microscopy/tissuenet.py` for details)
    - DynamicNuclearNet (see `torch_em/data/datasets/light_microscopy/dynamicnuclearnet.py` for details)
    - CellPose (see `torch_em/data/datasets/light_microscopy/cellpose.py` for details)
    - YeaZ (see `torch_em/data/datasets/light_microscopy/yeaz.py` for details)

    NOTE: To remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format,
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    # Get the datasets.
    generalist_train_dataset = get_concat_lm_datasets(input_path, patch_shape, "train")
    generalist_val_dataset = get_concat_lm_datasets(input_path, patch_shape, "val")

    # Get the dataloaders.
    train_loader = torch_em.get_data_loader(generalist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(generalist_val_dataset, batch_size=1, shuffle=True, num_workers=16)

    return train_loader, val_loader
