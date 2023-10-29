import os
import numpy as np
from math import ceil, floor

import torch
import torch_em
import torch_em.data.datasets as datasets
from torch_em.transform.label import label_consecutive
from torch_em.data import MinInstanceSampler, ConcatDataset
from torch_em.transform.raw import standardize, normalize_percentile


def neurips_raw_trafo(raw):
    raw = datasets.neurips_cell_seg.to_rgb(raw)  # ensures 3 channels for the neurips data
    raw = normalize_percentile(raw)
    raw = np.mean(raw, axis=0)
    raw = standardize(raw)
    return raw


def tissuenet_raw_trafo(raw, desired_shape=(512, 512)):
    raw = normalize_percentile(raw, axis=(1, 2))
    raw = np.mean(raw, axis=0)
    raw = standardize(raw)

    tmp_ddim = (desired_shape[0] - raw.shape[0], desired_shape[1] - raw.shape[1])
    ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
    raw = np.pad(raw, pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))), mode="reflect")
    assert raw.shape == desired_shape
    return raw


def tissuenet_label_trafo(labels, desired_shape=(512, 512)):
    labels = label_consecutive(labels)

    tmp_ddim = (desired_shape[0] - labels.shape[0], desired_shape[1] - labels.shape[1])
    ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
    labels = np.pad(
        labels,
        pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
        mode="constant"
    )
    assert labels.shape == desired_shape
    return labels


def raw_padding_trafo(raw, desired_shape=(512, 512)):
    tmp_ddim = (desired_shape[0] - raw.shape[0], desired_shape[1] - raw.shape[1])
    ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
    raw = np.pad(raw, pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))), mode="reflect")
    assert raw.shape == desired_shape
    return raw


def get_concat_lm_datasets(input_path, patch_shape, split_choice):
    assert split_choice in ["train", "val"]

    label_dtype = torch.int64
    label_transform = label_consecutive
    sampler = MinInstanceSampler()

    generalist_dataset = ConcatDataset(
        datasets.get_tissuenet_dataset(
            path=os.path.join(input_path, "tissuenet"), split=split_choice, download=True,
            patch_shape=patch_shape if split_choice == "train" else (256, 256), raw_channel="rgb",
            label_channel="cell", raw_transform=tissuenet_raw_trafo, label_transform=tissuenet_label_trafo,
            sampler=sampler, label_dtype=label_dtype, n_samples=1000 if split_choice == "train" else 100
        ),
        datasets.get_livecell_dataset(
            path=os.path.join(input_path, "livecell"), split=split_choice, patch_shape=patch_shape,
            label_transform=label_transform, sampler=sampler, label_dtype=label_dtype,
            n_samples=1000 if split_choice == "train" else 100, download=True
        ),
        datasets.get_deepbacs_dataset(
            path=os.path.join(input_path, "deepbacs"), split=split_choice if split_choice == "train" else "test",
            patch_shape=patch_shape, label_transform=label_transform, sampler=sampler, label_dtype=label_dtype,
            download=True
        ),
        datasets.get_neurips_cellseg_supervised_dataset(
            root=os.path.join(input_path, "neurips-cell-seg"), split=split_choice, patch_shape=patch_shape,
            raw_transform=neurips_raw_trafo, label_transform=label_transform, label_dtype=label_dtype, sampler=sampler
        ),
        datasets.get_dsb_dataset(
            path=os.path.join(input_path, "dsb"), split=split_choice if split_choice == "train" else "test",
            patch_shape=(1, patch_shape[0], patch_shape[1]), label_transform=label_transform, sampler=sampler,
            label_dtype=label_dtype, download=True
        ),
        datasets.get_plantseg_dataset(
            path=os.path.join(input_path, "plantseg"), name="root", sampler=MinInstanceSampler(min_num_instances=10),
            label_transform=tissuenet_label_trafo, ndim=2, split=split_choice, label_dtype=label_dtype,
            raw_transform=raw_padding_trafo, patch_shape=(1, patch_shape[0], patch_shape[1]), download=True,
            n_samples=1000 if split_choice == "train" else 100
        )

    )
    generalist_dataset.datasets[3].max_sampling_attempts = 5000

    return generalist_dataset


def get_generalist_lm_loaders(input_path, patch_shape):
    """This returns the concatenated light microscopy datasets implemented in torch_em:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets
    It will automatically download all the datasets
        - expect NeurIPS CellSeg (Multi-Modal Microscopy Images) (https://neurips22-cellseg.grand-challenge.org/)

    NOTE: to remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    generalist_train_dataset = get_concat_lm_datasets(input_path, patch_shape, "train")
    generalist_val_dataset = get_concat_lm_datasets(input_path, patch_shape, "val")
    train_loader = torch_em.get_data_loader(generalist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(generalist_val_dataset, batch_size=1, shuffle=True, num_workers=16)
    return train_loader, val_loader
