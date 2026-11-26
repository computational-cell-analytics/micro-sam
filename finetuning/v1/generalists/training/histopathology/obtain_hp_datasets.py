import os
import numpy as np
from math import ceil, floor
from typing import Optional, List

from skimage import measure

import torch
import torch.utils.data as data_util

import torch_em
from torch_em.transform.raw import standardize
from torch_em.data import datasets, MinInstanceSampler, ConcatDataset


"""NOTE: test sets for in-domain histopathology evaluation
    - monuseg test split
    - monusac test split
    - bcss test samples (split intrinsically - in the new PR)

length of individual loaders: @all (3 channel input images)
    - lizard:  train - 718;  val - 179
    - bcss:    train - 108;   val - 28
    - monuseg: train - 30;   val - 7
    - monusac: train - 168;   val - 41
    - pannuke: train - 1294;  val - 680
"""


def _get_train_val_split(ds, val_fraction: float = 0.2):
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds


class BCSSLabelTrafo:
    def __init__(self, label_choices: Optional[List[int]] = None, do_connected_components: bool = False):
        self.label_choices = label_choices
        self.do_connected_components = do_connected_components

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Returns the transformed bcss data labels (use-case for SAM)"""
        if self.label_choices is not None:
            labels[~np.isin(labels, self.label_choices)] = 0

        if self.do_connected_components:
            segmentation = measure.label(labels)
        else:
            segmentation = label_consecutive_trafo(labels)

        return segmentation


def raw_padding_trafo(raw, desired_shape=(3, 512, 512)):
    assert raw.shape[0] == 3, "The input shape isn't channels first, expected: (3, H, W)"
    raw = standardize(raw)
    tmp_ddim = (desired_shape[1] - raw.shape[1], desired_shape[2] - raw.shape[2])
    ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
    raw = np.pad(
        raw,
        pad_width=((0, 0), (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
        mode="reflect"
    )
    assert raw.shape == desired_shape
    return raw


def label_padding_trafo(labels, desired_shape=(512, 512)):
    tmp_ddim = (desired_shape[0] - labels.shape[0], desired_shape[1] - labels.shape[1])
    ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
    labels = np.pad(
        labels,
        pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
        mode="reflect"
    )
    assert labels.shape == desired_shape
    labels = label_consecutive_trafo(labels)
    return labels


def label_consecutive_trafo(labels):
    labels = labels.astype(int)
    labels = torch_em.transform.label.label_consecutive(labels)  # to ensure consecutive IDs
    return labels


def get_concat_hp_datasets(path, patch_shape):
    label_dtype = torch.int64
    sampler = MinInstanceSampler(min_num_instances=5)

    # make lizard dataset splits into fractions
    lizard_ds = datasets.get_lizard_dataset(
        path=os.path.join(path, "lizard"), patch_shape=patch_shape, sampler=sampler, label_dtype=label_dtype,
        raw_transform=raw_padding_trafo, label_transform=label_padding_trafo
    )
    lizard_train_ds, lizard_val_ds = _get_train_val_split(ds=lizard_ds)
    lizard_train_ds.ndim = 2
    lizard_val_ds.ndim = 2

    # get bcss internal splits
    bcss_train_ds = datasets.get_bcss_dataset(
        path=os.path.join(path, "bcss"), patch_shape=patch_shape, split="train", sampler=MinInstanceSampler(),
        label_transform=BCSSLabelTrafo(do_connected_components=True), label_dtype=label_dtype
    )
    bcss_val_ds = datasets.get_bcss_dataset(
        path=os.path.join(path, "bcss"), patch_shape=patch_shape, split="val", sampler=MinInstanceSampler(),
        label_transform=BCSSLabelTrafo(do_connected_components=True), label_dtype=label_dtype
    )

    # make monuseg train dataset splits into fractions
    monuseg_ds = datasets.get_monuseg_dataset(
        path=os.path.join(path, "monuseg"), patch_shape=patch_shape, split="train", sampler=sampler,
        label_transform=label_consecutive_trafo, ndim=2, label_dtype=label_dtype
    )
    monuseg_train_ds, monuseg_val_ds = _get_train_val_split(ds=monuseg_ds)

    # make monusac train dataset splits into fractions
    monusac_ds = datasets.get_monusac_dataset(
        path=os.path.join(path, "monusac"), patch_shape=patch_shape, split="train", sampler=MinInstanceSampler(),
        label_transform=label_consecutive_trafo, ndim=2, label_dtype=label_dtype
    )
    monusac_train_ds, monusac_val_ds = _get_train_val_split(ds=monusac_ds)

    # out of three folds (sets of data) of provided data, we use two for training and 1 for validation
    pannuke_train_ds = datasets.get_pannuke_dataset(
        path=os.path.join(path, "pannuke"), patch_shape=(1, *patch_shape), sampler=sampler, folds=["fold_1", "fold_2"],
        label_transform=label_padding_trafo, raw_transform=raw_padding_trafo, ndim=2, label_dtype=label_dtype
    )
    pannuke_val_ds = datasets.get_pannuke_dataset(
        path=os.path.join(path, "pannuke"), patch_shape=(1, *patch_shape), sampler=sampler, folds=["fold_3"],
        label_transform=label_padding_trafo, raw_transform=raw_padding_trafo, ndim=2, label_dtype=label_dtype
    )

    generalist_hp_train_dataset = ConcatDataset(
        lizard_train_ds, bcss_train_ds, monuseg_train_ds, monusac_train_ds, pannuke_train_ds
    )

    generalist_hp_val_dataset = ConcatDataset(
        lizard_val_ds, bcss_val_ds, monuseg_val_ds, monusac_val_ds, pannuke_val_ds
    )

    return generalist_hp_train_dataset, generalist_hp_val_dataset


def get_generalist_hp_loaders(patch_shape, data_path):
    """This returns the concatenated histopathology datasets implemented in `torch_em`:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets
    It will automatically download all the datasets

    NOTE: to remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    generalist_train_dataset, generalist_val_dataset = get_concat_hp_datasets(path=data_path, patch_shape=patch_shape)
    train_loader = torch_em.get_data_loader(generalist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(generalist_val_dataset, batch_size=1, shuffle=True, num_workers=16)
    return train_loader, val_loader
