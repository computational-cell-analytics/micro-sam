import os
import numpy as np
from math import ceil, floor

from elf.io import open_file
from skimage.measure import label

from torch_em import get_data_loader
from torch_em.transform.raw import standardize
from torch_em.transform.label import label_consecutive
from torch_em.data import ConcatDataset, MinInstanceSampler, datasets


def raw_trafo_for_padding(raw, desired_shape=(512, 512)):
    raw = standardize(raw)
    tmp_ddim = (desired_shape[0] - raw.shape[0], desired_shape[1] - raw.shape[1])
    ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
    raw = np.pad(raw,
                 pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
                 mode="reflect")
    assert raw.shape == desired_shape
    return raw


def label_trafo_for_padding(labels, desired_shape=(512, 512)):
    labels = label(labels)
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


def standard_label_trafo(labels):
    labels = label(labels)
    labels = label_consecutive(labels)
    return labels


def compute_platy_rois(root, sample_ids, ignore_label, file_template, label_key):
    rois = {}
    for sample_id in sample_ids:
        path = os.path.join(root, (file_template % sample_id))
        with open_file(path, "r") as f:
            labels = f[label_key][:]
        valid_coordinates = np.where(labels != ignore_label)
        roi = tuple(slice(
            int(coord.min()), int(coord.max()) + 1
        ) for coord in valid_coordinates)
        rois[sample_id] = roi
    return rois


def _check_dataset_available_for_rois(path, patch_shape):
    """This function checks whether or not all the expected datasets are available, else downloads them
    We do this only for "platynereis - nuclei", "mitoem" datasets - as we expect specific RoIs only from them
    """
    datasets.get_mitoem_dataset(
        path=os.path.join(path, "mitoem"), patch_shape=patch_shape, download=True, splits="train"
    )
    datasets.get_platynereis_nuclei_dataset(
        path=os.path.join(path, "platynereis"), patch_shape=patch_shape, download=True
    )
    print("All the datasets are available for RoI splitting")


def get_concat_mito_nuc_datasets(input_path, patch_shape):
    _check_dataset_available_for_rois(path=input_path, patch_shape=patch_shape)

    sampler = MinInstanceSampler()

    # mitoem parameters
    mitoem_train_rois = [np.s_[100:120, :, :], np.s_[100:120, :, :]]
    mitoem_val_rois = [np.s_[0:20, :, :], np.s_[0:20, :, :]]

    # platynereis cell dataset parameters
    platy_root = os.path.join(input_path, "platynereis")

    # platynereis nuclei dataset parameters
    platy_nuclei_template = "nuclei/train_data_nuclei_%02i.h5"
    platy_nuclei_label_key = "volumes/labels/nucleus_instance_labels"

    platy_nuclei_train_samples = [1, 2, 3, 4, 5, 6, 7, 8]
    platy_nuclei_train_rois = compute_platy_rois(platy_root, platy_nuclei_train_samples, ignore_label=-1,
                                                 file_template=platy_nuclei_template, label_key=platy_nuclei_label_key)
    platy_nuclei_val_samples = [9, 10]
    platy_nuclei_val_rois = compute_platy_rois(platy_root, platy_nuclei_val_samples, ignore_label=-1,
                                               file_template=platy_nuclei_template, label_key=platy_nuclei_label_key)

    generalist_em_train_dataset = ConcatDataset(
        datasets.get_mitoem_dataset(
            path=os.path.join(input_path, "mitoem"), splits="train", download=True, patch_shape=patch_shape,
            rois=mitoem_train_rois, label_transform=standard_label_trafo, ndim=2,
            sampler=MinInstanceSampler(min_num_instances=5)
        ),
        datasets.get_platynereis_nuclei_dataset(
            path=platy_root, patch_shape=patch_shape, download=True, sampler=sampler, ndim=2,
            label_transform=label_trafo_for_padding, rois=platy_nuclei_train_rois, raw_transform=raw_trafo_for_padding,
            sample_ids=platy_nuclei_train_samples
        )
    )

    generalist_em_val_dataset = ConcatDataset(
        datasets.get_mitoem_dataset(
            path=os.path.join(input_path, "mitoem"), splits="val", download=True, patch_shape=patch_shape,
            rois=mitoem_val_rois, label_transform=standard_label_trafo, ndim=2,
            sampler=MinInstanceSampler(min_num_instances=5)
        ),
        datasets.get_platynereis_nuclei_dataset(
            path=platy_root, patch_shape=patch_shape, download=True, sampler=sampler, ndim=2,
            label_transform=label_trafo_for_padding, rois=platy_nuclei_val_rois, raw_transform=raw_trafo_for_padding,
            sample_ids=platy_nuclei_val_samples
        )
    )

    return generalist_em_train_dataset, generalist_em_val_dataset


def get_generalist_mito_nuc_loaders(input_path, patch_shape):
    """This returns the concatenated electron microscopy datasets implemented in torch_em:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets
    It will automatically download all the datasets
    NOTE: to remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    generalist_train_dataset, generalist_val_dataset = get_concat_mito_nuc_datasets(input_path, patch_shape)
    train_loader = get_data_loader(generalist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = get_data_loader(generalist_val_dataset, batch_size=1, shuffle=True, num_workers=16)
    return train_loader, val_loader
