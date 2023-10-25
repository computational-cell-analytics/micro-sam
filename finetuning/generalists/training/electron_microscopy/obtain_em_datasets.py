import os
import numpy as np
from math import ceil, floor

from elf.io import open_file
from skimage.measure import label
from skimage.segmentation import watershed

from torch_em import get_data_loader
from torch_em.transform.label import label_consecutive
from torch_em.data import ConcatDataset, MinInstanceSampler, datasets


def axondeepseg_label_trafo(labels):
    # after checking, labels look like this : 0 is bg, 1 is myelins and 2 is axons
    foreground_seeds = label((labels == 2))
    boundary_prediction = (labels == 1)
    seg = watershed(boundary_prediction, markers=foreground_seeds, mask=(foreground_seeds + boundary_prediction) > 0)
    seg = label_consecutive(seg)
    return seg


def raw_trafo_for_padding(raw):
    desired_shape = (512, 512)
    tmp_ddim = (desired_shape[0] - raw.shape[0], desired_shape[1] - raw.shape[1])
    ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
    raw = np.pad(raw,
                 pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
                 mode="reflect")
    assert raw.shape == desired_shape
    return raw


def label_trafo_for_padding(labels):
    desired_shape = (512, 512)
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
    We do this only for "platynereis", "cremi" and "mitoem" datasets - as we expect specific RoIs only from them
    """
    datasets.get_cremi_dataset(path=os.path.join(path, "cremi"), patch_shape=patch_shape, download=True)
    datasets.get_mitoem_dataset(path=os.path.join(path, "mitoem"), patch_shape=patch_shape, download=True, splits="train")
    datasets.get_platynereis_cell_dataset(path=os.path.join(path, "platynereis"), patch_shape=patch_shape, download=True)
    datasets.get_platynereis_nuclei_dataset(path=os.path.join(path, "platynereis"), patch_shape=patch_shape, download=True)
    datasets.get_platynereis_cilia_dataset(path=os.path.join(path, "platynereis"), patch_shape=patch_shape, download=True)
    print("All the datasets are available for RoI splitting")


def get_concat_em_datasets(input_path, patch_shape):
    _check_dataset_available_for_rois(path=input_path, patch_shape=patch_shape)

    sampler = MinInstanceSampler()

    # cremi dataset parameters
    cremi_train_rois = {"A": np.s_[0:75, :, :], "B": np.s_[0:75, :, :], "C": np.s_[0:75, :, :]}
    cremi_val_rois = {"A": np.s_[75:100, :, :], "B": np.s_[75:100, :, :], "C": np.s_[75:100, :, :]}

    # mitoem parameters
    mitoem_train_rois = [np.s_[100:120, :, :], np.s_[100:120, :, :]]
    mitoem_val_rois = [np.s_[0:20, :, :], np.s_[0:20, :, :]]

    # platynereis cell dataset parameters
    platy_root = os.path.join(input_path, "platynereis")
    platy_cell_template = "membrane/train_data_membrane_%02i.n5"
    platy_cell_label_key = "volumes/labels/segmentation/s1"

    platy_cell_train_samples = [1, 2, 3, 4, 5, 6]
    platy_cell_train_rois = compute_platy_rois(platy_root, platy_cell_train_samples, ignore_label=0,
                                               file_template=platy_cell_template, label_key=platy_cell_label_key)
    platy_cell_val_samples = [7, 8]
    platy_cell_val_rois = compute_platy_rois(platy_root, platy_cell_val_samples, ignore_label=0,
                                             file_template=platy_cell_template, label_key=platy_cell_label_key)

    # platynereis cilia dataset parameters
    platy_cilia_template = "cilia/train_data_cilia_%02i.h5"
    platy_cilia_label_key = "volumes/labels/segmentation"

    platy_cilia_train_samples = [1, 2]
    platy_cilia_train_rois = compute_platy_rois(platy_root, platy_cilia_train_samples, ignore_label=-1,
                                                file_template=platy_cilia_template, label_key=platy_cilia_label_key)
    platy_cilia_val_samples = [3]
    platy_cilia_val_rois = compute_platy_rois(platy_root, platy_cilia_val_samples, ignore_label=-1,
                                              file_template=platy_cilia_template, label_key=platy_cilia_label_key)

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
        datasets.get_cremi_dataset(
            path=os.path.join(input_path, "cremi"), patch_shape=patch_shape, download=True,
            label_transform=standard_label_trafo, rois=cremi_train_rois, sampler=sampler, ndim=2,
            defect_augmentation_kwargs=None
        ),
        datasets.get_platynereis_cell_dataset(
            path=platy_root, sample_ids=platy_cell_train_samples, patch_shape=patch_shape,
            download=True, sampler=sampler, ndim=2, label_transform=label_trafo_for_padding,
            rois=platy_cell_train_rois, raw_transform=raw_trafo_for_padding
        ),
        datasets.get_platynereis_cilia_dataset(
            path=platy_root, download=True, patch_shape=patch_shape, ndim=2, rois=platy_cilia_train_rois,
            raw_transform=raw_trafo_for_padding, label_transform=label_trafo_for_padding, sampler=sampler,
            sample_ids=platy_cilia_train_samples
        ),
        datasets.get_mitoem_dataset(
            path=os.path.join(input_path, "mitoem"), splits="train", download=True, patch_shape=patch_shape,
            rois=mitoem_train_rois, label_transform=standard_label_trafo, ndim=2,
            sampler=MinInstanceSampler(min_num_instances=5)
        ),
        datasets.get_axondeepseg_dataset(
            path=os.path.join(input_path, "axondeepseg"), name=["sem"], patch_shape=patch_shape[1:], download=True,
            label_transform=axondeepseg_label_trafo, sampler=sampler, data_fraction=0.9, split="train"
        ),
        datasets.get_uro_cell_dataset(
            path=os.path.join(input_path, "uro_cell"), target="mito", patch_shape=patch_shape, download=True,
            sampler=sampler, label_transform=label_trafo_for_padding, ndim=2, raw_transform=raw_trafo_for_padding
        ),
        datasets.get_platynereis_nuclei_dataset(
            path=platy_root, patch_shape=patch_shape, download=True, sampler=sampler, ndim=2,
            label_transform=label_trafo_for_padding, rois=platy_nuclei_train_rois, raw_transform=raw_trafo_for_padding,
            sample_ids=platy_nuclei_train_samples
        )
    )

    generalist_em_val_dataset = ConcatDataset(
        datasets.get_cremi_dataset(
            path=os.path.join(input_path, "cremi"), patch_shape=patch_shape,
            download=True, label_transform=standard_label_trafo, rois=cremi_val_rois,
            sampler=sampler, ndim=2, defect_augmentation_kwargs=None
        ),
        datasets.get_platynereis_cell_dataset(
            path=platy_root, sample_ids=platy_cell_val_samples, patch_shape=patch_shape,
            download=True, sampler=sampler, ndim=2, label_transform=label_trafo_for_padding,
            rois=platy_cell_val_rois, raw_transform=raw_trafo_for_padding
        ),
        datasets.get_platynereis_cilia_dataset(
            path=platy_root, download=True, patch_shape=patch_shape, ndim=2, rois=platy_cilia_val_rois,
            raw_transform=raw_trafo_for_padding, label_transform=label_trafo_for_padding, sampler=sampler,
            sample_ids=platy_cilia_val_samples
        ),
        datasets.get_mitoem_dataset(
            path=os.path.join(input_path, "mitoem"), splits="val", download=True, patch_shape=patch_shape,
            rois=mitoem_val_rois, label_transform=standard_label_trafo, ndim=2,
            sampler=MinInstanceSampler(min_num_instances=5)
        ),
        datasets.get_axondeepseg_dataset(
            path=os.path.join(input_path, "axondeepseg"), name=["sem"], patch_shape=patch_shape[1:], download=True,
            label_transform=axondeepseg_label_trafo, sampler=sampler, data_fraction=0.1, split="val"
        ),
        datasets.get_platynereis_nuclei_dataset(
            path=platy_root, patch_shape=patch_shape, download=True, sampler=sampler, ndim=2,
            label_transform=label_trafo_for_padding, rois=platy_nuclei_val_rois, raw_transform=raw_trafo_for_padding,
            sample_ids=platy_nuclei_val_samples
        )
    )

    return generalist_em_train_dataset, generalist_em_val_dataset


def get_generalist_em_loaders(input_path, patch_shape):
    """This returns the concatenated electron microscopy datasets implemented in torch_em:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets
    It will automatically download all the datasets
    NOTE: to remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    generalist_train_dataset, generalist_val_dataset = get_concat_em_datasets(input_path, patch_shape)
    train_loader = get_data_loader(generalist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = get_data_loader(generalist_val_dataset, batch_size=1, shuffle=True, num_workers=16)
    return train_loader, val_loader
