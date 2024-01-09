import os
import numpy as np

from elf.io import open_file

from skimage.measure import label
from skimage.segmentation import watershed

from torch_em import get_data_loader
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data import ConcatDataset, MinInstanceSampler, datasets

from micro_sam.training import identity
from micro_sam.training.util import ResizeRawTrafo, ResizeLabelTrafo


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


def axondeepseg_label_trafo(labels):
    # after checking, labels look like this : 0 is bg, 1 is myelins and 2 is axons
    foreground_seeds = label((labels == 2))
    boundary_prediction = (labels == 1)
    seg = watershed(boundary_prediction, markers=foreground_seeds, mask=(foreground_seeds + boundary_prediction) > 0)
    dist_trafo = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=0
    )
    seg = dist_trafo(seg)
    return seg


def _check_dataset_available_for_rois(path, patch_shape):
    """This function checks whether or not all the expected datasets are available, else downloads them
    We do this only for "platynereis - cells", "cremi" datasets - as we expect specific RoIs only from them
    """
    datasets.get_cremi_dataset(path=os.path.join(path, "cremi"), patch_shape=patch_shape, download=True)
    datasets.get_platynereis_cell_dataset(
        path=os.path.join(path, "platynereis"), patch_shape=patch_shape, download=True
    )
    print("All the datasets are available for RoI splitting")


def get_concat_boundaries_datasets(input_path, patch_shape):
    _check_dataset_available_for_rois(path=input_path, patch_shape=patch_shape)

    sampler = MinInstanceSampler()
    standard_label_trafo = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=0
    )

    # cremi dataset parameters
    cremi_train_rois = {"A": np.s_[0:75, :, :], "B": np.s_[0:75, :, :], "C": np.s_[0:75, :, :]}
    cremi_val_rois = {"A": np.s_[75:100, :, :], "B": np.s_[75:100, :, :], "C": np.s_[75:100, :, :]}

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

    def cremi_dataset(rois):
        return datasets.get_cremi_dataset(
            path=os.path.join(input_path, "cremi"), patch_shape=patch_shape, label_transform=standard_label_trafo,
            rois=rois, sampler=sampler, ndim=2, defect_augmentation_kwargs=None, download=True, raw_transform=identity
        )

    cremi_train_dataset = cremi_dataset(cremi_train_rois)
    cremi_val_dataset = cremi_dataset(cremi_val_rois)

    def platy_cell_dataset(rois, sample_ids):
        return datasets.get_platynereis_cell_dataset(
            path=platy_root, sample_ids=sample_ids, patch_shape=patch_shape, download=True, sampler=sampler,
            ndim=2, rois=rois, raw_transform=ResizeRawTrafo(patch_shape[1:], do_rescaling=False),
            label_transform=ResizeLabelTrafo(patch_shape[1:])
        )

    platy_cell_train_dataset = platy_cell_dataset(platy_cell_train_rois, platy_cell_train_samples)
    platy_cell_val_dataset = platy_cell_dataset(platy_cell_val_rois, platy_cell_val_samples)

    def axondeepseg_dataset(split, data_fraction):
        return datasets.get_axondeepseg_dataset(
            path=os.path.join(input_path, "axondeepseg"), name=["sem"], patch_shape=patch_shape[1:],
            label_transform=axondeepseg_label_trafo, sampler=sampler, data_fraction=data_fraction,
            raw_transform=identity, split=split, download=True
        )

    axondeepseg_train_dataset = axondeepseg_dataset("train", 0.9)
    axondeepseg_val_dataset = axondeepseg_dataset("val", 0.1)

    train_datasets = [cremi_train_dataset, platy_cell_train_dataset, axondeepseg_train_dataset]
    val_datasets = [cremi_val_dataset, platy_cell_val_dataset, axondeepseg_val_dataset]

    generalist_em_train_dataset = ConcatDataset(*train_datasets)
    generalist_em_val_dataset = ConcatDataset(*val_datasets)

    return generalist_em_train_dataset, generalist_em_val_dataset


def get_generalist_boundaries_loaders(input_path, patch_shape):
    """This returns the concatenated electron microscopy datasets implemented in torch_em:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets
    It will automatically download all the datasets
    NOTE: to remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    generalist_train_dataset, generalist_val_dataset = get_concat_boundaries_datasets(input_path, patch_shape)
    train_loader = get_data_loader(generalist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = get_data_loader(generalist_val_dataset, batch_size=1, shuffle=True, num_workers=16)
    return train_loader, val_loader
