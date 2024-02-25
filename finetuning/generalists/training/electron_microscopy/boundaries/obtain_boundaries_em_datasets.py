import os
import numpy as np

from elf.io import open_file

from skimage.measure import label
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

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

    # use the distance to the myelinated axons as height map to assign pixels to nearest myelinated axon
    hmap = distance_transform_edt(labels != 2)
    seg = watershed(image=hmap, markers=foreground_seeds, mask=(foreground_seeds + boundary_prediction) > 0)

    dist_trafo = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=0
    )
    seg = dist_trafo(seg)
    return seg


def _check_dataset_available_for_rois(path, patch_shape):
    """This function checks whether or not all the expected datasets are available, else downloads them
    We do this for "platynereis - cells", "cremi", "platynereis - nuclei", "mitoem" datasets
        - as we expect specific RoIs only from them
    """
    datasets.get_cremi_dataset(path=os.path.join(path, "cremi"), patch_shape=patch_shape, download=True)
    datasets.get_platynereis_cell_dataset(
        path=os.path.join(path, "platynereis"), patch_shape=patch_shape, download=True
    )
    datasets.get_mitoem_dataset(
        path=os.path.join(path, "mitoem"), patch_shape=patch_shape, download=True, splits="train"
    )
    datasets.get_platynereis_nuclei_dataset(
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

    # platynereis nuclei dataset parameters
    platy_root = os.path.join(input_path, "platynereis")
    platy_nuclei_template = "nuclei/train_data_nuclei_%02i.h5"
    platy_nuclei_label_key = "volumes/labels/nucleus_instance_labels"

    platy_nuclei_train_samples = [1, 2, 3, 4, 5, 6, 7, 8]
    platy_nuclei_train_rois = compute_platy_rois(platy_root, platy_nuclei_train_samples, ignore_label=-1,
                                                 file_template=platy_nuclei_template, label_key=platy_nuclei_label_key)
    platy_nuclei_val_samples = [9, 10]
    platy_nuclei_val_rois = compute_platy_rois(platy_root, platy_nuclei_val_samples, ignore_label=-1,
                                               file_template=platy_nuclei_template, label_key=platy_nuclei_label_key)

    # mitoem parameters
    mitoem_train_rois = [np.s_[100:110, :, :], np.s_[100:110, :, :]]
    mitoem_val_rois = [np.s_[0:5, :, :], np.s_[0:5, :, :]]

    def axondeepseg_dataset(split):
        # train is oversampled by ~10 times and val by ~15 times
        n_samples = 500 if split == "train" else 100
        return datasets.get_axondeepseg_dataset(
            path=os.path.join(input_path, "axondeepseg"), name=["sem"], patch_shape=patch_shape[1:],
            label_transform=axondeepseg_label_trafo, sampler=sampler, split=split,
            raw_transform=identity, download=True, val_fraction=0.1, n_samples=n_samples
        )

    axondeepseg_train_dataset = axondeepseg_dataset("train")
    axondeepseg_val_dataset = axondeepseg_dataset("val")

    def cremi_dataset(rois, n_samples):
        return datasets.get_cremi_dataset(
            path=os.path.join(input_path, "cremi"), patch_shape=patch_shape, label_transform=standard_label_trafo,
            n_samples=n_samples, rois=rois, sampler=sampler, ndim=2, defect_augmentation_kwargs=None,
            download=True, raw_transform=identity
        )

    cremi_train_dataset = cremi_dataset(cremi_train_rois, n_samples=750)  # taking ~50% of all training samples
    cremi_val_dataset = cremi_dataset(cremi_val_rois, n_samples=250)  # # taking ~50% of all val samples

    def platy_cell_dataset(rois, sample_ids):
        return datasets.get_platynereis_cell_dataset(
            path=platy_root, sample_ids=sample_ids, patch_shape=patch_shape, download=True, sampler=sampler,
            ndim=2, rois=rois, raw_transform=ResizeRawTrafo(patch_shape[1:], do_rescaling=False),
            label_transform=ResizeLabelTrafo(patch_shape[1:])
        )

    platy_cell_train_dataset = platy_cell_dataset(platy_cell_train_rois, platy_cell_train_samples)
    platy_cell_val_dataset = platy_cell_dataset(platy_cell_val_rois, platy_cell_val_samples)

    def mitoem_dataset(split, roi_choice):
        return datasets.get_mitoem_dataset(
            path=os.path.join(input_path, "mitoem"), splits=split, download=True, patch_shape=patch_shape,
            rois=roi_choice, label_transform=standard_label_trafo, ndim=2, raw_transform=identity,
            sampler=MinInstanceSampler(min_num_instances=5)
        )

    mitoem_train_dataset = mitoem_dataset("train", mitoem_train_rois)
    mitoem_val_dataset = mitoem_dataset("val", mitoem_val_rois)

    def platy_nuclei_dataset(roi_choice, sample_ids):
        return datasets.get_platynereis_nuclei_dataset(
            path=platy_root, patch_shape=patch_shape, download=True, sampler=sampler, ndim=2,
            label_transform=ResizeLabelTrafo(patch_shape[1:]), rois=roi_choice,
            raw_transform=ResizeRawTrafo(patch_shape[1:], do_rescaling=False), sample_ids=sample_ids
        )

    platy_nuclei_train_dataset = platy_nuclei_dataset(platy_nuclei_train_rois, sample_ids=platy_nuclei_train_samples)
    platy_nuclei_val_dataset = platy_nuclei_dataset(platy_nuclei_val_rois, sample_ids=platy_nuclei_val_samples)

    def cem_dataset(split):
        # 10% of the total training set, 1/3 of the total val set
        n_samples = 1620 if split == "train" else 600
        return datasets.cem.get_mitolab_dataset(
            path=os.path.join(input_path, "mitolab"), split=split, val_fraction=0.1, sampler=sampler,
            raw_transform=ResizeRawTrafo(patch_shape[1:], do_rescaling=False), patch_shape=patch_shape[1:],
            label_transform=ResizeLabelTrafo(patch_shape[1:]), n_samples=n_samples
        )

    cem_train_dataset = cem_dataset("train")
    cem_val_dataset = cem_dataset("val")

    train_datasets = [
        axondeepseg_train_dataset, cremi_train_dataset, platy_cell_train_dataset,
        mitoem_train_dataset, platy_nuclei_train_dataset, cem_train_dataset
    ]
    val_datasets = [
        axondeepseg_val_dataset, cremi_val_dataset, platy_cell_val_dataset,
        mitoem_val_dataset, platy_nuclei_val_dataset, cem_val_dataset
    ]

    for train_dataset in train_datasets:
        train_dataset.max_sampling_attempts = 5000

    for val_dataset in val_datasets:
        val_dataset.max_sampling_attempts = 5000

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
    train_loader = get_data_loader(generalist_train_dataset, batch_size=2, num_workers=16, shuffle=True)
    val_loader = get_data_loader(generalist_val_dataset, batch_size=1, num_workers=16, shuffle=True)
    return train_loader, val_loader
