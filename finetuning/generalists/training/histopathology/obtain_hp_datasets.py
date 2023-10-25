import os

import torch_em
from torch_em.data import datasets
import torch.utils.data as data_util
from torch_em.data.sampler import MinInstanceSampler


# TODO use other datasets than lizard
# need to add: pannuke, bcss, monuseg, monusac
def get_generalist_hp_loaders(patch_shape, data_path):
    label_transform = torch_em.transform.label.label_consecutive  # to ensure consecutive IDs
    sampler = MinInstanceSampler(min_num_instances=5)
    dataset = datasets.get_lizard_dataset(
        path=os.path.join(data_path, "lizard"), download=False, patch_shape=patch_shape,
        label_transform=label_transform, sampler=sampler
    )
    train_ds, val_ds = data_util.random_split(dataset, [0.9, 0.1])
    train_loader = torch_em.get_data_loader(train_ds, batch_size=1)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=1)
    return train_loader, val_loader
