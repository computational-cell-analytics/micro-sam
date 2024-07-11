import argparse

import torch

from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data.datasets.light_microscopy import get_livecell_loader

import micro_sam.training as sam_training


def get_loaders(path, patch_shape, for_sam=False):
    kwargs = {
        "label_transform": PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            min_size=25,
        ),
        "label_dtype": torch.float32,
        "num_workers": 16,
        "patch_shape": patch_shape
    }

    if for_sam:
        kwargs["raw_transform"] = sam_training.identity

    train_loader = get_livecell_loader(path=path, split="train", batch_size=2, **kwargs)
    val_loader = get_livecell_loader(path=path, split="val", batch_size=1, **kwargs)

    return train_loader, val_loader


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="/scratch/projects/nim00007/sam/data/livecell")
    parser.add_argument("-s", "--save_root", type=str, default=None)
    parser.add_argument("-p", "--phase", type=str, default=None, choices=["train", "predict", "evaluate"])
    parser.add_argument("--iterations", type=str, default=1e5)
    parser.add_argument("--sam", action="store_true")
    args = parser.parse_args()
    return args
