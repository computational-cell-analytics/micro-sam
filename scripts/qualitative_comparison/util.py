import os

from torch_em.data import datasets
from torch_em.transform.label import connected_components


ROOT = "/media/anwai/ANWAI/data"


def fetch_data_loaders(dataset_name):
    if dataset_name == "lucchi":
        loader = datasets.get_lucchi_loader(
            os.path.join(ROOT, "lucchi", "t"), "train", (1, 512, 512), 1, ndim=2, download=True,
            label_transform=connected_components
        )

    elif dataset_name == "livecell":
        loader = datasets.get_livecell_loader(os.path.join(ROOT, "livecell"), "train", (512, 512), 1)

    elif dataset_name == "deepbacs":
        loader = datasets.get_deepbacs_loader(
            os.path.join(ROOT, "deepbacs"), "test", bac_type="mixed", download=True,
            patch_shape=(512, 512), batch_size=1, shuffle=False, n_samples=100
        )

    elif dataset_name == "tissuenet":
        loader = datasets.get_tissuenet_loader(
            os.path.join(ROOT, "tissuenet"), "train", raw_channel="rgb", label_channel="cell",
            patch_shape=(256, 256), batch_size=1, shuffle=True,
        )

    elif dataset_name == "plantseg_root":
        loader = datasets.get_plantseg_loader(
            os.path.join(ROOT, "plantseg"), "root", "test", (1, 512, 512), 1, ndim=2, download=True
        )

    return loader
