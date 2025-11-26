import os

from torch_em.data import datasets

from micro_sam.evaluation.livecell import _get_livecell_paths


DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/data"


def get_image_label_paths(dataset_name, split):
    assert split in ["val", "test"]

    # Label-free
    if dataset_name == "livecell":
        image_paths, label_paths = _get_livecell_paths(
            input_folder=os.path.join(DATA_DIR, dataset_name), split=split,
        )

    # Histopathology
    elif dataset_name == "pannuke":
        ...

    # Fluroscence (Nuclei)
    elif dataset_name == "dsb":
        if split == "val":  # Since 'val' does not exist for this data.
            split = "test"

        image_paths, label_paths = datasets.light_microscopy.dsb.get_dsb_paths(
            os.path.join(DATA_DIR, "dsb"), source="reduced", split=split,
        )

    # Fluorescence (Cells)
    elif dataset_name == "tissuenet":
        ...

    else:
        raise ValueError

    return image_paths, label_paths
