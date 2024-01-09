import os


DATA_ROOT = {
    "deepbacs": [
        "/scratch/projects/nim00007/sam/data/deepbacs/mixed/val/source/*",
        "/scratch/projects/nim00007/sam/data/deepbacs/mixed/val/target/*",
        "/scratch/projects/nim00007/sam/data/deepbacs/mixed/test/source/*"
    ],
    "tissuenet": [
        "/scratch/projects/nim00007/sam/data/tissuenet/val/*",
        "/scratch/projects/nim00007/sam/data/tissuenet/val/*",
        "/scratch/projects/nim00007/sam/data/tissuenet/test/*"
    ],
    "plantseg_root": [
        "/scratch/projects/nim00007/sam/data/plantseg/root_val/*",
        "/scratch/projects/nim00007/sam/data/plantseg/root_val/*",
        "/scratch/projects/nim00007/sam/data/plantseg/root_test/*"
    ]
}

EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/"

PROMPT_FOLDER = "/scratch/projects/nim00007/sam/experiments/prompts/livecell"


# we need to create tiff for val images, val gt and test gt out of the h5 or zarr file storage formats
# note: compress the files (zlib), do not store empty label files


def get_experiment_folder(domain, name, model_type):
    return os.path.join(EXPERIMENT_ROOT, domain, name, model_type)
