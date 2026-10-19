"""Download 64 x 1024 x 1024 mitochondria boxes of the Parlakgul fasting liver samples (EMPIAR-12017, 8 nm).

One central tissue box per sample is fetched with the torch-em loader (bounding-box streaming), which caches it as
<data>/parlakgul_fasting_liver/<sample>_mito_<hash>.zarr. The task index of a slurm array selects the box.
"""
import argparse
import os

from torch_em.data.datasets.electron_microscopy import parlakgul_fasting_liver

from figure_utils import DATA_ROOT


# (sample, (x_min, x_max, y_min, y_max, z_min, z_max)): the center of each sample volume.
BOXES = [
    ("8970", (4189, 5213, 2962, 3986, 4193, 4257)),
    ("2679", (2738, 3762, 2738, 3762, 3820, 3884)),
    ("1858", (4175, 5199, 4800, 5824, 2865, 2929)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    args = parser.parse_args()
    sample, box = BOXES[args.index]
    path = parlakgul_fasting_liver.get_parlakgul_fasting_liver_data(
        os.path.join(DATA_ROOT, "parlakgul_fasting_liver"), box, sample=sample, label_choice="mito", download=True,
    )
    print(sample, box, path, flush=True)


if __name__ == "__main__":
    main()
