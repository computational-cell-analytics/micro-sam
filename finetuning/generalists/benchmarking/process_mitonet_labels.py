import os
from glob import glob
from tqdm import tqdm
from pathlib import Path

import imageio.v3 as imageio
from skimage.segmentation import relabel_sequential

ROOT = "/scratch-grete/projects/nim00007/sam/experiments/benchmarking/mitonet/"


def make_slices_from_mitolab_predictions(labels_path, save_dir, counter=0):
    labels = imageio.imread(labels_path)
    label_id = Path(labels_path).stem

    os.makedirs(save_dir, exist_ok=True)

    for _slice in tqdm(labels, total=labels.shape[0]):
        relabeled_slice = relabel_sequential(_slice)[0]
        save_path = os.path.join(save_dir, f"{label_id}_{counter+1:05}.tif")
        imageio.imwrite(save_path, relabeled_slice, compression="zlib")
        counter += 1


def main():
    # for lucchi
    make_slices_from_mitolab_predictions(
        labels_path=os.path.join(ROOT, "lucchi", "lucchi_test_batch_segs_from_mitonet_default_params.tif"),
        save_dir=os.path.join(ROOT, "lucchi", "slices")
    )

    # for mitoem
    mitoem_seg_paths = glob(os.path.join(ROOT, "mitoem", "mitoem_*_test_batch_segs.tif"))
    for seg_path in mitoem_seg_paths:
        make_slices_from_mitolab_predictions(
            labels_path=seg_path, save_dir=os.path.join(ROOT, "mitoem", "slices"), counter=5
        )


if __name__ == "__main__":
    main()
