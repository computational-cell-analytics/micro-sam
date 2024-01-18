import os
from tqdm import tqdm

import imageio.v3 as imageio
from skimage.segmentation import relabel_sequential

ROOT = "/scratch-grete/projects/nim00007/sam/experiments/benchmarking/mitonet/lucchi/"


def make_slices_from_mitolab_predictions(labels_path, save_dir):
    labels = imageio.imread(labels_path)
    for i, _slice in tqdm(enumerate(labels), total=labels.shape[0]):
        relabeled_slice = relabel_sequential(_slice)[0]
        save_path = os.path.join(save_dir, f"lucchi_test_{i+1:05}.tif")
        imageio.imwrite(save_path, relabeled_slice, compression="zlib")


def main():
    # we slice the label stack and save it here
    os.makedirs(os.path.join(ROOT, "slices"), exist_ok=True)

    make_slices_from_mitolab_predictions(
        labels_path=os.path.join(ROOT, "lucchi_test_batch_segs_from_mitonet_default_params.tif"),
        save_dir=os.path.join(ROOT, "slices")
    )


if __name__ == "__main__":
    main()
