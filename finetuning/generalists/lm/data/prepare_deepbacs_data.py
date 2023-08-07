import os
from glob import glob

import imageio.v3 as imageio
import numpy as np

from sklearn.model_selection import train_test_split

ROOT = "/scratch-grete/projects/nim00007/data/deepbacs"


def download_deepbacs():
    from torch_em.data.datasets import get_deepbacs_loader
    get_deepbacs_loader(ROOT, "train", bac_type="mixed", download=True, patch_shape=(256, 256), batch_size=1)
    get_deepbacs_loader(ROOT, "test", bac_type="mixed", download=True, patch_shape=(256, 256), batch_size=1)


# old code from Anwai
def get_deepbacs_test_images():
    root = ROOT
    output_root = "/scratch-grete/projects/nim00007/sam/ood/LM/deepbacs"

    def write_split(images, labels, split):
        out_folder = os.path.join(output_root, split)
        os.makedirs(out_folder, exist_ok=True)
        for ii, (im, lab) in enumerate(zip(images, labels)):
            out_im = os.path.join(out_folder, f"image_{ii:04}.tif")
            out_lab = os.path.join(out_folder, f"labels_{ii:04}.tif")
            im, lab = imageio.imread(im), imageio.imread(lab)
            imageio.imwrite(out_im, im)
            imageio.imwrite(out_lab, lab)

    root_imgs = glob(os.path.join(root, "mixed", "test", "source", "*"))
    root_gts = glob(os.path.join(root, "mixed", "test", "target", "*"))
    np.random.seed(0)

    val_images = np.random.choice(root_imgs, size=5, replace=False).tolist()
    val_labels = [gt_p for gt_p in root_gts if os.path.basename(gt_p) in [os.path.basename(x) for x in val_images]]

    test_images = [ip for ip in root_imgs if ip not in val_images]
    test_labels = [gp for gp in root_gts if gp not in val_labels]

    write_split(val_images, val_labels, "val")
    write_split(test_images, test_labels, "test")


# new simplified code
def get_deepbacs_test_images_new():
    root = ROOT
    output_root = "/scratch-grete/projects/nim00007/sam/ood/LM/deepbacs"

    def write_split(images, labels, split):
        out_folder = os.path.join(output_root, split)
        os.makedirs(out_folder, exist_ok=True)
        for ii, (im, lab) in enumerate(zip(images, labels)):
            out_im = os.path.join(out_folder, f"image_{ii:04}.tif")
            out_lab = os.path.join(out_folder, f"labels_{ii:04}.tif")
            im, lab = imageio.imread(im), imageio.imread(lab)
            imageio.imwrite(out_im, im)
            imageio.imwrite(out_lab, lab)

    images = sorted(glob(os.path.join(root, "mixed", "test", "source", "*")))
    labels = sorted(glob(os.path.join(root, "mixed", "test", "target", "*")))

    test_images, val_images, test_labels, val_labels = train_test_split(
        images, labels, test_size=0.15, random_state=42
    )

    write_split(val_images, val_labels, "val")
    write_split(test_images, test_labels, "test")


def main():
    # download_deepbacs()
    get_deepbacs_test_images_new()


if __name__ == "__main__":
    main()
