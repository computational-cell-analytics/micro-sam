import os
from glob import glob

import imageio.v3 as imageio

ROOT = "/scratch-grete/projects/nim00007/data/tissuenet"


def get_tissuenet_images(split):
    assert split in ["val", "test"]
    val_set, test_set = glob(os.path.join(ROOT, "val", "*")), glob(os.path.join(ROOT, "test", "*"))
    if split == "val":
        return sorted(val_set)
    else:
        return sorted(test_set)


# TODO
def create_tissuenet_splits():
    output_root = "/scratch-grete/projects/nim00007/sam/ood/LM/tissuenet"

    def write_split(images, labels, split):
        out_folder = os.path.join(output_root, split)
        os.makedirs(out_folder, exist_ok=True)
        for ii, (im, lab) in enumerate(zip(images, labels)):
            out_im = os.path.join(out_folder, f"image_{ii:04}.tif")
            out_lab = os.path.join(out_folder, f"labels_{ii:04}.tif")
            im, lab = imageio.imread(im), imageio.imread(lab)
            imageio.imwrite(out_im, im)
            imageio.imwrite(out_lab, lab)

    val_set = get_tissuenet_images("val")

    write_split(val_images, val_labels, "val")
    write_split(test_images, test_labels, "test")


if __name__ == "__main__":
    create_tissuenet_splits()
