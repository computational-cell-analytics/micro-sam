
import os
from tqdm import tqdm
import imageio.v2 as imageio
import numpy as np

from torch_em.data import MinInstanceSampler
from torch_em.transform.label import label_consecutive
from torch_em.data.datasets import get_tissuenet_loader
from torch_em.transform.raw import standardize, normalize_percentile


def rgb_to_gray_transform(raw):
    raw = normalize_percentile(raw, axis=(1, 2))
    raw = np.mean(raw, axis=0)
    raw = standardize(raw)
    return raw


def get_tissuenet_loaders(input_path):
    sampler = MinInstanceSampler()
    label_transform = label_consecutive
    raw_transform = rgb_to_gray_transform
    val_loader = get_tissuenet_loader(path=input_path, split="val", raw_channel="rgb", label_channel="cell",
                                      batch_size=1, patch_shape=(256, 256), num_workers=0,
                                      sampler=sampler, label_transform=label_transform, raw_transform=raw_transform)
    test_loader = get_tissuenet_loader(path=input_path, split="test", raw_channel="rgb", label_channel="cell",
                                       batch_size=1, patch_shape=(256, 256), num_workers=0,
                                       sampler=sampler, label_transform=label_transform, raw_transform=raw_transform)
    return val_loader, test_loader


def extract_images(loader, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        img_path = os.path.join(out_folder, "image_{:04d}.tif".format(i))
        gt_path = os.path.join(out_folder, "label_{:04d}.tif".format(i))

        img = x.squeeze().detach().cpu().numpy()
        gt = y.squeeze().detach().cpu().numpy()

        imageio.imwrite(img_path, img)
        imageio.imwrite(gt_path, gt)


def main():
    val_loader, test_loader = get_tissuenet_loaders("/scratch-grete/projects/nim00007/data/tissuenet")
    print("Length of val loader is:", len(val_loader))
    print("Length of test loader is:", len(test_loader))

    root_save_dir = "/scratch/projects/nim00007/sam/datasets/tissuenet"

    # we use the val set for test because there are some issues with the test set
    # out_folder = os.path.join(root_save_dir, "test")
    # extract_images(val_loader, out_folder)

    # we use the test folder for val and just use as many images as we can sample
    out_folder = os.path.join(root_save_dir, "val")
    extract_images(test_loader, out_folder)


if __name__ == "__main__":
    main()
