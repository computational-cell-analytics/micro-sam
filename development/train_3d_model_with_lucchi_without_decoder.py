import numpy as np
from glob import glob
import h5py
from micro_sam.training import train_sam, default_sam_dataset
from torch_em.data.sampler import MinInstanceSampler
from torch_em.segmentation import get_data_loader
from torch_em.transform.raw import normalize
import torch
import torch_em
import os
import argparse
from skimage.measure import regionprops
from torch_em.util.debug import check_loader


def get_rois_coordinates_skimage(file, label_key, min_shape, euler_threshold=None, min_amount_pixels=None):
    """
    Calculates the average coordinates for each unique label in a 3D label image using skimage.regionprops.

    Args:
        file (h5py.File): Handle to the open HDF5 file.
        label_key (str): Key for the label data within the HDF5 file.
        min_shape (tuple): A tuple representing the minimum size for each dimension of the ROI.
        euler_threshold (int, optional): The Euler number threshold. If provided, only regions with the specified Euler number will be considered.
        min_amount_pixels (int, optional): The minimum amount of pixels. If provided, only regions with at least this many pixels will be considered.

    Returns:
        dict or None: A dictionary mapping unique labels to lists of average coordinates for each dimension, or None if no labels are found.
    """

    label_data = file[label_key]
    label_shape = label_data.shape

    # Ensure data type is suitable for regionprops (usually uint labels)
    # if label_data.dtype != np.uint:
    #     label_data = label_data.astype(np.uint).value

    # Find connected regions (objects) using regionprops
    regions = regionprops(label_data)

    # Check if any regions were found
    if not regions:
        return None

    label_extents = {}
    for region in regions:
        if euler_threshold is not None:
            if region.euler_number != euler_threshold:
                continue
        if min_amount_pixels is not None:
            if region["area"] < min_amount_pixels:
                continue
        
        # # Extract relevant information for ROI calculation
        label = region.label  # Get the label value
        min_coords = region.bbox[:3]  # Minimum coordinates (excluding intensity channel)
        max_coords = region.bbox[3:6]  # Maximum coordinates (excluding intensity channel)

        # Clip coordinates and create ROI extent (similar to previous approach)
        clipped_min_coords = np.clip(min_coords, 0, label_shape[0] - min_shape[0])
        clipped_max_coords = np.clip(max_coords, min_shape[1], label_shape[1])
        roi_extent = tuple(slice(min_val, min_val + min_shape[dim]) for dim, (min_val, max_val) in enumerate(zip(clipped_min_coords, clipped_max_coords)))

        # Check for labels within the ROI extent (new part)
        roi_data = file[label_key][roi_extent]
        amount_label_pixels = np.count_nonzero(roi_data)
        if amount_label_pixels < 100 or amount_label_pixels < min_amount_pixels:  # Check for any non-zero values (labels)
            continue  # Skip this ROI if no labels present

        label_extents[label] = roi_extent

    return label_extents


def get_data_paths_and_rois(data_dir, min_shape,
                            data_format="*.h5",
                            image_key="raw",
                            label_key_mito="labels/mitochondria",
                            label_key_cristae="labels/cristae",
                            with_thresholds=True):

    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    rois_list = []
    new_data_paths = [] # one data path for each ROI

    for data_path in data_paths:
        try:
            # Open the HDF5 file in read-only mode
            with h5py.File(data_path, "r") as f:
                # Check for existence of image and label datasets (considering key flexibility)
                if image_key not in f or (label_key_mito is not None and label_key_mito not in f):
                    print(f"Warning: Key(s) missing in {data_path}. Skipping {image_key}")
                    continue

                #label_data_mito = f[label_key_mito][()] if label_key_mito is not None else None

                # Extract ROIs (assuming ndim of label data is the same as image data)
                if with_thresholds:
                    rois = get_rois_coordinates_skimage(f, label_key_mito, min_shape, min_amount_pixels=100) # euler_threshold=1,
                else:
                    rois = get_rois_coordinates_skimage(f, label_key_mito, min_shape, euler_threshold=None, min_amount_pixels=None)
                for label_id, roi in rois.items():
                    rois_list.append(roi)
                    new_data_paths.append(data_path)
        except OSError:
            print(f"Error accessing file: {data_path}. Skipping...")

    return new_data_paths, rois_list


def split_data_paths_to_dict(data_paths, rois_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits data paths and ROIs into training, validation, and testing sets without shuffling.

    Args:
        data_paths (list): List of paths to all HDF5 files.
        rois_list (list): List of ROIs corresponding to the data paths.
        train_ratio (float, optional): Proportion of data for training (0.0-1.0) (default: 0.8).
        val_ratio (float, optional): Proportion of data for validation (0.0-1.0) (default: 0.1).
        test_ratio (float, optional): Proportion of data for testing (0.0-1.0) (default: 0.1).

    Returns:
        tuple: A tuple containing two dictionaries:
            - data_split: Dictionary containing "train", "val", and "test" keys with data paths.
            - rois_split: Dictionary containing "train", "val", and "test" keys with corresponding ROIs.
    """

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=0.01):
        raise ValueError(f"Sum of train, validation, and test ratios must equal 1.0. But instead got:{train_ratio + val_ratio + test_ratio}")
    num_data = len(data_paths)
    if rois_list is not None:
        if len(rois_list) != num_data:
            raise ValueError(f"Length of data paths and number of ROIs in the dictionary must match: len rois {len(rois_list)}, len data_paths {len(data_paths)}")

    train_size = int(num_data * train_ratio)
    val_size = int(num_data * val_ratio)  # Optional validation set
    test_size = num_data - train_size - val_size

    data_split = {
        "train": data_paths[:train_size],
        "val": data_paths[train_size:train_size+val_size],
        "test": data_paths[train_size+val_size:]
    }

    if rois_list is not None:
        rois_split = {
            "train": rois_list[:train_size],
            "val": rois_list[train_size:train_size+val_size],
            "test": rois_list[train_size+val_size:]
        }

        return data_split, rois_split
    else:
        return data_split


def get_data_paths(data_dir, data_format="*.h5"):
    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    return data_paths


def raw_transform(image):
    image = normalize(image)
    image = image * 255
    return image



def train(args):
    n_workers = 4 if torch.cuda.is_available() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.input_path
    with_rois = True if args.without_rois is False else False 
    with_rois = False
    patch_shape = args.patch_shape
    bs = args.batch_size
    #label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=False)
    label_transform = torch_em.transform.label.labels_to_binary
    ndim = 3

    if with_rois:
        data_paths, rois_dict = get_data_paths_and_rois(data_dir, min_shape=patch_shape, with_thresholds=True)
        data, rois_dict = split_data_paths_to_dict(data_paths, rois_dict, train_ratio=.7, val_ratio=0.2, test_ratio=0.1)
    else:
        data_paths = get_data_paths(data_dir)
        data = split_data_paths_to_dict(data_paths, rois_list=None, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    #path = "/scratch-emmy/projects/nim00007/fruit-fly-data/cambridge_data/parker_s2_soma_roi_z472-600_y795-1372_x1122-1687_clahed.zarr"
    label_key = "labels/mitochondria" # "./annotations1.tif"
    train_ds = default_sam_dataset(
        raw_paths=data["train"], raw_key="raw",
        label_paths=data["train"], label_key=label_key,
        patch_shape=args.patch_shape, with_segmentation_decoder=False,
        sampler=MinInstanceSampler(2),
        raw_transform=raw_transform,
        #rois=np.s_[64:, :, :],
        #n_samples=200,
    )
    train_loader = get_data_loader(train_ds, shuffle=True, batch_size=2)

    val_ds = default_sam_dataset(
        raw_paths=data["val"], raw_key="raw",
        label_paths=data["val"], label_key=label_key,
        patch_shape=args.patch_shape, with_segmentation_decoder=False,
        sampler=MinInstanceSampler(2),
        raw_transform=raw_transform,
        #rois=np.s_[64:, :, :],
        is_train=False, 
        #n_samples=25,
    )
    val_loader = get_data_loader(val_ds, shuffle=True, batch_size=1)
    # if with_rois:
    #     train_loader = torch_em.default_segmentation_loader(
    #         raw_paths=data["train"], raw_key="raw",
    #         label_paths=data["train"], label_key="labels/mitochondria",
    #         patch_shape=patch_shape, ndim=ndim, batch_size=bs,
    #         label_transform=label_transform, raw_transform=raw_transform,
    #         num_workers=n_workers,
    #         rois=rois_dict["train"]
    #         #rois=[np.s_[64:, :, :]] * len(data["train"])
    #     )
    #     val_loader = torch_em.default_segmentation_loader(
    #         raw_paths=data["val"], raw_key="raw",
    #         label_paths=data["val"], label_key="labels/mitochondria",
    #         patch_shape=patch_shape, ndim=ndim, batch_size=bs,
    #         label_transform=label_transform, raw_transform=raw_transform,
    #         num_workers=n_workers,
    #         rois=rois_dict["val"]
    #         # rois=[np.s_[64:, :, :]] * len(data["val"])
    #     )
    # else:
    #     train_loader = torch_em.default_segmentation_loader(
    #         raw_paths=data["train"], raw_key="raw",
    #         label_paths=data["train"], label_key=label_key,
    #         patch_shape=patch_shape, ndim=ndim, batch_size=bs,
    #         label_transform=label_transform, raw_transform=raw_transform,
    #         num_workers=n_workers,
    #     )
    #     print("len data[val]", len(data["val"]))
    #     val_loader = torch_em.default_segmentation_loader(
    #         raw_paths=data["val"], raw_key="raw",
    #         label_paths=data["val"], label_key=label_key,
    #         patch_shape=patch_shape, ndim=ndim, batch_size=bs,
    #         label_transform=label_transform, raw_transform=raw_transform,
    #         num_workers=n_workers,
    #     )
    
    
    #check_loader(train_loader, n_samples=3)
    # x,y =next(iter(train_loader))
    # print("shapes of x and y", x.shape, y.shape)
    # breakpoint()

    train_sam(
        name="mito_model", model_type="vit_b",
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=50, n_objects_per_batch=10,
        with_segmentation_decoder=False,
        save_root=args.save_root,
    )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the Mitochondria dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo/",
        help="The filepath to the LiveCELL data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h."
    )
    parser.add_argument("--without_lora", action="store_true", help="Whether to use LoRA for finetuning SAM for semantic segmentation.") 
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(1, 512, 512), help="Patch shape for data loading (3D tuple)")
    
    parser.add_argument("--n_epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--n_classes", type=int, default=3, help="Number of classes to predict")
    parser.add_argument("--batch_size", "-bs", type=int, default=1, help="Batch size") # masam 3 
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="base learning rate") # MASAM 0.0008
    parser.add_argument(
        "--save_root", "-s", default="/scratch-grete/usr/nimlufre/micro-sam_training_on_mitotomo",
        help="The filepath to where the logs and the checkpoints will be saved."
    )
    parser.add_argument(
        "--exp_name", default="vitb_3d_lora4-microsam-hypam-lucchi",
        help="The filepath to where the logs and the checkpoints will be saved."
    )
    parser.add_argument("--without_rois", action="store_true", help="Train without Regions Of Interest (ROI)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
