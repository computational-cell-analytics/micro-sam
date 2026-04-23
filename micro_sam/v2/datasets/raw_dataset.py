import os
from glob import glob
from pathlib import Path
from typing import Union

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

import torch

from torch_em.util.image import load_data
from torch_em.transform.raw import normalize_percentile
from torch_em.data.datasets import light_microscopy, electron_microscopy
from torch_em.data.datasets.electron_microscopy import axondeepseg as axondeepseg_module
from torch_em.data.datasets.light_microscopy import ctc as ctc_module

from training.dataset.vos_raw_dataset import VOSRawDataset, VOSFrame, VOSVideo

from .segment_loader import ImageSegmentLoader, VolumeSegmentLoader


class VolumeRawDataset(VOSRawDataset):
    """VolumeRawDataset for multi-dimensional datasets.

    Supported dataset_name values:
        lucchi, embedseg, nis3d, cremi, snemi,
        plantseg_root, plantseg_ovules, emneuron, platynereis

    Args:
        path: Filepath where the data is stored / downloaded.
        dataset_name: The choice of dataset.
        split: The choice of data split for training.
        sample_rate:
        truncate_video:
        download: Whether to download the dataset.
    """
    def __init__(
        self,
        path: Union[os.PathLike, str],
        dataset_name: str,
        split: str,
        sample_rate: float = 1,
        truncate_video: float = -1,
        download: bool = False,
    ):
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        if dataset_name == "lucchi":
            volume_paths = electron_microscopy.lucchi.get_lucchi_paths(path=path, split=split, download=download)
            self.raw_paths, self.label_paths = [volume_paths], [volume_paths]
            self.raw_key, self.label_key = "raw", "labels"

        elif dataset_name == "embedseg":
            self.raw_key, self.label_key = None, None
            self.raw_paths, self.label_paths = [], []
            for name in light_microscopy.embedseg_data.URLS.keys():
                curr_raw_paths, curr_label_paths = light_microscopy.embedseg_data.get_embedseg_paths(
                    path=path, name=name, split="train"
                )
                self.raw_paths.extend(curr_raw_paths)
                self.label_paths.extend(curr_label_paths)

        elif dataset_name == "nis3d":
            self.raw_paths, self.label_paths = light_microscopy.nis3d.get_nis3d_paths(path=path)
            self.raw_key, self.label_key = None, None

        elif dataset_name == "cremi":
            volume_paths = electron_microscopy.cremi.get_cremi_paths(path=path)
            self.raw_paths = self.label_paths = volume_paths
            self.raw_key, self.label_key = "volumes/raw", "volumes/labels/neuron_ids"

        elif dataset_name == "snemi":
            volume_paths = electron_microscopy.snemi.get_snemi_paths(path=path, sample="train")
            self.raw_paths = self.label_paths = [volume_paths]
            self.raw_key, self.label_key = "volumes/raw", "volumes/labels/neuron_ids"

        elif dataset_name == "plantseg_root":
            paths = light_microscopy.plantseg.get_plantseg_paths(path=path, name="root", split=split)
            self.raw_paths = self.label_paths = paths
            self.raw_key, self.label_key = "raw", "segmentation/corrected"

        elif dataset_name == "plantseg_ovules":
            # Ovules files use key "label", not "segmentation/corrected" like root.
            paths = light_microscopy.plantseg.get_plantseg_paths(path=path, name="ovules", split=split)
            self.raw_paths = self.label_paths = paths
            self.raw_key, self.label_key = "raw", "label"

        elif dataset_name == "emneuron":
            # Each .tif file is a 3D block (Z, H, W) with instance-labeled neurons.
            self.raw_paths, self.label_paths = electron_microscopy.emneuron.get_emneuron_paths(
                path=path, split=split
            )
            self.raw_key, self.label_key = None, None

        elif dataset_name == "platynereis":
            # All 8 training samples — split by convention (1-6 train, 7-8 val).
            sample_ids = [1, 2, 3, 4, 5, 6] if split == "train" else [7, 8]
            paths = electron_microscopy.platynereis.get_platynereis_paths(
                path=path, sample_ids=sample_ids, name="cells"
            )
            self.raw_paths = self.label_paths = paths
            self.raw_key, self.label_key = "volumes/raw/s1", "volumes/labels/segmentation/s1"

        else:
            raise ValueError(f"'{dataset_name}' is not a supported dataset.")

    def get_video(self, idx: int):
        """Given the VOSVideo object, return the mask tensors.
        """
        # Assign the relevant filepaths.
        raw_path = self.raw_paths[idx]
        label_path = self.label_paths[idx]
        volume_name = str(Path(raw_path).stem)

        # Load data from the file.
        raw = np.array(load_data(raw_path, self.raw_key))
        labels = np.array(load_data(label_path, self.label_key))

        # NOTE: This will change this in future to automate this based on the volume.
        assert raw.ndim == 3 and labels.ndim == 3, "We currently only support 3 dimensional inputs for this loader."

        # Normalize non-uint8 volumes (e.g. uint16 LM data) per-volume before slicing,
        # so all frames share the same intensity scale (Convention 2: output is uint8 [0, 255]).
        if raw.dtype != np.uint8:
            raw = normalize_percentile(raw.astype(np.float32))
            raw = np.clip(raw, 0, 1)
            raw = (raw * 255).astype(np.uint8)

        # Let's expand the one channel images to three channels for training.
        raw = np.repeat(raw[:, np.newaxis, :, :], 3, axis=1)  # (img_num, 3, H, W)

        if self.truncate_video > 0:
            raw = raw[:self.truncate_video]
            labels = labels[:self.truncate_video]

        # Let's prepare the volume in VOSFrame objects style for storing per slice in expected format.
        all_frames = []
        for i, per_slice in enumerate(raw[::self.sample_rate]):
            slice_idx = i * self.sample_rate
            all_frames.append(VOSFrame(slice_idx, image_path=None, data=torch.from_numpy(per_slice)))

        # Let's create the volume as a VOSVideo object.
        volume = VOSVideo(volume_name, idx, all_frames)

        # Finally, let's wrap the corresponding labels with a custom 'VolumeSegmentLoader'.
        segment_loader = VolumeSegmentLoader(labels[::self.sample_rate])

        return volume, segment_loader

    def __len__(self):
        return len(self.raw_paths)


class ImageRawDataset(VOSRawDataset):
    """ImagesRawDataset for 2d image datasets.

    Supported dataset_name values:
        livecell, cellpose, cvz_fluo, dsb,
        deepbacs, orgasegment, organoidnet, omnipose

    Args:
        path: Filepath where the data is stored / downloaded.
        dataset_name: The choice of dataset.
        split: The choice of data split for training.
        num_frames: The number of frames to replicate.
        download: Whether to download the dataset.
    """
    def __init__(
        self,
        path: Union[os.PathLike, str],
        dataset_name: str,
        split: str,
        num_frames: int = 1,
        download: bool = False,
    ):
        self.path = path
        self.num_frames = num_frames

        if dataset_name == "livecell":
            self.raw_paths, self.label_paths = light_microscopy.livecell.get_livecell_paths(
                path=path, split=split, download=download,
            )

        elif dataset_name == "cellpose":
            self.raw_paths, self.label_paths = light_microscopy.cellpose.get_cellpose_paths(
                path=path, split=split, choice="cyto", download=download,
            )

        elif dataset_name == "cvz_fluo":
            self.raw_paths, self.label_paths = light_microscopy.cvz_fluo.get_cvz_fluo_paths(
                path=path, stain_choice="cell",
            )
            curr_raw_paths, curr_label_paths = light_microscopy.cvz_fluo.get_cvz_fluo_paths(
                path=path, stain_choice="dapi",
            )
            self.raw_paths.extend(curr_raw_paths)
            self.label_paths.extend(curr_label_paths)

        elif dataset_name == "dsb":
            self.raw_paths, self.label_paths = light_microscopy.dsb.get_dsb_paths(
                path=path, source="reduced",
            )

        elif dataset_name == "deepbacs":
            # Returns (raw_folder, label_folder) — glob .tif files from each.
            raw_folder, label_folder = light_microscopy.deepbacs.get_deepbacs_paths(
                path=path, bac_type="mixed", split=split,
            )
            self.raw_paths = sorted(glob(os.path.join(raw_folder, "*.tif")))
            self.label_paths = sorted(glob(os.path.join(label_folder, "*.tif")))

        elif dataset_name == "orgasegment":
            self.raw_paths, self.label_paths = light_microscopy.orgasegment.get_orgasegment_paths(
                path=path, split=split,
            )

        elif dataset_name == "organoidnet":
            self.raw_paths, self.label_paths = light_microscopy.organoidnet.get_organoidnet_paths(
                path=path, split=split,
            )

        elif dataset_name == "omnipose":
            self.raw_paths, self.label_paths = light_microscopy.omnipose.get_omnipose_paths(
                path=path, split=split,
            )

        else:
            raise ValueError(f"'{dataset_name}' is not a supported dataset.")

        # Let's sort all the filepaths for loading data.
        self.raw_paths, self.label_paths = sorted(self.raw_paths), sorted(self.label_paths)

    def get_video(self, idx):
        """Given a VOSVideo object, return the mask tensors.
        """
        # Assign all relevant filepaths.
        raw_path = self.raw_paths[idx]
        label_path = self.label_paths[idx]
        fname = str(Path(label_path).stem)

        # We prepare the image in VOSFrame object style for storing it in expected format.
        raw = np.array(load_data(raw_path))

        # Normalize non-uint8 images per-image before RGB conversion
        # (Convention 2: output uint8 [0, 255] for ToTensorAPI → NormalizeAPI pipeline).
        if raw.dtype != np.uint8:
            raw = normalize_percentile(raw.astype(np.float32))
            raw = np.clip(raw, 0, 1)
            raw = (raw * 255).astype(np.uint8)

        # Ensure the images are in RGB format.
        raw = light_microscopy.neurips_cell_seg.to_rgb(raw)

        frames = [VOSFrame(0, image_path=None, data=torch.from_numpy(raw))]

        # Let's create the image as a VOSVideo object.
        video = VOSVideo(fname, idx, frames)

        # Finally, let's wrap the corresponding label image with a custom 'ImageSegmentLoader'.
        segment_loader = ImageSegmentLoader(label_path)

        return video, segment_loader

    def __len__(self):
        return len(self.raw_paths)


_CTC_DATASETS = [
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa",
    "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-N2DH-SIM+",
    "PhC-C2DH-U373", "PhC-C2DL-PSC",
]


class TissueNetDataset(VOSRawDataset):
    """TissueNetDataset for 2D multi-channel fluorescence microscopy images.

    Each zarr file contains an RGB image (raw/rgb) and instance labels (labels/cell).
    The data is treated as a single-frame video for SAM2 training.

    Args:
        path: Filepath where the data is stored / downloaded.
        split: The choice of data split (train, val, test).
        download: Whether to download the dataset.
    """
    def __init__(
        self,
        path: Union[os.PathLike, str],
        split: str,
        download: bool = False,
    ):
        self.zarr_paths = sorted(
            light_microscopy.tissuenet.get_tissuenet_paths(path=path, split=split, download=download)
        )

    def get_video(self, idx: int):
        zarr_path = self.zarr_paths[idx]
        fname = Path(zarr_path).stem

        raw = np.array(load_data(zarr_path, "raw/rgb"))        # (3, H, W) float64
        labels = np.array(load_data(zarr_path, "labels/cell"))  # (H, W) int32

        # Percentile normalize per channel → [0, 1] → uint8
        raw = normalize_percentile(raw, axis=(1, 2))
        raw = np.clip(raw, 0, 1)
        raw = (raw * 255).astype(np.uint8)  # (3, H, W)

        frames = [VOSFrame(0, image_path=None, data=torch.from_numpy(raw))]
        video = VOSVideo(fname, idx, frames)
        segment_loader = VolumeSegmentLoader(labels[np.newaxis])

        return video, segment_loader

    def __len__(self):
        return len(self.zarr_paths)


class AxonDeepSegDataset(VOSRawDataset):
    """AxonDeepSegDataset for 2D electron microscopy axon segmentation.

    Each h5 file contains a grayscale raw image and semantic labels where class 2
    corresponds to axon interiors. Instance labels are derived via connected components.

    Args:
        path: Filepath where the data is stored / downloaded.
        split: The choice of data split (train or val).
        download: Whether to download the dataset.
    """
    def __init__(
        self,
        path: Union[os.PathLike, str],
        split: str,
        download: bool = False,
    ):
        self.h5_paths = sorted(
            axondeepseg_module.get_axondeepseg_paths(
                path=path, name=["sem", "tem"], val_fraction=0.2, split=split, download=download,
            )
        )

    def get_video(self, idx: int):
        h5_path = self.h5_paths[idx]
        fname = Path(h5_path).stem

        raw = np.array(load_data(h5_path, "raw"))        # (H, W) uint8
        labels = np.array(load_data(h5_path, "labels"))   # (H, W) uint8, semantic 0/1/2

        # Stack grayscale to 3-channel CHW
        raw = np.stack([raw, raw, raw], axis=0)  # (3, H, W)

        # Derive instance labels from axon class (2) via connected components
        axon_mask = (labels == 2)
        instances = connected_components(axon_mask).astype(np.int32)  # (H, W)

        frames = [VOSFrame(0, image_path=None, data=torch.from_numpy(raw))]
        video = VOSVideo(fname, idx, frames)
        segment_loader = VolumeSegmentLoader(instances[np.newaxis])

        return video, segment_loader

    def __len__(self):
        return len(self.h5_paths)


class CTCDataset(VOSRawDataset):
    """CTCDataset for Cell Tracking Challenge multi-frame sequences.

    Each sequence is treated as a temporal video. Raw images and instance labels
    are loaded from the GT-masked IM and SEG folders respectively.

    Args:
        path: Filepath where the data is stored / downloaded.
        split: The choice of data split (currently only 'train').
        min_frames: Minimum number of labeled frames a sequence must have to be included.
            Sequences shorter than this are skipped to avoid over-sampling from repeated frames.
        download: Whether to download the dataset.
    """
    def __init__(
        self,
        path: Union[os.PathLike, str],
        split: str = "train",
        min_frames: int = 8,
        download: bool = False,
    ):
        self.sequence_pairs = []  # List of (image_folder, label_folder, dataset_name, vol_id)
        for dataset_name in _CTC_DATASETS:
            try:
                image_folders, label_folders = ctc_module.get_ctc_segmentation_paths(
                    path=path, dataset_name=dataset_name, split=split, download=download,
                )
                for img_folder, lbl_folder in zip(image_folders, label_folders):
                    n_frames = len(glob(os.path.join(img_folder, "*.tif")))
                    if n_frames < min_frames:
                        continue
                    vol_id = Path(lbl_folder).parent.name.rstrip("_GT")
                    self.sequence_pairs.append((img_folder, lbl_folder, dataset_name, vol_id))
            except Exception:
                pass

    def get_video(self, idx: int):
        img_folder, lbl_folder, dataset_name, vol_id = self.sequence_pairs[idx]
        seq_name = f"{dataset_name}_{vol_id}"

        img_files = sorted(glob(os.path.join(img_folder, "*.tif")))
        lbl_files = sorted(glob(os.path.join(lbl_folder, "*.tif")))
        assert len(img_files) == len(lbl_files) and len(img_files) > 0

        all_raw = [imageio.imread(f) for f in img_files]
        all_labels = [imageio.imread(f).astype(np.int32) for f in lbl_files]

        # Normalize non-uint8 sequences per-sequence so all frames share the same
        # intensity scale (avoids frame-to-frame brightness flicker in SAM2 video mode).
        # Convention 2 → uint8 [0, 255], consistent with ToTensorAPI → NormalizeAPI pipeline.
        if all_raw[0].dtype != np.uint8:
            seq_stack = np.stack(all_raw).astype(np.float32)  # (T, H, W)
            seq_stack = normalize_percentile(seq_stack)        # percentiles over whole sequence
            seq_stack = np.clip(seq_stack, 0, 1)
            all_raw = [(seq_stack[i] * 255).astype(np.uint8) for i in range(len(all_raw))]

        all_frames = []
        for frame_idx, raw_frame in enumerate(all_raw):
            raw_chw = np.stack([raw_frame, raw_frame, raw_frame], axis=0)  # (3, H, W)
            all_frames.append(VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(raw_chw)))

        video = VOSVideo(seq_name, idx, all_frames)
        segment_loader = VolumeSegmentLoader(np.stack(all_labels))

        return video, segment_loader

    def __len__(self):
        return len(self.sequence_pairs)
