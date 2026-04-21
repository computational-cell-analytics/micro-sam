import os
from pathlib import Path
from typing import Union

import numpy as np

import torch

from torch_em.util.image import load_data
from torch_em.data.datasets import light_microscopy, electron_microscopy

from training.dataset.vos_raw_dataset import VOSRawDataset, VOSFrame, VOSVideo

from .segment_loader import ImageSegmentLoader, VolumeSegmentLoader


class VolumeRawDataset(VOSRawDataset):
    """VolumeRawDataset for multi-dimensional datasets.

    Args:
        path: Filepath where the data is downloaded for further processing.
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
        raw = load_data(raw_path, self.raw_key)
        labels = load_data(label_path, self.label_key)

        # NOTE: This will change this in future to automate this based on the volume.
        assert raw.ndim == 3 and labels.ndim == 3, "We currently only support 3 dimensional inputs for this loader."

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

    Args:
        path: Filepath where the data is download for further processing.
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
        raw = load_data(raw_path)

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
