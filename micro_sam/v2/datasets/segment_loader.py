import os
from typing import Union

import numpy as np
import imageio.v3 as imageio

import torch


class VolumeSegmentLoader:
    def __init__(self, masks: np.ndarray):
        """Initialize the VolumeSegmentLoader.

        Args:
            masks: Array of masks with shape (img_num, H, W)
        """
        self.masks = masks

    def load(self, frame_idx: int):
        """Load the single masks for the given frame index and convert it to binary segments.

        Args:
            frame_idx: The index of the frame to load.

        Returns:
            Dictionary where keys are object IDs and values are binary masks.
        """
        mask = self.masks[frame_idx]

        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]

        binary_segments = {}
        for obj_id in object_ids:
            binary_mask = (mask == obj_id)
            binary_segments[int(obj_id)] = torch.from_numpy(binary_mask).bool()

        return binary_segments


class ImageSegmentLoader:
    def __init__(self, label_path: Union[os.PathLike, str]):
        """Initialize the ImageSegmentLoader.

        Args:
            label_path: Filepath containing the mask stored for a corresponding image.
        """
        self.label_path = label_path

    def load(self, frame_idx: int):
        """Load the single image masks.

        Args:
            frame_idx: The index of the frame (unused for single images).

        Returns:
            Dictionary where keys are object IDs and values are binary masks.
        """
        mask = imageio.imread(self.label_path)

        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]

        binary_segments = {}
        for obj_id in object_ids:
            binary_mask = (mask == obj_id)
            binary_segments[int(obj_id)] = torch.from_numpy(binary_mask).bool()

        return binary_segments

    def __len__(self):
        return
