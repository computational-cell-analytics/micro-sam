import os
from glob import glob
from natsort import natsorted

from .raw_dataset import VolumeRawDataset


class IgorAortaDataset(VolumeRawDataset):
    def __init__(self, path, split, sample_rate=1, truncate_video=-1):
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        self.volume_paths = natsorted(glob(os.path.join(path, "*.h5")))

        if split == "train":
            self.volume_paths = self.volume_paths[:-1]
        else:
            self.volume_paths = self.volume_paths[-1]

        self.raw_key, self.label_key = "raw", "labels/tissues"
        self.file_extension = ".h5"
