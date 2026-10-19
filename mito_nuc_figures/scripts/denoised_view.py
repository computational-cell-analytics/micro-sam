"""Lazy display-only denoising for noisy cryo-ET volumes, used by the cryo-ET figure scripts."""
import numpy as np
from scipy import ndimage


class DenoisedView:
    """Lazy wrapper that returns a slab mean + gaussian smoothed version of a (z, y, x) volume.

    For a z-slice the mean over `half_width` neighboring z-slices is taken, for a ZX view (integer y index)
    the mean over neighboring y-rows. The result is smoothed with a gaussian and then strided. Display only.
    """

    def __init__(self, volume, half_width=2, sigma=1.5):
        self.volume = volume
        self.shape = tuple(volume.shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype("float32")
        self.half_width = half_width
        self.sigma = sigma

    def slab(self, center, length):
        return slice(max(center - self.half_width, 0), min(center + self.half_width + 1, length))

    def __getitem__(self, index):
        z, y, x = index
        if isinstance(z, (int, np.integer)):
            data = np.asarray(self.volume[self.slab(int(z), self.shape[0])]).astype("float32").mean(axis=0)
            return ndimage.gaussian_filter(data, self.sigma)[y, x]
        if isinstance(y, (int, np.integer)):
            data = np.asarray(self.volume[:, self.slab(int(y), self.shape[1])]).astype("float32").mean(axis=1)
            return ndimage.gaussian_filter(data, self.sigma)[z, x]
        return np.asarray(self.volume[index])


class DilatedView:
    """Lazy wrapper that grey-dilates each 2D section of a thin (membrane) label volume before striding. Display only.

    The dilation keeps the label values. The larger value wins where classes touch. Thus, membranes that are 1-2 voxels
    thick stay visible after the figure helper subsamples the slices.
    """

    def __init__(self, volume, size=3):
        self.volume = volume
        self.shape = tuple(volume.shape)
        self.ndim = len(self.shape)
        self.dtype = volume.dtype
        self.size = size

    def __getitem__(self, index):
        z, y, x = index
        if isinstance(z, (int, np.integer)):
            data = np.asarray(self.volume[int(z)])
            return ndimage.grey_dilation(data, size=(self.size, self.size))[y, x]
        if isinstance(y, (int, np.integer)):
            data = np.asarray(self.volume[:, int(y)])
            return ndimage.grey_dilation(data, size=(self.size, self.size))[z, x]
        return np.asarray(self.volume[index])
