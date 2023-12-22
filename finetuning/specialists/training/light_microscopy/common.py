import numpy as np
from math import ceil, floor

from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.transform.raw import normalize_percentile, normalize


label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25
    )


class ResizeRawTrafo:
    def __init__(self, desired_shape, do_rescaling=True, padding="constant"):
        self.desired_shape = desired_shape
        self.padding = padding
        self.do_rescaling = do_rescaling

    def __call__(self, raw):
        if self.do_rescaling:
            raw = normalize_percentile(raw, axis=(1, 2))
            raw = np.mean(raw, axis=0)
            raw = normalize(raw)
            raw = raw * 255

        tmp_ddim = (self.desired_shape[0] - raw.shape[0], self.desired_shape[1] - raw.shape[1])
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
        raw = np.pad(
            raw,
            pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
            mode=self.padding
        )
        assert raw.shape == self.desired_shape
        return raw


class ResizeLabelTrafo:
    def __init__(self, desired_shape, padding="constant"):
        self.desired_shape = desired_shape
        self.padding = padding

    def __call__(self, labels):
        distance_trafo = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=25
        )
        labels = distance_trafo(labels)

        # choosing H and W from labels (4, H, W), from above dist trafo outputs
        tmp_ddim = (self.desired_shape[0] - labels.shape[1], self.desired_shape[0] - labels.shape[2])
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
        labels = np.pad(
            labels,
            pad_width=((0, 0), (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
            mode=self.padding
        )
        assert labels.shape[1:] == self.desired_shape, labels.shape
        return labels
