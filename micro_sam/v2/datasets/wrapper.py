"""Wrapper for ensuring all inputs are in a 3d pyramid structure for training UniSAM2.
"""

import torch


class UniDataWrapper(torch.utils.data.Dataset):
    def __init__(self, ds, source_ndim=None, group_key=None):
        self.ds = ds
        self.ndim = 3

        # Track the source dimensionality for batching (2D vs 3D grouping).
        if source_ndim is not None:
            self.source_ndim = source_ndim
        elif hasattr(ds, "ndim"):
            self.source_ndim = ds.ndim
        else:
            self.source_ndim = None

        # Group key for the batch sampler (defaults to source_ndim for backward compat).
        # Use a finer-grained key (e.g. (3, z_slices)) when batches must match on more than just ndim.
        self.group_key = group_key if group_key is not None else self.source_ndim

    def __len__(self):
        return len(self.ds)

    def _to_czyx(self, t):
        t = t if torch.is_tensor(t) else torch.as_tensor(t)

        if t.ndim == 2:  # (Y, X)
            t = t.unsqueeze(0).unsqueeze(1)  # converts to (1, 1, Y, X)
        elif t.ndim == 3:
            if t.shape[0] <= 4:  # (C, Y, X)
                t = t.unsqueeze(1)  # converts to (C, 1, Y, X)
            else:  # (Z, Y, X)
                t = t.unsqueeze(0)  # converts to (1, Z, Y, X)
        elif t.ndim == 4:  # assumes (C, Z, Y, X) and life goes on.
            pass
        else:
            raise ValueError(f"Unsupported ndim {t.ndim}")
        return t

    def __getitem__(self, i):
        raw, label = self.ds[i]
        raw = self._to_czyx(raw).to(torch.float32)
        label = self._to_czyx(label).to(torch.float32)

        # Ensure 3 channels for 3D raw inputs (eg. grayscale volume comes out with 1 channel; triplicate to match 2D).
        if self.source_ndim == 3 and raw.shape[0] == 1:
            raw = torch.cat([raw] * 3, dim=0)

        return raw, label
