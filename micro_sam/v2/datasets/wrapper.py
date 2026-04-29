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
            if self.source_ndim == 3:  # (Z, Y, X)
                t = t.unsqueeze(0)  # converts to (1, Z, Y, X)
            else:  # (C, Y, X) — 2D data, any number of channels
                t = t.unsqueeze(1)  # converts to (C, 1, Y, X)
        elif t.ndim == 4:  # assumes (C, Z, Y, X) and life goes on.
            pass
        else:
            raise ValueError(f"Unsupported ndim {t.ndim}")
        return t

    def __getitem__(self, i):
        raw, label = self.ds[i]
        raw = self._to_czyx(raw).to(torch.float32)
        label = self._to_czyx(label).to(torch.float32)

        # Diagnostic: warn when the instance channel (ch 0) of joint labels is all-zero.
        # Drills through nested ConcatDatasets to find the actual raw/label file path.
        if label.shape[0] >= 5 and label[0].max() == 0:
            import warnings
            import numpy as np

            def _find_leaf(ds, idx):
                offsets = getattr(ds, "ds_offsets", None)
                datasets = getattr(ds, "datasets", None)
                if offsets is None or datasets is None:
                    return ds, idx
                ds_idx = int(np.searchsorted(offsets, idx, side="right"))
                offset = offsets[ds_idx - 1] if ds_idx > 0 else 0
                return _find_leaf(datasets[ds_idx], idx - offset)

            leaf_ds, leaf_idx = _find_leaf(self.ds, i)
            raw_path = getattr(leaf_ds, "raw_path", "unknown")
            label_path = getattr(leaf_ds, "label_path", "unknown")
            ch1_vals = label[1].unique().tolist()  # foreground channel for comparison
            warnings.warn(
                f"UniDataWrapper: instance channel all-zero at index {i} "
                f"(leaf_idx={leaf_idx}, raw={raw_path}, label={label_path}, "
                f"label.shape={tuple(label.shape)}, ch1_unique={ch1_vals}, group_key={self.group_key})",
                stacklevel=2,
            )

        # Encoder patch_embed expects 3 input channels — triplicate any single-channel input.
        if raw.shape[0] == 1:
            raw = torch.cat([raw] * 3, dim=0)

        return raw, label
