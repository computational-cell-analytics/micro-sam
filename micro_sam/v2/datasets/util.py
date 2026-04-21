def check_loader(loader, n_samples: int = 50, n_target_channels: int = 4):
    """Check that raw inputs have 3 channels and targets have the expected number of channels.

    Args:
        loader: A dataloader yielding (raw, label) batches of shape (B, C, Z, H, W).
        n_samples: Number of batches to check.
            The default is set to 50.
        n_target_channels: Expected number of target channels.
            The current default is set to 4, i.e. foreground, and the three directed distances.
    """
    raw_channel_errors, label_channel_errors = [], []

    for i, (x, y) in enumerate(loader):
        if i >= n_samples:
            break

        if x.shape[1] != 3:
            raw_channel_errors.append(
                f"  batch {i}: raw has {x.shape[1]} channels, expected 3 — shape {tuple(x.shape)}"
            )

        if y.shape[1] != n_target_channels:
            label_channel_errors.append(
                f"  batch {i}: target has {y.shape[1]} channels, expected {n_target_channels} — shape {tuple(y.shape)}"
            )

    print(f"Checked {min(n_samples, i + 1)} batches.")
    if raw_channel_errors:
        print(f"Raw channel errors ({len(raw_channel_errors)}):\n" + "\n".join(raw_channel_errors))
    else:
        print("Raw channels: OK (all have 3 channels)")

    if label_channel_errors:
        print(f"Target channel errors ({len(label_channel_errors)}):\n" + "\n".join(label_channel_errors))
    else:
        print(f"Target channels: OK (all have {n_target_channels} channels)")
