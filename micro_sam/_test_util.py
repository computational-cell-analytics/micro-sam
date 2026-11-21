import numpy as np


def check_layer_initialization(viewer, expected_shape):
    """Utility function to check the initial layer setup is correct."""

    assert len(viewer.layers) == 6
    expected_layer_names = [
        "image", "auto_segmentation", "committed_objects", "current_object", "point_prompts", "prompts"
    ]

    for layer_name in expected_layer_names:
        assert layer_name in viewer.layers

    # Check prompt layers
    assert viewer.layers["prompts"].data == []  # shape data is list, not numpy array
    np.testing.assert_equal(viewer.layers["point_prompts"].data, 0)

    # Check segmentation layers.
    for layer_name in ["auto_segmentation", "committed_objects", "current_object"]:
        assert viewer.layers[layer_name].data.shape == expected_shape
        np.testing.assert_equal(viewer.layers[layer_name].data, 0)


def write_container(path, raw, labels):
    """Write an image and its labels as the datasets 'raw' and 'labels' of a h5, n5 or zarr file."""
    import h5py
    import zarr
    import z5py

    if path.endswith(".h5"):
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("labels", data=labels)
    elif path.endswith(".n5"):
        with z5py.File(path, "w") as f:
            f.create_dataset("raw", data=raw, chunks=raw.shape)
            f.create_dataset("labels", data=labels, chunks=labels.shape)
    else:
        group = zarr.open_group(path, mode="w")
        group.create_array("raw", data=raw)
        group.create_array("labels", data=labels)


def write_ome_zarr(path, raw, labels=None, version="0.5", n_scales=1, label_names=("cells",)):
    """Write an image and its label images in the ome-zarr layout, with zarr only.

    Version 0.5 uses zarr v3 and keeps the metadata under 'ome', version 0.4 uses zarr v2 and keeps it at the top.
    Each scale level halves the shape of the previous one. Every label image in `label_names` gets `labels`.
    """
    import zarr

    zarr_format, scale_prefix = (3, "s") if version == "0.5" else (2, "")
    scale_names = [f"{scale_prefix}{level}" for level in range(n_scales)]
    axes = [{"name": name, "type": "space"} for name in ("zyx" if raw.ndim == 3 else "yx")]
    multiscales = [{
        "version": version, "axes": axes,
        "datasets": [
            {"path": name, "coordinateTransformations": [{"type": "scale", "scale": [2.0 ** level] * raw.ndim}]}
            for level, name in enumerate(scale_names)
        ],
    }]

    def _attrs(attrs):
        return {"ome": {"version": version, **attrs}} if version == "0.5" else attrs

    def _write_scales(group, data):
        for level, name in enumerate(scale_names):
            group.create_array(name, data=data[(slice(None, None, 2 ** level),) * data.ndim])

    root = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    root.attrs.update(_attrs({"multiscales": multiscales}))
    _write_scales(root, raw)
    if labels is None:
        return

    label_root = root.create_group("labels")
    label_root.attrs.update(_attrs({"labels": list(label_names)}))
    for name in label_names:
        label_group = label_root.create_group(name)
        label_group.attrs.update(_attrs({"multiscales": multiscales, "image-label": {"version": version}}))
        _write_scales(label_group, labels)
