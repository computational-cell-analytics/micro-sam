"""Napari visualization for UniSAM2 full-volume predictions."""

import argparse
import os

import h5py
import napari
import numpy as np
from skimage.transform import rescale as sk_rescale


OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/grid-search-experiments"

DATASET_H5 = {
    "nis3d": os.path.join(OUTPUT_ROOT, "nis3d/automatic_best_full.h5"),
    "ovules": os.path.join(OUTPUT_ROOT, "plantseg_ovules/automatic_best_full.h5"),
    "mitoem": os.path.join(OUTPUT_ROOT, "mitoem/automatic_best_full.h5"),
    "cremi_padded": "/home/anwai/data/for_usam2/cremi_padded_automatic_best_full.h5",
    "liconn": "/home/anwai/data/for_usam2/liconn_automatic_best_full.h5",
    "microns": "/home/anwai/data/for_usam2/microns_automatic_best_full.h5",
}

# Physical voxel sizes (ZYX)
DATASET_SCALE = {
    "nis3d": (1.0, 1.0, 1.0),
    "ovules": (0.235, 0.075, 0.075),
    "mitoem": (30.0, 8.0, 8.0),
    "cremi_padded": (40.0, 4.0, 4.0),
    "liconn": (8.0, 8.0, 8.0),
    "microns": (12.95, 9.7, 9.7),
}

DATASET_SCALE_UNIT = {
    "nis3d": "um", "ovules": "um",
    "mitoem": "nm", "cremi_padded": "nm", "liconn": "nm", "microns": "nm",
}

# Uniform rescale factor applied to all spatial dims for visualization
DATASET_DS = {
    "nis3d": 0.5,
    "ovules": 0.5,
    "mitoem": 0.125,
    "cremi_padded": 0.25,
    "liconn": 0.5,
    "microns": 0.25,
}

NIS3D_GAP = 100
OVULES_GAP = 25
EM_GAP = 100
CREMI_GAP = 50
LICONN_Z_MAX = 312
LICONN_GAP = 150
EM_TOP_N = 25
EM_SHOW_3D = True
EM_Z_2D = None
EM_BORDER_WIDTH = 20
AXES_LABEL_OFFSET = 0.25

PALETTE = [
    (1.00, 0.10, 0.10), (0.10, 0.60, 1.00), (0.10, 0.85, 0.10), (1.00, 0.85, 0.00), (0.70, 0.10, 1.00),
    (0.00, 0.90, 0.90), (1.00, 0.45, 0.00), (1.00, 0.10, 0.85), (0.45, 1.00, 0.10), (0.00, 0.30, 0.90),
    (1.00, 0.65, 0.75), (0.55, 0.35, 0.10), (0.00, 0.90, 0.50), (0.80, 0.00, 0.30), (0.85, 1.00, 0.10),
    (0.10, 0.10, 0.70), (1.00, 0.55, 0.55), (0.10, 0.70, 0.50), (0.90, 0.70, 0.00), (0.50, 0.00, 0.50),
    (0.00, 0.70, 0.30), (1.00, 0.70, 0.20), (0.30, 0.00, 0.70), (0.00, 1.00, 0.75), (0.75, 0.20, 0.00),
]


def _load_rescaled(h5_path, key, factor, order=1):
    """Load a 3-D dataset from H5 and uniformly rescale with skimage.transform.rescale.

    Pre-strides in h5py for large arrays (>500M voxels) so the in-memory size
    stays manageable before skimage sees it.
    """
    with h5py.File(h5_path, "r") as f:
        shape = f[key].shape
        n_voxels = int(np.prod(shape))
        pre_stride = max(1, int(round((n_voxels / 500_000_000) ** (1 / 3))))
        arr = f[key][::pre_stride, ::pre_stride, ::pre_stride][:]

    effective = factor * pre_stride
    if abs(effective - 1.0) < 1e-6:
        return arr.astype("uint32" if order == 0 else "float32")

    out = sk_rescale(arr.astype("float32"), effective, order=order,
                     anti_aliasing=(order > 0), channel_axis=None)
    return out.astype("uint32" if order == 0 else "float32")


def _vis_scale(name, ds_factor):
    return tuple(s / ds_factor for s in DATASET_SCALE[name])


def _split_seg_topn(seg, n):
    fg = seg.ravel()
    fg = fg[fg > 0]
    if fg.size == 0:
        return seg.copy(), np.zeros_like(seg)
    labels, counts = np.unique(fg, return_counts=True)
    top_labels = set(labels[np.argsort(counts)[-min(n, len(labels)):]])
    mask = np.isin(seg, list(top_labels))
    topn = np.where(mask, seg, 0).astype(seg.dtype)
    rest = np.where(mask | (seg == 0), 0, seg).astype(seg.dtype)
    return topn, rest


def _make_color_dict(top_label_ids):
    d = {0: np.zeros(4, dtype="float32"), None: np.zeros(4, dtype="float32")}
    for i, lid in enumerate(top_label_ids):
        r, g, b = PALETTE[i % len(PALETTE)]
        d[lid] = np.array([r, g, b, 1.0], dtype="float32")
    return d


def _pad_z(arr, before, after):
    pads = [(before, after)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pads, constant_values=0)


def _set_axes_label_offset(viewer):
    try:
        axes_vispy = viewer.window._qt_viewer.canvas._overlay_to_visual[viewer.axes][0]
        axes_vispy.node._text_offsets = AXES_LABEL_OFFSET * np.array([1, 1, 1])
        axes_vispy._on_data_change()
    except Exception:
        pass


def run_nis3d():
    ds = DATASET_DS["nis3d"]
    h5_path = DATASET_H5["nis3d"]
    scale = _vis_scale("nis3d", ds)
    unit = DATASET_SCALE_UNIT["nis3d"]

    print("Loading nis3d ...")
    seg_ds = _load_rescaled(h5_path, "predicted_instances", ds, order=0)
    n_z_ds, n_y_ds, n_x_ds = seg_ds.shape
    first_seg_z_ds = next(z for z in range(n_z_ds) if np.any(seg_ds[z]))
    last_seg_z_ds = next(z for z in range(n_z_ds - 1, -1, -1) if np.any(seg_ds[z]))

    def _rescale_slice(sl):
        return sk_rescale(sl.astype("float32"), ds, order=1, anti_aliasing=True, channel_axis=None)

    with h5py.File(h5_path, "r") as f:
        raw_first = _rescale_slice(f["raw"][int(first_seg_z_ds / ds)][:])
        raw_last = _rescale_slice(f["raw"][int(last_seg_z_ds / ds)][:])

    raw_min = min(float(raw_first.min()), float(raw_last.min()))
    raw_max = max(float(raw_first.max()), float(raw_last.max()))
    clim = (raw_min, raw_max + 1e-6)

    pad_start = max(0, NIS3D_GAP - first_seg_z_ds)
    raw_z_first = max(0, first_seg_z_ds - NIS3D_GAP)
    raw_z_last = pad_start + last_seg_z_ds + NIS3D_GAP
    total_z = raw_z_last + 1
    pad_end = total_z - pad_start - n_z_ds

    seg_padded = _pad_z(seg_ds, pad_start, pad_end)
    raw_vol = np.zeros((total_z, n_y_ds, n_x_ds), dtype=raw_first.dtype)
    raw_vol[raw_z_first] = raw_first
    raw_vol[raw_z_last] = raw_last

    viewer = napari.Viewer(title=f"nis3d (ds={ds})")
    viewer.add_image(raw_vol, name="raw", scale=scale, contrast_limits=clim)
    viewer.add_labels(seg_padded, name="seg", scale=scale)
    viewer.dims.current_step = (raw_z_first, 0, 0)
    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.axes.visible = True
    _set_axes_label_offset(viewer)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = unit
    napari.run()


def run_ovules():
    ds = DATASET_DS["ovules"]
    h5_path = DATASET_H5["ovules"]
    scale = _vis_scale("ovules", ds)
    unit = DATASET_SCALE_UNIT["ovules"]

    print("Loading ovules ...")
    seg_ds = _load_rescaled(h5_path, "predicted_instances", ds, order=0)
    n_z_ds, n_y_ds, n_x_ds = seg_ds.shape
    mid_z = int((n_z_ds // 2) / ds)

    with h5py.File(h5_path, "r") as f:
        raw_2d = sk_rescale(f["raw"][mid_z][:].astype("float32"), ds, order=1, anti_aliasing=True, channel_axis=None)

    clim = (float(raw_2d.min()), float(raw_2d.max()) + 1e-6)
    seg_padded = _pad_z(seg_ds, OVULES_GAP + 1, 0)
    raw_vol = _pad_z(raw_2d[np.newaxis], 0, n_z_ds + OVULES_GAP)

    viewer = napari.Viewer(title=f"ovules (ds={ds})")
    viewer.add_image(raw_vol, name="raw", scale=scale, contrast_limits=clim)
    viewer.add_labels(seg_padded, name="seg", scale=scale)
    viewer.dims.current_step = (0, 0, 0)
    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.axes.visible = True
    _set_axes_label_offset(viewer)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = unit
    napari.run()


def _run_em_style(name):
    from napari.utils.colormaps.colormap import DirectLabelColormap

    ds = DATASET_DS[name]
    h5_path = DATASET_H5[name]
    scale = _vis_scale(name, ds)
    unit = DATASET_SCALE_UNIT[name]

    print(f"Loading {name} ...")
    seg_ds = _load_rescaled(h5_path, "predicted_instances", ds, order=0)
    raw_ds = _load_rescaled(h5_path, "raw", ds, order=1)
    n_z_ds = seg_ds.shape[0]

    clim = (float(raw_ds.min()), float(raw_ds.max()) + 1e-6)
    topn_np, rest_np = _split_seg_topn(seg_ds, EM_TOP_N)
    fg = topn_np.ravel()
    fg = fg[fg > 0]
    ids, cnts = np.unique(fg, return_counts=True)
    top_ids = ids[np.argsort(cnts)[::-1]].tolist()
    color_dict = _make_color_dict(top_ids)

    raw_pos = n_z_ds + EM_GAP
    raw_vol = _pad_z(raw_ds, 0, EM_GAP + 1)
    seg_topn = _pad_z(topn_np, 0, EM_GAP + 1)
    seg_rest = _pad_z(rest_np, 0, EM_GAP + 1)

    if EM_SHOW_3D:
        viewer = napari.Viewer(title=f"{name} 3D (ds={ds})")
        viewer.add_image(raw_vol, name="raw", scale=scale, contrast_limits=clim)
        viewer.add_labels(seg_rest, name="seg rest", scale=scale, opacity=0.25)
        topn_layer = viewer.add_labels(seg_topn, name=f"seg top{EM_TOP_N}", scale=scale, opacity=1.0)
        topn_layer.colormap = DirectLabelColormap(color_dict=color_dict)
        viewer.dims.current_step = (raw_pos, 0, 0)
        viewer.dims.axis_labels = ("z", "y", "x")
        viewer.axes.visible = True
        _set_axes_label_offset(viewer)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = unit
        napari.run()

    z_2d_ds = (n_z_ds // 2) if EM_Z_2D is None else EM_Z_2D
    z_2d_full = int(z_2d_ds / ds)
    with h5py.File(h5_path, "r") as f:
        raw_2d = f["raw"][z_2d_full][:].astype("float32")
        seg_2d = f["predicted_instances"][z_2d_full][:]

    top_set = set(top_ids)
    mask_2d = np.isin(seg_2d, list(top_set))
    topn_2d = np.where(mask_2d, seg_2d, 0).astype(seg_2d.dtype)
    rest_2d = np.where(mask_2d | (seg_2d == 0), 0, seg_2d).astype(seg_2d.dtype)
    clim_2d = (float(raw_2d.min()), float(raw_2d.max()) + 1e-6)

    viewer2d = napari.Viewer(title=f"{name} 2D z={z_2d_full}")
    viewer2d.add_image(raw_2d, name="raw", scale=scale[1:], contrast_limits=clim_2d)
    viewer2d.add_labels(rest_2d, name="seg rest", scale=scale[1:], opacity=0.25)
    fill_layer = viewer2d.add_labels(
        topn_2d, name=f"seg top{EM_TOP_N}", scale=scale[1:], opacity=0.7, blending="additive"
    )
    fill_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer = viewer2d.add_labels(
        topn_2d, name=f"seg top{EM_TOP_N} border", scale=scale[1:], opacity=1.0, blending="additive"
    )
    border_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer.contour = EM_BORDER_WIDTH
    viewer2d.dims.axis_labels = ("y", "x")
    viewer2d.axes.visible = True
    _set_axes_label_offset(viewer2d)
    viewer2d.scale_bar.visible = True
    viewer2d.scale_bar.unit = unit
    napari.run()


def run_mitoem():
    _run_em_style("mitoem")


def run_cremi_padded():
    from napari.utils.colormaps.colormap import DirectLabelColormap

    name = "cremi_padded"
    ds = DATASET_DS[name]
    h5_path = DATASET_H5[name]
    scale = _vis_scale(name, ds)
    unit = DATASET_SCALE_UNIT[name]

    print(f"Loading {name} ...")
    seg_full = _load_rescaled(h5_path, "predicted_instances", ds, order=0)

    nonempty_z = np.where(seg_full.any(axis=(1, 2)))[0]
    z0_ds, z1_ds = int(nonempty_z[0]), int(nonempty_z[-1]) + 1
    seg_ds = seg_full[z0_ds:z1_ds]
    n_z_ds, n_y_ds, n_x_ds = seg_ds.shape

    first_z_full = round(z0_ds / ds)
    with h5py.File(h5_path, "r") as f:
        raw_2d = sk_rescale(
            f["raw"][first_z_full][:].astype("float32"), ds, order=1, anti_aliasing=True, channel_axis=None
        )[:n_y_ds, :n_x_ds]

    clim = (float(raw_2d.min()), float(raw_2d.max()) + 1e-6)
    topn_np, rest_np = _split_seg_topn(seg_ds, EM_TOP_N)
    fg = topn_np.ravel()
    fg = fg[fg > 0]
    ids, cnts = np.unique(fg, return_counts=True)
    top_ids = ids[np.argsort(cnts)[::-1]].tolist()
    color_dict = _make_color_dict(top_ids)

    seg_start = CREMI_GAP + 1
    raw_vol = np.zeros((seg_start + n_z_ds, n_y_ds, n_x_ds), dtype=raw_2d.dtype)
    raw_vol[0] = raw_2d
    seg_topn = _pad_z(topn_np, seg_start, 0)
    seg_rest = _pad_z(rest_np, seg_start, 0)

    if EM_SHOW_3D:
        viewer = napari.Viewer(title=f"{name} 3D (ds={ds})")
        viewer.add_image(raw_vol, name="raw", scale=scale, contrast_limits=clim)
        viewer.add_labels(seg_rest, name="seg rest", scale=scale, opacity=0.25)
        topn_layer = viewer.add_labels(seg_topn, name=f"seg top{EM_TOP_N}", scale=scale, opacity=1.0)
        topn_layer.colormap = DirectLabelColormap(color_dict=color_dict)
        viewer.dims.current_step = (0, 0, 0)
        viewer.dims.axis_labels = ("z", "y", "x")
        viewer.axes.visible = True
        _set_axes_label_offset(viewer)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = unit
        napari.run()

    z_2d_ds = z0_ds + (n_z_ds // 2)
    z_2d_full = round(z_2d_ds / ds)
    with h5py.File(h5_path, "r") as f:
        raw_2d_view = f["raw"][z_2d_full][:].astype("float32")
        seg_2d = f["predicted_instances"][z_2d_full][:]

    top_set = set(top_ids)
    mask_2d = np.isin(seg_2d, list(top_set))
    topn_2d = np.where(mask_2d, seg_2d, 0).astype(seg_2d.dtype)
    rest_2d = np.where(mask_2d | (seg_2d == 0), 0, seg_2d).astype(seg_2d.dtype)
    clim_2d = (float(raw_2d_view.min()), float(raw_2d_view.max()) + 1e-6)

    viewer2d = napari.Viewer(title=f"{name} 2D z={z_2d_full}")
    viewer2d.add_image(raw_2d_view, name="raw", scale=scale[1:], contrast_limits=clim_2d)
    viewer2d.add_labels(rest_2d, name="seg rest", scale=scale[1:], opacity=0.25)
    fill_layer = viewer2d.add_labels(
        topn_2d, name=f"seg top{EM_TOP_N}", scale=scale[1:], opacity=0.7, blending="additive"
    )
    fill_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer = viewer2d.add_labels(
        topn_2d, name=f"seg top{EM_TOP_N} border", scale=scale[1:], opacity=1.0, blending="additive"
    )
    border_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer.contour = EM_BORDER_WIDTH
    viewer2d.dims.axis_labels = ("y", "x")
    viewer2d.axes.visible = True
    _set_axes_label_offset(viewer2d)
    viewer2d.scale_bar.visible = True
    viewer2d.scale_bar.unit = unit
    napari.run()


def run_liconn():
    from napari.utils.colormaps.colormap import DirectLabelColormap

    name = "liconn"
    ds = DATASET_DS[name]
    h5_path = DATASET_H5[name]
    scale = _vis_scale(name, ds)
    unit = DATASET_SCALE_UNIT[name]

    print(f"Loading {name} ...")
    seg_full = _load_rescaled(h5_path, "predicted_instances", ds, order=0)
    seg_ds = seg_full[:LICONN_Z_MAX]
    n_z_ds, n_y_ds, n_x_ds = seg_ds.shape

    with h5py.File(h5_path, "r") as f:
        raw_2d = sk_rescale(
            f["raw"][0][:].astype("float32"), ds, order=1, anti_aliasing=True, channel_axis=None
        )[:n_y_ds, :n_x_ds]

    clim = (float(raw_2d.min()), float(raw_2d.max()) + 1e-6)

    last_slice = seg_ds[LICONN_Z_MAX - 1]
    fg_last = last_slice.ravel()
    fg_last = fg_last[fg_last > 0]
    ids_last, cnts_last = np.unique(fg_last, return_counts=True)
    top_ids = ids_last[np.argsort(cnts_last)[::-1]][:EM_TOP_N].tolist()
    top_set = set(top_ids)
    mask = np.isin(seg_ds, list(top_set))
    topn_np = np.where(mask, seg_ds, 0).astype(seg_ds.dtype)
    rest_np = np.where(mask | (seg_ds == 0), 0, seg_ds).astype(seg_ds.dtype)
    color_dict = _make_color_dict(top_ids)

    seg_start = LICONN_GAP + 1
    raw_vol = np.zeros((seg_start + n_z_ds, n_y_ds, n_x_ds), dtype=raw_2d.dtype)
    raw_vol[0] = raw_2d
    seg_topn = _pad_z(topn_np, seg_start, 0)
    seg_rest = _pad_z(rest_np, seg_start, 0)

    if EM_SHOW_3D:
        viewer = napari.Viewer(title=f"{name} 3D z=0..{LICONN_Z_MAX - 1} top10-last-slice (ds={ds})")
        viewer.add_image(raw_vol, name="raw", scale=scale, contrast_limits=clim)
        viewer.add_labels(seg_rest, name="seg rest", scale=scale, opacity=0.25)
        topn_layer = viewer.add_labels(seg_topn, name=f"seg top{EM_TOP_N}", scale=scale, opacity=1.0)
        topn_layer.colormap = DirectLabelColormap(color_dict=color_dict)
        viewer.dims.current_step = (0, 0, 0)
        viewer.dims.axis_labels = ("z", "y", "x")
        viewer.axes.visible = True
        _set_axes_label_offset(viewer)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = unit
        napari.run()

    z_2d_ds = LICONN_Z_MAX - 1
    z_2d_full = round(z_2d_ds / ds)
    with h5py.File(h5_path, "r") as f:
        raw_2d_view = f["raw"][z_2d_full][:].astype("float32")
        seg_2d = f["predicted_instances"][z_2d_full][:]

    mask_2d = np.isin(seg_2d, list(top_set))
    topn_2d = np.where(mask_2d, seg_2d, 0).astype(seg_2d.dtype)
    rest_2d = np.where(mask_2d | (seg_2d == 0), 0, seg_2d).astype(seg_2d.dtype)
    clim_2d = (float(raw_2d_view.min()), float(raw_2d_view.max()) + 1e-6)

    viewer2d = napari.Viewer(title=f"{name} 2D z={z_2d_full}")
    viewer2d.add_image(raw_2d_view, name="raw", scale=scale[1:], contrast_limits=clim_2d)
    viewer2d.add_labels(rest_2d, name="seg rest", scale=scale[1:], opacity=0.25)
    fill_layer = viewer2d.add_labels(
        topn_2d, name=f"seg top{EM_TOP_N}", scale=scale[1:], opacity=0.7, blending="additive"
    )
    fill_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer = viewer2d.add_labels(
        topn_2d, name=f"seg top{EM_TOP_N} border", scale=scale[1:], opacity=1.0, blending="additive"
    )
    border_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer.contour = EM_BORDER_WIDTH
    viewer2d.dims.axis_labels = ("y", "x")
    viewer2d.axes.visible = True
    _set_axes_label_offset(viewer2d)
    viewer2d.scale_bar.visible = True
    viewer2d.scale_bar.unit = unit
    napari.run()


def run_microns():
    from napari.utils.colormaps.colormap import DirectLabelColormap

    name = "microns"
    ds = DATASET_DS[name]
    h5_path = DATASET_H5[name]
    scale = _vis_scale(name, ds)
    unit = DATASET_SCALE_UNIT[name]

    print(f"Loading {name} ...")
    seg_full = _load_rescaled(h5_path, "predicted_instances", ds, order=0)

    nonempty_z = np.where(seg_full.any(axis=(1, 2)))[0]
    z0_ds, z1_ds = int(nonempty_z[0]), int(nonempty_z[-1]) + 1
    seg_ds = seg_full[z0_ds:z1_ds]
    n_z_ds, n_y_ds, n_x_ds = seg_ds.shape

    first_z_full = round(z0_ds / ds)
    with h5py.File(h5_path, "r") as f:
        raw_2d = sk_rescale(
            f["raw"][first_z_full][:].astype("float32"), ds, order=1, anti_aliasing=True, channel_axis=None
        )[:n_y_ds, :n_x_ds]

    clim = (float(raw_2d.min()), float(raw_2d.max()) + 1e-6)

    last_slice = seg_ds[-1]
    fg_last = last_slice.ravel()
    fg_last = fg_last[fg_last > 0]
    ids_last, cnts_last = np.unique(fg_last, return_counts=True)
    top_ids = ids_last[np.argsort(cnts_last)[::-1]][:EM_TOP_N].tolist()
    top_set = set(top_ids)
    mask = np.isin(seg_ds, list(top_set))
    topn_np = np.where(mask, seg_ds, 0).astype(seg_ds.dtype)
    rest_np = np.where(mask | (seg_ds == 0), 0, seg_ds).astype(seg_ds.dtype)
    color_dict = _make_color_dict(top_ids)

    seg_start = EM_GAP + 1
    raw_vol = np.zeros((seg_start + n_z_ds, n_y_ds, n_x_ds), dtype=raw_2d.dtype)
    raw_vol[0] = raw_2d
    seg_topn = _pad_z(topn_np, seg_start, 0)
    seg_rest = _pad_z(rest_np, seg_start, 0)

    if EM_SHOW_3D:
        viewer = napari.Viewer(title=f"{name} 3D (ds={ds})")
        viewer.add_image(raw_vol, name="raw", scale=scale, contrast_limits=clim)
        viewer.add_labels(seg_rest, name="seg rest", scale=scale, opacity=0.25)
        topn_layer = viewer.add_labels(seg_topn, name=f"seg top{EM_TOP_N}", scale=scale, opacity=1.0)
        topn_layer.colormap = DirectLabelColormap(color_dict=color_dict)
        viewer.dims.current_step = (0, 0, 0)
        viewer.dims.axis_labels = ("z", "y", "x")
        viewer.axes.visible = True
        _set_axes_label_offset(viewer)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = unit
        napari.run()

    z_last_full = round((z0_ds + n_z_ds - 1) / ds)
    with h5py.File(h5_path, "r") as f:
        raw_2d_view = f["raw"][z_last_full][:].astype("float32")
        seg_2d = f["predicted_instances"][z_last_full][:]

    mask_2d = np.isin(seg_2d, list(top_set))
    topn_2d = np.where(mask_2d, seg_2d, 0).astype(seg_2d.dtype)
    rest_2d = np.where(mask_2d | (seg_2d == 0), 0, seg_2d).astype(seg_2d.dtype)
    clim_2d = (float(raw_2d_view.min()), float(raw_2d_view.max()) + 1e-6)

    viewer2d = napari.Viewer(title=f"{name} 2D z={z_last_full}")
    viewer2d.add_image(raw_2d_view, name="raw", scale=scale[1:], contrast_limits=clim_2d)
    viewer2d.add_labels(rest_2d, name="seg rest", scale=scale[1:], opacity=0.25)
    fill_layer = viewer2d.add_labels(
        topn_2d, name=f"seg top{EM_TOP_N}", scale=scale[1:], opacity=0.7, blending="additive"
    )
    fill_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer = viewer2d.add_labels(
        topn_2d, name=f"seg top{EM_TOP_N} border", scale=scale[1:], opacity=1.0, blending="additive"
    )
    border_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer.contour = EM_BORDER_WIDTH
    viewer2d.dims.axis_labels = ("y", "x")
    viewer2d.axes.visible = True
    _set_axes_label_offset(viewer2d)
    viewer2d.scale_bar.visible = True
    viewer2d.scale_bar.unit = unit
    napari.run()


def main():
    choices = list(DATASET_H5)
    parser = argparse.ArgumentParser(description="Napari visualization for UniSAM2 predictions.")
    parser.add_argument("--datasets", nargs="+", choices=choices, default=choices)
    args = parser.parse_args()

    dispatch = {
        "nis3d": run_nis3d, "ovules": run_ovules, "mitoem": run_mitoem,
        "cremi_padded": run_cremi_padded, "liconn": run_liconn, "microns": run_microns,
    }
    for ds in args.datasets:
        dispatch[ds]()


if __name__ == "__main__":
    main()
