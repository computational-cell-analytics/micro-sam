#!/usr/bin/env python
"""Napari visualization: nis3d and mitoem in nis3d style; ovules with full seg."""
import argparse
import dask.array as da
import h5py
import napari
import numpy as np
import tifffile
import zarr

NIS3D_H5 = '/home/anwai/data/for_usam2/nis3d_automatic_best_full.h5'
NIS3D_TIF = '/home/anwai/data/for_usam2/prediction_nis3d.tif'
NIS3D_SCALE = (1, 1, 1)  # isotropic: 1 um x 1 um x 1 um
NIS3D_GAP = 100  # gap slices between raw plane and nearest seg slice
NIS3D_DS = 2  # spatial downsample factor to fit GPU texture limits

OVULES_H5 = '/home/anwai/data/for_usam2/plantseg_ovules_automatic_best_full.h5'
OVULES_TIF = '/home/anwai/data/for_usam2/prediction_ovules.tif'
OVULES_SCALE = (0.235, 0.075, 0.075)  # anisotropic: 0.235 um x 0.075 um x 0.075 um (ZYX)
OVULES_GAP = 25  # gap slices between raw plane and first seg slice
OVULES_DS = 2  # spatial downsample factor to fit GPU texture limits

MITOEM_H5 = '/home/anwai/data/for_usam2/mitoem_automatic_segmentation.h5'
MITOEM_SCALE = (30, 8, 8)  # anisotropic: 30 nm x 8 nm x 8 nm (ZYX)
MITOEM_GAP = 100  # gap slices between raw plane and first seg slice
MITOEM_DS_Z = 1  # z downsample factor (keep full z resolution for anisotropy)
MITOEM_DS_XY = 8  # xy downsample factor to fit GPU texture limits
MITOEM_TOP_N = 25  # number of largest segments shown at full opacity
MITOEM_PALETTE = [
    (1.00, 0.10, 0.10), (0.10, 0.60, 1.00), (0.10, 0.85, 0.10), (1.00, 0.85, 0.00), (0.70, 0.10, 1.00),
    (0.00, 0.90, 0.90), (1.00, 0.45, 0.00), (1.00, 0.10, 0.85), (0.45, 1.00, 0.10), (0.00, 0.30, 0.90),
    (1.00, 0.65, 0.75), (0.55, 0.35, 0.10), (0.00, 0.90, 0.50), (0.80, 0.00, 0.30), (0.85, 1.00, 0.10),
    (0.10, 0.10, 0.70), (1.00, 0.55, 0.55), (0.10, 0.70, 0.50), (0.90, 0.70, 0.00), (0.50, 0.00, 0.50),
    (0.00, 0.70, 0.30), (1.00, 0.70, 0.20), (0.30, 0.00, 0.70), (0.00, 1.00, 0.75), (0.75, 0.20, 0.00),
]
MITOEM_2D_Z = None  # z slice for 2D view (None = middle of downsampled volume)
MITOEM_SHOW_3D = True  # set True to also open the 3D viewer before the 2D one
MITOEM_BORDER_WIDTH = 20  # contour width for the top-N border layer

AXES_LABEL_OFFSET = 0.25  # distance of axis labels from arrow tips (default napari: 0.1)


def _split_seg_topn(seg_np, n):
    fg = seg_np.ravel()
    fg = fg[fg > 0]
    if fg.size == 0:
        return seg_np.copy(), np.zeros_like(seg_np)
    labels, counts = np.unique(fg, return_counts=True)
    top_labels = set(labels[np.argsort(counts)[-min(n, len(labels)):]])
    mask = np.isin(seg_np, list(top_labels))
    topn = np.where(mask, seg_np, 0).astype(seg_np.dtype)
    rest = np.where(mask | (seg_np == 0), 0, seg_np).astype(seg_np.dtype)
    return topn, rest


def _pad_seg(arr_np, pad_before, pad_after):
    n_y, n_x = arr_np.shape[1], arr_np.shape[2]
    parts = []
    if pad_before > 0:
        parts.append(da.zeros((pad_before, n_y, n_x), dtype=arr_np.dtype, chunks=(1, n_y, n_x)))
    parts.append(da.from_array(arr_np, chunks=(1, n_y, n_x)))
    if pad_after > 0:
        parts.append(da.zeros((pad_after, n_y, n_x), dtype=arr_np.dtype, chunks=(1, n_y, n_x)))
    return da.concatenate(parts, axis=0)


def _set_axes_label_offset(viewer):
    try:
        axes_vispy = viewer.window._qt_viewer.canvas._overlay_to_visual[viewer.axes][0]
        axes_vispy.node._text_offsets = AXES_LABEL_OFFSET * np.array([1, 1, 1])
        axes_vispy._on_data_change()
    except Exception:
        pass


def run_nis3d():
    tif_nis = tifffile.TiffFile(NIS3D_TIF)
    store_nis = tif_nis.aszarr()
    z_arr_nis = zarr.open(store_nis, mode='r')
    n_z, n_y, n_x = z_arr_nis.shape
    first_seg_z = next(z for z in range(n_z) if np.any(z_arr_nis[z]))
    last_seg_z = next(z for z in range(n_z - 1, -1, -1) if np.any(z_arr_nis[z]))

    seg_nis_ds = da.from_zarr(z_arr_nis, chunks=(1, n_y, n_x))[::NIS3D_DS, ::NIS3D_DS, ::NIS3D_DS]
    n_z_ds, n_y_ds, n_x_ds = seg_nis_ds.shape
    first_seg_z_ds = first_seg_z // NIS3D_DS
    last_seg_z_ds = last_seg_z // NIS3D_DS
    nis3d_scale = tuple(s * NIS3D_DS for s in NIS3D_SCALE)

    pad_start = max(0, NIS3D_GAP - first_seg_z_ds)
    raw_z_first = max(0, first_seg_z_ds - NIS3D_GAP)
    raw_z_last = pad_start + last_seg_z_ds + NIS3D_GAP
    total_z = raw_z_last + 1
    pad_end = total_z - pad_start - n_z_ds

    with h5py.File(NIS3D_H5, 'r') as f:
        raw_first = np.array(f['raw'][first_seg_z])[::NIS3D_DS, ::NIS3D_DS]
        raw_last = np.array(f['raw'][last_seg_z])[::NIS3D_DS, ::NIS3D_DS]

    raw_min = min(float(raw_first.min()), float(raw_last.min()))
    raw_max = max(float(raw_first.max()), float(raw_last.max()))
    clim_nis = (raw_min, raw_max if raw_max > raw_min else raw_min + 1.0)

    gap_between = raw_z_last - raw_z_first - 1
    raw_parts = []
    if raw_z_first > 0:
        raw_parts.append(da.zeros((raw_z_first, n_y_ds, n_x_ds), dtype=raw_first.dtype, chunks=(1, n_y_ds, n_x_ds)))
    raw_parts.append(da.from_array(raw_first[np.newaxis], chunks=(1, n_y_ds, n_x_ds)))
    if gap_between > 0:
        raw_parts.append(da.zeros((gap_between, n_y_ds, n_x_ds), dtype=raw_first.dtype, chunks=(1, n_y_ds, n_x_ds)))
    raw_parts.append(da.from_array(raw_last[np.newaxis], chunks=(1, n_y_ds, n_x_ds)))
    raw_vol_nis = da.concatenate(raw_parts, axis=0)

    seg_parts = []
    if pad_start > 0:
        seg_parts.append(da.zeros((pad_start, n_y_ds, n_x_ds), dtype=z_arr_nis.dtype, chunks=(1, n_y_ds, n_x_ds)))
    seg_parts.append(seg_nis_ds)
    if pad_end > 0:
        seg_parts.append(da.zeros((pad_end, n_y_ds, n_x_ds), dtype=z_arr_nis.dtype, chunks=(1, n_y_ds, n_x_ds)))
    seg_nis = da.concatenate(seg_parts, axis=0)

    viewer = napari.Viewer(title=f'nis3d - raw at {raw_z_first} and {raw_z_last} (ds={NIS3D_DS})')
    viewer.add_image(raw_vol_nis, name='raw', scale=nis3d_scale, contrast_limits=clim_nis)
    viewer.add_labels(seg_nis, name='seg', scale=nis3d_scale)
    viewer.dims.current_step = (raw_z_first, 0, 0)
    viewer.dims.axis_labels = ('z', 'y', 'x')
    viewer.axes.visible = True
    _set_axes_label_offset(viewer)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'
    napari.run()
    store_nis.close()
    tif_nis.close()


def run_ovules():
    tif_ov = tifffile.TiffFile(OVULES_TIF)
    store_ov = tif_ov.aszarr()
    z_arr_ov = zarr.open(store_ov, mode='r')
    n_z_ov, n_y_ov, n_x_ov = z_arr_ov.shape

    seg_ov = da.from_zarr(z_arr_ov, chunks=(1, n_y_ov, n_x_ov))[::OVULES_DS, ::OVULES_DS, ::OVULES_DS]
    n_z_ds, n_y_ds, n_x_ds = seg_ov.shape
    mid_z_ds = n_z_ds // 2

    with h5py.File(OVULES_H5, 'r') as f:
        raw_2d_ov = np.array(f['raw'][mid_z_ds * OVULES_DS])[::OVULES_DS, ::OVULES_DS]

    raw_min_ov, raw_max_ov = float(raw_2d_ov.min()), float(raw_2d_ov.max())
    clim_ov = (raw_min_ov, raw_max_ov if raw_max_ov > raw_min_ov else raw_min_ov + 1.0)

    seg_start = OVULES_GAP + 1

    seg_ov_new = da.concatenate([
        da.zeros((seg_start, n_y_ds, n_x_ds), dtype=seg_ov.dtype, chunks=(1, n_y_ds, n_x_ds)),
        seg_ov,
    ], axis=0)
    raw_vol_ov = da.concatenate([
        da.from_array(raw_2d_ov[np.newaxis], chunks=(1, n_y_ds, n_x_ds)),
        da.zeros((n_z_ds + OVULES_GAP, n_y_ds, n_x_ds), dtype=raw_2d_ov.dtype, chunks=(1, n_y_ds, n_x_ds)),
    ], axis=0)

    viewer = napari.Viewer(title=f'ovules - raw at 0, seg z={seg_start}..{seg_start + n_z_ds - 1} (ds={OVULES_DS})')
    viewer.add_image(raw_vol_ov, name='raw', scale=OVULES_SCALE, contrast_limits=clim_ov)
    viewer.add_labels(seg_ov_new, name='seg', scale=OVULES_SCALE)
    viewer.dims.current_step = (0, 0, 0)
    viewer.dims.axis_labels = ('z', 'y', 'x')
    viewer.axes.visible = True
    _set_axes_label_offset(viewer)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'
    napari.run()
    store_ov.close()
    tif_ov.close()


def run_mitoem():
    f = h5py.File(MITOEM_H5, 'r')
    seg_h5 = f['predicted_instances']
    raw_h5 = f['raw']
    n_z, n_y, n_x = seg_h5.shape

    first_seg_z = next(z for z in range(n_z) if np.any(seg_h5[z, ::32, ::32]))

    seg_ds = da.from_array(seg_h5, chunks=(1, n_y, n_x))[::MITOEM_DS_Z, ::MITOEM_DS_XY, ::MITOEM_DS_XY]
    n_z_ds, n_y_ds, n_x_ds = seg_ds.shape
    mitoem_scale = (MITOEM_SCALE[0] * MITOEM_DS_Z, MITOEM_SCALE[1] * MITOEM_DS_XY, MITOEM_SCALE[2] * MITOEM_DS_XY)

    raw_slice = np.array(raw_h5[first_seg_z])[::MITOEM_DS_XY, ::MITOEM_DS_XY]
    raw_min, raw_max = float(raw_slice.min()), float(raw_slice.max())
    clim = (raw_min, raw_max if raw_max > raw_min else raw_min + 1.0)

    raw_pos = n_z_ds + MITOEM_GAP

    topn_np, rest_np = _split_seg_topn(seg_ds.compute(), MITOEM_TOP_N)
    import napari.utils.colormaps.colormap as napari_cmap
    DirectLabelColormap = napari_cmap.DirectLabelColormap
    fg = topn_np.ravel()
    fg = fg[fg > 0]
    unique_top, top_counts = np.unique(fg, return_counts=True)
    top_label_ids = unique_top[np.argsort(top_counts)[::-1]].tolist()
    topn_color_dict = {0: np.zeros(4, dtype=np.float32), None: np.zeros(4, dtype=np.float32)}
    for i, lid in enumerate(top_label_ids):
        r, g, b = MITOEM_PALETTE[i % len(MITOEM_PALETTE)]
        topn_color_dict[lid] = np.array([r, g, b, 1.0], dtype=np.float32)
    seg_topn = _pad_seg(topn_np, 0, MITOEM_GAP + 1)
    seg_rest = _pad_seg(rest_np, 0, MITOEM_GAP + 1)

    raw_vol = da.concatenate([
        da.zeros((raw_pos, n_y_ds, n_x_ds), dtype=raw_slice.dtype, chunks=(1, n_y_ds, n_x_ds)),
        da.from_array(raw_slice[np.newaxis], chunks=(1, n_y_ds, n_x_ds)),
    ], axis=0)

    if MITOEM_SHOW_3D:
        title = f'mitoem - seg z=0..{n_z_ds - 1}, raw at {raw_pos} (ds_z={MITOEM_DS_Z}, ds_xy={MITOEM_DS_XY})'
        viewer = napari.Viewer(title=title)
        viewer.add_image(raw_vol, name='raw', scale=mitoem_scale, contrast_limits=clim)
        viewer.add_labels(seg_rest, name='seg rest', scale=mitoem_scale, opacity=0.25)
        topn_layer = viewer.add_labels(seg_topn, name=f'seg top{MITOEM_TOP_N}', scale=mitoem_scale, opacity=1.0)
        DirectLabelColormap = type(topn_layer._direct_colormap)
        topn_layer.colormap = DirectLabelColormap(color_dict=topn_color_dict)
        viewer.dims.current_step = (raw_pos, 0, 0)
        viewer.dims.axis_labels = ('z', 'y', 'x')
        viewer.axes.visible = True
        _set_axes_label_offset(viewer)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = 'nm'
        napari.run()

    z_2d = (n_z_ds // 2) if MITOEM_2D_Z is None else MITOEM_2D_Z
    z_2d_full = z_2d * MITOEM_DS_Z
    raw_2d = np.array(raw_h5[z_2d_full])
    seg_2d = np.array(seg_h5[z_2d_full])

    top_labels_2d = set(np.unique(topn_np).tolist()) - {0}
    mask_2d = np.isin(seg_2d, list(top_labels_2d))
    topn_2d = np.where(mask_2d, seg_2d, 0).astype(seg_2d.dtype)
    rest_2d = np.where(mask_2d | (seg_2d == 0), 0, seg_2d).astype(seg_2d.dtype)

    raw_min_2d, raw_max_2d = float(raw_2d.min()), float(raw_2d.max())
    clim_2d = (raw_min_2d, raw_max_2d if raw_max_2d > raw_min_2d else raw_min_2d + 1.0)
    scale_2d = (MITOEM_SCALE[1], MITOEM_SCALE[2])

    viewer2d = napari.Viewer(title=f'mitoem 2D - z={z_2d_full} full res')
    viewer2d.add_image(raw_2d, name='raw', scale=scale_2d, contrast_limits=clim_2d)
    viewer2d.add_labels(rest_2d, name='seg rest', scale=scale_2d, opacity=0.25)
    topn_fill_2d = viewer2d.add_labels(
        topn_2d, name=f'seg top{MITOEM_TOP_N}', scale=scale_2d, opacity=0.7, blending='additive'
    )
    topn_fill_2d.colormap = DirectLabelColormap(color_dict=topn_color_dict)
    topn_border_2d = viewer2d.add_labels(
        topn_2d, name=f'seg top{MITOEM_TOP_N} border', scale=scale_2d, opacity=1.0, blending='additive'
    )
    topn_border_2d.colormap = DirectLabelColormap(color_dict=topn_color_dict)
    topn_border_2d.contour = MITOEM_BORDER_WIDTH
    viewer2d.dims.axis_labels = ('y', 'x')
    viewer2d.axes.visible = True
    _set_axes_label_offset(viewer2d)
    viewer2d.scale_bar.visible = True
    viewer2d.scale_bar.unit = 'nm'
    napari.run()
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Napari visualization for microscopy datasets.')
    parser.add_argument(
        '--datasets', nargs='+', choices=['nis3d', 'ovules', 'mitoem'], default=['nis3d', 'ovules', 'mitoem']
    )
    args = parser.parse_args()

    dispatch = {'nis3d': run_nis3d, 'ovules': run_ovules, 'mitoem': run_mitoem}
    for ds in args.datasets:
        dispatch[ds]()
