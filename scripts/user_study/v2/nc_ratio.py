"""The nucleus-to-cytoplasm (N/C) ratio of cells from a cell and a nucleus instance segmentation.

Definition:
    cytoplasm volume = cell volume - nucleus volume
    N/C ratio = nucleus volume / cytoplasm volume

The cell volume is the volume of the whole cell, including its nucleus. The volumes are in physical units if the
pixel / voxel size is given and in pixels / voxels otherwise. The ratio does not depend on the unit.

Usage:
    from nc_ratio import compute_nc_ratio

    table = compute_nc_ratio(cells, nuclei, voxel_size=(0.1, 0.0332, 0.0332), unit="um")
    table = table[table["n_nuclei"] == 1]

The segmentations can be numpy arrays or chunked arrays (zarr, h5py, ...), which are read in slabs, so that volumes
that do not fit into memory work as well. Both must be on the same grid (same shape and pixel / voxel size).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# The pairs of cell and nucleus ids are counted with a dense table up to this number of entries.
_MAX_TABLE_SIZE = 2 ** 26


def _count_pairs(cells, nuclei):
    """Count the pixels of each pair (cell id, nucleus id) of a block, including the id 0 (background).

    Returns:
        The cell ids of the pairs.
        The nucleus ids of the pairs.
        The number of pixels of the pairs.
    """
    cells, nuclei = cells.ravel(), nuclei.ravel()
    n_cells, n_nuclei = int(cells.max()) + 1, int(nuclei.max()) + 1
    if n_cells * n_nuclei <= _MAX_TABLE_SIZE:
        keys = cells.astype("int64")
        keys *= n_nuclei
        keys += nuclei
        counts = np.bincount(keys, minlength=n_cells * n_nuclei)
        keys = np.flatnonzero(counts)
        counts = counts[keys]
    else:  # Large or sparse ids.
        keys = cells.astype("uint64")
        keys *= np.uint64(n_nuclei)
        keys += nuclei.astype("uint64")
        keys, counts = np.unique(keys, return_counts=True)
    return (keys // n_nuclei).astype("int64"), (keys % n_nuclei).astype("int64"), counts.astype("int64")


def _get_border_ids(block, start, stop, size):
    """The ids of a slab of the first axis (from start to stop of size) on the border of the full array."""
    faces = [np.take(block, index, axis=axis) for axis in range(1, block.ndim) for index in (0, -1)]
    if start == 0:
        faces.append(block[0])
    if stop == size:
        faces.append(block[-1])
    return np.unique(np.concatenate([np.unique(face) for face in faces]))


def _check_labels(labels, name):
    dtype = np.dtype(labels.dtype)
    if not (np.issubdtype(dtype, np.integer) or dtype == bool):
        raise ValueError(f"The {name} must be an integer label array, got the data type {dtype}.")


def count_overlaps(cells, nuclei, slab_size=None, num_workers=1):
    """Count the pixels of each pair of cell and nucleus, and find the objects on the border of the array.

    The arrays are read in slabs along the first axis, so they can be larger than memory.

    Args:
        cells: The instance segmentation of the cells, 0 is background. A 2D or 3D array (numpy, zarr, h5py, ...).
        nuclei: The instance segmentation of the nuclei, with the same shape.
        slab_size: The number of slices along the first axis that are read at once. By default the chunk size of
            the first axis, or 32 for arrays without chunks.
        num_workers: The number of threads.

    Returns:
        A table with the columns 'cell_id', 'nucleus_id' and 'size' (number of pixels). The id 0 is the background,
            so the rows with the cell id 0 are the parts of nuclei outside of all cells.
        The ids of the cells on the border of the array.
        The ids of the nuclei on the border of the array.
    """
    if tuple(cells.shape) != tuple(nuclei.shape):
        raise ValueError(
            f"The cells and nuclei must be on the same grid, got the shapes {cells.shape} and {nuclei.shape}."
        )
    if len(cells.shape) not in (2, 3):
        raise ValueError(f"Expected 2D or 3D segmentations, got {len(cells.shape)} dimensions.")
    _check_labels(cells, "cells")
    _check_labels(nuclei, "nuclei")
    size = cells.shape[0]
    if slab_size is None:
        chunks = getattr(cells, "chunks", None)
        slab_size = 32 if chunks is None else chunks[0]

    def process(start):
        stop = min(start + slab_size, size)
        cell_block, nucleus_block = np.asarray(cells[start:stop]), np.asarray(nuclei[start:stop])
        if min(cell_block.min(), nucleus_block.min()) < 0:
            raise ValueError("The segmentations must not have negative ids.")
        return (
            _count_pairs(cell_block, nucleus_block),
            _get_border_ids(cell_block, start, stop, size), _get_border_ids(nucleus_block, start, stop, size),
        )

    with ThreadPoolExecutor(num_workers) as pool:
        results = list(pool.map(process, range(0, size, slab_size)))

    pairs = pd.DataFrame({
        name: np.concatenate([result[0][i] for result in results])
        for i, name in enumerate(("cell_id", "nucleus_id", "size"))
    })
    pairs = pairs.groupby(["cell_id", "nucleus_id"], as_index=False)["size"].sum()
    cell_border_ids = np.unique(np.concatenate([result[1] for result in results]))
    nucleus_border_ids = np.unique(np.concatenate([result[2] for result in results]))
    return pairs, cell_border_ids[cell_border_ids != 0], nucleus_border_ids[nucleus_border_ids != 0]


def compute_nc_ratio(cells, nuclei, voxel_size=None, unit=None, min_overlap=0.5, slab_size=None, num_workers=1):
    """Compute the cell, nucleus and cytoplasm volume and the nucleus-to-cytoplasm (N/C) ratio of each cell.

    The cytoplasm volume is the cell volume minus the nucleus volume and the N/C ratio is the nucleus volume
    divided by the cytoplasm volume. The cell is the whole cell, including its nucleus.

    The nuclei are matched to the cells by overlap, the ids of a cell and its nucleus do not have to be the same:
    a nucleus belongs to the cell that holds most of its pixels, if this is at least the fraction 'min_overlap'
    of the nucleus. Nuclei that are not matched to a cell are not in the result.

    The volumes are measured so that they stay consistent if the two segmentations disagree:
    - The nucleus volume is the volume of the nuclei of the cell, wherever their pixels are.
    - The cell volume is the volume of the cell label plus the volume of its nuclei outside of the cell label,
      because the nucleus is part of the cell. So the cytoplasm is exactly the part of the cell label that is not
      covered by the nuclei of the cell, and 'nucleus_fraction_in_cell' gives the fraction of the nucleus volume
      inside the cell label (1 if the segmentations are consistent).
    The parts of the nuclei of other cells that reach into a cell label count as cytoplasm of that cell, they show
    up as a 'nucleus_fraction_in_cell' below 1 for the other cell.

    The result has one row per cell, also for the cells that should not be used: check 'n_nuclei' (0: the ratio is
    not defined, > 1: the nucleus volume is the sum over the nuclei), 'cell_on_border' and 'nucleus_on_border' (the
    object is cut by the border of the array, so its volume is incomplete) and 'nucleus_fraction_in_cell'.

    Args:
        cells: The instance segmentation of the cells, 0 is background. A 2D or 3D array (numpy, zarr, h5py, ...).
        nuclei: The instance segmentation of the nuclei, on the same grid (same shape and pixel / voxel size).
        voxel_size: The size of a pixel / voxel, one number for isotropic data or one number per axis. If given the
            volumes are in physical units (for 2D data these are areas), otherwise in pixels / voxels.
        unit: The unit of the voxel size, e.g. 'um'. Only used for the unit in the attributes of the result.
        min_overlap: The fraction of a nucleus that has to be inside of the cell it is matched to.
        slab_size: The number of slices along the first axis that are read at once. By default the chunk size of
            the first axis, or 32 for arrays without chunks.
        num_workers: The number of threads.

    Returns:
        The table with one row per cell and the columns
        - cell_id: The id of the cell.
        - nucleus_id: The id of its nucleus, the largest one if it has several nuclei, 0 if it has none.
        - n_nuclei: The number of nuclei that are matched to the cell.
        - cell_volume: The volume of the cell, including its nuclei.
        - nucleus_volume: The volume of its nuclei.
        - cytoplasm_volume: The cell volume minus the nucleus volume.
        - nc_ratio: The nucleus volume divided by the cytoplasm volume. NaN for a cell without nucleus or without
          cytoplasm.
        - nucleus_fraction_in_cell: The fraction of the nucleus volume that is inside of the cell label.
        - cell_on_border: Whether the cell touches the border of the array.
        - nucleus_on_border: Whether one of its nuclei touches the border of the array.
        The attribute 'volume_unit' of the table is the unit of the volumes.
    """
    ndim = len(cells.shape)
    if voxel_size is None:
        scale, volume_unit = 1.0, "pixel" if ndim == 2 else "voxel"
    else:
        voxel_size = np.broadcast_to(np.asarray(voxel_size, dtype="float64"), (ndim,))
        if not (np.isfinite(voxel_size).all() and (voxel_size > 0).all()):
            raise ValueError(f"Invalid voxel size {tuple(voxel_size)}.")
        scale, volume_unit = float(np.prod(voxel_size)), f"{'unit' if unit is None else unit}^{ndim}"
    if not 0 < min_overlap <= 1:
        raise ValueError(f"The minimal overlap must be in (0, 1], got {min_overlap}.")

    pairs, cell_border_ids, nucleus_border_ids = count_overlaps(cells, nuclei, slab_size, num_workers)
    cell_sizes = pairs[pairs["cell_id"] != 0].groupby("cell_id")["size"].sum()
    nucleus_sizes = pairs[pairs["nucleus_id"] != 0].groupby("nucleus_id")["size"].sum()

    # Match each nucleus to the cell that holds most of it.
    overlaps = pairs[(pairs["cell_id"] != 0) & (pairs["nucleus_id"] != 0)]
    overlaps = overlaps.sort_values(["nucleus_id", "size", "cell_id"], ascending=[True, False, True])
    matches = overlaps.drop_duplicates("nucleus_id").rename(columns={"size": "overlap"})
    matches = matches.assign(nucleus_size=nucleus_sizes.reindex(matches["nucleus_id"]).to_numpy())
    matches = matches[matches["overlap"] >= min_overlap * matches["nucleus_size"]]
    matches = matches.assign(nucleus_on_border=np.isin(matches["nucleus_id"], nucleus_border_ids))

    # The largest nucleus comes first for each cell.
    matches = matches.sort_values(["cell_id", "nucleus_size", "nucleus_id"], ascending=[True, False, True])
    per_cell = matches.groupby("cell_id").agg(
        nucleus_id=("nucleus_id", "first"), n_nuclei=("nucleus_id", "size"), nucleus_size=("nucleus_size", "sum"),
        overlap=("overlap", "sum"), nucleus_on_border=("nucleus_on_border", "any"),
    ).reindex(cell_sizes.index)

    has_nucleus = per_cell["n_nuclei"].notna().to_numpy()
    label_size = cell_sizes.to_numpy().astype("float64")
    nucleus_size = per_cell["nucleus_size"].fillna(0).to_numpy().astype("float64")
    overlap = per_cell["overlap"].fillna(0).to_numpy().astype("float64")
    cytoplasm_size = label_size - overlap
    cell_size = cytoplasm_size + nucleus_size
    valid = has_nucleus & (cytoplasm_size > 0)

    table = pd.DataFrame({
        "cell_id": cell_sizes.index.to_numpy().astype("int64"),
        "nucleus_id": per_cell["nucleus_id"].fillna(0).to_numpy().astype("int64"),
        "n_nuclei": per_cell["n_nuclei"].fillna(0).to_numpy().astype("int64"),
        "cell_volume": cell_size * scale,
        "nucleus_volume": nucleus_size * scale,
        "cytoplasm_volume": cytoplasm_size * scale,
        "nc_ratio": np.divide(nucleus_size, cytoplasm_size, out=np.full(len(cell_size), np.nan), where=valid),
        "nucleus_fraction_in_cell": np.divide(
            overlap, nucleus_size, out=np.full(len(cell_size), np.nan), where=has_nucleus
        ),
        "cell_on_border": np.isin(cell_sizes.index.to_numpy(), cell_border_ids),
        "nucleus_on_border": per_cell["nucleus_on_border"].fillna(False).to_numpy().astype(bool),
    })
    table.attrs["volume_unit"] = volume_unit
    return table
