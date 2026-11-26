"""Update the nuclei and cells of the EM spheroid of the Beghin lab with the results for the partial nuclei.

The third step after detect_partial_nuclei.py (classes of the joined nuclei: complete, partial, fragment) and
reprompt_partial_nuclei.py (candidates for the partial nuclei from prompts). It writes new nuclei (s1) and cells (s0)
next to the joined nuclei and their cells, which stay as they are; get_nuclei_path('updated') and
get_cell_volume_path(0, border_seed=True, nuclei='updated') of explore_beghin_em_data.py.

Nuclei, from the joined nuclei:
- Each partial nucleus that was segmented again is replaced by its suggested candidate (the voxels of the old object
  outside of the candidate are removed).
- The objects that are pieces of the same nucleus (SAME_NUCLEUS) become one nucleus with the lowest id of the pieces,
  the candidate of that piece (or of the piece in CANDIDATE_SOURCE). The other pieces are removed.
- The fragments that lie in a candidate ('absorbs' of the candidate) are part of that nucleus.
- The fragments of SMALL_FRAGMENTS are removed.
All other nuclei keep their voxels and ids.

Cells, from the cells of the joined nuclei with the background seed, only relabeled (a cell has the id of its nucleus):
- The cell of a removed piece or of a fragment in a candidate becomes part of the cell of its nucleus.
- The cell of a fragment of SMALL_FRAGMENTS becomes part of the neighboring cell with the largest contact area. If it
  only has contact to the background it is removed.

The objects of EXCLUDED are kept as they are (the nuclei 17 and 238 are their candidates), and listed with a reason:
they are not full nuclei or cells and should be excluded from downstream analysis. The list is in the attributes of
the nuclei ('excluded') and in PARTIAL_NUCLEI_DIR/excluded.csv; PARTIAL_NUCLEI_DIR/update.csv has one row per object
that is changed, removed or excluded.

Example:
    python update_partial_nuclei.py
    python explore_beghin_em_data.py export_segmentations --nuclei updated
"""
import os
import time
import argparse
from itertools import product
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec

import explore_beghin_em_data as explore
from detect_partial_nuclei import PARTIAL_NUCLEI_DIR, get_detection_path
from reprompt_partial_nuclei import get_reprompted_path, get_table_paths

SOURCE_NUCLEI = "joined"
NUCLEI = "updated"
# The objects that are pieces of the same nucleus ('same_nucleus' of reprompted.csv): the id of the nucleus (the
# lowest id of the pieces) -> the other pieces.
SAME_NUCLEUS = {53: (81,), 99: (132,), 115: (146,), 211: (213,), 214: (239,)}
# The nuclei that take the candidate of another piece. The candidate of 214 holds the small body 238 next to the
# nucleus, the one of 239 does not; 238 stays an object of its own (EXCLUDED).
CANDIDATE_SOURCE = {214: 239}
# The fragments with a cell of 15 - 66 um^3: the nucleus is removed, the cell goes to a neighbor (see above).
SMALL_FRAGMENTS = (47, 80, 109, 112, 127, 139, 155, 173, 183, 203, 218)
# The objects that are kept but are not full nuclei or cells: id -> (what to exclude, reason).
_FRAGMENT = "Nucleus fragment, not a full nucleus. Its cell is small and mainly borders one other cell."
EXCLUDED = {
    83: ("nucleus and cell", _FRAGMENT), 122: ("nucleus and cell", _FRAGMENT), 141: ("nucleus and cell", _FRAGMENT),
    145: ("nucleus and cell", _FRAGMENT), 154: ("nucleus and cell", _FRAGMENT), 236: ("nucleus and cell", _FRAGMENT),
    244: ("nucleus and cell", _FRAGMENT), 247: ("nucleus and cell", _FRAGMENT),
    260: ("nucleus and cell", "Artifact in the resin at the border of the volume, its cell only borders background."),
    108: ("nucleus and cell", "Nucleus fragment at the edge of a nucleus that is mostly not segmented. The cell has "
                              "the size of a full cell."),
    17: ("nucleus and cell", "Small compact body that stays the same under all prompts, not a full nucleus. Its cell "
                             "is a small piece inside the cell 15."),
    238: ("nucleus and cell", "Small round body next to the nucleus 214. Its cell is a small piece next to the cell "
                              "214."),
    262: ("nucleus", "Only a seed for its cell, not a full nucleus (seed_only). The cell is a full cell."),
    263: ("nucleus", "Only a seed for its cell, not a full nucleus (seed_only). The cell is a full cell."),
}
MAX_ID = 2 ** 16 - 1
CONTACT_SLAB = 8


def get_table_path(name):
    return os.path.join(PARTIAL_NUCLEI_DIR, f"{name}.csv")


def load_labels(array, num_workers):
    """Load a label array as uint16, in slabs."""
    labels = np.zeros(array.shape, dtype="uint16")
    step = array.chunks[0]

    def load(z):
        data = array[z:z + step]
        assert data.max() <= MAX_ID
        labels[z:z + step] = data

    with ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(load, range(0, array.shape[0], step)))
    return labels


def write_labels(array, labels, num_workers):
    """Write a label volume to an array, in slabs of whole chunks."""
    step = array.chunks[0]

    def write(z):
        array[z:z + step] = labels[z:z + step]

    with ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(write, range(0, array.shape[0], step)))


def get_volumes(labels, voxel_size, num_workers):
    """The volume of each label in um^3, indexed by the label."""
    step = 16
    with ThreadPoolExecutor(num_workers) as pool:
        counts = list(pool.map(
            lambda z: np.bincount(labels[z:z + step].ravel(), minlength=MAX_ID + 1), range(0, labels.shape[0], step),
        ))
    return np.sum(counts, axis=0) * np.prod(voxel_size) * 1e-9


def get_nucleus_plan(detected, reprompted):
    """The changes of the nuclei.

    Returns:
        The reprompted nuclei that are kept: id -> id of the nucleus with the candidate.
        The nuclei that become part of another nucleus (pieces and fragments in a candidate): id -> id of the nucleus.
    """
    pieces = {piece: nucleus_id for nucleus_id, others in SAME_NUCLEUS.items() for piece in others}
    assert set(pieces) | set(SAME_NUCLEUS) <= set(reprompted.index)
    for nucleus_id, others in SAME_NUCLEUS.items():
        group = set(str(reprompted.loc[nucleus_id, "same_nucleus"]).split())
        assert {str(piece) for piece in others} <= group, (nucleus_id, others, group)
    candidates = {
        int(nucleus_id): CANDIDATE_SOURCE.get(nucleus_id, nucleus_id) for nucleus_id in reprompted.index
        if nucleus_id not in pieces
    }
    merged = dict(pieces)
    for nucleus_id, source in candidates.items():
        absorbs = reprompted.loc[source, "absorbs"]
        for other in ([] if pd.isna(absorbs) else [int(i) for i in str(absorbs).split()]):
            if other in reprompted.index:
                assert other in pieces or other == nucleus_id, (nucleus_id, other)
            else:
                assert detected.loc[other, "class"] == "fragment", (nucleus_id, other)
                assert merged.get(other, nucleus_id) == nucleus_id, (nucleus_id, other)
                merged[other] = nucleus_id
    assert not set(merged) & set(SMALL_FRAGMENTS) and not set(merged) & set(EXCLUDED)
    return candidates, merged


def update_nuclei(labels, candidates, merged):
    """Replace the partial nuclei by their candidates and remove the pieces, the fragments in candidates and the small
    fragments, in place. The candidates do not overlap each other or the nuclei that stay."""
    out = zarr.open_group(get_reprompted_path(), mode="r")
    lut = np.arange(MAX_ID + 1, dtype="uint16")
    lut[list(candidates) + list(merged) + list(SMALL_FRAGMENTS)] = 0
    labels[:] = lut[labels]
    for nucleus_id, source in candidates.items():
        group = out["candidates"][str(source)]
        bb = tuple(slice(o, o + sh) for o, sh in zip(group.attrs["offset"], group.attrs["shape"]))
        candidate = group[group.attrs["suggested"]][:]
        current = labels[bb]
        assert not current[candidate].any(), f"The candidate of {nucleus_id} overlaps {np.unique(current[candidate])}."
        current[candidate] = nucleus_id
        labels[bb] = current


def get_cell_contacts(cells, cell_ids, voxel_size, num_workers):
    """The contact areas (um^2) of some cells to all of their neighbors, counted over the faces between voxels.

    Returns:
        Per cell: neighbor id (0 is the background) -> contact area.
    """
    is_cell = np.zeros(MAX_ID + 1, dtype=bool)
    is_cell[list(cell_ids)] = True
    areas = [voxel_size[1] * voxel_size[2] * 1e-6, voxel_size[0] * voxel_size[2] * 1e-6,
             voxel_size[0] * voxel_size[1] * 1e-6]

    def get_slab_contacts(z):
        # One more slice for the faces to the next slab, the faces in its plane belong to that slab.
        data = cells[z:z + CONTACT_SLAB + 1]
        assert data.max() <= MAX_ID
        data = data.astype("uint16")
        contacts = {}
        if not is_cell[data].any():
            return contacts
        for axis, area in enumerate(areas):
            block = data if axis == 0 else data[:CONTACT_SLAB]
            first = block[tuple(slice(None, -1) if ax == axis else slice(None) for ax in range(3))]
            second = block[tuple(slice(1, None) if ax == axis else slice(None) for ax in range(3))]
            faces = (first != second) & (is_cell[first] | is_cell[second])
            pairs, counts = np.unique(np.stack([first[faces], second[faces]], axis=1), axis=0, return_counts=True)
            for (a, b), count in zip(pairs.tolist(), counts.tolist()):
                for cell_id, neighbor in ((a, b), (b, a)):
                    if is_cell[cell_id]:
                        contacts[(cell_id, neighbor)] = contacts.get((cell_id, neighbor), 0.0) + count * area
        return contacts

    result = {cell_id: {} for cell_id in cell_ids}
    with ThreadPoolExecutor(num_workers) as pool:
        for contacts in pool.map(get_slab_contacts, range(0, cells.shape[0], CONTACT_SLAB)):
            for (cell_id, neighbor), area in contacts.items():
                result[cell_id][neighbor] = result[cell_id].get(neighbor, 0.0) + area
    return result


def get_cell_plan(merged, contacts):
    """The new id of each cell that changes: the cell of the nucleus it belongs to, or for the cells of the small
    fragments the neighbor with the largest contact area (0, so removed, if there is only background)."""
    mapping = dict(merged)
    for cell_id in SMALL_FRAGMENTS:
        neighbors = {neighbor: area for neighbor, area in contacts[cell_id].items() if neighbor != 0}
        mapping[cell_id] = max(neighbors, key=neighbors.get) if neighbors else 0
    # A neighbor can itself become part of another cell.
    mapping = {cell_id: mapping.get(target, target) for cell_id, target in mapping.items()}
    assert not set(mapping.values()) & set(mapping), mapping
    return mapping


def update_cells(cells, mapping, path, params, num_workers):
    """Write the relabeled cells block-wise as the array 'segmentation' of a new group."""
    lut = np.arange(MAX_ID + 1, dtype=cells.dtype)
    for cell_id, target in mapping.items():
        lut[cell_id] = target
    out = zarr.open_group(path, mode="w", zarr_format=3)
    array = out.create_array(
        "segmentation", shape=cells.shape, dtype=cells.dtype, chunks=cells.chunks, fill_value=0,
        compressors=BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"), dimension_names=["z", "y", "x"],
    )
    # Blocks of whole chunks, so the threads write different chunks.
    block_shape = explore.CELL_VOLUME_BLOCK_SHAPE
    assert all(bs % ch == 0 for bs, ch in zip(block_shape, cells.chunks)), (block_shape, cells.chunks)
    blocks = [
        tuple(slice(start, min(start + bs, sh)) for start, bs, sh in zip(starts, block_shape, cells.shape))
        for starts in product(*[range(0, sh, bs) for sh, bs in zip(cells.shape, block_shape)])
    ]

    def relabel(block):
        array[block] = lut[cells[block]]

    with ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(relabel, blocks))
    out.attrs.update({"params": params, "done_stages": ["segmentation"]})
    return array


def get_update_table(detected, reprompted, candidates, merged, mapping, contacts, volumes):
    """One row per object that is changed, removed or excluded."""
    rows = []
    ids = sorted(set(candidates) | set(merged) | set(SMALL_FRAGMENTS) | set(EXCLUDED))
    for object_id in ids:
        if object_id in candidates:
            source = candidates[object_id]
            pieces = [other for other, nucleus_id in merged.items() if nucleus_id == object_id]
            nucleus = f"candidate {reprompted.loc[source, 'suggested']}"
            nucleus += f" of {source}" if source != object_id else ""
            nucleus += f", with {' '.join(str(i) for i in sorted(pieces))}" if pieces else ""
        elif object_id in merged:
            nucleus = f"part of {merged[object_id]}"
        elif object_id in SMALL_FRAGMENTS:
            nucleus = "removed"
        else:
            nucleus = "kept"
        if object_id in mapping:
            cell = f"part of {mapping[object_id]}" if mapping[object_id] else "removed"
        else:
            parts = [other for other, target in mapping.items() if target == object_id]
            cell = "kept" + (f", with {' '.join(str(i) for i in sorted(parts))}" if parts else "")
        row = {
            "id": object_id, "class": detected.loc[object_id, "class"], "nucleus": nucleus, "cell": cell,
            **{name: round(float(values[object_id]), 2) for name, values in volumes.items()},
            "exclude": EXCLUDED.get(object_id, ("", ""))[0], "reason": EXCLUDED.get(object_id, ("", ""))[1],
        }
        if object_id in SMALL_FRAGMENTS:
            total = sum(contacts[object_id].values())
            neighbors = sorted(contacts[object_id].items(), key=lambda item: -item[1])
            row["cell_contacts"] = " ".join(f"{neighbor}:{area / total:.2f}" for neighbor, area in neighbors[:4])
        rows.append(row)
    return pd.DataFrame(rows)


def update_partial_nuclei(num_workers=1):
    """Write the updated nuclei and cells and the tables of the changes, see the description of the script."""
    zarr.config.set({"threading.max_workers": num_workers})
    detected = pd.read_csv(get_detection_path(SOURCE_NUCLEI), index_col="id")
    reprompted = pd.read_csv(get_table_paths()[0], index_col="id")
    source_path, nuclei_path = explore.get_nuclei_path(SOURCE_NUCLEI), explore.get_nuclei_path(NUCLEI)
    cells_source_path = explore.get_cell_volume_path(0, border_seed=True, nuclei=SOURCE_NUCLEI)
    cells_path = explore.get_cell_volume_path(0, border_seed=True, nuclei=NUCLEI)
    source_group = zarr.open_group(source_path, mode="r")
    assert zarr.open_group(get_reprompted_path(), mode="r").attrs["nuclei"] == source_path
    voxel_size_s1, voxel_size_s0 = explore.get_level_voxel_size(1), explore.get_level_voxel_size(0)

    # The nuclei.
    t0 = time.time()
    candidates, merged = get_nucleus_plan(detected, reprompted)
    source = source_group["segmentation"]
    old = load_labels(source, num_workers)
    new = old.copy()
    update_nuclei(new, candidates, merged)
    volumes = {
        "nucleus_um3_old": get_volumes(old, voxel_size_s1, num_workers),
        "nucleus_um3": get_volumes(new, voxel_size_s1, num_workers),
    }
    out = zarr.open_group(nuclei_path, mode="w", zarr_format=3)
    array = out.create_array(
        "segmentation", shape=source.shape, dtype=source.dtype, chunks=source.chunks, fill_value=0,
        compressors=BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"), dimension_names=["z", "y", "x"],
    )
    write_labels(array, new, num_workers)
    nucleus_ids = np.flatnonzero(volumes["nucleus_um3"])[1:]
    print(
        f"Nuclei: {np.count_nonzero(volumes['nucleus_um3_old']) - 1} -> {len(nucleus_ids)}, {len(candidates)} from "
        f"candidates, {len(merged)} part of another nucleus, {len(SMALL_FRAGMENTS)} small fragments removed, in "
        f"{time.time() - t0:.0f} s", flush=True,
    )
    del old, new

    # The cells.
    t0 = time.time()
    cells = zarr.open_group(cells_source_path, mode="r")["segmentation"]
    contacts = get_cell_contacts(cells, SMALL_FRAGMENTS, voxel_size_s0, num_workers)
    mapping = get_cell_plan(merged, contacts)
    print(f"Cells: contacts in {time.time() - t0:.0f} s, changes {mapping}", flush=True)
    t0 = time.time()
    params = {
        "nuclei": nuclei_path, "cells": cells_source_path, "level": 0, "voxel_size": list(voxel_size_s0),
        "border_seed": True, "relabeled": {str(cell_id): int(target) for cell_id, target in mapping.items()},
    }
    new_cells = update_cells(cells, mapping, cells_path, params, num_workers)
    for name, array in (("cell_um3_old", cells), ("cell_um3", new_cells)):
        step = CONTACT_SLAB
        with ThreadPoolExecutor(num_workers) as pool:
            counts = list(pool.map(
                lambda z: np.bincount(array[z:z + step].ravel().astype("int64"), minlength=MAX_ID + 1),
                range(0, array.shape[0], step),
            ))
        volumes[name] = np.sum(counts, axis=0) * np.prod(voxel_size_s0) * 1e-9
    cell_ids = np.flatnonzero(volumes["cell_um3"])[1:]
    assert np.array_equal(cell_ids, nucleus_ids), set(cell_ids) ^ set(nucleus_ids)
    print(
        f"Cells: {np.count_nonzero(volumes['cell_um3_old']) - 1} -> {len(cell_ids)}, each with the id of its nucleus, "
        f"in {time.time() - t0:.0f} s", flush=True,
    )

    # The tables and the notes.
    table = get_update_table(detected, reprompted, candidates, merged, mapping, contacts, volumes)
    table.to_csv(get_table_path("update"), index=False)
    excluded = table.loc[table["exclude"] != "", ["id", "exclude", "reason", "nucleus_um3", "cell_um3"]]
    assert set(excluded["id"]) == set(EXCLUDED)
    excluded.to_csv(get_table_path("excluded"), index=False)
    source_attrs = dict(source_group.attrs)
    out.attrs.update({
        "nuclei": source_path, "level": 1, "voxel_size": list(voxel_size_s1), "reprompted": get_reprompted_path(),
        "objects": source_attrs["objects"], "seed_only": source_attrs["seed_only"],
        "joined": source_attrs["joined"],
        "candidates": {str(nucleus_id): int(source) for nucleus_id, source in candidates.items()},
        "merged": {str(other): int(nucleus_id) for other, nucleus_id in merged.items()},
        "removed": [int(i) for i in SMALL_FRAGMENTS],
        "excluded": excluded.to_dict("records"),
    })
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_rows", 200, "display.max_colwidth", 60)
    print(table.drop(columns=["reason"]).to_string(index=False))
    print(f"Nuclei: {nuclei_path}\nCells: {cells_path}")
    print(f"Tables: {get_table_path('update')}, {get_table_path('excluded')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    args = parser.parse_args()
    update_partial_nuclei(args.num_workers)


if __name__ == "__main__":
    main()
