"""Detect the nuclei of the EM spheroid of the Beghin lab that cover only a part of their nucleus, and the fragments.

The nuclei (join_nucleus_fragments of explore_beghin_em_data.py, s1) come from a block-wise watershed on predictions
that were computed in tiles of four slices. Where the foreground of a nucleus is missing in some tiles the object is
cut along z: it ends in a flat face, or it is striped, with slices that hold only a part of the cross-section. Both
show up as an abrupt change of the mask between two neighboring slices, which a nucleus that is segmented in full
does not have: its cross-section grows and shrinks by a few percent per slice.

Each object gets three measurements:
- volume: its size, compared to the volume of the consistent nuclei (the reference, see below).
- max_change: the largest change between two neighboring slices, the number of pixels that are in only one of the two
  slices, relative to the largest cross-section of the object. The slices before the first and after the last one are
  empty, so an object that starts or ends with a large cross-section has a large change as well. The faces on the
  border of the volume are not counted, there the data ends.
- z_extent: the number of slices it spans.

They give three classes (column 'class' of the table):
- fragment: smaller than FRAGMENT_MAX_VOLUME or thinner than FRAGMENT_MIN_Z_EXTENT. On this data these are thin sheets
  in the cytoplasm or in the resin, they are meant to be removed.
- partial: not a fragment, and inconsistent across the slices (max_change > MAX_SLICE_CHANGE) or small (more than
  SIZE_N_DEVIATIONS robust standard deviations below the median of the log volume of the consistent nuclei). These
  are the nuclei to segment again, see reprompt_partial_nuclei.py.
- complete: the rest.

The reference volume is the median volume of the consistent objects (max_change <= MAX_SLICE_CHANGE) that are not
flat; the spread is the robust standard deviation (1.4826 * MAD) of their log volume.

The thresholds were set by eye on galleries of the 252 joined nuclei: 102 nuclei have max_change <= 0.15 and look
complete, from 0.25 on the missing part is visible in the orthogonal views (a flat face of a few slices up to
stripes through the whole nucleus), in between the nuclei are complete or have small dents. The class is a proposal
for the inspection, not a ground truth: the table is sorted by max_change and has all measurements, so the
thresholds can be moved (--max_slice_change, --size_n_deviations, --fragment_max_volume, --fragment_min_z_extent).

The outputs are written to PARTIAL_NUCLEI_DIR: the table (detection.csv), a plot of volume against max_change
(detection_overview.png) and galleries of the partial nuclei and of the fragments (orthogonal views through each
object and its area per slice).

Example:
    python detect_partial_nuclei.py
    python detect_partial_nuclei.py --nuclei joined --max_slice_change 0.3 --no_galleries
"""
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import zarr
from scipy import ndimage

DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids"
ZARR_PATH = os.path.join(DATA_DIR, "PF_Dish_C_pin3new_Spheroid3.ome.zarr")
PREDICTION_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids_volume_predictions"
PARTIAL_NUCLEI_DIR = os.path.join(PREDICTION_DIR, "partial_nuclei")
# The nuclei are segmented at s1.
LEVEL = 1
VOXEL_SIZE = (100.0, 66.4, 66.4)

# An object is inconsistent if more than this fraction of its largest cross-section changes between two slices.
MAX_SLICE_CHANGE = 0.25
# An object is small if its log volume is more than this many robust standard deviations below the median of the
# consistent nuclei (2 deviations are 65% of the reference volume on the joined nuclei).
SIZE_N_DEVIATIONS = 2.0
# The fragments: objects below this volume (cubic micrometer, about 5% of a nucleus) or this extent in z (nanometer).
FRAGMENT_MAX_VOLUME = 15.0
FRAGMENT_MIN_Z_EXTENT = 1500.0

CLASSES = ("complete", "partial", "fragment")
GALLERY_HALF_SHAPE = (70, 150, 150)
GALLERY_PAGE_SIZE = 14


def get_nuclei_path(nuclei="joined"):
    """The path of a version of the nuclei at s1 ('watershed', 'merged' or 'joined')."""
    return os.path.join(PREDICTION_DIR, f"nuclei_s{LEVEL}_{nuclei}.zarr")


def get_detection_path(nuclei="joined"):
    name = "detection.csv" if nuclei == "joined" else f"detection_{nuclei}.csv"
    return os.path.join(PARTIAL_NUCLEI_DIR, name)


def load_labels(array, num_workers=1):
    """Load a label volume into memory, slab by slab, with the smallest unsigned integer type that holds its ids."""
    step = array.chunks[0]
    slabs = [slice(z, min(z + step, array.shape[0])) for z in range(0, array.shape[0], step)]

    def load(slab):
        data = array[slab]
        return data.astype(np.min_scalar_type(int(data.max())))

    with ThreadPoolExecutor(num_workers) as pool:
        data = list(pool.map(load, slabs))
    return np.concatenate(data).astype(np.result_type(*data), copy=False)


def get_slice_changes(mask, open_start=False, open_stop=False):
    """The change of a mask from slice to slice, relative to its largest cross-section.

    Args:
        mask: The mask of one object, (Z, Y, X), with at least one voxel.
        open_start: Whether the object continues before the first slice (the border of the volume), then the change
            into the first slice is not counted.
        open_stop: The same for the last slice.

    Returns:
        The number of voxels per slice, (Z,).
        The changes, (Z + 1,): the entry i is the number of voxels that are in only one of the slices i - 1 and i,
        divided by the largest number of voxels of a slice. The slices -1 and Z are empty.
    """
    areas = mask.sum(axis=(1, 2))
    changes = np.concatenate([areas[:1], (mask[1:] ^ mask[:-1]).sum(axis=(1, 2)), areas[-1:]]).astype("float64")
    if open_start:
        changes[0] = 0
    if open_stop:
        changes[-1] = 0
    return areas, changes / areas.max()


def get_object_features(mask, offset, shape, voxel_size=VOXEL_SIZE, max_slice_change=MAX_SLICE_CHANGE):
    """The measurements of one object for the detection of partial nuclei.

    Args:
        mask: The mask of the object in a crop of the volume, (Z, Y, X), with at least one voxel.
        offset: The position of the crop in the volume.
        shape: The shape of the volume.
        voxel_size: The voxel size in nanometer.
        max_slice_change: The change between two slices that counts as a jump ('n_jumps').

    Returns:
        A dictionary with the volume (voxels and cubic micrometer), the bounding box in the volume ('z0' ... 'x1',
        stop exclusive), the extent in z (slices and nanometer), the largest cross-section ('max_area' and its slice
        'max_area_z'), the largest change between two slices ('max_change', see get_slice_changes; it is between the
        slices 'max_change_z' - 1 and 'max_change_z'), the number of changes above max_slice_change ('n_jumps'), the
        total variation of the area per slice relative to that of a profile with a single maximum ('area_variation',
        1 if the area only grows and then only shrinks) and the faces of the volume the object touches ('at_border').
    """
    coords = np.nonzero(mask)
    bb = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
    start = [b.start + o for b, o in zip(bb, offset)]
    stop = [b.stop + o for b, o in zip(bb, offset)]
    mask = mask[bb]
    faces = [f"{axis}0" for axis, sta in zip("zyx", start) if sta == 0]
    faces += [f"{axis}1" for axis, sto, sh in zip("zyx", stop, shape) if sto == sh]
    areas, changes = get_slice_changes(mask, open_start="z0" in faces, open_stop="z1" in faces)
    volume = int(areas.sum())
    profile = np.concatenate([[0], areas, [0]]).astype("float64")
    features = {
        "volume": volume, "volume_um3": volume * float(np.prod(voxel_size)) / 1e9,
        "n_slices": len(areas), "z_extent_nm": len(areas) * float(voxel_size[0]),
        "max_area": int(areas.max()), "max_area_z": int(np.argmax(areas)) + start[0],
        "max_change": float(changes.max()), "max_change_z": int(np.argmax(changes)) + start[0],
        "n_jumps": int((changes > max_slice_change).sum()),
        "area_variation": float(np.abs(np.diff(profile)).sum() / (2 * areas.max())),
        "at_border": " ".join(faces),
    }
    features.update({f"{axis}0": sta for axis, sta in zip("zyx", start)})
    features.update({f"{axis}1": sto for axis, sto in zip("zyx", stop)})
    return features


def compute_features(labels, voxel_size=VOXEL_SIZE, max_slice_change=MAX_SLICE_CHANGE):
    """The table of the measurements of get_object_features for all objects of a label volume (index 'id')."""
    rows = []
    for label_id, bb in enumerate(ndimage.find_objects(labels), 1):
        if bb is None:
            continue
        offset = [b.start for b in bb]
        features = get_object_features(labels[bb] == label_id, offset, labels.shape, voxel_size, max_slice_change)
        rows.append({"id": label_id, **features})
    return pd.DataFrame(rows).set_index("id")


def get_reference(table, max_slice_change=MAX_SLICE_CHANGE, fragment_min_z_extent=FRAGMENT_MIN_Z_EXTENT):
    """The reference volume of a nucleus and the spread of the volumes, from the consistent objects of a table.

    Returns:
        The median volume (cubic micrometer) of the objects with max_change <= max_slice_change that are not flat.
        The robust standard deviation (1.4826 * MAD) of their log volume.
    """
    consistent = (table["max_change"] <= max_slice_change) & (table["z_extent_nm"] >= fragment_min_z_extent)
    log_volume = np.log(table.loc[consistent, "volume_um3"].to_numpy())
    median = np.median(log_volume)
    return float(np.exp(median)), float(1.4826 * np.median(np.abs(log_volume - median)))


def classify(
    table, reference, max_slice_change=MAX_SLICE_CHANGE, size_n_deviations=SIZE_N_DEVIATIONS,
    fragment_max_volume=FRAGMENT_MAX_VOLUME, fragment_min_z_extent=FRAGMENT_MIN_Z_EXTENT,
):
    """Assign the class of each object of a table of get_object_features, see the description of the script.

    Args:
        table: The table of the objects.
        reference: The reference volume and the spread of the log volumes (get_reference).
        max_slice_change: The largest change between two slices of a consistent object.
        size_n_deviations: The number of deviations below the reference volume from which an object is small.
        fragment_max_volume: The volume below which an object is a fragment, in cubic micrometer.
        fragment_min_z_extent: The extent in z below which an object is a fragment, in nanometer.

    Returns:
        The table with the columns 'volume_ratio' (volume / reference volume), 'small', 'inconsistent', 'class' and
        'reason'.
    """
    reference_volume, deviation = reference
    table = table.copy()
    table["volume_ratio"] = table["volume_um3"] / reference_volume
    table["small"] = table["volume_um3"] < reference_volume * np.exp(-size_n_deviations * deviation)
    table["inconsistent"] = table["max_change"] > max_slice_change
    tiny = table["volume_um3"] < fragment_max_volume
    flat = table["z_extent_nm"] < fragment_min_z_extent
    fragment = tiny | flat
    partial = ~fragment & (table["small"] | table["inconsistent"])
    table["class"] = np.select([fragment, partial], ["fragment", "partial"], default="complete")
    reasons = {"tiny": tiny, "flat": flat, "small": ~fragment & table["small"],
               "inconsistent": ~fragment & table["inconsistent"]}
    table["reason"] = [
        " ".join(name for name, flags in reasons.items() if flags.loc[label_id]) for label_id in table.index
    ]
    return table


def plot_overview(table, reference, thresholds, plot_path):
    """Plot the volume against the largest change between two slices for all objects, with the thresholds."""
    import matplotlib.pyplot as plt

    reference_volume, deviation = reference
    colors = {"complete": "tab:gray", "partial": "tab:red", "fragment": "tab:blue"}
    fig, ax = plt.subplots(figsize=(11, 7))
    for name in CLASSES:
        objects = table[table["class"] == name]
        ax.scatter(objects["volume_um3"], objects["max_change"], s=14, color=colors[name],
                   label=f"{name} ({len(objects)})")
    for label_id, row in table[table["class"] != "complete"].iterrows():
        ax.annotate(str(label_id), (row["volume_um3"], row["max_change"]), fontsize=5, xytext=(2, 2),
                    textcoords="offset points")
    ax.axhline(thresholds["max_slice_change"], color="k", lw=0.8, ls="--")
    ax.axvline(reference_volume * np.exp(-thresholds["size_n_deviations"] * deviation), color="k", lw=0.8, ls="--")
    ax.axvline(thresholds["fragment_max_volume"], color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("volume [cubic micrometer]")
    ax.set_ylabel("largest change between two slices / largest cross-section")
    ax.set_title(f"reference volume {reference_volume:.0f} cubic micrometer, robust sd of the log volume "
                 f"{deviation:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def get_gallery_crop(row, shape, half_shape=GALLERY_HALF_SHAPE):
    """The crop of a gallery panel: around the center of the bounding box of an object, clipped to the volume."""
    center = [(int(row[f"{axis}0"]) + int(row[f"{axis}1"])) // 2 for axis in "zyx"]
    return tuple(slice(max(c - h, 0), min(c + h, sh)) for c, h, sh in zip(center, half_shape, shape))


def get_contour_overlay(labels, contours):
    """An RGBA image with the contours of masks of a 2d label image.

    Args:
        labels: The 2d label image, only used for its shape.
        contours: Pairs of a 2d mask and the RGB color of its contour, drawn in this order.
    """
    from skimage.segmentation import find_boundaries

    overlay = np.zeros(labels.shape + (4,))
    for mask, color in contours:
        overlay[find_boundaries(mask, mode="inner")] = tuple(color) + (1.0,)
    return overlay


def plot_views(axes, raw, views, z_aspect):
    """Plot three orthogonal views (XY, XZ, YZ) of a crop of the inverted raw with contours.

    Args:
        axes: The three axes.
        raw: The crop of the raw, (Z, Y, X).
        views: Per view the index of its plane and the list of (2d label image or mask, color) to draw; a label
            image gets a contour around each of its objects.
        z_aspect: The size of a voxel in z relative to y and x.
    """
    from skimage.segmentation import find_boundaries

    selections = [lambda vol, i: vol[i], lambda vol, i: vol[:, i], lambda vol, i: vol[:, :, i]]
    for ax, select, (index, layers), aspect in zip(axes, selections, views, (1, z_aspect, z_aspect)):
        image = select(raw, index)
        ax.imshow(255 - image, cmap="gray", aspect=aspect, vmin=0, vmax=255)
        overlay = np.zeros(image.shape + (4,))
        for volume, color in layers:
            overlay[find_boundaries(select(volume, index), mode="inner")] = tuple(color) + (1.0,)
        ax.imshow(overlay, aspect=aspect, interpolation="nearest")
        ax.axis("off")


def plot_gallery(labels, raw, table, label_ids, plot_path, voxel_size=VOXEL_SIZE):
    """Plot one row per object: orthogonal views through it and its number of voxels per slice.

    The views are the slice with the largest cross-section of the object (XY) and the planes through the center of
    this cross-section (XZ, YZ) of the inverted raw, with the object in red and the other nuclei in cyan. In the
    profile the largest change between two slices is marked.
    """
    import matplotlib.pyplot as plt

    per_row = 2
    n_rows = int(np.ceil(len(label_ids) / per_row))
    fig, axes = plt.subplots(n_rows, 4 * per_row, figsize=(4 * per_row * 2.2, n_rows * 2.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for index, label_id in enumerate(label_ids):
        row = table.loc[label_id]
        panel = axes[index // per_row, 4 * (index % per_row): 4 * (index % per_row + 1)]
        bb = get_gallery_crop(row, labels.shape)
        crop, mask = labels[bb], labels[bb] == label_id
        other = np.where(mask, 0, crop)
        z = int(row["max_area_z"]) - bb[0].start
        y, x = np.argwhere(mask[z]).mean(axis=0).astype(int)
        layers = [(other, (0, 1, 1)), (mask, (1, 0.2, 0.2))]
        plot_views(panel, raw[bb], [(z, layers), (y, layers), (x, layers)], voxel_size[0] / voxel_size[1])
        panel[0].set_title(
            f"{label_id} ({row['class']}): {row['volume_um3']:.0f} um3, {int(row['n_slices'])} slices, "
            f"change {row['max_change']:.2f}", fontsize=7, loc="left",
        )
        panel[3].axis("on")
        panel[3].plot(np.arange(bb[0].start, bb[0].stop), mask.sum(axis=(1, 2)), lw=1, color="tab:red")
        panel[3].axvline(row["max_change_z"] - 0.5, color="k", lw=0.5, ls="--")
        panel[3].tick_params(labelsize=5)
    fig.tight_layout(pad=0.3)
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)


def plot_galleries(labels, raw, table, name, label_ids, page_size=GALLERY_PAGE_SIZE):
    """Plot the galleries of some objects in pages, as PARTIAL_NUCLEI_DIR/<name>_<page>.png."""
    paths = []
    for page, start in enumerate(range(0, len(label_ids), page_size)):
        paths.append(os.path.join(PARTIAL_NUCLEI_DIR, f"{name}_{page:02d}.png"))
        plot_gallery(labels, raw, table, label_ids[start:start + page_size], paths[-1])
    return paths


def detect_partial_nuclei(nuclei="joined", num_workers=1, galleries=True, **thresholds):
    """Measure and classify all objects of the nuclei and write the table, the overview plot and the galleries.

    Args:
        nuclei: The version of the nuclei ('watershed', 'merged' or 'joined').
        num_workers: The number of threads for reading the nuclei.
        galleries: Whether to plot the galleries of the partial nuclei and of the fragments.
        thresholds: max_slice_change, size_n_deviations, fragment_max_volume and fragment_min_z_extent, see classify.

    Returns:
        The table of the objects.
    """
    os.makedirs(PARTIAL_NUCLEI_DIR, exist_ok=True)
    group = zarr.open_group(get_nuclei_path(nuclei), mode="r")
    labels = load_labels(group["segmentation"], num_workers)
    voxel_size = tuple(group.attrs.get("voxel_size", VOXEL_SIZE))

    table = compute_features(labels, voxel_size, thresholds["max_slice_change"])
    reference = get_reference(table, thresholds["max_slice_change"], thresholds["fragment_min_z_extent"])
    table = classify(table, reference, **thresholds)
    # The prompted nuclei that are only seeds for their cells are not full nuclei by the decision of the annotator.
    table["seed_only"] = table.index.isin(group.attrs.get("seed_only", []))
    table = table.sort_values("max_change", ascending=False)
    table.to_csv(get_detection_path(nuclei))

    reference_volume, deviation = reference
    small_volume = reference_volume * np.exp(-thresholds["size_n_deviations"] * deviation)
    print(f"{len(table)} objects in {get_nuclei_path(nuclei)}")
    print(f"Reference volume {reference_volume:.1f} um3, robust sd of the log volume {deviation:.3f}: small below "
          f"{small_volume:.1f} um3 ({small_volume / reference_volume:.2f} of the reference)")
    for name in CLASSES:
        objects = table[table["class"] == name]
        print(f"{name}: {len(objects)}, volume {objects['volume_um3'].min():.1f} - "
              f"{objects['volume_um3'].max():.1f} um3, ids {sorted(objects.index.tolist())}")
    partial = table[table["class"] == "partial"]
    print(f"partial by reason: {partial['reason'].value_counts().to_dict()}")
    print(f"partial and only a seed for the cell: {sorted(partial.index[partial['seed_only']].tolist())}")
    print(f"partial or fragment on the border of the volume: "
          f"{table[(table['class'] != 'complete') & (table['at_border'] != '')]['at_border'].to_dict()}")

    plot_overview(table, reference, thresholds, os.path.join(PARTIAL_NUCLEI_DIR, "detection_overview.png"))
    if galleries:
        raw = zarr.open_array(os.path.join(ZARR_PATH, f"s{LEVEL}"), mode="r")
        for name in ("partial", "fragment"):
            objects = table[table["class"] == name]
            order = objects.sort_values("max_change" if name == "partial" else "volume", ascending=False)
            paths = plot_galleries(labels, raw, table, f"detection_{name}", order.index.tolist())
            print(f"Galleries of the class {name}: {paths}")
    print(f"Table: {get_detection_path(nuclei)}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nuclei", default="joined", choices=("watershed", "merged", "joined"))
    parser.add_argument("--max_slice_change", type=float, default=MAX_SLICE_CHANGE)
    parser.add_argument("--size_n_deviations", type=float, default=SIZE_N_DEVIATIONS)
    parser.add_argument("--fragment_max_volume", type=float, default=FRAGMENT_MAX_VOLUME)
    parser.add_argument("--fragment_min_z_extent", type=float, default=FRAGMENT_MIN_Z_EXTENT)
    parser.add_argument("--no_galleries", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    detect_partial_nuclei(
        nuclei=args.nuclei, num_workers=args.num_workers, galleries=not args.no_galleries,
        max_slice_change=args.max_slice_change, size_n_deviations=args.size_n_deviations,
        fragment_max_volume=args.fragment_max_volume, fragment_min_z_extent=args.fragment_min_z_extent,
    )


if __name__ == "__main__":
    main()
