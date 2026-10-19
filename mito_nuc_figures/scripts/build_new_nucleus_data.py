"""Write the derived nucleus data of the new torch-em datasets for the nucleus EM generalist.

- cardiomyocyte_tem: the 9 TEM micrographs with a nucleus mask. They are downscaled CARDIOMYOCYTE_SCALE times, because
  nuclei are 900-1600 px wide at full resolution. The holes are filled and specks below MIN_OBJECT_SIZE are dropped.
- perez: the nucleus task tiles (train 500 px, resized to 512 px; test 1000 px), holes filled.
- vem_nuclei: the 2D array tomography images and the two 3D FIB-SEM volumes. Each 2D image is downscaled by a power of
  two, so that nuclei are about VEM_TARGET_DIAMETER px wide. The 3D volumes are downscaled VEM_3D_SCALE times.
  Most 2D label images (all except VEM_EDGE_NUCLEI_INCLUDED) do not contain the nuclei that touch the image border.
  Thus, the unlabeled pixels in a border band of the largest nucleus diameter carry IGNORE_LABEL.
  Image id_9 is truncated in the archive and left out.
- u2os_fibsem: the 44 nuclear envelope slices of one U2-OS cell (evaluation), filled into nucleus masks.
  The crops cut the nucleus. Thus, the nucleus part in a crop is the convex hull of the envelope arc and the right
  border that the arc spans. The right border is the last column with data, before the constant padding of some slices.
- etchacell: the HeLa SBF-SEM ROIs (evaluation, 10 x 10 x 50 nm) with nuclear envelope consensus labels.
  The labels are filled into nucleus masks, per slice after closing small gaps, and in 3D. The 3D fill also fills the
  cap slices, where the cut through the envelope is tangential. The ROIs are downscaled ETCHACELL_SCALE times in xy,
  so that a nucleus fits a 512 px field of view.
- aive: the nucleus of the three AIVE FIB-SEM overview volumes with a nucleus label (6.5 x 6.5 x 10 nm).
  The 'chromatin' class marks the nucleoli inside the nucleus, so it is folded into the nucleus class. The holes are
  filled. A nucleus is about 3000 px long at full resolution. Thus, the volumes are downscaled AIVE_SCALE times
  (z, y, x) to 30 x 26 x 26 nm.
- cerebellum_fibsem: the mouse cerebellum FIB-SEM volume (10 nm isotropic) with its nucleus instances. It is
  downscaled CEREBELLUM_SCALE times, because nuclei are up to 600 px wide at full resolution. The labels below
  CEREBELLUM_MIN_NUCLEUS_SIZE voxels get the ignore label. They are three pieces of one dark object inside a myelinated
  axon at the end of the volume, not a nucleus by eye.
- macrophage_a431: the SBF-SEM volumes with the nucleus class (quick semantic MIB annotations). They are downscaled
  MACROPHAGE_SCALE times in xy. Each volume has one nucleus of about 1300 px. The voxel size is not stored.
  The nucleus extents suggest sections about 10 times thicker than the pixel size. Thus, z / xy is about 2.5 after
  downscaling.

Every file is written under a temporary name and moved in place when complete.
"""
import os
import glob
import json
import argparse

import h5py
import tifffile
import numpy as np
from scipy import ndimage
import imageio.v3 as imageio

from skimage.measure import label
from skimage.morphology import convex_hull_image, disk
from skimage.transform import downscale_local_mean, resize

from figure_utils import DATA_ROOT


IGNORE_LABEL = np.iinfo("uint32").max
MIN_OBJECT_SIZE = 200
CARDIOMYOCYTE_SCALE = 4
VEM_TARGET_DIAMETER = 250
VEM_3D_SCALE = 2
VEM_EDGE_NUCLEI_INCLUDED = (8, 10)
VEM_MISSING = (9,)
VEM_BORDER_PERCENTILE = 90
U2OS_MIN_ENVELOPE_PIXELS = 500
ETCHACELL_SCALE = 4
ETCHACELL_CLOSING_RADIUS = 5
AIVE_SCALE = (3, 4, 4)
AIVE_NUCLEUS_CLASSES = (6, 11)
AIVE_VOLUMES = (1, 2, 3)
CEREBELLUM_SCALE = (2, 2, 2)
CEREBELLUM_MIN_NUCLEUS_SIZE = 100000
MACROPHAGE_SCALE = (1, 4, 4)
MACROPHAGE_VOLUMES = ("sbf_a431", "sbf_oa_chol_7t", "sbf_oa_chol_av1", "sbf_oa_chol_av3", "sbf_no_load")
U2OS_CLOSING_RADIUS = 10
U2OS_MIN_PIECE_FRACTION = 0.1
U2OS_MAX_PIECE_GAP = 50
U2OS_MIN_FILLED_FRACTION = 0.05


def to_uint8(raw):
    if raw.dtype == np.uint8:
        return raw
    lo, hi = np.percentile(raw, (0.5, 99.5))
    return (np.clip((raw.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")


def downscale(raw, labels, factor):
    """Mean-downscale the raw data and subsample the labels by an integer factor along all axes."""
    if factor == 1:
        return raw, labels
    factors = (factor,) * raw.ndim
    small = downscale_local_mean(raw.astype("float32"), factors).astype("uint8")
    sl = tuple(slice(0, s * factor, factor) for s in small.shape)
    return small, labels[sl]


def clean_instances(labels, fill=True):
    """Fill holes per object, drop objects below MIN_OBJECT_SIZE and relabel consecutively."""
    out = np.zeros(labels.shape, dtype="uint32")
    next_id = 1
    for i, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        mask = labels[box] == i
        if fill:
            mask = ndimage.binary_fill_holes(mask)
        if mask.sum() < MIN_OBJECT_SIZE:
            continue
        out[box][mask] = next_id
        next_id += 1
    return out


def write_h5(path, raw, labels, attrs=None):
    tmp_path = f"{path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
    os.replace(tmp_path, path)


def build_cardiomyocyte(out_root):
    src = os.path.join(DATA_ROOT, "cardiomyocyte_tem")
    out_dir = os.path.join(out_root, "cardiomyocyte_tem_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for image_path in sorted(glob.glob(os.path.join(src, "images", "*.tif"))):
        name = os.path.basename(image_path)[:-4]
        masks = [
            p for p in glob.glob(os.path.join(src, "Nucleus_Masks", "*.tif"))
            if os.path.basename(p) in (f"{name}_nucleus.tif", f"KO_{name}_nucleus.tif")
        ]
        if not masks:
            continue
        raw = to_uint8(tifffile.imread(image_path))
        labels = label(tifffile.imread(masks[0]) > 0).astype("uint32")
        raw, labels = downscale(raw, labels, CARDIOMYOCYTE_SCALE)
        labels = clean_instances(labels)
        write_h5(os.path.join(out_dir, f"{name}.h5"), raw, labels, {"scale": CARDIOMYOCYTE_SCALE})
        infos[name] = {"shape": list(raw.shape), "n_nuclei": int(labels.max())}
        print("cardiomyocyte", name, infos[name], flush=True)
    return infos


def build_perez(out_root):
    src = os.path.join(DATA_ROOT, "perez", "perez", "nuclei")
    infos = {}
    for split in ("train", "test"):
        out_dir = os.path.join(out_root, "perez_nucleus", split)
        os.makedirs(out_dir, exist_ok=True)
        for image_path in sorted(glob.glob(os.path.join(src, split, "images", "*.png"))):
            name = os.path.basename(image_path)[:-4]
            raw = imageio.imread(image_path)
            raw = raw[..., 0] if raw.ndim == 3 else raw
            mask = imageio.imread(os.path.join(src, split, "masks", os.path.basename(image_path)))
            mask = (mask[..., 0] if mask.ndim == 3 else mask) > 0
            if raw.shape[0] < 512:
                raw = resize(raw, (512, 512), preserve_range=True, anti_aliasing=True).astype("uint8")
                mask = resize(mask, (512, 512), order=0, anti_aliasing=False).astype(bool)
            labels = clean_instances(label(mask).astype("uint32"))
            write_h5(os.path.join(out_dir, f"{name}.h5"), raw, labels)
            infos[f"{split}/{name}"] = int(labels.max())
    print("perez", {split: sum(1 for k in infos if k.startswith(split)) for split in ("train", "test")}, flush=True)
    return infos


def border_ignore(labels, width):
    """IGNORE_LABEL for the unlabeled pixels within width px of the image border."""
    band = np.zeros(labels.shape, dtype=bool)
    band[:width] = band[-width:] = True
    band[:, :width] = band[:, -width:] = True
    out = labels.copy()
    out[band & (labels == 0)] = IGNORE_LABEL
    return out


def build_vem(out_root):
    src = os.path.join(DATA_ROOT, "vem_nuclei")
    out_dir = os.path.join(out_root, "vem_nuclei_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for folder in sorted(glob.glob(os.path.join(src, "id_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        sample_id = int(folder.rsplit("_", 1)[1])
        if sample_id in VEM_MISSING:
            continue
        raw = tifffile.imread(glob.glob(os.path.join(folder, "img_*"))[0])
        labels = tifffile.imread(glob.glob(os.path.join(folder, "lbl_*"))[0])
        if raw.ndim == 3:
            raw, labels = downscale(to_uint8(raw), label(labels > 0).astype("uint32"), VEM_3D_SCALE)
            labels = clean_instances(labels, fill=False)
            factor = VEM_3D_SCALE
        else:
            labels = labels.astype("uint32")
            sizes = np.bincount(labels.ravel())[1:]
            diameter = float(np.sqrt(4 * np.median(sizes[sizes > 0]) / np.pi))
            factor = int(max(1, 2 ** round(np.log2(diameter / VEM_TARGET_DIAMETER))))
            raw, labels = downscale(to_uint8(raw), labels, factor)
            labels = clean_instances(labels)
            if sample_id not in VEM_EDGE_NUCLEI_INCLUDED and labels.max() > 0:
                boxes = [b for b in ndimage.find_objects(labels) if b is not None]
                extents = [max(b[0].stop - b[0].start, b[1].stop - b[1].start) for b in boxes]
                width = int(np.percentile(extents, VEM_BORDER_PERCENTILE))
                labels = border_ignore(labels, width)
        write_h5(os.path.join(out_dir, f"id_{sample_id}.h5"), raw, labels, {"scale": factor})
        n = len(np.unique(labels[(labels > 0) & (labels != IGNORE_LABEL)]))
        infos[f"id_{sample_id}"] = {
            "shape": list(raw.shape), "scale": factor, "n_nuclei": n,
            "ignore_fraction": float((labels == IGNORE_LABEL).mean()),
        }
        print("vem", sample_id, infos[f"id_{sample_id}"], flush=True)
    return infos


def fill_envelope(envelope):
    """The nucleus part of a crop that cuts the nucleus, from the envelope arc, or None if it is not reliable.

    The crops cut the nucleus, so the envelope is an arc (often dashed) with the nucleus on its right. A nucleus is
    convex. Thus, its part in the crop is the convex hull of the arc and the right border stretch that the arc spans in
    y. The arc has these parts:
    - the envelope pieces (after closing small gaps) of at least U2OS_MIN_PIECE_FRACTION of the envelope.
    - the smaller pieces within U2OS_MAX_PIECE_GAP px of them, which are the ends of a dashed arc.

    Stray fragments farther away are dropped, since they would pull the hull across the cytoplasm.
    """
    if envelope.sum() < U2OS_MIN_ENVELOPE_PIXELS:
        return None
    pieces = label(ndimage.binary_closing(envelope, structure=disk(U2OS_CLOSING_RADIUS)))
    piece_sizes = np.bincount(pieces[envelope])
    piece_sizes[0] = 0
    kept = set(np.nonzero(piece_sizes >= U2OS_MIN_PIECE_FRACTION * envelope.sum())[0].tolist())
    while True:
        near = ndimage.distance_transform_edt(~np.isin(pieces, list(kept))) <= U2OS_MAX_PIECE_GAP
        added = set(np.unique(pieces[near & envelope]).tolist()) - kept - {0}
        if not added:
            break
        kept |= added
    arc = envelope & np.isin(pieces, list(kept))
    ys = np.nonzero(arc)[0]
    border = np.zeros_like(arc)
    border[ys.min():ys.max() + 1, -1] = True
    nucleus = convex_hull_image(arc | border)
    if not U2OS_MIN_FILLED_FRACTION <= nucleus.mean() <= 1 - U2OS_MIN_FILLED_FRACTION:
        return None
    return nucleus.astype("uint32")


def build_u2os(out_root):
    src = os.path.join(DATA_ROOT, "u2os_fibsem", "1_NE", "1_Train_NE")
    out_dir = os.path.join(out_root, "u2os_fibsem_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for label_path in sorted(glob.glob(os.path.join(src, "labels_tif", "*.tif"))):
        name = os.path.basename(label_path)[:-4]
        raw = to_uint8(tifffile.imread(os.path.join(src, "images_tif", f"{name}.tif")))
        # Some slices end in constant padding columns on the right; the nucleus reaches the last column with data.
        width = int(np.nonzero(raw.std(axis=0) >= 1)[0].max()) + 1
        filled = fill_envelope(tifffile.imread(label_path)[:, :width] > 0)
        nucleus = None if filled is None else np.pad(filled, ((0, 0), (0, raw.shape[1] - width)))
        out_path = os.path.join(out_dir, f"{name}.h5")
        if nucleus is None:
            # The output of an earlier build is removed, so a rejected slice is not evaluated.
            if os.path.exists(out_path):
                os.remove(out_path)
            infos[name] = {"used": False}
            continue
        write_h5(out_path, raw, nucleus)
        infos[name] = {"used": True, "nucleus_fraction": float((nucleus > 0).mean())}
    print("u2os", sum(i["used"] for i in infos.values()), "of", len(infos), "slices filled", flush=True)
    return infos


def build_etchacell(out_root):
    src = os.path.join(DATA_ROOT, "etchacell")
    out_dir = os.path.join(out_root, "etchacell_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for raw_path in sorted(glob.glob(os.path.join(src, "raw", "*.tiff"))):
        name = os.path.basename(raw_path)[:-5]
        out_path = os.path.join(out_dir, f"{name}.h5")
        envelope = tifffile.imread(os.path.join(src, "labels", f"{name}.tiff")) > 0
        closing = disk(ETCHACELL_CLOSING_RADIUS)[None]
        envelope = ndimage.binary_closing(envelope, structure=closing)
        filled = np.stack([ndimage.binary_fill_holes(plane) for plane in envelope])
        filled |= ndimage.binary_fill_holes(envelope)
        del envelope
        raw = tifffile.imread(raw_path)
        small = downscale_local_mean(raw.astype("float32"), (1, ETCHACELL_SCALE, ETCHACELL_SCALE)).astype("uint8")
        del raw
        labels = filled[:, ::ETCHACELL_SCALE, ::ETCHACELL_SCALE][:, :small.shape[1], :small.shape[2]]
        labels = clean_instances(label(labels).astype("uint32"), fill=False)
        write_h5(out_path, small, labels, {"scale_xy": ETCHACELL_SCALE, "voxel_size_nm": [50, 40, 40]})
        infos[name] = {
            "shape": list(small.shape), "n_nuclei": int(labels.max()), "nucleus_fraction": float((labels > 0).mean()),
        }
        print("etchacell", name, infos[name], flush=True)
    return infos


def build_aive(out_root):
    out_dir = os.path.join(out_root, "aive_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for volume in AIVE_VOLUMES:
        with h5py.File(os.path.join(DATA_ROOT, "aive", f"aive_dataset{volume}.h5"), "r") as f:
            raw_ds, label_ds = f["raw"], f["labels"]
            fz, fy, fx = AIVE_SCALE
            nz, ny, nx = (s // factor for s, factor in zip(raw_ds.shape, AIVE_SCALE))
            raw = np.zeros((nz, ny, nx), dtype="uint8")
            mask = np.zeros((nz, ny, nx), dtype=bool)
            step = 16 * fz
            for z0 in range(0, nz * fz, step):
                z1 = min(z0 + step, nz * fz)
                block = raw_ds[z0:z1, :ny * fy, :nx * fx].astype("float32")
                raw[z0 // fz:z1 // fz] = downscale_local_mean(block, AIVE_SCALE).astype("uint8")
                labels = label_ds[z0:z1:fz, :ny * fy:fy, :nx * fx:fx]
                mask[z0 // fz:z1 // fz] = np.isin(labels, AIVE_NUCLEUS_CLASSES)
        mask = np.stack([ndimage.binary_fill_holes(plane) for plane in mask])
        mask = ndimage.binary_fill_holes(mask)
        instances = clean_instances(label(mask).astype("uint32"), fill=False)
        write_h5(os.path.join(out_dir, f"aive_dataset{volume}.h5"), raw, instances, {"voxel_size_nm": [30, 26, 26]})
        infos[f"aive_dataset{volume}"] = {
            "shape": list(raw.shape), "n_nuclei": int(instances.max()),
            "nucleus_fraction": float((instances > 0).mean()),
        }
        print("aive", volume, infos[f"aive_dataset{volume}"], flush=True)
    return infos


def downscale_volume(path, raw_key, label_key, scale, binary):
    """Read a volume slab-wise and downscale it: mean for the raw data, subsampling for the labels."""
    with h5py.File(path, "r") as f:
        raw_ds, label_ds = f[raw_key], f[label_key]
        fz, fy, fx = scale
        nz, ny, nx = (s // factor for s, factor in zip(raw_ds.shape, scale))
        raw = np.zeros((nz, ny, nx), dtype="uint8")
        labels = np.zeros((nz, ny, nx), dtype="uint32")
        step = 16 * fz
        for z0 in range(0, nz * fz, step):
            z1 = min(z0 + step, nz * fz)
            block = raw_ds[z0:z1, :ny * fy, :nx * fx].astype("float32")
            raw[z0 // fz:z1 // fz] = downscale_local_mean(block, scale).astype("uint8")
            block_labels = label_ds[z0:z1:fz, :ny * fy:fy, :nx * fx:fx]
            labels[z0 // fz:z1 // fz] = block_labels > 0 if binary else block_labels
    return raw, label(labels > 0).astype("uint32") if binary else labels


def build_volumes(out_root, folder, volumes, scale, binary, min_nucleus_size=None):
    out_dir = os.path.join(out_root, folder)
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for name, path, label_key in volumes:
        if not os.path.exists(path):
            print("missing", path, flush=True)
            continue
        raw, labels = downscale_volume(path, "raw", label_key, scale, binary)
        labels = clean_instances(labels, fill=binary)
        if min_nucleus_size is not None:
            sizes = np.bincount(labels.ravel())
            small = np.nonzero(sizes < min_nucleus_size)[0]
            labels[np.isin(labels, small[small > 0])] = IGNORE_LABEL
        write_h5(os.path.join(out_dir, f"{name}.h5"), raw, labels, {"scale_zyx": list(scale)})
        n_nuclei = len(np.setdiff1d(np.unique(labels), [0, IGNORE_LABEL]))
        infos[name] = {"shape": list(raw.shape), "n_nuclei": n_nuclei, "ignored": bool((labels == IGNORE_LABEL).any())}
        print(folder, name, infos[name], flush=True)
    return infos


def build_cerebellum(out_root):
    path = os.path.join(DATA_ROOT, "cerebellum_fibsem", "cerebellum_fibsem.h5")
    return build_volumes(
        out_root, "cerebellum_fibsem_nucleus", [("cerebellum_fibsem", path, "labels/nuclei")], CEREBELLUM_SCALE, False,
        min_nucleus_size=CEREBELLUM_MIN_NUCLEUS_SIZE,
    )


def build_macrophage(out_root):
    root = os.path.join(DATA_ROOT, "macrophage_a431")
    volumes = [(name, os.path.join(root, f"{name}.h5"), "labels/nucleus") for name in MACROPHAGE_VOLUMES]
    return build_volumes(out_root, "macrophage_a431_nucleus", volumes, MACROPHAGE_SCALE, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DATA_ROOT, help="Root of the derived data folders.")
    parser.add_argument(
        "--datasets", nargs="+", default=["cardiomyocyte", "perez", "vem", "u2os"], help="The datasets to build."
    )
    args = parser.parse_args()
    builders = {
        "cardiomyocyte": (build_cardiomyocyte, "cardiomyocyte_tem_nucleus"), "perez": (build_perez, "perez_nucleus"),
        "vem": (build_vem, "vem_nuclei_nucleus"), "u2os": (build_u2os, "u2os_fibsem_nucleus"),
        "etchacell": (build_etchacell, "etchacell_nucleus"), "aive": (build_aive, "aive_nucleus"),
        "cerebellum": (build_cerebellum, "cerebellum_fibsem_nucleus"),
        "macrophage": (build_macrophage, "macrophage_a431_nucleus"),
    }
    for name in args.datasets:
        build, folder = builders[name]
        infos = build(args.output)
        with open(os.path.join(args.output, folder, "build_info.json"), "w") as f:
            json.dump(infos, f, indent=2)


if __name__ == "__main__":
    main()
