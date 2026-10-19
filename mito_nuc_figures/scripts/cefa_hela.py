import os
import struct
import time
from glob import glob

import h5py
import numpy as np
import requests
import scipy.io

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info


DATASET = "cefa_hela"
ZENODO_RECORD = "https://zenodo.org/api/records/4590903"
EMPIAR_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/10094/data/Micrographs_ROI_00"
DM4_NAME = "Prefix_3VBSED_roi_00_slice_{:04d}.dm4"
N_SAMPLES = 12
N_MATCHED = 4
# Estimated by normalized cross-correlation of the GT background class (4) against the 8x downsampled full
# EMPIAR slice 150, for GT slices 124-163 (all gave y=5016-5096, x=5512-5600).
# The loader assumes the center (3096, 3096).
MATCHED_ORIGIN = (5050, 5580)
NOTES = "SBF-SEM HeLa (EMPIAR-10094), 10x10x50 nm, manual semantic labels of one central cell (2000x2000 ROI)"
ISSUES = (
    "Loader not usable as is: (1) ZENODO_URL 'records/4590903/files-archive' returns 404 (cefa_hela.py ZENODO_URL / "
    "get_cefa_hela_data util.download_source), worked around by downloading the 300 GT_Slice_*.mat files one by one "
    "(Zenodo rate-limits with HTTP 429, needs retries). (2) EMPIAR-10094 download needs aspera 'ascp' "
    "(not installed) and is 518 DM4 slices x 257 MB = ~133 GB; HTTPS range reads from ftp.ebi.ac.uk work instead. "
    "(3) _preprocess_data needs ncempy (not installed); a minimal DM4 reader in this script works (uint32, "
    "8192x8192, 10.01 nm). (4) Correctness bug: _preprocess_data center-crops the 8192x8192 raw to 2000x2000, but "
    "the GT ROI is not centered: template matching puts it at about y=5050, x=5580 (center crop starts at 3096, "
    "3096), so loader pairs would be misaligned. The z pairing (GT_Slice_k <-> sorted dm4 index k-1) and the exact "
    "xy offset (+-~30 px) are not verified."
)
OBSERVATIONS = (
    "Nucleus class 2 is a FILLED nucleus (separate thin nuclear_envelope ring, class 1, around it); one nucleus of the "
    "single central cell, present in 239/300 slices; label stack is smooth and consistent in 3D (ZX view). Other cells "
    "are class 3 (no nucleus labels), so nuclei of neighbouring cells visible in the raw are unlabeled. "
    "With the loader's center crop the labels do not match the raw at all (slices_loader_center_crop.png); with the "
    "template-matched origin (slices_matched_origin.png) the nucleus label sits on the nucleus, though some slices "
    "still look offset in z or by tens of px, so a proper registration is needed before training."
)
DM4_SIMPLE_TYPES = {2: "h", 3: "i", 4: "H", 5: "I", 6: "f", 7: "d", 8: "?", 9: "b", 10: "B", 11: "q", 12: "Q"}
DM4_DTYPES = {2: "<i2", 3: "<i4", 4: "<u2", 5: "<u4", 6: "<f4", 7: "<f8", 9: "<i1", 10: "<u1", 11: "<i8", 12: "<u8"}


def get_data_root():
    return os.path.join(DATA_ROOT, DATASET)


def get_with_retry(url, n_tries=20):
    """Zenodo rate-limits single file downloads (HTTP 429), so back off and retry."""
    for attempt in range(n_tries):
        resp = requests.get(url, timeout=120)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.content
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Too many retries for {url}")


def download_gt():
    """Try the torch-em loader first. Its Zenodo 'files-archive' URL returns 404, so fall back to single files."""
    from torch_em.data.datasets.electron_microscopy import cefa_hela

    try:
        cefa_hela.get_cefa_hela_data(get_data_root(), download=True)
    except Exception as e:
        print(f"Loader download failed: {e!r}")

    gt_dir = os.path.join(get_data_root(), "gt_labels")
    os.makedirs(gt_dir, exist_ok=True)
    files = requests.get(ZENODO_RECORD, timeout=60).json()["files"]
    for file_info in files:
        out_path = os.path.join(gt_dir, file_info["key"])
        if not file_info["key"].endswith(".mat") or os.path.exists(out_path):
            continue
        content = get_with_retry(file_info["links"]["self"])
        with open(out_path, "wb") as f:
            f.write(content)
    return sorted(glob(os.path.join(gt_dir, "GT_Slice_*.mat")))


class RemoteFile:
    """Read byte ranges of a remote file via HTTP range requests, with a small read-ahead buffer."""

    def __init__(self, url):
        self.url = url
        self.start, self.buffer = 0, b""

    def fetch(self, offset, n):
        resp = requests.get(self.url, headers={"Range": f"bytes={offset}-{offset + n - 1}"}, timeout=600)
        resp.raise_for_status()
        return resp.content

    def read(self, offset, n):
        if offset < self.start or offset + n > self.start + len(self.buffer):
            self.start, self.buffer = offset, self.fetch(offset, max(n, 1 << 20))
        return self.buffer[offset - self.start:offset - self.start + n]


def parse_dm4_group(f, pos, path, tags):
    """Parse a DM4 tag group. Large arrays are not read, only their offset is recorded."""
    n_tags = struct.unpack(">Q", f.read(pos + 2, 8))[0]
    pos += 10
    for i in range(n_tags):
        tag_type = f.read(pos, 1)[0]
        name_length = struct.unpack(">H", f.read(pos + 1, 2))[0]
        name = f.read(pos + 3, name_length).decode("latin1") or str(i)
        pos += 3 + name_length
        size = struct.unpack(">Q", f.read(pos, 8))[0]
        start = pos + 8
        if tag_type == 20:
            parse_dm4_group(f, start, path + [name], tags)
        elif tag_type == 21:
            n_info = struct.unpack(">Q", f.read(start + 4, 8))[0]
            info = struct.unpack(f">{n_info}Q", f.read(start + 12, 8 * n_info))
            data_pos = start + 12 + 8 * n_info
            key = "/".join(path + [name])
            if info[0] in DM4_SIMPLE_TYPES:
                fmt = "<" + DM4_SIMPLE_TYPES[info[0]]
                tags[key] = struct.unpack(fmt, f.read(data_pos, struct.calcsize(fmt)))[0]
            elif info[0] == 20 and info[1] in DM4_DTYPES:
                tags[key] = ("array", DM4_DTYPES[info[1]], info[2], data_pos)
        pos = start + size


def read_dm4_crop(url, crop_shape, origin=None):
    """Read a crop (the center crop if origin is None) of the main image of a remote DM4 file.

    Only the rows of the crop are fetched.
    """
    f = RemoteFile(url)
    version, _, byte_order = struct.unpack(">IQI", f.read(0, 16))
    assert version == 4 and byte_order == 1, (version, byte_order)
    tags = {}
    parse_dm4_group(f, 16, [], tags)
    _, dtype, length, offset = tags["ImageList/1/ImageData/Data"]
    width, height = tags["ImageList/1/ImageData/Dimensions/0"], tags["ImageList/1/ImageData/Dimensions/1"]
    assert width * height == length
    pixel_size_um = tags["ImageList/1/ImageData/Calibrations/Dimension/0/Scale"]
    h, w = crop_shape
    y0, x0 = (height // 2 - h // 2, width // 2 - w // 2) if origin is None else origin
    itemsize = np.dtype(dtype).itemsize
    data = f.fetch(offset + y0 * width * itemsize, h * width * itemsize)
    rows = np.frombuffer(data, dtype=dtype).reshape(h, width)
    return rows[:, x0:x0 + w].astype("float32"), (height, width), pixel_size_um


def load_raw_slice(z, crop_shape, origin=None):
    """Load (and cache) a crop of EMPIAR slice z. The center crop (origin=None) is what the torch-em loader takes."""
    folder = "empiar_center_crops" if origin is None else f"empiar_crops_y{origin[0]}_x{origin[1]}"
    cache_dir = os.path.join(get_data_root(), folder)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"slice_{z:04d}.h5")
    if not os.path.exists(cache_path):
        raw, full_shape, pixel_size_um = read_dm4_crop(f"{EMPIAR_URL}/{DM4_NAME.format(z)}", crop_shape, origin)
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.attrs["full_shape"] = full_shape
            f.attrs["pixel_size_um"] = pixel_size_um
            f.attrs["source"] = f"{EMPIAR_URL}/{DM4_NAME.format(z)}"
    with h5py.File(cache_path, "r") as f:
        return f["raw"][:], tuple(int(n) for n in f.attrs["full_shape"]), float(f.attrs["pixel_size_um"])


def load_labels(gt_files):
    return np.stack([scipy.io.loadmat(path)["groundTruth"] for path in gt_files])


def per_class_labels(labels):
    return {"nucleus": (labels == 2).astype("uint8"), "nuclear_envelope": (labels == 1).astype("uint8")}


def render_samples(labels, indices, out_path, group_name, notes, origin=None):
    samples, full_shape, pixel_size_um = [], None, None
    for i in indices:
        raw, full_shape, pixel_size_um = load_raw_slice(i, labels.shape[1:], origin)
        samples.append((f"z={i}", raw, per_class_labels(labels[i])))
    save_image_grid(samples, out_path, DATASET, group_name, notes=notes)
    return full_shape, pixel_size_um


def render_all_samples(labels, out_dir):
    """Render random slices with the loader's center crop, and slices with nucleus at the matched ROI origin."""
    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(len(labels), size=N_SAMPLES, replace=False).tolist())
    notes = f"{NOTES}; raw = torch-em loader center crop"
    out_path = os.path.join(out_dir, "slices_loader_center_crop.png")
    full_shape, pixel_size_um = render_samples(labels, indices, out_path, "roi_00 random slices", notes)
    with_nucleus = [i for i in indices if (labels[i] == 2).any()]
    matched = with_nucleus[::2][:N_MATCHED]
    notes = f"{NOTES}; raw crop at template-matched origin y={MATCHED_ORIGIN[0]}, x={MATCHED_ORIGIN[1]}"
    out_path = os.path.join(out_dir, "slices_matched_origin.png")
    render_samples(labels, matched, out_path, "roi_00 slices with nucleus", notes, origin=MATCHED_ORIGIN)
    return indices, matched, full_shape, pixel_size_um


def render_label_volume(labels, out_dir):
    """Show the label stack as a volume, with the label map as the image, to check 3D consistency."""
    image = np.where(labels == 4, 0, labels).astype("float32")
    out_path = os.path.join(out_dir, "label_volume.png")
    notes = "image = label map (no raw for all 300 slices); nucleus = class 2"
    save_volume_figure(image, {"nucleus": (labels == 2).astype("uint8")}, out_path, DATASET, "gt stack", notes=notes)


def summarize_labels(labels):
    values, counts = np.unique(labels, return_counts=True)
    fractions = {int(v): round(float(c) / labels.size, 4) for v, c in zip(values, counts)}
    nucleus_per_slice = (labels == 2).reshape(len(labels), -1).sum(axis=1)
    return fractions, int((nucleus_per_slice > 0).sum())


def main():
    out_dir = get_output_dir(DATASET)
    gt_files = download_gt()
    labels = load_labels(gt_files)
    fractions, n_with_nucleus = summarize_labels(labels)
    indices, matched, full_shape, pixel_size_um = render_all_samples(labels, out_dir)
    render_label_volume(labels, out_dir)
    write_info(DATASET, {
        "classes": ["nucleus"], "modality": "SBF-SEM", "dimensionality": "2d (300 consecutive slices of one ROI)",
        "label_quality": "manual, semantic (1 nuclear_envelope, 2 nucleus, 3 other_cells, 4 background, 5 cell)",
        "n_volumes": 1,
        "volumes": [{
            "name": "roi_00", "shape": labels.shape, "voxel_size_nm": (50, 10, 10),
            "path": os.path.dirname(gt_files[0]),
            "label_dtype": str(labels.dtype), "class_fractions": fractions, "slices_with_nucleus": n_with_nucleus,
            "empiar_full_slice_shape": full_shape, "empiar_pixel_size_um": pixel_size_um,
            "rendered_slices": indices, "rendered_slices_matched_origin": matched,
            "matched_roi_origin_yx": MATCHED_ORIGIN,
        }],
        "data_path": get_data_root(),
        "download": (
            "300 GT_Slice_*.mat from Zenodo 4590903 (19 MB) into gt_labels/; 2000x2000 crops of "
            f"{len(indices)} EMPIAR-10094 DM4 slices (loader center crop) into empiar_center_crops/ and "
            f"{len(matched)} at the matched origin into empiar_crops_y5050_x5580/, via HTTP range requests "
            "(~65 MB transferred per slice, ~11 MB stored); full EMPIAR-10094 is ~133 GB"
        ),
        "notes": NOTES,
        "issues": ISSUES,
        "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
