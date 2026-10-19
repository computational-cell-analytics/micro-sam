"""Overview figures for the Parlakgul liver FIB-SEM mitochondria labels (EMPIAR-10791).

The torch-em loader only resolves the EMPIAR file names of sample 6461. For 6464, 9430 and 1857 the raw
folder, the raw file pattern and the segmentation zip names in PARLAKGUL_SAMPLES do not match EMPIAR, and the
segmentation slices are not zero-padded consistently, so sorting the zip members by name does not give z order.
This script streams one central subvolume per sample with the correct file names and caches it as zarr v3 in
the same layout and file naming as `get_parlakgul_liver_data` (in a separate folder, because the dataset folder
is read-only for this user), so the loader can use the cached boxes when pointed to that folder.
"""
import io
import os
import re
import struct
import zipfile
import zlib

import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "parlakgul_liver"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
# DATA_DIR is not writable for this user, so the new boxes are cached next to it.
BOX_DIR = os.path.join(DATA_ROOT, f"{DATASET}_boxes")
NOTES = "FIB-SEM mouse liver, 8 nm isotropic, semantic mito masks (0/255) from {condition} sample {sample}"
BOX_XY = 1536
BOX_Z = 16
EMPIAR_BASE = (
    "https://ftp.ebi.ac.uk/empiar/world_availability/10791/data/Parlakgul%20et%20al%20-%20Regulation%20of%20liver"
    "%20subcellular%20architecture%20controls%20metabolic%20homeostasis/FIB-SEM%20Raw%20and%20Segmentation%20Data"
)
SAMPLES = {
    "6464": {
        "condition": "obese1",
        "shape": (7896, 10200, 9112),
        "raw_url": "6464%20-%20Obese1%20Liver/6464%20Obese1%20Liver%20-%20Raw/6464%20Ob%20Fed%20Raw%20with%20All_T001_Z{z:03d}_C09.tif",  # noqa
        "seg_url": "6464%20-%20Obese1%20Liver/6464%20Obese1%20Liver%20-%20Segmentation/6464%20Obese1%20Mitochondria.zip",  # noqa
        "raw_offset": 1,
        "seg_offset": 0,
    },
    "9430": {
        "condition": "obese2",
        "shape": (8501, 8050, 8000),
        "raw_url": "9430%20-%20Obese2%20Liver/9430%20Obese2%20Liver%20-%20Raw/9430%20raw/9430%20Ob%20Fed%20Raw_T001_Z{z:03d}_C07.tif",  # noqa
        "seg_url": "9430%20-%20Obese2%20Liver/9430%20Obese2%20Liver%20-%20Segmentation/9430%20Obese2%20Mitochondria.zip",  # noqa
        "raw_offset": 1,
        "seg_offset": 0,
    },
    "1857": {
        "condition": "obese_climp63",
        "shape": (3629, 9650, 9700),
        "raw_url": "1857%20-%20Obese%20Climp-63%20Liver/1857%20Obese%20Climp-63%20Raw/1857_Obese_Climp63_raw_T001_Z{z:03d}_C07.tif",  # noqa
        "seg_url": "1857%20-%20Obese%20Climp-63%20Liver/1857%20Obese%20Climp-63%20Segmentation/1857_Obese_Climp63_Mitochondria.zip",  # noqa
        "raw_offset": 1,
        "seg_offset": 1,
    },
}


def get_center_box(shape):
    """Get a (x_min, x_max, y_min, y_max, z_min, z_max) box in the volume center."""
    nz, ny, nx = shape
    z0, y0, x0 = nz // 2 - BOX_Z // 2, ny // 2 - BOX_XY // 2, nx // 2 - BOX_XY // 2
    return (x0, x0 + BOX_XY, y0, y0 + BOX_XY, z0, z0 + BOX_Z)


def index_zip_members(zf):
    """Map the z index parsed from each segmentation tiff name to the zip member name."""
    members = {}
    for name in zf.namelist():
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        stem = os.path.splitext(os.path.basename(name))[0]
        match = re.search(r"Z(\d+)_C\d+$", stem) or re.fullmatch(r"(\d+)", stem)
        members[int(match.group(1))] = name
    return members


def read_zip_member(fh, info):
    """Read one zip member with a single HTTP range request, zipfile itself reads in 4 kB requests."""
    fh.seek(info.header_offset)
    name_length, extra_length = struct.unpack("<HH", fh.read(30)[26:30])
    fh.seek(info.header_offset + 30 + name_length + extra_length)
    data = fh.read(info.compress_size)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(data)
    else:
        assert info.compress_type == zipfile.ZIP_STORED
    assert len(data) == info.file_size
    return data


def read_raw_rows(url, x_min, x_max, y_min, y_max):
    """Read only the rows y_min:y_max of an uncompressed raw tiff with one HTTP range request."""
    import tifffile
    from torch_em.data.datasets.electron_microscopy.parlakgul_liver import _HttpFile

    fh = _HttpFile(url)
    with tifffile.TiffFile(fh) as tif:
        page = tif.pages[0]
        height, width = page.shape
        offsets, counts = page.dataoffsets, page.databytecounts
        assert page.compression == 1 and page.dtype == np.uint8 and page.rowsperstrip == 1
    assert offsets[y_max - 1] - offsets[y_min] == (y_max - 1 - y_min) * width and counts[y_min] == width
    fh.seek(offsets[y_min])
    data = fh.read((y_max - y_min) * width)
    return np.frombuffer(data, dtype="uint8").reshape(y_max - y_min, width)[:, x_min:x_max], (height, width)


def stream_box(sample, bbox, zarr_path):
    """Stream raw and mito labels for one box and cache them like `get_parlakgul_liver_data`."""
    import tifffile
    import zarr
    from zarr.codecs import BloscCodec
    from torch_em.data.datasets.electron_microscopy.parlakgul_liver import PARLAKGUL_CHUNK_SHAPE, _HttpFile

    root = zarr.open_group(zarr_path, mode="a")
    info = SAMPLES[sample]
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    raw = np.zeros(shape, dtype="uint8")
    labels = np.zeros(shape, dtype="uint8")
    seg_fh = _HttpFile(f"{EMPIAR_BASE}/{info['seg_url']}")
    zf = zipfile.ZipFile(seg_fh)
    members = index_zip_members(zf)
    seg_shape, raw_shape = None, None
    for i, z in enumerate(range(z_min, z_max)):
        raw_url = f"{EMPIAR_BASE}/{info['raw_url'].format(z=z + info['raw_offset'])}"
        raw[i], raw_shape = read_raw_rows(raw_url, x_min, x_max, y_min, y_max)
        member = zf.getinfo(members[z + info["seg_offset"]])
        seg = tifffile.imread(io.BytesIO(read_zip_member(seg_fh, member)))
        seg_shape = seg.shape
        labels[i] = seg[y_min:y_max, x_min:x_max]
        print(f"{sample}: slice {i + 1}/{shape[0]}", flush=True)

    for name, data, shuffle in (("raw", raw, "shuffle"), ("labels", labels, "bitshuffle")):
        arr = root.create_array(
            name, shape=data.shape, chunks=PARLAKGUL_CHUNK_SHAPE, dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data
    root.attrs.update({
        "bounding_box": list(bbox), "sample": sample, "label_choice": "mito", "resolution_nm": [8, 8, 8],
        "raw_slice_shape": list(raw_shape), "segmentation_slice_shape": list(seg_shape),
    })


def download_missing_samples():
    """Stream one central box for each sample that has no cached mito box yet."""
    from torch_em.data.datasets.electron_microscopy.parlakgul_liver import _bbox_to_str

    os.makedirs(BOX_DIR, exist_ok=True)
    for sample, info in SAMPLES.items():
        bbox = get_center_box(info["shape"])
        zarr_path = os.path.join(BOX_DIR, f"{sample}_mito_{_bbox_to_str(bbox)}.zarr")
        if os.path.exists(os.path.join(zarr_path, "labels")):
            continue
        stream_box(sample, bbox, zarr_path)


def render_box(zarr_path, out_dir):
    import zarr

    root = zarr.open_group(zarr_path, mode="r")
    sample, bbox = root.attrs["sample"], root.attrs["bounding_box"]
    raw, labels = root["raw"], root["labels"]
    condition = SAMPLES[sample]["condition"] if sample in SAMPLES else "lean"
    name = f"{sample}_{condition}_x{bbox[0]}_y{bbox[2]}_z{bbox[4]}"
    notes = NOTES.format(condition=condition, sample=sample)
    save_volume_figure(raw, {"mito": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    fg_fraction = float(np.mean([(labels[z] > 0).mean() for z in range(0, labels.shape[0], 4)]))
    return {
        "name": name, "sample": sample, "condition": condition, "shape": raw.shape, "voxel_size_nm": (8, 8, 8),
        "bounding_box_xyz": bbox, "mito_foreground_fraction": round(fg_fraction, 4), "path": zarr_path,
    }


def main():
    download_missing_samples()
    out_dir = get_output_dir(DATASET)
    paths = [
        os.path.join(folder, name) for folder in (DATA_DIR, BOX_DIR) if os.path.exists(folder)
        for name in sorted(os.listdir(folder)) if "_mito_" in name and name.endswith(".zarr")
    ]
    volumes = [render_box(path, out_dir) for path in paths]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "FIB-SEM", "dimensionality": "3d",
        "label_quality": "semantic binary masks (0/255), produced by the authors with ML + proofreading",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": [DATA_DIR, BOX_DIR],
        "download": f"streamed one central {BOX_Z}x{BOX_XY}x{BOX_XY} box for 6464, 9430, 1857 (mito only) to {BOX_DIR}",
        "notes": "4 full volumes (6461 lean, 6464 obese1, 9430 obese2, 1857 obese Climp63) of 3.6k-8.5k slices",
        "issues": (
            "torch-em parlakgul_liver.py only works for 6461: for 6464 / 9430 / 1857 the raw_pattern (lines 72, 88, "
            "104) does not exist on EMPIAR (real names are e.g. '6464 Ob Fed Raw with All_T001_Z001_C09.tif', "
            "1-based z), 9430 raw lives in a '9430 raw' subfolder, the 1857 raw_dir / seg_dir and zip names differ "
            "('1857 Obese Climp-63 Raw', '1857_Obese_Climp63_Mitochondria.zip'), and _read_zip_slice (line 186) sorts "
            "member names lexicographically, which breaks z order for 6464 (000..7895) and 1857 (Z1..Z3629); "
            "_read_zip_slice also reopens the zip per slice and zipfile reads it in 4 kB HTTP requests (very slow); "
            "labels are stored as 0/255, not 0/1 as the docstring says; EMPIAR throughput was 0.2-3 MB/s, so the "
            "new boxes are 16 x 1536 x 1536"
        ),
        "observations": (
            "all 5 boxes (2 existing 6461, new 6464 / 9430 / 1857) aligned and exhaustive, with smooth, slightly "
            "generous mask outlines; numeric z mapping of the segmentation zips is correct (labels track mitos "
            "across z); semantic only, touching mitos merge (e.g. 9430), so instances need connected components "
            "or are not reliable"
        ),
    })


if __name__ == "__main__":
    main()
