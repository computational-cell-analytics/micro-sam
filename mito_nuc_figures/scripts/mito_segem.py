"""Overview figures for MitoSegEM (EMPIAR-12535): one representative subvolume per tissue.

The torch-em loader downloads every full raw slice (33-217 MB each) and crops it afterwards.
This script fetches only the TIFF strips that overlap the bounding box with HTTP range requests
and writes the same zarr store as `get_mito_segem_data` (same file name, arrays, chunks and attrs),
so `get_mito_segem_dataset(path, ..., bounding_boxes=[box], tissue=tissue)` reuses the cache.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import imagecodecs
import numpy as np
import tifffile
import zarr
from zarr.codecs import BloscCodec

from torch_em.data.datasets.electron_microscopy import mito_segem

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "mito_segem"
DATA_PATH = os.path.join(DATA_ROOT, DATASET)
# Boxes as (z_min, z_max, y_min, y_max, x_min, x_max) at native resolution, picked as the 1024 x 1024 window
# with the highest labeled-mito fraction in one label slice of the volume center.
BOXES = {
    "intestine": [(468, 532, 2304, 3328, 2816, 3840)],
    "testis": [(1268, 1332, 1792, 2816, 7680, 8704)],
    "brainstem": [(1868, 1932, 7936, 8960, 7168, 8192)],
}
N_WORKERS = 4

ISSUES = (
    "Loader is very slow: mito_segem.py:72 downloads each full raw slice (intestine 33 MB, testis 49 MB, brainstem "
    "217 MB) to crop it, serially; EBI gives ~1.6 MB/s through the cluster proxy (64 brainstem slices = ~2.5 h). "
    "This script reads only the TIFF strips that overlap the box (uncompressed for intestine/testis, PackBits for "
    "brainstem) with 4 threads and writes the identical zarr cache (~11 min per 64-slice box). Brainstem label "
    "slices are 17408 x 15360 while raw is 16449 x 15303; the crop at the origin is aligned (labels are padded at "
    "the end), so the loader's crop works."
)
OBSERVATIONS = (
    "All 3 tissues aligned. Despite the 'sparse' docstring, the labels in these boxes look near-exhaustive: most "
    "visible mito profiles are labeled (intestine: a few mito-like profiles unlabeled per slice, roughly 80-90% "
    "coverage; testis and brainstem look dense). Raw is noisy SBF-SEM, 50 nm z; brainstem has many myelinated axons."
)


def enable_proxy():
    """Let the aiohttp-based fsspec HTTP filesystem use the http(s)_proxy environment variables."""
    for protocol in ("http", "https"):
        fsspec.config.conf[protocol] = {"client_kwargs": {"trust_env": True}}


def read_raw_crop(url, y_min, y_max, x_min, x_max):
    """Read only the TIFF strips that overlap the rows y_min:y_max and crop them."""
    fs = fsspec.filesystem("https")
    for attempt in range(5):
        try:
            with fs.open(url, "rb", block_size=2**16) as f:
                page = tifffile.TiffFile(f).pages[0]
                assert page.predictor == 1 and page.compression in (1, 32773), (page.predictor, page.compression)
                height, width = page.shape
                rows_per_strip = min(page.rowsperstrip, height)
                first, last = y_min // rows_per_strip, (y_max - 1) // rows_per_strip
                start = page.dataoffsets[first]
                stop = page.dataoffsets[last] + page.databytecounts[last]
                f.seek(start)
                data = f.read(stop - start)
                strips = []
                for s in range(first, last + 1):
                    blob = data[page.dataoffsets[s] - start:page.dataoffsets[s] - start + page.databytecounts[s]]
                    if page.compression == 32773:
                        blob = imagecodecs.packbits_decode(blob)
                    strips.append(np.frombuffer(blob, dtype="uint8").reshape(-1, width))
                band = np.concatenate(strips, axis=0)
                offset = first * rows_per_strip
                return band[y_min - offset:y_max - offset, x_min:x_max]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def fetch_slice(tissue, z, box):
    info = mito_segem.TISSUES[tissue]
    _, _, y_min, y_max, x_min, x_max = box
    raw_url = f"{mito_segem.BASE_URL}/{info['raw_dir']}/{info['raw_pattern'].format(z=z + info['first_slice_index'])}"
    label_url = f"{mito_segem.BASE_URL}/{info['seg_dir']}/{z:04d}.h5"
    raw = read_raw_crop(raw_url, y_min, y_max, x_min, x_max)
    labels = mito_segem._read_label_slice(label_url, y_min, y_max, x_min, x_max)
    return raw, labels


def download_box(tissue, box):
    """Cache one box in the same zarr layout as `mito_segem.get_mito_segem_data`."""
    os.makedirs(DATA_PATH, exist_ok=True)
    zarr_path = os.path.join(DATA_PATH, f"{tissue}_{mito_segem._bbox_to_str(box)}.zarr")
    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    z_min, z_max, y_min, y_max, x_min, x_max = box
    shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    raw = np.zeros(shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint32)
    t0 = time.time()
    with ThreadPoolExecutor(N_WORKERS) as executor:
        futures = {executor.submit(fetch_slice, tissue, z, box): i for i, z in enumerate(range(z_min, z_max))}
        for n_done, future in enumerate(futures, start=1):
            i = futures[future]
            raw[i], labels[i] = future.result()
            print(f"{tissue} slice {n_done}/{len(futures)} ({time.time() - t0:.0f}s)", flush=True)

    for name, data, shuffle in (("raw", raw, "shuffle"), ("labels", labels, "bitshuffle")):
        array = root.create_array(
            name, shape=data.shape, chunks=(32, 512, 512), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        array[:] = data
    root.attrs["bounding_box"] = list(box)
    root.attrs["tissue"] = tissue
    root.attrs["resolution_nm"] = list(mito_segem.TISSUES[tissue]["resolution_nm"])
    root.attrs["labels_are_exhaustive"] = False
    return zarr_path


def folder_size_gb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        total += sum(os.path.getsize(os.path.join(dirpath, name)) for name in filenames)
    return total / 1e9


def render_box(tissue, box, out_dir):
    zarr_path = download_box(tissue, box)
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["raw"], root["labels"]
    resolution = mito_segem.TISSUES[tissue]["resolution_nm"]
    name = f"{tissue}_z{box[0]}-{box[1]}"
    voxel_size = f"{resolution[1]}x{resolution[2]}x{resolution[0]} nm (xyz)"
    notes = f"SBF-SEM mouse {tissue}, {voxel_size}, sparse instance mito labels (prediction + webKnossos)"
    save_volume_figure(raw, {"mito": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    label_data = labels[:]
    ids = np.unique(label_data)
    return {
        "name": name, "tissue": tissue, "shape": raw.shape, "voxel_size_nm": resolution,
        "bounding_box_zyx": list(box), "full_volume_shape_yx": mito_segem.TISSUES[tissue]["shape_yx"],
        "full_volume_n_slices": mito_segem.TISSUES[tissue]["num_slices"], "path": zarr_path,
        "n_objects": int((ids != 0).sum()), "foreground_fraction": float((label_data > 0).mean()),
    }


def main():
    enable_proxy()
    out_dir = get_output_dir(DATASET)
    volumes = [render_box(tissue, box, out_dir) for tissue, boxes in BOXES.items() for box in boxes]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "SBF-SEM (serial block-face SEM)", "dimensionality": "3d",
        "label_quality": "network prediction + webKnossos instance assignment, sparse (not exhaustive), instance",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_PATH,
        "download": f"one box per tissue (64 z-slices x 1024 x 1024), {folder_size_gb(DATA_PATH):.2f} GB on disk",
        "notes": "3 tissues (intestine 8 nm, testis and brainstem 15 nm xy, 50 nm z); boxes can be passed to "
        "get_mito_segem_dataset(bounding_boxes=[...], tissue=...) and hit the cache",
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
