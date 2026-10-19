"""Select, preview, download and downsample the human liver EM nucleus evaluation boxes.

The nucleus labels are the automated segmentation of the authors. They only cover the hepatocyte nuclei in the upper
part of the tissue. Thus, the boxes are placed where nuclei are labeled (see human_liver_em_nucleus_map.py). Every box
gets a visual check before use. The modes:
- 'select' scores 4096 x 4096 windows of BOX_DEPTH slices by the labeled nuclei in the sampled slices.
- 'preview' renders a strided view (every PREVIEW_STRIDE-th row and column) of the raw and the nucleus mask of a
  window. It reads one small range request per raw row, to check the window before the full download.
- 'download' streams a box at full resolution slice by slice (resumable) into <output>/fullres.
- 'finalize' downsamples the full resolution boxes 4x in xy and writes the evaluation boxes with nucleus instances.
"""
import io
import os
import json
import time
import shutil
import socket
import argparse
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import find_boundaries

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

from figure_utils import DATA_ROOT, get_output_dir  # noqa


EMPIAR_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/13356/data"
BOX_SIZE = 4096
SCALE = 4
PREVIEW_STRIDE = 8
OUTPUT = os.path.join(DATA_ROOT, "human_liver_em_nucleus_boxes")
FIGURE_DIR = os.path.join(get_output_dir("human_liver_em"), "nucleus_boxes")
# The lower part of the tissue (fibrous / portal region) has unlabeled nuclei, and the lower right holds an artefact.
MAX_Y = 11000


def select_windows(map_path, depth, step, n_best):
    data = np.load(map_path)
    maps, z_ids, block = data["maps"], data["z"], int(data["block"])
    win = BOX_SIZE // block
    candidates = []
    for z0 in range(0, int(z_ids.max()) - depth + 2, step):
        sel = (z_ids >= z0) & (z_ids < z0 + depth)
        if sel.sum() == 0:
            continue
        stack = maps[sel]
        for by in range(0, (MAX_Y - BOX_SIZE) // block + 1, 4):
            for bx in range(0, stack.shape[2] - win + 1, 4):
                window = stack[:, by:by + win, bx:bx + win]
                n_nuclei = np.mean([ndimage.label(w > 0.5)[1] for w in window])
                candidates.append({
                    "z": [z0, z0 + depth], "y": [by * block, by * block + BOX_SIZE],
                    "x": [bx * block, bx * block + BOX_SIZE],
                    "nucleus_fraction": float(window.mean()), "n_nuclei": float(n_nuclei),
                })
    candidates.sort(key=lambda c: (c["n_nuclei"], c["nucleus_fraction"]), reverse=True)
    return candidates[:n_best]


def with_retries(fn, n_attempts=12, max_wait=300):
    """Call *fn*, retrying with an exponential backoff on request errors.

    Under load the proxy answers 503 or drops the connection, and EMPIAR answers 403 for a while.
    """
    for attempt in range(n_attempts):
        try:
            return fn()
        except requests.RequestException as e:
            if attempt == n_attempts - 1:
                raise
            wait = min(2 ** attempt, max_wait)
            print(f"Request failed ({type(e).__name__}), retrying in {wait} s", flush=True)
            time.sleep(wait)


def tiff_strips(z):
    import tifffile
    from torch_em.data.datasets.electron_microscopy.human_liver_em import _HttpFile

    url = f"{EMPIAR_BASE}/humanliver_raw_images/humanliver_raw_{z:03d}.tif"

    def read_header():
        with tifffile.TiffFile(_HttpFile(url)) as tif:
            page = tif.pages[0]
            return url, page.dataoffsets, page.databytecounts, page.imagewidth, np.dtype(page.dtype)

    return with_retries(read_header)


def read_raw_row(url, offset, bytecount, width, dtype):
    from imagecodecs import delta_decode, lzw_decode

    def get():
        r = requests.get(url, headers={"Range": f"bytes={offset}-{offset + bytecount - 1}"}, timeout=120)
        r.raise_for_status()
        return r

    r = with_retries(get)
    decoded = np.frombuffer(lzw_decode(r.content), dtype=dtype).copy()
    return delta_decode(decoded, axis=-1, dist=1, out=decoded).reshape(width)


def read_nucleus_mask(z):
    Image.MAX_IMAGE_PIXELS = None
    url = f"{EMPIAR_BASE}/humanliver_nucleus_mask/humanliver_nucleus_mask_{z:03d}.png"

    def get():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return r

    r = with_retries(get)
    mask = np.array(Image.open(io.BytesIO(r.content)))
    return (mask[..., 0] if mask.ndim == 3 else mask) > 0


def preview_window(z, y0, x0, size=BOX_SIZE, stride=PREVIEW_STRIDE):
    url, offsets, bytecounts, width, dtype = tiff_strips(z)
    rows = list(range(y0, y0 + size, stride))
    with ThreadPoolExecutor(6) as pool:
        raw = list(pool.map(lambda y: read_raw_row(url, offsets[y], bytecounts[y], width, dtype)[x0:x0 + size], rows))
    raw = np.stack(raw)[:, ::stride]
    mask = read_nucleus_mask(z)[y0:y0 + size:stride, x0:x0 + size:stride]
    return raw, mask


def normalize(raw):
    raw = ndimage.gaussian_filter(raw.astype("float32"), 1.0)
    lo, hi = np.percentile(raw, [1, 99])
    return np.clip((raw - lo) / (hi - lo + 1e-6), 0, 1)


def save_preview(windows, out_path):
    fig, axes = plt.subplots(len(windows), 2, figsize=(14, 7 * len(windows)), squeeze=False)
    for row, (name, z, y0, x0) in zip(axes, windows):
        raw, mask = preview_window(z, y0, x0)
        overlay = np.zeros(raw.shape + (4,), dtype="float32")
        overlay[find_boundaries(mask)] = (0.0, 0.6, 1.0, 1.0)
        for ax in row:
            ax.imshow(normalize(raw), cmap="gray")
            ax.axis("off")
        row[1].imshow(overlay)
        row[0].set_title(f"{name}: z={z}, y={y0}, x={x0} (raw)", fontsize=9)
        row[1].set_title(f"{name}: labeled nuclei (blue)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=60)
    plt.close(fig)


def read_raw_rows(z, x0, x1, y0, y1):
    """Read rows y0:y1 of a raw slice with one range request over their strips (as the torch-em loader does)."""
    from imagecodecs import delta_decode, lzw_decode

    url, offsets, bytecounts, width, dtype = tiff_strips(z)
    start, end = offsets[y0], offsets[y1 - 1] + bytecounts[y1 - 1] - 1

    def get():
        r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=600)
        r.raise_for_status()
        if len(r.content) != end - start + 1:
            raise requests.RequestException("Incomplete range response.")
        return r

    data = with_retries(get).content
    rows = []
    for y in range(y0, y1):
        strip = data[offsets[y] - start:offsets[y] - start + bytecounts[y]]
        decoded = np.frombuffer(lzw_decode(strip), dtype=dtype).copy()
        rows.append(delta_decode(decoded, axis=-1, dist=1, out=decoded).reshape(width)[x0:x1])
    return np.stack(rows)


def download_box(name, bbox, output):
    """Download the raw and the nucleus mask of a box slice by slice, then write them to <output>/fullres.

    Every slice is saved on its own under a temporary name that is moved in place when complete. Thus, a failed or
    requeued job only downloads the slices that are missing.
    """
    import zarr

    fullres = os.path.join(output, "fullres")
    final_path = os.path.join(fullres, f"{name}.zarr")
    if os.path.exists(final_path):
        return final_path
    x0, x1, y0, y1, z0, z1 = bbox
    slice_dir = os.path.join(fullres, f"{name}_slices")
    os.makedirs(slice_dir, exist_ok=True)
    for z in range(z0, z1):
        slice_path = os.path.join(slice_dir, f"z{z:03d}.npz")
        if os.path.exists(slice_path):
            continue
        raw = read_raw_rows(z, x0, x1, y0, y1)
        nucleus = read_nucleus_mask(z)[y0:y1, x0:x1]
        tmp_path = f"{slice_path[:-4]}.{socket.gethostname()}.{os.getpid()}.npz"
        np.savez(tmp_path, raw=raw, nucleus=nucleus.astype("uint8"))
        os.replace(tmp_path, slice_path)
        print(f"{name}: slice {z} done", flush=True)

    tmp_path = f"{final_path[:-5]}.{socket.gethostname()}.{os.getpid()}.incomplete.zarr"
    group = zarr.open_group(tmp_path, mode="w")
    slices = [np.load(os.path.join(slice_dir, f"z{z:03d}.npz")) for z in range(z0, z1)]
    group.create_array("raw", data=np.stack([s["raw"] for s in slices]), chunks=(16, 512, 512))
    group.create_array("nucleus", data=np.stack([s["nucleus"] for s in slices]), chunks=(16, 512, 512))
    group.attrs["bounding_box"] = list(bbox)
    if os.path.exists(final_path):
        shutil.rmtree(tmp_path)
    else:
        os.replace(tmp_path, final_path)
    return final_path


def save_box_figure(name, raw, labels, out_path, n_slices=4):
    z_ids = np.linspace(0, raw.shape[0] - 1, n_slices).astype(int)
    fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 6.4), squeeze=False)
    for ax, z in zip(axes[0], z_ids):
        overlay = np.zeros(raw.shape[1:] + (4,), dtype="float32")
        overlay[find_boundaries(labels[z])] = (0.0, 0.6, 1.0, 1.0)
        ax.imshow(raw[z], cmap="gray")
        ax.imshow(overlay)
        ax.set_title(f"{name} z={z}: {len(np.unique(labels[z])) - 1} nuclei", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=60)
    plt.close(fig)


def finalize_box(name, role, output):
    import zarr

    source = zarr.open_group(os.path.join(output, "fullres", f"{name}.zarr"), mode="r")
    bbox = list(source.attrs["bounding_box"])
    raw = source["raw"][:].astype("float32")
    nucleus = source["nucleus"][:] > 0
    nz, ny, nx = raw.shape
    raw = raw.reshape(nz, ny // SCALE, SCALE, nx // SCALE, SCALE).mean(axis=(2, 4))
    nucleus = nucleus.reshape(nz, ny // SCALE, SCALE, nx // SCALE, SCALE).mean(axis=(2, 4)) > 0.5
    lo, hi = np.percentile(raw, [0.1, 99.9])
    raw = (np.clip((raw - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
    labels, n_nuclei = ndimage.label(nucleus)

    out_path = os.path.join(output, f"{name}.zarr")
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    group = zarr.open_group(tmp_path, mode="w")
    group.create_array("raw", data=raw, chunks=(16, 256, 256))
    group.create_array("labels", data=labels.astype("uint32"), chunks=(16, 256, 256))
    group.attrs.update({"bounding_box": bbox, "voxel_size_nm_zyx": [55, 32, 32], "role": role})
    # A rerun rebuilds the box: a directory cannot replace a non-empty one.
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)
    save_box_figure(name, raw, labels, os.path.join(FIGURE_DIR, f"{name}_final.png"))
    return {
        "bounding_box_xxyyzz": bbox, "role": role, "shape": list(raw.shape), "n_nuclei": int(n_nuclei),
        "nucleus_fraction": float(nucleus.mean()), "zarr": f"{name}.zarr",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "preview", "download", "finalize"])
    parser.add_argument("--boxes", default=os.path.join(FIGURE_DIR, "boxes.json"), help="The box definitions.")
    parser.add_argument("--names", nargs="+", default=None, help="Restrict 'preview' / 'download' to these boxes.")
    parser.add_argument("--depth", type=int, default=32, help="Number of slices of a box in 'select'.")
    parser.add_argument("--n_best", type=int, default=40, help="Number of windows that 'select' prints.")
    parser.add_argument("--output", default=OUTPUT, help="Output folder of the boxes.")
    args = parser.parse_args()
    os.makedirs(FIGURE_DIR, exist_ok=True)

    if args.mode == "select":
        for cand in select_windows(os.path.join(FIGURE_DIR, "nucleus_map.npz"), args.depth, 10, args.n_best):
            print(cand)
        return

    with open(args.boxes) as f:
        boxes = json.load(f)
    names = args.names or list(boxes)
    if args.mode == "preview":
        windows = []
        for name in names:
            x0, _, y0, _, z0, z1 = boxes[name]["bbox_xxyyzz"]
            windows.extend([(name, z, y0, x0) for z in (z0, (z0 + z1) // 2, z1 - 1)])
        save_preview(windows, os.path.join(FIGURE_DIR, f"preview_{'_'.join(names)}.png"))
    elif args.mode == "download":
        for name in names:
            print(name, download_box(name, boxes[name]["bbox_xxyyzz"], args.output), flush=True)
    else:
        infos = {name: finalize_box(name, boxes[name]["role"], args.output) for name in names}
        for name, info in infos.items():
            print(name, info)
        with open(os.path.join(args.output, "boxes.json"), "w") as f:
            json.dump(infos, f, indent=2)


if __name__ == "__main__":
    main()
