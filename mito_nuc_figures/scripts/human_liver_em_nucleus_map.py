"""Map where the human liver EM nucleus labels are, to place the nucleus evaluation boxes.

The nucleus mask PNG of every n-th slice (about 0.85 MB each) is reduced to the nucleus fraction per block of
BLOCK x BLOCK pixels. The maps of all sampled slices are saved to one npz file, together with the sampled z indices.
"""
import io
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
from PIL import Image

from figure_utils import get_output_dir


EMPIAR_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/13356/data"
NUCLEUS_DIR = "humanliver_nucleus_mask"
N_SLICES = 597
BLOCK = 128


def fetch_mask(z):
    Image.MAX_IMAGE_PIXELS = None
    url = f"{EMPIAR_BASE}/{NUCLEUS_DIR}/{NUCLEUS_DIR}_{z:03d}.png"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 4:
                raise
    mask = np.array(Image.open(io.BytesIO(r.content)))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def block_fraction(mask):
    ny, nx = mask.shape[0] // BLOCK, mask.shape[1] // BLOCK
    return mask[:ny * BLOCK, :nx * BLOCK].reshape(ny, BLOCK, nx, BLOCK).mean(axis=(1, 3)).astype("float32")


def map_slice(z):
    fraction = block_fraction(fetch_mask(z))
    print(f"z={z}: nucleus fraction {fraction.mean():.4f}", flush=True)
    return fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=10, help="Map every n-th slice.")
    parser.add_argument("--output", default=None, help="The npz file. Defaults to the human_liver_em figure folder.")
    args = parser.parse_args()
    out_dir = os.path.join(get_output_dir("human_liver_em"), "nucleus_boxes")
    os.makedirs(out_dir, exist_ok=True)
    output = args.output or os.path.join(out_dir, "nucleus_map.npz")
    z_ids = list(range(0, N_SLICES, args.step))
    with ThreadPoolExecutor(4) as pool:
        maps = list(pool.map(map_slice, z_ids))
    np.savez_compressed(output, maps=np.stack(maps), z=np.array(z_ids), block=BLOCK)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
