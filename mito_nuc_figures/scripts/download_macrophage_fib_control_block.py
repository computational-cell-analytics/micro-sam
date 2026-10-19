"""Download a contiguous block of slices of the Macrophage / A431 'fib_control' volume with all organelle labels.

The full volume has 2888 slices of 1054 x 5408 pixels (6 x 6 x 12 nm). EMPIAR delivers about 0.5 MB/s per stream.
Thus, the script downloads only a block of slices around z = 911, where the sampled slices hold the most mitochondria
labels. The output file is <data>/macrophage_a431/fib_control_z<start>-<stop>.h5. It holds 'raw' and the binary
'labels/<organelle>' of the torch-em loader. It has a temporary name that is moved in place when complete.
"""
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from torch_em.data.datasets.electron_microscopy import macrophage_a431

from figure_utils import DATA_ROOT


def download_slice(args, n_attempts=8):
    """Download one slice, with retries: the proxy and EMPIAR drop connections under load."""
    for attempt in range(n_attempts):
        try:
            return macrophage_a431._download_slice(args)
        except Exception as error:
            if attempt == n_attempts - 1:
                raise
            print(f"retrying {os.path.basename(args[0])} after {type(error).__name__}", flush=True)
            if os.path.exists(args[1]):
                os.remove(args[1])
            time.sleep(min(300, 10 * 2 ** attempt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=848, help="The first slice (zero-based).")
    parser.add_argument("--stop", type=int, default=976, help="The slice after the last one.")
    parser.add_argument("--n_threads", type=int, default=8, help="Parallel downloads.")
    args = parser.parse_args()
    info = macrophage_a431.VOLUMES["fib_control"]
    out_dir = os.path.join(DATA_ROOT, "macrophage_a431")
    out_path = os.path.join(out_dir, f"fib_control_z{args.start}-{args.stop}.h5")
    tmp_dir = f"{out_path}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    raw_args = macrophage_a431._slice_args(info, "raw", tmp_dir, True)[args.start:args.stop]
    label_args = macrophage_a431._slice_args(info, "label", tmp_dir, True)[args.start:args.stop]
    tmp_path = f"{out_path}.incomplete"
    step = 16
    with h5py.File(tmp_path, "w") as f, ThreadPoolExecutor(args.n_threads) as pool:
        datasets = {}
        for offset in range(0, len(raw_args), step):
            raw = np.stack(list(pool.map(download_slice, raw_args[offset:offset + step])))
            labels = np.stack(list(pool.map(download_slice, label_args[offset:offset + step])))
            assert raw.shape == labels.shape, f"{raw.shape} != {labels.shape}"
            if not datasets:
                shape = (len(raw_args),) + raw.shape[1:]
                chunks = (min(8, shape[0]), 512, 512)
                datasets["raw"] = f.create_dataset(
                    "raw", shape=shape, dtype=raw.dtype, chunks=chunks, compression="gzip"
                )
                for organelle in info["organelles"]:
                    datasets[organelle] = f.create_dataset(
                        f"labels/{organelle}", shape=shape, dtype="uint8", chunks=chunks, compression="gzip"
                    )
            stop = offset + len(raw)
            datasets["raw"][offset:stop] = raw
            for organelle, ids in info["organelles"].items():
                datasets[organelle][offset:stop] = np.isin(labels, ids).astype("uint8")
            print(f"slices {args.start + offset}-{args.start + stop} done", flush=True)
        f.attrs["z_range"] = [args.start, args.stop]
        f.attrs["voxel_size_nm"] = [12, 6, 6]
    os.replace(tmp_path, out_path)
    for name in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, name))
    os.rmdir(tmp_dir)
    print(out_path)


if __name__ == "__main__":
    main()
