import os
import time
import argparse
import statistics

import numpy as np
from skimage.draw import disk

import torch

from micro_sam.v1.util import precompute_image_embeddings
from micro_sam.v1.instance_segmentation import InstanceSegmentationWithDecoder, get_predictor_and_decoder


def get_image(shape=(256, 256)):
    image = np.zeros(shape, dtype="uint8")
    for center, radius in [((64, 64), 29), ((128, 128), 33), ((192, 192), 35)]:
        image[disk(center, radius, shape=shape)] = 255
    return image


def time_call(fn, n_repeats):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def baseline_speed(n_repeats):
    # Single-threaded reference that does not depend on torch or OpenMP.
    def python_loop():
        x = 0
        for i in range(5_000_000):
            x += i * i
        return x

    a = np.random.rand(1024, 1024).astype("float32")
    t_py = time_call(python_loop, n_repeats)
    torch.set_num_threads(1)
    ta = torch.from_numpy(a)
    t_mm = time_call(lambda: ta @ ta, n_repeats)
    print(f"baseline: python loop {t_py[0]:.3f} s (min {t_py[1]:.3f}, max {t_py[2]:.3f})")
    print(f"baseline: torch matmul 1 thread {t_mm[0]:.3f} s (min {t_mm[1]:.3f}, max {t_mm[2]:.3f})")


def benchmark_model(model_type, n_threads_list, n_repeats):
    image = get_image()
    predictor, decoder = get_predictor_and_decoder(model_type=model_type, device="cpu")
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)
    embeddings = precompute_image_embeddings(predictor, image, verbose=False)

    for n_threads in n_threads_list:
        torch.set_num_threads(n_threads)
        t_enc = time_call(lambda: precompute_image_embeddings(predictor, image, verbose=False), n_repeats)
        t_dec = time_call(lambda: segmenter.initialize(image, image_embeddings=embeddings), n_repeats)
        print(
            f"{model_type} threads={n_threads}: encoder {t_enc[0]:.2f} s (min {t_enc[1]:.2f}, max {t_enc[2]:.2f}), "
            f"decoder {t_dec[0]:.2f} s (min {t_dec[1]:.2f}, max {t_dec[2]:.2f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="vit_t_lm")
    parser.add_argument("--n_repeats", type=int, default=3)
    args = parser.parse_args()

    print(f"torch {torch.__version__}, default threads {torch.get_num_threads()}, os.cpu_count {os.cpu_count()}")
    print(f"mps available {torch.backends.mps.is_available()}, mps built {torch.backends.mps.is_built()}")
    print(f"OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY')}, KMP_BLOCKTIME={os.environ.get('KMP_BLOCKTIME')}")
    print(torch.__config__.parallel_info())

    n_cpus = os.cpu_count()
    baseline_speed(args.n_repeats)
    benchmark_model(args.model_type, sorted({1, 2, n_cpus}), args.n_repeats)
    baseline_speed(args.n_repeats)


if __name__ == "__main__":
    main()
