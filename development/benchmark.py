"""
Run benchmarks
--------------
1. Install pandas tabulate dependency `python -m pip install tabulate`
2. Run benchmark script, eg: `python benchmark.py --model_type vit_h --device cpu`

Line profiling
--------------
1. Install line profiler: `python -m pip install line_profiler`
2. Add `@profile` decorator to any function in the call stack
3. Run `kernprof -lv benchmark.py --model_type vit_h --device cpu`

Snakeviz visualization
----------------------
https://jiffyclub.github.io/snakeviz/
1. Install snakeviz: `python -m pip install snakeviz`
2. Generate profile file: `python -m cProfile -o program.prof benchmark.py --model_type vit_h --device cpu`
3. Visualize profile file: `snakeviz program.prof`
"""
import argparse
import time

import imageio.v3 as imageio
import micro_sam.instance_segmentation as instance_seg
import micro_sam.prompt_based_segmentation as seg
import micro_sam.util as util
import numpy as np
import pandas as pd

from micro_sam.sample_data import fetch_livecell_example_data


def _get_image_and_predictor(model_type, device):
    example_data = fetch_livecell_example_data("../examples/data")
    image = imageio.imread(example_data)
    predictor = util.get_sam_model(device, model_type)
    return image, predictor


def _add_result(benchmark_results, model_type, device, name, runtimes):
    nres = len(name)
    assert len(name) == len(runtimes)
    res = {
        "model": [model_type] * nres,
        "device": [device] * nres,
        "benchmark": name,
        "runtime": runtimes,
    }
    tab = pd.DataFrame(res)
    benchmark_results.append(tab)
    return benchmark_results


def benchmark_embeddings(image, predictor, n):
    print("Running benchmark_embeddings ...")
    n = 3 if n is None else n
    times = []
    for _ in range(n):
        t0 = time.time()
        util.precompute_image_embeddings(predictor, image)
        times.append(time.time() - t0)
    runtime = np.mean(times)
    return ["embeddings"], [runtime]


def benchmark_prompts(image, predictor, n):
    print("Running benchmark_prompts ...")
    n = 10 if n is None else n
    names, runtimes = [], []

    embeddings = util.precompute_image_embeddings(predictor, image)
    np.random.seed(42)

    names, runtimes = [], []

    # from random single point
    times = []
    for _ in range(n):
        t0 = time.time()
        points = np.array([
            np.random.randint(0, image.shape[0]),
            np.random.randint(0, image.shape[1]),
        ])[None]
        labels = np.array([1])
        seg.segment_from_points(predictor, points, labels, embeddings)
        times.append(time.time() - t0)
    names.append("prompt-p1n0")
    runtimes.append(np.min(times))

    # from random 2p4n
    times = []
    for _ in range(n):
        t0 = time.time()
        points = np.concatenate([
            np.random.randint(0, image.shape[0], size=6)[:, None],
            np.random.randint(0, image.shape[1], size=6)[:, None],
        ], axis=1)
        labels = np.array([1, 1, 0, 0, 0, 0])
        seg.segment_from_points(predictor, points, labels, embeddings)
        times.append(time.time() - t0)
    names.append("prompt-p2n4")
    runtimes.append(np.min(times))

    # from bounding box
    times = []
    for _ in range(n):
        t0 = time.time()
        box_size = np.random.randint(20, 100, size=2)
        box_start = [
            np.random.randint(0, image.shape[0] - box_size[0]),
            np.random.randint(0, image.shape[1] - box_size[1]),
        ]
        box = np.array([
            box_start[0], box_start[1],
            box_start[0] + box_size[0], box_start[1] + box_size[1],
        ])
        seg.segment_from_box(predictor, box, embeddings)
        times.append(time.time() - t0)
    names.append("prompt-box")
    runtimes.append(np.min(times))

    # from bounding box and points
    times = []
    for _ in range(n):
        t0 = time.time()
        points = np.concatenate([
            np.random.randint(0, image.shape[0], size=6)[:, None],
            np.random.randint(0, image.shape[1], size=6)[:, None],
        ], axis=1)
        labels = np.array([1, 1, 0, 0, 0, 0])
        box_size = np.random.randint(20, 100, size=2)
        box_start = [
            np.random.randint(0, image.shape[0] - box_size[0]),
            np.random.randint(0, image.shape[1] - box_size[1]),
        ]
        box = np.array([
            box_start[0], box_start[1],
            box_start[0] + box_size[0], box_start[1] + box_size[1],
        ])
        seg.segment_from_box_and_points(predictor, box, points, labels, embeddings)
        times.append(time.time() - t0)
    names.append("prompt-box-and-points")
    runtimes.append(np.min(times))

    return names, runtimes


def benchmark_amg(image, predictor, n):
    print("Running benchmark_amg ...")
    n = 1 if n is None else n
    embeddings = util.precompute_image_embeddings(predictor, image)
    amg = instance_seg.AutomaticMaskGenerator(predictor)
    times = []
    for _ in range(n):
        t0 = time.time()
        amg.initialize(image, embeddings)
        amg.generate()
        times.append(time.time() - t0)
    runtime = np.mean(times)
    return ["amg"], [runtime]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d",
                        choices=['cpu', 'cuda', 'mps'],
                        help="Which PyTorch backend device to use (REQUIRED)")
    parser.add_argument("--model_type", "-m", default="vit_h",
                        choices=list(util._MODEL_URLS),
                        help="Which deep learning model to use")
    parser.add_argument("--benchmark_embeddings", "-e", action="store_false",
                        help="Skip embedding benchmark test, do not run")
    parser.add_argument("--benchmark_prompts", "-p", action="store_false",
                        help="Skip prompt benchmark test, do not run")
    parser.add_argument("--benchmark_amg", "-a", action="store_false",
                        help="Skip automatic mask generation (amg) benchmark test, do not run")
    parser.add_argument("-n", "--n", type=int, default=None,
                        help="Number of times to repeat benchmark tests")

    args = parser.parse_args()

    model_type = args.model_type
    device = util.get_device(args.device)
    print("Running benchmarks for", model_type)
    print("with device:", device)

    image, predictor = _get_image_and_predictor(model_type, device)

    benchmark_results = []
    if args.benchmark_embeddings:
        name, rt = benchmark_embeddings(image, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, model_type, device, name, rt)

    if args.benchmark_prompts:
        name, rt = benchmark_prompts(image, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, model_type, device, name, rt)

    if args.benchmark_amg:
        name, rt = benchmark_amg(image, predictor, args.n)
        benchmark_results = _add_result(benchmark_results, model_type, device, name, rt)

    benchmark_results = pd.concat(benchmark_results)
    print(benchmark_results.to_markdown(index=False))


if __name__ == "__main__":
    main()
