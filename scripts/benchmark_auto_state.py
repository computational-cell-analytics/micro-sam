"""Benchmark the automatic-segmentation state caching ('auto_state') for SAM2.

For both automatic segmentation modes - AMG (grid-based mask generation) and AIS (UniSAM2
decoder) - and for 2d and 3d data, this compares the time to compute the automatic-segmentation
state from scratch against the time to load a precomputed state from disk. The embeddings are
precomputed once and shared, so the reported speedup isolates the benefit of caching the state.

Run with the finetuned 'hvit_t_cells' SAM2 model (which ships a UniSAM2 decoder) and the bundled
micro-sam sample data. Execute inside the managed environment, e.g.:

    micromamba run -n super python scripts/benchmark_auto_state.py
"""

import os
import shutil
import tempfile
import time

import imageio.v3 as iio
import numpy as np

from micro_sam.sam_annotator._state import _get_sam_model
from micro_sam.v2.util import precompute_image_embeddings
from micro_sam.v2.instance_segmentation import get_amg_segmenter, automatic_3d_segmentation
from micro_sam.precompute_state import cache_amg_state_v2, cache_ais_state_v2, _resolve_unisam2_decoder

MODEL_TYPE = "hvit_t_cells"
SAMPLE_DIR = os.path.expanduser("~/Library/Caches/micro_sam/sample_data")
AMG_PARAMS = dict(points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.9)
AMG_GENERATE = dict(min_object_size=50, with_background=True)


def _timed(fn):
    """Run `fn` and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _report(name, t_compute, t_load, match):
    speedup = (t_compute / t_load) if t_load > 0 else float("inf")
    status = "match" if match else "MISMATCH"
    print(f"{name:10s}  compute {t_compute:8.2f}s   load {t_load:8.3f}s   speedup {speedup:7.1f}x   [{status}]")


def _load_images():
    image = iio.imread(os.path.join(SAMPLE_DIR, "hela-2d-image.png"))
    volume = iio.imread(os.path.join(SAMPLE_DIR, "3d-nucleus-data.tif"))
    # Crop the (18, 1576, 1576) volume to a smaller, representative block to keep the run fast.
    volume = volume[:8, :512, :512]
    return image, volume


def benchmark_amg_2d(image, predictor, save_dir):
    model = getattr(predictor, "model", predictor)
    embeddings = precompute_image_embeddings(predictor, image, save_path=save_dir, ndim=2, verbose=False)
    params = dict(model_type=MODEL_TYPE, **AMG_PARAMS)

    # Warm up the device / kernels and write the state to disk (not timed).
    cache_amg_state_v2(model, image, embeddings, save_dir, verbose=False, **params)

    segmenter_l, t_load = _timed(
        lambda: cache_amg_state_v2(model, image, embeddings, save_dir, verbose=False, **params)
    )
    segmenter_c, t_compute = _timed(
        lambda: cache_amg_state_v2(model, image, embeddings, None, verbose=False, **params)
    )
    seg_l, seg_c = segmenter_l.generate(**AMG_GENERATE), segmenter_c.generate(**AMG_GENERATE)
    _report("AMG 2d", t_compute, t_load, np.array_equal(seg_l, seg_c))


def benchmark_ais_2d(image, predictor, decoder, save_dir):
    device = next(decoder.parameters()).device
    embeddings = precompute_image_embeddings(predictor, image, save_path=save_dir, ndim=2, verbose=False)

    cache_ais_state_v2(decoder, image, embeddings, save_dir, ndim=2, device=device, verbose=False)

    segmenter_l, t_load = _timed(
        lambda: cache_ais_state_v2(decoder, image, embeddings, save_dir, ndim=2, device=device, verbose=False)
    )
    segmenter_c, t_compute = _timed(
        lambda: cache_ais_state_v2(decoder, image, embeddings, None, ndim=2, device=device, verbose=False)
    )
    seg_l, seg_c = segmenter_l.generate(mode="sparse"), segmenter_c.generate(mode="sparse")
    _report("AIS 2d", t_compute, t_load, np.array_equal(seg_l, seg_c))


def benchmark_ais_3d(volume, predictor, decoder, save_dir):
    device = next(decoder.parameters()).device
    embeddings = precompute_image_embeddings(predictor, volume, save_path=save_dir, ndim=3, verbose=False)

    cache_ais_state_v2(decoder, volume, embeddings, save_dir, ndim=3, device=device, verbose=False)

    segmenter_l, t_load = _timed(
        lambda: cache_ais_state_v2(decoder, volume, embeddings, save_dir, ndim=3, device=device, verbose=False)
    )
    segmenter_c, t_compute = _timed(
        lambda: cache_ais_state_v2(decoder, volume, embeddings, None, ndim=3, device=device, verbose=False)
    )
    seg_l, seg_c = segmenter_l.generate(mode="sparse"), segmenter_c.generate(mode="sparse")
    _report("AIS 3d", t_compute, t_load, np.array_equal(seg_l, seg_c))


def benchmark_amg_3d(volume, predictor, save_dir):
    model = getattr(predictor, "model", predictor)
    embeddings = precompute_image_embeddings(predictor, volume, save_path=save_dir, ndim=3, verbose=False)

    def run(state_save_path):
        segmenter = get_amg_segmenter(model, model_type=MODEL_TYPE, **AMG_PARAMS)
        return automatic_3d_segmentation(
            volume, segmenter, image_embeddings=embeddings, state_save_path=state_save_path,
            verbose=False, **AMG_GENERATE,
        )

    run(save_dir)  # warm up and write the per-slice state to disk (not timed).
    seg_l, t_load = _timed(lambda: run(save_dir))
    seg_c, t_compute = _timed(lambda: run(None))
    _report("AMG 3d", t_compute, t_load, np.array_equal(seg_l, seg_c))


def main():
    image, volume = _load_images()
    print(f"2d image {image.shape}, 3d volume {volume.shape}, model '{MODEL_TYPE}'\n")

    predictor_2d, _ = _get_sam_model(MODEL_TYPE, 2, None, None, None, True)
    predictor_3d, _ = _get_sam_model(MODEL_TYPE, 3, None, None, None, True)
    decoder = _resolve_unisam2_decoder(MODEL_TYPE, None, None)
    if decoder is None:
        raise RuntimeError(f"Could not load a UniSAM2 decoder for '{MODEL_TYPE}'; AIS benchmarks need it.")

    tmp = tempfile.mkdtemp(prefix="auto_state_bench_")
    try:
        benchmark_amg_2d(image, predictor_2d, os.path.join(tmp, "amg_2d.zarr"))
        benchmark_ais_2d(image, predictor_2d, decoder, os.path.join(tmp, "ais_2d.zarr"))
        benchmark_ais_3d(volume, predictor_3d, decoder, os.path.join(tmp, "ais_3d.zarr"))
        benchmark_amg_3d(volume, predictor_3d, os.path.join(tmp, "amg_3d.zarr"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
