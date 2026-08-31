"""Batched, pipelined, multi-GPU SAM2 inference: scheduling engine, encoder embeddings, and decoder passes."""

import gc
import os
import time
import queue
import warnings
import threading
import contextlib
from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import torch

from .util import Devices, autocast, recommend_batch_size, to_float32
from micro_sam.util import _create_dataset_without_data
from .normalization import IMAGE_PREPROCESSING, VIDEO_PREPROCESSING, compute_percentile_bounds, to_image


STOP = object()

# Slices sampled to estimate whole-volume normalization statistics, so a lazy volume (dask / zarr /
# h5py) is never materialized all at once just to compute them.
NORMALIZATION_SAMPLE_SLICES = 32


def _volume_normalization_bounds(input_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile bounds from a sample of z slices spanning the whole volume.

    Every slice or tile the volume is later split into is normalized with these bounds (see
    `_compute_3d` / `_compute_tiled_3d`), so they all share one normalization instead of each
    estimating its own from a smaller, biased crop.
    """
    n_slices = int(input_.shape[0])
    step = max(1, n_slices // NORMALIZATION_SAMPLE_SLICES)
    sample = np.stack([np.asarray(input_[z]) for z in range(0, n_slices, step)])
    if sample.ndim == 4:
        bounds = compute_percentile_bounds(sample, axis=(0, 1, 2))
        return tuple(bound[0] for bound in bounds)
    return compute_percentile_bounds(sample)


class _PipelineAborted(Exception):
    """Raised inside workers when another pipeline worker fails."""


class _AtomicCounter:
    """Lock-guarded counter used to send completion sentinels exactly once."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.lock = threading.Lock()

    def decrement(self) -> int:
        with self.lock:
            self.value -= 1
            return self.value


@dataclass
class _PipelineJob:
    """A work item moving from the input loader to inference and output writing."""

    spec: Any
    data: Any


def _normalize_device(device: Union[str, torch.device]) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return _normalize_device(next(model.parameters()).device)
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _all_cuda_devices() -> List[torch.device]:
    return [torch.device("cuda", index) for index in range(torch.cuda.device_count())]


def _resolve_devices(model: torch.nn.Module, devices: Devices = None) -> List[torch.device]:
    """Resolve the inference devices, using every visible CUDA device by default.

    Only `devices=None` fans out. Anything given explicitly stays on the one device it names, so a bare
    'cuda' resolves to the current CUDA device and inference never allocates on a GPU the caller did not
    select. The GUI passes None for its 'auto' entry and lists the visible GPUs individually, so both
    intents are reachable from it.

    Automatic multi-GPU execution is enabled only when the supplied model already lives on CUDA.
    This preserves an explicitly CPU- or MPS-loaded model.

    Args:
        model: The model whose device decides the default. Only used when `devices` is None.
        devices: A single device to force one device, or a sequence to select an explicit set.
            By default all visible CUDA devices are used if the model is on CUDA.

    Returns:
        The resolved devices, in the order they will be assigned to inference workers.

    Raises:
        ValueError: If no device is given, or if the same device is given more than once.
        RuntimeError: If a CUDA device is requested but CUDA is unavailable.
    """
    if devices is None:
        device = _model_device(model)
        resolved = _all_cuda_devices() if device.type == "cuda" and torch.cuda.device_count() > 1 else [device]
    elif isinstance(devices, (str, torch.device)):
        resolved = [_normalize_device(devices)]
    else:
        resolved = [_normalize_device(device) for device in devices]

    if len(resolved) == 0:
        raise ValueError("At least one inference device is required.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Inference devices must be unique, got {resolved}.")
    if any(device.type == "cuda" for device in resolved) and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested, but PyTorch CUDA support is unavailable.")
    return resolved


def _prepare_models(
    model: torch.nn.Module, devices: Sequence[torch.device],
) -> List[Tuple[torch.nn.Module, torch.device]]:
    """Create one eval-mode model replica per device, reusing the original where possible.

    Replicas are deep-copied from a CPU copy of the model rather than from the source-device model,
    so a second full copy is never allocated on the source GPU (which could OOM during setup). This
    is safe because `_prepare_models` runs only at synchronous pipeline setup, with no concurrent use.

    Args:
        model: The model to replicate.
        devices: The devices to place the replicas on (see `_resolve_devices`).

    Returns:
        One (model, device) pair per device. The pair for the model's own device holds the original
        model, all others hold a deep copy. Release them with `_release_model_replicas`.
    """
    source_device = _model_device(model)
    replicas = {}
    other_devices = [device for device in devices if device != source_device]
    if other_devices:
        # Move the original to CPU (in place) so each deepcopy lands in host RAM, then restore it.
        model.to("cpu")
        try:
            for device in other_devices:
                replicas[device] = deepcopy(model).to(device)
        finally:
            model.to(source_device)

    models = []
    for device in devices:
        replica = model if device == source_device else replicas[device]
        if hasattr(replica, "eval"):
            replica.eval()
        models.append((replica, device))
    return models


def _release_model_replicas(model_devices: List[Tuple[torch.nn.Module, torch.device]]) -> None:
    """Drop the replicas built by `_prepare_models` and free their GPU memory.

    Clearing the list releases only the per-device copies; the caller's own model stays alive
    through its other references.

    Args:
        model_devices: The (model, device) pairs returned by `_prepare_models`. Cleared in place.
    """
    model_devices.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom_error(exc: Exception) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def _release_probe_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _release_fragmented_cuda_cache(device: torch.device) -> None:
    """Release inactive cache when it leaves the logical CUDA device with too little free VRAM."""
    if device.type != "cuda":
        return
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    inactive_memory = torch.cuda.memory_reserved(device) - allocated_memory
    if free_memory < 0.25 * total_memory and inactive_memory > 0.25 * total_memory:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _measure_batch_throughput(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    patch_shape: Tuple[int, ...],
    in_channels: int,
    prediction_function: Callable[[torch.nn.Module, torch.Tensor], Any],
    n_repeats: int = 2,
) -> Optional[float]:
    """Measure samples per second for one synthetic batch, returning None on CUDA OOM."""
    inputs = output = None
    was_training = model.training
    model.eval()
    try:
        inputs = torch.empty(
            (batch_size, in_channels, *patch_shape), dtype=torch.float32, device=device,
        ).normal_()
        timings = []
        for repeat in range(n_repeats + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                output = prediction_function(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if repeat > 0:
                timings.append(time.perf_counter() - start)
            del output
            output = None
        return float(batch_size) / float(np.median(timings))
    except Exception as exc:
        if _is_oom_error(exc):
            return None
        raise
    finally:
        model.train(was_training)
        del inputs, output
        _release_probe_memory(device)


def _select_throughput_batch_size(
    measurements: Sequence[Tuple[int, float]], min_relative_improvement: float = 0.1,
) -> int:
    """Prefer the smallest batch unless a larger one improves throughput materially."""
    if not measurements:
        raise ValueError("At least one batch-throughput measurement is required.")
    selected_batch, selected_throughput = measurements[0]
    for batch_size, throughput in measurements[1:]:
        if throughput >= selected_throughput * (1.0 + min_relative_improvement):
            selected_batch, selected_throughput = batch_size, throughput
    return int(selected_batch)


def _compute_auto_batch_sizes(
    model_devices: Sequence[Tuple[torch.nn.Module, torch.device]],
    n_jobs: int,
    patch_shape: Tuple[int, ...],
    in_channels: int,
    prediction_function: Callable[[torch.nn.Module, torch.Tensor], Any],
) -> List[int]:
    """Select a throughput-efficient batch size independently on every CUDA device.

    Powers of two are benchmarked from one upwards. A larger batch is selected only when it improves
    measured throughput by at least 10 percent, and probing stops after two consecutive candidates
    fail to do so. CPU and MPS retain the conservative batch size of one.

    Args:
        model_devices: The (model, device) pairs to benchmark (see `_prepare_models`).
        n_jobs: The total number of jobs; caps the candidates so a batch never exceeds the work.
        patch_shape: The spatial shape of a single input to the model.
        in_channels: The number of channels of a single input to the model.
        prediction_function: Called as `prediction_function(model, inputs)` to run one probe batch.

    Returns:
        One batch size per entry in `model_devices`.

    Raises:
        RuntimeError: If a device runs out of memory already at batch size one.
    """
    if n_jobs < 1:
        return [1] * len(model_devices)

    jobs_per_device = (int(n_jobs) + len(model_devices) - 1) // len(model_devices)
    upper_bound = min(256, jobs_per_device)
    batch_sizes = []
    for model, device in model_devices:
        if device.type != "cuda":
            batch_sizes.append(1)
            continue

        measurements = []
        non_improving_candidates = 0
        candidate = 1
        while candidate <= upper_bound:
            previous_selection = (
                _select_throughput_batch_size(measurements) if measurements else None
            )
            throughput = _measure_batch_throughput(
                model=model,
                device=device,
                batch_size=candidate,
                patch_shape=patch_shape,
                in_channels=in_channels,
                prediction_function=prediction_function,
            )
            if throughput is None:
                break
            measurements.append((candidate, throughput))
            selection = _select_throughput_batch_size(measurements)
            if previous_selection is not None:
                if selection == candidate:
                    non_improving_candidates = 0
                else:
                    non_improving_candidates += 1
            if non_improving_candidates >= 2:
                break
            candidate *= 2

        if not measurements:
            raise RuntimeError(
                f"The model does not fit batch size 1 on {device}. Reduce the patch shape or use a smaller model."
            )
        batch_sizes.append(_select_throughput_batch_size(measurements))
    return batch_sizes


def _safe_get(input_queue: queue.Queue, stop_event: threading.Event, timeout: float = 0.2) -> Any:
    while not stop_event.is_set():
        try:
            return input_queue.get(timeout=timeout)
        except queue.Empty:
            continue
    raise _PipelineAborted()


def _safe_put(output_queue: queue.Queue, item: Any, stop_event: threading.Event, timeout: float = 0.2) -> None:
    while not stop_event.is_set():
        try:
            output_queue.put(item, timeout=timeout)
            return
        except queue.Full:
            continue
    raise _PipelineAborted()


def _predict_with_oom_backoff(
    model: torch.nn.Module,
    items: List[Any],
    device: torch.device,
    predict_fn: Callable[[torch.nn.Module, List[Any], torch.device], List[Any]],
) -> Tuple[List[Any], int]:
    """Retry an OOMing batch in halves and return the largest size proven safe."""
    try:
        return predict_fn(model, items, device), len(items)
    except Exception as exc:
        if not _is_oom_error(exc):
            raise

    _release_probe_memory(device)
    if len(items) == 1:
        return predict_fn(model, items, device), 1
    split = len(items) // 2
    warnings.warn(
        f"Batch size {len(items)} ran out of memory on {device}; retrying with smaller batches.",
        RuntimeWarning,
        stacklevel=2,
    )
    left, left_size = _predict_with_oom_backoff(model, items[:split], device, predict_fn)
    right, right_size = _predict_with_oom_backoff(model, items[split:], device, predict_fn)
    return left + right, min(left_size, right_size)


def _run_batched_pipeline(
    jobs: Iterable[Any],
    model_devices: Sequence[Tuple[torch.nn.Module, torch.device]],
    batch_sizes: Sequence[int],
    load_fn: Callable[[Any], Any],
    predict_fn: Callable[[torch.nn.Module, List[Any], torch.device], List[Any]],
    write_fn: Callable[[Any, Any], None],
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
    update_progress: Optional[Callable[[int], None]] = None,
    progress_increment: Optional[Callable[[Any], int]] = None,
) -> None:
    """Run load/preprocess, batched inference, and writing as a bounded threaded pipeline.

    Loading runs on `num_prefetch_workers` threads, inference on one thread per device, and writing
    on `num_write_workers` threads, so input I/O and output I/O overlap with the model forward pass.
    The queues between the stages are bounded, so a slow stage throttles the others instead of
    growing memory. Jobs are written in completion order, not in input order. If any worker raises,
    the whole pipeline is stopped and the first error is re-raised on the calling thread.

    Args:
        jobs: The job specifications, e.g. tile ids or slice indices. Passed to `load_fn` and, with
            the prediction, to `write_fn`.
        model_devices: The (model, device) pairs to run inference on (see `_prepare_models`).
        batch_sizes: One batch size per entry in `model_devices` (see `_prepare_encoder_pipeline` for
            the encoder and `_compute_auto_batch_sizes` for the decoder).
        load_fn: Called as `load_fn(job)` to read and preprocess one job.
        predict_fn: Called as `predict_fn(model, items, device)` with the loaded data of one batch.
            It must return one output per input. Batches that run out of memory are automatically
            retried in halves, and the device's batch size is reduced accordingly.
        write_fn: Called as `write_fn(job, prediction)` to store one result. It must be safe to call
            from multiple threads if `num_write_workers` is larger than one.
        num_prefetch_workers: The number of threads used to read and preprocess jobs.
        num_write_workers: The number of threads used to write results. Multiple writers speed up
            compressed / chunked outputs; keep it at one for outputs that are not thread-safe.
        update_progress: Optional callback advancing external progress by the given number of steps.
            It is called on the calling thread, so it is safe to use for Qt / napari progress bars.
        progress_increment: Optional callback returning the number of steps a job contributes.
            By default every job counts as one step.

    Raises:
        ValueError: If the batch sizes do not match the devices, or are not positive.
    """
    jobs = list(jobs)
    if len(jobs) == 0:
        return
    if len(model_devices) != len(batch_sizes):
        raise ValueError("Expected one batch size for every model/device pair.")
    if any(int(batch_size) < 1 for batch_size in batch_sizes):
        raise ValueError(f"Batch sizes must be positive, got {batch_sizes}.")

    num_prefetch_workers = max(1, min(int(num_prefetch_workers), len(jobs)))
    num_write_workers = max(1, min(int(num_write_workers), len(jobs)))
    batch_sizes = [int(batch_size) for batch_size in batch_sizes]

    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in range(num_prefetch_workers):
        job_queue.put(STOP)

    input_queue = queue.Queue(maxsize=max(2 * sum(batch_sizes), 2))
    # Sized to hold a full batch from every device: a consumer that cannot hand off all of its
    # predictions blocks mid-batch and idles the GPU until the writers catch up.
    output_queue = queue.Queue(maxsize=max(2 * sum(batch_sizes), 2 * num_write_workers, 2))
    progress_queue = queue.Queue()
    stop_event = threading.Event()
    error_box = []
    error_lock = threading.Lock()
    remaining_producers = _AtomicCounter(num_prefetch_workers)
    remaining_consumers = _AtomicCounter(len(model_devices))

    def record_error(exc):
        with error_lock:
            if not error_box:
                error_box.append(exc)
        stop_event.set()

    def producer():
        try:
            while True:
                spec = job_queue.get()
                if spec is STOP or stop_event.is_set():
                    break
                _safe_put(input_queue, _PipelineJob(spec, load_fn(spec)), stop_event)
        except _PipelineAborted:
            pass
        except Exception as exc:  # noqa
            record_error(exc)
        finally:
            # A timed put: a consumer that fails while the queue is full leaves nobody to drain it,
            # and a blocking put would then hang this thread (and the join in the outer cleanup).
            if remaining_producers.decrement() == 0:
                with contextlib.suppress(_PipelineAborted):
                    for _ in model_devices:
                        _safe_put(input_queue, STOP, stop_event)

    def consumer(worker_id):
        model, device = model_devices[worker_id]
        batch_size = batch_sizes[worker_id]
        try:
            while True:
                batch = []
                got_stop = False
                while len(batch) < batch_size:
                    item = _safe_get(input_queue, stop_event)
                    if item is STOP:
                        got_stop = True
                        break
                    batch.append(item)

                if batch:
                    with torch.no_grad():
                        predictions, safe_batch_size = _predict_with_oom_backoff(
                            model, [item.data for item in batch], device, predict_fn,
                        )
                    batch_size = min(batch_size, safe_batch_size)
                    if len(predictions) != len(batch):
                        raise RuntimeError(
                            f"The batch predictor returned {len(predictions)} outputs for {len(batch)} inputs."
                        )
                    for item, prediction in zip(batch, predictions):
                        item.data = prediction
                        _safe_put(output_queue, item, stop_event)

                if got_stop:
                    break
        except _PipelineAborted:
            pass
        except Exception as exc:  # noqa
            record_error(exc)
        finally:
            # Timed as well, for the same reason: the writers can already have failed and stopped.
            if remaining_consumers.decrement() == 0:
                with contextlib.suppress(_PipelineAborted):
                    for _ in range(num_write_workers):
                        _safe_put(output_queue, STOP, stop_event)

    def writer():
        try:
            while True:
                item = _safe_get(output_queue, stop_event)
                if item is STOP:
                    break
                write_fn(item.spec, item.data)
                if update_progress is not None:
                    increment = 1 if progress_increment is None else int(progress_increment(item.spec))
                    progress_queue.put(increment)
        except _PipelineAborted:
            pass
        except Exception as exc:  # noqa
            record_error(exc)

    writer_threads = [
        threading.Thread(target=writer, name=f"sam2-writer-{worker_id}")
        for worker_id in range(num_write_workers)
    ]
    consumer_threads = [
        threading.Thread(target=consumer, args=(worker_id,), name=f"sam2-consumer-{worker_id}")
        for worker_id in range(len(model_devices))
    ]
    producer_threads = [
        threading.Thread(target=producer, name=f"sam2-producer-{worker_id}")
        for worker_id in range(num_prefetch_workers)
    ]
    threads = [*writer_threads, *consumer_threads, *producer_threads]

    def forward_progress(block):
        increments = 0
        try:
            increments = progress_queue.get(timeout=0.05 if block else 0)
        except queue.Empty:
            return
        while True:
            try:
                increments += progress_queue.get_nowait()
            except queue.Empty:
                break
        update_progress(increments)

    try:
        for thread in threads:
            thread.start()
        if update_progress is None:
            for thread in threads:
                thread.join()
        else:
            # The writer performs I/O off-thread, but UI progress callbacks must run on the calling
            # thread. Poll completed writes here instead of queuing Qt signals behind this wait.
            while any(thread.is_alive() for thread in threads):
                forward_progress(block=True)
            for thread in threads:
                thread.join()
            forward_progress(block=False)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    if error_box:
        raise error_box[0]


def _clear_group(group) -> None:
    for key in list(group.keys()):
        del group[key]


def _embedding_cache_complete(features, root) -> bool:
    return bool(features.attrs.get("complete", "input_size" in root.attrs))


def _prepare_encoder_pipeline(
    predictor, n_jobs: int, batch_size: Optional[int], devices: Devices,
) -> Tuple[torch.nn.Module, List[Tuple[torch.nn.Module, torch.device]], List[int]]:
    # Validated before the replicas are created, so an invalid argument never allocates on a GPU.
    if batch_size is not None and int(batch_size) < 1:
        raise ValueError(f"batch_size must be positive or None, got {batch_size}.")

    model = getattr(predictor, "model", predictor)
    resolved_devices = _resolve_devices(model, devices)
    model_devices = _prepare_models(model, resolved_devices)
    try:
        if batch_size is None:
            # Read after the replicas are placed, so their weights already count against the free VRAM.
            model_type = getattr(model, "model_type", None) or getattr(predictor, "model_type", "")
            # Cap by the share of the work a device receives, not by the total: a consumer fills its
            # batch before it runs, so a batch as large as all jobs would keep the other devices idle.
            jobs_per_device = (max(int(n_jobs), 0) + len(model_devices) - 1) // len(model_devices)
            batch_sizes = [
                recommend_batch_size(model_type, device, n_jobs=jobs_per_device) for _, device in model_devices
            ]
        else:
            batch_sizes = [int(batch_size)] * len(model_devices)
    except Exception:
        _release_model_replicas(model_devices)
        raise
    return model, model_devices, batch_sizes


def _forward_image_batch(
    model: torch.nn.Module, items: List[Dict], device: torch.device, feature_sizes: Sequence,
) -> List[Dict]:
    batch = torch.stack([item["tensor"] for item in items]).to(device, non_blocking=True)
    # The embedding cache is fp32 and numpy has no bfloat16.
    with autocast(device):
        backbone_out = to_float32(model.forward_image(batch))
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    batch_size = len(items)
    features = [
        feat.permute(1, 2, 0).reshape(batch_size, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], feature_sizes[::-1])
    ][::-1]
    features = [feature.detach().cpu().numpy() for feature in features]
    return [
        {
            "features": features[-1][index:index + 1],
            "high_res_feats": [feature[index:index + 1] for feature in features[:-1]],
            "original_size": items[index]["original_size"],
        }
        for index in range(batch_size)
    ]


def _compute_tiled_2d(
    input_: np.ndarray,
    predictor,
    tile_shape: Tuple[int, int],
    halo: Tuple[int, int],
    root,
    save_path: Optional[Union[str, os.PathLike]],
    pbar_init: Callable,
    pbar_update: Callable,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
) -> Dict:
    """Compute 2d tile embeddings with batched encoders and queued input / output I/O.

    Args:
        input_: The input image, shape (Y, X) or (Y, X, C).
        predictor: The SAM2 image predictor.
        tile_shape: The in-plane tile shape.
        halo: The in-plane tile halo (overlap).
        root: The zarr container the embeddings are written to (see `micro_sam.util._open_embeddings`).
        save_path: The path backing `root`, or None for an in-memory container. A complete cache at
            this path is returned as is instead of being recomputed.
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        batch_size: The number of tiles per encoder call. By default it is looked up from the
            free VRAM of each CUDA device (see `util.recommend_batch_size`).
        devices: The device or devices to run inference on (see `_resolve_devices`).
        num_prefetch_workers: The number of threads used to read and preprocess tiles.
        num_write_workers: The number of threads used to write embedding tiles.

    Returns:
        The tiled image embeddings. 'features' and 'high_res_feats' are the zarr groups holding the
        per-tile datasets; 'input_size' and 'original_size' are None because they are stored per tile.
    """
    from bioimage_cpp.utils import Blocking
    from micro_sam.v2.util import _write_embedding_signature

    features = root.require_group("features")
    high_res_group = root.require_group("high_res_feats")
    if save_path is not None and "shape" in features.attrs and _embedding_cache_complete(features, root):
        return {"features": features, "high_res_feats": high_res_group, "input_size": None, "original_size": None}

    _clear_group(features)
    _clear_group(high_res_group)

    tiling = Blocking([0, 0], list(input_.shape[:2]), list(tile_shape))
    n_tiles = tiling.number_of_blocks
    features.attrs["shape"] = list(input_.shape[:2])
    features.attrs["tile_shape"] = list(tile_shape)
    features.attrs["halo"] = list(halo)
    features.attrs["complete"] = False

    pbar_init(n_tiles, "Compute Image Embeddings 2D tiled")
    model, model_devices, batch_sizes = _prepare_encoder_pipeline(predictor, n_tiles, batch_size, devices)
    feature_sizes = predictor._bb_feat_sizes

    def load_tile(tile_id):
        block = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
        bb = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))
        image = to_image(np.asarray(input_[bb]))
        return {"tensor": predictor._transforms(image), "original_size": tuple(image.shape[:2])}

    def predict_tiles(this_model, items, device):
        return _forward_image_batch(this_model, items, device, feature_sizes)

    # Creating a dataset mutates the shared group, so it is serialized. The data (and with it the
    # compression) is written outside the lock, which is what multiple write workers speed up.
    creation_lock = threading.Lock()

    def write_tile(tile_id, result):
        name = str(tile_id)
        tile_features = result["features"]
        high_res_feats = result["high_res_feats"]
        with creation_lock:
            dataset = _create_dataset_without_data(
                features, name, shape=tile_features.shape, dtype=tile_features.dtype, chunks=tile_features.shape,
            )
            dataset.attrs["input_size"] = model.image_size
            dataset.attrs["original_size"] = [list(result["original_size"])]
            tile_high_res = high_res_group.require_group(name)
            high_res_datasets = [
                _create_dataset_without_data(
                    tile_high_res, str(level), shape=feature.shape, dtype=feature.dtype, chunks=feature.shape,
                ) for level, feature in enumerate(high_res_feats)
            ]

        dataset[:] = tile_features
        for high_res_dataset, feature in zip(high_res_datasets, high_res_feats):
            high_res_dataset[:] = feature

    try:
        _run_batched_pipeline(
            jobs=range(n_tiles),
            model_devices=model_devices,
            batch_sizes=batch_sizes,
            load_fn=load_tile,
            predict_fn=predict_tiles,
            write_fn=write_tile,
            num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
            update_progress=pbar_update,
        )
    finally:
        _release_model_replicas(model_devices)

    if save_path is not None:
        _write_embedding_signature(
            root, input_, predictor, tile_shape=tile_shape, halo=halo, input_size=None,
            original_size=None, preprocessing=IMAGE_PREPROCESSING,
        )
    features.attrs["complete"] = True
    return {"features": features, "high_res_feats": high_res_group, "input_size": None, "original_size": None}


def _prepare_video_frame(raw: np.ndarray, image_size: int, bounds=None) -> torch.Tensor:
    from micro_sam.v2.models._video_predictor import _prepare_frame

    return _prepare_frame(np.asarray(raw), image_size, bounds=bounds)


def _forward_video_batch(model: torch.nn.Module, items: List[Dict], device: torch.device) -> List[Dict]:
    batch = torch.stack([item["tensor"] for item in items]).to(device, non_blocking=True)
    with autocast(device):
        backbone_out = to_float32(model.forward_image(batch))

    vision_features = backbone_out["vision_features"].detach().cpu().numpy()
    # Positional encodings depend only on the input shape, so every batch element is identical.
    pos_enc = [value[:1].detach().cpu().numpy() for value in backbone_out["vision_pos_enc"]]
    # The last FPN level is the same tensor as 'vision_features', so it is not copied or stored twice.
    fpn = [value.detach().cpu().numpy() for value in backbone_out["backbone_fpn"][:-1]]
    return [
        {
            "features": vision_features[index:index + 1],
            "pos_enc": pos_enc,
            "fpn": [value[index:index + 1] for value in fpn],
            "original_size": items[index]["original_size"],
        }
        for index in range(len(items))
    ]


def _create_feature_dataset(group, name: str, n_slices: int, tensor: np.ndarray):
    shape = (n_slices,) + tuple(tensor.shape)
    chunks = (1,) + tuple(tensor.shape)
    return _create_dataset_without_data(group, name, shape=shape, dtype="float32", chunks=chunks)


def _check_pos_enc_shapes(pos_enc: Sequence[np.ndarray], reference: Sequence) -> None:
    """Verify a slice's positional encodings match the single stored copy.

    Args:
        pos_enc: The positional encodings of the slice being written, each (1, C, H, W).
        reference: The stored encodings, either the datasets or the in-memory tensors.

    Raises:
        RuntimeError: If the number of levels or any (C, H, W) differs from the stored copy.
    """
    expected = [tuple(value.shape[-3:]) for value in reference]
    actual = [tuple(value.shape[-3:]) for value in pos_enc]
    if actual != expected:
        raise RuntimeError(
            f"The positional encodings differ between slices, got {actual} but stored {expected}. "
            "A single stored copy is only valid if every frame is encoded at the same resolution."
        )


def _create_feature_levels(group, n_slices: int, tensors: Sequence[np.ndarray]) -> List:
    return [
        _create_feature_dataset(group, str(level), n_slices, tensor)
        for level, tensor in enumerate(tensors)
    ]


def _load_feature_levels(group, lazy_loading: bool) -> List:
    values = []
    level = 0
    while str(level) in group:
        dataset = group[str(level)]
        values.append(dataset if lazy_loading else dataset[:])
        level += 1
    return values


def _compute_3d(
    input_: np.ndarray,
    predictor,
    root,
    save_path: Optional[Union[str, os.PathLike]],
    lazy_loading: bool,
    pbar_init: Callable,
    pbar_update: Callable,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
    norm_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict:
    """Compute volume embeddings by batching slices and overlapping preprocessing and zarr writes.

    Args:
        input_: The input volume, shape (Z, Y, X).
        predictor: The SAM2 video predictor.
        root: The zarr container the embeddings are written to (see `micro_sam.util._open_embeddings`).
        save_path: The path backing `root`, or None to keep the embeddings in memory. A complete cache
            at this path is returned as is instead of being recomputed.
        lazy_loading: Whether to return the zarr datasets instead of materializing the embeddings.
            Only has an effect if `save_path` is given.
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        batch_size: The number of slices per encoder call. By default it is looked up from the
            free VRAM of each CUDA device (see `util.recommend_batch_size`).
        devices: The device or devices to run inference on (see `_resolve_devices`).
        num_prefetch_workers: The number of threads used to read and preprocess slices.
        num_write_workers: The number of threads used to write embedding slices.
        norm_bounds: Precomputed (lower, upper) percentile bounds (see `_volume_normalization_bounds`).
            Computed from `input_` when not given; pass this to share one volume's bounds across a
            caller's own tiling/blocking of it, e.g. `TiledAutomaticPromptGenerator`'s per-block calls.

    Returns:
        The volume embeddings, with the per-slice 'features', 'pos_enc' and 'fpn' outputs of the
        image encoder as well as 'input_size' and 'original_size'.
    """
    from micro_sam.v2.util import _write_embedding_signature

    if save_path is not None and "original_size" in root.attrs:
        features = root["features"] if lazy_loading else root["features"][:]
        return {
            "features": features,
            "pos_enc": _load_feature_levels(root["pos_enc"], lazy_loading),
            "fpn": _load_feature_levels(root["fpn"], lazy_loading),
            "input_size": root.attrs["input_size"],
            "original_size": root.attrs["original_size"],
        }

    for key in ("features", "pos_enc", "fpn"):
        if key in root:
            del root[key]

    n_slices = int(input_.shape[0])
    image_size = int(predictor.image_size)
    pbar_init(n_slices, "Compute Image Embeddings 3D")
    model, model_devices, batch_sizes = _prepare_encoder_pipeline(predictor, n_slices, batch_size, devices)

    # Computed once over the whole volume, so every slice normalizes against the same statistics
    # instead of each one estimating its own percentiles.
    if norm_bounds is None:
        norm_bounds = _volume_normalization_bounds(input_)

    if save_path is None:
        feature_values = [None] * n_slices
        fpn_values = [None] * n_slices
    else:
        feature_values = fpn_values = None
    # Only one positional encoding is kept, whichever slice is written first; they are all equal.
    pos_shared = None
    feature_dataset = None
    pos_datasets = None
    fpn_datasets = None

    def load_slice(z):
        raw = np.asarray(input_[z])
        return {
            "tensor": _prepare_video_frame(raw, image_size, bounds=norm_bounds),
            "original_size": tuple(int(value) for value in raw.shape[:2]),
        }

    # The tool creates the datasets from the first result, so only their creation is serialized. The
    # per-slice writes run in parallel (they go to separate chunks).
    creation_lock = threading.Lock()

    def write_slice(z, result):
        nonlocal feature_dataset, pos_datasets, fpn_datasets, pos_shared
        if save_path is None:
            with creation_lock:
                if pos_shared is None:
                    pos_shared = [torch.from_numpy(value) for value in result["pos_enc"]]
            _check_pos_enc_shapes(result["pos_enc"], pos_shared)
            feature_values[z] = torch.from_numpy(result["features"])
            fpn_values[z] = [torch.from_numpy(value) for value in result["fpn"]]
            return

        with creation_lock:
            if feature_dataset is None:
                feature_dataset = _create_feature_dataset(root, "features", n_slices, result["features"])
                # One entry, not one per slice: all slices share the same positional encoding.
                pos_datasets = _create_feature_levels(root.require_group("pos_enc"), 1, result["pos_enc"])
                for dataset, value in zip(pos_datasets, result["pos_enc"]):
                    dataset[0] = value
                fpn_datasets = _create_feature_levels(root.require_group("fpn"), n_slices, result["fpn"])
        _check_pos_enc_shapes(result["pos_enc"], pos_datasets)
        feature_dataset[z] = result["features"]
        for dataset, value in zip(fpn_datasets, result["fpn"]):
            dataset[z] = value

    try:
        _run_batched_pipeline(
            jobs=range(n_slices),
            model_devices=model_devices,
            batch_sizes=batch_sizes,
            load_fn=load_slice,
            predict_fn=_forward_video_batch,
            write_fn=write_slice,
            num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
            update_progress=pbar_update,
        )
    finally:
        _release_model_replicas(model_devices)

    original_size = tuple(int(value) for value in input_.shape[-2:])
    if save_path is None:
        features = torch.cat(feature_values).numpy()
        n_levels = len(fpn_values[0])
        # Shaped like the on-disk layout, (1, 1, C, H, W), so both are read back the same way.
        pos_enc = [value.unsqueeze(0) for value in pos_shared]
        fpn = [torch.stack([value[level] for value in fpn_values]) for level in range(n_levels)]
    else:
        _write_embedding_signature(
            root, input_, predictor, tile_shape=None, halo=None,
            input_size=image_size, original_size=original_size, preprocessing=VIDEO_PREPROCESSING,
        )
        features = feature_dataset if lazy_loading else feature_dataset[:]
        pos_enc = _load_feature_levels(root["pos_enc"], lazy_loading)
        fpn = _load_feature_levels(root["fpn"], lazy_loading)

    return {
        "features": features,
        "pos_enc": pos_enc,
        "fpn": fpn,
        "input_size": image_size,
        "original_size": original_size,
    }


def _compute_tiled_3d(
    input_: np.ndarray,
    predictor,
    tile_shape: Tuple[int, int],
    halo: Tuple[int, int],
    root,
    save_path: Optional[Union[str, os.PathLike]],
    pbar_init: Callable,
    pbar_update: Callable,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
    norm_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict:
    """Compute tile / slice embeddings as one pipelined job stream across all available GPUs.

    Every (tile, slice) pair is a separate job, so tiles and slices are batched together instead of
    tile column by tile column. This keeps all devices busy even for few tiles or few slices.

    Args:
        input_: The input volume, shape (Z, Y, X).
        predictor: The SAM2 video predictor.
        tile_shape: The in-plane tile shape. The volume is not tiled along z.
        halo: The in-plane tile halo (overlap).
        root: The zarr container the embeddings are written to (see `micro_sam.util._open_embeddings`).
        save_path: The path backing `root`, or None for an in-memory container. A complete cache at
            this path is returned as is instead of being recomputed.
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        batch_size: The number of tile slices per encoder call. By default it is looked up from
            the free VRAM of each CUDA device (see `util.recommend_batch_size`).
        devices: The device or devices to run inference on (see `_resolve_devices`).
        num_prefetch_workers: The number of threads used to read and preprocess tile slices.
        num_write_workers: The number of threads used to write embedding tile slices.
        norm_bounds: Precomputed (lower, upper) percentile bounds (see `_volume_normalization_bounds`).
            Computed from `input_` when not given; pass this to share one volume's bounds across a
            caller's own tiling/blocking of it, e.g. `TiledAutomaticPromptGenerator`'s per-block calls.

    Returns:
        The tiled volume embeddings. 'features', 'pos_enc' and 'fpn' are the zarr groups holding the
        per-tile datasets; 'input_size' and 'original_size' are None because they are stored per tile.
    """
    from bioimage_cpp.utils import Blocking
    from micro_sam.v2.util import _write_embedding_signature

    features = root.require_group("features")
    pos_enc_group = root.require_group("pos_enc")
    fpn_group = root.require_group("fpn")
    if save_path is not None and "shape" in features.attrs and _embedding_cache_complete(features, root):
        return {
            "features": features,
            "pos_enc": pos_enc_group,
            "fpn": fpn_group,
            "input_size": None,
            "original_size": None,
        }

    _clear_group(features)
    _clear_group(pos_enc_group)
    _clear_group(fpn_group)

    tiling = Blocking([0, 0], list(input_.shape[1:]), list(tile_shape))
    n_tiles = tiling.number_of_blocks
    n_slices = int(input_.shape[0])
    image_size = int(predictor.image_size)
    jobs = [(tile_id, z) for tile_id in range(n_tiles) for z in range(n_slices)]

    features.attrs["shape"] = list(input_.shape)
    features.attrs["tile_shape"] = list(tile_shape)
    features.attrs["halo"] = list(halo)
    features.attrs["complete"] = False
    pbar_init(len(jobs), "Compute Image Embeddings 3D tiled")

    tile_bounds = {}
    for tile_id in range(n_tiles):
        block = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
        tile_bounds[tile_id] = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))

    # Computed once over the whole volume, so every tile normalizes against the same statistics
    # instead of each one estimating its own percentiles from its own, smaller crop.
    if norm_bounds is None:
        norm_bounds = _volume_normalization_bounds(input_)

    model, model_devices, batch_sizes = _prepare_encoder_pipeline(predictor, len(jobs), batch_size, devices)
    tile_datasets = {}

    def load_tile_slice(job):
        tile_id, z = job
        bb = tile_bounds[tile_id]
        raw = np.asarray(input_[z, bb[0], bb[1]])
        return {
            "tensor": _prepare_video_frame(raw, image_size, bounds=norm_bounds),
            "original_size": tuple(int(value) for value in raw.shape[:2]),
        }

    # The tool creates a tile's datasets from its first slice, so only their creation is serialized. The
    # per-slice writes run in parallel (they go to separate chunks).
    creation_lock = threading.Lock()

    def write_tile_slice(job, result):
        tile_id, z = job
        with creation_lock:
            if tile_id not in tile_datasets:
                name = str(tile_id)
                feature_dataset = _create_feature_dataset(features, name, n_slices, result["features"])
                feature_dataset.attrs["input_size"] = image_size
                feature_dataset.attrs["original_size"] = list(result["original_size"])
                # One entry per tile rather than one per slice, see `write_slice` in `_compute_3d`.
                pos_datasets = _create_feature_levels(
                    pos_enc_group.require_group(name), 1, result["pos_enc"],
                )
                for dataset, value in zip(pos_datasets, result["pos_enc"]):
                    dataset[0] = value
                fpn_datasets = _create_feature_levels(
                    fpn_group.require_group(name), n_slices, result["fpn"],
                )
                tile_datasets[tile_id] = feature_dataset, pos_datasets, fpn_datasets

        feature_dataset, pos_datasets, fpn_datasets = tile_datasets[tile_id]
        _check_pos_enc_shapes(result["pos_enc"], pos_datasets)
        feature_dataset[z] = result["features"]
        for dataset, value in zip(fpn_datasets, result["fpn"]):
            dataset[z] = value

    try:
        _run_batched_pipeline(
            jobs=jobs,
            model_devices=model_devices,
            batch_sizes=batch_sizes,
            load_fn=load_tile_slice,
            predict_fn=_forward_video_batch,
            write_fn=write_tile_slice,
            num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
            update_progress=pbar_update,
        )
    finally:
        _release_model_replicas(model_devices)

    if save_path is not None:
        _write_embedding_signature(
            root, input_, predictor, tile_shape=tile_shape, halo=halo, input_size=None,
            original_size=None, preprocessing=VIDEO_PREPROCESSING,
        )
    features.attrs["complete"] = True
    return {
        "features": features,
        "pos_enc": pos_enc_group,
        "fpn": fpn_group,
        "input_size": None,
        "original_size": None,
    }


class _EmptyEncoder(torch.nn.Module):
    """Lightweight placeholder used while replicating decoder-only UniSAM2 models."""

    def __init__(self, img_size: int) -> None:
        super().__init__()
        self.img_size = img_size

    def forward(self, x: torch.Tensor):
        raise RuntimeError("The decoder-only placeholder must be replaced before inference.")


def _normalize_feature_block(block: np.ndarray) -> torch.Tensor:
    block = np.asarray(block)
    if block.ndim == 5 and block.shape[1] == 1:
        block = block[:, 0]
    if block.ndim == 3:
        block = block[None]
    if block.ndim != 4:
        raise ValueError(f"Expected a (Z, C, H, W) feature block, got shape {block.shape}.")
    return torch.from_numpy(np.asarray(block, dtype="float32"))


def _load_job(job: Dict) -> Dict:
    source = job["source"]
    selection = job.get("selection")
    block = np.asarray(source) if selection is None else np.asarray(source[selection])
    return {
        "feature": _normalize_feature_block(block),
        "original_size": job["original_size"],
    }


def _predict_jobs(model: torch.nn.Module, items: List[Dict], device: torch.device) -> List[np.ndarray]:
    from micro_sam.v2.instance_segmentation import _decode_3d_feature_batch

    original_size = items[0]["original_size"]
    if any(item["original_size"] != original_size for item in items):
        raise RuntimeError("Decoder jobs with different output sizes cannot share a batch.")
    features = torch.stack([item["feature"] for item in items]).to(device, non_blocking=True)
    output = _decode_3d_feature_batch(model, features, original_size, device)
    predictions = [np.array(value) for value in output]
    _release_fragmented_cuda_cache(device)
    return predictions


def _feature_shape(job: Dict) -> Tuple[int, ...]:
    source_shape = tuple(job["source"].shape)
    selection = job.get("selection")
    if selection is None:
        shape = source_shape
    else:
        start = 0 if selection.start is None else selection.start
        stop = source_shape[0] if selection.stop is None else selection.stop
        shape = (stop - start,) + source_shape[1:]

    if len(shape) == 5 and shape[1] == 1:
        shape = (shape[0],) + shape[2:]
    if len(shape) == 3:
        shape = (1,) + shape
    if len(shape) != 4:
        raise ValueError(f"Expected feature shape (Z, C, H, W), got {shape}.")
    return shape


@contextlib.contextmanager
def _decoder_only(model: torch.nn.Module):
    """Replace the model's image encoder for the duration of the block, so only the decoder is run.

    The encoder is not needed when decoding precomputed embeddings, and replicating it per device
    would waste memory. Restoring it is a context manager (rather than scattered cleanup) so the
    caller's model gets its encoder back on every exit path.
    """
    encoder = model.encoder
    model.encoder = _EmptyEncoder(getattr(encoder, "img_size", 1024))
    try:
        yield
    finally:
        model.encoder = encoder


def _prepare_decoder_pipeline(
    model: torch.nn.Module,
    jobs: List[Dict],
    batch_size: Optional[int],
    resolved_devices: Sequence[torch.device],
) -> Tuple[List[Tuple[torch.nn.Module, torch.device]], List[int]]:
    """Replicate the decoder and select its batch size. Must run inside `_decoder_only`."""
    model_devices = _prepare_models(model, resolved_devices)
    try:
        if batch_size is not None:
            if int(batch_size) < 1:
                raise ValueError(f"batch_size must be positive or None, got {batch_size}.")
            return model_devices, [int(batch_size)] * len(model_devices)

        representative = max(
            jobs,
            key=lambda job: np.prod(_feature_shape(job)) * np.prod(job["original_size"]),
        )
        z, channels, height, width = _feature_shape(representative)
        original_size = representative["original_size"]

        def prediction_function(this_model, inputs):
            from micro_sam.v2.instance_segmentation import _decode_3d_feature_batch

            features = inputs.permute(0, 2, 1, 3, 4)
            return _decode_3d_feature_batch(
                this_model, features, original_size, inputs.device,
            )

        batch_sizes = _compute_auto_batch_sizes(
            model_devices=model_devices,
            n_jobs=len(jobs),
            patch_shape=(z, height, width),
            in_channels=channels,
            prediction_function=prediction_function,
        )
    except Exception:
        _release_model_replicas(model_devices)
        raise
    return model_devices, batch_sizes


def _run_decoder_jobs(
    model: torch.nn.Module,
    jobs: List[Dict],
    write_fn: Callable[[Dict, np.ndarray], None],
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
    update_progress: Optional[Callable[[int], None]] = None,
    progress_increment: Optional[Callable[[Dict], int]] = None,
) -> None:
    if len(jobs) == 0:
        return

    # Resolved before the encoder is swapped out, so the device is read off the full model.
    resolved_devices = _resolve_devices(model, devices)
    with _decoder_only(model):
        groups = defaultdict(list)
        for job in jobs:
            groups[(_feature_shape(job), job["original_size"])].append(job)

        # Run the largest shape first so later, smaller groups can reuse its allocator blocks. Starting
        # with a boundary z-block and growing to the full context fragmented 10 GB MIG allocators.
        group_keys = sorted(groups, key=lambda key: np.prod(key[0]) * np.prod(key[1]), reverse=True)

        model_devices, batch_sizes = _prepare_decoder_pipeline(model, jobs, batch_size, resolved_devices)
        try:
            for group_key in group_keys:
                group_jobs = groups[group_key]
                _run_batched_pipeline(
                    jobs=group_jobs,
                    model_devices=model_devices,
                    batch_sizes=batch_sizes,
                    load_fn=_load_job,
                    predict_fn=_predict_jobs,
                    write_fn=write_fn,
                    num_prefetch_workers=num_prefetch_workers,
                    num_write_workers=num_write_workers,
                    update_progress=update_progress,
                    progress_increment=progress_increment,
                )
        finally:
            _release_model_replicas(model_devices)


def _resolve_z_blocking(z_block: Optional[int], z_halo: Optional[int]) -> Tuple[int, int]:
    """Resolve the z block / halo used to decode a volume, falling back to the defaults."""
    from micro_sam.v2.util import DEFAULT_HALO_Z, DEFAULT_TILE_Z

    z_block = DEFAULT_TILE_Z if z_block is None else int(z_block)
    z_halo = DEFAULT_HALO_Z if z_halo is None else int(z_halo)
    if z_block < 1 or z_halo < 0:
        raise ValueError(f"z_block must be positive and z_halo non-negative, got {z_block}, {z_halo}.")
    return z_block, z_halo


def _decode_volume_embeddings(
    model: torch.nn.Module,
    image_embeddings: Dict,
    z_block: Optional[int] = None,
    z_halo: Optional[int] = None,
    pbar_init: Optional[Callable] = None,
    pbar_update: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
) -> np.ndarray:
    """Decode z blocks from non-tiled volume embeddings with queued reads and batched inference.

    The volume is decoded in blocks of `z_block` slices, each extended by `z_halo` context slices
    that are cropped away again when the block is written.

    Args:
        model: The UniSAM2 model. Its encoder is bypassed, only the decoder is run.
        image_embeddings: The volume embeddings (see `_compute_3d`).
        device: Unused, kept for interface compatibility. Use `devices` instead.
        z_block: The number of slices decoded per block. By default `micro_sam.v2.util.DEFAULT_TILE_Z`.
        z_halo: The number of context slices per block. By default `micro_sam.v2.util.DEFAULT_HALO_Z`.
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        batch_size: The number of z blocks per decoder call. By default candidate sizes are
            benchmarked and a throughput-efficient value is selected on each CUDA device.
        devices: The device or devices to run inference on (see `_resolve_devices`).
        num_prefetch_workers: The number of threads used to read embedding blocks.
        num_write_workers: The number of threads used to write decoded blocks.

    Returns:
        The decoder predictions, shape (4, Z, Y, X): foreground and the three distance channels.

    Raises:
        ValueError: If the features are not 3d, if `z_block` is not positive or `z_halo` is negative.
    """
    features = image_embeddings["features"]
    n_dims = len(features.shape)
    if n_dims not in (4, 5):
        raise ValueError(
            f"Decoder-from-embeddings (3d) requires features with ndim 4 or 5, got {n_dims}."
        )

    n_slices = int(features.shape[0])
    z_block, z_halo = _resolve_z_blocking(z_block, z_halo)

    original_size = tuple(int(value) for value in np.asarray(image_embeddings["original_size"]).reshape(-1)[:2])
    output = np.zeros((4, n_slices, *original_size), dtype="float32")
    jobs = []
    for z0 in range(0, n_slices, z_block):
        z1 = min(z0 + z_block, n_slices)
        c0, c1 = max(0, z0 - z_halo), min(n_slices, z1 + z_halo)
        jobs.append({
            "source": features,
            "selection": slice(c0, c1),
            "original_size": original_size,
            "z0": z0,
            "z1": z1,
            "c0": c0,
        })

    if pbar_init is not None:
        pbar_init(n_slices, "Automatic segmentation (volume)")

    def write_prediction(job, prediction):
        inner = job["z0"] - job["c0"]
        count = job["z1"] - job["z0"]
        output[:, job["z0"]:job["z1"]] = prediction[:, inner:inner + count]

    _run_decoder_jobs(
        model, jobs, write_prediction, batch_size=batch_size, devices=devices,
        num_prefetch_workers=num_prefetch_workers, num_write_workers=num_write_workers,
        update_progress=pbar_update, progress_increment=lambda job: job["z1"] - job["z0"],
    )
    return output


def _tiled_metadata(image_embeddings: Dict, is_3d: bool) -> Tuple:
    from bioimage_cpp.utils import Blocking

    features = image_embeddings["features"]
    shape = tuple(int(value) for value in features.attrs["shape"])
    tile_shape = tuple(int(value) for value in features.attrs["tile_shape"])
    halo = tuple(int(value) for value in features.attrs["halo"])
    spatial_shape = shape[1:] if is_3d else shape
    tiling = Blocking([0, 0], list(spatial_shape), list(tile_shape))
    return features, shape, halo, tiling


def _decode_tiled_2d_embeddings(
    model: torch.nn.Module,
    image_embeddings: Dict,
    pbar_init: Optional[Callable] = None,
    pbar_update: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
) -> np.ndarray:
    """Decode and stitch 2d embedding tiles in automatically sized batches.

    Args:
        model: The UniSAM2 model. Its encoder is bypassed, only the decoder is run.
        image_embeddings: The tiled image embeddings (see `_compute_tiled_2d`).
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        batch_size: The number of tiles per decoder call. By default candidate sizes are benchmarked
            and a throughput-efficient value is selected on each CUDA device.
        devices: The device or devices to run inference on (see `_resolve_devices`).
        num_prefetch_workers: The number of threads used to read embedding tiles.
        num_write_workers: The number of threads used to stitch decoded tiles into the output.

    Returns:
        The stitched decoder predictions, shape (4, Y, X): foreground and the three distance channels.
    """
    features, shape, halo, tiling = _tiled_metadata(image_embeddings, is_3d=False)
    output = np.zeros((4, *shape), dtype="float32")
    jobs = []
    for tile_id in range(tiling.number_of_blocks):
        tile_features = features[str(tile_id)]
        original_size = tuple(int(value) for value in np.asarray(tile_features.attrs["original_size"]).reshape(-1)[:2])
        jobs.append({
            "source": tile_features,
            "selection": None,
            "original_size": original_size,
            "tile_id": tile_id,
        })

    if pbar_init is not None:
        pbar_init(len(jobs), "Automatic segmentation (tiles)")

    def write_prediction(job, prediction):
        block = tiling.get_block_with_halo(job["tile_id"], halo=list(halo))
        local = tuple(slice(begin, end) for begin, end in zip(
            block.inner_block_local.begin, block.inner_block_local.end,
        ))
        inner = tuple(slice(begin, end) for begin, end in zip(
            block.inner_block.begin, block.inner_block.end,
        ))
        output[(slice(None),) + inner] = prediction[(slice(None), slice(0, 1)) + local][:, 0]

    _run_decoder_jobs(
        model, jobs, write_prediction, batch_size=batch_size, devices=devices,
        num_prefetch_workers=num_prefetch_workers, num_write_workers=num_write_workers,
        update_progress=pbar_update,
    )
    return output


def _tiled_3d_jobs(features, tiling, n_slices: int, z_block: int, z_halo: int) -> List[Dict]:
    jobs = []
    for tile_id in range(tiling.number_of_blocks):
        tile_features = features[str(tile_id)]
        original_size = tuple(int(value) for value in np.asarray(tile_features.attrs["original_size"]).reshape(-1)[:2])
        for z0 in range(0, n_slices, z_block):
            z1 = min(z0 + z_block, n_slices)
            c0, c1 = max(0, z0 - z_halo), min(n_slices, z1 + z_halo)
            jobs.append({
                "source": tile_features,
                "selection": slice(c0, c1),
                "original_size": original_size,
                "tile_id": tile_id,
                "z0": z0,
                "z1": z1,
                "c0": c0,
            })
    return jobs


def _decode_tiled_3d_embeddings(
    model: torch.nn.Module,
    image_embeddings: Dict,
    pbar_init: Optional[Callable] = None,
    pbar_update: Optional[Callable] = None,
    z_block: Optional[int] = None,
    z_halo: Optional[int] = None,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
) -> np.ndarray:
    """Batch over both tile columns and z blocks while decoding tiled 3d embeddings.

    Args:
        model: The UniSAM2 model. Its encoder is bypassed, only the decoder is run.
        image_embeddings: The tiled volume embeddings (see `_compute_tiled_3d`).
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        z_block: The number of slices decoded per block. By default `micro_sam.v2.util.DEFAULT_TILE_Z`.
        z_halo: The number of context slices per block. By default `micro_sam.v2.util.DEFAULT_HALO_Z`.
        batch_size: The number of (tile, z block) jobs per decoder call. By default candidate sizes
            are benchmarked and a throughput-efficient value is selected on each CUDA device.
        devices: The device or devices to run inference on (see `_resolve_devices`).
        num_prefetch_workers: The number of threads used to read embedding blocks.
        num_write_workers: The number of threads used to stitch decoded blocks into the output.

    Returns:
        The stitched decoder predictions, shape (4, Z, Y, X): foreground and the three distance channels.

    Raises:
        ValueError: If `z_block` is not positive or `z_halo` is negative.
    """
    features, shape, halo, tiling = _tiled_metadata(image_embeddings, is_3d=True)
    n_slices = shape[0]
    z_block, z_halo = _resolve_z_blocking(z_block, z_halo)
    jobs = _tiled_3d_jobs(features, tiling, n_slices, z_block, z_halo)
    output = np.zeros((4, *shape), dtype="float32")

    if pbar_init is not None:
        pbar_init(tiling.number_of_blocks * n_slices, "Automatic segmentation (tiles)")

    def write_prediction(job, prediction):
        block = tiling.get_block_with_halo(job["tile_id"], halo=list(halo))
        local = tuple(slice(begin, end) for begin, end in zip(
            block.inner_block_local.begin, block.inner_block_local.end,
        ))
        inner = tuple(slice(begin, end) for begin, end in zip(
            block.inner_block.begin, block.inner_block.end,
        ))
        z_count = job["z1"] - job["z0"]
        local_z = job["z0"] - job["c0"]
        prediction = prediction[:, local_z:local_z + z_count]
        output[(slice(None), slice(job["z0"], job["z1"])) + inner] = prediction[(slice(None), slice(None)) + local]

    _run_decoder_jobs(
        model, jobs, write_prediction, batch_size=batch_size, devices=devices,
        num_prefetch_workers=num_prefetch_workers, num_write_workers=num_write_workers,
        update_progress=pbar_update, progress_increment=lambda job: job["z1"] - job["z0"],
    )
    return output


def _decode_tiled_3d_slice(
    model: torch.nn.Module,
    image_embeddings: Dict,
    index: int,
    pbar_init: Optional[Callable] = None,
    pbar_update: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
) -> np.ndarray:
    """Decode one volume slice across all embedding tiles in batches.

    Used for slice-wise automatic segmentation of a tiled volume, e.g. when only the current slice
    is segmented in the annotator.

    Args:
        model: The UniSAM2 model. Its encoder is bypassed, only the decoder is run.
        image_embeddings: The tiled volume embeddings (see `_compute_tiled_3d`).
        index: The index of the slice to decode.
        pbar_init: Callback to initialize an external progress bar.
        pbar_update: Callback to update an external progress bar.
        batch_size: The number of tiles per decoder call. By default candidate sizes are benchmarked
            and a throughput-efficient value is selected on each CUDA device.
        devices: The device or devices to run inference on (see `_resolve_devices`).
        num_prefetch_workers: The number of threads used to read embedding tiles.
        num_write_workers: The number of threads used to stitch decoded tiles into the output.

    Returns:
        The stitched decoder predictions for the slice, shape (4, Y, X).

    Raises:
        ValueError: If `index` is outside the volume.
    """
    features, shape, halo, tiling = _tiled_metadata(image_embeddings, is_3d=True)
    n_slices = shape[0]
    index = int(index)
    # A negative index would silently decode from the end, an index past the volume nothing at all.
    if not 0 <= index < n_slices:
        raise ValueError(f"The slice index must be in [0, {n_slices}), got {index}.")

    output = np.zeros((4, *shape[1:]), dtype="float32")
    jobs = []
    for tile_id in range(tiling.number_of_blocks):
        tile_features = features[str(tile_id)]
        original_size = tuple(int(value) for value in np.asarray(tile_features.attrs["original_size"]).reshape(-1)[:2])
        jobs.append({
            "source": tile_features,
            "selection": slice(index, index + 1),
            "original_size": original_size,
            "tile_id": tile_id,
        })

    if pbar_init is not None:
        pbar_init(len(jobs), "Automatic segmentation (tiles)")

    def write_prediction(job, prediction):
        block = tiling.get_block_with_halo(job["tile_id"], halo=list(halo))
        local = tuple(slice(begin, end) for begin, end in zip(
            block.inner_block_local.begin, block.inner_block_local.end,
        ))
        inner = tuple(slice(begin, end) for begin, end in zip(
            block.inner_block.begin, block.inner_block.end,
        ))
        output[(slice(None),) + inner] = prediction[(slice(None), 0) + local]

    _run_decoder_jobs(
        model, jobs, write_prediction, batch_size=batch_size, devices=devices,
        num_prefetch_workers=num_prefetch_workers, num_write_workers=num_write_workers,
        update_progress=pbar_update,
    )
    return output
