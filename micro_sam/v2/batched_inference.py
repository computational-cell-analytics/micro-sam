"""Batched, pipelined, multi-GPU SAM2 inference: scheduling engine, encoder embeddings, and decoder passes."""

import gc
import os
import queue
import threading
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from micro_sam.util import _create_dataset_with_data, _create_dataset_without_data
from .normalization import to_image
from .util import Device, Devices


STOP = object()


class PipelineAborted(Exception):
    """Raised inside workers when another pipeline worker has failed."""


class AtomicCounter:
    """Lock-guarded counter used to send completion sentinels exactly once."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.lock = threading.Lock()

    def decrement(self) -> int:
        with self.lock:
            self.value -= 1
            return self.value


@dataclass
class PipelineJob:
    """A work item moving from the input loader to inference and output writing."""

    spec: Any
    data: Any


def _normalize_device(device: Union[str, torch.device]) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return _normalize_device(next(model.parameters()).device)
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def resolve_devices(model: torch.nn.Module, devices: Devices = None) -> List[torch.device]:
    """Resolve inference devices, using every visible CUDA device by default.

    Automatic multi-GPU execution is enabled only when the supplied model already lives on CUDA.
    This preserves an explicitly CPU- or MPS-loaded model. Pass a scalar device to force one device,
    or a sequence to select an explicit set of devices.
    """
    if devices is None:
        device = _model_device(model)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            resolved = [torch.device("cuda", index) for index in range(torch.cuda.device_count())]
        else:
            resolved = [device]
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


def prepare_models(
    model: torch.nn.Module, devices: Sequence[torch.device],
) -> List[Tuple[torch.nn.Module, torch.device]]:
    """Create one eval-mode model replica per device, reusing the original where possible."""
    source_device = _model_device(model)
    models = []
    for device in devices:
        replica = model if device == source_device else deepcopy(model).to(device)
        if hasattr(replica, "eval"):
            replica.eval()
        models.append((replica, device))
    return models


def release_model_replicas(model_devices: List[Tuple[torch.nn.Module, torch.device]]) -> None:
    """Drop the replicas built by :func:`prepare_models` and free their GPU memory.

    Clearing the list releases only the per-device copies; the caller's own model stays alive
    through its other references.
    """
    model_devices.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_auto_batch_sizes(
    model_devices: Sequence[Tuple[torch.nn.Module, torch.device]],
    n_jobs: int,
    patch_shape: Tuple[int, ...],
    in_channels: int,
    prediction_function: Callable[[torch.nn.Module, torch.Tensor], Any],
) -> List[int]:
    """Probe the largest safe batch size independently on every CUDA device.

    The probe delegates its exponential and binary OOM search to
    :func:`torch_em.util.compute_max_batch_size`. CPU and MPS use a conservative batch size of one,
    because the torch-em probe intentionally supports CUDA OOM detection only.
    """
    if n_jobs < 1:
        return [1] * len(model_devices)

    # Cap the probe at the per-device job count and at 256 (no run needs a larger batch).
    jobs_per_device = (int(n_jobs) + len(model_devices) - 1) // len(model_devices)
    upper_bound = min(256, jobs_per_device)
    batch_sizes = []
    from torch_em.util import compute_max_batch_size

    for model, device in model_devices:
        if device.type != "cuda":
            batch_sizes.append(1)
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The batch size search reached the upper bound.*", category=UserWarning,
            )
            batch_size = compute_max_batch_size(
                model=model,
                patch_shape=patch_shape,
                in_channels=in_channels,
                device=device,
                dtype=torch.float32,
                safety_factor=0.8,  # leave GPU-memory headroom below the probed maximum
                max_batch_size=upper_bound,
                prediction_function=prediction_function,
            )
        batch_sizes.append(batch_size)
    return batch_sizes


def _safe_get(input_queue: queue.Queue, stop_event: threading.Event, timeout: float = 0.2) -> Any:
    while not stop_event.is_set():
        try:
            return input_queue.get(timeout=timeout)
        except queue.Empty:
            continue
    raise PipelineAborted()


def _safe_put(output_queue: queue.Queue, item: Any, stop_event: threading.Event, timeout: float = 0.2) -> None:
    while not stop_event.is_set():
        try:
            output_queue.put(item, timeout=timeout)
            return
        except queue.Full:
            continue
    raise PipelineAborted()


def run_batched_pipeline(
    jobs: Iterable[Any],
    model_devices: Sequence[Tuple[torch.nn.Module, torch.device]],
    batch_sizes: Sequence[int],
    load_fn: Callable[[Any], Any],
    predict_fn: Callable[[torch.nn.Module, List[Any], torch.device], List[Any]],
    write_fn: Callable[[Any, Any], None],
    num_prefetch_workers: int = 4,
    update_progress: Optional[Callable[[int], None]] = None,
) -> None:
    """Run load/preprocess, batched inference, and writing as a bounded threaded pipeline.

    There is one inference consumer per device and one writer. Keeping output writes on one thread
    is safe for zarr, N5, HDF5, and in-memory outputs, while still overlapping I/O with GPU work.
    """
    jobs = list(jobs)
    if len(jobs) == 0:
        return
    if len(model_devices) != len(batch_sizes):
        raise ValueError("Expected one batch size for every model/device pair.")
    if any(int(batch_size) < 1 for batch_size in batch_sizes):
        raise ValueError(f"Batch sizes must be positive, got {batch_sizes}.")

    num_prefetch_workers = max(1, min(int(num_prefetch_workers), len(jobs)))
    batch_sizes = [int(batch_size) for batch_size in batch_sizes]

    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in range(num_prefetch_workers):
        job_queue.put(STOP)

    input_queue = queue.Queue(maxsize=max(2 * sum(batch_sizes), 2))
    output_queue = queue.Queue(maxsize=max(2 * len(model_devices), 2))
    stop_event = threading.Event()
    error_box = []
    error_lock = threading.Lock()
    remaining_producers = AtomicCounter(num_prefetch_workers)
    remaining_consumers = AtomicCounter(len(model_devices))

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
                _safe_put(input_queue, PipelineJob(spec, load_fn(spec)), stop_event)
        except PipelineAborted:
            pass
        except Exception as exc:  # noqa
            record_error(exc)
        finally:
            if remaining_producers.decrement() == 0 and not stop_event.is_set():
                for _ in model_devices:
                    input_queue.put(STOP)

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
                        predictions = predict_fn(model, [item.data for item in batch], device)
                    if len(predictions) != len(batch):
                        raise RuntimeError(
                            f"The batch predictor returned {len(predictions)} outputs for {len(batch)} inputs."
                        )
                    for item, prediction in zip(batch, predictions):
                        item.data = prediction
                        _safe_put(output_queue, item, stop_event)

                if got_stop:
                    break
        except PipelineAborted:
            pass
        except Exception as exc:  # noqa
            record_error(exc)
        finally:
            if remaining_consumers.decrement() == 0 and not stop_event.is_set():
                output_queue.put(STOP)

    def writer():
        try:
            while True:
                item = _safe_get(output_queue, stop_event)
                if item is STOP:
                    break
                write_fn(item.spec, item.data)
                if update_progress is not None:
                    update_progress(1)
        except PipelineAborted:
            pass
        except Exception as exc:  # noqa
            record_error(exc)

    writer_thread = threading.Thread(target=writer, name="sam2-writer")
    consumer_threads = [
        threading.Thread(target=consumer, args=(worker_id,), name=f"sam2-consumer-{worker_id}")
        for worker_id in range(len(model_devices))
    ]
    producer_threads = [
        threading.Thread(target=producer, name=f"sam2-producer-{worker_id}")
        for worker_id in range(num_prefetch_workers)
    ]
    threads = [writer_thread, *consumer_threads, *producer_threads]

    try:
        for thread in threads:
            thread.start()
        for thread in producer_threads:
            thread.join()
        for thread in consumer_threads:
            thread.join()
        writer_thread.join()
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    if error_box:
        raise error_box[0]


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _clear_group(group) -> None:
    for key in list(group.keys()):
        del group[key]


def _embedding_cache_complete(features, root) -> bool:
    return bool(features.attrs.get("complete", "input_size" in root.attrs))


def _prepare_encoder_pipeline(
    predictor, n_jobs: int, batch_size: Optional[int], devices: Devices,
) -> Tuple[torch.nn.Module, List[Tuple[torch.nn.Module, torch.device]], List[int]]:
    model = getattr(predictor, "model", predictor)
    resolved_devices = resolve_devices(model, devices)
    model_devices = prepare_models(model, resolved_devices)
    if batch_size is None:
        image_size = int(getattr(model, "image_size", getattr(predictor, "image_size", 1024)))
        batch_sizes = compute_auto_batch_sizes(
            model_devices=model_devices,
            n_jobs=n_jobs,
            patch_shape=(image_size, image_size),
            in_channels=3,
            prediction_function=lambda this_model, inputs: this_model.forward_image(inputs),
        )
    else:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be positive or None, got {batch_size}.")
        batch_sizes = [int(batch_size)] * len(model_devices)
    return model, model_devices, batch_sizes


def _forward_image_batch(
    model: torch.nn.Module, items: List[Dict], device: torch.device, feature_sizes: Sequence,
) -> List[Dict]:
    batch = torch.stack([item["tensor"] for item in items]).to(device, non_blocking=True)
    backbone_out = model.forward_image(batch)
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


def compute_tiled_2d(
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
) -> Dict:
    """Compute 2D tile embeddings with batched encoders and queued input/output I/O."""
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

    def write_tile(tile_id, result):
        name = str(tile_id)
        dataset = _create_dataset_with_data(features, name, data=result["features"])
        dataset.attrs["input_size"] = model.image_size
        dataset.attrs["original_size"] = [list(result["original_size"])]

        tile_high_res = high_res_group.require_group(name)
        for level, feature in enumerate(result["high_res_feats"]):
            _create_dataset_with_data(tile_high_res, str(level), data=feature)

    try:
        run_batched_pipeline(
            jobs=range(n_tiles),
            model_devices=model_devices,
            batch_sizes=batch_sizes,
            load_fn=load_tile,
            predict_fn=predict_tiles,
            write_fn=write_tile,
            num_prefetch_workers=num_prefetch_workers,
            update_progress=pbar_update,
        )
    finally:
        release_model_replicas(model_devices)

    if save_path is not None:
        _write_embedding_signature(
            root, input_, predictor, tile_shape=tile_shape, halo=halo, input_size=None, original_size=None,
        )
    features.attrs["complete"] = True
    return {"features": features, "high_res_feats": high_res_group, "input_size": None, "original_size": None}


def _prepare_video_frame(raw: np.ndarray, image_size: int) -> torch.Tensor:
    from micro_sam.v2.models._video_predictor import _load_img_as_tensor

    image, _, _ = _load_img_as_tensor(np.asarray(raw), image_size)
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGE_STD, dtype=torch.float32)[:, None, None]
    return (image - mean) / std


def _forward_video_batch(model: torch.nn.Module, items: List[Dict], device: torch.device) -> List[Dict]:
    batch = torch.stack([item["tensor"] for item in items]).to(device, non_blocking=True)
    backbone_out = model.forward_image(batch)

    vision_features = backbone_out["vision_features"].detach().cpu().numpy()
    pos_enc = [value.detach().cpu().numpy() for value in backbone_out["vision_pos_enc"]]
    fpn = [value.detach().cpu().numpy() for value in backbone_out["backbone_fpn"]]
    return [
        {
            "features": vision_features[index:index + 1],
            "pos_enc": [value[index:index + 1] for value in pos_enc],
            "fpn": [value[index:index + 1] for value in fpn],
            "original_size": items[index]["original_size"],
        }
        for index in range(len(items))
    ]


def _create_feature_dataset(group, name: str, n_slices: int, tensor: np.ndarray):
    shape = (n_slices,) + tuple(tensor.shape)
    chunks = (1,) + tuple(tensor.shape)
    return _create_dataset_without_data(group, name, shape=shape, dtype="float32", chunks=chunks)


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


def compute_3d(
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
) -> Dict:
    """Compute volume embeddings by batching slices and overlapping preprocessing and zarr writes."""
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

    if save_path is None:
        feature_values = [None] * n_slices
        pos_values = [None] * n_slices
        fpn_values = [None] * n_slices
        feature_dataset = None
        pos_datasets = None
        fpn_datasets = None
    else:
        feature_values = pos_values = fpn_values = None
        feature_dataset = None
        pos_datasets = None
        fpn_datasets = None

    def load_slice(z):
        raw = np.asarray(input_[z])
        return {
            "tensor": _prepare_video_frame(raw, image_size),
            "original_size": tuple(int(value) for value in raw.shape[:2]),
        }

    def write_slice(z, result):
        nonlocal feature_dataset, pos_datasets, fpn_datasets
        if save_path is None:
            feature_values[z] = torch.from_numpy(result["features"])
            pos_values[z] = [torch.from_numpy(value) for value in result["pos_enc"]]
            fpn_values[z] = [torch.from_numpy(value) for value in result["fpn"]]
            return

        if feature_dataset is None:
            feature_dataset = _create_feature_dataset(root, "features", n_slices, result["features"])
            pos_datasets = _create_feature_levels(root.require_group("pos_enc"), n_slices, result["pos_enc"])
            fpn_datasets = _create_feature_levels(root.require_group("fpn"), n_slices, result["fpn"])
        feature_dataset[z] = result["features"]
        for dataset, value in zip(pos_datasets, result["pos_enc"]):
            dataset[z] = value
        for dataset, value in zip(fpn_datasets, result["fpn"]):
            dataset[z] = value

    try:
        run_batched_pipeline(
            jobs=range(n_slices),
            model_devices=model_devices,
            batch_sizes=batch_sizes,
            load_fn=load_slice,
            predict_fn=_forward_video_batch,
            write_fn=write_slice,
            num_prefetch_workers=num_prefetch_workers,
            update_progress=pbar_update,
        )
    finally:
        release_model_replicas(model_devices)

    original_size = tuple(int(value) for value in input_.shape[-2:])
    if save_path is None:
        features = torch.cat(feature_values).numpy()
        n_levels = len(pos_values[0])
        pos_enc = [torch.stack([value[level] for value in pos_values]) for level in range(n_levels)]
        fpn = [torch.stack([value[level] for value in fpn_values]) for level in range(n_levels)]
    else:
        _write_embedding_signature(
            root, input_, predictor, tile_shape=None, halo=None,
            input_size=image_size, original_size=original_size,
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


def compute_tiled_3d(
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
) -> Dict:
    """Compute tile/slice embeddings as one pipelined job stream across all available GPUs."""
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

    bounds = {}
    for tile_id in range(n_tiles):
        block = tiling.get_block_with_halo(tile_id, list(halo)).outer_block
        bounds[tile_id] = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))

    model, model_devices, batch_sizes = _prepare_encoder_pipeline(predictor, len(jobs), batch_size, devices)
    tile_datasets = {}

    def load_tile_slice(job):
        tile_id, z = job
        bb = bounds[tile_id]
        raw = np.asarray(input_[z, bb[0], bb[1]])
        return {
            "tensor": _prepare_video_frame(raw, image_size),
            "original_size": tuple(int(value) for value in raw.shape[:2]),
        }

    def write_tile_slice(job, result):
        tile_id, z = job
        if tile_id not in tile_datasets:
            name = str(tile_id)
            feature_dataset = _create_feature_dataset(features, name, n_slices, result["features"])
            feature_dataset.attrs["input_size"] = image_size
            feature_dataset.attrs["original_size"] = list(result["original_size"])
            pos_datasets = _create_feature_levels(
                pos_enc_group.require_group(name), n_slices, result["pos_enc"],
            )
            fpn_datasets = _create_feature_levels(
                fpn_group.require_group(name), n_slices, result["fpn"],
            )
            tile_datasets[tile_id] = feature_dataset, pos_datasets, fpn_datasets

        feature_dataset, pos_datasets, fpn_datasets = tile_datasets[tile_id]
        feature_dataset[z] = result["features"]
        for dataset, value in zip(pos_datasets, result["pos_enc"]):
            dataset[z] = value
        for dataset, value in zip(fpn_datasets, result["fpn"]):
            dataset[z] = value

    try:
        run_batched_pipeline(
            jobs=jobs,
            model_devices=model_devices,
            batch_sizes=batch_sizes,
            load_fn=load_tile_slice,
            predict_fn=_forward_video_batch,
            write_fn=write_tile_slice,
            num_prefetch_workers=num_prefetch_workers,
            update_progress=pbar_update,
        )
    finally:
        release_model_replicas(model_devices)

    if save_path is not None:
        _write_embedding_signature(
            root, input_, predictor, tile_shape=tile_shape, halo=halo, input_size=None, original_size=None,
        )
    features.attrs["complete"] = True
    return {
        "features": features,
        "pos_enc": pos_enc_group,
        "fpn": fpn_group,
        "input_size": None,
        "original_size": None,
    }


class EmptyEncoder(torch.nn.Module):
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
    return [np.array(value) for value in output]


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


def _prepare_decoder_pipeline(
    model: torch.nn.Module, jobs: List[Dict], batch_size: Optional[int], devices: Devices,
) -> Tuple[List[Tuple[torch.nn.Module, torch.device]], List[int], torch.nn.Module]:
    resolved_devices = resolve_devices(model, devices)
    encoder = model.encoder
    model.encoder = EmptyEncoder(getattr(encoder, "img_size", 1024))
    try:
        model_devices = prepare_models(model, resolved_devices)
    except Exception:
        model.encoder = encoder
        raise

    if batch_size is not None:
        if int(batch_size) < 1:
            release_model_replicas(model_devices)
            model.encoder = encoder
            raise ValueError(f"batch_size must be positive or None, got {batch_size}.")
        return model_devices, [int(batch_size)] * len(model_devices), encoder

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

    try:
        batch_sizes = compute_auto_batch_sizes(
            model_devices=model_devices,
            n_jobs=len(jobs),
            patch_shape=(z, height, width),
            in_channels=channels,
            prediction_function=prediction_function,
        )
    except Exception:
        release_model_replicas(model_devices)
        model.encoder = encoder
        raise
    return model_devices, batch_sizes, encoder


def _run_decoder_jobs(
    model: torch.nn.Module,
    jobs: List[Dict],
    write_fn: Callable[[Dict, np.ndarray], None],
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
) -> None:
    if len(jobs) == 0:
        return

    model_devices, batch_sizes, encoder = _prepare_decoder_pipeline(model, jobs, batch_size, devices)
    groups = defaultdict(list)
    for job in jobs:
        groups[(_feature_shape(job), job["original_size"])].append(job)

    try:
        for group_jobs in groups.values():
            run_batched_pipeline(
                jobs=group_jobs,
                model_devices=model_devices,
                batch_sizes=batch_sizes,
                load_fn=_load_job,
                predict_fn=_predict_jobs,
                write_fn=write_fn,
                num_prefetch_workers=num_prefetch_workers,
            )
    finally:
        release_model_replicas(model_devices)
        model.encoder = encoder


def decode_volume_embeddings(
    model: torch.nn.Module,
    image_embeddings: Dict,
    device: Device = None,
    z_block: Optional[int] = None,
    z_halo: Optional[int] = None,
    pbar_init: Optional[Callable] = None,
    pbar_update: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
) -> np.ndarray:
    """Decode z blocks from non-tiled volume embeddings with queued reads and batched inference."""
    from micro_sam.v2.util import DEFAULT_HALO_Z, DEFAULT_TILE_Z

    features = image_embeddings["features"]
    n_dims = len(features.shape)
    if n_dims not in (4, 5):
        raise ValueError(
            f"Decoder-from-embeddings (3d) requires features with ndim 4 or 5, got {n_dims}."
        )

    n_slices = int(features.shape[0])
    z_block = DEFAULT_TILE_Z if z_block is None else int(z_block)
    z_halo = DEFAULT_HALO_Z if z_halo is None else int(z_halo)
    if z_block < 1 or z_halo < 0:
        raise ValueError(f"z_block must be positive and z_halo non-negative, got {z_block}, {z_halo}.")

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
        if pbar_update is not None:
            pbar_update(count)

    _run_decoder_jobs(
        model, jobs, write_prediction, batch_size=batch_size, devices=devices,
        num_prefetch_workers=num_prefetch_workers,
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


def decode_tiled_2d_embeddings(
    model: torch.nn.Module,
    image_embeddings: Dict,
    device: Device = None,
    pbar_init: Optional[Callable] = None,
    pbar_update: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
) -> np.ndarray:
    """Decode and stitch 2D embedding tiles in automatically sized batches."""
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
        if pbar_update is not None:
            pbar_update(1)

    _run_decoder_jobs(
        model, jobs, write_prediction, batch_size=batch_size, devices=devices,
        num_prefetch_workers=num_prefetch_workers,
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


def decode_tiled_3d_embeddings(
    model: torch.nn.Module,
    image_embeddings: Dict,
    device: Device = None,
    pbar_init: Optional[Callable] = None,
    pbar_update: Optional[Callable] = None,
    z_block: Optional[int] = None,
    z_halo: Optional[int] = None,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
) -> np.ndarray:
    """Batch over both tile columns and z blocks while decoding tiled 3D embeddings."""
    from micro_sam.v2.util import DEFAULT_HALO_Z, DEFAULT_TILE_Z

    features, shape, halo, tiling = _tiled_metadata(image_embeddings, is_3d=True)
    n_slices = shape[0]
    z_block = DEFAULT_TILE_Z if z_block is None else int(z_block)
    z_halo = DEFAULT_HALO_Z if z_halo is None else int(z_halo)
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
        if pbar_update is not None:
            pbar_update(z_count)

    _run_decoder_jobs(
        model, jobs, write_prediction, batch_size=batch_size, devices=devices,
        num_prefetch_workers=num_prefetch_workers,
    )
    return output


def decode_tiled_3d_slice(
    model: torch.nn.Module,
    image_embeddings: Dict,
    index: int,
    device: Device = None,
    pbar_init: Optional[Callable] = None,
    pbar_update: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
) -> np.ndarray:
    """Decode one volume slice across all embedding tiles in batches."""
    features, shape, halo, tiling = _tiled_metadata(image_embeddings, is_3d=True)
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
        if pbar_update is not None:
            pbar_update(1)

    _run_decoder_jobs(
        model, jobs, write_prediction, batch_size=batch_size, devices=devices,
        num_prefetch_workers=num_prefetch_workers,
    )
    return output
