"""Worker processes that propagate a volume's prompts, one per inference device.

Volumetric prompt generation spends about ninety percent of its time propagating candidates, and that
loop is Python-heavy enough that threads do not scale over devices: four propagation threads keep
four A100s about half busy, because one interpreter lock is what they queue on. Four *processes* on
the same four devices each run at the speed they reach alone (measured within 1.5%), so the
propagation is handed to processes instead.

A worker receives nothing live. It rebuilds the model from the recipe that built it
(`get_sam2_model`'s `build_kwargs`) and reads the embeddings from the store they were written to, so
the only things that cross a process boundary are the volume, once per volume, and the candidates
and mask records of a job. When any of that is missing - one device, embeddings held in memory, a
model built by hand - `build_pool` returns None and the caller keeps its thread path.
"""

import os
import queue
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch

# Model attributes the workers mirror. They change what the model computes but are set on the instance
# rather than passed to its constructor, so rebuilding from 'build_kwargs' alone would lose them.
MIRRORED_MODEL_ATTRIBUTES = ("num_maskmem",)

STOP = "stop"
DONE = "done"


def _available_cpus() -> int:
    """The cores this process may actually run on.

    'os.cpu_count' reports the machine, not the allocation, so on a scheduler that pins a job to a
    subset of the cores it oversubscribes every worker by the ratio between the two.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Not POSIX.
        return os.cpu_count() or 1


def model_overrides(model) -> Dict[str, Any]:
    """The instance-level model settings a worker has to reproduce, see MIRRORED_MODEL_ATTRIBUTES."""
    return {name: getattr(model, name) for name in MIRRORED_MODEL_ATTRIBUTES if hasattr(model, name)}


def _worker_devices(devices: Sequence) -> List[str]:
    """The devices as strings, which is what survives a process boundary unambiguously."""
    return [str(torch.device(device)) for device in devices]


def _build_worker_propagator(model, setup: Dict[str, Any], device: str):
    """Rebuild one worker's tiled propagator over the volume it was given."""
    from micro_sam.v2.util import precompute_image_embeddings
    from micro_sam.v2.prompt_based_segmentation import TiledPromptableSegmentation3D

    # The store is complete, so this reads it rather than encoding anything.
    embeddings = precompute_image_embeddings(
        model, setup["volume"], save_path=setup["embedding_path"], ndim=3,
        tile_shape=setup["tile_shape"], halo=setup["halo"], verbose=False, lazy_loading=True,
        devices=device,
    )
    return TiledPromptableSegmentation3D(
        model, setup["volume"], embeddings, devices=device,
        offload_state_to_cpu=setup["offload_state_to_cpu"], max_cached_frames=setup["max_cached_frames"],
    )


def _take_while_parented(channel, parent: int, poll: float = 5.0):
    """The next item from a queue, or None once the parent process is gone.

    A worker outlives a parent that was killed rather than closed - 'daemon' only covers a normal
    interpreter shutdown, not a signal - and a blocking 'get' would hold its share of a device for as
    long as it waited. So the wait polls, and reparenting to init ends it.
    """
    while True:
        try:
            return channel.get(timeout=poll)
        except queue.Empty:
            if os.getppid() != parent:
                return None


def _run_job(propagator, job: Tuple, n_slices: int) -> Tuple:
    """Propagate one tile's run of passes, exactly as the in-process path does."""
    from micro_sam.v2.automatic_prompt_generation import propagate_passes

    _, tile_id, passes, early_stop_patience = job
    return propagate_passes(propagator, tile_id, passes, early_stop_patience, n_slices)


def _worker_main(build_kwargs, overrides, device, n_threads, commands, jobs, results, worker_id):
    """Build the model once, then serve one volume at a time and the jobs that go with it."""
    from micro_sam.v2.util import get_sam2_model

    try:
        torch.set_num_threads(n_threads)
        model = get_sam2_model(device=device, **build_kwargs)
        for name, value in overrides.items():
            setattr(model, name, value)
    except Exception:
        results.put((worker_id, None, traceback.format_exc()))
        return

    parent = os.getppid()
    while True:
        command = _take_while_parented(commands, parent)
        if command is None or command == STOP:
            break

        try:
            propagator = _build_worker_propagator(model, command, device)
            n_slices = int(command["volume"].shape[0])
            results.put((worker_id, None, None))
        except Exception:
            results.put((worker_id, None, traceback.format_exc()))
            continue

        while True:
            job = _take_while_parented(jobs, parent)
            if job is None or job == DONE:
                break
            try:
                results.put((job[0], _run_job(propagator, job, n_slices), None))
            except Exception:
                results.put((worker_id, None, traceback.format_exc()))
                break

        # The volume's states go, the model stays: the next volume reuses it.
        propagator.reset_predictor()
        propagator = None


class PropagationPool:
    """One propagation worker process per inference device, reused across volumes.

    The model is the expensive part of starting a worker, so a pool is built once and kept: a volume
    is loaded into the workers with `set_volume`, its jobs run with `map_jobs`, and the next volume
    replaces it. `close` ends them.

    Args:
        build_kwargs: The arguments that rebuild the video predictor, from `model.build_kwargs`.
        devices: The devices to run one worker on each.
        overrides: Instance-level model settings to reapply after rebuilding, see `model_overrides`.
        n_threads: CPU threads per worker. By default the cores this process may use, split evenly.
    """

    def __init__(
        self, build_kwargs: Dict[str, Any], devices: Sequence, overrides: Optional[Dict[str, Any]] = None,
        n_threads: Optional[int] = None,
    ):
        import multiprocessing

        self._devices = _worker_devices(devices)
        self._context = multiprocessing.get_context("spawn")
        if n_threads is None:
            n_threads = max(1, _available_cpus() // len(self._devices))

        self._results = self._context.Queue()
        self._jobs = self._context.Queue()
        self._commands = [self._context.Queue() for _ in self._devices]
        self._workers = [
            self._context.Process(
                target=_worker_main,
                args=(
                    build_kwargs, overrides or {}, device, n_threads,
                    self._commands[index], self._jobs, self._results, index,
                ),
                daemon=True,
            )
            for index, device in enumerate(self._devices)
        ]
        for worker in self._workers:
            worker.start()
        self._loaded = False

    def _take(self):
        """The next message from a worker, or a failure if one died without sending one."""
        while True:
            try:
                return self._results.get(timeout=1.0)
            except queue.Empty:
                if not any(worker.is_alive() for worker in self._workers):
                    raise RuntimeError("Every propagation worker exited before returning a result.")

    def set_volume(
        self, volume: np.ndarray, embedding_path: str, tile_shape: Sequence[int], halo: Sequence[int],
        offload_state_to_cpu: Optional[bool], max_cached_frames: Optional[int],
    ) -> None:
        """Give every worker the volume to propagate, replacing the previous one."""
        setup = {
            "volume": volume, "embedding_path": str(embedding_path),
            "tile_shape": tuple(int(s) for s in tile_shape), "halo": tuple(int(s) for s in halo),
            "offload_state_to_cpu": offload_state_to_cpu, "max_cached_frames": max_cached_frames,
        }
        for command in self._commands:
            command.put(setup)

        for _ in self._workers:
            worker_id, _, error = self._take()
            if error is not None:
                raise RuntimeError(f"Propagation worker {worker_id} failed to load the volume:\n{error}")
        self._loaded = True

    def map_jobs(self, jobs: List[Tuple], early_stop_patience: Optional[int]) -> List:
        """Run one job per tile-and-passes, returning the results in the order the jobs were given."""
        if not self._loaded:
            raise RuntimeError("The pool has no volume. Call 'set_volume' first.")

        for index, (tile_id, _, passes) in enumerate(jobs):
            self._jobs.put((index, tile_id, passes, early_stop_patience))
        for _ in self._workers:
            self._jobs.put(DONE)
        self._loaded = False

        if not jobs:
            return []

        results = [None] * len(jobs)
        for _ in jobs:
            index, payload, error = self._take()
            if error is not None:
                raise RuntimeError(f"Propagation worker {index} failed:\n{error}")
            results[index] = payload
        return results

    def close(self) -> None:
        """End every worker and release the queues they were served through."""
        for command in self._commands:
            try:
                command.put(STOP)
            except (ValueError, OSError):  # The queue is already closed.
                pass
        for worker in self._workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
        for channel in [*self._commands, self._jobs, self._results]:
            channel.close()
            channel.join_thread()
        self._workers = []
        self._commands = []
        self._loaded = False

    def __del__(self):
        if getattr(self, "_workers", None):
            self.close()


def build_pool(
    model, devices: Sequence, n_threads: Optional[int] = None, n_worker_processes: Optional[int] = None,
) -> Optional[PropagationPool]:
    """A pool for this model, or None when the propagation has to stay in this process.

    Args:
        model: The SAM2 video predictor the workers rebuild.
        devices: The inference devices, one worker each.
        n_threads: CPU threads per worker.
        n_worker_processes: The requested number of workers. None uses every device when at least two exist.

    Returns:
        The pool, or None when automatic selection has one device or the model has no build recipe.
    """
    devices = list(devices)
    if n_worker_processes is not None:
        if n_worker_processes < 0 or n_worker_processes > len(devices):
            raise ValueError(
                f"The worker process count {n_worker_processes} must be between 0 and {len(devices)}."
            )
        if n_worker_processes == 0:
            return None
        devices = devices[:n_worker_processes]

    build_kwargs = getattr(model, "build_kwargs", None)
    if build_kwargs is None or (n_worker_processes is None and len(devices) < 2):
        return None
    if not all(torch.device(device).type == "cuda" for device in devices):
        return None
    return PropagationPool(build_kwargs, devices, overrides=model_overrides(model), n_threads=n_threads)
