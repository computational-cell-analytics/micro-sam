"""Shared settings of the decoder training scripts."""

import json
import os
import subprocess

import torch

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
SAVE_ROOT = os.environ.get(
    "DECODER_SAVE_ROOT", "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training"
)

# The settings of the automatic segmentation branch in the joint training (finetuning/v2/generalist/train_joint.py),
# except for the learning rate: the decoder is trained from scratch, with the rate of the decoder experiments.
LEARNING_RATE = 1e-4
BOUNDARY_DICE_WEIGHT = 0.5
N_EPOCHS = {"hvit_t": 75, "hvit_s": 73, "hvit_b": 82, "hvit_l": 79}
# The batch size for 2D and the number of slices for 3D of the joint training, per GPU.
JOINT_BATCH_SETTINGS = {"hvit_t": (14, 14), "hvit_s": (12, 14), "hvit_b": (10, 10), "hvit_l": (8, 8)}


def set_start_method():
    """Use fork for the loader workers. Python 3.14 changed the default, which the datasets do not support."""
    torch.multiprocessing.set_start_method("fork", force=True)


def run_name(model_type, decoder):
    return f"{decoder}_{model_type}"


def build_loaders(batch_size_2d, z_slice, batch_size_3d=1, dataset_choice="all", n_workers=8, label_trafo_threads=6):
    """The loaders of the joint training (micro_sam.v2.training.train_joint_sam2).

    The targets have the channels instance ids, foreground, three directed distances (geodesic) and boundaries.
    The decoder training uses all but the first one, like the automatic segmentation branch of the joint training.
    Each validation dataset is limited to 50 samples.

    Returns:
        The training loader and the validation loader.
    """
    from micro_sam.v2.datasets.generalist_loader import _build_joint_datasets, _prepare_data_loader

    train_ds, val_ds = _build_joint_datasets(
        DATA_ROOT, [z_slice], dataset_choice, "geodesic", with_boundaries=True,
        label_trafo_threads=label_trafo_threads,
    )
    batch_size_per_group = {2: batch_size_2d} if batch_size_2d != batch_size_3d else None
    train_loader = _prepare_data_loader(
        train_ds, batch_size=batch_size_3d, shuffle=True, batch_size_per_group=batch_size_per_group,
        num_workers=n_workers,
    )
    val_loader = _prepare_data_loader(
        val_ds, batch_size=batch_size_3d, shuffle=False, batch_size_per_group=batch_size_per_group,
        num_workers=n_workers, deterministic=True,
    )
    return _without_persistent_workers(train_loader), val_loader


def _without_persistent_workers(loader):
    """Rebuild a loader so that its workers are restarted in every epoch.

    The host memory of the workers grows while they draw samples (the datasets take about 14 GB in the main
    process, which the forked workers copy step by step, plus per-dataset state). With persistent workers this
    accumulates over the epochs: the first training job ran out of memory (160 GB) in its fourth epoch.
    """
    import torch

    if loader.num_workers == 0 or not loader.persistent_workers:
        return loader
    shuffle = getattr(loader, "shuffle", None)
    if loader.batch_sampler is not None and loader.batch_size is None:  # Built with a batch sampler.
        new_loader = torch.utils.data.DataLoader(
            loader.dataset, batch_sampler=loader.batch_sampler, num_workers=loader.num_workers,
            pin_memory=loader.pin_memory, collate_fn=loader.collate_fn, worker_init_fn=loader.worker_init_fn,
            persistent_workers=False,
        )
    else:
        new_loader = torch.utils.data.DataLoader(
            loader.dataset, batch_size=loader.batch_size, sampler=loader.sampler, drop_last=loader.drop_last,
            num_workers=loader.num_workers, pin_memory=loader.pin_memory, collate_fn=loader.collate_fn,
            worker_init_fn=loader.worker_init_fn, persistent_workers=False,
        )
    if shuffle is not None:
        new_loader.shuffle = shuffle
    return new_loader


def log_host_memory(interval=900):
    """Print the host memory of this process and its children (the loader workers) every 'interval' seconds."""
    import threading
    import time

    import psutil

    def _log():
        process = psutil.Process()
        while True:
            time.sleep(interval)
            try:
                children = process.children(recursive=True)
                # PSS counts shared pages only once over all processes, unlike RSS.
                pss = sum(p.memory_full_info().pss for p in [process] + children) / 2 ** 30
                rss = process.memory_info().rss / 2 ** 30
                print(f"[memory] {time.strftime('%Y-%m-%d %H:%M:%S')} total pss {pss:.1f} GiB, "
                      f"main rss {rss:.1f} GiB, {len(children)} children", flush=True)
            except Exception as e:  # E.g. a worker that ended in between.
                print(f"[memory] failed: {e}", flush=True)

    threading.Thread(target=_log, daemon=True).start()


def describe_loader(loader):
    """Count the datasets, samples, batches and slices per group (2D or 3D with z slices) of a loader."""
    from micro_sam.v2.datasets.sampler import _build_group_map

    dataset = loader.dataset
    groups = {}
    for ds in dataset.datasets:
        group = groups.setdefault(str(ds.group_key), {"datasets": 0, "samples": 0})
        group["datasets"] += 1
        group["samples"] += len(ds)
    assert len(_build_group_map(dataset)) == sum(group["samples"] for group in groups.values())
    return {"groups": groups, "batches": len(loader), "samples": len(dataset)}


def _git_commit(folder):
    try:
        return subprocess.run(
            ["git", "-C", folder, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def code_versions():
    import micro_sam
    import torch_em
    return {
        "micro_sam": _git_commit(os.path.dirname(micro_sam.__file__)),
        "torch_em": _git_commit(os.path.dirname(torch_em.__file__)),
        "torch": torch.__version__,
    }


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
