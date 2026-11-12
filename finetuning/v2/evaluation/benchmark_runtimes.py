"""Runtime benchmark of micro-sam2 and its references on CPU, GPU and Apple silicon (Extended Data Figure 2).

The benchmark times the evaluation pipelines of 'evaluate_automatic_segmentation.py' (AIS and APG with the library
defaults) and of 'evaluate_interactive_segmentation.py' (first prompt only) for the four '_cells' models of
micro-sam2, and the automatic references CellposeSAM (LIVECell, GoNuclear) and SegNeuron (CREMI).

Each sample is timed once, after one untimed warm-up run per loaded model. The parts are:
- automatic: 'encoder' (image embeddings), 'decoder' (UniSAM2 decoder), 'postprocessing' (instances from the
  decoder output; for APG also the prompting of SAM2 with the decoder seeds). The references have 'network'
  instead of 'encoder' and 'decoder'. 'total' is the time of the whole call, so 'total' minus the parts is the
  time spent outside of them.
- interactive: 'encoder' once per sample ('other' in 3d: the set-up of the video predictor state on the
  embeddings), and the 'prompting' time of each object from its first prompt (a point or a box): one mask in 2d,
  the propagation through all slices in 3d. The objects of a sample are capped at MAX_OBJECTS,
  chosen at random with a fixed seed.

Every sample is written to its own JSON file, with the device, the host and the per-object times, so that the
numbers can be regrouped later without a rerun ('collect' writes them into CSV tables).

Usage:

    # Once, on the cluster: write the samples, so that every device times the same inputs.
    python benchmark_runtimes.py export
    # One group (dataset, method, model, prompt) on one device:
    python benchmark_runtimes.py run --device_tag a100 -d livecell --method apg -m hvit_t
    # All groups of a device in turn (e.g. on a Mac, see 'run_runtimes_mac.sh'):
    python benchmark_runtimes.py run --device_tag m2 --device mps --all
    python benchmark_runtimes.py collect
"""

import os
import sys
import json
import time
import glob
import socket
import argparse
import platform
import contextlib

import h5py
import numpy as np
import torch

ROOT = os.environ.get(
    "RUNTIME_BENCHMARK_ROOT", "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/runtime_benchmark"
)
MODEL_DIR = os.environ.get("MICRO_SAM2_MODEL_DIR", os.path.expanduser("~/.cache/micro_sam/models"))
DATASETS = {"livecell": 2, "gonuclear": 3, "cremi": 3}
MAX_SAMPLES = 50
MAX_OBJECTS = 50
# The number of slices of the 3d warm-up, per device type. The CPU needs few, so that a volume fits a 1 h job.
WARMUP_SLICES = {"cuda": 8, "cpu": 2}
SIZES = ["hvit_t", "hvit_s", "hvit_b", "hvit_l"]
REFERENCES = {"livecell": "cellposesam", "gonuclear": "cellposesam", "cremi": "segneuron"}
INTERACTIVE = ["interactive_point", "interactive_box"]


def groups():
    """All (dataset, method, model) groups of one device, the GPU-heavy ones first."""
    result = []
    for dataset in DATASETS:
        for size in reversed(SIZES):
            for method in ["apg", "ais"] + INTERACTIVE:
                result.append((dataset, method, size))
        result.append((dataset, REFERENCES[dataset], "default"))
    return result


#
# Timing
#


class Timer:
    """Wall-clock timer that waits for the device before reading the clock."""

    def __init__(self, device):
        self.device = str(device)

    def sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device.startswith("mps"):
            torch.mps.synchronize()

    @contextlib.contextmanager
    def __call__(self, times, key):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            times[key] = times.get(key, 0.0) + time.perf_counter() - start


def timed_method(owner, name, timer, times, key):
    """Replace 'owner.name' by a version that adds its run time to 'times[key]'. Returns the original."""
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        with timer(times, key):
            return original(*args, **kwargs)

    setattr(owner, name, wrapper)
    return original


def environment(device, threads):
    """The device and the host a number was measured on."""
    info = {
        "host": socket.gethostname(), "platform": platform.platform(), "processor": platform.processor(),
        "torch": torch.__version__, "device": str(device), "threads": threads,
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu_model"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu_model"] = platform.processor()
    if platform.system() == "Darwin":
        import subprocess
        info["cpu_model"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    if str(device).startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name()
    elif str(device).startswith("mps"):
        info["gpu"] = info["cpu_model"]
    return info


#
# Data
#


def sample_paths(dataset):
    return sorted(glob.glob(os.path.join(ROOT, "data", dataset, "sample_*.h5")))


# In 3d, all methods are timed on the same three volumes on every device: APG takes up to 4 h per volume on the
# CPU. The volumes are those with the A100 APG time closest to the mean over all volumes (averaged over the sizes):
# their mean is within 13% of the mean over all volumes for every size.
VOLUMES_3D = {
    "gonuclear": ["sample_0009.h5", "sample_0012.h5", "sample_0014.h5"],
    "cremi": ["sample_0001.h5", "sample_0003.h5", "sample_0004.h5"],
}
# Groups that are timed on fewer volumes. CREMI APG L needs more than 12 h per volume on CPU, so only one volume.
VOLUMES_3D_GROUP = {
    ("cpu", "cremi", "apg", "hvit_l"): ["sample_0001.h5"],
    # CREMI is not in the figure; its CPU APG B runs are not completed.
    ("cpu", "cremi", "apg", "hvit_b"): ["sample_0001.h5", "sample_0003.h5"],
}
# Devices that are timed on fewer volumes. The M2 takes about 1 h per volume for APG, so only the volume with the
# object count closest to the mean of the three.
VOLUMES_3D_DEVICE = {
    ("m2", "gonuclear"): ["sample_0014.h5"],
}


def group_paths(device_tag, dataset, method, model=None):
    """The samples that are timed for a group: all in 2d, the volumes of VOLUMES_3D in 3d.

    VOLUMES_3D_GROUP and then VOLUMES_3D_DEVICE replace the volumes of VOLUMES_3D for a group or for a device.
    """
    paths = sample_paths(dataset)
    volumes = VOLUMES_3D_GROUP.get(
        (device_tag, dataset, method, model), VOLUMES_3D_DEVICE.get((device_tag, dataset), VOLUMES_3D.get(dataset))
    )
    if volumes is not None:
        paths = [p for p in paths if os.path.basename(p) in volumes]
    return paths


def read_sample(path):
    with h5py.File(path, "r") as f:
        return f["raw"][:], f["labels"][:]


def export():
    """Write the first MAX_SAMPLES test samples with labels of each dataset, as the evaluation loads them."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import DATA_ROOT, load_data
    for dataset, ndim in DATASETS.items():
        folder = os.path.join(ROOT, "data", dataset)
        os.makedirs(folder, exist_ok=True)
        n_written = len(sample_paths(dataset))
        if n_written >= MAX_SAMPLES:
            continue
        written = 0
        for sample_id, (raw, labels, valid_roi) in enumerate(load_data(dataset, DATA_ROOT, ndim)):
            if labels.max() == 0:
                continue
            path = os.path.join(folder, f"sample_{sample_id:04d}.h5")
            if not os.path.exists(path):
                with h5py.File(path + ".tmp", "w") as f:
                    f.create_dataset("raw", data=raw, compression="gzip")
                    f.create_dataset("labels", data=labels.astype("uint32"), compression="gzip")
                    f.attrs["sample_id"] = sample_id
                os.replace(path + ".tmp", path)
            written += 1
            if written == MAX_SAMPLES:
                break
        print(f"{dataset}: {written} samples in '{folder}'.")


#
# Methods. Each runner is built once per group and returns the times of one sample.
#


def _common():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common
    common.use_tmpdir_for_embeddings()
    return common


def automatic_runner(method, size, dataset, device, timer):
    """AIS or APG of micro-sam2, in the pipeline of 'evaluate_automatic_segmentation.py' with the defaults."""
    common = _common()
    ndim = DATASETS[dataset]
    decoder = os.path.join(MODEL_DIR, f"{size}_cells_decoder")
    import micro_sam.v2.automatic_prompt_generation as apg_module
    from micro_sam.v2.instance_segmentation import UniSAM2InstanceSegmentation
    if method == "apg":
        model = common.build_model(
            "apg", size, device, ndim, checkpoint_path=decoder,
            interactive_checkpoint_path=os.path.join(MODEL_DIR, f"{size}_cells"), devices=[device],
        )
        spacing = common.DATASET_SPACING.get(dataset)

        def run(raw):
            times = {}
            # The encoder is the embedding of the image or of the volume, the decoder the UniSAM2 decoder on it.
            originals = [
                (model, "_encode", timed_method(model, "_encode", timer, times, "encoder")),
                (apg_module, "precompute_image_embeddings",
                 timed_method(apg_module, "precompute_image_embeddings", timer, times, "encoder")),
                (UniSAM2InstanceSegmentation, "initialize",
                 timed_method(UniSAM2InstanceSegmentation, "initialize", timer, times, "decoder")),
            ]
            try:
                model.clear_state()
                with timer(times, "total"), timer(times, "initialize"):
                    model.initialize(raw, ndim=ndim, **(common.VOLUME_SPEED_OPTIONS if ndim == 3 else {}))
                with timer(times, "total"), timer(times, "generate"):
                    seg = model.generate(**({"spacing": spacing} if ndim == 3 else {}))
            finally:
                for owner, name, original in originals:
                    if owner is model:
                        del model._encode  # The instance attribute hid the bound method.
                    else:
                        setattr(owner, name, original)
            # The rest of 'initialize' builds the video predictor that 'generate' prompts, so it goes with it.
            times["postprocessing"] = times["generate"] + times["initialize"] - times["encoder"] - times["decoder"]
            model.clear_state()
            return times, int(len(np.unique(seg)) - 1)
        return run

    # AIS runs like the first stage of APG: the embeddings of the image or of the volume (once per slice, untiled),
    # then the UniSAM2 decoder on them, then the AIS postprocessing.
    from micro_sam.util import make_temp_embedding_path
    model = common.build_model(
        "apg", size, device, ndim, checkpoint_path=decoder,
        interactive_checkpoint_path=os.path.join(MODEL_DIR, f"{size}_cells"), devices=[device],
    )

    def run(raw):
        times = {}
        path = make_temp_embedding_path() if ndim == 3 else None
        try:
            model.clear_state()
            with timer(times, "total"):
                with timer(times, "encoder"):
                    if ndim == 3:
                        # As in 'AutomaticPromptGenerator._initialize_volume'.
                        embeddings = apg_module.precompute_image_embeddings(
                            model._video_predictor, raw, save_path=path, ndim=3, verbose=False,
                            lazy_loading=True, devices=model._inference_devices(None),
                        )
                    else:
                        embeddings = model._encode(raw)
                with timer(times, "decoder"):
                    UniSAM2InstanceSegmentation.initialize(model, raw, ndim=ndim, image_embeddings=embeddings)
                with timer(times, "postprocessing"):
                    seg = common.postprocess_unisam2(model.get_state()["prediction"], dataset, model_type=size)
        finally:
            model.clear_state()
            if path is not None:
                import shutil
                shutil.rmtree(path, ignore_errors=True)
        return times, int(len(np.unique(seg)) - 1)
    return run


def cellposesam_runner(dataset, device, timer):
    """CellposeSAM as in 'evaluate_automatic_baselines.py': 2d, or 'do_3D' for a volume."""
    from cellpose import models
    gpu = not str(device).startswith("cpu")
    model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam", device=torch.device(device) if gpu else None)
    ndim = DATASETS[dataset]

    def run(raw):
        times = {}
        originals = [
            ("_run_net", timed_method(model, "_run_net", timer, times, "network")),
            ("_compute_masks", timed_method(model, "_compute_masks", timer, times, "postprocessing")),
        ]
        try:
            with timer(times, "total"):
                if ndim == 3:
                    masks = model.eval(raw, diameter=None, do_3D=True, z_axis=0)[0]
                else:
                    masks = model.eval(raw, diameter=None)[0]
        finally:
            for name, _ in originals:
                delattr(model, name)
        return times, int(len(np.unique(masks)) - 1)
    return run


def segneuron_runner(dataset, device, timer):
    """SegNeuron as in 'evaluate_automatic_baselines.py': the network on blocks, then its multicut."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import evaluate_automatic_baselines as baselines
    checkpoint = os.environ.get("SEGNEURON_CHECKPOINT", baselines.SEGNEURON_CHECKPOINT)
    if os.environ.get("SEGNEURON_ROOT"):
        baselines.SEGNEURON_ROOT = os.environ["SEGNEURON_ROOT"]
    model = baselines._load_segneuron(checkpoint, device)

    def run(raw):
        times = {}
        original = timed_method(baselines, "_segneuron_multicut", timer, times, "postprocessing")
        try:
            with timer(times, "total"):
                seg = baselines._segment_segneuron(raw, model, device)
        finally:
            baselines._segneuron_multicut = original
        times["network"] = times["total"] - times["postprocessing"]
        return times, int(len(np.unique(seg)) - 1)
    return run


def pick_objects(ids, seed=0):
    """At most MAX_OBJECTS of the object ids, chosen at random with a fixed seed, in increasing order."""
    ids = np.asarray(ids)
    if len(ids) <= MAX_OBJECTS:
        return ids
    return np.sort(np.random.default_rng(seed).choice(ids, MAX_OBJECTS, replace=False))


def interactive_runner(method, size, dataset, device, timer):
    """The first prompt of 'evaluate_interactive_segmentation.py', one object at a time."""
    from micro_sam.v1.evaluation.inference import _get_batched_prompts
    from micro_sam.v2.normalization import normalize_raw
    from micro_sam.v2.util import encode_image, get_sam2_image_predictor, get_sam2_model, precompute_image_embeddings
    from micro_sam.v2.evaluation.inference import _embedding_tensors_to_numpy, _extract_prompts_per_object
    from bioimage_cpp.segmentation import label as connected_components

    use_boxes = method == "interactive_box"
    prompt_kwargs = dict(use_points=not use_boxes, use_boxes=use_boxes, n_positive=0 if use_boxes else 1)
    checkpoint = os.path.join(MODEL_DIR, f"{size}_cells")
    ndim = DATASETS[dataset]

    def to_uint8(raw):
        return normalize_raw(raw, output_dtype="uint8")

    if ndim == 2:
        predictor = get_sam2_image_predictor(get_sam2_model(model_type=size, device=device, checkpoint_path=checkpoint))

        def run(raw, labels):
            times = {}
            # As 'run_interactive_segmentation_2d': a grayscale image is stacked to RGB, the labels are relabelled.
            image = to_uint8(raw).astype("uint8")
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            labels = connected_components(labels).astype("uint32")
            np.random.seed(0)
            gt_ids = pick_objects(np.setdiff1d(np.unique(labels), [0]))
            points, point_labels, boxes = _get_batched_prompts(
                gt=labels, gt_ids=gt_ids, use_points=not use_boxes, use_boxes=use_boxes,
                n_positives=prompt_kwargs["n_positive"], n_negatives=0, dilation=5,
            )
            with timer(times, "encoder"):
                encode_image(predictor, image)
            object_times = []
            for i in range(len(gt_ids)):
                one = {}
                with timer(one, "object"):
                    predictor.predict(
                        point_coords=None if use_boxes else points[i:i + 1],
                        point_labels=None if use_boxes else point_labels[i:i + 1],
                        box=boxes[i:i + 1] if use_boxes else None, multimask_output=False, return_logits=True,
                    )
                object_times.append(one["object"])
            return times, object_times, 1
        return run

    predictor = get_sam2_model(model_type=size, device=device, checkpoint_path=checkpoint, input_type="videos")
    predictor.add_all_frames_to_correct_as_cond = True

    def run(raw, labels):
        times = {}
        volume = np.stack([to_uint8(frame) for frame in raw])
        labels = connected_components(labels).astype(labels.dtype)
        gt_ids = [int(i) for i in pick_objects(np.setdiff1d(np.unique(labels), [0]))]
        with timer(times, "encoder"):
            embeddings = precompute_image_embeddings(predictor=predictor, input_=volume, ndim=3)
            embeddings = _embedding_tensors_to_numpy(embeddings)
        # The video predictor state holds the embeddings for the propagation: the 'other' part of the encoding.
        with timer(times, "other"):
            state = predictor.init_state(volume=volume, volume_embeddings=embeddings)
        np.random.seed(0)
        prompts = _extract_prompts_per_object(
            labels=labels, gt_ids=gt_ids, first_prompt_at_center=True, dilation=5, **prompt_kwargs,
        )
        object_times = []
        with torch.no_grad():
            for obj_id, (z, point, label, box) in prompts.items():
                predictor.reset_state(state)
                one = {}
                with timer(one, "object"):
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=z, obj_id=obj_id, points=point, labels=label, box=box,
                    )
                    n_frames = 0
                    for _, _, logits in predictor.propagate_in_video(state):
                        (logits > 0.0).cpu().numpy()
                        n_frames += 1
                    if n_frames < volume.shape[0]:
                        for _, _, logits in predictor.propagate_in_video(state, reverse=True):
                            (logits > 0.0).cpu().numpy()
                object_times.append(one["object"])
        return times, object_times, int(volume.shape[0])
    return run


#
# Running
#


def result_path(device_tag, dataset, method, model, sample_path):
    name = os.path.basename(sample_path).replace(".h5", ".json")
    return os.path.join(ROOT, "results", device_tag, dataset, f"{method}_{model}", name)


def release_memory(device):
    """Empty the MPS cache between samples, untimed: without a watermark it grows into swap over the samples."""
    if str(device).startswith("mps"):
        import gc
        gc.collect()
        torch.mps.empty_cache()


def run_group(device_tag, device, dataset, method, model, deadline, threads, max_samples=None):
    """Time all samples of one group that have no result yet. Stops before 'deadline' (time.time())."""
    paths = group_paths(device_tag, dataset, method, model)[:max_samples]
    paths = [p for p in paths if open_sample(result_path(device_tag, dataset, method, model, p))]
    if not paths:
        return True
    # A job that has not enough time left for any sample of this group leaves it to a longer job.
    shortest = min(expected_seconds(device_tag, dataset, method, model, p) for p in paths)
    if time.time() + FIT_MARGIN * shortest > deadline:
        print(f"Skipping the group: one sample takes about {shortest / 60:.0f} min.", flush=True)
        return True
    timer = Timer(device)
    if method in ("ais", "apg"):
        runner, interactive = automatic_runner(method, model, dataset, device, timer), False
    elif method == "cellposesam":
        runner, interactive = cellposesam_runner(dataset, device, timer), False
    elif method == "segneuron":
        runner, interactive = segneuron_runner(dataset, device, timer), False
    else:
        runner, interactive = interactive_runner(method, model, dataset, device, timer), True
    info = environment(device, threads)

    def call(raw, labels):
        return runner(raw, labels) if interactive else runner(raw)

    # The warm-up loads the weights to the device and compiles the kernels; its times are not kept. For 3d, it runs
    # on the first slices of a volume only, which use the same kernels at a fraction of the time.
    raw, labels = read_sample(paths[0])
    if raw.ndim == 3:
        n_slices = WARMUP_SLICES.get(str(device), WARMUP_SLICES["cuda"])
        raw, labels = raw[:n_slices], labels[:n_slices]
    call(raw, labels)
    release_memory(device)
    for path in paths:
        # A sample that does not fit in the time left is left to a longer job; a shorter one may still fit.
        if time.time() + FIT_MARGIN * expected_seconds(device_tag, dataset, method, model, path) > deadline:
            print(f"Leaving {os.path.basename(path)} to a longer job.", flush=True)
            continue
        out = result_path(device_tag, dataset, method, model, path)
        started = out + ".started"
        # Several jobs can run the same group: each sample is claimed by one job.
        if not claim_sample(out):
            continue
        start = time.time()
        raw, labels = read_sample(path)
        record = {
            "wall_seconds": None, "dataset": dataset, "method": method, "model": model, "device_tag": device_tag,
            "sample": os.path.basename(path), "shape": list(raw.shape), **info,
        }
        if interactive:
            times, object_times, n_slices = call(raw, labels)
            record.update({
                "times": times, "object_times": object_times, "n_slices": n_slices,
                "n_objects_total": int(len(np.setdiff1d(np.unique(labels), [0]))),
            })
        else:
            times, n_objects = call(raw, labels)
            record.update({"times": times, "n_objects": n_objects})
        if str(device).startswith("mps"):
            record["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
            record["mps_driver_gb"] = torch.mps.driver_allocated_memory() / 1e9
        release_memory(device)
        record["wall_seconds"] = time.time() - start
        with open(out + ".tmp", "w") as f:
            json.dump(record, f, indent=1)
        os.replace(out + ".tmp", out)
        os.remove(started)
        print(f"{dataset} {method} {model} {os.path.basename(path)}: {json.dumps(times)}", flush=True)
    return True


def _owner():
    return os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")


def _started_owner(started):
    try:
        with open(started) as f:
            return f.read().strip()
    except OSError:
        return None


def open_sample(out):
    """Whether a sample has no result and no job works on it.

    A sample whose job ended without a result took longer than that job had left. Short jobs leave it to the long
    jobs, so that it does not restart in every short job.
    """
    if os.path.exists(out):
        return False
    started = out + ".started"
    if not os.path.exists(started):
        return True
    if _job_alive(_started_owner(started)):
        return False
    return os.environ.get("RUNTIME_SHORT_JOB") != "1"


def claim_sample(out):
    """Claim a sample for this job with an atomic create. A claim of a job that no longer runs is taken over."""
    started = out + ".started"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        fd = os.open(started, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if not open_sample(out):
            return False
        # The claim of an ended job: take it over.
        with open(started, "w") as f:
            f.write(_owner())
        return True
    with os.fdopen(fd, "w") as f:
        f.write(_owner())
    return True


# For a group without results on the CPU yet, the time of a sample is estimated from the A100 times. APG measured a
# CPU / A100 ratio of 41 to 50 for T to B on GoNuclear, and the ratio grows with the model size.
CPU_SLOWDOWN = 70
# A job starts a sample only if this multiple of its expected time is left.
FIT_MARGIN = 1.15


def expected_seconds(device_tag, dataset, method, model, path):
    """The expected wall time of one sample, to start it only in a job with enough time left.

    On the CPU, it is the A100 time of the sample times the largest CPU / A100 ratio of the samples of the group with
    both times (CPU_SLOWDOWN if there are none). Else, it is the longest wall time of a sample of the group so far.
    """
    times = _wall_seconds(device_tag, dataset, method, model)
    if device_tag == "cpu":
        gpu_times = _wall_seconds("a100", dataset, method, model)
        sample = os.path.basename(path)
        if sample in gpu_times:
            ratios = [times[name] / gpu_times[name] for name in times if gpu_times.get(name)]
            return max(ratios, default=CPU_SLOWDOWN) * gpu_times[sample]
        if not times:
            return CPU_SLOWDOWN * max(gpu_times.values(), default=0.0)
    return max(times.values(), default=0.0)


def _wall_seconds(device_tag, dataset, method, model):
    """The wall time of each finished sample of a group, by the file name of the sample."""
    times = {}
    for path in glob.glob(os.path.join(ROOT, "results", device_tag, dataset, f"{method}_{model}", "*.json")):
        try:
            with open(path) as f:
                record = json.load(f)
        except (OSError, ValueError):
            continue
        seconds = record.get("wall_seconds")
        if seconds is None:
            parts = record["times"]
            seconds = parts.get("total", sum(parts.values())) + sum(record.get("object_times", []))
        times[os.path.splitext(os.path.basename(path))[0] + ".h5"] = seconds
    return times


def claim(device_tag, group):
    """Claim a group for this job with an atomic mkdir. A lock of a job that no longer runs is taken over."""
    lock = os.path.join(ROOT, "locks", device_tag, "__".join(group))
    owner = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    try:
        os.mkdir(lock)
    except FileExistsError:
        try:
            with open(os.path.join(lock, "owner")) as f:
                previous = f.read().strip()
        except OSError:
            return None
        if previous == owner or _job_alive(previous):
            return None
    with open(os.path.join(lock, "owner"), "w") as f:
        f.write(owner)
    return lock


def _job_alive(job):
    if not job or job.startswith("local"):
        return False
    import subprocess
    out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True).stdout
    return bool(out.strip())


def group_done(device_tag, group):
    dataset, method, model = group
    return all(
        os.path.exists(result_path(device_tag, dataset, method, model, p))
        for p in group_paths(device_tag, dataset, method, model)
    )


def run(args):
    threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    torch.set_num_threads(threads)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    deadline = time.time() + args.seconds if args.seconds else float("inf")
    if not args.all:
        todo = [(args.dataset_name, args.method, args.model_type)]
    else:
        todo = [g for g in groups() if not args.only or g[1] in args.only or g[0] in args.only]
    # First the groups that no job runs, then, with --all, the groups of other jobs that still have open samples:
    # the jobs share the samples of a group.
    passes = [(group, True) for group in todo] + ([(group, False) for group in todo] if args.all else [])
    for group, own in passes:
        if group_done(args.device_tag, group):
            continue
        lock = claim(args.device_tag, group) if args.all and own else None
        if args.all and own and lock is None:
            continue
        print(f"Group {group} on {device} ({args.device_tag}).", flush=True)
        try:
            finished = run_group(args.device_tag, device, *group, deadline, threads, args.max_samples)
        except Exception:
            # One failing group must not end the job: the error is logged, and the next group runs.
            import traceback
            traceback.print_exc()
            os.makedirs(os.path.join(ROOT, "failed"), exist_ok=True)
            with open(os.path.join(ROOT, "failed", f"{args.device_tag}__{'__'.join(group)}.txt"), "a") as f:
                f.write(traceback.format_exc())
            finished = True
        finally:
            if lock is not None:
                import shutil
                shutil.rmtree(lock, ignore_errors=True)
        if not finished:
            return


def collect():
    """Write the per-sample parts and the per-object times of all devices into two CSV tables."""
    import pandas as pd
    samples, objects = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "*", "*", "*", "*.json"))):
        with open(path) as f:
            record = json.load(f)
        # All timed samples are kept; 'selected' marks the samples of the figure (see VOLUMES_3D).
        selected = [os.path.basename(p) for p in group_paths(
            record["device_tag"], record["dataset"], record["method"], record["model"])]
        keys = ["device_tag", "dataset", "method", "model", "sample", "host", "cpu_model", "gpu", "threads"]
        base = {key: record.get(key) for key in keys}
        base["selected"] = record["sample"] in selected
        for part, seconds in record["times"].items():
            samples.append({**base, "part": part, "seconds": seconds,
                            "n_objects": record.get("n_objects", len(record.get("object_times", [])))})
        for k, seconds in enumerate(record.get("object_times", [])):
            objects.append({**base, "object": k, "seconds": seconds, "n_slices": record["n_slices"],
                            "seconds_per_slice": seconds / record["n_slices"]})
    pd.DataFrame(samples).to_csv(os.path.join(ROOT, "runtimes_per_sample.csv"), index=False)
    pd.DataFrame(objects).to_csv(os.path.join(ROOT, "runtimes_per_object.csv"), index=False)
    print(f"{len(samples)} part rows and {len(objects)} object rows in '{ROOT}'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "run", "collect", "status"])
    parser.add_argument("--device_tag", help="The name of the device in the results, e.g. 'cpu', 'a100', 'm2'.")
    parser.add_argument("--device", default=None, help="The torch device. Default: cuda if available, else cpu.")
    parser.add_argument("-d", "--dataset_name", choices=list(DATASETS))
    parser.add_argument("--method", choices=["ais", "apg", "cellposesam", "segneuron"] + INTERACTIVE)
    parser.add_argument("-m", "--model_type", default="default")
    parser.add_argument("--all", action="store_true", help="Run all open groups, claiming each with a lock.")
    parser.add_argument("--only", nargs="*", help="With --all: only the groups of these methods or datasets.")
    parser.add_argument("--threads", type=int, default=None, help="The torch threads. Default: the Slurm CPUs.")
    parser.add_argument("--max_samples", type=int, default=None, help="Time only the first samples, for a check.")
    parser.add_argument("--seconds", type=float, default=None, help="Stop starting new samples after this time.")
    args = parser.parse_args()
    if args.command == "export":
        export()
    elif args.command == "collect":
        collect()
    elif args.command == "status":
        # The number of open groups of the device, for the job loops.
        print(sum(not group_done(args.device_tag, group) for group in groups()))
    else:
        if args.device_tag is None:
            parser.error("'run' needs --device_tag.")
        run(args)


if __name__ == "__main__":
    main()
