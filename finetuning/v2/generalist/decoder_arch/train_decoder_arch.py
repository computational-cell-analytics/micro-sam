"""Train one arm of the decoder architecture evaluation, or calibrate its speed.

Calibration (all arms of a step in one job; writes <root>/calibration/arm<X>.json):
    python train_decoder_arch.py --arm A --calibrate 200
Training:
    python train_decoder_arch.py --arm A --seed 0 --iterations 24000

Checkpoints: <root>/checkpoints/<run name>/{best,latest}.pt (torch-em layout, with 'decoder_config'),
validation history: <root>/checkpoints/<run name>/validation.csv, TensorBoard: <root>/logs/<run name>.
Import decoder_arch_lib before torch.load of a trainer checkpoint.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
import torch.multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_arch_lib as lib  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(lib.ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--calibrate", type=int, default=None, help="Run this many iterations and report the speed.")
    parser.add_argument("--manifest", default=lib.MANIFEST_PATH)
    parser.add_argument("--save-root", default=lib.EXPERIMENT_ROOT)
    parser.add_argument("--n-workers", type=int, default=14)
    parser.add_argument("--val-workers", type=int, default=4)
    parser.add_argument("--resume", default=None, help="Trainer checkpoint to resume from (a longer run).")
    return parser.parse_args()


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _git_commit():
    try:
        root = os.path.dirname(os.path.abspath(lib.__file__))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def time_loader(loader, n_batches):
    """Seconds per batch of the train loader alone, separately for 2D and 3D batches."""
    times = {2: [], 3: []}
    iterator = iter(loader)
    next(iterator)  # worker start-up
    last = time.perf_counter()
    for _ in range(n_batches):
        x, _ = next(iterator)
        now = time.perf_counter()
        times[2 if x.shape[2] == 1 else 3].append(now - last)
        last = now
    return {f"{k}d_seconds_per_batch": sum(v) / max(1, len(v)) for k, v in times.items()} | {
        f"{k}d_batches": len(v) for k, v in times.items()
    }


def time_gpu_steps(trainer, x, y, n_steps=10):
    """GPU-only seconds per training step (forward, loss, backward, step) on a fixed batch, and peak memory."""
    trainer.model.train()
    x, y = x.to(trainer.device), y.to(trainer.device)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(n_steps + 3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        trainer.optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = trainer._forward_and_loss(x, y)
        loss.backward()
        trainer.optimizer.step()
        torch.cuda.synchronize()
        if step >= 3:
            times.append(time.perf_counter() - started)
    return sum(times) / len(times), torch.cuda.max_memory_allocated() / 2**30


def calibrate(args, trainer, train_loader, device):
    result = {"arm": args.arm, "decoder_type": lib.ARMS[args.arm]["decoder_type"], "iterations": args.calibrate,
              "gpu": torch.cuda.get_device_name(device), "n_workers": args.n_workers}
    model = trainer.model
    result["decoder_parameters_M"] = sum(p.numel() for n, p in model.named_parameters()
                                         if not n.startswith("encoder.")) / 1e6
    result["loader"] = time_loader(train_loader, 60)
    print("[calibrate] loader:", result["loader"], flush=True)

    batches = {}
    for x, y in train_loader:
        batches.setdefault(2 if x.shape[2] == 1 else 3, (x, y))
        if len(batches) == 2:
            break
    for ndim, (x, y) in sorted(batches.items()):
        seconds, peak = time_gpu_steps(trainer, x, y)
        result[f"gpu_{ndim}d"] = {"input_shape": list(x.shape), "seconds_per_step": seconds, "peak_GiB": peak}
        print(f"[calibrate] gpu {ndim}d {tuple(x.shape)}: {seconds:.3f} s/step, peak {peak:.1f} GiB", flush=True)

    # The real loop: loader, training steps and one validation pass.
    model.load_state_dict(lib.build_model(args.arm, "cpu").state_dict())  # undo the timing steps
    trainer.step_timestamps = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(iterations=args.calibrate, overwrite_training=True)
    fit_seconds = time.perf_counter() - started
    stamps = trainer.step_timestamps
    skip = min(50, len(stamps) // 4)
    result["seconds_per_iteration"] = (stamps[-1] - stamps[skip]) / (len(stamps) - 1 - skip)
    result["fit_seconds"] = fit_seconds
    result["peak_GiB_fit"] = torch.cuda.max_memory_allocated() / 2**30
    validation_started = time.perf_counter()
    trainer._validate_mixed()
    result["validation_seconds"] = time.perf_counter() - validation_started
    result["validation_batches"] = len(trainer.val_loader)
    print("[calibrate]", json.dumps(result, indent=1), flush=True)
    out_dir = os.path.join(args.save_root, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"arm{args.arm}.json"), "w") as f:
        json.dump(result, f, indent=1)


def main():
    args = parse_args()
    if (args.iterations is None) == (args.calibrate is None):
        raise SystemExit("Pass exactly one of --iterations and --calibrate.")
    # Python 3.14 starts loader workers through a fork server, which re-imports everything per worker.
    torch.multiprocessing.set_start_method("fork", force=True)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    lib.seed_everything(args.seed)

    manifest = lib.load_manifest(args.manifest)
    train_loader, val_loader = lib.build_loaders(manifest, args.n_workers, args.val_workers)
    print(f"train: {len(train_loader.dataset)} samples in {len(train_loader)} iterations per epoch; "
          f"validation: {len(val_loader.dataset)} samples in {len(val_loader)} batches", flush=True)

    model = lib.build_model(args.arm, device)
    name = f"calibration_arm{args.arm}" if args.calibrate else lib.run_name(args.arm, args.seed)
    total = args.calibrate or args.iterations
    trainer = lib.build_trainer(
        name, model, train_loader, val_loader, total, device, args.save_root,
        aux_weights=lib.ARMS[args.arm].get("aux_weights"),
    )

    if args.calibrate:
        calibrate(args, trainer, train_loader, device)
        return

    checkpoint_dir = os.path.join(args.save_root, "checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    info = {
        "args": vars(args), "arm": lib.ARMS[args.arm], "run_name": name, "manifest_md5": _md5(args.manifest),
        "git_commit": _git_commit(), "code_dir": os.path.dirname(os.path.abspath(lib.__file__)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decoder_config": model.decoder_config,
        "encoder_lr": lib.ENCODER_LR, "decoder_lr": lib.DECODER_LR, "warmup": lib.WARMUP_ITERATIONS,
    }
    with open(os.path.join(checkpoint_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=1)

    started = time.perf_counter()
    if args.resume:
        trainer.fit(iterations=args.iterations, load_from_checkpoint=args.resume)
    else:
        trainer.fit(iterations=args.iterations, overwrite_training=True)
    print(f"training finished after {(time.perf_counter() - started) / 3600:.2f} h; "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
