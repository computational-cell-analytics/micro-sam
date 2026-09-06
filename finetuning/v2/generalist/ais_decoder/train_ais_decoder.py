"""Train one AIS decoder variant (decoder only, encoder frozen at the v4 joint weights).

Example (smoke test on the session GPU, then a full run):
    python train_ais_decoder.py --variant contact --smoke 30 --batch-size 4 --n-workers 1
    python train_ais_decoder.py --variant contact --iterations 30000 --batch-size 8 --n-workers 12

Checkpoints: <save-root>/checkpoints/ais_decoder_<variant>/{best,latest}.pt (torch_em layout), the file
lists behind the loaders: <save-root>/checkpoints/ais_decoder_<variant>/data_manifest.json.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ais_decoder_lib as lib  # noqa: E402

from micro_sam.util import training_autocast_dtype  # noqa: E402
from micro_sam.v2.datasets.util import check_loader  # noqa: E402
from micro_sam.v2.loss import DirectedDistanceLoss  # noqa: E402
from micro_sam.v2.training.sam2_trainer import UniSAM2Logger, UniSAM2Trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(lib.VARIANTS))
    parser.add_argument("--iterations", type=int, default=None, help="Training iterations (required unless --smoke).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-workers", type=int, default=12, help="Train loader workers.")
    parser.add_argument("--val-workers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epoch-scale", type=float, default=1.0,
                        help="Multiplier of the per-dataset samples per epoch (base ~1450 samples).")
    parser.add_argument("--save-root", default=lib.CAMPAIGN_ROOT)
    parser.add_argument("--data-root", default=lib.DATA_ROOT)
    parser.add_argument("--init-checkpoint", default=lib.V4_CHECKPOINT, help="Joint checkpoint to warm-start from.")
    parser.add_argument("--no-warm-start", action="store_true", help="Random decoder init (not used in the campaign).")
    parser.add_argument("--name", default=None, help="Checkpoint name, default ais_decoder_<variant>.")
    parser.add_argument("--log-image-interval", type=int, default=100)
    parser.add_argument("--resume", default=None, help="Trainer checkpoint to resume from.")
    parser.add_argument("--smoke", type=int, default=None,
                        help="Smoke test: time the loader, run this many iterations plus one validation, report.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_trainer(args, model, train_loader, val_loader, device, name):
    settings = lib.VARIANTS[args.variant]
    loss = DirectedDistanceLoss(
        mask_distances_in_bg=True, contact=settings["contact"], boundary_weight=settings["boundary_weight"],
        boundary_radius=lib.BOUNDARY_RADIUS,
    )
    optimizer = torch.optim.AdamW(lib.decoder_parameters(model), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10)
    return UniSAM2Trainer(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader, loss=loss, metric=loss,
        optimizer=optimizer, device=device, lr_scheduler=scheduler,
        mixed_precision=training_autocast_dtype(device) is not None, mixed_precision_dtype="bfloat16",
        early_stopping=None, log_image_interval=args.log_image_interval, logger=UniSAM2Logger, logger_kwargs=None,
        id_=None, save_root=args.save_root, compile_model=False, rank=None,
    )


def time_loader(loader, n_batches):
    started = time.perf_counter()
    n_samples = 0
    for index, (x, y) in enumerate(loader):
        n_samples += x.shape[0]
        if index + 1 >= n_batches:
            break
    seconds = time.perf_counter() - started
    return seconds, n_samples


def time_gpu_step(trainer, x, y, n_steps):
    """The GPU-only cost of one training step on a fixed batch (forward, loss, backward, optimizer step)."""
    trainer.model.train()
    x, y = x.to(trainer.device), y.to(trainer.device)
    dtype = torch.bfloat16 if trainer.mixed_precision else None
    times = []
    for step in range(n_steps + 3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        trainer.optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            prediction = trainer.model(x)
            loss = trainer.loss(prediction, y)
        loss.backward()
        trainer.optimizer.step()
        torch.cuda.synchronize()
        if step >= 3:
            times.append(time.perf_counter() - started)
    return sum(times) / len(times), float(loss.detach())


def main():
    args = parse_args()
    if args.iterations is None and args.smoke is None:
        raise SystemExit("Pass --iterations or --smoke.")
    torch.set_num_threads(2)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = args.name or (f"smoke_{args.variant}" if args.smoke else f"ais_decoder_{args.variant}")
    n_channels = lib.n_output_channels(args.variant)
    print(f"variant {args.variant}: {n_channels} output channels, loss settings {lib.VARIANTS[args.variant]}")

    train_loader, val_loader, manifest = lib.build_loaders(
        args.variant, args.data_root, args.batch_size, args.n_workers, args.val_workers, scale=args.epoch_scale,
    )
    print(f"train: {len(train_loader.dataset)} samples per epoch in {len(train_loader)} iterations of "
          f"{args.batch_size}; validation: {len(val_loader.dataset)} samples")
    for dataset, lists in manifest.items():
        print(f"  {dataset:22s} train files {len(lists['train']):5d}  val files {len(lists['val']):3d}")

    unetr_state = None
    if not args.no_warm_start:
        unetr_state = lib.load_lean_v4_states(args.init_checkpoint, args.save_root)["unetr_state"]
    model = lib.build_model(args.variant, device, unetr_state)
    n_trainable = sum(p.numel() for p in lib.decoder_parameters(model))
    n_frozen = sum(p.numel() for p in model.encoder.parameters())
    print(f"model: {n_trainable / 1e6:.2f} M trainable decoder parameters, "
          f"{n_frozen / 1e6:.2f} M frozen encoder parameters")

    trainer = build_trainer(args, model, train_loader, val_loader, device, name)
    checkpoint_dir = os.path.join(args.save_root, "checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "data_manifest.json"), "w") as f:
        json.dump({"variant": args.variant, "args": vars(args), "datasets": manifest}, f, indent=2)

    if args.smoke:
        check_loader(train_loader, n_samples=3, n_target_channels=n_channels)
        seconds, n_samples = time_loader(train_loader, n_batches=max(2, args.smoke // 5))
        print(f"[smoke] loader: {n_samples / seconds:.2f} samples/s with {args.n_workers} workers "
              f"({seconds / max(1, n_samples // args.batch_size):.2f} s per batch of {args.batch_size})")
        x, y = next(iter(train_loader))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        step_seconds, loss_value = time_gpu_step(trainer, x, y, n_steps=max(5, args.smoke // 3))
        print(f"[smoke] gpu step: {step_seconds:.3f} s per iteration at batch {args.batch_size} "
              f"(loss {loss_value:.4f})")
        if device.type == "cuda":
            print(f"[smoke] peak memory after gpu steps: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB "
                  f"allocated, {torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB reserved")
        started = time.perf_counter()
        trainer.fit(iterations=args.smoke, overwrite_training=True)
        fit_seconds = time.perf_counter() - started
        print(f"[smoke] fit: {args.smoke} iterations + validation ({len(val_loader)} batches) in {fit_seconds:.1f} s")
        if device.type == "cuda":
            print(f"[smoke] peak memory overall: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB allocated, "
                  f"{torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB reserved")
        return

    started = time.perf_counter()
    trainer.fit(iterations=args.iterations, overwrite_training=args.resume is None, load_from_checkpoint=args.resume)
    print(f"training finished after {(time.perf_counter() - started) / 3600:.2f} h")
    if device.type == "cuda":
        print(f"[peak-memory] {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB allocated, "
              f"{torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB reserved")


if __name__ == "__main__":
    main()
