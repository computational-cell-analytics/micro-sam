"""Calibrate the batch size for 2D and the number of slices for 3D of the decoder training on the current GPU.

    python calibrate_decoder.py sweep    # memory and time per training step for the candidate settings
    python calibrate_decoder.py loop --batch_size_2d <B> --z_slice <Z>    # the full loop with the data loaders

'sweep' takes one batch for 2D and one for 3D from the training loader and repeats their content to the candidate
sizes, so memory and time are measured for training steps on real data. It then selects the largest settings that
stay below the memory limit. 'loop' runs training iterations and a validation pass with the loaders, which shows
whether the loaders keep up, and extrapolates the duration of the training. Without settings it uses the
selection of 'sweep'.
"""

import argparse
import json
import os
import time

import torch

import common

CANDIDATES_2D = (14, 24, 32, 40, 48, 64)
CANDIDATES_3D = (8, 14, 16, 24, 32)


def get_training_step(model_type, decoder, device):
    from micro_sam.v2.training import get_decoder_train_model, get_decoder_trainer

    model = get_decoder_train_model(model_type, decoder=decoder, device=device)
    # The trainer is only used to get the optimizer and the loss with the settings of the training.
    trainer = get_decoder_trainer("calibration", model, None, None, device=device)
    optimizer, loss_function = trainer.optimizer, trainer.loss
    model.train()

    def step(x, y):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(x)
            loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()
        return loss

    def validation_step(x, y):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return loss_function(model(x), y)

    return model, step, validation_step


def _repeat(tensor, size, dim):
    """Repeat the content of a tensor along a dimension to the given size, mirrored so that it stays continuous."""
    pieces, flip = [], False
    while sum(piece.shape[dim] for piece in pieces) < size:
        pieces.append(tensor.flip(dim) if flip else tensor)
        flip = not flip
    return torch.cat(pieces, dim=dim).narrow(dim, 0, size).contiguous()


def measure(step, x, y, n_warmup=3, n_steps=20):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(n_warmup + n_steps):
            if i == n_warmup:
                torch.cuda.synchronize()
                start = time.time()
            loss = step(x, y)
        loss = loss.item()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        return {"oom": True}
    except RuntimeError as e:  # Any other failure ends the candidates as well, but is recorded as such.
        return {"oom": True, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
    seconds = (time.time() - start) / n_steps
    n_slices = x.shape[0] * x.shape[2]
    return {
        "oom": False, "shape": list(x.shape), "seconds_per_step": seconds, "slices_per_second": n_slices / seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30, "loss": loss,
    }


def eligible_z_slices(output):
    """The numbers of slices for which all datasets could be sampled, according to check_data.py."""
    path = os.path.join(common.SAVE_ROOT, "data_check", "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        summary = json.load(f)
    return [int(z) for z, result in summary.items() if result.get("n_failed_datasets") == 0]


def sweep(args):
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    total_gib = properties.total_memory / 2 ** 30
    limit_gib = args.memory_fraction * total_gib
    print(f"GPU: {properties.name}, {total_gib:.1f} GiB, limit {limit_gib:.1f} GiB")

    backbone = args.model_type[:6]
    batch_size_2d, z_slice = common.JOINT_BATCH_SETTINGS[backbone]
    train_loader, _ = common.build_loaders(
        batch_size_2d, z_slice, n_workers=args.n_workers, dataset_choice=args.dataset_choice,
    )
    batches = {}
    for x, y in train_loader:
        batches.setdefault(2 if x.shape[2] == 1 else 3, (x, y))
        if len(batches) == 2:
            break
    del train_loader

    model, step, _ = get_training_step(args.model_type, args.decoder, device)
    results = {"gpu": properties.name, "total_gib": total_gib, "limit_gib": limit_gib, "2d": {}, "3d": {}}
    for key, candidates, dim in (("2d", CANDIDATES_2D, 0), ("3d", CANDIDATES_3D, 2)):
        x, y = batches[int(key[0])]
        for candidate in candidates:
            result = measure(step, _repeat(x, candidate, dim), _repeat(y, candidate, dim))
            results[key][candidate] = result
            print(key, candidate, result)
            if result["oom"]:
                break

    def select(key, allowed=None):
        fitting = [
            candidate for candidate, result in results[key].items()
            if not result["oom"] and result["peak_reserved_gib"] <= limit_gib
            and (allowed is None or candidate in allowed)
        ]
        return max(fitting) if fitting else None

    eligible = eligible_z_slices(args.output)
    results["eligible_z_slices"] = eligible
    results["selected"] = {"batch_size_2d": select("2d"), "z_slice": select("3d", eligible)}
    print("Selected:", results["selected"], "- eligible z:", eligible)
    common.write_json(os.path.join(args.output, f"sweep_{args.decoder}_{args.model_type}.json"), results)
    return results["selected"]


def loop(args, batch_size_2d, z_slice):
    device = torch.device("cuda")
    train_loader, val_loader = common.build_loaders(
        batch_size_2d, z_slice, n_workers=args.n_workers, label_trafo_threads=args.label_trafo_threads,
        dataset_choice=args.dataset_choice,
    )
    train_info = common.describe_loader(train_loader)
    model, step, validation_step = get_training_step(args.model_type, args.decoder, device)

    torch.cuda.reset_peak_memory_stats()
    times = {2: {"wait": [], "step": []}, 3: {"wait": [], "step": []}}
    start = last = time.time()
    for i, (x, y) in enumerate(train_loader):
        if i == args.iterations + args.warmup:
            break
        loaded = time.time()
        step(x, y).item()
        torch.cuda.synchronize()
        done = time.time()
        if i >= args.warmup:
            group = times[2 if x.shape[2] == 1 else 3]
            group["wait"].append(loaded - last)
            group["step"].append(done - loaded)
        else:
            start = done
        last = done
    train_seconds = last - start
    peak_gib = torch.cuda.max_memory_reserved() / 2 ** 30

    name = f"loop_{args.decoder}_{args.model_type}_b{batch_size_2d}_z{z_slice}_w{args.n_workers}.json"
    # The training part is stored before the validation pass, in case the job ends during the validation.
    partial = {
        "batch_size_2d": batch_size_2d, "z_slice": z_slice, "seconds_per_iteration": train_seconds / args.iterations,
        "peak_reserved_gib": peak_gib, "train_loader": train_info,
        "groups": {ndim: {key: sum(values) / max(len(values), 1) for key, values in group.items()}
                   for ndim, group in times.items()},
    }
    common.write_json(os.path.join(args.output, "partial_" + name), partial)
    print(json.dumps(partial, indent=2))

    model.eval()
    start = time.time()
    n_val, val_loss = 0, 0.0
    for x, y in val_loader:
        if args.max_val_batches is not None and n_val == args.max_val_batches:
            break
        val_loss += validation_step(x, y).item()
        n_val += 1
    val_seconds = time.time() - start

    def mean(values):
        return sum(values) / max(len(values), 1)

    # The number of batches per group in an epoch. Incomplete batches are kept.
    n_batches = {}
    for group, info in train_info["groups"].items():
        is_2d = group == "2"
        n_batches[2 if is_2d else 3] = -(-info["samples"] // (batch_size_2d if is_2d else 1))
    per_group = {
        ndim: {
            "measured_steps": len(group["step"]), "step_seconds": mean(group["step"]),
            "wait_seconds": mean(group["wait"]), "batches_per_epoch": n_batches.get(ndim, 0),
        } for ndim, group in times.items()
    }
    epoch_seconds = sum(
        group["batches_per_epoch"] * (group["step_seconds"] + group["wait_seconds"]) for group in per_group.values()
    ) + val_seconds
    n_epochs = common.N_EPOCHS[args.model_type[:6]]
    result = {
        "batch_size_2d": batch_size_2d, "z_slice": z_slice, "n_workers": args.n_workers,
        "label_trafo_threads": args.label_trafo_threads, "iterations": args.iterations,
        "seconds_per_iteration": train_seconds / args.iterations, "peak_reserved_gib": peak_gib,
        "groups": per_group, "validation_seconds": val_seconds, "validation_batches": n_val,
        "validation_loss": val_loss / max(n_val, 1), "batches_per_epoch": len(train_loader),
        "epoch_hours": epoch_seconds / 3600, "n_epochs": n_epochs, "training_hours": n_epochs * epoch_seconds / 3600,
        "epochs_in_48_hours": 48 * 3600 / epoch_seconds,
    }
    print(json.dumps(result, indent=2))
    common.write_json(os.path.join(args.output, name), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["sweep", "loop", "both"])
    parser.add_argument("--model_type", default="hvit_t_cells")
    parser.add_argument("--decoder", default="hiera_u")
    parser.add_argument("--batch_size_2d", type=int, default=None)
    parser.add_argument("--z_slice", type=int, default=None)
    parser.add_argument("--memory_fraction", type=float, default=0.8,
                        help="The fraction of the GPU memory that the selected settings may use.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max_val_batches", type=int, default=None, help="Limit the validation pass, for tests.")
    parser.add_argument("--dataset_choice", default="all")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--label_trafo_threads", type=int, default=6)
    parser.add_argument("--output", default=os.path.join(common.SAVE_ROOT, "calibration"))
    args = parser.parse_args()
    common.set_start_method()

    selected = {"batch_size_2d": args.batch_size_2d, "z_slice": args.z_slice}
    if args.mode in ("sweep", "both"):
        selected = sweep(args)
    if args.mode in ("loop", "both"):
        if args.mode == "loop" and None in selected.values():
            path = os.path.join(args.output, f"sweep_{args.decoder}_{args.model_type}.json")
            with open(path) as f:
                swept = json.load(f)["selected"]
            selected = {key: swept[key] if value is None else value for key, value in selected.items()}
        loop(args, selected["batch_size_2d"], selected["z_slice"])


if __name__ == "__main__":
    main()
