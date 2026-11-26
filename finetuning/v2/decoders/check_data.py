"""Check that the training data of the cells model is available and yields valid targets for the decoder training.

For each number of slices the loaders of the joint training are built, the datasets per group are counted and
samples are drawn from every dataset of the training and validation split.

    python check_data.py --z_slices 14 --n_workers 16
    python check_data.py --z_slices 8 14 16 24 32 --only_3d_after_first    # which z do all 3D datasets support
"""

import argparse
import os
import time
import traceback

import numpy as np
import torch

import common


def dataset_label(dataset):
    """A short description of the data source of a dataset, from the first raw path that can be found."""
    seen = 0
    while seen < 8:
        seen += 1
        for attribute in ("raw_path", "raw_paths", "image_paths", "path", "paths"):
            value = getattr(dataset, attribute, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                value = value[0]
            return os.path.relpath(str(value), common.DATA_ROOT)
        inner = getattr(dataset, "ds", None) or getattr(dataset, "dataset", None)
        if inner is None and getattr(dataset, "datasets", None):
            inner = dataset.datasets[0]
        if inner is None:
            break
        dataset = inner
    return type(dataset).__name__


class Probe(torch.utils.data.Dataset):
    """Draw samples from given positions of a dataset and summarize them. Errors are reported, not raised."""
    def __init__(self, dataset, positions):
        self.dataset, self.positions = dataset, positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        dataset_id, position = self.positions[index]
        result = {"dataset": dataset_id}
        start = time.time()
        try:
            x, y = self.dataset[position]
            x, y = np.asarray(x), np.asarray(y)
            result.update(
                x_shape=list(x.shape), y_shape=list(y.shape), x_min=float(x.min()), x_max=float(x.max()),
                finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),
                n_instances=int(len(np.unique(y[0])) - 1),
                foreground=float((y[1] > 0).mean()), ignored=float((y[1] < 0).mean()),
                distance_range=[float(y[2:5].min()), float(y[2:5].max())],
                boundary=float((y[5] > 0).mean()),
            )
        except Exception:
            result["error"] = traceback.format_exc(limit=3)
        result["seconds"] = time.time() - start
        return result


def check_loader(loader, z_slice, n_samples, n_workers, only_3d):
    concat = loader.dataset
    positions, labels, offset = [], [], 0
    for dataset_id, dataset in enumerate(concat.datasets):
        labels.append({"label": dataset_label(dataset), "group": str(dataset.group_key), "samples": len(dataset)})
        if not (only_3d and dataset.group_key == 2):
            step = max(1, len(dataset) // n_samples)
            positions.extend((dataset_id, offset + k * step) for k in range(min(n_samples, len(dataset))))
        offset += len(dataset)

    probe_loader = torch.utils.data.DataLoader(
        Probe(concat, positions), batch_size=None, num_workers=n_workers, collate_fn=lambda sample: sample,
    )
    # The target channels: instance ids, foreground, three directed distances, boundaries.
    expected = {2: [6, 1, 512, 512], 3: [6, z_slice, 512, 512]}
    for result in probe_loader:
        entry = labels[result.pop("dataset")]
        ndim = 2 if entry["group"] == "2" else 3
        if "error" not in result:
            if result["y_shape"] != expected[ndim] or result["x_shape"] != [3, *expected[ndim][1:]]:
                result["error"] = f"Unexpected shapes {result['x_shape']} {result['y_shape']}"
            elif not result["finite"]:
                result["error"] = "The sample contains values that are not finite."
            elif not (-1e-4 <= result["x_min"] and result["x_max"] <= 1 + 1e-4):
                result["error"] = f"The raw data is outside of [0, 1]: [{result['x_min']}, {result['x_max']}]"
            elif result["foreground"] == 0:
                result["error"] = "The sample has no foreground."
        entry.setdefault("draws", []).append(result)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--z_slices", type=int, nargs="+", default=[14])
    parser.add_argument("--dataset_choice", default="all")
    parser.add_argument("--n_samples", type=int, default=2, help="The number of samples to draw per dataset.")
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--only_3d_after_first", action="store_true",
                        help="Draw only from the 3D datasets for all but the first number of slices.")
    parser.add_argument("--output", default=os.path.join(common.SAVE_ROOT, "data_check"))
    args = parser.parse_args()
    common.set_start_method()

    summary = {}
    for i, z_slice in enumerate(args.z_slices):
        start = time.time()
        report = {"z_slice": z_slice}
        try:
            train_loader, val_loader = common.build_loaders(
                14, z_slice, n_workers=0, label_trafo_threads=1, dataset_choice=args.dataset_choice,
            )
        except Exception:
            report["error"] = traceback.format_exc(limit=5)
            print(f"z = {z_slice}: the loaders cannot be built\n{report['error']}")
            common.write_json(os.path.join(args.output, f"data_check_z{z_slice}.json"), report)
            summary[z_slice] = {"build_error": True}
            continue

        only_3d = args.only_3d_after_first and i > 0
        failed = []
        for split, loader in (("train", train_loader), ("val", val_loader)):
            report[f"{split}_loader"] = common.describe_loader(loader)
            report[split] = check_loader(loader, z_slice, args.n_samples, args.n_workers, only_3d)
            for entry in report[split]:
                errors = [draw["error"] for draw in entry.get("draws", []) if "error" in draw]
                if errors:
                    failed.append((split, entry["label"], entry["group"], errors[0].strip().splitlines()[-1]))
        report["failed"] = failed
        common.write_json(os.path.join(args.output, f"data_check_z{z_slice}.json"), report)

        seconds = [draw["seconds"] for split in ("train", "val") for entry in report[split]
                   for draw in entry.get("draws", [])]
        summary[z_slice] = {
            "train": report["train_loader"], "val": report["val_loader"], "n_failed_datasets": len(failed),
            "mean_seconds_per_sample": float(np.mean(seconds)), "max_seconds_per_sample": float(np.max(seconds)),
        }
        print(f"z = {z_slice}: {time.time() - start:.0f} s, train {report['train_loader']}")
        print(f"z = {z_slice}: val {report['val_loader']}")
        print(f"z = {z_slice}: {len(failed)} datasets with failed draws")
        for failure in failed:
            print("   ", *failure)
    common.write_json(os.path.join(args.output, "summary.json"), summary)


if __name__ == "__main__":
    main()
