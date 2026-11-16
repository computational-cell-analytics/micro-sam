"""Evaluate one trained run of the decoder architecture comparison (plan section 6).

    python evaluate_decoder_arch.py place-val-crops                  # once, before any run is evaluated
    python evaluate_decoder_arch.py predict-val --run armA_current_seed0     # GPU
    python evaluate_decoder_arch.py tune --run armA_current_seed0 --workers 16   # CPU
    python evaluate_decoder_arch.py test --run armA_current_seed0 --workers 14   # GPU + CPU workers
    python report_decoder_arch.py

Outputs go to '<eval root>/<run>/': val_cache/ (decoder outputs of the validation data), tuning/ (score matrices,
rankings, best_params.json), test/ (one row per test sample and dataset, decode_benchmark.json). Every stage resumes
where it stopped. '--limit N' restricts every dataset to its first N samples for a check; use it with a separate
'--eval-root'.
"""

import argparse
import csv
import os
import sys
import time
from concurrent import futures

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_arch_lib as lib  # noqa: E402
import eval_lib as ev  # noqa: E402


def run_dir(args):
    return os.path.join(args.eval_root, args.run)


def checkpoint_path(args):
    return args.checkpoint or os.path.join(lib.EXPERIMENT_ROOT, "checkpoints", args.run, "best.pt")


def checkpoint_identity(path):
    from common import checkpoint_checksum
    return {"checkpoint": os.path.abspath(path), "checksum": checkpoint_checksum(path)}


def make_pool(workers):
    """A fork pool whose workers all start now, before the parent initializes CUDA."""
    import multiprocessing
    pool = futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"))
    list(pool.map(_noop, range(workers)))
    return pool


def _noop(_):
    return None


#
# place-val-crops
#


def place_val_crops_command(args):
    if os.path.exists(args.val_crops) and not args.force:
        raise SystemExit(f"{args.val_crops} exists; every run must use the same crops (--force to replace).")
    crops = ev.place_val_crops(lib.load_manifest(args.manifest))
    ev.save_json({"manifest": os.path.abspath(args.manifest), "crop_shape": list(ev.VAL_CROP_SHAPE),
                  "max_per_volume": ev.MAX_VAL_CROPS_PER_VOLUME, "datasets": crops}, args.val_crops)
    for dataset, items in crops.items():
        print(f"{dataset:24s} {len(items)} crops, objects {[c['n_objects'] for c in items]}")
    print(f"wrote {args.val_crops}")


#
# predict-val
#


def cache_path(args, dataset, index):
    return os.path.join(run_dir(args), "val_cache", dataset, f"{index:04d}.npz")


def val_specs(args, ndim):
    specs = ev.val_samples(lib.load_manifest(args.manifest), ev.load_json(args.val_crops)["datasets"], ndim)
    if args.limit:
        specs = {d: s[:args.limit] for d, s in specs.items()}
    return specs


def predict_val_command(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    identity = checkpoint_identity(checkpoint_path(args))
    index_path = os.path.join(run_dir(args), "val_cache", "index.json")
    if os.path.exists(index_path) and ev.load_json(index_path)["checksum"] != identity["checksum"]:
        raise SystemExit(f"{index_path} was predicted with another checkpoint; remove the cache first.")
    model = ev.load_model(identity["checkpoint"], device)
    counts = {}
    for ndim in args.ndim:
        for dataset, specs in val_specs(args, ndim).items():
            started, n_new = time.perf_counter(), 0
            for index, spec in enumerate(specs):
                path = cache_path(args, dataset, index)
                if os.path.exists(path):
                    continue
                raw, labels, valid_roi = ev.load_val_sample(spec)
                prediction = ev.predict(model, raw, ndim, device)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
                np.savez(tmp, prediction=prediction.astype("float16"), labels=labels,
                         valid_roi=np.zeros(0, bool) if valid_roi is None else valid_roi)
                os.replace(tmp, path)
                n_new += 1
            counts[dataset] = len(specs)
            print(f"{dataset}: {len(specs)} samples ({n_new} predicted, {time.perf_counter() - started:.0f} s)")
    previous = ev.load_json(index_path)["datasets"] if os.path.exists(index_path) else {}
    ev.save_json({**identity, "datasets": {**previous, **counts}, "limit": args.limit}, index_path)


#
# tune
#


def _score_cached(path, params_list, border_min_size, spacing):
    data = np.load(path)
    valid_roi = data["valid_roi"] if data["valid_roi"].size else None
    labels = data["labels"]
    if labels.max() == 0:
        return None
    return ev.score_grid(data["prediction"].astype("float32"), labels, params_list, valid_roi=valid_roi,
                         border_min_size=border_min_size, spacing=spacing)


def flow_chunks(params_list, n_chunks):
    """Split the combination indices into at most 'n_chunks' chunks of about equal cost that never split a flow
    setting, so that no two chunks compute the same flow density.

    The cost of a flow setting is estimated from a profile of a 32 x 512 x 512 crop: about 0.055 s per integration
    step for the density and 0.5 s per combination for the watershed, the size filter and the scoring.
    """
    groups = {}
    for i, params in enumerate(params_list):
        groups.setdefault(tuple(params[k] for k in ev.FLOW_KEYS), []).append(i)
    n_chunks = max(1, min(n_chunks, len(groups)))
    chunks, costs = [[] for _ in range(n_chunks)], [0.0] * n_chunks
    ranked = sorted(groups.items(), key=lambda item: -(0.055 * item[0][2] + 0.5 * len(item[1])))
    for key, members in ranked:  # longest first, each to the cheapest chunk so far
        c = int(np.argmin(costs))
        chunks[c] += members
        costs[c] += 0.055 * key[2] + 0.5 * len(members)
    return [sorted(chunk) for chunk in chunks]


def score_round(pool, args, ndim, datasets, params_list, path):
    """The (combinations x samples) mSA matrix of every dataset for one list of combinations, saved to 'path'.

    Every sample is split into chunks of flow settings, enough to give every worker about two jobs.
    """
    from common import DATASET_SPACING, GT_MIN_SIZE_2D
    if os.path.exists(path):
        data = np.load(path, allow_pickle=False)
        return {d: data[d] for d in datasets}
    started = time.perf_counter()
    n_samples = sum(datasets.values())
    chunks = flow_chunks(params_list, -(-2 * args.workers // max(1, n_samples)))
    jobs = {}
    for dataset, n in datasets.items():
        border = GT_MIN_SIZE_2D.get(dataset, 0) if ndim == 2 else 0
        spacing = DATASET_SPACING.get(dataset)
        for index in range(n):
            for chunk in chunks:
                future = pool.submit(_score_cached, cache_path(args, dataset, index),
                                     [params_list[i] for i in chunk], border, spacing)
                jobs[future] = (dataset, index, chunk)
    columns = {d: {} for d in datasets}
    for done, future in enumerate(futures.as_completed(jobs), 1):
        dataset, index, chunk = jobs[future]
        scores = future.result()
        if scores is not None:
            column = columns[dataset].setdefault(index, np.full(len(params_list), np.nan))
            column[chunk] = scores
        if done % max(1, len(jobs) // 10) == 0:
            print(f"  {done}/{len(jobs)} jobs ({time.perf_counter() - started:.0f} s)", flush=True)
    matrices = {
        d: np.stack([cols[i] for i in sorted(cols)], axis=1) if cols else np.zeros((len(params_list), 0))
        for d, cols in columns.items()
    }
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **matrices)
    os.replace(tmp, path)
    return matrices


def tune_command(args):
    tuning = os.path.join(run_dir(args), "tuning")
    os.makedirs(tuning, exist_ok=True)
    index = ev.load_json(os.path.join(run_dir(args), "val_cache", "index.json"))
    best_path = os.path.join(tuning, "best_params.json")
    best = ev.load_json(best_path) if os.path.exists(best_path) else {}
    pool = make_pool(args.workers)
    for ndim in args.ndim:
        datasets = {d: index["datasets"][d] for d in ev.VAL_DATASETS[ndim]}
        params_list, matrices = [], {d: [] for d in datasets}
        new = ev.grid_params(ndim, small=args.grid == "small")
        for round_index in range(ev.MAX_EXTENSION_ROUNDS + 1):
            round_path = os.path.join(tuning, f"{ndim}d_round{round_index}.npz")
            params_path = round_path[:-4] + "_params.json"
            if os.path.exists(round_path) and ev.load_json(params_path) != new:
                raise RuntimeError(f"{round_path} holds other combinations; remove the tuning folder.")
            ev.save_json(new, params_path)
            print(f"{ndim}D round {round_index}: {len(new)} combinations on {sum(datasets.values())} samples")
            scores = score_round(pool, args, ndim, datasets, new, round_path)
            params_list += new
            for d in datasets:
                matrices[d].append(scores[d])
            stacked = {d: np.concatenate(m, axis=0) for d, m in matrices.items()}
            winner, relative, means, group_means = ev.select_params(stacked, ev.SELECTION_GROUPS[ndim])
            swept = {k: [p[k] for p in params_list] for k in ev.PARAM_KEYS}
            seen = {ev.params_key(p) for p in params_list}
            new = [p for p in ev.edge_extension(params_list[winner], swept) if ev.params_key(p) not in seen]
            if not new:
                break
        default_index = [ev.params_key(p) for p in params_list].index(ev.params_key(ev.default_params(ndim)))
        at_edge = [
            k for k in ev.PARAM_KEYS
            if len(set(swept[k])) > 1 and params_list[winner][k] in (min(swept[k]), max(swept[k]))
        ]
        best[f"{ndim}d"] = {
            "params": params_list[winner],
            "relative_score": float(relative[winner]),
            "val_msa": {d: float(means[i, winner]) for i, d in enumerate(datasets)},
            "val_group_msa": {g: float(group_means[i, winner]) for i, g in enumerate(ev.SELECTION_GROUPS[ndim])},
            "default_params": params_list[default_index],
            "default_relative_score": float(relative[default_index]),
            "default_val_msa": {d: float(means[i, default_index]) for i, d in enumerate(datasets)},
            "default_val_group_msa": {
                g: float(group_means[i, default_index]) for i, g in enumerate(ev.SELECTION_GROUPS[ndim])},
            "n_samples": {d: int(stacked[d].shape[1]) for d in datasets},
            "n_combinations": len(params_list),
            "rounds": round_index + 1,
            "at_grid_edge": at_edge,
            "grid": args.grid,
        }
        order = np.argsort(-relative)[:args.top]
        with open(os.path.join(tuning, f"{ndim}d_ranking.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([*ev.PARAM_KEYS, "relative", *datasets])
            for i in order:
                writer.writerow([params_list[i][k] for k in ev.PARAM_KEYS] + [f"{relative[i]:.5f}"]
                                + [f"{means[j, i]:.5f}" for j in range(len(datasets))])
        print(f"{ndim}D best (relative {relative[winner]:.4f}, defaults {relative[default_index]:.4f}): "
              f"{params_list[winner]}; at grid edge: {at_edge or 'none'}")
        for i, d in enumerate(datasets):
            print(f"  {d:24s} tuned {means[i, winner]:.4f}  defaults {means[i, default_index]:.4f}")
    pool.shutdown()
    best.update({"checkpoint": index["checkpoint"], "checksum": index["checksum"], "limit": index.get("limit")})
    ev.save_json(best, best_path)


#
# test
#


TEST_DIAGNOSTICS = ("fg_iou", "boundary_recall")
TEST_PER_PARAMS = ("msa", "sa50", "sa75", "merged", "split", "missed", "n_pred", "seconds_post")


def test_fields(param_sets):
    return (["sample", "skipped", "n_gt", "seconds_predict", *TEST_DIAGNOSTICS]
            + [f"{k}_{name}" for name in param_sets for k in TEST_PER_PARAMS])


def _score_test_sample(index, prediction, labels, valid_roi, param_sets, border_min_size, spacing, seconds_predict):
    row = {"sample": index, "skipped": 0, "n_gt": int(len(np.unique(labels[labels > 0]))),
           "seconds_predict": seconds_predict}
    row.update(ev.field_diagnostics(prediction, labels, valid_roi))
    for name, params in param_sets.items():
        started = time.perf_counter()
        seg = ev.mask_segmentation(ev.flow_segmentation(prediction, params, spacing=spacing), valid_roi,
                                   border_min_size)
        scores = ev.msa_scores(seg, labels)
        errors = ev.object_errors(seg, labels, valid_roi)
        row.update({f"{k}_{name}": v for k, v in scores.items()})
        row.update({f"{k}_{name}": errors[k] for k in ("merged", "split", "missed")})
        row[f"n_pred_{name}"] = int(len(np.unique(seg[seg > 0])))
        row[f"seconds_post_{name}"] = time.perf_counter() - started
    return row


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def test_dataset(pool, model, device, args, dataset, param_sets, identity):
    from common import DATASET_SPACING, GT_MIN_SIZE_2D
    out_dir = os.path.join(run_dir(args), "test")
    final, partial = os.path.join(out_dir, f"{dataset}.csv"), os.path.join(out_dir, f"{dataset}.partial.csv")
    meta_path = os.path.join(out_dir, f"{dataset}.json")
    if os.path.exists(final):
        print(f"{dataset}: done")
        return
    total = ev.n_test_samples(dataset)
    wanted = list(range(total if not args.limit else min(total, args.limit)))
    done = {int(r["sample"]) for r in read_rows(partial)}
    todo = [i for i in wanted if i not in done]
    ndim = ev.ndim_of(dataset)
    border = GT_MIN_SIZE_2D.get(dataset, 0) if ndim == 2 else 0
    spacing = DATASET_SPACING.get(dataset) if ndim == 3 else None
    print(f"{dataset}: {len(todo)} of {len(wanted)} samples to score", flush=True)
    started = time.perf_counter()
    fields = test_fields(param_sets)
    if os.path.exists(partial):
        with open(partial) as f:
            if next(csv.reader(f)) != fields:
                raise RuntimeError(f"{partial} was written with other parameter sets; remove it.")
    pending = set()

    def write(rows):
        if not rows:
            return
        exists = os.path.exists(partial)
        with open(partial, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def drain(block):
        nonlocal pending
        finished, pending = futures.wait(
            pending, return_when=futures.ALL_COMPLETED if block else futures.FIRST_COMPLETED)
        write([future.result() for future in finished])

    os.makedirs(out_dir, exist_ok=True)
    for index, (raw, labels, valid_roi) in ev.test_samples(dataset, todo):
        if labels.max() == 0:
            write([{"sample": index, "skipped": 1, "n_gt": 0}])
            continue
        t0 = time.perf_counter()
        prediction = ev.predict(model, raw, ndim, device)
        seconds = time.perf_counter() - t0
        pending.add(pool.submit(
            _score_test_sample, index, prediction, labels, valid_roi, param_sets, border, spacing, seconds))
        while len(pending) >= args.max_pending:
            drain(block=False)
    while pending:
        drain(block=True)

    all_rows = read_rows(partial)
    rows = [r for r in all_rows if r["skipped"] != "1"]
    scored = sorted({int(r["sample"]) for r in all_rows})
    if scored != wanted:
        raise RuntimeError(f"{dataset}: scored {len(scored)} of {len(wanted)} samples.")
    rows.sort(key=lambda r: int(r["sample"]))
    with open(final + ".tmp", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in fields if k != "skipped"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(final + ".tmp", final)
    os.remove(partial)
    ev.save_json({**identity, "dataset": dataset, "n_samples": total, "n_scored": len(rows),
                  "n_without_labels": len(wanted) - len(rows), "limit": args.limit, "param_sets": param_sets,
                  "seconds": time.perf_counter() - started}, meta_path)
    msa = {name: np.mean([float(r[f"msa_{name}"]) for r in rows]) for name in param_sets}
    print(f"{dataset}: {len(rows)} samples, " + ", ".join(f"mSA {k} {v:.4f}" for k, v in msa.items())
          + f" ({time.perf_counter() - started:.0f} s)", flush=True)


def test_command(args):
    import torch
    identity = checkpoint_identity(checkpoint_path(args))
    best_path = os.path.join(run_dir(args), "tuning", "best_params.json")
    tuned = ev.load_json(best_path) if os.path.exists(best_path) else None
    if tuned is None and not args.defaults_only:
        raise SystemExit(f"{best_path} is missing; run 'tune' first or pass --defaults-only.")
    if tuned is not None and tuned["checksum"] != identity["checksum"]:
        raise SystemExit("The tuned parameters were selected for another checkpoint.")
    checks = [f"{n}d" for n in (2, 3) if tuned is not None and tuned.get(f"{n}d", {}).get("grid", "full") != "full"]
    if tuned is not None and (tuned.get("limit") or checks) and not args.limit:
        raise SystemExit("The tuned parameters come from a check ('--limit' or '--grid small'); tune in full.")

    pool = make_pool(args.workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ev.load_model(identity["checkpoint"], device)
    out_dir = os.path.join(run_dir(args), "test")
    os.makedirs(out_dir, exist_ok=True)

    bench_path = os.path.join(out_dir, "decode_benchmark.json")
    if device == "cuda" and not args.skip_benchmark and not os.path.exists(bench_path):
        rows = []
        for n_z in (1, 8):
            try:
                rows.append(ev.decode_benchmark(model, device, n_z))
            except torch.cuda.OutOfMemoryError as e:
                rows.append({"n_z": n_z, "error": f"out of memory: {e}"[:200]})
            torch.cuda.empty_cache()
        ev.save_json({"gpu": torch.cuda.get_device_name(), "rows": rows, **identity}, bench_path)
        print("decode benchmark:", rows)

    for dataset in args.datasets:
        ndim = ev.ndim_of(dataset)
        param_sets = {"default": ev.normalize_params(ev.default_params(ndim))}
        if tuned is not None:
            param_sets["tuned"] = ev.normalize_params(tuned[f"{ndim}d"]["params"])
        test_dataset(pool, model, device, args, dataset, param_sets, identity)
    pool.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["place-val-crops", "predict-val", "tune", "test"])
    parser.add_argument("--run", help="the run name, e.g. armA_current_seed0")
    parser.add_argument("--checkpoint", help="default: <experiment root>/checkpoints/<run>/best.pt")
    parser.add_argument("--eval-root", default=ev.EVAL_ROOT)
    parser.add_argument("--manifest", default=lib.MANIFEST_PATH)
    parser.add_argument("--val-crops", default=ev.VAL_CROPS_PATH)
    parser.add_argument("--ndim", type=int, nargs="+", default=[2, 3], choices=[2, 3])
    parser.add_argument("--datasets", nargs="+", default=list(ev.SCORED_TEST_DATASETS))
    parser.add_argument("--limit", type=int, default=None, help="only the first N samples per dataset (a check)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--max-pending", type=int, default=None, help="test: predictions waiting for a worker")
    parser.add_argument("--top", type=int, default=50, help="tune: rows of the ranking CSV")
    parser.add_argument("--grid", choices=["full", "small"], default="full", help="tune: 'small' for checks only")
    parser.add_argument("--defaults-only", action="store_true", help="test: score only the library defaults")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command != "place-val-crops" and not args.run:
        parser.error("--run is required")
    unknown = sorted(set(args.datasets) - set(ev.SCORED_TEST_DATASETS))
    if unknown:
        parser.error(f"unknown test datasets {unknown}; choose from {ev.SCORED_TEST_DATASETS}")
    args.max_pending = args.max_pending or 2 * args.workers
    {"place-val-crops": place_val_crops_command, "predict-val": predict_val_command, "tune": tune_command,
     "test": test_command}[args.command](args)


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method("fork", force=True)
    main()
