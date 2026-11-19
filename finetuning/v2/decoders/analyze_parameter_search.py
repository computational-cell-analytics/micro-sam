"""Offline analysis of the AIS post-processing parameter search, on sweeps that already exist.

It reads the per-image score matrices (combinations x samples, mSA) of two earlier searches and asks why a tuned
parameter set transfers poorly, and which selection rule would transfer better. Nothing is predicted or segmented.

Sources:
- 'v6_2d', 'v6_3d': the global search of the micro-sam2 hvit_t cells model (production decoder) on the validation
  data of 30 2D and 15 3D in-domain datasets (`parameter_search.py run`, broad grid).
- 'arch_<run>_2d', 'arch_<run>_3d': round 0 (the full grid) of the tuning of the decoder comparison on the branch
  `decoder-tests` (`decoder_arch/eval_lib.py`), 3 2D datasets and 3 3D dataset groups (EmbedSeg = 4 collections).

Analyses (subcommand 'run', then 'report'):
1. Sensitivity: per parameter, the best score reachable with the parameter pinned to each value ("profile").
2. Flatness: how many combinations lie within 1 / 2 / 5 % of the best one, and the parameter ranges they span.
3. Transfer of selection rules, selected on one part of the data and scored on another:
   - 'datasets': repeated random halves of the datasets, and leave-one-dataset-out (the v6 sources only).
   - 'samples': repeated random halves of the samples of every dataset.
   - 'seeds': selected on one training seed, scored on the other (arch A0 / A1 and C0 / C1).
4. Per-dataset optima of every parameter, against the median object size of the validation data where the
   labels are available (the arch sources cache them).

Usage:
    python analyze_parameter_search.py cache   # read the sweeps once into float32 caches
    python analyze_parameter_search.py run     # the analyses, writes CSVs
    python analyze_parameter_search.py report  # report.md from the CSVs
"""

import os
import json
import glob
import argparse

import numpy as np
import pandas as pd

V6_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/v6_global_search"
V6_CHECKPOINT = "69dee608d040cf051e2b497f8a0fa803"
V6_DIRS = {
    "v6_2d": os.path.join(V6_ROOT, "tuning", "ais", "hvit_t", V6_CHECKPOINT),
    "v6_3d": os.path.join(V6_ROOT, "tuning_3d32", "ais", "hvit_t", V6_CHECKPOINT),
}
# The dense (multicut) sweeps in the 3D folder use another grid and metric.
V6_DENSE = ("cremi", "fafb", "hemibrain", "liconn", "xpress", "zebrafinch_j0126")
ARCH_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_arch/evaluation"
ARCH_RUNS = {
    "A0": "armA_current_seed0", "A1": "armA_current_seed1", "B0": "armB_current_skips_seed0",
    "C0": "armC_hiera_u_seed0", "C1": "armC_hiera_u_seed1", "D0": "armD_cagd_3d_seed0", "E0": "armE_c3f2_seed0",
}
ARCH_GROUPS_3D = {
    "nis3d": ("nis3d",), "plantseg_root": ("plantseg_root",),
    "embedseg": ("embedseg_organoid", "embedseg_platy_nuclei", "embedseg_mouse_skull", "embedseg_platy_ish"),
}
SEED_PAIRS = (("A0", "A1"), ("C0", "C1"))
OUT_ROOT = os.environ.get(
    "PARAM_ANALYSIS_ROOT",
    "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training/param_analysis",
)
CACHE = os.path.join(OUT_ROOT, "cache")

# The library defaults of hvit_t cells (micro_sam/_model_settings.py), as travel = n_iter * dt.
DEFAULTS = {
    "2d": {"foreground_threshold": 0.5, "density_threshold": 30.0, "min_size": 50, "sigma": 0.5, "travel": 50.0,
           "foreground_weight": 0.9},
    "3d": {"foreground_threshold": 0.5, "density_threshold": 80.0, "min_size": 500, "sigma": 0.5, "travel": 50.0,
           "foreground_weight": 0.9},
}
AXES = ("foreground_threshold", "density_threshold", "min_size", "sigma", "travel", "foreground_weight")
N_SPLITS = 200
N_BOOT = 200
RNG_SEED = 0


#
# Loading
#


def combo_key(params):
    """A combination as a tuple over AXES; n_iter and dt act only through their product (the travel)."""
    p = dict(params)
    p["travel"] = round(float(p.pop("n_iter")) * float(p.pop("dt")), 6)
    return tuple(round(float(p[a]), 6) for a in AXES)


def _load_v6(name):
    folder = V6_DIRS[name]
    files = sorted(f for f in glob.glob(os.path.join(folder, "*.per_image.npz")) if ".shard" not in f)
    keys, scores = None, {}
    for path in files:
        dataset = os.path.basename(path).split(".")[0]
        if dataset in V6_DENSE:
            continue
        data = np.load(path)  # Every array is read once: indexing an NpzFile per row decompresses it each time.
        params, matrix = json.loads(str(data["params"])), data["scores"].astype("float32")
        order = {combo_key(p): i for i, p in enumerate(params)}
        if keys is None:
            keys = sorted(order)
        missing = [k for k in keys if k not in order]
        if missing:
            raise RuntimeError(f"{dataset}: {len(missing)} combinations of the first dataset are missing.")
        scores[dataset] = matrix[[order[k] for k in keys]]
        print(f"{name} {dataset}: {matrix.shape}")
    return keys, scores


def _load_arch(run, ndim):
    folder = os.path.join(ARCH_ROOT, ARCH_RUNS[run], "tuning")
    with open(os.path.join(folder, f"{ndim}d_round0_params.json")) as f:
        params = json.load(f)
    data = np.load(os.path.join(folder, f"{ndim}d_round0.npz"))
    # All runs share the grid file, so the combinations line up across runs without reordering.
    return [combo_key(p) for p in params], {d: data[d].astype("float32") for d in data.files}


def cache_command(args):
    os.makedirs(CACHE, exist_ok=True)
    sources = {}
    for name in V6_DIRS:
        sources[name] = _load_v6(name)
    for run in ARCH_RUNS:
        for ndim in (2, 3):
            sources[f"arch_{run}_{ndim}d"] = _load_arch(run, ndim)
    for name, (keys, scores) in sources.items():
        np.savez(os.path.join(CACHE, f"{name}.npz"), keys=np.array(keys, dtype="float64"), **scores)
        print(f"Cached {name}: {len(keys)} combinations, {len(scores)} datasets.")
    _cache_object_sizes()


def _cache_object_sizes():
    """Median ground-truth object size (pixels / voxels) per dataset, from the arch validation caches."""
    from skimage.measure import regionprops
    rows = []
    run_dir = os.path.join(ARCH_ROOT, ARCH_RUNS["C0"], "val_cache")
    for dataset in sorted(os.listdir(run_dir)):
        files = sorted(glob.glob(os.path.join(run_dir, dataset, "*.npz")))
        sizes = []
        for path in files:
            labels = np.load(path)["labels"]
            # 6 of the 80 cellpose validation images hold thousands of 1-pixel label fragments; they are ignored.
            sizes.extend(r.area for r in regionprops(labels.astype("int64")) if r.area >= 10)
        if sizes:
            rows.append({"dataset": dataset, "median_object_size": float(np.median(sizes)), "n_objects": len(sizes)})
            print(f"object sizes {dataset}: median {np.median(sizes):.0f} over {len(sizes)}")
    pd.DataFrame(rows).to_csv(os.path.join(CACHE, "object_sizes_arch.csv"), index=False)


class Source:
    """The score matrices of one search: per dataset (combinations x samples), and the selection units."""

    def __init__(self, name):
        data = np.load(os.path.join(CACHE, f"{name}.npz"))
        self.name = name
        self.keys = data["keys"]
        self.scores = {k: data[k] for k in data.files if k != "keys"}
        self.ndim = "3d" if name.endswith("3d") else "2d"
        if name.startswith("arch") and self.ndim == "3d":
            self.units = {u: tuple(d for d in ds if d in self.scores) for u, ds in ARCH_GROUPS_3D.items()}
        else:
            self.units = {d: (d,) for d in self.scores}
        # Combinations that failed on every sample of a dataset cannot be ranked.
        means = self.dataset_means()
        self.valid = np.all(np.isfinite(np.stack(list(means.values()))), axis=0)
        self.default_index = self._find(DEFAULTS[self.ndim])

    def _find(self, params):
        target = np.array([params[a] for a in AXES])
        hits = np.where(np.all(np.isclose(self.keys, target), axis=1))[0]
        return int(hits[0]) if len(hits) else None

    def dataset_means(self, sample_masks=None):
        out = {}
        for d, s in self.scores.items():
            if sample_masks is not None:
                s = s[:, sample_masks[d]]
            with np.errstate(all="ignore"):
                out[d] = np.nanmean(s, axis=1) if s.shape[1] else np.full(s.shape[0], np.nan)
        return out

    def unit_matrix(self, means, units=None):
        """(units x combinations) of the unit scores: the mean of the member datasets' mean mSA."""
        units = units or list(self.units)
        return np.stack([np.mean([means[d] for d in self.units[u]], axis=0) for u in units]), units


#
# Selection rules. Each takes the (units x combinations) score matrix of the tuning data and returns a combination.
#


def relative(matrix):
    best = np.nanmax(matrix, axis=1, keepdims=True)
    return matrix / best


def neighbours(keys):
    """(combinations x 13) indices of the grid neighbours (one step on one axis) and the combination itself."""
    values = [np.unique(keys[:, j]) for j in range(keys.shape[1])]
    index = np.stack([np.searchsorted(values[j], keys[:, j]) for j in range(keys.shape[1])], axis=1)
    lookup = {tuple(row): i for i, row in enumerate(index)}
    nbrs = np.full((len(keys), 2 * keys.shape[1] + 1), -1, dtype="int64")
    nbrs[:, 0] = np.arange(len(keys))
    for i, row in enumerate(index):
        col = 1
        for j in range(keys.shape[1]):
            for step in (-1, 1):
                other = list(row)
                other[j] += step
                nbrs[i, col] = lookup.get(tuple(other), -1)
                col += 1
    return nbrs


def smooth(score, nbrs):
    vals = np.where(nbrs >= 0, score[np.clip(nbrs, 0, None)], np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(vals, axis=1)


def _argmax(score, valid):
    score = np.where(valid & np.isfinite(score), score, -np.inf)
    return int(np.argmax(score))


def make_rules(source, rng):
    nbrs = neighbours(source.keys)

    def bootstrap_quantile(matrix, valid, q=0.1):
        rel = relative(matrix)
        n = rel.shape[0]
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype("float32")
        boot = counts @ np.nan_to_num(rel) / n
        return _argmax(np.quantile(boot, q, axis=0), valid)

    rules = {
        "relative_mean (current)": lambda m, v: _argmax(relative(m).mean(axis=0), v),
        "absolute_mean": lambda m, v: _argmax(m.mean(axis=0), v),
        "relative_worst": lambda m, v: _argmax(relative(m).min(axis=0), v),
        "relative_mean_minus_std": lambda m, v: _argmax(
            relative(m).mean(axis=0) - (relative(m).std(axis=0) if m.shape[0] > 1 else 0), v
        ),
        "relative_mean_smoothed": lambda m, v: _argmax(smooth(relative(m).mean(axis=0), nbrs), v),
        "absolute_mean_smoothed": lambda m, v: _argmax(smooth(m.mean(axis=0), nbrs), v),
        "bootstrap_q10": bootstrap_quantile,
    }
    if source.default_index is not None:
        rules["defaults"] = lambda m, v: source.default_index
    return rules


#
# Evaluation of the rules
#


def score_choice(matrix, index):
    """Scores of a chosen combination on held-out units: mean relative to each unit's own best, and mean mSA."""
    return {
        "heldout_relative": float(np.mean(matrix[:, index] / np.nanmax(matrix, axis=1))),
        "heldout_msa": float(np.mean(matrix[:, index])),
        # The best single combination for the held-out units: the ceiling of any global choice.
        "heldout_oracle_relative": float(np.nanmax(relative(matrix).mean(axis=0))),
    }


def transfer_datasets(source, rng):
    means = source.dataset_means()
    matrix, units = source.unit_matrix(means)
    rules = make_rules(source, rng)
    rows = []
    n = len(units)
    if n < 4:
        splits = [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)]
        kind = "lodo"
    else:
        splits = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(n)
            splits.append((perm[: n // 2], perm[n // 2:]))
        kind = "halves"
    lodo = [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)] if n >= 4 else []
    for split_kind, split_list in ((kind, splits), ("lodo", lodo)):
        for k, (tune, test) in enumerate(split_list):
            valid = source.valid
            for rule, fn in rules.items():
                index = fn(matrix[tune], valid)
                rows.append({
                    "source": source.name, "protocol": f"datasets_{split_kind}", "split": k, "rule": rule,
                    "heldout_units": ",".join(units[i] for i in test), "index": index,
                    **score_choice(matrix[test], index),
                })
    return rows


def transfer_samples(source, rng):
    rules = make_rules(source, rng)
    rows = []
    n_splits = N_SPLITS // 4
    for k in range(n_splits):
        masks_a, masks_b = {}, {}
        for d, s in source.scores.items():
            perm = rng.permutation(s.shape[1])
            a = np.zeros(s.shape[1], bool)
            a[perm[: max(1, s.shape[1] // 2)]] = True
            masks_a[d], masks_b[d] = a, ~a
        # Datasets with one sample cannot be split; they stay in both halves.
        for d, s in source.scores.items():
            if s.shape[1] == 1:
                masks_b[d] = masks_a[d]
        tune, _ = source.unit_matrix(source.dataset_means(masks_a))
        test, _ = source.unit_matrix(source.dataset_means(masks_b))
        for rule, fn in rules.items():
            index = fn(tune, source.valid)
            rows.append({
                "source": source.name, "protocol": "samples_halves", "split": k, "rule": rule, "index": index,
                **score_choice(test, index),
            })
    return rows


def transfer_seeds(sources, rng):
    rows = []
    for a, b in SEED_PAIRS:
        for ndim in ("2d", "3d"):
            for tune_run, test_run in ((a, b), (b, a)):
                tune_src, test_src = sources[f"arch_{tune_run}_{ndim}"], sources[f"arch_{test_run}_{ndim}"]
                tune, _ = tune_src.unit_matrix(tune_src.dataset_means())
                test, _ = test_src.unit_matrix(test_src.dataset_means())
                valid = tune_src.valid & test_src.valid
                tune_rules, test_rules = make_rules(tune_src, rng), make_rules(test_src, rng)
                for rule, fn in tune_rules.items():
                    index = fn(tune, valid)
                    own = test_rules[rule](test, valid)
                    rows.append({
                        "source": f"arch_{ndim}", "protocol": "seeds", "split": f"{tune_run}->{test_run}",
                        "rule": rule, "index": index, "same_choice_as_target_seed": index == own,
                        **score_choice(test, index),
                    })
    return rows


#
# Sensitivity, flatness, per-dataset optima
#


def sensitivity(source):
    matrix, _ = source.unit_matrix(source.dataset_means())
    rel = relative(matrix).mean(axis=0)
    rel = np.where(source.valid, rel, np.nan)
    best = np.nanmax(rel)
    rows = []
    for j, axis in enumerate(AXES):
        for value in np.unique(source.keys[:, j]):
            sel = source.keys[:, j] == value
            rows.append({
                "source": source.name, "axis": axis, "value": value,
                "best_relative": float(np.nanmax(rel[sel])), "loss_vs_best": float(best - np.nanmax(rel[sel])),
                "median_relative": float(np.nanmedian(rel[sel])),
            })
    return rows


def flatness(source):
    matrix, _ = source.unit_matrix(source.dataset_means())
    rel = np.where(source.valid, relative(matrix).mean(axis=0), np.nan)
    best = np.nanmax(rel)
    row = {"source": source.name, "n_combinations": int(source.valid.sum()), "n_units": matrix.shape[0],
           "best_relative": float(best)}
    if source.default_index is not None:
        row["default_relative"] = float(rel[source.default_index])
        row["default_rank"] = int(np.sum(rel > rel[source.default_index]) + 1)
    for tol in (0.01, 0.02, 0.05):
        sel = rel >= best - tol
        row[f"n_within_{int(tol * 100)}pct"] = int(np.sum(sel))
        for j, axis in enumerate(AXES):
            vals = np.unique(source.keys[sel, j])
            row[f"range_{int(tol * 100)}pct_{axis}"] = f"{vals.min():g}-{vals.max():g} ({len(vals)})"
    return row


def per_dataset_optima(source):
    means = source.dataset_means()
    rows = []
    for d, m in means.items():
        m = np.where(source.valid, m, np.nan)
        i = int(np.nanargmax(m))
        row = {"source": source.name, "dataset": d, "best_msa": float(m[i]), "n_samples": source.scores[d].shape[1]}
        if source.default_index is not None:
            row["default_msa"] = float(m[source.default_index])
        row.update({a: source.keys[i, j] for j, a in enumerate(AXES)})
        rows.append(row)
    return rows


def run_command(args):
    os.makedirs(OUT_ROOT, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CACHE, "*.npz")))
    sources = {n: Source(n) for n in names}
    transfer, sens, flat, optima = [], [], [], []
    for name, source in sources.items():
        print(f"{name}: {len(source.keys)} combinations, {len(source.units)} units, default index "
              f"{source.default_index}")
        sens += sensitivity(source)
        flat.append(flatness(source))
        optima += per_dataset_optima(source)
        main = name.startswith("v6") or name.startswith("arch_C0") or name.startswith("arch_A0")
        if main:
            transfer += transfer_datasets(source, rng)
            transfer += transfer_samples(source, rng)
    transfer += transfer_seeds(sources, rng)
    pd.DataFrame(transfer).to_csv(os.path.join(OUT_ROOT, "transfer.csv"), index=False)
    pd.DataFrame(sens).to_csv(os.path.join(OUT_ROOT, "sensitivity.csv"), index=False)
    pd.DataFrame(flat).to_csv(os.path.join(OUT_ROOT, "flatness.csv"), index=False)
    pd.DataFrame(optima).to_csv(os.path.join(OUT_ROOT, "per_dataset_optima.csv"), index=False)
    print(f"Wrote the tables to {OUT_ROOT}.")


def _md(df, floatfmt=".3f", index=False):
    """A markdown table, without the optional 'tabulate' dependency of `DataFrame.to_markdown`."""
    if index:
        df = df.reset_index()

    def cell(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)

    header = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def report_command(args):
    transfer = pd.read_csv(os.path.join(OUT_ROOT, "transfer.csv"))
    sens = pd.read_csv(os.path.join(OUT_ROOT, "sensitivity.csv"))
    flat = pd.read_csv(os.path.join(OUT_ROOT, "flatness.csv"))
    optima = pd.read_csv(os.path.join(OUT_ROOT, "per_dataset_optima.csv"))
    sizes = pd.read_csv(os.path.join(CACHE, "object_sizes_arch.csv"))
    lines = ["# Offline analysis of the AIS post-processing parameter search", ""]

    lines += ["## Transfer of the selection rules", "",
              "Mean over splits of the held-out score: relative to each held-out unit's own best combination, "
              "and mean mSA. 'oracle' is the best single combination for the held-out units.", ""]
    agg = transfer.groupby(["source", "protocol", "rule"]).agg(
        heldout_relative=("heldout_relative", "mean"), relative_sd=("heldout_relative", "std"),
        heldout_msa=("heldout_msa", "mean"), oracle=("heldout_oracle_relative", "mean"),
        n_splits=("heldout_relative", "size"),
    ).reset_index()
    for (source, protocol), group in agg.groupby(["source", "protocol"]):
        lines += [f"### {source}, {protocol}", "", _md(group.drop(columns=["source", "protocol"])), ""]
    seeds = transfer[transfer.protocol == "seeds"]
    if len(seeds):
        lines += ["### Same choice as on the target seed", "",
                  _md(seeds.groupby(["source", "rule"]).same_choice_as_target_seed.mean().reset_index()), ""]

    lines += ["## Flatness", "", _md(flat), ""]
    lines += ["## Sensitivity (loss of the best mean relative score when an axis is pinned to a value)", ""]
    for source in ("v6_2d", "v6_3d", "arch_C0_2d", "arch_C0_3d", "arch_C1_3d", "arch_A0_3d"):
        s = sens[sens.source == source]
        if len(s):
            table = s.pivot_table(index="axis", columns="value", values="loss_vs_best")
            lines += [f"### {source}", "", _md(table, index=True), ""]
    optima = optima.merge(sizes, on="dataset", how="left")
    lines += ["## Per-dataset optima", "", _md(optima), ""]
    path = os.path.join(OUT_ROOT, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cache")
    sub.add_parser("run")
    sub.add_parser("report")
    args = parser.parse_args()
    {"cache": cache_command, "run": run_command, "report": report_command}[args.command](args)


if __name__ == "__main__":
    main()
