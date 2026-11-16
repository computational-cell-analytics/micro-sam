"""Compare the evaluated runs of the decoder architecture comparison (plan section 6).

    python report_decoder_arch.py                       # all runs with test results under the evaluation root
    python report_decoder_arch.py --runs armA_current_seed0 armB_current_skips_seed0

Writes '<eval root>/report/report.md' and CSV tables next to it:
- scores.csv: mSA per run, reported dataset and parameter set (tuned = the run's own tuned parameters);
- comparisons.csv: paired bootstrap of every run against the reference, per dataset, group and headline;
- diagnostics.csv: foreground IoU, boundary recall, merged / split / missed objects per run and dataset.

The headline is the mean over the twelve reported datasets at each run's own tuned parameters; EmbedSeg is one
dataset, the mean of its four collections. The seed spread (|seed 0 - seed 1| of the reference arm) is the noise
floor of the decision rule:
- the headline mean is higher than the reference's by more than the seed spread;
- neither the 2D nor the 3D mean is lower by more than the seed spread;
- no dataset loses more than 5 % relative mSA beyond its bootstrap interval, read as: no dataset whose relative
  change is below -5 % with a 95 % interval entirely below zero.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_lib as ev  # noqa: E402

PARAM_SETS = ("tuned", "default")
MAX_DATASET_LOSS = 0.05
DIMENSION_GROUPS = {
    "2d": ev.TEST_GROUPS["2d_in_domain"] + ev.TEST_GROUPS["2d_external"],
    "3d": ev.TEST_GROUPS["3d_in_domain"] + ev.TEST_GROUPS["3d_external"],
}
SUMMARIES = {**{g: list(d) for g, d in ev.TEST_GROUPS.items()}, **DIMENSION_GROUPS,
             "headline": list(ev.REPORTED_DATASETS)}


def members(dataset):
    """The scored datasets of a reported dataset and their weight within it."""
    return ev.EMBEDSEG_COLLECTIONS if dataset == "embedseg" else (dataset,)


def load_run(eval_root, run, allow_limited):
    """Per scored dataset, the test rows of a run as {column: array}, ordered by sample; None if incomplete."""
    tables = {}
    for dataset in ev.SCORED_TEST_DATASETS:
        path = os.path.join(eval_root, run, "test", f"{dataset}.csv")
        meta_path = os.path.join(eval_root, run, "test", f"{dataset}.json")
        if not os.path.exists(path):
            return None, f"missing {dataset}"
        meta = ev.load_json(meta_path)
        if meta.get("limit") and not allow_limited:
            return None, f"{dataset} was scored with --limit {meta['limit']}"
        with open(path) as f:
            rows = list(csv.DictReader(f))
        columns = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}
        columns["meta"] = meta
        tables[dataset] = columns
    return tables, None


def dataset_score(tables, dataset, column):
    return float(np.mean([np.nanmean(tables[m][column]) for m in members(dataset)]))


def summary_parts(tables, reference, datasets, column):
    """The groups, aligned samples and weights that a summary over 'datasets' bootstraps over."""
    values, refs, weights = [], [], []
    for dataset in datasets:
        group = members(dataset)
        for m in group:
            ids, ref_ids = tables[m]["sample"], reference[m]["sample"]
            if not np.array_equal(ids, ref_ids):
                raise RuntimeError(f"{m}: the run and the reference scored different samples.")
            values.append(tables[m][column])
            refs.append(reference[m][column])
            weights.append(1 / (len(datasets) * len(group)))
    return values, refs, weights


def fmt(value, digits=4):
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", default=ev.EVAL_ROOT)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--reference", default="armA_current_seed0")
    parser.add_argument("--seed-pair", nargs=2, default=["armA_current_seed0", "armA_current_seed1"])
    parser.add_argument("--out", default=None, help="default: <eval root>/report")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--allow-limited", action="store_true", help="accept '--limit' results (checks only)")
    args = parser.parse_args()
    out = args.out or os.path.join(args.eval_root, "report")
    os.makedirs(out, exist_ok=True)

    candidates = args.runs or sorted(
        d for d in os.listdir(args.eval_root) if os.path.isdir(os.path.join(args.eval_root, d, "test"))
    )
    runs, skipped = {}, {}
    for run in candidates:
        tables, reason = load_run(args.eval_root, run, args.allow_limited)
        if tables is None:
            skipped[run] = reason
        else:
            runs[run] = tables
    if not runs:
        raise SystemExit(f"No complete run under {args.eval_root}: {skipped}")
    for run, reason in skipped.items():
        print(f"skipping {run}: {reason}")

    # Scores per run, dataset and parameter set.
    score_rows = []
    for run, tables in runs.items():
        for name in PARAM_SETS:
            column = f"msa_{name}"
            if column not in tables[ev.SCORED_TEST_DATASETS[0]]:
                continue
            row = {"run": run, "params": name}
            row.update({d: dataset_score(tables, d, column) for d in ev.REPORTED_DATASETS})
            row.update({c: dataset_score(tables, c, column) for c in ev.EMBEDSEG_COLLECTIONS})
            row.update({s: float(np.mean([row[d] for d in ds])) for s, ds in SUMMARIES.items()})
            score_rows.append(row)
    score_columns = ["run", "params", *ev.REPORTED_DATASETS, *ev.EMBEDSEG_COLLECTIONS, *SUMMARIES]
    with open(os.path.join(out, "scores.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_columns)
        writer.writeheader()
        writer.writerows(score_rows)
    scores = {(r["run"], r["params"]): r for r in score_rows}

    # Seed spread of the reference arm.
    spread = {}
    if all(run in runs for run in args.seed_pair):
        for summary in ("headline", "2d", "3d"):
            spread[summary] = abs(scores[(args.seed_pair[0], "tuned")][summary]
                                  - scores[(args.seed_pair[1], "tuned")][summary])

    # Paired bootstrap against the reference.
    comparison_rows, verdicts = [], {}
    if args.reference in runs:
        reference = runs[args.reference]
        for run, tables in runs.items():
            if run == args.reference:
                continue
            for name in PARAM_SETS:
                column = f"msa_{name}"
                if column not in tables[ev.SCORED_TEST_DATASETS[0]]:
                    continue
                for target in (*ev.REPORTED_DATASETS, *SUMMARIES):
                    datasets = SUMMARIES.get(target, [target])
                    values, refs, weights = summary_parts(tables, reference, datasets, column)
                    stats = ev.paired_bootstrap(values, refs, weights=weights, n_boot=args.n_boot)
                    comparison_rows.append({"run": run, "params": name, "target": target, **stats})
            if run in args.seed_pair:
                verdicts[run] = {"verdict": "seed replicate of the reference (defines the seed spread)"}
            else:
                verdicts[run] = decide(run, comparison_rows, spread)
    with open(os.path.join(out, "comparisons.csv"), "w", newline="") as f:
        fields = ["run", "params", "target", "value", "reference", "diff", "diff_lo", "diff_hi", "rel", "rel_lo",
                  "rel_hi"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Diagnostics at the tuned parameters (or the defaults where a run was not tuned).
    diag_rows = []
    for run, tables in runs.items():
        name = "tuned" if "msa_tuned" in tables[ev.SCORED_TEST_DATASETS[0]] else "default"
        for dataset in ev.REPORTED_DATASETS:
            row = {"run": run, "params": name, "dataset": dataset}
            for key, column in (("fg_iou", "fg_iou"), ("boundary_recall", "boundary_recall"),
                                ("merged", f"merged_{name}"), ("split", f"split_{name}"),
                                ("missed", f"missed_{name}")):
                row[key] = dataset_score(tables, dataset, column)
            row["pred_per_gt"] = float(np.mean([
                np.nansum(tables[m][f"n_pred_{name}"]) / np.nansum(tables[m]["n_gt"]) for m in members(dataset)
            ]))
            diag_rows.append(row)
    with open(os.path.join(out, "diagnostics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0]))
        writer.writeheader()
        writer.writerows(diag_rows)

    write_markdown(out, args, runs, scores, spread, comparison_rows, verdicts, diag_rows, skipped)
    print(f"wrote {out}/report.md")


def decide(run, comparison_rows, spread):
    """The decision rule of plan section 6 for one run against the reference, at the tuned parameters."""
    rows = {r["target"]: r for r in comparison_rows if r["run"] == run and r["params"] == "tuned"}
    if not rows:
        return {"verdict": "not tuned"}
    losers = [d for d in ev.REPORTED_DATASETS if rows[d]["rel"] < -MAX_DATASET_LOSS and rows[d]["rel_hi"] < 0]
    if not spread:
        return {"verdict": "pending: no seed spread yet", "dataset_losses": losers}
    checks = {
        "headline_up": rows["headline"]["diff"] > spread["headline"],
        "2d_not_down": rows["2d"]["diff"] >= -spread["2d"],
        "3d_not_down": rows["3d"]["diff"] >= -spread["3d"],
        "no_dataset_loss": not losers,
    }
    return {"verdict": "beats the reference" if all(checks.values()) else "does not beat the reference",
            "checks": checks, "dataset_losses": losers}


def write_markdown(out, args, runs, scores, spread, comparison_rows, verdicts, diag_rows, skipped):
    lines = ["# Decoder architecture comparison", ""]
    lines += [f"Evaluation root: `{args.eval_root}`. Reference: `{args.reference}`. mSA on the test sets; "
              "'tuned' uses each run's own parameters (selected on the validation data), 'default' the library "
              "defaults of hvit_t.", ""]
    if skipped:
        lines += ["Incomplete runs (not reported): " + ", ".join(f"`{r}` ({why})" for r, why in skipped.items()), ""]

    for name in PARAM_SETS:
        rows = [scores[(run, name)] for run in runs if (run, name) in scores]
        if not rows:
            continue
        lines += [f"## Scores ({name} parameters)", ""]
        header = ["run", "headline", "2d", "3d", *ev.TEST_GROUPS, *ev.REPORTED_DATASETS]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for row in rows:
            lines.append("| " + " | ".join([row["run"]] + [fmt(row[h]) for h in header[1:]]) + " |")
        lines.append("")

    lines += ["## Seed spread", ""]
    if spread:
        lines += [f"|seed 0 - seed 1| of `{args.seed_pair[0]}` / `{args.seed_pair[1]}` (tuned): "
                  + ", ".join(f"{k} {fmt(v)}" for k, v in spread.items()), ""]
    else:
        lines += ["Not available: the second seed of the reference arm is missing.", ""]

    if comparison_rows:
        lines += ["## Against the reference (tuned parameters, paired bootstrap 95 %)", ""]
        header = ["run", "target", "mSA", "reference", "diff [95 %]", "relative [95 %]"]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for r in comparison_rows:
            if r["params"] != "tuned":
                continue
            lines.append(f"| {r['run']} | {r['target']} | {fmt(r['value'])} | {fmt(r['reference'])} | "
                         f"{r['diff']:+.4f} [{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}] | "
                         f"{100 * r['rel']:+.1f} % [{100 * r['rel_lo']:+.1f}, {100 * r['rel_hi']:+.1f}] |")
        lines.append("")
        lines += ["## Decision rule", ""]
        for run, verdict in verdicts.items():
            detail = ", ".join(f"{k}: {v}" for k, v in verdict.get("checks", {}).items())
            losses = verdict.get("dataset_losses")
            lines.append(f"- `{run}`: **{verdict['verdict']}**" + (f" ({detail})" if detail else "")
                         + (f"; datasets losing > 5 %: {', '.join(losses)}" if losses else ""))
        lines.append("")

    lines += ["## Diagnostics (test sets)", "",
              "Foreground IoU at 0.5, boundary recall at 0.5, and the share of ground-truth objects that are merged, "
              "split or missed (see `eval_lib.object_errors`); predicted objects per ground-truth object.", ""]
    header = ["run", "dataset", "fg_iou", "boundary_recall", "merged", "split", "missed", "pred_per_gt"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in diag_rows:
        lines.append("| " + " | ".join([row["run"], row["dataset"]] + [fmt(row[h], 3) for h in header[2:]]) + " |")
    lines.append("")

    lines += ["## Tuned parameters and decode cost", ""]
    header = ["run", "2d parameters", "3d parameters", "edges (2d / 3d)", "decode ms Z=1", "Z=8", "peak GiB Z=1",
              "Z=8"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run in runs:
        tuned_path = os.path.join(args.eval_root, run, "tuning", "best_params.json")
        bench_path = os.path.join(args.eval_root, run, "test", "decode_benchmark.json")
        tuned = ev.load_json(tuned_path) if os.path.exists(tuned_path) else {}
        bench = {r["n_z"]: r for r in ev.load_json(bench_path)["rows"]} if os.path.exists(bench_path) else {}

        def params(ndim):
            p = tuned.get(f"{ndim}d", {}).get("params")
            if not p:
                return "n/a"
            return (f"fg {p['foreground_threshold']}, dens {p['density_threshold']}, min {p['min_size']}, "
                    f"sigma {p['sigma']}, travel {p['n_iter'] * p['dt']:g}, fw {p['foreground_weight']}")
        edges = " / ".join(",".join(tuned.get(f"{n}d", {}).get("at_grid_edge", [])) or "-" for n in (2, 3))
        cost = [fmt(bench.get(z, {}).get(k), 1 if k == "decode_ms" else 2)
                for k in ("decode_ms", "decode_peak_GiB") for z in (1, 8)]
        lines.append("| " + " | ".join([run, params(2), params(3), edges, *cost]) + " |")
    lines.append("")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
