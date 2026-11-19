"""Build the 3d tuning set of the AIS post-processing search for the cells models.

The v6 search tuned the 3d parameters on one center crop per validation volume, 15 datasets with 68 crops. This
script tiles the validation regions instead, scores the tiles of the low-scoring datasets with two models to see
whether other regions score better, and writes the chosen tiles to `evaluation/tuning_crops.json`, from where
`common.load_data(..., split="val")` reads them.

    python tune_set.py candidates                    # tiles + label statistics of all in-domain 3d datasets
    python tune_set.py score --model hvit_t_cells    # predict + score the tiles of the low-scoring datasets (GPU)
    python tune_set.py score --model hiera_u --checkpoint <decoder_training>/checkpoints/epoch-10.pt
    python tune_set.py submit --model hiera_u --checkpoint ...   # the same as a grete:shared job
    python tune_set.py report                        # per-tile tables for choosing the tiles
    python tune_set.py write [--exclude picks.json]  # place at most 10 tiles per dataset in tuning_crops.json

Outputs go to OUT_ROOT (env TUNE_SET_ROOT).
"""
import os
import sys
import json
import time
import shutil
import argparse
import functools
import itertools
import subprocess

import numpy as np
import pandas as pd

import evaluate_decoder

OUT_ROOT = os.environ.get(
    "TUNE_SET_ROOT",
    "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training/param_analysis/tune_set",
)
# The datasets whose best mSA over the whole v6 grid stayed below 0.3 (see analyze_parameter_search.py).
LOW_SCORING = (
    "cartocell", "platynereis_cells", "plantseg_root", "celegans_atlas", "synapsenet_compartments", "nucverse3d",
)
MAX_TILES = 10
MIN_OBJECT_SIZE = 10  # Voxels; smaller label fragments are not counted as objects in the statistics.
# The hvit_t cells defaults for volumes (micro_sam/_model_settings.py).
DEFAULTS_3D = {
    "foreground_threshold": 0.5, "sigma": 0.5, "n_iter": 50, "dt": 1.0, "density_threshold": 80.0,
    "foreground_weight": 0.9, "min_size": 500,
}
# A compact grid around the hvit_t defaults and the Hiera-U optimum of the decoder tests: 36 flow densities x 45.
GRID = {
    "foreground_threshold": [0.3, 0.5, 0.7], "sigma": [0.25, 0.5, 1.0], "n_iter": [25, 50, 100, 200], "dt": [1.0],
    "density_threshold": [5.0, 10.0, 20.0, 40.0, 80.0], "foreground_weight": [0.25, 0.5, 0.9],
    "min_size": [50, 200, 500],
}
# Hiera-U is evaluated on z blocks of 14 slices (see evaluate_decoder.py).
HIERA_U_TILING = {"tile_shape_3d": (evaluate_decoder.Z_TILE, 384, 384), "halo_3d": (evaluate_decoder.Z_HALO, 64, 64)}


@functools.lru_cache(maxsize=None)
def import_search():
    """Import 'parameter_search' with the 'common' module of the evaluation folder, which this folder shadows.

    The modules are imported once: a second import would find 'parameter_search' cached without its 'common'.
    """
    folder = evaluate_decoder.evaluation_dir()
    saved = sys.modules.pop("common", None)
    sys.path.insert(0, folder)
    try:
        import parameter_search
        search_common = sys.modules["common"]
    finally:
        sys.path.remove(folder)
        sys.modules.pop("common", None)
        if saved is not None:
            sys.modules["common"] = saved
    return parameter_search, search_common


def grid_params():
    keys = list(GRID)
    return [dict(zip(keys, combo)) for combo in itertools.product(*GRID.values())]


def tile_key(dataset, volume, start):
    return f"{dataset}|{volume}|" + ("center" if start is None else "_".join(map(str, start)))


def _start(value):
    return None if value == "center" else tuple(int(v) for v in value.split("_"))


#
# Candidates
#


def _label_statistics(labels, valid_roi):
    ids, sizes = np.unique(labels[labels > 0], return_counts=True)
    keep = sizes >= MIN_OBJECT_SIZE
    # The objects that touch a lateral face of the tile are cut by it (or by the end of the volume).
    faces = np.concatenate([
        labels[:, 0].ravel(), labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel()
    ])
    cut = np.isin(ids[keep], np.unique(faces))
    return {
        "shape": "x".join(map(str, labels.shape)), "n_objects": int(keep.sum()),
        "n_fragments": int((~keep).sum()), "median_size": float(np.median(sizes[keep])) if keep.any() else 0.0,
        "foreground_fraction": float((labels > 0).mean()),
        "annotated_fraction": 1.0 if valid_roi is None else float(valid_roi.mean()),
        "cut_fraction": float(cut.mean()) if keep.any() else 0.0,
    }


def candidates_command(args):
    search, search_common = import_search()
    datasets = args.datasets or search.IN_DOMAIN_CELLS["3d"]
    rows = []
    for dataset in datasets:
        t0 = time.time()
        for volume, paths, z_range, region in search.tile_regions(dataset, search_common.DATA_ROOT, 3):
            # The v6 crop (the center crop) first, then the tiles.
            starts = [None] + search.tile_starts(region, search.TUNING_CROP_3D, edge_aligned=True)
            for start in starts:
                _, labels, valid_roi = search_common.load_evaluation_sample_3d(
                    *paths, crop_shape=search.TUNING_CROP_3D, z_range=z_range, split="val", crop_start=start,
                )
                rows.append({
                    "key": tile_key(dataset, volume, start), "dataset": dataset, "volume": volume,
                    "start": "center" if start is None else "_".join(map(str, start)),
                    "region": "x".join(str(sl.stop - sl.start) for sl in region),
                    **_label_statistics(labels, valid_roi),
                })
        n = sum(row["dataset"] == dataset for row in rows)
        print(f"{dataset}: {n} candidates ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    table = pd.DataFrame(rows)
    path = os.path.join(OUT_ROOT, "candidates.csv")
    if args.datasets and os.path.exists(path):  # Replace only the datasets of this run.
        old = pd.read_csv(path)
        table = pd.concat([old[~old["dataset"].isin(datasets)], table], ignore_index=True)
    table.to_csv(path, index=False)
    print(f"Written to {path}")


#
# Scoring
#


def load_model(model, checkpoint, device):
    """The model and its 3d tiling: Hiera-U from a training checkpoint, or the registry hvit_t_cells decoder."""
    if model == "hiera_u":
        staged, checksum, state = evaluate_decoder.stage_checkpoint(checkpoint, OUT_ROOT)
        print(f"Hiera-U checkpoint {staged} (epoch {state['epoch'] + 1}, checksum {checksum}).")
        return evaluate_decoder.load_decoder_model(state, device), HIERA_U_TILING
    search, search_common = import_search()
    _, decoder_path = search_common.registry_weight_paths("hvit_t_cells")
    print(f"hvit_t_cells decoder {decoder_path}.")
    return search_common.build_model("ais", "hvit_t_cells", device, 3, checkpoint_path=decoder_path), {}


def _foreground_statistics(prediction, labels, valid_roi, threshold=0.5):
    """Predicted foreground outside the labelled foreground, and labelled foreground the prediction misses."""
    fg = prediction[0] > threshold
    scored = np.ones_like(fg) if valid_roi is None else valid_roi
    predicted = (fg & scored).sum()
    labelled = ((labels > 0) & scored).sum()
    return {
        "pred_fg_outside_gt": float((fg & scored & (labels == 0)).sum() / max(predicted, 1)),
        "gt_fg_missed": float(((labels > 0) & scored & ~fg).sum() / max(labelled, 1)),
    }


def score_command(args):
    import torch

    search, search_common = import_search()
    candidates = pd.read_csv(os.path.join(OUT_ROOT, "candidates.csv"))
    candidates = candidates[candidates["dataset"].isin(args.datasets) & (candidates["n_objects"] > 0)]
    if args.limit:
        candidates = candidates.groupby("dataset").head(args.limit)
    params = grid_params() if args.n_combos is None else grid_params()[:args.n_combos]
    defaults_index = params.index(DEFAULTS_3D) if DEFAULTS_3D in params else None

    out = os.path.join(OUT_ROOT, "scores", args.model)
    cache = os.path.join(OUT_ROOT, "predictions", args.model)
    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "grid.json"), "w") as f:
        json.dump(params, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tiling = None, None
    for dataset, rows in candidates.groupby("dataset", sort=False):
        regions = {volume: (paths, z_range) for volume, paths, z_range, _ in search.tile_regions(
            dataset, search_common.DATA_ROOT, 3
        )}
        for row in rows.itertuples():
            name = row.key.replace("|", "__")
            result_path = os.path.join(out, f"{name}.json")
            if os.path.exists(result_path):
                continue
            t0 = time.time()
            paths, z_range = regions[row.volume]
            raw, labels, valid_roi = search_common.load_evaluation_sample_3d(
                *paths, crop_shape=search.TUNING_CROP_3D, z_range=z_range, split="val", crop_start=_start(row.start),
            )
            prediction_path = os.path.join(cache, f"{name}.npy")
            if os.path.exists(prediction_path):
                prediction = np.load(prediction_path).astype("float32")
            else:
                if model is None:
                    model, tiling = load_model(args.model, args.checkpoint, device)
                prediction = search_common.predict_unisam2(model, raw, ndim=3, device=device, **tiling)
                np.save(prediction_path + ".tmp.npy", prediction.astype("float16"))
                os.replace(prediction_path + ".tmp.npy", prediction_path)
            t_predict = time.time() - t0
            msa = [
                np.nan if m is None else m["msa"] for m in search.score_image(
                    prediction, labels, "sparse", params, n_threads=args.threads, valid_roi=valid_roi,
                )
            ]
            result = {"key": row.key, "msa": msa, **_foreground_statistics(prediction, labels, valid_roi)}
            if valid_roi is not None:  # The score the search gave before it respected the annotated region.
                result["msa_defaults_unmasked"] = search.score_image(
                    prediction, labels, "sparse", [DEFAULTS_3D], n_threads=args.threads,
                )[0]["msa"]
            result["seconds_predict"], result["seconds_total"] = t_predict, time.time() - t0
            with open(result_path + ".tmp", "w") as f:
                json.dump(result, f)
            os.replace(result_path + ".tmp", result_path)
            best = np.nanargmax(msa)
            defaults = "" if defaults_index is None else f", defaults {msa[defaults_index]:.3f}"
            print(f"{row.key}: best {msa[best]:.3f}{defaults} ({time.time() - t0:.0f} s)", flush=True)


SBATCH = """#!/bin/bash
#SBATCH --job-name=tuneset_{name}
#SBATCH -p {partition}
#SBATCH --gres=gpu:{gres}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH --constraint=inet
#SBATCH -A nim00007
#SBATCH -o {out}/logs/tuneset_{name}_%j.out
#SBATCH -e {out}/logs/tuneset_{name}_%j.err
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate new-stack
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True TUNE_SET_ROOT={out}
cd {code}
{commands}
"""


def submit_command(args):
    """Submit 'score' as one job that scores the models one after the other, from a frozen copy of the code."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = "_".join(args.model)
    code = os.path.join(OUT_ROOT, "jobs", f"code_{name}_{stamp}")
    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(here, code, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        evaluate_decoder.evaluation_dir(), os.path.join(code, "evaluation"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    commands = []
    for model in args.model:
        command = f"python tune_set.py score --model {model} --threads {args.cpus} --datasets {' '.join(args.datasets)}"
        if model == "hiera_u":
            staged, _, _ = evaluate_decoder.stage_checkpoint(args.checkpoint, OUT_ROOT)  # A fixed copy for the job.
            command += f" --checkpoint {staged}"
        commands.append(command)
    script = SBATCH.format(
        name=name, partition=args.partition, gres=args.gres, cpus=args.cpus, mem=args.mem, time=args.time,
        out=OUT_ROOT, code=code, commands="\n".join(commands),
    )
    os.makedirs(os.path.join(OUT_ROOT, "logs"), exist_ok=True)
    script_path = os.path.join(code, "submit.sbatch")
    with open(script_path, "w") as f:
        f.write(script)
    print(f"Script: {script_path}")
    if not args.dry_run:
        print(subprocess.run(["sbatch", script_path], capture_output=True, text=True, check=True).stdout.strip())


#
# Report
#


def load_scores(model):
    folder = os.path.join(OUT_ROOT, "scores", model)
    if not os.path.isdir(folder):
        return {}
    scores = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            with open(os.path.join(folder, name)) as f:
                result = json.load(f)
            scores[result["key"]] = result
    return scores


def _md(table):
    def fmt(value):
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.3f}"
        return str(value)
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def report_command(args):
    candidates = pd.read_csv(os.path.join(OUT_ROOT, "candidates.csv"))
    with open(os.path.join(OUT_ROOT, "grid.json")) as f:
        params = json.load(f)
    defaults_index = params.index(DEFAULTS_3D)
    models = [model for model in ("hiera_u", "hvit_t_cells") if load_scores(model)]
    for model in models:
        scores = load_scores(model)
        candidates[f"{model}_defaults"] = [
            scores[k]["msa"][defaults_index] if k in scores else np.nan for k in candidates["key"]
        ]
        candidates[f"{model}_best"] = [
            np.nanmax(scores[k]["msa"]) if k in scores else np.nan for k in candidates["key"]
        ]
        candidates[f"{model}_fg_outside"] = [
            scores[k]["pred_fg_outside_gt"] if k in scores else np.nan for k in candidates["key"]
        ]
        candidates[f"{model}_unmasked"] = [
            scores[k].get("msa_defaults_unmasked", np.nan) if k in scores else np.nan for k in candidates["key"]
        ]
    candidates.to_csv(os.path.join(OUT_ROOT, "candidates_scored.csv"), index=False)

    lines = ["# 3d tune set: candidate tiles", ""]
    lines.append(
        "Tiles are 32x512x512 (non-overlapping, plus edge-aligned last tiles); `center` is the crop of the v6 search. "
        "Label statistics count objects of at least 10 voxels; `cut` is the fraction of objects touching a lateral "
        "tile face, `annotated` the fraction inside `valid_roi`. Scores are mSA, scored inside `valid_roi`: "
        "`defaults` at the hvit_t cells defaults, `best` the best of the compact grid "
        f"({len(params)} combinations), `fg_outside` the predicted foreground (> 0.5) outside the labelled "
        "foreground, `unmasked` the defaults scored without `valid_roi` (how the search scored before)."
    )
    overview = candidates.groupby("dataset", sort=False).agg(
        candidates=("key", "size"), with_objects=("n_objects", lambda n: int((n > 0).sum())),
        objects=("n_objects", "sum"),
    ).reset_index()
    lines += ["", "## Overview", "", _md(overview)]
    columns = ["volume", "start", "shape", "n_objects", "median_size", "annotated_fraction", "cut_fraction"]
    for model in models:
        columns += [f"{model}_defaults", f"{model}_best", f"{model}_fg_outside", f"{model}_unmasked"]
    for dataset in LOW_SCORING:
        table = candidates[candidates["dataset"] == dataset]
        if table.empty:
            continue
        table = table[columns].rename(columns={"annotated_fraction": "annotated", "cut_fraction": "cut"})
        table = table.dropna(axis=1, how="all")
        lines += ["", f"## {dataset}", "", _md(table)]
    path = os.path.join(OUT_ROOT, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written to {path}")


#
# Writing the tuning crops
#


def select_tiles(rows, max_tiles=MAX_TILES):
    """Up to 'max_tiles' tiles, taken round-robin over the volumes, each volume's tiles by decreasing object count."""
    queues = [
        list(group.sort_values(["n_objects", "start"], ascending=[False, True])["start"])
        for _, group in rows.groupby("volume", sort=True)
    ]
    volumes = sorted(rows["volume"].unique())
    chosen = []
    while len(chosen) < max_tiles and any(queues):
        for volume, queue in zip(volumes, queues):
            if queue and len(chosen) < max_tiles:
                chosen.append((volume, queue.pop(0)))
    return chosen


def write_command(args):
    search, search_common = import_search()
    candidates = pd.read_csv(os.path.join(OUT_ROOT, "candidates.csv"))
    exclude, include = set(), {}
    if args.picks:
        with open(args.picks) as f:
            picks = json.load(f)
        exclude = set(picks.get("exclude", []))
        include = picks.get("include", {})  # dataset -> list of keys, used instead of the automatic choice.
    placed = {}
    for dataset in args.datasets or search.IN_DOMAIN_CELLS["3d"]:
        rows = candidates[(candidates["dataset"] == dataset) & (candidates["n_objects"] > 0)]
        rows = rows[(rows["start"] != "center") & ~rows["key"].isin(exclude)]
        if dataset in include:
            chosen = [(key.split("|")[1], key.split("|")[2]) for key in include[dataset]]
        else:
            chosen = select_tiles(rows, args.max_tiles)
        # Every tuning volume needs an entry, also the ones without a chosen tile.
        volumes = [volume for volume, *_ in search.tile_regions(dataset, search_common.DATA_ROOT, 3)]
        starts = {volume: [] for volume in volumes}
        for volume, start in sorted(chosen):
            starts[volume].append(list(_start(start)))
        objects = rows.set_index("key").loc[[tile_key(dataset, v, _start(s)) for v, s in chosen], "n_objects"]
        note = (
            f"{len(chosen)} tiles of {len(volumes)} volumes, holding {int(objects.min())}-{int(objects.max())} "
            "objects; tune_set.py: round-robin over volumes, most objects first" + (
                ", tiles picked by hand" if dataset in include else ""
            ) + "."
        )
        placed[dataset] = {"crop_shape": list(search.TUNING_CROP_3D), "note": note, "starts": starts}
        print(f"{dataset}: {note}")
    if args.dry_run:
        return
    search.write_tuning_crops(placed)
    print(f"Written to {search.TUNING_CROPS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("candidates", help="Tile the validation regions and compute label statistics.")
    sub.add_argument("--datasets", nargs="+", default=None)

    for name in ("score", "submit"):
        sub = commands.add_parser(name, help="Predict and score the candidate tiles." if name == "score" else
                                  "Submit 'score' as a Slurm job.")
        sub.add_argument(
            "--model", choices=("hiera_u", "hvit_t_cells"), required=True, nargs="+" if name == "submit" else None,
            help="The model; 'submit' takes several, which the job scores one after the other.",
        )
        sub.add_argument("--checkpoint", default=None, help="The Hiera-U training checkpoint.")
        sub.add_argument("--datasets", nargs="+", default=list(LOW_SCORING))
        if name == "score":
            sub.add_argument("--threads", type=int, default=8)
            sub.add_argument("--limit", type=int, default=None, help="At most this many tiles per dataset.")
            sub.add_argument("--n_combos", type=int, default=None, help="Only the first combos, for a check.")
        else:
            sub.add_argument("--partition", default="grete:shared")
            sub.add_argument("--gres", default="A100:1")
            sub.add_argument("--cpus", type=int, default=16)
            sub.add_argument("--mem", default="128G")
            sub.add_argument("--time", default="08:00:00")
            sub.add_argument("--dry-run", action="store_true", dest="dry_run")

    sub = commands.add_parser("report", help="Write the per-tile tables.")

    sub = commands.add_parser("write", help="Place the tuning crops in tuning_crops.json.")
    sub.add_argument("--datasets", nargs="+", default=None)
    sub.add_argument("--picks", default=None, help="JSON with 'exclude' (tile keys) and 'include' (dataset: keys).")
    sub.add_argument("--max_tiles", type=int, default=MAX_TILES)
    sub.add_argument("--dry-run", action="store_true", dest="dry_run")

    args = parser.parse_args()
    models = [] if args.command not in ("score", "submit") else (
        args.model if isinstance(args.model, list) else [args.model]
    )
    if "hiera_u" in models and args.checkpoint is None:
        parser.error("--model hiera_u needs --checkpoint.")
    {"candidates": candidates_command, "score": score_command, "submit": submit_command,
     "report": report_command, "write": write_command}[args.command](args)


if __name__ == "__main__":
    main()
