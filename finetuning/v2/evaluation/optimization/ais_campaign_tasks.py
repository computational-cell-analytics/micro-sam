"""Build the task lists of the AIS optimization campaign and hand them to the submitter.

Each subcommand turns a few arguments into '(tag, command)' pairs for `benchmark_ais_optimization.py`
and submits them through `submit_optimization_jobs.submit_tasks`. '--extra' appends verbatim arguments
to every command.

Usage examples:
    # Cache the predictions of two subsets, one task per subset, on the session GPU.
    python ais_campaign_tasks.py predict --name ais_predict --subsets primary training_extra --local

    # One CPU task per (subset, configuration): the screen of a candidate family against the baseline.
    python ais_campaign_tasks.py screen --name s0_travel --preset cpu --subsets primary training_extra \\
        --configs configs/ais_control_registry_defaults.json configs/ais_s0_*.json

    # A parameter sweep, one task per (subset, dataset, shard).
    python ais_campaign_tasks.py sweep --name lm_grid --preset cpu --subsets primary --grid configs/ais_grid_lm.json \\
        --datasets livecell tissuenet --num-shards 4
"""

from __future__ import annotations

import argparse
import glob
import shlex
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPTIMIZATION_ROOT))

from submit_optimization_jobs import add_submit_arguments, sanitize, submit_from_args  # noqa

SCRIPT = OPTIMIZATION_ROOT / "benchmark_ais_optimization.py"

Task = Tuple[str, str]


def _command(*args: object) -> str:
    return shlex.join(["python", str(SCRIPT), *[str(arg) for arg in args]])


def _config_stem(path: Optional[Path]) -> str:
    if path is None:
        return "defaults"
    stem = Path(path).stem
    return sanitize(stem[4:] if stem.startswith("ais_") else stem)


def _expand(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No configuration matches '{pattern}'.")
        paths.extend(Path(match).resolve() for match in matches)
    return paths


def predict_tasks(kind: str, subsets: Sequence[str], extra: Sequence[str] = ()) -> List[Task]:
    """One `predict` task per subset."""
    return [
        (f"predict_{kind}_{sanitize(subset)}", _command("predict", "--kind", kind, "--subset", subset, *extra))
        for subset in subsets
    ]


def run_tasks(
    kind: str, subsets: Sequence[str], configs: Sequence[Optional[Path]], trial_ids: Sequence[str],
    extra: Sequence[str] = (),
) -> List[Task]:
    """One `run` task per (subset, configuration, trial)."""
    tasks = []
    for subset in subsets:
        for config in configs:
            for trial in trial_ids:
                args: List[object] = ["run", "--kind", kind, "--subset", subset, "--trial-id", trial]
                if config is not None:
                    args.extend(["--config", config])
                args.extend(extra)
                tag = f"run_{kind}_{sanitize(subset)}_{_config_stem(config)}_{sanitize(trial)}"
                tasks.append((tag, _command(*args)))
    return tasks


def sweep_tasks(
    kind: str, subsets: Sequence[str], grid: Path, datasets: Sequence[str], num_shards: int, extra: Sequence[str] = (),
) -> List[Task]:
    """One `sweep` task per (subset, dataset, shard)."""
    tasks = []
    for subset in subsets:
        for dataset in datasets:
            for shard in range(num_shards):
                args: List[object] = [
                    "sweep", "--kind", kind, "--subset", subset, "--grid", grid, "--datasets", dataset,
                    "--shard-index", shard, "--num-shards", num_shards, *extra,
                ]
                tag = f"sweep_{kind}_{sanitize(subset)}_{sanitize(dataset)}_{shard}of{num_shards}"
                tasks.append((tag, _command(*args)))
    return tasks


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Cache the decoder predictions of subsets.")
    screen = subparsers.add_parser("screen", help="Run configurations on the cache, one task each.")
    screen.add_argument("--configs", nargs="*", default=[], help="Configuration files or globs.")
    screen.add_argument("--no-defaults", action="store_true", help="Do not add the library-defaults baseline.")
    screen.add_argument("--trial-ids", nargs="*", default=["trial-1"])
    sweep = subparsers.add_parser("sweep", help="Sweep a grid on the cache, one task per dataset and shard.")
    sweep.add_argument("--grid", type=Path, required=True)
    sweep.add_argument("--datasets", nargs="+", required=True)
    sweep.add_argument("--num-shards", type=int, default=1)

    for sub in (predict, screen, sweep):
        sub.add_argument("--kind", choices=("v5", "apg3d"), default="v5")
        sub.add_argument("--subsets", nargs="+", default=["primary"])
        sub.add_argument("--extra", default="", help="Arguments appended verbatim to every command.")
        sub.add_argument("--print-only", action="store_true", help="Print the tasks and stop.")
        add_submit_arguments(sub)

    args = parser.parse_args(list(argv) if argv is not None else None)
    extra = shlex.split(args.extra) if args.extra else []
    if args.command == "predict":
        tasks = predict_tasks(args.kind, args.subsets, extra)
    elif args.command == "screen":
        configs: List[Optional[Path]] = list(_expand(args.configs))
        if not args.no_defaults:
            configs = [None, *configs]
        tasks = run_tasks(args.kind, args.subsets, configs, args.trial_ids, extra)
    else:
        tasks = sweep_tasks(args.kind, args.subsets, args.grid.resolve(), args.datasets, args.num_shards, extra)
    for tag, command in tasks:
        print(f"{tag}\t{command}")
    if args.print_only:
        return 0
    submit_from_args(tasks, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
