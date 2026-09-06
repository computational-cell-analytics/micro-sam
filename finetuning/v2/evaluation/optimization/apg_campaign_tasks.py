"""Build the task lists of the APG optimization campaign and hand them to the submitter.

Each subcommand turns a few arguments into '(tag, command)' pairs for one script of this directory
and submits them through `submit_optimization_jobs.submit_tasks`. '--extra' appends verbatim
arguments to every command, which is how artifact paths and time budgets reach the scripts.

Usage examples:
    # Three serialized, bracketed 2d timing trials of one config on the holdout, one at a time.
    python apg_campaign_tasks.py benchmark --name holdout_timing --preset 2d --gres 1g.20gb:1 \\
        --ndim 2 --subset holdout --trial-ids trial-1 trial-2 trial-3 --serialize --bracket --throttle 1 \\
        --config configs/apg_control_registry_defaults.json

    # One array task per crop of the 3d runner (the sample index is appended after '--extra').
    python apg_campaign_tasks.py per-sample --name apg3d_primary --preset 3d \\
        --script optimization/benchmark_apg_3d.py --indices 0-56 --throttle 12 \\
        --extra "run --subset primary --config configs/apg3d_defaults.json"
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPTIMIZATION_ROOT))

from submit_optimization_jobs import add_submit_arguments, sanitize, submit_from_args  # noqa

CONFIG_ROOT = OPTIMIZATION_ROOT / "configs"
SCRIPTS = {
    "benchmark": OPTIMIZATION_ROOT / "benchmark_apg_optimization.py",
    "benchmark-3d": OPTIMIZATION_ROOT / "benchmark_apg_3d.py",
}

Task = Tuple[str, str]


def _command(script: Path, *args: object) -> str:
    return shlex.join(["python", str(script), *[str(arg) for arg in args]])


def _config_stem(path: Optional[Path]) -> str:
    if path is None:
        return "defaults"
    stem = Path(path).stem
    return sanitize(stem[4:] if stem.startswith("apg_") else stem)


def chain(tag: str, tasks: Sequence[Task]) -> Task:
    """Join several tasks into one that runs them in order on one node."""
    return tag, " && ".join(command for _, command in tasks)


def benchmark_tasks(
    configs: Sequence[Optional[Path]], trial_ids: Sequence[str], ndim: str = "2", subset: str = "primary",
    crops_3d: str = "standard", extra: Sequence[str] = (), serialize: bool = False, bracket: bool = False,
) -> List[Task]:
    """One `benchmark_apg_optimization.py` run per (trial, config).

    'serialize' collapses each trial into one task so the whole trial runs on one node in order, and
    'bracket' surrounds it with a defaults run before and after, which is how a timing trial detects
    node-level drift. Completed runs short-circuit, so a chain re-run after preemption is cheap.
    """
    script = SCRIPTS["benchmark"]
    tasks = []
    for trial in trial_ids:
        trial_tasks = []
        entries: List[Tuple[Optional[Path], str]] = [(config, trial) for config in configs]
        if bracket:
            entries = [(None, f"{trial}-bracket-pre"), *entries, (None, f"{trial}-bracket-post")]
        for config, trial_id in entries:
            args: List[object] = ["--ndim", ndim, "--subset", subset, "--crops-3d", crops_3d, "--trial-id", trial_id]
            if config is not None:
                args.extend(["--config", Path(config).resolve()])
            args.extend(extra)
            tag = f"bench{ndim}d_{subset}_{_config_stem(config)}_{trial_id}"
            trial_tasks.append((tag, _command(script, *args)))
        if serialize:
            tasks.append(chain(f"bench{ndim}d_{subset}_{sanitize(trial)}_serial", trial_tasks))
        else:
            tasks.extend(trial_tasks)
    return tasks


def parse_indices(spec: str) -> List[int]:
    """'1-3,7' -> [1, 2, 3, 7]."""
    indices: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = part.split("-")
            indices.extend(range(int(start), int(stop) + 1))
        else:
            indices.append(int(part))
    return indices


def per_sample_tasks(
    script: Path, indices: Iterable[int], sample_flag: str = "--sample-index", extra: Sequence[str] = (),
    tag_prefix: str = "sample",
) -> List[Task]:
    """One task per sample index of a script that processes one manifest sample per invocation."""
    script = Path(script).resolve()
    return [(f"{tag_prefix}_{index:03d}", _command(script, *extra, sample_flag, index)) for index in indices]


def _extra(args: argparse.Namespace) -> List[str]:
    return shlex.split(args.extra) if args.extra else []


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark", help="Canonical benchmark runs.")
    bench.add_argument("--config", type=Path, nargs="*", default=[], help="Config JSON files.")
    bench.add_argument("--defaults", action="store_true", help="Also run the library defaults.")
    bench.add_argument("--trial-ids", nargs="*", default=None)
    bench.add_argument("--trials", type=int, default=1, help="Number of trials when --trial-ids is not given.")
    bench.add_argument("--ndim", choices=("2", "3", "both"), default="2")
    bench.add_argument("--subset", default="primary")
    bench.add_argument("--crops-3d", default="standard")
    bench.add_argument("--serialize", action="store_true")
    bench.add_argument("--bracket", action="store_true")

    per_sample = subparsers.add_parser("per-sample", help="One task per sample index of a script.")
    per_sample.add_argument("--script", type=Path, required=True)
    per_sample.add_argument("--indices", required=True, help="e.g. 0-30 or 1,4,7")
    per_sample.add_argument("--sample-flag", default="--sample-index")
    per_sample.add_argument("--tag-prefix", default="sample")

    for sub in (bench, per_sample):
        sub.add_argument("--extra", default="", help="Arguments appended verbatim to every command.")
        sub.add_argument("--print-only", action="store_true", help="Print the tasks and stop.")
        add_submit_arguments(sub)

    args = parser.parse_args(list(argv) if argv is not None else None)
    extra = _extra(args)
    if args.command == "benchmark":
        configs: List[Optional[Path]] = list(args.config)
        if args.defaults or not configs:
            configs = [None, *configs]
        trial_ids = args.trial_ids or [f"trial-{index}" for index in range(1, args.trials + 1)]
        tasks = benchmark_tasks(
            configs, trial_ids, ndim=args.ndim, subset=args.subset, crops_3d=args.crops_3d, extra=extra,
            serialize=args.serialize, bracket=args.bracket,
        )
    else:
        tasks = per_sample_tasks(
            args.script, parse_indices(args.indices), sample_flag=args.sample_flag, extra=extra,
            tag_prefix=args.tag_prefix,
        )
    for tag, command in tasks:
        print(f"{tag}\t{command}")
    if args.print_only:
        return 0
    submit_from_args(tasks, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
