"""Submit the APG optimization scripts to Slurm as array jobs, or run them locally.

Every task is one 'tag<TAB>command' line of a tasks file. One array script dispatches the lines by
SLURM_ARRAY_TASK_ID, optionally packs several commands into a full-node allocation, retries failures,
and records the outcome of every task as a
'.done' or '.failed' marker beside the logs, so a dependent stage can wait for a marker rather than
for a file that may still be half written. Preemption restarts the script from the top through
'--requeue'; the marker check and the scripts' own per-sample resume make that idempotent.

Usage examples:
    python submit_optimization_jobs.py submit --name smoke --preset 2d-short --tasks-file tasks.txt --dry-run
    python submit_optimization_jobs.py submit --name smoke --preset 2d --tasks-file tasks.txt --local
    python submit_optimization_jobs.py status <job_dir> --tail 3

The presets encode the cluster facts of Grete and standard96s. Canonical timing trials must share one
hardware identity, so run them with '--throttle 1' on a fixed GRES type.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
# The benchmark's DEFAULT_OUTPUT_ROOT, duplicated so this module does not import torch.
OUTPUT_ROOT = Path("/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization")
JOBS_ROOT = OUTPUT_ROOT / "jobs"
ENV = "new-stack"
PARTITION = "grete:preemptible"
CONSTRAINT = "inet"
N_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 30
DEFAULT_THROTTLE = 8
DEFAULT_TASKS_PER_JOB = 1
PRESET_TASKS_PER_JOB = {"cpu-test": 48}
# Read by common.py at call time; pinned into the job script so a job resolves the same checkpoints.
PINNED_ENV_VARS = ("MICRO_SAM2_JOINT_CHECKPOINT_ROOT", "MICRO_SAM2_JOINT_EXPORT_ROOT")


@dataclasses.dataclass(frozen=True)
class SlurmResources:
    gres: Optional[str]
    mem: str
    time_limit: str
    qos: Optional[str] = None
    cpus: int = 4
    partition: str = PARTITION
    account: Optional[str] = None


PRESETS = {
    "2d": SlurmResources("1g.10gb:1", "16G", "08:00:00"),
    "2d-short": SlurmResources("1g.10gb:1", "16G", "02:00:00", qos="2h"),
    "3d": SlurmResources("2g.20gb:1", "32G", "12:00:00"),
    "3d-large": SlurmResources("2g.20gb:1", "64G", "12:00:00"),
    # Cached AIS/APG sweeps and screens never load the model. The test partition has a hard one-hour
    # limit, which is sufficient for the cache-aware 2d shards and avoids reserving an idle MIG slice.
    "cpu-test": SlurmResources(None, "500G", "00:59:00", cpus=192, partition="standard96s:test"),
    # Small cached screens do not justify an exclusive test node.
    "cpu-shared": SlurmResources(None, "16G", "01:00:00", cpus=4, partition="standard96s:shared"),
    # Legacy long-running CPU preset on the GPU partition; retained for existing campaign commands.
    "cpu": SlurmResources("1g.10gb:1", "64G", "04:00:00", cpus=16),
}

Task = Tuple[str, str]


def sanitize(name: str) -> str:
    """Make a name safe for file names and Slurm job names."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "job"


def validate_tasks(tasks: Sequence[Task]) -> None:
    if not tasks:
        raise ValueError("At least one task is required.")
    seen = set()
    for tag, command in tasks:
        if not tag or not command:
            raise ValueError(f"Empty tag or command in task {tag!r}.")
        if "\t" in tag or "\n" in tag or "\t" in command or "\n" in command:
            raise ValueError(f"Tabs and newlines are not allowed in task {tag!r}.")
        key = sanitize(tag)
        if key in seen:
            raise ValueError(f"Duplicate task tag after sanitizing: {key!r}.")
        seen.add(key)


def write_tasks_file(job_dir: Path, tasks: Sequence[Task]) -> Path:
    """Write one 'tag<TAB>command' line per task."""
    validate_tasks(tasks)
    path = job_dir / "tasks.txt"
    with open(path, "w") as f:
        for tag, command in tasks:
            f.write(f"{sanitize(tag)}\t{command}\n")
    return path


def parse_tasks_file(path: Path) -> List[Task]:
    tasks = []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"{path}:{number}: expected 'tag<TAB>command'.")
            tag, command = line.split("\t", 1)
            tasks.append((tag, command))
    validate_tasks(tasks)
    return tasks


def env_exports() -> str:
    lines = [f"export {name}={shlex.quote(os.environ[name])}" for name in PINNED_ENV_VARS if name in os.environ]
    return "\n".join(lines)


def render_job_script(
    name: str, job_dir: Path, n_tasks: int, resources: SlurmResources, throttle: int = DEFAULT_THROTTLE,
    dependency: Optional[str] = None, attempts: int = N_ATTEMPTS,
    tasks_per_job: int = DEFAULT_TASKS_PER_JOB,
) -> str:
    """Render the Slurm array script. Every '#SBATCH' line precedes the first command."""
    if tasks_per_job < 1:
        raise ValueError("tasks_per_job must be positive.")
    n_array_jobs = (n_tasks + tasks_per_job - 1) // tasks_per_job
    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={sanitize(name)}",
        f"#SBATCH -p {resources.partition}",
    ]
    if resources.gres is not None:
        header.append(f"#SBATCH -G {resources.gres}")
    header.extend([
        f"#SBATCH -c {resources.cpus}",
        f"#SBATCH --mem={resources.mem}",
        f"#SBATCH -t {resources.time_limit}",
        f"#SBATCH --constraint={CONSTRAINT}",
        "#SBATCH --requeue",
        "#SBATCH --open-mode=append",
        f"#SBATCH --array=0-{n_array_jobs - 1}%{throttle}",
        f"#SBATCH -o {job_dir}/logs/{sanitize(name)}_%A_%a.out",
        f"#SBATCH -e {job_dir}/logs/{sanitize(name)}_%A_%a.err",
    ])
    if resources.qos:
        header.append(f"#SBATCH --qos={resources.qos}")
    if resources.account:
        header.append(f"#SBATCH -A {resources.account}")
    if dependency:
        header.append(f"#SBATCH --dependency={dependency}")
    body = f"""
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate {ENV}
cd {REPOSITORY_ROOT}
export PYTHONUNBUFFERED=1
{env_exports()}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {job_dir}/tasks.txt)
tag=$(cut -f1 <<< "$line")
command=$(cut -f2- <<< "$line")
markers={job_dir}/logs
echo "[$(date -Is)] task $SLURM_ARRAY_TASK_ID '$tag' job $SLURM_JOB_ID" \\
  "restart ${{SLURM_RESTART_COUNT:-0}} node $SLURMD_NODENAME"
if [ -f "$markers/$tag.done" ]; then echo "'$tag' is already done."; exit 0; fi
rm -f "$markers/$tag.failed"
started=$SECONDS
rc=1
attempt=0
for attempt in $(seq 1 {attempts}); do
  rc=0
  eval "$command" || rc=$?
  [ $rc -eq 0 ] && break
  echo "[$(date -Is)] attempt $attempt of '$tag' failed with exit $rc."
  sleep {RETRY_SLEEP_SECONDS}
done
elapsed=$((SECONDS - started))
if [ $rc -eq 0 ]; then
  printf 'exit=0 elapsed=%s attempts=%s job=%s restarts=%s node=%s\\n' "$elapsed" "$attempt" "$SLURM_JOB_ID" \\
    "${{SLURM_RESTART_COUNT:-0}}" "$SLURMD_NODENAME" > "$markers/$tag.done"
else
  printf 'exit=%s elapsed=%s attempts=%s job=%s\\n' "$rc" "$elapsed" "$attempt" "$SLURM_JOB_ID" \\
    > "$markers/$tag.failed"
fi
exit $rc
"""
    if tasks_per_job > 1:
        body = f"""
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate {ENV}
cd {REPOSITORY_ROOT}
export PYTHONUNBUFFERED=1
{env_exports()}
markers={job_dir}/logs

run_task() {{
  local task_index="$1"
  local line tag command started rc attempt elapsed
  line=$(sed -n "$((task_index + 1))p" {job_dir}/tasks.txt)
  tag=$(cut -f1 <<< "$line")
  command=$(cut -f2- <<< "$line")
  echo "[$(date -Is)] task $task_index '$tag' array $SLURM_ARRAY_TASK_ID job $SLURM_JOB_ID" \\
    "restart ${{SLURM_RESTART_COUNT:-0}} node $SLURMD_NODENAME"
  if [ -f "$markers/$tag.done" ]; then echo "'$tag' is already done."; return 0; fi
  rm -f "$markers/$tag.failed"
  started=$SECONDS
  rc=1
  attempt=0
  for attempt in $(seq 1 {attempts}); do
    rc=0
    eval "$command" >> "$markers/$tag.out" 2>> "$markers/$tag.err" || rc=$?
    [ $rc -eq 0 ] && break
    echo "[$(date -Is)] attempt $attempt of '$tag' failed with exit $rc."
    sleep {RETRY_SLEEP_SECONDS}
  done
  elapsed=$((SECONDS - started))
  if [ $rc -eq 0 ]; then
    printf 'exit=0 elapsed=%s attempts=%s job=%s restarts=%s node=%s\\n' \\
      "$elapsed" "$attempt" "$SLURM_JOB_ID" "${{SLURM_RESTART_COUNT:-0}}" "$SLURMD_NODENAME" \\
      > "$markers/$tag.done"
  else
    printf 'exit=%s elapsed=%s attempts=%s job=%s\\n' "$rc" "$elapsed" "$attempt" "$SLURM_JOB_ID" \\
      > "$markers/$tag.failed"
  fi
  return "$rc"
}}

first_task=$((SLURM_ARRAY_TASK_ID * {tasks_per_job}))
last_task=$((first_task + {tasks_per_job}))
[ "$last_task" -gt {n_tasks} ] && last_task={n_tasks}
pids=()
for ((task_index=first_task; task_index<last_task; task_index++)); do
  run_task "$task_index" &
  pids+=("$!")
done
rc=0
for pid in "${{pids[@]}}"; do
  if ! wait "$pid"; then rc=1; fi
done
exit "$rc"
"""
    return "\n".join(header) + "\n" + body


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=REPOSITORY_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_job_dir(
    name: str, tasks: Sequence[Task], resources: SlurmResources, jobs_root: Path = JOBS_ROOT,
    throttle: int = DEFAULT_THROTTLE, dependency: Optional[str] = None, attempts: int = N_ATTEMPTS,
    argv: Optional[Sequence[str]] = None, tasks_per_job: int = DEFAULT_TASKS_PER_JOB,
) -> Path:
    """Create '<jobs_root>/<timestamp>_<name>/' with tasks.txt, job.sh, logs/ and submit.json."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = jobs_root / f"{stamp}_{sanitize(name)}"
    (job_dir / "logs").mkdir(parents=True, exist_ok=False)
    write_tasks_file(job_dir, tasks)
    script = render_job_script(
        name, job_dir, len(tasks), resources, throttle, dependency, attempts, tasks_per_job,
    )
    (job_dir / "job.sh").write_text(script)
    record = {
        "name": name,
        "argv": list(argv) if argv is not None else sys.argv,
        "resources": dataclasses.asdict(resources),
        "n_tasks": len(tasks),
        "tasks_per_job": tasks_per_job,
        "n_array_jobs": (len(tasks) + tasks_per_job - 1) // tasks_per_job,
        "throttle": throttle,
        "dependency": dependency,
        "attempts": attempts,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "cwd": os.getcwd(),
        "created": stamp,
    }
    (job_dir / "submit.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return job_dir


def submit_script(job_dir: Path) -> str:
    """Submit job.sh with sbatch and record the job id."""
    result = subprocess.run(
        ["sbatch", "--parsable", str(job_dir / "job.sh")], capture_output=True, text=True, check=False,
    )
    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0 or not output:
        raise RuntimeError(f"sbatch failed: {output}")
    job_id = output.split(";")[0].strip()
    (job_dir / "job_id.txt").write_text(job_id + "\n")
    print(f"Submitted {job_id}: {job_dir}")
    return job_id


def _marker_state(job_dir: Path, tag: str) -> Tuple[str, str]:
    for state in ("done", "failed"):
        marker = job_dir / "logs" / f"{sanitize(tag)}.{state}"
        if marker.exists():
            return state, marker.read_text().strip()
    return "pending", ""


def run_local(job_dir: Path, tasks: Sequence[Task], attempts: int = 1) -> int:
    """Run the tasks sequentially in the current environment; return the number of failures."""
    logs = job_dir / "logs"
    failures = 0
    for tag, command in tasks:
        tag = sanitize(tag)
        state, _ = _marker_state(job_dir, tag)
        if state == "done":
            print(f"'{tag}' is already done.")
            continue
        (logs / f"{tag}.failed").unlink(missing_ok=True)
        started = datetime.datetime.now()
        rc = 1
        for attempt in range(1, attempts + 1):
            with open(logs / f"{tag}.out", "a") as out, open(logs / f"{tag}.err", "a") as err:
                out.write(f"[{datetime.datetime.now().isoformat()}] local attempt {attempt}: {command}\n")
                out.flush()
                rc = subprocess.run(
                    ["bash", "-o", "pipefail", "-c", command], cwd=REPOSITORY_ROOT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}, stdout=out, stderr=err, check=False,
                ).returncode
            if rc == 0:
                break
        elapsed = int((datetime.datetime.now() - started).total_seconds())
        marker = "done" if rc == 0 else "failed"
        (logs / f"{tag}.{marker}").write_text(f"exit={rc} elapsed={elapsed} attempts={attempt} job=local\n")
        print(f"{tag}: {marker} (exit {rc}, {elapsed} s)")
        failures += int(rc != 0)
    return failures


def filter_resume(tasks: Sequence[Task], resume_from: Optional[Path]) -> List[Task]:
    """Drop tasks that already have a '.done' marker in an earlier job dir."""
    if resume_from is None:
        return list(tasks)
    kept = [task for task in tasks if _marker_state(resume_from, task[0])[0] != "done"]
    print(f"Resuming: {len(tasks) - len(kept)} of {len(tasks)} tasks are already done in {resume_from}.")
    return kept


def submit_tasks(
    tasks: Sequence[Task], name: str, resources: SlurmResources, throttle: int = DEFAULT_THROTTLE,
    dependency: Optional[str] = None, attempts: int = N_ATTEMPTS, dry_run: bool = False, local: bool = False,
    jobs_root: Path = JOBS_ROOT, resume_from: Optional[Path] = None, argv: Optional[Sequence[str]] = None,
    tasks_per_job: int = DEFAULT_TASKS_PER_JOB,
) -> Tuple[Optional[Path], Optional[str]]:
    """The Python entry point the job builders call. Returns (job_dir, job_id)."""
    tasks = filter_resume(tasks, resume_from)
    if not tasks:
        print("Nothing to do.")
        return None, None
    job_dir = write_job_dir(
        name, tasks, resources, jobs_root, throttle, dependency, attempts, argv, tasks_per_job,
    )
    n_array_jobs = (len(tasks) + tasks_per_job - 1) // tasks_per_job
    print(f"Job directory: {job_dir} ({len(tasks)} tasks packed into {n_array_jobs} array jobs)")
    if dry_run:
        print((job_dir / "job.sh").read_text())
        return job_dir, None
    if local:
        failures = run_local(job_dir, tasks, attempts)
        if failures:
            print(f"{failures} task(s) failed; see {job_dir / 'logs'}.")
        return job_dir, None
    return job_dir, submit_script(job_dir)


def warn_missing_env() -> None:
    try:
        listing = subprocess.run(["micromamba", "env", "list"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return
    if listing and ENV not in listing:
        print(f"Warning: micromamba environment '{ENV}' was not found on this host.", file=sys.stderr)


# ----------------------------------------------------------------------------------------------
# status


def _sacct_rows(job_id: str) -> List[Dict[str, str]]:
    fields = ["JobID", "State", "ExitCode", "Elapsed", "Restarts", "NodeList"]
    result = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-P", "-n", "--format=" + ",".join(fields)],
        capture_output=True, text=True, check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= len(fields):
            rows.append(dict(zip(fields, parts[:len(fields)])))
    return rows


def _expand_array_ids(job_id: str) -> List[int]:
    """'15465049_[2-5,9%8]' -> [2, 3, 4, 5, 9]; '15465049_3' -> [3]; '15465049' -> []."""
    match = re.match(r"^\d+_(\[(.*)\]|(\d+))$", job_id)
    if match is None:
        return []
    if match.group(3) is not None:
        return [int(match.group(3))]
    spec = match.group(2).split("%")[0]
    ids: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            start, stop = part.split("-")
            ids.extend(range(int(start), int(stop) + 1))
        elif part:
            ids.append(int(part))
    return ids


def _last_line(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = f.read()[-4096:]
    lines = [line for line in data.decode("utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def status(job_dir: Path, tail: int = 1) -> int:
    """Print one line per task with the Slurm state, the marker and the last log line. Exit 1 on failures."""
    tasks = parse_tasks_file(job_dir / "tasks.txt")
    job_id_path = job_dir / "job_id.txt"
    states: Dict[int, Dict[str, str]] = {}
    if job_id_path.exists():
        for row in _sacct_rows(job_id_path.read_text().strip()):
            for index in _expand_array_ids(row["JobID"]):
                states[index] = row
    failing = 0
    submission = json.loads((job_dir / "submit.json").read_text())
    name = sanitize(submission["name"])
    tasks_per_job = int(submission.get("tasks_per_job", 1))
    for index, (tag, _) in enumerate(tasks):
        row = states.get(index // tasks_per_job, {})
        marker, marker_text = _marker_state(job_dir, tag)
        slurm_state = row.get("State", "-")
        log = job_dir / "logs" / f"{tag}.out"
        if not log.exists() and row.get("JobID"):
            log = job_dir / "logs" / f"{name}_{row['JobID'].replace('_', '_')}.out"
        last = _last_line(log) if tail else ""
        restarts = row.get("Restarts", "0")
        flag = " RESTARTED" if restarts not in ("", "0") else ""
        print(f"{index:4d} {tag:40s} {slurm_state:12s} exit={row.get('ExitCode', '-'):6s} "
              f"{row.get('Elapsed', '-'):10s} {marker:8s}{flag} {marker_text} | {last[:80]}")
        if marker == "failed" or any(word in slurm_state for word in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF")):
            failing += 1
    print(f"{failing} failing task(s) of {len(tasks)}.")
    return int(failing > 0)


# ----------------------------------------------------------------------------------------------
# CLI


def add_submit_arguments(parser: argparse.ArgumentParser) -> None:
    """The submission options shared by this CLI and the job builders."""
    parser.add_argument("--name", required=True, help="Job name; also names the job directory.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="2d")
    parser.add_argument("--gres", default=None, help="Override the preset GRES, e.g. 1g.20gb:1.")
    parser.add_argument("--mem", default=None)
    parser.add_argument("--time", default=None, help="Slurm time limit, e.g. 04:00:00.")
    parser.add_argument("--qos", default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--throttle", type=int, default=DEFAULT_THROTTLE, help="Concurrent array tasks.")
    parser.add_argument("--dependency", default=None, help="Slurm dependency, e.g. afterok:123.")
    parser.add_argument("--attempts", type=int, default=N_ATTEMPTS, help="In-process retries per task.")
    parser.add_argument(
        "--tasks-per-job", type=int, default=None,
        help="Concurrent task-file commands per Slurm array element (preset-dependent by default).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the job directory, do not submit.")
    parser.add_argument("--local", action="store_true", help="Run the tasks here, sequentially.")
    parser.add_argument("--jobs-root", type=Path, default=JOBS_ROOT)
    parser.add_argument("--resume-from", type=Path, default=None, help="Skip tasks done in this job dir.")


def resolve_resources(args: argparse.Namespace) -> SlurmResources:
    resources = PRESETS[args.preset]
    overrides = {
        "gres": args.gres, "mem": args.mem, "time_limit": args.time, "qos": args.qos, "cpus": args.cpus,
        "partition": args.partition, "account": args.account,
    }
    return dataclasses.replace(resources, **{key: value for key, value in overrides.items() if value is not None})


def submit_from_args(tasks: Sequence[Task], args: argparse.Namespace) -> Tuple[Optional[Path], Optional[str]]:
    if not args.local and not args.dry_run:
        warn_missing_env()
    tasks_per_job = args.tasks_per_job
    if tasks_per_job is None:
        tasks_per_job = PRESET_TASKS_PER_JOB.get(args.preset, DEFAULT_TASKS_PER_JOB)
    if tasks_per_job < 1:
        raise ValueError("--tasks-per-job must be positive.")
    return submit_tasks(
        tasks, args.name, resolve_resources(args), throttle=args.throttle, dependency=args.dependency,
        attempts=args.attempts, dry_run=args.dry_run, local=args.local, jobs_root=args.jobs_root,
        resume_from=args.resume_from, tasks_per_job=tasks_per_job,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit", help="Submit a tasks file as one array job.")
    submit.add_argument("--tasks-file", type=Path, required=True)
    add_submit_arguments(submit)
    show = subparsers.add_parser("status", help="Show the state of every task of a job directory.")
    show.add_argument("job_dir", type=Path)
    show.add_argument("--tail", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "status":
        return status(args.job_dir, args.tail)
    submit_from_args(parse_tasks_file(args.tasks_file), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
