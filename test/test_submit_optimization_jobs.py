import sys
import shlex
from pathlib import Path

import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

import submit_optimization_jobs as soj  # noqa
import apg_campaign_tasks as campaign  # noqa


def _tasks_file(tmp_path, lines):
    path = tmp_path / "tasks.txt"
    path.write_text("".join(f"{tag}\t{command}\n" for tag, command in lines))
    return path


def _only_job_dir(root):
    dirs = [path for path in root.iterdir() if path.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_dry_run_writes_job_dir(tmp_path):
    tasks = _tasks_file(tmp_path, [("a", "echo a"), ("b", "echo b")])
    jobs_root = tmp_path / "jobs"
    soj.main(["submit", "--name", "unit", "--preset", "2d", "--tasks-file", str(tasks),
              "--dry-run", "--jobs-root", str(jobs_root)])
    job_dir = _only_job_dir(jobs_root)
    assert job_dir.name.endswith("_unit") and job_dir.name[:8].isdigit()
    for name in ("tasks.txt", "job.sh", "logs", "submit.json"):
        assert (job_dir / name).exists()
    assert not (job_dir / "job_id.txt").exists()


def test_job_script_header_and_activation_order(tmp_path):
    resources = soj.PRESETS["2d"]
    script = soj.render_job_script("unit", tmp_path, 2, resources)
    lines = script.splitlines()
    sbatch = [line for line in lines if line.startswith("#SBATCH")]
    for expected in (
        "#SBATCH -p grete:preemptible", "#SBATCH -G 1g.10gb:1", "#SBATCH --mem=16G", "#SBATCH -t 08:00:00",
        "#SBATCH --constraint=inet", "#SBATCH --requeue", "#SBATCH --open-mode=append", "#SBATCH --array=0-1%8",
    ):
        assert expected in sbatch, expected
    assert not any("--qos" in line for line in sbatch)
    first_command = next(i for i, line in enumerate(lines) if line and not line.startswith("#"))
    assert all(i < first_command for i, line in enumerate(lines) if line.startswith("#SBATCH"))
    order = [
        lines.index("set -eo pipefail"), lines.index("source ~/.bashrc"), lines.index("set -u"),
        lines.index("micromamba activate super"), lines.index(f"cd {soj.REPOSITORY_ROOT}"),
        lines.index("export PYTHONUNBUFFERED=1"),
    ]
    assert order == sorted(order)
    assert "set -euo" not in script
    assert "${SLURM_RESTART_COUNT:-0}" in script
    assert "$SLURM_RESTART_COUNT " not in script and "$SLURM_RESTART_COUNT\"" not in script


def test_preset_overrides_and_optional_lines(tmp_path):
    tasks = _tasks_file(tmp_path, [("a", "echo a")])
    jobs_root = tmp_path / "jobs"
    soj.main(["submit", "--name", "unit", "--preset", "3d", "--time", "04:00:00", "--qos", "2h",
              "--dependency", "afterok:123", "--tasks-file", str(tasks), "--dry-run", "--jobs-root", str(jobs_root)])
    script = (_only_job_dir(jobs_root) / "job.sh").read_text()
    for expected in ("#SBATCH -G 2g.20gb:1", "#SBATCH --mem=32G", "#SBATCH -t 04:00:00", "#SBATCH --qos=2h",
                     "#SBATCH --dependency=afterok:123", "#SBATCH --array=0-0%8"):
        assert expected in script, expected


def test_tasks_file_roundtrip_and_validation(tmp_path):
    tasks = [("first task", "echo 1"), ("second", "echo 2 | cat")]
    path = soj.write_tasks_file(tmp_path, tasks)
    for line in path.read_text().splitlines():
        assert line.count("\t") == 1
    parsed = soj.parse_tasks_file(path)
    assert parsed == [("first-task", "echo 1"), ("second", "echo 2 | cat")]
    with pytest.raises(ValueError):
        soj.validate_tasks([("a", "echo"), ("a", "echo")])
    with pytest.raises(ValueError):
        soj.validate_tasks([("a", "echo\tb")])
    with pytest.raises(ValueError):
        soj.validate_tasks([])


@pytest.mark.skipif(sys.platform == "win32", reason="bash")
def test_local_mode_marks_done_and_failed(tmp_path):
    tasks = _tasks_file(tmp_path, [("ok", "echo hello"), ("bad", "false")])
    jobs_root = tmp_path / "jobs"
    soj.main(["submit", "--name", "unit", "--tasks-file", str(tasks), "--local", "--attempts", "1",
              "--jobs-root", str(jobs_root)])
    job_dir = _only_job_dir(jobs_root)
    assert "exit=0" in (job_dir / "logs" / "ok.done").read_text()
    assert "hello" in (job_dir / "logs" / "ok.out").read_text()
    assert (job_dir / "logs" / "bad.failed").exists()
    # A second run skips the finished task and retries the failed one.
    failures = soj.run_local(job_dir, soj.parse_tasks_file(job_dir / "tasks.txt"), attempts=1)
    assert failures == 1
    assert [line for line in (job_dir / "logs" / "ok.out").read_text().splitlines() if line == "hello"] == ["hello"]


def test_status_maps_sacct_rows_to_tags(tmp_path, monkeypatch, capsys):
    tasks = _tasks_file(tmp_path, [("a", "echo"), ("b", "echo"), ("c", "echo"), ("d", "echo")])
    jobs_root = tmp_path / "jobs"
    soj.main(["submit", "--name", "unit", "--tasks-file", str(tasks), "--dry-run", "--jobs-root", str(jobs_root)])
    job_dir = _only_job_dir(jobs_root)
    (job_dir / "job_id.txt").write_text("15465049\n")
    rows = [
        {"JobID": "15465049_0", "State": "COMPLETED", "ExitCode": "0:0", "Elapsed": "00:12:54", "Restarts": "0",
         "NodeList": "ggpu101"},
        {"JobID": "15465049_1", "State": "FAILED", "ExitCode": "1:0", "Elapsed": "00:00:40", "Restarts": "1",
         "NodeList": "ggpu102"},
        {"JobID": "15465049_[2-3%8]", "State": "PENDING", "ExitCode": "0:0", "Elapsed": "00:00:00", "Restarts": "0",
         "NodeList": "None assigned"},
    ]
    monkeypatch.setattr(soj, "_sacct_rows", lambda job_id: rows)
    assert soj.status(job_dir) == 1
    out = capsys.readouterr().out
    assert "a " in out and "COMPLETED" in out and "FAILED" in out and "RESTARTED" in out
    assert out.count("PENDING") == 2
    assert soj._expand_array_ids("15465049_[2-5,9%8]") == [2, 3, 4, 5, 9]
    assert soj._expand_array_ids("15465049_3") == [3]
    assert soj._expand_array_ids("15465049") == []


def test_benchmark_builder(tmp_path):
    config = tmp_path / "apg_my_config.json"
    config.write_text("{}")
    tasks = campaign.benchmark_tasks([None, config], ["trial-1", "trial-2"], ndim="2", subset="holdout")
    tags = [tag for tag, _ in tasks]
    assert len(tags) == len(set(tags)) == 4
    for _, command in tasks:
        assert "--trial-id" in command and "--ndim 2" in command and "--subset holdout" in command
    arguments = [shlex.split(command) for _, command in tasks]
    assert any(
        args[args.index("--config") + 1] == str(config.resolve()) for args in arguments if "--config" in args
    )
    assert any("my_config" in tag for tag in tags)
    serial = campaign.benchmark_tasks([config], ["trial-1"], serialize=True, bracket=True)
    assert len(serial) == 1
    command = serial[0][1]
    parts = command.split(" && ")
    assert len(parts) == 3
    assert "bracket-pre" in parts[0] and "--config" not in parts[0]
    assert "--config" in parts[1]
    assert "bracket-post" in parts[2] and "--config" not in parts[2]


def test_per_sample_builder_and_indices():
    assert campaign.parse_indices("1-3,7") == [1, 2, 3, 7]
    tasks = campaign.per_sample_tasks(Path("optimization/x.py"), [1, 2, 3, 7], extra=["--subset", "primary"])
    assert len(tasks) == 4
    assert tasks[-1][1].endswith("--sample-index 7")
    assert "--subset primary" in tasks[0][1]
    assert all(path.exists() for path in campaign.SCRIPTS.values())
