"""Check that the documented conda-forge install commands give the latest micro_sam on all platforms.

Solves every command as a dry run with micromamba for the given platform. The check fails if a solve fails,
if it selects an older micro_sam than the latest one on conda-forge, or if a GPU command selects a CPU pytorch.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

PLATFORMS = ["linux-64", "linux-aarch64", "osx-64", "osx-arm64", "win-64"]

CPU_SPECS = ["micro_sam>=2", "python=3.12"]
LINUX_GPU_SPECS = CPU_SPECS + ["pytorch=*=cuda*", "cuda-version=12"]
WINDOWS_GPU_SPECS = CPU_SPECS + ["pytorch=*=cuda*"]

# Known problems, reported as warnings instead of failures. Remove an entry when the problem is fixed.
KNOWN_ISSUES = {
    ("linux-aarch64", "cpu"): "conda-forge has no PyQt6 for linux-aarch64 that napari accepts.",
    ("linux-aarch64", "unpinned"): "conda-forge has no PyQt6 for linux-aarch64 that napari accepts.",
    ("osx-64", "unpinned"): "segment-anything needs onnxruntime, which has no osx-64 build for python 3.14.",
}


def get_cases(platform):
    """Get the install commands to check for a platform.

    Args:
        platform: The conda platform, e.g. 'linux-64'.

    Returns:
        A list of (name, specs, cuda version of the virtual driver, whether pytorch must be a CUDA build).
    """
    cases = [("unpinned", ["micro_sam"], None, False), ("cpu", CPU_SPECS, None, False)]
    if platform == "linux-64":
        cases += [(f"gpu-cuda{cuda}", LINUX_GPU_SPECS, cuda, True) for cuda in ("12.4", "12.8", "13.0")]
    elif platform == "win-64":
        cases += [(f"gpu-cuda{cuda}", WINDOWS_GPU_SPECS, cuda, True) for cuda in ("12.4", "12.8", "13.0")]
    return cases


def get_latest_version():
    """Get the latest micro_sam version on conda-forge."""
    url = "https://api.anaconda.org/package/conda-forge/micro_sam"
    with urllib.request.urlopen(url, timeout=60) as f:
        return json.load(f)["latest_version"]


def solve(platform, specs, cuda):
    """Solve an install command as a dry run.

    Args:
        platform: The conda platform.
        specs: The package specs.
        cuda: The CUDA version of the virtual NVIDIA driver, or None for no driver.

    Returns:
        A dict that maps the package names to (version, build), or the error message if the solve fails.
    """
    env = dict(os.environ, CONDA_SUBDIR=platform, CONDA_OVERRIDE_CUDA=cuda or "", CONDA_OVERRIDE_GLIBC="2.28")
    if platform.startswith("osx"):
        env["CONDA_OVERRIDE_OSX"] = "13.0"
    cmd = [
        "micromamba", "create", "-n", "check-install", "--dry-run", "--json", "-y",
        "--override-channels", "-c", "conda-forge", "--platform", platform,
    ] + specs
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        return (result.stderr or result.stdout).strip()[-1000:]
    if result.returncode != 0 or not output.get("success", False):
        return json.dumps(output)[-1000:]
    return {pkg["name"]: (pkg["version"], pkg["build_string"]) for pkg in output["actions"]["LINK"]}


def check_case(platform, name, specs, cuda, needs_cuda_torch, latest):
    """Check one install command.

    Returns:
        The table row and the list of problems.
    """
    packages = solve(platform, specs, cuda)
    if isinstance(packages, str):
        row = f"| {platform} | {name} | solve failed | - | - |"
        return row, [f"{platform} {name}: the solve failed: {packages}"]

    problems = []
    micro_sam_version = packages.get("micro_sam", ("missing", ""))[0]
    torch_version, torch_build = packages.get("pytorch", ("missing", ""))
    python_version = packages.get("python", ("missing", ""))[0]
    if micro_sam_version != latest:
        problems.append(f"{platform} {name}: micro_sam {micro_sam_version} instead of {latest}")
    if needs_cuda_torch and "cuda" not in torch_build:
        problems.append(f"{platform} {name}: a CPU pytorch ({torch_version} {torch_build}) for a CUDA {cuda} driver")
    row = f"| {platform} | {name} | {micro_sam_version} | {torch_version} {torch_build} | {python_version} |"
    return row, problems


def write_summary(lines):
    """Write the result table to the GitHub step summary, if available."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    args = parser.parse_args()

    latest = get_latest_version()
    lines = [
        f"### {args.platform} (latest micro_sam: {latest})", "",
        "| platform | command | micro_sam | pytorch | python |", "|---|---|---|---|---|",
    ]
    failures, warnings = [], []
    for name, specs, cuda, needs_cuda_torch in get_cases(args.platform):
        row, problems = check_case(args.platform, name, specs, cuda, needs_cuda_torch, latest)
        print(row, flush=True)
        lines.append(row)
        known_issue = KNOWN_ISSUES.get((args.platform, name))
        if problems and known_issue is not None:
            warnings += [f"{problem} (known issue: {known_issue})" for problem in problems]
        else:
            failures += problems

    lines += [""] + [f"- warning: {warning}" for warning in warnings] + [f"- error: {failure}" for failure in failures]
    write_summary(lines)
    for warning in warnings:
        print(f"::warning::{warning}")
    for failure in failures:
        print(f"::error::{failure}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
