"""Derive the common iteration count N from the calibration results (plan section 5).

N = 0.8 x budget / (seconds per iteration of the slowest arm), rounded down to whole epochs. The other 20 % of
the budget covers start-up, validation and checkpointing; the script also reports whether the measured validation
time fits into it.

N comes from the step-1 arms only (`--arms`); step-2 arms keep that N and are listed with their expected training
hours, so their wall time can be set.
"""

import argparse
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_arch_lib as lib  # noqa: E402

ITERATIONS_PER_EPOCH = 600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=lib.EXPERIMENT_ROOT)
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--arms", nargs="+", default=list(lib.STEP1_ARMS), help="the arms that determine N")
    args = parser.parse_args()

    results = {}
    for path in sorted(glob(os.path.join(args.root, "calibration", "arm*.json"))):
        with open(path) as f:
            r = json.load(f)
        results[r["arm"]] = r
    if not results:
        raise SystemExit("No calibration results.")

    print(f"{'arm':4s} {'decoder':12s} {'M params':>8s} {'s/it':>6s} {'gpu2d':>6s} {'gpu3d':>6s} {'load2d':>7s} "
          f"{'load3d':>7s} {'peak GiB':>8s} {'val s':>6s}")
    for arm, r in sorted(results.items()):
        print(f"{arm:4s} {r['decoder_type']:12s} {r['decoder_parameters_M']:8.2f} {r['seconds_per_iteration']:6.3f} "
              f"{r['gpu_2d']['seconds_per_step']:6.3f} {r['gpu_3d']['seconds_per_step']:6.3f} "
              f"{r['loader']['2d_seconds_per_batch']:7.3f} {r['loader']['3d_seconds_per_batch']:7.3f} "
              f"{max(r['peak_GiB_fit'], r['gpu_2d']['peak_GiB'], r['gpu_3d']['peak_GiB']):8.1f} "
              f"{r['validation_seconds']:6.0f}")

    slowest = max((r for arm, r in results.items() if arm in args.arms), key=lambda r: r["seconds_per_iteration"])
    seconds = slowest["seconds_per_iteration"]
    n_raw = args.fraction * args.budget_hours * 3600 / seconds
    n_epochs = int(n_raw // ITERATIONS_PER_EPOCH)
    n = n_epochs * ITERATIONS_PER_EPOCH
    validation_hours = n_epochs * slowest["validation_seconds"] / 3600
    expected_hours = (n * seconds) / 3600 + validation_hours
    summary = {
        "slowest_arm": slowest["arm"], "seconds_per_iteration": seconds, "iterations": n, "epochs": n_epochs,
        "expected_hours_slowest": expected_hours, "validation_hours_slowest": validation_hours,
        "budget_hours": args.budget_hours, "arms": sorted(a for a in results if a in args.arms),
        "expected_hours": {
            arm: (n * r["seconds_per_iteration"] + n_epochs * r["validation_seconds"]) / 3600
            for arm, r in sorted(results.items())
        },
    }
    print(json.dumps(summary, indent=1))
    if expected_hours > 0.95 * args.budget_hours:
        print("WARNING: training plus validation exceeds 95 % of the budget.")
    name = "summary.json" if sorted(args.arms) == sorted(lib.STEP1_ARMS) else f"summary_{''.join(args.arms)}.json"
    with open(os.path.join(args.root, "calibration", name), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
