"""Stage a trained AIS decoder as a lean joint-format checkpoint for the evaluation harness.

Writes <staged-root>/joint_sam2_hvit_t_multi_gpu/<variant>.pt with the v4 SAM2 'model_state' (the frozen
encoder equals the v4 encoder, so the interactive half is the production one) and the trained 'unetr_state'.
Then `export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<staged-root>` and select the model with
`--joint-checkpoint <variant>` in benchmark_ais_optimization.py; the checkpoint id is the file's checksum.

    python stage_ais_decoder_checkpoint.py --variant contact [--which best|latest]
"""

import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ais_decoder_lib as lib  # noqa: E402,F401  (registers the classes the trainer pickled)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(lib.VARIANTS))
    parser.add_argument("--which", default="best", choices=["best", "latest"])
    parser.add_argument("--save-root", default=lib.CAMPAIGN_ROOT)
    parser.add_argument("--staged-root", default=None, help="Default <save-root>/staged.")
    parser.add_argument("--name", default=None, help="Checkpoint name, default ais_decoder_<variant>.")
    parser.add_argument("--v4-checkpoint", default=lib.V4_CHECKPOINT)
    args = parser.parse_args()

    name = args.name or f"ais_decoder_{args.variant}"
    checkpoint_path = os.path.join(args.save_root, "checkpoints", name, f"{args.which}.pt")
    staged_root = args.staged_root or os.path.join(args.save_root, "staged")
    staged_dir = os.path.join(staged_root, f"joint_sam2_{lib.MODEL_TYPE}_multi_gpu")
    os.makedirs(staged_dir, exist_ok=True)
    staged_path = os.path.join(staged_dir, f"{args.variant}.pt")

    trained = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    unetr_state = {k: v for k, v in trained["model_state"].items()}
    v4 = lib.load_lean_v4_states(args.v4_checkpoint, args.save_root)
    # The encoder was frozen: the trained state must carry the v4 encoder unchanged.
    for key, value in v4["unetr_state"].items():
        if key.startswith("encoder.") and not torch.equal(value, unetr_state[key]):
            raise RuntimeError(f"The encoder weights changed during training ({key}); refusing to stage.")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        revision = None
    lean = {
        "model_state": v4["model_state"],
        "unetr_state": unetr_state,
        "source": {
            "variant": args.variant, "checkpoint": checkpoint_path, "which": args.which,
            "iteration": int(trained.get("iteration", -1)), "epoch": int(trained.get("epoch", -1)),
            "best_epoch": int(trained.get("best_epoch", -1)),
            "best_metric": float(trained.get("best_metric", float("nan"))),
            "current_metric": float(trained.get("current_metric", float("nan"))), "git_revision": revision,
            "output_channels": int(unetr_state["out_conv.weight"].shape[0]),
        },
    }
    tmp_path = f"{staged_path}.tmp.{os.getpid()}"
    torch.save(lean, tmp_path)
    os.replace(tmp_path, staged_path)
    with open(os.path.join(staged_dir, f"{args.variant}.json"), "w") as f:
        json.dump(lean["source"], f, indent=2)
    print(f"staged {checkpoint_path} -> {staged_path}")
    print(json.dumps(lean["source"], indent=2))
    print(f"export MICRO_SAM2_JOINT_CHECKPOINT_ROOT={staged_root}")


if __name__ == "__main__":
    main()
