"""Export a jointly-trained SAM2 + UniSAM2 checkpoint into loadable micro-sam weights.

The joint multi-GPU trainer (`micro_sam.v2.training.joint_sam2_trainer`) stores a single
torch_em checkpoint that bundles the interactive SAM2 weights (under 'model_state'), the
automatic UniSAM2 decoder weights (under 'unetr_state') and a lot of non-tensor trainer state
(optimizer, datasets, ...). That checkpoint cannot be loaded directly because:

- `sam2.build_sam._load_checkpoint` reads `torch.load(...)["model"]` with `weights_only=True`,
  so it needs the SAM2 weights under a 'model' key and rejects the pickled trainer objects.
- `micro_sam.v2.automatic_segmentation.get_unisam2_model` needs the UniSAM2 state dict.

This script splits the joint checkpoint into the two-file, micro-sam-v1-style layout:

- `<name>.pt` -> {'model': <SAM2 weights>, 'model_type': 'hvit_t'} (interactive predictor)
- `<name>_decoder.pt` -> <UniSAM2 state dict> (automatic instance segmentation decoder)

Both files are plain tensor dicts, so they load with `weights_only=True` and can be registered
in the micro-sam SAM2 model download console (see `micro_sam.v2.util.models`).
"""

import os
import argparse
from collections import OrderedDict

import torch
import xxhash


def _strip_ddp_prefix(state_dict):
    """Remove a leading 'module.' from keys, in case the checkpoint was saved DDP-wrapped."""
    return OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v) for k, v in state_dict.items()
    )


def _compute_xxh128(path):
    """Compute the xxh128 hash of a file, matching the format used by the micro-sam registry."""
    hasher = xxhash.xxh128()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return f"xxh128:{hasher.hexdigest()}"


def export_joint_model(checkpoint_path, output_folder, name, base_model_type):
    """Split a joint SAM2 + UniSAM2 checkpoint into a predictor file and a decoder file.

    Args:
        checkpoint_path: Path to the joint trainer checkpoint (e.g. 'best.pt').
        output_folder: Directory where the exported files are written.
        name: Base name for the exported files, e.g. 'hvit_t_cells'.
        base_model_type: The SAM2 backbone the model was trained from, e.g. 'hvit_t'.

    Returns:
        A dict mapping each registry key to (filepath, xxh128 hash).
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in state or "unetr_state" not in state:
        raise RuntimeError(
            f"Expected a joint checkpoint with 'model_state' and 'unetr_state', got keys: {list(state.keys())}"
        )

    model_state = _strip_ddp_prefix(state["model_state"])
    unetr_state = _strip_ddp_prefix(state["unetr_state"])

    os.makedirs(output_folder, exist_ok=True)
    encoder_path = os.path.join(output_folder, f"{name}.pt")
    decoder_path = os.path.join(output_folder, f"{name}_decoder.pt")

    # The interactive predictor file mirrors a native SAM2 checkpoint: weights under 'model'.
    torch.save({"model": model_state, "model_type": base_model_type}, encoder_path)
    # The decoder file is the raw UniSAM2 state dict, mirroring the v1 '<model>_decoder.pt' layout.
    torch.save(unetr_state, decoder_path)

    return {
        name: (encoder_path, _compute_xxh128(encoder_path)),
        f"{name}_decoder": (decoder_path, _compute_xxh128(decoder_path)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="Path to the joint SAM2 + UniSAM2 trainer checkpoint.",
    )
    parser.add_argument(
        "-o", "--output_folder", default=".",
        help="Folder where the exported '<name>.pt' and '<name>_decoder.pt' files are written.",
    )
    parser.add_argument("-n", "--name", default="hvit_t_cells", help="Base name for the exported files.")
    parser.add_argument("--base_model_type", default="hvit_t", help="The SAM2 backbone the model was trained from.")
    args = parser.parse_args()

    results = export_joint_model(args.checkpoint, args.output_folder, args.name, args.base_model_type)

    print("Exported the following files. Add these entries to 'micro_sam.v2.util.models':")
    for key, (path, file_hash) in results.items():
        size_mb = os.path.getsize(path) / 1e6
        print(f"{key}: {path} ({size_mb:.1f} MB)")
        print(f"registry hash for {key}: \"{file_hash}\"")


if __name__ == "__main__":
    main()
