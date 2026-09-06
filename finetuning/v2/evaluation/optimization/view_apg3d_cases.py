"""Open one packaged 3d case (`package_apg3d_cases.py`) in napari.

Layers: the raw volume, the ground truth, one labels layer per segmentation (v2 / v4 checkpoint, volume
defaults / points+boxes refinement), and per run three points layers with the anchors: every proposed
density-ladder candidate (grey), the candidates that passed the anchor scoring and were propagated (yellow),
and the ones whose track is in the output (green). All segmentation layers but the v4 defaults start hidden;
toggle them with the eye icons. The scores of every run are printed to the terminal.

Usage:
    python view_apg3d_cases.py /path/to/3d_cases/primary/gonuclear__gonuclear_1234abcd.h5
    python view_apg3d_cases.py <file.h5> --no-prompts
"""

from __future__ import annotations

import argparse
import json
import sys

import h5py
import napari


COLORS = {"all": "lightgray", "scored": "yellow", "merged": "lime"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--no-prompts", action="store_true", help="Skip the anchor point layers.")
    args = parser.parse_args(argv)

    with h5py.File(args.path, "r") as f:
        sample_id, dataset = f.attrs["sample_id"], f.attrs["dataset"]
        spacing = json.loads(f.attrs.get("spacing", "null")) or (1.0, 1.0, 1.0)
        scale = tuple(float(value) for value in spacing)
        raw = f["raw"][:]
        labels = f["labels"][:]
        segmentations = {name: f["segmentation"][name][:] for name in f["segmentation"]} if "segmentation" in f else {}
        prompts = {}
        if "prompts" in f:
            for name in f["prompts"]:
                group = f["prompts"][name]
                prompts[name] = {key: group[key][:] for key in group}
        scores = {key: float(value) for key, value in f.attrs.items() if "/" in key}
        reasons = f.attrs.get("reasons", "")

    print(f"{dataset}  {sample_id}  selected as: {reasons}")
    runs = sorted({key.split("/")[0] for key in scores})
    header = ("run", "mSA", "gt", "predicted", "merged", "misses", "passes", "seconds")
    print("  ".join(f"{h:>12s}" for h in header))
    for run in runs:
        keys = (
            "msa", "gt_objects", "predicted_objects", "merged", "genuine_misses", "propagation_passes", "total_seconds",
        )
        values = [scores.get(f"{run}/{key}", float("nan")) for key in keys]
        counts = "  ".join(f"{int(v):12d}" if v == v else f"{'-':>12s}" for v in values[1:6])
        print(f"{run:>12s}  {values[0]:12.4f}  {counts}  {values[6]:12.1f}")

    viewer = napari.Viewer(title=f"{dataset} {sample_id}")
    viewer.add_image(raw, name="raw", scale=scale, colormap="gray")
    viewer.add_labels(labels, name="ground truth", scale=scale, opacity=0.5)
    for name in sorted(segmentations):
        layer = viewer.add_labels(segmentations[name], name=f"seg {name}", scale=scale, opacity=0.6)
        layer.visible = name == "v4_defaults"
    if not args.no_prompts:
        for name in sorted(prompts):
            arrays = prompts[name]
            anchors = arrays.get("anchors")
            if anchors is None or len(anchors) == 0:
                continue
            scored = set(int(i) for i in arrays.get("scored_prompt_index", []) if i >= 0)
            merged = set(int(i) for i in arrays.get("merged_prompt_index", []) if i >= 0)
            groups = {
                "all": [i for i in range(len(anchors)) if i not in scored],
                "scored": [i for i in scored if i not in merged],
                "merged": sorted(merged),
            }
            for kind, indices in groups.items():
                if not indices:
                    continue
                layer = viewer.add_points(
                    anchors[indices].astype("float32"), name=f"anchors {name} {kind} ({len(indices)})", scale=scale,
                    size=4, face_color=COLORS[kind], border_color="black", out_of_slice_display=False,
                )
                layer.visible = name == "v4_defaults"
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
