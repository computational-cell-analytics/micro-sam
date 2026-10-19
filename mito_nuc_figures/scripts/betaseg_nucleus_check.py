"""Check every slice of the BetaSeg nucleus boxes for unlabeled nuclei.

Only the nucleus of the central target cell is labeled in each volume. A random forest pixel classifier, trained per
volume on the labeled nucleus (positive) and the rest of the target cell (negative), predicts nucleus texture in every
z-slice of the box around the labeled nucleus. Predicted nucleus voxels away from the labeled nucleus are grouped into
3D connected components, and components above a minimum size are reported as candidate unlabeled nuclei, together
with the margin at which they would enter the box.
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
from scipy import ndimage  # noqa
from skimage.feature import multiscale_basic_features  # noqa
from skimage.measure import block_reduce, label, regionprops  # noqa
from skimage.segmentation import find_boundaries  # noqa
from sklearn.ensemble import RandomForestClassifier  # noqa

from betaseg_nucleus_boxes import nucleus_box  # noqa
from figure_utils import DATA_ROOT, get_output_dir, normalize  # noqa


SCALE = 4


def compute_features(image, prob=None):
    feats = multiscale_basic_features(normalize(image.astype("float32")), sigma_min=1, sigma_max=32, num_sigma=6)
    if prob is not None:
        prob_feats = multiscale_basic_features(prob, sigma_min=2, sigma_max=32, num_sigma=5, texture=False)
        feats = np.concatenate([feats, prob[..., None], prob_feats], axis=-1)
    return feats


def downscale_mask(mask):
    return block_reduce(mask, (1, SCALE, SCALE), np.mean) > 0.5


def train_classifier(raw, nucleus, cell, z_ids, prob=None, seed=0):
    rng = np.random.default_rng(seed)
    features, targets = [], []
    for z in z_ids:
        feats = compute_features(raw[z], None if prob is None else prob[z])
        pos = ndimage.binary_erosion(nucleus[z], iterations=2)
        neg = cell[z] & ~ndimage.binary_dilation(nucleus[z], iterations=4)
        for mask, target in ((pos, 1), (neg, 0)):
            idx = np.flatnonzero(mask)
            idx = rng.choice(idx, min(len(idx), 2000), replace=False)
            features.append(feats.reshape(-1, feats.shape[-1])[idx])
            targets.append(np.full(len(idx), target))
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1, random_state=seed)
    rf.fit(np.concatenate(features), np.concatenate(targets))
    return rf


def predict(rf, raw, prev=None):
    prob = np.zeros(raw.shape, dtype="float32")
    for z in range(raw.shape[0]):
        feats = compute_features(raw[z], None if prev is None else prev[z])
        prob[z] = rf.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1].reshape(raw.shape[1:])
    return prob


def margin_to_enter(bbox, nuc_box, scale):
    """The smallest box margin (in full resolution voxels) at which the component enters the box."""
    gaps = []
    for ax, (lo, hi) in enumerate(zip(bbox[:3], bbox[3:])):
        start, stop = nuc_box[ax].start, nuc_box[ax].stop
        f = 1 if ax == 0 else scale
        gaps.append(max(0, start - hi * f, lo * f - stop))
    return int(max(gaps))


def check_volume(path, margin, min_size, out_dir):
    name = os.path.basename(path).replace("_source.h5", "")
    with h5py.File(path, "r") as f:
        nucleus_full = f["labels/nucleus"][:] > 0
        nuc_box = nucleus_box(nucleus_full, 0)
        box = nucleus_box(nucleus_full, margin)
        raw = f["raw"][box]
        cell = f["labels/membrane"][box] > 0
    nucleus = nucleus_full[box]
    raw = block_reduce(raw, (1, SCALE, SCALE), np.mean).astype("float32")
    nucleus, cell = downscale_mask(nucleus), downscale_mask(cell)

    # Train on slices through the nucleus and on a few slices of the target cell above and below it.
    nuc_z = np.where(nucleus.any(axis=(1, 2)))[0]
    z_ids = np.unique(np.concatenate([
        nuc_z[np.linspace(0, len(nuc_z) - 1, 24).astype(int)],
        np.linspace(0, raw.shape[0] - 1, 8).astype(int),
    ]))
    # Two-stage autocontext: the second classifier also sees the smoothed prediction of the first one.
    prob = None
    for _ in range(2):
        rf = train_classifier(raw, nucleus, cell, z_ids, prob)
        prob = predict(rf, raw, prob)
    recall = float((prob[nucleus] > 0.5).mean())
    false_pos = float((prob[cell & ~nucleus] > 0.5).mean())

    # Predicted nucleus voxels that are not adjacent to the labeled nucleus.
    near = ndimage.binary_dilation(nucleus, structure=np.ones((3, 3, 3)), iterations=6)
    candidates = ndimage.binary_opening(ndimage.gaussian_filter(prob, 1.0) > 0.5, iterations=2) & ~near
    components = label(candidates)
    offset = [b.start for b in box]
    nuc_box_local = [slice(n.start - o, n.stop - o) for n, o in zip(nuc_box, offset)]
    found = []
    for prop in regionprops(components):
        size = prop.area * SCALE * SCALE
        if size < min_size:
            continue
        z0, y0, x0, z1, y1, x1 = prop.bbox
        coords = prop.coords
        found.append({
            "size_voxels": int(size),
            "z_range": [z0, z1], "y_range": [y0 * SCALE, y1 * SCALE], "x_range": [x0 * SCALE, x1 * SCALE],
            "outside_cell_fraction": float((~cell[tuple(coords.T)]).mean()),
            "margin_to_enter": margin_to_enter(prop.bbox, nuc_box_local, SCALE),
            "center_slice": int((z0 + z1) // 2),
        })
    found.sort(key=lambda c: c["margin_to_enter"])
    result = {
        "volume": name, "box_margin": margin, "box": [[b.start, b.stop] for b in box],
        "nucleus_recall": recall, "false_positive_rate_in_cell": false_pos, "candidates": found,
    }
    save_candidate_figure(name, raw, prob, nucleus, cell, found, out_dir)
    return result


def save_candidate_figure(name, raw, prob, nucleus, cell, found, out_dir):
    shown = found[:12]
    if not shown:
        return
    fig, axes = plt.subplots(1, len(shown), figsize=(3.6 * len(shown), 3.8), squeeze=False)
    for ax, cand in zip(axes[0], shown):
        z = cand["center_slice"]
        overlay = np.zeros(raw.shape[1:] + (4,), dtype="float32")
        overlay[~cell[z]] = (1.0, 0.2, 0.2, 0.25)
        overlay[prob[z] > 0.5] = (1.0, 0.85, 0.0, 0.45)
        overlay[find_boundaries(nucleus[z], mode="inner")] = (0.2, 0.6, 1.0, 1.0)
        ax.imshow(normalize(raw[z]), cmap="gray")
        ax.imshow(overlay)
        y0, y1 = (v / SCALE for v in cand["y_range"])
        x0, x1 = (v / SCALE for v in cand["x_range"])
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="lime", lw=1.5))
        ax.set_title(f"z={z}, {cand['size_voxels']} vox, enters at margin {cand['margin_to_enter']}", fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"betaseg / {name} - candidate unlabeled nuclei (yellow: predicted nucleus, blue: labeled nucleus, "
        "red: outside the target cell, green: candidate)", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_candidates.png"), dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=int, default=96, help="Margin of the checked box around the nucleus.")
    parser.add_argument("--min_size", type=int, default=50000, help="Minimal candidate size in full-res voxels.")
    parser.add_argument("--volumes", nargs="+", default=None, help="The volumes to check, e.g. low_c2.")
    args = parser.parse_args()
    out_dir = os.path.join(get_output_dir("betaseg"), "nucleus_boxes")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for path in sorted(glob.glob(os.path.join(DATA_ROOT, "betaseg", "data", "*.h5"))):
        name = os.path.basename(path).replace("_source.h5", "")
        if args.volumes is not None and name not in args.volumes:
            continue
        res = check_volume(path, args.margin, args.min_size, out_dir)
        results[name] = res
        print(
            f"{name}: nucleus recall {res['nucleus_recall']:.3f}, "
            f"false positives in cell {res['false_positive_rate_in_cell']:.4f}, {len(res['candidates'])} candidates"
        )
        for cand in res["candidates"]:
            print(cand)
    with open(os.path.join(out_dir, f"candidates_margin{args.margin}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
