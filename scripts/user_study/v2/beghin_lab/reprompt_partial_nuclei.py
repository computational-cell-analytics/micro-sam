"""Segment the partial nuclei of the EM spheroid of the Beghin lab again from prompts.

The partial nuclei are the objects of the class 'partial' of detect_partial_nuclei.py (run it first): nuclei that are
cut or striped along z or too small. Each of them is segmented again with the SAM2 video predictor of the nuclei
model, as in segment_nuclei_from_prompts and join_nucleus_fragments of explore_beghin_em_data.py: in a crop of
CROP_SHAPE of the inverted raw at s1 around the object, prompts in one or several slices are propagated through the
crop in both directions, and the result is made of the connected components that hold a prompt point.

The prompts are derived from the existing object in several ways (STRATEGIES), the embeddings of the crop are shared:
- center: one positive point at the voxel of the object closest to its centroid.
- largest_slice: one positive point in the slice with the largest cross-section of the object, at the pixel of the
  cross-section that is farthest from its border (its interior point).
- three_slices: the interior points of three slices, at 20%, 50% and 80% of the volume of the object along z.
- box: the bounding box of the largest cross-section and its interior point.
- mask: the largest cross-section as a mask prompt (refined on its slice with its box, then propagated).

Each result is a candidate for the nucleus. The voxels of other nuclei are left to them, unless more than
FRAGMENT_FRACTION of the other nucleus lies inside the result: then it is a fragment of the same nucleus, it becomes
part of the candidate and is listed in 'absorbs'. For each candidate the measurements of the detection are computed
again, and how it relates to the object (iou_old, covers_old: the fraction of the object inside the candidate) and to
the candidates of the other strategies (agreement: its mean IoU with them). The candidate with the highest agreement
is the suggested one: a nucleus that comes out the same from different prompts was stable in the earlier tests, a
result that only one strategy gives was not. This is a proposal for the inspection, nothing is merged here.

The outputs are written to PARTIAL_NUCLEI_DIR:
- reprompted.zarr: 'candidates/<id>/<strategy>', the masks of the candidates in the crop of the nucleus (the
  attributes of 'candidates/<id>' hold the position of the crop and the prompts in voxels of s1), and 'suggested',
  a label volume of s1 with the suggested candidates under the ids of their nuclei. Where two suggested candidates
  overlap the one of the lower id keeps the voxels, the overlap is in the table ('overlaps_suggested').
- reprompted_candidates.csv: one row per nucleus and strategy.
- reprompted.csv: one row per nucleus, the object before and the suggested candidate. Objects that are pieces of
  one nucleus get the same candidate, they are listed in 'same_nucleus'.
- reprompted_<page>.png: galleries, one row per nucleus with the object (red) and the suggested candidate (yellow)
  in orthogonal views and the area per slice of the object and of all candidates.
View single nuclei with view_beghin_em_reprompted_nuclei.py.

Example:
    python reprompt_partial_nuclei.py
    python reprompt_partial_nuclei.py --ids 25 100 --strategies center box
    python reprompt_partial_nuclei.py --resume
    python reprompt_partial_nuclei.py --assemble_only

Several processes can share the work, each with --resume --skip_assemble and its own --ids, followed by one run with
--assemble_only.
"""
import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import zarr
from scipy import ndimage
from zarr.codecs import BloscCodec

import detect_partial_nuclei as detection
from detect_partial_nuclei import PARTIAL_NUCLEI_DIR, ZARR_PATH, LEVEL

MODEL_TYPE = "hvit_s_em_nuclei"
# The crop that is segmented, around the center of the bounding box of the object (NUCLEUS_PROMPT_CROP_SHAPE of
# explore_beghin_em_data.py): 19 x 34 x 34 micrometer, the largest nucleus is 13 x 15 x 13 micrometer.
CROP_SHAPE = (192, 512, 512)
# Another nucleus with more than this fraction of its voxels inside a candidate is a fragment of the same nucleus.
FRAGMENT_FRACTION = 0.5
# The positions of the prompted slices of 'three_slices', as fractions of the volume of the object along z.
SLICE_QUANTILES = (0.2, 0.5, 0.8)
STRATEGIES = ("center", "largest_slice", "three_slices", "box", "mask")
CHUNKS = (64, 256, 256)
COLORS = {
    "center": "tab:blue", "largest_slice": "tab:orange", "three_slices": "tab:green", "box": "tab:purple",
    "mask": "tab:brown",
}


def get_reprompted_path():
    return os.path.join(PARTIAL_NUCLEI_DIR, "reprompted.zarr")


def get_table_paths():
    """The paths of the table of the nuclei and of the table of the candidates."""
    return (os.path.join(PARTIAL_NUCLEI_DIR, "reprompted.csv"),
            os.path.join(PARTIAL_NUCLEI_DIR, "reprompted_candidates.csv"))


def get_interior_point(mask):
    """The pixel of a 2d mask that is farthest from its border, (y, x)."""
    distances = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    return tuple(int(i) for i in np.unravel_index(np.argmax(distances), mask.shape))


def get_center_point(mask, voxel_size):
    """The voxel of a mask that is closest to its centroid (in nanometer), (z, y, x)."""
    coords = np.argwhere(mask)
    distances = (((coords - coords.mean(axis=0)) * np.array(voxel_size)) ** 2).sum(axis=1)
    return tuple(int(i) for i in coords[np.argmin(distances)])


def derive_prompts(mask, strategy, voxel_size):
    """Derive the prompts of a strategy from the mask of an object, see the description of the script.

    Args:
        mask: The mask of the object in the crop, (Z, Y, X).
        strategy: One of STRATEGIES.
        voxel_size: The voxel size in nanometer.

    Returns:
        A list with one prompt per conditioned slice: a dictionary with 'frame', and with 'points' ((N, 2) in y, x)
        and 'labels', optionally with 'box' (y0, x0, y1, x1), or with 'mask' (2d).
        The points (z, y, x) that mark the object: the result is made of the components that hold one of them.
    """
    areas = mask.sum(axis=(1, 2))
    z_max = int(np.argmax(areas))
    interior = get_interior_point(mask[z_max])
    if strategy == "center":
        z, y, x = get_center_point(mask, voxel_size)
        return [{"frame": z, "points": [(y, x)], "labels": [1]}], [(z, y, x)]
    if strategy == "largest_slice":
        return [{"frame": z_max, "points": [interior], "labels": [1]}], [(z_max,) + interior]
    if strategy == "three_slices":
        cumulative = np.cumsum(areas) / areas.sum()
        frames = sorted({int(np.searchsorted(cumulative, quantile)) for quantile in SLICE_QUANTILES})
        points = [(z,) + get_interior_point(mask[z]) for z in frames]
        return [{"frame": z, "points": [(y, x)], "labels": [1]} for z, y, x in points], points
    if strategy == "box":
        ys, xs = np.nonzero(mask[z_max])
        box = [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1]
        return [{"frame": z_max, "points": [interior], "labels": [1], "box": box}], [(z_max,) + interior]
    if strategy == "mask":
        return [{"frame": z_max, "mask": mask[z_max]}], [(z_max,) + interior]
    raise ValueError(f"Unknown strategy {strategy}, choose from {STRATEGIES}.")


def segment_from_prompts(predictor, volume, embeddings, prompts):
    """Segment one object with the SAM2 video predictor from the prompts of derive_prompts.

    Returns:
        The mask of the propagation through the whole volume, in both directions.
    """
    from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D

    segmenter = PromptableSegmentation3D(predictor, volume, embeddings)
    for prompt in prompts:
        if "mask" in prompt:
            segmenter.add_mask_prompts(frame_ids=int(prompt["frame"]), masks=[prompt["mask"]], object_id=1)
        else:
            box = prompt.get("box")
            segmenter.add_prompt_set(
                frame_id=int(prompt["frame"]), points=np.array(prompt["points"], dtype="float32"),
                point_labels=np.array(prompt["labels"], dtype="int32"),
                box=None if box is None else np.array(box, dtype="float32"), object_id=1,
            )
    video_segments = segmenter.propagate_prompts()
    segmenter.reset_predictor()
    mask = np.zeros(volume.shape, dtype=bool)
    for frame, per_object in video_segments.items():
        mask[frame] = np.asarray(per_object[1]).squeeze()[:volume.shape[1], :volume.shape[2]]
    return mask


def get_components(mask, points):
    """The connected components of a mask that hold one of the points (empty if no point is in the mask)."""
    labels, _ = ndimage.label(mask)
    label_ids = {int(labels[tuple(point)]) for point in points} - {0}
    return np.isin(labels, list(label_ids)) if label_ids else np.zeros_like(mask)


def iou(mask_a, mask_b):
    union = int((mask_a | mask_b).sum())
    return float((mask_a & mask_b).sum() / union) if union > 0 else 0.0


def get_candidate(mask, labels, nucleus_id, volumes):
    """Make the candidate of a nucleus from a prompted mask: without the voxels of the other nuclei it only touches.

    Args:
        mask: The prompted mask in the crop.
        labels: The nuclei in the crop.
        nucleus_id: The id of the nucleus that was prompted.
        volumes: The number of voxels of each nucleus in the whole volume (indexed by id).

    Returns:
        The candidate.
        The ids of the other nuclei with more than FRAGMENT_FRACTION of their voxels in the mask (part of it).
        The fraction of the mask in other nuclei that are not part of it (these voxels are removed).
    """
    ids, counts = np.unique(labels[mask], return_counts=True)
    absorbs = [
        int(i) for i, count in zip(ids, counts)
        if i not in (0, nucleus_id) and count > FRAGMENT_FRACTION * volumes.loc[i]
    ]
    other = mask & (labels > 0) & ~np.isin(labels, [nucleus_id] + absorbs)
    in_other = float(other.sum() / mask.sum()) if mask.any() else 0.0
    return mask & ~other, absorbs, in_other


def get_crop(row, shape, crop_shape=CROP_SHAPE):
    """The crop around the center of the bounding box of an object, shifted to stay inside the volume."""
    center = [(int(row[f"{axis}0"]) + int(row[f"{axis}1"])) // 2 for axis in "zyx"]
    starts = [int(min(max(c - cs // 2, 0), sh - cs)) for c, cs, sh in zip(center, crop_shape, shape)]
    return tuple(slice(start, start + cs) for start, cs in zip(starts, crop_shape))


def touches_crop_border(mask, bb, shape):
    """Whether a mask touches a face of its crop that is not a face of the volume."""
    for axis, (b, sh) in enumerate(zip(bb, shape)):
        planes = np.moveaxis(mask, axis, 0)
        if (b.start > 0 and planes[0].any()) or (b.stop < sh and planes[-1].any()):
            return True
    return False


def reprompt_nucleus(predictor, raw, nuclei, nucleus_id, row, volumes, strategies, voxel_size):
    """Segment one nucleus again with all strategies.

    Returns:
        The crop (slices in the volume).
        The candidates: strategy -> mask in the crop.
        The records of the candidates: strategy -> measurements.
    """
    from micro_sam.v2.util import precompute_image_embeddings

    shape = raw.shape
    bb = get_crop(row, shape)
    offset = [b.start for b in bb]
    labels = nuclei[bb]
    old = labels == nucleus_id
    assert old.sum() == row["volume"], f"The nucleus {nucleus_id} does not fit into the crop."

    volume = 255 - raw[bb]
    embeddings = precompute_image_embeddings(predictor, volume, ndim=3, verbose=False)
    candidates, records = {}, {}
    with embeddings:
        for strategy in strategies:
            prompts, points = derive_prompts(old, strategy, voxel_size)
            mask = get_components(segment_from_prompts(predictor, volume, embeddings, prompts), points)
            candidate, absorbs, in_other = get_candidate(mask, labels, nucleus_id, volumes)
            record = {
                "points": [[int(p + o) for p, o in zip(point, offset)] for point in points],
                "box": [int(b + o) for b, o in zip(prompts[0]["box"], offset[1:] * 2)] if "box" in prompts[0] else None,
                "empty": not bool(candidate.any()), "absorbs": absorbs, "fraction_in_other": in_other,
                "at_crop_border": touches_crop_border(candidate, bb, shape),
                "iou_old": iou(candidate, old), "covers_old": float((candidate & old).sum() / old.sum()),
            }
            if candidate.any():
                record.update(detection.get_object_features(candidate, offset, shape, voxel_size))
            candidates[strategy], records[strategy] = candidate, record
    for strategy in strategies:
        others = [iou(candidates[strategy], candidates[other]) for other in strategies if other != strategy]
        records[strategy]["agreement"] = float(np.mean(others)) if others else np.nan
    return bb, candidates, records


def get_suggested(records):
    """The strategy whose candidate agrees best with the candidates of the other strategies (None if all are empty)."""
    valid = [strategy for strategy, record in records.items() if not record["empty"]]
    if not valid:
        return None
    return max(valid, key=lambda strategy: np.nan_to_num(records[strategy]["agreement"]))


def to_json(value):
    """Make the measurements of a candidate serializable: python types, None for nan."""
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value


def write_candidates(out, nucleus_id, bb, candidates, records, suggested):
    """Write the candidates of one nucleus with their measurements ('records') into 'candidates/<id>'."""
    group = out.require_group("candidates").create_group(str(nucleus_id), overwrite=True)
    for strategy, candidate in candidates.items():
        group.create_array(
            strategy, data=candidate, chunks=CHUNKS, compressors=BloscCodec(cname="zstd", clevel=5),
            dimension_names=["z", "y", "x"],
        )
    group.attrs.update(to_json({
        "offset": [b.start for b in bb], "shape": [b.stop - b.start for b in bb], "suggested": suggested,
        "prompts": {strategy: {"points": record["points"], "box": record["box"]}
                    for strategy, record in records.items()},
        "records": records,
    }))


def write_suggested(out, nucleus_id, shape):
    """Add the suggested candidate of one nucleus to the label volume 'suggested'.

    Returns:
        The other nuclei that are already in 'suggested' where the candidate is: id -> fraction of the candidate.
        These voxels are not overwritten.
    """
    group = out["candidates"][str(nucleus_id)]
    if group.attrs["suggested"] is None:
        return {}
    bb = tuple(slice(o, o + sh) for o, sh in zip(group.attrs["offset"], group.attrs["shape"]))
    candidate = group[group.attrs["suggested"]][:]
    current = out["suggested"][bb]
    ids, counts = np.unique(current[candidate], return_counts=True)
    overlaps = {int(i): float(count / candidate.sum()) for i, count in zip(ids, counts) if i not in (0, nucleus_id)}
    current[candidate & (current == 0)] = nucleus_id
    out["suggested"][bb] = current
    return overlaps


def get_candidate_rows(nucleus_id, records):
    """The rows of the table of the candidates for one nucleus."""
    columns = ("volume_um3", "n_slices", "max_change", "n_jumps", "area_variation", "z0", "z1")
    return [{
        "id": nucleus_id, "strategy": strategy, "empty": record["empty"],
        **{column: record.get(column, np.nan) for column in columns},
        "iou_old": record["iou_old"], "covers_old": record["covers_old"], "agreement": record["agreement"],
        "fraction_in_other": record["fraction_in_other"], "absorbs": " ".join(str(i) for i in record["absorbs"]),
        "at_crop_border": record["at_crop_border"], "points": json.dumps(record["points"]),
        "box": json.dumps(record["box"]),
    } for strategy, record in records.items()]


def get_nucleus_row(nucleus_id, row, records, suggested, overlaps, reference, thresholds):
    """The row of the table of the nuclei: the object before and its suggested candidate."""
    volumes = [record["volume_um3"] for record in records.values() if not record["empty"]]
    result = {
        "id": nucleus_id, "reason": row["reason"], "volume_um3": row["volume_um3"], "n_slices": row["n_slices"],
        "max_change": row["max_change"], "suggested": suggested, "n_empty": len(records) - len(volumes),
        "volume_um3_min": min(volumes, default=np.nan), "volume_um3_max": max(volumes, default=np.nan),
    }
    if suggested is None:
        return result
    record = records[suggested]
    features = pd.DataFrame([{key: record[key] for key in ("volume_um3", "max_change", "z_extent_nm")}])
    result.update({
        "new_volume_um3": record["volume_um3"], "volume_factor": record["volume_um3"] / row["volume_um3"],
        "new_n_slices": record["n_slices"], "new_max_change": record["max_change"],
        "new_class": detection.classify(features, reference, **thresholds)["class"].iloc[0],
        "agreement": record["agreement"], "iou_old": record["iou_old"], "covers_old": record["covers_old"],
        "fraction_in_other": record["fraction_in_other"], "absorbs": " ".join(str(i) for i in record["absorbs"]),
        "at_crop_border": record["at_crop_border"],
        "overlaps_suggested": " ".join(f"{i}:{fraction:.2f}" for i, fraction in overlaps.items()),
    })
    return result


def reprompt_partial_nuclei(nuclei="joined", ids=None, strategies=STRATEGIES, include_seed_only=False, resume=False):
    """Segment the partial nuclei of the detection table again and write their candidates.

    Several processes can run this at the same time with resume=True and different ids (start them one after the
    other, so that the output exists). Run assemble_reprompted when all nuclei are done.

    Args:
        nuclei: The version of the nuclei the detection was run on.
        ids: The ids of the nuclei to segment. By default all of the class 'partial'.
        strategies: The strategies for the prompts.
        include_seed_only: Whether to segment the nuclei that are only seeds for their cells as well. They are
            rounded cells without a clear nucleus, the prompts were not stable for them.
        resume: Whether to keep the nuclei that are already in the output and segment only the missing ones.
    """
    from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter

    table = pd.read_csv(detection.get_detection_path(nuclei), index_col="id")
    if ids is None:
        selected = table[(table["class"] == "partial") & (include_seed_only | ~table["seed_only"])]
        ids = sorted(selected.index.tolist())
    group = zarr.open_group(detection.get_nuclei_path(nuclei), mode="r")
    labels = group["segmentation"]
    voxel_size = tuple(group.attrs.get("voxel_size", detection.VOXEL_SIZE))
    raw = zarr.open_array(os.path.join(ZARR_PATH, f"s{LEVEL}"), mode="r")
    assert raw.shape == labels.shape, (raw.shape, labels.shape)

    out = zarr.open_group(get_reprompted_path(), mode="a" if resume else "w", zarr_format=3)
    out.attrs.update({
        "nuclei": detection.get_nuclei_path(nuclei), "level": LEVEL, "voxel_size": list(voxel_size),
        "model_type": MODEL_TYPE, "inverted": True, "crop_shape": list(CROP_SHAPE),
    })
    done = out.require_group("candidates")
    ids = [i for i in ids if not (str(i) in done and "records" in done[str(i)].attrs)]

    print(f"Segmenting {len(ids)} nuclei again with the strategies {list(strategies)}", flush=True)
    predictor, _ = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="apg", ndim=3)
    for nucleus_id in ids:
        t0 = time.time()
        row = table.loc[nucleus_id]
        bb, candidates, records = reprompt_nucleus(
            predictor, raw, labels, nucleus_id, row, table["volume"], strategies, voxel_size
        )
        suggested = get_suggested(records)
        write_candidates(out, nucleus_id, bb, candidates, records, suggested)
        sizes = {strategy: round(record.get("volume_um3", 0.0)) for strategy, record in records.items()}
        new = records.get(suggested)
        print(
            f"Nucleus {nucleus_id}: {row['volume_um3']:.0f} um3, change {row['max_change']:.2f} -> {sizes} um3, "
            f"suggested {suggested}"
            + (f" (change {new['max_change']:.2f}, agreement {new['agreement']:.2f}, covers "
               f"{new['covers_old']:.2f} of the object, absorbs {new['absorbs']})" if suggested else "")
            + f", in {time.time() - t0:.0f} s", flush=True,
        )


def get_same_nucleus(links):
    """Group the objects whose suggested candidates are the same nucleus.

    Args:
        links: Per nucleus that was segmented again the ids of the objects that are the same nucleus: the objects
            its suggested candidate absorbs and the earlier suggested candidates that hold most of it.

    Returns:
        Per nucleus that was segmented again the other ids of its group (sorted).
    """
    parents = {}

    def find(i):
        parents.setdefault(i, i)
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for nucleus_id, linked in links.items():
        for other in linked:
            parents[find(other)] = find(nucleus_id)
    members = {}
    for i in list(parents):
        members.setdefault(find(i), []).append(i)
    return {i: sorted(set(members.get(find(i), [])) - {i}) for i in links}


def assemble_reprompted(nuclei="joined", **thresholds):
    """Write the tables and the label volume 'suggested' from the candidates of all nuclei in the output.

    The suggested candidates are written in the order of the ids of their nuclei. Two objects are the same nucleus
    (column 'same_nucleus' of the table of the nuclei) if the suggested candidate of one absorbs the other or if
    more than FRAGMENT_FRACTION of it is in the suggested candidate of the other.

    Args:
        nuclei: The version of the nuclei the detection was run on.
        thresholds: The thresholds of the detection (see detect_partial_nuclei.classify), for the class of the
            suggested candidates.
    """
    table = pd.read_csv(detection.get_detection_path(nuclei), index_col="id")
    reference = detection.get_reference(table, thresholds["max_slice_change"], thresholds["fragment_min_z_extent"])
    shape = zarr.open_group(detection.get_nuclei_path(nuclei), mode="r")["segmentation"].shape
    out = zarr.open_group(get_reprompted_path(), mode="a", zarr_format=3)
    out.create_array(
        "suggested", shape=shape, dtype="uint32", chunks=CHUNKS, fill_value=0, overwrite=True,
        compressors=BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"), dimension_names=["z", "y", "x"],
    )
    groups = {int(name): group for name, group in out["candidates"].groups() if "records" in group.attrs}
    nucleus_rows, candidate_rows, links = [], [], {}
    for nucleus_id in sorted(groups):
        attrs = groups[nucleus_id].attrs
        records = {
            strategy: {key: np.nan if value is None and key != "box" else value for key, value in record.items()}
            for strategy, record in attrs["records"].items()
        }
        overlaps = write_suggested(out, nucleus_id, shape)
        candidate_rows.extend(get_candidate_rows(nucleus_id, records))
        nucleus_rows.append(get_nucleus_row(
            nucleus_id, table.loc[nucleus_id], records, attrs["suggested"], overlaps, reference, thresholds
        ))
        absorbs = records[attrs["suggested"]]["absorbs"] if attrs["suggested"] is not None else []
        links[nucleus_id] = set(absorbs) | {i for i, fraction in overlaps.items() if fraction > FRAGMENT_FRACTION}
    same_nucleus = get_same_nucleus(links)
    for row in nucleus_rows:
        row["same_nucleus"] = " ".join(str(i) for i in same_nucleus[row["id"]])
    table_path, candidate_path = get_table_paths()
    pd.DataFrame(nucleus_rows).to_csv(table_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
    print(f"{len(nucleus_rows)} nuclei\nTables: {table_path}, {candidate_path}\nCandidates: {get_reprompted_path()}",
          flush=True)


def plot_reprompted_gallery(labels, raw, out, table, detected, label_ids, plot_path, voxel_size):
    """Plot one row per nucleus: orthogonal views with the object and the suggested candidate and the area profiles.

    The views are the slice with the largest cross-section of the suggested candidate (XY) and the planes through
    the center of this cross-section (XZ, YZ) of the inverted raw, with the other nuclei in cyan, the object in red
    and the suggested candidate in yellow. The profile shows the number of voxels per slice of the object (black)
    and of the candidates of all strategies, the suggested one with a thick line.
    """
    import matplotlib.pyplot as plt

    per_row = 2
    n_rows = int(np.ceil(len(label_ids) / per_row))
    fig, axes = plt.subplots(n_rows, 4 * per_row, figsize=(4 * per_row * 2.2, n_rows * 2.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for index, label_id in enumerate(label_ids):
        row = table.loc[label_id]
        panel = axes[index // per_row, 4 * (index % per_row): 4 * (index % per_row + 1)]
        group = out["candidates"][str(label_id)]
        offset, suggested = group.attrs["offset"], group.attrs["suggested"]
        candidates = {name: array[:] for name, array in group.arrays()}
        crop = tuple(slice(o, o + sh) for o, sh in zip(offset, group.attrs["shape"]))
        old = labels[crop] == label_id
        shown = candidates[suggested] if suggested is not None else old
        # The panel is centered on the shown mask and has the size of the galleries of the detection.
        center = [int(c) for c in np.argwhere(shown).mean(axis=0)]
        local = tuple(
            slice(max(c - h, 0), min(c + h, sh)) for c, h, sh in zip(center, detection.GALLERY_HALF_SHAPE, old.shape)
        )
        bb = tuple(slice(lo.start + o, lo.stop + o) for lo, o in zip(local, offset))
        z = int(np.argmax(shown[local].sum(axis=(1, 2))))
        y, x = np.argwhere(shown[local][z]).mean(axis=0).astype(int)
        other = np.where(old[local], 0, labels[bb])
        layers = [(other, (0, 1, 1)), (old[local], (1, 0.2, 0.2))]
        if suggested is not None:
            layers.append((candidates[suggested][local], (1, 1, 0)))
        detection.plot_views(panel, raw[bb], [(z, layers), (y, layers), (x, layers)], voxel_size[0] / voxel_size[1])
        title = f"{label_id}: {row['volume_um3']:.0f} um3, change {row['max_change']:.2f}"
        if suggested is not None:
            title += (f" -> {suggested}: {row['new_volume_um3']:.0f} um3, change {row['new_max_change']:.2f}, "
                      f"agreement {row['agreement']:.2f}")
        panel[0].set_title(title, fontsize=6, loc="left")
        panel[3].axis("on")
        z_axis = np.arange(crop[0].start, crop[0].stop)
        for name, candidate in candidates.items():
            panel[3].plot(z_axis, candidate.sum(axis=(1, 2)), lw=2 if name == suggested else 0.7,
                          color=COLORS.get(name, "tab:gray"), label=name)
        panel[3].plot(z_axis, old.sum(axis=(1, 2)), lw=1, color="k", label="object")
        covered = np.nonzero(np.any([c.any(axis=(1, 2)) for c in candidates.values()] + [old.any(axis=(1, 2))], 0))[0]
        panel[3].set_xlim(z_axis[covered.min()] - 3, z_axis[covered.max()] + 3)
        panel[3].tick_params(labelsize=5)
        if index == 0:
            panel[3].legend(fontsize=4, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)


def plot_reprompted(nuclei="joined", num_workers=1, page_size=detection.GALLERY_PAGE_SIZE):
    """Plot the galleries of all nuclei of the table, sorted by the gain of volume of the suggested candidate."""
    table = pd.read_csv(get_table_paths()[0], index_col="id")
    detected = pd.read_csv(detection.get_detection_path(nuclei), index_col="id")
    group = zarr.open_group(detection.get_nuclei_path(nuclei), mode="r")
    labels = detection.load_labels(group["segmentation"], num_workers)
    voxel_size = tuple(group.attrs.get("voxel_size", detection.VOXEL_SIZE))
    raw = zarr.open_array(os.path.join(ZARR_PATH, f"s{LEVEL}"), mode="r")
    out = zarr.open_group(get_reprompted_path(), mode="r")
    label_ids = table.sort_values("volume_factor", ascending=False).index.tolist()
    paths = []
    for page, start in enumerate(range(0, len(label_ids), page_size)):
        paths.append(os.path.join(PARTIAL_NUCLEI_DIR, f"reprompted_{page:02d}.png"))
        plot_reprompted_gallery(
            labels, raw, out, table, detected, label_ids[start:start + page_size], paths[-1], voxel_size
        )
    print(f"Galleries: {paths}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nuclei", default="joined", choices=("watershed", "merged", "joined"))
    parser.add_argument("--ids", type=int, nargs="+", help="The nuclei to segment, by default the partial ones.")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--include_seed_only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep the nuclei that are already in the output.")
    parser.add_argument("--skip_assemble", action="store_true", help="Only segment (for several processes).")
    parser.add_argument("--assemble_only", action="store_true", help="Only write the tables and the galleries.")
    parser.add_argument("--max_slice_change", type=float, default=detection.MAX_SLICE_CHANGE)
    parser.add_argument("--size_n_deviations", type=float, default=detection.SIZE_N_DEVIATIONS)
    parser.add_argument("--fragment_max_volume", type=float, default=detection.FRAGMENT_MAX_VOLUME)
    parser.add_argument("--fragment_min_z_extent", type=float, default=detection.FRAGMENT_MIN_Z_EXTENT)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    if not args.assemble_only:
        reprompt_partial_nuclei(
            nuclei=args.nuclei, ids=args.ids, strategies=tuple(args.strategies),
            include_seed_only=args.include_seed_only, resume=args.resume,
        )
    if not args.skip_assemble:
        assemble_reprompted(
            nuclei=args.nuclei, max_slice_change=args.max_slice_change, size_n_deviations=args.size_n_deviations,
            fragment_max_volume=args.fragment_max_volume, fragment_min_z_extent=args.fragment_min_z_extent,
        )
        plot_reprompted(args.nuclei, args.num_workers)


if __name__ == "__main__":
    main()
