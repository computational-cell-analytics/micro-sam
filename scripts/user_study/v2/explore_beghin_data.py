"""Automatic instance segmentation with micro-sam2 and 'hvit_l_cells' on the beghin data, scored on the ground truth.

The data is the benchmark of the Digitalized Organoid paper in 'nus_organoids': four uint16 volumes (ZYX) with an
instance ground truth each, from a 15 slice monolayer to a 100 x 1080 x 1080 organoid. The MCF7 spheroid stacks in
'mcf7_spheroids' have no ground truth. The layout of the data folder is in its 'README.md'.

Usage:
    python explore_beghin_data.py  # AIS on every volume
    python explore_beghin_data.py --best -o beghin_best_segmentations  # AIS with the best settings of each volume
    python explore_beghin_data.py -e ais apg -n ZeroG_breast_cancer_spheroid  # AIS and APG on one volume
    python explore_beghin_data.py --clean -n HCT116_cells_monolayer  # the spheroid cleanup, stored as 'ais_clean'
    python explore_beghin_data.py --view -n TM00099_cell  # the raw data, ground truth and segmentations in napari
    python explore_beghin_data.py --spheroids  # AIS on DAPI of the MCF7 spheroids, as '<stack>_nucleus_prediction.tif'
    python explore_beghin_data.py --membrane  # AIS on the cells of the composites, '<stack>_membrane_prediction.tif'
    python explore_beghin_data.py --cellpose -o beghin_best_segmentations  # cellposeSAM on every volume
    python explore_beghin_data.py --export  # the segmentations as '<name>_prediction[_<method>].tif' in the data folder
    python explore_beghin_data.py --sweep -n HCT116_cells_monolayer --scale 1 2 2  # the APG variants at this scale
"""

import os
import time
import shutil
import argparse
import tempfile

import h5py
import tifffile
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from skimage.segmentation import relabel_sequential

from elf.evaluation import matching, mean_segmentation_accuracy
from bioimage_py.evaluation import symmetric_best_dice_score

from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/nus_organoids"
OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data"
MODEL_TYPE = "hvit_l_cells"
ENGINES = ("ais", "apg")
# The embeddings are written to the tmp folder of the job and removed after each segmentation.
EMBEDDING_ROOT = os.environ.get("TMPDIR", tempfile.gettempdir())
# The MCF7 spheroid stacks (Z, C, Y, X), z 1.5 um, xy 0.414 um. Channel 0 is DAPI in both sets.
# The sets were '39_Spheroids_MCF7_GCp24_DAPI_Ki67_B4' and '3_Spheroids_MCF7_GCp24_DAPI_eCadh_btub'.
SPHEROID_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/mcf7_spheroids"
SPHEROID_SETS = ("dapi_ki67", "dapi_ecadh_btub")
DAPI_CHANNEL = 0
# The AIS post-processing splits small fragments off the nuclei, and the spheroids sit in a frame of bright speckles
# like ZeroG. A minimum size of 2000 removed both, but also small nuclei. Instead, objects below LARGE_SIZE (nuclei
# are >= 3500 voxels in 90%) are merged into the nucleus that encloses them, removed when they are far from every
# large object (the frame), smaller than SMALL_MIN_SIZE or grainy like the speckles (small nuclei are smooth).
LARGE_SIZE = 2000
FRAGMENT_MIN_ENCLOSED = 0.6
NEAR_XY, NEAR_Z = 15, 5
SMALL_MIN_SIZE = 300
SMALL_MAX_CV = 0.45
# The cells of the 'dapi_ecadh_btub' stacks are segmented on (beta-tubulin, DAPI, E-cadherin) as RGB, the order of the
# channels in the files is DAPI, E-cadherin, beta-tubulin by the folder name. beta-tubulin sets the borders between
# the cells, DAPI keeps the cells whole (beta-tubulin alone cuts 10 - 15% of the nuclei) and E-cadherin closes the
# outer rim of the spheroid. On the 3 stacks this covered the most of the spheroid (0.68 - 0.83, beta-tubulin alone
# 0.62 - 0.77) and kept 94 - 99% of the nuclei inside one cell (85 - 90%).
MEMBRANE_SET = "dapi_ecadh_btub"
MEMBRANE_CHANNELS = (2, 0, 1)
# The cell segmentation holds small grit inside the cells: 35 - 45% of the objects are below 200 voxels, the cells
# above 2000 and nearly nothing in between. Objects below GRIT_MAX_SIZE are merged into the neighbor that covers at
# least GRIT_MIN_ENCLOSED of their border (the cell around them), else removed.
GRIT_MAX_SIZE = 1000
GRIT_MIN_ENCLOSED = 0.5
# The slices below and above the spheroid hold two kinds of noise that survive the minimum size: flat blobs no brighter
# than the background ring around them (contrast 0.98 - 1.0, the dim nuclei on top of the spheroid 1.15 and more),
# and a layer of bright speckles below it, grainy (intensity CV >= 0.5, nuclei <= 0.39 in 90%) and thin (<= 9
# slices, nuclei 10 - 18).
NOISE_MIN_CONTRAST = 1.1
SPECKLE_MIN_CV = 0.5
SPECKLE_MAX_Z_EXTENT = 9
NOISE_RING_WIDTH = 3
NAMES = ("HCT116_cells_monolayer", "Human_colon_organoid", "TM00099_cell", "ZeroG_breast_cancer_spheroid")
# The AIS settings that scored best on the ground truth of each volume. 'scale' resamples the input (z, y, x) and the
# segmentation back, 'anisotropy' is the z spacing over the xy spacing of the original grid for the flow
# post-processing, 'min_size' is in voxels of the original grid and 'norm_lower' / 'norm_upper' replace the 2nd / 98th
# percentile bounds of the normalization, or 'norm_percentiles' sets other percentiles. 'background_sigma' subtracts the
# background, a gaussian of this sigma in xy, before the normalization. 'tile_shape' and 'halo' (z, y, x) set the
# tiling instead of the default. 'clean' replaces the minimum size by 'clean_spheroid_segmentation'. 'engine' overrides
# the engine of the command line for '--best'. 'apg' holds more keyword arguments for the APG 'generate'.
DEFAULT_SETTINGS = {
    "scale": (1, 1, 1), "anisotropy": 1, "min_size": None, "norm_lower": None, "norm_upper": None,
    "norm_percentiles": None, "background_sigma": None, "tile_shape": None, "halo": None, "clean": False, "apg": None,
}
BEST_SETTINGS = {
    # The nuclei are segmented at 3x in yx, every one is found already at 1x (mSA 0.648 -> 0.704 at 2x -> 0.750).
    "HCT116_cells_monolayer": {"scale": (1, 3, 3), "min_size": 500},
    # 0.390 -> 0.493.
    "Human_colon_organoid": {"scale": (2, 2, 2), "anisotropy": 4, "min_size": 500},
    # Removes the bright speckles in the frame around the spheroid. A min_size of 2000 removed them, 254 of the 362
    # objects (mSA 0.129 -> 0.389, F1 0.843), the spheroid cleanup also keeps small nuclei (mSA 0.417, F1 0.886).
    "ZeroG_breast_cancer_spheroid": {"clean": True},
    # The nuclei take 0.2% of the volume and sit in the dim body of cell clusters, which the model segments as one
    # object. Subtracting the background removes the cluster bodies, and the 99th - 99.99th percentile keeps only the
    # brightest 1% of the voxels visible. On a crop of 77 nuclei: default AIS mSA 0.038, this mSA 0.210 (F1 0.730).
    # The tiles of 256 + 2 * 52 = 360 pixels reach the encoder at the scale of that untiled crop.
    "TM00099_cell": {
        "background_sigma": 8, "norm_percentiles": (99, 99.99),
        "tile_shape": (4, 256, 256), "halo": (2, 52, 52),
    },
}


def load_data(name):
    raw = tifffile.imread(os.path.join(DATA_ROOT, f"{name}.tif"))
    labels = tifffile.imread(os.path.join(DATA_ROOT, f"{name}_groundtruth.tif"))
    assert raw.shape == labels.shape, f"{name}: raw {raw.shape} and ground truth {labels.shape} differ."
    return raw, labels


def evaluate(segmentation, labels):
    msa, accuracies = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
    # Precision: the share of objects that match a ground-truth object with IoU >= 0.5, recall: the share of the
    # ground-truth objects that are matched.
    scores = matching(segmentation, labels, threshold=0.5)
    return {
        "objects": len(np.unique(segmentation)) - 1, "gt_objects": len(np.unique(labels)) - 1,
        "mSA": msa, "SA50": accuracies[0], "SA75": accuracies[5],
        "precision": scores["precision"], "recall": scores["recall"], "F1": scores["f1"],
        "SBD": symmetric_best_dice_score(segmentation, labels),
    }


def segment(raw, engine, settings=DEFAULT_SETTINGS, models=None, embedding_path=None):
    """Segment a volume. 'models' reuses a (predictor, segmenter) pair of this engine, 'embedding_path' caches the
    embeddings there and keeps them, by default they are written to a tmp folder and removed."""
    settings = {**DEFAULT_SETTINGS, **settings}
    settings.pop("engine", None)
    scale = tuple(settings["scale"])
    if models is None:
        models = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine, ndim=3)
    predictor, segmenter = models
    start = time.time()

    if settings["background_sigma"] is not None:
        sigma = settings["background_sigma"]
        raw = raw.astype("float32")
        raw = np.clip(raw - ndimage.gaussian_filter(raw, (0, sigma, sigma)), 0, None)
    volume = raw if scale == (1, 1, 1) else ndimage.zoom(raw.astype("float32"), scale, order=1)
    norm_bounds = None
    if settings["norm_percentiles"] is not None:
        lower, upper = np.percentile(raw, settings["norm_percentiles"])
        norm_bounds = (np.array([lower], dtype="float32"), np.array([upper], dtype="float32"))
    elif settings["norm_lower"] is not None or settings["norm_upper"] is not None:
        lower = np.percentile(raw, 2) if settings["norm_lower"] is None else settings["norm_lower"]
        upper = np.percentile(raw, 98) if settings["norm_upper"] is None else settings["norm_upper"]
        norm_bounds = (np.array([lower], dtype="float32"), np.array([upper], dtype="float32"))
    generate_kwargs = {}
    if settings["min_size"] is not None:
        generate_kwargs["min_size"] = int(settings["min_size"] * np.prod(scale))
    spacing_z = settings["anisotropy"] / scale[0]
    if spacing_z > 1:
        generate_kwargs["spacing"] = (spacing_z, 1.0, 1.0)
    if engine == "apg" and settings["apg"] is not None:
        generate_kwargs.update(settings["apg"])

    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT) if embedding_path is None else None
    try:
        # 'batch_size=None' picks the batch size per device.
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=volume, ndim=3, batch_size=None, verbose=False,
            embedding_path=os.path.join(embedding_dir, "embeddings.zarr") if embedding_path is None else embedding_path,
            norm_bounds=norm_bounds, tile_shape=settings["tile_shape"], halo=settings["halo"], **generate_kwargs,
        )
    finally:
        if embedding_dir is not None:
            shutil.rmtree(embedding_dir, ignore_errors=True)
    if segmentation.shape != raw.shape:
        segmentation = resize(
            segmentation, raw.shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype("uint32")
    return segmentation, time.time() - start


def run(names, engines, best, output_folder, clean=False, scale=None):
    """Segment and score the volumes. 'engines' overrides the engine of the best settings, 'clean' applies
    'clean_spheroid_segmentation' like the 'clean' setting, but stores the result as '<engine>_clean'."""
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for name in names:
        raw, labels = load_data(name)
        settings = {**DEFAULT_SETTINGS, **(BEST_SETTINGS[name] if best else {})}
        if scale is not None:
            settings["scale"] = tuple(scale)
        if clean or settings["clean"]:
            settings.update({"min_size": None, "clean": True})
        output_path = os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5")
        for engine in (engines or [settings.get("engine", "ais")]):
            segmentation, runtime = segment(raw, engine, settings)
            if settings["clean"]:
                start = time.time()
                segmentation = clean_spheroid_segmentation(segmentation, raw)
                runtime += time.time() - start
            dataset = f"{engine}_clean" if clean else engine
            scores = evaluate(segmentation, labels)
            results.append({"name": name, "engine": dataset, "time [s]": runtime, **scores})
            print(f"{name} {dataset.upper()}: " + ", ".join(
                f"{key} {value:.3f}" if isinstance(value, float) else f"{key} {value}" for key, value in scores.items()
            ) + f", {runtime:.1f} s", flush=True)

            with h5py.File(output_path, "a") as f:
                for key, data in (("raw", raw), ("gt", labels)):
                    if key not in f:
                        f.create_dataset(key, data=data, compression="gzip")
                if dataset in f:
                    del f[dataset]
                f.create_dataset(dataset, data=segmentation, compression="gzip")
                f[dataset].attrs.update(scores)
                f[dataset].attrs["settings"] = str({**settings, "engine": engine})

    print("\n" + "\t".join(results[0].keys()))
    for result in results:
        print("\t".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in result.values()))


# The APG variants of '--sweep', each one change of the 'generate' defaults (see 'DEFAULT_PROMPT_GENERATION_3D' in
# 'micro_sam.v2.automatic_prompt_generation'), on top of the best settings of the volume.
APG_VARIANTS = {
    "default": {},
    "refine": {"refinement": "points+boxes"},
    "score0.5": {"score_threshold": 0.5},
    "score0.7": {"score_threshold": 0.7},
    "score0.8": {"score_threshold": 0.8},
    "score0.9": {"score_threshold": 0.9},
    "overlap0.3": {"max_overlap": 0.3},
    "overlap0.7": {"max_overlap": 0.7},
    "sigma1": {"sigma": 1.0},
    "ladder_low": {"candidate_threshold": (1.0, 5.0)},
    "ladder_high": {"candidate_threshold": (3.0, 10.0)},
    "fg0.5": {"foreground_threshold": 0.5},
    "flow_long": {"n_iter": 100, "dt": 1.0},
    "sigma0.5": {"sigma": 0.5},
    "sigma2": {"sigma": 2.0},
    "fg0.3": {"foreground_threshold": 0.3},
    # 'min_size' and 'spacing' are on the scaled grid here and replace the ones of the best settings.
    "min100": {"min_size": 100},
    "min1000": {"min_size": 1000},
    "spacing1": {"spacing": None},
    "spacing2": {"spacing": (2.0, 1.0, 1.0)},
    "spacing4": {"spacing": (4.0, 1.0, 1.0)},
    "spacing6": {"spacing": (6.0, 1.0, 1.0)},
    "nomulti": {"multimasking": False},
    "maxsize1.5": {"max_size_factor": 1.5},
    "maxsize2": {"max_size_factor": 2.0},
    "maxsize3": {"max_size_factor": 3.0},
}


def parse_variant(variant):
    """A '+'-joined combination of 'APG_VARIANTS' keys, e.g. 'refine+score0.7', applied from left to right."""
    kwargs = {}
    for key in variant.split("+"):
        kwargs.update(APG_VARIANTS[key])
    return kwargs


def sweep(name, scale, variants, output_folder, tag=None):
    """Run the APG variants on one volume at one scale, from the same embeddings. Each result is scored and saved
    to 'apg_sweep/<name>_s<z><y><x>[_<tag>].h5' with its scores as attributes, and a variant already in it is skipped.
    Parallel jobs need their own 'tag'."""
    raw, labels = load_data(name)
    settings = {**DEFAULT_SETTINGS, **BEST_SETTINGS[name], "scale": tuple(scale)}
    os.makedirs(os.path.join(output_folder, "apg_sweep"), exist_ok=True)
    file_name = f"{name}_s{''.join(str(x) for x in scale)}" + ("" if tag is None else f"_{tag}")
    output_path = os.path.join(output_folder, "apg_sweep", f"{file_name}.h5")
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            variants = [variant for variant in variants if variant not in f]
    if not variants:
        return
    models = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="apg", ndim=3)
    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
    try:
        for variant in variants:
            variant_settings = {**settings, "apg": parse_variant(variant)}
            segmentation, runtime = segment(
                raw, "apg", variant_settings, models=models,
                embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
            )
            # With the cleanup, the result before it is stored too, as '<variant>_noclean'.
            results = [(variant, segmentation)]
            if settings["clean"]:
                cleaned = clean_spheroid_segmentation(segmentation, raw)
                results = [(variant, cleaned), (f"{variant}_noclean", segmentation)]
            for key, result in results:
                scores = evaluate(result, labels)
                print(f"{name} scale {scale} APG {key}: " + ", ".join(
                    f"{metric} {value:.3f}" if isinstance(value, float) else f"{metric} {value}"
                    for metric, value in scores.items()
                ) + f", {runtime:.1f} s", flush=True)
                with h5py.File(output_path, "a") as f:
                    f.create_dataset(key, data=result, compression="gzip")
                    f[key].attrs.update({**scores, "time": runtime, "settings": str(variant_settings)})
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)


def remove_noise_objects(segmentation, raw):
    """Remove the flat noise blobs and the speckle layer objects, see 'NOISE_MIN_CONTRAST', in place."""
    raw = raw.astype("float32")
    bboxes = ndimage.find_objects(segmentation)
    removed = []
    for object_id, bbox in enumerate(bboxes, 1):
        if bbox is None:
            continue
        bbox = tuple(slice(max(axis.start - NOISE_RING_WIDTH, 0), axis.stop + NOISE_RING_WIDTH) for axis in bbox)
        labels = segmentation[bbox]
        mask = labels == object_id
        ring = ndimage.binary_dilation(mask, iterations=NOISE_RING_WIDTH) & (labels == 0)
        values = raw[bbox][mask]
        contrast = values.mean() / raw[bbox][ring].mean() if ring.any() else np.inf
        z_extent = int(mask.any(axis=(1, 2)).sum())
        is_speckle = values.std() / values.mean() >= SPECKLE_MIN_CV and z_extent <= SPECKLE_MAX_Z_EXTENT
        if contrast < NOISE_MIN_CONTRAST or is_speckle:
            removed.append(object_id)
    if removed:
        lut = np.arange(len(bboxes) + 1, dtype=segmentation.dtype)
        lut[removed] = 0
        segmentation = lut[segmentation]
    return segmentation


def clean_spheroid_segmentation(segmentation, raw):
    """Merge the fragments into their nuclei and remove the frame, small and grainy objects and the end noise."""
    raw = raw.astype("float32")
    bboxes = ndimage.find_objects(segmentation)
    lut = np.arange(len(bboxes) + 1, dtype=segmentation.dtype)
    for object_id, bbox in enumerate(bboxes, 1):
        if bbox is None or (segmentation[bbox] == object_id).sum() >= LARGE_SIZE:
            continue
        padded = tuple(slice(max(axis.start - 1, 0), axis.stop + 1) for axis in bbox)
        labels = segmentation[padded]
        mask = labels == object_id
        border = labels[ndimage.binary_dilation(mask) & ~mask]
        neighbors, counts = np.unique(border[border != 0], return_counts=True)
        if len(neighbors) and counts.max() / len(border) >= FRAGMENT_MIN_ENCLOSED:
            lut[object_id] = neighbors[counts.argmax()]
    segmentation = lut[segmentation]

    ids, counts = np.unique(segmentation, return_counts=True)
    near = np.isin(segmentation, ids[(counts >= LARGE_SIZE) & (ids != 0)])
    near = ndimage.binary_dilation(near, structure=np.ones((2 * NEAR_Z + 1, 1, 1), bool))
    near = ndimage.binary_dilation(near, structure=np.ones((1, 3, 3), bool), iterations=NEAR_XY)
    bboxes = ndimage.find_objects(segmentation)
    lut = np.arange(len(bboxes) + 1, dtype=segmentation.dtype)
    for object_id, bbox in enumerate(bboxes, 1):
        if bbox is None:
            continue
        mask = segmentation[bbox] == object_id
        size = int(mask.sum())
        if size >= LARGE_SIZE:
            continue
        values = raw[bbox][mask]
        if size < SMALL_MIN_SIZE or near[bbox][mask].mean() < 0.5 or values.std() / values.mean() >= SMALL_MAX_CV:
            lut[object_id] = 0
    return remove_noise_objects(lut[segmentation], raw)


def clean_grit(segmentation):
    """Merge the objects below GRIT_MAX_SIZE into the neighbor that encloses them, or remove them."""
    segmentation = segmentation.copy()
    object_ids, sizes = np.unique(segmentation[segmentation > 0], return_counts=True)
    bboxes = ndimage.find_objects(segmentation)
    for object_id in object_ids[sizes < GRIT_MAX_SIZE]:
        bbox = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in bboxes[object_id - 1])
        crop = segmentation[bbox]
        mask = crop == object_id
        if not mask.any():
            continue
        border = crop[ndimage.binary_dilation(mask) & ~mask]
        neighbors, counts = np.unique(border, return_counts=True)
        best = np.argmax(np.where(neighbors > 0, counts, 0))
        enclosed = neighbors[best] > 0 and counts[best] >= GRIT_MIN_ENCLOSED * border.size
        crop[mask] = neighbors[best] if enclosed else 0
    return relabel_sequential(segmentation)[0]


def segment_spheroid(predictor, segmenter, raw):
    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
    try:
        return automatic_instance_segmentation(
            predictor, segmenter, input_path=raw, ndim=3, batch_size=None, verbose=False,
            embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
        )
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)


def spheroid_paths(spheroid_set):
    from glob import glob

    paths = sorted(glob(os.path.join(SPHEROID_ROOT, spheroid_set, "*.tif")))
    return [path for path in paths if "_prediction" not in os.path.basename(path)]


def run_spheroids():
    """Segment the nuclei on the DAPI channel of every spheroid stack, as '<stack>_nucleus_prediction.tif'."""
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais", ndim=3)
    for spheroid_set in SPHEROID_SETS:
        for path in spheroid_paths(spheroid_set):
            name = os.path.splitext(os.path.basename(path))[0]
            raw = tifffile.imread(path)[:, DAPI_CHANNEL]
            start = time.time()
            segmentation = segment_spheroid(predictor, segmenter, raw)
            cleaned = clean_spheroid_segmentation(segmentation, raw)
            runtime = time.time() - start
            print(f"{spheroid_set} {name} {raw.shape}: AIS {len(np.unique(segmentation)) - 1} objects, cleaned "
                  f"{len(np.unique(cleaned)) - 1} objects, {runtime:.1f} s", flush=True)
            write_prediction(os.path.join(SPHEROID_ROOT, spheroid_set, f"{name}_nucleus_prediction.tif"), cleaned)


def run_spheroid_membranes():
    """Segment the cells of the composite stacks on MEMBRANE_CHANNELS, as '<stack>_membrane_prediction.tif'."""
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais", ndim=3)
    for path in spheroid_paths(MEMBRANE_SET):
        name = os.path.splitext(os.path.basename(path))[0]
        raw = np.moveaxis(tifffile.imread(path)[:, list(MEMBRANE_CHANNELS)], 1, -1)
        start = time.time()
        segmentation = segment_spheroid(predictor, segmenter, raw)
        cleaned = clean_grit(segmentation)
        runtime = time.time() - start
        print(f"{MEMBRANE_SET} {name} {raw.shape}: AIS {len(np.unique(segmentation)) - 1} objects, cleaned "
              f"{len(np.unique(cleaned)) - 1} objects, {runtime:.1f} s", flush=True)
        write_prediction(os.path.join(SPHEROID_ROOT, MEMBRANE_SET, f"{name}_membrane_prediction.tif"), cleaned)


# The cellposeSAM modes: '3d' predicts the flows on the yx, zx and zy planes and follows their average in 3D,
# '2d_stitch' segments every z plane in 2D and stitches the masks of neighboring planes with IoU >=
# CELLPOSE_STITCH_THRESHOLD.
CELLPOSE_MODES = ("3d", "2d_stitch")
CELLPOSE_STITCH_THRESHOLD = 0.5


def run_cellpose(names, output_folder):
    """Segment and score the volumes with cellposeSAM in both modes, with its default settings. The results are
    stored as 'cellposeSAM_<mode>' next to the micro-sam2 segmentations."""
    from cellpose import models

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for name in names:
        raw, labels = load_data(name)
        for mode in CELLPOSE_MODES:
            start = time.time()
            segmentation = model.eval(
                raw, z_axis=0, do_3D=mode == "3d",
                stitch_threshold=CELLPOSE_STITCH_THRESHOLD if mode == "2d_stitch" else 0.0,
            )[0].astype("uint32")
            runtime = time.time() - start
            scores = evaluate(segmentation, labels)
            results.append({"name": name, "mode": mode, "time [s]": runtime, **scores})
            print(f"{name} cellposeSAM {mode}: " + ", ".join(
                f"{key} {value:.3f}" if isinstance(value, float) else f"{key} {value}" for key, value in scores.items()
            ) + f", {runtime:.1f} s", flush=True)

            dataset = f"cellposeSAM_{mode}"
            with h5py.File(os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5"), "a") as f:
                for key, data in (("raw", raw), ("gt", labels)):
                    if key not in f:
                        f.create_dataset(key, data=data, compression="gzip")
                if dataset in f:
                    del f[dataset]
                f.create_dataset(dataset, data=segmentation, compression="gzip")
                f[dataset].attrs.update(scores)
                f[dataset].attrs["time"] = runtime

    print("\n" + "\t".join(results[0].keys()))
    for result in results:
        print("\t".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in result.values()))


def write_prediction(path, segmentation):
    """Write uint16 instance labels like the Fiji files of the collaborators: deflate in strips, and read it back."""
    assert segmentation.max() <= np.iinfo("uint16").max, f"{path}: {segmentation.max()} labels do not fit uint16."
    tifffile.imwrite(path, segmentation.astype("uint16"), compression="zlib")
    assert np.array_equal(tifffile.imread(path), segmentation), f"{path} differs after reading it back."
    os.chmod(path, 0o664)
    print(f"Saved {path}: {len(np.unique(segmentation)) - 1} objects")


def export_predictions(names, output_folder):
    """Export the final micro-sam2 segmentation (the engine of the best settings), both cellposeSAM modes and the
    better cellposeSAM mode by mSA as '_cellposeSAM'."""
    for name in names:
        engine = BEST_SETTINGS[name].get("engine", "ais")
        with h5py.File(os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5"), "r") as f:
            assert f[engine].shape == tifffile.imread(os.path.join(DATA_ROOT, f"{name}.tif")).shape
            modes = [f"cellposeSAM_{mode}" for mode in CELLPOSE_MODES]
            best = max(modes, key=lambda key: f[key].attrs["mSA"])
            for key, suffix in [(engine, ""), (best, "_cellposeSAM")] + [(key, f"_{key}") for key in modes]:
                write_prediction(os.path.join(DATA_ROOT, f"{name}_prediction{suffix}.tif"), f[key][:])


def view(name, output_folder):
    import napari

    viewer = napari.Viewer(title=name)
    with h5py.File(os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5"), "r") as f:
        viewer.add_image(f["raw"][:], name="raw")
        for key in f:
            if key != "raw":
                viewer.add_labels(f[key][:], name=key)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--names", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument(
        "-e", "--engines", nargs="+", choices=ENGINES, default=None,
        help="The engines, by default AIS, or the engine of the best settings with '--best'.",
    )
    parser.add_argument("--best", action="store_true", help="Use the best AIS settings of each volume.")
    parser.add_argument(
        "--clean", action="store_true", help="Clean up with 'clean_spheroid_segmentation' instead of the min_size.",
    )
    parser.add_argument("-o", "--output_folder", default=OUTPUT_ROOT, help="The folder for the h5 files.")
    parser.add_argument("--spheroids", action="store_true", help="Segment the DAPI channel of the spheroid stacks.")
    parser.add_argument("--membrane", action="store_true", help="Segment the cells of the composite spheroid stacks.")
    parser.add_argument("--cellpose", action="store_true", help="Segment the volumes with cellposeSAM in both modes.")
    parser.add_argument("--view", action="store_true", help="Show the first volume and its segmentations in napari.")
    parser.add_argument("--export", action="store_true", help="Export the segmentations as tifs to the data folder.")
    parser.add_argument("--sweep", action="store_true", help="Run the APG variants on the first volume.")
    parser.add_argument("--scale", nargs=3, type=int, default=None, help="The scale (z, y, x), overrides the settings.")
    parser.add_argument(
        "--variants", nargs="+", default=list(APG_VARIANTS), help="The variants of '--sweep', see 'parse_variant'."
    )
    parser.add_argument("--tag", default=None, help="Appended to the file name of '--sweep', for parallel jobs.")
    args = parser.parse_args()

    if args.sweep:
        scale = args.scale or BEST_SETTINGS[args.names[0]].get("scale", (1, 1, 1))
        sweep(args.names[0], scale, args.variants, args.output_folder, args.tag)
    elif args.export:
        export_predictions(args.names, args.output_folder)
    elif args.cellpose:
        run_cellpose(args.names, args.output_folder)
    elif args.spheroids:
        run_spheroids()
    elif args.membrane:
        run_spheroid_membranes()
    elif args.view:
        view(args.names[0], args.output_folder)
    else:
        run(args.names, args.engines, args.best, args.output_folder, args.clean, args.scale)


if __name__ == "__main__":
    main()
