"""Automatic instance segmentation with micro-sam2 and 'hvit_l_cells' on the beghin data, scored on the ground truth.

The data is the benchmark of the Digitalized Organoid paper in 'nus_organoids': four uint16 volumes (ZYX) with an
instance ground truth each, from a 15 slice monolayer to a 100 x 1080 x 1080 organoid.

Usage:
    python explore_beghin_data.py  # AIS on every volume
    python explore_beghin_data.py --best -o beghin_best_segmentations  # AIS with the best settings of each volume
    python explore_beghin_data.py -e ais apg -n ZeroG_breast_cancer_spheroid  # AIS and APG on one volume
    python explore_beghin_data.py --view -n TM00099_cell  # the raw data, ground truth and segmentations in napari
    python explore_beghin_data.py --spheroids  # AIS on the DAPI channel of the MCF7 spheroid stacks, no ground truth
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
SPHEROID_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data"
SPHEROID_SETS = ("39_Spheroids_MCF7_GCp24_DAPI_Ki67_B4", "3_Spheroids_MCF7_GCp24_DAPI_eCadh_btub")
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
# tiling instead of the default. 'engine' overrides the engine of the command line for '--best'.
DEFAULT_SETTINGS = {
    "scale": (1, 1, 1), "anisotropy": 1, "min_size": None, "norm_lower": None, "norm_upper": None,
    "norm_percentiles": None, "background_sigma": None, "tile_shape": None, "halo": None,
}
BEST_SETTINGS = {
    # The nuclei are segmented at 2x, every one is found already at 1x (mSA 0.648 -> 0.704).
    "HCT116_cells_monolayer": {"scale": (1, 2, 2), "min_size": 500},
    # 0.390 -> 0.493.
    "Human_colon_organoid": {"scale": (2, 2, 2), "anisotropy": 4, "min_size": 500},
    # Removes the bright speckles in the frame around the spheroid, 254 of the 362 objects (0.129 -> 0.393).
    "ZeroG_breast_cancer_spheroid": {"min_size": 2000},
    # The nuclei take 0.2% of the volume and sit in the dim body of cell clusters, which the model segments as one
    # object. Subtracting the background removes the cluster bodies, and the 99th - 99.99th percentile keeps only the
    # brightest 1% of the voxels visible. On a crop of 77 nuclei: default AIS mSA 0.038, this mSA 0.210 (F1 0.730).
    # The tiles of 256 + 2 * 52 = 360 pixels reach the encoder at the scale of that untiled crop.
    "TM00099_cell": {
        "engine": "apg", "background_sigma": 8, "norm_percentiles": (99, 99.99),
        "tile_shape": (4, 256, 256), "halo": (2, 52, 52),
    },
}


def load_data(name):
    raw = tifffile.imread(os.path.join(DATA_ROOT, "Data", f"{name}.tif"))
    labels = tifffile.imread(os.path.join(DATA_ROOT, "Groundtruth", f"{name}_GT.tif"))
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


def segment(raw, engine, settings=DEFAULT_SETTINGS):
    settings = {**DEFAULT_SETTINGS, **settings}
    settings.pop("engine", None)
    scale = tuple(settings["scale"])
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine, ndim=3)
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
    if engine == "ais":
        if settings["min_size"] is not None:
            generate_kwargs["min_size"] = int(settings["min_size"] * np.prod(scale))
        spacing_z = settings["anisotropy"] / scale[0]
        if spacing_z > 1:
            generate_kwargs["spacing"] = (spacing_z, 1.0, 1.0)

    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
    try:
        # 'batch_size=None' picks the batch size per device.
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=volume, embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
            ndim=3, batch_size=None, verbose=False, norm_bounds=norm_bounds, tile_shape=settings["tile_shape"],
            halo=settings["halo"], **generate_kwargs,
        )
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)
    if segmentation.shape != raw.shape:
        segmentation = resize(
            segmentation, raw.shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype("uint32")
    return segmentation, time.time() - start


def run(names, engines, best, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for name in names:
        raw, labels = load_data(name)
        settings = BEST_SETTINGS[name] if best else DEFAULT_SETTINGS
        output_path = os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5")
        for engine in ([settings.get("engine", engines[0])] if best else engines):
            segmentation, runtime = segment(raw, engine, settings)
            scores = evaluate(segmentation, labels)
            results.append({"name": name, "engine": engine, "time [s]": runtime, **scores})
            print(f"{name} {engine.upper()}: " + ", ".join(
                f"{key} {value:.3f}" if isinstance(value, float) else f"{key} {value}" for key, value in scores.items()
            ) + f", {runtime:.1f} s", flush=True)

            with h5py.File(output_path, "a") as f:
                for key, data in (("raw", raw), ("gt", labels)):
                    if key not in f:
                        f.create_dataset(key, data=data, compression="gzip")
                if engine in f:
                    del f[engine]
                f.create_dataset(engine, data=segmentation, compression="gzip")
                f[engine].attrs.update({key: value for key, value in scores.items()})
                f[engine].attrs["settings"] = str({**DEFAULT_SETTINGS, **settings})

    print("\n" + "\t".join(results[0].keys()))
    for result in results:
        print("\t".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in result.values()))


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


def run_spheroids(output_folder):
    from glob import glob

    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais", ndim=3)
    for spheroid_set in SPHEROID_SETS:
        paths = sorted(glob(os.path.join(SPHEROID_ROOT, spheroid_set, "*.tif")))
        set_folder = os.path.join(output_folder, spheroid_set)
        os.makedirs(set_folder, exist_ok=True)
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            raw = tifffile.imread(path)[:, DAPI_CHANNEL]
            start = time.time()
            embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
            try:
                segmentation = automatic_instance_segmentation(
                    predictor, segmenter, input_path=raw, ndim=3, batch_size=None, verbose=False,
                    embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
                )
            finally:
                shutil.rmtree(embedding_dir, ignore_errors=True)
            cleaned = clean_spheroid_segmentation(segmentation, raw)
            runtime = time.time() - start
            print(f"{spheroid_set} {name} {raw.shape}: AIS {len(np.unique(segmentation)) - 1} objects, cleaned "
                  f"{len(np.unique(cleaned)) - 1} objects, {runtime:.1f} s", flush=True)
            with h5py.File(os.path.join(set_folder, f"{name}_{MODEL_TYPE}.h5"), "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("ais_clean", data=cleaned, compression="gzip")


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
    parser.add_argument("-e", "--engines", nargs="+", choices=ENGINES, default=["ais"])
    parser.add_argument("--best", action="store_true", help="Use the best AIS settings of each volume.")
    parser.add_argument("-o", "--output_folder", default=OUTPUT_ROOT, help="The folder for the h5 files.")
    parser.add_argument("--spheroids", action="store_true", help="Segment the DAPI channel of the spheroid stacks.")
    parser.add_argument("--view", action="store_true", help="Show the first volume and its segmentations in napari.")
    args = parser.parse_args()

    if args.spheroids:
        run_spheroids(os.path.join(args.output_folder, "spheroids"))
    elif args.view:
        view(args.names[0], args.output_folder)
    else:
        run(args.names, args.engines, args.best, args.output_folder)


if __name__ == "__main__":
    main()
