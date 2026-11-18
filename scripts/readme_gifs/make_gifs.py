"""Record the demo GIFs of the README with the napari annotation tools.

Usage: python make_gifs.py 2d 3d tracking objects pixels

Every GIF runs in its own process, so pass one name per call to start each tool with a fresh state.
It needs a display (the viewer window is shown to render the canvas) and a GPU.
The pixel classification GIF uses UNI2-h, which needs your own HuggingFace access, and the BCSS data in BCSS_ROOT.
"""
import os
import sys

import micro_sam.util as util  # noqa, import before napari to avoid the triton / Mesa LLVM crash

import numpy as np
import imageio.v3 as imageio
from skimage.measure import regionprops

import napari

from micro_sam.sam_annotator._state import AnnotatorState

from recorder import Recorder

DATA_CACHE = os.path.join(util.get_cache_directory(), "sample_data")
# The GIFs of the README.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "doc", "gifs")
EMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
CELL_MODEL = "hvit_l_cells"
EM_MODEL = "hvit_s_em_mito"
# UNI2-h (MahmoodLab) for histopathology. Its weights are gated: it needs your own HuggingFace access.
HISTO_MODEL = "vit_univ2"


def auto_segment(image, model_type, name, ndim=2):
    cache = os.path.join(EMB_DIR, f"{name}-seg.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, ndim=ndim)
    seg = automatic_instance_segmentation(predictor, segmenter, image, ndim=ndim, verbose=False)
    np.save(cache, seg)
    return seg


def pick_objects(seg, n, min_area, margin=8):
    """Pick the n largest objects that keep a margin to the border. Pass n=None to pick all."""
    h, w = seg.shape
    props = [
        p for p in regionprops(seg) if p.area >= min_area and
        p.bbox[0] > margin and p.bbox[1] > margin and p.bbox[2] < h - margin and p.bbox[3] < w - margin
    ]
    props.sort(key=lambda p: p.area, reverse=True)
    return props if n is None else props[:n]


def gif_2d_boxes():
    from micro_sam.sample_data import fetch_hela_2d_example_data
    from micro_sam.sam_annotator import annotator

    image = imageio.imread(fetch_hela_2d_example_data(DATA_CACHE))
    seg = auto_segment(image, CELL_MODEL, "hela2d")
    # Box all cells, also the cells at the image border.
    objects = pick_objects(seg, n=None, min_area=300, margin=-1)
    # Order the boxes from top to bottom, like a user goes through the image.
    objects.sort(key=lambda p: (p.centroid[0] // 100, p.centroid[1]))

    viewer = napari.Viewer(show=False)
    annotator(
        image, model_type=CELL_MODEL, embedding_path=os.path.join(EMB_DIR, "hela2d.zarr"),
        return_viewer=True, viewer=viewer,
    )
    interactive = AnnotatorState().widgets["interactive"]
    interactive.batched = True
    rec = Recorder(viewer)
    rec.fit(image.shape)

    prompts = viewer.layers["prompts"]
    rec.snap(600, cursor=(image.shape[0] * 0.8, image.shape[1] * 0.1))
    for p in objects:
        y0, x0, y1, x1 = p.bbox
        h, w = image.shape[:2]
        y0, x0, y1, x1 = max(y0 - 4, 0), max(x0 - 4, 0), min(y1 + 4, h - 1), min(x1 + 4, w - 1)
        rec.move((y0, x0), steps=3)
        drawing = None
        for t in np.linspace(0.25, 1, 4):
            ye, xe = y0 + t * (y1 - y0), x0 + t * (x1 - x0)
            rect = np.array([[y0, x0], [y0, xe], [ye, xe], [ye, x0]])
            if drawing is None:
                prompts.add_rectangles(rect)
                drawing = len(prompts.data) - 1
            else:
                data = list(prompts.data)
                data[drawing] = rect
                prompts.data = data
            rec.snap(40, cursor=(ye, xe))
    rec.hold(300)
    interactive.segment(viewer)
    rec.snap(2500, cursor=None)
    rec.save(os.path.join(OUT_DIR, "interactive_2d.gif"))
    viewer.close()


def set_points(layer, coords, labels, **extra):
    layer.data = np.array(coords, dtype="float32")
    properties = {"label": np.array(labels)}
    properties.update({k: np.array(v) for k, v in extra.items()})
    layer.properties = properties
    layer.refresh_colors()


def lineage_targets(mother, daughters, division_frame):
    """Get prompt points for one lineage from the CTC ground-truth markers of DIC-HeLa 01.

    Returns the point (y, x) of the mother on the first frame, of the mother on its division frame,
    and of each daughter on the frame after the division.
    """
    import io
    import zipfile
    zip_path = os.path.join(DATA_CACHE, "DIC-C2DH-HeLa.zip")
    with zipfile.ZipFile(zip_path) as f:
        def markers(t):
            return imageio.imread(io.BytesIO(f.read(f"DIC-C2DH-HeLa/01_GT/TRA/man_track{t:03d}.tif")))
        first = inner_point(markers(0) == mother)[0]
        last = inner_point(markers(division_frame) == mother)[0]
        after = markers(division_frame + 1)
        children = [inner_point(after == d)[0] for d in daughters]
    return first, last, children


def gif_tracking():
    from elf.io import open_file
    from micro_sam.sam_annotator import annotator_tracking
    from micro_sam.sample_data import fetch_tracking_example_data

    with open_file(fetch_tracking_example_data(DATA_CACHE), mode="r") as f:
        timeseries = f["*.tif"][:]
    # Lineage 1 of the CTC ground truth: the mother divides after frame 29 and both daughters live until the last
    # frame.
    division_frame = 29
    first, last, children = lineage_targets(mother=1, daughters=(3, 4), division_frame=division_frame)

    viewer = napari.Viewer(show=False)
    annotator_tracking(
        timeseries, model_type=CELL_MODEL, embedding_path=os.path.join(EMB_DIR, "hela-track.zarr"),
        return_viewer=True, viewer=viewer,
    )
    state = AnnotatorState()
    interactive = state.widgets["interactive"]
    interactive.apply_to_volume_checkbox.setChecked(True)
    rec = Recorder(viewer)
    rec.fit(timeseries.shape)
    viewer.dims.current_step = (0, 0, 0)

    points = viewer.layers["point_prompts"]
    h, w = timeseries.shape[1:]
    coords, labels, states, track_ids = [], [], [], []

    def add_point(t, yx, point_state, track_id):
        coords.append((t, *yx))
        labels.append("positive")
        states.append(point_state)
        track_ids.append(str(track_id))
        set_points(points, coords, labels, state=states, track_id=track_ids)

    def scrub(t0, t1, step=3, duration=150):
        for t in list(range(t0, t1, step)) + [t1]:
            viewer.dims.current_step = (t, 0, 0)
            rec.snap(duration)

    # 1. Track the mother from a point on the first frame.
    rec.snap(600, cursor=(h * 0.85, w * 0.15))
    rec.move(first, steps=5)
    add_point(0, first, "track", 1)
    rec.snap(300)
    interactive.segment(viewer)
    rec.snap(500)

    # 2. Go to the frame where the mother divides and mark the division.
    scrub(0, division_frame)
    rec.move(last, steps=4)
    add_point(division_frame, last, "division", 1)
    rec.snap(300)
    interactive.segment(viewer)  # Ends the mother at the division and adds the track ids of the daughters.
    rec.snap(500)

    # 3. Track each daughter from a point on the frame after the division.
    viewer.dims.current_step = (division_frame + 1, 0, 0)
    rec.snap(300)
    for track_id, target in zip(state.lineage[1], children):
        state.current_track_id = track_id
        rec.move(target, steps=4)
        add_point(division_frame + 1, target, "track", track_id)
        rec.snap(300)
        interactive.segment(viewer)
        rec.snap(400)
    print("Lineage", state.lineage)
    state.widgets["commit"](viewer)
    rec.snap(400, cursor=None)

    # 4. Play the movie (every 3rd frame, which keeps the GIF small) to show the lineage.
    scrub(0, timeseries.shape[0] - 1, duration=180)
    rec.hold(1500)
    rec.save(os.path.join(OUT_DIR, "interactive_tracking.gif"), out_size=360)
    viewer.close()


def inner_point(mask):
    """The point inside the mask with the largest distance to its border."""
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask)
    return np.unravel_index(np.argmax(dist), mask.shape), dist.max()


def best_window(seg, size, min_area, margin=12, stride=32):
    """Find the square window that fully contains the most objects."""
    props = [p for p in regionprops(seg) if p.area >= min_area]
    h, w = seg.shape
    best, best_objects = None, []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            inside = [
                p for p in props if p.bbox[0] >= y + margin and p.bbox[1] >= x + margin and
                p.bbox[2] <= y + size - margin and p.bbox[3] <= x + size - margin
            ]
            if len(inside) > len(best_objects):
                best, best_objects = (y, x), inside
    return best, best_objects


def gif_3d_em():
    from micro_sam.sample_data import fetch_3d_example_data
    from micro_sam.sam_annotator import annotator

    raw = imageio.imread(fetch_3d_example_data(DATA_CACHE))
    zc = raw.shape[0] // 2
    seg = auto_segment(raw[zc], EM_MODEL, "lucchi-mid")

    size, dz = 512, 20
    (y0, x0), objects = best_window(seg, size, min_area=800)
    crop = raw[zc - dz:zc + dz + 1, y0:y0 + size, x0:x0 + size]
    seg_crop = seg[y0:y0 + size, x0:x0 + size]
    print("Segmenting", len(objects), "mitochondria in the window", (y0, x0))
    targets = [inner_point(seg_crop == p.label)[0] for p in objects]
    targets.sort(key=lambda yx: (yx[0] // 80, yx[1]))

    viewer = napari.Viewer(show=False)
    annotator(
        crop, ndim=3, model_type=EM_MODEL, embedding_path=os.path.join(EMB_DIR, f"lucchi-crop-{y0}-{x0}.zarr"),
        return_viewer=True, viewer=viewer,
    )
    interactive = AnnotatorState().widgets["interactive"]
    # Batched mode segments one object per positive point.
    interactive.batched_checkbox.setChecked(True)
    interactive.apply_to_volume_checkbox.setChecked(True)
    rec = Recorder(viewer)
    rec.fit(crop.shape)
    zm = dz
    viewer.dims.current_step = (zm, 0, 0)

    points = viewer.layers["point_prompts"]
    rec.snap(600, cursor=(size * 0.85, size * 0.15))
    for i in range(len(targets)):
        rec.move(targets[i], steps=4)
        set_points(points, [(zm, *t) for t in targets[:i + 1]], ["positive"] * (i + 1))
        rec.snap(200)
    rec.hold(200)
    interactive.segment(viewer)
    n_objects = len(np.unique(viewer.layers["current_object"].data)) - 1
    print("The batched segmentation returned", n_objects, "objects")
    rec.snap(900, cursor=None)
    # Scroll through the volume to show the objects in all slices.
    steps = list(range(zm, crop.shape[0])) + list(range(crop.shape[0] - 1, -1, -1)) + list(range(0, zm + 1))
    # Show every 3rd slice, which keeps the GIF small.
    for z in steps[::3]:
        viewer.dims.current_step = (z, 0, 0)
        rec.snap(165)
    rec.hold(1500)
    rec.save(os.path.join(OUT_DIR, "interactive_3d.gif"), out_size=360)
    viewer.close()


def paint_disc(layer, center, value, radius=8):
    from skimage.draw import disk
    data = layer.data
    rr, cc = disk(center, radius, shape=data.shape)
    data[rr, cc] = value
    layer.data = data


def gif_object_classifier():
    from micro_sam.sample_data import fetch_livecell_example_data
    from micro_sam.sam_annotator.object_classifier import object_classifier

    image = imageio.imread(fetch_livecell_example_data(DATA_CACHE))
    h, w = image.shape[:2]
    y0, x0 = (h - 512) // 2, (w - 512) // 2
    image = image[y0:y0 + 512, x0:x0 + 512]
    seg = auto_segment(image, CELL_MODEL, "livecell-crop")

    # The task: round cells are class 1 and elongated cells are class 2, split at the median eccentricity.
    # Cells close to the median are ambiguous, so only the clear cases are corrected.
    cells = pick_objects(seg, n=None, min_area=250, margin=4)
    cells.sort(key=lambda p: p.eccentricity)
    median = np.median([p.eccentricity for p in cells])
    reference = {p.label: 1 if p.eccentricity < median else 2 for p in cells}
    clear = {p.label for p in cells if abs(p.eccentricity - median) > 0.08}
    # Start with three clicks per class, on the clearest examples.
    examples = [(p, 1) for p in cells[:3]] + [(p, 2) for p in cells[-3:]]

    viewer = napari.Viewer(show=False)
    object_classifier(
        image, seg, model_type=CELL_MODEL, embedding_path=os.path.join(EMB_DIR, "livecell-crop.zarr"),
        return_viewer=True, viewer=viewer,
    )
    classifier = AnnotatorState().annotator
    # The SAM embeddings do not describe the shape, so add the classical shape features (as in the GUI).
    classifier._set_options(
        False, None, False,
        classical_features_val=["eccentricity", "solidity", "axis_major_length", "axis_minor_length"],
    )
    rec = Recorder(viewer)
    rec.fit(image.shape)

    def click(p, value):
        pos = inner_point(seg == p.label)[0]
        rec.move(pos, steps=5)
        paint_disc(annotations, pos, value)
        rec.snap(250)

    def mistakes():
        prediction = viewer.layers["prediction"].data
        wrong = [p for p in cells if p.label in clear and np.median(prediction[seg == p.label]) != reference[p.label]]
        return sorted(wrong, key=lambda p: abs(p.eccentricity - median), reverse=True)

    annotations = viewer.layers["annotations"]
    rec.snap(600, cursor=(400, 80))
    for p, value in examples:
        click(p, value)
    rec.hold(200)
    classifier._run_train_and_predict()
    viewer.layers["segmentation"].visible = False
    rec.snap(2000)

    # Correct the clearest mistakes and train again, up to two times.
    for _ in range(2):
        wrong = mistakes()
        print("Clearly wrong cells:", len(wrong), "of", len(clear))
        if not wrong:
            break
        for p in wrong[:3]:
            click(p, reference[p.label])
        rec.hold(200)
        classifier._run_train_and_predict()
        rec.snap(2000)
    print("Clearly wrong cells at the end:", len(mistakes()), "of", len(clear))
    rec.hold(1000)
    rec.snap(100, cursor=None)
    rec.save(os.path.join(OUT_DIR, "object_classification.gif"))
    viewer.close()


def stroke_points(start, end, n):
    return [(start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])) for t in np.linspace(0, 1, n)]


def paint_stroke(layer, points, value, radius=3):
    from skimage.draw import disk
    data = layer.data
    for p in points:
        rr, cc = disk(p, radius, shape=data.shape)
        data[rr, cc] = value
    layer.data = data


# A 1024 x 1024 window (0.25 um / pixel) of a BCSS breast cancer region, shown at half resolution.
# The classes, grouped as in Figure 4 of the micro-sam2 paper: 1 = tumor, 2 = stroma, 3 = inflammatory infiltrate,
# 4 = necrosis. The window has all four classes as large regions and no unlabeled area.
BCSS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/bcss"
BCSS_REGION = "TCGA-E2-A1B6-DX1_xmin16266_ymin50634_MPP-0.2500.png"
BCSS_WINDOW = (3072, 2048, 1024)  # y, x, size
BCSS_GROUPS = {1: 1, 19: 1, 20: 1, 2: 2, 3: 3, 10: 3, 11: 3, 4: 4}


def load_bcss_window():
    from skimage.transform import resize
    y, x, size = BCSS_WINDOW
    image = imageio.imread(os.path.join(BCSS_ROOT, "rgbs_colorNormalized", BCSS_REGION))[y:y + size, x:x + size, :3]
    labels = imageio.imread(os.path.join(BCSS_ROOT, "masks", BCSS_REGION))[y:y + size, x:x + size]
    reference = np.zeros_like(labels)
    for source, target in BCSS_GROUPS.items():
        reference[labels == source] = target
    image = (resize(image, (size // 2, size // 2, 3), anti_aliasing=True, preserve_range=True)).astype("uint8")
    reference = reference[::2, ::2]
    return image, reference


def freehand_points(start, end, n, rng):
    """Points of a stroke from start to end that look drawn by hand: a gentle bend and a small wobble."""
    start, end = np.asarray(start, dtype="float32"), np.asarray(end, dtype="float32")
    direction = end - start
    normal = np.array([-direction[1], direction[0]]) / (np.linalg.norm(direction) + 1e-6)
    t = np.linspace(0, 1, n)
    length = np.linalg.norm(direction)
    bend = rng.uniform(-0.12, 0.12) * length * np.sin(np.pi * t)
    wobble = 0.03 * length * np.sin(2 * np.pi * rng.uniform(1.5, 2.5) * t + rng.uniform(0, 2 * np.pi))
    # Uneven speed along the stroke, as a hand slows down at the ends.
    t = t + 0.08 * np.sin(2 * np.pi * t)
    points = start + t[:, None] * direction + (bend + wobble)[:, None] * normal
    return [tuple(p) for p in points]


def interior_stroke(mask, length=70):
    """A straight stroke (start, end) through the deepest point of the mask, in the direction that stays inside."""
    from scipy.ndimage import distance_transform_edt
    # Pad with background, so that the image border counts as the border of the region.
    dist = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    center = np.array(np.unravel_index(np.argmax(dist), mask.shape), dtype="float32")
    best, best_depth = None, -1
    for angle in np.linspace(0, np.pi, 12, endpoint=False):
        direction = np.array([np.sin(angle), np.cos(angle)])
        ends = center - direction * length / 2, center + direction * length / 2
        points = np.array(stroke_points(*ends, 24))
        inside = np.clip(np.round(points).astype(int), 0, np.array(mask.shape) - 1)
        depth = dist[inside[:, 0], inside[:, 1]].min()
        if depth > best_depth:
            best, best_depth = ends, depth
    return best


def gif_pixel_classifier():
    from skimage.measure import label as connected_components
    from micro_sam.sam_annotator.pixel_classifier import pixel_classifier

    image, reference = load_bcss_window()
    n_classes = 4

    viewer = napari.Viewer(show=False)
    pixel_classifier(
        image, model_type=HISTO_MODEL,
        embedding_path=os.path.join(
            EMB_DIR, f"bcss-{BCSS_REGION[:12]}-{BCSS_WINDOW[0]}-{BCSS_WINDOW[1]}-{HISTO_MODEL}.zarr"
        ),
        return_viewer=True, viewer=viewer,
    )
    classifier = AnnotatorState().annotator
    rec = Recorder(viewer)
    rec.fit(image.shape[:2])
    annotations = viewer.layers["annotations"]

    rng = np.random.default_rng(0)

    def draw(start, end, value, mask):
        from scipy.ndimage import distance_transform_edt
        # Keep the part of the freehand stroke that stays inside the region (with a margin for the brush).
        dist = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        points = freehand_points(start, end, 32, rng)
        pixels = np.clip(np.round(points).astype(int), 0, np.array(mask.shape) - 1)
        points = [p for p, (y, x) in zip(points, pixels) if dist[y, x] > 7]
        if len(points) < 4:
            points = stroke_points(start, end, 24)
        rec.move(points[0], steps=4)
        for i in range(0, len(points), 4):
            paint_stroke(annotations, points[i:i + 5], value, radius=6)
            rec.snap(40, cursor=points[min(i + 4, len(points) - 1)])

    def train():
        rec.hold(200)
        classifier._run_train_and_predict()
        prediction = viewer.layers["prediction"]
        # Show the tissue below the prediction.
        prediction.opacity = 0.5
        rec.snap(2000)
        labeled = np.isin(reference, range(1, n_classes + 1))
        print("Pixel accuracy:", round(float((prediction.data[labeled] == reference[labeled]).mean()), 3))
        return prediction.data

    # Start with two strokes per class, in the middle of the two largest regions of the class.
    rec.snap(600, cursor=(400, 80))
    for value in range(1, n_classes + 1):
        regions = connected_components(reference == value)
        sizes = np.bincount(regions.ravel())[1:]
        for region_id in np.argsort(sizes)[::-1][:2] + 1:
            if sizes[region_id - 1] > 2000:
                draw(*interior_stroke(regions == region_id, length=90), value, regions == region_id)
    prediction = train()

    # Correct the largest wrong region of each class and train again.
    corrections = []
    for value in range(1, n_classes + 1):
        regions = connected_components((reference == value) & (prediction != value))
        if regions.max() == 0:
            continue
        largest = np.argmax(np.bincount(regions.ravel())[1:]) + 1
        if (regions == largest).sum() > 500:
            corrections.append((regions == largest, value))
    for mask, value in corrections:
        draw(*interior_stroke(mask, length=60), value, mask)
    train()
    rec.hold(1000)
    rec.snap(100, cursor=None)
    rec.save(os.path.join(OUT_DIR, "pixel_classification.gif"))
    viewer.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)
    gifs = {
        "2d": gif_2d_boxes, "3d": gif_3d_em, "tracking": gif_tracking,
        "objects": gif_object_classifier, "pixels": gif_pixel_classifier,
    }
    for name in sys.argv[1:]:
        gifs[name]()


if __name__ == "__main__":
    main()
