"""Record the demo GIFs of the README with the napari annotation tools.

Usage: python make_gifs.py 2d 3d tracking objects pixels batch

Run each GIF in its own process, so that each tool starts with a fresh state.
The script needs a display and a GPU.
The pixel classification GIF needs your own HuggingFace access to UNI2-h.
The pixel and batch classification GIFs need the BCSS and PUMA data.
"""
import io
import os
import sys
import zipfile

import h5py
import numpy as np
import imageio.v3 as imageio
from skimage.draw import disk
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as connected_components, regionprops

import napari
from qtpy.QtWidgets import QApplication

from elf.io import open_file

import micro_sam.util as util
from micro_sam.classical_features import OBJECT_FEATURES
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator import annotator, annotator_tracking
from micro_sam.sam_annotator.pixel_classifier import pixel_classifier
from micro_sam.sam_annotator.object_classifier import object_classifier, batch_object_classifier
from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.sample_data import (
    fetch_3d_example_data, fetch_hela_2d_example_data, fetch_livecell_example_data, fetch_tracking_example_data
)

DATA_CACHE = os.path.join(util.get_cache_directory(), "sample_data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "doc", "gifs")
EMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")

CELL_MODEL = "hvit_l_cells"
EM_MODEL = "hvit_s_em_mito"
# UNI2-h from MahmoodLab. Its weights are gated.
HISTO_MODEL = "vit_univ2"

# A window of a BCSS breast cancer region with 4 large tissue regions, at 0.25 um per pixel.
BCSS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/bcss"
BCSS_REGION = "TCGA-E2-A1B6-DX1_xmin16266_ymin50634_MPP-0.2500.png"
BCSS_WINDOW = (3072, 2048, 1024)  # y, x, size
# The BCSS classes in the groups of Figure 4: 1 = tumor, 2 = stroma, 3 = inflammatory infiltrate, 4 = necrosis.
BCSS_GROUPS = {1: 1, 19: 1, 20: 1, 2: 2, 3: 3, 10: 3, 11: 3, 4: 4}

# 4 tiles (512 x 512 pixels at 40x) of one metastatic melanoma ROI of PUMA, with its nucleus classes.
PUMA_TILES = "/mnt/lustre-grete/usr/u12090/few_shot_sam/data/puma/train.h5"
PUMA_NAMES = [f"training_set_metastatic_roi_047_{y}_{x}" for y, x in [(0, 0), (0, 1), (1, 0), (1, 1)]]
# The PUMA classes stroma (1), tumor (2) and lymphocyte (5). Class 1 is the lymphocytes, class 2 the other nuclei.
PUMA_GROUPS = {5: 1, 1: 2, 2: 2}

# The arrow cursor as a polygon in pixels, with the tip at (0, 0).
CURSOR = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]


class Recorder:
    """Record frames of the napari canvas and save them as a GIF.

    Args:
        viewer: The napari viewer.
        size: The size of the square canvas in pixels.
    """

    def __init__(self, viewer, size=512):
        self.viewer = viewer
        self.size = size
        self.frames, self.durations = [], []
        self.cursor = None
        # An optional text in the top left corner of the frames.
        self.label = None
        self.font = ImageFont.load_default(size=28)
        # A hidden viewer renders a black canvas.
        viewer.show()
        viewer.window._qt_window.resize(size + 700, size + 300)
        for _ in range(5):
            QApplication.processEvents()

    def fit(self, image_shape):
        """Center the image and zoom it to fill the square canvas."""
        self.viewer.reset_view()
        h, w = image_shape[-2:]
        self.viewer.camera.center = (h / 2 - 0.5, w / 2 - 0.5)
        self.viewer.camera.zoom = self.size / max(h, w)

    def to_screen(self, yx, shot_shape):
        """Map a data position (y, x) to a pixel position (x, y) in the screenshot."""
        cam = self.viewer.camera
        cy, cx = cam.center[-2:]
        h, w = shot_shape[:2]
        return ((yx[1] - cx) * cam.zoom + w / 2, (yx[0] - cy) * cam.zoom + h / 2)

    def grab(self):
        QApplication.processEvents()
        shot = self.viewer.screenshot(canvas_only=True, flash=False)[..., :3]
        # Crop the center square, so that all frames have the same shape.
        h, w = shot.shape[:2]
        s = min(h, w, self.size)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return shot, (y0, x0, s)

    def snap(self, duration=60, cursor="keep"):
        """Add one frame.

        Args:
            duration: The display time of the frame in milliseconds.
            cursor: The data position (y, x) of the cursor, None to hide it, or 'keep'.
        """
        if not (isinstance(cursor, str) and cursor == "keep"):
            self.cursor = cursor
        shot, (y0, x0, s) = self.grab()
        img = Image.fromarray(shot)
        if self.cursor is not None:
            px, py = self.to_screen(self.cursor, shot.shape)
            poly = [(px + dx, py + dy) for dx, dy in CURSOR]
            ImageDraw.Draw(img).polygon(poly, fill=(255, 255, 255), outline=(0, 0, 0))
        img = img.crop((x0, y0, x0 + s, y0 + s))
        if self.label is not None:
            ImageDraw.Draw(img).text(
                (14, 10), self.label, font=self.font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0)
            )
        self.frames.append(img)
        self.durations.append(duration)

    def hold(self, duration):
        """Make the display time of the last frame longer."""
        self.durations[-1] += duration

    def move(self, target, steps=6, duration=40):
        """Move the cursor to the data position (y, x) in a few frames."""
        start = self.cursor if self.cursor is not None else target
        for t in np.linspace(0, 1, steps + 1)[1:]:
            pos = (start[0] + t * (target[0] - start[0]), start[1] + t * (target[1] - start[1]))
            self.snap(duration, cursor=pos)

    def save(self, path, out_size=400):
        frames = [f.resize((out_size, out_size), Image.LANCZOS) for f in self.frames]
        # One palette from all frames prevents flicker.
        # Median cut gives the rare label colors no entries, so use octree.
        mosaic = Image.new("RGB", (out_size * len(frames), out_size))
        for i, f in enumerate(frames):
            mosaic.paste(f, (i * out_size, 0))
        palette = mosaic.quantize(colors=255, method=Image.FASTOCTREE)
        frames = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=self.durations, loop=0, optimize=True)


def auto_segment(image, model_type, name, ndim=2):
    """Segment the image automatically, and cache the result in EMB_DIR."""
    cache = os.path.join(EMB_DIR, f"{name}-seg.npy")
    if os.path.exists(cache):
        return np.load(cache)
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, ndim=ndim)
    seg = automatic_instance_segmentation(predictor, segmenter, image, ndim=ndim, verbose=False)
    np.save(cache, seg)
    return seg


def pick_objects(seg, n, min_area, margin=8):
    """Pick the n largest objects that keep a margin to the border. Pass n=None to pick all objects."""
    h, w = seg.shape
    props = [
        p for p in regionprops(seg) if p.area >= min_area and
        p.bbox[0] > margin and p.bbox[1] > margin and p.bbox[2] < h - margin and p.bbox[3] < w - margin
    ]
    props.sort(key=lambda p: p.area, reverse=True)
    return props if n is None else props[:n]


def inner_point(mask):
    """Get the point in the mask with the largest distance to the border of the mask."""
    dist = distance_transform_edt(mask)
    return np.unravel_index(np.argmax(dist), mask.shape), dist.max()


def best_window(seg, size, min_area, margin=12, stride=32):
    """Find the square window that contains the most objects."""
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


def set_points(layer, coords, labels, **extra):
    layer.data = np.array(coords, dtype="float32")
    properties = {"label": np.array(labels)}
    properties.update({k: np.array(v) for k, v in extra.items()})
    layer.properties = properties
    layer.refresh_colors()


def paint_disc(layer, center, value, radius=8):
    data = layer.data
    rr, cc = disk(center, radius, shape=data.shape)
    data[rr, cc] = value
    layer.data = data


def paint_stroke(layer, points, value, radius=3):
    data = layer.data
    for p in points:
        rr, cc = disk(p, radius, shape=data.shape)
        data[rr, cc] = value
    layer.data = data


def stroke_points(start, end, n):
    return [(start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])) for t in np.linspace(0, 1, n)]


def freehand_points(start, end, n, rng):
    """Get the points of a stroke that looks like a hand drew it, with a small bend and a small wave."""
    start, end = np.asarray(start, dtype="float32"), np.asarray(end, dtype="float32")
    direction = end - start
    length = np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]]) / (length + 1e-6)
    t = np.linspace(0, 1, n)
    bend = rng.uniform(-0.12, 0.12) * length * np.sin(np.pi * t)
    wave = 0.03 * length * np.sin(2 * np.pi * rng.uniform(1.5, 2.5) * t + rng.uniform(0, 2 * np.pi))
    # A hand moves slower at the ends of a stroke.
    t = t + 0.08 * np.sin(2 * np.pi * t)
    points = start + t[:, None] * direction + (bend + wave)[:, None] * normal
    return [tuple(p) for p in points]


def interior_stroke(mask, length=70):
    """Get the start and the end of a straight stroke through the deepest point of the mask."""
    # The padding makes the image border a border of the mask.
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


def lineage_targets(mother, daughters, division_frame):
    """Get the prompt points of a lineage from the CTC ground truth of DIC-HeLa 01.

    Returns:
        The point (y, x) of the mother on the first frame.
        The point of the mother on the frame of the division.
        The points of the daughters on the frame after the division.
    """
    with zipfile.ZipFile(os.path.join(DATA_CACHE, "DIC-C2DH-HeLa.zip")) as f:
        def markers(t):
            return imageio.imread(io.BytesIO(f.read(f"DIC-C2DH-HeLa/01_GT/TRA/man_track{t:03d}.tif")))
        first = inner_point(markers(0) == mother)[0]
        last = inner_point(markers(division_frame) == mother)[0]
        after = markers(division_frame + 1)
        children = [inner_point(after == d)[0] for d in daughters]
    return first, last, children


def load_bcss_window():
    """Load the BCSS window at half resolution, with the tissue classes as reference."""
    y, x, size = BCSS_WINDOW
    image = imageio.imread(os.path.join(BCSS_ROOT, "rgbs_colorNormalized", BCSS_REGION))[y:y + size, x:x + size, :3]
    labels = imageio.imread(os.path.join(BCSS_ROOT, "masks", BCSS_REGION))[y:y + size, x:x + size]
    reference = np.zeros_like(labels)
    for source, target in BCSS_GROUPS.items():
        reference[labels == source] = target
    image = resize(image, (size // 2, size // 2, 3), anti_aliasing=True, preserve_range=True).astype("uint8")
    return image, reference[::2, ::2]


def gif_2d_boxes():
    image = imageio.imread(fetch_hela_2d_example_data(DATA_CACHE))
    seg = auto_segment(image, CELL_MODEL, "hela2d")
    # Draw a box around each cell, also around the cells at the image border.
    objects = pick_objects(seg, n=None, min_area=300, margin=-1)
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
    h, w = image.shape[:2]
    rec.snap(600, cursor=(h * 0.8, w * 0.1))
    for p in objects:
        y0, x0, y1, x1 = p.bbox
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


def gif_3d_em():
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
    # The batched mode segments one object for each positive point.
    interactive.batched_checkbox.setChecked(True)
    interactive.apply_to_volume_checkbox.setChecked(True)
    rec = Recorder(viewer)
    rec.fit(crop.shape)
    viewer.dims.current_step = (dz, 0, 0)

    points = viewer.layers["point_prompts"]
    rec.snap(600, cursor=(size * 0.85, size * 0.15))
    for i in range(len(targets)):
        rec.move(targets[i], steps=4)
        set_points(points, [(dz, *t) for t in targets[:i + 1]], ["positive"] * (i + 1))
        rec.snap(200)
    rec.hold(200)
    interactive.segment(viewer)
    n_objects = len(np.unique(viewer.layers["current_object"].data)) - 1
    print("The batched segmentation returned", n_objects, "objects")
    rec.snap(900, cursor=None)
    # Show every 3rd slice of a scroll through the volume, to keep the GIF small.
    steps = list(range(dz, crop.shape[0])) + list(range(crop.shape[0] - 1, -1, -1)) + list(range(0, dz + 1))
    for z in steps[::3]:
        viewer.dims.current_step = (z, 0, 0)
        rec.snap(165)
    rec.hold(1500)
    rec.save(os.path.join(OUT_DIR, "interactive_3d.gif"), out_size=360)
    viewer.close()


def gif_tracking():
    with open_file(fetch_tracking_example_data(DATA_CACHE), mode="r") as f:
        timeseries = f["*.tif"][:]
    # In lineage 1, the mother divides after frame 29 and both daughters live until the last frame.
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

    # Track the mother from a point on the first frame.
    rec.snap(600, cursor=(h * 0.85, w * 0.15))
    rec.move(first, steps=5)
    add_point(0, first, "track", 1)
    rec.snap(300)
    interactive.segment(viewer)
    rec.snap(500)

    # Mark the division. This ends the mother and adds the track ids of the daughters.
    scrub(0, division_frame)
    rec.move(last, steps=4)
    add_point(division_frame, last, "division", 1)
    rec.snap(300)
    interactive.segment(viewer)
    rec.snap(500)

    # Track each daughter from a point on the frame after the division.
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

    # Play every 3rd frame of the movie, to keep the GIF small.
    scrub(0, timeseries.shape[0] - 1, duration=180)
    rec.hold(1500)
    rec.save(os.path.join(OUT_DIR, "interactive_tracking.gif"), out_size=360)
    viewer.close()


def gif_object_classifier():
    image = imageio.imread(fetch_livecell_example_data(DATA_CACHE))
    h, w = image.shape[:2]
    y0, x0 = (h - 512) // 2, (w - 512) // 2
    image = image[y0:y0 + 512, x0:x0 + 512]
    seg = auto_segment(image, CELL_MODEL, "livecell-crop")

    # Class 1 is the round cells and class 2 is the elongated cells, split at the median eccentricity.
    # The cells close to the median are ambiguous, so the corrections use only the clear cells.
    cells = pick_objects(seg, n=None, min_area=250, margin=4)
    cells.sort(key=lambda p: p.eccentricity)
    median = np.median([p.eccentricity for p in cells])
    reference = {p.label: 1 if p.eccentricity < median else 2 for p in cells}
    clear = {p.label for p in cells if abs(p.eccentricity - median) > 0.08}
    examples = [(p, 1) for p in cells[:3]] + [(p, 2) for p in cells[-3:]]

    viewer = napari.Viewer(show=False)
    object_classifier(
        image, seg, model_type=CELL_MODEL, embedding_path=os.path.join(EMB_DIR, "livecell-crop.zarr"),
        return_viewer=True, viewer=viewer,
    )
    classifier = AnnotatorState().annotator
    # The embeddings do not describe the shape, so add the classical shape features.
    shape_features = ["eccentricity", "solidity", "axis_major_length", "axis_minor_length"]
    classifier._set_options(False, None, False, classical_features_val=shape_features)
    rec = Recorder(viewer)
    rec.fit(image.shape)
    annotations = viewer.layers["annotations"]

    def click(p, value):
        pos = inner_point(seg == p.label)[0]
        rec.move(pos, steps=5)
        paint_disc(annotations, pos, value)
        rec.snap(250)

    def mistakes():
        prediction = viewer.layers["prediction"].data
        wrong = [p for p in cells if p.label in clear and np.median(prediction[seg == p.label]) != reference[p.label]]
        return sorted(wrong, key=lambda p: abs(p.eccentricity - median), reverse=True)

    rec.snap(600, cursor=(400, 80))
    for p, value in examples:
        click(p, value)
    rec.hold(200)
    classifier._run_train_and_predict()
    viewer.layers["segmentation"].visible = False
    rec.snap(2000)

    # Correct the clearest mistakes and train again, two times.
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


def gif_pixel_classifier():
    image, reference = load_bcss_window()
    n_classes = 4

    viewer = napari.Viewer(show=False)
    embedding_name = f"bcss-{BCSS_REGION[:12]}-{BCSS_WINDOW[0]}-{BCSS_WINDOW[1]}-{HISTO_MODEL}.zarr"
    pixel_classifier(
        image, model_type=HISTO_MODEL, embedding_path=os.path.join(EMB_DIR, embedding_name),
        return_viewer=True, viewer=viewer,
    )
    classifier = AnnotatorState().annotator
    rec = Recorder(viewer)
    rec.fit(image.shape[:2])
    annotations = viewer.layers["annotations"]
    rng = np.random.default_rng(0)

    def draw(start, end, value, mask):
        # Keep the points of the stroke that stay inside the region, with a margin for the brush.
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

    # Draw a stroke in the two largest regions of each class.
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


def load_puma_tile(f, name):
    """Load a PUMA tile with the nuclei of PUMA_GROUPS, and the reference class of each nucleus."""
    image, instances, semantic = f[name]["raw"][:], f[name]["instances"][:], f[name]["semantic"][:]
    seg, reference = np.zeros_like(instances), {}
    for p in regionprops(instances):
        value = int(np.bincount(semantic[p.coords[:, 0], p.coords[:, 1]]).argmax())
        if value in PUMA_GROUPS:
            seg[p.coords[:, 0], p.coords[:, 1]] = p.label
            reference[p.label] = PUMA_GROUPS[value]
    return image, seg, reference


def gif_batch_nuclei():
    with h5py.File(PUMA_TILES, "r") as f:
        images, segmentations, references = zip(*[load_puma_tile(f, name) for name in PUMA_NAMES])
    n_images, n_classes = len(images), 2

    viewer = napari.Viewer(show=False)
    output_folder = os.path.join(EMB_DIR, "batch-nuclei")
    embedding_paths = [os.path.join(EMB_DIR, f"puma-{name}-{CELL_MODEL}.zarr") for name in PUMA_NAMES]
    batch_object_classifier(
        list(images), list(segmentations), output_folder, embedding_paths=embedding_paths, model_type=CELL_MODEL,
        viewer=viewer, return_viewer=True,
    )
    state = AnnotatorState()
    # The size, the intensity and the shape of a nucleus tell the lymphocytes apart from the other nuclei.
    state.annotator._set_options(False, None, False, classical_features_val=list(OBJECT_FEATURES))
    rec = Recorder(viewer)
    rec.fit(images[0].shape[:2])
    annotations = viewer.layers["annotations"]

    def click(index, label, value):
        pos = inner_point(segmentations[index] == label)[0]
        rec.move(pos, steps=5)
        paint_disc(annotations, pos, value, radius=5)
        rec.snap(250)

    def mistakes(index):
        seg, prediction = segmentations[index], viewer.layers["prediction"].data
        wrong = [
            p for p in regionprops(seg)
            if np.median(prediction[p.coords[:, 0], p.coords[:, 1]]) != references[index][p.label]
        ]
        accuracy = 1 - len(wrong) / len(references[index])
        print("Image", index + 1, "accuracy:", round(accuracy, 2))
        return wrong

    def train():
        rec.hold(200)
        state.annotator._run_train_and_predict()
        viewer.layers["segmentation"].visible = False
        rec.snap(2000)

    # On the first image, click 5 nuclei of each class.
    rec.label = f"Image 1 / {n_images}"
    rec.snap(600, cursor=(400, 80))
    for value in range(1, n_classes + 1):
        for label in [label for label, ref in references[0].items() if ref == value][:5]:
            click(0, label, value)
    train()
    mistakes(0)

    # The next images start with the prediction of the classifier, which keeps the earlier annotations.
    for index in range(1, n_images):
        state.widgets["batch_next"]()
        rec.label = f"Image {index + 1} / {n_images}"
        rec.snap(2000)
        wrong = mistakes(index)
        if index < n_images - 1:
            for p in wrong[:5]:
                click(index, p.label, references[index][p.label])
            train()
            mistakes(index)
    rec.hold(1000)
    rec.snap(100, cursor=None)
    rec.save(os.path.join(OUT_DIR, "batch_object_classification.gif"))
    viewer.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)
    gifs = {
        "2d": gif_2d_boxes, "3d": gif_3d_em, "tracking": gif_tracking,
        "objects": gif_object_classifier, "pixels": gif_pixel_classifier, "batch": gif_batch_nuclei,
    }
    for name in sys.argv[1:]:
        gifs[name]()


if __name__ == "__main__":
    main()
