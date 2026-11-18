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
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
EMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
CELL_MODEL = "hvit_l_cells"
EM_MODEL = "hvit_s_em_mito"


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


def gif_tracking():
    from elf.io import open_file
    from micro_sam.sam_annotator import annotator_tracking
    from micro_sam.sample_data import fetch_tracking_example_data

    with open_file(fetch_tracking_example_data(DATA_CACHE), mode="r") as f:
        timeseries = f["*.tif"][:]
    seg = auto_segment(timeseries[0], CELL_MODEL, "hela-track-t0")
    cells = pick_objects(seg, n=4, min_area=1500, margin=20)
    targets = [inner_point(seg == p.label)[0] for p in cells]
    targets.sort(key=lambda yx: (yx[0] // 100, yx[1]))
    print("Tracking", len(targets), "cells in", timeseries.shape)

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
    rec.snap(600, cursor=(h * 0.85, w * 0.15))
    for target in targets:
        rec.move(target, steps=5)
        set_points(points, [(0, *target)], ["positive"], state=["track"], track_id=["1"])
        rec.snap(250)
        interactive.segment(viewer)
        rec.snap(250)
        # Commit the track, so that the next cell starts a new track.
        state.widgets["commit"](viewer)
        rec.snap(300)
    rec.snap(400, cursor=None)
    # Play the timeseries to show the tracks.
    for t in range(timeseries.shape[0]):
        viewer.dims.current_step = (t, 0, 0)
        rec.snap(70)
    rec.hold(1500)
    rec.save(os.path.join(OUT_DIR, "interactive_tracking.gif"))
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
    for z in steps:
        viewer.dims.current_step = (z, 0, 0)
        rec.snap(55)
    rec.hold(1500)
    rec.save(os.path.join(OUT_DIR, "interactive_3d.gif"))
    viewer.close()


def paint_disc(layer, center, value, radius=5):
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

    # Annotate round cells as class 1 and elongated cells as class 2.
    cells = [p for p in pick_objects(seg, n=None, min_area=250, margin=4)]
    cells.sort(key=lambda p: p.eccentricity)
    examples = [(p, 1) for p in cells[:3]] + [(p, 2) for p in cells[-3:]]
    examples = [(inner_point(seg == p.label)[0], c) for p, c in examples]

    viewer = napari.Viewer(show=False)
    object_classifier(
        image, seg, model_type=CELL_MODEL, embedding_path=os.path.join(EMB_DIR, "livecell-crop.zarr"),
        return_viewer=True, viewer=viewer,
    )
    classifier = AnnotatorState().annotator
    rec = Recorder(viewer)
    rec.fit(image.shape)

    annotations = viewer.layers["annotations"]
    rec.snap(600, cursor=(400, 80))
    for pos, value in examples:
        rec.move(pos, steps=5)
        paint_disc(annotations, pos, value)
        rec.snap(250)
    rec.hold(200)
    classifier._run_train_and_predict()
    viewer.layers["segmentation"].visible = False
    rec.snap(3000, cursor=None)
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


def gif_pixel_classifier():
    from scipy.ndimage import distance_transform_edt
    from micro_sam.sample_data import fetch_3d_example_data
    from micro_sam.sam_annotator.pixel_classifier import pixel_classifier

    raw = imageio.imread(fetch_3d_example_data(DATA_CACHE))
    zc = raw.shape[0] // 2
    seg = auto_segment(raw[zc], EM_MODEL, "lucchi-mid")
    (y0, x0), _ = best_window(seg, 512, min_area=800)
    image = raw[zc, y0:y0 + 512, x0:x0 + 512]
    seg = seg[y0:y0 + 512, x0:x0 + 512]

    # Strokes along the long axis of three mitochondria (class 1), and in three places far from them (class 2).
    strokes = []
    for p in pick_objects(seg, n=3, min_area=800, margin=12):
        cy, cx = p.centroid
        dy, dx = np.sin(-p.orientation + np.pi / 2), np.cos(-p.orientation + np.pi / 2)
        half = 0.3 * p.axis_major_length
        strokes.append(((cy - half * dx, cx - half * dy), (cy + half * dx, cx + half * dy), 1))
    dist = distance_transform_edt(seg == 0)
    dist[:40], dist[-40:], dist[:, :40], dist[:, -40:] = 0, 0, 0, 0
    for _ in range(3):
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        strokes.append(((cy, cx - 30), (cy, cx + 30), 2))
        dist[max(cy - 90, 0):cy + 90, max(cx - 90, 0):cx + 90] = 0

    viewer = napari.Viewer(show=False)
    pixel_classifier(
        image, model_type=EM_MODEL, embedding_path=os.path.join(EMB_DIR, "lucchi-slice.zarr"),
        return_viewer=True, viewer=viewer,
    )
    classifier = AnnotatorState().annotator
    rec = Recorder(viewer)
    rec.fit(image.shape)

    annotations = viewer.layers["annotations"]
    rec.snap(600, cursor=(400, 80))
    for start, end, value in strokes:
        rec.move(start, steps=4)
        points = stroke_points(start, end, 24)
        for i in range(0, len(points), 4):
            paint_stroke(annotations, points[i:i + 5], value)
            rec.snap(40, cursor=points[min(i + 4, len(points) - 1)])
    rec.hold(200)
    classifier._run_train_and_predict()
    rec.snap(3000, cursor=None)
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
