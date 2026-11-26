"""Make the micro-sam2 release video: 3d segmentations of micro-sam2, whose objects fly apart and come back together.

Usage: python make_release_video.py [step ...]

The steps run in this order, and without arguments all of them run:
- segment: segment an Arabidopsis ovule (PlantSeg, confocal) and a MitoEM block (rat, serial section EM)
  with the automatic segmentation of micro-sam2, and save the segmentations in OUT_DIR.
- ovule, em: render one video each in OUT_DIR, as an mp4 (1080 x 1080, 30 fps) and a smaller gif.
  Each video is a seamless loop: the camera turns once around the objects.
- release: put the two videos side by side with the micro-sam2 logo, as one mp4 (2160 x 1080),
  for example for LinkedIn, which takes one video only.

The segment step needs a GPU. It takes about 5 minutes for the ovule (2 GPUs) and 20 minutes for the MitoEM block.
The render steps need a display with at least 1700 x 1500 pixels, for example a virtual one:
Xvnc :7 -geometry 2400x1800 -depth 24 -SecurityTypes None, and then DISPLAY=:7 python make_release_video.py.
Each render takes about 25 minutes.
"""
import os
import subprocess
import sys
import time

import h5py
import numpy as np
from PIL import Image
from elf.io import open_file
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes, regionprops
from skimage.measure import label as connected_components

import napari
from qtpy.QtWidgets import QApplication
from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

# The videos and the segmentations are large, so they are written outside of the repository.
OUT_DIR = "/mnt/vast-nhr/home/archit/u12090/micro_sam2_release_media"
FRAME_DIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "release_frames")
FFMPEG = "/mnt/lustre-grete/usr/u12090/micromamba/envs/super/bin/ffmpeg"
LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../plotting/v2/for_paper/assets/micro_sam2_logo.png")

# A full confocal volume of an Arabidopsis ovule from PlantSeg, with stained cell membranes.
OVULE_FILE = "/mnt/vast-nhr/projects/cidas/cca/data/plantseg_ovules/ovules_test/N_511_final_crop_ds2.h5"
OVULE_MODEL = "hvit_l_cells"
OVULE_SEG = os.path.join(OUT_DIR, "ovule_segmentation.npy")
# A block of the MitoEM rat validation volume (100 x 1024 x 1024 voxels, 30 nm sections, 8 nm pixels).
MITOEM_FILE = "/mnt/vast-nhr/projects/cidas/cca/data/mitoem/rat_val.n5"
MITOEM_BLOCK = np.s_[:, 1024:2048, 2304:3328]
MITOEM_MODEL = "hvit_s_em_mito"
MITOEM_SEG = os.path.join(OUT_DIR, "mitoem_segmentation.npy")

CANVAS, FPS, SECONDS = 1080, 30, 6
BACKGROUND = "#0b0b12"
# The colors of the micro-sam2 logo: purples, teals and a few coral accents.
PALETTE = np.array([
    (91, 76, 154), (119, 103, 169), (150, 132, 204), (73, 159, 129), (96, 190, 160),
    (52, 120, 112), (176, 160, 226), (232, 124, 93), (240, 160, 120),
]) / 255
PALETTE_WEIGHTS = np.array([3, 3, 2, 3, 3, 2, 2, 1, 0.6])


def keep_largest_component(seg):
    """Keep the objects that touch the largest connected component of the foreground."""
    components = connected_components(seg[::2, ::2, ::2] > 0)
    largest = np.argmax(np.bincount(components.ravel())[1:]) + 1
    return np.where(np.isin(seg, np.unique(seg[::2, ::2, ::2][components == largest])), seg, 0)


def object_meshes(seg, spacing, step=2, sigma=1.0):
    """Build one smooth surface for all objects, and return the object index of each vertex.

    The masks are interpolated along z to the xy spacing first, so that the thick sections do not show as steps.
    """
    factor = max(1, int(round(spacing[0] / spacing[1])))
    vertices, faces, owner, n_vertices = [], [], [], 0
    for p in regionprops(seg):
        start = np.maximum(np.array(p.bbox[:3]) - 3, 0)
        stop = np.minimum(np.array(p.bbox[3:]) + 3, seg.shape)
        crop = tuple(slice(a, b) for a, b in zip(start, stop))
        mask = zoom((seg[crop] == p.label).astype("float32"), (factor, 1, 1), order=1)
        # Smooth more along z, where the interpolation leaves faint steps.
        mask = gaussian_filter(mask, (sigma * max(1, 1.2 * factor), 1.2 * sigma, 1.2 * sigma))
        if mask.max() < 0.5 or min(mask.shape) < 3:
            continue
        fine_spacing = (spacing[0] / factor, spacing[1], spacing[2])
        v, f, _, _ = marching_cubes(mask, 0.5, spacing=fine_spacing, step_size=step)
        vertices.append(v + start * np.array(spacing))
        faces.append(f + n_vertices)
        # The objects without a surface are left out, so the index counts the surfaces only.
        owner.append(np.full(len(v), len(owner)))
        n_vertices += len(v)
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(owner)


def object_colors(n_objects, seed=0):
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(PALETTE), n_objects, p=PALETTE_WEIGHTS / PALETTE_WEIGHTS.sum())
    # A little jitter of the brightness, so that neighbors of one color stay apart.
    colors = PALETTE[choice] * rng.uniform(0.85, 1.15, (n_objects, 1))
    return np.clip(np.concatenate([colors, np.ones((n_objects, 1))], axis=1), 0, 1)


def explosion(t):
    """The explosion factor at the loop time t in [0, 1): rest, fly apart, hold, come back, rest."""
    def ease(x):
        return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))
    return ease((t - 0.15) / 0.25) - ease((t - 0.58) / 0.25)


def view_direction(azimuth, elevation):
    """The view direction (z, y, x) of a camera that looks down at an elevation, with z pointing up."""
    a, e = np.radians(azimuth), np.radians(elevation)
    return (-np.sin(e), -np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a))


def framing(points):
    """The center and the diagonal of the bounding box of points."""
    low, high = points.min(axis=0), points.max(axis=0)
    return (low + high) / 2, np.linalg.norm(high - low)


def render(name, vertices, faces, owner, offsets, elevation, setup=None, fixed_points=None):
    """Render the frames of one loop and write the mp4 and the gif.

    The camera frames the bounding box of the objects (and of fixed_points) at rest and at the full explosion,
    and moves between the two with the explosion.
    """
    n_objects = owner.max() + 1
    colors = object_colors(n_objects)[owner]
    viewer = napari.Viewer(show=False)
    viewer.show()
    viewer.window._qt_window.resize(CANVAS + 500, CANVAS + 300)
    for _ in range(5):
        QApplication.processEvents()
    canvas = viewer.window._qt_viewer.canvas.native
    assert min(canvas.width(), canvas.height()) >= CANVAS, f"The display is too small: {canvas.size()}"
    viewer.window._qt_viewer.canvas.background_color_override = BACKGROUND
    if setup is not None:
        setup(viewer)
    surface = viewer.add_surface((vertices, faces), vertex_colors=colors, shading="smooth", name=name)
    viewer.dims.ndisplay = 3
    for _ in range(5):
        QApplication.processEvents()

    fixed = np.zeros((0, 3)) if fixed_points is None else fixed_points
    center0, extent0 = framing(np.concatenate([vertices[::50], fixed]))
    center1, extent1 = framing(np.concatenate([(vertices + offsets[owner])[::50], fixed]))

    frame_dir = os.path.join(FRAME_DIR, name)
    os.makedirs(frame_dir, exist_ok=True)
    # N_FRAMES renders a short preview.
    n_frames = int(os.environ.get("N_FRAMES", FPS * SECONDS))
    # The first screenshot comes before the camera is in place, so frame 0 is rendered twice.
    for i in [0] + list(range(n_frames)):
        t = i / n_frames if "N_FRAMES" not in os.environ else [0.0, 0.3, 0.45, 0.9][i % 4]
        f = explosion(t)
        surface.data = (vertices + f * offsets[owner], faces)
        viewer.camera.center = tuple((1 - f) * center0 + f * center1)
        viewer.camera.set_view_direction(view_direction(360 * t, elevation), up_direction=(1, 0, 0))
        # The diagonal fits in the frame from all directions, so 1.05 still leaves a margin.
        viewer.camera.zoom = 1.05 * CANVAS / ((1 - f) * extent0 + f * extent1)
        QApplication.processEvents()
        shot = viewer.screenshot(canvas_only=True, flash=False)[..., :3]
        h, w = shot.shape[:2]
        y0, x0 = (h - CANVAS) // 2, (w - CANVAS) // 2
        Image.fromarray(shot[y0:y0 + CANVAS, x0:x0 + CANVAS]).save(os.path.join(frame_dir, f"{i:04d}.png"))
        print(i, end=" ", flush=True)
    viewer.close()

    mp4, gif = os.path.join(OUT_DIR, f"{name}.mp4"), os.path.join(OUT_DIR, f"{name}.gif")
    frames = os.path.join(frame_dir, "%04d.png")
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", frames,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", "-movflags", "+faststart", mp4,
    ], check=True)
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", frames, "-vf",
        "fps=20,scale=600:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=200[p];[b][p]paletteuse=dither=none",
        "-loop", "0", gif,
    ], check=True)
    print("\nwrote", mp4, os.path.getsize(mp4) // 1000, "KB,", gif, os.path.getsize(gif) // 1000, "KB")


def segment():
    """Segment the ovule and the MitoEM block with micro-sam2 (automatic segmentation, APG)."""
    with h5py.File(OVULE_FILE, "r") as f:
        raw = f["raw"][:]
    start = time.time()
    # All visible GPUs are used.
    predictor, segmenter = get_predictor_and_segmenter(model_type=OVULE_MODEL, ndim=3)
    seg = automatic_instance_segmentation(predictor, segmenter, raw, ndim=3, verbose=False)
    np.save(OVULE_SEG, seg)
    print("ovule:", len(np.unique(seg)) - 1, "cells in", round(time.time() - start), "s")

    raw = open_file(MITOEM_FILE, "r")["raw"][MITOEM_BLOCK]
    start = time.time()
    # The block is larger than the model input, so it is segmented in tiles.
    predictor, segmenter = get_predictor_and_segmenter(MITOEM_MODEL, segmentation_mode="apg", is_tiled=True, ndim=3)
    seg = automatic_instance_segmentation(
        predictor, segmenter, raw, ndim=3, tile_shape=(384, 384), halo=(64, 64), verbose=False
    )
    np.save(MITOEM_SEG, seg.astype("uint32"))
    print("mitoem:", len(np.unique(seg)) - 1, "mitochondria in", round(time.time() - start), "s")


def ovule():
    # The z spacing is about 2 times the xy spacing. The data does not store it, so it is for the visuals only.
    seg = keep_largest_component(np.load(OVULE_SEG))
    spacing = (2.0, 1.0, 1.0)
    vertices, faces, owner = object_meshes(seg, spacing, step=2)
    centroids = np.stack([vertices[owner == i].mean(axis=0) for i in range(owner.max() + 1)])
    offsets = 1.6 * (centroids - centroids.mean(axis=0))
    print("ovule:", owner.max() + 1, "cells,", len(vertices), "vertices")
    render("ovule_explosion", vertices, faces, owner, offsets, elevation=20)


def em():
    seg = np.load(MITOEM_SEG)
    # The bottom section of the block, for the floor.
    raw = open_file(MITOEM_FILE, "r")["raw"][MITOEM_BLOCK][:1]
    # 30 nm sections and 8 nm pixels.
    spacing = (30 / 8, 1.0, 1.0)
    vertices, faces, owner = object_meshes(seg, spacing, step=2)
    centroids = np.stack([vertices[owner == i].mean(axis=0) for i in range(owner.max() + 1)])
    center = centroids.mean(axis=0)
    # The mitochondria fly up out of the block, and apart.
    offsets = 1.2 * (centroids - center)
    offsets[:, 0] = np.abs(offsets[:, 0]) + 0.8 * seg.shape[0] * spacing[0]
    # The corners of the floor.
    floor = np.array([(0, y, x) for y in (0, seg.shape[1]) for x in (0, seg.shape[2])], dtype="float64")
    print("em:", owner.max() + 1, "mitochondria,", len(vertices), "vertices")

    def setup(viewer):
        # The EM image stays in place as an opaque floor: the bottom slice of the block.
        viewer.add_image(
            raw, scale=spacing, colormap="gray", contrast_limits=np.percentile(raw[0], (1, 99.5)),
            depiction="plane", blending="opaque",
            plane={"position": (0, raw.shape[1] / 2, raw.shape[2] / 2), "normal": (1, 0, 0)},
        )

    render("em_explosion", vertices, faces, owner, offsets, elevation=35, setup=setup, fixed_points=floor)


def release():
    """Put the EM and the ovule video side by side, with the logo at the bottom center between them.

    The objects never come into this area, so nothing hides the logo.
    """
    em_mp4, ovule_mp4 = os.path.join(OUT_DIR, "em_explosion.mp4"), os.path.join(OUT_DIR, "ovule_explosion.mp4")
    out = os.path.join(OUT_DIR, "micro_sam2_release.mp4")
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error", "-i", em_mp4, "-i", ovule_mp4, "-i", LOGO, "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v];[2:v]scale=380:-1:flags=lanczos[logo];[v][logo]overlay=(W-w)/2:H-h-36",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
    ], check=True)
    print("wrote", out, os.path.getsize(out) // 1000, "KB")


def main():
    steps = {"segment": segment, "ovule": ovule, "em": em, "release": release}
    for name in sys.argv[1:] or list(steps):
        steps[name]()


if __name__ == "__main__":
    main()
