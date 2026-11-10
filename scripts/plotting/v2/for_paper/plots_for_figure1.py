"""Plots for Figure 1 of the micro-sam2 paper.

Panel a shows the architecture of micro-sam2: a shared image encoder, the prompt encoder, memory attention and mask
decoder for interactive segmentation, and the segmentation decoder for automatic segmentation.
Panel b shows the diversity of the training data: example crops of the datasets, grouped by domain, with their
annotations. It reads the crops from the training data.
Panel c shows the tasks that micro-sam2 supports around its hub: interactive analysis (segmentation in 2D and 3D,
tracking in 2D+T, pixel and object classification, with the prompts that the tools support) on the left, and
automatic analysis (segmentation in 2D and 3D, tracking in 2D+T and 3D+T, batch processing) on the right.
Panel d shows examples from the user studies as studio renders: the objects in 3D with distinct colours for close
objects, on a light grey floor with walls and shadows (the expanded Platynereis nuclei, the MCF7 spheroid nuclei and
the cells of a heart section). The renders run off screen with pyvista on a GPU node (--render) and are saved in
panel_d of the output folder; panel d puts the examples together from them, in rows.
The panels and the whole figure are saved as svg, pdf and png.
"""

import io
import os
import re
import glob
import json
import time
import colorsys
import base64
import argparse
import subprocess
import importlib.util
import urllib.request
from functools import lru_cache

import zarr
import tifffile
import numpy as np
import vtk
import matplotlib
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import imageio.v3 as imageio
from matplotlib.path import Path
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline
from skimage.color import rgb2lab
from skimage.segmentation import find_boundaries
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, Polygon

from elf.io import open_file

from micro_sam.v2.transforms.labels import DirectedPerObjectBoundaryDistanceTransform


PNG_DPI = 600
# the fonts that render the text: Liberation Sans has the widths of Arial (see plot_style.py)
FONT_DIR = os.path.expanduser("~/.local/share/fonts/liberation")
# the smallest and the largest text in the figure, 5 pt and 7 pt in mm, as Nature Methods asks
MIN_TEXT, MAX_TEXT = 5 * 25.4 / 72, 7 * 25.4 / 72
# the layout of the figure in mm: panel a on top, panels b and c below it; the letters sit in a column on the left,
# and panel c is scaled down to fit next to panel b
FIGURE_W, LETTER_W, ROW_GAP, PANEL_GAP, LETTER_SIZE = 183.0, 2.8, 3.5, 2.0, 8 * 25.4 / 72
# the height of the figure: an A4 page with the same margins as the width leaves (13.5 mm)
FIGURE_H = 270.0
SIZE = 8.5
STACK_FRAME, STACK_OFFSET = 7.4, 1.4
ITEM_GAP = 1.5

BLUE = "#3d4596"
PURPLE = "#8e4c9f"
RIM = "#b49ad6"
# the object segmented from the box prompt in the 2D output of the interactive band
BOX_OBJECT = "#e0a030"
TEAL = "#38b2c4"
TEAL_DARK = "#2a8f9c"
ORANGE = "#ef6f45"
TEXT = "#212427"
SUBTEXT = "#55505e"
INTERACTIVE_BACKGROUND = "#e7dcdd"
AUTOMATIC_BACKGROUND = "#dbe3e0"
FONT = "font-family:Arial,'Liberation Sans',sans-serif"

# a cell body among the neurites of the EM block, and its nucleus, as spheres in the unit cube
SOMA = ((0.24, 0.6, 0.05), 0.3)
NUCLEUS = ((0.21, 0.62, 0.05), 0.17)
N_AXIAL, N_LATERAL = 34, 4
SOMA_LABEL, FEATURED_LABEL = N_AXIAL + N_LATERAL, N_AXIAL + N_LATERAL + 1

# the logo of micro-sam, taken from the documentation of this repository
REPOSITORY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4))
# the μSAM2 logo (the horizontal light version, 901 x 381 px, from the user, 2026-11-10)
LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "micro_sam2_logo.png")
# the official Python logo, which is downloaded once into the assets folder next to this script
PYTHON_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "python_logo.svg")
PYTHON_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg"
NAPARI_LOGO = os.path.join(
    importlib.util.find_spec("napari").submodule_search_locations[0], "resources", "logos", "gradient-plain-light.svg"
)


def fmt(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


@lru_cache
def font(bold):
    name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), 1000)


def text_width(text, size, bold=False):
    """Measure the width of a text in mm, for a font size in mm."""
    return font(bold).getlength(text) / 1000 * size


def blob_points(cx, cy, r, aspect, angle, seed, n=28):
    """Get the points of a slightly irregular closed outline."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    noise = sum(rng.normal(0, 0.07 / k) * np.sin(k * t + rng.uniform(0, 2 * np.pi)) for k in (2, 3, 4))
    rad = r * (1 + noise)
    px, py = rad * np.cos(t) * aspect, rad * np.sin(t) / aspect
    ca, sa = np.cos(angle), np.sin(angle)
    return np.stack([cx + ca * px - sa * py, cy + sa * px + ca * py], axis=1)


def blob_path(cx, cy, r, aspect, angle, seed, n=28):
    """Get a smooth, slightly irregular closed outline as a cubic bezier path."""
    pts = blob_points(cx, cy, r, aspect, angle, seed, n)
    d = f"M{fmt(pts[0, 0])},{fmt(pts[0, 1])}"
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
        d += f"C{fmt(c1[0])},{fmt(c1[1])} {fmt(c2[0])},{fmt(c2[1])} {fmt(p2[0])},{fmt(p2[1])}"
    return d + "Z"


class Drawing:
    """Collect the defs and the elements of an svg drawing."""

    def __init__(self, text_scale=1.0, scale=1.0):
        # the text scale enlarges the main text, from labels upwards, and leaves the small annotations as they are;
        # the scale maps the coordinates to mm in the figure, so that all text is between MIN_TEXT and MAX_TEXT there
        self.defs, self.body, self.count, self.text_scale, self.scale = [], [], 0, text_scale, scale

    def uid(self, prefix):
        self.count += 1
        return f"f1c-{prefix}-{self.count}"

    def add(self, element):
        self.body.append(element)

    def text_size(self, size):
        """Get the size that a text of this size gets: enlarged by the text scale, and between MIN_TEXT and MAX_TEXT in
        print."""
        size = size * self.text_scale if size >= 1.5 else size
        return min(max(size, MIN_TEXT / self.scale), MAX_TEXT / self.scale)

    def text(self, x, y, s, size, weight="normal", color=TEXT, anchor="middle", rotate=None):
        size = self.text_size(size)
        transform = f' transform="rotate({rotate},{fmt(x)},{fmt(y)})"' if rotate is not None else ""
        self.add(
            f'<text x="{fmt(x)}" y="{fmt(y)}"{transform} style="{FONT};font-size:{fmt(size)}px;'
            f'font-weight:{weight};fill:{color};text-anchor:{anchor}">{s}</text>'
        )

    def linear_gradient(self, x1, y1, x2, y2, start=BLUE, stop=PURPLE):
        gid = self.uid("grad")
        self.defs.append(
            f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{fmt(x1)}" y1="{fmt(y1)}" '
            f'x2="{fmt(x2)}" y2="{fmt(y2)}"><stop offset="0" stop-color="{start}"/>'
            f'<stop offset="1" stop-color="{stop}"/></linearGradient>'
        )
        return gid

    def clip_rect(self, x, y, w, h, rx):
        cid = self.uid("clip")
        self.defs.append(
            f'<clipPath id="{cid}"><rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
            f'rx="{fmt(rx)}"/></clipPath>'
        )
        return cid


def shared_defs():
    return (
        '<linearGradient id="f1c-tile" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#c6cacc"/><stop offset="1" stop-color="#a9aeb1"/></linearGradient>'
        '<linearGradient id="f1c-cell" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#5d6165"/><stop offset="1" stop-color="#8b8f93"/></linearGradient>'
        f'<linearGradient id="f1c-mask" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{BLUE}"/>'
        f'<stop offset="1" stop-color="{PURPLE}"/></linearGradient>'
        f'<linearGradient id="f1c-mask-teal" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{TEAL_DARK}"/>'
        '<stop offset="1" stop-color="#7fd0c4"/></linearGradient>'
        '<linearGradient id="f1c-mask-pink" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#b8457d"/>'
        '<stop offset="1" stop-color="#f19cc3"/></linearGradient>'
        '<linearGradient id="f1c-mask-blue" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2f62a8"/>'
        '<stop offset="1" stop-color="#86b4e8"/></linearGradient>'
        '<linearGradient id="f1c-mask-orange" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#d4582c"/>'
        '<stop offset="1" stop-color="#f6a77f"/></linearGradient>'
        '<linearGradient id="f1c-mask-gold" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#c9962c"/>'
        '<stop offset="1" stop-color="#f3d98a"/></linearGradient>'
    )


def point_prompt(drawing, x, y, positive=True, r=0.75):
    color = TEAL if positive else ORANGE
    s = r * 0.5
    sign = f"M{fmt(x - s)},{fmt(y)}H{fmt(x + s)}"
    if positive:
        sign += f"M{fmt(x)},{fmt(y - s)}V{fmt(y + s)}"
    drawing.add(
        f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}" fill="{color}" stroke="#ffffff" '
        f'stroke-width="{fmt(0.16 * r)}"/>'
        f'<path d="{sign}" stroke="#ffffff" stroke-width="{fmt(0.35 * r)}" stroke-linecap="round"/>'
    )


def in_sphere(x, y, z, sphere):
    (cx, cy, cz), r = sphere
    return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r ** 2


def neurite_labels(x, y, z, seed, n_axial=N_AXIAL, n_lateral=N_LATERAL, soma=None, featured=False):
    """Label points of the unit cube by their closest neurite; most neurites run along z, a few thin ones along x.

    A soma, given as a sphere, gets the label after the last neurite. The featured neurite starts near the center of
    the front face and bends up to run just below the top face, so that it trails along the top face.
    """
    rng = np.random.default_rng(seed)
    # a smooth warp of space makes the membranes wavy
    warped = []
    for coordinate in (x, y, z):
        offset = 0.0
        for _ in range(3):
            f, phase = rng.normal(0, 7, 3), rng.uniform(0, 2 * np.pi)
            offset = offset + 0.018 * np.sin(f[0] * x + f[1] * y + f[2] * z + phase)
        warped.append(coordinate + offset)
    x, y, z = warped
    distances = []
    for _ in range(n_axial):
        c, v, w = rng.uniform(-0.1, 1.1, 2), rng.normal(0, 0.25, 2), rng.uniform(0, 0.05)
        distances.append(np.hypot(x - c[0] - v[0] * (z - 0.5), y - c[1] - v[1] * (z - 0.5)) - w)
    for _ in range(n_lateral):
        c, v = rng.uniform(0.1, 0.9, 2), rng.normal(0, 0.15, 2)
        distances.append(1.8 * np.hypot(y - c[0] - v[0] * (x - 0.5), z - c[1] - v[1] * (x - 0.5)))
    if featured:
        distances.append(np.full_like(x, np.inf))
        center_x, center_y = 0.62 + 0.1 * z, 0.0 + 0.48 * np.exp(-z / 0.16)
        distances.append(np.hypot(x - center_x, y - center_y) - 0.12)
    labels = np.argmin(distances, axis=0)
    if soma is not None:
        labels = np.where(in_sphere(x, y, z, soma), n_axial + n_lateral, labels)
    return labels


def face_coordinates(face, n):
    """Get the cube coordinates of the pixels of a visible face; u runs along the columns and v along the rows."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    if face == "front":
        return u, v, np.zeros_like(u)
    if face == "top":
        return u, np.zeros_like(u), 1 - v
    return np.ones_like(u), v, u


def membranes(labels):
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    return ndimage.binary_dilation(boundary, iterations=1)


def stamp(canvas, points, radius):
    """Mark the pixels around a dense list of (row, col) points."""
    n = canvas.shape[0]
    for r, c in points:
        r0, r1 = int(max(r - radius, 0)), int(min(r + radius + 1, n))
        c0, c1 = int(max(c - radius, 0)), int(min(c + radius + 1, n))
        canvas[r0:r1, c0:c1] = True


def arc_points(center, radius, start, span, rng, wobble=1.2, step=0.4):
    angles = np.arange(start, start + span, step / radius)
    rad = radius + wobble * np.sin(angles * rng.uniform(5, 9) + rng.uniform(0, 2 * np.pi))
    return np.stack([center[0] + rad * np.sin(angles), center[1] + rad * np.cos(angles)], axis=1)


def cell_body_organelles(image, rng, soma, nucleus):
    """Draw the nucleus, rough ER, a Golgi stack and lysosomes of a cell body into an EM section."""
    n = image.shape[0]
    rows, cols = np.mgrid[:n, :n]
    if nucleus.any():
        depth = ndimage.distance_transform_edt(nucleus)
        chromatin = ndimage.gaussian_filter(rng.normal(0, 1, image.shape), 2.0)
        chromatin /= chromatin.std()
        image[nucleus] = 0.56 + 0.05 * chromatin[nucleus]
        # heterochromatin lines the envelope in patches and forms a few clumps inside
        image[nucleus & (((depth < 5) & (chromatin > -0.3)) | (chromatin > 1.6))] = 0.34
        image[nucleus & (depth <= 1.4)] = 0.2
        r0, c0 = np.unravel_index(np.argmax(depth), depth.shape)
        nucleolus = np.hypot(rows - r0 - 0.2 * depth.max(), cols - c0 + 0.25 * depth.max()) < 0.28 * depth.max()
        image[nucleolus] = np.where(chromatin[nucleolus] > 0, 0.22, 0.3)
        center, radius = (rows[nucleus].mean(), cols[nucleus].mean()), np.sqrt(nucleus.sum() / np.pi)
    else:
        center, radius = (rows[soma].mean(), cols[soma].mean()), 0.3 * np.sqrt(soma.sum() / np.pi)
    cytoplasm = soma & ~ndimage.binary_dilation(nucleus, iterations=3) & ndimage.binary_erosion(soma, iterations=3)
    image[cytoplasm] -= 0.05
    er, ribosomes = np.zeros(image.shape, dtype=bool), np.zeros(image.shape, dtype=bool)
    for _ in range(9):
        r, start, span = radius + rng.uniform(7, 30), rng.uniform(0, 2 * np.pi), rng.uniform(0.4, 0.9)
        for offset in (0.0, 2.6):
            points = arc_points(center, r + offset, start, span, rng)
            stamp(er, points, 1)
            stamp(ribosomes, points[rng.random(len(points)) < 0.15] + rng.normal(0, 1.8, 2), 1)
    image[ribosomes & cytoplasm] = 0.3
    image[er & cytoplasm] = 0.24
    golgi, start = np.zeros(image.shape, dtype=bool), rng.uniform(0, 2 * np.pi)
    for k in range(5):
        stamp(golgi, arc_points(center, radius + 12 + 3.2 * k, start, 0.6 - 0.06 * k, rng, wobble=0.3), 1)
    image[ndimage.binary_dilation(golgi) & cytoplasm] = 0.52
    image[golgi & cytoplasm] = 0.24
    for k in (-1, 1):
        angle = start + (0.3 + 0.12 * k) + k * 0.35
        for d in range(3):
            distance = radius + 14 + 3 * d
            pr, pc = center[0] + distance * np.sin(angle), center[1] + distance * np.cos(angle)
            image[(np.hypot(rows - pr, cols - pc) < 2.0) & cytoplasm] = 0.28
    candidates = np.argwhere(ndimage.binary_erosion(cytoplasm, iterations=5))
    for r, c in candidates[rng.integers(len(candidates), size=3)] if len(candidates) else []:
        lysosome = np.hypot(rows - r, cols - c) < rng.uniform(3, 4.5)
        image[lysosome] = 0.28
        image[lysosome & (rng.random(image.shape) < 0.15)] = 0.5
    return image


def em_texture(labels, boundary, seed, n_mitochondria=9, soma=None, nucleus=None, mitochondria=None, vesicles=None):
    """Simulate an EM section: neurite profiles with dark membranes, a cell body, mitochondria, vesicles and grain.

    The mitochondria and the clusters of vesicles are placed at random, unless masks of their sections are given.
    """
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    image = rng.uniform(0.6, 0.74, labels.max() + 1)[labels]
    nucleus = np.zeros(labels.shape, dtype=bool) if nucleus is None else nucleus
    if soma is not None and soma.any():
        image = cell_body_organelles(image, rng, soma, nucleus)
    rows, cols = np.mgrid[:n, :n]
    if mitochondria is not None:
        mito = mitochondria & ~boundary
        cristae = np.sin((0.8 * rows + 0.6 * cols) * 1.3) > 0.55
        image[mito] = np.where(cristae[mito], 0.3, 0.42)
        image[mito & ~ndimage.binary_erosion(mito, iterations=1)] = 0.22
        n_mitochondria = 0
    for _ in range(n_mitochondria):
        cy, cx = rng.integers(10, n - 10, 2)
        a, b, angle = rng.uniform(7, 15), rng.uniform(3, 5.5), rng.uniform(0, np.pi)
        du, dv = cols - cx, rows - cy
        ru, rv = du * np.cos(angle) + dv * np.sin(angle), -du * np.sin(angle) + dv * np.cos(angle)
        mito = ((ru / a) ** 2 + (rv / b) ** 2 < 1) & (labels == labels[cy, cx]) & ~boundary
        mito &= ~ndimage.binary_dilation(nucleus, iterations=3)
        cristae = np.sin(ru * 1.3) > 0.55
        image[mito] = np.where(cristae[mito], 0.3, 0.42)
        image[mito & ~ndimage.binary_erosion(mito, iterations=1)] = 0.22
    if vesicles is not None:
        image[vesicles & ~boundary] = 0.3
    for _ in range(0 if vesicles is not None else 7):
        center = rng.integers(8, n - 8, 2)
        for dy, dx in rng.normal(0, 3.0, (8, 2)):
            image[np.hypot(rows - center[0] - dy, cols - center[1] - dx) < 1.3] = 0.34
    image[boundary] = 0.2
    image = ndimage.gaussian_filter(image, 0.9) + rng.normal(0, 0.045, image.shape)
    image += 0.05 * ndimage.gaussian_filter(rng.normal(0, 1, image.shape), n / 8) * n / 8
    return np.clip(ndimage.gaussian_filter(image, 0.5), 0, 1)


def hex_to_rgb(color):
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


# the images are drawn at a small size in pixels; they are upsampled to this size of their longer side before they are
# embedded, so that they do not look pixelated in print
IMAGE_SIZE = 360
JPEG_QUALITY = 92


def upscale(rgb, factor, step=0.25):
    """Upsample an rgb image by an integer factor without pixel steps.

    Strong color steps, like the edges of masks and labels, are taken from a median filtered nearest neighbour upscale,
    which rounds their staircase into smooth contours. Everything else, like the texture of the images, is taken from a
    cubic upscale. Steps in any channel larger than 'step' count as strong.
    """
    rgb = np.clip(np.asarray(rgb, dtype=float), 0, 1)
    cubic = np.clip(ndimage.zoom(rgb, (factor, factor, 1), order=3), 0, 1)
    smooth = np.repeat(np.repeat(rgb, factor, 0), factor, 1)
    for _ in range(3):
        smooth = ndimage.median_filter(smooth, size=(factor + 1, factor + 1, 1))
    contrast = np.max([
        ndimage.maximum_filter(rgb[..., c], 2) - ndimage.minimum_filter(rgb[..., c], 2) for c in range(3)
    ], axis=0)
    weight = ndimage.maximum_filter(np.clip((contrast - step) / step, 0, 1), 3)
    weight = np.clip(ndimage.zoom(weight, factor, order=1), 0, 1)[..., None]
    return weight * smooth + (1 - weight) * cubic


def array_uri(rgb):
    buffer = io.BytesIO()
    rgb = np.clip(rgb, 0, 1)
    factor = int(np.ceil(IMAGE_SIZE / max(rgb.shape[:2])))
    # the median filter is slow at large factors, so the images are upsampled by at most 4 with it, and the rest of
    # the way with Lanczos interpolation
    if factor > 1:
        rgb = upscale(rgb, min(factor, 4))
    image = Image.fromarray((rgb * 255).astype("uint8"))
    if factor > 4:
        size = (rgb.shape[1] * factor // 4, rgb.shape[0] * factor // 4)
        image = image.resize(size, Image.LANCZOS)
    # the images have no transparency, so a high quality jpeg keeps the files small
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def em_faces(seed, overlay, n=220):
    """Render the three visible faces of a simulated EM block; overlay maps the labels and u, v to colors."""
    faces = {}
    for k, (face, shade) in enumerate((("front", 1.0), ("top", 1.12), ("side", 0.82))):
        x, y, z = face_coordinates(face, n)
        labels = neurite_labels(x, y, z, seed, soma=SOMA, featured=True)
        boundary = membranes(labels)
        soma, nucleus = labels == SOMA_LABEL, in_sphere(x, y, z, NUCLEUS)
        gray = em_texture(labels, boundary, seed + 17 * (k + 1), n_mitochondria=12, soma=soma, nucleus=nucleus)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        color, alpha = overlay(labels, u, v)
        alpha = np.where(boundary, alpha * 0.3, alpha)[..., None]
        rgb = (1 - alpha) * rgb + alpha * color * (0.55 + 0.6 * gray[..., None])
        faces[face] = (rgb * shade, labels)
    return faces


def em_block(drawing, x, y, faces, front=8.5, depth=3.5):
    """Draw an oblique EM block whose back top edge starts at (x + depth, y)."""
    fx, fy, d = x, y + depth, depth
    matrices = {
        "front": (front, 0, 0, front, fx, fy), "top": (front, 0, -d, d, fx + d, fy - d),
        "side": (d, -d, 0, front, fx + front, fy),
    }
    for face, matrix in matrices.items():
        drawing.add(
            f'<image width="1" height="1" preserveAspectRatio="none" transform="matrix({",".join(map(fmt, matrix))})" '
            f'xlink:href="{array_uri(faces[face][0])}"/>'
        )
    outline = (
        f"M{fmt(fx)},{fmt(fy)}h{fmt(front)}v{fmt(front)}h{fmt(-front)}Z"
        f"M{fmt(fx)},{fmt(fy)}l{fmt(d)},{fmt(-d)}h{fmt(front)}l{fmt(-d)},{fmt(d)}"
        f"M{fmt(fx + front + d)},{fmt(fy - d)}v{fmt(front)}l{fmt(-d)},{fmt(d)}"
    )
    drawing.add(f'<path d="{outline}" fill="none" stroke="#2d2b36" stroke-width="0.18" stroke-linejoin="round"/>')
    return fx, fy


def neurite_colors(n_labels):
    palette = [
        "#4f5bd5", "#9b59b6", "#38b2c4", "#e0679b", "#7fc97f", "#f0a35e", "#6c8ebf", "#c77dd6", "#2a9d8f",
        "#d4a5e8", "#5fb3e8", "#e8c547", "#8e4c9f", "#57c49a",
    ]
    return np.array([hex_to_rgb(palette[k % len(palette)]) for k in range(n_labels)])


EM_SEED = 9
# the mother cell (index 3) is blue before the division, its daughters (indices 3 and 4) are pink and orange
TRACK_COLORS = {0: "f1c-mask", 1: "f1c-mask-teal", 2: "f1c-mask-gold"}
MOTHER_COLOR, DAUGHTER_COLORS = "f1c-mask-blue", ("f1c-mask-pink", "f1c-mask-orange")


def lineage_colors(t, tracked):
    """Get the mask fill of each tracked cell in frame t; frame 0 is the last time point, after the division."""
    colors = {k: TRACK_COLORS[k] for k in tracked if k in TRACK_COLORS}
    if t == 0:
        colors.update({3: DAUGHTER_COLORS[0], 4: DAUGHTER_COLORS[1]})
    else:
        colors[3] = MOTHER_COLOR
    return colors


def frames_2dt():
    """Get the nuclei of the 2D+T example, front to back; the front frame is the last time point."""
    frames = []
    for t in range(3):
        if t == 0:
            cells = [
                (0.3, 0.3, 0.14, 1.1, 0.6), (0.78, 0.3, 0.13, 1.05, 2.0), (0.25, 0.8, 0.13, 1.1, 1.4),
                (0.62, 0.68, 0.1, 1.1, 0.4), (0.84, 0.86, 0.1, 1.1, 0.9),
            ]
        else:
            # the mother nucleus condenses and elongates before it divides
            cells = [
                (0.3 - 0.05 * t, 0.3 + 0.04 * t, 0.14, 1.1, 0.6), (0.78, 0.3 - 0.05 * t, 0.13, 1.05, 2.0),
                (0.25 + 0.04 * t, 0.8, 0.13, 1.1, 1.4), (0.72, 0.76, 0.13, 1.1 + 0.2 * (t == 1), 0.7),
            ]
        frames.append(cells)
    return frames


# the label free 2D example of panel c, in the style of LIVECell: a dense culture of wild cells, given as (x, y, type)
# in unit coordinates. Spiky cells bristle with filopodia like microglia, spread cells grow branched processes like
# glioma cells, neuron-like cells have long thin neurites, and spindle cells are long with spiky tips.
LIVECELL_CELLS = [
    (0.15, 0.17, "spread"), (0.46, 0.12, "spiky"), (0.8, 0.14, "neuron"), (0.12, 0.52, "spiky"),
    (0.4, 0.44, "spread"), (0.7, 0.45, "spiky"), (0.93, 0.52, "spindle"), (0.22, 0.83, "neuron"),
    (0.5, 0.76, "spiky"), (0.77, 0.8, "spread"), (0.96, 0.95, "spiky"), (0.55, 0.98, "spindle"),
]
# the cell that the box prompt encloses, the cell whose body holds the positive point prompt, and the neighbouring
# cell that a negative point prompt keeps out of its mask
LIVECELL_BOX, LIVECELL_POINT, LIVECELL_NEGATIVE = 5, 7, 8


def stamp_disks(labels, label, points, radii):
    """Mark the pixels within the radii of points in unit coordinates, without overwriting other cells."""
    n = labels.shape[0]
    for (px, py), r in zip(np.asarray(points) * n, np.asarray(radii) * n):
        r = max(r, 0.7)
        r0, r1 = int(max(py - r, 0)), int(min(py + r + 1, n))
        c0, c1 = int(max(px - r, 0)), int(min(px + r + 1, n))
        if r1 <= r0 or c1 <= c0:
            continue
        rows, cols = np.mgrid[r0:r1, c0:c1]
        block = labels[r0:r1, c0:c1]
        block[(np.hypot(cols - px, rows - py) <= r) & (block == 0)] = label


def branch(rng, start, heading, length, radius, depth, out):
    """Grow a process with a wandering path and a tapering radius, which may fork into two thinner branches."""
    steps = int(length * 500) + 8
    turns = np.cumsum(rng.normal(0, 0.09, steps))
    angles = heading + turns
    points = start + np.cumsum(np.stack([np.cos(angles), np.sin(angles)], axis=1) * length / steps, axis=0)
    radii = np.linspace(radius, 0.4 * radius, steps)
    out.append((points, radii))
    if depth > 0 and rng.uniform() < 0.85:
        for side in (-1, 1):
            spread = side * rng.uniform(0.35, 0.8)
            branch(rng, points[-1], angles[-1] + spread, 0.6 * length, 0.55 * radius, depth - 1, out)
    elif depth == 0 and rng.uniform() < 0.5:
        # a ruffled lamellipodium at the tip
        tip = points[-1]
        out.append((tip + rng.normal(0, 0.008, (6, 2)), np.full(6, 2.2 * radius)))


def wild_cell(rng, cx, cy, kind):
    """Describe one cell by the outline of its body, its nucleus and the strokes of its processes."""
    strokes = []
    if kind == "spindle":
        angle = rng.uniform(0, np.pi)
        t = np.linspace(-1, 1, 200)[:, None]
        axis = np.array([np.cos(angle), np.sin(angle)])
        bend = np.array([-axis[1], axis[0]]) * 0.03 * np.sin(np.pi * t)
        points = np.array([cx, cy]) + 0.16 * t * axis + bend
        strokes.append((points, 0.04 * (1 - np.abs(t[:, 0])) ** 0.7 + 0.003))
        body = (cx, cy, 0.03, 1.6, angle)
        tips = [points[0], points[-1]]
        spikes = [(tip, angle + np.pi * (k == 0)) for k, tip in enumerate(tips)]
    else:
        size = {"spread": 0.075, "spiky": 0.05, "neuron": 0.04}[kind]
        aspect = rng.uniform(1.0, 1.5) if kind != "spiky" else rng.uniform(1.0, 1.25)
        body = (cx, cy, size, aspect, rng.uniform(0, np.pi))
        spikes = []
        if kind == "spread":
            for heading in rng.uniform(0, 2 * np.pi, rng.integers(3, 6)):
                start = np.array([cx, cy]) + 0.8 * size * np.array([np.cos(heading), np.sin(heading)])
                branch(rng, start, heading, rng.uniform(0.08, 0.16), 0.014, 2, strokes)
        elif kind == "neuron":
            for heading in rng.uniform(0, 2 * np.pi, rng.integers(2, 4)):
                start = np.array([cx, cy]) + 0.8 * size * np.array([np.cos(heading), np.sin(heading)])
                branch(rng, start, heading, rng.uniform(0.18, 0.3), 0.007, 1, strokes)
        else:
            spikes = [(None, heading) for heading in rng.uniform(0, 2 * np.pi, rng.integers(14, 26))]
    # thin straight filopodia, around the whole body or at the tips of a spindle
    for tip, heading in spikes:
        count = 1 if tip is None else 4
        for _ in range(count):
            angle = heading + (0.0 if tip is None else rng.normal(0, 0.5))
            base = tip if tip is not None else np.array([cx, cy]) + 0.85 * body[2] * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            length = rng.uniform(0.02, 0.065)
            t = np.linspace(0, 1, 30)[:, None]
            strokes.append((base + t * length * np.array([np.cos(angle), np.sin(angle)]), np.full(30, 0.0035)))
    nucleus = (cx, cy, 0.5 * body[2], 1.25, body[4])
    return body, nucleus, strokes


@lru_cache
def livecell_scene(n=260):
    """Label the cells of the label free 2D example and render them as a phase contrast image."""
    rng = np.random.default_rng(EM_SEED + 7)
    labels = np.zeros((n, n), dtype="uint32")
    rows, cols = np.mgrid[:n, :n]
    grid = np.stack([cols.ravel() + 0.5, rows.ravel() + 0.5], axis=1) / n
    cells = [wild_cell(rng, cx, cy, kind) for cx, cy, kind in LIVECELL_CELLS]
    nuclei = np.zeros((n, n), dtype=bool)
    # the bodies go first, so that processes grow around their neighbours
    for label, (body, nucleus, _) in enumerate(cells, start=1):
        outline = blob_points(*body, seed=label, n=40)
        mask = Path(outline).contains_points(grid).reshape(n, n)
        labels[mask & (labels == 0)] = label
        nuclei |= Path(blob_points(*nucleus, seed=label + 50)).contains_points(grid).reshape(n, n)
    for label, (_, _, strokes) in enumerate(cells, start=1):
        for points, radii in strokes:
            stamp_disks(labels, label, points, radii)
    nuclei &= labels > 0
    # phase contrast: an even gray background, grainy cells with a dark rim and a bright halo, and nuclei with bright
    # nucleoli
    inside = labels > 0
    image = 0.58 + 0.03 * ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 12) * 12
    grain = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 1.2)
    image = np.where(inside, 0.4 + 0.12 * grain, image)
    image[nuclei] = 0.33 + 0.05 * grain[nuclei]
    spots = ndimage.binary_erosion(nuclei, iterations=3) & (rng.uniform(size=(n, n)) < 0.012)
    image[ndimage.binary_dilation(spots, iterations=1)] = 0.62
    halo = ndimage.binary_dilation(inside, iterations=4) & ~inside
    image = image + 0.42 * ndimage.gaussian_filter(halo.astype(float), 1.6)
    image[inside & ~ndimage.binary_erosion(inside, iterations=1)] = 0.2
    image = ndimage.gaussian_filter(image, 0.6) + rng.normal(0, 0.02, (n, n))
    gray = np.clip(image, 0, 1)
    return labels, np.repeat(gray[..., None], 3, axis=-1)


def livecell_tile(drawing, x, y, rgb):
    clip = drawing.clip_rect(x, y, SIZE, SIZE, 0.9)
    drawing.add(
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(SIZE)}" height="{fmt(SIZE)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(rgb)}"/>'
    )


def interactive_2d(drawing, x, cy):
    """A spiky cell segmented from a box prompt, a neuron-like cell with its neurites from a positive point prompt,
    and a negative point prompt on the neighbouring cell."""
    y = cy - SIZE / 2
    labels, image = livecell_scene()
    n = labels.shape[0]
    box, point = LIVECELL_BOX + 1, LIVECELL_POINT + 1
    livecell_tile(drawing, x, y, overlay(image, [labels == box, labels == point]))
    rows, cols = np.nonzero(labels == box)
    x0, y0 = x + cols.min() / n * SIZE - 0.3, y + rows.min() / n * SIZE - 0.3
    x1, y1 = x + cols.max() / n * SIZE + 0.3, y + rows.max() / n * SIZE + 0.3
    drawing.add(
        f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" height="{fmt(y1 - y0)}" rx="0.15" fill="none" '
        'stroke="#ffffff" stroke-width="0.22"/>'
    )
    cx, cy_point, _ = LIVECELL_CELLS[LIVECELL_POINT]
    point_prompt(drawing, x + cx * SIZE, y + cy_point * SIZE, r=0.5)
    cx, cy_point, _ = LIVECELL_CELLS[LIVECELL_NEGATIVE]
    point_prompt(drawing, x + cx * SIZE, y + cy_point * SIZE, positive=False, r=0.5)


def interactive_3d(drawing, x, cy, scribble=True, masked=True):
    """A neuron segmented from a point prompt, and optionally the larger cell body segmented from a scribble."""
    blue, purple = hex_to_rgb(BLUE), hex_to_rgb(PURPLE)
    teal_dark, teal_light = hex_to_rgb(TEAL_DARK), hex_to_rgb("#7fd0c4")

    def overlay(labels, u, v):
        g = ((u + v) / 2)[..., None]
        neuron, soma = (1 - g) * blue + g * purple, (1 - g) * teal_dark + g * teal_light
        color = np.where((labels == SOMA_LABEL)[..., None], soma, neuron)
        segmented = (labels == FEATURED_LABEL) | ((labels == SOMA_LABEL) & scribble)
        return color, np.where(segmented & masked, 0.75, 0.0)

    faces = em_faces(seed=EM_SEED, overlay=overlay)
    fx, fy = em_block(drawing, x, cy - (SIZE + 3.5) / 2, faces)
    labels = faces["front"][1]
    n = labels.shape[0]
    rows, cols = np.nonzero(labels == FEATURED_LABEL)
    point_prompt(drawing, fx + cols.mean() / n * SIZE, fy + rows.mean() / n * SIZE, r=0.62)
    if not scribble:
        return
    region = ndimage.binary_erosion(labels == SOMA_LABEL, iterations=int(0.06 * n))
    points = freehand_stroke(region, np.random.default_rng(2), length=int(0.45 * n))
    if points is not None:
        xs, ys = fx + points[:, 1] / n * SIZE, fy + points[:, 0] / n * SIZE
        brush_stroke(drawing, xs, ys, "#ffffff", width=0.46)
        brush_stroke(drawing, xs, ys, TEAL, width=0.3)


def frame_stack(drawing, x, cy, frames, seed):
    """Draw frames as a stack; frames go from front to back and hold (cells, masked, mask fills).

    Returns the top left corner of each frame.
    """
    size, offset = STACK_FRAME, STACK_OFFSET
    top = cy - (size + 2 * offset) / 2
    origins = [(x + offset * t, top + offset * (len(frames) - 1 - t)) for t in range(len(frames))]
    for t in reversed(range(len(frames))):
        cells, masked, fills = frames[t]
        nucleus_frame(drawing, *origins[t], size, cells, masked, fills, seed + t, mitotic=(t == 1,))
    return origins


def draw_tracks(drawing, tracks, split=None):
    """Draw tracks through time on top of the frames; tracks are (points, color) and split marks a division."""
    parts = []
    for points, color in tracks:
        path = "M" + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in points)
        parts.append(
            f'<path d="{path}" fill="none" stroke="#ffffff" stroke-width="0.3" stroke-linecap="round" '
            'stroke-linejoin="round" stroke-opacity="0.35"/>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="0.18" stroke-linecap="round" '
            'stroke-linejoin="round" stroke-dasharray="0.45,0.35"/>'
        )
    for points, color in tracks:
        for px, py in points:
            parts.append(
                f'<circle cx="{fmt(px)}" cy="{fmt(py)}" r="0.16" fill="{color}" stroke="#ffffff" stroke-width="0.07"/>'
            )
    if split is not None:
        parts.append(
            f'<circle cx="{fmt(split[0])}" cy="{fmt(split[1])}" r="0.28" fill="{ORANGE}" stroke="#ffffff" '
            'stroke-width="0.1"/>'
        )
    drawing.add("".join(parts))


def tracks_2dt(origins, tracked):
    """Get the tracks of the 2D+T example through the frames, oldest first, and the point of the division."""
    frames, s = frames_2dt(), STACK_FRAME

    def position(t, k):
        return origins[t][0] + frames[t][k][0] * s, origins[t][1] + frames[t][k][1] * s

    tracks = [([position(t, k) for t in (2, 1, 0)], MASK_EDGES[TRACK_COLORS[k]]) for k in tracked]
    split = position(1, 3)
    tracks.append(([position(2, 3), split], MASK_EDGES[MOTHER_COLOR]))
    for k, color in zip((3, 4), DAUGHTER_COLORS):
        tracks.append(([split, position(0, k)], MASK_EDGES[color]))
    return tracks, split


def interactive_2dt(drawing, x, cy):
    # a prompted nucleus is followed through time, and a prompted mother nucleus is followed through its division
    frames = []
    for t, cells in enumerate(frames_2dt()):
        colors = lineage_colors(t, tracked=(1,))
        frames.append((cells, list(colors), colors))
    origins = frame_stack(drawing, x, cy, frames, seed=81)
    draw_tracks(drawing, *tracks_2dt(origins, tracked=(1,)))
    fx, fy = origins[0]
    point_prompt(drawing, fx + 0.78 * STACK_FRAME, fy + 0.3 * STACK_FRAME, r=0.55)


def split_line(drawing, x, y):
    drawing.add(
        f'<path d="M{fmt(x + SIZE / 2)},{fmt(y)}V{fmt(y + SIZE)}" stroke="#ffffff" stroke-width="0.25" '
        'stroke-dasharray="0.6,0.4"/>'
    )


def freehand_stroke(region, rng, length, step=1.0, tries=300):
    """Simulate a hand drawn stroke: the longest gently curved arc with a little jitter that stays inside a region."""
    candidates = np.argwhere(region)
    if len(candidates) < 20:
        return None
    best = None
    for _ in range(tries):
        point = candidates[rng.integers(len(candidates))].astype(float)
        heading, curvature = rng.uniform(0, 2 * np.pi), rng.normal(0, 0.012)
        points = [point]
        for _ in range(length):
            heading += curvature
            point = point + step * np.array([np.sin(heading), np.cos(heading)])
            r, c = int(round(point[0])), int(round(point[1]))
            if not (0 <= r < region.shape[0] and 0 <= c < region.shape[1] and region[r, c]):
                break
            points.append(point)
        if best is None or len(points) > len(best):
            best = points
    if len(best) < 10:
        return None
    jitter = ndimage.gaussian_filter1d(rng.normal(0, 1.2, (len(best), 2)), 4, axis=0)
    return np.array(best) + jitter


def brush_stroke(drawing, xs, ys, color, width=0.24):
    path = "M" + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in zip(xs[::2], ys[::2]))
    drawing.add(
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{fmt(width)}" stroke-opacity="0.85" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def stamp_nuclei(image, mask, rng, centers, size, elongation, angles=None):
    """Stamp hematoxylin stained nuclei with some chromatin texture at the given centers inside a mask."""
    n = image.shape[0]
    for k, (r, c) in enumerate(centers):
        a = rng.uniform(*size)
        angle = rng.uniform(0, np.pi) if angles is None else angles[k]
        extent = int(np.ceil(a * elongation)) + 2
        r0, r1 = max(int(r) - extent, 0), min(int(r) + extent, n)
        c0, c1 = max(int(c) - extent, 0), min(int(c) + extent, n)
        rows, cols = np.mgrid[r0:r1, c0:c1]
        du, dv = cols - c, rows - r
        ru, rv = du * np.cos(angle) + dv * np.sin(angle), -du * np.sin(angle) + dv * np.cos(angle)
        nucleus = ((ru / (a * elongation)) ** 2 + (rv / a) ** 2 < 1) & mask[r0:r1, c0:c1]
        tone = rng.uniform(0.75, 1.15) * np.array([0.36, 0.25, 0.55])
        speckle = 1 + 0.25 * rng.normal(0, 1, nucleus.shape)
        patch = image[r0:r1, c0:c1]
        patch[nucleus] = np.clip(tone[None] * speckle[nucleus][:, None], 0, 1)


def he_tissue(seed, n=300):
    """Simulate an H&E stained tissue patch; returns the image and the classes (0 stroma, 1 tumor, 2 fat)."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[:n, :n]
    field = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), n / 12)
    tumor = field > 0.35 * field.std()
    fat_field = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), n / 10)
    fat = (fat_field > 0.5 * fat_field.std()) & ~ndimage.binary_dilation(tumor, iterations=8)
    classes = np.where(tumor, 1, np.where(fat, 2, 0))

    # eosin: stroma with wavy collagen fibres, tumor with paler, granular cytoplasm
    fibres = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), (0.7, 5))
    fibres = ndimage.map_coordinates(
        fibres, [rows + 6 * np.sin(cols / 23.0), cols + 4 * np.sin(rows / 31.0)], order=1, mode="reflect"
    )
    fibres /= fibres.std()
    stroma = np.array([0.94, 0.62, 0.76])[None, None] * (1 + 0.08 * fibres[..., None])
    granules = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 1.2)
    cytoplasm = np.array([0.86, 0.62, 0.8])[None, None] * (1 + 0.6 * granules[..., None])
    image = np.where(tumor[..., None], cytoplasm, stroma)

    # tumor glands: a lumen lined by a ring of crowded, polarized nuclei
    lumens = np.zeros((n, n), dtype=bool)
    candidates = np.argwhere(ndimage.binary_erosion(tumor, iterations=18))
    ring_centers, ring_angles = [], []
    for r, c in candidates[rng.integers(len(candidates), size=min(6, len(candidates)))] if len(candidates) else []:
        a, b = rng.uniform(5, 10), rng.uniform(4, 7)
        lumen = ((rows - r) / b) ** 2 + ((cols - c) / a) ** 2 < 1
        lumens |= lumen
        for t in np.arange(0, 2 * np.pi, 0.5):
            ring_centers.append((r + (b + 3.2) * np.sin(t), c + (a + 3.2) * np.cos(t)))
            ring_angles.append(t)
    image[lumens] = [0.97, 0.93, 0.96]
    stamp_nuclei(image, tumor & ~lumens, rng, ring_centers, (1.8, 2.4), 1.5, ring_angles)
    tumor_points = np.argwhere(tumor & ~ndimage.binary_dilation(lumens, iterations=6))
    stamp_nuclei(image, tumor, rng, tumor_points[rng.integers(len(tumor_points), size=330)], (1.8, 3.0), 1.2)
    stroma_points = np.argwhere(classes == 0)
    stamp_nuclei(image, classes == 0, rng, stroma_points[rng.integers(len(stroma_points), size=70)], (0.8, 1.2), 3.2)

    # adipocytes: large empty cells with thin pink walls
    seeds = rng.uniform(0, n, (55, 2))
    nearest = np.argmin([np.hypot(rows - r, cols - c) for r, c in seeds], axis=0)
    walls = membranes(nearest)
    image[fat] = [0.98, 0.96, 0.98]
    image[fat & walls] = [0.9, 0.6, 0.75]

    # slide appearance: slight blur, uneven staining and noise
    stain = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), n / 6)
    image = ndimage.gaussian_filter(image, (0.6, 0.6, 0)) * (1 + 0.6 * stain[..., None])
    image += rng.normal(0, 0.02, image.shape)
    return np.clip(image, 0, 1), classes


def pixel_classification(drawing, x, cy, n=300):
    """Tissue classification in a histopathology slide: scribbles on the left, the predicted tissue map on the right."""
    y = cy - SIZE / 2
    image, classes = he_tissue(seed=12, n=n)
    colors = np.array([hex_to_rgb(TEAL), hex_to_rgb(ORANGE), hex_to_rgb("#e8c547")])
    # the prediction is a light overlay with outlined classes, so that both halves show the same tissue
    outlines = ndimage.binary_dilation(membranes(classes), iterations=2)[..., None]
    predicted = 0.65 * image + 0.35 * colors[classes]
    predicted = np.where(outlines, colors[classes], predicted)
    image[:, n // 2:] = predicted[:, n // 2:]
    clip = drawing.clip_rect(x, y, SIZE, SIZE, 0.9)
    drawing.add(
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(SIZE)}" height="{fmt(SIZE)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(image)}"/>'
    )
    split_line(drawing, x, y)
    # freehand brush strokes for each tissue class, inside its region in the left half
    rng = np.random.default_rng(4)
    left = np.zeros(classes.shape, dtype=bool)
    left[int(0.05 * n):int(0.95 * n), int(0.05 * n):int(0.45 * n)] = True
    for k, color, n_strokes in ((0, TEAL, 1), (1, ORANGE, 1), (2, "#d9a520", 1)):
        region = ndimage.binary_erosion(classes == k, iterations=6) & left
        for _ in range(n_strokes):
            points = freehand_stroke(region, rng, length=int(0.3 * n))
            if points is not None:
                brush_stroke(drawing, x + points[:, 1] / n * SIZE, y + points[:, 0] / n * SIZE, color)


NUCLEUS_CLASSES = {"tumor": ORANGE, "lymphocyte": PURPLE, "fibroblast": TEAL}


def place_nuclei(rng, region, count, radius, aspect, nuclei, angle=None):
    """Place non-overlapping nuclei, given as fractions of the patch, at random positions inside a region."""
    n = region.shape[0]
    placed = 0
    for _ in range(3000):
        if placed == count:
            break
        cy, cx = rng.uniform(0.04, 0.96, 2)
        r = rng.uniform(*radius)
        if not region[int(cy * n), int(cx * n)]:
            continue
        if any(np.hypot(cx - o[0], cy - o[1]) < (r * aspect + o[2] * o[3]) * 1.05 for o in nuclei):
            continue
        theta = rng.uniform(0, np.pi) if angle is None else angle + rng.normal(0, 0.2)
        nuclei.append((cx, cy, r, aspect * rng.uniform(0.9, 1.15), theta))
        placed += 1


def he_nuclei(seed, n=220):
    """Simulate a zoomed H&E patch with tumor nuclei, lymphocytes and fibroblasts; returns image and nuclei."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[:n, :n]
    # a tumor nest in the upper left, stroma with collagen fibres elsewhere
    boundary = 0.55 + 0.08 * np.sin(cols / n * 7) + 0.05 * np.sin(rows / n * 11)
    tumor = (rows / n + 0.8 * cols / n) < boundary * 1.25
    fibres = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), (0.8, 7))
    fibres = ndimage.rotate(fibres, -25, reshape=False, mode="reflect")
    fibres /= fibres.std()
    stroma = np.array([0.95, 0.66, 0.79])[None, None] * (1 + 0.07 * fibres[..., None])
    granules = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 1.5)
    cytoplasm = np.array([0.87, 0.66, 0.83])[None, None] * (1 + 0.5 * granules[..., None])
    image = np.where(tumor[..., None], cytoplasm, stroma)

    nuclei, classes = [], []
    for name, region, count, radius, aspect, angle in (
        ("tumor", ndimage.binary_erosion(tumor, iterations=8), 11, (0.05, 0.065), 1.2, None),
        ("lymphocyte", ndimage.binary_erosion(~tumor, iterations=6), 9, (0.026, 0.03), 1.05, None),
        ("fibroblast", ndimage.binary_erosion(~tumor, iterations=6), 6, (0.018, 0.022), 2.6, -25 / 180 * np.pi),
    ):
        before = len(nuclei)
        place_nuclei(rng, region, count, radius, aspect, nuclei, angle)
        classes += [name] * (len(nuclei) - before)

    grid = np.stack([cols.ravel() + 0.5, rows.ravel() + 0.5], axis=1)
    hematoxylin = np.array([0.3, 0.2, 0.5])
    for k, ((cx, cy, r, aspect, angle), name) in enumerate(zip(nuclei, classes)):
        mask = Path(blob_points(cx, cy, r, aspect, angle, seed + k) * n).contains_points(grid).reshape(n, n)
        depth = ndimage.distance_transform_edt(mask)
        grain = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 0.8)
        if name == "tumor":
            # vesicular chromatin: a dark rim, a paler center and a prominent nucleolus
            density = 0.55 + 0.35 * (depth < 2.5) + 0.25 * grain / grain.std()
            r0, c0 = np.unravel_index(np.argmax(depth), depth.shape)
            density += 0.6 * (np.hypot(rows - r0 - 1, cols - c0 + 1) < 2.2)
        elif name == "lymphocyte":
            density = 1.15 + 0.12 * grain / grain.std()
        else:
            density = 1.0 + 0.1 * grain / grain.std()
        stain = 1 - np.clip(density, 0, 1.4)[..., None] * (1 - hematoxylin)[None, None] / 1.4 * 1.25
        image = np.where(mask[..., None], np.clip(stain, 0, 1), image)
    stain = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), n / 5)
    image = ndimage.gaussian_filter(image, (0.6, 0.6, 0)) * (1 + 0.5 * stain[..., None])
    image += rng.normal(0, 0.015, image.shape)
    return np.clip(image, 0, 1), nuclei, classes


def object_classification(drawing, x, cy):
    """Nucleus classification in a histopathology slide: tumor nuclei, lymphocytes and fibroblasts."""
    y = cy - SIZE / 2
    seed = 23
    image, nuclei, classes = he_nuclei(seed)
    clip = drawing.clip_rect(x, y, SIZE, SIZE, 0.9)
    parts = [
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(SIZE)}" height="{fmt(SIZE)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(image)}"/>', f'<g clip-path="url(#{clip})">',
    ]
    for k, ((cx, cyy, r, aspect, angle), name) in enumerate(zip(nuclei, classes)):
        path = blob_path(x + cx * SIZE, y + cyy * SIZE, r * SIZE, aspect, angle, seed + k)
        color = NUCLEUS_CLASSES[name]
        parts.append(f'<path d="{path}" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="0.26"/>')
    parts.append("</g>")
    # one nucleus per class that the user labeled by hand
    for name in NUCLEUS_CLASSES:
        k = classes.index(name)
        cx, cyy = nuclei[k][:2]
        parts.append(
            f'<circle cx="{fmt(x + cx * SIZE)}" cy="{fmt(y + cyy * SIZE)}" r="0.42" fill="{NUCLEUS_CLASSES[name]}" '
            'stroke="#ffffff" stroke-width="0.16"/>'
        )
    drawing.add("".join(parts))


def automatic_2d(drawing, x, cy):
    """All cells of the label free 2D example, segmented automatically."""
    labels, image = livecell_scene()
    livecell_tile(drawing, x, cy - SIZE / 2, overlay(image, [labels == k for k in range(1, labels.max() + 1)]))


def automatic_3d(drawing, x, cy):
    colors = neurite_colors(64)
    faces = em_faces(seed=EM_SEED, overlay=lambda labels, u, v: (colors[labels], np.full(labels.shape, 0.6)))
    em_block(drawing, x, cy - (SIZE + 3.5) / 2, faces)


def automatic_2dt(drawing, x, cy):
    frames = []
    for t, cells in enumerate(frames_2dt()):
        colors = lineage_colors(t, tracked=(0, 1, 2))
        frames.append((cells, list(colors), colors))
    origins = frame_stack(drawing, x, cy, frames, seed=81)
    draw_tracks(drawing, *tracks_2dt(origins, tracked=(0, 1, 2)))


# solid colors of the track masks, for their outlines
NUCLEUS_COLORS = {
    "purple": "#7b68d6", "teal": "#34b4c0", "gold": "#e0b43c", "pink": "#e0679b", "blue": "#4f86d6",
    "orange": "#ef7a45",
}
MASK_EDGES = {
    "f1c-mask": "#9a86e0", "f1c-mask-teal": "#5fd0d8", "f1c-mask-gold": "#f3cf63", "f1c-mask-pink": "#f08dbd",
    "f1c-mask-blue": "#7fb2f0", "f1c-mask-orange": "#f69a6c",
}


def nucleus_texture(mask, rng, brightness, condensed=False):
    """Fill a nucleus mask with DNA stain: grainy chromatin, a few bright foci and a brighter rim."""
    grain = ndimage.gaussian_filter(rng.normal(0, 1, mask.shape), 0.9)
    grain /= grain.std()
    foci = np.zeros(mask.shape)
    points = np.argwhere(ndimage.binary_erosion(mask, iterations=3))
    for r, c in points[rng.integers(len(points), size=min(5, len(points)))] if len(points) else []:
        foci[r, c] = 1.0
    foci = ndimage.gaussian_filter(foci, 1.1) * 18
    rim = ndimage.binary_dilation(~mask, iterations=2) & mask
    texture = brightness * (1 + (0.32 if condensed else 0.18) * grain) + 0.35 * foci + 0.08 * rim
    if condensed:
        texture *= 1.35
    return np.where(mask, texture, 0.0)


def stain_rgb(value):
    """Map a DNA stain intensity to the blue-white lookup table of a Hoechst channel."""
    value = np.clip(value, 0, 1)[..., None] ** 0.85
    return np.clip(value * np.array([0.62, 0.74, 1.0]) + np.clip(value - 0.75, 0, 1) * 1.2, 0, 1)


def nucleus_frame(drawing, x, y, size, cells, masked, fills, seed, mitotic=(), n=170, rx=0.8):
    """Draw a fluorescence frame of stained nuclei with outlined, lightly filled track masks."""
    rng = np.random.default_rng(seed)
    image = np.zeros((n, n))
    rows, cols = np.mgrid[:n, :n]
    grid = np.stack([cols.ravel() + 0.5, rows.ravel() + 0.5], axis=1)
    paths = []
    for k, (cx, cy, r, aspect, angle) in enumerate(cells):
        points = blob_points(cx, cy, r, aspect, angle, seed + k)
        mask = Path(points * n).contains_points(grid).reshape(n, n)
        condensed = k == 3 and bool(mitotic) and mitotic[0]
        image += nucleus_texture(mask, rng, rng.uniform(0.42, 0.6), condensed)
        paths.append(blob_path(x + cx * size, y + cy * size, r * size, aspect, angle, seed + k))
    image = ndimage.gaussian_filter(image, 0.9) + 0.035 + rng.normal(0, 0.018, image.shape)
    clip = drawing.clip_rect(x, y, size, size, rx)
    parts = [
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(size)}" height="{fmt(size)}" rx="{fmt(rx)}" fill="#0b0b12"/>',
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(size)}" height="{fmt(size)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(stain_rgb(image))}"/>',
        f'<g clip-path="url(#{clip})">',
    ]
    for k in masked:
        fill = fills[k] if isinstance(fills, dict) else fills
        parts.append(
            f'<path d="{paths[k]}" fill="url(#{fill})" fill-opacity="0.35" stroke="{MASK_EDGES[fill]}" '
            'stroke-width="0.32"/>'
        )
    parts.append("</g>")
    drawing.add("".join(parts))


def nucleus_sprite(color, seed, n=64):
    """Render a textured, shaded nucleus in 3D, tinted with the color of its track."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[:n, :n]
    u, v = (cols + 0.5) / n * 2 - 1, (rows + 0.5) / n * 2 - 1
    angle = np.arctan2(v, u)
    edge = 0.92 + 0.04 * np.sin(3 * angle + rng.uniform(0, 6)) + 0.03 * np.sin(5 * angle + rng.uniform(0, 6))
    radius = np.hypot(u, v) / edge
    mask = radius < 1
    shading = np.sqrt(np.clip(1 - radius ** 2, 0, 1)) * (0.75 + 0.25 * np.clip(-u - v, -1, 1))
    texture = nucleus_texture(mask, rng, 0.62) * (0.45 + 0.55 * shading)
    rgb = 0.45 * stain_rgb(1.5 * texture) + 0.55 * hex_to_rgb(color) * (0.6 + 0.9 * texture[..., None])
    alpha = np.clip((1 - radius) * 14, 0, 1)
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)
    buffer = io.BytesIO()
    Image.fromarray((rgba * 255).astype("uint8"), mode="RGBA").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# the nuclei of the 3D+T example as (x, y, z) in the unit cube, oldest time point first; the blue mother divides
# into the pink and the orange daughter in the last time point. They move mostly in the image plane, so that no two
# nuclei overlap in the oblique projection.
NUCLEI_3DT = {
    "purple": [(0.06, 0.18, 0.2), (0.3, 0.2, 0.3), (0.54, 0.22, 0.4)],
    "teal": [(0.84, 0.0, 0.55), (0.82, 0.24, 0.5), (0.8, 0.48, 0.45)],
    "gold": [(0.04, 0.98, 0.15), (0.14, 0.78, 0.2), (0.24, 0.58, 0.25)],
    "blue": [(0.36, 0.66, 0.75), (0.5, 0.74, 0.5)],
    "pink": [(0.44, 0.98, 0.3)],
    "orange": [(0.82, 0.9, 0.3)],
}


def automatic_3dt(drawing, x, cy, front=SIZE, depth=3.5):
    """A dark volume with the segmented nuclei of the last time point and their tracks through earlier ones."""
    fx, fy, d = x, cy - (front + depth) / 2 + depth, depth

    def project(nx, ny, nz):
        return fx + nx * front + nz * d, fy + ny * front - nz * d

    def polygon(points, fill):
        return '<path d="M' + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in points) + f'Z" fill="{fill}"/>'

    top = [(fx, fy), (fx + d, fy - d), (fx + front + d, fy - d), (fx + front, fy)]
    side = [(fx + front, fy), (fx + front + d, fy - d), (fx + front + d, fy - d + front), (fx + front, fy + front)]
    parts = [
        f'<rect x="{fmt(fx)}" y="{fmt(fy)}" width="{fmt(front)}" height="{fmt(front)}" fill="#0b0b12"/>',
        polygon(top, "#262536"), polygon(side, "#1a1925"),
        f'<path d="M{fmt(fx + d)},{fmt(fy - d)}v{fmt(front)}l{fmt(-d)},{fmt(d)}M{fmt(fx + d)},{fmt(fy - d + front)}'
        f'h{fmt(front)}" fill="none" stroke="#5a5770" stroke-width="0.1" stroke-dasharray="0.4,0.3"/>',
    ]
    # earlier time points are smaller and faded, the last time point is solid; draw from back to front
    nuclei = []
    for color, positions in NUCLEI_3DT.items():
        for k, position in enumerate(positions):
            is_last = color != "blue" and k == len(positions) - 1
            nuclei.append((position, color, 0.13 if is_last else 0.075, 1.0 if is_last else 0.45))
    for (nx, ny, nz), color, r, opacity in sorted(nuclei, key=lambda n: -n[0][2]):
        px, py = project(nx, ny, nz)
        w, h = 2 * r * front, 1.75 * r * front
        parts.append(
            f'<image x="{fmt(px - w / 2)}" y="{fmt(py - h / 2)}" width="{fmt(w)}" height="{fmt(h)}" '
            f'preserveAspectRatio="none" opacity="{fmt(opacity)}" '
            f'xlink:href="{nucleus_sprite(NUCLEUS_COLORS[color], seed=len(parts))}"/>'
        )
    drawing.add("".join(parts))

    split = project(*NUCLEI_3DT["blue"][-1])
    tracks = [([project(*p) for p in NUCLEI_3DT[c]], NUCLEUS_COLORS[c]) for c in ("purple", "teal", "gold", "blue")]
    tracks += [([split, project(*NUCLEI_3DT[c][0])], NUCLEUS_COLORS[c]) for c in ("pink", "orange")]
    draw_tracks(drawing, tracks, split)
    drawing.add(
        f'<path d="M{fmt(fx)},{fmt(fy)}h{fmt(front)}v{fmt(front)}h{fmt(-front)}ZM{fmt(fx)},{fmt(fy)}l{fmt(d)},'
        f'{fmt(-d)}h{fmt(front)}l{fmt(-d)},{fmt(d)}M{fmt(fx + front + d)},{fmt(fy - d)}v{fmt(front)}l{fmt(-d)},'
        f'{fmt(d)}" fill="none" stroke="#2d2b36" stroke-width="0.18" stroke-linejoin="round"/>'
    )


def batch_stack(drawing, x, cy, draw_slide):
    """Draw a series of three slides, each processed automatically and marked as done; draw_slide draws slide k."""
    size, offset = STACK_FRAME, STACK_OFFSET
    top = cy - (size + 2 * offset) / 2
    for k in reversed(range(3)):
        fx, fy = x + offset * k, top + offset * (2 - k)
        clip = drawing.clip_rect(fx, fy, size, size, 0.8)
        draw_slide(drawing, k, fx, fy, size, clip)
        cx, cyy = fx + size - 0.75, fy + 0.75
        drawing.add(
            f'<rect x="{fmt(fx)}" y="{fmt(fy)}" width="{fmt(size)}" height="{fmt(size)}" rx="0.8" fill="none" '
            'stroke="#ffffff" stroke-width="0.2"/>'
            f'<circle cx="{fmt(cx)}" cy="{fmt(cyy)}" r="0.55" fill="#4fb98a" stroke="#ffffff" stroke-width="0.12"/>'
            f'<path d="M{fmt(cx - 0.26)},{fmt(cyy)}l0.2,0.22 0.36,-0.42" fill="none" stroke="#ffffff" '
            'stroke-width="0.18" stroke-linecap="round" stroke-linejoin="round"/>'
        )


def tissue_slide(drawing, k, x, y, size, clip):
    """A slide classified into tissue types, with the colors of the pixel classification."""
    colors = np.array([hex_to_rgb(TEAL), hex_to_rgb(ORANGE), hex_to_rgb("#e8c547")])
    image, classes = he_tissue(seed=31 + 6 * k, n=160)
    outlines = ndimage.binary_dilation(membranes(classes), iterations=1)[..., None]
    predicted = np.where(outlines, colors[classes], 0.65 * image + 0.35 * colors[classes])
    drawing.add(
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(size)}" height="{fmt(size)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(predicted)}"/>'
    )


def nuclei_slide(drawing, k, x, y, size, clip):
    """A slide whose nuclei are classified, with the colors of the object classification."""
    seed = 41 + 5 * k
    image, nuclei, classes = he_nuclei(seed)
    parts = [
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(size)}" height="{fmt(size)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(image)}"/>', f'<g clip-path="url(#{clip})">',
    ]
    for j, ((cx, cyy, r, aspect, angle), name) in enumerate(zip(nuclei, classes)):
        path = blob_path(x + cx * size, y + cyy * size, r * size, aspect, angle, seed + j)
        color = NUCLEUS_CLASSES[name]
        parts.append(f'<path d="{path}" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="0.24"/>')
    parts.append("</g>")
    drawing.add("".join(parts))


def automatic_batch(drawing, x, cy):
    """A series of histopathology slides, each classified into tissue types automatically."""
    batch_stack(drawing, x, cy, tissue_slide)


def automatic_batch_objects(drawing, x, cy):
    """A series of histopathology slides, whose nuclei are each classified automatically."""
    batch_stack(drawing, x, cy, nuclei_slide)


ITEMS = {
    "interactive_2d": (interactive_2d, SIZE, "2D"),
    "interactive_3d": (interactive_3d, SIZE + 3.5, "3D"),
    "interactive_2dt": (interactive_2dt, STACK_FRAME + 2 * STACK_OFFSET, "2D time-lapse"),
    "pixel_classification": (pixel_classification, SIZE, "pixels"),
    "object_classification": (object_classification, SIZE, "objects"),
    "automatic_2d": (automatic_2d, SIZE, "2D"),
    "automatic_3d": (automatic_3d, SIZE + 3.5, "3D"),
    "automatic_2dt": (automatic_2dt, STACK_FRAME + 2 * STACK_OFFSET, "2D time-lapse"),
    "automatic_3dt": (automatic_3dt, SIZE + 3.5, "3D time-lapse"),
    "automatic_batch": (automatic_batch, STACK_FRAME + 2 * STACK_OFFSET, "pixels"),
    "automatic_batch_objects": (automatic_batch_objects, STACK_FRAME + 2 * STACK_OFFSET, "objects"),
}


def group_width(names, scale=1.0):
    return scale * (sum(ITEMS[name][1] for name in names) + ITEM_GAP * (len(names) - 1))


def superellipse(cx, cy, a, b, power=4.0, samples=240):
    """Get the points of a rounded rectangle-like superellipse."""
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = cx + a * np.sign(c) * np.abs(c) ** (2 / power)
    y = cy + b * np.sign(s) * np.abs(s) ** (2 / power)
    return np.stack([x, y], axis=1)


def loop_badge(drawing, center, r=2.7):
    cx, cy = center
    gid = drawing.linear_gradient(cx, cy + r, cx, cy - r)
    drawing.add(
        f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" fill="#ffffff" stroke="url(#{gid})" stroke-width="0.4"/>'
    )


def correction_icon(drawing, center):
    """Two touching cells, where a positive and a negative point prompt separate the target from its neighbour."""
    cx, cy = center
    target = blob_path(cx - 0.6, cy + 0.45, 1.15, 1.25, 0.5, seed=5)
    neighbour = blob_path(cx + 0.75, cy - 0.6, 1.1, 1.2, 0.5, seed=8)
    drawing.add(
        f'<path d="{neighbour}" fill="#6b6c73"/>'
        f'<path d="{target}" fill="url(#f1c-mask)" stroke="{RIM}" stroke-width="0.3"/>'
    )
    point_prompt(drawing, cx - 0.65, cy + 0.5, r=0.45)
    px, py = cx + 0.75, cy - 0.6
    point_prompt(drawing, px, py, positive=False, r=0.45)
    cursor = [(0, 0), (0, 0.9), (0.24, 0.69), (0.4, 1.03), (0.54, 0.97), (0.38, 0.63), (0.67, 0.61)]
    d = "M" + "L".join(f"{fmt(px + 0.25 + x)},{fmt(py + 0.25 + y)}" for x, y in cursor) + "Z"
    drawing.add(f'<path d="{d}" fill="#ffffff" stroke="#2d2b36" stroke-width="0.12" stroke-linejoin="round"/>')


def logo_data_uri(path, width_px=480):
    image = Image.open(path).convert("RGBA")
    image = image.resize((width_px, round(image.height * width_px / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), image.width / image.height


def draw_interfaces(drawing, cx, cy, logo_path, napari_logo, python_logo, logo_h=7.0, height=8.2):
    """Draw the μSAM2 ecosystem as one white pill centered at (cx, cy): the logo (an svg is kept as vectors), then a
    round badge with the name to its right for each of its three interfaces. Returns the lower edge of the pill."""
    if logo_path.endswith(".svg"):
        uri = svg_uri(logo_path)
        with open(logo_path) as f:
            _, _, w, h = map(float, re.search(r'viewBox="([^"]*)"', f.read()).group(1).split())
        aspect = w / h
    else:
        uri, aspect = logo_data_uri(logo_path, width_px=900)
    logo_w = logo_h * aspect
    interfaces = [("napari", svg_uri(napari_logo)), ("Python", svg_uri(python_logo)), ("CLI", None)]
    # the three badges and their icons have one size, a little larger than first drawn (user, 2026-11-09); the napari
    # icon is a filled square and looks larger than the Python snakes at the same size, so it is drawn a bit smaller
    badge, icon, gap, name_size = 2.5, 3.4, 0.8, 2.12
    icon_scale = {"napari": 0.86, "Python": 1.0}
    widths = [2 * badge + gap + text_width(name, name_size) for name, _ in interfaces]
    total = logo_w + sum(3.0 + w for w in widths)
    x = cx - total / 2
    margin = height / 2
    drawing.add(
        f'<rect x="{fmt(x - margin)}" y="{fmt(cy - height / 2)}" width="{fmt(total + 2 * margin)}" '
        f'height="{fmt(height)}" rx="{fmt(height / 2)}" fill="#ffffff" stroke="{RIM}" stroke-width="0.4"/>'
    )
    drawing.add(
        f'<image x="{fmt(x)}" y="{fmt(cy - logo_h / 2)}" width="{fmt(logo_w)}" height="{fmt(logo_h)}" '
        f'preserveAspectRatio="xMidYMid meet" xlink:href="{uri}"/>'
    )
    x += logo_w
    for (name, icon_uri), w in zip(interfaces, widths):
        x += 3.0
        bx = x + badge
        drawing.add(
            f'<circle cx="{fmt(bx)}" cy="{fmt(cy)}" r="{fmt(badge)}" fill="#ffffff" stroke="{RIM}" '
            'stroke-width="0.3"/>'
        )
        if icon_uri is None:
            # the prompt of a terminal
            s_ = 2.6
            drawing.add(
                f'<path d="M{fmt(bx - 0.55 * s_)},{fmt(cy - 0.4 * s_)}l{fmt(0.4 * s_)},{fmt(0.32 * s_)}'
                f'l{fmt(-0.4 * s_)},{fmt(0.32 * s_)}M{fmt(bx + 0.05 * s_)},{fmt(cy + 0.42 * s_)}h{fmt(0.5 * s_)}" '
                f'fill="none" stroke="#2d2b36" stroke-width="0.3" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        else:
            size = icon * icon_scale[name]
            drawing.add(
                f'<image x="{fmt(bx - size / 2)}" y="{fmt(cy - size / 2)}" width="{fmt(size)}" height="{fmt(size)}" '
                f'xlink:href="{icon_uri}"/>'
            )
        drawing.text(bx + badge + gap, cy + 0.75, name, name_size, color=TEXT, anchor="start")
        x += w
    return cy + height / 2


def svg_uri(path):
    with open(path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode()


A_W = 169.5
# the panel is scaled to the usable width of an A4 page, and its main text is enlarged a little
A4_WIDTH, A_TEXT_SCALE = 180.0, 1.12
A_HEADER_Y = 5.5
# the inputs stack in a narrow column; the automatic band sits right of the image encoder, the interactive band
# spans the panel below them
A_ENCODER_X = (18.0, 40.0)
A_BAND_X, A_BAND_GAP = 61.5, 3.0
AUTO_BAND_H, INTER_H = 57.0, 45.0
INTER_Y = AUTO_BAND_H + A_BAND_GAP
A_H = INTER_Y + INTER_H
A_CONTENT_X = 69.0
# the embeddings enter the automatic head at the height of the 2D embedding, which is its first decoder stage
A_ENTRY = 19.0
A_OUTPUT_X = 151.5
A_SCALE = 1.45
# the width of the inputs and of the automatic outputs, in mm of panel a
INPUT_SIZE = 14.6
# the prompt encoder of the interactive band (left edge and width); APG leads into it from the decoder maps
PROMPT_ENCODER_X, PROMPT_ENCODER_W = 27.0, 17.0
# the images of the interactive band are a little smaller, so that the band stays thin
# the rows of the 2D and the 3D outputs, relative to the top of a band
A_ROW_2D, A_ROW_3D, AUTO_ROW_3D = 11.0, 31.0, 44.0
# the 3D input and its embeddings, relative to the top of the panel
A_TRUNK_3D = 44.0
# the Hiera stages as (tokens per side, size, layers, global attention)
# the gap between the Hiera stages, in mm of panel a (user feedback 2026-11-09: the encoder taller in its box)
HIERA_GAP = 3.0
# the stages are 7% larger than first drawn, in the room that the lower attention key leaves (user, 2026-11-09)
HIERA_STAGES = [(16, 7.9, 2, False), (8, 6.2, 3, False), (4, 4.8, 5, True), (2, 3.65, 3, True)]
# the decoder stages as (channels, height, depth) of their feature maps
STAGES = [(0.9, 22.0, 5.6), (1.5, 16.0, 4.2), (2.2, 11.5, 3.0), (3.0, 8.0, 2.2)]
ARROW = "#5f5a6b"
# the fills and outlines of the modules; the heads take the colors of the bands of panel c
ENCODER_STYLE = ("#d9d4ee", "#7c74ab")
INTERACTIVE_STYLE = ("#d7b9c3", "#8f6676")
AUTOMATIC_STYLE = ("#b9d1c8", "#6f9186")
GATE_STYLE = ("#b8e2e8", "#2a8f9c")
SLICE_STYLE = ("#dcdce1", "#8d8d96")


def scaled(drawing, scale, draw, x, cy):
    """Draw an item of panel c at a larger scale, with its left edge at x and its center at cy."""
    start = len(drawing.body)
    drawing.scale *= scale
    draw(drawing, 0, 0)
    drawing.scale /= scale
    inner = "".join(drawing.body[start:])
    del drawing.body[start:]
    drawing.add(f'<g transform="translate({fmt(x)},{fmt(cy)}) scale({fmt(scale)})">{inner}</g>')


def flow_arrow(drawing, points, head=0.9, width=0.25, color=ARROW, dash=None):
    """Draw a thin polyline arrow with a filled head at its last point; 'dash' is an svg dash pattern of the line."""
    points = [np.asarray(p, dtype=float) for p in points]
    direction = points[-1] - points[-2]
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    end, tip = points[-1] - direction * head * 0.8, points[-1]
    left, right = tip - direction * head + normal * head * 0.45, tip - direction * head - normal * head * 0.45
    line = "M" + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in points[:-1] + [end])
    drawing.add(
        f'<path d="{line}" fill="none" stroke="{color}" stroke-width="{fmt(width)}" stroke-linejoin="round"'
        + (f' stroke-dasharray="{dash}"' if dash else "") + '/>'
        f'<path d="M{fmt(tip[0])},{fmt(tip[1])}L{fmt(left[0])},{fmt(left[1])}L{fmt(right[0])},{fmt(right[1])}Z" '
        f'fill="{color}"/>'
    )


def module(drawing, x, y, w, h, lines, style, size=1.94):
    """Draw a network module as a flat rounded box with a thin outline and its name inside."""
    fill, edge = style
    drawing.add(
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="{fmt(min(0.8, h / 4))}" '
        f'fill="{fill}" stroke="{edge}" stroke-width="0.22"/>'
    )
    for k, line in enumerate(lines):
        drawing.text(x + w / 2, y + h / 2 + (k - (len(lines) - 1) / 2) * 1.2 * size + 0.34 * size, line, size)


def feature_map(drawing, x, cy, w, h, d, style):
    """Draw a feature map as a box seen at an angle: w channels deep, h high, and d into the page."""
    fill, edge = style
    dx, dy = 0.7 * d, -0.5 * d
    y = cy - (h + dy) / 2
    light, dark = mix(fill, "#ffffff", 0.45), mix(fill, edge, 0.35)
    drawing.add(
        f'<path d="M{fmt(x)},{fmt(y)}l{fmt(dx)},{fmt(dy)}h{fmt(w)}l{fmt(-dx)},{fmt(-dy)}Z" fill="{light}"/>'
        f'<path d="M{fmt(x + w)},{fmt(y)}l{fmt(dx)},{fmt(dy)}v{fmt(h)}l{fmt(-dx)},{fmt(-dy)}Z" fill="{dark}"/>'
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" fill="{fill}"/>'
        f'<path d="M{fmt(x)},{fmt(y)}h{fmt(w)}v{fmt(h)}h{fmt(-w)}ZM{fmt(x)},{fmt(y)}l{fmt(dx)},{fmt(dy)}h{fmt(w)}'
        f'l{fmt(-dx)},{fmt(-dy)}M{fmt(x + w + dx)},{fmt(y + dy)}v{fmt(h)}l{fmt(-dx)},{fmt(-dy)}" fill="none" '
        f'stroke="{edge}" stroke-width="0.15" stroke-linejoin="round"/>'
    )
    return x + w + dx


def mix(a, b, t):
    rgb = (1 - t) * hex_to_rgb(a) + t * hex_to_rgb(b)
    return "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgb)


def embedding_slab(drawing, x, y, w, d, n=5, style=ENCODER_STYLE):
    """Draw an embedding as a flat grid seen at an angle; (x, y) is its front left corner."""
    fill, edge = style
    dx, dy = 0.75 * d, -0.55 * d
    top = [(x, y), (x + w, y), (x + w + dx, y + dy), (x + dx, y + dy)]
    grid = "".join(
        f"M{fmt(x + w * k / n)},{fmt(y)}l{fmt(dx)},{fmt(dy)}"
        f"M{fmt(x + dx * k / n)},{fmt(y + dy * k / n)}h{fmt(w)}" for k in range(1, n)
    )
    drawing.add(
        '<path d="M' + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in top) + f'Z" fill="{fill}" stroke="{edge}" '
        'stroke-width="0.15" stroke-linejoin="round"/>'
        f'<path d="{grid}" stroke="{edge}" stroke-width="0.08"/>'
    )
    return x + w + dx


# the volume of the 3D examples of panel a: a neuron with a cell body and its processes among other cells and neurites,
# all as smooth shapes in the unit cube, given as (x, y, z) with y pointing down and z into the page
VOLUME_N = 128
NEURON_LABEL = 1
# the neuron that is prompted: its cell body lies in the middle slice of the volume, which holds the prompt
NEURON_SOMA = ((0.33, 0.6, 0.5), 0.15)
NEURON_NUCLEUS = ((0.31, 0.62, 0.5), 0.085)
PROMPT = (0.42, 0.56, 0.5)
# the processes of the cells as (control points, radius, taper)
NEURON_PROCESSES = [
    ([(0.38, 0.54, 0.42), (0.47, 0.46, 0.26), (0.56, 0.42, 0.1), (0.6, 0.4, -0.1)], 0.055, (1.0, 0.75)),
    ([(0.3, 0.5, 0.55), (0.27, 0.3, 0.62), (0.3, 0.1, 0.7), (0.33, -0.1, 0.76)], 0.05, (1.0, 0.7)),
    ([(0.42, 0.68, 0.55), (0.62, 0.8, 0.62), (0.84, 0.9, 0.7), (1.1, 0.98, 0.76)], 0.034, (1.0, 1.0)),
    ([(0.24, 0.64, 0.58), (0.12, 0.72, 0.76), (0.05, 0.76, 0.94), (-0.1, 0.8, 1.1)], 0.045, (1.0, 0.65)),
]
# a big cell body that the front face cuts through, and a smaller one that the side face cuts through
BIG_CELL_LABEL, SIDE_CELL_LABEL = 2, 3
BIG_PROCESSES = [
    ([(0.74, 0.12, 0.2), (0.71, 0.02, 0.32), (0.69, -0.04, 0.42), (0.67, -0.1, 0.52)], 0.07, (1.0, 0.6)),
    ([(0.86, 0.36, 0.18), (0.93, 0.52, 0.32), (0.99, 0.66, 0.46), (1.1, 0.76, 0.58)], 0.05, (1.0, 0.7)),
]
SIDE_PROCESSES = [
    ([(0.85, 0.45, 0.76), (0.7, 0.42, 0.71), (0.55, 0.4, 0.73), (0.4, 0.38, 0.79)], 0.045, (1.0, 0.7)),
]
# the cells as (label, soma, nucleus, processes)
CELLS = [
    (NEURON_LABEL, NEURON_SOMA, NEURON_NUCLEUS, NEURON_PROCESSES),
    (BIG_CELL_LABEL, ((0.8, 0.22, 0.08), 0.2), ((0.79, 0.23, 0.07), 0.11), BIG_PROCESSES),
    (SIDE_CELL_LABEL, ((0.96, 0.45, 0.8), 0.17), ((0.97, 0.46, 0.8), 0.095), SIDE_PROCESSES),
]
CAMERA_DIRECTION = (0.8, 0.62, 1.0)
EDGE = "#2d2b36"


def spline(points, samples=240):
    points = np.asarray(points, dtype=float)
    t = np.linspace(0, 1, len(points))
    return make_interp_spline(t, points, k=min(3, len(points) - 1))(np.linspace(0, 1, samples))


def radius_profile(rng, base, samples, taper=(1.0, 1.0)):
    """Let the thickness of a neurite swell and shrink smoothly along its path, with a few bead-like varicosities."""
    t = np.linspace(0, 1, samples)
    radius = base * np.linspace(*taper, samples)
    radius = radius * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(1, 3) * t + rng.uniform(0, 2 * np.pi)))
    for center in rng.uniform(0.1, 0.9, rng.integers(0, 3)):
        radius = radius * (1 + 0.6 * np.exp(-((t - center) / 0.03) ** 2))
    return radius


def stamp_path(labels, label, path, radius):
    """Mark the voxels within the radius of a path in unit coordinates, without overwriting other objects."""
    n = labels.shape[0]
    for (px, py, pz), r in zip(path * (n - 1), radius * (n - 1)):
        low = np.maximum(np.floor([pz - r, py - r, px - r]).astype(int), 0)
        high = np.minimum(np.ceil([pz + r, py + r, px + r]).astype(int) + 1, n)
        if np.any(high <= low):
            continue
        zz, yy, xx = np.ogrid[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        block = labels[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
        block[((zz - pz) ** 2 + (yy - py) ** 2 + (xx - px) ** 2 <= r ** 2) & (block == 0)] = label


def candidate_neurites(rng):
    """Propose neurites: most run along z with a few bends, some along x or y, and some pairs twist around each
    other."""
    while True:
        kind = rng.uniform()
        if kind < 0.2:
            # two strands that spiral around a common axis along z
            x0, y0 = rng.uniform(0.05, 0.95, 2)
            drift, helix, turns = rng.normal(0, 0.1, 2), rng.uniform(0.05, 0.075), rng.uniform(1.0, 2.0)
            t = np.linspace(0, 1, 240)
            for phase in (0.0, np.pi):
                angle = 2 * np.pi * turns * t + phase
                path = np.stack([
                    x0 + drift[0] * t + helix * np.cos(angle), y0 + drift[1] * t + helix * np.sin(angle), 1.2 * t - 0.1
                ], axis=1)
                yield [(path, radius_profile(rng, rng.uniform(0.022, 0.032), len(path)))]
            continue
        axis = 2 if kind < 0.8 else (0 if kind < 0.92 else 1)
        start = rng.uniform(0.0, 1.0, 3)
        points = []
        for k, t in enumerate(np.linspace(-0.1, 1.1, 6)):
            point = start + rng.normal(0, 0.06, 3) * (k > 0)
            point[axis] = t
            start = point
            points.append(point.copy())
        path = spline(points)
        yield [(path, radius_profile(rng, rng.uniform(0.016, 0.045), len(path)))]


def collides(path, radius, occupied, margin=0.004):
    for other, other_radius in occupied:
        gaps = cdist(path[::4], other[::4]) - radius[::4, None] - other_radius[None, ::4]
        if gaps.min() < margin:
            return True
    return False


@lru_cache
def volume_data(count=220, grow=8):
    """Label a volume with the cells and the other neurites, indexed as (z, y, x), and mark their mitochondria and
    synaptic vesicles."""
    rng = np.random.default_rng(EM_SEED)
    n = VOLUME_N
    labels = np.zeros((n, n, n), dtype="uint32")
    occupied = []
    for label, (center, radius), _, _ in CELLS:
        soma = (np.array([center]), np.array([radius]))
        stamp_path(labels, label, *soma)
        occupied.append(soma)
    for label, _, _, processes in CELLS:
        for points, base, taper in processes:
            path = spline(points)
            process = (path, radius_profile(rng, base, len(path), taper))
            stamp_path(labels, label, *process)
            occupied.append(process)
    label, tries, neurites = len(CELLS) + 1, 0, []
    for group in candidate_neurites(rng):
        tries += 1
        if label > count or tries > 20000:
            break
        # a pair of twisted strands is proposed one strand after the other
        for path, radius in group:
            if collides(path, radius, occupied):
                continue
            stamp_path(labels, label, path, radius)
            occupied.append((path, radius))
            neurites.append((label, path, radius))
            label += 1
    # the neurites grow into the free space around them, like the densely packed neuropil of EM
    distance, (zi, yi, xi) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    labels = np.where(distance <= grow, labels[zi, yi, xi], 0).astype("uint32")
    return labels, mitochondria(labels, neurites), vesicles(labels, neurites)


def mitochondria(labels, neurites):
    """Place mitochondria as short capsules along the paths of some neurites, inside their neurite."""
    rng = np.random.default_rng(EM_SEED + 1)
    marks = np.zeros(labels.shape, dtype="uint8")
    for label, path, radius in neurites:
        for _ in range(rng.integers(1, 4)):
            length = rng.integers(25, 60)
            start = rng.integers(0, len(path) - length)
            segment = path[start:start + length] + rng.normal(0, 0.15, 3) * radius[start]
            marks_label = np.zeros_like(marks)
            stamp_path(marks_label, 1, segment, np.maximum(0.55 * radius[start:start + length], 0.012))
            marks |= marks_label & (labels == label)
    return marks.astype(bool)


def vesicles(labels, neurites, per_bouton=40):
    """Fill the varicosities of the neurites, their synaptic boutons, with clusters of small vesicles."""
    rng = np.random.default_rng(EM_SEED + 2)
    marks = np.zeros(labels.shape, dtype="uint8")
    for label, path, radius in neurites:
        peaks = np.nonzero(
            (radius[1:-1] > radius[:-2]) & (radius[1:-1] >= radius[2:]) & (radius[1:-1] > 1.3 * np.median(radius))
        )[0] + 1
        for index in peaks:
            direction = rng.normal(0, 1, (per_bouton, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            points = path[index] + direction * 0.6 * radius[index] * rng.uniform(0, 1, (per_bouton, 1)) ** (1 / 3)
            marks_label = np.zeros_like(marks)
            stamp_path(marks_label, 1, points, np.full(per_bouton, 0.007))
            marks |= marks_label & (labels == label)
    return marks.astype(bool)


def neuron_volume():
    return volume_data()[0]


def volume_faces(volume, axis=0):
    """Get the front, top and side faces of a volume whose spatial axes start at axis; top and side have z as rows."""
    index = (slice(None),) * axis
    front, top, side = volume[index + (0,)], volume[index + (slice(None), 0)], volume[index + (Ellipsis, -1)]
    return {"front": front, "top": top, "side": side}


def face_coordinates_3d(face, n):
    """Get the unit coordinates (x, y, z) of the pixels of a face from volume_faces."""
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    if face == "front":
        return a, b, np.zeros_like(a)
    if face == "top":
        return a, np.zeros_like(a), b
    return np.ones_like(a), a, b


def em_section(labels, coordinates, mito, vesicle, seed):
    """Simulate the EM section of a plane through the volume, with the nuclei and organelles of the cell bodies that it
    cuts, and the sections of the mitochondria and vesicles."""
    soma, nucleus = np.zeros(labels.shape, dtype=bool), np.zeros(labels.shape, dtype=bool)
    for label, cell_soma, cell_nucleus, _ in CELLS:
        soma |= (labels == label) & in_sphere(*coordinates, cell_soma)
        nucleus |= in_sphere(*coordinates, cell_nucleus)
    gray = em_texture(
        labels, membranes(labels), seed, soma=soma, nucleus=nucleus, mitochondria=mito & ~nucleus, vesicles=vesicle
    )
    return np.repeat(gray[..., None], 3, axis=-1)


@lru_cache
def em_volume_faces():
    """Simulate EM sections on the visible faces of the volume."""
    n, (labels, mito, vesicle) = VOLUME_N, volume_data()
    planes = zip(volume_faces(labels).items(), volume_faces(mito).values(), volume_faces(vesicle).values())
    return {
        face: em_section(face_labels, face_coordinates_3d(face, n), face_mito, face_vesicle, EM_SEED + 17 * (k + 1))
        for k, ((face, face_labels), face_mito, face_vesicle) in enumerate(planes)
    }


@lru_cache
def em_slice(index):
    """Simulate the EM section of the slice at a z index."""
    n, (labels, mito, vesicle) = VOLUME_N, volume_data()
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    coordinates = (a, b, np.full_like(a, index / (n - 1)))
    return em_section(labels[index], coordinates, mito[index], vesicle[index], EM_SEED + 5 * index)


# the face images of the volume are upsampled by this factor for the renders, and the renders use this window size
TEXTURE_FACTOR, RENDER_SIZE = 4, 800


def to_world(points):
    """Map unit coordinates (x, y, z) to the world coordinates of the renders, where y points up and z to the viewer."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.stack([points[:, 0], 1 - points[:, 1], 1 - points[:, 2]], axis=1) * (VOLUME_N - 1)


def color_grid(dimensions, origin, data, factor=TEXTURE_FACTOR):
    """Make a flat grid with the colors of an image. The image is upsampled by factor along its two axes with
    'upscale', so that the voxels of the volume do not show as steps in the renders."""
    flat = [k for k, d in enumerate(dimensions) if d == 1][0]
    # the image has its axes in the reverse order of the dimensions
    image = upscale(np.squeeze(np.asarray(data, dtype=float), axis=2 - flat), factor)
    data = np.expand_dims(image, axis=2 - flat)
    size = [d * factor if d > 1 else 1 for d in dimensions]
    spacing = [(d - 1) / (m - 1) if d > 1 else 1.0 for d, m in zip(dimensions, size)]
    grid = pv.ImageData(dimensions=size, origin=origin, spacing=spacing)
    grid.point_data["rgb"] = (np.clip(data, 0, 1) * 255).astype("uint8").reshape(-1, 3)
    return grid


def face_slabs(rgb):
    """Turn the face images of volume_faces into flat grids at the faces of the volume, with their colors."""
    n = VOLUME_N
    return [
        color_grid((n, n, 1), (0, 0, n - 1), rgb["front"][::-1][None]),
        color_grid((n, 1, n), (0, n - 1, 0), rgb["top"][::-1][:, None]),
        color_grid((1, n, n), (n - 1, 0, 0), rgb["side"][::-1, ::-1][:, :, None]),
    ]


def slice_slab(rgb, index, spread=1.0):
    """Place the image of the slice at a z index in the volume, with the slices pulled apart by spread."""
    n = VOLUME_N
    return color_grid((n, n, 1), (0, 0, (n - 1) / 2 + spread * ((n - 1) / 2 - index)), rgb[::-1][None])


@lru_cache
def label_surfaces():
    """Extract a smooth, closed surface for each object of the volume, slightly shrunk so that touching objects stay
    apart, in the world coordinates of the renders."""
    labels = neuron_volume()
    n = labels.shape[0]
    meshes = {}
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        # a margin of empty voxels closes the surfaces where the objects leave the volume
        mask = np.pad(ndimage.binary_erosion(labels[box] == label), 3)
        field = ndimage.gaussian_filter(mask.astype("float32"), 1.0)
        origin = [box[2].start - 3, box[1].start - 3, box[0].start - 3]
        grid = pv.ImageData(dimensions=field.shape[::-1], origin=origin)
        grid.point_data["field"] = field.ravel()
        mesh = grid.contour([0.5], scalars="field")
        if mesh.n_points == 0:
            continue
        x, y, z = mesh.points.T
        mesh.points = np.stack([x, n - 1 - y, n - 1 - z], axis=1)
        meshes[label] = mesh
    return meshes


def render(add, size=RENDER_SIZE, edges=True):
    """Render a scene off screen from the viewpoint of the blocks, inside the edges of the volume, and crop it."""
    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    # translucent objects need depth peeling to be drawn in the right order
    plotter.enable_depth_peeling()
    add(plotter)
    n = VOLUME_N - 1
    if edges:
        plotter.add_mesh(pv.Box(bounds=(0, n, 0, n, 0, n)).extract_all_edges(), color=EDGE, line_width=3 * size / 900)
    focal = np.full(3, n / 2)
    direction = np.asarray(CAMERA_DIRECTION) / np.linalg.norm(CAMERA_DIRECTION)
    plotter.camera.focal_point = focal
    plotter.camera.position = focal + direction * n * 4.2
    plotter.camera.up = (0, 1, 0)
    # the view must hold the whole cube, whose nearest corner reaches furthest out
    plotter.camera.view_angle = 30
    plotter.reset_camera_clipping_range()
    plotter.render()
    image = plotter.screenshot(transparent_background=True, return_img=True)
    plotter.close()
    rows, cols = np.nonzero(image[..., 3])
    return image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def add_colors(plotter, grids, opacity=1.0):
    for grid in grids:
        plotter.add_mesh(grid, scalars="rgb", rgb=True, ambient=0.35, diffuse=0.75, specular=0.05, opacity=opacity)


def add_point_prompt(plotter, point, radius=6.5, world=False):
    """Lay a positive point prompt flat onto the slice that holds it, facing the viewer. 'point' is in unit
    coordinates, or in world coordinates with 'world'."""
    center = (np.asarray(point, float) if world else to_world(point)[0]) + (0, 0, 0.6)
    ring = pv.Disc(center=center, inner=0, outer=1.25 * radius, normal=(0, 0, 1), c_res=60)
    disc = pv.Disc(center=center + (0, 0, 0.2), inner=0, outer=radius, normal=(0, 0, 1), c_res=60)
    plotter.add_mesh(ring, color="white", ambient=0.6)
    plotter.add_mesh(disc, color=TEAL, ambient=0.6)
    arm, half = 0.55 * radius, 0.13 * radius
    cx, cy, cz = center + (0, 0, 0.5)
    for bounds in ((cx - arm, cx + arm, cy - half, cy + half), (cx - half, cx + half, cy - arm, cy + arm)):
        plotter.add_mesh(pv.Box(bounds=bounds + (cz, cz + 0.3)), color="white", ambient=0.8)


@lru_cache
def render_em_block():
    return render(lambda plotter: add_colors(plotter, face_slabs(em_volume_faces())))


# eight slices, like the eight embeddings that the encoder makes of them
SLICE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)


@lru_cache
@lru_cache
def render_prompted_slice():
    """The volume with its middle slice, on which the point prompt is placed."""
    middle = VOLUME_N // 2

    def add(plotter):
        # the volume behind the prompted slice is an opaque EM block whose front face is the slice, the volume in
        # front of it only a very faint shell, so that the prompt sits on the cut face of the volume
        n, z, faces = VOLUME_N, VOLUME_N - 1 - middle, em_volume_faces()
        back = n - middle
        add_colors(plotter, [
            color_grid((n, 1, back), (0, n - 1, 0), faces["top"][middle:][::-1][:, None]),
            color_grid((1, n, back), (n - 1, 0, 0), faces["side"][middle:][::-1, ::-1][:, :, None]),
        ])
        front = pv.Box(bounds=(0, n - 1, 0, n - 1, z + 0.5, n - 1))
        plotter.add_mesh(front, color="#c9c3e3", opacity=0.12, ambient=0.5)
        add_colors(plotter, [slice_slab(em_slice(middle), middle)])
        add_point_prompt(plotter, PROMPT, radius=9.0)
    return render(add)


@lru_cache
def apg_points():
    """One positive point per object of the middle slice, at its most interior pixel, as (row, col): the prompts
    that APG derives from the decoder maps. Every object gets one (user feedback 2026-11-09); only specks of fewer
    than 20 pixels at the border are left out."""
    labels, points = slice_labels(), []
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        mask = labels == label
        if mask.sum() < 20:
            continue
        distance = ndimage.distance_transform_edt(mask)
        points.append(np.unravel_index(distance.argmax(), distance.shape))
    return points


def slice_apg(drawing, x, cy):
    """The middle slice with a positive point on each object, as APG prompts it."""
    tx, ty = slice_tile(drawing, x, cy, em_slice(MIDDLE))
    for point in apg_points():
        # the points of objects at the border stay inside the tile
        px, py = slice_point(tx, ty, point)
        px, py = np.clip(px, tx + 0.6, tx + SIZE - 0.6), np.clip(py, ty + 0.6, ty + SIZE - 0.6)
        point_prompt(drawing, px, py, r=0.32)


def slice_apg_sheets(drawing, x, cy, n_sheets=3, step=0.85):
    """The APG plane with the neighbouring slices of the volume behind it, as translucent sheets of paper shifted up
    and to the right."""
    for k in range(n_sheets, 0, -1):
        sx, sy = x + k * step, cy - SIZE / 2 - k * step
        clip = drawing.clip_rect(sx, sy, SIZE, SIZE, 0.9)
        drawing.add(
            f'<image x="{fmt(sx)}" y="{fmt(sy)}" width="{fmt(SIZE)}" height="{fmt(SIZE)}" preserveAspectRatio="none" '
            f'clip-path="url(#{clip})" opacity="0.45" xlink:href="{array_uri(em_slice(MIDDLE + 4 * k))}"/>'
            f'<rect x="{fmt(sx)}" y="{fmt(sy)}" width="{fmt(SIZE)}" height="{fmt(SIZE)}" rx="0.9" fill="none" '
            f'stroke="#8d8d96" stroke-width="0.15"/>'
        )
    slice_apg(drawing, x, cy)


@lru_cache
def render_apg_slice():
    """The volume with its middle slice, with a positive point on each object of the slice, as APG prompts it."""
    middle = VOLUME_N // 2

    def add(plotter):
        n, z, faces = VOLUME_N, VOLUME_N - 1 - middle, em_volume_faces()
        back = n - middle
        add_colors(plotter, [
            color_grid((n, 1, back), (0, n - 1, 0), faces["top"][middle:][::-1][:, None]),
            color_grid((1, n, back), (n - 1, 0, 0), faces["side"][middle:][::-1, ::-1][:, :, None]),
        ])
        front = pv.Box(bounds=(0, n - 1, 0, n - 1, z + 0.5, n - 1))
        plotter.add_mesh(front, color="#c9c3e3", opacity=0.12, ambient=0.5)
        add_colors(plotter, [slice_slab(em_slice(middle), middle)])
        for row, col in apg_points():
            add_point_prompt(plotter, (col / (n - 1), row / (n - 1), PROMPT[2]), radius=4.0)
    return render(add)


@lru_cache
def render_map_block(name):
    return render(lambda plotter: add_colors(plotter, face_slabs(decoder_maps()[1][name])))


@lru_cache
def render_neuron():
    """The neuron that the prompt on the middle slice segments through the whole volume, among faint other objects."""
    n = VOLUME_N - 1

    def add(plotter):
        for label, mesh in label_surfaces().items():
            if label == NEURON_LABEL:
                plotter.add_mesh(mesh, color=PURPLE, smooth_shading=True, specular=0.3, specular_power=20)
            else:
                plotter.add_mesh(mesh, color="#9a9aa2", opacity=0.1, smooth_shading=True)
        plane = pv.Box(bounds=(0, n, 0, n, n / 2, n / 2)).extract_all_edges()
        plotter.add_mesh(plane, color=TEAL, line_width=4 * RENDER_SIZE / 900)
    return render(add)


@lru_cache
def render_instances():
    """All objects of the volume, each in its own color."""
    colors = neurite_colors(int(neuron_volume().max()) + 1)

    def add(plotter):
        for label, mesh in label_surfaces().items():
            color = PURPLE if label == NEURON_LABEL else colors[label]
            plotter.add_mesh(mesh, color=color, smooth_shading=True, specular=0.3, specular_power=20)
    return render(add)


def colored_slice_rgb(index):
    """A slice of the volume with its profiles in the colors of their 3d objects, as in 'automatic_slice_rgb', so
    that an object keeps its color from slice to slice."""
    labels = neuron_volume()[index]
    colors = neurite_colors(int(neuron_volume().max()) + 1)
    base = em_slice(index)
    image = np.where((labels > 0)[..., None], 0.45 * base + 0.55 * colors[labels], base)
    image[membranes(labels)] *= 0.45
    return image


def slice_apg_output_sheets(drawing, x, cy, n_sheets=3, step=0.85):
    """The APG output as a stack like the APG input: the middle slice in front with all objects and their points,
    and the slices behind it as translucent sheets with the same objects in the same colors, so that the objects
    chain through the volume (user feedback 2026-11-09)."""
    for k in range(n_sheets, 0, -1):
        sx, sy = x + k * step, cy - SIZE / 2 - k * step
        clip = drawing.clip_rect(sx, sy, SIZE, SIZE, 0.9)
        drawing.add(
            f'<image x="{fmt(sx)}" y="{fmt(sy)}" width="{fmt(SIZE)}" height="{fmt(SIZE)}" preserveAspectRatio="none" '
            f'clip-path="url(#{clip})" opacity="0.6" xlink:href="{array_uri(colored_slice_rgb(MIDDLE + 4 * k))}"/>'
            f'<rect x="{fmt(sx)}" y="{fmt(sy)}" width="{fmt(SIZE)}" height="{fmt(SIZE)}" rx="0.9" fill="none" '
            f'stroke="#8d8d96" stroke-width="0.15"/>'
        )
    tx, ty = slice_tile(drawing, x, cy, colored_slice_rgb(MIDDLE))
    # the stack is drawn larger than the APG input, so its points are smaller in its own units
    for point in apg_points():
        px, py = slice_point(tx, ty, point)
        px, py = np.clip(px, tx + 0.45, tx + SIZE - 0.45), np.clip(py, ty + 0.45, ty + SIZE - 0.45)
        point_prompt(drawing, px, py, r=0.25)


def rgba_uri(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def place_render(drawing, image, x, cy, width):
    """Draw a render with its left edge at x and its center at cy, and return its height."""
    height = width * image.shape[0] / image.shape[1]
    drawing.add(
        f'<image x="{fmt(x)}" y="{fmt(cy - height / 2)}" width="{fmt(width)}" height="{fmt(height)}" '
        f'xlink:href="{rgba_uri(image)}"/>'
    )
    return height


def interactive_3d_render(drawing, x, cy):
    place_render(drawing, render_neuron(), x, cy, SIZE + 3.5)


def automatic_3d_render(drawing, x, cy):
    place_render(drawing, render_instances(), x, cy, SIZE + 3.5)


MIDDLE = VOLUME_N // 2


@lru_cache
def slice_labels():
    """The labels of the middle slice of the volume, which is the 2D image of panel a."""
    return neuron_volume()[MIDDLE]


@lru_cache
def slice_prompts():
    """Find the prompts on the middle slice: a positive point on the cell body of the neuron, a negative point on a
    neurite that touches it, and a box around another profile. Points are (row, col) and the box is (r0, c0, r1, c1)
    in pixels."""
    labels, n = slice_labels(), VOLUME_N
    positive = (PROMPT[1] * (n - 1), PROMPT[0] * (n - 1))
    ring = ndimage.binary_dilation(labels == NEURON_LABEL, iterations=3) & (labels != NEURON_LABEL) & (labels > 0)
    neighbour = np.bincount(labels[ring]).argmax()
    distance = ndimage.distance_transform_edt(labels == neighbour)
    negative = np.unravel_index(distance.argmax(), distance.shape)
    best, box_label = 0.0, None
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None or label in (NEURON_LABEL, neighbour):
            continue
        (r0, r1), (c0, c1) = (box[0].start, box[0].stop), (box[1].start, box[1].stop)
        mask = labels == label
        far = np.hypot((r0 + r1) / 2 - positive[0], (c0 + c1) / 2 - positive[1]) > 0.3 * n
        if not far or min(r0, c0) < 3 or max(r1, c1) > n - 3 or not 250 < mask.sum() < 1200:
            continue
        # the roundest of the large profiles makes a clear box
        thickness = ndimage.distance_transform_edt(mask).max()
        if thickness > best:
            best, box_label = thickness, (label, (r0, c0, r1, c1))
    return {"positive": positive, "negative": negative, "box": box_label}


def overlay(base, masks, colors=None):
    """Lay masks with a blue to purple gradient (or the rgb of 'colors', one per mask) and a light rim over a gray
    image."""
    n = base.shape[0]
    rows, cols = np.mgrid[:n, :n]
    g = ((rows + cols) / (2 * n))[..., None]
    gradient = (1 - g) * hex_to_rgb(BLUE) + g * hex_to_rgb(PURPLE)
    image = base.copy()
    for k, mask in enumerate(masks):
        rim = mask & ~ndimage.binary_erosion(mask, iterations=2)
        fill = gradient if colors is None or colors[k] is None else hex_to_rgb(colors[k])
        image = np.where(mask[..., None], 0.35 * image + 0.65 * fill, image)
        image[rim] = hex_to_rgb(RIM)
    return image


def slice_tile(drawing, x, cy, rgb):
    """Draw an image of the middle slice as a rounded tile with its left edge at x and its center at cy."""
    y = cy - SIZE / 2
    clip = drawing.clip_rect(x, y, SIZE, SIZE, 0.9)
    drawing.add(
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(SIZE)}" height="{fmt(SIZE)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(rgb)}"/>'
    )
    return x, y


def slice_point(x, y, point):
    return x + point[1] / VOLUME_N * SIZE, y + point[0] / VOLUME_N * SIZE


def draw_slice_prompts(drawing, x, y, negative=True):
    prompts = slice_prompts()
    r0, c0, r1, c1 = prompts["box"][1]
    (x0, y0), (x1, y1) = slice_point(x, y, (r0 - 2, c0 - 2)), slice_point(x, y, (r1 + 2, c1 + 2))
    drawing.add(
        f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" height="{fmt(y1 - y0)}" rx="0.15" fill="none" '
        'stroke="#ffffff" stroke-width="0.22"/>'
    )
    point_prompt(drawing, *slice_point(x, y, prompts["positive"]), r=0.55)
    if negative:
        point_prompt(drawing, *slice_point(x, y, prompts["negative"]), positive=False, r=0.55)


def slice_input(drawing, x, cy):
    slice_tile(drawing, x, cy, em_slice(MIDDLE))


def slice_prompted(drawing, x, cy):
    """The middle slice with the prompts that μSAM2 is trained with: a box, and positive and negative points."""
    draw_slice_prompts(drawing, *slice_tile(drawing, x, cy, em_slice(MIDDLE)))


def slice_interactive(drawing, x, cy):
    """The neuron and the boxed profile, segmented from their prompts on the middle slice."""
    labels, prompts = slice_labels(), slice_prompts()
    masks = [labels == NEURON_LABEL, labels == prompts["box"][0]]
    # the object of the point keeps the purple, the object of the box is amber (user feedback 2026-11-09)
    image = overlay(em_slice(MIDDLE), masks, colors=[None, BOX_OBJECT])
    draw_slice_prompts(drawing, *slice_tile(drawing, x, cy, image), negative=False)


def automatic_slice_rgb():
    """The middle slice with all its profiles, each in its own color."""
    labels = slice_labels()
    colors = neurite_colors(int(labels.max()) + 1)
    base = em_slice(MIDDLE)
    image = np.where((labels > 0)[..., None], 0.45 * base + 0.55 * colors[labels], base)
    image[membranes(labels)] *= 0.45
    return image


def slice_automatic(drawing, x, cy):
    """All profiles of the middle slice, each in its own color."""
    slice_tile(drawing, x, cy, automatic_slice_rgb())


# the fire lookup table of ImageJ, from dark purple over red and orange to pale yellow, as (position, rgb)
FIRE = [
    (0.0, (0.16, 0.0, 0.32)), (0.18, (0.55, 0.0, 0.55)), (0.38, (0.9, 0.12, 0.08)), (0.58, (1.0, 0.5, 0.0)),
    (0.78, (1.0, 0.85, 0.1)), (1.0, (1.0, 0.98, 0.75)),
]


def fire(value):
    """Map values in [0, 1] to the colors of the fire lookup table."""
    positions = [position for position, _ in FIRE]
    return np.stack([np.interp(value, positions, [rgb[c] for _, rgb in FIRE]) for c in range(3)], axis=-1)


def colorize(channels):
    """Map the foreground, directed distance and boundary channels of micro-sam to rgb images."""
    inside = channels[0][..., None] > 0
    background = hex_to_rgb("#2d2b36")
    mid = hex_to_rgb("#f2eff7")
    maps = {
        # the object boundaries are dark, so that touching objects of dense data stay apart
        "foreground": np.where(inside & (channels[4] == 0)[..., None], mid, background),
        "boundary": np.where((channels[4] > 0)[..., None], mid, background),
    }
    shown = inside[..., 0] & (channels[4] == 0)
    for name, channel in zip(("z-distance", "y-distance", "x-distance"), channels[1:4]):
        # the distances crowd around zero, so they are stretched by their rank over the whole fire table
        ranks = np.zeros(channel.shape)
        if shown.any():
            values = channel[shown]
            ordered = np.sort(values)
            # equal values, like the many zeros of the z-distance, share their middle rank
            low, high = np.searchsorted(ordered, values, side="left"), np.searchsorted(ordered, values, side="right")
            ranks[shown] = (low + high) / 2 / len(values)
        maps[name] = np.where(shown[..., None], fire(ranks), background)
    return maps


@lru_cache
def decoder_maps():
    """Compute the targets of the automatic segmentation decoder with micro-sam, for the 2D image and the volume.

    The maps of the volume are given for its front, top and side faces.
    """
    transform = DirectedPerObjectBoundaryDistanceTransform(with_boundaries=True)
    maps_2d = colorize(transform(slice_labels()))
    faces = {face: colorize(channels) for face, channels in volume_faces(transform(neuron_volume()), axis=1).items()}
    maps_3d = {name: {face: faces[face][name] for face in faces} for name in maps_2d}
    return maps_2d, maps_3d


def map_tile(drawing, x, y, size, rgb):
    clip = drawing.clip_rect(x, y, size, size, 0.5)
    drawing.add(
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(size)}" height="{fmt(size)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(rgb)}"/>'
    )


def item_label(drawing, x, width, y, text, size=1.76):
    drawing.text(x + width / 2, y, text, size, color=SUBTEXT)


def outputs(drawing, draw_2d, draw_3d, y, row_2d, row_3d, scale=A_SCALE):
    """Draw the 2D output above the 3D output, and return the left edge of each."""
    # the outputs are as large as the inputs (user feedback 2026-11-09), centered in the column of the widest output
    w2 = w3 = INPUT_SIZE
    wide = (SIZE + 3.5) * A_SCALE
    x2, x3 = A_OUTPUT_X + (wide - w2) / 2, A_OUTPUT_X + (wide - w3) / 2
    scaled(drawing, w2 / SIZE, draw_2d, x2, y + row_2d)
    item_label(drawing, A_OUTPUT_X, wide, y + row_2d + w2 / 2 + 2.2, "2D")
    scaled(drawing, w3 / (SIZE + 3.5), draw_3d, x3, y + row_3d)
    item_label(drawing, A_OUTPUT_X, wide, y + row_3d + w3 / 2 + 2.3, "3D")
    return x2, x3


def a_band(drawing, x, y, height, title, background, title_y, horizontal=False, title_x=None):
    drawing.add(
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(A_W - x)}" height="{fmt(height)}" rx="1.2" '
        f'fill="{background}"/>'
    )
    # a little smaller than the headers, so that it fits the height of the thin interactive band
    if horizontal:
        # at the top center inside the band (user feedback 2026-11-09)
        drawing.text((x + A_W) / 2 if title_x is None else title_x, y + 3.0, title, 1.8, weight="bold", color=BLUE)
    else:
        drawing.text(x + 3.1, title_y, title, 1.8, weight="bold", color=BLUE, rotate=-90)


def token_grid(drawing, x, y, size, n, highlight=None, style=ENCODER_STYLE):
    """Draw one transformer block as a square grid of n x n tokens; highlight is the side of an attention window
    in tokens, or n for global attention."""
    fill, edge = style
    lines = "".join(
        f"M{fmt(x + size * k / n)},{fmt(y)}v{fmt(size)}M{fmt(x)},{fmt(y + size * k / n)}h{fmt(size)}"
        for k in range(1, n)
    )
    parts = [
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(size)}" height="{fmt(size)}" fill="{fill}" stroke="{edge}" '
        'stroke-width="0.15"/>',
        f'<path d="{lines}" stroke="{edge}" stroke-width="0.05" stroke-opacity="0.8"/>',
    ]
    if highlight:
        w = size * highlight / n
        offset = 0.0 if highlight == n else size / n
        parts.append(
            f'<rect x="{fmt(x + offset)}" y="{fmt(y + offset)}" width="{fmt(w)}" height="{fmt(w)}" fill="{BLUE}" '
            f'fill-opacity="0.22" stroke="{BLUE}" stroke-width="0.2"/>'
        )
    drawing.add("".join(parts))


def hiera(drawing, cx, top):
    """Draw the Hiera stages from top to bottom: transformer blocks on token grids that pooling halves from stage to
    stage (the arrows between the stages, without a label, user feedback 2026-11-09). The early stages attend within
    local windows, the late stages globally, and a neck right next to the stages merges all of them. Return the bottom
    of the stages."""
    shift, y, links = 0.5, top, []
    widest = max(size + (layers - 1) * shift for _, size, layers, _ in HIERA_STAGES)
    neck_x = cx + widest / 2 + 2.0
    for k, (n, size, layers, is_global) in enumerate(HIERA_STAGES):
        width = size + (layers - 1) * shift
        x = cx - width / 2
        y += (layers - 1) * shift
        for layer in range(layers):
            last = layer == layers - 1
            window = (n if is_global else n // 4) if last else None
            token_grid(drawing, x + layer * shift, y - layer * shift, size, n, highlight=window)
        links.append((x + width, y + size / 2))
        y += size
        if k < len(HIERA_STAGES) - 1:
            flow_arrow(drawing, [(cx, y + 0.3), (cx, y + HIERA_GAP - 0.3)], head=0.55, width=0.2)
            y += HIERA_GAP
    neck_top, neck_bottom = top - 0.5, y + 0.5
    fill, edge = ENCODER_STYLE
    drawing.add(
        f'<rect x="{fmt(neck_x)}" y="{fmt(neck_top)}" width="3.6" height="{fmt(neck_bottom - neck_top)}" rx="0.6" '
        f'fill="{fill}" stroke="{edge}" stroke-width="0.2"/>'
    )
    # the rotated text reads upwards with its letters left of its baseline, so the baseline moves right by half the
    # height of the letters to center the word in the neck
    drawing.text(neck_x + 1.8 + 0.36 * drawing.text_size(1.5), (neck_top + neck_bottom) / 2, "neck", 1.5, rotate=-90)
    for x, y_link in links:
        flow_arrow(drawing, [(x + 0.3, y_link), (neck_x - 0.2, y_link)], head=0.5, width=0.18)
    return y


def attention_legend(drawing, x, y):
    """Explain the two kinds of attention of the Hiera stages with small token grids."""
    for k, (highlight, text) in enumerate(((1, "local attention"), (4, "global attention"))):
        gy = y + k * 2.8
        token_grid(drawing, x, gy - 1.5, 2.2, 4, highlight=highlight)
        drawing.text(x + 2.8, gy + 0.1, text, 1.3, color=SUBTEXT, anchor="start")


def draw_trunk(drawing):
    """Draw the inputs, the hierarchical image encoder and the embeddings, which both heads share. The image and the
    volume stack in one column; the image encoder takes a volume slice by slice."""
    # the 2D embedding lines up with the entry of the automatic head, so that its line runs straight through
    cy_2d, cy_3d = A_ENTRY, A_TRUNK_3D
    # the inputs are as large as the column allows (user feedback 2026-11-09: larger images and volumes)
    column = 15.4
    drawing.text(column / 2, A_HEADER_Y, "Input", 2.12, weight="bold")
    scaled(drawing, INPUT_SIZE / SIZE, slice_input, 0.3, cy_2d)
    item_label(drawing, 0.0, column, cy_2d + INPUT_SIZE / 2 + 2.4, "2D image")
    height = place_render(drawing, render_em_block(), 0.4, cy_3d, INPUT_SIZE)
    item_label(drawing, 0.0, column, cy_3d + height / 2 + 2.4, "3D volume")
    # the Hiera encoder, a hierarchical vision transformer, downsamples each image or slice in stages
    (x0, x1), top = A_ENCODER_X, 1.5
    box_index = len(drawing.body)
    drawing.text((x0 + x1) / 2, A_HEADER_Y, "Image encoder", 2.12, weight="bold")
    # the box fits the stages, its legend and its caption, and still takes the arrow of the volume
    bottom = max(hiera(drawing, x0 + 7.8, top + 7.5) + 6.6, cy_3d + 6.0) + 2.0
    fill, edge = ENCODER_STYLE
    drawing.body.insert(box_index, (
        f'<rect x="{fmt(x0)}" y="{fmt(top)}" width="{fmt(x1 - x0)}" height="{fmt(bottom - top)}" rx="1.2" '
        f'fill="{mix(fill, "#ffffff", 0.55)}" stroke="{edge}" stroke-width="0.25"/>'
    ))
    attention_legend(drawing, x0 + 1.4, bottom - 4.5)
    # the name of the encoder stands under its box (user feedback 2026-11-09)
    item_label(drawing, x0, x1 - x0, bottom + 2.3, "Hierarchical ViT (Hiera)", size=1.5)
    flow_arrow(drawing, [(column + 0.6, cy_2d), (x0 - 0.7, cy_2d)])
    flow_arrow(drawing, [(column + 0.6, cy_3d), (x0 - 0.7, cy_3d)])
    emb_x = x1 + 3.5
    for cy in (cy_2d, cy_3d):
        flow_arrow(drawing, [(x1 + 0.6, cy), (emb_x - 0.6, cy)])
    drawing.text(emb_x + 5.9, A_HEADER_Y, "Embeddings", 2.12, weight="bold")
    embedding_slab(drawing, emb_x, cy_2d + 1.5, 8.0, 5.0)
    item_label(drawing, emb_x, 11.75, cy_2d + 6.0, "1 × embedding")
    for k in range(len(SLICE_INDICES)):
        embedding_slab(drawing, emb_x, cy_3d + 4.0 - 0.95 * k, 8.0, 5.0)
    item_label(drawing, emb_x, 11.75, cy_3d + 8.6, "Z × embeddings")
    # both embeddings go to the automatic head on the right and down to the interactive head
    bus_x = emb_x + 15.0
    drawing.add(
        f'<path d="M{fmt(emb_x + 12.0)},{fmt(cy_2d)}H{fmt(bus_x)}M{fmt(emb_x + 12.0)},{fmt(cy_3d)}H{fmt(bus_x)}'
        f'M{fmt(bus_x)},{fmt(cy_2d)}V{fmt(INTER_Y)}" fill="none" stroke="{ARROW}" stroke-width="0.25"/>'
    )
    return bus_x, (top, bottom)


def recycle_icon(drawing, cx, cy, r, color):
    """Two arrows that chase each other around a circle: the memory is used again and again."""
    for start in (20.0, 200.0):
        a0, a1 = np.radians(start), np.radians(start + 130.0)
        x0, y0 = cx + r * np.cos(a0), cy + r * np.sin(a0)
        x1, y1 = cx + r * np.cos(a1), cy + r * np.sin(a1)
        drawing.add(
            f'<path d="M{fmt(x0)},{fmt(y0)}A{fmt(r)},{fmt(r)} 0 0 1 {fmt(x1)},{fmt(y1)}" fill="none" '
            f'stroke="{color}" stroke-width="0.3" stroke-linecap="round"/>'
        )
        # the head points along the circle, clockwise
        tangent = np.array([-np.sin(a1), np.cos(a1)])
        normal = np.array([np.cos(a1), np.sin(a1)])
        tip = np.array([x1, y1]) + 0.55 * tangent
        left, right = tip - 0.75 * tangent + 0.38 * normal, tip - 0.75 * tangent - 0.38 * normal
        drawing.add(
            f'<path d="M{fmt(tip[0])},{fmt(tip[1])}L{fmt(left[0])},{fmt(left[1])}L{fmt(right[0])},{fmt(right[1])}Z" '
            f'fill="{color}"/>'
        )


def memory_bank(drawing, cx, cy, size=5.6, gap=0.7, step=14, squeeze=0.45, shear=-0.32):
    """Draw the memory bank centered at (cx, cy) on a card with rounded corners: a recycle sign, the masks of the
    slices that were already processed, standing side by side like books with a small gap and facing left, the way
    the memory flows, and its label. Returns the left and right edges of the card."""
    labels = neuron_volume()
    # the prompted middle slice stands in front, the slices that the memory reached from it behind
    indices = [MIDDLE, MIDDLE + step, MIDDLE - step, MIDDLE + 2 * step]
    w = size * squeeze
    shift = w + gap
    icon_r, pad = 1.5, 1.4
    label_w = max(text_width("memory bank", drawing.text_size(1.25), bold=True),
                  text_width("masks of processed slices", drawing.text_size(1.2)))
    slices_w = (len(indices) - 1) * shift + w
    total = pad + 2 * icon_r + 1.2 + slices_w + 1.2 + label_w + pad
    x0, x1 = cx - total / 2, cx + total / 2
    y0, y1 = cy - size / 2 - 1.6, cy + size / 2 + 1.6
    fill, edge = INTERACTIVE_STYLE
    drawing.add(
        f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(total)}" height="{fmt(y1 - y0)}" rx="1.6" '
        f'fill="{mix(fill, "#ffffff", 0.6)}" stroke="{edge}" stroke-width="0.25"/>'
    )
    recycle_icon(drawing, x0 + pad + icon_r, cy, icon_r, mix(edge, "#000000", 0.2))
    x = x0 + pad + 2 * icon_r + 1.2
    top = cy - size / 2 - shear * size * squeeze / 2
    # the slices further right stand behind, so they are drawn first
    for k, index in reversed(list(enumerate(indices))):
        # the memory keeps the masks themselves, shown in pastel red on a light background
        mask = labels[index] == NEURON_LABEL
        image = np.where(mask[..., None], hex_to_rgb("#ec8a84"), hex_to_rgb("#f4f1f6"))
        matrix = ",".join(fmt(v) for v in (squeeze, shear * squeeze, 0, 1, x + k * shift, top))
        drawing.add(
            f'<g transform="matrix({matrix})"><image width="{fmt(size)}" height="{fmt(size)}" '
            f'preserveAspectRatio="none" xlink:href="{array_uri(image)}"/><rect width="{fmt(size)}" '
            f'height="{fmt(size)}" fill="none" stroke="{SLICE_STYLE[1]}" stroke-width="0.3"/></g>'
        )
    label_x = x + slices_w + 1.2
    drawing.text(label_x, cy - 0.3, "memory bank", 1.25, weight="bold", color=TEXT, anchor="start")
    drawing.text(label_x, cy + 1.7, "masks of processed slices", 1.2, color=SUBTEXT, anchor="start")
    return x0, x1


def apg_frame():
    """The dashed frame of the APG stack (x0, y0, x1, y1), centered on the APG arrow above the prompt encoder, the
    left edge of the stack and the center height of its front sheet."""
    line_x, shift = PROMPT_ENCODER_X + PROMPT_ENCODER_W / 2, 3 * 0.85 * 9.0 / SIZE
    apg_cy = INTER_Y + 9.6
    width = 9.0 + shift
    stack_x = line_x - width / 2
    frame = (stack_x - 0.7, apg_cy - 4.5 - shift - 0.7, stack_x + width + 0.7, apg_cy + 4.5 + 0.7)
    return frame, stack_x, apg_cy


def draw_interactive_head(drawing, bus_x):
    """Draw the interactive band below the trunk: the prompts below the inputs, the prompt encoder below the image
    encoder, the memory attention that propagates a mask through volumes, and the mask decoder."""
    y0 = INTER_Y
    a_band(drawing, 0.0, y0, INTER_H, "Interactive segmentation", INTERACTIVE_BACKGROUND, y0 + INTER_H / 2,
           horizontal=True)
    decoder_x = A_OUTPUT_X - 30.0
    decoder_right = decoder_x + 13.5
    module(
        drawing, decoder_x, y0 + 1.5, decoder_right - decoder_x, INTER_H - 3.0, ["Mask", "decoder"], INTERACTIVE_STYLE
    )
    # 2D embeddings go straight to the mask decoder, the slices of a volume pass the memory attention first
    # the 3D path goes straight down from the embeddings (user, 2026-11-09): no step between the 2D and the 3D path
    split, bypass, memory = bus_x, y0 + 0.14 * INTER_H, y0 + 0.34 * INTER_H
    drawing.add(
        f'<path d="M{fmt(bus_x)},{fmt(y0 - A_BAND_GAP)}V{fmt(bypass)}H{fmt(split)}M{fmt(split)},{fmt(bypass)}'
        f'V{fmt(memory)}" fill="none" stroke="{ARROW}" stroke-width="0.25" stroke-linejoin="round"/>'
    )
    flow_arrow(drawing, [(split, bypass), (decoder_x - 0.7, bypass)])
    drawing.text(split + 1.5, bypass - 0.8, "2D", 1.3, color=SUBTEXT, anchor="start")
    memory_x = split + 6.0
    flow_arrow(drawing, [(split, memory), (memory_x - 0.7, memory)])
    drawing.text(split + 1.5, memory - 0.8, "3D", 1.3, color=SUBTEXT, anchor="start")
    module(drawing, memory_x, memory - 3.6, 26.0, 7.2, ["Memory attention"], INTERACTIVE_STYLE)
    flow_arrow(drawing, [(memory_x + 26.7, memory), (decoder_x - 0.7, memory)])
    # the APG prompts of a volume take the memory path too (user feedback 2026-11-09): dashed lines next to it
    apg = dict(color=TEXT, dash="0.9,0.5")
    flow_arrow(drawing, [(memory_x + 26.7, memory + 1.0), (decoder_x - 0.7, memory + 1.0)], **apg)
    # the masks of the slices go into a memory bank, which conditions the next slices
    # the loop runs left from the mask decoder, through the memory bank in its middle, and turns up with a round
    # corner into the memory attention
    loop_y, loop_x, bend = y0 + 0.72 * INTER_H, memory_x + 13.0, 2.0
    drawing.add(
        f'<path d="M{fmt(decoder_x - 0.3)},{fmt(loop_y)}H{fmt(loop_x + bend)}Q{fmt(loop_x)},{fmt(loop_y)} '
        f'{fmt(loop_x)},{fmt(loop_y - bend)}V{fmt(memory + 5.4)}" fill="none" stroke="{ARROW}" stroke-width="0.25"/>'
    )
    flow_arrow(drawing, [(loop_x, memory + 5.5), (loop_x, memory + 4.2)])
    # the dashed APG loop runs inside the solid loop, under the memory bank
    # with a round bend, as the solid loop
    dx, dy = loop_x + 1.0, loop_y - 1.0
    drawing.add(
        f'<path d="M{fmt(decoder_x - 0.3)},{fmt(dy)}H{fmt(dx + bend)}Q{fmt(dx)},{fmt(dy)} {fmt(dx)},{fmt(dy - bend)}'
        f'V{fmt(memory + 5.5)}" fill="none" stroke="{TEXT}" stroke-width="0.25" stroke-dasharray="0.9,0.5"/>'
    )
    flow_arrow(drawing, [(dx, memory + 5.5), (dx, memory + 4.2)], color=TEXT)
    memory_bank(drawing, (loop_x + decoder_x) / 2, loop_y)
    # prompts are placed on a 2D image, or on one slice of a volume, at the left
    row_2d, row_3d = y0 + A_ROW_2D, y0 + A_ROW_3D
    # the prompted volume sits lower, in the room at the bottom left of the band (user feedback 2026-11-09)
    in_row_3d = row_3d + 3.0
    prompt = 12.5  # the size of the prompted image
    # the prompted inputs move left into the room of the former vertical band title (user feedback 2026-11-09)
    in_2d, in_3d = 3.0, 2.0
    scaled(drawing, prompt / SIZE, slice_prompted, in_2d, row_2d)
    place_render(drawing, render_prompted_slice(), in_3d, in_row_3d, 1.2 * prompt)
    encoder_x, encoder_y, elbow = PROMPT_ENCODER_X, y0 + INTER_H / 2, 23.5
    # the APG prompts right above the prompt encoder: a positive point on each object of the image, with the slices
    # of a volume as translucent sheets behind it. The stack sits in a dashed frame on the APG arrow, which enters it
    # from above and leaves it into the prompt encoder, so that both read as one unit (user feedback 2026-11-09)
    (fx0, fy0, fx1, fy1), stack_x, apg_cy = apg_frame()
    scaled(drawing, 9.0 / SIZE, slice_apg_sheets, stack_x, apg_cy)
    flow_arrow(drawing, [((fx0 + fx1) / 2, fy1), ((fx0 + fx1) / 2, encoder_y - 5.0 - 0.7)], color=TEXT,
               dash="0.9,0.5")
    drawing.add(
        f'<rect x="{fmt(fx0)}" y="{fmt(fy0)}" width="{fmt(fx1 - fx0)}" height="{fmt(fy1 - fy0)}" rx="1.0" '
        f'fill="none" stroke="{TEXT}" stroke-width="0.25" stroke-dasharray="0.9,0.5"/>'
    )
    module(drawing, encoder_x, encoder_y - 5.0, PROMPT_ENCODER_W, 10.0, ["Prompt", "encoder"], INTERACTIVE_STYLE)
    inputs = ((in_2d + 0.6 + prompt, row_2d, encoder_y - 2.0), (in_3d + 0.6 + 1.2 * prompt, in_row_3d, encoder_y + 2.0))
    for start, row, end in inputs:
        flow_arrow(drawing, [(start, row), (elbow, row), (elbow, end), (encoder_x - 0.7, end)])
    # the prompt embeddings run below the memory to the mask decoder
    lane = y0 + INTER_H - 2.5
    flow_arrow(drawing, [(encoder_x + 8.5, encoder_y + 5.6), (encoder_x + 8.5, lane), (decoder_x - 0.7, lane)])
    # the APG prompts take the same way: a dashed line next to it continues the APG arrow from the prompt encoder to
    # the mask decoder (user feedback 2026-11-09)
    apg_lane = lane - 1.0
    flow_arrow(drawing, [(encoder_x + 9.5, encoder_y + 5.6), (encoder_x + 9.5, apg_lane), (decoder_x - 0.7, apg_lane)],
               color=TEXT, dash="0.9,0.5")
    # the outputs in two rows (user feedback 2026-11-09): the promptable segmentation of an image and a volume on
    # top, the automatic segmentation from the APG prompts below, each row with the 2D output left of the 3D output
    tile, width_3d = 10.5, 12.0
    # the outputs keep to the right edge, so that the arrows from the mask decoder get room
    x_3d = A_W - width_3d - 0.3
    x_2d = x_3d - tile - 2.5
    # the interactive outputs: the 2D output left of the 3D output
    row = row_2d
    flow_arrow(drawing, [(decoder_right + 0.7, row), (x_2d - 0.6, row)])
    scaled(drawing, tile / SIZE, slice_interactive, x_2d, row)
    height = place_render(drawing, render_neuron(), x_3d, row, width_3d)
    top, bottom = row - max(tile, height) / 2, row + max(tile, height) / 2 + 2.0
    drawing.text((x_2d + x_3d + width_3d) / 2, top - 1.0, "Interactive segmentation", 1.3, weight="bold",
                 color=SUBTEXT)
    drawing.text(x_2d + tile / 2, bottom, "2D", 1.5, color=SUBTEXT)
    drawing.text(x_3d + width_3d / 2, bottom, "3D", 1.5, color=SUBTEXT)
    # the APG output: one larger stack of slices, as the APG input above the prompt encoder (user feedback
    # 2026-11-09), a little lower than the 3D input row
    row, stack = row_3d + 3.5, 13.0
    shift = 3 * 0.85 * stack / SIZE
    x_stack = (x_2d + A_W - 0.3) / 2 - (stack + shift) / 2
    flow_arrow(drawing, [(decoder_right + 0.7, row), (x_stack - 0.6, row)], color=TEXT, dash="0.9,0.5")
    scaled(drawing, stack / SIZE, slice_apg_output_sheets, x_stack, row)
    drawing.text(x_stack + (stack + shift) / 2, row - stack / 2 - shift - 1.0, "APG: automatic segmentation", 1.3,
                 weight="bold", color=SUBTEXT)
    drawing.text(x_stack + stack / 2, row + stack / 2 + 2.0, "2D / 3D", 1.5, color=SUBTEXT)


def draw_automatic_head(drawing, bus_x, encoder_box):
    """Draw the automatic band: the segmentation decoder, as tall as its stages, and the maps it predicts for an
    image and for a volume, with the instance segmentation made from each: the 2D maps right of the decoder, the 3D
    maps in a row below it."""
    # the title at the top inside the band, as in the interactive band (user feedback 2026-11-09); the decoder box
    # takes the top left of the band, so the title is centered in the room right of it
    a_band(drawing, A_BAND_X, 0.0, AUTO_BAND_H, "Automatic segmentation", AUTOMATIC_BACKGROUND, AUTO_BAND_H / 2 + 4.0,
           horizontal=True, title_x=(A_CONTENT_X + 40.0 + A_OUTPUT_X) / 2 + 2.0)
    top = encoder_box[0]
    # the box ends a little below the largest feature map (drawn at 0.85 of its height and 0.75 of its depth)
    bottom = A_ENTRY + max(0.85 * h / 2 + 0.75 * d for _, h, d in STAGES) + 2.0
    x0, x1 = A_CONTENT_X, A_CONTENT_X + 40.0
    fill, edge = AUTOMATIC_STYLE
    drawing.add(
        f'<rect x="{fmt(x0)}" y="{fmt(top)}" width="{fmt(x1 - x0)}" height="{fmt(bottom - top)}" rx="1.2" '
        f'fill="{mix(fill, "#ffffff", 0.55)}" stroke="{edge}" stroke-width="0.25"/>'
    )
    drawing.text((x0 + x1) / 2, A_HEADER_Y, "Segmentation decoder", 2.12, weight="bold")
    flow_arrow(drawing, [(bus_x, A_ENTRY), (x0 + 1.3, A_ENTRY)])
    # the decoder upsamples the embeddings back to the image resolution: an image passes straight from stage to
    # stage, a volume takes the detour through the DSP block after each stage
    cy, x, markers = A_ENTRY, x0 + 2.0, []
    # the DSP circles are wide enough for their name at the smallest print size
    gap, drop, marker = 5.6, 3.8, 2.3
    for w, h, d in STAGES[::-1]:
        x = feature_map(drawing, x, cy, w, 0.85 * h, 0.75 * d, AUTOMATIC_STYLE)
        mx = x + gap / 2
        markers.append(mx)
        flow_arrow(drawing, [(x + 0.25, cy), (x + gap - 0.3, cy)], head=0.5, width=0.2)
        drawing.add(
            f'<path d="M{fmt(x + 0.8)},{fmt(cy)}V{fmt(cy + drop)}H{fmt(mx - marker)}'
            f'M{fmt(mx + marker)},{fmt(cy + drop)}H{fmt(x + gap - 0.8)}V{fmt(cy)}" fill="none" stroke="{ARROW}" '
            'stroke-width="0.2" stroke-linejoin="round" stroke-dasharray="0.35,0.3"/>'
        )
        x += gap
    # a key in the lower left corner of the decoder: the solid path is taken by images and volumes, the dashed path
    # through the DSP blocks only by volumes (user feedback 2026-11-09: DSP is applied only in 3D)
    # Only the dashed line is named (user, 2026-11-09: no "2D + 3D" key).
    for dash, text in (("0.35,0.3", "3D only: DSP = Depth Strip Pooling"),):
        ky = bottom - 1.2
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        drawing.add(
            f'<path d="M{fmt(x0 + 1.2)},{fmt(ky - 0.45)}H{fmt(x0 + 4.2)}" fill="none" stroke="{ARROW}" '
            f'stroke-width="0.22"{dashed}/>'
        )
        drawing.text(x0 + 4.8, ky, text, 1.3, color=SUBTEXT, anchor="start")
    for mx in markers:
        drawing.add(
            f'<circle cx="{fmt(mx)}" cy="{fmt(cy + drop)}" r="{fmt(marker)}" fill="#ffffff" stroke="{edge}" '
            'stroke-width="0.22"/>'
        )
        # darker than the outline, for a contrast above 4.5 on white
        drawing.text(mx, cy + drop + 0.55, "DSP", 0.85, weight="bold", color=mix(edge, "#000000", 0.4))
    # the outputs of the decoder, computed with the training targets of micro-sam; z only exists in volumes
    maps_2d = decoder_maps()[0]
    names = ["foreground", "boundary", "x-distance", "y-distance", "z-distance"]
    front, depth, map_gap = 4.8, 1.6, 0.7
    pitch, mx = front + depth + map_gap, x1 + 4.0
    row_2d, row_3d = A_ENTRY, AUTO_ROW_3D
    # the four 2D maps stretch over the whole row, the five 3D maps sit in their own rows; each map is named
    width = len(names) * pitch - map_gap
    tile = (width - 3 * map_gap) / 4
    for k, name in enumerate(names[:4]):
        tx = mx + k * (tile + map_gap)
        drawing.text(tx + tile / 2, row_2d - tile / 2 - 1.2, name, 1.3, color=SUBTEXT)
        map_tile(drawing, tx, row_2d - tile / 2, tile, maps_2d[name])
    maps_end = mx + width
    # the five 3D maps sit in one row below the decoder, from its left edge to the end of the 2D maps, each named
    # below it; the volume branches off the arrow to the 2D maps, just right of the decoder, down to them
    # the blocks are a little larger and closer, and the row ends before the 2D maps do, so that the arrow to the 3D
    # output gets room (user feedback 2026-11-09)
    # a little larger again and closer together, centered under the decoder and the 2D maps (user, 2026-11-09)
    block, spacing = 12.4, 1.6
    row_width = len(names) * block + (len(names) - 1) * spacing
    row_start = (x0 + 0.5 + maps_end - 4.0) / 2 - row_width / 2
    row_end = row_start + row_width
    for k, name in enumerate(names):
        tx = row_start + k * (block + spacing)
        height = place_render(drawing, render_map_block(name), tx, row_3d, block)
        drawing.text(tx + block / 2, row_3d + height / 2 + 2.0, name, 1.3, color=SUBTEXT)
    flow_arrow(drawing, [(x - 0.3, row_2d), (mx - 0.7, row_2d)])
    branch = x1 + 1.6
    flow_arrow(drawing, [(branch, row_2d), (branch, row_3d - block / 2 - 0.6)])
    x2, x3 = outputs(drawing, slice_automatic, automatic_3d_render, 0.0, row_2d=row_2d, row_3d=row_3d)
    # two ways from the maps to the instances (user feedback 2026-11-09): AIS post-processes the maps directly (the
    # solid arrows to the outputs), APG derives prompts from the maps and segments them with the interactive branch
    # (the dashed path from the maps down to the prompt encoder, as in 'automatic_prompt_generation.py')
    for row, start, end in ((row_2d, maps_end, x2), (row_3d, row_end, x3)):
        flow_arrow(drawing, [(start + 0.5, row), (end - 1.0, row)])
    drawing.text((maps_end + x2) / 2 - 0.4, row_2d - 1.0, "AIS", 1.15, weight="bold", color=SUBTEXT)
    apg_x = maps_end + 1.6
    gap_y = AUTO_BAND_H + A_BAND_GAP / 2
    target_x = PROMPT_ENCODER_X + PROMPT_ENCODER_W / 2
    # the arrow ends on the top of the dashed frame of the APG stack (see 'apg_frame')
    target_y = apg_frame()[0][1]
    # the path is drawn after the interactive band (see 'draw_apg'), which would cover its lower part
    # the path starts on the arrow to the 2D output and passes through the arrow to the 3D output, with a dot at
    # each, so that APG reads as taking the maps of both (user feedback 2026-11-09)
    return [(apg_x, row_2d), (apg_x, gap_y), (target_x, gap_y), (target_x, target_y)], [row_2d, row_3d]


def draw_apg(drawing, points, taps):
    """The dashed APG path from the decoder maps down to the prompt encoder, with its name along the line between
    the arrows to the 2D and the 3D output."""
    # as thick as the other arrows (user, 2026-11-09), darker, so that it stays visible on the interactive band
    # no arrow head: the line runs into the dashed frame of the APG stack, which leads on to the prompt encoder
    line = "M" + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in points)
    drawing.add(
        f'<path d="{line}" fill="none" stroke="{TEXT}" stroke-width="0.25" stroke-linejoin="round" '
        'stroke-dasharray="0.9,0.5"/>'
    )
    apg_x = points[0][0]
    for y in taps:
        drawing.add(f'<circle cx="{fmt(apg_x)}" cy="{fmt(y)}" r="0.55" fill="{TEXT}"/>')
    # the name runs along the dashed line between the two taps, as AIS names its arrow (user feedback 2026-11-09)
    drawing.text(apg_x - 0.7, (taps[0] + taps[1]) / 2, "APG", 1.3, weight="bold", color=SUBTEXT, rotate=-90)


def fetch_python_logo(path, url=PYTHON_LOGO_URL):
    """Download the Python logo to path, unless it is there already."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "micro-sam-figures"})
    with urllib.request.urlopen(request) as response:
        data = response.read()
    with open(path, "wb") as f:
        f.write(data)
    print(f"Downloaded the Python logo to {path}")


# panel b: example crops of the training data, one per dataset or imaging technique, grouped by domain
DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
# the number of datasets of the generalist model; the four LM datasets with both modalities count once here
B_TRAIN_DATASETS = 91
# the datasets of the generalist model per domain, for the slices of the pie chart; each of the four LM datasets with
# both modalities counts in the one of most of its training images (Omnipose and NeurIPS CellSeg in label-free, CTC and
# DeepBacs in fluorescence), so that the slices add up to all datasets
B_DOMAIN_DATASETS = {"fluorescence": 34, "label-free": 18, "em": 21, "histopathology": 18}
# the annotated objects of the datasets of the generalist model per domain: all annotated images and volumes on disk,
# in all splits and including the subsets that the loader leaves out (level C of figures/figure1/data_stats/full,
# counted 2026-11-05), without the EM mitochondria and nucleus data; the connectomes count the boxes on disk, LICONN
# (expansion microscopy, 83,837 objects) counts with electron microscopy, as in the text and the dataset counts
# (feedback 2026-11-09; full_stats.csv counts it as fluorescence), and XPRESS (X-ray, 51 objects) counts with
# electron microscopy, since the model trains on it in the EM pool
B_OBJECTS = {
    "fluorescence": 21653662 - 83837, "label-free": 4443510, "em": 2083391 + 83837, "histopathology": 1707969,
}
# the pixels of all 2D images and the voxels of all 3D volumes of the same data
B_PIXELS, B_VOXELS = 57086325840, 230487473654
# the domains with their titles, their quadrant (column, row) around the hub, and their fill and edge colors
B_DOMAINS = (
    ("fluorescence", "Fluorescence microscopy", (0, 0), ("#e2ece6", "#5f8a78")),
    ("label-free", "Label-free microscopy", (1, 0), ("#efe9dc", "#9a8256")),
    ("em", "Electron microscopy", (0, 1), ("#e6e4ee", "#736ca3")),
    ("histopathology", "Histopathology", (1, 1), ("#f1e2e8", "#8f6676")),
)
# the crops are resized to this size in pixels before they are drawn
B_TILE_PX = 256
# each quadrant holds a grid of tiles without the cell in its inner corner, which the pie chart covers
B_COLUMNS, B_ROWS = 3, 2
B_GAP, B_PAD, B_QUADRANT_GAP = 2.2, 1.6, 1.4
# the gap between the rows of tiles, larger than between the columns, for some air
B_ROW_GAP = 5.0
# the cards behind the tiles of volumes and movies reach this far to the right of their tiles, so the right margin of
# a quadrant is wider by it
B_CARD_REACH = 1.5
# the tiles are as large as needed for their captions at 5 pt; panel c takes the rest of the width
B_TILE = 12.6
B_CHANNEL_COLORS = {
    "gray": (1.0, 1.0, 1.0), "red": (1.0, 0.25, 0.2), "green": (0.2, 1.0, 0.35), "blue": (0.3, 0.45, 1.0),
    "magenta": (1.0, 0.2, 1.0), "cyan": (0.1, 0.9, 1.0), "yellow": (1.0, 0.9, 0.1),
}
# the example crops of the training data (see micro_sam/v2/datasets/generalist_loader.py); 'crop' is [y0, x0, size]
# in source pixels and 'index' selects the slice, frame or image in front of the two spatial axes
B_DATASETS = [
    dict(
        dataset="Omnipose", domain="fluorescence", caption=("widefield", "bacteria"),
        raw_path="omnipose/data/bact_fluor/train_sorted/cex/s_flex_cex_xy2_tile12_cyto.tif", raw_key=None,
        label_path="omnipose/data/bact_fluor/train_sorted/cex/s_flex_cex_xy2_tile12_cyto_masks.tif", label_key=None,
        crop=(0, 0, 508),
    ),
    dict(
        dataset="TissueNet", domain="fluorescence", caption=("multiplexed", "tissue"),
        raw_path="tissuenet/train/image_1100.zarr", raw_key="raw/rgb", label_key="labels/cell",
        channels=[(1, "green"), (0, "magenta")], channel_axis=0, crop=(0, 0, 512),
    ),
    dict(
        dataset="PlantSeg", domain="fluorescence", caption=("light-sheet", "plant root"), ndim=3,
        # the volumes are the time points of three movies; the cards behind the tile show further slices
        time_lapse=True,
        raw_path="plantseg/root_train/Movie1_t00049_crop_gt.h5", raw_key="raw", label_key="label", index=139,
        crop=(130, 280, 320),
    ),
    dict(
        dataset="DynamicNuclearNet", domain="fluorescence", caption=("widefield", "nuclei"), time_lapse=True,
        raw_path="dynamicnuclearnet/train/image_2520.zarr", raw_key="raw", label_key="labels", crop=(0, 0, 512),
        # the frames of a movie are consecutive files; 2520, 2522 and 2524 belong to the same movie
        frame_pattern="dynamicnuclearnet/train/image_{:04d}.zarr", frame=2520,
    ),
    # LICONN (expansion microscopy) counts as EM-like in the text and in the dataset accounting, so it is no
    # fluorescence example (feedback 2026-11-09); the C. elegans atlas, a confocal volume, takes its place
    dict(
        dataset="C. elegans atlas", domain="fluorescence", caption=("confocal", "C. elegans"), ndim=3,
        raw_path="celegans_atlas/c_elegans_nuclei/train/images/cnd1threeL1_1213061.tif", raw_key=None,
        label_path="celegans_atlas/c_elegans_nuclei/train/masks/cnd1threeL1_1213061.tif", label_key=None, index=70,
        crop=(25, 250, 90),
    ),
    dict(
        dataset="OrgLine", domain="label-free", caption=("brightfield", "organoids"),
        raw_path="orgline/data1/Intestine/train/20210727_38203_ckufhb5ev1aqr0y8e55bv30ld_img.jpg.h5", raw_key="image",
        label_key="masks", crop=(1488, 1104, 384),
    ),
    dict(
        dataset="NeurIPS CellSeg", domain="label-free", caption=("stained BF", "bone marrow"),
        raw_path="neurips_cellseg/Training-labeled/images/cell_00011.bmp", raw_key=None,
        label_path="neurips_cellseg/Training-labeled/labels/cell_00011_label.tiff", label_key=None, rgb=True,
        crop=(360, 360, 960),
    ),
    dict(
        dataset="NeurIPS CellSeg", domain="label-free", caption=("DIC", "cultured cells"),
        raw_path="neurips_cellseg/Training-labeled/images/cell_00318.tif", raw_key=None,
        label_path="neurips_cellseg/Training-labeled/labels/cell_00318_label.tiff", label_key=None,
        crop=(704, 1188, 352),
    ),
    dict(
        dataset="VICAR", domain="label-free", caption=("QPI", "cancer cells"),
        raw_path="vicar/labelled/PC3/00231_PC3_img.tif", raw_key=None,
        label_path="vicar/labelled/PC3/00231_PC3_mask.png", label_key=None, crop=(216, 0, 384),
    ),
    dict(
        dataset="Cell-ACDC", domain="label-free", caption=("phase contrast", "budding yeast"), time_lapse=True,
        raw_path=(
            "cell_acdc/test_data_budding_yeast/TimeLapse_2D/MIA_KC_htb1_mCitrine_labeled/Position_4/Images/"
            "19-03-2021_KCY050_SCGE_s04_phase_contr.tif"
        ),
        raw_key=None,
        label_path=(
            "cell_acdc/test_data_budding_yeast/TimeLapse_2D/MIA_KC_htb1_mCitrine_labeled/Position_4/Images/"
            "19-03-2021_KCY050_SCGE_s04_segm.tif"
        ),
        label_key=None, index=168, crop=(0, 24, 192),
    ),
    dict(
        dataset="CREMI", domain="em", caption=("ssTEM", "fly brain"), ndim=3, raw_path="cremi/sampleA.h5",
        raw_key="volumes/raw", label_key="volumes/labels/neuron_ids", index=24, crop=(672, 0, 384),
    ),
    dict(
        dataset="Platynereis cells", domain="em", caption=("SBEM", "platynereis"), ndim=3,
        raw_path="platynereis/membrane/train_data_membrane_01.n5", raw_key="volumes/raw/s1",
        label_key="volumes/labels/segmentation_corrected/v1/ignore_16777215/s1", index=64, crop=(96, 96, 384),
    ),
    dict(
        dataset="Wafer4", domain="em", caption=("mSEM", "cortex"), ndim=3,
        raw_path="wafer4/wafer4_inputs.h5", raw_key="main", label_path="wafer4/wafer4_labels.h5", label_key="main",
        index=40, crop=(400, 400, 384),
    ),
    dict(
        dataset="ASTIH", domain="em", caption=("SEM", "spinal cord"),
        raw_path="astih/SEM1/train/sub-rat4_sample-data12_SEM.h5", raw_key="raw", label_key="labels",
        instances=["class", 2], crop=(100, 100, 512),
    ),
    dict(
        dataset="Tumor spheroid EM", domain="em", caption=("SBF-SEM", "spheroid"),
        raw_path="tumor_spheroid_em/2d_manual/50-50-50/Au_01-vol_01-z_0034.h5", raw_key="raw",
        label_key="labels/cells", crop=(0, 0, 2000),
    ),
    dict(
        dataset="LyNSeC", domain="histopathology", caption=("IHC", "lymphoma"),
        raw_path="lynsec/data/ihc/images/302_l1.tif", raw_key=None, label_path="lynsec/data/ihc/labels/302_l1.tif",
        label_key=None, rgb=True, channel_axis=2, crop=(0, 0, 256),
    ),
    dict(
        dataset="SRSA-Net", domain="histopathology", caption=("IHC", "lung TMA"),
        raw_path="srsanet/IHC_TMA_dataset/images/fold1_122_CD20.png", raw_key=None,
        label_path="srsanet/IHC_TMA_dataset/preprocessed_labels/fold1_122_CD20.tif", label_key=None, rgb=True,
        channel_axis=2, crop=(0, 0, 256),
    ),
    dict(
        dataset="CryoNuSeg", domain="histopathology", caption=("cryosection", "thymus"),
        raw_path="cryonuseg/tissue images/Human_Thymus_01.tif", raw_key=None,
        label_path="cryonuseg/Annotator 1 (biologist)/label masks modify/Human_Thymus_01.tif", label_key=None, rgb=True,
        channel_axis=2, crop=(0, 256, 256),
    ),
    dict(
        dataset="SPATCH", domain="histopathology", caption=("Visium HD", "colon"),
        raw_path="spatch/visium_hd_coad/data/tile4.h5", raw_key="raw/rgb", label_key="labels/nuclei", rgb=True,
        channel_axis=0, crop=(64, 192, 256),
    ),
    dict(
        dataset="PUMA", domain="histopathology", caption=("H&amp;E", "melanoma"),
        raw_path="puma/train/preprocessed/training_set_primary_roi_081.h5", raw_key="raw",
        label_key="labels/instances/nuclei", rgb=True, channel_axis=0, crop=(768, 288, 256),
    ),
]


def open_array(path, key):
    """Open an array lazily from h5, zarr, n5 or tif files, or read a small image file."""
    extension = os.path.splitext(path.rstrip("/"))[1].lower()
    if extension in (".h5", ".hdf5", ".hdf", ".zarr", ".n5"):
        data = open_file(path, "r")
        return data if key is None else data[key]
    if extension in (".tif", ".tiff"):
        store = tifffile.imread(path, aszarr=True)
        data = zarr.open(store, mode="r")
        # pyramidal tifs open as a group of levels; the first level has the full resolution
        return data if hasattr(data, "shape") else data["0"]
    return imageio.imread(path)


def read_crop(path, key, index, crop, channel_axis=None):
    """Read the crop [y0, x0, size] of an image, with its channels last if it has a channel axis.

    The channel axis counts in the full array. The last two other axes are the spatial axes, and 'index' selects along
    the axes in front of them, such as the slice of a volume or the image of a stack.
    """
    data = open_array(path, key)
    ndim = len(data.shape)
    index = [] if index is None else list(index) if isinstance(index, (list, tuple)) else [index]
    # an axis beyond the index and the two spatial axes holds the channels: the last one if short, else the first one
    if channel_axis is None and ndim == len(index) + 3:
        channel_axis = -1 if data.shape[-1] <= 4 else 0
    if channel_axis is not None:
        channel_axis %= ndim
    others = [axis for axis in range(ndim) if axis != channel_axis]
    if len(index) != len(others) - 2:
        raise ValueError(f"The index {index} does not fit the shape {data.shape} of {path}.")
    y0, x0, size = crop
    box = [slice(None)] * ndim
    for axis, value in zip(others[:-2], index):
        box[axis] = value
    box[others[-2]], box[others[-1]] = slice(y0, y0 + size), slice(x0, x0 + size)
    crop = np.asarray(data[tuple(box)])
    if channel_axis is None:
        return crop
    # the indexed axes are gone, so the channel axis moves by the number of indexed axes in front of it
    shift = sum(axis < channel_axis for axis in others[:-2])
    return np.moveaxis(crop, channel_axis - shift, -1)


def percentile_normalize(image, low=1.0, high=99.5):
    image = image.astype("float32")
    lo, hi = np.percentile(image, [low, high])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def to_display(raw, spec):
    """Map a raw crop to rgb in [0, 1]: rgb images as they are, channels composited in their colors, gray otherwise."""
    if spec.get("rgb") and raw.ndim == 3:
        rgb = raw[..., :3].astype("float32")
        return rgb / 255.0 if 0 <= rgb.min() and rgb.max() <= 255 else percentile_normalize(rgb, 0.5, 99.8)
    if spec.get("channels"):
        rgb = np.zeros(raw.shape[:2] + (3,), dtype="float32")
        for channel, color in spec["channels"]:
            rgb += percentile_normalize(raw[..., channel])[..., None] * np.array(B_CHANNEL_COLORS[color])
        return np.clip(rgb, 0, 1)
    if raw.ndim == 3:
        raw = raw[..., 0]
    return np.repeat(percentile_normalize(raw)[..., None], 3, axis=-1)


def resize(image, size, nearest=False):
    if nearest:
        rows = (np.arange(size) * image.shape[0] / size).astype(int)
        cols = (np.arange(size) * image.shape[1] / size).astype(int)
        return image[rows][:, cols]
    channels = [
        np.asarray(Image.fromarray(image[..., c].astype("float32")).resize((size, size), Image.BILINEAR))
        for c in range(image.shape[-1])
    ]
    return np.clip(np.stack(channels, axis=-1), 0, 1)


# the ignore labels of the training data
IGNORE_LABELS = (2 ** 64 - 1, 2 ** 32 - 1, 2 ** 24 - 1, 2 ** 16 - 1, -1)


def instance_labels(labels, instances=None):
    """Get the instances of a label crop: as they are, as the connected components of the foreground ('components'),
    or as the connected components of one class (['class', k])."""
    labels = labels.astype("int64") if labels.dtype != np.uint64 else labels.astype("float64")
    labels[np.isin(labels, IGNORE_LABELS)] = 0
    labels = labels.astype("int64")
    if instances == "components":
        labels = ndimage.label(labels > 0)[0]
    elif instances is not None:
        labels = ndimage.label(labels == instances[1])[0]
    # consecutive ids keep the color table of the overlay small
    return np.unique(np.maximum(labels, 0), return_inverse=True)[1].reshape(labels.shape)


@lru_cache
def dataset_tile(k):
    """Load the crop of the k-th example and return its rgb image and its instance labels at the tile size."""
    spec = B_DATASETS[k]
    raw_path = os.path.join(DATA_ROOT, spec["raw_path"])
    label_path = os.path.join(DATA_ROOT, spec.get("label_path", spec["raw_path"]))
    raw = read_crop(raw_path, spec.get("raw_key"), spec.get("index"), spec["crop"], spec.get("channel_axis"))
    labels = read_crop(label_path, spec.get("label_key"), spec.get("label_index", spec.get("index")), spec["crop"])
    if labels.ndim == 3:
        labels = labels[..., 0]
    labels = instance_labels(labels, spec.get("instances"))
    rgb = resize(to_display(raw, spec), B_TILE_PX)
    labels = resize(labels, B_TILE_PX, nearest=True)
    return rgb, labels


@lru_cache
def neighbor_slice(k, step, spacing=4):
    """Load the image of the slice or frame 'step * spacing' away from the one of the k-th example, in the direction
    that stays inside the volume or movie, at a smaller size and a little darker, for the cards behind its tile."""
    spec = B_DATASETS[k]
    if "frame_pattern" in spec:
        # a movie stored as one file per frame: the later frames are the files with the next numbers
        path = os.path.join(DATA_ROOT, spec["frame_pattern"].format(spec["frame"] + step * spacing))
        raw = read_crop(path, spec.get("raw_key"), None, spec["crop"], spec.get("channel_axis"))
        return 0.85 * resize(to_display(raw, spec), B_TILE_PX // 2)
    path = os.path.join(DATA_ROOT, spec["raw_path"])
    depth = open_array(path, spec.get("raw_key")).shape[0]
    index = spec["index"] + step * spacing
    if index >= depth:
        index = spec["index"] - step * spacing
    raw = read_crop(path, spec.get("raw_key"), index, spec["crop"], spec.get("channel_axis"))
    return 0.85 * resize(to_display(raw, spec), B_TILE_PX // 2)


def label_overlay(rgb, labels, seed):
    """Lay the instances over the lower right half of the image, split by the diagonal, in the label colors."""
    n = rgb.shape[0]
    ids = np.unique(labels[labels > 0])
    rng = np.random.default_rng(seed)
    palette = neurite_colors(14)
    colors = np.zeros((int(labels.max()) + 1, 3))
    colors[ids] = palette[rng.integers(0, len(palette), len(ids))]
    rows, cols = np.mgrid[:n, :n]
    half = (rows + cols >= n)[..., None]
    gray = rgb.mean(axis=-1, keepdims=True)
    inside = (labels > 0)[..., None]
    boundary = (ndimage.grey_dilation(labels, size=3) != ndimage.grey_erosion(labels, size=3)) & (labels > 0)
    filled = np.where(inside, 0.4 * gray + 0.6 * colors[labels], 0.75 * gray)
    filled = np.where(boundary[..., None], 0.25 * gray + 0.75 * colors[labels] * 1.1, filled)
    return np.where(half, np.clip(filled, 0, 1), rgb)


def b_tile(drawing, x, y, size, k, spec):
    """Draw one example as a rounded tile, with a stack behind it for volumes and movies, and its caption on it in
    small text: the modality on the upper left, the tissue or organism on the lower right."""
    if spec.get("ndim", 2) == 3 or spec.get("time_lapse"):
        # the cards behind the tile show the slices further in the volume, or the frames later in the movie
        spacing = 2 if spec.get("time_lapse") else 4
        for step in (2, 1):
            o = 0.75 * step
            clip = drawing.clip_rect(x + o, y - o, size, size, 0.6)
            drawing.add(
                f'<image x="{fmt(x + o)}" y="{fmt(y - o)}" width="{fmt(size)}" height="{fmt(size)}" '
                f'preserveAspectRatio="none" clip-path="url(#{clip})" '
                f'xlink:href="{array_uri(neighbor_slice(k, step, spacing))}"/>'
                f'<rect x="{fmt(x + o)}" y="{fmt(y - o)}" width="{fmt(size)}" height="{fmt(size)}" rx="0.6" '
                f'fill="none" stroke="#ffffff" stroke-width="0.2"/>'
            )
    rgb, labels = dataset_tile(k)
    clip = drawing.clip_rect(x, y, size, size, 0.6)
    drawing.add(
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(size)}" height="{fmt(size)}" preserveAspectRatio="none" '
        f'clip-path="url(#{clip})" xlink:href="{array_uri(label_overlay(rgb, labels, k))}"/>'
        f'<path d="M{fmt(x + size)},{fmt(y)}L{fmt(x)},{fmt(y + size)}" stroke="#ffffff" stroke-width="0.15" '
        f'clip-path="url(#{clip})"/>'
    )
    modality, tissue = spec["caption"]
    tile_tag(drawing, x + 0.2, y + 0.2, modality, "start")
    tile_tag(drawing, x + size - 0.2, y + size - 2.2, tissue, "end")


def tile_tag(drawing, x, y, text, anchor):
    """Write a small text (5 pt) on a tile, white on a tight dark box with its top at y, from x to the right (start)
    or to the left (end)."""
    size = drawing.text_size(1.76)
    w = text_width(text, size) + 0.4
    x0 = x if anchor == "start" else x - w
    drawing.add(f'<rect x="{fmt(x0)}" y="{fmt(y)}" width="{fmt(w)}" height="2.0" rx="0.45" fill="{TEXT}" '
                'fill-opacity="0.72"/>')
    drawing.text(x0 + w / 2, y + 1.48, text, 1.76, color="#ffffff")


def format_count(value):
    """Format a number of objects in millions or thousands."""
    return f"{value / 1e6:.1f} M" if value >= 1e6 else f"{value / 1e3:.0f} k"


def quadrant_cells(column, row):
    """Get the grid cells of a quadrant in reading order, without the cell of its inner corner."""
    inner = (B_ROWS - 1 if row == 0 else 0, B_COLUMNS - 1 if column == 0 else 0)
    return [(r, c) for r in range(B_ROWS) for c in range(B_COLUMNS) if (r, c) != inner]


def arc_path(cx, cy, radius, start, stop):
    """Get the svg path of a circular arc, with the angles in degrees clockwise from the top."""
    a0, a1 = np.radians(start), np.radians(stop)
    x0, y0 = cx + radius * np.sin(a0), cy - radius * np.cos(a0)
    x1, y1 = cx + radius * np.sin(a1), cy - radius * np.cos(a1)
    large = 1 if stop - start > 180 else 0
    return f"M{fmt(x0)},{fmt(y0)}A{fmt(radius)},{fmt(radius)} 0 {large} 1 {fmt(x1)},{fmt(y1)}"


def b_pie(drawing, cx, cy, radius):
    """Draw the pie chart of the training data: a ring split into the domains by their number of datasets, in the
    edge colors of their quadrants, with the totals of the datasets, objects, pixels and voxels inside it."""
    counts = B_DOMAIN_DATASETS
    total = sum(counts.values())
    assert total == B_TRAIN_DATASETS
    width = 2.6
    ring = radius - width / 2 - 0.2
    # the white disc ends at the outer edge of the ring, so that the outlines of the cards run up to the ring and
    # merge into it (feedback 2026-11-09)
    drawing.add(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(ring + width / 2)}" fill="#ffffff"/>')
    # the slices run clockwise in the order of the quadrants: upper right, lower right, lower left, upper left; the
    # ring is turned so that each slice points roughly to its quadrant
    order = ("label-free", "histopathology", "em", "fluorescence")
    start = 20.0
    colors = {domain: mix(edge, "#000000", 0.15) for domain, _, _, (_, edge) in B_DOMAINS}
    angle = start
    for domain in order:
        span = 360.0 * counts[domain] / total
        drawing.add(
            f'<path d="{arc_path(cx, cy, ring, angle, angle + span)}" fill="none" stroke="{colors[domain]}" '
            f'stroke-width="{fmt(width)}"/>'
        )
        # the number of datasets sits in the middle of its slice, in white, named "datasets" so that it is clear what
        # it counts (feedback 2026-11-09); the text runs along the ring and reads from left to right
        middle = angle + span / 2
        turn = middle if middle % 360 < 90 or middle % 360 > 270 else middle - 180
        # the text curves along the ring (feedback 2026-11-09): each letter stands on the ring at its own angle,
        # clockwise on the upper half and counterclockwise on the lower half, so that it reads from left to right
        # (rsvg-convert does not draw svg text paths); the baseline sits a third of the letter height off the middle
        # of the ring, so that the letters are centered in it
        size = drawing.text_size(1.5)
        upper = turn == middle
        radius = ring - 0.36 * size if upper else ring + 0.36 * size
        label = f"{counts[domain]} datasets"
        widths = [text_width(c, size, bold=True) for c in label]
        position = -sum(widths) / 2
        for char, w in zip(label, widths):
            along = np.degrees((position + w / 2) / radius) * (1 if upper else -1)
            theta = middle + along
            lx, ly = cx + radius * np.sin(np.radians(theta)), cy - radius * np.cos(np.radians(theta))
            rot = theta if upper else theta - 180
            drawing.add(
                f'<text x="{fmt(lx)}" y="{fmt(ly)}" transform="rotate({fmt(rot)},{fmt(lx)},{fmt(ly)})" '
                f'style="{FONT};font-size:{fmt(size)}px;font-weight:bold;fill:#ffffff;text-anchor:middle">'
                f'{char}</text>'
            )
            position += w
        angle += span
    # white gaps between the slices
    angle = start
    for domain in order:
        t = np.radians(angle)
        x0, y0 = cx + (ring - width / 2) * np.sin(t), cy - (ring - width / 2) * np.cos(t)
        x1, y1 = cx + (ring + width / 2) * np.sin(t), cy - (ring + width / 2) * np.cos(t)
        drawing.add(f'<path d="M{fmt(x0)},{fmt(y0)}L{fmt(x1)},{fmt(y1)}" stroke="#ffffff" stroke-width="0.3"/>')
        angle += 360.0 * counts[domain] / total
    drawing.text(cx, cy - 3.0, "Training data", 2.12, weight="bold", color=BLUE)
    drawing.text(cx, cy - 0.7, f"{B_TRAIN_DATASETS} datasets", 1.94, weight="bold")
    drawing.text(cx, cy + 1.4, f"{format_count(sum(B_OBJECTS.values()))} objects", 1.76, color=SUBTEXT)
    drawing.text(cx, cy + 3.3, f"{B_PIXELS / 1e9:.0f} G pixels (2D)", 1.76, color=SUBTEXT)
    drawing.text(cx, cy + 5.2, f"{B_VOXELS / 1e9:.0f} G voxels (3D)", 1.76, color=SUBTEXT)


def b_size():
    """Get the size of a quadrant of panel b, the margin of the panel and the size of the panel. The domain names sit
    on the outer edges, so the quadrants need no header; the panel keeps a margin for the outer halves of the names."""
    width_q = 2 * B_PAD + B_CARD_REACH + B_COLUMNS * B_TILE + (B_COLUMNS - 1) * B_GAP
    height_q = 5.4 + B_ROWS * B_TILE + (B_ROWS - 1) * B_ROW_GAP
    margin = 0.8 * Drawing().text_size(2.12)
    return width_q, height_q, margin, 2 * width_q + B_QUADRANT_GAP, 2 * height_q + B_QUADRANT_GAP + 2 * margin


def build_panel_b():
    """Draw panel b in its own coordinates in mm: the training data as a world of four domains around a pie chart of
    their training objects. Each domain fills a quadrant with example crops of its datasets; each tile shows the
    image in its upper left half and the annotated instances in its lower right half. Returns the content and the
    size of the panel."""
    drawing = Drawing()
    pitch_x, pitch_y = B_TILE + B_GAP, B_TILE + B_ROW_GAP
    width_q, height_q, margin, width, height = b_size()
    gap = B_QUADRANT_GAP
    cx, cy = width / 2, height / 2
    tiles, boxes = [], []
    for domain, title, (column, row), (fill, edge) in B_DOMAINS:
        qx, qy = column * (width_q + gap), margin + row * (height_q + gap)
        drawing.add(
            f'<rect x="{fmt(qx)}" y="{fmt(qy)}" width="{fmt(width_q)}" height="{fmt(height_q)}" rx="1.2" '
            f'fill="#ffffff" stroke="{edge}" stroke-width="0.3" stroke-opacity="0.8"/>'
        )
        # the cards have no fill color, only their colored outline (feedback 2026-11-09)
        # the titles sit on the outer edges: on the upper edge of the top quadrants, on the lower edge of the bottom
        # ones, at their outer corners
        top = qy + (2.7 if row == 0 else 1.9)
        title_y = qy if row == 0 else qy + height_q
        title_x, anchor = (qx + B_PAD, "start") if column == 0 else (qx + width_q - B_PAD - B_CARD_REACH, "end")
        edge_label(drawing, title_x, title_y, title, 2.12, "#ffffff", edge, mix(edge, "#000000", 0.35), anchor=anchor)
        members = [(k, spec) for k, spec in enumerate(B_DATASETS) if spec["domain"] == domain]
        for (r, c), (k, spec) in zip(quadrant_cells(column, row), members):
            x, y = qx + B_PAD + c * pitch_x, top + r * pitch_y
            tiles.append((x, y, k, spec))
            boxes.append((x - 1.0, y - 1.6, x + B_TILE + 1.0, y + B_TILE + 0.4))
    # the pie chart reaches up to the nearest tile or caption
    halo = min(
        np.hypot(max(x0 - cx, 0, cx - x1), max(y0 - cy, 0, cy - y1)) for x0, y0, x1, y1 in boxes
    ) - 0.5
    # no white halo: the outlines of the cards run up to the ring of the pie chart and merge into it (feedback
    # 2026-11-09); the pie chart draws its own white disc
    for x, y, k, spec in tiles:
        b_tile(drawing, x, y, B_TILE, k, spec)
    b_pie(drawing, cx, cy, halo - 0.6)
    content = f'<defs>{shared_defs()}{"".join(drawing.defs)}</defs>{"".join(drawing.body)}'
    # the ids must differ from the ones of panels a and c, since all panels end up in the same figure
    return content.replace("f1c-", "f1b-"), width, height


def build_panel_a(width=A4_WIDTH):
    """Draw panel a, the architecture of μSAM2, in its own coordinates, scaled to the given width in mm (the text
    stays between MIN_TEXT and MAX_TEXT in print)."""
    drawing = Drawing(text_scale=A_TEXT_SCALE, scale=width / A_W)
    bus_x, encoder_box = draw_trunk(drawing)
    apg_path, apg_taps = draw_automatic_head(drawing, bus_x, encoder_box)
    draw_interactive_head(drawing, bus_x)
    draw_apg(drawing, apg_path, apg_taps)
    w3 = (SIZE + 3.5) * A_SCALE
    drawing.text(A_OUTPUT_X + w3 / 2, A_HEADER_Y, "Output", 2.12, weight="bold")
    body = f'<g transform="scale({fmt(width / A_W)})">{"".join(drawing.body)}</g>'
    content = f'<defs>{shared_defs()}{"".join(drawing.defs)}</defs>{body}'
    # the ids must differ from the ones of panel c, since both panels end up in the same figure
    return content.replace("f1c-", "f1a-")


def draw_items(drawing, cx, cy, names, scale):
    """Draw the items of one cell of panel c centered at (cx, cy) and enlarged by scale, each with its label right
    below it (the items are about as high as they are wide)."""
    x = cx - group_width(names, scale) / 2
    for name in names:
        draw, item_width, label = ITEMS[name]
        scaled(drawing, scale, draw, x, cy)
        drawing.text(x + scale * item_width / 2, cy + scale * item_width / 2 + 1.9, label, 1.6, color=SUBTEXT)
        x += scale * (item_width + ITEM_GAP)


def edge_label(drawing, x, y, text, size, fill, edge, color, anchor="middle"):
    """Draw a title stitched onto the edge of a card: a small pill of the card's colors, centered on the edge at y,
    half inside and half outside the card."""
    size = drawing.text_size(size)
    w, h = text_width(text, size, bold=True) + 2.6, 1.5 * size
    x0 = x - w / 2 if anchor == "middle" else (x if anchor == "start" else x - w)
    drawing.add(
        f'<rect x="{fmt(x0)}" y="{fmt(y - h / 2)}" width="{fmt(w)}" height="{fmt(h)}" rx="{fmt(h / 2)}" '
        f'fill="{fill}" stroke="{edge}" stroke-width="0.3"/>'
    )
    # the size is already the printed one, so it is passed on unchanged
    drawing.add(
        f'<text x="{fmt(x0 + w / 2)}" y="{fmt(y + 0.36 * size)}" style="{FONT};font-size:{fmt(size)}px;'
        f'font-weight:bold;fill:{color};text-anchor:middle">{text}</text>'
    )
    return w


def correction_badge(drawing, x, y, label_dy=4.2):
    """Draw the correction badge at (x, y), with its name in italics below it (or above it, for a negative
    label_dy)."""
    # the badge is 35% larger than first drawn (user, 2026-11-09): it is drawn at the origin and scaled into place
    start = len(drawing.body)
    loop_badge(drawing, (0, 0), r=2.1)
    correction_icon(drawing, (0, 0))
    inner = "".join(drawing.body[start:])
    del drawing.body[start:]
    drawing.add(f'<g transform="translate({fmt(x)},{fmt(y)}) scale(1.35)">{inner}</g>')
    label_dy *= 1.25
    size = drawing.text_size(1.76)
    # a white plate behind the name, so that the outlines of the bands do not run through it
    w = text_width("correction", size, bold=True) + 1.0
    drawing.add(
        f'<rect x="{fmt(x - w / 2)}" y="{fmt(y + label_dy - 0.85 * size)}" width="{fmt(w)}" height="{fmt(1.1 * size)}" '
        f'rx="{fmt(0.3 * size)}" fill="#ffffff"/>'
    )
    drawing.add(
        f'<text x="{fmt(x)}" y="{fmt(y + label_dy)}" style="{FONT};font-size:{fmt(size)}px;font-weight:bold;'
        f'font-style:italic;fill:{SUBTEXT};text-anchor:middle">correction</text>'
    )


# the rows of panel c: the shared label in the middle, the interactive items on the left and the automatic items on
# the right; the automatic classification runs as batch processing
C_ROWS = [
    ("Segmentation", ["interactive_2d", "interactive_3d"], ["automatic_2d", "automatic_3d"]),
    ("Tracking", ["interactive_2dt"], ["automatic_2dt", "automatic_3dt"]),
    ("Classification", ["pixel_classification", "object_classification"],
     ["automatic_batch", "automatic_batch_objects"]),
]


def build_panel_c(logo_path, napari_logo, python_logo, width, height):
    """Draw panel c in mm as a table: the name of each row rotated in a narrow column on the left, the interactive
    tasks in the left band and the automatic tasks in the right band, with the μSAM2 logo and its interfaces on top.

    Each row holds the same task in both bands, so the two modes of each task face each other; the correction runs
    above all rows from the automatic to the interactive band, since every automatic result can be corrected
    interactively. Returns the content, the size of the panel and its scale.
    """
    drawing = Drawing()
    title_size, label_size = 2.12, 2.12
    # the row names stand rotated in a narrow column on the left, so that the two bands take the rest of the width
    # and their items get larger (feedback 2026-11-09); a small gap separates the bands
    names_w, gap = 3.6, 1.6
    band = (width - names_w - gap) / 2
    lefts = (names_w, names_w + band + gap)
    cx = names_w + band + gap / 2
    # the pill of the ecosystem overlaps the upper edges of both bands, which both build on it; the titles of the bands
    # sit on their lower edges
    pill_y, bands_top = 4.0, 5.2
    bands_bottom = height - 0.8 * drawing.text_size(title_size)
    # the rows start a little below the pill, so that it has some air
    rows_top, rows_bottom = bands_top + 5.2, bands_bottom - 1.6
    row_h = (rows_bottom - rows_top) / len(C_ROWS)
    # the items take the largest scale at which the widest cell fits its band and the highest one its row
    widest = max(group_width(names) for _, *cells in C_ROWS for names in cells)
    scale = min((band - 2.5) / widest, (row_h - 3.6) / 12.0)
    bands = ((lefts[0], INTERACTIVE_STYLE[1], "Interactive analysis"),
             (lefts[1], AUTOMATIC_STYLE[1], "Automatic analysis"))
    # the bands have no fill color, only their colored outline (feedback 2026-11-09)
    for x0, edge, title in bands:
        drawing.add(
            f'<rect x="{fmt(x0 + 0.15)}" y="{fmt(bands_top)}" width="{fmt(band - 0.3)}" '
            f'height="{fmt(bands_bottom - bands_top)}" rx="1.2" fill="#ffffff" stroke="{edge}" stroke-width="0.3" '
            'stroke-opacity="0.8"/>'
        )
        edge_label(drawing, x0 + band / 2, bands_bottom, title, title_size, "#ffffff", edge, BLUE)
    draw_interfaces(drawing, cx, pill_y, logo_path, napari_logo, python_logo)
    # the correction badge stands between the two bands on their lower edge, without an arrow, named in italics:
    # the user corrects the automatic results interactively (feedback 2026-11-09)
    correction_badge(drawing, cx, bands_bottom, label_dy=-3.0)
    for k, (name, interactive, automatic) in enumerate(C_ROWS):
        top = rows_top + k * row_h
        # the items sit a little above the middle of the row, since their labels hang below them
        cy = top + row_h / 2 - 1.6
        draw_items(drawing, lefts[0] + band / 2, cy, interactive, scale)
        draw_items(drawing, lefts[1] + band / 2, cy, automatic, scale)
        drawing.text(names_w / 2 + 0.36 * drawing.text_size(label_size), cy, name, label_size, weight="bold",
                     color=TEXT, rotate=-90)
    content = f'<defs>{shared_defs()}{"".join(drawing.defs)}</defs>{"".join(drawing.body)}'
    return content, width, height, 1.0


def save_panel(content, output_folder, name, width, height):
    """Save the panel as a standalone svg and export it to pdf and png with rsvg-convert."""
    svg_path = os.path.join(output_folder, f"{name}.svg")
    with open(svg_path, "w") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{fmt(width)}mm" height="{fmt(height)}mm" viewBox="0 0 {fmt(width)} {fmt(height)}">'
            f"{content}</svg>\n"
        )
    # a fixed date keeps the pdf the same between runs
    env = dict(os.environ, SOURCE_DATE_EPOCH="0")
    base = os.path.join(output_folder, name)
    subprocess.run(["rsvg-convert", "-u", "-f", "pdf", "-o", f"{base}.pdf", svg_path], check=True, env=env)
    dpi = str(PNG_DPI)
    subprocess.run(
        ["rsvg-convert", "-u", "-f", "png", "-d", dpi, "-p", dpi, "-b", "white", "-o", f"{base}.png", svg_path],
        check=True, env=env,
    )
    # rsvg-convert writes no resolution into the png
    with Image.open(f"{base}.png") as image:
        image.load()
    image.save(f"{base}.png", dpi=(PNG_DPI, PNG_DPI))
    print(f"Saved {name} as svg, pdf and png in {output_folder}")


def build_figure(panels):
    """Put the panels together: panel a on top, panels b and c next to each other below it, and panel d below them,
    each with its letter. Returns the content and the size of the figure."""
    (a, _, a_h), (b, b_w, b_h), (c, c_w, c_h, c_scale) = panels["a"], panels["b"], panels["c"]
    c_x, row_y = LETTER_W + b_w + PANEL_GAP + LETTER_W, a_h + ROW_GAP
    if c_x + c_scale * c_w > FIGURE_W + 0.01:
        raise ValueError(f"Panel c is {c_x + c_scale * c_w - FIGURE_W:.1f} mm too wide for the figure.")
    parts = [
        f'<g transform="translate({fmt(LETTER_W)},0)">{a}</g>',
        f'<g transform="translate({fmt(LETTER_W)},{fmt(row_y)})">{b}</g>',
        f'<g transform="translate({fmt(c_x)},{fmt(row_y)}) scale({fmt(c_scale)})">{c}</g>',
    ]
    d, _, d_h = panels["d"]
    d_y = row_y + max(b_h, c_scale * c_h) + ROW_GAP
    parts.append(f'<g transform="translate({fmt(LETTER_W)},{fmt(d_y)})">{d}</g>')
    for letter, x, y in (("a", 0.0, 0.0), ("b", 0.0, row_y), ("c", c_x - LETTER_W, row_y), ("d", 0.0, d_y)):
        parts.append(
            f'<text x="{fmt(x)}" y="{fmt(y + LETTER_SIZE)}" style="{FONT};font-size:{fmt(LETTER_SIZE)}px;'
            f'font-weight:bold;fill:{TEXT}">{letter}</text>'
        )
    # a small margin keeps the outlines of the lowest panel inside the page
    height = d_y + d_h + 0.6
    width = max(FIGURE_W, LETTER_W + panels["d"][1])
    return '<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(parts), width, height


#
# Panel d: examples from the user studies, as studio renders: the objects in 3D with distinct colours for close
# objects, on a light grey floor with walls and shadows. The renders run off screen with pyvista (on a GPU node);
# the panels are put together from the saved renders with matplotlib.
#

D_LINE = "#1a1a1a"
# the top margin of each example of panel d in mm: room for its number
D_TOP = {"platynereis": 2.5, "spheroid": 2.5, "gruen": 2.5, "histai": 2.5}
D_START = time.time()
# the expanded Platynereis (Huisken lab): nuclei of the whole animal (s1 grid: z, y, x)
PLATY = os.path.join(DATA_ROOT, "huisken_data/exm_lsm/high_res/stitched/Expanded_Platynereis.ome.zarr")
PLATY_SHAPE = (770, 477, 1197)
PLATY_VOXEL_UM = 2.5112882406416843  # s1 voxel size, expanded
PLATY_EXPANSION = 10.0  # assumed expansion factor, for the pre-expansion scale bars
PLATY_SLAB_UM = 11.0  # section thickness, pre-expansion
# the upright oblique studio view: volume axes x, y, z = s1 axes 2, 1, 0; the round end (low x) on top. The window
# is twice the size of the saved render, for smooth edges and shadows; PLATY_RES scales all renders of the example,
# so that they print at more than 1200 dpi
PLATY_RES = 1.25
PLATY_SIZE = (round(3300 * PLATY_RES), round(6000 * PLATY_RES))
PLATY_BOUNDS = (0, PLATY_SHAPE[2], 0, PLATY_SHAPE[1], 0, PLATY_SHAPE[0])
PLATY_CENTER = np.array([PLATY_SHAPE[2], PLATY_SHAPE[1], PLATY_SHAPE[0]], float) / 2
PLATY_DIRECTION = np.array([-0.3, 1.0, -0.5]) / np.linalg.norm([-0.3, 1.0, -0.5])
PLATY_UP = (-1, 0, 0)
PLATY_ZOOM = 1.55
# the crop of the studio render (rows r0:r1, columns c0:c1): the animal, its shadows and its raw signal, with margins
PLATY_FRAME = tuple(round(v * PLATY_RES) for v in (334, 2639, 0, 1468))
# the section is seen from its normal (z, the studio camera side), tilted a little like the studio view
PLATY_DIRECTION_B = np.array([-0.15, 0.25, -1.0]) / np.linalg.norm([-0.15, 0.25, -1.0])
# the MCF7 spheroid (Beghin lab): the nuclei of the largest stack, DAPI in channel 0 of the raw
# the MCF7 spheroid with a nuclear (DAPI) and a membrane (E-cadherin) stain, and the micro-sam2 segmentations of
# its nuclei and its cells (feedback 2026-11-09: an organoid with cell and nuclear staining, both segmentations)
SPHEROID = os.path.join(DATA_ROOT, "beghin_data/mcf7_spheroids/dapi_ecadh_btub/Composite_s23")
SPHEROID_XY_UM, SPHEROID_Z_UM = 0.414, 1.5
# the heart section (Gruen lab): the normalized volume (z, y, x, channel), three crops and their cell masks
GRUEN = os.path.join(DATA_ROOT, "gruen_lab_data")
# no voxel size in the files: z from the 10 um section in 16 planes, xy estimated from the size of the nuclei
GRUEN_XY_UM, GRUEN_Z_UM = 0.12, 10.0 / 16
# the Ki67 core (HistAI, case_13673): OME-Zarr in a zip with the image pyramid (c, y, x), labels/instances and
# labels/classes (1: positive tumor nucleus, 2: negative tumor nucleus, 3: other cells); 0.5 um per level 0 px
HISTAI = os.path.join(DATA_ROOT, "histai_specimen_omezarr_2026-11-06", "case_13673.ome.zarr.zip")
# one crop of 256 x 256 px (128 um), smaller than first drawn (feedback and user, 2026-11-09)
HISTAI_UM, HISTAI_CROP = 0.5, 256
# the classes of the Ki67 labels, from the label metadata of the OME-Zarr, as (name, rgb)
HISTAI_CLASSES = [("positive", (214, 39, 40)), ("negative", (31, 90, 214)), ("others", (240, 180, 0))]
HISTAI_CLASS_RGB = np.array([[200, 200, 200], [214, 39, 40], [31, 90, 214], [240, 180, 0]], np.uint8)


def d_log(*args):
    print(*args, round(time.time() - D_START), flush=True)


def d_separate(labels):
    """Remove the voxels where two objects touch, so that each object becomes its own surface; also return the
    pairs of touching objects."""
    pairs = set()
    cut = np.zeros(labels.shape, bool)
    for axis in range(3):
        a = np.moveaxis(labels, axis, 0)
        lo, hi = a[:-1], a[1:]
        touch = (lo != hi) & (lo > 0) & (hi > 0)
        found = np.unique(np.stack([lo[touch], hi[touch]], 1), axis=0)
        pairs.update(map(tuple, found))
        c = np.moveaxis(cut, axis, 0)
        c[:-1] |= touch
        c[1:] |= touch
    labels = labels.copy()
    labels[cut] = 0
    return labels, pairs


def d_near_pairs(coarse, radius=2):
    """Pairs of objects closer than about radius voxels."""
    pairs = set()
    offsets = [(dz, dy, dx) for dz in range(0, radius + 1) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)
               if (dz, dy, dx) > (0, 0, 0) and dz * dz + dy * dy + dx * dx <= radius * radius + 1]
    nz, ny, nx = coarse.shape
    for dz, dy, dx in offsets:
        a = coarse[0:nz - dz, max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)]
        b = coarse[dz:nz, max(0, dy):ny - max(0, -dy), max(0, dx):nx - max(0, -dx)]
        hit = (a != b) & (a > 0) & (b > 0)
        code = np.unique(np.minimum(a[hit], b[hit]).astype("int64") * 100000 + np.maximum(a[hit], b[hit]))
        pairs.update(zip((code // 100000).tolist(), (code % 100000).tolist()))
    return pairs


def d_distinct_colors(n_labels, pairs, seed=1):
    """A palette of clear colours; each object gets a colour that is clearly different (in Lab) from the colours of
    its neighbours, so that close objects never look alike."""
    palette = []
    for i in range(60):
        hue = (i * 0.6180339887) % 1.0
        sat = (0.42, 0.58, 0.30)[i % 3]
        val = (0.97, 0.88, 0.78)[(i // 3) % 3]
        palette.append(colorsys.hsv_to_rgb(hue, sat, val))
    palette = np.array(palette)
    lab = rgb2lab(palette[None])[0]
    dist = np.linalg.norm(lab[:, None] - lab[None], axis=-1)
    neighbours = [[] for _ in range(n_labels + 1)]
    for a, b in pairs:
        neighbours[a].append(b)
        neighbours[b].append(a)
    rng = np.random.default_rng(seed)
    choice = np.full(n_labels + 1, -1)
    # the most connected objects first
    order = sorted(range(1, n_labels + 1), key=lambda i: (-len(neighbours[i]), rng.random()))
    for label in order:
        used = [choice[n] for n in neighbours[label] if choice[n] >= 0]
        # a random colour among those clearly different from all neighbours, else the most different one
        gap = dist[:, used].min(axis=1) if used else np.full(len(palette), np.inf)
        clear = np.flatnonzero(gap >= 30)
        choice[label] = int(rng.choice(clear)) if len(clear) else int(np.argmax(gap))
    choice[0] = 0
    return palette[choice]


def d_label_surfaces(labels):
    """Smooth surfaces of all objects in one pass; the label of each face is in cell_data["label"]."""
    grid = pv.ImageData(dimensions=labels.shape[::-1])
    grid.point_data["values"] = labels.ravel(order="C")
    nets = vtk.vtkSurfaceNets3D()
    nets.SetInputData(grid)
    nets.SetInputArrayToProcess(0, 0, 0, 0, "values")
    nets.GenerateLabels(int(labels.max()), 1, int(labels.max()))
    nets.SetOutputMeshTypeToTriangles()
    nets.SetOutputStyleToBoundary()
    nets.SmoothingOn()
    nets.SetNumberOfIterations(40)
    nets.Update()
    mesh = pv.wrap(nets.GetOutput())
    mesh.cell_data["label"] = np.asarray(mesh.cell_data["BoundaryLabels"]).max(axis=1)
    return mesh


def d_colored_mesh(labels, colors, z_scale=1.0):
    """The surfaces with one rgb colour per face; a colour map would be resampled to 256 colours and merge
    neighbouring ids."""
    mesh = d_label_surfaces(labels)
    mesh.points[:, 2] *= z_scale
    mesh.cell_data["rgb"] = (colors[mesh.cell_data["label"]] * 255).astype("uint8")
    return mesh


def d_add_objects(plotter, mesh):
    plotter.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.2, specular_power=20,
                     ambient=0.35, diffuse=0.75, show_scalar_bar=False)


def d_project(plotter, points):
    """Image coordinates (column, row) of world points, at the saved (half) image size."""
    renderer = plotter.renderer
    height = plotter.window_size[1]
    out = []
    for p in points:
        renderer.SetWorldPoint(*p, 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        out.append((x / 2, (height - y) / 2))
    return out


def d_px_per_unit(plotter, centre, step=50.0):
    """Pixels of the saved render per scene unit at the depth of the centre, along x."""
    a, b = d_project(plotter, [centre, np.asarray(centre) + np.array([step, 0, 0])])
    return float(np.hypot(b[0] - a[0], b[1] - a[1]) / step)


def d_screenshot(plotter, size, transparent=False):
    """Render, and scale the double size window down to the saved size, for smooth edges."""
    plotter.render()
    image = plotter.screenshot(return_img=True, transparent_background=transparent)
    return np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS))


def d_view(elevation, azimuth):
    """The direction from the scene to the camera, and its horizontal part."""
    flat = np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth)), 0.0])
    return np.cos(np.radians(elevation)) * flat + np.array([0, 0, np.sin(np.radians(elevation))]), flat


def d_quad(corners):
    """A rectangle from four corners, finely split so that the shadows look smooth."""
    return pv.PolyData(np.array(corners, float), faces=[4, 0, 1, 2, 3]).triangulate().subdivide(4)


def d_backdrop(plotter, bounds, view, margin, reach, height):
    """A floor under the scene (z up) and two walls behind it, on the sides away from the camera."""
    x0, x1, y0, y1, z0, z1 = bounds
    floor_z = z0 - margin
    wx = x0 - margin if view[0] > 0 else x1 + margin
    wy = y0 - margin if view[1] > 0 else y1 + margin
    fx = x1 + reach if view[0] > 0 else x0 - reach
    fy = y1 + reach if view[1] > 0 else y0 - reach
    top = z1 + height
    for plane in (d_quad([(wx, wy, floor_z), (fx, wy, floor_z), (fx, fy, floor_z), (wx, fy, floor_z)]),
                  d_quad([(wx, wy, floor_z), (wx, fy, floor_z), (wx, fy, top), (wx, wy, top)]),
                  d_quad([(wx, wy, floor_z), (fx, wy, floor_z), (fx, wy, top), (wx, wy, top)])):
        plotter.add_mesh(plane, color=(0.95, 0.95, 0.95), ambient=0.45, diffuse=0.6, specular=0.0)


def d_studio(size, bounds, view, distance, zoom):
    """A plotter with the studio lights and shadows (z up), the camera from the view direction onto the bounds."""
    x0, x1, y0, y1, z0, z1 = bounds
    middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    plotter = pv.Plotter(off_screen=True, window_size=(size[0] * 2, size[1] * 2), lighting="none")
    plotter.set_background("white")
    # a key light from above, a little to the side of the camera, and a soft fill light from the camera
    side = np.cross(view, [0, 0, 1.0])
    key = middle + distance * (0.5 * view + 0.45 * side / np.linalg.norm(side) + np.array([0, 0, 1.0]))
    plotter.add_light(pv.Light(position=tuple(key), focal_point=tuple(middle), intensity=0.7, positional=True,
                               cone_angle=50, light_type="scene light"))
    plotter.add_light(pv.Light(position=tuple(middle + 2 * distance * view), focal_point=tuple(middle),
                               intensity=0.55, light_type="scene light"))
    plotter.enable_shadows()
    plotter.renderer._render_passes._shadow_map_pass.GetShadowMapBakerPass().SetResolution(8192)
    plotter.camera.focal_point = middle
    plotter.camera.position = middle + 3 * distance * view
    plotter.camera.up = (0, 0, 1)
    plotter.camera.view_angle = 20
    plotter.reset_camera(bounds=bounds)
    plotter.camera.zoom(zoom)
    return plotter


def d_section_shot(add, centre, direction, up, key, bounds, size, zoom, transparent=False, points=()):
    """A render of a section from its camera direction, with ambient occlusion and no shadows; returns the image,
    the image positions of the points and the pixels per scene unit at the centre."""
    plotter = pv.Plotter(off_screen=True, window_size=(size[0] * 2, size[1] * 2), lighting="none")
    plotter.set_background("white")
    add(plotter)
    plotter.add_light(pv.Light(position=tuple(centre + key), focal_point=tuple(centre), intensity=0.7,
                               light_type="scene light"))
    plotter.add_light(pv.Light(position=tuple(centre + 3000 * direction), focal_point=tuple(centre),
                               intensity=0.55, light_type="scene light"))
    plotter.enable_ssao(radius=5, bias=0.05, kernel_size=128, blur=True)
    plotter.camera.focal_point = centre
    plotter.camera.position = centre + 3000 * direction
    plotter.camera.up = tuple(up)
    plotter.camera.view_angle = 20
    plotter.reset_camera(bounds=bounds)
    plotter.camera.zoom(zoom)
    image = d_screenshot(plotter, size, transparent)
    projected, ppu = d_project(plotter, points), d_px_per_unit(plotter, centre)
    plotter.close()
    return image, projected, ppu


def d_card(plotter, image, origin, step, thickness):
    """A raw card: a dark box with the image on its top face. Image row 0 is at the far (high y) edge; the box spans
    x in [ox, ox + width * step] and y in [oy - height * step, oy]."""
    h, w = image.shape[:2]
    ox, oy, oz = origin
    box = pv.Box(bounds=(ox, ox + w * step, oy - h * step, oy, oz, oz + thickness))
    plotter.add_mesh(box, color=(0.08, 0.08, 0.08), ambient=0.6, diffuse=0.4, specular=0.0)
    grid = pv.ImageData(dimensions=(w, h, 1), spacing=(step, step, 1.0),
                        origin=(ox + step / 2, oy - h * step + step / 2, oz + thickness + 0.5))
    grid.point_data["rgb"] = image[::-1].reshape(-1, 3)
    plotter.add_mesh(grid, scalars="rgb", rgb=True, ambient=1.0, diffuse=0.0, specular=0.0, show_scalar_bar=False)


def d_rectangle(bounds):
    """The four corners of a box in its middle z plane."""
    x0, x1, y0, y1, z0, z1 = bounds
    z = (z0 + z1) / 2
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def d_upsample(volume, factor, shape):
    """Nearest upsampling of a coarse volume into the given shape; a missing edge is filled from the last voxel."""
    up = volume
    for axis in range(3):
        up = np.repeat(up, factor, axis)
    up = np.pad(up, [(0, max(0, s - u)) for s, u in zip(shape, up.shape)], mode="edge")
    return up[:shape[0], :shape[1], :shape[2]]


def platy_raw_s0(cache):
    """The Platynereis raw at s0, divided by its smooth local level, scaled to uint8 (20th to 99.9th percentile) and
    masked to the body. It is large (3.5 GB), so it is cached."""
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    group = zarr.open_group(PLATY, mode="r")
    shape = tuple(2 * s for s in PLATY_SHAPE)
    coarse = zarr.open_group(PLATY + "/labels/nuclei", mode="r")["s2"][:] > 0
    body = ndimage.binary_dilation(ndimage.binary_closing(coarse, iterations=3), iterations=3)
    level = ndimage.gaussian_filter(group["s2"][:].astype("float32"), 12) + 1.0
    # the normalised volume in half precision, then the contrast limits from a sparse sample of it
    norm = np.zeros(shape, "float16")
    slab = 64
    for z0 in range(0, shape[0], slab):
        z1 = min(z0 + slab, shape[0])
        c0, c1 = z0 // 4, (z1 + 3) // 4
        block = group["s0"][z0:z1, :shape[1], :shape[2]].astype("float32")
        lev = d_upsample(level[c0:c1], 4, (c1 * 4 - c0 * 4,) + shape[1:])[z0 - c0 * 4:z1 - c0 * 4]
        mask = d_upsample(body[c0:c1], 4, (c1 * 4 - c0 * 4,) + shape[1:])[z0 - c0 * 4:z1 - c0 * 4]
        block = block / lev
        block[~mask] = np.nan
        norm[z0:z1, :block.shape[1], :block.shape[2]] = block
    lo, hi = np.nanpercentile(norm[::3, ::3, ::3].astype("float32"), [20, 99.9])
    raw = np.zeros(shape, "uint8")
    for z0 in range(0, shape[0], slab):
        block = norm[z0:z0 + slab].astype("float32")
        raw[z0:z0 + slab] = np.nan_to_num(np.clip((block - lo) / (hi - lo), 0, 1) * 255, nan=0).astype("uint8")
    del norm
    np.save(cache, raw)
    d_log("raw s0", raw.shape)
    return raw


def platy_backdrop(plotter):
    """A floor below the tail end and two walls behind the animal, in light grey."""
    x1, y1, z1 = PLATY_SHAPE[2], PLATY_SHAPE[1], PLATY_SHAPE[0]
    m, far, top = 60, 350, -250
    floor_x, wall_y, wall_z = x1 + m, -m, z1 + m
    # the floor spans from the walls to the front; the walls from the floor to above the round end
    front_y, front_z = y1 + far, -far
    for plane in (d_quad([(floor_x, wall_y, wall_z), (floor_x, front_y, wall_z), (floor_x, front_y, front_z),
                          (floor_x, wall_y, front_z)]),
                  d_quad([(top, wall_y, front_z), (top, wall_y, wall_z), (floor_x, wall_y, wall_z),
                          (floor_x, wall_y, front_z)]),
                  d_quad([(top, wall_y, wall_z), (top, front_y, wall_z), (floor_x, front_y, wall_z),
                          (floor_x, wall_y, wall_z)])):
        plotter.add_mesh(plane, color=(0.95, 0.95, 0.95), ambient=0.45, diffuse=0.6, specular=0.0)


def render_platynereis(out, raw_cache):
    """The renders of the Platynereis panel: the nuclei in the studio, an 11 um longitudinal section (the nuclei with
    their centre in a slab, and the raw of its middle plane), and close-ups of its anterior and posterior ends."""
    os.makedirs(out, exist_ok=True)
    nuclei = zarr.open_group(PLATY + "/labels/nuclei", mode="r")
    labels = nuclei["s1"][:]
    n_labels = int(labels.max())
    separated, pairs = d_separate(labels)
    pairs |= d_near_pairs(nuclei["s2"][:])
    colors = d_distinct_colors(n_labels, pairs)
    d_log("colours")
    # the centre of each nucleus, in s1 voxels (z, y, x)
    counts = np.zeros(n_labels + 1)
    sums = np.zeros((3, n_labels + 1))
    for z0 in range(0, labels.shape[0], 64):
        block = labels[z0:z0 + 64]
        zz, yy, xx = np.indices(block.shape, dtype="float32")
        flat = block.ravel()
        counts += np.bincount(flat, minlength=n_labels + 1)
        for axis, c in enumerate((zz + z0, yy, xx)):
            sums[axis] += np.bincount(flat, weights=c.ravel(), minlength=n_labels + 1)
    del labels
    cz, cy, cx = (sums / np.maximum(counts, 1))[:, 1:]
    ids = np.arange(1, n_labels + 1)
    # the slab: the nuclei with their centre within half the thickness of the middle of the body in z; the studio
    # camera sees this plane almost edge on
    half = PLATY_SLAB_UM * PLATY_EXPANSION / PLATY_VOXEL_UM / 2
    z_mid = float(np.median(cz))
    in_slab = np.abs(cz - z_mid) <= half
    slab_ids = ids[in_slab]
    layout = {"nuclei": n_labels, "slab_nuclei": int(len(slab_ids)), "slab_um": PLATY_SLAB_UM,
              "expansion": PLATY_EXPANSION, "voxel_um": PLATY_VOXEL_UM}

    # the studio, cropped to the frame, and the outline of the slab plane in it (from above the round end down to the
    # floor, like a sheet standing in the studio)
    plotter = pv.Plotter(off_screen=True, window_size=PLATY_SIZE, lighting="none")
    plotter.set_background("white")
    d_add_objects(plotter, d_colored_mesh(separated, colors))
    platy_backdrop(plotter)
    plotter.add_light(pv.Light(position=tuple(PLATY_CENTER + np.array([-2500.0, 1200.0, -1600.0])),
                               focal_point=tuple(PLATY_CENTER), intensity=0.7, positional=True, cone_angle=40,
                               light_type="scene light"))
    plotter.add_light(pv.Light(position=tuple(PLATY_CENTER + PLATY_DIRECTION * 3000 + np.array([0.0, 600.0, 0.0])),
                               focal_point=tuple(PLATY_CENTER), intensity=0.55, light_type="scene light"))
    plotter.enable_shadows()
    plotter.renderer._render_passes._shadow_map_pass.GetShadowMapBakerPass().SetResolution(8192)
    plotter.camera.focal_point = PLATY_CENTER
    plotter.camera.position = PLATY_CENTER + PLATY_DIRECTION * 4000
    plotter.camera.up = PLATY_UP
    plotter.camera.view_angle = 20
    plotter.reset_camera(bounds=PLATY_BOUNDS)
    plotter.camera.zoom(PLATY_ZOOM)
    image = d_screenshot(plotter, (PLATY_SIZE[0] // 2, PLATY_SIZE[1] // 2))
    r0, r1, c0, c1 = PLATY_FRAME
    Image.fromarray(image[r0:r1, c0:c1]).save(f"{out}/seg_studio.png")
    sub = np.where(np.isin(separated, slab_ids), separated, 0)
    del separated
    mesh = d_colored_mesh(sub, colors)
    del sub
    x0, x1, y0, y1, _, _ = mesh.bounds
    outline = d_project(plotter, d_rectangle((x0 - 40, x1 + 60, y0 - 25, y1 + 25, z_mid, z_mid)))
    layout["studio"] = {"slab_outline": [(c - c0, r - r0) for c, r in outline],
                        "px_per_voxel": d_px_per_unit(plotter, PLATY_CENTER)}
    plotter.close()
    d_log("studio")

    # the anterior (round end, low x) and posterior boxes, from the centres of the slab nuclei
    sx, sy = cx[in_slab], cy[in_slab]
    x_lo, x_hi = sx.min(), sx.max()
    length = x_hi - x_lo
    box_ids, box_points = {}, []
    for name, (b0, b1) in (("anterior", (x_lo - 8, x_lo + 0.30 * length)),
                           ("posterior", (x_hi - 0.22 * length, x_hi + 8))):
        pick = (sx >= b0) & (sx <= b1)
        box_ids[name] = slab_ids[pick]
        box_points += d_rectangle((b0, b1, sy[pick].min() - 14, sy[pick].max() + 14, z_mid - half, z_mid + half))

    # the section on a transparent background (for the stack of cards in the figure), and the raw of its middle
    # plane, from the same camera: a dark card over the slab area
    slab_bounds = mesh.bounds

    def shot(add, bounds, size, zoom, transparent=False, points=()):
        centre = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        return d_section_shot(add, centre, PLATY_DIRECTION_B, PLATY_UP, np.array([-2500.0, 1500.0, -1500.0]), bounds,
                              size, zoom, transparent, points)

    # the section prints narrow and tall, so it is rendered larger than the studio
    section_size = (round(1100 * 2.5), round(2000 * 2.5))
    image, projected, ppv = shot(lambda p: d_add_objects(p, mesh), slab_bounds, section_size, 1.0, True, box_points)
    Image.fromarray(image).save(f"{out}/section_rgba.png")
    layout["section"] = {"anterior_box": projected[:4], "posterior_box": projected[4:], "px_per_voxel": ppv,
                         "raw_slice_s0_z": int(round(2 * z_mid))}
    raw = platy_raw_s0(raw_cache)
    pad = 12
    ia, ib = int(max(0, 2 * (x0 - pad))), int(min(raw.shape[2], 2 * (x1 + pad)))
    ja, jb = int(max(0, 2 * (y0 - pad))), int(min(raw.shape[1], 2 * (y1 + pad)))
    plane = np.asarray(raw[int(round(2 * z_mid)), ja:jb, ia:ib], dtype="float32")
    # the contrast of the single slice, which is dimmer than the projection the raw was scaled for
    raw_hi = float(np.percentile(plane[plane > 0], 99.0))

    def add_raw(plotter):
        grid = pv.ImageData(dimensions=(plane.shape[1], plane.shape[0], 1), spacing=(0.5, 0.5, 1.0),
                            origin=(ia / 2, ja / 2, z_mid))
        grid.point_data["values"] = plane.ravel(order="C")
        plotter.add_mesh(grid, scalars="values", cmap="gray", clim=(0, raw_hi), ambient=1.0, diffuse=0.0,
                         specular=0.0, show_scalar_bar=False)

    Image.fromarray(shot(add_raw, slab_bounds, section_size, 1.0)[0]).save(f"{out}/section_raw.png")
    d_log("section")
    for name in ("anterior", "posterior"):
        part = mesh.extract_cells(np.flatnonzero(np.isin(mesh.cell_data["label"], box_ids[name])))
        image, _, ppv = shot(lambda p, part=part: d_add_objects(p, part), part.bounds, (1300, 1300), 1.05)
        Image.fromarray(image).save(f"{out}/{name}.png")
        layout[name] = {"nuclei": int(len(box_ids[name])), "px_per_voxel": ppv}
    json.dump(layout, open(f"{out}/layout.json", "w"), indent=1)
    d_log("platynereis done")


# the colours of the cells and the nuclei on the plane of the spheroid example, and of their outlines
SPHEROID_CELL_RGB, SPHEROID_CELL_EDGE = (0.84, 0.42, 0.76), (0.52, 0.16, 0.46)
SPHEROID_NUCLEUS_RGB, SPHEROID_NUCLEUS_EDGE = (1.0, 1.0, 1.0), (0.62, 0.62, 0.68)


def spheroid_cells(seg):
    """Keep only the cells of the spheroid: the objects that lie mostly in the largest connected foreground component,
    that are at least a fifth of the median cell size and that fill at least a fifth of their bounding box. The loose
    pieces around the spheroid and the flat sheets on its surface go."""
    components, _ = ndimage.label(seg > 0)
    largest = np.argmax(np.bincount(components[components > 0]))
    ids, sizes = np.unique(seg[seg > 0], return_counts=True)
    inside = np.bincount(seg[components == largest], minlength=int(seg.max()) + 1)[ids]
    boxes = ndimage.find_objects(seg)
    fill = np.array([sizes[k] / np.prod([b.stop - b.start for b in boxes[i - 1]]) for k, i in enumerate(ids)])
    keep = ids[(inside > 0.5 * sizes) & (sizes >= 0.2 * np.median(sizes)) & (fill >= 0.2)]
    return np.where(np.isin(seg, keep), seg, 0)


def render_spheroid(out):
    """The renders of the spheroid example: its cells and its nuclei in the studio, side by side with one camera, a
    nucleus in the colour of the cell that holds most of it; and the middle z plane of both channels (E-cadherin
    magenta, DAPI cyan), the raw on its left half and the outlines of the cells (white) and the nuclei (yellow) on its
    right half."""
    os.makedirs(out, exist_ok=True)
    z_scale = SPHEROID_Z_UM / SPHEROID_XY_UM  # the scene is in xy voxel units
    raw = tifffile.imread(f"{SPHEROID}.tif")  # z, c, y, x: DAPI, E-cadherin, beta-tubulin

    def relabel(labels):
        ids, inverse = np.unique(labels, return_inverse=True)
        return inverse.reshape(labels.shape).astype("uint16"), len(ids) - 1

    cells, n_cells = relabel(spheroid_cells(tifffile.imread(f"{SPHEROID}_membrane_prediction.tif")))
    nuclei, n_nuclei = relabel(tifffile.imread(f"{SPHEROID}_nucleus_prediction.tif"))
    cells_sep, pairs = d_separate(cells)
    pairs |= d_near_pairs(cells, radius=3)
    cell_colors = d_distinct_colors(n_cells, pairs)
    both = (nuclei > 0) & (cells > 0)
    votes = np.zeros((n_nuclei + 1, n_cells + 1))
    np.add.at(votes, (nuclei[both], cells[both]), 1)
    # a nucleus outside every cell is grey
    nucleus_colors = np.full((n_nuclei + 1, 3), 0.75)
    has = votes[:, 1:].sum(1) > 0
    nucleus_colors[has] = cell_colors[votes[has, 1:].argmax(1) + 1]
    meshes = {"cells": d_colored_mesh(cells_sep, cell_colors, z_scale),
              "nuclei": d_colored_mesh(d_separate(nuclei)[0], nucleus_colors, z_scale)}

    # one studio with the cells and the nuclei side by side (user, 2026-11-09): the nuclei are shifted across the line
    # of sight, so that the two spheroids stand next to each other
    view, flat = d_view(30, -60)
    across = np.array([-flat[1], flat[0], 0.0])
    zz, yy, xx = np.nonzero(cells)
    span = max(np.ptp(xx), np.ptp(yy))
    # a little apart, closer than first drawn (user, 2026-11-09)
    shift = 0.95 * span * across
    meshes["nuclei"].points += shift
    z = int(np.median(zz))
    z_plane = (z + 0.5) * z_scale
    # the plane as one rectangle through both spheroids, long across the line of sight: the box of the footprints of
    # both in the frame of the line of sight
    footprint = np.c_[xx, yy].astype(float)
    footprint = np.r_[footprint, footprint + shift[:2]]
    u, v = footprint @ across[:2], footprint @ flat[:2]
    pad = 8.0
    corners = [(u0, v0) for u0, v0 in ((u.min() - pad, v.max() + pad), (u.max() + pad, v.max() + pad),
                                       (u.max() + pad, v.min() - pad), (u.min() - pad, v.min() - pad))]
    rectangle = [tuple(cu * across[:2] + cv * flat[:2]) + (z_plane,) for cu, cv in corners]
    points = np.r_[np.array([m.bounds for m in meshes.values()])]
    bounds = (points[:, 0].min() - 20, points[:, 1].max() + 20, points[:, 2].min() - 20, points[:, 3].max() + 20, 0,
              points[:, 5].max())
    layout = {"xy_um": SPHEROID_XY_UM, "cells": n_cells, "nuclei": n_nuclei}
    plotter = d_studio((2400, 1600), bounds, view, 900, 1.0)
    for mesh in meshes.values():
        d_add_objects(plotter, mesh)
    d_backdrop(plotter, bounds, view, 10, 600, 40)
    Image.fromarray(d_screenshot(plotter, (2400, 1600))).save(f"{out}/studio.png")
    layout["rectangle"] = d_project(plotter, rectangle)
    # the top centre of each spheroid, for its name
    tops = {name: ((m.bounds[0] + m.bounds[1]) / 2, (m.bounds[2] + m.bounds[3]) / 2, m.bounds[5])
            for name, m in meshes.items()}
    layout["labels"] = {name: d_project(plotter, [top])[0] for name, top in tops.items()}
    centre = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, bounds[5] / 2)
    layout["px_per_voxel"] = d_px_per_unit(plotter, np.array(centre))
    plotter.close()
    d_log("spheroid studio")

    norm = [np.clip(c / np.percentile(c, 99.5), 0, 1) for c in raw[z].astype("float32")]
    rgb = np.clip(norm[1][..., None] * np.array([1.0, 0.25, 0.8]) + norm[0][..., None] * np.array([0.2, 0.75, 1.0]),
                  0, 1)
    # the right half (user, 2026-11-09): all cells solid in one pale magenta, as the E-cadherin stain, and the nuclei
    # on top in white; each with an outline in a darker shade of its fill
    lined = rgb.copy()
    lined[cells[z] > 0] = SPHEROID_CELL_RGB
    lined[find_boundaries(cells[z], mode="inner")] = SPHEROID_CELL_EDGE
    lined[nuclei[z] > 0] = SPHEROID_NUCLEUS_RGB
    lined[find_boundaries(nuclei[z], mode="inner")] = SPHEROID_NUCLEUS_EDGE
    rows, cols = np.nonzero(cells[z])
    r0, r1 = max(rows.min() - 15, 0), min(rows.max() + 15, rgb.shape[0])
    c0, c1 = max(cols.min() - 15, 0), min(cols.max() + 15, rgb.shape[1])
    half = (c0 + c1) // 2
    plane = np.concatenate([rgb[r0:r1, c0:half], lined[r0:r1, half:c1]], 1)
    Image.fromarray((plane * 255).astype("uint8")).save(f"{out}/plane.png")
    layout["plane_z"] = z
    json.dump(layout, open(f"{out}/layout.json", "w"), indent=1)
    d_log("spheroid done")


def render_gruen(out):
    """The renders of the heart section panel: the volume as one card of its merged channels (on white) with the
    outlines of the three crops on it, and for each crop its cells alone, in the studio, from the same direction."""
    os.makedirs(out, exist_ok=True)
    z_scale = GRUEN_Z_UM / GRUEN_XY_UM  # the scene is in xy pixel units
    view, _ = d_view(35, -60)
    down = 2  # the downscaling of the raw card of the volume
    volume = tifffile.imread(glob.glob(f"{GRUEN}/*_normalized.tif")[0])
    height, width = volume.shape[1:3]
    thickness = volume.shape[0] * z_scale
    mip = volume.max(0)
    # the nuclear channel (blue) is dim and blue looks dark: stretch it to its 99.5th percentile, with a gamma
    nuclei = mip[..., 2].astype("float32")
    mip[..., 2] = (np.clip(nuclei / np.percentile(nuclei, 99.5), 0, 1) ** 0.7 * 255).astype("uint8")
    del volume
    small = np.asarray(Image.fromarray(mip).resize((width // down, height // down), Image.LANCZOS))
    crops = []
    for path in sorted(glob.glob(f"{GRUEN}/RGB/*crop*.tif")):
        name = os.path.basename(path)
        crops.append({"name": f"crop{name.split('_crop')[1].split('_')[0]}",
                      "y": int(name.split("_y")[1].split("_")[0]), "x": int(name.split("_x")[1].split(".")[0]),
                      "mask": f"{GRUEN}/masks/{name.replace('.tif', '_mask.tif')}"})

    # one card of the merged channels
    top_z = 0.0
    plotter = d_studio((2400, 2400), (0, width, -height, 0, 0, top_z + thickness), view, 4000, 1.0)
    d_card(plotter, small, (0, 0, top_z), down, thickness)
    Image.fromarray(d_screenshot(plotter, (2400, 2400))).save(f"{out}/volume.png")
    layout = {"xy_um": GRUEN_XY_UM, "z_um": GRUEN_Z_UM, "volume": [int(height), int(width)], "crops": {}}
    for crop in crops:
        # the crop footprint on the top face of the top card: top left, top right, bottom right, bottom left
        x, y, z = crop["x"], -crop["y"], top_z + thickness + 0.5
        corners = [(x, y, z), (x + 512, y, z), (x + 512, y - 512, z), (x, y - 512, z)]
        layout["crops"][crop["name"]] = {"outline": d_project(plotter, corners), "y": crop["y"], "x": crop["x"]}
    layout["volume_px_per_pixel"] = d_px_per_unit(plotter, (width / 2, -height / 2, top_z), 500.0)
    plotter.close()
    d_log("volume")

    # each crop: its cells alone, on the studio floor
    for crop in crops:
        labels = tifffile.imread(crop["mask"])
        ids, inverse = np.unique(labels, return_inverse=True)
        labels = inverse.reshape(labels.shape).astype("uint16")
        n_labels = len(ids) - 1
        separated, pairs = d_separate(labels)
        pairs |= d_near_pairs(labels, radius=3)
        colors = d_distinct_colors(n_labels, pairs)
        # the cells fill all planes and reach the crop edges: pad with background, so that their surfaces close
        mesh = d_label_surfaces(np.pad(separated, 1))
        mesh.points -= 1
        # y down the image rows as in the volume render; the mirror flips the faces, so turn them back
        mesh.points[:, 1] *= -1
        mesh.points[:, 2] *= z_scale
        mesh.flip_faces(inplace=True)
        mesh.cell_data["rgb"] = (colors[mesh.cell_data["label"]] * 255).astype("uint8")
        bounds = (0, 512, -512, 0, 0, labels.shape[0] * z_scale)
        plotter = d_studio((1400, 1400), bounds, view, 700, 1.0)
        d_add_objects(plotter, mesh)
        d_backdrop(plotter, bounds, view, 30, 300, 180)
        Image.fromarray(d_screenshot(plotter, (1400, 1400))).save(f"{out}/{crop['name']}.png")
        # the corners of the top face of the cell block (the cells fill most of the crop), in the same order as
        # the crop outline
        z = bounds[5]
        face = d_project(plotter, [(0, 0, z), (512, 0, z), (512, -512, z), (0, -512, z)])
        layout["crops"][crop["name"]].update({"top_face": face, "cells": n_labels,
                                              "px_per_pixel": d_px_per_unit(plotter, (256, -256, 0))})
        plotter.close()
        # one exploded view of the crop (feedback 2026-11-09): its three channels as thin cards at the bottom, the
        # merged channels as a block above them, and the segmented cells floating above the block
        y, x = crop["y"], crop["x"]
        merged = mip[y:y + 512, x:x + 512]
        # the channel cards spread like a dealt deck (user, 2026-11-09): farther apart, and each lower card shifted
        # further to the left, so that more of each channel shows
        height_cells, card, gap, spread = bounds[5], 10.0, 110.0, 110.0
        levels = [k * (card + gap) for k in range(3)]
        block_z = 3 * (card + gap)
        cells_z = block_z + height_cells + 1.4 * gap
        exploded = mesh.copy()
        exploded.points[:, 2] += cells_z
        scene = (-3 * spread, 512, -512, 0, 0, cells_z + height_cells)
        plotter = d_studio((1400, 2000), scene, view, 900, 1.0)
        # without the studio: no walls, floor or shadows (user, 2026-11-09)
        plotter.disable_shadows()
        # red, green and blue from the bottom up, each in its own color; pure blue looks dark, so the channels are
        # stretched to their 99.5th percentile and drawn in lighter tints (user, 2026-11-09)
        tints = np.array([(1.0, 0.25, 0.2), (0.25, 1.0, 0.35), (0.35, 0.6, 1.0)])
        for k, z0 in enumerate(levels):
            value = merged[..., k].astype("float32")
            value = np.clip(value / max(np.percentile(value, 99.5), 1.0), 0, 1) ** 0.8
            single = (value[..., None] * tints[k] * 255).astype("uint8")
            d_card(plotter, single, (-(3 - k) * spread, 0, z0), 1.0, card)
        d_card(plotter, merged, (0, 0, block_z), 1.0, height_cells)
        d_add_objects(plotter, exploded)
        Image.fromarray(d_screenshot(plotter, (1400, 2000))).save(f"{out}/{crop['name']}_exploded.png")
        zb = block_z + height_cells
        layout["crops"][crop["name"]]["block_face"] = d_project(
            plotter, [(0, 0, zb), (512, 0, zb), (512, -512, zb), (0, -512, zb)])
        layout["crops"][crop["name"]]["exploded_px_per_pixel"] = d_px_per_unit(plotter, (256, -256, zb))
        # the top face of the lowest channel card, for the lines from the outline on the section
        x0 = -3 * spread
        layout["crops"][crop["name"]]["deck_face"] = d_project(
            plotter, [(x0, 0, card), (x0 + 512, 0, card), (x0 + 512, -512, card), (x0, -512, card)])
        plotter.close()
        d_log(crop["name"], n_labels, "cells")
    json.dump(layout, open(f"{out}/layout.json", "w"), indent=1)
    d_log("gruen done")


def render_histai(out):
    """The render of the Ki67 core example: the whole core upright in the studio (scan orientation), and in a column on
    its right one full resolution crop, the window with the most of both tumor classes, joined by lines from its box
    on the core: the image data on top and the classification below it, every segmented nucleus with a light fill and
    an outline in the colour of its class (feedback 2026-11-09: one smaller zoom, image and classification)."""
    os.makedirs(out, exist_ok=True)
    group = zarr.open_group(zarr.storage.ZipStore(HISTAI, mode="r"), mode="r", zarr_format=2)
    h0, w0 = group["0"].shape[1:]
    size_px = HISTAI_CROP
    # the crops, found on level 3 (1 px = 8 level 0 px), at least two crop sizes apart
    classes3 = group["labels/classes/3"][:]
    k = size_px // 8
    maps = {"positive": ndimage.uniform_filter((classes3 == 1).astype("float32"), k),
            "negative": ndimage.uniform_filter((classes3 == 2).astype("float32"), k),
            "other": ndimage.uniform_filter((classes3 == 3).astype("float32"), k)}
    # the most diverse window: the one where the rarest of the three classes is densest
    score = np.minimum(np.minimum(maps["positive"], maps["negative"]), maps["other"])
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    crop = (int(np.clip(cy * 8 - size_px // 2, 0, h0 - size_px)), int(np.clip(cx * 8 - size_px // 2, 0, w0 - size_px)))

    def raw_crop(y0, x0):
        return np.moveaxis(group["0"][:, y0:y0 + size_px, x0:x0 + size_px], 0, -1).copy()

    def outlined(y0, x0):
        """The crop with each nucleus in the colour of its class (the class of most of its pixels): a light fill and a
        thick outline. Also return the number of nuclei per class."""
        raw = np.moveaxis(group["0"][:, y0:y0 + size_px, x0:x0 + size_px], 0, -1).copy()
        inst = group["labels/instances/0"][y0:y0 + size_px, x0:x0 + size_px]
        cls = group["labels/classes/0"][y0:y0 + size_px, x0:x0 + size_px]
        ids, inverse = np.unique(inst, return_inverse=True)
        inst = inverse.reshape(inst.shape)
        n = len(ids) - 1
        votes = np.stack([np.bincount(inst[cls == c].ravel(), minlength=n + 1) for c in range(4)], 1)
        obj_class = votes[:, 1:].argmax(1) + 1
        obj_class[votes[:, 1:].sum(1) == 0] = 0
        inside = inst > 0
        raw[inside] = (0.6 * raw[inside] + 0.4 * HISTAI_CLASS_RGB[obj_class[inst[inside]]]).astype(np.uint8)
        edge = ndimage.binary_dilation(find_boundaries(inst, mode="inner"), iterations=2) & inside
        raw[edge] = HISTAI_CLASS_RGB[obj_class[inst[edge]]]
        return raw, [int((obj_class[1:] == c).sum()) for c in (1, 2, 3)]

    def rect(x, y, w, h):
        """Top left, top right, bottom right, bottom left; y is the top edge (scene y grows upwards)."""
        return [(x, y), (x + w, y), (x + w, y - h), (x, y - h)]

    t, core_h, card, gap_x, gap_y = 20.0, 3000.0, 1100.0, 250.0, 120.0
    step0 = core_h / h0  # scene units per level 0 px on the core
    core_w = w0 * step0
    column_x = core_w + gap_x
    column_top = -(core_h - (2 * card + gap_y)) / 2
    view, _ = d_view(55, -90)
    size = (3900, 3900)  # large enough to print at more than 1200 dpi
    plotter = d_studio(size, (0, column_x + card, -core_h, 0, 0, t), view, 8000, 1.0)
    d_card(plotter, np.moveaxis(group["3"][:], 0, -1), (0, 0, 0), step0 * 8, t)
    layout = {"um_per_px": HISTAI_UM, "crops": {}}
    y0, x0 = crop
    classified, counts = outlined(y0, x0)
    box = rect(x0 * step0, -y0 * step0, size_px * step0, size_px * step0)
    ring = [(x, y, t + 1) for x, y in box + box[:1]]
    plotter.add_mesh(pv.lines_from_points(np.array(ring)), color=(0.1, 0.1, 0.1), line_width=8)
    # the image data on top, the classification below it
    for i, image in enumerate((raw_crop(y0, x0), classified)):
        top = column_top - i * (card + gap_y)
        d_card(plotter, image, (column_x, top, 0), card / size_px, t)
        if i == 0:
            # from the right corners of the box to the left corners of the image card
            target = rect(column_x, top, card, card)
            for a, b in ((box[1], target[0]), (box[2], target[3])):
                plotter.add_mesh(pv.Line((a[0], a[1], t + 1), (b[0], b[1], t + 1)), color=(0.15, 0.15, 0.15),
                                 line_width=5)
    layout["crops"]["mixed"] = {"y0": y0, "x0": x0, "positive": counts[0], "negative": counts[1],
                                "other": counts[2]}
    image = d_screenshot(plotter, size)
    # pixels of the render per micrometer, at the centre of the core and of the middle crop
    layout["core_px_per_um"] = d_px_per_unit(plotter, (core_w / 2, -core_h / 2, t)) * step0 / HISTAI_UM
    middle = (column_x + card / 2, column_top - card / 2, t)
    layout["crop_px_per_um"] = d_px_per_unit(plotter, middle) * card / size_px / HISTAI_UM
    left, bottom_px = d_project(plotter, [(column_x, column_top - 2 * card - gap_y, t)])[0]
    # the top of the image card, for the legend above it
    column_top_px = d_project(plotter, [(column_x, column_top, t)])[0][1]
    plotter.close()
    rows, cols = np.nonzero((image < 245).any(-1))
    r0, c0 = max(rows.min() - 8, 0), max(cols.min() - 8, 0)
    image = image[r0:rows.max() + 8, c0:cols.max() + 8]
    layout["crop_column_left"] = (left - c0) / image.shape[1]
    layout["crop_column_top"] = (column_top_px - r0) / image.shape[0]
    layout["crop_column_bottom"] = (bottom_px - r0) / image.shape[0]
    Image.fromarray(image).save(f"{out}/zoom.png")
    json.dump(layout, open(f"{out}/layout.json", "w"), indent=1)
    d_log("histai done")


def d_fonts():
    for name in os.listdir(FONT_DIR):
        if name.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONT_DIR, name))
    matplotlib.rcParams["font.family"] = "Liberation Sans"
    matplotlib.rcParams["font.size"] = 7


def d_load(path, mode="RGB"):
    return np.asarray(Image.open(path).convert(mode))


def d_crop_white(image, margin):
    """Crop a white background image to its content plus a margin; also return the offset (column, row)."""
    rows, cols = np.nonzero((image < 245).any(-1))
    r0, c0 = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
    return image[r0:rows.max() + margin, c0:cols.max() + margin], np.array([c0, r0])


def d_crop_to(image, points, keep, margin):
    """Crop to the pixels to keep and the given points, plus a margin; also return the offset (column, row)."""
    rows, cols = np.nonzero(keep)
    pts = np.asarray(points).reshape(-1, 2)
    c0, c1 = min(cols.min(), pts[:, 0].min()), max(cols.max(), pts[:, 0].max())
    r0, r1 = min(rows.min(), pts[:, 1].min()), max(rows.max(), pts[:, 1].max())
    c0, r0 = int(max(c0 - margin, 0)), int(max(r0 - margin, 0))
    c1, r1 = int(min(c1 + margin, image.shape[1])), int(min(r1 + margin, image.shape[0]))
    return image[r0:r1, c0:c1], np.array([c0, r0])


def d_colourful(image):
    """The scene without the grey studio: coloured or dark pixels."""
    rgb = image.astype(int)
    return (np.ptp(rgb, -1) > 30) | (rgb.max(-1) < 60)


def d_card_corners(image):
    """The corners of a dark tilted card on white: top left, top right, bottom right, bottom left."""
    rows, cols = np.nonzero((image < 250).any(-1))
    s, d = cols + rows, cols - rows
    return np.array([(cols[i], rows[i]) for i in (np.argmin(s), np.argmax(d), np.argmax(s), np.argmin(d))], float)


def d_stretch(raw, on_card):
    """More contrast in a raw card: stretch to its 98th percentile on the card, with a slight gamma."""
    grey = raw[..., 0].astype(float)
    hi = np.percentile(grey[on_card & (grey > 8)], 98)
    stretched = (np.clip(grey / hi, 0, 1) ** 0.8 * 255)[..., None].repeat(3, -1)
    return np.where(on_card[..., None], stretched, raw).astype("uint8")


def d_layered(raw, seg, shift):
    """A stack of two cards of the same tilted shape: the raw at the back with its edge drawn, and the objects (with
    their own alpha, rendered on a transparent background) on a clear card in front, moved by the shift vector.
    Returns the image, the card corners and the origins of the back and front layers."""
    h, w = raw.shape[:2]
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    back = np.array([0, max(0, -dy)])
    front = back + [dx, dy]
    out = np.full((h + abs(dy), w + dx, 3), 255, "uint8")
    out[back[1]:back[1] + h, back[0]:back[0] + w] = raw
    card = d_card_corners(raw)
    edge = Image.fromarray(out)
    ImageDraw.Draw(edge).polygon([tuple(p) for p in card + back], outline=(138, 138, 138), width=3)
    out = np.asarray(edge).astype(float)
    alpha = seg[..., 3:4].astype(float) / 255
    region = out[front[1]:front[1] + h, front[0]:front[0] + w]
    out[front[1]:front[1] + h, front[0]:front[0] + w] = region * (1 - alpha) + seg[..., :3] * alpha
    return np.clip(out, 0, 255).astype("uint8"), card, back, front


# the print resolutions of the images placed in panel d
D_DPI = []


class DFigure:
    """A matplotlib figure laid out in mm from the top left."""

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=300)

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / self.width, 1 - (y + h) / self.height, w / self.width, h / self.height])

    def image(self, image, x, y, w, h, frame=False):
        """The image as large as fits in a box, aligned to the top left; with a thin frame or without axes."""
        ih, iw = image.shape[:2]
        scale = min(w / iw, h / ih)
        # the print resolution of the image (pixels per inch), for the check that panel d stays above 1200 dpi
        D_DPI.append(25.4 / scale)
        ax = self.axes(x, y, iw * scale, ih * scale)
        # no interpolation: the svg keeps the full resolution of the render, instead of resampling it to 72 dpi
        ax.imshow(image, interpolation="none")
        if frame:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color(D_LINE)
        else:
            ax.axis("off")
        return ax

    def title(self, ax, text):
        ax.text(0.5, 1.0 + 1.6 / (ax.get_position().height * self.height), text, transform=ax.transAxes,
                ha="center", va="baseline", fontsize=7, color=D_LINE)

    def connect(self, ax_a, point_a, ax_b, point_b):
        self.fig.add_artist(ConnectionPatch(xyA=point_a, coordsA=ax_a.transData, xyB=point_b, coordsB=ax_b.transData,
                                            lw=0.5, color=D_LINE))

    def svg(self):
        """The figure as an svg element sized in mm, with editable text, to place in the svg of the figure."""
        buffer = io.StringIO()
        # transparent, so that an example placed over the empty margin of another does not hide it
        with matplotlib.rc_context({"svg.fonttype": "none", "image.composite_image": False}):
            self.fig.savefig(buffer, format="svg", dpi=300, transparent=True)
        plt.close(self.fig)
        text = buffer.getvalue()
        root = text[text.index("<svg"):]
        head, body = root.split(">", 1)
        head = re.sub(r'\s(width|height)="[^"]*"', "", head)
        # the text asks for Arial first, as in the other panels; Liberation Sans has its metrics
        body = re.sub(r"font-family:\s*'?Liberation Sans'?", "font-family:Arial,'Liberation Sans',sans-serif", body)
        return f'{head} width="{fmt(self.width)}" height="{fmt(self.height)}">{body}'


def d_scale_bar(fig, ax, um, px_per_um, below=1.6, inset=0.03):
    """A scale bar a fixed distance (in mm) below the image of the axes, at its left, with its label to its right,
    so that neither covers the image at any size. Whole millimeters are labelled in mm."""
    h, w = ax.get_ylim()[0] + 0.5, ax.get_xlim()[1] + 0.5
    px_per_mm = h / (ax.get_position().height * fig.height)
    x0, y = inset * w, h + below * px_per_mm
    ax.plot([x0, x0 + um * px_per_um], [y, y], color=D_LINE, lw=1.6, solid_capstyle="butt", clip_on=False)
    label = f"{um // 1000} mm" if um >= 1000 and um % 1000 == 0 else f"{um} µm"
    ax.text(x0 + um * px_per_um + 0.8 * px_per_mm, y, label, ha="left", va="center", color=D_LINE,
            fontsize=6.5, clip_on=False)
    ax.set_ylim(h - 0.5, -0.5)


def d_box(ax, corners, lw=0.7):
    """A dark outline with a white halo under it, so that it shows on dark and on white."""
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="white", lw=lw + 1.1, joinstyle="miter"))
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=D_LINE, lw=lw, joinstyle="miter"))


def d_right_edge(corners):
    """The upper and the lower of the two rightmost corners."""
    right = corners[np.argsort(corners[:, 0])[-2:]]
    return right[np.argmin(right[:, 1])], right[np.argmax(right[:, 1])]


def build_platynereis(folder, height, measure=False):
    """The Platynereis example: a) the nuclei in the studio with the section plane, b) the section as a stack of the
    raw card at the back and its nuclei on a clear card in front, joined to the plane by thin lines."""
    layout = json.load(open(f"{folder}/layout.json"))
    um_pre = layout["voxel_um"] / layout["expansion"]  # pre-expansion micrometers per s1 voxel
    raw = d_load(f"{folder}/section_raw.png")
    raw = d_stretch(raw, (raw < 250).any(-1))
    corners = d_card_corners(raw)
    section, card, back, front = d_layered(raw, d_load(f"{folder}/section_rgba.png", "RGBA"),
                                           0.42 * (corners[1] - corners[0]))
    section, offset = d_crop_white(section, 8)
    glass, back_card = card + front - offset, card + back - offset
    # a: the studio cropped to the animal and the section plane, with a small margin
    seg = d_load(f"{folder}/seg_studio.png")
    outline = np.array(layout["studio"]["slab_outline"])
    cols = np.flatnonzero((np.ptp(seg.astype(int), -1) > 40).any(0))
    a0 = max(int(min(cols.min(), outline[:, 0].min())) - 4, 0)
    a1 = min(int(max(cols.max(), outline[:, 0].max())) + 4, seg.shape[1])
    seg, outline = seg[:, a0:a1], outline - [a0, 0]

    # the widths follow from one common height of a and b
    top, gap = D_TOP["platynereis"], 2.5
    row_h = height - top - 4
    wa = row_h * seg.shape[1] / seg.shape[0]
    wb = row_h * section.shape[1] / section.shape[0]
    xa, xb = 3.0, 3.0 + wa + gap
    width = xb + wb + 1.5
    if measure:
        return None, width
    fig = DFigure(width, height)
    ax_a = fig.image(seg, xa, top, wa, row_h)
    ax_b = fig.image(section, xb, top, wb, row_h)
    # the section plane in a, and lines from its right edge to the left corners of the raw card in b (corners: top
    # left, top right, bottom right, bottom left)
    ax_a.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=D_LINE, lw=0.8, joinstyle="miter"))
    upper, lower = d_right_edge(outline)
    fig.connect(ax_a, upper, ax_b, back_card[0])
    fig.connect(ax_a, lower, ax_b, back_card[3])
    ax_b.add_patch(Polygon(glass, closed=True, fill=False, edgecolor="#8a8a8a", lw=0.5, joinstyle="miter"))
    d_scale_bar(fig, ax_a, 50, layout["studio"]["px_per_voxel"] / um_pre)
    d_scale_bar(fig, ax_b, 20, layout["section"]["px_per_voxel"] / um_pre)
    return fig.svg(), width


# the height of the studio of the spheroid example, as a share of the height of its plane
SPHEROID_STUDIO_SHARE = 0.72


def build_spheroid(folder, height, measure=False):
    """The spheroid example: its cells and its nuclei in one studio, side by side in shared colours, with one
    rectangle of the middle plane through both; and the plane of both channels, larger, with the outlines of both
    segmentations on its right half, joined to the rectangle by lines (user's picks 2026-11-09)."""
    layout = json.load(open(f"{folder}/layout.json"))
    studio = d_load(f"{folder}/studio.png")
    rectangle = np.array(layout["rectangle"])
    studio, offset = d_crop_to(studio, rectangle, np.ptp(studio.astype(int), -1) > 40, 12)
    # a little more of the studio around the spheroids, so that both walls and the floor show (user, 2026-11-09):
    # the crop grows by these shares of its height, at the top, the sides and the bottom
    full = d_load(f"{folder}/studio.png")
    h = studio.shape[0]
    r0, c0 = max(int(offset[1] - 0.22 * h), 0), max(int(offset[0] - 0.08 * h), 0)
    r1 = min(int(offset[1] + h + 0.06 * h), full.shape[0])
    c1 = min(int(offset[0] + studio.shape[1] + 0.08 * h), full.shape[1])
    studio, offset = full[r0:r1, c0:c1], np.array([c0, r0])
    rectangle = rectangle - offset
    plane = d_load(f"{folder}/plane.png")

    # the plane takes the full height of the row, the studio a part of it, centred (user, 2026-11-09: a larger plane)
    top, gap = D_TOP["spheroid"] + 1.5, 4.0
    row_h = height - top - 4.0
    ha = SPHEROID_STUDIO_SHARE * row_h
    wa = ha * studio.shape[1] / studio.shape[0]
    wb = row_h * plane.shape[1] / plane.shape[0]
    width = 3.0 + wa + gap + wb + 1.5
    if measure:
        return None, width
    fig = DFigure(width, height)
    ax_a = fig.image(studio, 3.0, top + (row_h - ha) / 2, wa, ha)
    ax_b = fig.image(plane, 3.0 + wa + gap, top, wb, row_h)
    ax_a.add_patch(Polygon(rectangle, closed=True, fill=False, edgecolor=D_LINE, lw=0.8, joinstyle="miter"))
    # the right corners of the rectangle to the left corners of the plane
    right = rectangle[np.argsort(rectangle[:, 0])[-2:]]
    right = right[np.argsort(right[:, 1])]
    ph = plane.shape[0]
    fig.connect(ax_a, right[0], ax_b, (-0.5, -0.5))
    fig.connect(ax_a, right[1], ax_b, (-0.5, ph - 0.5))
    # the names stand above the studio, over their spheroids (user, 2026-11-09)
    px_per_mm = studio.shape[0] / ha
    for name, (x, _) in layout["labels"].items():
        ax_a.text(x - offset[0], -0.8 * px_per_mm, name, ha="center", va="bottom", fontsize=7, color=D_LINE,
                  clip_on=False)
    d_scale_bar(fig, ax_a, 50, layout["px_per_voxel"] / layout["xy_um"])
    d_scale_bar(fig, ax_b, 50, 1 / layout["xy_um"])
    return fig.svg(), width


def build_gruen(folder, height, measure=False):
    """The heart section panel: the volume as one card of its merged channels with the outline of one crop on it
    (the lowest, user's pick 2026-11-09), and on its right the exploded view of the crop: its three channels at the
    bottom, the merged block above them and the segmented cells floating on top, joined to the outline by lines to
    the top face of the block (feedback 2026-11-09)."""
    layout = json.load(open(f"{folder}/layout.json"))
    volume = d_load(f"{folder}/volume.png")
    crops = layout["crops"]
    name = max(crops, key=lambda n: np.mean(np.array(crops[n]["outline"])[:, 1]))
    # tight crops and a small gap, so that the example is as large as its box allows (user, 2026-11-09)
    volume, offset = d_crop_to(volume, [crops[name]["outline"]], d_colourful(volume), 2)
    image = d_load(f"{folder}/{name}_exploded.png")
    image, off = d_crop_to(image, crops[name]["block_face"], d_colourful(image), 2)
    deck = np.array(crops[name]["deck_face"]) - off

    top, gap = D_TOP["gruen"], 2.5
    row_h = height - top - 4
    wa = row_h * volume.shape[1] / volume.shape[0]
    wc = row_h * image.shape[1] / image.shape[0]
    xa, xc = 3.0, 3.0 + wa + gap
    width = xc + wc + 1.5
    if measure:
        return None, width
    fig = DFigure(width, height)
    ax_a = fig.image(volume, xa, top, wa, row_h)
    outline = np.array(crops[name]["outline"]) - offset
    d_box(ax_a, outline)
    ax = fig.image(image, xc, top, wc, row_h)
    # the pixel size is estimated (GRUEN_XY_UM), not from the files
    d_scale_bar(fig, ax, 20, crops[name]["exploded_px_per_pixel"] / layout["xy_um"])
    # the lower edge of the outline to the lower edge of the lowest channel card (user, 2026-11-09): the two lowest
    # corners of each, left to right
    low_a = outline[np.argsort(outline[:, 1])[-2:]]
    low_b = deck[np.argsort(deck[:, 1])[-2:]]
    for a, b in zip(low_a[np.argsort(low_a[:, 0])], low_b[np.argsort(low_b[:, 0])]):
        fig.connect(ax_a, a, ax, b)
    d_scale_bar(fig, ax_a, 100, layout["volume_px_per_pixel"] / layout["xy_um"])
    return fig.svg(), width


def build_histai(folder, height, measure=False):
    """The Ki67 core example: its render with a scale bar for the core and one for the crop below it, and the legend of
    the classes above the crop (feedback 2026-11-09)."""
    layout = json.load(open(f"{folder}/layout.json"))
    image = d_load(f"{folder}/zoom.png")
    top = D_TOP["histai"]
    row_h = height - top - 4.0
    width = 3.0 + row_h * image.shape[1] / image.shape[0] + 1.5
    if measure:
        return None, width
    fig = DFigure(width, height)
    ax = fig.image(image, 3.0, top, width - 4.5, row_h)
    d_scale_bar(fig, ax, 1000, layout["core_px_per_um"], inset=0.02)
    d_scale_bar(fig, ax, 50, layout["crop_px_per_um"], inset=layout["crop_column_left"])
    h, w = ax.get_ylim()[0] + 0.5, ax.get_xlim()[1] + 0.5
    px_per_mm = h / (ax.get_position().height * fig.height)
    x0 = layout["crop_column_left"] * w
    # the legend below the classification, centred on it, in two rows: positive and negative, then others (user,
    # 2026-11-09); the texts are measured first
    renderer = fig.fig.canvas.get_renderer()
    data_per_display = w / ax.get_window_extent(renderer=renderer).width
    swatch, inner, outer = 1.4 * px_per_mm, 0.6 * px_per_mm, 1.4 * px_per_mm
    rows = (HISTAI_CLASSES[:2], HISTAI_CLASSES[2:])
    for r, entries in enumerate(rows):
        y = layout["crop_column_bottom"] * h + (1.6 + 2.4 * r) * px_per_mm
        texts, widths = [], []
        for name, _ in entries:
            text = ax.text(0, y, name, ha="left", va="center", color=D_LINE, fontsize=6.5, clip_on=False)
            texts.append(text)
            widths.append(text.get_window_extent(renderer=renderer).width * data_per_display)
        total = sum(swatch + inner + tw for tw in widths) + outer * (len(widths) - 1)
        x = x0 + (w - x0 - total) / 2
        for (name, rgb), text, tw in zip(entries, texts, widths):
            ax.add_patch(matplotlib.patches.Rectangle((x, y - 0.7 * px_per_mm), swatch, swatch,
                                                      facecolor=np.array(rgb) / 255, edgecolor="none",
                                                      clip_on=False))
            text.set_x(x + swatch + inner)
            x += swatch + inner + tw + outer
    return fig.svg(), width


D_RENDERS = {"platynereis": render_platynereis, "spheroid": render_spheroid, "gruen": render_gruen,
             "histai": render_histai}
# the rows of panel d with their examples, in reading order; each row fills the width of the figure
# panel d is one row of columns, from left to right; the examples of a column are stacked, with the same width
# the spheroid (ii) above the heart section (iii) (user, 2026-11-09)
D_COLUMNS = (("platynereis",), ("spheroid", "gruen"), ("histai",))
D_BUILDERS = {"platynereis": build_platynereis, "spheroid": build_spheroid, "gruen": build_gruen,
              "histai": build_histai}
D_EXAMPLE_GAP = 1.5
# the gap between stacked examples (the one above already ends with its scale bar strip)
D_STACK_GAP = 0.0


def build_panel_d(folder, width=A4_WIDTH, height=None):
    """Panel d: the examples in one row of columns, from their renders in the folder, each with its number (i, ii,
    ...) at the top left of its first image. All columns have the same height, at which the row fills the width (at
    most the given height); the examples of a column are stacked at the same width. The width of an example grows
    linearly with its height. Returns the svg content and the size of the panel."""
    d_fonts()
    line = {}
    for column in D_COLUMNS:
        for name in column:
            build, path = D_BUILDERS[name], os.path.join(folder, name)
            low, high = build(path, 50.0, measure=True)[1], build(path, 100.0, measure=True)[1]
            line[name] = ((high - low) / 50.0, low - (high - low))  # slope and offset

    def column_width(column, h):
        """The width of a column of height h: its examples stacked at one width, filling the height."""
        inverse = sum(1 / line[n][0] for n in column)
        return (h - D_STACK_GAP * (len(column) - 1) + sum(line[n][1] / line[n][0] for n in column)) / inverse

    def row_width(h):
        return sum(column_width(c, h) for c in D_COLUMNS) + D_EXAMPLE_GAP * (len(D_COLUMNS) - 1)

    # the height at which the row fills the width (the row width grows linearly with the height)
    h50, h100 = row_width(50.0), row_width(100.0)
    panel_h = 50.0 + (width - h50) * 50.0 / (h100 - h50)
    if height is not None:
        panel_h = min(panel_h, height)
    widths = [column_width(c, panel_h) for c in D_COLUMNS]
    gap = (width - sum(widths)) / (len(D_COLUMNS) - 1)
    parts, x, numbers = [], 0.0, iter(("i", "ii", "iii", "iv", "v", "vi"))
    size = MAX_TEXT  # 7 pt, below the 8 pt panel letters
    for column, column_w in zip(D_COLUMNS, widths):
        y = 0.0
        for name in column:
            example_h = (column_w - line[name][1]) / line[name][0]
            svg, _ = D_BUILDERS[name](os.path.join(folder, name), example_h)
            parts.append(f'<g transform="translate({fmt(x)},{fmt(y)})">{svg}</g>')
            # right next to the first image of the example (which starts 3 mm in and D_TOP down), at its top
            parts.append(f'<text x="{fmt(x + 2.6)}" y="{fmt(y + D_TOP[name] + 0.75 * size)}" text-anchor="end" '
                         f'style="{FONT};font-size:{fmt(size)}px;font-weight:bold;fill:{TEXT}">{next(numbers)}</text>')
            y += example_h + D_STACK_GAP
        x += column_w + gap
    return "".join(parts), width, panel_h


def main():
    parser = argparse.ArgumentParser(description="Make the panels of Figure 1 and put them together.")
    parser.add_argument("--output_folder", default="figures/figure1", help="The folder for the panel files.")
    parser.add_argument("--final_folder", default="figures/final", help="The folder for the whole figure.")
    parser.add_argument("--logo", default=LOGO, help="The micro-sam2 logo.")
    parser.add_argument("--napari_logo", default=NAPARI_LOGO, help="The napari logo, by default the one of napari.")
    parser.add_argument("--python_logo", default=PYTHON_LOGO, help="The Python logo.")
    parser.add_argument("--panels", nargs="+", default=["a", "b", "c", "d"], help="The panels to make.")
    parser.add_argument("--render", nargs="+", default=[], choices=sorted(D_RENDERS),
                        help="The panel d examples to render again, before panel d is put together (on a GPU node).")
    parser.add_argument("--platy_raw_cache", default=os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                                                  "platynereis_raw_s0.npy"),
                        help="The cache of the normalised Platynereis raw at s0 (3.5 GB), for the section.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    panels = {}
    # panel d first: its examples may be rendered again
    d_folder = os.path.join(args.output_folder, "panel_d")
    for name in args.render:
        if name == "platynereis":
            render_platynereis(os.path.join(d_folder, name), args.platy_raw_cache)
        else:
            D_RENDERS[name](os.path.join(d_folder, name))
    d = build_panel_d(d_folder)
    if "d" in args.panels:
        panels["d"] = d
    if "a" in args.panels:
        panels["a"] = (build_panel_a(), A4_WIDTH, A_H * A4_WIDTH / A_W)
    # panel c takes the size of panel b, so panel b comes first
    if "b" in args.panels or "c" in args.panels:
        panels["b"] = build_panel_b()
    if "c" in args.panels:
        fetch_python_logo(args.python_logo)
        _, b_w, b_h = panels["b"]
        c_w = FIGURE_W - 2 * LETTER_W - PANEL_GAP - b_w
        panels["c"] = build_panel_c(args.logo, args.napari_logo, args.python_logo, c_w, b_h)
    for name, (content, width, height, *_) in panels.items():
        if name in args.panels:
            save_panel(content, args.output_folder, f"fig1_panel_{name}", width, height)
    # the whole figure needs all panels
    if {"a", "b", "c", "d"} <= set(args.panels):
        os.makedirs(args.final_folder, exist_ok=True)
        content, width, height = build_figure(panels)
        save_panel(content, args.final_folder, "figure_1", width, height)


if __name__ == "__main__":
    main()
