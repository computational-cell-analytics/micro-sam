"""Plots for Figure 1 of the micro-sam2 paper.

Panel a shows the architecture of micro-sam2: a shared image encoder, the prompt encoder, memory attention and mask
decoder for interactive segmentation, and the segmentation decoder for automatic segmentation.
Panel c shows the tasks that micro-sam2 supports around its hub: interactive analysis (segmentation in 2D and 3D,
tracking in 2D+T, pixel and object classification, with the prompts that the tools support) on the left, and
automatic analysis (segmentation in 2D and 3D, tracking in 2D+T and 3D+T, batch processing) on the right.
The panels are saved as svg, pdf and png, and replace their layers in the Figure 1 draft.
"""

import io
import os
import re
import base64
import argparse
import subprocess
import importlib.util
import urllib.request
from functools import lru_cache

import numpy as np
import pyvista as pv
from PIL import Image
from scipy import ndimage
from matplotlib.path import Path
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline

from micro_sam.v2.transforms.labels import DirectedPerObjectBoundaryDistanceTransform


# the top left corners of the panels in the Figure 1 draft, in mm
PANEL_ORIGINS = {"a": (15.0, 6.0), "c": (100.0, 143.0)}
SIZE = 8.5
STACK_FRAME, STACK_OFFSET = 7.4, 1.4
ITEM_GAP = 1.5

BLUE = "#3d4596"
PURPLE = "#8e4c9f"
RIM = "#b49ad6"
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
LOGO = os.path.join(REPOSITORY, "doc", "logo", "logo_and_text.png")
# the official Python logo, which is downloaded once into the assets folder next to this script
PYTHON_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "python_logo.svg")
PYTHON_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg"
NAPARI_LOGO = os.path.join(
    importlib.util.find_spec("napari").submodule_search_locations[0], "resources", "logos", "gradient-plain-light.svg"
)


def fmt(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


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

    def __init__(self, text_scale=1.0):
        # the scale enlarges the main text, from labels upwards, and leaves the small annotations as they are
        self.defs, self.body, self.count, self.text_scale = [], [], 0, text_scale

    def uid(self, prefix):
        self.count += 1
        return f"f1c-{prefix}-{self.count}"

    def add(self, element):
        self.body.append(element)

    def text(self, x, y, s, size, weight="normal", color=TEXT, anchor="middle", rotate=None):
        size = size * self.text_scale if size >= 1.5 else size
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
        f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}" fill="{color}" stroke="#ffffff" stroke-width="0.12"/>'
        f'<path d="{sign}" stroke="#ffffff" stroke-width="0.26" stroke-linecap="round"/>'
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


def array_uri(rgb):
    buffer = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


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

# the task groups of panel c: the interactive ones on the left half, the automatic ones on the right half
INTERACTIVE_GROUPS = [
    ("Segmentation", ["interactive_2d", "interactive_3d"]),
    ("Tracking", ["interactive_2dt"]),
    ("Classification", ["pixel_classification", "object_classification"]),
]
AUTOMATIC_GROUPS = [
    ("Segmentation", ["automatic_2d", "automatic_3d"]),
    ("Tracking", ["automatic_2dt", "automatic_3dt"]),
    ("Batch Processing", ["automatic_batch", "automatic_batch_objects"]),
]


def group_width(names, scale=1.0):
    return scale * (sum(ITEMS[name][1] for name in names) + ITEM_GAP * (len(names) - 1))


def draw_group(drawing, cx, cy, header, names, scale=1.0, color=TEXT):
    """Draw a group of items centered at (cx, cy) and enlarged by scale, with its header above and labels below."""
    width = group_width(names, scale)
    drawing.text(cx, cy - 6.0 * scale - 1.7, header, 2.0, weight="bold", color=color)
    x = cx - width / 2
    for name in names:
        draw, item_width, label = ITEMS[name]
        scaled(drawing, scale, draw, x, cy)
        drawing.text(x + scale * item_width / 2, cy + 6.0 * scale + 2.3, label, 1.6, color=SUBTEXT)
        x += scale * (item_width + ITEM_GAP)


def superellipse(cx, cy, a, b, power=4.0, samples=240):
    """Get the points of a rounded rectangle-like superellipse."""
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = cx + a * np.sign(c) * np.abs(c) ** (2 / power)
    y = cy + b * np.sign(s) * np.abs(s) ** (2 / power)
    return np.stack([x, y], axis=1)


def compact_layout(scale, halo, rows):
    """Place the groups around the hub at x = 0: the middle group beside the hub, the others above and below it.

    Returns the centers of the groups of each side and how far each side reaches from the hub.
    """
    layout, reach = {}, {}
    for side, groups in ((-1, INTERACTIVE_GROUPS), (1, AUTOMATIC_GROUPS)):
        halves = [group_width(names, scale) / 2 for _, names in groups]
        # the upper and lower groups share one column, and the middle group stands as close to the hub as it can,
        # so that the three groups of a side trace one curve around the hub
        column = max(20.5, 6.5 + max(halves[0], halves[2]))
        offsets = [column, halo + 1.2 + halves[1], column]
        centers = [(side * offset, row) for offset, row in zip(offsets, rows)]
        layout[side] = centers
        reach[side] = max(offset + half for offset, half in zip(offsets, halves))
    return layout, reach


def prompt_icon(drawing, kind, x, cy):
    """Draw the icon of one kind of prompt with its left edge at x, and return its right edge."""
    if kind == "points":
        point_prompt(drawing, x + 0.55, cy, r=0.55)
        point_prompt(drawing, x + 2.0, cy, positive=False, r=0.55)
        return x + 2.6
    if kind == "scribbles":
        t = np.linspace(0, 1, 20)
        for dy, color in ((-0.45, TEAL), (0.45, ORANGE)):
            brush_stroke(drawing, x + 0.1 + 2.4 * t, cy + dy + 0.3 * np.sin(2 * np.pi * t), color, width=0.32)
        return x + 2.6
    # the shape prompts are drawn around a small cell, in the teal of the positive prompts
    cell = blob_path(x + 1.3, cy, 0.62, 1.25, 0.4, seed=7)
    if kind in ("box", "lasso"):
        drawing.add(f'<path d="{cell}" fill="#7b7c84"/>')
    if kind == "box":
        drawing.add(
            f'<rect x="{fmt(x + 0.15)}" y="{fmt(cy - 0.95)}" width="2.3" height="1.9" rx="0.15" fill="none" '
            f'stroke="{TEAL}" stroke-width="0.24"/>'
        )
        for px, py in ((x + 0.15, cy - 0.95), (x + 2.45, cy - 0.95), (x + 0.15, cy + 0.95), (x + 2.45, cy + 0.95)):
            drawing.add(
                f'<rect x="{fmt(px - 0.22)}" y="{fmt(py - 0.22)}" width="0.44" height="0.44" fill="#ffffff" '
                f'stroke="{TEAL}" stroke-width="0.14"/>'
            )
    elif kind == "lasso":
        t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        rr = 1.0 + 0.1 * np.sin(3 * t + 0.6) + 0.06 * np.sin(5 * t + 1.3)
        points = [(x + 1.3 + 1.2 * r * np.cos(a), cy + 0.82 * r * np.sin(a)) for r, a in zip(rr, t)]
        path = "M" + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in points) + "Z"
        drawing.add(
            f'<path d="{path}" fill="{TEAL}" fill-opacity="0.15" stroke="{TEAL}" stroke-width="0.24" '
            'stroke-dasharray="0.45,0.3" stroke-linejoin="round"/>'
        )
    else:
        # three slices of a stack, the middle one prompted, the mask carried to the others
        for k, dy in enumerate((-0.75, 0.0, 0.75)):
            middle = k == 1
            fill, edge = ("#d9f0f3", TEAL) if middle else ("#eceaf0", "#8d8d96")
            drawing.add(
                f'<path d="M{fmt(x + 0.1)},{fmt(cy + dy + 0.35)}l0.6,-0.7h1.8l-0.6,0.7Z" fill="{fill}" '
                f'stroke="{edge}" stroke-width="0.14" stroke-linejoin="round"/>'
            )
        point_prompt(drawing, x + 1.3, cy, r=0.28)
    return x + 2.6


PROMPTS = [
    ("points", "+ / − points"), ("box", "box"), ("scribbles", "+ / − scribbles"), ("lasso", "lasso / ellipse"),
    ("volume", "3D propagation"),
]
PROMPTS_W = 13.6


PROMPTS = [
    ("points", "+ / − points"), ("box", "box"), ("scribbles", "+ / − scribbles"), ("lasso", "lasso / ellipse"),
    ("volume", "3D propagation"),
]
PROMPTS_W = 13.6


def prompt_portfolio(drawing, x, header_y, top, pitch=2.75):
    """List the prompts of the annotation tools in a column, as checked in micro_sam/sam_annotator: positive and
    negative points and scribbles, boxes, lasso and ellipse shapes as mask prompts, and the propagation of the prompts
    of one slice or frame through a volume or a timeseries."""
    color = mix(INTERACTIVE_STYLE[1], "#000000", 0.3)
    # a very subtle card holds the legend together
    x0, y0 = x - 1.4, header_y - 2.8
    x1, y1 = x + PROMPTS_W + 1.6, top + (len(PROMPTS) - 1) * pitch + 2.0
    drawing.add(
        f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" height="{fmt(y1 - y0)}" rx="1.4" fill="#ffffff" '
        f'fill-opacity="0.35" stroke="{INTERACTIVE_STYLE[1]}" stroke-opacity="0.35" stroke-width="0.18"/>'
    )
    drawing.text(x + PROMPTS_W / 2, header_y, "Prompts", 2.0, weight="bold", color=color)
    for k, (kind, text) in enumerate(PROMPTS):
        cy = top + k * pitch
        right = prompt_icon(drawing, kind, x, cy)
        drawing.text(right + 0.7, cy + 0.45, text, 1.3, color=SUBTEXT, anchor="start")


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


def draw_hub(drawing, cx, cy, radius, logo_path, napari_logo, python_logo):
    """Draw the μSAM2 hub as a white disc with the logo and the name, and its three interfaces as round badges on the
    lower rim of the disc, like the correction badge on its upper rim."""
    drawing.add(
        f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(radius)}" fill="#ffffff" stroke="{RIM}" '
        'stroke-width="0.35"/>'
    )
    # the full logo of micro-sam with its name, centered in the disc
    uri, aspect = logo_data_uri(logo_path, width_px=900)
    logo_w = 16.0
    logo_h = logo_w / aspect
    drawing.add(
        f'<image x="{fmt(cx - logo_w / 2)}" y="{fmt(cy - logo_h / 2)}" width="{fmt(logo_w)}" height="{fmt(logo_h)}" '
        f'preserveAspectRatio="xMidYMid meet" xlink:href="{uri}"/>'
    )
    interfaces = [("napari", svg_uri(napari_logo)), ("Python", svg_uri(python_logo)), ("CLI", None)]
    badge, icon = 2.05, 2.5
    for angle, (label, icon_uri) in zip((236, 270, 304), interfaces):
        t = np.radians(angle)
        bx, by = cx + radius * np.cos(t), cy - radius * np.sin(t)
        drawing.add(
            f'<circle cx="{fmt(bx)}" cy="{fmt(by)}" r="{fmt(badge)}" fill="#ffffff" stroke="{RIM}" '
            'stroke-width="0.3"/>'
        )
        if icon_uri is None:
            # the prompt of a terminal
            s_ = 1.5
            drawing.add(
                f'<path d="M{fmt(bx - 0.55 * s_)},{fmt(by - 0.4 * s_)}l{fmt(0.4 * s_)},{fmt(0.32 * s_)}'
                f'l{fmt(-0.4 * s_)},{fmt(0.32 * s_)}M{fmt(bx + 0.05 * s_)},{fmt(by + 0.42 * s_)}h{fmt(0.5 * s_)}" '
                f'fill="none" stroke="#2d2b36" stroke-width="0.24" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        else:
            drawing.add(
                f'<image x="{fmt(bx - icon / 2)}" y="{fmt(by - icon / 2)}" width="{fmt(icon)}" height="{fmt(icon)}" '
                f'xlink:href="{icon_uri}"/>'
            )
        # the name of each interface sits right below its badge
        drawing.text(bx, by + badge + 1.55, label, 1.3, color=SUBTEXT)


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
AUTO_BAND_H, INTER_H = 71.0, 52.0
INTER_Y = AUTO_BAND_H + A_BAND_GAP
A_H = INTER_Y + INTER_H
A_CONTENT_X = 69.0
# the embeddings enter the automatic head at the height of the 2D embedding, which is its first decoder stage
A_ENTRY = 19.0
A_OUTPUT_X = 151.5
A_SCALE = 1.35
# the rows of the 2D and the 3D outputs, relative to the top of a band
A_ROW_2D, A_ROW_3D, AUTO_ROW_3D = 15.5, 40.0, 48.5
# the Hiera stages as (tokens per side, size, layers, global attention), and the decoder stages as (channels,
# height, depth) of their feature maps
HIERA_STAGES = [(16, 10.8, 2, False), (8, 8.4, 3, False), (4, 6.5, 5, True), (2, 4.9, 3, True)]
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
    draw(drawing, 0, 0)
    inner = "".join(drawing.body[start:])
    del drawing.body[start:]
    drawing.add(f'<g transform="translate({fmt(x)},{fmt(cy)}) scale({fmt(scale)})">{inner}</g>')


def flow_arrow(drawing, points, head=0.9, width=0.25, color=ARROW):
    """Draw a thin polyline arrow with a filled head at its last point."""
    points = [np.asarray(p, dtype=float) for p in points]
    direction = points[-1] - points[-2]
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    end, tip = points[-1] - direction * head * 0.8, points[-1]
    left, right = tip - direction * head + normal * head * 0.45, tip - direction * head - normal * head * 0.45
    line = "M" + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in points[:-1] + [end])
    drawing.add(
        f'<path d="{line}" fill="none" stroke="{color}" stroke-width="{fmt(width)}" stroke-linejoin="round"/>'
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


def slab_stack(drawing, x, cy, count, style, w=3.4, d=2.2, gap=0.65):
    """Draw a stack of slices, as the features of a volume with their z axis pointing up; return its right edge."""
    right = x
    for k in range(count):
        right = embedding_slab(drawing, x, cy + 0.6 * (count - 1) * gap / 2 + 1.0 - gap * k, w, d, n=3, style=style)
    return right


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


def to_world(points):
    """Map unit coordinates (x, y, z) to the world coordinates of the renders, where y points up and z to the viewer."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.stack([points[:, 0], 1 - points[:, 1], 1 - points[:, 2]], axis=1) * (VOLUME_N - 1)


def color_grid(dimensions, origin, data):
    grid = pv.ImageData(dimensions=dimensions, origin=origin)
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


def render(add, size=900, edges=True):
    """Render a scene off screen from the viewpoint of the blocks, inside the edges of the volume, and crop it."""
    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    # translucent objects need depth peeling to be drawn in the right order
    plotter.enable_depth_peeling()
    add(plotter)
    n = VOLUME_N - 1
    if edges:
        plotter.add_mesh(pv.Box(bounds=(0, n, 0, n, 0, n)).extract_all_edges(), color=EDGE, line_width=3)
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


def add_point_prompt(plotter, point, radius=6.5):
    """Lay a positive point prompt flat onto the slice that holds it."""
    center = to_world(point)[0] + (0, 0, 0.6)
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
def render_slices():
    """The slices of the volume, pulled apart, as the image encoder sees them one after the other."""
    slabs = [slice_slab(em_slice(k), k, spread=1.6) for k in SLICE_INDICES]
    return render(lambda plotter: add_colors(plotter, slabs), edges=False)


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
        plotter.add_mesh(plane, color=TEAL, line_width=4)
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


def overlay(base, masks):
    """Lay masks with a blue to purple gradient and a light rim over a gray image."""
    n = base.shape[0]
    rows, cols = np.mgrid[:n, :n]
    g = ((rows + cols) / (2 * n))[..., None]
    gradient = (1 - g) * hex_to_rgb(BLUE) + g * hex_to_rgb(PURPLE)
    image = base.copy()
    for mask in masks:
        rim = mask & ~ndimage.binary_erosion(mask, iterations=2)
        image = np.where(mask[..., None], 0.35 * image + 0.65 * gradient, image)
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
    draw_slice_prompts(drawing, *slice_tile(drawing, x, cy, overlay(em_slice(MIDDLE), masks)), negative=False)


def slice_automatic(drawing, x, cy):
    """All profiles of the middle slice, each in its own color."""
    labels = slice_labels()
    colors = neurite_colors(int(labels.max()) + 1)
    base = em_slice(MIDDLE)
    image = np.where((labels > 0)[..., None], 0.45 * base + 0.55 * colors[labels], base)
    image[membranes(labels)] *= 0.45
    slice_tile(drawing, x, cy, image)


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


def outputs(drawing, draw_2d, draw_3d, y, row_2d=None, row_3d=None):
    """Draw the 2D output above the 3D output, and return the left edge of each."""
    row_2d = A_ROW_2D if row_2d is None else row_2d
    row_3d = A_ROW_3D if row_3d is None else row_3d
    w2, w3 = SIZE * A_SCALE, (SIZE + 3.5) * A_SCALE
    x2 = A_OUTPUT_X + (w3 - w2) / 2
    scaled(drawing, A_SCALE, draw_2d, x2, y + row_2d)
    item_label(drawing, A_OUTPUT_X, w3, y + row_2d + w2 / 2 + 2.2, "2D")
    scaled(drawing, A_SCALE, draw_3d, A_OUTPUT_X, y + row_3d)
    item_label(drawing, A_OUTPUT_X, w3, y + row_3d + w3 / 2 + 2.3, "3D")
    return x2, A_OUTPUT_X


def a_band(drawing, x, y, height, title, background, title_y):
    drawing.add(
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(A_W - x)}" height="{fmt(height)}" rx="1.2" '
        f'fill="{background}"/>'
    )
    drawing.text(x + 3.3, title_y, title, 2.12, weight="bold", color=BLUE, rotate=-90)


def token_grid(drawing, x, y, size, n, highlight=None):
    """Draw one transformer block as a square grid of n x n tokens; highlight is the side of an attention window
    in tokens, or n for global attention."""
    fill, edge = ENCODER_STYLE
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
    stage. The early stages attend within local windows, the late stages globally, and a neck right next to the
    stages merges all of them. Return the bottom of the stages."""
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
            flow_arrow(drawing, [(cx, y + 0.3), (cx, y + 2.1)], head=0.6, width=0.2)
            drawing.text(cx + 0.8, y + 1.7, "pool", 1.25, color=SUBTEXT, anchor="start")
            y += 2.4
    neck_top, neck_bottom = top - 0.5, y + 0.5
    fill, edge = ENCODER_STYLE
    drawing.add(
        f'<rect x="{fmt(neck_x)}" y="{fmt(neck_top)}" width="3.6" height="{fmt(neck_bottom - neck_top)}" rx="0.6" '
        f'fill="{fill}" stroke="{edge}" stroke-width="0.2"/>'
    )
    drawing.text(neck_x + 1.8, (neck_top + neck_bottom) / 2, "neck", 1.5, rotate=-90)
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
    """Draw the inputs, the hierarchical image encoder and the embeddings, which both heads share. The inputs stack in
    one column, where a volume reaches the image encoder slice by slice."""
    # the 2D embedding lines up with the entry of the automatic head, so that its line runs straight through
    cy_2d, cy_block, cy_3d = A_ENTRY, 37.5, 58.5
    column = 14.0
    drawing.text(column / 2, A_HEADER_Y, "Input", 2.12, weight="bold")
    scaled(drawing, 13.0 / SIZE, slice_input, 0.5, cy_2d)
    item_label(drawing, 0.0, column, cy_2d + 6.5 + 2.4, "2D image")
    height = place_render(drawing, render_em_block(), 0.5, cy_block, 13.0)
    item_label(drawing, 0.0, column, cy_block + height / 2 + 2.4, "3D volume")
    flow_arrow(drawing, [(column / 2, cy_block + height / 2 + 3.3), (column / 2, cy_3d - 6.6)])
    height = place_render(drawing, render_slices(), 0.25, cy_3d, 13.5)
    item_label(drawing, 0.0, column, cy_3d + height / 2 + 2.4, "Z slices")
    # the Hiera encoder, a hierarchical vision transformer, downsamples each image or slice in stages
    (x0, x1), top = A_ENCODER_X, 1.5
    box_index = len(drawing.body)
    drawing.text((x0 + x1) / 2, A_HEADER_Y, "Image encoder", 2.12, weight="bold")
    # the box fits the stages, its legend and its caption, and still takes the arrow of the volume
    bottom = max(hiera(drawing, x0 + 7.8, top + 8.0) + 13.5, cy_3d + 6.0)
    fill, edge = ENCODER_STYLE
    drawing.body.insert(box_index, (
        f'<rect x="{fmt(x0)}" y="{fmt(top)}" width="{fmt(x1 - x0)}" height="{fmt(bottom - top)}" rx="1.2" '
        f'fill="{mix(fill, "#ffffff", 0.55)}" stroke="{edge}" stroke-width="0.25"/>'
    ))
    attention_legend(drawing, x0 + 1.4, bottom - 9.6)
    item_label(drawing, x0, x1 - x0, bottom - 2.2, "Hierarchical ViT (Hiera)", size=1.5)
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
        f'M{fmt(bus_x)},{fmt(cy_2d)}V{fmt(INTER_Y + 5.0)}" fill="none" stroke="{ARROW}" stroke-width="0.25"/>'
    )
    return bus_x


def propagation(drawing, x, top, count=7, size=8.6, pitch=10.4, step=14):
    """Show how the memory carries the mask of the neuron from the prompted middle slice to the slices on both sides,
    as EM slices of the volume with the mask that each of them gets."""
    middle, labels = count // 2, neuron_volume()
    for k in range(count):
        index = MIDDLE + (k - middle) * step
        image = overlay(em_slice(index), [labels[index] == NEURON_LABEL])
        sx = x + k * pitch
        clip = drawing.clip_rect(sx, top, size, size, 0.6)
        color, width = (TEAL, 0.45) if k == middle else (SLICE_STYLE[1], 0.2)
        drawing.add(
            f'<image x="{fmt(sx)}" y="{fmt(top)}" width="{fmt(size)}" height="{fmt(size)}" preserveAspectRatio="none" '
            f'clip-path="url(#{clip})" xlink:href="{array_uri(image)}"/>'
            f'<rect x="{fmt(sx)}" y="{fmt(top)}" width="{fmt(size)}" height="{fmt(size)}" rx="0.6" fill="none" '
            f'stroke="{color}" stroke-width="{fmt(width)}"/>'
        )
        if k == middle:
            point_prompt(drawing, sx + PROMPT[0] * size, top + PROMPT[1] * size, r=0.5)
        drawing.text(sx + size / 2, top + size + 1.9, f"z = {index}", 1.25, color=SUBTEXT)
    # the mask hops from slice to slice, forward and backward from the prompted slice, numbered in their order
    t = np.linspace(0, 1, 24)[:, None]
    for step in range(1, middle + 1):
        for direction in (1, -1):
            a, b = middle + direction * (step - 1), middle + direction * step
            start = np.array([x + a * pitch + size / 2 + direction * 1.2, top - 0.3])
            end = np.array([x + b * pitch + size / 2 - direction * 1.2, top - 0.3])
            c1, c2 = start + (0, -2.6), end + (0, -2.6)
            curve = (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * end
            flow_arrow(drawing, list(curve), head=0.6, width=0.22)
            drawing.text((start[0] + end[0]) / 2, top - 2.6, str(step), 1.05, weight="bold", color=SUBTEXT)
    center, half = x + middle * pitch + size / 2, (middle + 1) / 2 * pitch
    label_y, reach = top + size + 4.4, 7.9
    for direction, text in ((-1, "backward propagation"), (1, "forward propagation")):
        anchor_x = center + direction * half
        drawing.text(anchor_x, label_y, text, 1.3, weight="bold", color=SUBTEXT)
        # a small arrow on the outer side of the label points the way of the propagation
        arrow = [(anchor_x + direction * reach, label_y - 0.45), (anchor_x + direction * (reach + 2.4), label_y - 0.45)]
        flow_arrow(drawing, arrow, head=0.6, width=0.22)
    drawing.text(center, top + size + 4.4, "prompted", 1.3, color=TEAL_DARK)


def memory_bank(drawing, x, cy, size=5.6, gap=0.7, step=14, squeeze=0.45, shear=-0.32):
    """Draw the memory bank as the masks of the slices that were already processed, standing side by side like books
    with a small gap and facing left, the way the memory flows, with its label."""
    labels = neuron_volume()
    # the prompted middle slice stands in front, the slices that the memory reached from it behind
    indices = [MIDDLE, MIDDLE + step, MIDDLE - step, MIDDLE + 2 * step]
    top = cy - size / 2 - shear * size * squeeze / 2
    w = size * squeeze
    shift = w + gap
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
    label_x = x + (len(indices) - 1) * shift + w + 1.2
    # the label sits above the loop that runs through the slices
    drawing.text(label_x, cy - 2.9, "memory bank", 1.25, weight="bold", color=TEXT, anchor="start")
    drawing.text(label_x, cy - 0.9, "masks of processed slices", 1.2, color=SUBTEXT, anchor="start")


def draw_interactive_head(drawing, bus_x):
    """Draw the interactive band below the trunk: the prompts below the inputs, the prompt encoder below the image
    encoder, the memory attention that propagates a mask through volumes, and the mask decoder."""
    y0 = INTER_Y
    a_band(drawing, 0.0, y0, INTER_H, "Interactive Segmentation", INTERACTIVE_BACKGROUND, y0 + 26.0)
    decoder_x = A_OUTPUT_X - 19.5
    decoder_right = decoder_x + 13.5
    module(drawing, decoder_x, y0 + 2.0, decoder_right - decoder_x, 46.0, ["Mask", "decoder"], INTERACTIVE_STYLE)
    # 2D embeddings go straight to the mask decoder, the slices of a volume pass the memory attention first
    split, bypass, memory = bus_x + 3.0, y0 + 3.5, y0 + 10.5
    drawing.add(
        f'<path d="M{fmt(bus_x)},{fmt(y0 - A_BAND_GAP)}V{fmt(y0 + 5.0)}H{fmt(split)}M{fmt(split)},{fmt(bypass)}'
        f'V{fmt(memory)}" fill="none" stroke="{ARROW}" stroke-width="0.25" stroke-linejoin="round"/>'
    )
    flow_arrow(drawing, [(split, bypass), (decoder_x - 0.7, bypass)])
    drawing.text(split + 1.5, bypass - 0.8, "2D", 1.3, color=SUBTEXT, anchor="start")
    memory_x = split + 6.0
    flow_arrow(drawing, [(split, memory), (memory_x - 0.7, memory)])
    drawing.text(split + 1.5, memory - 0.8, "3D", 1.3, color=SUBTEXT, anchor="start")
    module(drawing, memory_x, memory - 3.6, 26.0, 7.2, ["Memory attention", "(3D only)"], INTERACTIVE_STYLE)
    flow_arrow(drawing, [(memory_x + 26.7, memory), (decoder_x - 0.7, memory)])
    # the masks of the slices go into a memory bank, which conditions the next slices
    loop_y, loop_x = y0 + 17.5, memory_x + 20.0
    flow_arrow(drawing, [(decoder_x - 0.3, loop_y), (loop_x, loop_y), (loop_x, memory + 4.2)])
    memory_bank(drawing, loop_x + 10.0, loop_y - 0.4)
    propagation(drawing, decoder_x - 82.0, y0 + 24.5)
    # prompts are placed on a 2D image, or on one slice of a volume, right below the inputs
    row_2d, row_3d = y0 + 15.5, y0 + 39.0
    drawing.text(12.0, y0 + 5.0, "Prompts", 1.76, color=SUBTEXT)
    scaled(drawing, 12.0 / SIZE, slice_prompted, 6.0, row_2d)
    place_render(drawing, render_prompted_slice(), 4.5, row_3d, 15.0)
    encoder_x, encoder_y, elbow = 27.0, y0 + 27.0, 23.5
    module(drawing, encoder_x, encoder_y - 5.0, 18.0, 10.0, ["Prompt", "encoder"], INTERACTIVE_STYLE)
    for start, row, end in ((18.4, row_2d, encoder_y - 2.0), (19.8, row_3d, encoder_y + 2.0)):
        flow_arrow(drawing, [(start, row), (elbow, row), (elbow, end), (encoder_x - 0.7, end)])
    # the prompt embeddings run below the memory to the mask decoder
    lane = y0 + 46.5
    flow_arrow(drawing, [(encoder_x + 9.0, encoder_y + 5.6), (encoder_x + 9.0, lane), (decoder_x - 0.7, lane)])
    x2, x3 = outputs(drawing, slice_interactive, interactive_3d_render, y0)
    flow_arrow(drawing, [(decoder_right + 0.7, y0 + A_ROW_2D), (x2 - 1.2, y0 + A_ROW_2D)])
    flow_arrow(drawing, [(decoder_right + 0.7, y0 + A_ROW_3D), (x3 - 1.2, y0 + A_ROW_3D)])


def operator(drawing, x, y, symbol):
    drawing.text(x, y + 0.6, symbol, 1.8, weight="bold")


def strip_pooling(drawing, x, y, width):
    """Sketch the DSP block (Depth Strip Pooling) after each 3D block of the decoder, as in
    torch_em.model.unetr: the features
    x of all slices are averaged over z, two 1x1x1 convolutions with batch norm and ReLU in between and a sigmoid turn
    the average into a gate g, and DSP(x) = x + x * g applies the gate to every slice as a gated residual."""
    fill, edge = AUTOMATIC_STYLE
    small = 1.05
    bold = '<tspan style="font-weight:bold">{}</tspan>'.format
    title = f"{bold('DSP')} ({bold('D')}epth {bold('S')}trip {bold('P')}ooling) {bold('Block')}"
    drawing.text(x + width / 2, y + 3.6, title, 1.45)
    row = y + 10.2
    right = slab_stack(drawing, x + 1.2, row, 4, AUTOMATIC_STYLE, w=3.0, d=2.0)
    drawing.text(x + 3.4, row + 3.8, "x: Z × H × W", small, color=SUBTEXT)
    flow_arrow(drawing, [(right + 0.6, row), (right + 3.4, row)], head=0.55, width=0.2)
    drawing.text(right + 2.0, row + 1.6, "mean z", small, color=SUBTEXT)
    slice_x = right + 3.7
    right = embedding_slab(drawing, slice_x, row + 0.8, 3.0, 2.0, n=3, style=AUTOMATIC_STYLE)
    drawing.text(slice_x + 2.2, row + 3.8, "1 × H × W", small, color=SUBTEXT)
    flow_arrow(drawing, [(right + 0.3, row), (right + 1.5, row)], head=0.55, width=0.2)
    # the small network that turns the average into the gate, as one capsule with a cell per layer
    cells = [
        (3.8, ("conv", "1×1×1"), fill), (3.4, ("BN,", "ReLU"), fill), (3.8, ("conv", "1×1×1"), fill),
        (2.4, ("σ",), GATE_STYLE[0]),
    ]
    cx, top, height = right + 1.8, row - 1.9, 3.8
    total = sum(w for w, _, _ in cells)
    clip = drawing.clip_rect(cx, top, total, height, 1.1)
    parts = [f'<g clip-path="url(#{clip})">']
    offset = cx
    for w, _, color in cells:
        parts.append(f'<rect x="{fmt(offset)}" y="{fmt(top)}" width="{fmt(w)}" height="{fmt(height)}" fill="{color}"/>')
        offset += w
    parts.append("</g>")
    separators = "".join(f"M{fmt(cx + sum(w for w, _, _ in cells[:k]))},{fmt(top)}v{fmt(height)}" for k in range(1, 4))
    parts.append(f'<path d="{separators}" stroke="{edge}" stroke-width="0.15"/>')
    parts.append(
        f'<rect x="{fmt(cx)}" y="{fmt(top)}" width="{fmt(total)}" height="{fmt(height)}" rx="1.1" fill="none" '
        f'stroke="{edge}" stroke-width="0.22"/>'
    )
    drawing.add("".join(parts))
    offset = cx
    for w, lines, _ in cells:
        for k, line in enumerate(lines):
            drawing.text(offset + w / 2, row + 0.38 + (k - (len(lines) - 1) / 2) * 1.2, line, 0.98)
        offset += w
    flow_arrow(drawing, [(cx + total + 0.3, row), (cx + total + 1.5, row)], head=0.55, width=0.2)
    gate_x = cx + total + 1.8
    embedding_slab(drawing, gate_x, row + 0.8, 3.0, 2.0, n=3, style=GATE_STYLE)
    drawing.text(gate_x + 2.2, row + 3.8, "gate g", small, color=GATE_STYLE[1])
    # the gated residual, with the gate broadcast to every slice
    g = f'<tspan style="fill:{GATE_STYLE[1]}">g</tspan>'
    drawing.text(x + width / 2, y + 19.8, f"DSP(x) = x + x ⊙ {g}", 1.8, weight="bold")
    drawing.text(x + width / 2, y + 22.8, "g is broadcast over z", 1.15, color=SUBTEXT)
    drawing.text(x + width / 2, y + 25.3, "3D only (2D: DSP(x) = x)", 1.15, weight="bold", color=SUBTEXT)


def draw_automatic_head(drawing, bus_x):
    """Draw the segmentation decoder, its foreground, distance and boundary predictions, and the segmentation."""
    a_band(drawing, A_BAND_X, 0.0, AUTO_BAND_H, "Automatic Segmentation", AUTOMATIC_BACKGROUND, 40.0)
    # the content sits a little lower, so that the first decoder stage lines up with the 2D embedding
    y = 0.0
    # the box closes right below the sketch of the DSP block
    x0, x1, top = A_CONTENT_X, A_CONTENT_X + 40.0, y + 4.0
    bottom = top + 60.0
    fill, edge = AUTOMATIC_STYLE
    drawing.add(
        f'<rect x="{fmt(x0)}" y="{fmt(top)}" width="{fmt(x1 - x0)}" height="{fmt(bottom - top)}" rx="1.2" '
        f'fill="{mix(fill, "#ffffff", 0.55)}" stroke="{edge}" stroke-width="0.25"/>'
    )
    drawing.text((x0 + x1) / 2, top + 4.5, "Segmentation decoder", 2.12, weight="bold")
    flow_arrow(drawing, [(bus_x, A_ENTRY), (x0 - 0.7, A_ENTRY)])
    # the decoder upsamples the embeddings back to the image resolution, and every 3D block ends with a DSP block
    cy, x, markers = A_ENTRY, x0 + 2.0, []
    zoom_index = len(drawing.body)
    # a 2D input passes straight on from stage to stage, a volume takes the detour through the DSP block
    gap, drop = 5.6, 2.7
    for k, (w, h, d) in enumerate(STAGES[::-1]):
        x = feature_map(drawing, x, cy, w, 0.85 * h, 0.75 * d, AUTOMATIC_STYLE)
        mx = x + gap / 2
        markers.append(mx)
        flow_arrow(drawing, [(x + 0.25, cy), (x + gap - 0.3, cy)], head=0.5, width=0.2)
        drawing.add(
            f'<path d="M{fmt(x + 0.8)},{fmt(cy)}V{fmt(cy + drop)}H{fmt(mx - 1.25)}M{fmt(mx + 1.25)},{fmt(cy + drop)}'
            f'H{fmt(x + gap - 0.8)}V{fmt(cy)}" fill="none" stroke="{ARROW}" stroke-width="0.2" '
            'stroke-linejoin="round" stroke-dasharray="0.35,0.3"/>'
        )
        if k == 0:
            drawing.text(mx, cy + drop + 2.4, "3D", 0.85, color=SUBTEXT)
        x += gap
    for mx in markers:
        drawing.add(
            f'<circle cx="{fmt(mx)}" cy="{fmt(cy + drop)}" r="1.25" fill="#ffffff" stroke="{edge}" '
            'stroke-width="0.22"/>'
        )
        drawing.text(mx, cy + drop + 0.32, "DSP", 0.85, weight="bold", color=edge)
    # the sketch of the DSP block opens from one DSP block like a magnifying cone, drawn behind the feature maps
    frame_x, frame_top, frame_w = x0 + 1.2, top + 31.0, x1 - x0 - 2.4
    frame_bottom = frame_top + 27.8
    mx = markers[2]
    cone = [
        (mx - 1.0, cy + drop + 0.7), (mx + 1.0, cy + drop + 0.7), (frame_x + frame_w, frame_top), (frame_x, frame_top)
    ]
    drawing.body.insert(zoom_index, (
        '<path d="M' + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in cone) + f'Z" fill="#ffffff" '
        f'fill-opacity="0.55" stroke="none"/><path d="M{fmt(cone[0][0])},{fmt(cone[0][1])}L{fmt(cone[3][0])},'
        f'{fmt(cone[3][1])}M{fmt(cone[1][0])},{fmt(cone[1][1])}L{fmt(cone[2][0])},{fmt(cone[2][1])}" '
        f'stroke="{edge}" stroke-width="0.15" stroke-opacity="0.7"/>'
    ))
    frame_h = frame_bottom - frame_top
    drawing.add(
        f'<rect x="{fmt(frame_x)}" y="{fmt(frame_top)}" width="{fmt(frame_w)}" height="{fmt(frame_h)}" '
        f'rx="0.8" fill="#ffffff" fill-opacity="0.85" stroke="{edge}" stroke-width="0.2"/>'
    )
    strip_pooling(drawing, frame_x, frame_top, frame_w)
    # the outputs of the decoder, computed with the training targets of micro-sam; z only exists in volumes
    maps_2d = decoder_maps()[0]
    names = ["foreground", "boundary", "x-distance", "y-distance", "z-distance"]
    front, depth, gap = 4.8, 1.6, 0.7
    pitch, mx = front + depth + gap, x1 + 4.0
    # the 3D row sits lower than in the interactive band, since the automatic band is as tall as the inputs
    row_2d, row_3d = A_ENTRY, y + AUTO_ROW_3D
    # the four 2D maps stretch over the whole row, the five 3D maps sit in their own row; each row names its maps
    width = len(names) * pitch - gap
    tile = (width - 3 * gap) / 4
    for k, name in enumerate(names[:4]):
        tx = mx + k * (tile + gap)
        drawing.text(tx + tile / 2, row_2d - tile / 2 - 1.2, name, 1.3, color=SUBTEXT)
        map_tile(drawing, tx, row_2d - tile / 2, tile, maps_2d[name])
    # the 3D maps are larger and split in two lines, the foreground and boundary above the three distances, and the
    # 3D output sits between the two lines
    block, lines = 10.0, [(names[:2], row_3d - 7.2), (names[2:], row_3d + 6.9)]
    for line, cy in lines:
        spacing = (width - len(line) * block) / (len(line) - 1) if len(line) > 2 else 4.0
        start = mx + (width - len(line) * block - (len(line) - 1) * spacing) / 2
        for k, name in enumerate(line):
            tx = start + k * (block + spacing)
            height = place_render(drawing, render_map_block(name), tx, cy, block)
            drawing.text(tx + block / 2, cy - height / 2 - 1.1, name, 1.3, color=SUBTEXT)
    end = mx + width
    for row in (row_2d, row_3d):
        flow_arrow(drawing, [(x1 + 0.7, row), (mx - 0.7, row)])
    x2, x3 = outputs(drawing, slice_automatic, automatic_3d_render, y, row_2d=A_ENTRY, row_3d=AUTO_ROW_3D)
    flow_arrow(drawing, [(end + 0.7, row_2d), (x2 - 1.2, row_2d)])
    flow_arrow(drawing, [(end + 0.7, row_3d), (x3 - 1.2, row_3d)])


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


def build_panel_b():
    """Draw panel b, the overview of the training data. Not implemented yet, it needs access to the datasets."""
    raise NotImplementedError("Panel b is not implemented yet.")


def build_panel_a():
    """Draw panel a, the architecture of μSAM2, in its own coordinates in mm."""
    drawing = Drawing(text_scale=A_TEXT_SCALE)
    bus_x = draw_trunk(drawing)
    draw_automatic_head(drawing, bus_x)
    draw_interactive_head(drawing, bus_x)
    w3 = (SIZE + 3.5) * A_SCALE
    drawing.text(A_OUTPUT_X + w3 / 2, A_HEADER_Y, "Output", 2.12, weight="bold")
    body = f'<g transform="scale({fmt(A4_WIDTH / A_W)})">{"".join(drawing.body)}</g>'
    content = f'<defs>{shared_defs()}{"".join(drawing.defs)}</defs>{body}'
    # the ids must differ from the ones of panel c, since both panels end up in the same figure
    return content.replace("f1c-", "f1a-")


def build_panel_c(logo_path, napari_logo, python_logo):
    """Draw panel c in its own coordinates in mm: the μSAM2 hub in the middle, and the tasks stacked in two halves that
    wrap it, the interactive ones on the left and the automatic ones on the right, each half as wide as its content.
    Returns the content and the size of the panel."""
    height, scale = 80.0, 1.22
    # the halo leaves room for the badges of the interfaces and their names on the lower rim of the hub
    hub_r, gap, halo, margin = 10.5, 1.4, 15.6, 6.3
    rows = (-24.5, 0.0, 24.5)
    layout, reach = compact_layout(scale, halo, rows)
    # the prompts stand as a small group of their own beside the interactive tracking, on its outer side
    tracking_dx = layout[-1][1][0]
    prompts_x = tracking_dx - group_width(INTERACTIVE_GROUPS[1][1], scale) / 2 - 2.4 - PROMPTS_W
    reach[-1] = max(reach[-1], -prompts_x)
    a_left, a_right = reach[-1] + margin, reach[1] + margin
    width = a_left + a_right + 1.4
    cx, cy, b = a_left + 0.7, height / 2, 39.4
    drawing = Drawing()
    halves = ((-1, a_left, INTERACTIVE_BACKGROUND, INTERACTIVE_STYLE[1]),
              (1, a_right, AUTOMATIC_BACKGROUND, AUTOMATIC_STYLE[1]))
    for side, a, color, edge in halves:
        shape = "M" + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in superellipse(cx, cy, a, b)) + "Z"
        x0 = 0.0 if side < 0 else cx + gap
        clip = drawing.clip_rect(x0, 0, (cx - gap) if side < 0 else (width - cx - gap), height, 0)
        # a thin outline in the darker band color traces the outer edge of each half
        drawing.add(
            f'<path d="{shape}" fill="{color}" stroke="{edge}" stroke-width="0.3" stroke-opacity="0.8" '
            f'clip-path="url(#{clip})"/>'
        )
    # a white halo around the hub cuts the two halves into crescents
    drawing.add(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(halo)}" fill="#ffffff"/>')
    drawing.text(cx - a_left + 2.4, cy, "Interactive Analysis", 2.12, weight="bold", color=BLUE, rotate=-90)
    drawing.text(cx + a_right - 2.4, cy, "Automatic Analysis", 2.12, weight="bold", color=BLUE, rotate=90)
    # the headers take a dark shade of the color of their half
    header_colors = {
        -1: mix(INTERACTIVE_STYLE[1], "#000000", 0.3), 1: mix(AUTOMATIC_STYLE[1], "#000000", 0.35)
    }
    for side, groups in ((-1, INTERACTIVE_GROUPS), (1, AUTOMATIC_GROUPS)):
        for (header, names), (dx, dy) in zip(groups, layout[side]):
            draw_group(drawing, cx + dx, cy + dy + 1.0, header, names, scale=scale, color=header_colors[side])
    draw_hub(drawing, cx, cy, hub_r, logo_path, napari_logo, python_logo)
    # correction runs over the hub, from the automatic half to the interactive half
    radius = halo + 0.4
    angles = np.radians(np.linspace(62, 118, 60))
    points = [(cx + radius * np.cos(t), cy - radius * np.sin(t)) for t in angles]
    gid = drawing.linear_gradient(*points[0], *points[-1])
    tip = np.array(points[-1])
    direction = tip - np.array(points[-5])
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    head = 1.4
    left, right = tip - direction * head + normal * head * 0.6, tip - direction * head - normal * head * 0.6
    line = "M" + "L".join(f"{fmt(px)},{fmt(py)}" for px, py in points[:-4])
    drawing.add(
        f'<path d="{line}" fill="none" stroke="url(#{gid})" stroke-width="0.6" stroke-linecap="round"/>'
        f'<path d="M{fmt(tip[0])},{fmt(tip[1])}L{fmt(left[0])},{fmt(left[1])}'
        f'L{fmt(right[0])},{fmt(right[1])}Z" fill="url(#{gid})"/>'
    )
    badge = (cx, cy - radius)
    loop_badge(drawing, badge, r=2.3)
    correction_icon(drawing, badge)
    drawing.text(cx, badge[1] - 3.3, "correction", 1.76, weight="bold", color=SUBTEXT)
    # the card of the legend sits in the middle between the title of the half and the tracking stack
    # the vertical title extends to the left of its baseline, so its right edge is just past the baseline
    title_right = cx - a_left + 2.4 + 0.3
    stack_left = cx + tracking_dx - group_width(INTERACTIVE_GROUPS[1][1], scale) / 2
    card_left, card_right = cx + prompts_x - 1.4, cx + prompts_x + PROMPTS_W + 1.6
    shift = ((stack_left - card_right) - (card_left - title_right)) / 2
    prompt_portfolio(drawing, cx + prompts_x + shift, cy + rows[1] + 1.0 - 6.0 * scale - 1.7, cy + rows[1] - 4.5)
    content = f'<defs>{shared_defs()}{"".join(drawing.defs)}</defs>{"".join(drawing.body)}'
    return content, width, height


def save_panel(content, output_folder, name, width, height):
    """Save the panel as a standalone svg and export it to pdf and png."""
    svg_path = os.path.join(output_folder, f"{name}.svg")
    with open(svg_path, "w") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{fmt(width)}mm" height="{fmt(height)}mm" viewBox="0 0 {fmt(width)} {fmt(height)}">'
            f"{content}</svg>\n"
        )
    for extension, options in (("pdf", []), ("png", ["--export-dpi=600", "--export-background=white"])):
        out_path = os.path.join(output_folder, f"{name}.{extension}")
        subprocess.run(
            ["inkscape", svg_path, f"--export-type={extension}", f"--export-filename={out_path}"] + options,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    print(f"Saved {name} as svg, pdf and png in {output_folder}")


def insert_panel(figure_path, content, panel):
    """Replace the layer of a panel in the figure draft with the new panel."""
    with open(figure_path) as f:
        figure = f.read()
    figure = re.sub(rf'<g id="panel-{panel}".*?<!-- /panel-{panel} -->\n?', "", figure, flags=re.S)
    x, y = PANEL_ORIGINS[panel]
    layer = (
        f'<g id="panel-{panel}" inkscape:label="Panel {panel}" inkscape:groupmode="layer" '
        f'transform="translate({fmt(x)},{fmt(y)})">{content}</g><!-- /panel-{panel} -->\n'
    )
    idx = figure.rstrip().rfind("</svg>")
    with open(figure_path, "w") as f:
        f.write(figure[:idx] + layer + figure[idx:])
    print(f"Updated panel {panel} in {figure_path}")


def main():
    parser = argparse.ArgumentParser(description="Make the panels of Figure 1.")
    parser.add_argument("--output_folder", default=".", help="The folder for the panel files.")
    parser.add_argument("--figure", default="fig1.svg", help="The Figure 1 draft whose panels a and c are replaced.")
    parser.add_argument("--logo", default=LOGO, help="The micro-sam2 logo.")
    parser.add_argument("--napari_logo", default=NAPARI_LOGO, help="The napari logo, by default the one of napari.")
    parser.add_argument("--python_logo", default=PYTHON_LOGO, help="The Python logo.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    fetch_python_logo(args.python_logo)
    panel_a = build_panel_a()
    save_panel(panel_a, args.output_folder, "fig1_panel_a", A4_WIDTH, A_H * A4_WIDTH / A_W)
    panel_c, width, height = build_panel_c(args.logo, args.napari_logo, args.python_logo)
    save_panel(panel_c, args.output_folder, "fig1_panel_c", width, height)
    if os.path.exists(args.figure):
        insert_panel(args.figure, panel_a, "a")
        insert_panel(args.figure, panel_c, "c")


if __name__ == "__main__":
    main()
