"""The MCF7 spheroid panel, in the layout of the Platynereis panel: a) the nuclei in the studio with the outline of
the middle z plane, b) a stack of two cards: the DAPI raw of the plane at the back and the nuclei cut by the plane
on a clear card in front. Usage: spheroid_panel.py RENDER_DIR OUT_PNG (also saves a pdf)."""
import json
import os
import sys

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, Polygon
from matplotlib.path import Path
from PIL import Image, ImageDraw

FONT_DIR = os.path.expanduser("~/.local/share/fonts/liberation")
for name in os.listdir(FONT_DIR):
    if name.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONT_DIR, name))
matplotlib.rcParams["font.family"] = "Liberation Sans"
matplotlib.rcParams["font.size"] = 7
LINE = "#1a1a1a"
D, OUT = sys.argv[1:3]
layout = json.load(open(f"{D}/layout.json"))
SHIFT = 0.42  # of the top edge of the card


def load(path, mode="RGB"):
    return np.asarray(Image.open(path).convert(mode))


def crop_white(image, margin):
    """Crop a white background image to its content plus a margin; also return the offset (column, row)."""
    rows, cols = np.nonzero((image < 245).any(-1))
    r0, c0 = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
    return image[r0:rows.max() + margin, c0:cols.max() + margin], np.array([c0, r0])


def card_corners(image):
    """The corners of the dark raw card: top left, top right, bottom right, bottom left."""
    rows, cols = np.nonzero((image < 250).any(-1))
    s, d = cols + rows, cols - rows
    return np.array([(cols[i], rows[i]) for i in (np.argmin(s), np.argmax(d), np.argmax(s), np.argmin(d))], float)


def layered(raw, seg, shift):
    """The raw card at the back, its edge drawn, and the nuclei (with their own alpha) on a clear card in front,
    moved by the shift vector. Returns the image, the card corners and the origins of both layers."""
    h, w = raw.shape[:2]
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    back = np.array([0, max(0, -dy)])
    front = back + [dx, dy]
    out = np.full((h + abs(dy), w + dx, 3), 255, "uint8")
    out[back[1]:back[1] + h, back[0]:back[0] + w] = raw
    card = card_corners(raw)
    edge = Image.fromarray(out)
    ImageDraw.Draw(edge).polygon([tuple(p) for p in card + back], outline=(138, 138, 138), width=3)
    out = np.asarray(edge).astype(float)
    alpha = seg[..., 3:4].astype(float) / 255
    region = out[front[1]:front[1] + h, front[0]:front[0] + w]
    out[front[1]:front[1] + h, front[0]:front[0] + w] = region * (1 - alpha) + seg[..., :3] * alpha
    return np.clip(out, 0, 255).astype("uint8"), card, back, front


# b: more contrast in the raw (stretch to its 98th percentile on the card, slight gamma), then the stack
raw = load(f"{D}/section_raw.png")
grey = raw[..., 0].astype(float)
on_card = Path(card_corners(raw)).contains_points(
    np.c_[np.indices(grey.shape)[1].ravel(), np.indices(grey.shape)[0].ravel()]).reshape(grey.shape)
hi = np.percentile(grey[on_card & (grey > 8)], 98)
stretched = (np.clip(grey / hi, 0, 1) ** 0.8 * 255)[..., None].repeat(3, -1)
raw = np.where(on_card[..., None], stretched, raw).astype("uint8")
corners = card_corners(raw)
section, card, back, front = layered(raw, load(f"{D}/section_rgba.png", "RGBA"), SHIFT * (corners[1] - corners[0]))
section, offset = crop_white(section, 22)
glass = card + front - offset
back_card = card + back - offset
section = np.pad(section, ((0, int(0.11 * section.shape[0])), (0, 0), (0, 0)), constant_values=255)

# a: the studio cropped to the spheroid and the plane outline, with room for the floor and the shadows
studio = load(f"{D}/studio.png")
outline = np.array(layout["studio"]["slab_outline"])
rgb = studio.astype(int)
colourful = (rgb.max(-1) - rgb.min(-1)) > 40
rows, cols = np.flatnonzero(colourful.any(1)), np.flatnonzero(colourful.any(0))
r0, r1 = min(rows.min(), outline[:, 1].min()), max(rows.max(), outline[:, 1].max())
c0, c1 = min(cols.min(), outline[:, 0].min()), max(cols.max(), outline[:, 0].max())
r0, r1 = int(max(r0 - 0.06 * (r1 - r0), 0)), int(min(r1 + 0.10 * (r1 - r0), studio.shape[0]))
c0, c1 = int(max(c0 - 0.05 * (c1 - c0), 0)), int(min(c1 + 0.05 * (c1 - c0), studio.shape[1]))
studio = studio[r0:r1, c0:c1]
outline = outline - [c0, r0]

H = 62.0  # mm
TOP, GAP = 6.0, 6.0
row_h = H - TOP - 2
wa = row_h * studio.shape[1] / studio.shape[0]
wb = row_h * section.shape[1] / section.shape[0]
xa, xb = 3.0, 3.0 + wa + GAP
W = xb + wb + 1.5
print("width", round(W, 1), "mm")
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=300)


def place(image, x, y, w, h):
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.imshow(image, interpolation="lanczos")
    ax.axis("off")
    return ax


def scale_bar(ax, image, um, px_per_voxel, inset=0.05):
    h, w = image.shape[:2]
    px = um / layout["xy_um"] * px_per_voxel
    x0, y = inset * w, (1 - inset) * h
    ax.plot([x0, x0 + px], [y, y], color=LINE, lw=1.6, solid_capstyle="butt")
    ax.text(x0 + px / 2, y - 0.012 * h, f"{um} µm", ha="center", va="bottom", color=LINE, fontsize=6.5)


def title(ax, text):
    ax.text(0.5, 1.0 + 1.6 / (ax.get_position().height * H), text, transform=ax.transAxes, ha="center",
            va="baseline", fontsize=7, color=LINE)


def letter(x, y, text):
    fig.text(x / W, 1 - y / H, text, fontsize=8, fontweight="bold", va="top", ha="left", color=LINE)


ax_a = place(studio, xa, TOP, wa, row_h)
ax_b = place(section, xb, TOP, wb, row_h)
letter(0.5, 1, "a")
letter(xb - 4, 1, "b")
title(ax_a, f"{layout['nuclei']} nuclei")
title(ax_b, f"{layout['slab_nuclei']} nuclei in one z plane")

# the plane in a, and lines from its right corners to the left corners of the raw card in b, so that no line
# crosses a (corners: top left, top right, bottom right, bottom left in both)
ax_a.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=LINE, lw=0.8, joinstyle="miter"))
for i, j in ((1, 0), (2, 3)):
    fig.add_artist(ConnectionPatch(xyA=outline[i], coordsA=ax_a.transData, xyB=back_card[j],
                                   coordsB=ax_b.transData, lw=0.5, color=LINE))
ax_b.add_patch(Polygon(glass, closed=True, fill=False, edgecolor="#8a8a8a", lw=0.5, joinstyle="miter"))
scale_bar(ax_a, studio, 50, layout["studio"]["px_per_voxel"])
scale_bar(ax_b, section, 50, layout["section"]["px_per_voxel"], inset=0.035)
fig.savefig(OUT, dpi=300)
fig.savefig(os.path.splitext(OUT)[0] + ".pdf")
print("saved", OUT)
