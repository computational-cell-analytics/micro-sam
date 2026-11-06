"""Mock-up of the Platynereis user study panel, in the layout of the reference figure: a) the nuclei in the studio
with the section plane, b) the longitudinal section, c, d) the anterior and posterior close-ups, joined by thin
lines. Usage: platy_section_mockup.py STUDIO_DIR SECTION_DIR OUT_PNG."""
import json
import os
import sys

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, Polygon
from PIL import Image

FONT_DIR = os.path.expanduser("~/.local/share/fonts/liberation")
for name in os.listdir(FONT_DIR):
    if name.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONT_DIR, name))
matplotlib.rcParams["font.family"] = "Liberation Sans"
matplotlib.rcParams["font.size"] = 7
LINE = "#1a1a1a"
STUDIO, SECTION, OUT = sys.argv[1:4]
layout = json.load(open(f"{SECTION}/layout.json"))
um_pre = layout["voxel_um"] / layout["expansion"]  # pre-expansion micrometers per s1 voxel


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def crop_white(image, margin):
    """Crop a white background render to its content plus a margin; also return the offset."""
    rows, cols = np.nonzero((image < 245).any(-1))
    r0, c0 = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
    return image[r0:rows.max() + margin, c0:cols.max() + margin], np.array([c0, r0])


def pad_to(image, aspect):
    """Pad a white background image to the given width / height ratio, centred."""
    h, w = image.shape[:2]
    if w / h < aspect:
        extra = int(round(h * aspect - w))
        return np.pad(image, ((0, 0), (extra // 2, extra - extra // 2), (0, 0)), constant_values=255), extra // 2, 0
    extra = int(round(w / aspect - h))
    return np.pad(image, ((extra // 2, extra - extra // 2), (0, 0), (0, 0)), constant_values=255), 0, extra // 2


seg = load(f"{STUDIO}/seg_studio.png")
def card_corners(image):
    """The corners of the tilted dark raw card: top left, top right, bottom right, bottom left."""
    rows, cols = np.nonzero((image < 250).any(-1))
    s, d = cols + rows, cols - rows
    return np.array([(cols[i], rows[i]) for i in (np.argmin(s), np.argmax(d), np.argmax(s), np.argmin(d))], float)


def layered(raw, seg, shift):
    """A stack of two cards of the same tilted shape: the raw at the back, the nuclei on a clear card in front,
    moved by the shift vector (along the top edge of the card, so that both follow the same perspective). The front
    card is transparent, so the raw shows through between the nuclei. Returns the image and the origins of the back and front layers."""
    from matplotlib.path import Path
    from scipy import ndimage
    h, w = raw.shape[:2]
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    back = np.array([0, max(0, -dy)])
    front = back + [dx, dy]
    H_, W_ = h + abs(dy), w + dx
    out = np.full((H_, W_, 3), 255.0)
    out[back[1]:back[1] + h, back[0]:back[0] + w] = raw
    card = card_corners(raw)
    # the edge of the back card, under the front card
    from PIL import ImageDraw
    edge = Image.fromarray(out.astype("uint8"))
    ImageDraw.Draw(edge).polygon([tuple(p) for p in card + back], outline=(138, 138, 138), width=3)
    out = np.asarray(edge).astype(float)
    yy, xx = np.mgrid[0:H_, 0:W_]
    points = np.c_[xx.ravel(), yy.ravel()]
    inside = Path(card + front).contains_points(points).reshape(H_, W_)
    # the front card is transparent: only its edge and the nuclei are drawn
    # the nuclei with their own alpha (rendered on a transparent background), so their edges blend into the raw
    nuclei = seg[..., :3].astype(float)
    alpha = (seg[..., 3:4].astype(float) / 255)
    region = out[front[1]:front[1] + h, front[0]:front[0] + w]
    out[front[1]:front[1] + h, front[0]:front[0] + w] = region * (1 - alpha) + nuclei * alpha
    return np.clip(out, 0, 255).astype("uint8"), card, back, front


SHIFT = 0.42  # of the top edge of the card
raw_card = load(f"{SECTION}/section_raw.png")
# more contrast in the raw slice: stretch to its 98th percentile inside the card, with a slight gamma
grey = raw_card[..., 0].astype(float)
on_card = (raw_card < 250).any(-1)
hi = np.percentile(grey[on_card & (grey > 8)], 98)
stretched = (np.clip(grey / hi, 0, 1) ** 0.8 * 255)[..., None].repeat(3, -1)
raw_card = np.where(on_card[..., None], stretched, raw_card).astype("uint8")
corners = card_corners(raw_card)
section_rgba = np.asarray(Image.open(f"{SECTION}/section_rgba.png").convert("RGBA"))
section, card, back, front = layered(raw_card, section_rgba, SHIFT * (corners[1] - corners[0]))
section, offset = crop_white(section, 22)
back_card = card + back - offset
glass = card + front - offset
offset = offset - front
# room below the nuclei for the scale bar
section = np.pad(section, ((0, int(0.06 * section.shape[0])), (0, 0), (0, 0)), constant_values=255)
boxes = {name: np.array(layout["section"][f"{name}_box"]) - offset for name in ("anterior", "posterior")}
close = {}
for name in ("anterior", "posterior"):
    image, _ = crop_white(load(f"{SECTION}/{name}.png"), 14)
    image = np.pad(image, ((0, int(0.12 * image.shape[0])), (0, 0), (0, 0)), constant_values=255)
    close[name] = pad_to(image, 1.0)[0]

# a: the studio cropped to the animal and the section plane, with a small margin
outline = np.array(layout["studio"]["slab_outline"])
rgb = seg.astype(int)
colourful = (rgb.max(-1) - rgb.min(-1)) > 40
cols = np.flatnonzero(colourful.any(0))
a0 = max(int(min(cols.min(), outline[:, 0].min())) - 10, 0)
a1 = min(int(max(cols.max(), outline[:, 0].max())) + 10, seg.shape[1])
seg = seg[:, a0:a1]
outline = outline - [a0, 0]

# the widths follow from one common height of a and b; c and d share that height
H = 112.0  # mm
TOP, GAP = 6.0, 5.5
row_h = H - TOP - 4
wa = row_h * seg.shape[1] / seg.shape[0]
wb = row_h * section.shape[1] / section.shape[0]
# c and d smaller than half the row: c at the top, d at the bottom
c_size = 38.0
xa, xb = 3.0, 3.0 + wa + GAP
xc = xb + wb + GAP
W = xc + c_size + 1.5
print("width", round(W, 1), "mm")
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=300)


def axes_mm(x, y, w, h):
    """Axes at a position in mm from the top left."""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def fit(image, x, y, w, h):
    """Axes of the image's aspect, as large as fits in a box in mm, aligned to the top left."""
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    ax = axes_mm(x, y, iw * scale, ih * scale)
    ax.imshow(image, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(LINE)
    return ax


def scale_bar(ax, image, um, px_per_voxel, inset=0.035):
    """A scale bar in the lower left corner, its label above it."""
    h, w = image.shape[:2]
    px = um / um_pre * px_per_voxel
    x0, y = inset * w, (1 - inset) * h
    ax.plot([x0, x0 + px], [y, y], color=LINE, lw=1.6, solid_capstyle="butt")
    ax.text(x0 + px / 2, y - 0.012 * h, f"{um} µm", ha="center", va="bottom", color=LINE, fontsize=6.5)


def title(ax, text):
    ax.text(0.5, 1.0, text, transform=ax.transAxes, ha="center", va="baseline", fontsize=7, color=LINE)
    ax.texts[-1].set_y(1.0 + 1.6 / (ax.get_position().height * H))


def letter(x, y, text):
    fig.text(x / W, 1 - y / H, text, fontsize=8, fontweight="bold", va="top", ha="left", color=LINE)


top = TOP
ax_a = fit(seg, xa, top, wa, row_h)
ax_a.axis("off")
ax_b = fit(section, xb, top, wb, row_h)
ax_c = fit(close["anterior"], xc, top, c_size, c_size)
ax_d = fit(close["posterior"], xc, top + row_h - c_size, c_size, c_size)
letter(0.5, 1, "a")
letter(xb - 4, 1, "b")
letter(xc - 4, 1, "c")
letter(xc - 4, top + row_h - c_size - 5, "d")
title(ax_a, f"{8625:,} nuclei")
title(ax_b, f"{layout['slab_um']:g} µm longitudinal section")
title(ax_c, "anterior")
title(ax_d, "posterior")

# the section plane in a, and lines from its right edge to the frame of b
ax_a.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=LINE, lw=0.8, joinstyle="miter"))
right = outline[np.argsort(outline[:, 0])[-2:]]
upper, lower = right[np.argmin(right[:, 1])], right[np.argmax(right[:, 1])]
bh = section.shape[0]
for point, corner in ((upper, (0, 0)), (lower, (0, bh))):
    fig.add_artist(ConnectionPatch(xyA=point, coordsA=ax_a.transData, xyB=corner, coordsB=ax_b.transData,
                                   lw=0.5, color=LINE))
for edge in (glass,):
    ax_b.add_patch(Polygon(edge, closed=True, fill=False, edgecolor="#8a8a8a", lw=0.5, joinstyle="miter"))
# the close-up boxes in b, and lines from their right edges to the frames of c and d
for name, ax in (("anterior", ax_c), ("posterior", ax_d)):
    box = boxes[name]
    # a white halo under the dark box line, so that it shows on the raw and on white
    ax_b.add_patch(Polygon(box, closed=True, fill=False, edgecolor="white", lw=1.8, joinstyle="miter"))
    ax_b.add_patch(Polygon(box, closed=True, fill=False, edgecolor=LINE, lw=0.7, joinstyle="miter"))
    right = box[np.argsort(box[:, 0])[-2:]]
    upper, lower = right[np.argmin(right[:, 1])], right[np.argmax(right[:, 1])]
    ch = close[name].shape[0]
    for point, corner in ((upper, (0, 0)), (lower, (0, ch))):
        fig.add_artist(ConnectionPatch(xyA=point, coordsA=ax_b.transData, xyB=corner, coordsB=ax.transData,
                                       lw=0.5, color=LINE))

scale_bar(ax_a, seg, 50, layout["studio"]["px_per_voxel"], inset=0.05)
scale_bar(ax_b, section, 20, layout["section"]["px_per_voxel"])
for name, ax in (("anterior", ax_c), ("posterior", ax_d)):
    # the close-ups were cropped and padded but not scaled, so the render scale holds
    scale_bar(ax, close[name], 10, layout[name]["px_per_voxel"], inset=0.06)
fig.savefig(OUT, dpi=300)
fig.savefig(os.path.splitext(OUT)[0] + ".pdf")
print("saved", OUT)
