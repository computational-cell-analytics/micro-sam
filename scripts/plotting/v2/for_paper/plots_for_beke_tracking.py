"""Candidate studio renders of the tracked nuclei of the Beke time-lapse (Platynereis, 3-6 hpf), for a tracking figure.

The script renders four layouts, each with three colour schemes, for the old and the new defaults:

- A: the nuclei at four time points in the studio, with the lineage tree below.
- B: a space-time cube: the nucleus paths as tubes that rise with time and split at divisions, over the t0 raw.
- C: the nuclei at the last time point with trails of their last positions, with the lineage tree beside.
- D: stacked raw cards at four time points with the outlines of the nuclei, and the paths between the cards.

The colour schemes: 'tree' (one colour per lineage tree, shades for the daughters), 't0' (only the trees that start
at t0 in colour, the rest grey) and 'ratio' (the cell to nucleus volume ratio at each time point).
The renders run off screen with pyvista on a GPU node.

Panel a of the tracking figure combines A and D with the tree colours for the new defaults. The renders of the panel
run with --panel_render (GPU node), the panel is put together from them with --panel.

Usage:
    python plots_for_beke_tracking.py
    python plots_for_beke_tracking.py --versions new_defaults --layouts A B
    python plots_for_beke_tracking.py --panel_render --panel
    python plots_for_beke_tracking.py --panel_render nuclei --panel  # only the nuclei views again
"""

import os
import json
import argparse
import colorsys

import h5py
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from skimage.segmentation import find_boundaries

from plot_style import TEXT_COLOR, set_style
from plots_for_figure1 import (
    d_backdrop, d_card, d_colored_mesh, d_crop_white, d_log, d_project, d_px_per_unit, d_quad, d_screenshot, d_studio,
    d_view,
)

TRACK_DIR = "../../../user_study/v2/huisken_tracks_h5"
DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/huisken_data/platynereis_timelapse_3-6hpf"
VERSIONS = ["old_defaults", "new_defaults"]
LAYOUTS = ["A", "B", "C", "D"]
SCHEMES = ["tree", "t0", "ratio"]
# z spacing over the xy spacing (2.5 / 0.406 um, xy downsampled 2x)
Z_SCALE = 2.5 / 0.812
START_HPF, MINUTES_PER_TIMEPOINT = 3.0, 2.0
TIMEPOINTS = (0, 30, 60, 88)
# the scene height of one time point in the space-time views
T_HEIGHT = 3.5
TRAIL = 20
GREY = np.array([0.78, 0.78, 0.76])
RATIO_CMAP = LinearSegmentedColormap.from_list(
    "ratio", ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)
RENDER_SIZE = (1200, 1200)
VIEW = d_view(35, -60)[0]
XY_UM = 0.812
PANEL_VERSION = "new_defaults"
PANEL_FOLDER = "figures/beke_tracking"
MM = 1 / 25.4


def to_hpf(t):
    return START_HPF + np.asarray(t) * MINUTES_PER_TIMEPOINT / 60


def load(version):
    """The track file (downsampled 2x in y and x), the lineage as a table and the ratio table."""
    path = os.path.join(TRACK_DIR, f"tracks_{version}.h5")
    with h5py.File(path, "r") as f:
        lineage = pd.DataFrame(f["lineage"][:], columns=["track_id", "t_start", "t_end", "parent_id"])
        tracks = pd.DataFrame(f["tracks"][:], columns=["track_id", "t", "z", "y", "x"])
    tracks[["track_id", "t"]] = tracks[["track_id", "t"]].astype(int)
    ratio = pd.read_csv(os.path.join(DATA_DIR, version, "volume_ratio_cells.tsv"), sep="\t", keep_default_na=False)
    ratio = ratio[ratio["outlier"] == ""]
    return path, lineage.set_index("track_id"), tracks, ratio


def tree_info(lineage):
    """The root and the generation of each track, and the children of each track."""
    children = {i: [] for i in lineage.index}
    for i, parent in lineage["parent_id"].items():
        if parent > 0:
            children[parent].append(i)
    roots, depth = {}, {}
    for i in lineage.index:
        j, d = i, 0
        while lineage.at[j, "parent_id"] > 0:
            j, d = lineage.at[j, "parent_id"], d + 1
        roots[i], depth[i] = j, d
    return roots, depth, children


def track_colors(lineage, scheme):
    """One rgb colour per track: the hue of its tree, shades of it for the daughters."""
    roots, depth, children = tree_info(lineage)
    root_ids = sorted(set(roots.values()), key=lambda r: (lineage.at[r, "t_start"], r))
    shade = {}
    for i in lineage.index:
        siblings = children.get(lineage.at[i, "parent_id"], [i])
        shade[i] = siblings.index(i) if i in siblings else 0
    colors = {}
    for k, root in enumerate(root_ids):
        if scheme == "t0" and lineage.at[root, "t_start"] > 0:
            continue
        hue = (k * 0.6180339887) % 1.0
        colors[root] = hue
    out = {}
    for i in lineage.index:
        if roots[i] not in colors:
            out[i] = GREY
            continue
        d = depth[i]
        sat = 0.75 - 0.08 * min(d, 4)
        val = 0.92 - 0.12 * shade[i] if d > 0 else 0.92
        hue = (colors[roots[i]] + 0.025 * d * (1 if shade[i] == 0 else -1)) % 1.0
        out[i] = np.array(colorsys.hsv_to_rgb(hue, sat, val))
    return out


def color_fn(lineage, ratio, scheme):
    """A function of the track id and the time point that returns the rgb colour."""
    if scheme != "ratio":
        colors = track_colors(lineage, scheme)
        return lambda i, t: colors[i]
    values = dict(zip(zip(ratio["nucleus_id"], ratio["t"]), ratio["ratio"]))
    lo, hi = np.percentile(ratio["ratio"], [2, 98])
    norm = LogNorm(lo, hi, clip=True)

    def fn(i, t):
        value = values.get((i, t))
        return GREY if value is None else np.array(RATIO_CMAP(norm(value))[:3])
    fn.norm = norm
    return fn


def paths(lineage, tracks, smooth=2.0):
    """The smoothed centroid path of each track, with the last point of the parent first."""
    out = {}
    for i, group in tracks.sort_values("t").groupby("track_id"):
        zyx = group[["z", "y", "x"]].to_numpy(float)
        if len(zyx) > 2:
            zyx = ndimage.gaussian_filter1d(zyx, smooth, axis=0, mode="nearest")
        out[i] = (group["t"].to_numpy(), zyx)
    for i in list(out):
        parent = lineage.at[i, "parent_id"]
        if parent > 0 and parent in out:
            t, zyx = out[i]
            pt, pzyx = out[parent]
            out[i] = (np.r_[pt[-1], t], np.r_[pzyx[-1:], zyx])
    return out


def tube(points, colors, radius):
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-3]
    points, colors = points[keep], np.asarray(colors)[keep]
    if len(points) < 2:
        return None
    mesh = pv.lines_from_points(points).tube(radius=radius, n_sides=20, capping=True)
    # the tube filter drops the colours, so each tube point takes the colour of the nearest path point
    nearest = cKDTree(points).query(mesh.points)[1]
    mesh.point_data["rgb"] = (colors[nearest] * 255).astype("uint8")
    return mesh


def add_tube(plotter, mesh):
    if mesh is None:
        return
    plotter.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.3, specular_power=30,
                     ambient=0.35, diffuse=0.75, show_scalar_bar=False)


def nuclei_mesh(segmentation, t, fn):
    labels = segmentation[t].astype("uint32")
    ids = np.unique(labels)
    remap = np.zeros(int(ids.max()) + 1, "uint32")
    remap[ids] = np.arange(len(ids))
    colors = np.array([GREY] + [fn(int(i), t) for i in ids[1:]])
    return d_colored_mesh(remap[labels], colors, Z_SCALE)


def volume_bounds(shape):
    return (0, shape[3], 0, shape[2], 0, shape[1] * Z_SCALE)


def studio_shot(add, bounds, distance=900, zoom=1.6):
    plotter = d_studio(RENDER_SIZE, bounds, VIEW, distance, zoom)
    add(plotter)
    d_backdrop(plotter, bounds, VIEW, 20, 260, 80)
    image = d_screenshot(plotter, RENDER_SIZE)
    plotter.close()
    return image


def crop_white(image, margin=20):
    mask = (image < 245).any(axis=-1)
    rows, cols = np.nonzero(mask)
    r0, r1 = max(rows.min() - margin, 0), rows.max() + margin
    c0, c1 = max(cols.min() - margin, 0), cols.max() + margin
    return image[r0:r1, c0:c1]


def raw_mip(raw, t):
    mip = raw[t].max(axis=0).astype("float32")
    lo, hi = np.percentile(mip, [1, 99.8])
    return np.clip((mip - lo) / max(hi - lo, 1), 0, 1)


def draw_tree(ax, lineage, fn, marks=()):
    """The lineage trees over time: one row per leaf, a vertical line at each division."""
    roots, _, children = tree_info(lineage)
    root_ids = sorted({r for r in roots.values()}, key=lambda r: (lineage.at[r, "t_start"], r))
    y, count = {}, [0]

    def place(i):
        if not children[i]:
            y[i] = count[0]
            count[0] += 1
        else:
            for c in children[i]:
                place(c)
            y[i] = np.mean([y[c] for c in children[i]])
    for r in root_ids:
        place(r)
        count[0] += 0.8
    for i, row in lineage.iterrows():
        ts = np.arange(row["t_start"], row["t_end"] + 1)
        for a, b in zip(ts[:-1], ts[1:]):
            ax.plot(to_hpf([a, b]), [y[i], y[i]], color=fn(i, a), lw=1.2, solid_capstyle="butt")
        if children[i]:
            ys = [y[c] for c in children[i]]
            ax.plot(to_hpf([row["t_end"]] * 2), [min(ys), max(ys)], color=fn(i, row["t_end"]), lw=0.8)
            for c in children[i]:
                ax.plot(to_hpf([row["t_end"], lineage.at[c, "t_start"]]), [y[c], y[c]], color=fn(c, row["t_end"]),
                        lw=1.2)
    for t in marks:
        ax.axvline(to_hpf(t), color="#52514e", lw=0.6, ls=(0, (2, 2)))
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("time (hpf)")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def save_with_tree(out, renders, labels, lineage, fn, marks, beside=False):
    """The renders in a row, the lineage tree below (or beside) them."""
    n = len(renders)
    if beside:
        fig = plt.figure(figsize=(12, 6))
        axes = [fig.add_axes([0.0, 0.0, 0.55, 1.0])]
        tree_ax = fig.add_axes([0.6, 0.1, 0.38, 0.85])
    else:
        fig = plt.figure(figsize=(3.2 * n, 6.4))
        axes = [fig.add_axes([k / n, 0.42, 1 / n, 0.58]) for k in range(n)]
        tree_ax = fig.add_axes([0.04, 0.08, 0.92, 0.32])
    for ax, image, label in zip(axes, renders, labels):
        ax.imshow(image)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    draw_tree(tree_ax, lineage, fn, marks)
    add_ratio_bar(fig, fn)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def add_ratio_bar(fig, fn):
    if not hasattr(fn, "norm"):
        return
    cax = fig.add_axes([0.9, 0.95, 0.08, 0.015])
    bar = fig.colorbar(plt.cm.ScalarMappable(fn.norm, RATIO_CMAP), cax=cax, orientation="horizontal")
    bar.set_label("cell / nucleus volume", fontsize=7)
    bar.ax.tick_params(labelsize=6)


def save_single(out, image, fn):
    fig = plt.figure(figsize=(6, 6 * image.shape[0] / image.shape[1]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    add_ratio_bar(fig, fn)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def layout_a(out, segmentation, lineage, fn):
    bounds = volume_bounds(segmentation.shape)
    renders = []
    for t in TIMEPOINTS:
        mesh = nuclei_mesh(segmentation, t, fn)
        renders.append(crop_white(studio_shot(lambda plotter: add_mesh_rgb(plotter, mesh), bounds)))
    labels = [f"{to_hpf(t):.1f} hpf" for t in TIMEPOINTS]
    save_with_tree(out, renders, labels, lineage, fn, TIMEPOINTS)


def add_mesh_rgb(plotter, mesh):
    plotter.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.2, specular_power=20,
                     ambient=0.35, diffuse=0.75, show_scalar_bar=False)


def layout_b(out, raw, lineage, tracks, fn):
    """Space-time cube: x, y on the floor, time going up."""
    tubes = []
    for i, (t, zyx) in paths(lineage, tracks).items():
        points = np.c_[zyx[:, 2], -zyx[:, 1], 6 + t * T_HEIGHT]
        tubes.append(tube(points, [fn(i, tt) for tt in t], 1.6))
        if lineage.at[i, "t_end"] == raw.shape[0] - 1:
            tubes.append(pv.Sphere(3.2, points[-1]).compute_normals())
            tubes[-1].point_data["rgb"] = np.tile((fn(i, t[-1]) * 255).astype("uint8"), (tubes[-1].n_points, 1))
    mip = raw_mip(raw, 0)
    card = (np.stack([mip] * 3, -1) * 255).astype("uint8")
    height = 6 + (raw.shape[0] - 1) * T_HEIGHT
    bounds = (0, raw.shape[3], -raw.shape[2], 0, 0, height)

    def add(plotter):
        d_card(plotter, card, (0, 0, 0), 1.0, 3)
        for mesh in tubes:
            add_tube(plotter, mesh)
    view = d_view(18, -60)[0]
    plotter = d_studio(RENDER_SIZE, bounds, view, 900, 1.0)
    add(plotter)
    d_backdrop(plotter, bounds, view, 10, 300, 40)
    image = crop_white(d_screenshot(plotter, RENDER_SIZE))
    plotter.close()
    save_single(out, image, fn)


def layout_c(out, segmentation, lineage, tracks, fn):
    """The last time point with trails of the last TRAIL time points."""
    t_last = segmentation.shape[0] - 1
    mesh = nuclei_mesh(segmentation, t_last, fn)
    trails = []
    for i, (t, zyx) in paths(lineage, tracks).items():
        keep = t >= t_last - TRAIL
        if keep.sum() < 2:
            continue
        points = np.c_[zyx[keep, 2], zyx[keep, 1], zyx[keep, 0] * Z_SCALE]
        trails.append(tube(points, [fn(i, tt) * 0.8 for tt in t[keep]], 1.0))
    bounds = volume_bounds(segmentation.shape)

    def add(plotter):
        # translucent meshes vanish in the shadow pass, so the nuclei are opaque
        add_mesh_rgb(plotter, mesh)
        for trail in trails:
            add_tube(plotter, trail)
    image = crop_white(studio_shot(add, bounds))
    save_with_tree(out, [image], [f"{to_hpf(t_last):.1f} hpf, trails of {TRAIL * 2} min"], lineage, fn, (t_last,),
                   beside=True)


def layout_d(out, raw, segmentation, lineage, tracks, fn):
    """Stacked raw cards with the nucleus outlines, the paths running between them."""
    gap = 30 * T_HEIGHT
    cards = []
    for k, t in enumerate(TIMEPOINTS):
        mip = raw_mip(raw, t)
        labels = segmentation[t].max(axis=0)
        image = np.stack([mip] * 3, -1) * 0.85
        edge = find_boundaries(labels, mode="inner")
        for i in np.unique(labels[edge]):
            image[edge & (labels == i)] = fn(int(i), t)
        cards.append(((image * 255).astype("uint8"), k * gap))
    tubes = []
    for i, (t, zyx) in paths(lineage, tracks).items():
        h = np.interp(t, TIMEPOINTS, np.arange(len(TIMEPOINTS)) * gap)
        points = np.c_[zyx[:, 2], -zyx[:, 1], h + 2]
        tubes.append(tube(points, [fn(i, tt) for tt in t], 1.0))
    bounds = (0, raw.shape[3], -raw.shape[2], 0, 0, (len(TIMEPOINTS) - 1) * gap + 4)

    def add(plotter):
        for image, z in cards:
            d_card(plotter, image, (0, 0, z), 1.0, 2)
        for mesh in tubes:
            add_tube(plotter, mesh)
    view = d_view(22, -60)[0]
    plotter = d_studio(RENDER_SIZE, bounds, view, 900, 1.0)
    add(plotter)
    d_backdrop(plotter, bounds, view, 10, 300, 40)
    image = crop_white(d_screenshot(plotter, RENDER_SIZE))
    plotter.close()
    save_single(out, image, fn)


def contact_sheet(folder, version):
    """All candidates of one version on one page: a row per layout, a column per colour scheme."""
    fig, axes = plt.subplots(len(LAYOUTS), len(SCHEMES), figsize=(5 * len(SCHEMES), 4 * len(LAYOUTS)))
    for r, layout in enumerate(LAYOUTS):
        for c, scheme in enumerate(SCHEMES):
            ax = axes[r, c]
            ax.axis("off")
            path = os.path.join(folder, f"{layout}_{scheme}.png")
            if os.path.exists(path):
                ax.imshow(Image.open(path))
            ax.set_title(f"{layout} / {scheme}", fontsize=10)
    fig.suptitle(version)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"overview_{version}.png"), dpi=150)
    plt.close(fig)


def render_version(version, layouts, schemes, output_folder):
    path, lineage, tracks, ratio = load(version)
    folder = os.path.join(output_folder, version)
    os.makedirs(folder, exist_ok=True)
    with h5py.File(path, "r") as f:
        segmentation = f["segmentation"][:]
        raw = f["raw"][:] if {"B", "D"} & set(layouts) else None
    d_log(version, "loaded", segmentation.shape)
    for scheme in schemes:
        fn = color_fn(lineage, ratio, scheme)
        for layout in layouts:
            out = os.path.join(folder, f"{layout}_{scheme}.png")
            if layout == "A":
                layout_a(out, segmentation, lineage, fn)
            elif layout == "B":
                layout_b(out, raw, lineage, tracks, fn)
            elif layout == "C":
                layout_c(out, segmentation, lineage, tracks, fn)
            else:
                layout_d(out, raw, segmentation, lineage, tracks, fn)
            d_log(version, layout, scheme)
    contact_sheet(folder, version)


def crop_to_embryo(raw, segmentation, tracks, margin=10):
    """Crop y and x to the nuclei of all time points plus a margin, and shift the tracks."""
    rows, cols = np.nonzero((segmentation > 0).any(axis=(0, 1)))
    y0, x0 = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
    y1, x1 = rows.max() + margin + 1, cols.max() + margin + 1
    tracks = tracks.copy()
    tracks["y"] -= y0
    tracks["x"] -= x0
    return raw[:, :, y0:y1, x0:x1], segmentation[:, :, y0:y1, x0:x1], tracks


def panel_render_stack(folder, raw, segmentation, lineage, tracks, fn):
    """The D stack at print resolution; saves the render and the image positions of the card corners."""
    gap = 30 * T_HEIGHT
    h, w = raw.shape[2:]
    cards = []
    for k, t in enumerate(TIMEPOINTS):
        mip = raw_mip(raw, t)
        labels = segmentation[t].max(axis=0)
        image = np.stack([mip] * 3, -1) * 0.85
        edge = find_boundaries(labels, mode="inner")
        for i in np.unique(labels[edge]):
            image[edge & (labels == i)] = fn(int(i), t)
        cards.append(((image * 255).astype("uint8"), k * gap))
    tubes = [tube(np.c_[zyx[:, 2], -zyx[:, 1], np.interp(t, TIMEPOINTS, np.arange(len(TIMEPOINTS)) * gap) + 2],
                  [fn(i, tt) for tt in t], 1.0) for i, (t, zyx) in paths(lineage, tracks).items()]
    bounds = (0, w, -h, 0, 0, (len(TIMEPOINTS) - 1) * gap + 4)
    size = (1800, 2200)
    view = d_view(22, -60)[0]
    plotter = d_studio(size, bounds, view, 900, 1.15)
    for image, z in cards:
        d_card(plotter, image, (0, 0, z), 1.0, 2)
    for mesh in tubes:
        add_tube(plotter, mesh)
    d_backdrop(plotter, bounds, view, 10, 300, 40)
    image = d_screenshot(plotter, size)
    corners = [d_project(plotter, [(0, 0, z + 2), (w, 0, z + 2), (w, -h, z + 2), (0, -h, z + 2)]) for _, z in cards]
    bar = d_project(plotter, [(8, -h, 0), (8 + 50 / XY_UM, -h, 0)])
    plotter.close()
    image, offset = d_crop_white(image, 10)
    Image.fromarray(image).save(f"{folder}/stack.png")
    corners = [[(x - offset[0], y - offset[1]) for x, y in c] for c in corners + [bar]]
    return {"card_corners": corners[:-1], "stack_bar": corners[-1]}


def panel_render_nuclei(folder, segmentation, fn):
    """The nuclei at the time points of the cards, with the same camera; y is flipped to match the cards."""
    _, _, h, w = segmentation.shape
    bounds = (0, w, 0, h, 0, segmentation.shape[1] * Z_SCALE)
    view = d_view(22, -60)[0]
    images, ppu = [], None
    for t in TIMEPOINTS:
        labels = segmentation[t][:, ::-1].astype("uint32")
        ids = np.unique(labels)
        remap = np.zeros(int(ids.max()) + 1, "uint32")
        remap[ids] = np.arange(len(ids))
        colors = np.array([GREY] + [fn(int(i), t) for i in ids[1:]])
        mesh = d_colored_mesh(remap[labels], colors, Z_SCALE)
        plotter = d_studio((1400, 1400), bounds, view, 700, 1.05)
        add_mesh_rgb(plotter, mesh)
        add_floor(plotter, bounds, 10, 600)
        images.append(d_screenshot(plotter, (1400, 1400)))
        ppu = d_px_per_unit(plotter, np.array([w / 2, h / 2, bounds[5] / 2]))
        plotter.close()
        d_log("panel nuclei", t)
    # the empty floor with the same camera, to make the floor white without losing the shadows
    plotter = d_studio((1400, 1400), bounds, view, 700, 1.05)
    add_floor(plotter, bounds, 10, 600)
    Image.fromarray(d_screenshot(plotter, (1400, 1400))).save(f"{folder}/floor.png")
    plotter.close()
    for t, image in zip(TIMEPOINTS, images):
        Image.fromarray(image).save(f"{folder}/nuclei_t{t:03d}.png")
    return {"nuclei_px_per_um": ppu / XY_UM}


def add_floor(plotter, bounds, margin, reach):
    """A floor under the scene for the shadows, without walls."""
    x0, x1, y0, y1, z0, _ = bounds
    z = z0 - margin
    floor = d_quad([(x0 - reach, y0 - reach, z), (x1 + reach, y0 - reach, z), (x1 + reach, y1 + reach, z),
                    (x0 - reach, y1 + reach, z)])
    plotter.add_mesh(floor, color=(0.95, 0.95, 0.95), ambient=0.45, diffuse=0.6, specular=0.0)


def panel_render(parts):
    os.makedirs(PANEL_FOLDER, exist_ok=True)
    layout_path = f"{PANEL_FOLDER}/layout.json"
    layout = {}
    if os.path.exists(layout_path):
        with open(layout_path) as f:
            layout = json.load(f)
    path, lineage, tracks, ratio = load(PANEL_VERSION)
    with h5py.File(path, "r") as f:
        raw, segmentation = f["raw"][:], f["segmentation"][:]
    raw, segmentation, tracks = crop_to_embryo(raw, segmentation, tracks)
    fn = color_fn(lineage, ratio, "tree")
    if "stack" in parts:
        layout.update(panel_render_stack(PANEL_FOLDER, raw, segmentation, lineage, tracks, fn))
        d_log("panel stack")
    if "nuclei" in parts:
        layout.update(panel_render_nuclei(PANEL_FOLDER, segmentation, fn))
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=1)


def mm_axes(fig, size, box):
    """Axes at a box (left, bottom, width, height) in mm of a figure of the size (width, height) in mm."""
    (width, height), (left, bottom, w, h) = size, box
    return fig.add_axes([left / width, bottom / height, w / width, h / height])


def crop_box(image, keep, points=(), margins=(30, 30, 30, 30)):
    """The box (c0, r0, c1, r1) around the pixels to keep and the points, with margins (left, top, right, bottom)."""
    rows, cols = np.nonzero(keep)
    xs = np.r_[cols, np.asarray(points, float).reshape(-1, 2)[:, 0]]
    ys = np.r_[rows, np.asarray(points, float).reshape(-1, 2)[:, 1]]
    c0, c1, r0, r1 = xs.min(), xs.max(), ys.min(), ys.max()
    left, top, right, bottom = margins
    h, w = image.shape[:2]
    return int(max(c0 - left, 0)), int(max(r0 - top, 0)), int(min(c1 + right, w)), int(min(r1 + bottom, h))


def white_floor(image, floor):
    """Divide the grey pixels (floor and shadows) by the render of the empty floor: the floor becomes white, the
    shadows stay."""
    rgb, floor = image.astype(float), np.maximum(floor.astype(float), 1)
    grey = np.ptp(rgb, -1) < 10
    lifted = np.clip(rgb / floor * 255, 0, 255)
    return np.where(grey[..., None], lifted, rgb).astype("uint8")


def scale_bar(ax, p0, p1, label):
    """A dark scale bar between two image points, with its label to the right."""
    (x0, y0), (x1, y1) = p0, p1
    ax.plot([x0, x1], [y0, y1], color=TEXT_COLOR, lw=1.2, solid_capstyle="butt", clip_on=False)
    ax.text(x1 + 12, y1, label, ha="left", va="center", fontsize=6, color=TEXT_COLOR, clip_on=False)


def panel():
    """Panel a: the stack of raw cards with the nucleus paths, and beside each card the nuclei of its time point."""
    set_style()
    with open(f"{PANEL_FOLDER}/layout.json") as f:
        layout = json.load(f)
    stack = np.asarray(Image.open(f"{PANEL_FOLDER}/stack.png").convert("RGB"))
    corners = np.array(layout["card_corners"])
    bar = np.array(layout["stack_bar"])
    # the bar runs along the front edge of the lowest card; it moves down into the white below the card
    bar_shift = np.array([0, 70.0])
    c0, r0, c1, r1 = crop_box(stack, np.zeros(stack.shape[:2], bool), np.r_[corners.reshape(-1, 2), bar + bar_shift],
                              margins=(20, 20, 20, 40))
    stack = stack[r0:r1, c0:c1]
    corners = corners - [c0, r0]
    bar = bar + bar_shift - [c0, r0]

    floor = np.asarray(Image.open(f"{PANEL_FOLDER}/floor.png").convert("RGB"))
    nuclei = [white_floor(np.asarray(Image.open(f"{PANEL_FOLDER}/nuclei_t{t:03d}.png").convert("RGB")), floor)
              for t in TIMEPOINTS]
    # one crop for all time points, so that they keep the same scale
    # the crop keeps the nuclei and their shadows
    keep = np.any([image.min(-1) < 235 for image in nuclei], axis=0)
    n0, m0, n1, m1 = crop_box(nuclei[0], keep, margins=(20, 20, 20, 20))
    nuclei = [image[m0:m1, n0:n1] for image in nuclei]
    n_ppu = layout["nuclei_px_per_um"]

    width, height = 183, 150
    label_h, view_gap = 3.5, 1.5
    stack_h = height - 8
    view_h = (stack_h - 4 * label_h - 3 * view_gap) / 4
    view_w = view_h * nuclei[0].shape[1] / nuclei[0].shape[0]
    stack_w = stack_h * stack.shape[1] / stack.shape[0]
    gap = 12
    total = stack_w + gap + view_w
    if total > width - 4:
        scale = (width - 4 - gap - view_w) / stack_w
        stack_w, stack_h = stack_w * scale, stack_h * scale
    left = (width - (stack_w + gap + view_w)) / 2
    fig = plt.figure(figsize=(width * MM, height * MM))
    stack_ax = mm_axes(fig, (width, height), (left, (height - stack_h) / 2, stack_w, stack_h))
    stack_ax.imshow(stack)
    stack_ax.axis("off")
    scale_bar(stack_ax, bar[0], bar[1], "50 µm")

    view_left = left + stack_w + gap
    for k, (t, image) in enumerate(zip(TIMEPOINTS, nuclei)):
        # the latest time point on top, as in the stack
        bottom = 4 + k * (view_h + label_h + view_gap)
        ax = mm_axes(fig, (width, height), (view_left, bottom, view_w, view_h))
        ax.imshow(image)
        ax.axis("off")
        ax.text(0, 1.02, f"{to_hpf(t):.1f} hpf", transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
                color=TEXT_COLOR, fontweight="bold")
        if k == 0:
            y = image.shape[0] - 35
            scale_bar(ax, (25, y), (25 + 50 * n_ppu, y), "50 µm")
        right = corners[k][np.argmax(corners[k][:, 0])]
        # the connector runs from the right corner of the card to the left edge of its view, touching both
        fig.add_artist(ConnectionPatch(right, (0, 0.5), "data", "axes fraction", axesA=stack_ax, axesB=ax,
                                       color="#52514e", lw=1.0, ls=(0, (3, 1.5)), capstyle="butt", zorder=5))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{PANEL_FOLDER}/beke_tracking_panel_a.{ext}", dpi=600)
    plt.close(fig)
    print(f"Saved panel a in {PANEL_FOLDER}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", nargs="+", default=VERSIONS, choices=VERSIONS)
    parser.add_argument("--layouts", nargs="+", default=LAYOUTS, choices=LAYOUTS)
    parser.add_argument("--schemes", nargs="+", default=SCHEMES, choices=SCHEMES)
    parser.add_argument("--output_folder", default="figures/beke_candidates")
    parser.add_argument("--panel_render", nargs="*", choices=["stack", "nuclei"],
                        help="Render the views of panel a (GPU node): the stack, the nuclei, or both if empty.")
    parser.add_argument("--panel", action="store_true", help="Put panel a together from its renders.")
    args = parser.parse_args()
    if args.panel_render is not None or args.panel:
        if args.panel_render is not None:
            panel_render(args.panel_render or ["stack", "nuclei"])
        if args.panel:
            panel()
        return
    for version in args.versions:
        render_version(version, args.layouts, args.schemes, args.output_folder)


if __name__ == "__main__":
    main()
