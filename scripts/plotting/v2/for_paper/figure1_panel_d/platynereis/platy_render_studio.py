"""Upright oblique render of the expanded Platynereis nuclei: separated nuclei, distinct colours for touching
nuclei, and optionally a light grey studio backdrop with shadows. Usage: platy_render_studio.py OUT_DIR."""
import colorsys
import os
import sys
import time

import numpy as np
import pyvista as pv
import vtk
import zarr
from matplotlib.colors import ListedColormap
from PIL import Image

P = "/mnt/vast-nhr/projects/cidas/cca/data/huisken_data/exm_lsm/high_res/stitched/Expanded_Platynereis.ome.zarr"
MODE, OUT = (sys.argv[1], sys.argv[2]) if __name__ == "__main__" else ("lib", None)
# rendered at twice the size and scaled down, for smooth edges and shadows
SIZE = (3300, 6000)
t0 = time.time()


def log(*args):
    print(*args, round(time.time() - t0), flush=True)


def separate(labels):
    """Remove the voxels where two nuclei touch, so that each nucleus becomes its own surface; also return the
    pairs of touching nuclei."""
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


def near_pairs(coarse, radius=2):
    """Pairs of nuclei closer than about radius voxels on the coarse grid (2x the s1 voxel size)."""
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


def distinct_colors(n_labels, pairs, seed=1):
    """A palette of clear colours; each nucleus gets the colour that is most different (in Lab) from the colours
    of its neighbours, so that close nuclei never look alike."""
    from skimage.color import rgb2lab
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
    # the most connected nuclei first
    order = sorted(range(1, n_labels + 1), key=lambda i: (-len(neighbours[i]), rng.random()))
    for label in order:
        used = [choice[n] for n in neighbours[label] if choice[n] >= 0]
        # a random colour among those clearly different from all neighbours, else the most different one
        gap = dist[:, used].min(axis=1) if used else np.full(len(palette), np.inf)
        clear = np.flatnonzero(gap >= 30)
        choice[label] = int(rng.choice(clear)) if len(clear) else int(np.argmax(gap))
    choice[0] = 0
    return palette[choice]


def label_surfaces(labels):
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


shape = (770, 477, 1197)
if MODE == "seg":
    labels = zarr.open_group(P + "/labels/nuclei", mode="r")["s1"][:]
    n_labels = int(labels.max())
    labels, pairs = separate(labels)
    coarse = zarr.open_group(P + "/labels/nuclei", mode="r")["s2"][:]
    pairs |= near_pairs(coarse)
    del coarse
    log("separated; neighbour pairs", len(pairs))
    colors = distinct_colors(n_labels, pairs)
    cmap = ListedColormap(colors)
    mesh = label_surfaces(labels)
    del labels
    # one rgb colour per face; a colour map would be resampled to 256 colours and merge neighbouring ids
    mesh.cell_data["rgb"] = (colors[mesh.cell_data["label"]] * 255).astype("uint8")
    log("mesh", mesh.n_cells)


# the camera of the upright oblique view; volume axes x, y, z = s1 axes 2, 1, 0; the round end (low x) on top
BOUNDS = (0, shape[2], 0, shape[1], 0, shape[0])
CENTER = np.array([shape[2], shape[1], shape[0]], float) / 2
DIRECTION = np.array([-0.3, 1.0, -0.5]) / np.linalg.norm([-0.3, 1.0, -0.5])
UP = (-1, 0, 0)
ZOOM = float(os.environ.get("ZOOM", 1.55))


def backdrop(plotter):
    """A floor below the tail end and two walls behind the animal, in light grey."""
    x1, y1, z1 = shape[2], shape[1], shape[0]
    m, far, top = 60, 350, -250
    grey = (0.95, 0.95, 0.95)
    floor_x, wall_y, wall_z = x1 + m, -m, z1 + m
    # the floor spans from the walls to the front; the walls from the floor to above the round end
    front_y, front_z = y1 + far, -far

    def quad(corners):
        """A rectangle from four corners in x, y, z, finely split so that the shadows look smooth."""
        return pv.PolyData(np.array(corners, float), faces=[4, 0, 1, 2, 3]).triangulate().subdivide(4)

    floor = quad([(floor_x, wall_y, wall_z), (floor_x, front_y, wall_z), (floor_x, front_y, front_z),
                  (floor_x, wall_y, front_z)])
    back_y = quad([(top, wall_y, front_z), (top, wall_y, wall_z), (floor_x, wall_y, wall_z),
                   (floor_x, wall_y, front_z)])
    back_z = quad([(top, wall_y, wall_z), (top, front_y, wall_z), (floor_x, front_y, wall_z),
                   (floor_x, wall_y, wall_z)])
    for plane in (floor, back_y, back_z):
        plotter.add_mesh(plane, color=grey, ambient=0.45, diffuse=0.6, specular=0.0)


def lights(plotter):
    # a key light from above and in front, and a soft fill light from the camera side
    key = pv.Light(position=tuple(CENTER + np.array([-2500.0, 1200.0, -1600.0])), focal_point=tuple(CENTER),
                   intensity=0.7, positional=True, cone_angle=40, light_type="scene light")
    fill = pv.Light(position=tuple(CENTER + DIRECTION * 3000 + np.array([0.0, 600.0, 0.0])),
                    focal_point=tuple(CENTER), intensity=0.55, light_type="scene light")
    plotter.add_light(key)
    plotter.add_light(fill)


def shoot(add, background="white", shadows=False):
    plotter = pv.Plotter(off_screen=True, window_size=SIZE, lighting="none")
    plotter.set_background(background)
    add(plotter)
    lights(plotter)
    if shadows:
        plotter.enable_shadows()
        plotter.renderer._render_passes._shadow_map_pass.GetShadowMapBakerPass().SetResolution(8192)
    plotter.camera.focal_point = CENTER
    plotter.camera.position = CENTER + DIRECTION * 4000
    plotter.camera.up = UP
    plotter.camera.view_angle = 20
    plotter.reset_camera(bounds=BOUNDS)
    plotter.camera.zoom(ZOOM)
    plotter.render()
    image = plotter.screenshot(return_img=True)
    plotter.close()
    # scaled down to half, for smooth edges
    return np.asarray(Image.fromarray(image).resize((SIZE[0] // 2, SIZE[1] // 2), Image.LANCZOS))


def add_seg(plotter):
    plotter.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.2, specular_power=20,
                     ambient=0.35, diffuse=0.75, show_scalar_bar=False)


def add_plane(plotter):
    """The section plane (from platy_render_section2.py), as a faint glass sheet with a thin edge."""
    if not os.environ.get("PLANE"):
        return
    import json
    plane = json.load(open(os.environ["PLANE"]))["plane"]
    (xa, xb), (za, zb), y = plane["x"], plane["z"], plane["y"]
    corners = np.array([(xa, y, za), (xa, y, zb), (xb, y, zb), (xb, y, za)], float)
    sheet = pv.PolyData(corners, faces=[4, 0, 1, 2, 3])
    plotter.add_mesh(sheet, color=(0.30, 0.50, 0.75), opacity=0.22, ambient=1.0, diffuse=0.0, specular=0.0)
    edge = pv.PolyData(np.vstack([corners, corners[:1]]), lines=[5, 0, 1, 2, 3, 4])
    plotter.add_mesh(edge, color=(0.20, 0.33, 0.52), line_width=5, ambient=1.0, diffuse=0.0)


if MODE != "lib":
    os.makedirs(OUT, exist_ok=True)
if MODE == "seg":
    # the nuclei in the studio, the empty studio, and the nuclei alone (for their outline)
    np.save(f"{OUT}/seg_studio.npy", shoot(lambda p: (add_seg(p), backdrop(p), add_plane(p)), shadows=True))
    log("seg studio")
    np.save(f"{OUT}/studio_empty.npy", shoot(backdrop))
    log("studio empty")
    np.save(f"{OUT}/seg_alone.npy", shoot(add_seg, background="black"))
    log("seg alone")
elif MODE == "raw":
    # the full resolution raw (from platy_render_hq.py), on black; the s0 voxels are half the s1 voxels
    raw = np.load(sys.argv[3])

    def add_raw(plotter):
        grid = pv.ImageData(dimensions=raw.shape[::-1], spacing=(0.5, 0.5, 0.5))
        grid.point_data["values"] = raw.ravel(order="C")
        plotter.add_volume(grid, scalars="values", cmap="gray", opacity="linear", clim=(0, 230),
                           show_scalar_bar=False, blending="maximum", mapper="gpu")
    np.save(f"{OUT}/raw.npy", shoot(add_raw, background="black"))
    log("raw")
if MODE != "lib":
    log("done")
