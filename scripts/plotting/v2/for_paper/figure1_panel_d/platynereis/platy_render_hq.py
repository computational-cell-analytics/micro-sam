"""High quality render of the upright oblique view of the expanded Platynereis: the raw at full resolution (s0) and
the nuclei with ambient occlusion. Usage: platy_render_hq.py raw|seg OUT_DIR. Saves the renders as npy, for the
combination step."""
import os
import sys
import time

import numpy as np
import pyvista as pv
import vtk
import zarr
from matplotlib.colors import ListedColormap
from scipy import ndimage

sys.path.insert(0, "/mnt/vast-nhr/home/archit/u12090/micro-sam/scripts/plotting/v2/for_paper")
from plots_for_figure1 import neurite_colors  # noqa: E402

P = "/mnt/vast-nhr/projects/cidas/cca/data/huisken_data/exm_lsm/high_res/stitched/Expanded_Platynereis.ome.zarr"
MODE, OUT = sys.argv[1], sys.argv[2]
SIZE = (1650, 3000)
# the s1 grid is the reference frame for the camera; s0 voxels get a spacing of 0.5
S1 = (770, 477, 1197)
BOUNDS = (0, S1[2], 0, S1[1], 0, S1[0])
CENTER = np.array([S1[2], S1[1], S1[0]], float) / 2
DIRECTION, UP = (0.0, -1.0, -0.6), (-1, 0, 0)
t0 = time.time()
LOW, HIGH = float(os.environ.get("LOW", 20)), float(os.environ.get("HIGH", 99.9))


def log(*args):
    print(*args, round(time.time() - t0), flush=True)


def image_data(volume, spacing=1.0):
    grid = pv.ImageData(dimensions=volume.shape[::-1], spacing=(spacing,) * 3)
    grid.point_data["values"] = volume.ravel(order="C")
    return grid


def shoot(add, background):
    plotter = pv.Plotter(off_screen=True, window_size=SIZE)
    plotter.set_background(background)
    add(plotter)
    direction = np.asarray(DIRECTION, float) / np.linalg.norm(DIRECTION)
    plotter.camera.focal_point = CENTER
    plotter.camera.position = CENTER + direction * 4000
    plotter.camera.up = UP
    plotter.camera.view_angle = 20
    plotter.reset_camera(bounds=BOUNDS)
    plotter.camera.zoom(1.35)
    plotter.render()
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def upsample(volume, factor, shape):
    """Nearest upsampling of a coarse volume into the given shape; a missing edge is filled from the last voxel."""
    up = volume
    for axis in range(3):
        up = np.repeat(up, factor, axis)
    up = np.pad(up, [(0, max(0, s - u)) for s, u in zip(shape, up.shape)], mode="edge")
    return up[:shape[0], :shape[1], :shape[2]]


def full_raw():
    """The raw at s0, divided by its smooth local level, scaled to uint8 and masked to the body."""
    group = zarr.open_group(P, mode="r")
    shape = tuple(2 * s for s in S1)
    coarse = zarr.open_group(P + "/labels/nuclei", mode="r")["s2"][:] > 0
    body = ndimage.binary_dilation(ndimage.binary_closing(coarse, iterations=3), iterations=3)
    level = ndimage.gaussian_filter(group["s2"][:].astype("float32"), 12) + 1.0
    # the normalised volume in half precision, then the contrast limits from a sparse sample of it at full resolution
    norm = np.zeros(shape, "float16")
    slab = 64
    for z0 in range(0, shape[0], slab):
        z1 = min(z0 + slab, shape[0])
        c0, c1 = z0 // 4, (z1 + 3) // 4
        block = group["s0"][z0:z1, :shape[1], :shape[2]].astype("float32")
        lev = upsample(level[c0:c1], 4, (c1 * 4 - c0 * 4,) + shape[1:])[z0 - c0 * 4:z1 - c0 * 4]
        mask = upsample(body[c0:c1], 4, (c1 * 4 - c0 * 4,) + shape[1:])[z0 - c0 * 4:z1 - c0 * 4]
        block = block / lev
        block[~mask] = np.nan
        norm[z0:z1, :block.shape[1], :block.shape[2]] = block
    sample = norm[::3, ::3, ::3].astype("float32")
    lo, hi = np.nanpercentile(sample, [LOW, HIGH])
    del sample
    log("limits", lo, hi)
    raw = np.zeros(shape, "uint8")
    for z0 in range(0, shape[0], slab):
        block = norm[z0:z0 + slab].astype("float32")
        block = np.nan_to_num(np.clip((block - lo) / (hi - lo), 0, 1) * 255, nan=0)
        raw[z0:z0 + slab] = block.astype("uint8")
    del norm
    log("raw s0", raw.shape)
    return raw


def label_surfaces(labels):
    grid = image_data(labels)
    nets = vtk.vtkSurfaceNets3D()
    nets.SetInputData(grid)
    nets.SetInputArrayToProcess(0, 0, 0, 0, "values")
    nets.GenerateLabels(int(labels.max()), 1, int(labels.max()))
    nets.SetOutputMeshTypeToTriangles()
    nets.SetOutputStyleToBoundary()
    nets.SmoothingOn()
    nets.SetNumberOfIterations(int(os.environ.get("SMOOTH", 40)))
    nets.Update()
    mesh = pv.wrap(nets.GetOutput())
    mesh.cell_data["label"] = np.asarray(mesh.cell_data["BoundaryLabels"]).max(axis=1)
    return mesh


os.makedirs(OUT, exist_ok=True)
if MODE == "raw":
    cache = f"{OUT}/raw_s0.npy"
    if os.path.exists(cache):
        raw = np.load(cache)
    else:
        raw = full_raw()
        np.save(cache, raw)
    for cmap, name in [("gray", "gray"), ("inferno", "inferno")]:
        def add(plotter, cmap=cmap):
            actor = plotter.add_volume(image_data(raw, 0.5), scalars="values", cmap=cmap, opacity="linear",
                                       clim=(0, 230), show_scalar_bar=False, blending="maximum",
                                       mapper="gpu")
        np.save(f"{OUT}/raw_{name}.npy", shoot(add, "black"))
        log("raw", name)
else:
    labels = zarr.open_group(P + "/labels/nuclei", mode="r")["s1"][:]
    n_labels = int(labels.max())
    mesh = label_surfaces(labels)
    del labels
    log("mesh", mesh.n_cells)
    rng = np.random.default_rng(3)
    palette = neurite_colors(14)
    cmap = ListedColormap(palette[rng.integers(0, len(palette), n_labels + 1)])
    for background in ["white", "black"]:
        def add(plotter):
            plotter.enable_lightkit()
            plotter.add_mesh(mesh, scalars="label", cmap=cmap, clim=(0, n_labels), smooth_shading=True,
                             specular=0.25, specular_power=25, ambient=0.1, diffuse=0.9, show_scalar_bar=False)
            plotter.enable_ssao(radius=6, bias=0.05, kernel_size=128, blur=True)
            plotter.enable_anti_aliasing("ssaa")
        np.save(f"{OUT}/seg_{background}.npy", shoot(add, background))
        log("seg", background)
log("done")
