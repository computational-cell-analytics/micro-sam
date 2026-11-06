"""Renders for the MCF7 spheroid panel, in the layout of the Platynereis panel: a) the nuclei in the studio (the mid
view, elevation 40) with the outline of the middle z plane, b) the nuclei cut by that plane, seen from above, and the
DAPI raw of the plane from the same camera.
Usage: render_spheroid_section.py LABELS_TIF RAW_TIF OUT_DIR, with the labels and raw of
/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/mcf7_spheroids/dapi_ki67: s2_nucleus_prediction.tif and s2.tif."""
import json
import os
import sys

import numpy as np
import pyvista as pv
import tifffile
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "platynereis"))
import platy_render_studio as L  # noqa: E402

LABELS, RAW, OUT = sys.argv[1:4]
XY_UM, Z_UM = 0.414, 1.5
Z_SCALE = Z_UM / XY_UM  # the mesh is in xy voxel units
ELEVATION, AZIMUTH = 40, 40  # the studio view
STUDIO_SIZE = (2000, 2000)
SECTION_SIZE = (1600, 1600)
os.makedirs(OUT, exist_ok=True)

labels = tifffile.imread(LABELS).astype("uint32")
ids, inverse = np.unique(labels, return_inverse=True)
labels = inverse.reshape(labels.shape).astype("uint16")
n_labels = len(ids) - 1
separated, pairs = L.separate(labels)
pairs |= L.near_pairs(labels, radius=3)
colors = L.distinct_colors(n_labels, pairs)


def surfaces(volume):
    mesh = L.label_surfaces(volume)
    mesh.points[:, 2] *= Z_SCALE
    mesh.cell_data["rgb"] = (colors[mesh.cell_data["label"]] * 255).astype("uint8")
    return mesh


mesh = surfaces(separated)
L.log("mesh", n_labels, "nuclei")

# the centre of each nucleus in voxels (z, y, x)
zz, yy, xx = np.nonzero(labels)
lab = labels[zz, yy, xx]
count = np.maximum(np.bincount(lab, minlength=n_labels + 1), 1)
cz = np.bincount(lab, weights=zz, minlength=n_labels + 1)[1:] / count[1:]

# the section: the nuclei cut by the middle plane (the plane of the raw in b), rendered in full
raw_z = int(round(np.median(cz)))
slab_ids = np.unique(labels[raw_z])[1:]
section = surfaces(np.where(np.isin(separated, slab_ids), separated, 0))
z_plane = raw_z * Z_SCALE
L.log("section", len(slab_ids), "nuclei at z", raw_z)

x0, x1, y0, y1, z0, z1 = mesh.bounds
middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
flat = np.array([np.cos(np.radians(AZIMUTH)), np.sin(np.radians(AZIMUTH)), 0.0])
view = np.cos(np.radians(ELEVATION)) * flat + np.array([0, 0, np.sin(np.radians(ELEVATION))])


def project(plotter, points):
    """Image coordinates (column, row) of world points, at the final (half) image size."""
    renderer = plotter.renderer
    height = plotter.window_size[1]
    out = []
    for p in points:
        renderer.SetWorldPoint(*p, 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        out.append((x / 2, (height - y) / 2))
    return out


def px_per_voxel(plotter, centre):
    a, b = project(plotter, [centre, centre + np.array([50.0, 0, 0])])
    return float(np.hypot(b[0] - a[0], b[1] - a[1]) / 50)


def backdrop(plotter):
    """A floor under the spheroid and two walls behind it, on the sides away from the camera."""
    m, reach = 25.0, 260.0
    floor_z = z0 - m
    wx = x0 - m if view[0] > 0 else x1 + m
    wy = y0 - m if view[1] > 0 else y1 + m
    fx = x1 + reach if view[0] > 0 else x0 - reach
    fy = y1 + reach if view[1] > 0 else y0 - reach
    top = z1 + 120

    def quad(corners):
        return pv.PolyData(np.array(corners, float), faces=[4, 0, 1, 2, 3]).triangulate().subdivide(4)

    for plane in (quad([(wx, wy, floor_z), (fx, wy, floor_z), (fx, fy, floor_z), (wx, fy, floor_z)]),
                  quad([(wx, wy, floor_z), (wx, fy, floor_z), (wx, fy, top), (wx, wy, top)]),
                  quad([(wx, wy, floor_z), (fx, wy, floor_z), (fx, wy, top), (wx, wy, top)])):
        plotter.add_mesh(plane, color=(0.95, 0.95, 0.95), ambient=0.45, diffuse=0.6, specular=0.0)


def add_nuclei(plotter, part):
    plotter.add_mesh(part, scalars="rgb", rgb=True, smooth_shading=True, specular=0.2, specular_power=20,
                     ambient=0.35, diffuse=0.75, show_scalar_bar=False)


def finish(plotter, size, transparent=False):
    plotter.render()
    image = plotter.screenshot(return_img=True, transparent_background=transparent)
    return np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS))


# a: the studio, and the outline of the section plane in it
studio = pv.Plotter(off_screen=True, window_size=(STUDIO_SIZE[0] * 2, STUDIO_SIZE[1] * 2), lighting="none")
studio.set_background("white")
add_nuclei(studio, mesh)
backdrop(studio)
side = np.cross(view, [0, 0, 1.0])
key = middle + 900 * (0.5 * view + 0.45 * side / np.linalg.norm(side) + np.array([0, 0, 1.0]))
studio.add_light(pv.Light(position=tuple(key), focal_point=tuple(middle), intensity=0.7, positional=True,
                          cone_angle=50, light_type="scene light"))
studio.add_light(pv.Light(position=tuple(middle + 1500 * view), focal_point=tuple(middle), intensity=0.55,
                          light_type="scene light"))
studio.enable_shadows()
studio.renderer._render_passes._shadow_map_pass.GetShadowMapBakerPass().SetResolution(8192)
studio.camera.focal_point = middle
studio.camera.position = middle + 2000 * view
studio.camera.up = (0, 0, 1)
studio.camera.view_angle = 20
studio.reset_camera(bounds=mesh.bounds)
image = finish(studio, STUDIO_SIZE)
# the section card: a square in the plane, its sides along the view (far side up) and across it, so that it
# shows as a near upright card in a and b
up_h = -flat
right_h = np.array([-flat[1], flat[0], 0.0])
sx0, sx1, sy0, sy1, _, _ = section.bounds
centre = np.array([(sx0 + sx1) / 2, (sy0 + sy1) / 2, z_plane])
rel = section.points - centre
R = float(np.abs(np.c_[rel @ right_h, rel @ up_h]).max()) + 12
corners = [tuple(centre + R * (a * right_h + b * up_h)) for a, b in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
layout = {"nuclei": n_labels, "slab_nuclei": int(len(slab_ids)), "xy_um": XY_UM, "z_um": Z_UM, "raw_z": raw_z,
          "studio": {"slab_outline": project(studio, corners), "px_per_voxel": px_per_voxel(studio, middle)}}
studio.close()
Image.fromarray(image).save(f"{OUT}/studio.png")
L.log("studio")

# b: the section from above, tilted a little towards the studio camera; the far side on top, as in the studio
raw = tifffile.imread(RAW)[raw_z, 0].astype("float32")
steps = np.arange(-R, R + 0.5, 0.5)
A, B = np.meshgrid(steps, steps, indexing="ij")
X = centre[0] + A * right_h[0] + B * up_h[0]
Y = centre[1] + A * right_h[1] + B * up_h[1]
# the background level of the raw is black, also outside the image
lo, hi = np.percentile(raw, [10, 99.5])
plane = ndimage.map_coordinates(raw, [Y, X], order=1, cval=lo)
DIRECTION_B = 0.5 * flat + np.array([0, 0, 1.0])
DIRECTION_B /= np.linalg.norm(DIRECTION_B)
right = np.cross(-DIRECTION_B, -flat)
right /= np.linalg.norm(right)
camera_up = np.cross(right, -DIRECTION_B)


def shoot_section(add, transparent=False):
    plotter = pv.Plotter(off_screen=True, window_size=(SECTION_SIZE[0] * 2, SECTION_SIZE[1] * 2), lighting="none")
    plotter.set_background("white")
    add(plotter)
    # a key light from the upper left of the camera, and a fill light from the camera
    key = centre + 3000 * (DIRECTION_B + 0.5 * camera_up - 0.5 * right)
    plotter.add_light(pv.Light(position=tuple(key), focal_point=tuple(centre), intensity=0.7,
                               light_type="scene light"))
    plotter.add_light(pv.Light(position=tuple(centre + 3000 * DIRECTION_B), focal_point=tuple(centre),
                               intensity=0.55, light_type="scene light"))
    plotter.enable_ssao(radius=5, bias=0.05, kernel_size=128, blur=True)
    plotter.camera.focal_point = centre
    plotter.camera.position = centre + 3000 * DIRECTION_B
    plotter.camera.up = tuple(camera_up)
    plotter.camera.view_angle = 20
    c = np.array(corners)
    plotter.reset_camera(bounds=(c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max(), z_plane - 5,
                                 z_plane + 5))
    plotter.camera.zoom(1.6)
    image = finish(plotter, SECTION_SIZE, transparent)
    ppv = px_per_voxel(plotter, centre)
    plotter.close()
    return image, ppv


def add_raw(plotter):
    grid = pv.StructuredGrid(X[..., None], Y[..., None], np.full(X.shape + (1,), z_plane))
    grid.point_data["values"] = plane.ravel(order="F")
    plotter.add_mesh(grid, scalars="values", cmap="gray", clim=(lo, hi), ambient=1.0, diffuse=0.0, specular=0.0,
                     show_scalar_bar=False)


image, ppv = shoot_section(lambda p: add_nuclei(p, section), transparent=True)
Image.fromarray(image).save(f"{OUT}/section_rgba.png")
image, _ = shoot_section(add_raw)
Image.fromarray(image).save(f"{OUT}/section_raw.png")
layout["section"] = {"px_per_voxel": ppv}
json.dump(layout, open(f"{OUT}/layout.json", "w"), indent=1)
L.log("done")
