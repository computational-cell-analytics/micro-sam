"""A thin longitudinal section of the expanded Platynereis nuclei (all nuclei with their centre in a slab), and
close-ups of its anterior and posterior ends, in the colours of the studio render. Also projects the slab outline
into the studio image. Usage: platy_render_section.py OUT_DIR STUDIO_FRAME_JSON."""
import json
import os
import sys

import numpy as np
import pyvista as pv
import zarr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platy_render_studio as L  # noqa: E402

OUT, FRAME, RAW = sys.argv[1], sys.argv[2], sys.argv[3]
VOXEL_UM = 2.5112882406416843  # s1 voxel size, expanded
EXPANSION = 10.0  # assumed expansion factor, for the pre-expansion scale bars
SLAB_UM = 11.0  # section thickness, pre-expansion
# the section is seen from its normal (z, the studio camera side), tilted a little like the studio view
DIRECTION_B = np.array([-0.15, 0.25, -1.0]) / np.linalg.norm([-0.15, 0.25, -1.0])
os.makedirs(OUT, exist_ok=True)

# the same separation and colours as the studio render
labels = zarr.open_group(L.P + "/labels/nuclei", mode="r")["s1"][:]
n_labels = int(labels.max())
separated, pairs = L.separate(labels)
pairs |= L.near_pairs(zarr.open_group(L.P + "/labels/nuclei", mode="r")["s2"][:])
colors = L.distinct_colors(n_labels, pairs)
L.log("colours")

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
centre = sums / np.maximum(counts, 1)
del labels
ids = np.arange(1, n_labels + 1)
cz, cy, cx = centre[:, 1:]

# the slab: nuclei with their centre within half the thickness of the middle of the body in z; the studio camera
# sees this plane almost edge on
half = SLAB_UM * EXPANSION / VOXEL_UM / 2
z_mid = float(np.median(cz))
in_slab = np.abs(cz - z_mid) <= half
slab_ids = ids[in_slab]
L.log("slab", len(slab_ids), "nuclei, half thickness", round(half, 1), "voxels at z", round(z_mid, 1))
full_mesh = L.label_surfaces(separated)
full_mesh.cell_data["rgb"] = (colors[full_mesh.cell_data["label"]] * 255).astype("uint8")
sub = np.where(np.isin(separated, slab_ids), separated, 0)
del separated
mesh = L.label_surfaces(sub)
del sub
mesh.cell_data["rgb"] = (colors[mesh.cell_data["label"]] * 255).astype("uint8")

# the anterior (round end, low x) and posterior boxes, from the centres of the slab nuclei
sx, sy = cx[in_slab], cy[in_slab]
x_lo, x_hi = sx.min(), sx.max()
length = x_hi - x_lo
boxes = {"anterior": (x_lo - 8, x_lo + 0.30 * length), "posterior": (x_hi - 0.22 * length, x_hi + 8)}
box_ids, box_bounds = {}, {}
for name, (b0, b1) in boxes.items():
    pick = (sx >= b0) & (sx <= b1)
    box_ids[name] = slab_ids[pick]
    y0, y1 = sy[pick].min() - 14, sy[pick].max() + 14
    box_bounds[name] = (b0, b1, y0, y1, z_mid - half, z_mid + half)


def project(plotter, points):
    """Image coordinates (column, row) of world points (x, y, z), at the final (half) image size."""
    renderer = plotter.renderer
    height = plotter.window_size[1]
    out = []
    for p in points:
        renderer.SetWorldPoint(*p, 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        out.append((x / 2, (height - y) / 2))
    return out


def rectangle(bounds):
    """The four corners of a box in its middle z plane, in x and y."""
    x0, x1, y0, y1, z0, z1 = bounds
    z = (z0 + z1) / 2
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def shoot(part, bounds, size, zoom, points=(), add=None, background="white", transparent=False):
    plotter = pv.Plotter(off_screen=True, window_size=(size[0] * 2, size[1] * 2), lighting="none")
    plotter.set_background(background)
    if add is not None:
        add(plotter)
    else:
        plotter.add_mesh(part, scalars="rgb", rgb=True, smooth_shading=True, specular=0.2, specular_power=20,
                         ambient=0.35, diffuse=0.75, show_scalar_bar=False)
    centre = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
    plotter.add_light(pv.Light(position=tuple(centre + np.array([-2500.0, 1500.0, -1500.0])),
                               focal_point=tuple(centre), intensity=0.7, light_type="scene light"))
    plotter.add_light(pv.Light(position=tuple(centre + DIRECTION_B * 3000), focal_point=tuple(centre),
                               intensity=0.55, light_type="scene light"))
    plotter.enable_ssao(radius=5, bias=0.05, kernel_size=128, blur=True)
    plotter.camera.focal_point = centre
    plotter.camera.position = centre + DIRECTION_B * 4000
    plotter.camera.up = L.UP
    plotter.camera.view_angle = 20
    plotter.reset_camera(bounds=bounds)
    plotter.camera.zoom(zoom)
    plotter.render()
    image = plotter.screenshot(return_img=True, transparent_background=transparent)
    projected = project(plotter, points)
    # pixels per voxel at the depth of the centre, from two points 50 voxels apart along x
    a, b = project(plotter, [centre, centre + np.array([50.0, 0, 0])])
    px_per_voxel = float(np.hypot(b[0] - a[0], b[1] - a[1]) / 50)
    plotter.close()
    image = np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS))
    return image, projected, px_per_voxel


layout = {"slab_nuclei": int(len(slab_ids)), "slab_um": SLAB_UM, "expansion": EXPANSION, "voxel_um": VOXEL_UM}
slab_bounds = mesh.bounds
image, projected, ppv = shoot(mesh, slab_bounds, (1100, 2000), 1.0,
                             rectangle(box_bounds["anterior"]) + rectangle(box_bounds["posterior"]))
Image.fromarray(image).save(f"{OUT}/section.png")
# the same render with a transparent background, for the stack in the figure (no white edges on the raw)
image, _, _ = shoot(mesh, slab_bounds, (1100, 2000), 1.0, transparent=True)
Image.fromarray(image).save(f"{OUT}/section_rgba.png")
layout["section"] = {"anterior_box": projected[:4], "posterior_box": projected[4:], "px_per_voxel": ppv}
L.log("section")

# the raw in the middle plane of the slab, from the same camera: a dark card over the slab area
raw = np.load(RAW, mmap_mode="r")
sx0, sx1, sy0, sy1, _, _ = slab_bounds
pad = 12
ia, ib = int(max(0, 2 * (sx0 - pad))), int(min(raw.shape[2], 2 * (sx1 + pad)))
ja, jb = int(max(0, 2 * (sy0 - pad))), int(min(raw.shape[1], 2 * (sy1 + pad)))
plane = np.asarray(raw[int(round(2 * z_mid)), ja:jb, ia:ib], dtype="float32")
# the contrast of the single slice, which is dimmer than the projection the raw was scaled for
RAW_HI = float(np.percentile(plane[plane > 0], 99.0))


def add_raw(plotter):
    grid = pv.ImageData(dimensions=(plane.shape[1], plane.shape[0], 1), spacing=(0.5, 0.5, 1.0),
                        origin=(ia / 2, ja / 2, z_mid))
    grid.point_data["values"] = plane.ravel(order="C")
    plotter.add_mesh(grid, scalars="values", cmap="gray", clim=(0, RAW_HI), ambient=1.0, diffuse=0.0,
                     specular=0.0, show_scalar_bar=False)


image, _, _ = shoot(None, slab_bounds, (1100, 2000), 1.0, add=add_raw)
Image.fromarray(image).save(f"{OUT}/section_raw.png")
layout["section"]["raw_slice_s0_z"] = int(round(2 * z_mid))
L.log("raw slice")
for name in ("anterior", "posterior"):
    keep = np.isin(mesh.cell_data["label"], box_ids[name])
    part = mesh.extract_cells(np.flatnonzero(keep))
    image, _, ppv = shoot(part, part.bounds, (1300, 1300), 1.05)
    Image.fromarray(image).save(f"{OUT}/{name}.png")
    layout[name] = {"nuclei": int(len(box_ids[name])), "px_per_voxel": ppv}
    L.log(name)

# the slab outline in the studio image, with the studio camera and its crop
frame = json.load(open(FRAME))
plotter = pv.Plotter(off_screen=True, window_size=L.SIZE)
plotter.add_mesh(pv.Sphere(radius=1, center=tuple(L.CENTER)))
plotter.camera.focal_point = L.CENTER
plotter.camera.position = L.CENTER + L.DIRECTION * 4000
plotter.camera.up = L.UP
plotter.camera.view_angle = 20
plotter.reset_camera(bounds=L.BOUNDS)
plotter.camera.zoom(L.ZOOM)
plotter.render()
x0, x1, y0, y1, _, _ = slab_bounds
# from above the round end down to the floor, like a sheet standing in the studio
outline = project(plotter, rectangle((x0 - 40, x1 + 60, y0 - 25, y1 + 25, z_mid, z_mid)))
a, b = project(plotter, [L.CENTER, L.CENTER + np.array([50.0, 0, 0])])
plotter.close()
corners = rectangle((x0 - 40, x1 + 60, y0 - 25, y1 + 25, z_mid, z_mid))
layout["studio_plane_world"] = corners
# the placement check: the nuclei with a real plane at the same place, from the studio camera, with the same crop
check = pv.Plotter(off_screen=True, window_size=L.SIZE)
check.set_background("white")
check.add_mesh(full_mesh, scalars="rgb", rgb=True, smooth_shading=True, show_scalar_bar=False)
check.add_mesh(pv.PolyData(np.array(corners, float), faces=[4, 0, 1, 2, 3]), color="red", opacity=0.45,
               ambient=1.0, diffuse=0.0)
check.camera.focal_point = L.CENTER
check.camera.position = L.CENTER + L.DIRECTION * 4000
check.camera.up = L.UP
check.camera.view_angle = 20
check.reset_camera(bounds=L.BOUNDS)
check.camera.zoom(L.ZOOM)
check.render()
shot = check.screenshot(return_img=True)
check.close()
shot = np.asarray(Image.fromarray(shot).resize((L.SIZE[0] // 2, L.SIZE[1] // 2), Image.LANCZOS))
Image.fromarray(shot[frame["r0"]:frame["r1"], frame["c0"]:frame["c1"]]).save(f"{OUT}/placement_check.png")
layout["studio"] = {"slab_outline": [(c - frame["c0"], r - frame["r0"]) for c, r in outline],
                    "px_per_voxel": float(np.hypot(b[0] - a[0], b[1] - a[1]) / 50)}
json.dump(layout, open(f"{OUT}/layout.json", "w"), indent=1)
L.log("done")
