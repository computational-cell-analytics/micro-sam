"""Put the raw into the same studio as the nuclei: the shadows of the nuclei render on the walls, and the raw on top,
either as dark signal on the light studio (inverted) or as bright signal in a dimmed studio (dark). Both images
are cropped to the same tight frame. Usage: platy_studio_compose.py RENDER_DIR OUT_DIR."""
import os
import sys

import numpy as np
from PIL import Image

D, OUT = sys.argv[1], sys.argv[2]
seg = np.load(f"{D}/seg_studio.npy").astype("float32")
empty = np.load(f"{D}/studio_empty.npy").astype("float32")
alone = np.load(f"{D}/seg_alone.npy").astype("float32")
raw = np.load(f"{D}/raw.npy").astype("float32")[..., 0] / 255

# the studio with the shadows of the animal, without the animal itself
body = alone.sum(-1) > 20
studio = np.where(body[..., None], empty, seg)
signal = raw[..., None]
inverted = studio * (1 - 0.92 * signal)
dim = studio * 0.16
dark = 255 - (255 - dim) * (1 - signal)

# a tight frame around the animal, with its shadows: a margin of 9 % of the animal size, more below for the floor
rows, cols = np.nonzero(body | (raw > 0.05))
height, width = rows.max() - rows.min(), cols.max() - cols.min()
r0 = max(rows.min() - int(0.09 * height), 0)
r1 = min(rows.max() + int(0.14 * height), seg.shape[0])
c0 = max(cols.min() - int(0.22 * width), 0)
c1 = min(cols.max() + int(0.22 * width), seg.shape[1])
os.makedirs(OUT, exist_ok=True)
for name, image in [("seg_studio", seg), ("raw_studio_inverted", inverted), ("raw_studio_dark", dark)]:
    Image.fromarray(np.clip(image[r0:r1, c0:c1], 0, 255).astype("uint8")).save(f"{OUT}/{name}.png")
import json
json.dump({"r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)}, open(f"{OUT}/frame.json", "w"))
print("frame", r1 - r0, c1 - c0)
