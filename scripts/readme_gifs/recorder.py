import numpy as np
from PIL import Image, ImageDraw

from qtpy.QtWidgets import QApplication

# The arrow cursor as a polygon in pixels, with the tip at (0, 0).
CURSOR = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]


class Recorder:
    """Record frames of the napari canvas and save them as a GIF.

    Args:
        viewer: The napari viewer.
        size: The size of the square canvas in pixels.
    """

    def __init__(self, viewer, size=512):
        self.viewer = viewer
        self.size = size
        self.frames, self.durations = [], []
        self.cursor = None
        # A hidden viewer renders a black canvas, so show the window.
        viewer.show()
        viewer.window._qt_window.resize(size + 700, size + 300)
        for _ in range(5):
            QApplication.processEvents()

    def fit(self, image_shape):
        """Center the image (y, x) and zoom it to fill the square crop."""
        self.viewer.reset_view()
        h, w = image_shape[-2:]
        self.viewer.camera.center = (h / 2 - 0.5, w / 2 - 0.5)
        self.viewer.camera.zoom = self.size / max(h, w)

    def to_screen(self, yx, shot_shape):
        """Map a data coordinate (y, x) to a pixel position (x, y) in the screenshot."""
        cam = self.viewer.camera
        cy, cx = cam.center[-2:]
        h, w = shot_shape[:2]
        return ((yx[1] - cx) * cam.zoom + w / 2, (yx[0] - cy) * cam.zoom + h / 2)

    def grab(self):
        QApplication.processEvents()
        shot = self.viewer.screenshot(canvas_only=True, flash=False)[..., :3]
        # Crop the center square, so that all frames have the same shape.
        h, w = shot.shape[:2]
        s = min(h, w, self.size)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return shot, (y0, x0, s)

    def snap(self, duration=60, cursor="keep"):
        """Add one frame.

        Args:
            duration: The display time of the frame in milliseconds.
            cursor: The data position (y, x) of the cursor, None to hide it, or 'keep'.
        """
        if not (isinstance(cursor, str) and cursor == "keep"):
            self.cursor = cursor
        shot, (y0, x0, s) = self.grab()
        img = Image.fromarray(shot)
        if self.cursor is not None:
            px, py = self.to_screen(self.cursor, shot.shape)
            poly = [(px + dx, py + dy) for dx, dy in CURSOR]
            draw = ImageDraw.Draw(img)
            draw.polygon(poly, fill=(255, 255, 255), outline=(0, 0, 0))
        img = img.crop((x0, y0, x0 + s, y0 + s))
        self.frames.append(img)
        self.durations.append(duration)

    def hold(self, duration):
        """Extend the display time of the last frame."""
        self.durations[-1] += duration

    def move(self, target, steps=6, duration=40):
        """Move the cursor to the data position (y, x) in a few frames."""
        start = self.cursor if self.cursor is not None else target
        for t in np.linspace(0, 1, steps + 1)[1:]:
            pos = (start[0] + t * (target[0] - start[0]), start[1] + t * (target[1] - start[1]))
            self.snap(duration, cursor=pos)

    def save(self, path, out_size=400):
        frames = [f.resize((out_size, out_size), Image.LANCZOS) for f in self.frames]
        # A shared palette from frames across the whole recording avoids the flicker of per-frame palettes.
        picks = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).astype(int)]
        mosaic = Image.new("RGB", (out_size * len(picks), out_size))
        for i, f in enumerate(picks):
            mosaic.paste(f, (i * out_size, 0))
        palette_src = mosaic.quantize(colors=255, method=Image.MEDIANCUT)
        frames = [f.quantize(palette=palette_src, dither=Image.NONE) for f in frames]
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=self.durations, loop=0, optimize=True,
        )
