"""Check napari workflows with cached microscopy data and a real SAM checkpoint."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import imageio.v3 as imageio

import napari
from napari.utils.key_bindings import coerce_keybinding

from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator import annotator_2d, annotator_3d, annotator_tracking, image_series_annotator


def check_rendering(viewer, path):
    viewer.window.resize(1200, 800)
    QApplication.processEvents()
    QTest.qWait(300)
    screenshot = viewer.screenshot(path=str(path), canvas_only=False)
    assert screenshot.std() > 0, "The viewer screenshot is empty."


def invoke_shortcut(viewer, key, layer=None):
    target = viewer if layer is None else viewer.layers[layer]
    target.keymap[coerce_keybinding(key)](viewer)
    QApplication.processEvents()


def check_prompt_segmentation(viewer):
    image = viewer.layers["image"].data
    point = [size // 2 for size in image.shape]
    if image.ndim == 3:
        viewer.dims.set_current_step(0, point[0])
    viewer.layers["point_prompts"].add(point)
    invoke_shortcut(viewer, "s", "point_prompts")
    assert viewer.layers["current_object"].data.any(), "Point segmentation is empty."
    invoke_shortcut(viewer, "c")
    assert viewer.layers["committed_objects"].data.any(), "Commit did not save the segmentation."
    invoke_shortcut(viewer, "t")
    assert viewer.layers["point_prompts"].current_properties["label"][0] == "negative"
    invoke_shortcut(viewer, "t")
    invoke_shortcut(viewer, "Shift-C")
    assert len(viewer.layers["point_prompts"].data) == 0
    if image.ndim == 2:
        viewer.layers["prompts"].add(np.array([[64, 64], [64, 192], [192, 192], [192, 64]]), shape_type="rectangle")
        invoke_shortcut(viewer, "s", "prompts")
        assert viewer.layers["current_object"].data.any(), "Box segmentation is empty."
        invoke_shortcut(viewer, "Shift-C")


def check_automatic(image, cache, device, output):
    AnnotatorState().reset_state()
    viewer = napari.Viewer()
    try:
        annotator_2d(
            image, viewer=viewer, return_viewer=True, model_type="vit_b_lm", device=device,
            checkpoint_path=str(cache / "models" / "vit_b_lm"),
            decoder_path=str(cache / "models" / "vit_b_lm_decoder"),
        )
        state = AnnotatorState()
        assert state.decoder is not None
        state.widgets["autosegment"]()
        assert viewer.layers["auto_segmentation"].data.any(), "Automatic segmentation is empty."
        state.widgets["commit"](viewer, layer="auto_segmentation")
        assert viewer.layers["committed_objects"].data.any()
        check_rendering(viewer, output / "automatic.png")
        print("PASS: automatic segmentation and commit with the microscopy decoder", flush=True)
    finally:
        viewer.close()


def check_plugins(output):
    names = [
        "Annotator 2d", "Annotator 3d", "Annotator Tracking", "Image Series Annotator",
        "Object Classifier", "Finetuning", "Settings",
    ]
    for index, name in enumerate(names):
        AnnotatorState().reset_state()
        viewer = napari.Viewer()
        try:
            dock, widget = viewer.window.add_plugin_dock_widget("micro_sam", name)
            check_rendering(viewer, output / f"plugin-{index}.png")
            assert dock.isVisible(), f"The {name} dock is hidden."
            assert dock.height() > 100, f"The {name} dock is too short."
            dock.setFloating(True)
            QApplication.processEvents()
            dock.setFloating(False)
            QApplication.processEvents()
            assert widget is not None
            print(f"PASS: plugin {name}", flush=True)
        finally:
            viewer.close()


def check_annotator(name, function, image, kwargs, output):
    AnnotatorState().reset_state()
    viewer = napari.Viewer()
    try:
        function(image, viewer=viewer, return_viewer=True, **kwargs)
        for layer in ("current_object", "committed_objects", "auto_segmentation"):
            assert viewer.layers[layer].data.shape == image.shape
        check_prompt_segmentation(viewer)
        check_rendering(viewer, output / f"{name}.png")
        if name == "3d":
            viewer.dims.ndisplay = 3
            check_rendering(viewer, output / "3d-rendering.png")
            viewer.dims.ndisplay = 2
        print(f"PASS: {name} annotation, segmentation, commit, and shortcuts", flush=True)
    finally:
        viewer.close()


def check_series(images, kwargs, output):
    AnnotatorState().reset_state()
    viewer = napari.Viewer()
    try:
        with TemporaryDirectory() as folder:
            image_series_annotator(images, folder, viewer=viewer, return_viewer=True, **kwargs)
            check_prompt_segmentation(viewer)
            invoke_shortcut(viewer, "n")
            saved = list(Path(folder).glob("*.tif"))
            assert len(saved) == 1 and imageio.imread(saved[0]).any()
            np.testing.assert_array_equal(viewer.layers["image"].data, images[1])
            assert not viewer.layers["committed_objects"].data.any()
            check_prompt_segmentation(viewer)
            check_rendering(viewer, output / "image-series.png")
            print("PASS: image series navigation, saved labels, and next-image segmentation", flush=True)
    finally:
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="The micro-sam cache with models and sample data.")
    parser.add_argument("--output", type=Path, required=True, help="The directory for screenshots.")
    parser.add_argument("--device", default="cpu", help="The device for SAM inference.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"napari: {napari.__version__}", flush=True)
    data = args.cache / "sample_data"
    image = imageio.imread(data / "hela-2d-image.png")[:256, :256]
    volume = imageio.imread(data / "3d-nucleus-data.tif")[:3, :256, :256]
    frames = sorted((data / "DIC-C2DH-HeLa.zip.unzip" / "DIC-C2DH-HeLa" / "01").glob("t*.tif"))[:3]
    assert len(frames) == 3, "The tracking sample data are missing."
    tracking = np.stack([imageio.imread(path)[:256, :256] for path in frames])
    kwargs = {"model_type": "vit_b", "checkpoint_path": str(args.cache / "models" / "vit_b"), "device": args.device}
    check_plugins(args.output)
    check_annotator("2d", annotator_2d, image, kwargs, args.output)
    check_annotator("3d", annotator_3d, volume, kwargs, args.output)
    check_annotator("tracking", annotator_tracking, tracking, kwargs, args.output)
    check_series(list(tracking[:2]), kwargs, args.output)
    check_automatic(image, args.cache, args.device, args.output)
    AnnotatorState().reset_state()
    print("All napari workflow checks passed.", flush=True)


if __name__ == "__main__":
    main()
