"""Shared navigation harness for the image series annotator.

Hosts any per-task annotator (segmentation, tracking, classification) and drives
Next/Previous navigation, the skip-already-done check and per-item loading/saving
through a small task-adapter interface (`SeriesAnnotatorTask`). The harness is
task-agnostic; everything task-specific lives in the concrete adapter.
"""

import os

import napari
import imageio.v3 as imageio
from magicgui import magicgui
from qtpy.QtCore import QTimer

from . import _widgets as widgets
from ._state import AnnotatorState


class SeriesAnnotatorTask:
    """Adapter encoding the task-specific parts of an image series session.

    The harness owns navigation, the skip-already-done check and the end-of-series dialog.
    The adapter owns precomputing the model and embeddings, loading an item into the viewer,
    deciding whether there is content worth saving, and saving the per-item result. Concrete
    tasks: segmentation, tracking, object/pixel classification.
    """

    #: Folder where per-item results are written. Set by the harness before the session starts.
    output_folder = None

    #: Whether the series inputs are in-memory arrays (True) or file paths (False). Set by the harness.
    have_inputs_as_arrays = False

    #: Whether backward (Previous) navigation is offered. Tasks that accumulate state forward across
    #: the series (e.g. the classifiers, which would double-count features on revisit) set this False.
    supports_previous = True

    def result_filename(self, entry, index: int) -> str:
        """Return the filename (relative to `output_folder`) of this item's saved result.

        Used to skip items that are already done. `entry` is the raw series entry
        (an array or a file path), `index` its position in the series.
        """
        raise NotImplementedError

    def precompute(self, images):
        """Prepare the shared model and per-item embeddings.

        Returns a list with one embedding path (or None) per image, aligned to `images`.
        """
        raise NotImplementedError

    def start(self, viewer, entry, image, embedding_path, index: int):
        """Initialize the predictor/state for the first item, dock the annotator and return it."""
        raise NotImplementedError

    def advance(self, viewer, annotator, entry, image, embedding_path, index: int):
        """Load a subsequent item into the existing viewer and annotator."""
        raise NotImplementedError

    def has_unsaved_content(self, viewer) -> bool:
        """Whether the user produced anything worth saving for the current item."""
        raise NotImplementedError

    def save_item(self, viewer, entry, index: int) -> None:
        """Persist the current item's result into `output_folder`."""
        raise NotImplementedError

    def on_leave_item(self, viewer, entry, index: int) -> None:
        """Hook called when navigating away from an item (e.g. classifier feature/RF accumulation)."""

    #: Message shown when the user advances without producing anything for the current item.
    empty_item_message = "Nothing is annotated yet. Do you wish to continue to the next image?"


def run_image_series(
    images,
    output_folder,
    task: SeriesAnnotatorTask,
    *,
    have_inputs_as_arrays: bool,
    viewer=None,
    return_viewer: bool = False,
    skip_done: bool = True,
):
    """Drive an image series annotation session for any task.

    Args:
        images: The series entries (in-memory arrays or file paths).
        output_folder: The folder where per-item results are saved.
        task: The task adapter that encodes the task-specific behavior.
        have_inputs_as_arrays: Whether `images` holds arrays (True) or file paths (False).
        viewer: An optional pre-initialized napari viewer.
        return_viewer: Whether to return the viewer instead of starting the napari event loop.
        skip_done: Whether to skip items whose result already exists in `output_folder`.

    Returns:
        The napari viewer, only if `return_viewer=True`.
    """
    end_msg = "You have annotated the last image. Do you wish to close napari?"
    os.makedirs(output_folder, exist_ok=True)
    task.output_folder = output_folder
    task.have_inputs_as_arrays = have_inputs_as_arrays
    n_images = len(images)

    def _is_done(index):
        path = os.path.join(output_folder, task.result_filename(images[index], index))
        return os.path.exists(path)

    def _load_pixels(index):
        entry = images[index]
        return entry if have_inputs_as_arrays else imageio.imread(entry)

    # Prepare the shared model and per-item embeddings (task-specific).
    embedding_paths = task.precompute(images)

    # Find the first item to annotate, optionally skipping items that are already done.
    current_index = 0
    if skip_done:
        while current_index < n_images and _is_done(current_index):
            current_index += 1
        if current_index == n_images:
            print("All images have already been annotated and 'skip_done' is set. Nothing to do.")
            return
        if current_index != 0:
            print("The first image to annotate is image number", current_index)

    if viewer is None:
        viewer = napari.Viewer()

    image = _load_pixels(current_index)
    annotator = task.start(viewer, images[current_index], image, embedding_paths[current_index], current_index)

    def _go_to(index):
        nonlocal current_index
        current_index = index
        image = _load_pixels(index)
        print("Loading image:", images[index] if not have_inputs_as_arrays else f"at index {index}")
        task.advance(viewer, annotator, images[index], image, embedding_paths[index], index)

    def _save_current():
        task.on_leave_item(viewer, images[current_index], current_index)
        task.save_item(viewer, images[current_index], current_index)

    @magicgui(call_button="Next Image [N]")
    def next_image(*args):
        # Prompt before advancing if nothing was produced for this item.
        if not task.has_unsaved_content(viewer):
            if widgets._generate_message("info", task.empty_item_message):
                return

        _save_current()

        # Find the next item, skipping already-done ones if requested.
        index = current_index + 1
        if skip_done:
            while index < n_images and _is_done(index):
                index += 1
        if index >= n_images:
            if not widgets._generate_message("info", end_msg):
                QTimer.singleShot(0, viewer.close)
            return
        _go_to(index)

    viewer.window.add_dock_widget(next_image)
    # Track the navigation controls in the shared state alongside the other widgets, so they can be
    # triggered programmatically (e.g. in tests) just like the annotator's own widgets.
    AnnotatorState().widgets["series_next"] = next_image

    @viewer.bind_key("n", overwrite=True)
    def _next_image(viewer):
        next_image(viewer)

    # Backward navigation is only offered for tasks that do not accumulate state forward.
    if task.supports_previous:
        @magicgui(call_button="Previous Image [P]")
        def prev_image(*args):
            if current_index == 0:
                widgets._generate_message("info", "This is already the first image.")
                return
            # Save the current item before stepping back so progress is not lost.
            _save_current()
            _go_to(current_index - 1)

        viewer.window.add_dock_widget(prev_image)
        AnnotatorState().widgets["series_prev"] = prev_image

        @viewer.bind_key("p", overwrite=True)
        def _prev_image(viewer):
            prev_image(viewer)

    if return_viewer:
        return viewer
    napari.run()
