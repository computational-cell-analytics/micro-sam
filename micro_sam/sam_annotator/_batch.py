"""Shared navigation harness for the batch annotator.

Hosts any per-task annotator (segmentation, tracking, classification) and drives
forward navigation, the skip-already-done check and per-item loading/saving
through a small task-adapter interface (`BatchAnnotatorTask`). The harness is
task-agnostic; everything task-specific lives in the concrete adapter.
"""

import os

import napari
import imageio.v3 as imageio
from magicgui.widgets import Container, PushButton
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QTimer

from . import _widgets as widgets
from ._state import AnnotatorState


def _hide_embedding_widget(annotator):
    """Hide the docked annotator's embedding section during a batch session.

    The launcher's advanced settings are the single source of truth for the model / tiling / device /
    embedding-path / ndim, and the harness computes embeddings itself in 'start'/'advance', so the
    annotator's embedding panel (and its 'Compute Embeddings' button) is redundant here. The widget
    object is kept alive (just hidden) because '_sync_embedding_widget' and the classifier spec read
    from it. Standalone (non-batch) annotators never call this, so their panel stays visible.
    """
    ew = getattr(annotator, "_embedding_widget", None)
    if ew is None:
        return
    # Each annotator wraps its widgets in a QGroupBox. Hide that wrapper so the whole section (frame
    # included) disappears rather than leaving an empty box.
    frame = ew
    while frame is not None and not isinstance(frame, QtWidgets.QGroupBox):
        frame = frame.parentWidget()
    (frame or ew).hide()


def _embed_navigation(viewer, annotator, nav_container):
    """Add the navigation controls as a 'Batch Navigation' section inside the docked annotator.

    Falls back to a standalone dock widget if the annotator has no embeddable inner layout.
    """
    inner = getattr(annotator, "_annotator_widget", None)
    if inner is None or inner.layout() is None:
        viewer.window.add_dock_widget(nav_container, name="Batch Navigation")
        return
    group = QtWidgets.QGroupBox("Batch Navigation")
    group_layout = QtWidgets.QVBoxLayout()
    # Add a top margin so the group title is not cramped against the navigation buttons.
    group_layout.setContentsMargins(8, 14, 8, 8)
    group_layout.addWidget(nav_container.native)
    group.setLayout(group_layout)
    # Pin to the top of the annotator panel so it stays visible (the task annotators are tall and
    # the navigation would otherwise sit below the fold at the bottom of the scroll area).
    inner.layout().insertWidget(0, group)


def _maximize_dock_vertically(viewer, annotator):
    """Expand the docked annotator to fill the available vertical space when it opens.

    napari sizes a freshly docked widget to its (small) size hint, leaving it shrunk; this makes the
    annotator claim the full window height instead. Best-effort and guarded, so it is a no-op in
    headless / test runs where the main window is not shown.
    """
    size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
    annotator.setSizePolicy(size_policy.Preferred, size_policy.Expanding)

    # Walk up to the QDockWidget that hosts the annotator.
    dock = annotator
    while dock is not None and not isinstance(dock, QtWidgets.QDockWidget):
        dock = dock.parentWidget()
    if dock is None:
        return

    def _resize():
        try:
            main_window = viewer.window._qt_window
            main_window.resizeDocks([dock], [main_window.height()], Qt.Vertical)
        except Exception:
            pass

    # Defer until the window is shown, otherwise the initial dock layout overrides the resize.
    QTimer.singleShot(0, _resize)


class BatchAnnotatorTask:
    """Adapter encoding the task-specific parts of a batch annotation session.

    The harness owns navigation, the skip-already-done check and the end-of-batch dialog.
    The adapter owns precomputing the model and embeddings, loading an item into the viewer,
    deciding whether there is content worth saving, and saving the per-item result. Concrete
    tasks: segmentation, tracking, object/pixel classification.
    """

    #: Folder where per-item results are written. Set by the harness before the session starts.
    output_folder = None

    #: Whether the batch inputs are in-memory arrays (True) or file paths (False). Set by the harness.
    have_inputs_as_arrays = False

    def result_filename(self, entry, index: int) -> str:
        """Return the filename (relative to `output_folder`) of this item's saved result.

        Used to skip items that are already done. `entry` is the raw batch entry
        (an array or a file path), `index` its position in the batch.
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

    def nav_extra_widgets(self):
        """Extra magicgui widgets to place next to the Next button in the Batch Navigation container.

        Task-specific (e.g. the classifiers' 'Forward Classifier State' checkbox); none by default.
        """
        return []


def run_batch(
    images,
    output_folder,
    task: BatchAnnotatorTask,
    *,
    have_inputs_as_arrays: bool,
    viewer=None,
    return_viewer: bool = False,
    skip_done: bool = True,
):
    """Drive a batch annotation session for any task.

    Args:
        images: The batch entries (in-memory arrays or file paths).
        output_folder: The folder where per-item results are saved.
        task: The task adapter that encodes the task-specific behavior.
        have_inputs_as_arrays: Whether `images` holds arrays (True) or file paths (False).
        viewer: An optional pre-initialized napari viewer.
        return_viewer: Whether to return the viewer instead of starting the napari event loop.
        skip_done: Whether to skip items whose result already exists in `output_folder`.

    Returns:
        The napari viewer, only if `return_viewer=True`.
    """
    end_msg = "You annotated the last image. Do you want to close napari?"
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
            # The batch launcher reports this in a dialog and stays open so its settings can be
            # changed. Keep the terminal message for direct Python / CLI use, where no viewer was
            # supplied to display that feedback.
            if viewer is None:
                print("All images have already been annotated and 'skip_done' is set. Nothing to do.")
            return
        if current_index != 0:
            print("The first image to annotate is image number", current_index)

    if viewer is None:
        viewer = napari.Viewer()

    image = _load_pixels(current_index)
    annotator = task.start(viewer, images[current_index], image, embedding_paths[current_index], current_index)

    # The launcher owns the model / embedding settings in a batch session, so hide the annotator's
    # (now redundant) embedding section to avoid duplicating those controls.
    _hide_embedding_widget(annotator)

    # Open the annotator maximized vertically instead of shrunk to its size hint.
    _maximize_dock_vertically(viewer, annotator)

    def _go_to(index):
        nonlocal current_index
        current_index = index
        image = _load_pixels(index)
        print("Loading image:", images[index] if not have_inputs_as_arrays else f"at index {index}")
        task.advance(viewer, annotator, images[index], image, embedding_paths[index], index)

    def _save_current():
        task.on_leave_item(viewer, images[current_index], current_index)
        task.save_item(viewer, images[current_index], current_index)

    def _do_next(*args):
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

    # Embed the navigation controls in the docked annotator, so they travel with the batch
    # annotator instead of as a separate floating dock widget. The action is also tracked in the
    # shared state, so it can be triggered programmatically (e.g. in tests) just like the annotator's
    # own widgets.
    state = AnnotatorState()
    next_button = PushButton(text="Next Image [N]")
    next_button.clicked.connect(lambda: _do_next())
    state.widgets["batch_next"] = _do_next

    nav_buttons = [next_button]
    # Task-specific controls placed next to Next (e.g. the classifiers' 'Keep Classifier').
    nav_buttons.extend(task.nav_extra_widgets())

    nav_container = Container(layout="horizontal", widgets=nav_buttons, labels=False)
    nav_container.native.layout().setContentsMargins(0, 0, 0, 0)
    _embed_navigation(viewer, annotator, nav_container)

    @viewer.bind_key("n", overwrite=True)
    def _next_image(viewer):
        _do_next()

    if return_viewer:
        return viewer
    napari.run()
