"""Shared navigation harness for the image series annotator.

Hosts any per-task annotator (segmentation, tracking, classification) and drives
Next/Previous navigation, the skip-already-done check and per-item loading/saving
through a small task-adapter interface (`SeriesAnnotatorTask`). The harness is
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
    """Hide the docked annotator's embedding section during a series session.

    The launcher's advanced settings are the single source of truth for the model / tiling / device /
    embedding-path / ndim, and the harness computes embeddings itself in 'start'/'advance', so the
    annotator's embedding panel (and its 'Compute Embeddings' button) is redundant here. The widget
    object is kept alive (just hidden) because '_sync_embedding_widget' and the classifier spec read
    from it. Standalone (non-series) annotators never call this, so their panel stays visible.
    """
    ew = getattr(annotator, "_embedding_widget", None)
    if ew is None:
        return
    # Each annotator wraps its widgets in a QGroupBox; hide that wrapper so the whole section (frame
    # included) disappears rather than leaving an empty box.
    frame = ew
    while frame is not None and not isinstance(frame, QtWidgets.QGroupBox):
        frame = frame.parentWidget()
    (frame or ew).hide()


def _embed_navigation(viewer, annotator, nav_container):
    """Add the navigation controls as a 'Series Navigation' section inside the docked annotator.

    Falls back to a standalone dock widget if the annotator has no embeddable inner layout.
    """
    inner = getattr(annotator, "_annotator_widget", None)
    if inner is None or inner.layout() is None:
        viewer.window.add_dock_widget(nav_container, name="Series Navigation")
        return
    group = QtWidgets.QGroupBox("Series Navigation")
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

    def nav_extra_widgets(self):
        """Extra magicgui widgets to place next to the Next button in the Series Navigation container.

        Task-specific (e.g. the classifiers' 'Forward Classifier State' checkbox); none by default.
        """
        return []


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

    # The launcher owns the model / embedding settings in a series session, so hide the annotator's
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

    def _do_prev(*args):
        if current_index == 0:
            widgets._generate_message("info", "This is already the first image.")
            return
        # Save the current item before stepping back so progress is not lost.
        _save_current()
        _go_to(current_index - 1)

    # Build a single navigation container (Previous + Next) and embed it as a section inside the
    # docked annotator, so the controls travel with the image series annotator instead of as
    # separate floating dock widgets. The actions are also tracked in the shared state, so they can
    # be triggered programmatically (e.g. in tests) just like the annotator's own widgets.
    state = AnnotatorState()
    next_button = PushButton(text="Next Image [N]")
    next_button.clicked.connect(lambda: _do_next())
    state.widgets["series_next"] = _do_next

    nav_buttons = []
    # Backward navigation is only offered for tasks that do not accumulate state forward.
    if task.supports_previous:
        prev_button = PushButton(text="Previous Image [P]")
        prev_button.clicked.connect(lambda: _do_prev())
        state.widgets["series_prev"] = _do_prev
        nav_buttons.append(prev_button)
    nav_buttons.append(next_button)
    # Task-specific controls placed next to Next (e.g. the classifiers' 'Forward Classifier State').
    nav_buttons.extend(task.nav_extra_widgets())

    nav_container = Container(layout="horizontal", widgets=nav_buttons, labels=False)
    nav_container.native.layout().setContentsMargins(0, 0, 0, 0)
    _embed_navigation(viewer, annotator, nav_container)

    @viewer.bind_key("n", overwrite=True)
    def _next_image(viewer):
        _do_next()

    if task.supports_previous:
        @viewer.bind_key("p", overwrite=True)
        def _prev_image(viewer):
            _do_prev()

    if return_viewer:
        return viewer
    napari.run()
