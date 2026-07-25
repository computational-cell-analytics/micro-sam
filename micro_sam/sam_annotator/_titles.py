"""Dock widget titles for the annotation tools.

These are the single source for the titles of the napari dock widgets. The CLI / API entry points pass
them to `add_dock_widget`. The napari plugin menu cannot import them, because its manifest
(`micro_sam/napari.yaml`) is static data, so the `display_name` entries there repeat these titles and
have to be updated together with this file.
"""

PREFIX = "Segment Anything for Microscopy"

dock_titles = {
    "segmentation": f"{PREFIX} (Segmentation)",
    "tracking": f"{PREFIX} (Tracking)",
    "pixel_classification": f"{PREFIX} (Pixel Classification)",
    "object_classification": f"{PREFIX} (Object Classification)",
    "batch_annotation": f"{PREFIX} (Batch Annotation)",
    "batch_segmentation": f"{PREFIX} (Batch Segmentation)",
    "batch_tracking": f"{PREFIX} (Batch Tracking)",
    "batch_pixel_classification": f"{PREFIX} (Batch Pixel Classification)",
    "batch_object_classification": f"{PREFIX} (Batch Object Classification)",
    "settings": f"{PREFIX} (Settings)",
    "batch_navigation": "Batch Navigation",
}


def get_dock_title(name: str) -> str:
    """Retrieves the dock widget title for a given tool.

    Args:
        name: The name of the tool, for example 'segmentation' or 'batch_tracking'.

    Returns:
        The title of the dock widget.

    Raises:
        KeyError: If the tool is not known.
    """
    if name not in dock_titles:
        raise KeyError(f"'{name}' is not a known dock widget. Choose one of {list(dock_titles)}.")
    return dock_titles[name]
