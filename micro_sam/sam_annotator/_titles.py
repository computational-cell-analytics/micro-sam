"""Dock widget titles for the annotation tools.

These are the single source for the titles of the napari dock widgets. The CLI / API entry points pass
them to `add_dock_widget`. The napari plugin menu builds the same title itself, as
'<tool name> (<plugin display name>)', from the static manifest (`micro_sam/napari.yaml`). That manifest
cannot import this module, so the tool names below are repeated as its `display_name` entries and have to
be updated together with this file.
"""

PLUGIN_NAME = "Segment Anything for Microscopy"

tool_names = {
    "segmentation": "Segmentation Annotator",
    "tracking": "Tracking Annotator",
    "pixel_classification": "Pixel Classifier",
    "object_classification": "Object Classifier",
    "batch_annotation": "Batch Annotator",
    "batch_segmentation": "Batch Segmentation Annotator",
    "batch_tracking": "Batch Tracking Annotator",
    "batch_pixel_classification": "Batch Pixel Classifier",
    "batch_object_classification": "Batch Object Classifier",
    "settings": "Settings",
    "batch_navigation": "Batch Navigation",
}

dock_titles = {name: f"{tool_name} ({PLUGIN_NAME})" for name, tool_name in tool_names.items()}


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
