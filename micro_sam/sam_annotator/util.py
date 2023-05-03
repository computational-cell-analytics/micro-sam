import numpy as np
from magicgui.widgets import ComboBox, Container


def create_prompt_menu(points_layer, labels):
    label_menu = ComboBox(label="prompts", choices=labels)
    label_widget = Container(widgets=[label_menu])

    def update_label_menu(event):
        new_label = str(points_layer.current_properties["label"][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    points_layer.events.current_properties.connect(update_label_menu)

    def label_changed(new_label):
        current_properties = points_layer.current_properties
        current_properties["label"] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    label_menu.changed.connect(label_changed)

    return label_widget


def prompt_layer_to_points(prompt_layer, i=None):
    """Extract point prompts for SAM from point layer.

    Argumtents:
        prompt_layer: the point layer
        i [int] - index for the data (required for 3d data)
    """

    points = prompt_layer.data
    labels = prompt_layer.properties["label"]
    assert len(points) == len(labels)

    if i is None:
        assert points.ndim == 2
        this_points, this_labels = points, labels
    else:
        assert points.ndim == 3
        mask = points[:, 0] == i
        this_points = points[mask][:, 1:]
        this_labels = labels[mask]
    assert len(this_points) == len(this_labels)

    this_labels = np.array([1 if label == "positive" else 0 for label in this_labels])
    # a single point with a negative label is interpreted as 'stop' signal
    # in this case we return None
    if len(this_points) == 1 and this_labels[0] == 0:
        return None

    return this_points, this_labels
