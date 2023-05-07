import numpy as np

from magicgui import magicgui
from magicgui.widgets import ComboBox, Container
from napari import Viewer

from ..segment_from_prompts import segment_from_points


@magicgui(call_button="Commit [C]", layer={"choices": ["current_object", "auto_segmentation"]})
def commit_segmentation_widget(v: Viewer, layer: str = "current_object"):
    seg = v.layers[layer].data

    id_offset = int(v.layers["committed_objects"].data.max())
    mask = seg != 0

    v.layers["committed_objects"].data[mask] = (seg[mask] + id_offset)
    v.layers["committed_objects"].refresh()

    shape = v.layers["raw"].data.shape
    v.layers[layer].data = np.zeros(shape, dtype="uint32")
    v.layers[layer].refresh()

    if layer == "current_object":
        v.layers["prompts"].data = []
        v.layers["prompts"].refresh()


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
        assert points.shape[1] == 2, f"{points.shape}"
        this_points, this_labels = points, labels
    else:
        assert points.shape[1] == 3, f"{points.shape}"
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


def segment_slices_with_prompts(predictor, prompt_layer, image_embeddings, shape, progress_bar=None):
    seg = np.zeros(shape, dtype="uint32")

    slices = np.unique(prompt_layer.data[:, 0]).astype("int")
    stop_lower, stop_upper = False, False

    def _update_progress():
        if progress_bar is not None:
            progress_bar.update(1)

    for i in slices:
        prompts_i = prompt_layer_to_points(prompt_layer, i)

        # TODO also take into account division properties once we have this implemented in tracking
        # do we end the segmentation at the outer slices?
        if prompts_i is None:

            if i == slices[0]:
                stop_lower = True
            elif i == slices[-1]:
                stop_upper = True
            else:
                raise RuntimeError("Stop slices can only be at the start or end")

            seg[i] = 0
            _update_progress()
            continue

        points, labels = prompts_i
        seg_i = segment_from_points(predictor, points, labels, image_embeddings=image_embeddings, i=i)
        seg[i] = seg_i
        _update_progress()

    return seg, slices, stop_lower, stop_upper
