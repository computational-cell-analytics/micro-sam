from ._annotator import _AnnotatorBase

# TODO: I don't really understand the reason behind this,
# we need napari anyways, why don't we just import it?
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari


class Annotator2d(_AnnotatorBase):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer=viewer)

        # TODO do all the extra stuff for the 2d annotator


# TODO dummy function as placeholder for CLI
def annotator_2d():
    pass
